from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from app.errors import ProxyError
from app.schemas_chat import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    ChatTool,
    ChatToolCall,
    ChatToolFunctionCall,
)
from app.schemas_responses import (
    ResponseContentPart,
    ResponseInputItem,
    ResponseOutputItem,
    ResponseTool,
    ResponseToolFunction,
    ResponseUsageDetails,
    ResponsesRequest,
    ResponsesResponse,
)
from app.settings import Settings
from app.tool_guard import (
    ToolCompatibilityResult,
    ToolCallValidationResult,
    classify_response_tools,
    validate_and_rewrite_tool_calls,
)


def _item_to_model(item: ResponseInputItem | dict[str, Any]) -> ResponseInputItem:
    if isinstance(item, ResponseInputItem):
        return item
    return ResponseInputItem.model_validate(item)


def _extract_text(content: str | list[ResponseContentPart | dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = part.text
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _stringify_output(output: str | dict[str, Any] | list[Any] | None) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=True)


def _strip_json_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _single_function_tool_name(tools: list[ResponseTool] | None) -> str | None:
    names: list[str] = []
    for tool in tools or []:
        if tool.type != "function":
            continue
        name = tool.function.name if tool.function else tool.name
        if name:
            names.append(name)
    if len(names) == 1:
        return names[0]
    return None


def _schema_for_tool_name(tool_name: str, tools: list[ResponseTool] | None) -> dict[str, Any] | None:
    for tool in tools or []:
        if tool.type != "function":
            continue
        name = tool.function.name if tool.function else tool.name
        if name == tool_name:
            if tool.function:
                return tool.function.parameters
            return tool.parameters or {}
    return None


def _normalize_recovered_arguments(
    tool_name: str,
    arguments: str | dict[str, Any] | list[Any] | None,
    tools: list[ResponseTool] | None,
) -> str | dict[str, Any] | list[Any] | None:
    if not isinstance(arguments, str):
        return arguments

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    schema = _schema_for_tool_name(tool_name, tools) or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []

    if "cmd" in properties:
        return {"cmd": arguments}
    if isinstance(required, list) and len(required) == 1 and required[0] in properties:
        return {required[0]: arguments}
    if len(properties) == 1:
        only_property = next(iter(properties))
        return {only_property: arguments}
    return arguments


def _recover_tool_calls_from_message_content(
    content: str | None,
    tools: list[ResponseTool] | None,
) -> list[ChatToolCall] | None:
    if not content:
        return None

    candidate = _strip_json_code_fence(content)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    tool_name: str | None = None
    arguments: str | dict[str, Any] | None = None

    if isinstance(parsed, list) and len(parsed) == 2 and isinstance(parsed[0], str):
        tool_name = parsed[0]
        arguments = parsed[1]
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("function"), dict):
            function = parsed["function"]
            maybe_name = function.get("name")
            if isinstance(maybe_name, str):
                tool_name = maybe_name
                arguments = function.get("arguments")
        if tool_name is None:
            maybe_name = parsed.get("name") or parsed.get("tool") or parsed.get("tool_name")
            if isinstance(maybe_name, str):
                tool_name = maybe_name
                arguments = (
                    parsed.get("arguments")
                    or parsed.get("input")
                    or parsed.get("parameters")
                    or parsed.get("tool_input")
                )
        if tool_name is None:
            tool_name = _single_function_tool_name(tools)
            if tool_name is not None:
                arguments = parsed

    if not tool_name or arguments is None:
        return None

    arguments = _normalize_recovered_arguments(tool_name, arguments, tools)

    if isinstance(arguments, str):
        raw_arguments = arguments
    else:
        raw_arguments = json.dumps(arguments, separators=(",", ":"))

    return [
        ChatToolCall(
            id=f"call_{uuid4().hex[:8]}",
            type="function",
            function=ChatToolFunctionCall(name=tool_name, arguments=raw_arguments),
        )
    ]


def build_tool_compatibility(
    tools: list[ResponseTool] | None,
    settings: Settings,
    *,
    require_forwardable: bool = False,
) -> ToolCompatibilityResult:
    return classify_response_tools(
        tools,
        allowed_function_names=settings.debug_tool_name_set,
        require_forwardable=require_forwardable,
    )


def translate_responses_request_to_chat(
    request: ResponsesRequest,
    settings: Settings,
    compatibility: ToolCompatibilityResult | None = None,
) -> ChatCompletionsRequest:
    messages: list[ChatMessage] = []
    hidden_reasoning: list[str] = []

    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))

    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    else:
        for raw_item in request.input:
            item = _item_to_model(raw_item)
            if item.type == "message":
                role = item.role or "user"
                if role == "developer":
                    role = "system"
                messages.append(ChatMessage(role=role, content=_extract_text(item.content)))
            elif item.type == "reasoning":
                if item.summary:
                    snippets = [entry.get("text", "") for entry in item.summary if isinstance(entry, dict)]
                    hidden_reasoning.extend(filter(None, snippets))
                else:
                    hidden_reasoning.append(_extract_text(item.content))
            elif item.type == "function_call":
                if not item.name or not item.call_id:
                    raise ProxyError(400, "invalid_input", "Function call items must include 'name' and 'call_id'.")
                arguments = item.arguments
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                elif arguments is None:
                    arguments = "{}"
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ChatToolCall(
                                id=item.call_id,
                                type="function",
                                function=ChatToolFunctionCall(name=item.name, arguments=arguments),
                            )
                        ],
                    )
                )
            elif item.type == "function_call_output":
                if not item.call_id:
                    raise ProxyError(400, "invalid_input", "Function call output items must include 'call_id'.")
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=item.call_id,
                        content=_stringify_output(item.output),
                    )
                )
            else:
                raise ProxyError(
                    400,
                    "unsupported_input_item",
                    f"Unsupported Responses input item type '{item.type}'.",
                )

    if hidden_reasoning:
        reasoning_prefix = "Prior reasoning context:\n" + "\n".join(filter(None, hidden_reasoning))
        messages.insert(0, ChatMessage(role="system", content=reasoning_prefix))

    if not messages:
        raise ProxyError(400, "invalid_input", "Responses requests must include at least one input item.")

    model = settings.resolve_model(request.model)
    max_tokens = request.max_output_tokens

    return ChatCompletionsRequest(
        model=model,
        messages=messages,
        stream=False,
        tools=compatibility.forwarded_tools or None if compatibility is not None else build_tool_compatibility(request.tools, settings).forwarded_tools or None,
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=max_tokens,
        stop=request.stop,
    )


def _usage_from_chat(chat_response: ChatCompletionsResponse) -> ResponseUsageDetails:
    usage = chat_response.usage
    if usage is None:
        return ResponseUsageDetails()
    return ResponseUsageDetails(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


def translate_chat_response_to_responses(
    request: ResponsesRequest,
    chat_response: ChatCompletionsResponse,
) -> tuple[ResponsesResponse, ToolCallValidationResult]:
    if not chat_response.choices:
        raise ProxyError(502, "invalid_upstream_payload", "Ollama returned no completion choices.")

    choice = chat_response.choices[0]
    message = choice.message
    output: list[ResponseOutputItem] = []

    if message.reasoning:
        output.append(
            ResponseOutputItem(
                id=f"rs_{uuid4().hex}",
                type="reasoning",
                summary=[{"type": "summary_text", "text": message.reasoning}],
            )
        )

    recovered_tool_calls = None
    if not message.tool_calls:
        recovered_tool_calls = _recover_tool_calls_from_message_content(message.content, request.tools)

    if message.tool_calls or recovered_tool_calls:
        validation = validate_and_rewrite_tool_calls(message.tool_calls or recovered_tool_calls, request.tools)
        for validated in validation.validated:
            output.append(
                ResponseOutputItem(
                    id=validated.tool_call.id,
                    type="function_call",
                    call_id=validated.tool_call.id,
                    name=validated.tool_call.function.name,
                    arguments=validated.tool_call.function.arguments,
                    status="completed",
                )
            )
    else:
        validation = ToolCallValidationResult(
            validated=[],
            diagnostics={
                "attempted": 0,
                "validated": 0,
                "rewritten": 0,
                "coercions": 0,
                "dropped_extra_fields": 0,
                "rewritten_names": 0,
            },
        )
        output.append(
            ResponseOutputItem(
                id=f"msg_{uuid4().hex}",
                type="message",
                role="assistant",
                content=[
                    {
                        "type": "output_text",
                        "text": message.content or "",
                        "annotations": [],
                    }
                ],
                status="completed",
            )
        )

    return (
        ResponsesResponse(
            id=chat_response.id.replace("chatcmpl", "resp", 1),
            created_at=ResponsesResponse.now_iso(),
            model=chat_response.model,
            output=output,
            usage=_usage_from_chat(chat_response),
        ),
        validation,
    )


def build_response_events(response: ResponsesResponse) -> list[dict[str, Any]]:
    payload = response.model_dump(mode="json")
    created_payload = dict(payload)
    created_payload["status"] = "in_progress"
    return [
        {"type": "response.created", "sequence_number": 0, "response": created_payload},
        {"type": "response.in_progress", "sequence_number": 1, "response": created_payload},
        {"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": payload["output"][0]}
        if payload.get("output")
        else {"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": None},
        {"type": "response.completed", "sequence_number": 3, "response": payload},
    ]


def build_sse_events(response: ResponsesResponse) -> list[str]:
    return [_format_sse(event) for event in build_response_events(response)] + ["data: [DONE]\n\n"]


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
