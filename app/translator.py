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
from app.tool_guard import validate_and_rewrite_tool_calls, validate_response_tools


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


def _translate_tools(tools: list[ResponseTool] | None) -> list[ChatTool] | None:
    registry = validate_response_tools(tools)
    if not registry:
        return None
    translated: list[ChatTool] = []
    for name, tool in registry.items():
        function = tool.function or ResponseToolFunction(
            name=name,
            description=tool.description,
            parameters=tool.parameters or {},
        )
        translated.append(
            ChatTool(
                type="function",
                function={
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters,
                },
            )
        )
    return translated


def _filter_translated_tools(tools: list[ChatTool] | None, settings: Settings) -> list[ChatTool] | None:
    allowed = settings.debug_tool_name_set
    if not tools or not allowed:
        return tools
    filtered = [tool for tool in tools if tool.function.name in allowed]
    return filtered or None


def translate_responses_request_to_chat(request: ResponsesRequest, settings: Settings) -> ChatCompletionsRequest:
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
        tools=_filter_translated_tools(_translate_tools(request.tools), settings),
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
) -> ResponsesResponse:
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

    if message.tool_calls:
        validated_calls = validate_and_rewrite_tool_calls(message.tool_calls, request.tools)
        for validated in validated_calls:
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

    return ResponsesResponse(
        id=chat_response.id.replace("chatcmpl", "resp", 1),
        created_at=ResponsesResponse.now_iso(),
        model=chat_response.model,
        output=output,
        usage=_usage_from_chat(chat_response),
    )


def build_response_events(response: ResponsesResponse) -> list[dict[str, Any]]:
    payload = response.model_dump(mode="json")
    created_payload = dict(payload)
    created_payload["status"] = "in_progress"
    return [
        {"type": "response.created", "sequence_number": 0, "response": created_payload},
        {"type": "response.in_progress", "sequence_number": 1, "response": created_payload},
        {"type": "response.completed", "sequence_number": 2, "response": payload},
    ]


def build_sse_events(response: ResponsesResponse) -> list[str]:
    return [_format_sse(event) for event in build_response_events(response)] + ["data: [DONE]\n\n"]


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
