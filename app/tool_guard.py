from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any

from app.errors import ProxyError
from app.schemas_chat import ChatToolCall
from app.schemas_responses import ResponseTool


@dataclass
class ValidatedToolCall:
    tool_call: ChatToolCall
    rewritten: bool = False


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def validate_response_tools(tools: list[ResponseTool] | None) -> dict[str, ResponseTool]:
    registry: dict[str, ResponseTool] = {}
    if not tools:
        return registry

    for tool in tools:
        if tool.type != "function":
            raise ProxyError(
                status_code=400,
                code="unsupported_tool",
                message=(
                    f"Unsupported Responses tool type '{tool.type}'. "
                    "This proxy only forwards function tools that can be represented safely "
                    "over chat completions."
                ),
            )
        definition_name = tool.function.name if tool.function else tool.name
        parameters = tool.function.parameters if tool.function else tool.parameters
        if not definition_name:
            raise ProxyError(400, "invalid_tool", "Function tools must include a name.")
        if definition_name in registry:
            raise ProxyError(400, "invalid_tool", f"Duplicate function tool '{definition_name}'.")
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ProxyError(400, "invalid_tool", f"Function tool '{definition_name}' parameters must be an object.")
        if parameters and parameters.get("type") not in (None, "object"):
            raise ProxyError(
                400,
                "invalid_tool",
                f"Function tool '{definition_name}' parameters must use a JSON schema object.",
            )
        registry[definition_name] = tool
    return registry


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProxyError(422, "invalid_tool_call", f"Tool call arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProxyError(422, "invalid_tool_call", "Tool call arguments must decode to a JSON object.")
    return data


def _schema_for(tool: ResponseTool) -> dict[str, Any]:
    if tool.function:
        return tool.function.parameters
    return tool.parameters or {}


def _coerce_primitive(value: Any, schema: dict[str, Any]) -> Any:
    expected = schema.get("type")
    if expected == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
    if expected == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    if expected == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    return value


def _repair_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            continue
        property_schema = properties.get(key, {})
        cleaned[key] = _coerce_primitive(value, property_schema)

    missing = sorted(required - cleaned.keys())
    if missing:
        raise ProxyError(
            422,
            "invalid_tool_call",
            f"Tool call is missing required arguments: {', '.join(missing)}.",
        )
    return cleaned


def _candidate_tool_names(tool_name: str, registry: dict[str, ResponseTool]) -> list[str]:
    exact = [name for name in registry if name == tool_name]
    if exact:
        return exact

    normalized_matches = [name for name in registry if _normalize_name(name) == _normalize_name(tool_name)]
    if normalized_matches:
        return normalized_matches

    close = difflib.get_close_matches(tool_name, list(registry.keys()), n=2, cutoff=0.75)
    if close:
        return close

    normalized = _normalize_name(tool_name)
    return [name for name in registry if normalized and normalized in _normalize_name(name)]


def validate_and_rewrite_tool_calls(
    tool_calls: list[ChatToolCall] | None,
    tools: list[ResponseTool] | None,
) -> list[ValidatedToolCall]:
    if not tool_calls:
        return []

    registry = validate_response_tools(tools)
    if not registry:
        raise ProxyError(422, "unexpected_tool_call", "The model proposed a tool call but no tools were advertised.")

    validated: list[ValidatedToolCall] = []
    for tool_call in tool_calls:
        candidates = _candidate_tool_names(tool_call.function.name, registry)
        if not candidates:
            raise ProxyError(
                422,
                "invalid_tool_call",
                f"Unknown tool '{tool_call.function.name}' proposed by the model.",
            )
        if len(candidates) > 1:
            raise ProxyError(
                422,
                "ambiguous_tool_call",
                f"Tool '{tool_call.function.name}' matched multiple candidates: {', '.join(candidates)}.",
            )
        canonical_name = candidates[0]
        tool = registry[canonical_name]
        arguments = _parse_arguments(tool_call.function.arguments)
        repaired = _repair_arguments(arguments, _schema_for(tool))
        rewritten_call = tool_call.model_copy(deep=True)
        rewritten_call.function.name = canonical_name
        rewritten_call.function.arguments = json.dumps(repaired, separators=(",", ":"))
        validated.append(
            ValidatedToolCall(
                tool_call=rewritten_call,
                rewritten=(canonical_name != tool_call.function.name or rewritten_call.function.arguments != tool_call.function.arguments),
            )
        )
    return validated
