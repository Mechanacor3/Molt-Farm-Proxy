from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.errors import ProxyError
from app.schemas_chat import ChatTool, ChatToolCall
from app.schemas_responses import ResponseTool, ResponseToolFunction

KNOWN_HOSTED_TOOL_TYPES = {
    "browser",
    "code_interpreter",
    "computer",
    "computer_use_preview",
    "file_search",
    "http",
    "http_request",
    "image_generation",
    "web_search_preview",
}
COMMON_ARGUMENT_ALIASES = {
    "cmd": {"command", "command_line", "cmdline", "shell_command"},
}
IGNORED_REQUEST_FIELDS = {
    "background",
    "conversation",
    "include",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "service_tier",
    "store",
    "text",
    "truncation",
    "user",
}


@dataclass
class ToolCompatibilityEntry:
    tool_type: str | None
    name: str | None
    disposition: str
    detail: str


@dataclass
class ToolCompatibilityResult:
    forwarded_registry: dict[str, ResponseTool] = field(default_factory=dict)
    forwarded_tools: list[ChatTool] = field(default_factory=list)
    entries: list[ToolCompatibilityEntry] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add_entry(
        self, tool_type: str | None, name: str | None, disposition: str, detail: str
    ) -> None:
        """Record one observed tool and increment the matching disposition count."""
        self.entries.append(
            ToolCompatibilityEntry(
                tool_type=tool_type,
                name=name,
                disposition=disposition,
                detail=detail,
            )
        )
        self.counts[disposition] = self.counts.get(disposition, 0) + 1

    def as_log_payload(self) -> dict[str, Any]:
        """Build the compact diagnostics payload stored in request logs."""
        return {
            "counts": {
                "observed": len(self.entries),
                "forwarded": self.counts.get("function_forwarded", 0),
                "filtered": self.counts.get("function_filtered", 0),
                "ignored": self.counts.get("web_search_disabled_ignored", 0)
                + self.counts.get("hosted_tool_observed_not_executed", 0),
                "rejected": self.counts.get("unsupported_tool_rejected", 0),
                **self.counts,
            },
            "entries": [
                {
                    "type": entry.tool_type,
                    "name": entry.name,
                    "disposition": entry.disposition,
                    "detail": entry.detail,
                }
                for entry in self.entries
            ],
        }


@dataclass
class ValidatedToolCall:
    tool_call: ChatToolCall
    rewritten: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallValidationResult:
    validated: list[ValidatedToolCall]
    diagnostics: dict[str, Any]


def _normalize_name(name: str) -> str:
    """Normalize tool names for loose matching across punctuation changes."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _function_tool_name(tool: ResponseTool) -> str | None:
    """Read the effective function name from either supported tool shape."""
    return tool.function.name if tool.function else tool.name


def _function_tool_parameters(tool: ResponseTool) -> dict[str, Any] | None:
    """Read the effective JSON schema from either supported tool shape."""
    return tool.function.parameters if tool.function else tool.parameters


def _build_forwarded_chat_tool(tool: ResponseTool, definition_name: str) -> ChatTool:
    """Convert a Responses function tool into the chat-completions tool shape."""
    function = tool.function or ResponseToolFunction(
        name=definition_name,
        description=tool.description,
        parameters=tool.parameters or {},
    )
    return ChatTool(
        type="function",
        function={
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters,
        },
    )


def classify_response_tools(
    tools: list[ResponseTool] | None,
    *,
    allowed_function_names: set[str] | None = None,
    require_forwardable: bool = False,
) -> ToolCompatibilityResult:
    """Classify which Responses tools can be forwarded to Ollama safely.

    The algorithm walks each advertised tool once, records what was observed
    for logging, forwards only function tools with compatible object schemas,
    ignores intentionally disabled hosted tools, and raises structured errors
    as soon as a tool definition would make the request ambiguous or unsafe.
    """
    result = ToolCompatibilityResult()
    if not tools:
        return result

    for tool in tools:
        tool_type = tool.type
        name = _function_tool_name(tool)
        if tool_type == "web_search":
            extra = tool.model_extra or {}
            if extra.get("external_web_access") is False:
                result.add_entry(
                    tool_type,
                    name,
                    "web_search_disabled_ignored",
                    "external access disabled",
                )
                continue
            result.add_entry(
                tool_type,
                name,
                "unsupported_tool_rejected",
                "external web access not supported",
            )
            raise ProxyError(
                400,
                "unsupported_tool",
                "Unsupported Responses tool type 'web_search'. This proxy cannot forward web search tools with external access.",
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )
        if tool_type in KNOWN_HOSTED_TOOL_TYPES:
            result.add_entry(
                tool_type,
                name,
                "hosted_tool_observed_not_executed",
                "hosted tool observed and not executed",
            )
            continue
        if tool_type != "function":
            result.add_entry(
                tool_type,
                name,
                "unsupported_tool_rejected",
                "tool type cannot be bridged over chat completions",
            )
            raise ProxyError(
                400,
                "unsupported_tool",
                (
                    f"Unsupported Responses tool type '{tool_type}'. "
                    "This proxy only forwards function tools that can be represented safely "
                    "over chat completions."
                ),
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )

        definition_name = name
        parameters = _function_tool_parameters(tool)
        if not definition_name:
            result.add_entry(
                tool_type,
                definition_name,
                "unsupported_tool_rejected",
                "function tool is missing a name",
            )
            raise ProxyError(
                400,
                "invalid_tool",
                "Function tools must include a name.",
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )
        if definition_name in result.forwarded_registry:
            result.add_entry(
                tool_type,
                definition_name,
                "unsupported_tool_rejected",
                "duplicate function tool name",
            )
            raise ProxyError(
                400,
                "invalid_tool",
                f"Duplicate function tool '{definition_name}'.",
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            result.add_entry(
                tool_type,
                definition_name,
                "unsupported_tool_rejected",
                "function parameters are not an object",
            )
            raise ProxyError(
                400,
                "invalid_tool",
                f"Function tool '{definition_name}' parameters must be an object.",
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )
        if parameters and parameters.get("type") not in (None, "object"):
            result.add_entry(
                tool_type,
                definition_name,
                "unsupported_tool_rejected",
                "function parameters use a non-object schema",
            )
            raise ProxyError(
                400,
                "invalid_tool",
                f"Function tool '{definition_name}' parameters must use a JSON schema object.",
                details={
                    "failure_detail": "tool_definition_problem",
                    "tool_diagnostics": result.as_log_payload(),
                },
            )

        if (
            allowed_function_names is not None
            and definition_name not in allowed_function_names
        ):
            result.add_entry(
                tool_type,
                definition_name,
                "function_filtered",
                "filtered by debug tool allowlist",
            )
            continue

        result.forwarded_registry[definition_name] = tool
        result.forwarded_tools.append(_build_forwarded_chat_tool(tool, definition_name))
        result.add_entry(
            tool_type,
            definition_name,
            "function_forwarded",
            "forwarded to chat completions",
        )

    if require_forwardable and tools and not result.forwarded_tools:
        raise ProxyError(
            400,
            "no_forwardable_tools",
            "Responses request contained tools, but none can be forwarded safely to chat completions.",
            details={
                "failure_detail": "tool_definition_policy_error",
                "tool_diagnostics": result.as_log_payload(),
            },
        )
    return result


def validate_response_tools(
    tools: list[ResponseTool] | None,
) -> dict[str, ResponseTool]:
    """Return only the forwarded function-tool registry for a request."""
    return classify_response_tools(tools).forwarded_registry


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Decode the upstream tool-call argument string into a JSON object."""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProxyError(
            422,
            "invalid_tool_call",
            f"Tool call arguments are not valid JSON: {exc.msg}",
            details={
                "failure_detail": "upstream_selected_invalid_tool",
                "tool_call_diagnostics": {"error": "invalid_json"},
            },
        ) from exc
    if not isinstance(data, dict):
        raise ProxyError(
            422,
            "invalid_tool_call",
            "Tool call arguments must decode to a JSON object.",
            details={
                "failure_detail": "upstream_selected_invalid_tool",
                "tool_call_diagnostics": {"error": "non_object_arguments"},
            },
        )
    return data


def _schema_for(tool: ResponseTool) -> dict[str, Any]:
    """Return the effective schema object for a function tool."""
    if tool.function:
        return tool.function.parameters
    return tool.parameters or {}


def _coerce_primitive(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    """Apply a few conservative primitive coercions before schema validation."""
    expected = schema.get("type")
    if expected == "string":
        if isinstance(value, str):
            return value, False
        if isinstance(value, (int, float, bool)):
            return str(value), True
    if expected == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, False
        if isinstance(value, str) and value.isdigit():
            return int(value), True
    if expected == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value, False
        if isinstance(value, str):
            try:
                return float(value), True
            except ValueError:
                return value, False
    if expected == "boolean":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true", True
    return value, False


def _resolve_property_name(
    argument_name: str, properties: dict[str, Any]
) -> str | None:
    """Match a model-supplied argument name to the canonical schema property.

    We try exact matches first, then normalization, a small alias table, and
    finally close matches so common punctuation or naming drift can be repaired
    without guessing across unrelated fields.
    """
    if argument_name in properties:
        return argument_name

    normalized_matches = [
        name
        for name in properties
        if _normalize_name(name) == _normalize_name(argument_name)
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    lowered = argument_name.lower()
    for canonical_name, aliases in COMMON_ARGUMENT_ALIASES.items():
        if canonical_name in properties and lowered in aliases:
            return canonical_name

    close = difflib.get_close_matches(
        argument_name, list(properties.keys()), n=1, cutoff=0.8
    )
    if len(close) == 1:
        return close[0]

    return None


def _repair_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite raw tool arguments into the advertised schema shape.

    The repair pass keeps only schema-known fields, renames close matches,
    performs narrow primitive coercions, and then fails fast if required fields
    are still missing after the cleanup.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    cleaned: dict[str, Any] = {}
    dropped_extra_fields: list[str] = []
    coercions: list[dict[str, str]] = []

    for key, value in arguments.items():
        property_name = _resolve_property_name(key, properties)
        if property_name is None:
            dropped_extra_fields.append(key)
            continue
        property_schema = properties.get(property_name, {})
        coerced, changed = _coerce_primitive(value, property_schema)
        cleaned[property_name] = coerced
        if changed:
            coercions.append(
                {
                    "field": property_name,
                    "target_type": str(property_schema.get("type")),
                }
            )

    missing = sorted(required - cleaned.keys())
    if missing:
        raise ProxyError(
            422,
            "invalid_tool_call",
            f"Tool call is missing required arguments: {', '.join(missing)}.",
            details={
                "failure_detail": "upstream_selected_invalid_tool",
                "tool_call_diagnostics": {
                    "error": "missing_required_arguments",
                    "missing_required": missing,
                    "dropped_extra_fields": dropped_extra_fields,
                    "coercions": coercions,
                },
            },
        )

    return cleaned, {
        "missing_required": [],
        "dropped_extra_fields": dropped_extra_fields,
        "coercions": coercions,
    }


def _candidate_tool_names(
    tool_name: str, registry: dict[str, ResponseTool]
) -> list[str]:
    """Produce the ordered list of plausible canonical tool names for a call."""
    exact = [name for name in registry if name == tool_name]
    if exact:
        return exact

    normalized_matches = [
        name for name in registry if _normalize_name(name) == _normalize_name(tool_name)
    ]
    if normalized_matches:
        return normalized_matches

    close = difflib.get_close_matches(
        tool_name, list(registry.keys()), n=2, cutoff=0.75
    )
    if close:
        return close

    normalized = _normalize_name(tool_name)
    return [
        name for name in registry if normalized and normalized in _normalize_name(name)
    ]


def validate_and_rewrite_tool_calls(
    tool_calls: list[ChatToolCall] | None,
    tools: list[ResponseTool] | None,
) -> ToolCallValidationResult:
    """Validate upstream tool calls against the advertised Responses tool list.

    Each proposed tool call is matched to a canonical tool name, its arguments
    are parsed and repaired against the JSON schema, and the final diagnostics
    summarize whether we rewrote names, coerced values, or dropped extras.
    """
    if not tool_calls:
        return ToolCallValidationResult(
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

    compatibility = classify_response_tools(tools)
    registry = compatibility.forwarded_registry
    if not registry:
        raise ProxyError(
            422,
            "unexpected_tool_call",
            "The model proposed a tool call but no tools were advertised.",
            details={
                "failure_detail": "upstream_selected_invalid_tool",
                "tool_diagnostics": compatibility.as_log_payload(),
            },
        )

    validated: list[ValidatedToolCall] = []
    total_coercions = 0
    total_dropped = 0
    rewritten_names = 0

    for tool_call in tool_calls:
        candidates = _candidate_tool_names(tool_call.function.name, registry)
        if not candidates:
            raise ProxyError(
                422,
                "invalid_tool_call",
                f"Unknown tool '{tool_call.function.name}' proposed by the model.",
                details={
                    "failure_detail": "upstream_selected_invalid_tool",
                    "tool_call_diagnostics": {
                        "error": "unknown_tool",
                        "tool_name": tool_call.function.name,
                    },
                },
            )
        if len(candidates) > 1:
            raise ProxyError(
                422,
                "ambiguous_tool_call",
                f"Tool '{tool_call.function.name}' matched multiple candidates: {', '.join(candidates)}.",
                details={
                    "failure_detail": "upstream_selected_invalid_tool",
                    "tool_call_diagnostics": {
                        "error": "ambiguous_tool_name",
                        "tool_name": tool_call.function.name,
                        "candidates": candidates,
                    },
                },
            )
        canonical_name = candidates[0]
        tool = registry[canonical_name]
        arguments = _parse_arguments(tool_call.function.arguments)
        repaired, repair_diagnostics = _repair_arguments(arguments, _schema_for(tool))
        rewritten_call = tool_call.model_copy(deep=True)
        rewritten_call.function.name = canonical_name
        rewritten_call.function.arguments = json.dumps(repaired, separators=(",", ":"))

        rewritten = (
            canonical_name != tool_call.function.name
            or rewritten_call.function.arguments != tool_call.function.arguments
        )
        if canonical_name != tool_call.function.name:
            rewritten_names += 1
        total_coercions += len(repair_diagnostics["coercions"])
        total_dropped += len(repair_diagnostics["dropped_extra_fields"])

        validated.append(
            ValidatedToolCall(
                tool_call=rewritten_call,
                rewritten=rewritten,
                diagnostics={
                    "original_name": tool_call.function.name,
                    "canonical_name": canonical_name,
                    "coercions": repair_diagnostics["coercions"],
                    "dropped_extra_fields": repair_diagnostics["dropped_extra_fields"],
                    "missing_required": repair_diagnostics["missing_required"],
                },
            )
        )

    rewritten_count = sum(1 for item in validated if item.rewritten)
    return ToolCallValidationResult(
        validated=validated,
        diagnostics={
            "attempted": len(tool_calls),
            "validated": len(validated),
            "rewritten": rewritten_count,
            "coercions": total_coercions,
            "dropped_extra_fields": total_dropped,
            "rewritten_names": rewritten_names,
            "calls": [
                {
                    "call_id": item.tool_call.id,
                    "rewritten": item.rewritten,
                    **item.diagnostics,
                }
                for item in validated
            ],
        },
    )


def summarize_request_ignored_fields(payload: object) -> dict[str, Any] | None:
    """Report which known-but-ignored request fields were present on input."""
    if not isinstance(payload, dict):
        return None

    ignored = sorted(field for field in payload if field in IGNORED_REQUEST_FIELDS)
    if not ignored:
        return None

    return {
        "recognized_ignored_fields": ignored,
        "count": len(ignored),
    }
