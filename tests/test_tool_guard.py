from __future__ import annotations

from app.errors import ProxyError
from app.schemas_chat import ChatToolCall, ChatToolFunctionCall
from app.schemas_responses import ResponseTool, ResponseToolFunction
from app.tool_guard import classify_response_tools, validate_and_rewrite_tool_calls, validate_response_tools


def _tool() -> ResponseTool:
    return ResponseTool(
        type="function",
        function=ResponseToolFunction(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        ),
    )


def _exec_tool() -> ResponseTool:
    return ResponseTool(
        type="function",
        function=ResponseToolFunction(
            name="exec_command",
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                },
                "required": ["cmd"],
            },
        ),
    )


def test_exact_tool_match_passes() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get_weather", arguments='{"city":"Boston"}'),
        )
    ]
    result = validate_and_rewrite_tool_calls(calls, [_tool()])
    assert result.validated[0].tool_call.function.name == "get_weather"
    assert result.validated[0].rewritten is False
    assert result.diagnostics["rewritten"] == 0


def test_near_match_rewrites_name() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get-weather", arguments='{"city":"Boston"}'),
        )
    ]
    result = validate_and_rewrite_tool_calls(calls, [_tool()])
    assert result.validated[0].tool_call.function.name == "get_weather"
    assert result.validated[0].rewritten is True
    assert result.diagnostics["rewritten_names"] == 1


def test_invalid_json_fails() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get_weather", arguments="{oops"),
        )
    ]
    try:
        validate_and_rewrite_tool_calls(calls, [_tool()])
    except ProxyError as exc:
        assert exc.code == "invalid_tool_call"
    else:  # pragma: no cover
        raise AssertionError("expected ProxyError")


def test_missing_required_argument_fails() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get_weather", arguments='{"days":2}'),
        )
    ]
    try:
        validate_and_rewrite_tool_calls(calls, [_tool()])
    except ProxyError as exc:
        assert exc.code == "invalid_tool_call"
    else:  # pragma: no cover
        raise AssertionError("expected ProxyError")


def test_coercions_and_extra_fields_are_tracked() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get_weather", arguments='{"city":7,"days":"2","extra":"ignored"}'),
        )
    ]
    result = validate_and_rewrite_tool_calls(calls, [_tool()])
    assert result.validated[0].tool_call.function.arguments == '{"city":"7","days":2}'
    assert result.diagnostics["coercions"] == 2
    assert result.diagnostics["dropped_extra_fields"] == 1


def test_common_argument_aliases_are_rewritten() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(
                name="exec_command",
                arguments='{"command":"pwd","workdir":"/tmp","timeout":5}',
            ),
        )
    ]
    result = validate_and_rewrite_tool_calls(calls, [_exec_tool()])
    assert result.validated[0].tool_call.function.arguments == '{"cmd":"pwd","workdir":"/tmp"}'
    assert result.diagnostics["dropped_extra_fields"] == 1


def test_disabled_web_search_tool_is_ignored() -> None:
    registry = validate_response_tools(
        [
            ResponseTool(type="web_search", external_web_access=False),
            ResponseTool(
                type="function",
                function=ResponseToolFunction(
                    name="exec_command",
                    parameters={"type": "object", "properties": {"cmd": {"type": "string"}}},
                ),
            ),
        ]
    )
    assert list(registry) == ["exec_command"]


def test_external_web_search_tool_is_rejected() -> None:
    try:
        validate_response_tools([ResponseTool(type="web_search", external_web_access=True)])
    except ProxyError as exc:
        assert exc.code == "unsupported_tool"
    else:  # pragma: no cover
        raise AssertionError("expected ProxyError")


def test_hosted_tool_is_classified_without_execution() -> None:
    result = classify_response_tools([ResponseTool(type="http", name="fetch_docs")])
    assert result.forwarded_tools == []
    assert result.as_log_payload()["counts"]["hosted_tool_observed_not_executed"] == 1


def test_only_non_forwardable_tools_fail_with_specific_error() -> None:
    try:
        classify_response_tools([ResponseTool(type="http", name="fetch_docs")], require_forwardable=True)
    except ProxyError as exc:
        assert exc.code == "no_forwardable_tools"
        assert exc.details["failure_detail"] == "tool_definition_policy_error"
    else:  # pragma: no cover
        raise AssertionError("expected ProxyError")
