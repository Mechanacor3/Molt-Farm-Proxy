from __future__ import annotations

from app.errors import ProxyError
from app.schemas_chat import ChatToolCall, ChatToolFunctionCall
from app.schemas_responses import ResponseTool, ResponseToolFunction
from app.tool_guard import validate_and_rewrite_tool_calls


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


def test_exact_tool_match_passes() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get_weather", arguments='{"city":"Boston"}'),
        )
    ]
    validated = validate_and_rewrite_tool_calls(calls, [_tool()])
    assert validated[0].tool_call.function.name == "get_weather"
    assert validated[0].rewritten is False


def test_near_match_rewrites_name() -> None:
    calls = [
        ChatToolCall(
            id="call_1",
            type="function",
            function=ChatToolFunctionCall(name="get-weather", arguments='{"city":"Boston"}'),
        )
    ]
    validated = validate_and_rewrite_tool_calls(calls, [_tool()])
    assert validated[0].tool_call.function.name == "get_weather"
    assert validated[0].rewritten is True


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
