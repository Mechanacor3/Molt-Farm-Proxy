from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "get_weather_tool_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("get_weather_tool_probe", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_parser = _MODULE.build_parser
build_request_headers = _MODULE.build_request_headers
build_responses_followup_input = _MODULE.build_responses_followup_input
responses_tool_output_value = _MODULE.responses_tool_output_value


def test_responses_tool_output_value_stringifies_objects() -> None:
    """Structured tool outputs should be stringified for the Responses follow-up turn."""
    assert (
        responses_tool_output_value({"temp_f": 61, "condition": "sunny"})
        == '{"temp_f": 61, "condition": "sunny"}'
    )
    assert responses_tool_output_value([1, 2, 3]) == "[1, 2, 3]"


def test_responses_tool_output_value_preserves_strings() -> None:
    """String tool outputs should pass through without re-encoding."""
    assert responses_tool_output_value('{"temp_f":61}') == '{"temp_f":61}'


def test_build_responses_followup_input_uses_string_tool_output() -> None:
    """Responses follow-up input should include stringified tool output content."""
    call = {
        "call_id": "call_123",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Boston", "unit": "f"}),
    }

    payload = build_responses_followup_input(
        "What is the weather in Boston?",
        call,
        {"temp_f": 61, "condition": "sunny"},
    )

    assert payload[2] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": '{"temp_f": 61, "condition": "sunny"}',
    }


def test_build_request_headers_includes_optional_bearer_token() -> None:
    """Probe helpers should add a bearer token only when an API key is supplied."""
    assert build_request_headers(None) == {"Content-Type": "application/json"}
    assert build_request_headers("local-dev-key") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer local-dev-key",
    }


def test_build_parser_accepts_api_key_flag() -> None:
    """The probe CLI should accept the authenticated-backend API key flag."""
    parser = build_parser()

    args = parser.parse_args(["--mode", "chat", "--api-key", "local-dev-key"])

    assert args.mode == "chat"
    assert args.api_key == "local-dev-key"
