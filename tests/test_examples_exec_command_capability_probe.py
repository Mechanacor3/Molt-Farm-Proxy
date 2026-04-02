from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import httpx

_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "exec_command_capability_probe.py"
_SPEC = importlib.util.spec_from_file_location("exec_command_capability_probe", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

OBSERVED_FULL_EXEC_SPEC = _MODULE.OBSERVED_FULL_EXEC_SPEC
build_surface_unavailable_results = _MODULE.build_surface_unavailable_results
build_case_matrix = _MODULE.build_case_matrix
build_tool_presets = _MODULE.build_tool_presets
extract_chat_function_call = _MODULE.extract_chat_function_call
extract_responses_function_call = _MODULE.extract_responses_function_call
preflight_surface = _MODULE.preflight_surface
summarize_flags = _MODULE.summarize_flags


def test_build_tool_presets_keep_variants_intentional() -> None:
    presets = build_tool_presets()

    assert list(presets["narrow"]["parameters"]["properties"]) == ["cmd"]
    assert list(presets["narrow"]["parameters"]["required"]) == ["cmd"]
    assert set(presets["narrow_runtime"]["parameters"]["properties"]) == {
        "cmd",
        "max_output_tokens",
        "workdir",
        "yield_time_ms",
    }
    assert presets["observed_full"] == OBSERVED_FULL_EXEC_SPEC
    assert set(presets["observed_full"]["parameters"]["properties"]) == {
        "cmd",
        "justification",
        "login",
        "max_output_tokens",
        "prefix_rule",
        "sandbox_permissions",
        "shell",
        "tty",
        "workdir",
        "yield_time_ms",
    }

    presets["observed_full"]["parameters"]["properties"]["cmd"]["description"] = "changed"
    assert build_tool_presets()["observed_full"]["parameters"]["properties"]["cmd"]["description"] == (
        "Shell command to execute."
    )


def test_build_case_matrix_is_stable() -> None:
    assert [case.name for case in build_case_matrix()] == [
        "pwd_narrow_plain",
        "pwd_narrow_strict",
        "pwd_narrow_runtime_plain",
        "pwd_observed_full_plain",
        "readme_narrow_plain",
        "readme_narrow_strict",
        "readme_narrow_runtime_plain",
        "readme_observed_full_strict",
    ]


def test_extract_responses_function_call_normalizes_fields() -> None:
    payload = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "no tool"}]},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd"}',
            },
        ]
    }

    assert extract_responses_function_call(payload) == {
        "call_id": "call_123",
        "name": "exec_command",
        "arguments": '{"cmd":"pwd"}',
    }


def test_extract_chat_function_call_normalizes_fields() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": '{"cmd":"pwd"}',
                            },
                        }
                    ]
                }
            }
        ]
    }

    assert extract_chat_function_call(payload) == {
        "call_id": "call_456",
        "name": "exec_command",
        "arguments": '{"cmd":"pwd"}',
    }


def test_summarize_flags_detects_strict_help_and_followup_success() -> None:
    results = [
        {
            "surface": "chat",
            "case_name": "pwd_narrow_plain",
            "tool_preset": "narrow",
            "command_family": "pwd",
            "case_success": False,
            "followup_success": False,
        },
        {
            "surface": "chat",
            "case_name": "pwd_narrow_strict",
            "tool_preset": "narrow",
            "command_family": "pwd",
            "case_success": True,
            "followup_success": True,
        },
        {
            "surface": "chat",
            "case_name": "pwd_narrow_runtime_plain",
            "tool_preset": "narrow_runtime",
            "command_family": "pwd",
            "case_success": True,
            "followup_success": True,
        },
        {
            "surface": "responses",
            "case_name": "readme_observed_full_strict",
            "tool_preset": "observed_full",
            "command_family": "readme",
            "case_success": True,
            "followup_success": True,
        },
    ]

    assert summarize_flags(results) == {
        "minimal_pwd_ok": False,
        "strict_prompt_helps": True,
        "runtime_fields_ok": True,
        "full_schema_ok": True,
        "file_read_via_exec_ok": True,
        "responses_end_to_end_ok": True,
    }


def test_summarize_flags_limits_full_schema_and_file_read_flags() -> None:
    results = [
        {
            "surface": "chat",
            "case_name": "pwd_narrow_plain",
            "tool_preset": "narrow",
            "command_family": "pwd",
            "case_success": True,
            "followup_success": False,
        },
        {
            "surface": "chat",
            "case_name": "pwd_narrow_strict",
            "tool_preset": "narrow",
            "command_family": "pwd",
            "case_success": True,
            "followup_success": False,
        },
    ]

    assert summarize_flags(results) == {
        "minimal_pwd_ok": True,
        "strict_prompt_helps": False,
        "runtime_fields_ok": False,
        "full_schema_ok": False,
        "file_read_via_exec_ok": False,
        "responses_end_to_end_ok": False,
    }


def test_build_surface_unavailable_results_marks_all_cases_failed() -> None:
    cases = build_case_matrix()[:2]

    results = build_surface_unavailable_results("responses", cases, "surface_preflight_failed: timed out")

    assert [item["case_name"] for item in results] == ["pwd_narrow_plain", "pwd_narrow_strict"]
    assert all(item["surface"] == "responses" for item in results)
    assert all(item["case_success"] is False for item in results)
    assert all(item["error"] == "surface_preflight_failed: timed out" for item in results)


def test_preflight_surface_accepts_expected_proxy_probe() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(405, request=request))
    with httpx.Client(transport=transport) as client:
        assert preflight_surface(client, "responses", "http://proxy.test") is None


def test_preflight_surface_reports_bad_status_and_timeouts() -> None:
    bad_status_transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    with httpx.Client(transport=bad_status_transport) as client:
        assert preflight_surface(client, "responses", "http://proxy.test") == (
            "surface_preflight_failed: expected 405 from GET /v1/responses, got 200"
        )

    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=request)

    timeout_transport = httpx.MockTransport(raise_timeout)
    with httpx.Client(transport=timeout_transport) as client:
        assert preflight_surface(client, "responses", "http://proxy.test") == (
            "surface_preflight_failed: timed out contacting responses endpoint"
        )
