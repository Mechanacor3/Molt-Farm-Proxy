from __future__ import annotations

import json
import sys

from app import bridge_report
from app.bridge_cli import build_codex_command
from app.devloop import base_run_payload, classify_proxy_failure


def test_build_codex_command_injects_model_and_endpoint() -> None:
    """The bridge command should inject the endpoint override and fallback model."""
    command = build_codex_command(
        "/tmp/codex",
        "http://127.0.0.1:8000/v1",
        "codex-bridge",
        ["exec", "--json", "ping"],
    )
    assert command[0] == "/tmp/codex"
    assert "exec" in command
    assert "--json" in command
    assert "-m" in command
    assert 'openai_base_url="http://127.0.0.1:8000/v1"' in command


def test_build_codex_command_respects_existing_model() -> None:
    """An explicit caller-provided model should not be duplicated by the bridge."""
    command = build_codex_command(
        "/tmp/codex",
        "http://127.0.0.1:8000/v1",
        "codex-bridge",
        ["exec", "-m", "custom-model", "ping"],
    )
    assert command.count("-m") == 1
    assert "custom-model" in command


def test_build_codex_command_preserves_explicit_resume_session_id() -> None:
    """Resume invocations should keep the explicit session id arguments intact."""
    command = build_codex_command(
        "/tmp/codex",
        "http://127.0.0.1:8000/v1",
        "codex-bridge",
        ["exec", "resume", "019d4521-cf09-7b61-add0-069a443ca23e", "--json", "thanks"],
    )
    assert "resume" in command
    assert "019d4521-cf09-7b61-add0-069a443ca23e" in command
    assert "--json" in command
    assert "thanks" in command


def test_failure_classification_covers_expected_paths() -> None:
    """Observed proxy outcomes should map into the expected dev-loop buckets."""
    assert (
        classify_proxy_failure(405, "websocket_not_supported", "GET")
        == "transport_failure"
    )
    assert classify_proxy_failure(200, None, "GET", "responses_websocket") is None
    assert (
        classify_proxy_failure(400, "request_parse_error", "GET", "responses_websocket")
        == "transport_failure"
    )
    assert (
        classify_proxy_failure(400, "request_validation_error", "POST")
        == "schema_mismatch"
    )
    assert (
        classify_proxy_failure(400, "request_parse_error", "POST") == "schema_mismatch"
    )
    assert classify_proxy_failure(400, "unsupported_tool", "POST") == "unsupported_tool"
    assert (
        classify_proxy_failure(400, "no_forwardable_tools", "POST")
        == "unsupported_tool"
    )
    assert (
        classify_proxy_failure(502, "upstream_bad_status", "POST")
        == "upstream_ollama_failure"
    )
    assert (
        classify_proxy_failure(422, "invalid_tool_call", "POST")
        == "proxy_validation_failure"
    )


def test_base_run_payload_is_stable() -> None:
    """The shared bridge run payload should preserve the supplied metadata."""
    payload = base_run_payload(
        "bridge-123", "cli", "codex-bridge", "http://127.0.0.1:8000/v1", "/tmp/codex"
    )
    assert payload["run_id"] == "bridge-123"
    assert payload["mode"] == "cli"
    assert payload["model"] == "codex-bridge"


def test_bridge_report_surfaces_tool_breakdowns(tmp_path, capsys, monkeypatch) -> None:
    """The report should surface tool-count and failure-detail aggregates."""
    (tmp_path / "bridge-runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "bridge-123",
                "event": "run_finished",
                "status": "failed",
                "exit_code": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "proxy-requests.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "bridge_run_id": "bridge-123",
                        "status_code": 400,
                        "request_kind": "responses_http",
                        "failure_class": "unsupported_tool",
                        "failure_detail": "tool_definition_policy_error",
                        "tool_count_observed": 2,
                        "tool_count_forwarded": 1,
                        "tool_count_ignored": 1,
                        "tool_count_rejected": 0,
                        "tool_diagnostics": {
                            "counts": {
                                "observed": 2,
                                "function_forwarded": 1,
                                "hosted_tool_observed_not_executed": 1,
                            }
                        },
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys, "argv", ["codex-bridge-report", "--log-dir", str(tmp_path), "--limit", "1"]
    )
    bridge_report.main()
    output = capsys.readouterr().out
    assert "failure_details={'tool_definition_policy_error': 1}" in output
    assert "'forwarded': 1" in output
    assert "'hosted_tool_observed_not_executed': 1" in output
