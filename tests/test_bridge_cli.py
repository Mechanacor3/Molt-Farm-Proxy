from __future__ import annotations

import json

from app.bridge_cli import build_codex_command
from app.devloop import base_run_payload, classify_proxy_failure, generate_run_id


def test_build_codex_command_injects_model_and_endpoint() -> None:
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
    command = build_codex_command(
        "/tmp/codex",
        "http://127.0.0.1:8000/v1",
        "codex-bridge",
        ["exec", "-m", "custom-model", "ping"],
    )
    assert command.count("-m") == 1
    assert "custom-model" in command


def test_build_codex_command_preserves_explicit_resume_session_id() -> None:
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
    assert classify_proxy_failure(405, "websocket_not_supported", "GET") == "transport_failure"
    assert classify_proxy_failure(400, "request_validation_error", "POST") == "schema_mismatch"
    assert classify_proxy_failure(400, "request_parse_error", "POST") == "schema_mismatch"
    assert classify_proxy_failure(400, "unsupported_tool", "POST") == "unsupported_tool"
    assert classify_proxy_failure(502, "upstream_bad_status", "POST") == "upstream_ollama_failure"
    assert classify_proxy_failure(422, "invalid_tool_call", "POST") == "proxy_validation_failure"


def test_base_run_payload_is_stable() -> None:
    payload = base_run_payload("bridge-123", "cli", "codex-bridge", "http://127.0.0.1:8000/v1", "/tmp/codex")
    assert payload["run_id"] == "bridge-123"
    assert payload["mode"] == "cli"
    assert payload["model"] == "codex-bridge"
