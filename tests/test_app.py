from __future__ import annotations

import json
from typing import Any

import pytest
import zstandard as zstd
from fastapi.testclient import TestClient

from app.devloop import request_log_path
from app.main import app, get_ollama_client
from app.ollama_client import _authorization_headers
from app.schemas_chat import ChatCompletionsRequest, ChatCompletionsResponse
from app.settings import get_settings


class FakeOllamaClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the canned payload and remember the last request for assertions."""
        self.payload = payload
        self.last_request: ChatCompletionsRequest | None = None
        self.last_authorization: str | None = None

    async def close(self) -> None:
        """Match the real client shutdown API without doing any work."""
        return None

    async def create_chat_completion(
        self,
        request: ChatCompletionsRequest,
        *,
        authorization: str | None = None,
    ) -> ChatCompletionsResponse:
        """Capture the translated request and return the canned upstream payload."""
        self.last_request = request
        self.last_authorization = authorization
        return ChatCompletionsResponse.model_validate(self.payload)


@pytest.fixture
def client() -> TestClient:
    """Create a test client with the Ollama dependency overridden by a fake."""
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app.dependency_overrides[get_ollama_client] = lambda: fake
    with TestClient(app) as test_client:
        original_ollama = test_client.app.state.ollama_client
        test_client.app.state.ollama_client = fake
        test_client.fake_ollama = fake  # type: ignore[attr-defined]
        yield test_client
        test_client.app.state.ollama_client = original_ollama
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Reset cached settings around each test so env overrides stay isolated."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _zstd_json(payload: dict[str, Any]) -> bytes:
    """Compress a JSON payload the same way the real fallback client does."""
    return zstd.ZstdCompressor().compress(json.dumps(payload).encode("utf-8"))


def test_healthz(client: TestClient) -> None:
    """The health endpoint should report liveness and the default model."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_alias_matches_healthz(client: TestClient) -> None:
    """The `/health` alias should match the existing liveness payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_settings_prefer_upstream_base_url_over_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new generic upstream env should win when both names are present."""
    monkeypatch.setenv("MOLT_UPSTREAM_BASE_URL", "http://upstream.example")
    monkeypatch.setenv("MOLT_OLLAMA_BASE_URL", "http://legacy.example")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.upstream_base_url == "http://upstream.example"


def test_settings_still_accept_legacy_ollama_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy upstream env should still configure the proxy when used alone."""
    monkeypatch.setenv("MOLT_OLLAMA_BASE_URL", "http://legacy.example")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.upstream_base_url == "http://legacy.example"


def test_non_streaming_response(client: TestClient) -> None:
    """A basic non-streaming Responses request should round-trip to text output."""
    response = client.post("/v1/responses", json={"input": "ping", "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["text"] == "pong"


def test_incoming_authorization_is_forwarded_when_no_upstream_api_key(
    client: TestClient,
) -> None:
    """Client auth should be forwarded upstream when no explicit upstream key is set."""
    response = client.post(
        "/v1/responses",
        json={"input": "ping", "stream": False},
        headers={"Authorization": "Bearer forwarded-key"},
    )

    assert response.status_code == 200
    assert client.fake_ollama.last_authorization == "Bearer forwarded-key"  # type: ignore[attr-defined]


def test_authorization_headers_prefer_explicit_upstream_api_key() -> None:
    """The upstream API key should override any forwarded client auth."""
    assert _authorization_headers(
        upstream_api_key="upstream-secret",
        incoming_authorization="Bearer forwarded-key",
    ) == {"Authorization": "Bearer upstream-secret"}


def test_streaming_response_returns_final_sse(client: TestClient) -> None:
    """Streaming Responses calls should emit the typed SSE event sequence plus DONE."""
    with client.stream(
        "POST", "/v1/responses", json={"input": "ping", "stream": True}
    ) as response:
        text = "".join(
            chunk.decode() if isinstance(chunk, bytes) else chunk
            for chunk in response.iter_text()
        )
    assert response.status_code == 200
    assert '"type":"response.created"' in text
    assert '"type":"response.output_item.added"' in text
    assert '"type":"response.completed"' in text
    assert "data: [DONE]" in text


def test_zstd_streaming_real_client_shape(client: TestClient) -> None:
    """The proxy should decode zstd requests and preserve Codex role translation."""
    payload = {
        "model": "codex-bridge",
        "instructions": "System instructions",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Follow repo rules"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                "strict": False,
            }
        ],
        "tool_choice": "auto",
        "stream": True,
        "store": False,
        "reasoning": None,
        "include": [],
        "parallel_tool_calls": False,
        "prompt_cache_key": "req-123",
    }

    with client.stream(
        "POST",
        "/v1/responses",
        content=_zstd_json(payload),
        headers={
            "content-type": "application/json",
            "content-encoding": "zstd",
            "accept": "text/event-stream",
        },
    ) as response:
        text = "".join(
            chunk.decode() if isinstance(chunk, bytes) else chunk
            for chunk in response.iter_text()
        )

    assert response.status_code == 200
    assert '"type":"response.completed"' in text

    upstream = client.fake_ollama.last_request  # type: ignore[attr-defined]
    assert upstream is not None
    assert upstream.messages[0].role == "system"
    assert upstream.messages[0].content == "System instructions"
    assert upstream.messages[1].role == "system"
    assert upstream.messages[1].content == "Follow repo rules"
    assert upstream.messages[2].role == "user"
    assert upstream.messages[2].content == "hi"
    assert upstream.tools is not None


def test_unsupported_tool_rejected(client: TestClient) -> None:
    """Hosted-only tools should fail with the no-forwardable-tools policy error."""
    response = client.post(
        "/v1/responses",
        json={"input": "ping", "tools": [{"type": "web_search_preview"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_forwardable_tools"


def test_hosted_tool_is_observed_but_ignored_when_function_tool_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Hosted tools should be logged but ignored when a forwardable function tool exists."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "ping",
            "tools": [
                {"type": "http", "name": "fetch_docs"},
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        entries[-1]["tool_diagnostics"]["counts"]["hosted_tool_observed_not_executed"]
        == 1
    )
    assert entries[-1]["tool_diagnostics"]["counts"]["function_forwarded"] == 1


def test_only_hosted_tool_returns_specific_policy_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A request with only hosted tools should log and return the policy failure detail."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={"input": "ping", "tools": [{"type": "http", "name": "fetch_docs"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_forwardable_tools"
    assert (
        response.json()["error"]["details"]["failure_detail"]
        == "tool_definition_policy_error"
    )

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["failure_detail"] == "tool_definition_policy_error"


def test_get_probe_returns_structured_transport_error(client: TestClient) -> None:
    """GET probe traffic should return the explicit websocket-not-supported error."""
    response = client.get("/v1/responses")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "websocket_not_supported"


def test_websocket_response_create_streams_events(client: TestClient) -> None:
    """A websocket response.create frame should stream the typed event sequence."""
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {"type": "response.create", "input": "ping", "stream": True}
        )
        created = json.loads(websocket.receive_text())
        in_progress = json.loads(websocket.receive_text())
        output_added = json.loads(websocket.receive_text())
        completed = json.loads(websocket.receive_text())

    assert created["type"] == "response.created"
    assert in_progress["type"] == "response.in_progress"
    assert output_added["type"] == "response.output_item.added"
    assert completed["type"] == "response.completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "pong"


def test_websocket_invalid_first_message_returns_error(client: TestClient) -> None:
    """Non-JSON websocket payloads should surface the parse error shape."""
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_text("not-json")
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "request_parse_error"


def test_websocket_invalid_message_type_returns_error(client: TestClient) -> None:
    """Unsupported websocket message types should return a structured error event."""
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json({"type": "response.cancel"})
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "unsupported_websocket_message"


def test_websocket_validation_error_returns_error(client: TestClient) -> None:
    """Invalid websocket request bodies should return the validation error event."""
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {"type": "response.create", "input": {"unexpected": "shape"}}
        )
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "request_validation_error"


def test_request_validation_error_is_structured(client: TestClient) -> None:
    """HTTP validation failures should use the structured proxy error payload."""
    response = client.post("/v1/responses", json={"input": {"unexpected": "shape"}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


def test_invalid_zstd_request_body_is_structured(client: TestClient) -> None:
    """Bad zstd payloads should fail with the structured parse error."""
    response = client.post(
        "/v1/responses",
        content=b"not-a-zstd-frame",
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_parse_error"


def test_zstd_request_validation_error_is_structured(client: TestClient) -> None:
    """Decoded zstd payloads should still surface schema validation errors cleanly."""
    response = client.post(
        "/v1/responses",
        content=_zstd_json({"input": {"unexpected": "shape"}}),
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


def test_failed_post_parsing_is_logged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Request-parse failures should still produce a structured request log entry."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        content=b"not-a-zstd-frame",
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )

    assert response.status_code == 400

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["error_code"] == "request_parse_error"
    assert entries[-1]["failure_class"] == "schema_mismatch"
    assert entries[-1]["status_code"] == 400


def test_zstd_request_logs_decoded_tool_summary(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Decoded zstd requests should log a summary of observed tool definitions."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    payload = {
        "model": "codex-bridge",
        "input": "ping",
        "stream": True,
        "tools": [
            {
                "type": "web_search",
                "external_web_access": False,
            },
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        ],
    }

    response = client.post(
        "/v1/responses",
        content=_zstd_json(payload),
        headers={
            "content-type": "application/json",
            "content-encoding": "zstd",
            "accept": "text/event-stream",
        },
    )

    assert response.status_code == 200

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["tool_summary"] == [
        {
            "type": "web_search",
            "name": None,
            "function_name": None,
            "external_web_access": False,
            "has_function": False,
            "parameter_type": None,
            "property_names": None,
            "required": None,
        },
        {
            "type": "function",
            "name": "exec_command",
            "function_name": None,
            "external_web_access": None,
            "has_function": False,
            "parameter_type": "object",
            "property_names": ["cmd"],
            "required": ["cmd"],
        },
    ]
    assert entries[-1]["tool_diagnostics"]["counts"]["web_search_disabled_ignored"] == 1
    assert entries[-1]["tool_diagnostics"]["counts"]["function_forwarded"] == 1


def test_request_log_includes_upstream_tool_use_summary(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Successful tool use should be summarized in the structured request log."""
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app.dependency_overrides[get_ollama_client] = lambda: fake

    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "read it",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["upstream_summary"] == {
        "tool_count": 1,
        "tool_names": ["read_file"],
        "tool_choice": "auto",
        "finish_reason": "tool_calls",
        "response_has_tool_calls": True,
        "response_tool_call_names": ["read_file"],
        "response_content_present": False,
        "tool_call_validation": {
            "attempted": 1,
            "validated": 1,
            "rewritten": 0,
            "coercions": 0,
            "dropped_extra_fields": 0,
            "rewritten_names": 0,
            "calls": [
                {
                    "call_id": "call_1",
                    "rewritten": False,
                    "original_name": "read_file",
                    "canonical_name": "read_file",
                    "coercions": [],
                    "dropped_extra_fields": [],
                    "missing_required": [],
                }
            ],
        },
    }


def test_textual_tool_call_is_recovered_into_responses_function_call(
    client: TestClient,
) -> None:
    """JSON text output should be recovered into a validated function call."""
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"command":"pwd","timeout":5}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app.dependency_overrides[get_ollama_client] = lambda: fake

    response = client.post(
        "/v1/responses",
        json={
            "input": "run pwd",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["name"] == "exec_command"
    assert body["output"][0]["arguments"] == '{"cmd":"pwd"}'


def test_fenced_tool_name_tool_input_is_recovered(client: TestClient) -> None:
    """Fenced JSON with tool_name/tool_input should be recovered into a function call."""
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '```json\n{"tool_name":"exec_command","tool_input":"pwd"}\n```',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app.dependency_overrides[get_ollama_client] = lambda: fake

    response = client.post(
        "/v1/responses",
        json={
            "input": "run pwd",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["name"] == "exec_command"
    assert body["output"][0]["arguments"] == '{"cmd":"pwd"}'


def test_debug_tool_filter_reduces_forwarded_tool_names(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The debug tool allowlist should reduce the forwarded tool set and log summary."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("MOLT_DEBUG_TOOL_NAMES", "exec_command")
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "ping",
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
                {
                    "type": "function",
                    "name": "write_stdin",
                    "parameters": {
                        "type": "object",
                        "properties": {"session_id": {"type": "integer"}},
                        "required": ["session_id"],
                    },
                },
            ],
        },
    )

    assert response.status_code == 200

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["upstream_summary"]["tool_count"] == 1
    assert entries[-1]["upstream_summary"]["tool_names"] == ["exec_command"]


def test_non_utf8_request_body_does_not_crash(client: TestClient) -> None:
    """Non-UTF-8 request bodies should fail cleanly instead of crashing the app."""
    response = client.post(
        "/v1/responses",
        content=b"\xb5\x00\x01",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_parse_error"


def test_request_features_log_recognized_ignored_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Known ignored request fields should be summarized in the request log."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "ping",
            "parallel_tool_calls": False,
            "include": [],
            "store": False,
            "prompt_cache_key": "req-123",
        },
    )

    assert response.status_code == 200

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["request_features"] == {
        "recognized_ignored_fields": [
            "include",
            "parallel_tool_calls",
            "prompt_cache_key",
            "store",
        ],
        "count": 4,
    }


def test_invalid_upstream_tool_call_logs_failure_detail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Invalid upstream tool calls should preserve the proxy validation failure detail."""
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-3",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"extra":"no-path"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app.dependency_overrides[get_ollama_client] = lambda: fake
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "read it",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 422
    assert (
        response.json()["error"]["details"]["failure_detail"]
        == "upstream_selected_invalid_tool"
    )

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["failure_detail"] == "upstream_selected_invalid_tool"


def test_websocket_request_is_logged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Completed websocket requests should be written to the structured log."""
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json(
            {"type": "response.create", "input": "ping", "stream": True}
        )
        websocket.receive_text()
        websocket.receive_text()
        websocket.receive_text()

    log_path = request_log_path(tmp_path)
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    entry = entries[-1]
    assert entry["request_kind"] == "responses_websocket"
    assert entry["websocket_status"] == "completed"
    assert entry["failure_class"] is None
    assert entry["websocket_headers"]["sec-websocket-key"] == "[redacted]"
