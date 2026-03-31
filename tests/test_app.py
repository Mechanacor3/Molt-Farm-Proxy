from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
import zstandard as zstd

from app.devloop import request_log_path
from app.main import app, get_ollama_client
from app.schemas_chat import ChatCompletionsRequest, ChatCompletionsResponse
from app.settings import get_settings


class FakeOllamaClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_request: ChatCompletionsRequest | None = None

    async def close(self) -> None:
        return None

    async def create_chat_completion(self, request: ChatCompletionsRequest) -> ChatCompletionsResponse:
        self.last_request = request
        return ChatCompletionsResponse.model_validate(self.payload)


@pytest.fixture
def client() -> TestClient:
    fake = FakeOllamaClient(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
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
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _zstd_json(payload: dict[str, Any]) -> bytes:
    return zstd.ZstdCompressor().compress(json.dumps(payload).encode("utf-8"))


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_non_streaming_response(client: TestClient) -> None:
    response = client.post("/v1/responses", json={"input": "ping", "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["text"] == "pong"


def test_streaming_response_returns_final_sse(client: TestClient) -> None:
    with client.stream("POST", "/v1/responses", json={"input": "ping", "stream": True}) as response:
        text = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())
    assert response.status_code == 200
    assert '"type":"response.created"' in text
    assert '"type":"response.completed"' in text
    assert "data: [DONE]" in text


def test_zstd_streaming_real_client_shape(client: TestClient) -> None:
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
        text = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

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
    response = client.post(
        "/v1/responses",
        json={"input": "ping", "tools": [{"type": "web_search_preview"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_tool"


def test_get_probe_returns_structured_transport_error(client: TestClient) -> None:
    response = client.get("/v1/responses")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "websocket_not_supported"


def test_websocket_response_create_streams_events(client: TestClient) -> None:
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json({"type": "response.create", "input": "ping", "stream": True})
        created = json.loads(websocket.receive_text())
        in_progress = json.loads(websocket.receive_text())
        completed = json.loads(websocket.receive_text())

    assert created["type"] == "response.created"
    assert in_progress["type"] == "response.in_progress"
    assert completed["type"] == "response.completed"
    assert completed["response"]["output"][0]["content"][0]["text"] == "pong"


def test_websocket_invalid_first_message_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_text("not-json")
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "request_parse_error"


def test_websocket_invalid_message_type_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json({"type": "response.cancel"})
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "unsupported_websocket_message"


def test_websocket_validation_error_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json({"type": "response.create", "input": {"unexpected": "shape"}})
        payload = json.loads(websocket.receive_text())

    assert payload["type"] == "error"
    assert payload["error"]["code"] == "request_validation_error"


def test_request_validation_error_is_structured(client: TestClient) -> None:
    response = client.post("/v1/responses", json={"input": {"unexpected": "shape"}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


def test_invalid_zstd_request_body_is_structured(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content=b"not-a-zstd-frame",
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_parse_error"


def test_zstd_request_validation_error_is_structured(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content=_zstd_json({"input": {"unexpected": "shape"}}),
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"


def test_failed_post_parsing_is_logged(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/v1/responses",
        content=b"not-a-zstd-frame",
        headers={"content-type": "application/json", "content-encoding": "zstd"},
    )

    assert response.status_code == 400

    log_path = request_log_path(tmp_path)
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["error_code"] == "request_parse_error"
    assert entries[-1]["failure_class"] == "schema_mismatch"
    assert entries[-1]["status_code"] == 400


def test_zstd_request_logs_decoded_tool_summary(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
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


def test_request_log_includes_upstream_tool_use_summary(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
                                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
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
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["upstream_summary"] == {
        "tool_count": 1,
        "tool_names": ["read_file"],
        "tool_choice": "auto",
        "finish_reason": "tool_calls",
        "response_has_tool_calls": True,
        "response_tool_call_names": ["read_file"],
        "response_content_present": False,
    }


def test_debug_tool_filter_reduces_forwarded_tool_names(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["upstream_summary"]["tool_count"] == 1
    assert entries[-1]["upstream_summary"]["tool_names"] == ["exec_command"]


def test_non_utf8_request_body_does_not_crash(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content=b"\xb5\x00\x01",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_parse_error"


def test_websocket_request_is_logged(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("MOLT_LOG_DIR", str(tmp_path))
    get_settings.cache_clear()

    with client.websocket_connect("/v1/responses") as websocket:
        websocket.send_json({"type": "response.create", "input": "ping", "stream": True})
        websocket.receive_text()
        websocket.receive_text()
        websocket.receive_text()

    log_path = request_log_path(tmp_path)
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    entry = entries[-1]
    assert entry["request_kind"] == "responses_websocket"
    assert entry["websocket_status"] == "completed"
    assert entry["failure_class"] is None
    assert entry["websocket_headers"]["sec-websocket-key"] == "[redacted]"
