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
        test_client.fake_ollama = fake  # type: ignore[attr-defined]
        yield test_client
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


def test_non_utf8_request_body_does_not_crash(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content=b"\xb5\x00\x01",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_parse_error"
