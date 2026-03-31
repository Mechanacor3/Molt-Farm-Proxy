from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_ollama_client
from app.schemas_chat import ChatCompletionsRequest, ChatCompletionsResponse


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
    assert "event: response.created" in text
    assert "event: response.completed" in text
    assert "data: [DONE]" in text


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


def test_non_utf8_request_body_does_not_crash(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        content=b"\xb5\x00\x01",
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_validation_error"
