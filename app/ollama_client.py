from __future__ import annotations

import httpx

from app.errors import ProxyError, UpstreamError
from app.schemas_chat import ChatCompletionsRequest, ChatCompletionsResponse
from app.settings import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def create_chat_completion(self, request: ChatCompletionsRequest) -> ChatCompletionsResponse:
        url = f"{self._settings.ollama_base_url}/v1/chat/completions"
        try:
            response = await self._client.post(url, json=request.model_dump(mode="json", exclude_none=True))
        except httpx.TimeoutException as exc:
            raise UpstreamError("Timed out waiting for Ollama.", status_code=504, code="upstream_timeout") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Failed to reach Ollama: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text.strip() or "Ollama returned an error."
            raise UpstreamError(detail, status_code=502, code="upstream_bad_status")

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Ollama returned malformed JSON.",
                status_code=502,
                code="invalid_upstream_payload",
            ) from exc
        try:
            return ChatCompletionsResponse.model_validate(payload)
        except Exception as exc:  # pragma: no cover - narrow pydantic exception not needed here
            raise ProxyError(502, "invalid_upstream_payload", f"Ollama payload validation failed: {exc}") from exc
