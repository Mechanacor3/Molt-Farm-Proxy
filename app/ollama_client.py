from __future__ import annotations

import httpx

from app.errors import ProxyError, UpstreamError
from app.schemas_chat import ChatCompletionsRequest, ChatCompletionsResponse
from app.settings import Settings


class ChatUpstreamClient:
    def __init__(self, settings: Settings) -> None:
        """Create the shared async HTTP client used for upstream chat calls."""
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def close(self) -> None:
        """Close the shared async HTTP client during app shutdown."""
        await self._client.aclose()

    async def create_chat_completion(
        self,
        request: ChatCompletionsRequest,
        *,
        authorization: str | None = None,
    ) -> ChatCompletionsResponse:
        """Send one translated chat completion request to the upstream API.

        This keeps transport failures, bad status codes, malformed JSON, and
        schema mismatches mapped into the proxy's structured upstream errors.
        """
        url = _chat_completions_url(self._settings.upstream_base_url)
        try:
            response = await self._client.post(
                url,
                json=request.model_dump(mode="json", exclude_none=True),
                headers=_authorization_headers(
                    upstream_api_key=self._settings.upstream_api_key,
                    incoming_authorization=authorization,
                ),
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Timed out waiting for the upstream chat completion endpoint.",
                status_code=504,
                code="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Failed to reach the upstream chat endpoint: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text.strip() or "The upstream chat endpoint returned an error."
            raise UpstreamError(detail, status_code=502, code="upstream_bad_status")

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "The upstream chat endpoint returned malformed JSON.",
                status_code=502,
                code="invalid_upstream_payload",
            ) from exc
        try:
            return ChatCompletionsResponse.model_validate(payload)
        except (
            Exception
        ) as exc:  # pragma: no cover - narrow pydantic exception not needed here
            raise ProxyError(
                502,
                "invalid_upstream_payload",
                f"Upstream chat payload validation failed: {exc}",
            ) from exc


def _chat_completions_url(base_url: str) -> str:
    """Resolve the upstream chat-completions URL from a root or `/v1` base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _authorization_headers(
    *,
    upstream_api_key: str | None,
    incoming_authorization: str | None,
) -> dict[str, str] | None:
    """Build the optional Authorization header for upstream requests."""
    if upstream_api_key:
        return {"Authorization": f"Bearer {upstream_api_key}"}
    if incoming_authorization:
        return {"Authorization": incoming_authorization}
    return None


OllamaClient = ChatUpstreamClient
