from __future__ import annotations


class ProxyError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        """Carry the structured error payload used by the HTTP and websocket layers."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class UnsupportedToolError(ProxyError):
    def __init__(self, tool_type: str) -> None:
        """Create the standard unsupported-tool error for unknown Responses tools."""
        super().__init__(
            status_code=400,
            code="unsupported_tool",
            message=(
                f"Unsupported Responses tool type '{tool_type}'. "
                "This proxy only forwards function tools that can be represented safely "
                "over chat completions."
            ),
        )


class UpstreamError(ProxyError):
    def __init__(
        self, message: str, status_code: int = 502, code: str = "upstream_error"
    ) -> None:
        """Wrap upstream Ollama failures in the proxy's structured error type."""
        super().__init__(status_code=status_code, code=code, message=message)
