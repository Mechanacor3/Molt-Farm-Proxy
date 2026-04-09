from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOLT_", extra="ignore")

    app_name: str = "Molt Farm Proxy"
    upstream_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices(
            "MOLT_UPSTREAM_BASE_URL",
            "MOLT_OLLAMA_BASE_URL",
        ),
    )
    upstream_api_key: str | None = None
    default_model: str = "nemotron-3-nano:4b"
    model_aliases_json: str = Field(default="{}")
    request_timeout_seconds: float = 120.0
    debug_payload_logging: bool = False
    debug_tool_names: str | None = None
    log_dir: str = ".molt-logs"

    @field_validator("upstream_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        """Normalize the Ollama base URL so request joins stay predictable."""
        return value.rstrip("/")

    @field_validator("upstream_api_key")
    @classmethod
    def blank_api_key_to_none(cls, value: str | None) -> str | None:
        """Treat blank upstream keys as unset so auth forwarding can take over."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @property
    def model_aliases(self) -> dict[str, str]:
        """Decode and validate the configured model alias mapping."""
        try:
            data = json.loads(self.model_aliases_json)
        except json.JSONDecodeError as exc:
            raise ValueError("MOLT_MODEL_ALIASES_JSON must be valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("MOLT_MODEL_ALIASES_JSON must decode to an object")
        aliases: dict[str, str] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Model aliases must be string to string mappings")
            aliases[key] = value
        return aliases

    @property
    def resolved_log_dir(self) -> Path:
        """Resolve the configured log directory into an absolute path."""
        return Path(self.log_dir).expanduser().resolve()

    def resolve_model(self, requested_model: str | None) -> str:
        """Map a Codex-facing model name onto the concrete upstream model."""
        if not requested_model:
            return self.default_model
        return self.model_aliases.get(requested_model, requested_model)

    @property
    def debug_tool_name_set(self) -> set[str] | None:
        """Parse the optional debug allowlist of function tool names."""
        if not self.debug_tool_names:
            return None
        names = {
            item.strip() for item in self.debug_tool_names.split(",") if item.strip()
        }
        return names or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cache settings so the app and tests share one resolved configuration."""
    return Settings()
