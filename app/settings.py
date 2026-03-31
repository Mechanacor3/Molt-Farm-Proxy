from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOLT_", extra="ignore")

    app_name: str = "Molt Farm Proxy"
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "nemotron-3-nano:4b"
    model_aliases_json: str = Field(default="{}")
    request_timeout_seconds: float = 120.0
    debug_payload_logging: bool = False
    debug_tool_names: str | None = None
    log_dir: str = ".molt-logs"

    @field_validator("ollama_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def model_aliases(self) -> dict[str, str]:
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
        return Path(self.log_dir).expanduser().resolve()

    def resolve_model(self, requested_model: str | None) -> str:
        if not requested_model:
            return self.default_model
        return self.model_aliases.get(requested_model, requested_model)

    @property
    def debug_tool_name_set(self) -> set[str] | None:
        if not self.debug_tool_names:
            return None
        names = {item.strip() for item in self.debug_tool_names.split(",") if item.strip()}
        return names or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
