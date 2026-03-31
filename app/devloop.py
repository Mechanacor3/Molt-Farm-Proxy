from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.jsonl import append_jsonl, utc_now_iso

DEFAULT_CODEX_BINARY = (
    "/home/user/.vscode-server/extensions/"
    "openai.chatgpt-26.325.31654-linux-x64/bin/linux-x86_64/codex"
)
DEFAULT_PROXY_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_BRIDGE_MODEL = "codex-bridge"
DEFAULT_LOG_DIR = ".molt-logs"
DEFAULT_ALIAS_MAP = {
    "codex-bridge": "nemotron-3-nano:4b",
    "gpt-5.4": "nemotron-3-nano:4b",
    "gpt-5.3-codex": "nemotron-3-nano:4b",
}


def resolve_codex_binary(explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get("CODEX_BRIDGE_BINARY") or DEFAULT_CODEX_BINARY
    return str(Path(candidate).expanduser())


def resolve_log_dir(explicit: str | None = None) -> Path:
    raw = explicit or os.environ.get("CODEX_BRIDGE_LOG_DIR") or DEFAULT_LOG_DIR
    return Path(raw).expanduser().resolve()


def generate_run_id(explicit: str | None = None) -> str:
    return explicit or f"bridge-{uuid4().hex[:12]}"


def state_file(log_dir: Path) -> Path:
    return log_dir / "active-run.json"


def runs_log_path(log_dir: Path) -> Path:
    return log_dir / "bridge-runs.jsonl"


def request_log_path(log_dir: Path) -> Path:
    return log_dir / "proxy-requests.jsonl"


def load_active_run(log_dir: Path) -> dict[str, Any] | None:
    path = state_file(log_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_active_run(log_dir: Path, payload: dict[str, Any]) -> None:
    path = state_file(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def clear_active_run(log_dir: Path, run_id: str) -> None:
    path = state_file(log_dir)
    current = load_active_run(log_dir)
    if current and current.get("run_id") == run_id and path.exists():
        path.unlink()


def build_model_aliases_json(explicit_json: str | None = None, upstream_model: str | None = None) -> str:
    if explicit_json:
        return explicit_json
    aliases = dict(DEFAULT_ALIAS_MAP)
    if upstream_model:
        aliases["codex-bridge"] = upstream_model
        aliases["gpt-5.4"] = upstream_model
        aliases["gpt-5.3-codex"] = upstream_model
    return json.dumps(aliases, ensure_ascii=True, separators=(",", ":"))


def classify_proxy_failure(
    status_code: int | None,
    error_code: str | None,
    method: str,
    request_kind: str | None = None,
) -> str | None:
    if request_kind == "responses_websocket":
        return None if error_code is None else "transport_failure"
    if method == "GET":
        return "transport_failure"
    if error_code == "unsupported_tool":
        return "unsupported_tool"
    if error_code in {"invalid_input", "unsupported_input_item", "request_validation_error", "request_parse_error"}:
        return "schema_mismatch"
    if error_code and error_code.startswith("upstream_"):
        return "upstream_ollama_failure"
    if error_code in {"invalid_tool_call", "ambiguous_tool_call", "unexpected_tool_call"}:
        return "proxy_validation_failure"
    if status_code and status_code >= 500:
        return "proxy_validation_failure"
    if status_code and status_code >= 400:
        return "schema_mismatch"
    return None


def log_bridge_event(log_dir: Path, payload: dict[str, Any]) -> None:
    append_jsonl(runs_log_path(log_dir), payload)


def base_run_payload(run_id: str, mode: str, model: str, endpoint: str, binary: str) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "endpoint": endpoint,
        "binary": binary,
    }
