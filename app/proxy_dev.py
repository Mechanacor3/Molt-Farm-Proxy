from __future__ import annotations

import argparse
import os
import subprocess
import sys

from app.devloop import DEFAULT_LOG_DIR, build_model_aliases_json


def main() -> None:
    """Start uvicorn with the dev-loop defaults wired into the environment."""
    parser = argparse.ArgumentParser(
        description="Start the proxy with dev-loop defaults."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--upstream-model",
        default=os.environ.get("MOLT_DEFAULT_MODEL", "nemotron-3-nano:4b"),
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MOLT_LOG_DIR", DEFAULT_LOG_DIR)
    env.setdefault(
        "MOLT_MODEL_ALIASES_JSON",
        build_model_aliases_json(upstream_model=args.upstream_model),
    )
    env.setdefault("MOLT_DEFAULT_MODEL", args.upstream_model)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        args.port,
    ]
    if args.reload:
        command.append("--reload")
    raise SystemExit(subprocess.run(command, env=env, check=False).returncode)
