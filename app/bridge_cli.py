from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time

from app.devloop import (
    DEFAULT_BRIDGE_MODEL,
    DEFAULT_PROXY_BASE_URL,
    base_run_payload,
    clear_active_run,
    generate_run_id,
    log_bridge_event,
    resolve_codex_binary,
    resolve_log_dir,
    write_active_run,
)


def _parser(default_mode: str) -> argparse.ArgumentParser:
    """Build the CLI parser shared by the generic and VS Code bridge entrypoints."""
    parser = argparse.ArgumentParser(
        description="Launch the real Codex binary against the local bridge."
    )
    parser.add_argument("--binary", help="Override the Codex binary path.")
    parser.add_argument(
        "--proxy-base-url",
        default=os.environ.get("CODEX_BRIDGE_BASE_URL", DEFAULT_PROXY_BASE_URL),
    )
    parser.add_argument(
        "--model", default=os.environ.get("CODEX_BRIDGE_MODEL", DEFAULT_BRIDGE_MODEL)
    )
    parser.add_argument("--log-dir", default=os.environ.get("CODEX_BRIDGE_LOG_DIR"))
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default=default_mode)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("codex_args", nargs=argparse.REMAINDER)
    return parser


def build_codex_command(
    binary: str, proxy_base_url: str, model: str, codex_args: list[str]
) -> list[str]:
    """Inject bridge overrides into the real Codex command line.

    The bridge only adds the runtime base URL and a fallback model flag when
    the caller did not already pin a model explicitly.
    """
    forwarded = list(codex_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    command = [binary]
    has_model = False
    for token in forwarded:
        if token in {"-m", "--model"}:
            has_model = True

    command.extend(["-c", f'openai_base_url="{proxy_base_url}"'])

    if not has_model:
        command.extend(["-m", model])

    command.extend(forwarded)
    return command


def _run(default_mode: str) -> int:
    """Run one bridged Codex invocation and log its start and finish events."""
    args = _parser(default_mode).parse_args()
    binary = resolve_codex_binary(args.binary)
    log_dir = resolve_log_dir(args.log_dir)
    run_id = generate_run_id(args.run_id)
    command = build_codex_command(
        binary, args.proxy_base_url, args.model, args.codex_args
    )

    if args.dry_run:
        print(shlex.join(command))
        return 0

    started = time.perf_counter()
    start_payload = base_run_payload(
        run_id, args.mode, args.model, args.proxy_base_url, binary
    )
    start_payload.update({"event": "run_started", "argv": command[1:]})
    log_bridge_event(log_dir, start_payload)
    write_active_run(
        log_dir,
        {
            "run_id": run_id,
            "started_at": start_payload["timestamp"],
            "mode": args.mode,
            "model": args.model,
            "endpoint": args.proxy_base_url,
        },
    )

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "codex-bridge")
    env["CODEX_BRIDGE_RUN_ID"] = run_id
    env["CODEX_BRIDGE_LOG_DIR"] = str(log_dir)

    try:
        completed = subprocess.run(command, env=env, check=False)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        finish_payload = base_run_payload(
            run_id, args.mode, args.model, args.proxy_base_url, binary
        )
        finish_payload.update(
            {
                "event": "run_finished",
                "exit_code": completed.returncode,
                "duration_ms": duration_ms,
                "status": "success" if completed.returncode == 0 else "failed",
            }
        )
        log_bridge_event(log_dir, finish_payload)
        return completed.returncode
    finally:
        clear_active_run(log_dir, run_id)


def main() -> None:
    """Launch the bridge in the default CLI mode."""
    raise SystemExit(_run("cli"))


def main_vscode() -> None:
    """Launch the bridge with the VS Code flavored mode label."""
    raise SystemExit(_run("vscode-cli"))
