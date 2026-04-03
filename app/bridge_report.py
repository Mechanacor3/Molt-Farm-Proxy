from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.devloop import (
    load_active_run,
    request_log_path,
    resolve_log_dir,
    runs_log_path,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file defensively, skipping blank or malformed lines."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _sum_numeric(records: list[dict[str, Any]], key: str) -> int:
    """Sum the integer values for one key across the provided records."""
    total = 0
    for record in records:
        value = record.get(key)
        if isinstance(value, int):
            total += value
    return total


def main() -> None:
    """Print a compact summary of recent bridge runs and their proxy outcomes."""
    parser = argparse.ArgumentParser(description="Summarize recent codex-bridge runs.")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    log_dir = resolve_log_dir(args.log_dir)
    runs = _read_jsonl(runs_log_path(log_dir))
    requests = _read_jsonl(request_log_path(log_dir))
    requests_by_run: dict[str, list[dict[str, Any]]] = {}
    for record in requests:
        run_id = record.get("bridge_run_id") or "unscoped"
        requests_by_run.setdefault(run_id, []).append(record)

    finished = [record for record in runs if record.get("event") == "run_finished"]
    recent = finished[-args.limit :]
    if not recent:
        print("No bridge runs found.")
        active = load_active_run(log_dir)
        if active:
            print(f"Active run: {active['run_id']} ({active['mode']})")
        return

    for run in recent:
        run_id = str(run["run_id"])
        related = requests_by_run.get(run_id, [])
        failure_counts = Counter(
            record["failure_class"] for record in related if record.get("failure_class")
        )
        request_count = len(related)
        statuses = Counter(
            str(record.get("status_code"))
            for record in related
            if record.get("status_code") is not None
        )
        request_kinds = Counter(
            str(record.get("request_kind"))
            for record in related
            if record.get("request_kind")
        )
        failure_details = Counter(
            str(record.get("failure_detail"))
            for record in related
            if record.get("failure_detail")
        )
        tool_dispositions = Counter()
        for record in related:
            tool_diagnostics = record.get("tool_diagnostics")
            if not isinstance(tool_diagnostics, dict):
                continue
            counts = tool_diagnostics.get("counts")
            if not isinstance(counts, dict):
                continue
            for key, value in counts.items():
                if isinstance(value, int):
                    tool_dispositions[key] += value
        note = ""
        if (
            request_count
            and request_kinds.get("responses_http", 0) == 0
            and request_kinds.get("responses_websocket", 0) == 0
        ):
            note = " note=no_http_post_seen"
        print(
            f"{run_id} {run['status']} exit={run['exit_code']} "
            f"requests={request_count} failures={dict(failure_counts)} statuses={dict(statuses)} "
            f"kinds={dict(request_kinds)} failure_details={dict(failure_details)} "
            f"tools={{'observed': {_sum_numeric(related, 'tool_count_observed')}, "
            f"'forwarded': {_sum_numeric(related, 'tool_count_forwarded')}, "
            f"'ignored': {_sum_numeric(related, 'tool_count_ignored')}, "
            f"'rejected': {_sum_numeric(related, 'tool_count_rejected')}, "
            f"'dispositions': {dict(tool_dispositions)}}}{note}"
        )
