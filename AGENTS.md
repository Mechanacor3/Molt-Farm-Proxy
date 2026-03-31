# AGENTS

## Purpose

This repo is a local compatibility harness for running the real Codex CLI
against a local bridge and Ollama backend.

The current stack is:

1. Real Codex binary from the VS Code extension
2. `codex-bridge`, which injects runtime endpoint and model overrides
3. FastAPI proxy in this repo, exposing `POST /v1/responses`
4. Ollama upstream via `POST /v1/chat/completions`

The goal is not just to proxy requests. The goal is to observe what the real
Codex client actually sends, classify failures, and tighten compatibility in a
fast loop.

## Important Commands

Install the local package and dev dependencies:

```bash
pip install -e ".[dev]"
```

Start the proxy in dev mode:

```bash
molt-proxy-dev --reload
```

Run the real Codex binary through the bridge:

```bash
codex-bridge exec --json "Reply with the single word pong."
```

Use the VS Code-flavored alias if you want the intent to be explicit:

```bash
vscode-codex-bridge exec --json "Reply with the single word pong."
```

Summarize recent bridge runs:

```bash
codex-bridge-report --limit 5
```

Run tests:

```bash
python3 -m pytest
```

## How The Bridge Works

`codex-bridge` is intentionally thin.

- It resolves the actual Codex binary path. By default this is the Codex binary
  bundled with the local VS Code extension.
- It does not edit `~/.codex/config.toml`.
- It injects runtime overrides with Codex `-c` flags, currently centered on
  `openai_base_url`.
- It sets a bridge run id and log directory in the environment so the proxy can
  correlate requests back to the exact Codex invocation.
- It writes JSONL session records to `.molt-logs/bridge-runs.jsonl`.

The proxy writes correlated request records to `.molt-logs/proxy-requests.jsonl`.

## Current Proxy Behavior

The proxy currently:

- Accepts `POST /v1/responses`
- Returns a structured `405` on `GET /v1/responses` for websocket probe traffic
- Translates supported Responses requests into Ollama chat completions
- Logs request kind, status code, latency, run id, and failure class

Known current behavior with real Codex:

- Codex first probes `GET /v1/responses` expecting websocket-style behavior
- Codex then falls back to HTTP
- The HTTP fallback body is not yet fully compatible with the proxy’s expected
  Responses schema, so the loop currently records a schema mismatch

This is expected. The bridge exists to make those failures concrete.

## Tight Improvement Loop

Use this loop when improving compatibility:

1. Start the proxy with `molt-proxy-dev --reload`
2. Run a real Codex prompt through `codex-bridge`
3. Inspect `codex-bridge-report`
4. If needed, inspect the raw JSONL logs in `.molt-logs/`
5. Update the proxy to support the next real request shape or failure mode
6. Add or update tests that lock the new behavior
7. Repeat with the same Codex prompt until the failure class changes or clears

Treat the report output as the primary scoreboard:

- `transport_failure` means Codex never reached a useful HTTP request shape
- `schema_mismatch` means Codex reached the proxy but the body shape does not
  match what we accept yet
- `unsupported_tool` means the request is understood but uses a tool type we
  intentionally reject
- `upstream_ollama_failure` means the bridge and proxy worked, but Ollama failed
- `proxy_validation_failure` means the proxy accepted the request but rejected
  the model/tool response during normalization

## What Good Iteration Looks Like

A strong feature-improvement pass should:

- reproduce the issue with `codex-bridge`
- identify the exact failing request kind from `.molt-logs/proxy-requests.jsonl`
- make the smallest proxy change that moves the real client forward
- add a regression test for that exact shape
- rerun the real bridge flow to confirm the failure class moved forward

Do not optimize for hypothetical compatibility first. Optimize for the next real
Codex request we can already observe.

## Files To Start With

- `app/main.py` for request handling, error mapping, and request logging
- `app/translator.py` for Responses-to-chat translation
- `app/tool_guard.py` for tool schema validation and repair
- `app/bridge_cli.py` for runtime Codex launch behavior
- `app/bridge_report.py` for dev-loop summaries
- `app/devloop.py` for run ids, paths, defaults, and failure classification

## Environment Knobs

These are the main bridge/proxy knobs:

- `CODEX_BRIDGE_BINARY`
- `CODEX_BRIDGE_BASE_URL`
- `CODEX_BRIDGE_MODEL`
- `CODEX_BRIDGE_LOG_DIR`
- `MOLT_DEFAULT_MODEL`
- `MOLT_MODEL_ALIASES_JSON`
- `MOLT_LOG_DIR`

Defaults are chosen so the local VS Code Codex binary talks to the local proxy
and the proxy maps common Codex-facing model names onto the local Ollama model.
