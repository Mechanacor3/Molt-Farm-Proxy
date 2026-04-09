# Molt Farm Proxy

FastAPI proxy that accepts OpenAI Responses API requests and translates them to
an upstream OpenAI-compatible chat-completions endpoint.

## Install `uv`

This project now uses `uv` for Python installation and environment management.

Install `uv` using Astral's standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you want a user-local install without shell profile edits, this also works:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
```

## End-User Setup

Provision Python 3.12 with `uv`, create the virtual environment, and install the
project dependencies:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev
```

The repo includes `.python-version` set to `3.12`, so `uv` will stay on the
expected interpreter family by default.

## Start The Proxy

Run the bridge-friendly proxy with the local model aliases:

```bash
uv run molt-proxy-dev --reload
```

By default this starts the FastAPI proxy on `http://127.0.0.1:8000`.

### Gemma On llama.cpp

To front a local authenticated `llama.cpp` server instead of Ollama, point the
proxy at the root server URL and provide the upstream bearer token:

```bash
env \
  MOLT_UPSTREAM_BASE_URL=http://127.0.0.1:8080 \
  MOLT_UPSTREAM_API_KEY=local-dev-key \
  uv run molt-proxy-dev --host 127.0.0.1 --port 8000 --upstream-model gemma-4-e4b
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/v1/responses
```

The proxy intentionally returns `405` from `GET /v1/responses`; that is the
expected preflight shape for a healthy Responses surface.

## Tight Dev Loop

Run the real Codex binary through the bridge and summarize what happened:

```bash
uv run codex-bridge exec --json "Reply with the single word pong."
uv run codex-bridge-report --limit 5
```

`codex-bridge` launches the exact VS Code-installed Codex binary with runtime
`openai_base_url` overrides pointing at this proxy. It leaves
`~/.codex/config.toml` unchanged and writes JSONL run logs to `.molt-logs/`.

`molt-proxy-dev` starts the FastAPI proxy with bridge-friendly model aliases so
default Codex model names such as `gpt-5.4` route to the configured local
upstream model.

## Continuation Tests

For a reliable multi-turn continuation test, capture the session id from the
first run's `thread.started` event and resume that exact session:

```bash
uv run codex-bridge exec --json "hi"
uv run codex-bridge exec resume <session-id> --json "thanks"
```

Prefer an explicit session id over `resume --last`. On a machine with other
Codex activity, `--last` can resume a different recent session and make the
result look flaky.

## Useful Commands

Run the tests:

```bash
uv run pytest
```

Probe function-calling without the full Codex client:

```bash
uv run python examples/get_weather_tool_probe.py
```

That script sends a single `get_weather` function tool to `POST /v1/responses`,
prints the first response, and if the model emits a function call it posts a
fake tool result back on a second turn.

To hit the upstream chat-completions server directly instead of the proxy:

```bash
uv run python examples/get_weather_tool_probe.py \
  --mode chat \
  --model gemma-4-e4b \
  --base-url http://127.0.0.1:8080 \
  --api-key local-dev-key
```

Probe the proxy-backed Responses path against the same Gemma server:

```bash
uv run python examples/get_weather_tool_probe.py \
  --mode responses \
  --model gemma-4-e4b \
  --base-url http://127.0.0.1:8000 \
  --api-key local-dev-key
```

Probe how far a narrower `exec_command` definition makes it through against
direct chat and proxy Responses:

```bash
uv run python examples/exec_command_capability_probe.py \
  --model gemma-4-e4b \
  --chat-base-url http://127.0.0.1:8080 \
  --proxy-base-url http://127.0.0.1:8000 \
  --api-key local-dev-key
```

`--surface chat` talks straight to the upstream chat endpoint. `--surface responses` and
`--surface both` expect the proxy to be running on `http://127.0.0.1:8000`.

For machine-readable pass/fail flags:

```bash
uv run python examples/exec_command_capability_probe.py \
  --model gemma-4-e4b \
  --chat-base-url http://127.0.0.1:8080 \
  --proxy-base-url http://127.0.0.1:8000 \
  --api-key local-dev-key \
  --json
```

See the agent/operator guide:

```bash
sed -n '1,220p' AGENTS.md
```
