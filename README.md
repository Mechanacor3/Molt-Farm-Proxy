# Molt Farm Proxy

FastAPI proxy that accepts OpenAI Responses API requests and translates them to
Ollama chat completions.

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

Run the bridge-friendly proxy with the local Ollama-backed model aliases:

```bash
uv run molt-proxy-dev --reload
```

By default this starts the FastAPI proxy on `http://127.0.0.1:8000`.

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
default Codex model names such as `gpt-5.4` route to the local Ollama model.

## Useful Commands

Run the tests:

```bash
uv run pytest
```

See the agent/operator guide:

```bash
sed -n '1,220p' AGENTS.md
```
