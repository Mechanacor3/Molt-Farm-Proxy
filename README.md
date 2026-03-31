# Molt Farm Proxy

FastAPI proxy that accepts OpenAI Responses API requests and translates them to
Ollama chat completions.

## Dev Loop

```bash
pip install -e ".[dev]"
molt-proxy-dev --reload
codex-bridge exec --json "Reply with the single word pong."
codex-bridge-report
```

`codex-bridge` launches the exact VS Code-installed Codex binary with runtime
`openai_base_url` overrides pointing at this proxy. It leaves
`~/.codex/config.toml` unchanged and writes JSONL run logs to `.molt-logs/`.

`molt-proxy-dev` starts the FastAPI proxy with bridge-friendly model aliases so
default Codex model names such as `gpt-5.4` route to the local Ollama model.
