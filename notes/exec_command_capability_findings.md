# exec_command Capability Findings

This note captures the current state after adding the narrow
`examples/exec_command_capability_probe.py` experiment and replaying the real
Codex CLI against the local bridge/proxy/Ollama stack.

## What We Confirmed

- Direct Ollama chat completions can call `exec_command` reliably with
  `nemotron-3-nano:4b`.
- The narrowed probe also succeeds through proxy `Responses` when the proxy is
  actually running.
- The model handles all three tested schema shapes:
  - `narrow`
  - `narrow_runtime`
  - `observed_full`
- The model handles both tested command families:
  - `pwd`
  - `sed -n '1p' README.md`
- The strict prompt variant did not outperform the plain prompt in the probe,
  because the plain prompt already worked.

## Important Correction

The earlier all-fail `responses` run in `probe.txt` was misleading. At the time
of that run, nothing was listening on `127.0.0.1:8000`, so every `responses`
case spent the full client timeout waiting on a dead endpoint.

The probe now does a fast preflight `GET /v1/responses` check before running any
`responses` cases. If the proxy is unavailable, it reports
`surface_preflight_failed: ...` immediately instead of waiting through the whole
matrix.

## What Still Fails

The real Codex CLI still does not reliably use tools with the local nano model,
even though the probe proves that:

- the proxy can translate tool calls correctly
- the model can emit tool calls correctly

The remaining problem appears when the full Codex harness prompt and tool
context are present.

Observed current behavior:

- Plain non-tool Codex prompts complete through the proxy.
- Tool-oriented Codex prompts also complete, but upstream often returns plain
  text with `finish_reason: "stop"` instead of structured `tool_calls`.
- This still happens even when the proxy forwards only `exec_command` via
  `MOLT_DEBUG_TOOL_NAMES=exec_command`.

## Current Read

The main blocker is no longer:

- `Responses` transport support
- the proxy's tool-call translation
- the `exec_command` schema itself

The main blocker is most likely the prompt and context shape produced by the
real Codex CLI for small local models.

## Best Next Steps

- Add a small-model mode that compacts or rewrites the Codex prompt stack before
  sending it upstream.
- Explore whether a provider-specific instruction shim can make the model emit
  tool calls more consistently under the full Codex harness.
- Keep using the narrow probe as the fast compatibility scoreboard before making
  app-level behavior changes.
