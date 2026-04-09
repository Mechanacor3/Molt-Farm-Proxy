from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from typing import Any

import httpx

OBSERVED_FULL_EXEC_SPEC: dict[str, Any] = {
    "type": "function",
    "name": "exec_command",
    "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to execute."},
            "justification": {
                "type": "string",
                "description": "Short summary of why the command is needed.",
            },
            "login": {
                "type": "boolean",
                "description": "Whether to run the shell with -l/-i semantics.",
            },
            "max_output_tokens": {
                "type": "integer",
                "description": "Maximum number of tokens to return.",
            },
            "prefix_rule": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Suggested command prefix for similar future requests.",
            },
            "sandbox_permissions": {
                "type": "string",
                "description": "Sandbox permission level for the command.",
            },
            "shell": {"type": "string", "description": "Shell binary to launch."},
            "tty": {"type": "boolean", "description": "Whether to allocate a TTY."},
            "workdir": {
                "type": "string",
                "description": "Working directory to run in.",
            },
            "yield_time_ms": {
                "type": "integer",
                "description": "How long to wait before yielding output.",
            },
        },
        "required": ["cmd"],
    },
}

PLAIN_PROMPT_TEMPLATE = "Use the exec_command tool to run {command} and then reply with only the resulting output."
STRICT_ONE_TOOL_PROMPT_TEMPLATE = (
    "Do not answer from memory. You have exactly one available tool: exec_command. "
    "You must call exec_command to run this exact command: {command}. "
    "After the tool returns, reply with the exact stdout and nothing else."
)


@dataclass(frozen=True)
class ProbeCase:
    name: str
    command_family: str
    command: str
    fake_stdout: str
    tool_preset: str
    prompt_preset: str


def build_tool_presets() -> dict[str, dict[str, Any]]:
    """Return the intentional exec_command schema variants used by the probe."""
    return {
        "narrow": {
            "type": "function",
            "name": "exec_command",
            "description": "Run one shell command and return its stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                },
                "required": ["cmd"],
            },
        },
        "narrow_runtime": {
            "type": "function",
            "name": "exec_command",
            "description": "Run one shell command and return its stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory to run in.",
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "description": "How long to wait before yielding output.",
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to return.",
                    },
                },
                "required": ["cmd"],
            },
        },
        "observed_full": copy.deepcopy(OBSERVED_FULL_EXEC_SPEC),
    }


def build_case_matrix() -> list[ProbeCase]:
    """Build the fixed matrix of tool, prompt, and command probe cases."""
    return [
        ProbeCase(
            name="pwd_narrow_plain",
            command_family="pwd",
            command="pwd",
            fake_stdout="/probe/pwd",
            tool_preset="narrow",
            prompt_preset="plain",
        ),
        ProbeCase(
            name="pwd_narrow_strict",
            command_family="pwd",
            command="pwd",
            fake_stdout="/probe/pwd",
            tool_preset="narrow",
            prompt_preset="strict_one_tool",
        ),
        ProbeCase(
            name="pwd_narrow_runtime_plain",
            command_family="pwd",
            command="pwd",
            fake_stdout="/probe/pwd",
            tool_preset="narrow_runtime",
            prompt_preset="plain",
        ),
        ProbeCase(
            name="pwd_observed_full_plain",
            command_family="pwd",
            command="pwd",
            fake_stdout="/probe/pwd",
            tool_preset="observed_full",
            prompt_preset="plain",
        ),
        ProbeCase(
            name="readme_narrow_plain",
            command_family="readme",
            command="sed -n '1p' README.md",
            fake_stdout="# Molt Farm Proxy",
            tool_preset="narrow",
            prompt_preset="plain",
        ),
        ProbeCase(
            name="readme_narrow_strict",
            command_family="readme",
            command="sed -n '1p' README.md",
            fake_stdout="# Molt Farm Proxy",
            tool_preset="narrow",
            prompt_preset="strict_one_tool",
        ),
        ProbeCase(
            name="readme_narrow_runtime_plain",
            command_family="readme",
            command="sed -n '1p' README.md",
            fake_stdout="# Molt Farm Proxy",
            tool_preset="narrow_runtime",
            prompt_preset="plain",
        ),
        ProbeCase(
            name="readme_observed_full_strict",
            command_family="readme",
            command="sed -n '1p' README.md",
            fake_stdout="# Molt Farm Proxy",
            tool_preset="observed_full",
            prompt_preset="strict_one_tool",
        ),
    ]


CASES = build_case_matrix()
CASES_BY_NAME = {case.name: case for case in CASES}


def build_prompt(case: ProbeCase) -> str:
    """Render the prompt text for one probe case."""
    if case.prompt_preset == "strict_one_tool":
        return STRICT_ONE_TOOL_PROMPT_TEMPLATE.format(command=case.command)
    return PLAIN_PROMPT_TEMPLATE.format(command=case.command)


def responses_tool_output_value(tool_output: str) -> str:
    """Keep fake tool output as-is for the Responses follow-up shape."""
    return tool_output


def build_chat_tool(tool_spec: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Responses-style tool definition in the chat-completions shape."""
    return {
        "type": "function",
        "function": {
            "name": tool_spec["name"],
            "description": tool_spec.get("description"),
            "parameters": copy.deepcopy(tool_spec.get("parameters") or {}),
        },
    }


def send_responses_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    tool_spec: dict[str, Any],
    input_items: str | list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send one non-streaming probe request to the proxy Responses endpoint."""
    response = client.post(
        f"{base_url}/v1/responses",
        headers=build_request_headers(api_key),
        json={
            "model": model,
            "stream": False,
            "tool_choice": "auto",
            "input": input_items,
            "tools": [tool_spec],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def send_chat_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    tool_spec: dict[str, Any],
    messages: list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send one non-streaming probe request to the upstream chat endpoint."""
    response = client.post(
        f"{base_url}/v1/chat/completions",
        headers=build_request_headers(api_key),
        json={
            "model": model,
            "stream": False,
            "tool_choice": "auto",
            "messages": messages,
            "tools": [build_chat_tool(tool_spec)],
        },
        timeout=90.0,
    )
    response.raise_for_status()
    return response.json()


def build_request_headers(api_key: str | None) -> dict[str, str]:
    """Build the optional auth header for authenticated direct or proxied probes."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_responses_function_call(payload: dict[str, Any]) -> dict[str, str] | None:
    """Return the first function call emitted in a Responses payload, if any."""
    for item in payload.get("output", []):
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if (
            isinstance(call_id, str)
            and isinstance(name, str)
            and isinstance(arguments, str)
        ):
            return {"call_id": call_id, "name": name, "arguments": arguments}
    return None


def extract_chat_function_call(payload: dict[str, Any]) -> dict[str, str] | None:
    """Return the first function call emitted in a chat-completions payload."""
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    call = tool_calls[0]
    function = call.get("function") or {}
    call_id = call.get("id")
    name = function.get("name")
    arguments = function.get("arguments")
    if (
        isinstance(call_id, str)
        and isinstance(name, str)
        and isinstance(arguments, str)
    ):
        return {"call_id": call_id, "name": name, "arguments": arguments}
    return None


def build_responses_followup_input(
    prompt: str, call: dict[str, str], tool_output: str
) -> list[dict[str, Any]]:
    """Build the second-turn Responses input that feeds back fake tool output."""
    return [
        {"type": "message", "role": "user", "content": prompt},
        {
            "type": "function_call",
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": call["arguments"],
        },
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": responses_tool_output_value(tool_output),
        },
    ]


def build_chat_followup_messages(
    prompt: str, call: dict[str, str], tool_output: str
) -> list[dict[str, Any]]:
    """Build the chat-completions follow-up message sequence for one tool call."""
    return [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call["call_id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call["call_id"],
            "content": tool_output,
        },
    ]


def extract_chat_finish_reason(payload: dict[str, Any]) -> str | None:
    """Return the first chat choice finish reason when present."""
    choices = payload.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def run_case(
    client: httpx.Client,
    *,
    surface: str,
    model: str,
    base_url: str,
    case: ProbeCase,
    api_key: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute one probe case and capture normalized success diagnostics.

    The flow issues the first request, looks for an `exec_command` call, then
    optionally runs the fake-tool follow-up turn and records enough detail to
    compare behavior across prompt and schema variants.
    """
    tool_spec = copy.deepcopy(build_tool_presets()[case.tool_preset])
    prompt = build_prompt(case)
    result: dict[str, Any] = {
        "surface": surface,
        "case_name": case.name,
        "command_family": case.command_family,
        "command": case.command,
        "fake_stdout": case.fake_stdout,
        "tool_preset": case.tool_preset,
        "prompt_preset": case.prompt_preset,
        "call_detected": False,
        "call_name": None,
        "call_arguments": None,
        "followup_attempted": False,
        "followup_success": False,
        "case_success": False,
        "first_finish_reason": None,
        "error": None,
    }
    try:
        if surface == "chat":
            first = send_chat_request(
                client,
                base_url,
                model,
                tool_spec,
                [{"role": "user", "content": prompt}],
                api_key,
            )
            call = extract_chat_function_call(first)
            result["first_finish_reason"] = extract_chat_finish_reason(first)
            if verbose:
                result["first_response"] = first
            if call is not None:
                result["call_detected"] = call["name"] == "exec_command"
                result["call_name"] = call["name"]
                result["call_arguments"] = call["arguments"]
            result["case_success"] = result["call_detected"]
            if result["call_detected"]:
                result["followup_attempted"] = True
                try:
                    second = send_chat_request(
                        client,
                        base_url,
                        model,
                        tool_spec,
                        build_chat_followup_messages(prompt, call, case.fake_stdout),
                        api_key,
                    )
                    result["followup_success"] = True
                    if verbose:
                        result["second_response"] = second
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text.strip()
                    if len(detail) > 240:
                        detail = detail[:237] + "..."
                    result["error"] = f"http_{exc.response.status_code}: {detail}"
            return result

        first = send_responses_request(
            client,
            base_url,
            model,
            tool_spec,
            prompt,
            api_key,
        )
        call = extract_responses_function_call(first)
        if verbose:
            result["first_response"] = first
        if call is not None:
            result["call_detected"] = call["name"] == "exec_command"
            result["call_name"] = call["name"]
            result["call_arguments"] = call["arguments"]
        if result["call_detected"]:
            result["followup_attempted"] = True
            second = send_responses_request(
                client,
                base_url,
                model,
                tool_spec,
                build_responses_followup_input(prompt, call, case.fake_stdout),
                api_key,
            )
            result["followup_success"] = True
            if verbose:
                result["second_response"] = second
        result["case_success"] = result["call_detected"] and result["followup_success"]
        return result
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        if len(detail) > 240:
            detail = detail[:237] + "..."
        result["error"] = f"http_{exc.response.status_code}: {detail}"
    except Exception as exc:  # pragma: no cover
        result["error"] = str(exc)
    return result


def summarize_flags(results: list[dict[str, Any]]) -> dict[str, bool]:
    """Collapse detailed probe results into a few high-level capability flags."""
    by_surface_and_case = {
        (item["surface"], item["case_name"]): item for item in results
    }

    def succeeded(surface: str, case_name: str) -> bool:
        """Check whether one surface/case pair completed successfully."""
        record = by_surface_and_case.get((surface, case_name))
        return bool(record and record.get("case_success"))

    strict_prompt_helps = False
    surfaces = sorted({item["surface"] for item in results})
    for surface in surfaces:
        for family in ("pwd", "readme"):
            if succeeded(surface, f"{family}_narrow_strict") and not succeeded(
                surface, f"{family}_narrow_plain"
            ):
                strict_prompt_helps = True

    return {
        "minimal_pwd_ok": any(
            item["case_name"] == "pwd_narrow_plain" and item["case_success"]
            for item in results
        ),
        "strict_prompt_helps": strict_prompt_helps,
        "runtime_fields_ok": any(
            item["tool_preset"] == "narrow_runtime" and item["case_success"]
            for item in results
        ),
        "full_schema_ok": any(
            item["tool_preset"] == "observed_full" and item["case_success"]
            for item in results
        ),
        "file_read_via_exec_ok": any(
            item["command_family"] == "readme" and item["case_success"]
            for item in results
        ),
        "responses_end_to_end_ok": any(
            item["surface"] == "responses" and item["followup_success"]
            for item in results
        ),
    }


def render_human_summary(results: list[dict[str, Any]], flags: dict[str, bool]) -> str:
    """Render the probe output as a compact human-readable report."""
    lines = ["=== exec_command capability probe ==="]
    for item in results:
        status = "PASS" if item["case_success"] else "FAIL"
        lines.append(
            " ".join(
                [
                    f"{status:4}",
                    f"surface={item['surface']}",
                    f"case={item['case_name']}",
                    f"tool_call={'yes' if item['call_detected'] else 'no'}",
                    f"followup={'yes' if item['followup_success'] else 'no'}",
                ]
            )
        )
        if item.get("error"):
            lines.append(f"  error={item['error']}")
    lines.append("")
    lines.append("=== capability flags ===")
    for key, value in flags.items():
        lines.append(f"{key}={'yes' if value else 'no'}")
    return "\n".join(lines)


def resolve_cases(case_name: str) -> list[ProbeCase]:
    """Resolve the CLI case selector into a list of concrete probe cases."""
    if case_name == "all":
        return CASES
    return [CASES_BY_NAME[case_name]]


def resolve_surfaces(surface: str) -> list[str]:
    """Resolve the CLI surface selector into one or both probe surfaces."""
    if surface == "both":
        return ["chat", "responses"]
    return [surface]


def build_surface_unavailable_results(
    surface: str,
    cases: list[ProbeCase],
    error: str,
) -> list[dict[str, Any]]:
    """Mark every requested case as failed when a surface preflight blocks all runs."""
    return [
        {
            "surface": surface,
            "case_name": case.name,
            "command_family": case.command_family,
            "command": case.command,
            "fake_stdout": case.fake_stdout,
            "tool_preset": case.tool_preset,
            "prompt_preset": case.prompt_preset,
            "call_detected": False,
            "call_name": None,
            "call_arguments": None,
            "followup_attempted": False,
            "followup_success": False,
            "case_success": False,
            "first_finish_reason": None,
            "error": error,
        }
        for case in cases
    ]


def preflight_surface(client: httpx.Client, surface: str, base_url: str) -> str | None:
    """Check whether a surface is reachable before running the full case matrix.

    Responses mode expects the proxy's websocket probe behavior specifically,
    so we require the known `405` GET response before spending time on cases.
    """
    if surface != "responses":
        return None

    try:
        response = client.get(f"{base_url}/v1/responses", timeout=5.0)
    except httpx.TimeoutException:
        return "surface_preflight_failed: timed out contacting responses endpoint"
    except httpx.RequestError as exc:
        return f"surface_preflight_failed: {exc}"

    if response.status_code == 405:
        return None

    return f"surface_preflight_failed: expected 405 from GET /v1/responses, got {response.status_code}"


def case_by_name_keys() -> list[str]:
    """Return sorted case names for stable CLI choice ordering."""
    return sorted(CASES_BY_NAME)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser so tests can lock the public probe flags."""
    parser = argparse.ArgumentParser(
        description="Probe narrow exec_command tool variants against direct chat and the proxy."
    )
    parser.add_argument(
        "--surface", choices=["chat", "responses", "both"], default="both"
    )
    parser.add_argument(
        "--model",
        default="nemotron-3-nano:4b",
        help="Model name to send for each request.",
    )
    parser.add_argument(
        "--chat-base-url",
        "--ollama-base-url",
        dest="chat_base_url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--proxy-base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token for authenticated local or proxied backends.",
    )
    parser.add_argument("--case", choices=["all", *case_by_name_keys()], default="all")
    parser.add_argument(
        "--json", action="store_true", help="Print the full results as JSON."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include raw response payloads in the results.",
    )
    return parser


def main() -> None:
    """Run the selected probe cases and print JSON or a human summary."""
    parser = build_parser()
    args = parser.parse_args()

    selected_cases = resolve_cases(args.case)
    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for surface in resolve_surfaces(args.surface):
            base_url = (
                args.chat_base_url if surface == "chat" else args.proxy_base_url
            )
            preflight_error = preflight_surface(client, surface, base_url)
            if preflight_error is not None:
                results.extend(
                    build_surface_unavailable_results(
                        surface, selected_cases, preflight_error
                    )
                )
                continue
            for case in selected_cases:
                results.append(
                    run_case(
                        client,
                        surface=surface,
                        model=args.model,
                        base_url=base_url,
                        case=case,
                        api_key=args.api_key,
                        verbose=args.verbose,
                    )
                )

    flags = summarize_flags(results)
    payload = {
        "model": args.model,
        "surface": args.surface,
        "selected_case": args.case,
        "results": results,
        "flags": flags,
        "cases": [asdict(case) for case in selected_cases],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(render_human_summary(results, flags))


if __name__ == "__main__":
    main()
