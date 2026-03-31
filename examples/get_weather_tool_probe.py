from __future__ import annotations

import argparse
import json
from typing import Any

import httpx


def build_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Return the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["f", "c"]},
            },
            "required": ["city"],
        },
    }


def send_response_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    input_items: str | list[dict[str, Any]],
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/v1/responses",
        json={
            "model": model,
            "stream": False,
            "tool_choice": "auto",
            "input": input_items,
            "tools": [build_tool()],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def send_chat_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "stream": False,
            "tool_choice": "auto",
            "messages": messages,
            "tools": [{"type": "function", "function": build_tool()}],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def first_function_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    for item in payload.get("output", []):
        if item.get("type") == "function_call":
            return item
    return None


def first_chat_tool_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    return tool_calls[0]


def run_responses_mode(client: httpx.Client, base_url: str, model: str, prompt: str, tool_output: dict[str, Any]) -> None:
    first = send_response_request(client, base_url, model, prompt)
    print("=== First response ===")
    print(json.dumps(first, indent=2))

    call = first_function_call(first)
    if call is None:
        print("\nNo function call returned.")
        return

    print("\n=== Function call detected ===")
    print(f"name: {call['name']}")
    print(f"call_id: {call['call_id']}")
    print(f"arguments: {call['arguments']}")

    followup_input = [
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
            "output": tool_output,
        },
    ]
    second = send_response_request(client, base_url, model, followup_input)
    print("\n=== Second response with fake tool output ===")
    print(json.dumps(second, indent=2))


def run_chat_mode(client: httpx.Client, base_url: str, model: str, prompt: str, tool_output: dict[str, Any]) -> None:
    first = send_chat_request(
        client,
        base_url,
        model,
        [{"role": "user", "content": prompt}],
    )
    print("=== First response ===")
    print(json.dumps(first, indent=2))

    call = first_chat_tool_call(first)
    if call is None:
        print("\nNo function call returned.")
        return

    function = call.get("function") or {}
    print("\n=== Function call detected ===")
    print(f"name: {function.get('name')}")
    print(f"call_id: {call.get('id')}")
    print(f"arguments: {function.get('arguments')}")

    second = send_chat_request(
        client,
        base_url,
        model,
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_output),
            },
        ],
    )
    print("\n=== Second response with fake tool output ===")
    print(json.dumps(second, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the proxy with one get_weather function tool.")
    parser.add_argument(
        "--mode",
        choices=["responses", "chat"],
        default="responses",
        help="Use the proxy Responses endpoint or hit chat completions directly.",
    )
    parser.add_argument("--base-url", default=None, help="Base URL for the selected mode.")
    parser.add_argument("--model", default="codex-bridge", help="Model name to send in the Responses request.")
    parser.add_argument("--city", default="Boston", help="City to ask about.")
    parser.add_argument(
        "--prompt",
        default="What is the weather in {city}? Use the get_weather tool if you need external data.",
        help="Prompt template. {city} will be replaced.",
    )
    parser.add_argument(
        "--tool-json",
        default='{"temp_f": 61, "condition": "sunny"}',
        help="JSON payload to return if the model calls get_weather.",
    )
    args = parser.parse_args()

    prompt = args.prompt.format(city=args.city)
    tool_output = json.loads(args.tool_json)
    base_url = args.base_url or (
        "http://127.0.0.1:8000" if args.mode == "responses" else "http://127.0.0.1:11434"
    )

    with httpx.Client() as client:
        if args.mode == "responses":
            run_responses_mode(client, base_url, args.model, prompt, tool_output)
        else:
            run_chat_mode(client, base_url, args.model, prompt, tool_output)


if __name__ == "__main__":
    main()
