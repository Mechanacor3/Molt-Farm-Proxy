from __future__ import annotations

from app.schemas_chat import ChatCompletionsResponse
from app.schemas_responses import ResponsesRequest
from app.settings import Settings
from app.translator import translate_chat_response_to_responses, translate_responses_request_to_chat


def test_message_only_request_translates_to_chat() -> None:
    request = ResponsesRequest(model="alias-model", input=[{"type": "message", "role": "user", "content": "hello"}])
    settings = Settings(default_model="nemotron-3-nano:4b", model_aliases_json='{"alias-model":"real-model"}')
    translated = translate_responses_request_to_chat(request, settings)
    assert translated.model == "real-model"
    assert translated.messages[0].role == "user"
    assert translated.messages[0].content == "hello"
    assert translated.stream is False


def test_reasoning_and_tool_result_translate_to_chat() -> None:
    request = ResponsesRequest(
        input=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Think carefully"}]},
            {"type": "message", "role": "user", "content": "Weather?"},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Boston"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": {"temp_f": 70}},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    translated = translate_responses_request_to_chat(request, Settings())
    assert translated.messages[0].role == "system"
    assert "Think carefully" in (translated.messages[0].content or "")
    assert translated.messages[1].role == "user"
    assert translated.messages[2].tool_calls is not None
    assert translated.messages[3].role == "tool"
    assert translated.tools is not None


def test_tool_list_translates_for_ollama_and_ignores_disabled_web_search() -> None:
    request = ResponsesRequest(
        input="tool?",
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
            {
                "type": "web_search",
                "external_web_access": False,
            },
        ],
    )

    translated = translate_responses_request_to_chat(request, Settings())

    assert translated.tool_choice == "auto"
    assert translated.tools is not None
    assert [tool.function.name for tool in translated.tools] == ["read_file", "exec_command"]
    assert translated.tools[0].function.parameters["required"] == ["path"]
    assert translated.tools[1].function.parameters["required"] == ["cmd"]


def test_debug_tool_name_filter_keeps_only_named_tool() -> None:
    request = ResponsesRequest(
        input="tool?",
        tools=[
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
            {
                "type": "function",
                "name": "write_stdin",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type": "integer"}},
                    "required": ["session_id"],
                },
            },
        ],
    )

    translated = translate_responses_request_to_chat(
        request,
        Settings(debug_tool_names="exec_command"),
    )

    assert translated.tools is not None
    assert [tool.function.name for tool in translated.tools] == ["exec_command"]


def test_chat_text_response_translates_to_responses() -> None:
    request = ResponsesRequest(input="hi")
    chat_response = ChatCompletionsResponse.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    translated = translate_chat_response_to_responses(request, chat_response)
    assert translated.object == "response"
    assert translated.output[0].type == "message"
    assert translated.output[0].content[0]["text"] == "hello"
    assert translated.usage.total_tokens == 15


def test_chat_tool_call_translates_to_responses() -> None:
    request = ResponsesRequest(
        input="tool?",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    chat_response = ChatCompletionsResponse.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1,
            "model": "nemotron-3-nano:4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Boston"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    translated = translate_chat_response_to_responses(request, chat_response)
    assert translated.output[0].type == "function_call"
    assert translated.output[0].name == "get_weather"
