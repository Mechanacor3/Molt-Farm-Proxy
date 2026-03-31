from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatTool(BaseModel):
    type: Literal["function"]
    function: ChatFunctionDefinition


class ChatToolFunctionCall(BaseModel):
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ChatToolFunctionCall
    index: int | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    tools: list[ChatTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage | None = None
    system_fingerprint: str | None = None
