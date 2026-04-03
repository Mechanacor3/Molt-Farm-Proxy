from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class ResponseInputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str
    role: str | None = None
    content: str | list[ResponseContentPart | dict[str, Any]] | None = None
    name: str | None = None
    call_id: str | None = None
    arguments: str | dict[str, Any] | None = None
    output: str | dict[str, Any] | list[Any] | None = None
    summary: list[dict[str, Any]] | None = None


class ResponseToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ResponseTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    function: ResponseToolFunction | None = None


class ResponseUsageDetails(BaseModel):
    input_tokens: int = 0
    input_tokens_details: dict[str, Any] = Field(
        default_factory=lambda: {"cached_tokens": 0}
    )
    output_tokens: int = 0
    output_tokens_details: dict[str, Any] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )
    total_tokens: int = 0


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[ResponseInputItem | dict[str, Any]]
    tools: list[ResponseTool] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    stop: str | list[str] | None = None


class ResponseOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    role: str | None = None
    content: list[dict[str, Any]] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    summary: list[dict[str, Any]] | None = None
    status: str | None = None


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: str
    status: str = "completed"
    model: str
    output: list[ResponseOutputItem]
    usage: ResponseUsageDetails

    @staticmethod
    def now_iso() -> str:
        """Return a Responses-compatible UTC timestamp for new proxy responses."""
        return datetime.now(timezone.utc).isoformat()
