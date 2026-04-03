from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4

import zstandard as zstd
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect

from app.devloop import classify_proxy_failure, load_active_run, request_log_path
from app.errors import ProxyError
from app.jsonl import append_jsonl, utc_now_iso
from app.ollama_client import OllamaClient
from app.schemas_responses import ResponsesRequest, ResponsesResponse
from app.settings import Settings, get_settings
from app.tool_guard import summarize_request_ignored_fields
from app.translator import (
    build_response_events,
    build_sse_events,
    build_tool_compatibility,
    translate_chat_response_to_responses,
    translate_responses_request_to_chat,
)

logger = logging.getLogger("molt_farm_proxy")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create one shared Ollama client for the FastAPI app lifespan.

    This keeps connection setup centralized and makes test overrides simpler.
    """
    settings = get_settings()
    app.state.ollama_client = OllamaClient(settings)
    yield
    await app.state.ollama_client.close()


app = FastAPI(title="Molt Farm Proxy", lifespan=lifespan)


def get_ollama_client(request: Request) -> OllamaClient:
    """Expose the lifespan-managed Ollama client through FastAPI DI."""
    return request.app.state.ollama_client


def _request_log_payload(
    request: Request, response_status: int, settings: Settings
) -> dict[str, object]:
    """Assemble the structured JSONL payload for HTTP request logging.

    The proxy stores both transport details and the bridge-specific tool and
    failure annotations so the dev loop can classify what moved forward.
    """
    request_body = getattr(request.state, "parsed_json_body", None)
    error_code = getattr(request.state, "error_code", None)
    request_kind = getattr(request.state, "request_kind", None)
    payload = {
        "timestamp": utc_now_iso(),
        "request_id": getattr(request.state, "request_id", None),
        "bridge_run_id": getattr(request.state, "bridge_run_id", None),
        "method": request.method,
        "path": request.url.path,
        "status_code": response_status,
        "latency_ms": getattr(request.state, "latency_ms", None),
        "model": getattr(request.state, "downstream_model", None),
        "stream": getattr(request.state, "stream", None),
        "request_kind": request_kind
        or (
            "responses_websocket_probe"
            if request.method == "GET" and request.url.path == "/v1/responses"
            else "responses_http"
        ),
        "error_code": error_code,
        "failure_class": classify_proxy_failure(
            response_status, error_code, request.method, request_kind
        ),
    }
    tool_summary = getattr(request.state, "tool_summary", None)
    if tool_summary is not None:
        payload["tool_summary"] = tool_summary
    tool_diagnostics = getattr(request.state, "tool_diagnostics", None)
    if tool_diagnostics is not None:
        payload["tool_diagnostics"] = tool_diagnostics
        counts = tool_diagnostics.get("counts")
        if isinstance(counts, dict):
            payload["tool_count_observed"] = counts.get("observed")
            payload["tool_count_forwarded"] = counts.get("forwarded")
            payload["tool_count_ignored"] = counts.get("ignored")
            payload["tool_count_rejected"] = counts.get("rejected")
    failure_detail = getattr(request.state, "failure_detail", None)
    if failure_detail is not None:
        payload["failure_detail"] = failure_detail
    tool_call_diagnostics = getattr(request.state, "tool_call_diagnostics", None)
    if tool_call_diagnostics is not None:
        payload["tool_call_diagnostics"] = tool_call_diagnostics
    upstream_summary = getattr(request.state, "upstream_summary", None)
    if upstream_summary is not None:
        payload["upstream_summary"] = upstream_summary
    request_features = getattr(request.state, "request_features", None)
    if request_features is not None:
        payload["request_features"] = request_features
    if settings.debug_payload_logging and request_body is not None:
        payload["request_body"] = request_body
    return payload


def _websocket_log_payload(
    websocket: WebSocket, settings: Settings
) -> dict[str, object]:
    """Assemble the structured JSONL payload for websocket request logging.

    Websocket requests carry a slightly different state surface, so we mirror
    the HTTP log shape while preserving websocket-specific status fields.
    """
    request_body = getattr(websocket.state, "parsed_json_body", None)
    error_code = getattr(websocket.state, "error_code", None)
    payload = {
        "timestamp": utc_now_iso(),
        "request_id": getattr(websocket.state, "request_id", None),
        "bridge_run_id": getattr(websocket.state, "bridge_run_id", None),
        "method": "GET",
        "path": websocket.url.path,
        "status_code": getattr(websocket.state, "status_code", None),
        "latency_ms": getattr(websocket.state, "latency_ms", None),
        "model": getattr(websocket.state, "downstream_model", None),
        "stream": getattr(websocket.state, "stream", None),
        "request_kind": getattr(websocket.state, "request_kind", "responses_websocket"),
        "error_code": error_code,
        "failure_class": classify_proxy_failure(
            getattr(websocket.state, "status_code", None),
            error_code,
            "GET",
            getattr(websocket.state, "request_kind", "responses_websocket"),
        ),
        "websocket_status": getattr(websocket.state, "websocket_status", None),
    }
    tool_summary = getattr(websocket.state, "tool_summary", None)
    if tool_summary is not None:
        payload["tool_summary"] = tool_summary
    tool_diagnostics = getattr(websocket.state, "tool_diagnostics", None)
    if tool_diagnostics is not None:
        payload["tool_diagnostics"] = tool_diagnostics
    failure_detail = getattr(websocket.state, "failure_detail", None)
    if failure_detail is not None:
        payload["failure_detail"] = failure_detail
    tool_call_diagnostics = getattr(websocket.state, "tool_call_diagnostics", None)
    if tool_call_diagnostics is not None:
        payload["tool_call_diagnostics"] = tool_call_diagnostics
    upstream_summary = getattr(websocket.state, "upstream_summary", None)
    if upstream_summary is not None:
        payload["upstream_summary"] = upstream_summary
    request_features = getattr(websocket.state, "request_features", None)
    if request_features is not None:
        payload["request_features"] = request_features
    websocket_headers = getattr(websocket.state, "websocket_headers", None)
    if websocket_headers is not None:
        payload["websocket_headers"] = websocket_headers
    if settings.debug_payload_logging and request_body is not None:
        payload["request_body"] = request_body
    return payload


def _sanitize_for_json(value: Any) -> Any:
    """Convert exception payload fragments into JSON-safe values."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


def _summarize_tools(parsed: object) -> list[dict[str, object]] | None:
    """Capture a lightweight description of inbound tool definitions.

    The summary intentionally keeps only the fields that help correlate schema
    mismatches without storing the full request body by default.
    """
    if not isinstance(parsed, dict):
        return None

    raw_tools = parsed.get("tools")
    if not isinstance(raw_tools, list):
        return None

    summary: list[dict[str, object]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            summary.append({"raw_type": type(tool).__name__})
            continue

        function = tool.get("function")
        function_name = function.get("name") if isinstance(function, dict) else None
        parameters = (
            function.get("parameters")
            if isinstance(function, dict)
            else tool.get("parameters")
        )
        required = parameters.get("required") if isinstance(parameters, dict) else None
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        property_names = (
            sorted(properties.keys()) if isinstance(properties, dict) else None
        )

        summary.append(
            {
                "type": tool.get("type"),
                "name": tool.get("name"),
                "function_name": function_name,
                "external_web_access": tool.get("external_web_access"),
                "has_function": isinstance(function, dict),
                "parameter_type": parameters.get("type")
                if isinstance(parameters, dict)
                else None,
                "property_names": property_names,
                "required": required if isinstance(required, list) else None,
            }
        )
    return summary


def _redact_websocket_headers(headers: list[tuple[str, str]]) -> dict[str, str]:
    """Lowercase websocket headers and redact the sensitive probe values."""
    sensitive = {"authorization", "cookie", "sec-websocket-key"}
    return {
        key.lower(): ("[redacted]" if key.lower() in sensitive else value)
        for key, value in headers
    }


def _decode_request_body(raw_body: bytes, content_encoding: str | None) -> bytes:
    """Decode supported request body encodings in the declared order.

    Real Codex HTTP fallback requests currently arrive as zstd-compressed JSON,
    so this helper peels those layers before schema validation and rejects
    encodings we do not intentionally support.
    """
    if not content_encoding:
        return raw_body

    encodings = [
        token.strip().lower() for token in content_encoding.split(",") if token.strip()
    ]
    decoded = raw_body
    for encoding in encodings:
        if encoding == "identity":
            continue
        if encoding == "zstd":
            try:
                with zstd.ZstdDecompressor().stream_reader(BytesIO(decoded)) as reader:
                    decoded = reader.read()
            except zstd.ZstdError as exc:
                raise ProxyError(
                    400,
                    "request_parse_error",
                    "There was an error parsing the body",
                ) from exc
            continue
        raise ProxyError(
            400,
            "request_parse_error",
            f"Unsupported Content-Encoding '{encoding}'.",
        )
    return decoded


async def _parse_responses_request(request: Request) -> ResponsesRequest:
    """Decode, summarize, and validate a POST `/v1/responses` payload."""
    raw_body = await request.body()
    decoded_body = _decode_request_body(
        raw_body, request.headers.get("content-encoding")
    )

    try:
        parsed = json.loads(decoded_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProxyError(
            400, "request_parse_error", "There was an error parsing the body"
        ) from exc

    request.state.parsed_json_body = parsed
    if isinstance(parsed, dict):
        request.state.downstream_model = parsed.get("model")
        request.state.stream = parsed.get("stream")
    request.state.tool_summary = _summarize_tools(parsed)
    request.state.request_features = summarize_request_ignored_fields(parsed)

    try:
        return ResponsesRequest.model_validate(parsed)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _set_state_from_payload(state: Any, parsed: object) -> None:
    """Populate request or websocket state from a parsed Responses payload."""
    state.parsed_json_body = parsed
    state.downstream_model = parsed.get("model") if isinstance(parsed, dict) else None
    state.stream = parsed.get("stream") if isinstance(parsed, dict) else None
    state.tool_summary = _summarize_tools(parsed)
    state.request_features = summarize_request_ignored_fields(parsed)


def _parse_websocket_responses_request(payload: object) -> ResponsesRequest:
    """Validate the first websocket frame as a Responses `response.create` request."""
    if not isinstance(payload, dict):
        raise ProxyError(
            400,
            "request_validation_error",
            "Request body did not match the expected Responses API schema.",
        )

    if payload.get("type") != "response.create":
        raise ProxyError(
            400,
            "unsupported_websocket_message",
            "Unsupported websocket message type. Expected 'response.create'.",
        )

    request_payload = {key: value for key, value in payload.items() if key != "type"}
    try:
        return ResponsesRequest.model_validate(request_payload)
    except ValidationError as exc:
        raise ProxyError(
            400,
            "request_validation_error",
            "Request body did not match the expected Responses API schema.",
        ) from exc


async def _execute_responses_request(
    responses_request: ResponsesRequest,
    state: Any,
    settings: Settings,
    ollama_client: OllamaClient,
) -> ResponsesResponse:
    """Run the full proxy pipeline from Responses request to normalized output.

    The algorithm is: classify and filter tools, translate the request into the
    upstream chat shape, call Ollama, then validate or repair any returned tool
    calls before translating the result back into Responses semantics.
    """
    compatibility = build_tool_compatibility(
        responses_request.tools,
        settings,
        require_forwardable=bool(responses_request.tools),
    )
    state.tool_diagnostics = (
        compatibility.as_log_payload() if responses_request.tools else None
    )
    upstream_request = translate_responses_request_to_chat(
        responses_request, settings, compatibility
    )
    state.upstream_summary = {
        "tool_count": len(upstream_request.tools or []),
        "tool_names": [tool.function.name for tool in (upstream_request.tools or [])],
        "tool_choice": upstream_request.tool_choice,
    }
    started = perf_counter()
    chat_response = await ollama_client.create_chat_completion(upstream_request)
    choice = chat_response.choices[0] if chat_response.choices else None
    if choice is not None:
        state.upstream_summary = {
            **state.upstream_summary,
            "finish_reason": choice.finish_reason,
            "response_has_tool_calls": bool(choice.message.tool_calls),
            "response_tool_call_names": [
                tool_call.function.name
                for tool_call in (choice.message.tool_calls or [])
            ],
            "response_content_present": bool(choice.message.content),
        }
    translated, tool_call_validation = translate_chat_response_to_responses(
        responses_request, chat_response
    )
    state.upstream_summary = {
        **state.upstream_summary,
        "tool_call_validation": tool_call_validation.diagnostics,
    }
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    logger.info(
        "completed request",
        extra={
            "downstream_model": responses_request.model or settings.default_model,
            "upstream_model": chat_response.model,
            "stream": responses_request.stream,
            "latency_ms": elapsed_ms,
        },
    )
    return translated


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach per-request state, then append a structured log after the response."""
    settings = get_settings()
    request.state.request_id = uuid4().hex
    request.state.bridge_run_id = None
    active = load_active_run(settings.resolved_log_dir)
    if active:
        request.state.bridge_run_id = active.get("run_id")

    request.state.parsed_json_body = None
    request.state.downstream_model = None
    request.state.stream = None
    request.state.error_code = None
    request.state.failure_detail = None
    request.state.tool_call_diagnostics = None
    request.state.tool_summary = None
    request.state.tool_diagnostics = None
    request.state.upstream_summary = None
    request.state.request_features = None
    request.state.request_kind = (
        "responses_http"
        if request.method == "POST" and request.url.path == "/v1/responses"
        else None
    )

    if request.url.path == "/v1/responses":
        raw_body = await request.body()
        if raw_body:
            try:
                parsed = json.loads(raw_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            _set_state_from_payload(request.state, parsed)

    started = perf_counter()
    response = await call_next(request)
    request.state.latency_ms = round((perf_counter() - started) * 1000, 2)
    response.headers["x-molt-request-id"] = request.state.request_id
    if request.url.path == "/v1/responses":
        append_jsonl(
            request_log_path(settings.resolved_log_dir),
            _request_log_payload(request, response.status_code, settings),
        )
    return response


@app.exception_handler(ProxyError)
async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
    """Translate internal proxy errors into the structured error response shape."""
    request.state.error_code = exc.code
    request.state.failure_detail = exc.details.get("failure_detail")
    if exc.details.get("tool_diagnostics") is not None:
        request.state.tool_diagnostics = exc.details["tool_diagnostics"]
    if exc.details.get("tool_call_diagnostics") is not None:
        request.state.tool_call_diagnostics = exc.details["tool_call_diagnostics"]
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "proxy_error",
                "code": exc.code,
                **({"details": exc.details} if exc.details else {}),
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize pydantic request validation failures into proxy error JSON."""
    request.state.error_code = "request_validation_error"
    request.state.failure_detail = "request_schema_validation"
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Request body did not match the expected Responses API schema.",
                "type": "invalid_request_error",
                "code": "request_validation_error",
                "details": _sanitize_for_json(exc.errors()),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Rewrite low-level parse failures on the Responses route into proxy errors."""
    if request.url.path == "/v1/responses" and exc.status_code == 400:
        request.state.error_code = "request_parse_error"
        request.state.failure_detail = "request_parse_error"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error",
                    "code": "request_parse_error",
                }
            },
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/healthz")
async def healthz(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Report a minimal liveness payload plus the current default model."""
    return {"status": "ok", "default_model": settings.default_model}


@app.get("/v1/responses", response_model=None)
async def responses_get_probe() -> JSONResponse:
    """Return the structured transport error used for websocket probe traffic."""
    raise ProxyError(
        status_code=405,
        code="websocket_not_supported",
        message="This proxy does not implement the Responses websocket transport. Use HTTP POST /v1/responses.",
    )


@app.websocket("/v1/responses")
async def responses_websocket(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
):
    """Handle the limited websocket handshake path the real client probes first.

    We accept the socket, require an initial `response.create` JSON frame, run
    the same translation pipeline as HTTP, and stream typed Responses events
    back over the socket before logging the full interaction.
    """
    websocket.state.request_id = uuid4().hex
    websocket.state.bridge_run_id = None
    active = load_active_run(settings.resolved_log_dir)
    if active:
        websocket.state.bridge_run_id = active.get("run_id")
    websocket.state.parsed_json_body = None
    websocket.state.downstream_model = None
    websocket.state.stream = None
    websocket.state.error_code = None
    websocket.state.failure_detail = None
    websocket.state.tool_call_diagnostics = None
    websocket.state.tool_summary = None
    websocket.state.tool_diagnostics = None
    websocket.state.upstream_summary = None
    websocket.state.request_features = None
    websocket.state.request_kind = "responses_websocket"
    websocket.state.status_code = 101
    websocket.state.websocket_status = "accepted"
    websocket.state.websocket_headers = _redact_websocket_headers(
        list(websocket.headers.items())
    )

    started = perf_counter()
    await websocket.accept()

    try:
        message = await websocket.receive()
        if message.get("type") != "websocket.receive":
            raise ProxyError(
                400, "websocket_protocol_error", "Expected a websocket message."
            )
        text = message.get("text")
        if text is None:
            raise ProxyError(
                400,
                "websocket_protocol_error",
                "Expected the initial websocket message to be text JSON.",
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProxyError(
                400, "request_parse_error", "There was an error parsing the body"
            ) from exc

        _set_state_from_payload(websocket.state, parsed)
        responses_request = _parse_websocket_responses_request(parsed)
        translated = await _execute_responses_request(
            responses_request,
            websocket.state,
            settings,
            websocket.app.state.ollama_client,
        )

        for event in build_response_events(translated):
            await websocket.send_text(json.dumps(event, separators=(",", ":")))

        websocket.state.status_code = 200
        websocket.state.websocket_status = "completed"
        await websocket.close()
    except WebSocketDisconnect:
        websocket.state.status_code = 499
        websocket.state.error_code = "websocket_client_disconnected"
        websocket.state.websocket_status = "client_disconnected"
    except ProxyError as exc:
        websocket.state.status_code = exc.status_code
        websocket.state.error_code = exc.code
        websocket.state.failure_detail = exc.details.get("failure_detail")
        if exc.details.get("tool_diagnostics") is not None:
            websocket.state.tool_diagnostics = exc.details["tool_diagnostics"]
        if exc.details.get("tool_call_diagnostics") is not None:
            websocket.state.tool_call_diagnostics = exc.details["tool_call_diagnostics"]
        websocket.state.websocket_status = "proxy_error"
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": exc.message,
                        "type": "proxy_error",
                        "code": exc.code,
                        **({"details": exc.details} if exc.details else {}),
                    },
                },
                separators=(",", ":"),
            )
        )
        await websocket.close(code=1008)
    finally:
        websocket.state.latency_ms = round((perf_counter() - started) * 1000, 2)
        append_jsonl(
            request_log_path(settings.resolved_log_dir),
            _websocket_log_payload(websocket, settings),
        )


@app.post("/v1/responses", response_model=None)
async def create_response(
    request: Request,
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Handle the HTTP Responses endpoint and optionally emit SSE events."""
    responses_request = await _parse_responses_request(request)
    translated = await _execute_responses_request(
        responses_request, request.state, settings, ollama_client
    )

    if responses_request.stream:

        async def event_stream() -> AsyncIterator[str]:
            """Yield typed Responses SSE frames in the order Codex expects."""
            for event in build_sse_events(translated):
                yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return JSONResponse(content=translated.model_dump(mode="json"))
