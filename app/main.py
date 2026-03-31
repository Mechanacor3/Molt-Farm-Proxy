from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import ProxyError
from app.devloop import classify_proxy_failure, load_active_run, request_log_path
from app.jsonl import append_jsonl, utc_now_iso
from app.ollama_client import OllamaClient
from app.schemas_responses import ResponsesRequest
from app.settings import Settings, get_settings
from app.translator import build_sse_events, translate_chat_response_to_responses, translate_responses_request_to_chat

logger = logging.getLogger("molt_farm_proxy")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.ollama_client = OllamaClient(settings)
    yield
    await app.state.ollama_client.close()


app = FastAPI(title="Molt Farm Proxy", lifespan=lifespan)


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def _request_log_payload(request: Request, response_status: int, settings: Settings) -> dict[str, object]:
    request_body = getattr(request.state, "parsed_json_body", None)
    error_code = getattr(request.state, "error_code", None)
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
        "request_kind": "responses_websocket_probe"
        if request.method == "GET" and request.url.path == "/v1/responses"
        else "responses_http",
        "error_code": error_code,
        "failure_class": classify_proxy_failure(response_status, error_code, request.method),
    }
    if settings.debug_payload_logging and request_body is not None:
        payload["request_body"] = request_body
    return payload


def _sanitize_for_json(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
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

    if request.url.path == "/v1/responses":
        raw_body = await request.body()
        if raw_body:
            try:
                parsed = json.loads(raw_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            request.state.parsed_json_body = parsed
            if isinstance(parsed, dict):
                request.state.downstream_model = parsed.get("model")
                request.state.stream = parsed.get("stream")

    started = perf_counter()
    response = await call_next(request)
    request.state.latency_ms = round((perf_counter() - started) * 1000, 2)
    response.headers["x-molt-request-id"] = request.state.request_id
    if request.url.path == "/v1/responses":
        append_jsonl(request_log_path(settings.resolved_log_dir), _request_log_payload(request, response.status_code, settings))
    return response


@app.exception_handler(ProxyError)
async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
    request.state.error_code = exc.code
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.message, "type": "proxy_error", "code": exc.code}},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request.state.error_code = "request_validation_error"
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
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if request.url.path == "/v1/responses" and exc.status_code == 400:
        request.state.error_code = "request_parse_error"
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
    return {"status": "ok", "default_model": settings.default_model}


@app.get("/v1/responses", response_model=None)
async def responses_get_probe() -> JSONResponse:
    raise ProxyError(
        status_code=405,
        code="websocket_not_supported",
        message="This proxy does not implement the Responses websocket transport. Use HTTP POST /v1/responses.",
    )


@app.post("/v1/responses", response_model=None)
async def create_response(
    request: ResponsesRequest,
    settings: Settings = Depends(get_settings),
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    upstream_request = translate_responses_request_to_chat(request, settings)
    started = perf_counter()
    chat_response = await ollama_client.create_chat_completion(upstream_request)
    translated = translate_chat_response_to_responses(request, chat_response)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    logger.info(
        "completed request",
        extra={
            "downstream_model": request.model or settings.default_model,
            "upstream_model": chat_response.model,
            "stream": request.stream,
            "latency_ms": elapsed_ms,
        },
    )

    if request.stream:
        async def event_stream() -> AsyncIterator[str]:
            for event in build_sse_events(translated):
                yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return JSONResponse(content=translated.model_dump(mode="json"))
