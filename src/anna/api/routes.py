from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from anna.api.schemas import ChatCompletionRequest, CompletionRequest
from anna.runtime.engine import (
    AnnaEngineError,
    GenerationConfig,
    StreamEvent,
    TextGenerationResult,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _engine(request: Request):
    return request.app.state.engine


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def _chat_response_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    result: TextGenerationResult,
) -> dict:
    message = {"role": "assistant", "content": result.text}
    if result.reasoning_text is not None:
        message["reasoning_content"] = result.reasoning_text
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def _completion_response_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    result: TextGenerationResult,
) -> dict:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def _error_payload_from_exception(exc: AnnaEngineError) -> dict:
    return {
        "error": {
            "message": str(exc),
            "type": exc.error_type,
            "code": exc.code,
        }
    }


def _sse_error_frame(exc: AnnaEngineError) -> str:
    return f"event: error\ndata: {json.dumps(_error_payload_from_exception(exc), ensure_ascii=False)}\n\n"


def _stream_sse_chat(
    *,
    response_id: str,
    created: int,
    model: str,
    events: Iterator[StreamEvent],
) -> Iterator[str]:
    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    try:
        for event in events:
            delta: dict[str, str] = {}
            if event.reasoning_text:
                delta["reasoning_content"] = event.reasoning_text
            if event.text:
                delta["content"] = event.text
            payload = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": event.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except AnnaEngineError as exc:
        yield _sse_error_frame(exc)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled chat streaming failure.")
        yield _sse_error_frame(
            AnnaEngineError(
                "Streaming response failed.",
                status_code=500,
                error_type="server_error",
                code="streaming_failed",
            )
        )

    yield "data: [DONE]\n\n"


def _stream_sse_completion(
    *,
    response_id: str,
    created: int,
    model: str,
    events: Iterator[StreamEvent],
) -> Iterator[str]:
    try:
        for event in events:
            payload = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": event.text, "finish_reason": event.finish_reason}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except AnnaEngineError as exc:
        yield _sse_error_frame(exc)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled text streaming failure.")
        yield _sse_error_frame(
            AnnaEngineError(
                "Streaming response failed.",
                status_code=500,
                error_type="server_error",
                code="streaming_failed",
            )
        )

    yield "data: [DONE]\n\n"


@router.get("/healthz")
def healthz(request: Request) -> dict:
    return _engine(request).health()


@router.get("/v1/models")
def list_models(request: Request) -> dict:
    engine = _engine(request)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "anna",
            }
            for model_id in engine.list_models()
        ],
    }


@router.post("/v1/chat/completions")
def chat_completions(request: Request, payload: ChatCompletionRequest):
    engine = _engine(request)
    config = GenerationConfig(
        max_new_tokens=payload.max_completion_tokens or payload.max_tokens or 256,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
        stop_strings=_normalize_stop(payload.stop),
    )
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = payload.model or engine.default_model_id

    try:
        if payload.stream:
            events = engine.stream_chat(
                payload.messages,
                config=config,
                enable_thinking=bool(payload.enable_thinking),
            )
            return StreamingResponse(
                _stream_sse_chat(
                    response_id=response_id,
                    created=created,
                    model=model_id,
                    events=events,
                ),
                media_type="text/event-stream",
            )

        result = engine.generate_chat(
            payload.messages,
            config=config,
            enable_thinking=bool(payload.enable_thinking),
        )
    except AnnaEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return JSONResponse(
        _chat_response_payload(
            response_id=response_id,
            created=created,
            model=model_id,
            result=result,
        )
    )


@router.post("/v1/completions")
def completions(request: Request, payload: CompletionRequest):
    engine = _engine(request)
    config = GenerationConfig(
        max_new_tokens=payload.max_tokens or 256,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
        stop_strings=_normalize_stop(payload.stop),
    )
    response_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = payload.model or engine.default_model_id

    try:
        if payload.stream:
            events = engine.stream_text(payload.prompt, config=config)
            return StreamingResponse(
                _stream_sse_completion(
                    response_id=response_id,
                    created=created,
                    model=model_id,
                    events=events,
                ),
                media_type="text/event-stream",
            )

        result = engine.generate_text(payload.prompt, config=config)
    except AnnaEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return JSONResponse(
        _completion_response_payload(
            response_id=response_id,
            created=created,
            model=model_id,
            result=result,
        )
    )
