from __future__ import annotations

import json
import logging
import io
import time
import uuid
from typing import Iterator

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from anna.api.schemas import ChatCompletionRequest, CompletionRequest, SpeechRequest
from anna.runtime.qwen3_5_text_engine import (
    AnnaEngineError,
    GenerationConfig,
    ReasoningFormat,
    StreamEvent,
    TextGenerationResult,
    normalize_reasoning_format,
)
from anna.runtime.qwen3_tts_engine import Qwen3TTSSynthesisConfig

router = APIRouter()
logger = logging.getLogger(__name__)
_MISSING = object()


def _engine(request: Request):
    return request.app.state.engine


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _memory_snapshot(engine: object):
    device_context = getattr(engine, "device_context", None)
    if device_context is None or not hasattr(device_context, "get_memory_info"):
        return None
    return device_context.get_memory_info()


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def _default_max_completion_tokens(engine: object) -> int | None:
    value = getattr(engine, "default_max_completion_tokens", None)
    return None if value is None else max(1, int(value))


def _default_reasoning_format(engine: object) -> ReasoningFormat:
    value = getattr(engine, "reasoning_format", None)
    return normalize_reasoning_format(value)


def _default_enable_thinking(engine: object) -> bool:
    return getattr(engine, "default_enable_thinking", True) is not False


def _stream_usage_requested(payload: ChatCompletionRequest | CompletionRequest) -> bool:
    stream_options = getattr(payload, "stream_options", None)
    return bool(getattr(stream_options, "include_usage", False) or getattr(payload, "stream_include_usage", False))


def _require_method(engine: object, method_name: str, *, message: str, code: str):
    if not hasattr(engine, method_name):
        raise AnnaEngineError(message, code=code)
    return getattr(engine, method_name)


def _model_family_name(engine: object) -> str:
    value = getattr(engine, "model_family", None)
    if isinstance(value, str) and value:
        return value
    return "unknown_model_family"


def _resolve_enable_thinking(engine: object, payload: ChatCompletionRequest) -> bool:
    if payload.enable_thinking is not None:
        return payload.enable_thinking
    if payload.chat_template_kwargs is not None and payload.chat_template_kwargs.enable_thinking is not None:
        return payload.chat_template_kwargs.enable_thinking
    return _default_enable_thinking(engine)


def _resolve_reasoning_format(engine: object, payload: ChatCompletionRequest) -> ReasoningFormat:
    return normalize_reasoning_format(payload.reasoning_format) if payload.reasoning_format else _default_reasoning_format(engine)


def _log_generation_result(
    *,
    route_name: str,
    model: str,
    result: TextGenerationResult,
    elapsed_seconds: float,
    memory_before,
    memory_after,
) -> None:
    perf = result.perf
    if perf is None:
        total_tokens_per_second = 0.0 if elapsed_seconds <= 0 else result.completion_tokens / elapsed_seconds
        logger.info(
            "%s model=%s prompt_tokens=%s completion_tokens=%s total_seconds=%.3f total_tokens_per_second=%.2f xpu_free_before=%s xpu_free_after=%s",
            route_name,
            model,
            result.prompt_tokens,
            result.completion_tokens,
            elapsed_seconds,
            total_tokens_per_second,
            _format_bytes(None if memory_before is None else memory_before.free_bytes),
            _format_bytes(None if memory_after is None else memory_after.free_bytes),
        )
        return

    logger.info(
        "%s model=%s prompt_tokens=%s completion_tokens=%s prefill_seconds=%.3f prefill_tokens_per_second=%.2f ttft_seconds=%.3f decode_seconds=%.3f decode_tokens=%s decode_tokens_per_second=%.2f total_seconds=%.3f total_tokens_per_second=%.2f xpu_free_before=%s xpu_free_after=%s xpu_allocated_after=%s xpu_reserved_after=%s",
        route_name,
        model,
        result.prompt_tokens,
        result.completion_tokens,
        perf.prefill_seconds,
        perf.prefill_tokens_per_second,
        perf.ttft_seconds,
        perf.decode_seconds,
        perf.decode_tokens,
        perf.decode_tokens_per_second,
        perf.total_seconds,
        perf.total_tokens_per_second,
        _format_bytes(None if memory_before is None else memory_before.free_bytes),
        _format_bytes(None if memory_after is None else memory_after.free_bytes),
        _format_bytes(None if memory_after is None else memory_after.allocated_bytes),
        _format_bytes(None if memory_after is None else memory_after.reserved_bytes),
    )


def _chat_response_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    result: TextGenerationResult,
) -> dict:
    message: dict[str, object] = {
        "role": "assistant",
        "content": None if result.tool_calls and not result.text else result.text,
    }
    if result.reasoning_text is not None:
        message["reasoning_content"] = result.reasoning_text
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
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


def _usage_payload(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
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
        "usage": _usage_payload(result.prompt_tokens, result.completion_tokens),
    }


def _encode_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    scaled = np.rint(clipped * 32767.0).astype(np.int16)
    return scaled.tobytes()


def _encode_audio_bytes(audio: np.ndarray, *, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    if response_format == "pcm":
        return _encode_pcm16(audio), "audio/pcm"

    buffer = io.BytesIO()
    if response_format == "flac":
        sf.write(buffer, audio, sample_rate, format="FLAC", subtype="PCM_16")
        return buffer.getvalue(), "audio/flac"

    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), "audio/wav"


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


def _chat_chunk_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    content: str,
    reasoning: str,
    tool_calls: list[dict[str, object]] | None,
    finish_reason: str | None,
    usage: dict[str, int] | None | object = _MISSING,
) -> dict:
    delta: dict[str, object] = {}
    if reasoning:
        delta["reasoning_content"] = reasoning
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not _MISSING:
        payload["usage"] = usage
    return payload


def _stream_sse_chat(
    *,
    response_id: str,
    created: int,
    model: str,
    events: Iterator[StreamEvent],
    include_usage: bool,
) -> Iterator[str]:
    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    if include_usage:
        role_chunk["usage"] = None
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    final_usage = None
    try:
        for event in events:
            if not event.text and not event.reasoning_text and not event.tool_calls and event.finish_reason is None:
                continue
            if event.finish_reason is not None and event.prompt_tokens is not None and event.completion_tokens is not None:
                final_usage = _usage_payload(event.prompt_tokens, event.completion_tokens)
            payload = _chat_chunk_payload(
                response_id=response_id,
                created=created,
                model=model,
                content=event.text or "",
                reasoning=event.reasoning_text or "",
                tool_calls=None if event.tool_calls is None else [tool_call.to_openai_dict() for tool_call in event.tool_calls],
                finish_reason=event.finish_reason,
                usage=None if include_usage else _MISSING,
            )
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

    if include_usage and final_usage is not None:
        usage_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": final_usage,
        }
        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def _stream_sse_completion(
    *,
    response_id: str,
    created: int,
    model: str,
    events: Iterator[StreamEvent],
    include_usage: bool,
) -> Iterator[str]:
    final_usage = None
    try:
        for event in events:
            payload = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": event.text, "finish_reason": event.finish_reason}],
            }
            if include_usage:
                payload["usage"] = None
            if event.finish_reason is not None and event.prompt_tokens is not None and event.completion_tokens is not None:
                final_usage = _usage_payload(event.prompt_tokens, event.completion_tokens)
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

    if include_usage and final_usage is not None:
        usage_chunk = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [],
            "usage": final_usage,
        }
        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

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
        max_new_tokens=payload.max_completion_tokens or payload.max_tokens or _default_max_completion_tokens(engine),
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
        stop_strings=_normalize_stop(payload.stop),
    )
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = payload.model or engine.default_model_id
    enable_thinking = _resolve_enable_thinking(engine, payload)
    reasoning_format = _resolve_reasoning_format(engine, payload)
    memory_before = _memory_snapshot(engine)
    started_at = time.perf_counter()

    try:
        if payload.stream:
            include_usage = _stream_usage_requested(payload)
            stream_chat = _require_method(
                engine,
                "stream_chat",
                message=f"The loaded {_model_family_name(engine)} model family does not support chat completions.",
                code="unsupported_chat_completions",
            )
            events = stream_chat(
                payload.messages,
                config=config,
                enable_thinking=enable_thinking,
                reasoning_format=reasoning_format,
                tools=payload.tools,
                tool_choice=payload.tool_choice,
                parallel_tool_calls=payload.parallel_tool_calls,
            )
            return StreamingResponse(
                _stream_sse_chat(
                    response_id=response_id,
                    created=created,
                    model=model_id,
                    events=events,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        generate_chat = _require_method(
            engine,
            "generate_chat",
            message=f"The loaded {_model_family_name(engine)} model family does not support chat completions.",
            code="unsupported_chat_completions",
        )
        result = generate_chat(
            payload.messages,
            config=config,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
            parallel_tool_calls=payload.parallel_tool_calls,
        )
    except AnnaEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    _log_generation_result(
        route_name="chat_completion",
        model=model_id,
        result=result,
        elapsed_seconds=time.perf_counter() - started_at,
        memory_before=memory_before,
        memory_after=_memory_snapshot(engine),
    )

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
        max_new_tokens=payload.max_tokens or _default_max_completion_tokens(engine),
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
        stop_strings=_normalize_stop(payload.stop),
    )
    response_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = payload.model or engine.default_model_id
    memory_before = _memory_snapshot(engine)
    started_at = time.perf_counter()

    try:
        if payload.stream:
            include_usage = _stream_usage_requested(payload)
            stream_text = _require_method(
                engine,
                "stream_text",
                message=f"The loaded {_model_family_name(engine)} model family does not support text completions.",
                code="unsupported_text_completions",
            )
            events = stream_text(payload.prompt, config=config)
            return StreamingResponse(
                _stream_sse_completion(
                    response_id=response_id,
                    created=created,
                    model=model_id,
                    events=events,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        generate_text = _require_method(
            engine,
            "generate_text",
            message=f"The loaded {_model_family_name(engine)} model family does not support text completions.",
            code="unsupported_text_completions",
        )
        result = generate_text(payload.prompt, config=config)
    except AnnaEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    _log_generation_result(
        route_name="completion",
        model=model_id,
        result=result,
        elapsed_seconds=time.perf_counter() - started_at,
        memory_before=memory_before,
        memory_after=_memory_snapshot(engine),
    )

    return JSONResponse(
        _completion_response_payload(
            response_id=response_id,
            created=created,
            model=model_id,
            result=result,
        )
    )


@router.post("/v1/audio/speech")
def audio_speech(request: Request, payload: SpeechRequest):
    engine = _engine(request)
    synthesize_qwen3_tts_speech = _require_method(
        engine,
        "synthesize_qwen3_tts_speech",
        message=f"The loaded {_model_family_name(engine)} model family does not support speech synthesis.",
        code="unsupported_speech_synthesis",
    )
    model_id = payload.model or engine.default_model_id
    speaker = payload.speaker or payload.voice
    memory_before = _memory_snapshot(engine)
    started_at = time.perf_counter()

    try:
        result = synthesize_qwen3_tts_speech(
            payload.input,
            language=payload.language,
            speaker=speaker,
            instruct=payload.instruct,
            ref_audio=payload.ref_audio,
            ref_text=payload.ref_text,
            x_vector_only_mode=payload.x_vector_only_mode,
            config=Qwen3TTSSynthesisConfig(
                max_new_tokens=payload.max_new_tokens,
                do_sample=payload.do_sample,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                repetition_penalty=payload.repetition_penalty,
                subtalker_do_sample=payload.subtalker_do_sample,
                subtalker_temperature=payload.subtalker_temperature,
                subtalker_top_p=payload.subtalker_top_p,
                subtalker_top_k=payload.subtalker_top_k,
                non_streaming_mode=payload.non_streaming_mode,
            ),
        )
    except AnnaEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    memory_after = _memory_snapshot(engine)
    elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "audio_speech model=%s audio_seconds=%.3f sample_rate=%s total_seconds=%.3f xpu_free_before=%s xpu_free_after=%s",
        model_id,
        result.duration_seconds,
        result.sample_rate,
        elapsed_seconds,
        _format_bytes(None if memory_before is None else memory_before.free_bytes),
        _format_bytes(None if memory_after is None else memory_after.free_bytes),
    )
    audio_bytes, media_type = _encode_audio_bytes(
        result.audio,
        sample_rate=result.sample_rate,
        response_format=payload.response_format,
    )
    extension = "pcm" if payload.response_format == "pcm" else payload.response_format
    headers = {
        "X-Model-Id": str(model_id),
        "X-Audio-Sample-Rate": str(result.sample_rate),
        "X-Audio-Duration-Seconds": f"{result.duration_seconds:.3f}",
        "Content-Disposition": f'inline; filename=\"speech.{extension}\"',
    }
    return Response(content=audio_bytes, media_type=media_type, headers=headers)
