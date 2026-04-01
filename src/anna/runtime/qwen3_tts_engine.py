from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import AnnaEngineError
from anna.runtime.service_metrics import AnnaServiceMetrics, ServiceMetricsSnapshot

logger = logging.getLogger(__name__)


def _stringify_dtype(dtype) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


@dataclass(slots=True)
class Qwen3TTSSynthesisConfig:
    max_new_tokens: int | None = None
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50
    non_streaming_mode: bool = True


@dataclass(slots=True)
class Qwen3TTSSynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    total_seconds: float


class AnnaQwen3TTSEngine:
    qwen_model_family = "qwen3_tts"
    supports_chat_completions = False
    supports_text_completions = False
    supports_speech_synthesis = True
    default_max_completion_tokens = None
    default_enable_thinking = False
    reasoning_format = "none"

    def __init__(
        self,
        *,
        model,
        model_id: str,
        device_context: DeviceContext,
        tokenizer_type: str,
        tts_model_type: str,
        tts_model_size: str,
    ) -> None:
        self.model = model
        self.default_model_id = model_id
        self.device_context = device_context
        self.tokenizer_type = tokenizer_type
        self.tts_model_type = tts_model_type
        self.tts_model_size = tts_model_size
        self.metrics = AnnaServiceMetrics()
        self.execution_lock = threading.Lock()

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_id: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        compile_mode: str = "none",
        compile_fullgraph: bool = False,
        prefill_chunk_size: int = 0,
        prompt_cache_size: int = 0,
        profile_runtime: bool = False,
        safety_policy: RuntimeSafetyPolicy | None = None,
        default_max_completion_tokens: int | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: str = "deepseek",
        offload_mode: str = "auto",
        offload_vision: bool = False,
        expert_quant: str = "auto",
        weight_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
    ) -> "AnnaQwen3TTSEngine":
        del (
            default_max_completion_tokens,
            default_enable_thinking,
            reasoning_format,
            resident_expert_layers,
            resident_expert_layer_indices,
            cached_experts_per_layer,
        )
        model_path = Path(model_dir)
        device_context = DeviceContext.resolve(
            device=device,
            dtype=dtype,
            model_dtype="bfloat16",
        )
        if safety_policy is not None:
            device_context.safety_policy = safety_policy

        ignored_options: list[str] = []
        if compile_mode != "none":
            ignored_options.append(f"compile_mode={compile_mode}")
        if compile_fullgraph:
            ignored_options.append("compile_fullgraph=True")
        if prefill_chunk_size:
            ignored_options.append(f"prefill_chunk_size={prefill_chunk_size}")
        if prompt_cache_size:
            ignored_options.append(f"prompt_cache_size={prompt_cache_size}")
        if profile_runtime:
            ignored_options.append("profile_runtime=True")
        if offload_mode != "auto":
            ignored_options.append(f"offload_mode={offload_mode}")
        if offload_vision:
            ignored_options.append("offload_vision=True")
        if expert_quant != "auto":
            ignored_options.append(f"expert_quant={expert_quant}")
        if weight_quant != "auto":
            ignored_options.append(f"weight_quant={weight_quant}")
        if ignored_options:
            logger.info(
                "Ignoring qwen3_5_text-family runtime options for qwen3_tts model load: %s",
                ", ".join(ignored_options),
            )

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover - exercised in environment setup, not unit tests
            raise RuntimeError(
                "Qwen3-TTS support requires the optional dependency 'qwen-tts'. Install project dependencies again to enable the qwen3_tts model family."
            ) from exc

        started_at = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device_context.device.type,
            dtype=device_context.dtype,
            attn_implementation="eager",
            local_files_only=True,
        )
        device_context.synchronize()
        elapsed = time.perf_counter() - started_at

        resolved_model_id = model_id or model_path.name
        logger.info(
            "Loaded Qwen3-TTS model %s on %s (compute=%s, requested=%s, tokenizer=%s, tts_model_type=%s, tts_model_size=%s) in %.2fs",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.requested_dtype,
            getattr(model.model, "tokenizer_type", "unknown"),
            getattr(model.model, "tts_model_type", "unknown"),
            getattr(model.model, "tts_model_size", "unknown"),
            elapsed,
        )

        return cls(
            model=model,
            model_id=resolved_model_id,
            device_context=device_context,
            tokenizer_type=str(getattr(model.model, "tokenizer_type", "unknown")),
            tts_model_type=str(getattr(model.model, "tts_model_type", "unknown")),
            tts_model_size=str(getattr(model.model, "tts_model_size", "unknown")),
        )

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def service_metrics_snapshot(self) -> ServiceMetricsSnapshot:
        snapshot = self.metrics.snapshot()
        return replace(
            snapshot,
            kv_cache_used_pages=0,
            kv_cache_total_pages=0,
            prompt_cache_entries=0,
        )

    def health(self) -> dict[str, Any]:
        memory_info = self.device_context.get_memory_info()
        service_metrics = self.service_metrics_snapshot()
        supported_languages = self._safe_supported_languages()
        supported_speakers = self._safe_supported_speakers()
        return {
            "status": "ok",
            "model": self.default_model_id,
            "qwen_model_family": self.qwen_model_family,
            "tts_model_type": self.tts_model_type,
            "tts_model_size": self.tts_model_size,
            "tokenizer_type": self.tokenizer_type,
            "device": str(self.device_context.device),
            "compute_dtype": _stringify_dtype(self.device_context.dtype),
            "requested_dtype": self.device_context.requested_dtype,
            "reported_dtype": self.device_context.reported_dtype,
            "supports_audio_speech": True,
            "supports_text_completions": False,
            "supports_chat_completions": False,
            "supported_languages": supported_languages,
            "supported_speakers": supported_speakers,
            "memory": None
            if memory_info is None
            else {
                "free_bytes": memory_info.free_bytes,
                "total_bytes": memory_info.total_bytes,
                "allocated_bytes": memory_info.allocated_bytes,
                "reserved_bytes": memory_info.reserved_bytes,
            },
            "service_metrics": {
                "requests_total": service_metrics.requests_total,
                "requests_in_flight": service_metrics.requests_in_flight,
                "requests_succeeded": service_metrics.requests_succeeded,
                "requests_failed": service_metrics.requests_failed,
            },
        }

    def synthesize_qwen3_tts_speech(
        self,
        text: str,
        *,
        config: Qwen3TTSSynthesisConfig,
        language: str | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
    ) -> Qwen3TTSSynthesisResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise AnnaEngineError("Qwen3-TTS speech synthesis input must not be empty.")

        self.metrics.record_request_submitted(waiting=False)
        success = False
        started_at = time.perf_counter()
        try:
            with self.execution_lock:
                result = self._synthesize_locked(
                    normalized_text,
                    config=config,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
            success = True
            return result
        finally:
            self.metrics.record_request_finished(success=success)
            self.device_context.release_unused_memory()
            logger.debug(
                "Completed speech synthesis request for %s in %.3fs",
                self.default_model_id,
                time.perf_counter() - started_at,
            )

    def _synthesize_locked(
        self,
        text: str,
        *,
        config: Qwen3TTSSynthesisConfig,
        language: str | None,
        speaker: str | None,
        instruct: str | None,
        ref_audio: str | None,
        ref_text: str | None,
        x_vector_only_mode: bool,
    ) -> Qwen3TTSSynthesisResult:
        generation_kwargs = self._build_generation_kwargs(config)
        language_value = language or "Auto"
        started_at = time.perf_counter()
        try:
            if self.tts_model_type == "base":
                if not ref_audio:
                    raise AnnaEngineError(
                        "The loaded Qwen3-TTS Base model requires ref_audio for voice cloning requests.",
                        code="missing_ref_audio",
                    )
                if not x_vector_only_mode and not ref_text:
                    raise AnnaEngineError(
                        "ref_text is required for Qwen3-TTS Base voice cloning unless x_vector_only_mode=true.",
                        code="missing_ref_text",
                    )
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=language_value,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **generation_kwargs,
                )
            elif self.tts_model_type == "custom_voice":
                if not speaker:
                    raise AnnaEngineError(
                        "The loaded Qwen3-TTS CustomVoice model requires speaker (or voice).",
                        code="missing_speaker",
                    )
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=text,
                    language=language_value,
                    speaker=speaker,
                    instruct=instruct,
                    **generation_kwargs,
                )
            elif self.tts_model_type == "voice_design":
                if not instruct:
                    raise AnnaEngineError(
                        "The loaded Qwen3-TTS VoiceDesign model requires instruct text.",
                        code="missing_instruct",
                    )
                wavs, sample_rate = self.model.generate_voice_design(
                    text=text,
                    language=language_value,
                    instruct=instruct,
                    **generation_kwargs,
                )
            else:
                raise AnnaEngineError(
                    f"Unsupported Qwen3-TTS model type: {self.tts_model_type}",
                    status_code=500,
                    error_type="server_error",
                    code="unsupported_tts_model_type",
                )
            self.device_context.synchronize()
        except AnnaEngineError:
            raise
        except RuntimeError as exc:
            raise self._handle_runtime_failure(exc) from exc
        except ValueError as exc:
            raise AnnaEngineError(str(exc), code="invalid_tts_request") from exc

        if not wavs:
            raise AnnaEngineError(
                "Speech synthesis produced no audio samples.",
                status_code=500,
                error_type="server_error",
                code="empty_audio_output",
            )

        audio = np.asarray(wavs[0], dtype=np.float32)
        total_seconds = time.perf_counter() - started_at
        return Qwen3TTSSynthesisResult(
            audio=audio,
            sample_rate=int(sample_rate),
            duration_seconds=0.0 if sample_rate <= 0 else float(audio.shape[0]) / float(sample_rate),
            total_seconds=total_seconds,
        )

    def _build_generation_kwargs(self, config: Qwen3TTSSynthesisConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "subtalker_dosample": config.subtalker_do_sample,
            "subtalker_temperature": config.subtalker_temperature,
            "subtalker_top_p": config.subtalker_top_p,
            "subtalker_top_k": config.subtalker_top_k,
            "non_streaming_mode": config.non_streaming_mode,
        }
        if config.max_new_tokens is not None:
            kwargs["max_new_tokens"] = config.max_new_tokens
        return kwargs

    def _handle_runtime_failure(self, exc: RuntimeError) -> AnnaEngineError:
        category = self.device_context.classify_runtime_error(exc)
        if self.device_context.should_recover(exc):
            try:
                self.device_context.recover()
            except Exception:  # pragma: no cover - best-effort recovery
                logger.exception("Failed to recover device context after Qwen3-TTS runtime failure.")

        if category == "out_of_memory":
            return AnnaEngineError(
                "XPU out of memory during speech synthesis. Reduce the request size or retry on CPU.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_memory",
            )
        if category == "device_lost":
            return AnnaEngineError(
                "XPU device was lost during speech synthesis. Retry the request after the device recovers.",
                status_code=503,
                error_type="server_error",
                code="device_lost",
            )
        if category == "out_of_resources":
            return AnnaEngineError(
                "XPU runtime ran out of resources during speech synthesis. Reduce the request size and retry.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_resources",
            )
        return AnnaEngineError(
            f"Speech synthesis failed: {exc}",
            status_code=500,
            error_type="server_error",
            code="speech_synthesis_failed",
        )

    def _safe_supported_languages(self) -> list[str] | None:
        getter = getattr(self.model.model, "get_supported_languages", None)
        if not callable(getter):
            return None
        values = getter()
        if values is None:
            return None
        return [str(value) for value in values]

    def _safe_supported_speakers(self) -> list[str] | None:
        getter = getattr(self.model.model, "get_supported_speakers", None)
        if not callable(getter):
            return None
        values = getter()
        if values is None:
            return None
        return [str(value) for value in values]
