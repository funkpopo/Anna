from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import AnnaEngineError
from anna.runtime.service_metrics import AnnaServiceMetrics, ServiceMetricsSnapshot

logger = logging.getLogger(__name__)


def _stringify_dtype(dtype) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


@dataclass(slots=True)
class Qwen3ASRTranscriptionConfig:
    language: str | None = None
    return_timestamps: bool = False


@dataclass(slots=True)
class Qwen3ASRTranscriptionResult:
    text: str
    language: str | None
    timestamps: Any | None
    total_seconds: float


class AnnaQwen3ASREngine:
    model_family = "qwen3_asr"
    supports_chat_completions = False
    supports_text_completions = False
    supports_speech_synthesis = False
    supports_audio_transcriptions = True
    default_max_completion_tokens = None
    default_enable_thinking = False
    reasoning_format = "none"

    def __init__(
        self,
        *,
        model,
        model_id: str,
        device_context: DeviceContext,
        supported_languages: list[str] | None,
    ) -> None:
        self.model = model
        self.default_model_id = model_id
        self.device_context = device_context
        self.supported_languages = supported_languages
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
        prompt_cache_max_tokens: int = 0,
        profile_runtime: bool = False,
        safety_policy: RuntimeSafetyPolicy | None = None,
        default_max_completion_tokens: int | None = None,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_top_k: int | None = None,
        default_min_p: float | None = None,
        default_presence_penalty: float | None = None,
        default_repetition_penalty: float | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: str = "deepseek",
        offload_mode: str = "auto",
        offload_vision: bool = False,
        expert_quant: str = "auto",
        weight_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
    ) -> "AnnaQwen3ASREngine":
        del (
            default_max_completion_tokens,
            default_temperature,
            default_top_p,
            default_top_k,
            default_min_p,
            default_presence_penalty,
            default_repetition_penalty,
            default_enable_thinking,
            reasoning_format,
            resident_expert_layers,
            resident_expert_layer_indices,
            cached_experts_per_layer,
        )
        model_path = Path(model_dir)
        device_context = DeviceContext.resolve(device=device, dtype=dtype, model_dtype="bfloat16")
        if device_context.device.type != "xpu":
            raise RuntimeError(
                "Qwen3-ASR support in Anna requires Intel XPU execution. "
                "Use --device xpu and ensure torch.xpu can see the Arc device."
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
        if prompt_cache_max_tokens:
            ignored_options.append(f"prompt_cache_max_tokens={prompt_cache_max_tokens}")
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
            logger.info("Ignoring text-generation runtime options for qwen3_asr model load: %s", ", ".join(ignored_options))

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
            raise RuntimeError(
                "Qwen3-ASR support requires the 'qwen-asr' dependency. Install project dependencies in the anna conda environment."
            ) from exc

        started_at = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=device_context.dtype,
            device_map=str(device_context.device),
            attn_implementation="eager",
            local_files_only=True,
        )
        device_context.synchronize()
        elapsed = time.perf_counter() - started_at

        resolved_model_id = model_id or model_path.name
        logger.info(
            "Loaded Qwen3-ASR model %s on %s (compute=%s, requested=%s) in %.2fs",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.requested_dtype,
            elapsed,
        )
        return cls(
            model=model,
            model_id=resolved_model_id,
            device_context=device_context,
            supported_languages=cls._read_supported_languages(model_path),
        )

    @staticmethod
    def _read_supported_languages(model_path: Path) -> list[str] | None:
        try:
            import json

            data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        except Exception:
            return None
        values = data.get("support_languages")
        if not isinstance(values, list):
            return None
        return [str(value) for value in values]

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def service_metrics_snapshot(self) -> ServiceMetricsSnapshot:
        snapshot = self.metrics.snapshot()
        return replace(snapshot, kv_cache_used_pages=0, kv_cache_total_pages=0, prompt_cache_entries=0)

    def health(self) -> dict[str, Any]:
        memory_info = self.device_context.get_memory_info()
        service_metrics = self.service_metrics_snapshot()
        return {
            "status": "ok",
            "model": self.default_model_id,
            "model_family": self.model_family,
            "device": str(self.device_context.device),
            "compute_dtype": _stringify_dtype(self.device_context.dtype),
            "requested_dtype": self.device_context.requested_dtype,
            "reported_dtype": self.device_context.reported_dtype,
            "supports_audio_transcriptions": True,
            "supports_audio_speech": False,
            "supports_text_completions": False,
            "supports_chat_completions": False,
            "supported_languages": self.supported_languages,
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

    def transcribe_qwen3_asr_audio(
        self,
        audio: str | bytes | bytearray,
        *,
        config: Qwen3ASRTranscriptionConfig,
        filename: str | None = None,
    ) -> Qwen3ASRTranscriptionResult:
        self.metrics.record_request_submitted(waiting=False)
        success = False
        started_at = time.perf_counter()
        try:
            with self.execution_lock:
                result = self._transcribe_locked(audio, config=config, filename=filename)
            success = True
            return result
        finally:
            self.metrics.record_request_finished(success=success)
            self.device_context.release_unused_memory()
            logger.debug("Completed ASR request for %s in %.3fs", self.default_model_id, time.perf_counter() - started_at)

    def _transcribe_locked(
        self,
        audio: str | bytes | bytearray,
        *,
        config: Qwen3ASRTranscriptionConfig,
        filename: str | None,
    ) -> Qwen3ASRTranscriptionResult:
        started_at = time.perf_counter()
        temp_path: Path | None = None
        audio_input: str
        if isinstance(audio, str):
            audio_input = audio
        else:
            suffix = Path(filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(prefix="anna-qwen3-asr-", suffix=suffix, delete=False) as handle:
                handle.write(bytes(audio))
                temp_path = Path(handle.name)
            audio_input = str(temp_path)

        try:
            raw = self.model.transcribe(
                audio=audio_input,
                language=config.language,
                return_time_stamps=config.return_timestamps,
            )
            self.device_context.synchronize()
        except RuntimeError as exc:
            raise self._handle_runtime_failure(exc) from exc
        except ValueError as exc:
            raise AnnaEngineError(str(exc), code="invalid_asr_request") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove temporary ASR upload file: %s", temp_path)

        return self._normalize_transcription_result(raw, config=config, total_seconds=time.perf_counter() - started_at)

    def _normalize_transcription_result(
        self,
        raw: Any,
        *,
        config: Qwen3ASRTranscriptionConfig,
        total_seconds: float,
    ) -> Qwen3ASRTranscriptionResult:
        item = raw[0] if isinstance(raw, list) and raw else raw
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("transcription") or item.get("sentence") or "")
            language = item.get("language", config.language)
            timestamps = item.get("timestamps", item.get("time_stamps", item.get("timestamp")))
        else:
            text = str(item)
            language = config.language
            timestamps = None
        if not text.strip():
            raise AnnaEngineError(
                "Qwen3-ASR produced an empty transcription.",
                status_code=500,
                error_type="server_error",
                code="empty_transcription",
            )
        return Qwen3ASRTranscriptionResult(
            text=text,
            language=None if language is None else str(language),
            timestamps=timestamps,
            total_seconds=total_seconds,
        )

    def _handle_runtime_failure(self, exc: RuntimeError) -> AnnaEngineError:
        category = self.device_context.classify_runtime_error(exc)
        if category == "out_of_memory":
            return AnnaEngineError(
                "XPU out of memory during speech recognition.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_memory",
            )
        if category == "device_lost":
            return AnnaEngineError(
                "XPU device was lost during speech recognition.",
                status_code=503,
                error_type="server_error",
                code="device_lost",
            )
        if category == "out_of_resources":
            return AnnaEngineError(
                "XPU runtime ran out of resources during speech recognition.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_resources",
            )
        return AnnaEngineError(
            f"Speech recognition failed: {exc}",
            status_code=500,
            error_type="server_error",
            code="speech_recognition_failed",
        )
