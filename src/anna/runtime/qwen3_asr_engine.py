from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import torch

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
    context: str = ""
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
        xpu_tensor_numel: int,
        xpu_tensor_devices: dict[str, int],
    ) -> None:
        self.model = model
        self.default_model_id = model_id
        self.device_context = device_context
        self.supported_languages = supported_languages
        self.xpu_tensor_numel = xpu_tensor_numel
        self.xpu_tensor_devices = xpu_tensor_devices
        self.metrics = AnnaServiceMetrics()
        self.execution_lock = threading.Lock()
        self._xpu_forward_observed_count = 0
        self._last_xpu_elapsed_ms: float | None = None
        self._xpu_guard_handles = self._install_xpu_execution_guards()

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
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
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
        asr_max_inference_batch_size: int = 1,
        asr_max_new_tokens: int = 512,
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
        if asr_max_inference_batch_size <= 0:
            raise ValueError("Qwen3-ASR asr_max_inference_batch_size must be > 0.")
        if asr_max_new_tokens <= 0:
            raise ValueError("Qwen3-ASR asr_max_new_tokens must be > 0.")

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
        if kv_cache_quantization != "none":
            ignored_options.append(f"kv_cache_quantization={kv_cache_quantization}")
        if kv_cache_quant_bits != 4:
            ignored_options.append(f"kv_cache_quant_bits={kv_cache_quant_bits}")
        if kv_cache_residual_len != 128:
            ignored_options.append(f"kv_cache_residual_len={kv_cache_residual_len}")
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
            max_inference_batch_size=asr_max_inference_batch_size,
            max_new_tokens=asr_max_new_tokens,
        )
        device_context.synchronize()
        xpu_tensor_numel, xpu_tensor_devices = cls._validate_xpu_only_model(model, device_context.device)
        elapsed = time.perf_counter() - started_at

        resolved_model_id = model_id or model_path.name
        logger.info(
            "Loaded Qwen3-ASR model %s on %s (compute=%s, requested=%s, max_inference_batch_size=%s, max_new_tokens=%s, xpu_tensor_numel=%s) in %.2fs",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.requested_dtype,
            asr_max_inference_batch_size,
            asr_max_new_tokens,
            xpu_tensor_numel,
            elapsed,
        )
        return cls(
            model=model,
            model_id=resolved_model_id,
            device_context=device_context,
            supported_languages=cls._read_supported_languages(model_path),
            xpu_tensor_numel=xpu_tensor_numel,
            xpu_tensor_devices=xpu_tensor_devices,
        )

    @staticmethod
    def _normalize_device(value: Any) -> torch.device | None:
        if value is None:
            return None
        try:
            return torch.device(value)
        except (TypeError, RuntimeError):
            return None

    @classmethod
    def _validate_xpu_only_model(cls, model: Any, expected_device: torch.device) -> tuple[int, dict[str, int]]:
        if getattr(model, "backend", None) != "transformers":
            raise RuntimeError(
                "Qwen3-ASR in Anna uses the transformers backend on Intel XPU only. "
                f"Unsupported qwen-asr backend: {getattr(model, 'backend', None)!r}."
            )

        wrapper_device = cls._normalize_device(getattr(model, "device", None))
        if wrapper_device is None or wrapper_device.type != "xpu":
            raise RuntimeError(f"Qwen3-ASR wrapper is not bound to XPU after load: device={getattr(model, 'device', None)!r}.")

        actual_model = getattr(model, "model", None)
        if not isinstance(actual_model, torch.nn.Module):
            raise RuntimeError("Qwen3-ASR backend did not expose a torch.nn.Module model for XPU validation.")

        module_device = cls._normalize_device(getattr(actual_model, "device", None))
        if module_device is not None and module_device.type != "xpu":
            raise RuntimeError(f"Qwen3-ASR torch module is not bound to XPU after load: device={module_device}.")

        hf_device_map = getattr(actual_model, "hf_device_map", None)
        if isinstance(hf_device_map, Mapping):
            non_xpu_entries = {
                str(key): str(value)
                for key, value in hf_device_map.items()
                if cls._normalize_device(value) is None or cls._normalize_device(value).type != "xpu"
            }
            if non_xpu_entries:
                raise RuntimeError(f"Qwen3-ASR loaded with non-XPU device_map entries: {non_xpu_entries}.")

        tensor_devices: dict[str, int] = {}
        bad_tensors: list[str] = []
        total_numel = 0
        tensor_count = 0
        for collection_name, iterator in (
            ("parameter", actual_model.named_parameters(recurse=True)),
            ("buffer", actual_model.named_buffers(recurse=True)),
        ):
            for name, tensor in iterator:
                tensor_count += 1
                numel = int(tensor.numel())
                total_numel += numel
                device_name = str(tensor.device)
                tensor_devices[device_name] = tensor_devices.get(device_name, 0) + numel
                if tensor.device.type != "xpu":
                    bad_tensors.append(f"{collection_name} {name} on {tensor.device} shape={tuple(tensor.shape)}")

        if tensor_count == 0:
            raise RuntimeError("Qwen3-ASR model exposes no tensors; refusing to run without an XPU tensor audit.")
        if bad_tensors:
            preview = "; ".join(bad_tensors[:8])
            suffix = "" if len(bad_tensors) <= 8 else f"; ... {len(bad_tensors) - 8} more"
            raise RuntimeError(f"Qwen3-ASR model has tensors outside XPU: {preview}{suffix}.")
        if expected_device.type != "xpu":
            raise RuntimeError(f"Qwen3-ASR expected device must be XPU, got {expected_device}.")
        return total_numel, tensor_devices

    def _install_xpu_execution_guards(self) -> list[Any]:
        actual_model = getattr(self.model, "model", None)
        if not isinstance(actual_model, torch.nn.Module):
            raise RuntimeError("Qwen3-ASR backend did not expose a torch.nn.Module model for execution guards.")

        candidates: list[tuple[str, torch.nn.Module]] = []
        for name in ("thinker", "thinker.audio_tower", "thinker.model"):
            module = actual_model
            for part in name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, torch.nn.Module) and all(module is not existing for _, existing in candidates):
                candidates.append((name, module))
        if not candidates:
            candidates.append(("model", actual_model))

        handles: list[Any] = []
        for name, module in candidates:
            handles.append(module.register_forward_pre_hook(self._make_xpu_guard_hook(name), with_kwargs=True))
        return handles

    def _make_xpu_guard_hook(self, module_name: str):
        def _guard(_module, args, kwargs):
            self._assert_tensor_tree_on_xpu(args, context=f"{module_name} args")
            self._assert_tensor_tree_on_xpu(kwargs, context=f"{module_name} kwargs")
            self._xpu_forward_observed_count += 1

        return _guard

    def _assert_tensor_tree_on_xpu(self, value: Any, *, context: str) -> int:
        bad_tensors: list[str] = []
        seen_objects: set[int] = set()

        def _walk(item: Any, path: str) -> int:
            object_id = id(item)
            if object_id in seen_objects:
                return 0
            if isinstance(item, (Mapping, list, tuple, set, frozenset)):
                seen_objects.add(object_id)
            if isinstance(item, torch.Tensor):
                if item.device.type != "xpu":
                    bad_tensors.append(f"{path} on {item.device} shape={tuple(item.shape)}")
                return 1
            if isinstance(item, Mapping):
                count = 0
                for key, child in item.items():
                    count += _walk(child, f"{path}.{key}")
                return count
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                count = 0
                for index, child in enumerate(item):
                    count += _walk(child, f"{path}[{index}]")
                return count
            return 0

        tensor_count = _walk(value, context)
        if bad_tensors:
            preview = "; ".join(bad_tensors[:8])
            suffix = "" if len(bad_tensors) <= 8 else f"; ... {len(bad_tensors) - 8} more"
            raise RuntimeError(f"Qwen3-ASR attempted to execute with non-XPU tensors: {preview}{suffix}.")
        return tensor_count

    def _start_xpu_execution_probe(self) -> tuple[Any | None, Any | None]:
        xpu = getattr(torch, "xpu", None)
        event_cls = getattr(xpu, "Event", None) if xpu is not None else None
        if event_cls is None:
            raise RuntimeError("Qwen3-ASR XPU execution verification requires torch.xpu.Event.")
        start_event = event_cls(enable_timing=True)
        end_event = event_cls(enable_timing=True)
        start_event.record()
        return start_event, end_event

    def _finish_xpu_execution_probe(self, probe: tuple[Any | None, Any | None], *, forward_count_before: int) -> None:
        start_event, end_event = probe
        if end_event is not None:
            end_event.record()
        self.device_context.synchronize()
        if self._xpu_forward_observed_count <= forward_count_before:
            raise RuntimeError("Qwen3-ASR transcription completed without observing an XPU model forward pass.")
        if start_event is not None and end_event is not None:
            try:
                self._last_xpu_elapsed_ms = float(start_event.elapsed_time(end_event))
            except RuntimeError as exc:
                raise RuntimeError("Qwen3-ASR XPU execution verification failed to read torch.xpu.Event timing.") from exc

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
            "xpu_execution_enforced": True,
            "xpu_tensor_numel": self.xpu_tensor_numel,
            "xpu_tensor_devices": self.xpu_tensor_devices,
            "xpu_forward_observed_count": self._xpu_forward_observed_count,
            "last_xpu_elapsed_ms": self._last_xpu_elapsed_ms,
            "xpu_device": None
            if self.device_context.xpu_info is None
            else self.device_context.xpu_info.as_log_fields(),
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
            forward_count_before = self._xpu_forward_observed_count
            probe = self._start_xpu_execution_probe()
            raw = self.model.transcribe(
                audio=audio_input,
                context=config.context,
                language=config.language,
                return_time_stamps=config.return_timestamps,
            )
            self._finish_xpu_execution_probe(probe, forward_count_before=forward_count_before)
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
            text_value = (
                getattr(item, "text", None)
                or getattr(item, "transcription", None)
                or getattr(item, "sentence", None)
            )
            text = "" if item is None else str(item) if text_value is None else str(text_value)
            language = getattr(item, "language", config.language)
            timestamps = (
                getattr(item, "timestamps", None)
                or getattr(item, "time_stamps", None)
                or getattr(item, "timestamp", None)
            )
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
