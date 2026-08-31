from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import torch

from anna.core.function_calling import ThinkingStreamParser, ToolCallDelta
from anna.core.format_utils import format_bytes
from anna.core.gguf_model import has_gguf_model
from anna.mm.prepared_inputs import PreparedInputs
from anna.mm.qwen3_5_text_processor import Qwen3_5TextMultimodalProcessor
from anna.model.fused_ops import (
    gqa_decode_splitkv_fused_out_is_available,
    gqa_decode_splitkv_turboquant_fused_out_is_available,
    maybe_load_gated_delta_library,
    paged_gqa_decode_fused_is_available,
)
from anna.model.quantization import (
    AutoRoundGPTQLinear,
    XPUInt4Linear,
    convert_module_linears_to_xpu_int4,
    estimate_module_xpu_int4_bytes,
    resolve_xpu_int4_layout_cache_dir,
    weight_quant_auto_usage_threshold,
)
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator, Qwen3SparseMoeBlock
from anna.model.turboquant import (
    VALID_KV_CACHE_QUANT_BITS,
    resolve_turboquant_runtime_settings,
    turboquant_is_available,
)
from anna.model.xpu_decode_profile import (
    _amortized_per_request_ms,
    get_or_create_scheduler_decode_steady_accum,
    record_steady_decode_step_if_applicable,
    steady_decode_accumulation,
)
from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.decode_executor import (
    DecodeGraphUnavailable,
    DecodeGraphRunner,
    normalize_decode_executor,
    resolve_decode_executor_mode,
)
from anna.runtime.memory_release import AdaptiveMemoryReleaser, release_conversion_artifacts
from anna.runtime.runtime_health import PROCESS_ADMISSION_GATE, RuntimeAdmissionGate
from anna.runtime.service_metrics import AnnaServiceMetrics, ServiceMetricsSnapshot
from anna.runtime.streaming import IncrementalTextAssembler
from anna.sampling.sampler import (
    sample_next_token,
    sample_next_token_from_candidates,
    token_ids_to_host,
)
from anna.weights.qwen3_5_text_weight_loader import build_qwen3_5_text_model, estimate_qwen3_5_text_model_weight_bytes, load_qwen3_5_text_model_config, load_qwen3_5_text_model_weights
from anna.weights.qwen3_5_text_tokenizer import Qwen3_5TextTokenizer

logger = logging.getLogger(__name__)

ReasoningFormat = Literal["none", "deepseek"]
_REASONING_FORMAT_VALUES = frozenset({"none", "deepseek"})
_DEFAULT_REASONING_FORMAT: ReasoningFormat = "deepseek"
_COMPILE_MODE_VALUES = frozenset({"none", "auto", "default", "reduce-overhead", "max-autotune"})
# Final runtime modes only (``auto`` is resolved before EngineOptimizationConfig is built).
_KV_CACHE_QUANTIZATION_VALUES = frozenset({"none", "turboquant"})


def _module_cpu_tensor_bytes(module: torch.nn.Module) -> int:
    total = 0
    seen: set[tuple[int, int, int]] = set()
    tensors = itertools.chain(module.named_parameters(), module.named_buffers())
    for _name, tensor in tensors:
        if tensor.device.type != "cpu":
            continue
        try:
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(tensor.storage_offset()), int(tensor.nbytes))
        except Exception:
            key = (id(tensor), 0, int(tensor.nelement() * tensor.element_size()))
        if key in seen:
            continue
        seen.add(key)
        total += int(tensor.nelement() * tensor.element_size())
    return total


def normalize_reasoning_format(value: str | None) -> ReasoningFormat:
    if value is None:
        return _DEFAULT_REASONING_FORMAT
    normalized = value.strip().lower()
    if normalized not in _REASONING_FORMAT_VALUES:
        allowed = ", ".join(sorted(_REASONING_FORMAT_VALUES))
        raise ValueError(f"Unsupported reasoning format: {value}. Expected one of: {allowed}.")
    return cast(ReasoningFormat, normalized)


def normalize_compile_mode(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = value.strip().lower()
    if normalized not in _COMPILE_MODE_VALUES:
        allowed = ", ".join(sorted(_COMPILE_MODE_VALUES))
        raise ValueError(f"Unsupported compile mode: {value}. Expected one of: {allowed}.")
    return normalized


def _qwen3_paged_full_attention_decode_enabled(*, device_type: str, kv_cache_quantization: str) -> tuple[bool, bool]:
    """Return (use_paged_full_attention_decode, maintain_full_attention_mirror) for Qwen3PageAllocator."""
    turboquant_kv_enabled = kv_cache_quantization == "turboquant"
    use_paged_full_attention_decode = False
    if device_type == "xpu" and turboquant_kv_enabled:
        maybe_load_gated_delta_library()
        if not gqa_decode_splitkv_fused_out_is_available() or not gqa_decode_splitkv_turboquant_fused_out_is_available():
            raise RuntimeError(
                "Anna Qwen3.5 TurboQuant decode on Intel XPU requires the native fused-op library with "
                "gqa_decode_splitkv_fused_out and gqa_decode_splitkv_turboquant_fused_out. "
                "Build anna_gated_delta_fused (see tools/build_gated_delta_fused_op.py) or set ANNA_GATED_DELTA_OP_LIB."
            )
    if device_type == "xpu" and not turboquant_kv_enabled:
        maybe_load_gated_delta_library()
        if not paged_gqa_decode_fused_is_available():
            raise RuntimeError(
                "Anna Qwen3.5 on Intel XPU requires the native fused-op library with paged_gqa_decode_fused. "
                "Build anna_gated_delta_fused (see tools/build_gated_delta_fused_op.py) or set ANNA_GATED_DELTA_OP_LIB."
            )
        use_paged_full_attention_decode = True
    maintain_full_attention_mirror = not use_paged_full_attention_decode and not turboquant_kv_enabled
    return use_paged_full_attention_decode, maintain_full_attention_mirror


def normalize_kv_cache_quantization(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = value.strip().lower()
    if normalized not in _KV_CACHE_QUANTIZATION_VALUES:
        allowed = ", ".join(sorted(_KV_CACHE_QUANTIZATION_VALUES))
        raise ValueError(f"Unsupported KV-cache quantization mode: {value}. Expected one of: {allowed}.")
    return normalized


@dataclass(slots=True)
class EngineOptimizationConfig:
    compile_mode: str = "none"
    compile_fullgraph: bool = False
    prefill_chunk_size: int = 0
    prompt_cache_size: int = 0
    prompt_cache_max_tokens: int = 0
    profile_runtime: bool = False
    kv_cache_quantization: str = "none"
    kv_cache_quant_bits: int = 4
    kv_cache_residual_len: int = 128
    # Size-tier name applied when bits/residual came from recommend_turboquant_preset.
    kv_cache_turboquant_preset: str | None = None
    # When True, skip paged prefix-block registration for prompts that are eligible
    # for exact prompt-cache reuse (avoids dual full-KV + page registry waste).
    prefer_prompt_cache_over_prefix: bool = True
    # Phase 2: decode step graph execution. "auto" enables graph capture when a
    # backend (CUDA graph / torch.xpu graph) is detected on the target device.
    decode_executor: str = "auto"


@dataclass(slots=True)
class PromptCacheEntry:
    logits: torch.Tensor
    past_key_values: object
    prompt_tokens: int


@dataclass(slots=True)
class PromptPrefillResult:
    logits: torch.Tensor
    past_key_values: object | None
    prefill_seconds: float
    prompt_cache_hit: bool = False


class AnnaEngineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    stop_strings: list[str] = field(default_factory=list)
    cancellation_event: threading.Event | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class GenerationPerfStats:
    total_seconds: float
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    prompt_tokens: int
    completion_tokens: int
    prefill_tokens_per_second: float
    decode_tokens: int
    decode_tokens_per_second: float
    total_tokens_per_second: float


@dataclass(slots=True)
class StreamEvent:
    text: str
    reasoning_text: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    perf: GenerationPerfStats | None = None


@dataclass(slots=True)
class TextGenerationResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_text: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    perf: GenerationPerfStats | None = None


class AnnaQwen3_5TextEngine:
    model_family = "qwen3_5_text"
    supports_chat_completions = True
    supports_text_completions = True
    supports_speech_synthesis = False

    def __init__(
        self,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        tokenizer: Qwen3_5TextTokenizer,
        processor: Qwen3_5TextMultimodalProcessor,
        model_id: str,
        device_context: DeviceContext,
        quantized_replacements: int = 0,
        default_max_completion_tokens: int | None = None,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_top_k: int | None = None,
        default_min_p: float | None = None,
        default_presence_penalty: float | None = None,
        default_repetition_penalty: float | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        offload_mode: str = "none",
        offload_vision: bool = False,
        expert_quant: str = "none",
        weight_quant: str = "none",
        resident_expert_layer_indices: tuple[int, ...] = (),
        cached_experts_per_layer: int = 0,
        optimization_config: EngineOptimizationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.default_model_id = model_id
        self.device_context = device_context
        self.config = model.config
        self.quantized_replacements = quantized_replacements
        self.default_max_completion_tokens = (
            None if default_max_completion_tokens is None else max(1, int(default_max_completion_tokens))
        )
        self.default_temperature = 0.7 if default_temperature is None else max(0.0, float(default_temperature))
        self.default_top_p = 0.8 if default_top_p is None else min(1.0, max(0.0, float(default_top_p)))
        self.default_top_k = 20 if default_top_k is None else max(0, int(default_top_k))
        self.default_min_p = 0.0 if default_min_p is None else min(1.0, max(0.0, float(default_min_p)))
        self.default_presence_penalty = 1.5 if default_presence_penalty is None else float(default_presence_penalty)
        self.default_repetition_penalty = (
            1.0 if default_repetition_penalty is None else max(0.1, float(default_repetition_penalty))
        )
        self.default_enable_thinking = bool(default_enable_thinking)
        self.reasoning_format = normalize_reasoning_format(reasoning_format)
        self.offload_mode = offload_mode
        self.offload_vision = offload_vision
        self.expert_quant = expert_quant
        self.weight_quant = weight_quant
        self.resident_expert_layer_indices = tuple(resident_expert_layer_indices)
        self.resident_expert_layers = len(self.resident_expert_layer_indices)
        self.cached_experts_per_layer = max(0, int(cached_experts_per_layer))
        self.optimization_config = self._normalize_optimization_config(optimization_config)
        _, maintain_full_attention_mirror = _qwen3_paged_full_attention_decode_enabled(
            device_type=self.device_context.device.type,
            kv_cache_quantization=self.optimization_config.kv_cache_quantization,
        )
        self.cache_allocator = Qwen3PageAllocator(
            self.config.text_config,
            maintain_full_attention_mirror=maintain_full_attention_mirror,
        )
        self.full_attention_cache_mirror = maintain_full_attention_mirror and bool(
            self.cache_allocator.full_attention_layer_indices
        )
        self.optimization_config = replace(
            self.optimization_config,
            prefill_chunk_size=self._resolve_prefill_chunk_size(self.optimization_config.prefill_chunk_size),
        )
        self._attach_cache_allocator()
        self.execution_lock = threading.Lock()
        self.scheduler = None
        self._compiled_text_forward = None
        self._prompt_cache: OrderedDict[tuple[int, ...], PromptCacheEntry] = OrderedDict()
        self.metrics = AnnaServiceMetrics()
        self._apply_runtime_optimizations()
        self._attach_prefix_share_gate()
        # P2-#6: idle-time allocator fragmentation control (reserved >> allocated).
        self._memory_releaser = self._create_memory_releaser()
        self._memory_releaser.start()

    def _create_memory_releaser(self) -> AdaptiveMemoryReleaser:
        """Build the idle memory sweeper (XPU only; a no-op provider elsewhere)."""
        if self.device_context.device.type != "xpu":
            return AdaptiveMemoryReleaser(
                snapshot_provider=lambda: None,
                release_callback=lambda: None,
                interval_seconds=0.0,
            )
        return AdaptiveMemoryReleaser(
            snapshot_provider=self._adaptive_memory_release_snapshot,
            release_callback=self._adaptive_memory_release_execute,
        )

    def _adaptive_memory_release_snapshot(self) -> dict[str, object] | None:
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            snap = metrics.snapshot()
            if snap.running_requests > 0 or snap.waiting_requests > 0:
                return {"idle": False}
        memory_info = self.device_context.get_memory_info()
        if memory_info is None:
            return None
        return {
            "idle": True,
            "free_bytes": memory_info.free_bytes,
            "total_bytes": memory_info.total_bytes,
            "reserved_bytes": memory_info.reserved_bytes,
            "allocated_bytes": memory_info.allocated_bytes,
            "min_free_bytes": self.device_context.safety_policy.min_free_bytes,
        }

    def _adaptive_memory_release_execute(self) -> None:
        """Release idle device memory: trim KV pages, evict prompt cache, empty_cache."""
        prompt_cache = getattr(self, "_prompt_cache", None)
        if prompt_cache:
            for key, entry in list(prompt_cache.items()):
                self._evict_prompt_cache_entry(key, entry)
        trimmed_pages = self.cache_allocator.trim()
        release_unused_memory = getattr(self.device_context, "release_unused_memory", None)
        if callable(release_unused_memory):
            release_unused_memory()
        if trimmed_pages > 0:
            logger.info("Adaptive memory release trimmed idle KV cache pages: released_pages=%s", trimmed_pages)

    @staticmethod
    def _normalize_optimization_config(config: EngineOptimizationConfig | None) -> EngineOptimizationConfig:
        if config is None:
            return EngineOptimizationConfig()
        kv_cache_quant_bits = int(config.kv_cache_quant_bits)
        if kv_cache_quant_bits not in VALID_KV_CACHE_QUANT_BITS:
            raise ValueError(
                f"Unsupported TurboQuant KV-cache bit-width: {config.kv_cache_quant_bits}. "
                f"Expected one of {sorted(VALID_KV_CACHE_QUANT_BITS)}."
            )
        return EngineOptimizationConfig(
            compile_mode=normalize_compile_mode(config.compile_mode),
            compile_fullgraph=bool(config.compile_fullgraph),
            prefill_chunk_size=max(0, int(config.prefill_chunk_size)),
            prompt_cache_size=max(0, int(config.prompt_cache_size)),
            prompt_cache_max_tokens=max(0, int(config.prompt_cache_max_tokens)),
            profile_runtime=bool(config.profile_runtime),
            kv_cache_quantization=normalize_kv_cache_quantization(config.kv_cache_quantization),
            kv_cache_quant_bits=kv_cache_quant_bits,
            kv_cache_residual_len=max(1, int(config.kv_cache_residual_len)),
            kv_cache_turboquant_preset=config.kv_cache_turboquant_preset,
            prefer_prompt_cache_over_prefix=bool(config.prefer_prompt_cache_over_prefix),
            decode_executor=normalize_decode_executor(config.decode_executor),
        )

    def _resolve_prefill_chunk_size(self, requested_chunk_size: int) -> int:
        if requested_chunk_size > 0:
            block = max(1, int(self.config.text_config.cache_block_size))
            raw = max(1, int(requested_chunk_size))
            aligned = max(block, (raw // block) * block)
            return aligned
        text_config = self.config.text_config
        bytes_per_elem = self.device_context.element_size(self.device_context.dtype)
        full_layers = sum(1 for layer_type in text_config.layer_types if layer_type == "full_attention")
        linear_layers = max(0, int(text_config.num_hidden_layers) - full_layers)
        full_layer_kv_bytes = (
            0
            if self.optimization_config.kv_cache_quantization == "turboquant"
            else (
                full_layers
                * 2
                * int(text_config.num_key_value_heads)
                * int(text_config.head_dim)
                * bytes_per_elem
            )
        )
        per_token_kv_bytes = (
            full_layer_kv_bytes
        )
        if self.full_attention_cache_mirror:
            per_token_kv_bytes *= 2
        per_token_hidden_bytes = int(text_config.hidden_size) * bytes_per_elem * 8
        per_token_linear_bytes = linear_layers * (
            (
                int(text_config.linear_num_key_heads) * int(text_config.linear_key_head_dim) * 2
            )
            + (
                int(text_config.linear_num_value_heads) * int(text_config.linear_value_head_dim)
            )
        ) * bytes_per_elem
        estimated_bytes_per_token = max(1, per_token_kv_bytes + per_token_hidden_bytes + per_token_linear_bytes)

        memory_info = self.device_context.get_memory_info()
        if memory_info is None:
            target_chunk_budget = 96 << 20
        else:
            policy = self.device_context.safety_policy
            available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
            available_budget = max(
                64 << 20,
                min(max(available_budget // 6, 256 << 20), 1024 << 20),
            )
            target_chunk_budget = available_budget

        auto_chunk = int(target_chunk_budget // estimated_bytes_per_token)
        resolved = max(128, min(2048, auto_chunk))
        block = max(1, int(text_config.cache_block_size))
        if block > 1:
            resolved = max(block, (resolved // block) * block)
        logger.info(
            "Enabled auto prefill chunking on %s: chunk_size=%s block_size=%s estimated_bytes_per_token=%s target_budget=%s",
            self.device_context.device,
            resolved,
            block,
            format_bytes(estimated_bytes_per_token),
            format_bytes(target_chunk_budget),
        )
        return resolved

    def _apply_runtime_optimizations(self) -> None:
        text_model = self._text_model(self.model)
        if text_model is not None and hasattr(text_model, "layers"):
            for layer in text_model.layers:
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is not None:
                    linear_attn.profile_runtime = self.optimization_config.profile_runtime
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and hasattr(self_attn, "profile_runtime"):
                    self_attn.profile_runtime = self.optimization_config.profile_runtime
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, Qwen3SparseMoeBlock):
                    mlp.profile_runtime = self.optimization_config.profile_runtime
        self._maybe_compile_text_model()

    def _maybe_compile_text_model(self) -> None:
        self._compiled_text_forward = None
        compile_mode = self.optimization_config.compile_mode
        if compile_mode == "auto":
            compile_mode = "reduce-overhead"
            logger.info("compile_mode=auto resolved to %s", compile_mode)
        if compile_mode == "none" or not hasattr(torch, "compile") or not hasattr(self.model, "forward_text_only"):
            return
        # XPU-only: Inductor dynamic_shapes pulls in Triton kernels that use fp64 (unsupported on XPU).
        # Give dynamo more cache slots so legitimate shape/batch variants (bs=1 vs scheduler max
        # batch, different prefill lengths) are not evicted into eager fallback mid-request.
        dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_config is not None:
            try:
                if int(getattr(dynamo_config, "cache_size_limit", 8)) < 32:
                    dynamo_config.cache_size_limit = 32
                if int(getattr(dynamo_config, "accumulated_cache_size_limit", 256)) < 512:
                    dynamo_config.accumulated_cache_size_limit = 512
            except (TypeError, ValueError):
                pass
        self._compiled_text_forward = torch.compile(
            self.model.forward_text_only,
            mode=compile_mode,
            fullgraph=self.optimization_config.compile_fullgraph,
        )
        logger.info(
            "Enabled torch.compile for XPU text path: mode=%s fullgraph=%s",
            compile_mode,
            self.optimization_config.compile_fullgraph,
        )

    def _kv_cache_runtime_info(self) -> dict[str, object]:
        info_getter = getattr(self.model, "kv_cache_runtime_info", None)
        if callable(info_getter):
            info = info_getter()
            if isinstance(info, dict):
                return info
        turboquant_enabled = self.optimization_config.kv_cache_quantization == "turboquant"
        full_attention_layers = [
            layer_idx
            for layer_idx, layer_type in enumerate(self.config.text_config.layer_types)
            if layer_type == "full_attention"
        ]
        return {
            "mode": self.optimization_config.kv_cache_quantization,
            "turboquant_enabled": turboquant_enabled,
            "turboquant_bits": self.optimization_config.kv_cache_quant_bits if turboquant_enabled else None,
            "turboquant_residual_len": self.optimization_config.kv_cache_residual_len if turboquant_enabled else None,
            "turboquant_preset": self.optimization_config.kv_cache_turboquant_preset if turboquant_enabled else None,
            "turboquant_applies_to": "full_attention_only" if turboquant_enabled else "disabled",
            "full_attention_layers": len(full_attention_layers),
            "full_attention_layer_indices": full_attention_layers,
            "turboquant_quantized_layers": len(full_attention_layers) if turboquant_enabled else 0,
            "turboquant_quantized_layer_indices": full_attention_layers if turboquant_enabled else [],
            "prompt_cache_vs_prefix": {
                "prompt_cache_size": self.optimization_config.prompt_cache_size,
                "prompt_cache_max_tokens": self.optimization_config.prompt_cache_max_tokens,
                "prefer_prompt_cache_over_prefix": self.optimization_config.prefer_prompt_cache_over_prefix,
                "prefix_kv_share_env": os.environ.get("ANNA_PREFIX_KV_SHARE", "1"),
                "strategy": (
                    "exact_prompt_cache_skips_prefix_registration"
                    if (
                        self.optimization_config.prompt_cache_size > 0
                        and self.optimization_config.prefer_prompt_cache_over_prefix
                    )
                    else "prefix_block_sharing"
                ),
            },
        }

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
        kv_cache_quant_bits_explicit: bool = False,
        kv_cache_residual_len_explicit: bool = False,
        safety_policy: RuntimeSafetyPolicy | None = None,
        default_max_completion_tokens: int | None = None,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_top_k: int | None = None,
        default_min_p: float | None = None,
        default_presence_penalty: float | None = None,
        default_repetition_penalty: float | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        offload_mode: str = "auto",
        offload_vision: bool = False,
        expert_quant: str = "auto",
        weight_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
        decode_executor: str | None = None,
    ) -> "AnnaQwen3_5TextEngine":
        model_path = Path(model_dir)
        config = load_qwen3_5_text_model_config(model_path)
        device_context = DeviceContext.resolve(
            device=device,
            dtype=dtype,
            model_dtype=config.text_config.dtype,
        )
        if device_context.device.type != "xpu":
            raise ValueError(
                "AnnaQwen3_5TextEngine requires Intel XPU (device='xpu', or device='auto' when torch.xpu is available). "
                f"Resolved device {device_context.device!r} from device={device!r}."
            )
        if safety_policy is not None:
            device_context.safety_policy = safety_policy
        weight_bytes = estimate_qwen3_5_text_model_weight_bytes(model_path)
        (
            resolved_kv_cache_quantization,
            kv_cache_quant_bits,
            kv_cache_residual_len,
            turboquant_preset,
        ) = cls._resolve_kv_cache_quantization(
            requested_mode=kv_cache_quantization,
            requested_bits=kv_cache_quant_bits,
            requested_residual_len=kv_cache_residual_len,
            bits_explicit=kv_cache_quant_bits_explicit,
            residual_explicit=kv_cache_residual_len_explicit,
            weight_bytes=weight_bytes,
            device_context=device_context,
        )
        resolved_offload_mode = cls._resolve_offload_mode(
            requested_mode=offload_mode,
            model_path=model_path,
            config=config,
            device_context=device_context,
        )
        resolved_offload_vision = cls._resolve_offload_vision(
            requested_offload_vision=offload_vision,
            resolved_offload_mode=resolved_offload_mode,
            config=config,
        )
        resolved_expert_quant = cls._resolve_expert_quant(
            requested_quant=expert_quant,
            resolved_offload_mode=resolved_offload_mode,
        )
        resolved_weight_quant = cls._resolve_weight_quant(
            requested_quant=weight_quant,
            resolved_offload_mode=resolved_offload_mode,
            model_path=model_path,
            config=config,
            device_context=device_context,
        )
        resolved_resident_expert_layer_indices = cls._resolve_resident_expert_layer_indices(
            requested_layers=resident_expert_layers,
            requested_indices=resident_expert_layer_indices,
            config=config,
            resolved_offload_mode=resolved_offload_mode,
        )
        resolved_cached_experts_per_layer = cls._resolve_cached_experts_per_layer(
            requested_cached_experts_per_layer=cached_experts_per_layer,
            resolved_offload_mode=resolved_offload_mode,
        )
        model_device = (
            torch.device("cpu")
            if resolved_offload_mode == "experts" or resolved_weight_quant == "int4"
            else device_context.device
        )
        uses_gguf_weights = has_gguf_model(model_path)

        def _direct_int4_placeholder_predicate(module_name: str, _module: torch.nn.Module) -> bool:
            normalized = module_name.replace("\\", "/")
            if ".visual." in normalized or normalized.startswith("model.visual."):
                return False
            if ".mlp._expert_cache." in normalized:
                return False
            if ".mlp.experts." in normalized:
                return resolved_expert_quant == "int4"
            return resolved_weight_quant == "int4"

        try:
            logger.info(
                "Building Qwen3.5 runtime: model_dir=%s compute_device=%s load_device=%s offload=%s expert_quant=%s weight_quant=%s resident_expert_layers=%s cached_experts_per_layer=%s kv_cache=%s",
                model_path,
                device_context.device,
                model_device,
                resolved_offload_mode,
                resolved_expert_quant,
                resolved_weight_quant,
                0 if resolved_resident_expert_layer_indices is None else len(resolved_resident_expert_layer_indices),
                resolved_cached_experts_per_layer,
                resolved_kv_cache_quantization,
            )
            model, model_quantized_replacements = build_qwen3_5_text_model(
                config,
                device=model_device,
                dtype=device_context.dtype,
                int4_placeholder_predicate=(
                    _direct_int4_placeholder_predicate
                    if uses_gguf_weights or resolved_weight_quant == "int4"
                    else None
                ),
            )
            report = load_qwen3_5_text_model_weights(model, model_path)
            logger.info(
                "Finished loading Qwen3.5 weights: tensors_loaded=%s tensors_skipped=%s quantized_placeholders=%s",
                report.loaded,
                report.skipped,
                model_quantized_replacements,
            )
            runtime_weight_quantized_replacements = 0
            if resolved_weight_quant == "int4":
                runtime_weight_quantized_replacements = cls._apply_runtime_weight_quantization(
                    model=model,
                    device=device_context.device,
                    compute_dtype=device_context.dtype,
                    cache_dir=resolve_xpu_int4_layout_cache_dir(model_path=model_path),
                )
                release_conversion_artifacts(device_context.device)
            total_quantized_replacements = model_quantized_replacements + runtime_weight_quantized_replacements
            report.quantized_replacements = total_quantized_replacements
            auto_resident_indices = resolved_resident_expert_layer_indices is None
            auto_cached_experts_per_layer = resolved_cached_experts_per_layer is None
            initial_resident_expert_layer_indices = () if auto_resident_indices else resolved_resident_expert_layer_indices
            initial_cached_experts_per_layer = 0 if auto_cached_experts_per_layer else resolved_cached_experts_per_layer

            logger.info(
                "Configuring Qwen3.5 runtime placement on %s: offload_experts=%s offload_vision=%s resident_expert_indices=%s cached_experts_per_layer=%s",
                device_context.device,
                resolved_offload_mode == "experts",
                resolved_offload_vision,
                list(initial_resident_expert_layer_indices),
                initial_cached_experts_per_layer,
            )
            model.configure_runtime(
                device_context.device,
                offload_experts=resolved_offload_mode == "experts",
                offload_vision=resolved_offload_vision,
                offload_token_io=False,
                resident_expert_layers=0,
                resident_expert_layer_indices=initial_resident_expert_layer_indices,
                expert_quant=resolved_expert_quant,
                cached_experts_per_layer=initial_cached_experts_per_layer,
                kv_cache_quantization=resolved_kv_cache_quantization,
                kv_cache_quant_bits=kv_cache_quant_bits,
                kv_cache_residual_len=kv_cache_residual_len,
            )
            int4_cache_dir = resolve_xpu_int4_layout_cache_dir(model_path=model_path)
            logger.info("Preparing loaded quantized Qwen3.5 modules for XPU execution.")
            cls._prepare_loaded_quantized_modules_for_execution(
                model=model,
                config=config,
                device=device_context.device,
                cache_dir=int4_cache_dir,
            )
            if auto_resident_indices:
                resolved_resident_expert_layer_indices = cls._estimate_resident_expert_layer_indices(
                    model=model,
                    device_context=device_context,
                    expert_quant=resolved_expert_quant,
                )
                model.configure_runtime(
                    device_context.device,
                    offload_experts=resolved_offload_mode == "experts",
                    offload_vision=resolved_offload_vision,
                    offload_token_io=False,
                    resident_expert_layers=0,
                    resident_expert_layer_indices=resolved_resident_expert_layer_indices,
                    expert_quant=resolved_expert_quant,
                    cached_experts_per_layer=initial_cached_experts_per_layer,
                    kv_cache_quantization=resolved_kv_cache_quantization,
                    kv_cache_quant_bits=kv_cache_quant_bits,
                    kv_cache_residual_len=kv_cache_residual_len,
                )
                cls._prepare_loaded_quantized_modules_for_execution(
                    model=model,
                    config=config,
                    device=device_context.device,
                    cache_dir=int4_cache_dir,
                )
            if auto_cached_experts_per_layer:
                resolved_cached_experts_per_layer = cls._estimate_cached_experts_per_layer(
                    model=model,
                    device_context=device_context,
                    expert_quant=resolved_expert_quant,
                )
                model.configure_runtime(
                    device_context.device,
                    offload_experts=resolved_offload_mode == "experts",
                    offload_vision=resolved_offload_vision,
                    offload_token_io=False,
                    resident_expert_layers=0,
                    resident_expert_layer_indices=resolved_resident_expert_layer_indices,
                    expert_quant=resolved_expert_quant,
                    cached_experts_per_layer=resolved_cached_experts_per_layer,
                    kv_cache_quantization=resolved_kv_cache_quantization,
                    kv_cache_quant_bits=kv_cache_quant_bits,
                    kv_cache_residual_len=kv_cache_residual_len,
                )
                cls._prepare_loaded_quantized_modules_for_execution(
                    model=model,
                    config=config,
                    device=device_context.device,
                    cache_dir=int4_cache_dir,
                )
            elif (
                resolved_offload_mode == "experts"
                and resolved_expert_quant == "int4"
                and device_context.device.type == "xpu"
                and int(resolved_cached_experts_per_layer or 0) > 0
            ):
                promoted_cached_experts_per_layer = cls._estimate_cached_experts_per_layer(
                    model=model,
                    device_context=device_context,
                    expert_quant=resolved_expert_quant,
                )
                if promoted_cached_experts_per_layer > int(resolved_cached_experts_per_layer or 0):
                    logger.info(
                        "Promoting XPU expert cache capacity from requested=%s to effective=%s based on free device memory.",
                        resolved_cached_experts_per_layer,
                        promoted_cached_experts_per_layer,
                    )
                    resolved_cached_experts_per_layer = promoted_cached_experts_per_layer
                    model.configure_runtime(
                        device_context.device,
                        offload_experts=resolved_offload_mode == "experts",
                        offload_vision=resolved_offload_vision,
                        offload_token_io=False,
                        resident_expert_layers=0,
                        resident_expert_layer_indices=resolved_resident_expert_layer_indices,
                        expert_quant=resolved_expert_quant,
                        cached_experts_per_layer=resolved_cached_experts_per_layer,
                        kv_cache_quantization=resolved_kv_cache_quantization,
                        kv_cache_quant_bits=kv_cache_quant_bits,
                        kv_cache_residual_len=kv_cache_residual_len,
                    )
            resolved_cached_experts_per_layer = cls._effective_cached_experts_per_layer(model)
            model.eval()
            release_conversion_artifacts(device_context.device)
            logger.info(
                "Post-load Qwen3.5 CPU tensor residency: %s",
                format_bytes(_module_cpu_tensor_bytes(model)),
            )
        except RuntimeError as exc:
            if device_context.should_recover(exc):
                try:
                    device_context.recover()
                except Exception:  # pragma: no cover - best-effort recovery
                    logger.exception("Failed to recover device context after model load failure.")
            raise

        tokenizer = Qwen3_5TextTokenizer.from_model_dir(model_path)
        processor = Qwen3_5TextMultimodalProcessor(config, tokenizer)
        resolved_model_id = model_id or model_path.name
        resolved_default_max_completion_tokens = (
            config.default_max_completion_tokens
            if default_max_completion_tokens is None
            else default_max_completion_tokens
        )
        if resolved_default_max_completion_tokens is not None:
            resolved_default_max_completion_tokens = max(1, int(resolved_default_max_completion_tokens))

        _, _maintain_full_attn_mirror = _qwen3_paged_full_attention_decode_enabled(
            device_type=device_context.device.type,
            kv_cache_quantization=resolved_kv_cache_quantization,
        )
        _logged_full_attention_cache_mirror = _maintain_full_attn_mirror and bool(
            config.text_config.layer_types and "full_attention" in config.text_config.layer_types
        )

        logger.info(
            "Loaded model %s on %s (compute=%s, requested=%s, default_max_completion_tokens=%s, default_temperature=%s, default_top_p=%s, default_top_k=%s, default_min_p=%s, default_presence_penalty=%s, default_repetition_penalty=%s, default_enable_thinking=%s, reasoning_format=%s, offload=%s, offload_vision=%s, expert_quant=%s, weight_quant=%s, resident_expert_layers=%s, resident_expert_layer_indices=%s, cached_experts_per_layer=%s, full_attention_cache_mirror=%s, weight_load_device=%s); tensors loaded=%s skipped=%s quantized=%s",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.reported_dtype,
            resolved_default_max_completion_tokens,
            default_temperature if default_temperature is not None else 0.7,
            default_top_p if default_top_p is not None else 0.8,
            default_top_k if default_top_k is not None else 20,
            default_min_p if default_min_p is not None else 0.0,
            default_presence_penalty if default_presence_penalty is not None else 1.5,
            default_repetition_penalty if default_repetition_penalty is not None else 1.0,
            bool(default_enable_thinking),
            normalize_reasoning_format(reasoning_format),
            resolved_offload_mode,
            resolved_offload_vision,
            resolved_expert_quant,
            resolved_weight_quant,
            len(resolved_resident_expert_layer_indices or ()),
            list(resolved_resident_expert_layer_indices or ()),
            resolved_cached_experts_per_layer,
            _logged_full_attention_cache_mirror,
            model_device,
            report.loaded,
            report.skipped,
            report.quantized_replacements,
        )

        if resolved_offload_vision:
            device_context.migration_policy.keep_media_on_host = True

        engine = cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_id=resolved_model_id,
            device_context=device_context,
            quantized_replacements=report.quantized_replacements,
            default_max_completion_tokens=resolved_default_max_completion_tokens,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_top_k=default_top_k,
            default_min_p=default_min_p,
            default_presence_penalty=default_presence_penalty,
            default_repetition_penalty=default_repetition_penalty,
            default_enable_thinking=default_enable_thinking,
            reasoning_format=reasoning_format,
            offload_mode=resolved_offload_mode,
            offload_vision=resolved_offload_vision,
            expert_quant=resolved_expert_quant,
            weight_quant=resolved_weight_quant,
            resident_expert_layer_indices=tuple(resolved_resident_expert_layer_indices or ()),
            cached_experts_per_layer=int(resolved_cached_experts_per_layer or 0),
            optimization_config=EngineOptimizationConfig(
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                prefill_chunk_size=prefill_chunk_size,
                prompt_cache_size=prompt_cache_size,
                prompt_cache_max_tokens=prompt_cache_max_tokens,
                profile_runtime=profile_runtime,
                kv_cache_quantization=resolved_kv_cache_quantization,
                kv_cache_quant_bits=kv_cache_quant_bits,
                kv_cache_residual_len=kv_cache_residual_len,
                kv_cache_turboquant_preset=turboquant_preset,
                prefer_prompt_cache_over_prefix=prompt_cache_size > 0,
                decode_executor=decode_executor or "auto",
            ),
        )
        kv_cache_info = engine._kv_cache_runtime_info()
        logger.info(
            "Qwen3.5 KV cache runtime: mode=%s turboquant_enabled=%s turboquant_bits=%s "
            "turboquant_residual_len=%s turboquant_preset=%s full_attention_layers=%s "
            "turboquant_quantized_layers=%s prompt_cache_size=%s prefer_prompt_cache_over_prefix=%s",
            kv_cache_info.get("mode"),
            kv_cache_info.get("turboquant_enabled"),
            kv_cache_info.get("turboquant_bits"),
            kv_cache_info.get("turboquant_residual_len"),
            kv_cache_info.get("turboquant_preset"),
            kv_cache_info.get("full_attention_layers"),
            kv_cache_info.get("turboquant_quantized_layers"),
            prompt_cache_size,
            engine.optimization_config.prefer_prompt_cache_over_prefix,
        )
        return engine

    @staticmethod
    def _resolve_offload_mode(
        *,
        requested_mode: str,
        model_path: Path,
        config: object,
        device_context: DeviceContext,
    ) -> str:
        normalized = requested_mode.lower()
        if normalized not in {"auto", "none", "experts"}:
            raise ValueError(f"Unsupported offload mode: {requested_mode}")
        if normalized != "auto":
            return normalized

        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return "none"

        weight_bytes = estimate_qwen3_5_text_model_weight_bytes(model_path)
        if config.text_config.is_moe_model and weight_bytes > int(memory_info.total_bytes * 0.85):
            return "experts"
        return "none"

    @staticmethod
    def _resolve_kv_cache_quantization(
        *,
        requested_mode: str,
        requested_bits: int = 4,
        requested_residual_len: int = 128,
        bits_explicit: bool = False,
        residual_explicit: bool = False,
        weight_bytes: int | None = None,
        device_context: DeviceContext | None = None,
    ) -> tuple[str, int, int, str | None]:
        del device_context  # reserved for future device-specific TurboQuant defaults
        try:
            return resolve_turboquant_runtime_settings(
                requested_mode=requested_mode,
                requested_bits=requested_bits,
                requested_residual_len=requested_residual_len,
                weight_bytes=weight_bytes,
                bits_explicit=bits_explicit,
                residual_explicit=residual_explicit,
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _resolve_offload_vision(
        *,
        requested_offload_vision: bool,
        resolved_offload_mode: str,
        config: object,
    ) -> bool:
        if getattr(config, "vision_config", None) is None:
            return False
        return bool(requested_offload_vision or resolved_offload_mode == "experts")

    @staticmethod
    def _resolve_expert_quant(
        *,
        requested_quant: str,
        resolved_offload_mode: str,
    ) -> str:
        normalized = requested_quant.lower()
        if normalized not in {"auto", "none", "int4"}:
            raise ValueError(f"Unsupported expert quant mode: {requested_quant}")
        if resolved_offload_mode != "experts":
            return "none"
        if normalized == "auto":
            return "int4"
        return normalized

    @staticmethod
    def _resolve_weight_quant(
        *,
        requested_quant: str,
        resolved_offload_mode: str,
        model_path: Path,
        config: object,
        device_context: DeviceContext,
    ) -> str:
        normalized = requested_quant.lower()
        if normalized not in {"auto", "none", "int4"}:
            raise ValueError(f"Unsupported weight quant mode: {requested_quant}")
        if normalized != "auto":
            return normalized
        if getattr(config, "quantization_config", None) is not None and config.quantization_config.is_enabled:
            return "none"

        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return "none"

        weight_bytes = estimate_qwen3_5_text_model_weight_bytes(model_path)
        is_moe_or_expert_offload = resolved_offload_mode == "experts" or bool(
            getattr(config.text_config, "is_moe_model", False)
        )
        usage_threshold = weight_quant_auto_usage_threshold(is_moe_or_expert_offload=is_moe_or_expert_offload)
        if weight_bytes > int(memory_info.total_bytes * usage_threshold):
            return "int4"
        return "none"

    @classmethod
    def _apply_runtime_weight_quantization(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        device: torch.device,
        compute_dtype: torch.dtype,
        cache_dir: Path | None = None,
    ) -> int:
        def _should_quantize(module_name: str, _module: torch.nn.Module) -> bool:
            normalized = module_name.replace("\\", "/")
            return (
                ".visual." not in normalized
                and not normalized.startswith("model.visual.")
                and ".mlp.experts." not in normalized
                and ".mlp._expert_cache." not in normalized
            )

        resolved_cache_dir = resolve_xpu_int4_layout_cache_dir(cache_dir)
        replacements = convert_module_linears_to_xpu_int4(
            model,
            compute_dtype=compute_dtype,
            device=device,
            include_predicate=_should_quantize,
            cache_dir=resolved_cache_dir,
        )
        logger.info(
            "Runtime dense XPU int4 quantization: replacements=%s device=%s compute_dtype=%s cache_dir=%s",
            replacements,
            device,
            compute_dtype,
            resolved_cache_dir if resolved_cache_dir is not None else "disabled",
        )
        return replacements

    @classmethod
    def _prepare_loaded_quantized_modules_for_execution(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        config: object,
        device: torch.device,
        cache_dir: Path | None = None,
    ) -> int:
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = (getattr(quantization_config, "quant_method", None) or "").strip().lower()
        if quant_method not in {"auto-round", "auto_round"}:
            return 0

        packing_format = (getattr(quantization_config, "packing_format", None) or "").strip().lower()
        if packing_format != "auto_round:auto_gptq":
            raise ValueError(
                f"Unsupported AutoRound packing format at runtime: {getattr(quantization_config, 'packing_format', None)!r}"
            )

        replacements = convert_module_linears_to_xpu_int4(
            model,
            device=device,
            include_predicate=lambda _module_name, module: (
                isinstance(module, AutoRoundGPTQLinear) and module.qweight.device.type == "xpu"
            ),
            cache_dir=resolve_xpu_int4_layout_cache_dir(cache_dir),
        )
        if replacements > 0:
            logger.info(
                "Prepared AutoRound modules for XPU execution: replacements=%s device=%s packing_format=%s",
                replacements,
                device,
                packing_format,
            )
        return replacements

    @staticmethod
    def _resolve_cached_experts_per_layer(
        *,
        requested_cached_experts_per_layer: int | None,
        resolved_offload_mode: str,
    ) -> int | None:
        if resolved_offload_mode != "experts":
            return 0
        if requested_cached_experts_per_layer is None:
            return None
        return max(0, int(requested_cached_experts_per_layer))

    @staticmethod
    def _sparse_moe_layer_indices(config: object) -> list[int]:
        return [
            layer_idx
            for layer_idx in range(config.text_config.num_hidden_layers)
            if config.text_config.uses_sparse_moe(layer_idx)
        ]

    @classmethod
    def _validate_resident_expert_layer_indices(
        cls,
        *,
        requested_indices: tuple[int, ...],
        config: object,
    ) -> tuple[int, ...]:
        if not requested_indices:
            return ()

        num_hidden_layers = int(config.text_config.num_hidden_layers)
        sparse_layer_indices = cls._sparse_moe_layer_indices(config)
        sparse_layer_index_set = set(sparse_layer_indices)
        requested_set: set[int] = set()
        for layer_idx in requested_indices:
            index = int(layer_idx)
            if index < 0 or index >= num_hidden_layers:
                raise ValueError(f"Resident expert layer index out of range: {index}")
            if index not in sparse_layer_index_set:
                raise ValueError(f"Decoder layer {index} does not use sparse MoE experts.")
            requested_set.add(index)
        return tuple(layer_idx for layer_idx in sparse_layer_indices if layer_idx in requested_set)

    @classmethod
    def _resolve_resident_expert_layer_indices(
        cls,
        *,
        requested_layers: int | None,
        requested_indices: tuple[int, ...] | None,
        config: object,
        resolved_offload_mode: str,
    ) -> tuple[int, ...] | None:
        if resolved_offload_mode != "experts":
            return ()

        if requested_indices is not None:
            return cls._validate_resident_expert_layer_indices(
                requested_indices=requested_indices,
                config=config,
            )

        if requested_layers is None:
            return None

        requested = max(0, int(requested_layers))
        if requested == 0:
            return ()

        sparse_layer_indices = cls._sparse_moe_layer_indices(config)
        return tuple(sparse_layer_indices[:requested])

    @staticmethod
    def _module_nbytes(module: torch.nn.Module) -> int:
        total = 0
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            total += tensor.nelement() * tensor.element_size()
        return total

    @staticmethod
    def _text_model(model: Qwen3_5TextForConditionalGeneration) -> object | None:
        text_model = getattr(getattr(model, "model", None), "language_model", None)
        if text_model is None:
            text_model = getattr(model, "model", None)
        return text_model

    @classmethod
    def _offloaded_sparse_moe_blocks(cls, model: Qwen3_5TextForConditionalGeneration) -> list[tuple[int, Qwen3SparseMoeBlock]]:
        text_model = cls._text_model(model)
        if text_model is None or not hasattr(text_model, "layers"):
            return []
        blocks: list[tuple[int, Qwen3SparseMoeBlock]] = []
        for layer_idx, layer in enumerate(text_model.layers):
            if isinstance(getattr(layer, "mlp", None), Qwen3SparseMoeBlock) and layer.mlp.offload_experts:
                blocks.append((layer_idx, layer.mlp))
        return blocks

    @classmethod
    def _effective_cached_experts_per_layer(cls, model: Qwen3_5TextForConditionalGeneration) -> int:
        offloaded_blocks = cls._offloaded_sparse_moe_blocks(model)
        if offloaded_blocks:
            return int(offloaded_blocks[0][1].cached_experts_per_layer)

        text_model = cls._text_model(model)
        if text_model is None or not hasattr(text_model, "layers"):
            return 0
        for layer in text_model.layers:
            if isinstance(getattr(layer, "mlp", None), Qwen3SparseMoeBlock):
                return int(layer.mlp.cached_experts_per_layer)
        return 0

    @staticmethod
    def _estimate_resident_budget_bytes(
        *,
        memory_info: object,
        safety: object,
        expert_quant: str,
    ) -> tuple[int, int, float]:
        if expert_quant == "int4":
            target_free_bytes = max(2304 << 20, int(memory_info.total_bytes * 0.16))
            budget_factor = 1.10
            budget_bytes = int(max(0, int(memory_info.free_bytes) - target_free_bytes) / budget_factor)
            return budget_bytes, target_free_bytes, budget_factor

        reserve_bytes = max(
            int(safety.min_free_bytes),
            int(safety.reserve_margin_bytes),
            int(memory_info.total_bytes * (1.0 - safety.max_estimated_usage_ratio)),
        )
        budget_factor = max(1.0, float(safety.generation_memory_safety_factor))
        budget_bytes = int(max(0, int(memory_info.free_bytes) - reserve_bytes) / budget_factor)
        return budget_bytes, reserve_bytes, budget_factor

    @staticmethod
    def select_resident_layers_by_heat(
        *,
        layer_candidates: list[tuple[int, int, float]],
        budget_bytes: int,
    ) -> tuple[int, ...]:
        """Pick resident sparse layers under a byte budget, preferring higher routing heat.

        ``layer_candidates`` is a list of ``(layer_idx, layer_bytes, heat)``.
        When all heats are zero (cold start), falls back to sequential first-fit so
        startup behaviour matches the historical front-N policy.
        """
        if budget_bytes <= 0 or not layer_candidates:
            return ()

        has_heat = any(heat > 0.0 for _, _, heat in layer_candidates)
        if has_heat:
            ordered = sorted(
                layer_candidates,
                key=lambda item: (item[2], -item[0]),
                reverse=True,
            )
        else:
            ordered = sorted(layer_candidates, key=lambda item: item[0])

        selected: list[int] = []
        consumed = 0
        for layer_idx, layer_bytes, _heat in ordered:
            if layer_bytes <= 0:
                continue
            if consumed + layer_bytes > budget_bytes:
                continue
            selected.append(int(layer_idx))
            consumed += int(layer_bytes)
        # Keep decoder order in the returned indices for stable logging/configure.
        return tuple(sorted(selected))

    @classmethod
    def _collect_sparse_layer_candidates(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        expert_quant: str,
        layer_heats: dict[int, float] | None = None,
    ) -> list[tuple[int, int, float]]:
        text_model = cls._text_model(model)
        if text_model is None or not hasattr(text_model, "layers"):
            return []
        heats = layer_heats or {}
        candidates: list[tuple[int, int, float]] = []
        for layer_idx, layer in enumerate(text_model.layers):
            if not isinstance(layer.mlp, Qwen3SparseMoeBlock):
                continue
            layer_bytes = (
                estimate_module_xpu_int4_bytes(layer.mlp.experts)
                if expert_quant == "int4"
                else cls._module_nbytes(layer.mlp.experts)
            )
            heat = float(heats.get(layer_idx, getattr(layer.mlp, "_layer_heat", 0.0) or 0.0))
            candidates.append((layer_idx, int(layer_bytes), heat))
        return candidates

    @classmethod
    def _estimate_resident_expert_layer_indices(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        device_context: DeviceContext,
        expert_quant: str,
        layer_heats: dict[int, float] | None = None,
    ) -> tuple[int, ...]:
        device_context.synchronize()
        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return ()

        text_model = cls._text_model(model)
        if text_model is None or not hasattr(text_model, "layers"):
            return ()

        safety = device_context.safety_policy
        budget_bytes, reserve_bytes, budget_factor = cls._estimate_resident_budget_bytes(
            memory_info=memory_info,
            safety=safety,
            expert_quant=expert_quant,
        )
        if budget_bytes <= 0:
            logger.info(
                "Auto resident expert placement skipped: expert_quant=%s free=%s reserve=%s budget_factor=%.2f budget=%s",
                expert_quant,
                format_bytes(memory_info.free_bytes),
                format_bytes(reserve_bytes),
                budget_factor,
                format_bytes(budget_bytes),
            )
            return ()

        layer_candidates = cls._collect_sparse_layer_candidates(
            model=model,
            expert_quant=expert_quant,
            layer_heats=layer_heats,
        )
        selected_indices = cls.select_resident_layers_by_heat(
            layer_candidates=layer_candidates,
            budget_bytes=budget_bytes,
        )
        selected_set = set(selected_indices)
        consumed_bytes = sum(layer_bytes for layer_idx, layer_bytes, _ in layer_candidates if layer_idx in selected_set)
        heat_map = {layer_idx: heat for layer_idx, _bytes, heat in layer_candidates if heat > 0.0}

        logger.info(
            "Auto resident expert placement: expert_quant=%s free=%s reserve=%s budget_factor=%.2f budget=%s selected_layers=%s selected_bytes=%s heat_guided=%s candidate_layer_bytes=%s",
            expert_quant,
            format_bytes(memory_info.free_bytes),
            format_bytes(reserve_bytes),
            budget_factor,
            format_bytes(budget_bytes),
            list(selected_indices),
            format_bytes(consumed_bytes),
            bool(heat_map),
            {
                layer_idx: format_bytes(layer_bytes)
                for layer_idx, layer_bytes, _heat in layer_candidates[:8]
            },
        )
        return selected_indices

    @classmethod
    def _estimate_cached_experts_per_layer(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        device_context: DeviceContext,
        expert_quant: str,
    ) -> int:
        offloaded_blocks = cls._offloaded_sparse_moe_blocks(model)
        if not offloaded_blocks:
            return 0

        device_context.synchronize()
        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return 0

        exemplar_block = offloaded_blocks[0][1]
        exemplar_expert = exemplar_block.experts[0]
        per_expert_bytes = (
            estimate_module_xpu_int4_bytes(exemplar_expert)
            if expert_quant == "int4"
            else cls._module_nbytes(exemplar_expert)
        )
        if per_expert_bytes <= 0:
            return max(exemplar_block.top_k, 0)

        safety = device_context.safety_policy
        reserve_bytes = max(
            int(safety.min_free_bytes),
            int(safety.reserve_margin_bytes),
            int(memory_info.total_bytes * (1.0 - safety.max_estimated_usage_ratio)),
        )
        if expert_quant == "int4":
            cache_target_free_bytes = max(reserve_bytes, 1536 << 20, int(memory_info.total_bytes * 0.10))
            cache_budget_fraction = 0.85
            budget_factor = max(1.0, min(float(safety.generation_memory_safety_factor), 1.25))
            minimum_cache = max(exemplar_block.top_k, exemplar_block.top_k * 8)
        else:
            cache_target_free_bytes = max(reserve_bytes, 768 << 20, int(memory_info.total_bytes * 0.06))
            cache_budget_fraction = 0.35
            budget_factor = max(1.0, float(safety.generation_memory_safety_factor))
            minimum_cache = exemplar_block.top_k
        cache_budget_bytes = int(
            max(0, int(memory_info.free_bytes) - cache_target_free_bytes) * cache_budget_fraction / budget_factor
        )
        auto_cached = cache_budget_bytes // max(1, per_expert_bytes * len(offloaded_blocks))

        max_cache = exemplar_block.num_experts
        minimum_budget_bytes = per_expert_bytes * len(offloaded_blocks) * minimum_cache
        if cache_budget_bytes < minimum_budget_bytes:
            resolved = max(0, min(max_cache, auto_cached))
        else:
            resolved = max(minimum_cache, min(max_cache, auto_cached))

        logger.info(
            "Auto expert cache sizing: expert_quant=%s free=%s target_free=%s cache_budget_fraction=%.2f budget_factor=%.2f cache_budget=%s offloaded_layers=%s per_expert=%s minimum_cache=%s cached_experts_per_layer=%s",
            expert_quant,
            format_bytes(memory_info.free_bytes),
            format_bytes(cache_target_free_bytes),
            cache_budget_fraction,
            budget_factor,
            format_bytes(cache_budget_bytes),
            len(offloaded_blocks),
            format_bytes(per_expert_bytes),
            minimum_cache,
            resolved,
        )
        return resolved

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def _attach_cache_allocator(self) -> None:
        text_model = getattr(getattr(self.model, "model", None), "language_model", None)
        if text_model is None:
            text_model = getattr(self.model, "model", None)
        if text_model is not None and hasattr(text_model, "cache_allocator"):
            text_model.cache_allocator = self.cache_allocator

    def _reserve_prefill_cache(self, prepared: PreparedInputs) -> Qwen3DynamicCache | None:
        config = getattr(self, "config", None)
        allocator = getattr(self, "cache_allocator", None)
        text_config = getattr(config, "text_config", None)
        if text_config is None or allocator is None:
            return None
        batch_size = int(prepared.input_ids.shape[0])
        cache = Qwen3DynamicCache(
            text_config,
            allocator=allocator,
            batch_size=batch_size,
            kv_cache_quantization=self.optimization_config.kv_cache_quantization,
            kv_cache_quant_bits=self.optimization_config.kv_cache_quant_bits,
            kv_cache_residual_len=self.optimization_config.kv_cache_residual_len,
        )
        cache.reserve_sequence_capacity(int(prepared.input_ids.shape[1]))
        # Exact prompt-cache eligible prompts skip prefix registration (dual-cache boundary).
        if not self._should_skip_prefix_registration(prepared):
            cache.set_prompt_token_ids(prepared.input_ids)
        return cache

    def _attach_prefix_share_gate(self) -> None:
        """Wire model-created caches to the same prompt-cache vs prefix boundary policy."""

        def _allow_prefix_share(input_ids: torch.Tensor) -> bool:
            if int(input_ids.shape[0]) != 1:
                return True
            if self.optimization_config.prompt_cache_size <= 0:
                return True
            if not self.optimization_config.prefer_prompt_cache_over_prefix:
                return True
            max_tokens = self.optimization_config.prompt_cache_max_tokens
            seq = int(input_ids.shape[1])
            if max_tokens > 0 and seq > max_tokens:
                return True  # long prompts are not exact-cached → allow prefix sharing
            return False  # exact-cache eligible → skip prefix registration

        language_model = getattr(getattr(self.model, "model", None), "language_model", None)
        if language_model is None:
            language_model = getattr(self.model, "model", None)
        if language_model is not None:
            language_model._anna_prefix_share_gate = _allow_prefix_share  # type: ignore[attr-defined]
        # Also attach on the outer conditional-generation module used by multimodal forward.
        setattr(self.model, "_anna_prefix_share_gate", _allow_prefix_share)

    def _trim_runtime_cache_if_idle(self) -> None:
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            snapshot = metrics.snapshot()
            if snapshot.running_requests > 0 or snapshot.waiting_requests > 0:
                return
        trimmed_pages = self.cache_allocator.trim()
        # P2-#6: adaptive release - when free memory is below the admission safety
        # floor after a request finished, actively empty the device allocator cache
        # instead of waiting for the idle sweeper (releases fragmented segments).
        memory_info = None
        get_memory_info = getattr(self.device_context, "get_memory_info", None)
        if callable(get_memory_info):
            memory_info = get_memory_info()
        memory_pressure = (
            memory_info is not None
            and memory_info.free_bytes < self.device_context.safety_policy.min_free_bytes
        )
        release_unused_memory = getattr(self.device_context, "release_unused_memory", None)
        if trimmed_pages <= 0 and not memory_pressure:
            return
        if callable(release_unused_memory):
            release_unused_memory()
        if trimmed_pages > 0:
            logger.info("Trimmed idle KV cache pages: released_pages=%s", trimmed_pages)
        elif memory_info is not None:
            reserved = memory_info.reserved_bytes
            allocated = memory_info.allocated_bytes
            if reserved is not None and allocated is not None:
                logger.debug(
                    "Memory pressure release: free=%.0fMiB reserved=%.2fGiB allocated=%.2fGiB",
                    memory_info.free_bytes >> 20,
                    (reserved - allocated) / (1 << 30),
                    allocated / (1 << 30),
                )

    def _reclaim_runtime_memory_for_admission(self) -> bool:
        released = False
        prompt_cache = getattr(self, "_prompt_cache", None)
        if prompt_cache:
            for key, entry in list(prompt_cache.items()):
                self._evict_prompt_cache_entry(key, entry)
                released = True

        allocator = getattr(self, "cache_allocator", None)
        trim = getattr(allocator, "trim", None)
        if callable(trim):
            try:
                released = int(trim()) > 0 or released
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Failed to trim KV cache allocator during memory admission.", exc_info=True)

        if released:
            release_unused_memory = getattr(self.device_context, "release_unused_memory", None)
            if callable(release_unused_memory):
                release_unused_memory()
            logger.info("Reclaimed runtime caches before memory admission.")
        return released

    def _clear_runtime_caches_after_recover(self, *, reason: str) -> None:
        prompt_cache = getattr(self, "_prompt_cache", None)
        prompt_entries = 0
        if prompt_cache is not None:
            for entry in list(prompt_cache.values()):
                past_key_values = getattr(entry, "past_key_values", None)
                release = getattr(past_key_values, "release", None)
                if callable(release):
                    try:
                        release()
                    except Exception:  # pragma: no cover - best-effort cleanup
                        logger.debug("Failed to release prompt cache entry during XPU recovery.", exc_info=True)
                prompt_entries += 1
            prompt_cache.clear()

        released_pages = 0
        allocator = getattr(self, "cache_allocator", None)
        clear = getattr(allocator, "clear", None)
        if callable(clear):
            try:
                released_pages = int(clear())
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("Failed to clear KV cache allocator after XPU recovery.")
        elif allocator is not None:
            try:
                released_pages = int(allocator.trim())
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("Failed to trim KV cache allocator after XPU recovery.")

        release_unused_memory = getattr(self.device_context, "release_unused_memory", None)
        if callable(release_unused_memory):
            release_unused_memory()
        logger.warning(
            "Cleared runtime caches after XPU recovery: reason=%s prompt_entries=%s released_pages=%s",
            reason,
            prompt_entries,
            released_pages,
        )

    def set_scheduler(self, scheduler: object | None) -> None:
        self.scheduler = scheduler

    def _kv_cache_page_counts(self) -> tuple[int, int, int, int]:
        """Return (used_pages, total_pages, turboquant_rows, turboquant_cached_tokens).

        P2-#7: under TurboQuant KV the full-attention layers bypass the paged
        allocator (quantized rows per request), so pages alone always read 0/0.
        The live-cache registry makes the turboquant usage observable instead.
        """
        used_pages = 0
        total_pages = 0
        for layer in getattr(self.cache_allocator, "layers", ()):
            key_pages = getattr(layer, "key_pages", None)
            if key_pages is None:
                continue
            capacity = int(key_pages.shape[0])
            free_pages = len(getattr(layer, "free_pages", ()))
            used_pages += max(0, min(capacity, capacity - free_pages))
            total_pages += capacity
        turboquant_rows = 0
        turboquant_tokens = 0
        live_caches = getattr(self.cache_allocator, "live_caches", None)
        if callable(live_caches):
            for cache in live_caches():
                rows = getattr(cache, "turboquant_rows", None)
                lengths = getattr(cache, "layer_lengths", None)
                if rows is None or lengths is None:
                    continue
                uses_turboquant = getattr(cache, "uses_turboquant_for_layer", None)
                for layer_idx, layer_rows in enumerate(rows):
                    if callable(uses_turboquant) and not uses_turboquant(layer_idx):
                        continue
                    for batch_idx, row in enumerate(layer_rows):
                        if row is None:
                            continue
                        turboquant_rows += 1
                        try:
                            turboquant_tokens += int(lengths[layer_idx][batch_idx])
                        except (IndexError, TypeError, ValueError):
                            pass
        return used_pages, total_pages, turboquant_rows, turboquant_tokens

    def service_metrics_snapshot(self) -> ServiceMetricsSnapshot:
        metrics = getattr(self, "metrics", None)
        prefix_stats = None
        pool = getattr(getattr(self, "cache_allocator", None), "prefix_block_pool", None)
        if pool is not None and hasattr(pool, "stats"):
            prefix_stats = pool.stats()
            if metrics is not None:
                metrics.set_prefix_block_stats(
                    lookups_total=prefix_stats.lookups_total,
                    hits_total=prefix_stats.hits_total,
                    misses_total=prefix_stats.misses_total,
                    registers_total=prefix_stats.registers_total,
                    entries=prefix_stats.entries,
                )
        snapshot = metrics.snapshot() if metrics is not None else ServiceMetricsSnapshot(timestamp=time.perf_counter())
        used_pages, total_pages, turboquant_rows, turboquant_tokens = self._kv_cache_page_counts()
        extra: dict[str, object] = {
            "kv_cache_used_pages": used_pages,
            "kv_cache_total_pages": total_pages,
            "kv_cache_turboquant_rows": turboquant_rows,
            "kv_cache_turboquant_tokens": turboquant_tokens,
            "prompt_cache_entries": len(getattr(self, "_prompt_cache", {})),
        }
        if prefix_stats is not None and metrics is None:
            extra.update(
                {
                    "prefix_block_lookups_total": prefix_stats.lookups_total,
                    "prefix_block_hits_total": prefix_stats.hits_total,
                    "prefix_block_misses_total": prefix_stats.misses_total,
                    "prefix_block_registers_total": prefix_stats.registers_total,
                    "prefix_block_entries": prefix_stats.entries,
                }
            )
        return replace(snapshot, **extra)  # type: ignore[arg-type]

    def _admission_gate(self) -> RuntimeAdmissionGate:
        gate = getattr(self, "admission_gate", None)
        if isinstance(gate, RuntimeAdmissionGate):
            return gate
        return PROCESS_ADMISSION_GATE

    def ensure_accepting_requests(self) -> None:
        gate = self._admission_gate()
        snap = gate.snapshot()
        if snap.accepting_requests:
            return
        raise AnnaEngineError(
            snap.degradation_reason
            or "Runtime is degraded after a device failure and is not accepting new requests.",
            status_code=503,
            error_type="server_error",
            code="runtime_degraded",
        )

    def health(self) -> dict[str, Any]:
        quant_method = self.config.quantization_config.quant_method or "dense"
        memory_info = self.device_context.get_memory_info()
        service_metrics = self.service_metrics_snapshot()
        kv_cache_runtime_info = self._kv_cache_runtime_info()
        admission = self._admission_gate().to_health_dict()
        status = "ok" if admission.get("accepting_requests", True) else "degraded"
        return {
            "status": status,
            "accepting_requests": bool(admission.get("accepting_requests", True)),
            "runtime_admission": admission,
            "model": self.default_model_id,
            "model_family": self.model_family,
            "device": str(self.device_context.device),
            "compute_dtype": str(self.device_context.dtype),
            "requested_dtype": self.device_context.requested_dtype,
            "reported_dtype": self.device_context.reported_dtype,
            "default_max_completion_tokens": self.default_max_completion_tokens,
            "default_temperature": self.default_temperature,
            "default_top_p": self.default_top_p,
            "default_top_k": self.default_top_k,
            "default_min_p": self.default_min_p,
            "default_presence_penalty": self.default_presence_penalty,
            "default_repetition_penalty": self.default_repetition_penalty,
            "default_enable_thinking": self.default_enable_thinking,
            "reasoning_format": self.reasoning_format,
            "quantization": quant_method,
            "weight_quant": self.weight_quant,
            "quantized_replacements": self.quantized_replacements,
            "offload_mode": self.offload_mode,
            "offload_vision": self.offload_vision,
            "expert_quant": self.expert_quant,
            "resident_expert_layers": self.resident_expert_layers,
            "resident_expert_layer_indices": self._resident_expert_layer_indices(),
            "cached_experts_per_layer": self.cached_experts_per_layer,
            "expert_offload": self.expert_offload_stats(),
            "full_attention_cache_mirror": self.full_attention_cache_mirror,
            "runtime_optimizations": {
                "compile_mode": self.optimization_config.compile_mode,
                "compile_fullgraph": self.optimization_config.compile_fullgraph,
                "compiled_text_forward": self._compiled_text_forward is not None,
                "prefill_chunk_size": self.optimization_config.prefill_chunk_size,
                "prompt_cache_size": self.optimization_config.prompt_cache_size,
                "prompt_cache_max_tokens": self.optimization_config.prompt_cache_max_tokens,
                "prompt_cache_entries": len(self._prompt_cache),
                "prefer_prompt_cache_over_prefix": self.optimization_config.prefer_prompt_cache_over_prefix,
                "profile_runtime": self.optimization_config.profile_runtime,
                # P2 hot-loop reduction: scheduler samples a whole decode batch in one
                # vectorized pass; non-streaming generation defers host token materialization.
                "batched_sampling": True,
                "device_decode_token_loop": True,
                "kv_cache_quantization": self.optimization_config.kv_cache_quantization,
                "kv_cache_quant_bits": self.optimization_config.kv_cache_quant_bits,
                "kv_cache_residual_len": self.optimization_config.kv_cache_residual_len,
                "kv_cache_turboquant_preset": self.optimization_config.kv_cache_turboquant_preset,
                "xpu_int4_kernels": {
                    "matmul_strategy": os.getenv("ANNA_XPU_INT4_MATMUL", "auto"),
                    "matmul_backend": XPUInt4Linear.resolve_matmul_backend(),
                    "layout_cache_dir": (
                        str(_layout_cache)
                        if (_layout_cache := resolve_xpu_int4_layout_cache_dir()) is not None
                        else None
                    ),
                    "lm_head_local_size": os.getenv("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"),
                    "lm_head_block_topk_threshold": os.getenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_TOPK_THRESHOLD", "65536"),
                    "lm_head_block_size": os.getenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_SIZE", "4096"),
                    "moe_gate_local_size": os.getenv("ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE"),
                    "moe_down_local_size": os.getenv("ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE"),
                    "lm_head_int4_topk_enabled": os.getenv("ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED"),
                    "lm_head_int4_topk_disabled": os.getenv("ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK"),
                    "moe_grouped_int4_disabled": os.getenv("ANNA_XPU_DISABLE_MOE_GROUPED_INT4"),
                },
            },
            "vision_enabled": self.config.vision_config is not None,
            "cache_device": str(self.device_context.migration_policy.execution_device),
            "preprocess_device": str(self.device_context.migration_policy.preprocess_device),
            "safety_policy": {
                "min_free_bytes": self.device_context.safety_policy.min_free_bytes,
                "reserve_margin_bytes": self.device_context.safety_policy.reserve_margin_bytes,
                "max_estimated_usage_ratio": self.device_context.safety_policy.max_estimated_usage_ratio,
                "generation_memory_safety_factor": self.device_context.safety_policy.generation_memory_safety_factor,
            },
            "memory": None
            if memory_info is None
            else {
                "free_bytes": memory_info.free_bytes,
                "total_bytes": memory_info.total_bytes,
                "allocated_bytes": memory_info.allocated_bytes,
                "reserved_bytes": memory_info.reserved_bytes,
            },
            "kv_cache": kv_cache_runtime_info,
            "service_metrics": {
                "requests_started_total": service_metrics.requests_started_total,
                "requests_completed_total": service_metrics.requests_completed_total,
                "requests_failed_total": service_metrics.requests_failed_total,
                "prompt_tokens_total": service_metrics.prompt_tokens_total,
                "generation_tokens_total": service_metrics.generation_tokens_total,
                "prompt_cache_queries_total": service_metrics.prompt_cache_queries_total,
                "prompt_cache_hits_total": service_metrics.prompt_cache_hits_total,
                "prompt_cache_entries": service_metrics.prompt_cache_entries,
                "prefix_block_lookups_total": service_metrics.prefix_block_lookups_total,
                "prefix_block_hits_total": service_metrics.prefix_block_hits_total,
                "prefix_block_misses_total": service_metrics.prefix_block_misses_total,
                "prefix_block_registers_total": service_metrics.prefix_block_registers_total,
                "prefix_block_entries": service_metrics.prefix_block_entries,
                "prefix_block_hit_rate": (
                    0.0
                    if service_metrics.prefix_block_lookups_total <= 0
                    else service_metrics.prefix_block_hits_total / service_metrics.prefix_block_lookups_total
                ),
                "queue_rejected_total": service_metrics.queue_rejected_total,
                "running_requests": service_metrics.running_requests,
                "waiting_requests": service_metrics.waiting_requests,
                "scheduler_queue_depth": service_metrics.scheduler_queue_depth,
                "kv_cache_used_pages": service_metrics.kv_cache_used_pages,
                "kv_cache_total_pages": service_metrics.kv_cache_total_pages,
                "kv_cache_turboquant_rows": service_metrics.kv_cache_turboquant_rows,
                "kv_cache_turboquant_tokens": service_metrics.kv_cache_turboquant_tokens,
                "cache_compact_count": service_metrics.cache_compact_count,
                "cache_compact_seconds_total": service_metrics.cache_compact_seconds_total,
                "scheduler_prefill_admitted_requests_total": (
                    service_metrics.scheduler_prefill_admitted_requests_total
                ),
                "scheduler_prefill_deferred_requests_total": (
                    service_metrics.scheduler_prefill_deferred_requests_total
                ),
                "scheduler_prefill_admitted_tokens_total": service_metrics.scheduler_prefill_admitted_tokens_total,
                "scheduler_prefill_admission_count": service_metrics.scheduler_prefill_admission_count,
                "scheduler_prefill_admitted_tokens_max": service_metrics.scheduler_prefill_admitted_tokens_max,
                "scheduler_decode_batch_count": service_metrics.scheduler_decode_batch_count,
                "scheduler_decode_batch_requests_total": service_metrics.scheduler_decode_batch_requests_total,
                "scheduler_decode_batch_requests_max": service_metrics.scheduler_decode_batch_requests_max,
                "scheduler_decode_batch_tokens_total": service_metrics.scheduler_decode_batch_tokens_total,
                "scheduler_decode_batch_tokens_max": service_metrics.scheduler_decode_batch_tokens_max,
                "scheduler_decode_profile": self._scheduler_decode_profile_snapshot(),
                "ttft_histogram": service_metrics.ttft_histogram(),
                "itl_histogram": service_metrics.itl_histogram(),
                "ttft_count": service_metrics.ttft_count,
                "itl_count": service_metrics.itl_count,
                "kernel_strategy_hits": service_metrics.kernel_strategy_hits,
            },
        }

    def _scheduler_decode_profile_snapshot(self) -> dict[str, object] | None:
        if not self.optimization_config.profile_runtime:
            return None
        accum = get_or_create_scheduler_decode_steady_accum(enabled=True)
        if accum is None:
            return None
        return accum.snapshot()

    def _resident_expert_layer_indices(self) -> list[int]:
        text_model = getattr(getattr(self.model, "model", None), "language_model", None)
        if text_model is None:
            text_model = getattr(self.model, "model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            return []

        indices: list[int] = []
        for layer_idx, layer in enumerate(text_model.layers):
            mlp = getattr(layer, "mlp", None)
            if getattr(mlp, "resident_experts", False):
                indices.append(layer_idx)
        return indices

    def expert_offload_stats(self) -> dict[str, object]:
        """Aggregate MoE expert cache hit/prefetch stats across offloaded sparse layers."""
        blocks = self._offloaded_sparse_moe_blocks(self.model)
        if not blocks:
            return {
                "offloaded_layers": 0,
                "lookups": 0,
                "hits": 0,
                "misses": 0,
                "staged": 0,
                "hit_rate": 0.0,
                "prefetch_requests": 0,
                "prefetch_hits": 0,
                "prefetch_staged": 0,
                "layer_heats": {},
            }

        lookups = hits = misses = staged = 0
        prefetch_requests = prefetch_hits = prefetch_staged = 0
        layer_heats: dict[str, float] = {}
        for layer_idx, block in blocks:
            stats = block.expert_cache_stats()
            lookups += stats.lookups
            hits += stats.hits
            misses += stats.misses
            staged += stats.staged
            prefetch_requests += stats.prefetch_requests
            prefetch_hits += stats.prefetch_hits
            prefetch_staged += stats.prefetch_staged
            layer_heats[str(layer_idx)] = float(stats.layer_heat)

        return {
            "offloaded_layers": len(blocks),
            "lookups": lookups,
            "hits": hits,
            "misses": misses,
            "staged": staged,
            "hit_rate": 0.0 if lookups <= 0 else hits / lookups,
            "prefetch_requests": prefetch_requests,
            "prefetch_hits": prefetch_hits,
            "prefetch_staged": prefetch_staged,
            "layer_heats": layer_heats,
        }

    def sparse_moe_layer_heats(self) -> dict[int, float]:
        text_model = self._text_model(self.model)
        if text_model is None or not hasattr(text_model, "layers"):
            return {}
        heats: dict[int, float] = {}
        for layer_idx, layer in enumerate(text_model.layers):
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Qwen3SparseMoeBlock):
                heats[layer_idx] = float(getattr(mlp, "_layer_heat", 0.0) or 0.0)
        return heats

    def rebalance_resident_experts_by_heat(self) -> tuple[int, ...]:
        """Re-pin sparse MoE layers using observed routing heat under the current memory budget.

        Safe to call after warmup traffic. No-op when expert offload is disabled.
        """
        if self.offload_mode != "experts":
            return tuple(self._resident_expert_layer_indices())

        selected = self._estimate_resident_expert_layer_indices(
            model=self.model,
            device_context=self.device_context,
            expert_quant=self.expert_quant,
            layer_heats=self.sparse_moe_layer_heats(),
        )
        current = tuple(self._resident_expert_layer_indices())
        if selected == current:
            return current

        logger.info(
            "Rebalancing resident expert layers by routing heat: previous=%s next=%s",
            list(current),
            list(selected),
        )
        self.model.configure_runtime(
            self.device_context.device,
            offload_experts=True,
            offload_vision=self.offload_vision,
            offload_token_io=False,
            resident_expert_layers=0,
            resident_expert_layer_indices=selected,
            expert_quant=self.expert_quant,
            cached_experts_per_layer=self.cached_experts_per_layer,
            kv_cache_quantization=self.optimization_config.kv_cache_quantization,
            kv_cache_quant_bits=self.optimization_config.kv_cache_quant_bits,
            kv_cache_residual_len=self.optimization_config.kv_cache_residual_len,
        )
        self.resident_expert_layer_indices = tuple(selected)
        self.resident_expert_layers = len(selected)
        return selected

    def generate_text(self, prompt: str, *, config: GenerationConfig) -> TextGenerationResult:
        prepared = self.processor.encode_text(
            prompt,
            tensor_device=self._preprocess_device(),
        )
        return self._generate(prepared, config=config)

    def stream_text(self, prompt: str, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        prepared = self.processor.encode_text(
            prompt,
            tensor_device=self._preprocess_device(),
        )
        events = self._stream(prepared, config=config)
        try:
            yield from events
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    def generate_chat(
        self,
        messages: list[object],
        *,
        config: GenerationConfig,
        enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str | None = None,
        tools: list[object] | None = None,
        tool_choice: object = None,
        parallel_tool_calls: bool | None = None,
    ) -> TextGenerationResult:
        prepare_kwargs: dict[str, object] = {"enable_thinking": enable_thinking}
        if tools is not None or tool_choice is not None or parallel_tool_calls is not None:
            prepare_kwargs.update(
                {
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                }
            )
        prepared = self._prepare_messages(messages, **prepare_kwargs)
        raw = self._generate(prepared, config=config)
        text, reasoning_text, tool_calls = self._project_chat_output(
            raw_text=raw.text,
            raw_reasoning_text=raw.reasoning_text,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
        )
        finish_reason = "tool_calls" if raw.finish_reason == "stop" and tool_calls else raw.finish_reason
        return TextGenerationResult(
            text=text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            perf=raw.perf,
        )

    def stream_chat(
        self,
        messages: list[object],
        *,
        config: GenerationConfig,
        enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str | None = None,
        tools: list[object] | None = None,
        tool_choice: object = None,
        parallel_tool_calls: bool | None = None,
    ) -> Iterator[StreamEvent]:
        prepare_kwargs: dict[str, object] = {"enable_thinking": enable_thinking}
        if tools is not None or tool_choice is not None or parallel_tool_calls is not None:
            prepare_kwargs.update(
                {
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                }
            )
        prepared = self._prepare_messages(messages, **prepare_kwargs)
        resolved_reasoning_format = self._resolve_reasoning_format(reasoning_format)
        reasoning_parser = None
        if resolved_reasoning_format != "none":
            reasoning_parser = self.tokenizer.create_reasoning_parser(enable_thinking=enable_thinking)
        tool_call_parser = self.tokenizer.create_tool_call_stream_parser()
        events = self._stream(prepared, config=config)
        try:
            for event in events:
                outputs: list[StreamEvent] = []
                if event.reasoning_text:
                    outputs.append(StreamEvent(text="", reasoning_text=event.reasoning_text))

                if reasoning_parser is None:
                    if event.text:
                        outputs.extend(self._project_tool_stream_outputs(tool_call_parser.feed(event.text)))
                else:
                    if event.text:
                        for kind, chunk in reasoning_parser.feed(event.text):
                            if kind == "reasoning":
                                outputs.append(StreamEvent(text="", reasoning_text=chunk))
                            elif chunk:
                                outputs.extend(self._project_tool_stream_outputs(tool_call_parser.feed(chunk)))
                    if event.finish_reason is not None:
                        for kind, chunk in reasoning_parser.flush():
                            if kind == "reasoning":
                                outputs.append(StreamEvent(text="", reasoning_text=chunk))
                            elif chunk:
                                outputs.extend(self._project_tool_stream_outputs(tool_call_parser.feed(chunk)))

                if event.finish_reason is not None:
                    outputs.extend(self._project_tool_stream_outputs(tool_call_parser.flush()))
                    for output in outputs:
                        yield output
                    finish_reason = (
                        "tool_calls" if event.finish_reason == "stop" and tool_call_parser.saw_tool_calls else event.finish_reason
                    )
                    yield StreamEvent(
                        text="",
                        finish_reason=finish_reason,
                        prompt_tokens=event.prompt_tokens,
                        completion_tokens=event.completion_tokens,
                        perf=event.perf,
                    )
                    continue

                for output in outputs:
                    yield output
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    def _project_tool_stream_outputs(
        self,
        outputs: list[tuple[str, str | ToolCallDelta]],
    ) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for kind, value in outputs:
            if kind == "content":
                text = cast(str, value)
                if text:
                    events.append(StreamEvent(text=text))
                continue
            events.append(StreamEvent(text="", tool_calls=[cast(ToolCallDelta, value)]))
        return events

    def _prepare_messages(
        self,
        messages: list[object],
        *,
        enable_thinking: bool = True,
        tools: list[object] | None = None,
        tool_choice: object = None,
        parallel_tool_calls: bool | None = None,
    ) -> PreparedInputs:
        try:
            prepare_kwargs: dict[str, object] = {
                "enable_thinking": enable_thinking,
                "tensor_device": self._preprocess_device(),
                "tensor_dtype": self.device_context.dtype,
            }
            if tools is not None or tool_choice is not None or parallel_tool_calls is not None:
                prepare_kwargs.update(
                    {
                        "tools": tools,
                        "tool_choice": tool_choice,
                        "parallel_tool_calls": parallel_tool_calls,
                    }
                )
            return self.processor.prepare_messages(
                messages,
                **prepare_kwargs,
            )
        except FileNotFoundError as exc:
            raise AnnaEngineError(str(exc), status_code=400, code="invalid_media_reference") from exc
        except ValueError as exc:
            raise AnnaEngineError(str(exc), status_code=400) from exc
        except RuntimeError as exc:
            raise AnnaEngineError(str(exc), status_code=500, error_type="server_error") from exc

    def _preprocess_device(self) -> torch.device:
        migration_policy = getattr(self.device_context, "migration_policy", None)
        preprocess_device = getattr(migration_policy, "preprocess_device", None)
        if isinstance(preprocess_device, torch.device):
            return preprocess_device
        if preprocess_device is not None:
            return torch.device(preprocess_device)
        return self.device_context.device

    def _can_use_scheduler(self, prepared: PreparedInputs) -> bool:
        # Multimodal requests are admitted through the scheduler so vision prefill can
        # interleave with text decode (see AnnaScheduler multimodal isolation).
        # Audio features (Gemma path) still bypass until feature batching exists.
        if self.scheduler is None:
            return False
        if prepared.input_features is not None:
            return False
        return True

    def _has_multimodal_inputs(self, prepared: PreparedInputs) -> bool:
        return any(getattr(prepared, key) is not None for key in self._forward_multimodal_input_keys())

    def _build_prefill_model_kwargs(
        self,
        prepared: PreparedInputs,
        *,
        token_slice: slice | None = None,
        include_media: bool = True,
    ) -> dict[str, object]:
        mm_token_type_ids = prepared.mm_token_type_ids
        if token_slice is not None and mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:, token_slice]
        return {
            "pixel_values": prepared.pixel_values if include_media else None,
            "pixel_values_videos": prepared.pixel_values_videos if include_media else None,
            "image_grid_thw": prepared.image_grid_thw if include_media else None,
            "video_grid_thw": prepared.video_grid_thw if include_media else None,
            "mm_token_type_ids": mm_token_type_ids,
        }

    def _forward_multimodal_input_keys(self) -> tuple[str, ...]:
        return ("pixel_values", "pixel_values_videos")

    @staticmethod
    def _profile_memory_stats_snapshot(memory_stats: dict[str, int | float] | None) -> dict[str, int | float] | None:
        if not memory_stats:
            return None
        keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "num_alloc_retries",
            "num_ooms",
        )
        snapshot = {key: memory_stats[key] for key in keys if key in memory_stats}
        return snapshot or None

    def _log_profiled_forward(
        self,
        *,
        stage: str,
        elapsed_seconds: float,
        input_ids: torch.Tensor,
        past_key_values: object | None,
        memory_before: object | None,
        memory_after: object | None,
        stats_before: dict[str, int | float] | None,
        stats_after: dict[str, int | float] | None,
        batch_size: int | None = None,
        token_cost: int | None = None,
        component_ms: dict[str, float] | None = None,
        steady_recorded: bool | None = None,
    ) -> None:
        cache_length = getattr(past_key_values, "get_seq_length", None)
        seen_tokens = cache_length() if callable(cache_length) else 0
        resolved_batch = int(batch_size) if batch_size is not None else int(input_ids.shape[0])
        # P0-#1: wall time not covered by tracked GPU categories = kernel-launch
        # gaps + host dispatch + untracked ops. The primary overhead signal.
        launch_gap = ""
        if component_ms and elapsed_seconds > 0:
            gap_ms = max(0.0, elapsed_seconds * 1000.0 - float(sum(component_ms.values())))
            launch_gap = f" cpu_launch_gap_ms={gap_ms:.3f}"
        extra = ""
        if stage.startswith("scheduler_decode"):
            per_req = _amortized_per_request_ms(component_ms or {}, resolved_batch) if component_ms else {}
            extra = (
                f" batch_size={resolved_batch} token_cost={token_cost if token_cost is not None else '-'} "
                f"component_ms_per_req={{{', '.join(f'{k}: {round(v, 3)}' for k, v in sorted(per_req.items()))}}} "
                f"steady={1 if steady_recorded else 0}"
            )
        logger.info(
            "xpu_profile stage=%s input_tokens=%s cache_tokens=%s elapsed_seconds=%.6f%s "
            "free_before=%s free_after=%s stats_before=%s stats_after=%s%s",
            stage,
            int(input_ids.shape[-1]),
            seen_tokens,
            elapsed_seconds,
            launch_gap,
            format_bytes(memory_before.free_bytes if memory_before is not None else None),
            format_bytes(memory_after.free_bytes if memory_after is not None else None),
            stats_before,
            stats_after,
            extra,
        )

    def _profiled_forward_generation_model(
        self,
        *,
        stage: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        model_kwargs: dict[str, object] | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
        batch_size: int | None = None,
        token_cost: int | None = None,
    ):
        if not self.optimization_config.profile_runtime:
            return self._forward_generation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                model_kwargs=model_kwargs,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
            )

        from anna.model.xpu_decode_profile import decode_profile_session

        resolved_batch = int(batch_size) if batch_size is not None else int(input_ids.shape[0])
        if stage.startswith("scheduler_decode"):
            get_or_create_scheduler_decode_steady_accum(enabled=True)

        self.device_context.synchronize()
        memory_before = self.device_context.get_memory_info()
        stats_before = self._profile_memory_stats_snapshot(self.device_context.get_memory_stats())
        started_at = time.perf_counter()
        with decode_profile_session() as decode_prof:
            outputs = self._forward_generation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                model_kwargs=model_kwargs,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
            )
            self.device_context.synchronize()
            ms = decode_prof.log_summary(log=logger)
        steady_recorded = None
        if ms:
            if stage.startswith("scheduler_decode"):
                accum = get_or_create_scheduler_decode_steady_accum(enabled=True)
                if accum is not None:
                    steady_recorded = accum.record(ms, batch_size=resolved_batch)
                else:
                    record_steady_decode_step_if_applicable(
                        stage,
                        ms,
                        batch_size=resolved_batch,
                        token_cost=token_cost,
                    )
            else:
                record_steady_decode_step_if_applicable(
                    stage,
                    ms,
                    batch_size=resolved_batch,
                    token_cost=token_cost,
                )
        elapsed_seconds = time.perf_counter() - started_at
        self.device_context.synchronize()
        memory_after = self.device_context.get_memory_info()
        stats_after = self._profile_memory_stats_snapshot(self.device_context.get_memory_stats())
        self._log_profiled_forward(
            stage=stage,
            elapsed_seconds=elapsed_seconds,
            input_ids=input_ids,
            past_key_values=past_key_values,
            memory_before=memory_before,
            memory_after=memory_after,
            stats_before=stats_before,
            stats_after=stats_after,
            batch_size=resolved_batch,
            token_cost=token_cost,
            component_ms=ms or None,
            steady_recorded=steady_recorded,
        )
        return outputs

    def _prompt_cache_key(self, prepared: PreparedInputs) -> tuple[int, ...] | None:
        if self.optimization_config.prompt_cache_size <= 0:
            return None
        if self._has_multimodal_inputs(prepared):
            return None
        prompt_cache_max_tokens = self.optimization_config.prompt_cache_max_tokens
        if prompt_cache_max_tokens > 0 and int(prepared.input_ids.shape[1]) > prompt_cache_max_tokens:
            return None
        return tuple(int(token_id) for token_id in prepared.input_ids[0].tolist())

    def _prompt_is_exact_cache_eligible(self, prepared: PreparedInputs) -> bool:
        """True when this prompt would use the exact prompt-cache path (not prefix-only)."""
        return self._prompt_cache_key(prepared) is not None

    def _should_skip_prefix_registration(self, prepared: PreparedInputs) -> bool:
        """Avoid dual caching: exact prompt-cache eligible prompts skip paged prefix registration.

        - Exact full-prompt reuse → prompt cache (whole KV clone)
        - Shared system-prefix across different prompts → paged PrefixBlockPool
        When both are enabled, prompts that fit the exact-cache policy skip prefix page
        registration so we do not keep two residency strategies for the same full prompt.
        """
        if not self.optimization_config.prefer_prompt_cache_over_prefix:
            return False
        return self._prompt_is_exact_cache_eligible(prepared)

    def _evict_prompt_cache_entry(self, key: tuple[int, ...], entry: PromptCacheEntry) -> None:
        release = getattr(entry.past_key_values, "release", None)
        if callable(release):
            release()
        self._prompt_cache.pop(key, None)

    def _clone_prompt_cache_entry(self, key: tuple[int, ...]) -> PromptPrefillResult | None:
        entry = self._prompt_cache.get(key)
        if entry is None:
            return None
        clone = getattr(entry.past_key_values, "clone", None)
        if not callable(clone):
            self._evict_prompt_cache_entry(key, entry)
            return None
        self._prompt_cache.move_to_end(key)
        started_at = time.perf_counter()
        try:
            cached_past = clone()
        except Exception:
            logger.exception("Failed to clone prompt cache entry; evicting stale cache.")
            self._evict_prompt_cache_entry(key, entry)
            return None
        return PromptPrefillResult(
            logits=entry.logits,
            past_key_values=cached_past,
            prefill_seconds=time.perf_counter() - started_at,
            prompt_cache_hit=True,
        )

    def _remember_prompt_cache_entry(
        self,
        *,
        key: tuple[int, ...] | None,
        logits: torch.Tensor,
        past_key_values: object | None,
        prompt_tokens: int,
    ) -> None:
        if key is None or past_key_values is None or self.optimization_config.prompt_cache_size <= 0:
            return
        clone = getattr(past_key_values, "clone", None)
        if not callable(clone):
            return
        try:
            cached_past = clone()
        except Exception:
            logger.exception("Failed to clone prompt cache state for reuse; skipping cache insert.")
            return

        existing = self._prompt_cache.pop(key, None)
        if existing is not None:
            release = getattr(existing.past_key_values, "release", None)
            if callable(release):
                release()

        self._prompt_cache[key] = PromptCacheEntry(
            logits=logits.detach(),
            past_key_values=cached_past,
            prompt_tokens=prompt_tokens,
        )

        while len(self._prompt_cache) > self.optimization_config.prompt_cache_size:
            stale_key, stale_entry = self._prompt_cache.popitem(last=False)
            release = getattr(stale_entry.past_key_values, "release", None)
            if callable(release):
                release()

    def lookup_prompt_cache(self, prepared: PreparedInputs) -> tuple[tuple[int, ...], PromptPrefillResult] | None:
        """Public exact prompt-cache lookup (scheduler path, P2-#7).

        Returns ``(cache_key, PromptPrefillResult)`` on a hit, ``None`` on a miss or
        when the prompt is not exact-cache eligible. Hit/miss counters are recorded
        so the prompt-cache hit-rate metric stays meaningful under continuous batching.
        """
        key = self._prompt_cache_key(prepared)
        if key is None:
            return None
        # Cache clones copy inference-mode tensors; keep the clone in the same mode.
        with torch.inference_mode():
            cached = self._clone_prompt_cache_entry(key)
        metrics = getattr(self, "metrics", None)
        if cached is not None:
            if metrics is not None:
                metrics.record_prompt_cache_lookup(hit=True)
            logger.info("Prompt cache hit: prompt_tokens=%s", int(prepared.input_ids.shape[1]))
            return key, cached
        if metrics is not None:
            metrics.record_prompt_cache_lookup(hit=False)
        return None

    def remember_prompt_cache(
        self,
        *,
        key: tuple[int, ...] | None,
        logits: torch.Tensor,
        past_key_values: object | None,
        prompt_tokens: int,
    ) -> None:
        """Public prompt-cache insert (scheduler path, P2-#7). Single-request caches only."""
        self._remember_prompt_cache_entry(
            key=key,
            logits=logits,
            past_key_values=past_key_values,
            prompt_tokens=prompt_tokens,
        )

    def _prefill_generation_prompt(self, prepared: PreparedInputs) -> PromptPrefillResult:
        started_at = time.perf_counter()
        prompt_tokens = int(prepared.input_ids.shape[1])
        metrics = getattr(self, "metrics", None)
        cache_key = self._prompt_cache_key(prepared)
        if cache_key is not None:
            cached = self._clone_prompt_cache_entry(cache_key)
            if metrics is not None:
                metrics.record_prompt_cache_lookup(hit=cached is not None)
            if cached is not None:
                logger.info("Prompt cache hit: prompt_tokens=%s", int(prepared.input_ids.shape[1]))
                if metrics is not None:
                    metrics.record_prompt_tokens(prompt_tokens)
                return cached

        input_ids = prepared.input_ids
        attention_mask = prepared.attention_mask
        past_key_values = None
        outputs = None
        chunk_size = self.optimization_config.prefill_chunk_size
        prompt_tokens_recorded = 0

        try:
            if chunk_size > 0 and not self._has_multimodal_inputs(prepared) and int(input_ids.shape[1]) > chunk_size:
                past_key_values = self._reserve_prefill_cache(prepared)
                total_tokens = int(input_ids.shape[1])
                for start_idx in range(0, total_tokens, chunk_size):
                    end_idx = min(total_tokens, start_idx + chunk_size)
                    outputs = self._profiled_forward_generation_model(
                        stage=f"prefill[{start_idx}:{end_idx}]",
                        input_ids=input_ids[:, start_idx:end_idx],
                        attention_mask=attention_mask[:, :end_idx] if start_idx == 0 else None,
                        past_key_values=past_key_values,
                        model_kwargs=self._build_prefill_model_kwargs(
                            prepared,
                            token_slice=slice(start_idx, end_idx),
                            include_media=start_idx == 0,
                        ),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past_key_values = outputs.past_key_values
                    if metrics is not None:
                        chunk_tokens = end_idx - start_idx
                        if chunk_tokens > 0:
                            metrics.record_prompt_tokens(chunk_tokens)
                            prompt_tokens_recorded += chunk_tokens
            else:
                outputs = self._profiled_forward_generation_model(
                    stage="prefill",
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    model_kwargs=self._build_prefill_model_kwargs(prepared, include_media=True),
                    use_cache=True,
                    logits_to_keep=1,
                )
                past_key_values = outputs.past_key_values
        except Exception:
            release = getattr(past_key_values, "release", None)
            if callable(release):
                release()
            raise

        if outputs is None:
            raise RuntimeError("Prompt prefill did not produce model outputs.")

        logits = outputs.logits[0, -1]
        self._remember_prompt_cache_entry(
            key=cache_key,
            logits=logits,
            past_key_values=past_key_values,
            prompt_tokens=int(prepared.input_ids.shape[1]),
        )
        if metrics is not None and prompt_tokens_recorded < prompt_tokens:
            metrics.record_prompt_tokens(prompt_tokens - prompt_tokens_recorded)
        return PromptPrefillResult(
            logits=logits,
            past_key_values=past_key_values,
            prefill_seconds=time.perf_counter() - started_at,
            prompt_cache_hit=False,
        )

    def _forward_generation_model(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        model_kwargs: dict[str, object] | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ):
        attention_mask = self._prune_trivial_attention_mask(attention_mask)
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        has_multimodal_inputs = any(
            model_kwargs.get(key) is not None
            for key in self._forward_multimodal_input_keys()
        )
        if not has_multimodal_inputs:
            forward_fn = getattr(self, "_compiled_text_forward", None)
            if forward_fn is None:
                if not hasattr(self.model, "forward_text_only"):
                    raise RuntimeError("Text-only generation requires Qwen3_5TextForConditionalGeneration.forward_text_only")
                forward_fn = self.model.forward_text_only
            return forward_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
            )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **model_kwargs,
        )

    def _fused_lm_head_candidate_count(self, config: GenerationConfig) -> int | None:
        if config.repetition_penalty != 1.0:
            return None
        if config.presence_penalty != 0.0:
            return None
        if config.temperature <= 0.0:
            return 1
        if config.top_k <= 0:
            return None
        return int(config.top_k)

    def _forward_generation_model_topk(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        model_kwargs: dict[str, object] | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
        top_k: int,
    ):
        attention_mask = self._prune_trivial_attention_mask(attention_mask)
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        has_multimodal_inputs = any(
            model_kwargs.get(key) is not None
            for key in self._forward_multimodal_input_keys()
        )
        if not has_multimodal_inputs:
            forward_fn = getattr(self.model, "forward_text_only_topk", None)
            if forward_fn is None:
                return None
            return forward_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                top_k=top_k,
            )
        forward_fn = getattr(self.model, "forward_topk", None)
        if forward_fn is None:
            return None
        return forward_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            top_k=top_k,
            **model_kwargs,
        )

    def warmup_inference_kernels(
        self,
        *,
        prefill_tokens: int = 2,
        decode_steps: int = 1,
        batch_size: int = 1,
        prefill_lengths: list[int] | None = None,
        batch_sizes: list[int] | None = None,
    ) -> None:
        from anna.model.fused_ops import maybe_load_gated_delta_library, resolve_flashqla_gdn_prefill_mode

        maybe_load_gated_delta_library()
        cfg = self.config.text_config
        pad = int(cfg.pad_token_id)
        if not (0 <= pad < cfg.vocab_size):
            pad = 0
        bos = getattr(self.tokenizer, "bos_token_id", None)
        token_id = int(bos) if bos is not None and 0 <= int(bos) < cfg.vocab_size else pad
        device = self.device_context.device
        prefill_tokens = max(2, int(prefill_tokens))
        decode_steps = max(1, int(decode_steps))
        batch_size = max(1, int(batch_size))
        # Multi-shape prefill ladder reduces first-request SYCL JIT when FlashQLA (or fused GDN) is on,
        # and avoids recompile / lazy-kernel-init stalls on the first real request (P1-#4).
        # ``prefill_lengths`` / ``batch_sizes`` (derived from the scheduler profile + chunk size by
        # ``derive_warmup_shape_table``) take precedence over the single-shape arguments.
        flashqla_mode = resolve_flashqla_gdn_prefill_mode()
        if prefill_lengths:
            prefill_lengths = sorted({max(2, int(length)) for length in prefill_lengths if int(length) > 1})
        else:
            prefill_lengths = [2, 64, prefill_tokens]
            chunk = int(getattr(self.optimization_config, "prefill_chunk_size", 0) or 0)
            if chunk > 1:
                prefill_lengths.append(chunk)
            # Dedup, keep ascending, clamp to a sane upper bound for warmup.
            # The resolved prefill chunk size is always kept: chunked long-prompt prefills
            # run at exactly that shape, so warming it avoids a first-request stall
            # regardless of --warmup-prefill-tokens.
            prefill_lengths = sorted({max(2, int(length)) for length in prefill_lengths if int(length) > 1})
            warmup_cap = max(prefill_tokens, 256)
            prefill_lengths = [
                length
                for length in prefill_lengths
                if length <= warmup_cap or (chunk > 1 and length == chunk)
            ]
            if prefill_tokens not in prefill_lengths:
                prefill_lengths.append(prefill_tokens)
                prefill_lengths.sort()
        if batch_sizes:
            warmup_batch_sizes = sorted({max(1, int(size)) for size in batch_sizes if int(size) > 0})
        else:
            # Serving is dominated by bs=1 (single-stream requests), while the scheduler
            # warms up at max_batch_size; warm both so neither shape compiles/JITs on the
            # first real request.
            warmup_batch_sizes = sorted({1, batch_size})
        with torch.inference_mode():
            # Fused causal_conv1d_prefill / gated_delta_prefill require seq_len > 1 (see SYCL TORCH_CHECK).
            for warm_batch in warmup_batch_sizes:
                past = None
                for length in prefill_lengths:
                    if past is not None:
                        release = getattr(past, "release", None)
                        if callable(release):
                            release()
                        past = None
                    input_ids = torch.full((warm_batch, length), token_id, device=device, dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    outputs = self._forward_generation_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        model_kwargs={},
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past = outputs.past_key_values
                if past is not None:
                    decode_ids = torch.full((warm_batch, 1), token_id, device=device, dtype=torch.long)
                    for _ in range(decode_steps):
                        outputs = self._forward_generation_model(
                            input_ids=decode_ids,
                            attention_mask=None,
                            past_key_values=past,
                            model_kwargs={},
                            use_cache=True,
                            logits_to_keep=1,
                        )
                        past = outputs.past_key_values
                    past.release()
            if device.type == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.synchronize()
        logger.info(
            "XPU inference warmup finished (prefill_tokens=%s prefill_shapes=%s decode_steps=%s "
            "batch_sizes=%s flashqla_mode=%s).",
            prefill_tokens,
            prefill_lengths,
            decode_steps,
            warmup_batch_sizes,
            flashqla_mode,
        )

    @staticmethod
    def _prune_trivial_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None or attention_mask.ndim != 2:
            return attention_mask
        if int(attention_mask.min().item()) > 0:
            return None
        return attention_mask

    @staticmethod
    def _tokens_per_second(token_count: int, elapsed_seconds: float) -> float:
        if token_count <= 0 or elapsed_seconds <= 0:
            return 0.0
        return token_count / elapsed_seconds

    def _build_generation_perf_stats(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_seconds: float,
        prefill_seconds: float,
        decode_seconds: float,
    ) -> GenerationPerfStats:
        prefill_seconds = max(0.0, prefill_seconds)
        total_seconds = max(prefill_seconds, total_seconds)
        decode_seconds = max(0.0, decode_seconds)
        decode_tokens = max(0, completion_tokens - 1)
        return GenerationPerfStats(
            total_seconds=total_seconds,
            prefill_seconds=prefill_seconds,
            ttft_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefill_tokens_per_second=self._tokens_per_second(prompt_tokens, prefill_seconds),
            decode_tokens=decode_tokens,
            decode_tokens_per_second=self._tokens_per_second(decode_tokens, decode_seconds),
            total_tokens_per_second=self._tokens_per_second(completion_tokens, total_seconds),
        )

    def _validate_generation_request(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> tuple[list[int], int, GenerationConfig]:
        prompt_ids = prepared.input_ids[0].tolist()
        prompt_length = int(prepared.input_ids.shape[1])
        if prompt_length == 0:
            raise AnnaEngineError("Prompt produced zero tokens.")

        context_limit = self.config.text_config.max_position_embeddings
        context_remaining = context_limit - prompt_length
        if context_remaining <= 0:
            raise AnnaEngineError(
                f"Prompt length {prompt_length} already reaches the model context limit {context_limit}.",
                status_code=400,
                code="context_length_exceeded",
            )
        if config.temperature < 0.0:
            raise AnnaEngineError("temperature must be >= 0.")
        if not 0.0 < config.top_p <= 1.0:
            raise AnnaEngineError("top_p must be in the range (0, 1].")
        if config.top_k < 0:
            raise AnnaEngineError("top_k must be >= 0.")
        if not 0.0 <= config.min_p <= 1.0:
            raise AnnaEngineError("min_p must be in the range [0, 1].")
        if config.repetition_penalty < 0.1:
            raise AnnaEngineError("repetition_penalty must be >= 0.1.")
        if config.max_new_tokens is None:
            resolved_max_new_tokens = self._resolve_auto_max_new_tokens(
                prepared,
                context_remaining=context_remaining,
            )
        else:
            resolved_max_new_tokens = max(1, int(config.max_new_tokens))
            max_total = prompt_length + resolved_max_new_tokens
            if max_total > context_limit:
                raise AnnaEngineError(
                    f"Requested sequence length {max_total} exceeds model context limit {context_limit}.",
                    status_code=400,
                    code="context_length_exceeded",
                )
        return prompt_ids, prompt_length, replace(config, max_new_tokens=resolved_max_new_tokens)

    def _resolve_auto_max_new_tokens(
        self,
        prepared: PreparedInputs,
        *,
        context_remaining: int,
    ) -> int:
        if context_remaining <= 0:
            raise AnnaEngineError(
                "No completion tokens remain within the model context window.",
                status_code=400,
                code="context_length_exceeded",
            )

        memory_info = self.device_context.get_memory_info()
        if memory_info is None:
            return context_remaining
        policy = self.device_context.safety_policy
        probe = GenerationConfig(max_new_tokens=1)

        for attempt in range(2):
            if memory_info.free_bytes >= policy.min_free_bytes:
                available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
                max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)
                memory_budget = min(available_budget, max_allowed)
                if memory_budget > 0:
                    low = 1
                    high = context_remaining
                    best = 0
                    while low <= high:
                        mid = (low + high) // 2
                        probe.max_new_tokens = mid
                        estimated_bytes = self._estimate_generation_memory_bytes(prepared, config=probe)
                        if estimated_bytes <= memory_budget:
                            best = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                    if best > 0:
                        return best

            if attempt == 0 and self._reclaim_runtime_memory_for_admission():
                memory_info = self.device_context.get_memory_info()
                if memory_info is None:
                    return context_remaining
                continue
            break

        if memory_info.free_bytes < policy.min_free_bytes:
            raise AnnaEngineError(
                f"Insufficient free XPU memory before generation: free={format_bytes(memory_info.free_bytes)}, "
                f"required reserve={format_bytes(policy.min_free_bytes)}. Reduce workload or restart the service.",
                status_code=503,
                error_type="server_error",
                code="insufficient_device_memory",
            )

        available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
        max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)
        memory_budget = min(available_budget, max_allowed)
        if memory_budget <= 0:
            raise AnnaEngineError(
                "No XPU memory budget remains for generation after applying safety margins.",
                status_code=503,
                error_type="server_error",
                code="insufficient_device_memory",
            )

        one_token_config = GenerationConfig(max_new_tokens=1)
        estimated_bytes = self._estimate_generation_memory_bytes(prepared, config=one_token_config)
        raise AnnaEngineError(
            f"Request rejected by memory guard: estimated={format_bytes(estimated_bytes)}, "
            f"free={format_bytes(memory_info.free_bytes)}, reserve={format_bytes(policy.reserve_margin_bytes)}. "
            "Reduce prompt length, image/video size, or max_completion_tokens.",
            status_code=400,
            error_type="invalid_request_error",
            code="estimated_device_oom",
        )

    def _move_prepared_for_generation(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> PreparedInputs:
        self._guard_generation_memory(prepared, config=config)
        try:
            return self.device_context.move_prepared_inputs(prepared)
        except RuntimeError as exc:
            raise self._handle_runtime_failure(exc) from exc

    def _estimate_generation_memory_bytes(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> int:
        text_config = self.config.text_config
        bytes_per_elem = self.device_context.element_size(self.device_context.dtype)
        total_tokens = int(prepared.input_ids.shape[1]) + config.max_new_tokens
        full_layers = sum(1 for layer_type in text_config.layer_types if layer_type == "full_attention")
        linear_layers = max(0, text_config.num_hidden_layers - full_layers)
        kv_elements_per_token = text_config.num_key_value_heads * text_config.head_dim
        optimization_config = getattr(self, "optimization_config", None)
        if optimization_config is not None and optimization_config.kv_cache_quantization == "turboquant":
            residual_tokens = min(total_tokens, max(1, int(optimization_config.kv_cache_residual_len)))
            quantized_tokens = max(0, total_tokens - residual_tokens)
            quant_bits = int(optimization_config.kv_cache_quant_bits)
            quantized_bytes = (2 * quantized_tokens * kv_elements_per_token * quant_bits + 7) // 8
            residual_bytes = 2 * residual_tokens * kv_elements_per_token * bytes_per_elem
            # TurboQuant keeps per-group scale/min metadata in floating point. The
            # exact group count is implementation-specific, so budget a modest
            # metadata margin while still reflecting the compressed KV footprint.
            kv_cache_bytes = full_layers * int((quantized_bytes + residual_bytes) * 1.20)
        else:
            kv_cache_bytes = (
                full_layers
                * 2
                * total_tokens
                * kv_elements_per_token
                * bytes_per_elem
            )
        linear_key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        linear_value_dim = text_config.linear_num_value_heads * text_config.linear_value_head_dim
        conv_cache_bytes = linear_layers * (linear_key_dim * 2 + linear_value_dim) * text_config.linear_conv_kernel_dim * bytes_per_elem
        recurrent_bytes = (
            linear_layers
            * text_config.linear_num_value_heads
            * text_config.linear_key_head_dim
            * text_config.linear_value_head_dim
            * bytes_per_elem
        )
        hidden_working_bytes = total_tokens * text_config.hidden_size * bytes_per_elem * 8
        media_bytes = 0
        if prepared.pixel_values is not None:
            media_bytes += prepared.pixel_values.numel() * bytes_per_elem
        if prepared.pixel_values_videos is not None:
            media_bytes += prepared.pixel_values_videos.numel() * bytes_per_elem
        if prepared.input_features is not None:
            media_bytes += prepared.input_features.numel() * bytes_per_elem

        if getattr(self, "full_attention_cache_mirror", False):
            kv_cache_bytes *= 2

        estimated = kv_cache_bytes + conv_cache_bytes + recurrent_bytes + hidden_working_bytes + media_bytes
        return int(estimated * self.device_context.safety_policy.generation_memory_safety_factor)

    def _guard_generation_memory(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> None:
        memory_info = self.device_context.get_memory_info()
        if memory_info is None:
            return

        estimated_bytes = self._estimate_generation_memory_bytes(prepared, config=config)
        policy = self.device_context.safety_policy
        available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
        max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)

        if (
            memory_info.free_bytes < policy.min_free_bytes
            or estimated_bytes > available_budget
            or estimated_bytes > max_allowed
        ) and self._reclaim_runtime_memory_for_admission():
            memory_info = self.device_context.get_memory_info()
            if memory_info is None:
                return
            available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
            max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)

        if memory_info.free_bytes < policy.min_free_bytes:
            raise AnnaEngineError(
                f"Insufficient free XPU memory before generation: free={format_bytes(memory_info.free_bytes)}, "
                f"required reserve={format_bytes(policy.min_free_bytes)}. Reduce workload or restart the service.",
                status_code=503,
                error_type="server_error",
                code="insufficient_device_memory",
            )

        if estimated_bytes > available_budget or estimated_bytes > max_allowed:
            raise AnnaEngineError(
                f"Request rejected by memory guard: estimated={format_bytes(estimated_bytes)}, "
                f"free={format_bytes(memory_info.free_bytes)}, reserve={format_bytes(policy.reserve_margin_bytes)}. "
                "Reduce prompt length, image/video size, or max_completion_tokens.",
                status_code=400,
                error_type="invalid_request_error",
                code="estimated_device_oom",
            )

    def _handle_runtime_failure(self, exc: RuntimeError) -> AnnaEngineError:
        category = self.device_context.classify_runtime_error(exc)
        if self.device_context.should_recover(exc):
            try:
                self.device_context.recover()
                self._clear_runtime_caches_after_recover(reason=category)
            except Exception:  # pragma: no cover - best-effort recovery
                logger.exception("Failed to recover device context after runtime failure.")
            # Unified recovery: stop admitting new work until the operator confirms
            # healthz / restarts. Device-lost and OOM both enter degraded mode.
            if category in {"out_of_memory", "device_lost", "out_of_resources"}:
                reason = (
                    f"Runtime degraded after {category}: caches cleared; "
                    "new requests are rejected until the process is restarted "
                    "or the admission gate is cleared."
                )
                self._admission_gate().enter_degraded(category=category, reason=reason)
                logger.error("Entered runtime degraded mode after %s.", category)

        if category == "out_of_memory":
            return AnnaEngineError(
                "XPU out of memory during generation. Reduce prompt length, media size, batch size, or max_completion_tokens. "
                "The service is not accepting new requests until recovery is confirmed.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_memory",
            )
        if category == "device_lost":
            return AnnaEngineError(
                "XPU device was lost during generation. The runtime cache was cleared; "
                "new requests are rejected until the device recovers and the process is restarted.",
                status_code=503,
                error_type="server_error",
                code="device_lost",
            )
        if category == "out_of_resources":
            return AnnaEngineError(
                "XPU runtime ran out of resources during generation. Reduce the request size and retry. "
                "The service is not accepting new requests until recovery is confirmed.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_resources",
            )
        return AnnaEngineError(
            f"Runtime execution failed: {exc}",
            status_code=500,
            error_type="server_error",
            code="runtime_execution_failed",
        )

    def _resolve_reasoning_format(self, reasoning_format: ReasoningFormat | str | None) -> ReasoningFormat:
        if reasoning_format is None:
            return normalize_reasoning_format(getattr(self, "reasoning_format", None))
        return normalize_reasoning_format(reasoning_format)

    def _project_chat_output(
        self,
        *,
        raw_text: str,
        raw_reasoning_text: str | None,
        enable_thinking: bool,
        reasoning_format: ReasoningFormat | str | None,
    ) -> tuple[str, str | None, list[dict[str, object]] | None]:
        resolved_reasoning_format = self._resolve_reasoning_format(reasoning_format)
        if resolved_reasoning_format == "none":
            text, tool_calls = self.tokenizer.extract_tool_calls(raw_text)
            return text, None, [tool_call.to_openai_dict() for tool_call in tool_calls] or None

        parsed_reasoning, parsed_content = self._split_chat_output(raw_text, enable_thinking=enable_thinking)
        text, tool_calls = self.tokenizer.extract_tool_calls(parsed_content)
        reasoning_text = raw_reasoning_text if raw_reasoning_text is not None else parsed_reasoning
        return text, reasoning_text, [tool_call.to_openai_dict() for tool_call in tool_calls] or None

    def _split_chat_output(self, raw_text: str, *, enable_thinking: bool) -> tuple[str | None, str]:
        return self.tokenizer.split_assistant_reasoning(raw_text, enable_thinking=enable_thinking)

    def _record_perf_latency(self, perf: GenerationPerfStats | None) -> None:
        if perf is None:
            return
        metrics = getattr(self, "metrics", None)
        if metrics is None or not hasattr(metrics, "record_generation_latency"):
            return
        metrics.record_generation_latency(
            ttft_seconds=perf.ttft_seconds,
            decode_seconds=perf.decode_seconds,
            decode_tokens=perf.decode_tokens,
        )

    def _generate(self, prepared: PreparedInputs, *, config: GenerationConfig) -> TextGenerationResult:
        self.ensure_accepting_requests()
        if self._can_use_scheduler(prepared):
            result = self.scheduler.generate(prepared, config=config)
            self._record_perf_latency(getattr(result, "perf", None))
            return result
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.record_request_submitted(waiting=False)
        success = False
        try:
            result = self._generate_direct(prepared, config=config)
            self._record_perf_latency(result.perf)
            success = True
            return result
        finally:
            if metrics is not None:
                metrics.record_request_finished(success=success)
            self._trim_runtime_cache_if_idle()

    def _generate_direct(self, prepared: PreparedInputs, *, config: GenerationConfig) -> TextGenerationResult:
        if not config.stop_strings:
            return self._generate_without_streaming_overhead(prepared, config=config)

        text_parts: list[str] = []
        finish_reason = "length"
        prompt_tokens = 0
        completion_tokens = 0
        perf = None

        for delta, finished, reason, prompt_count, completion_count, perf_stats in self._iter_generation(prepared, config):
            if delta:
                text_parts.append(delta)
            prompt_tokens = prompt_count
            completion_tokens = completion_count
            if perf_stats is not None:
                perf = perf_stats
            if finished:
                finish_reason = reason or "stop"
                break

        return TextGenerationResult(
            text="".join(text_parts),
            reasoning_text=None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            perf=perf,
        )

    def _generate_without_streaming_overhead(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> TextGenerationResult:
        completion_ids, finish_reason, prompt_tokens, completion_tokens, perf = self._generate_token_ids(
            prepared,
            config=config,
        )
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        return TextGenerationResult(
            text=text,
            reasoning_text=None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            perf=perf,
        )

    def _stream(self, prepared: PreparedInputs, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        self.ensure_accepting_requests()
        if self._can_use_scheduler(prepared):
            for event in self.scheduler.stream(prepared, config=config):
                if event.finish_reason is not None and event.perf is not None:
                    self._record_perf_latency(event.perf)
                yield event
            return
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.record_request_submitted(waiting=False)
        success = False
        try:
            for event in self._stream_direct(prepared, config=config):
                if event.finish_reason is not None and event.perf is not None:
                    self._record_perf_latency(event.perf)
                yield event
            success = True
        finally:
            if metrics is not None:
                metrics.record_request_finished(success=success)
            self._trim_runtime_cache_if_idle()

    def _stream_direct(self, prepared: PreparedInputs, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        try:
            for delta, finished, reason, prompt_tokens, completion_tokens, perf in self._iter_generation(prepared, config):
                if delta:
                    yield StreamEvent(text=delta, finish_reason=None)
                if finished:
                    yield StreamEvent(
                        text="",
                        finish_reason=reason or "stop",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        perf=perf,
                    )
                    return
        except RuntimeError as exc:
            raise self._handle_runtime_failure(exc) from exc

    def _init_repetition_penalty_state(
        self,
        prompt_ids: list[int],
        penalty: float,
        presence_penalty: float = 0.0,
    ) -> tuple[torch.Tensor | None, set[int] | None]:
        if penalty == 1.0 and presence_penalty == 0.0:
            return None, None
        unique_ids = list(dict.fromkeys(prompt_ids))
        if not unique_ids:
            return None, set()
        history_tensor = self.device_context.move_token_ids(
            torch.tensor(unique_ids, dtype=torch.long, device=self.device_context.device)
        )
        return history_tensor, set(unique_ids)

    def _append_repetition_penalty_token(
        self,
        *,
        history_tensor: torch.Tensor | None,
        history_ids: set[int] | None,
        next_token: torch.Tensor,
        token_id: int | None = None,
    ) -> tuple[torch.Tensor | None, set[int] | None]:
        if history_tensor is None or history_ids is None:
            return history_tensor, history_ids

        token_id = self._token_id_from_tensor(next_token) if token_id is None else int(token_id)
        if token_id in history_ids:
            return history_tensor, history_ids

        history_ids.add(token_id)
        appended = next_token.view(1)
        if history_tensor.device != appended.device:
            appended = appended.to(device=history_tensor.device)
        if history_tensor.numel() == 0:
            return appended, history_ids
        return torch.cat([history_tensor, appended]), history_ids

    @staticmethod
    def _append_repetition_penalty_token_device(
        history_tensor: torch.Tensor | None,
        next_token: torch.Tensor,
    ) -> torch.Tensor | None:
        """Device-only penalty-history append for deferred token-pull paths (P0-#12).

        The host token id is not known yet, so the host-side dedup set cannot be
        maintained here (it is backfilled when the bulk pull happens). Duplicate
        entries are harmless: apply_*_penalty runs torch.unique.
        """
        if history_tensor is None:
            return None
        flat = next_token.detach().reshape(1)
        if flat.device != history_tensor.device:
            flat = flat.to(device=history_tensor.device)
        if history_tensor.numel() == 0:
            return flat
        return torch.cat([history_tensor, flat])

    @staticmethod
    def _token_id_from_tensor(next_token: torch.Tensor) -> int:
        return token_ids_to_host(next_token)[0]

    @staticmethod
    def _stop_token_tensor(stop_token_ids: set[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not stop_token_ids:
            return torch.empty((0,), dtype=dtype, device=device)
        return torch.tensor(sorted(stop_token_ids), dtype=dtype, device=device)

    def _raise_if_generation_cancelled(self, config: GenerationConfig) -> None:
        if config.cancellation_event is not None and config.cancellation_event.is_set():
            raise AnnaEngineError(
                "Generation cancelled because the client disconnected.",
                status_code=499,
                error_type="server_error",
                code="client_disconnected",
            )

    def _stop_token_ids(self) -> set[int]:
        token_ids = set(self.tokenizer.eos_token_ids)
        text_config = getattr(getattr(self, "config", None), "text_config", None)
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids.add(int(eos_token_id))
        return token_ids

    def _generate_token_ids(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> tuple[list[int], str, int, int, GenerationPerfStats]:
        # Non-streaming path: keep sampled tokens on-device and bulk-sync once at the
        # end (or on stop) instead of a host round-trip every decode step.
        return self._generate_token_ids_device_loop(prepared, config=config)

    @torch.inference_mode()
    def _generate_token_ids_device_loop(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> tuple[list[int], str, int, int, GenerationPerfStats]:
        """Decode loop that defers host token materialization.

        Forward + sample stay on the execution device; stop checks use an on-device
        ``isin`` against EOS ids. Token ids are bulk-copied to host only when the
        sequence finishes (or cancellation forces an early exit). This cuts Python
        hot-loop overhead for non-streaming generation.
        """
        prompt_ids, prompt_length, config = self._validate_generation_request(prepared, config=config)
        prepared = self._move_prepared_for_generation(prepared, config=config)

        stop_token_ids = self._stop_token_ids()
        repetition_history, repetition_history_ids = self._init_repetition_penalty_state(
            prompt_ids,
            config.repetition_penalty,
            config.presence_penalty,
        )

        started_at = time.perf_counter()
        first_token_at = None
        past_key_values = None
        device_tokens: list[torch.Tensor] = []
        finish_reason = "length"
        input_ids = None
        # Deferred stop polling: accumulate an on-device EOS flag instead of a per-step
        # host sync; the flag is materialized every N steps (and on the final step) with
        # a single scalar transfer. Post-EOS tokens are trimmed after the final pull.
        stop_sync_every = max(1, int(os.environ.get("ANNA_STOP_SYNC_EVERY_STEPS", "16")))
        stop_hit: torch.Tensor | None = None

        try:
            with steady_decode_accumulation(enabled=self.optimization_config.profile_runtime, log=logger):
                try:
                    prefill = self._prefill_generation_prompt(prepared)
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc
                past_key_values = prefill.past_key_values
                current_logits = prefill.logits
                stop_tensor = self._stop_token_tensor(
                    stop_token_ids,
                    device=current_logits.device,
                    dtype=torch.long,
                )
                # Phase 2: optional decode-step graph execution. Only the plain
                # (non-topk) single-request forward is captured; sampling, stop
                # checks, and the fused lm-head candidate path stay eager.
                decode_graph_runner: DecodeGraphRunner | None = None
                decode_graph_backend: str | None = None
                if self.optimization_config.decode_executor != "eager":
                    resolved_mode, decode_graph_backend = resolve_decode_executor_mode(
                        self.optimization_config.decode_executor,
                        current_logits.device,
                    )
                    if resolved_mode == "graph" and decode_graph_backend is not None:
                        decode_graph_runner = DecodeGraphRunner(
                            current_logits.device,
                            backend=decode_graph_backend,
                        )
                        logger.info(
                            "Decode step graph executor enabled (backend=%s, device=%s).",
                            decode_graph_backend,
                            current_logits.device,
                        )

                for step_idx in range(config.max_new_tokens):
                    self._raise_if_generation_cancelled(config)
                    candidate_logits = None
                    candidate_token_ids = None
                    if step_idx > 0:
                        try:
                            with self.execution_lock:
                                candidate_count = self._fused_lm_head_candidate_count(config)
                                outputs = None
                                if candidate_count is not None:
                                    outputs = self._forward_generation_model_topk(
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                        top_k=candidate_count,
                                    )
                                elif decode_graph_runner is not None:
                                    try:
                                        outputs = decode_graph_runner.step(
                                            lambda static_ids: self._forward_generation_model(
                                                input_ids=static_ids,
                                                attention_mask=None,
                                                past_key_values=past_key_values,
                                                use_cache=True,
                                                logits_to_keep=1,
                                            ),
                                            input_ids,
                                        )
                                    except DecodeGraphUnavailable:
                                        # Permanent fallback for this generation: capture
                                        # or replay failed (dynamic shape, driver limits...).
                                        decode_graph_runner = None
                                if outputs is None:
                                    outputs = self._profiled_forward_generation_model(
                                        stage=f"decode[{step_idx}]",
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                    )
                            if hasattr(outputs, "candidate_logits"):
                                candidate_logits = outputs.candidate_logits[0, -1]
                                candidate_token_ids = outputs.candidate_token_ids[0, -1]
                            else:
                                current_logits = outputs.logits[0, -1]
                            past_key_values = outputs.past_key_values
                        except RuntimeError as exc:
                            raise self._handle_runtime_failure(exc) from exc

                    if candidate_logits is not None and candidate_token_ids is not None:
                        next_token = sample_next_token_from_candidates(
                            candidate_logits,
                            candidate_token_ids,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            min_p=config.min_p,
                        )
                    else:
                        next_token = sample_next_token(
                            current_logits,
                            generated_ids=repetition_history,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            min_p=config.min_p,
                            presence_penalty=config.presence_penalty,
                            repetition_penalty=config.repetition_penalty,
                        )

                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    # Deferred on-device stop check (no per-step .item() sync).
                    if stop_tensor is not None and stop_tensor.numel() > 0:
                        step_stop = torch.isin(next_token.detach().reshape(-1)[:1], stop_tensor).reshape(())
                        stop_hit = step_stop if stop_hit is None else (stop_hit | step_stop)
                        if (step_idx + 1) % stop_sync_every == 0 or step_idx + 1 >= config.max_new_tokens:
                            if bool(stop_hit.item()):
                                finish_reason = "stop"
                                break
                            stop_hit = None

                    # Keep penalty history on-device (no per-step host id). Duplicate
                    # entries are harmless: apply_*_penalty runs torch.unique.
                    if repetition_history is not None:
                        flat = next_token.detach().reshape(1)
                        if flat.device != repetition_history.device:
                            flat = flat.to(device=repetition_history.device)
                        if repetition_history.numel() == 0:
                            repetition_history = flat
                        else:
                            repetition_history = torch.cat([repetition_history, flat])

                    device_tokens.append(next_token.detach().reshape(()))
                    metrics = getattr(self, "metrics", None)
                    if metrics is not None:
                        metrics.record_generation_tokens(1)

                    input_ids = next_token.view(1, 1)

                    if step_idx + 1 >= config.max_new_tokens:
                        break

            if device_tokens:
                completion_ids = token_ids_to_host(torch.stack(device_tokens))
                if finish_reason == "stop" and stop_token_ids:
                    # Trim any tokens generated between the last stop poll and the break.
                    for token_offset, trimmed_id in enumerate(completion_ids):
                        if trimmed_id in stop_token_ids:
                            completion_ids = completion_ids[:token_offset]
                            break
            else:
                completion_ids = []

            total_seconds = time.perf_counter() - started_at
            prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
            perf = self._build_generation_perf_stats(
                prompt_tokens=prompt_length,
                completion_tokens=len(completion_ids),
                total_seconds=total_seconds,
                prefill_seconds=prefill_seconds,
                decode_seconds=max(0.0, total_seconds - prefill_seconds),
            )
            return completion_ids, finish_reason, prompt_length, len(completion_ids), perf
        finally:
            if past_key_values is not None:
                past_key_values.release()

    def _iter_generation(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> Iterator[tuple[str, bool, str | None, int, int, GenerationPerfStats | None]]:
        events = self._iter_generation_events(prepared, config, with_assembler=True)
        try:
            for delta, _token_id, finished, reason, prompt_length, completion_count, perf in events:
                if delta or finished:
                    yield delta, finished, reason, prompt_length, completion_count, perf
        finally:
            events.close()

    @torch.inference_mode()
    def _iter_generation_events(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
        *,
        with_assembler: bool,
    ) -> Iterator[tuple[str, int | None, bool, str | None, int, int, GenerationPerfStats | None]]:
        """Single-request generation loop shared by the token-ids and streaming paths.

        Yields ``(delta, token_id, finished, finish_reason, prompt_length, completion_count, perf)``.
        Token events carry ``token_id`` (with a possibly empty text delta); the final event carries
        ``finished=True`` plus the finish reason and perf stats. ``with_assembler`` enables
        incremental text decoding and stop-string detection.
        """
        prompt_ids, prompt_length, config = self._validate_generation_request(prepared, config=config)
        prepared = self._move_prepared_for_generation(prepared, config=config)

        completion_ids: list[int] = []
        stop_token_ids = self._stop_token_ids()
        repetition_history, repetition_history_ids = self._init_repetition_penalty_state(
            prompt_ids,
            config.repetition_penalty,
            config.presence_penalty,
        )
        text_assembler = (
            IncrementalTextAssembler(tokenizer=self.tokenizer, stop_strings=config.stop_strings)
            if with_assembler
            else None
        )

        started_at = time.perf_counter()
        first_token_at = None
        input_ids = None
        past_key_values = None
        stop_tensor: torch.Tensor | None = None

        def _finish_event(reason: str) -> tuple[str, int | None, bool, str | None, int, int, GenerationPerfStats]:
            total_seconds = time.perf_counter() - started_at
            prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
            return (
                "",
                None,
                True,
                reason,
                prompt_length,
                len(completion_ids),
                self._build_generation_perf_stats(
                    prompt_tokens=prompt_length,
                    completion_tokens=len(completion_ids),
                    total_seconds=total_seconds,
                    prefill_seconds=prefill_seconds,
                    decode_seconds=max(0.0, total_seconds - prefill_seconds),
                ),
            )

        def _flush_tail() -> str:
            if text_assembler is None:
                return ""
            tail, _ = text_assembler.flush()
            return tail

        try:
            with steady_decode_accumulation(enabled=self.optimization_config.profile_runtime, log=logger):
                try:
                    prefill = self._prefill_generation_prompt(prepared)
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc
                past_key_values = prefill.past_key_values
                current_logits = prefill.logits
                stop_tensor = self._stop_token_tensor(
                    stop_token_ids,
                    device=current_logits.device,
                    dtype=torch.long,
                )
                # Phase 2: optional decode-step graph execution (see the
                # non-streaming loop for the fallback contract).
                decode_graph_runner: DecodeGraphRunner | None = None
                if self.optimization_config.decode_executor != "eager":
                    resolved_mode, decode_graph_backend = resolve_decode_executor_mode(
                        self.optimization_config.decode_executor,
                        current_logits.device,
                    )
                    if resolved_mode == "graph" and decode_graph_backend is not None:
                        decode_graph_runner = DecodeGraphRunner(
                            current_logits.device,
                            backend=decode_graph_backend,
                        )
                        logger.info(
                            "Decode step graph executor enabled (backend=%s, device=%s).",
                            decode_graph_backend,
                            current_logits.device,
                        )

                for step_idx in range(config.max_new_tokens):
                    self._raise_if_generation_cancelled(config)
                    candidate_logits = None
                    candidate_token_ids = None
                    if step_idx > 0:
                        try:
                            with self.execution_lock:
                                candidate_count = self._fused_lm_head_candidate_count(config)
                                outputs = None
                                if candidate_count is not None:
                                    outputs = self._forward_generation_model_topk(
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                        top_k=candidate_count,
                                    )
                                elif decode_graph_runner is not None:
                                    try:
                                        outputs = decode_graph_runner.step(
                                            lambda static_ids: self._forward_generation_model(
                                                input_ids=static_ids,
                                                attention_mask=None,
                                                past_key_values=past_key_values,
                                                use_cache=True,
                                                logits_to_keep=1,
                                            ),
                                            input_ids,
                                        )
                                    except DecodeGraphUnavailable:
                                        # Permanent fallback for this generation: capture
                                        # or replay failed (dynamic shape, driver limits...).
                                        decode_graph_runner = None
                                if outputs is None:
                                    outputs = self._profiled_forward_generation_model(
                                        stage=f"decode[{step_idx}]",
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                    )
                            if hasattr(outputs, "candidate_logits"):
                                candidate_logits = outputs.candidate_logits[0, -1]
                                candidate_token_ids = outputs.candidate_token_ids[0, -1]
                            else:
                                current_logits = outputs.logits[0, -1]
                            past_key_values = outputs.past_key_values
                        except RuntimeError as exc:
                            raise self._handle_runtime_failure(exc) from exc

                    if candidate_logits is not None and candidate_token_ids is not None:
                        next_token = sample_next_token_from_candidates(
                            candidate_logits,
                            candidate_token_ids,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            min_p=config.min_p,
                        )
                    else:
                        next_token = sample_next_token(
                            current_logits,
                            generated_ids=repetition_history,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            min_p=config.min_p,
                            presence_penalty=config.presence_penalty,
                            repetition_penalty=config.repetition_penalty,
                        )
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    # Host-side stop check on the already-materialized token id: the
                    # streaming path pulls the id for text assembly anyway, so an extra
                    # device-side isin().item() sync per step would be pure overhead.
                    # EOS is still never fed to the assembler (checked before feed_token).
                    token_id = self._token_id_from_tensor(next_token)
                    if token_id in stop_token_ids:
                        tail = _flush_tail()
                        if tail:
                            yield tail, None, False, None, prompt_length, len(completion_ids), None
                        yield _finish_event("stop")
                        return

                    completion_ids.append(token_id)
                    metrics = getattr(self, "metrics", None)
                    if metrics is not None:
                        metrics.record_generation_tokens(1)
                    repetition_history, repetition_history_ids = self._append_repetition_penalty_token(
                        history_tensor=repetition_history,
                        history_ids=repetition_history_ids,
                        next_token=next_token,
                        token_id=token_id,
                    )
                    delta, hit_stop_string = (
                        text_assembler.feed_token(token_id) if text_assembler is not None else ("", False)
                    )

                    input_ids = next_token.view(1, 1)

                    yield delta, token_id, False, None, prompt_length, len(completion_ids), None

                    if hit_stop_string:
                        yield _finish_event("stop")
                        return

                    if step_idx + 1 >= config.max_new_tokens:
                        break

                tail = _flush_tail()
                if tail:
                    yield tail, None, False, None, prompt_length, len(completion_ids), None
                yield _finish_event("length")
        finally:
            if past_key_values is not None:
                past_key_values.release()
