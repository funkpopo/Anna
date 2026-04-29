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
from anna.model.fused_ops import maybe_load_gated_delta_library, paged_gqa_decode_fused_is_available
from anna.model.quantization import AutoRoundGPTQLinear, convert_module_linears_to_xpu_int4, estimate_module_xpu_int4_bytes
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator, Qwen3SparseMoeBlock
from anna.model.turboquant import VALID_KV_CACHE_QUANT_BITS, turboquant_is_available
from anna.model.xpu_decode_profile import record_steady_decode_step_if_applicable, steady_decode_accumulation
from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.memory_release import release_conversion_artifacts
from anna.runtime.service_metrics import AnnaServiceMetrics, ServiceMetricsSnapshot
from anna.runtime.streaming import IncrementalTextAssembler, strip_unstable_replacement_suffix
from anna.sampling.sampler import sample_next_token, sample_next_token_from_candidates
from anna.weights.qwen3_5_text_weight_loader import build_qwen3_5_text_model, estimate_qwen3_5_text_model_weight_bytes, load_qwen3_5_text_model_config, load_qwen3_5_text_model_weights
from anna.weights.qwen3_5_text_tokenizer import Qwen3_5TextTokenizer

logger = logging.getLogger(__name__)

ReasoningFormat = Literal["none", "deepseek"]
_REASONING_FORMAT_VALUES = frozenset({"none", "deepseek"})
_DEFAULT_REASONING_FORMAT: ReasoningFormat = "deepseek"
_COMPILE_MODE_VALUES = frozenset({"none", "auto", "default", "reduce-overhead", "max-autotune"})
_KV_CACHE_QUANTIZATION_VALUES = frozenset({"none", "turboquant"})


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


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
    top_p: float = 0.95
    top_k: int = 50
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
            "turboquant_applies_to": "full_attention_only" if turboquant_enabled else "disabled",
            "full_attention_layers": len(full_attention_layers),
            "full_attention_layer_indices": full_attention_layers,
            "turboquant_quantized_layers": len(full_attention_layers) if turboquant_enabled else 0,
            "turboquant_quantized_layer_indices": full_attention_layers if turboquant_enabled else [],
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
        safety_policy: RuntimeSafetyPolicy | None = None,
        default_max_completion_tokens: int | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        offload_mode: str = "auto",
        offload_vision: bool = False,
        expert_quant: str = "auto",
        weight_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
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
        resolved_kv_cache_quantization = cls._resolve_kv_cache_quantization(
            requested_mode=kv_cache_quantization,
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
                    cache_dir=model_path / ".anna" / "xpu_int4_cache",
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
            logger.info("Preparing loaded quantized Qwen3.5 modules for XPU execution.")
            cls._prepare_loaded_quantized_modules_for_execution(
                model=model,
                config=config,
                device=device_context.device,
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
            "Loaded model %s on %s (compute=%s, requested=%s, default_max_completion_tokens=%s, default_enable_thinking=%s, reasoning_format=%s, offload=%s, offload_vision=%s, expert_quant=%s, weight_quant=%s, resident_expert_layers=%s, resident_expert_layer_indices=%s, cached_experts_per_layer=%s, full_attention_cache_mirror=%s, weight_load_device=%s); tensors loaded=%s skipped=%s quantized=%s",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.reported_dtype,
            resolved_default_max_completion_tokens,
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

        engine = cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_id=resolved_model_id,
            device_context=device_context,
            quantized_replacements=report.quantized_replacements,
            default_max_completion_tokens=resolved_default_max_completion_tokens,
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
            ),
        )
        kv_cache_info = engine._kv_cache_runtime_info()
        logger.info(
            "Qwen3.5 KV cache runtime: mode=%s turboquant_enabled=%s turboquant_bits=%s turboquant_residual_len=%s full_attention_layers=%s turboquant_quantized_layers=%s",
            kv_cache_info.get("mode"),
            kv_cache_info.get("turboquant_enabled"),
            kv_cache_info.get("turboquant_bits"),
            kv_cache_info.get("turboquant_residual_len"),
            kv_cache_info.get("full_attention_layers"),
            kv_cache_info.get("turboquant_quantized_layers"),
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
        device_context: DeviceContext,
    ) -> str:
        normalized = normalize_kv_cache_quantization(requested_mode)
        if normalized == "turboquant" and not turboquant_is_available():
            raise ValueError(
                "TurboQuant KV-cache compression was requested, but the 'turboquant' dependency is not installed."
            )
        return normalized

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
        usage_threshold = 0.70 if resolved_offload_mode == "experts" or config.text_config.is_moe_model else 0.85
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

        replacements = convert_module_linears_to_xpu_int4(
            model,
            compute_dtype=compute_dtype,
            device=device,
            include_predicate=_should_quantize,
            cache_dir=cache_dir,
        )
        logger.info(
            "Runtime dense XPU int4 quantization: replacements=%s device=%s compute_dtype=%s cache_dir=%s",
            replacements,
            device,
            compute_dtype,
            cache_dir,
        )
        return replacements

    @classmethod
    def _prepare_loaded_quantized_modules_for_execution(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        config: object,
        device: torch.device,
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

    @classmethod
    def _estimate_resident_expert_layer_indices(
        cls,
        *,
        model: Qwen3_5TextForConditionalGeneration,
        device_context: DeviceContext,
        expert_quant: str,
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

        selected_indices: list[int] = []
        consumed_bytes = 0
        layer_sizes: list[tuple[int, int]] = []
        for layer_idx, layer in enumerate(text_model.layers):
            if not isinstance(layer.mlp, Qwen3SparseMoeBlock):
                continue
            layer_bytes = (
                estimate_module_xpu_int4_bytes(layer.mlp.experts)
                if expert_quant == "int4"
                else cls._module_nbytes(layer.mlp.experts)
            )
            layer_sizes.append((layer_idx, layer_bytes))
            if layer_bytes <= 0 or consumed_bytes + layer_bytes > budget_bytes:
                break
            selected_indices.append(layer_idx)
            consumed_bytes += layer_bytes

        logger.info(
            "Auto resident expert placement: expert_quant=%s free=%s reserve=%s budget_factor=%.2f budget=%s selected_layers=%s selected_bytes=%s candidate_layer_bytes=%s",
            expert_quant,
            format_bytes(memory_info.free_bytes),
            format_bytes(reserve_bytes),
            budget_factor,
            format_bytes(budget_bytes),
            selected_indices,
            format_bytes(consumed_bytes),
            {layer_idx: format_bytes(layer_bytes) for layer_idx, layer_bytes in layer_sizes[:8]},
        )
        return tuple(selected_indices)

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
        cache.set_prompt_token_ids(prepared.input_ids)
        return cache

    def _trim_runtime_cache_if_idle(self) -> None:
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            snapshot = metrics.snapshot()
            if snapshot.running_requests > 0 or snapshot.waiting_requests > 0:
                return
        trimmed_pages = self.cache_allocator.trim()
        if trimmed_pages <= 0:
            return
        release_unused_memory = getattr(self.device_context, "release_unused_memory", None)
        if callable(release_unused_memory):
            release_unused_memory()
        logger.info("Trimmed idle KV cache pages: released_pages=%s", trimmed_pages)

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

    def _kv_cache_page_counts(self) -> tuple[int, int]:
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
        return used_pages, total_pages

    def service_metrics_snapshot(self) -> ServiceMetricsSnapshot:
        metrics = getattr(self, "metrics", None)
        snapshot = metrics.snapshot() if metrics is not None else ServiceMetricsSnapshot(timestamp=time.perf_counter())
        used_pages, total_pages = self._kv_cache_page_counts()
        return replace(
            snapshot,
            kv_cache_used_pages=used_pages,
            kv_cache_total_pages=total_pages,
            prompt_cache_entries=len(getattr(self, "_prompt_cache", {})),
        )

    def health(self) -> dict[str, Any]:
        quant_method = self.config.quantization_config.quant_method or "dense"
        memory_info = self.device_context.get_memory_info()
        service_metrics = self.service_metrics_snapshot()
        kv_cache_runtime_info = self._kv_cache_runtime_info()
        return {
            "status": "ok",
            "model": self.default_model_id,
            "model_family": self.model_family,
            "device": str(self.device_context.device),
            "compute_dtype": str(self.device_context.dtype),
            "requested_dtype": self.device_context.requested_dtype,
            "reported_dtype": self.device_context.reported_dtype,
            "default_max_completion_tokens": self.default_max_completion_tokens,
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
            "full_attention_cache_mirror": self.full_attention_cache_mirror,
            "runtime_optimizations": {
                "compile_mode": self.optimization_config.compile_mode,
                "compile_fullgraph": self.optimization_config.compile_fullgraph,
                "compiled_text_forward": self._compiled_text_forward is not None,
                "prefill_chunk_size": self.optimization_config.prefill_chunk_size,
                "prompt_cache_size": self.optimization_config.prompt_cache_size,
                "prompt_cache_max_tokens": self.optimization_config.prompt_cache_max_tokens,
                "prompt_cache_entries": len(self._prompt_cache),
                "profile_runtime": self.optimization_config.profile_runtime,
                "kv_cache_quantization": self.optimization_config.kv_cache_quantization,
                "kv_cache_quant_bits": self.optimization_config.kv_cache_quant_bits,
                "kv_cache_residual_len": self.optimization_config.kv_cache_residual_len,
                "xpu_int4_kernels": {
                    "matmul_strategy": os.getenv("ANNA_XPU_INT4_MATMUL", "auto"),
                    "auto_gemv_enabled": os.getenv("ANNA_XPU_AUTO_INT4_GEMV"),
                    "gemv_kernel": os.getenv("ANNA_XPU_INT4_GEMV_KERNEL"),
                    "gemv_local_size": os.getenv("ANNA_XPU_INT4_GEMV_LOCAL_SIZE"),
                    "gemv_output_tile": os.getenv("ANNA_XPU_INT4_GEMV_OUTPUT_TILE"),
                    "lm_head_local_size": os.getenv("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"),
                    "lm_head_block_topk_threshold": os.getenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_TOPK_THRESHOLD", "65536"),
                    "lm_head_block_size": os.getenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_SIZE", "4096"),
                    "moe_gate_local_size": os.getenv("ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE"),
                    "moe_down_local_size": os.getenv("ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE"),
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
                "running_requests": service_metrics.running_requests,
                "waiting_requests": service_metrics.waiting_requests,
                "kv_cache_used_pages": service_metrics.kv_cache_used_pages,
                "kv_cache_total_pages": service_metrics.kv_cache_total_pages,
            },
        }

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
        yield from self._stream(prepared, config=config)

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
        for event in self._stream(prepared, config=config):
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
        return (
            self.scheduler is not None
            and prepared.pixel_values is None
            and prepared.pixel_values_videos is None
            and prepared.input_features is None
        )

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
    ) -> None:
        cache_length = getattr(past_key_values, "get_seq_length", None)
        seen_tokens = cache_length() if callable(cache_length) else 0
        logger.info(
            "xpu_profile stage=%s input_tokens=%s cache_tokens=%s elapsed_seconds=%.6f free_before=%s free_after=%s stats_before=%s stats_after=%s",
            stage,
            int(input_ids.shape[-1]),
            seen_tokens,
            elapsed_seconds,
            format_bytes(memory_before.free_bytes if memory_before is not None else None),
            format_bytes(memory_after.free_bytes if memory_after is not None else None),
            stats_before,
            stats_after,
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
        record_steady_decode_step_if_applicable(stage, ms)
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

    def warmup_inference_kernels(self) -> None:
        from anna.model.fused_ops import maybe_load_gated_delta_library

        maybe_load_gated_delta_library()
        cfg = self.config.text_config
        pad = int(cfg.pad_token_id)
        if not (0 <= pad < cfg.vocab_size):
            pad = 0
        bos = getattr(self.tokenizer, "bos_token_id", None)
        token_id = int(bos) if bos is not None and 0 <= int(bos) < cfg.vocab_size else pad
        device = self.device_context.device
        with torch.inference_mode():
            # Fused causal_conv1d_prefill / gated_delta_prefill require seq_len > 1 (see SYCL TORCH_CHECK).
            input_ids = torch.tensor([[token_id, token_id]], device=device, dtype=torch.long)
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
                self._forward_generation_model(
                    input_ids=torch.tensor([[token_id]], device=device, dtype=torch.long),
                    attention_mask=None,
                    past_key_values=past,
                    model_kwargs={},
                    use_cache=True,
                    logits_to_keep=1,
                )
                past.release()
        logger.info("XPU inference warmup finished (2-token prefill + 1-token decode).")

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

        if category == "out_of_memory":
            return AnnaEngineError(
                "XPU out of memory during generation. Reduce prompt length, media size, batch size, or max_completion_tokens.",
                status_code=503,
                error_type="server_error",
                code="device_out_of_memory",
            )
        if category == "device_lost":
            return AnnaEngineError(
                "XPU device was lost during generation. The runtime cache was cleared; retry the request after the device recovers.",
                status_code=503,
                error_type="server_error",
                code="device_lost",
            )
        if category == "out_of_resources":
            return AnnaEngineError(
                "XPU runtime ran out of resources during generation. Reduce the request size and retry.",
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

    def _generate(self, prepared: PreparedInputs, *, config: GenerationConfig) -> TextGenerationResult:
        if self._can_use_scheduler(prepared):
            return self.scheduler.generate(prepared, config=config)
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.record_request_submitted(waiting=False)
        success = False
        try:
            result = self._generate_direct(prepared, config=config)
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
        if self._can_use_scheduler(prepared):
            yield from self.scheduler.stream(prepared, config=config)
            return
        metrics = getattr(self, "metrics", None)
        if metrics is not None:
            metrics.record_request_submitted(waiting=False)
        success = False
        try:
            yield from self._stream_direct(prepared, config=config)
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

    def _trim_stop_strings(self, text: str, stop_strings: list[str]) -> tuple[str, bool]:
        if not stop_strings:
            return text, False
        indexes = [text.find(stop) for stop in stop_strings if stop and stop in text]
        if not indexes:
            return text, False
        trim_at = min(indexes)
        return text[:trim_at], True

    def _stable_decode_delta(
        self,
        *,
        previous_text: str,
        current_text: str,
        emitted_text: str,
    ) -> tuple[str, str]:
        stable_length = _common_prefix_length(previous_text, current_text)
        stable_text = strip_unstable_replacement_suffix(current_text[:stable_length])
        if not stable_text.startswith(emitted_text):
            return "", emitted_text
        return stable_text[len(emitted_text) :], stable_text

    def _flush_decode_tail(self, *, current_text: str, emitted_text: str) -> tuple[str, str]:
        current_text = strip_unstable_replacement_suffix(current_text)
        if not current_text.startswith(emitted_text):
            return "", emitted_text
        return current_text[len(emitted_text) :], current_text

    def _init_repetition_penalty_state(self, prompt_ids: list[int], penalty: float) -> tuple[torch.Tensor | None, set[int] | None]:
        if penalty == 1.0:
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
    ) -> tuple[torch.Tensor | None, set[int] | None]:
        if history_tensor is None or history_ids is None:
            return history_tensor, history_ids

        token_id = int(next_token.item())
        if token_id in history_ids:
            return history_tensor, history_ids

        history_ids.add(token_id)
        appended = next_token.view(1)
        if history_tensor.device != appended.device:
            appended = appended.to(device=history_tensor.device)
        if history_tensor.numel() == 0:
            return appended, history_ids
        return torch.cat([history_tensor, appended]), history_ids

    def _raise_if_generation_cancelled(self, config: GenerationConfig) -> None:
        if config.cancellation_event is not None and config.cancellation_event.is_set():
            raise AnnaEngineError(
                "Generation cancelled because the client disconnected.",
                status_code=499,
                error_type="server_error",
                code="client_disconnected",
            )

    @torch.inference_mode()
    def _generate_token_ids(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> tuple[list[int], str, int, int, GenerationPerfStats]:
        prompt_ids, prompt_length, config = self._validate_generation_request(prepared, config=config)
        prepared = self._move_prepared_for_generation(prepared, config=config)

        completion_ids: list[int] = []
        stop_token_ids = set(self.tokenizer.eos_token_ids)
        repetition_history, repetition_history_ids = self._init_repetition_penalty_state(
            prompt_ids,
            config.repetition_penalty,
        )
        started_at = time.perf_counter()
        first_token_at = None
        input_ids = None
        past_key_values = None

        try:
            with steady_decode_accumulation(enabled=self.optimization_config.profile_runtime, log=logger):
                try:
                    prefill = self._prefill_generation_prompt(prepared)
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc
                past_key_values = prefill.past_key_values
                current_logits = prefill.logits

                for step_idx in range(config.max_new_tokens):
                    self._raise_if_generation_cancelled(config)
                    candidate_logits = None
                    candidate_token_ids = None
                    if step_idx > 0:
                        try:
                            with self.execution_lock:
                                candidate_count = self._fused_lm_head_candidate_count(config)
                                outputs = (
                                    self._forward_generation_model_topk(
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                        top_k=candidate_count,
                                    )
                                    if candidate_count is not None
                                    else None
                                )
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
                        )
                    else:
                        next_token = sample_next_token(
                            current_logits,
                            generated_ids=repetition_history,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            repetition_penalty=config.repetition_penalty,
                        )
                    token_id = int(next_token.item())
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    if token_id in stop_token_ids:
                        total_seconds = time.perf_counter() - started_at
                        prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
                        return (
                            completion_ids,
                            "stop",
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

                    completion_ids.append(token_id)
                    metrics = getattr(self, "metrics", None)
                    if metrics is not None:
                        metrics.record_generation_tokens(1)
                    repetition_history, repetition_history_ids = self._append_repetition_penalty_token(
                        history_tensor=repetition_history,
                        history_ids=repetition_history_ids,
                        next_token=next_token,
                    )

                    input_ids = next_token.view(1, 1)

                    if step_idx + 1 >= config.max_new_tokens:
                        break

                total_seconds = time.perf_counter() - started_at
                prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
                return (
                    completion_ids,
                    "length",
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
        finally:
            if past_key_values is not None:
                past_key_values.release()

    @torch.inference_mode()
    def _iter_generation(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> Iterator[tuple[str, bool, str | None, int, int, GenerationPerfStats | None]]:
        prompt_ids, prompt_length, config = self._validate_generation_request(prepared, config=config)
        prepared = self._move_prepared_for_generation(prepared, config=config)
        completion_ids: list[int] = []
        stop_token_ids = set(self.tokenizer.eos_token_ids)
        repetition_history, repetition_history_ids = self._init_repetition_penalty_state(
            prompt_ids,
            config.repetition_penalty,
        )
        text_assembler = IncrementalTextAssembler(
            tokenizer=self.tokenizer,
            stop_strings=config.stop_strings,
        )

        started_at = time.perf_counter()
        first_token_at = None
        input_ids = None
        past_key_values = None

        try:
            with steady_decode_accumulation(enabled=self.optimization_config.profile_runtime, log=logger):
                try:
                    prefill = self._prefill_generation_prompt(prepared)
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc
                past_key_values = prefill.past_key_values
                current_logits = prefill.logits

                for step_idx in range(config.max_new_tokens):
                    self._raise_if_generation_cancelled(config)
                    candidate_logits = None
                    candidate_token_ids = None
                    if step_idx > 0:
                        try:
                            with self.execution_lock:
                                candidate_count = self._fused_lm_head_candidate_count(config)
                                outputs = (
                                    self._forward_generation_model_topk(
                                        input_ids=input_ids,
                                        attention_mask=None,
                                        past_key_values=past_key_values,
                                        use_cache=True,
                                        logits_to_keep=1,
                                        top_k=candidate_count,
                                    )
                                    if candidate_count is not None
                                    else None
                                )
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
                        )
                    else:
                        next_token = sample_next_token(
                            current_logits,
                            generated_ids=repetition_history,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            top_k=config.top_k,
                            repetition_penalty=config.repetition_penalty,
                        )
                    token_id = int(next_token.item())
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    if token_id in stop_token_ids:
                        tail, _ = text_assembler.flush()
                        if tail:
                            yield tail, False, None, prompt_length, len(completion_ids), None
                        total_seconds = time.perf_counter() - started_at
                        prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
                        yield (
                            "",
                            True,
                            "stop",
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
                        return

                    completion_ids.append(token_id)
                    metrics = getattr(self, "metrics", None)
                    if metrics is not None:
                        metrics.record_generation_tokens(1)
                    repetition_history, repetition_history_ids = self._append_repetition_penalty_token(
                        history_tensor=repetition_history,
                        history_ids=repetition_history_ids,
                        next_token=next_token,
                    )
                    delta, hit_stop_string = text_assembler.feed_token(token_id)

                    input_ids = next_token.view(1, 1)

                    if delta:
                        yield delta, False, None, prompt_length, len(completion_ids), None

                    if hit_stop_string:
                        total_seconds = time.perf_counter() - started_at
                        prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
                        yield (
                            "",
                            True,
                            "stop",
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
                        return

                    if step_idx + 1 >= config.max_new_tokens:
                        break

                tail, _ = text_assembler.flush()
                if tail:
                    yield tail, False, None, prompt_length, len(completion_ids), None
                total_seconds = time.perf_counter() - started_at
                prefill_seconds = total_seconds if first_token_at is None else first_token_at - started_at
                yield (
                    "",
                    True,
                    "length",
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
        finally:
            if past_key_values is not None:
                past_key_values.release()
