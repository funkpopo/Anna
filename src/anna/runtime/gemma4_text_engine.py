from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch

from anna.mm.gemma4_text_processor import Gemma4TextProcessor
from anna.model.gemma4_text_model import Gemma4DynamicCache, Gemma4ForConditionalGeneration
from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import (
    _DEFAULT_REASONING_FORMAT,
    _format_bytes,
    AnnaQwen3_5TextEngine,
    EngineOptimizationConfig,
    ReasoningFormat,
    normalize_reasoning_format,
)
from anna.runtime.service_metrics import AnnaServiceMetrics
from anna.weights.gemma4_text_weight_loader import (
    build_gemma4_text_model,
    estimate_gemma4_text_model_weight_bytes,
    load_gemma4_text_model_config,
    load_gemma4_text_model_weights,
)
from anna.weights.gemma4_tokenizer import Gemma4Tokenizer

logger = logging.getLogger(__name__)


class _NullCacheAllocator:
    layers: tuple[()] = ()

    def trim(self) -> int:
        return 0


class AnnaGemma4TextEngine(AnnaQwen3_5TextEngine):
    model_family = "gemma4_text"

    def __init__(
        self,
        *,
        model: Gemma4ForConditionalGeneration,
        tokenizer: Gemma4Tokenizer,
        processor: Gemma4TextProcessor,
        model_id: str,
        device_context: DeviceContext,
        default_max_completion_tokens: int | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        optimization_config: EngineOptimizationConfig | None = None,
        offload_per_layer_input_embeddings: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.default_model_id = model_id
        self.device_context = device_context
        self.config = model.config
        self.quantized_replacements = 0
        self.default_max_completion_tokens = (
            None if default_max_completion_tokens is None else max(1, int(default_max_completion_tokens))
        )
        self.default_enable_thinking = bool(default_enable_thinking)
        self.reasoning_format = normalize_reasoning_format(reasoning_format)
        self.offload_mode = "none"
        self.offload_vision = False
        self.expert_quant = "none"
        self.weight_quant = "none"
        self.resident_expert_layer_indices = ()
        self.resident_expert_layers = 0
        self.cached_experts_per_layer = 0
        self.offload_per_layer_input_embeddings = bool(offload_per_layer_input_embeddings)
        self.optimization_config = self._normalize_optimization_config(optimization_config)
        self.cache_allocator = _NullCacheAllocator()
        self.full_attention_cache_mirror = False
        self.optimization_config = replace(
            self.optimization_config,
            prefill_chunk_size=self._resolve_prefill_chunk_size(self.optimization_config.prefill_chunk_size),
        )
        self.execution_lock = threading.Lock()
        self.scheduler = None
        self._compiled_text_forward = None
        self._prompt_cache: OrderedDict[tuple[int, ...], object] = OrderedDict()
        self.metrics = AnnaServiceMetrics()
        self._apply_runtime_optimizations()

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
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        offload_mode: str = "auto",
        offload_vision: bool = False,
        expert_quant: str = "auto",
        weight_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
    ) -> "AnnaGemma4TextEngine":
        if offload_mode.lower() not in {"auto", "none"}:
            raise ValueError("Gemma4 text runtime does not support expert offload.")
        if offload_vision:
            raise ValueError("Gemma4 text runtime does not load the vision tower, so --offload-vision is unsupported.")
        if expert_quant.lower() not in {"auto", "none"}:
            raise ValueError("Gemma4 text runtime does not support expert quantization.")
        if weight_quant.lower() not in {"auto", "none"}:
            raise ValueError("Gemma4 text runtime does not support dense runtime quantization.")
        if resident_expert_layers not in {None, 0}:
            raise ValueError("Gemma4 text runtime does not use resident expert layers.")
        if resident_expert_layer_indices not in {None, ()}:
            raise ValueError("Gemma4 text runtime does not use resident expert layers.")
        if cached_experts_per_layer not in {None, 0}:
            raise ValueError("Gemma4 text runtime does not use expert caching.")

        model_path = Path(model_dir)
        config = load_gemma4_text_model_config(model_path)
        device_context = DeviceContext.resolve(
            device=device,
            dtype=dtype,
            model_dtype=config.text_config.dtype,
        )
        if safety_policy is not None:
            device_context.safety_policy = safety_policy

        resolved_offload_per_layer_input_embeddings = cls._resolve_per_layer_input_embedding_offload(
            model_path=model_path,
            config=config,
            device_context=device_context,
        )
        build_device = (
            torch.device("cpu") if resolved_offload_per_layer_input_embeddings else device_context.device
        )

        model, _ = build_gemma4_text_model(
            config,
            device=build_device,
            dtype=device_context.dtype,
        )
        report = load_gemma4_text_model_weights(model, model_path)
        model.configure_runtime(
            device_context.device,
            offload_token_io=False,
            offload_per_layer_input_embeddings=resolved_offload_per_layer_input_embeddings,
        )
        model.eval()

        tokenizer = Gemma4Tokenizer.from_model_dir(model_path)
        processor = Gemma4TextProcessor(tokenizer)
        engine = cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_id=model_id or model_path.name,
            device_context=device_context,
            default_max_completion_tokens=(
                default_max_completion_tokens
                if default_max_completion_tokens is not None
                else config.default_max_completion_tokens
            ),
            default_enable_thinking=default_enable_thinking,
            reasoning_format=reasoning_format,
            optimization_config=EngineOptimizationConfig(
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                prefill_chunk_size=prefill_chunk_size,
                prompt_cache_size=prompt_cache_size,
                profile_runtime=profile_runtime,
            ),
            offload_per_layer_input_embeddings=resolved_offload_per_layer_input_embeddings,
        )
        logger.info(
            "Loaded Gemma4 text runtime from %s: loaded_tensors=%s skipped_tensors=%s device=%s dtype=%s offload_per_layer_input_embeddings=%s",
            model_path,
            report.loaded,
            report.skipped,
            device_context.device,
            device_context.dtype,
            resolved_offload_per_layer_input_embeddings,
        )
        return engine

    def _attach_cache_allocator(self) -> None:
        return None

    def _resolve_prefill_chunk_size(self, requested_chunk_size: int) -> int:
        if requested_chunk_size > 0:
            return requested_chunk_size
        if self.device_context.device.type != "xpu":
            return 0

        text_config = self.config.text_config
        bytes_per_elem = self.device_context.element_size(self.device_context.dtype)
        full_layers = sum(1 for layer_type in text_config.layer_types if layer_type == "full_attention")
        sliding_layers = max(0, int(text_config.num_hidden_layers) - full_layers)
        per_token_kv_bytes = (
            full_layers * 2 * int(text_config.num_key_value_heads) * int(text_config.global_head_dim) * bytes_per_elem
            + sliding_layers * 2 * int(text_config.num_key_value_heads) * int(text_config.head_dim) * bytes_per_elem
        )
        per_token_hidden_bytes = int(text_config.hidden_size) * bytes_per_elem * 10
        per_token_ple_bytes = (
            int(text_config.num_hidden_layers) * int(text_config.hidden_size_per_layer_input) * bytes_per_elem
        )
        estimated_bytes_per_token = max(1, per_token_kv_bytes + per_token_hidden_bytes + per_token_ple_bytes)

        memory_info = self.device_context.get_memory_info()
        if memory_info is None:
            target_chunk_budget = 96 << 20
        else:
            policy = self.device_context.safety_policy
            available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
            target_chunk_budget = max(128 << 20, min(max(available_budget // 6, 384 << 20), 1536 << 20))

        auto_chunk = int(target_chunk_budget // estimated_bytes_per_token)
        resolved = max(64, min(1024, auto_chunk))
        logger.info(
            "Enabled Gemma4 auto prefill chunking on %s: chunk_size=%s estimated_bytes_per_token=%s target_budget=%s",
            self.device_context.device,
            resolved,
            _format_bytes(estimated_bytes_per_token),
            _format_bytes(target_chunk_budget),
        )
        return resolved

    def _reserve_prefill_cache(self, prepared) -> Gemma4DynamicCache | None:
        batch_size = int(prepared.input_ids.shape[0])
        cache = Gemma4DynamicCache(self.config.text_config, batch_size=batch_size)
        cache.reserve_sequence_capacity(int(prepared.input_ids.shape[1]))
        return cache

    def _estimate_generation_memory_bytes(self, prepared, *, config) -> int:
        text_config = self.config.text_config
        bytes_per_elem = self.device_context.element_size(self.device_context.dtype)
        batch_size = int(prepared.input_ids.shape[0])
        total_tokens = int(prepared.input_ids.shape[1]) + config.max_new_tokens
        full_layers = sum(1 for layer_type in text_config.layer_types if layer_type == "full_attention")
        sliding_layers = max(0, int(text_config.num_hidden_layers) - full_layers)
        sliding_cache_tokens = min(total_tokens, int(text_config.sliding_window))

        full_kv_bytes = (
            batch_size
            * full_layers
            * 2
            * total_tokens
            * int(text_config.num_key_value_heads)
            * int(text_config.global_head_dim)
            * bytes_per_elem
        )
        sliding_kv_bytes = (
            batch_size
            * sliding_layers
            * 2
            * sliding_cache_tokens
            * int(text_config.num_key_value_heads)
            * int(text_config.head_dim)
            * bytes_per_elem
        )
        hidden_working_bytes = batch_size * total_tokens * int(text_config.hidden_size) * bytes_per_elem * 10
        per_layer_input_bytes = (
            batch_size
            * total_tokens
            * int(text_config.num_hidden_layers)
            * int(text_config.hidden_size_per_layer_input)
            * bytes_per_elem
        )
        estimated = full_kv_bytes + sliding_kv_bytes + hidden_working_bytes + per_layer_input_bytes
        return int(estimated * self.device_context.safety_policy.generation_memory_safety_factor)

    def _kv_cache_page_counts(self) -> tuple[int, int]:
        return 0, 0

    @staticmethod
    def _resolve_per_layer_input_embedding_offload(
        *,
        model_path: Path,
        config,
        device_context: DeviceContext,
    ) -> bool:
        if device_context.device.type != "xpu":
            return False
        if int(config.text_config.hidden_size_per_layer_input) <= 0:
            return False

        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return False

        weight_bytes = estimate_gemma4_text_model_weight_bytes(model_path)
        return weight_bytes > int(memory_info.total_bytes * 0.85)

    def health(self) -> dict[str, object]:
        memory_info = self.device_context.get_memory_info()
        service_metrics = self.service_metrics_snapshot()
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
            "float8_available": self.device_context.float8_available,
            "quantization": "dense",
            "weight_quant": self.weight_quant,
            "quantized_replacements": self.quantized_replacements,
            "offload_mode": self.offload_mode,
            "offload_vision": self.offload_vision,
            "expert_quant": self.expert_quant,
            "resident_expert_layers": self.resident_expert_layers,
            "resident_expert_layer_indices": [],
            "cached_experts_per_layer": self.cached_experts_per_layer,
            "offload_per_layer_input_embeddings": self.offload_per_layer_input_embeddings,
            "full_attention_cache_mirror": self.full_attention_cache_mirror,
            "runtime_optimizations": {
                "compile_mode": self.optimization_config.compile_mode,
                "compile_fullgraph": self.optimization_config.compile_fullgraph,
                "compiled_text_forward": self._compiled_text_forward is not None,
                "prefill_chunk_size": self.optimization_config.prefill_chunk_size,
                "prompt_cache_size": self.optimization_config.prompt_cache_size,
                "prompt_cache_entries": len(self._prompt_cache),
                "profile_runtime": self.optimization_config.profile_runtime,
            },
            "vision_enabled": False,
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
