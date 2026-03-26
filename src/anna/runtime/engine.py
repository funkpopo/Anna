from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from anna.mm.processor import PreparedInputs, Qwen3MultimodalProcessor
from anna.model.quantization import estimate_module_xpu_int4_bytes
from anna.model.qwen import Qwen3ForConditionalGeneration
from anna.model.ops import Qwen3PageAllocator, Qwen3SparseMoeBlock
from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.streaming import IncrementalTextAssembler, strip_unstable_replacement_suffix
from anna.sampling.sampler import sample_next_token
from anna.weights.loader import build_model, estimate_model_weight_bytes, load_model_config, load_model_weights
from anna.weights.tokenizer import QwenTokenizer

logger = logging.getLogger(__name__)


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _strip_think_open_tag(text: str) -> str:
    normalized = text
    if normalized.startswith("<think>"):
        normalized = normalized[len("<think>") :]
        normalized = normalized.lstrip("\r\n")
    return normalized


def _strip_unstable_replacement_suffix(text: str) -> str:
    return strip_unstable_replacement_suffix(text)


class ThinkingStreamParser:
    CLOSE_TAG = "</think>"

    def __init__(self, *, enable_thinking: bool) -> None:
        self.enable_thinking = enable_thinking
        self.state = "reasoning" if enable_thinking else "content"
        self.buffer = ""

    def _emit_reasoning_prefix(self) -> list[tuple[str, str]]:
        if self.state != "reasoning":
            return []
        self.buffer = _strip_think_open_tag(self.buffer)
        safe_length = max(0, len(self.buffer) - (len(self.CLOSE_TAG) - 1))
        if safe_length <= 0:
            return []
        reasoning = self.buffer[:safe_length]
        self.buffer = self.buffer[safe_length:]
        return [("reasoning", reasoning)]

    def feed(self, text: str) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        self.buffer += text

        while True:
            if self.state == "reasoning":
                self.buffer = _strip_think_open_tag(self.buffer)
                close_index = self.buffer.find(self.CLOSE_TAG)
                if close_index == -1:
                    outputs.extend(self._emit_reasoning_prefix())
                    break
                reasoning = self.buffer[:close_index]
                if reasoning:
                    outputs.append(("reasoning", reasoning))
                self.buffer = self.buffer[close_index + len(self.CLOSE_TAG) :].lstrip("\r\n")
                self.state = "content"
                continue

            if self.buffer:
                outputs.append(("content", self.buffer))
                self.buffer = ""
            break

        return outputs

    def flush(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if self.state == "reasoning":
            self.buffer = _strip_think_open_tag(self.buffer)
            if self.buffer:
                outputs.append(("reasoning", self.buffer))
        elif self.buffer:
            outputs.append(("content", self.buffer))
        self.buffer = ""
        return outputs


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
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_strings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StreamEvent:
    text: str
    reasoning_text: str | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class TextGenerationResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_text: str | None = None


class AnnaEngine:
    def __init__(
        self,
        *,
        model: Qwen3ForConditionalGeneration,
        tokenizer: QwenTokenizer,
        processor: Qwen3MultimodalProcessor,
        model_id: str,
        device_context: DeviceContext,
        quantized_replacements: int = 0,
        offload_mode: str = "none",
        expert_quant: str = "none",
        resident_expert_layer_indices: tuple[int, ...] = (),
        cached_experts_per_layer: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.default_model_id = model_id
        self.device_context = device_context
        self.config = model.config
        self.quantized_replacements = quantized_replacements
        self.offload_mode = offload_mode
        self.expert_quant = expert_quant
        self.resident_expert_layer_indices = tuple(resident_expert_layer_indices)
        self.resident_expert_layers = len(self.resident_expert_layer_indices)
        self.cached_experts_per_layer = max(0, int(cached_experts_per_layer))
        self.cache_allocator = Qwen3PageAllocator(self.config.text_config)
        self._attach_cache_allocator()
        self.execution_lock = threading.Lock()
        self.scheduler = None

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_id: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        safety_policy: RuntimeSafetyPolicy | None = None,
        offload_mode: str = "auto",
        expert_quant: str = "auto",
        resident_expert_layers: int | None = None,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        cached_experts_per_layer: int | None = None,
    ) -> "AnnaEngine":
        model_path = Path(model_dir)
        config = load_model_config(model_path)
        device_context = DeviceContext.resolve(
            device=device,
            dtype=dtype,
            model_dtype=config.text_config.dtype,
        )
        if safety_policy is not None:
            device_context.safety_policy = safety_policy
        resolved_offload_mode = cls._resolve_offload_mode(
            requested_mode=offload_mode,
            model_path=model_path,
            config=config,
            device_context=device_context,
        )
        resolved_expert_quant = cls._resolve_expert_quant(
            requested_quant=expert_quant,
            resolved_offload_mode=resolved_offload_mode,
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
        model_device = torch.device("cpu") if resolved_offload_mode == "experts" else device_context.device
        try:
            model, quantized_replacements = build_model(
                config,
                device=model_device,
                dtype=device_context.dtype,
            )
            report = load_model_weights(model, model_path)
            report.quantized_replacements = quantized_replacements
            auto_resident_indices = resolved_resident_expert_layer_indices is None
            auto_cached_experts_per_layer = resolved_cached_experts_per_layer is None
            initial_resident_expert_layer_indices = () if auto_resident_indices else resolved_resident_expert_layer_indices
            initial_cached_experts_per_layer = 0 if auto_cached_experts_per_layer else resolved_cached_experts_per_layer

            model.configure_runtime(
                device_context.device,
                offload_experts=resolved_offload_mode == "experts",
                offload_vision=resolved_offload_mode == "experts",
                offload_token_io=False,
                resident_expert_layers=0,
                resident_expert_layer_indices=initial_resident_expert_layer_indices,
                expert_quant=resolved_expert_quant,
                cached_experts_per_layer=initial_cached_experts_per_layer,
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
                    offload_vision=resolved_offload_mode == "experts",
                    offload_token_io=False,
                    resident_expert_layers=0,
                    resident_expert_layer_indices=resolved_resident_expert_layer_indices,
                    expert_quant=resolved_expert_quant,
                    cached_experts_per_layer=initial_cached_experts_per_layer,
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
                    offload_vision=resolved_offload_mode == "experts",
                    offload_token_io=False,
                    resident_expert_layers=0,
                    resident_expert_layer_indices=resolved_resident_expert_layer_indices,
                    expert_quant=resolved_expert_quant,
                    cached_experts_per_layer=resolved_cached_experts_per_layer,
                )
            resolved_cached_experts_per_layer = cls._effective_cached_experts_per_layer(model)
            model.eval()
        except RuntimeError as exc:
            if device_context.should_recover(exc):
                try:
                    device_context.recover()
                except Exception:  # pragma: no cover - best-effort recovery
                    logger.exception("Failed to recover device context after model load failure.")
            raise

        tokenizer = QwenTokenizer.from_model_dir(model_path)
        processor = Qwen3MultimodalProcessor(config, tokenizer)
        resolved_model_id = model_id or model_path.name

        logger.info(
            "Loaded model %s on %s (compute=%s, requested=%s, offload=%s, expert_quant=%s, resident_expert_layers=%s, resident_expert_layer_indices=%s, cached_experts_per_layer=%s, weights=%s); tensors loaded=%s skipped=%s quantized=%s",
            resolved_model_id,
            device_context.device,
            device_context.dtype,
            device_context.reported_dtype,
            resolved_offload_mode,
            resolved_expert_quant,
            len(resolved_resident_expert_layer_indices or ()),
            list(resolved_resident_expert_layer_indices or ()),
            resolved_cached_experts_per_layer,
            model_device,
            report.loaded,
            report.skipped,
            report.quantized_replacements,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_id=resolved_model_id,
            device_context=device_context,
            quantized_replacements=quantized_replacements,
            offload_mode=resolved_offload_mode,
            expert_quant=resolved_expert_quant,
            resident_expert_layer_indices=tuple(resolved_resident_expert_layer_indices or ()),
            cached_experts_per_layer=int(resolved_cached_experts_per_layer or 0),
        )

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
        if device_context.device.type != "xpu":
            return "none"
        if normalized != "auto":
            return normalized

        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return "none"

        weight_bytes = estimate_model_weight_bytes(model_path)
        if config.text_config.is_moe_model and weight_bytes > int(memory_info.total_bytes * 0.85):
            return "experts"
        return "none"

    @staticmethod
    def _resolve_expert_quant(
        *,
        requested_quant: str,
        resolved_offload_mode: str,
        device_context: DeviceContext,
    ) -> str:
        normalized = requested_quant.lower()
        if normalized not in {"auto", "none", "int4"}:
            raise ValueError(f"Unsupported expert quant mode: {requested_quant}")
        if resolved_offload_mode != "experts" or device_context.device.type != "xpu":
            return "none"
        if normalized == "auto":
            return "int4"
        return normalized

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
    def _text_model(model: Qwen3ForConditionalGeneration) -> object | None:
        text_model = getattr(getattr(model, "model", None), "language_model", None)
        if text_model is None:
            text_model = getattr(model, "model", None)
        return text_model

    @classmethod
    def _offloaded_sparse_moe_blocks(cls, model: Qwen3ForConditionalGeneration) -> list[tuple[int, Qwen3SparseMoeBlock]]:
        text_model = cls._text_model(model)
        if text_model is None or not hasattr(text_model, "layers"):
            return []
        blocks: list[tuple[int, Qwen3SparseMoeBlock]] = []
        for layer_idx, layer in enumerate(text_model.layers):
            if isinstance(getattr(layer, "mlp", None), Qwen3SparseMoeBlock) and layer.mlp.offload_experts:
                blocks.append((layer_idx, layer.mlp))
        return blocks

    @classmethod
    def _effective_cached_experts_per_layer(cls, model: Qwen3ForConditionalGeneration) -> int:
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
        model: Qwen3ForConditionalGeneration,
        device_context: DeviceContext,
        expert_quant: str,
    ) -> tuple[int, ...]:
        if device_context.device.type != "xpu":
            return ()

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
                _format_bytes(memory_info.free_bytes),
                _format_bytes(reserve_bytes),
                budget_factor,
                _format_bytes(budget_bytes),
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
            _format_bytes(memory_info.free_bytes),
            _format_bytes(reserve_bytes),
            budget_factor,
            _format_bytes(budget_bytes),
            selected_indices,
            _format_bytes(consumed_bytes),
            {layer_idx: _format_bytes(layer_bytes) for layer_idx, layer_bytes in layer_sizes[:8]},
        )
        return tuple(selected_indices)

    @classmethod
    def _estimate_cached_experts_per_layer(
        cls,
        *,
        model: Qwen3ForConditionalGeneration,
        device_context: DeviceContext,
        expert_quant: str,
    ) -> int:
        if device_context.device.type != "xpu":
            return 0

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

        cache_target_free_bytes = max(768 << 20, int(memory_info.total_bytes * 0.06))
        cache_budget_fraction = 0.65 if expert_quant == "int4" else 0.35
        cache_budget_bytes = int(max(0, int(memory_info.free_bytes) - cache_target_free_bytes) * cache_budget_fraction)
        auto_cached = cache_budget_bytes // max(1, per_expert_bytes * len(offloaded_blocks))

        minimum_cache = exemplar_block.top_k
        max_cache = exemplar_block.num_experts
        minimum_budget_bytes = per_expert_bytes * len(offloaded_blocks) * minimum_cache
        if cache_budget_bytes < minimum_budget_bytes:
            resolved = max(0, min(max_cache, auto_cached))
        else:
            resolved = max(minimum_cache, min(max_cache, auto_cached))

        logger.info(
            "Auto expert cache sizing: expert_quant=%s free=%s target_free=%s cache_budget_fraction=%.2f cache_budget=%s offloaded_layers=%s per_expert=%s cached_experts_per_layer=%s",
            expert_quant,
            _format_bytes(memory_info.free_bytes),
            _format_bytes(cache_target_free_bytes),
            cache_budget_fraction,
            _format_bytes(cache_budget_bytes),
            len(offloaded_blocks),
            _format_bytes(per_expert_bytes),
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

    def set_scheduler(self, scheduler: object | None) -> None:
        self.scheduler = scheduler

    def health(self) -> dict[str, Any]:
        quant_method = self.config.quantization_config.quant_method or "dense"
        memory_info = self.device_context.get_memory_info()
        return {
            "status": "ok",
            "model": self.default_model_id,
            "device": str(self.device_context.device),
            "compute_dtype": str(self.device_context.dtype),
            "requested_dtype": self.device_context.requested_dtype,
            "reported_dtype": self.device_context.reported_dtype,
            "float8_available": self.device_context.float8_available,
            "quantization": quant_method,
            "quantized_replacements": self.quantized_replacements,
            "offload_mode": self.offload_mode,
            "expert_quant": self.expert_quant,
            "resident_expert_layers": self.resident_expert_layers,
            "resident_expert_layer_indices": self._resident_expert_layer_indices(),
            "cached_experts_per_layer": self.cached_experts_per_layer,
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
        prepared = self.processor.encode_text(prompt)
        return self._generate(prepared, config=config)

    def stream_text(self, prompt: str, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        prepared = self.processor.encode_text(prompt)
        yield from self._stream(prepared, config=config)

    def generate_chat(
        self,
        messages: list[object],
        *,
        config: GenerationConfig,
        enable_thinking: bool = False,
    ) -> TextGenerationResult:
        prepared = self._prepare_messages(messages, enable_thinking=enable_thinking)
        raw = self._generate(prepared, config=config)
        reasoning_text, content_text = self._split_chat_output(raw.text, enable_thinking=enable_thinking)
        return TextGenerationResult(
            text=content_text,
            reasoning_text=reasoning_text,
            finish_reason=raw.finish_reason,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
        )

    def stream_chat(
        self,
        messages: list[object],
        *,
        config: GenerationConfig,
        enable_thinking: bool = False,
    ) -> Iterator[StreamEvent]:
        prepared = self._prepare_messages(messages, enable_thinking=enable_thinking)
        parser = ThinkingStreamParser(enable_thinking=enable_thinking)

        for event in self._stream(prepared, config=config):
            if event.finish_reason is not None:
                for kind, chunk in parser.flush():
                    if kind == "reasoning":
                        yield StreamEvent(text="", reasoning_text=chunk, finish_reason=None)
                    else:
                        yield StreamEvent(text=chunk, reasoning_text=None, finish_reason=None)
                yield event
                return

            for kind, chunk in parser.feed(event.text):
                if kind == "reasoning":
                    yield StreamEvent(text="", reasoning_text=chunk, finish_reason=None)
                else:
                    yield StreamEvent(text=chunk, reasoning_text=None, finish_reason=None)

    def _prepare_messages(self, messages: list[object], *, enable_thinking: bool) -> PreparedInputs:
        try:
            return self.processor.prepare_messages(messages, enable_thinking=enable_thinking)
        except FileNotFoundError as exc:
            raise AnnaEngineError(str(exc), status_code=400, code="invalid_media_reference") from exc
        except ValueError as exc:
            raise AnnaEngineError(str(exc), status_code=400) from exc
        except RuntimeError as exc:
            raise AnnaEngineError(str(exc), status_code=500, error_type="server_error") from exc

    def _can_use_scheduler(self, prepared: PreparedInputs) -> bool:
        return (
            self.scheduler is not None
            and prepared.pixel_values is None
            and prepared.pixel_values_videos is None
        )

    def _validate_generation_request(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> tuple[list[int], int]:
        prompt_ids = prepared.input_ids[0].tolist()
        prompt_length = int(prepared.input_ids.shape[1])
        if prompt_length == 0:
            raise AnnaEngineError("Prompt produced zero tokens.")

        max_total = prompt_length + config.max_new_tokens
        if max_total > self.config.text_config.max_position_embeddings:
            raise AnnaEngineError(
                f"Requested sequence length {max_total} exceeds model context limit {self.config.text_config.max_position_embeddings}.",
                status_code=400,
                code="context_length_exceeded",
            )
        return prompt_ids, prompt_length

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

        kv_cache_bytes = (
            full_layers
            * 2
            * total_tokens
            * text_config.num_key_value_heads
            * text_config.head_dim
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

        if memory_info.free_bytes < policy.min_free_bytes:
            raise AnnaEngineError(
                f"Insufficient free XPU memory before generation: free={_format_bytes(memory_info.free_bytes)}, "
                f"required reserve={_format_bytes(policy.min_free_bytes)}. Reduce workload or restart the service.",
                status_code=503,
                error_type="server_error",
                code="insufficient_device_memory",
            )

        if estimated_bytes > available_budget or estimated_bytes > max_allowed:
            raise AnnaEngineError(
                f"Request rejected by memory guard: estimated={_format_bytes(estimated_bytes)}, "
                f"free={_format_bytes(memory_info.free_bytes)}, reserve={_format_bytes(policy.reserve_margin_bytes)}. "
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

    def _split_chat_output(self, raw_text: str, *, enable_thinking: bool) -> tuple[str | None, str]:
        normalized = raw_text
        if normalized.startswith("<think>"):
            normalized = normalized[len("<think>") :].lstrip("\r\n")
            enable_thinking = True

        closing_tag = "</think>"
        if enable_thinking:
            close_index = normalized.find(closing_tag)
            if close_index == -1:
                reasoning = normalized.strip()
                return (reasoning or None), ""
            reasoning = normalized[:close_index].strip()
            content = normalized[close_index + len(closing_tag) :].lstrip("\r\n")
            return (reasoning or None), content

        tagged_prefix = "<think>"
        if tagged_prefix in normalized and closing_tag in normalized:
            prefix_index = normalized.find(tagged_prefix)
            close_index = normalized.find(closing_tag, prefix_index)
            if prefix_index != -1 and close_index != -1:
                reasoning = normalized[prefix_index + len(tagged_prefix) : close_index].strip()
                content = normalized[close_index + len(closing_tag) :].lstrip("\r\n")
                return (reasoning or None), content
        return None, raw_text

    def _generate(self, prepared: PreparedInputs, *, config: GenerationConfig) -> TextGenerationResult:
        if self._can_use_scheduler(prepared):
            return self.scheduler.generate(prepared, config=config)
        return self._generate_direct(prepared, config=config)

    def _generate_direct(self, prepared: PreparedInputs, *, config: GenerationConfig) -> TextGenerationResult:
        if not config.stop_strings:
            return self._generate_without_streaming_overhead(prepared, config=config)

        text_parts: list[str] = []
        finish_reason = "length"
        prompt_tokens = 0
        completion_tokens = 0

        for delta, finished, reason, prompt_count, completion_count in self._iter_generation(prepared, config):
            if delta:
                text_parts.append(delta)
            prompt_tokens = prompt_count
            completion_tokens = completion_count
            if finished:
                finish_reason = reason or "stop"
                break

        return TextGenerationResult(
            text="".join(text_parts),
            reasoning_text=None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _generate_without_streaming_overhead(
        self,
        prepared: PreparedInputs,
        *,
        config: GenerationConfig,
    ) -> TextGenerationResult:
        completion_ids, finish_reason, prompt_tokens, completion_tokens = self._generate_token_ids(
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
        )

    def _stream(self, prepared: PreparedInputs, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        if self._can_use_scheduler(prepared):
            yield from self.scheduler.stream(prepared, config=config)
            return
        yield from self._stream_direct(prepared, config=config)

    def _stream_direct(self, prepared: PreparedInputs, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        for delta, finished, reason, _, _ in self._iter_generation(prepared, config):
            if delta:
                yield StreamEvent(text=delta, finish_reason=None)
            if finished:
                yield StreamEvent(text="", finish_reason=reason or "stop")
                return

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
        stable_text = _strip_unstable_replacement_suffix(current_text[:stable_length])
        if not stable_text.startswith(emitted_text):
            return "", emitted_text
        return stable_text[len(emitted_text) :], stable_text

    def _flush_decode_tail(self, *, current_text: str, emitted_text: str) -> tuple[str, str]:
        current_text = _strip_unstable_replacement_suffix(current_text)
        if not current_text.startswith(emitted_text):
            return "", emitted_text
        return current_text[len(emitted_text) :], current_text

    def _init_repetition_penalty_state(self, prompt_ids: list[int], penalty: float) -> tuple[torch.Tensor | None, set[int] | None]:
        if penalty == 1.0:
            return None, None
        unique_ids = list(dict.fromkeys(prompt_ids))
        if not unique_ids:
            return None, set()
        history_tensor = self.device_context.move_token_ids(torch.tensor(unique_ids, dtype=torch.long))
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

    @torch.inference_mode()
    def _generate_token_ids(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> tuple[list[int], str, int, int]:
        prompt_ids, prompt_length = self._validate_generation_request(prepared, config=config)
        prepared = self._move_prepared_for_generation(prepared, config=config)

        completion_ids: list[int] = []
        stop_token_ids = set(self.tokenizer.eos_token_ids)
        repetition_history, repetition_history_ids = self._init_repetition_penalty_state(
            prompt_ids,
            config.repetition_penalty,
        )

        input_ids = prepared.input_ids
        attention_mask = prepared.attention_mask
        mm_token_type_ids = prepared.mm_token_type_ids
        pixel_values = prepared.pixel_values
        image_grid_thw = prepared.image_grid_thw
        pixel_values_videos = prepared.pixel_values_videos
        video_grid_thw = prepared.video_grid_thw
        past_key_values = None

        try:
            for step_idx in range(config.max_new_tokens):
                try:
                    with self.execution_lock:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            pixel_values=pixel_values,
                            pixel_values_videos=pixel_values_videos,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            mm_token_type_ids=mm_token_type_ids,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                    logits = outputs.logits[0, -1]
                    next_token = sample_next_token(
                        logits,
                        generated_ids=repetition_history,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                    )
                    token_id = int(next_token.item())
                    past_key_values = outputs.past_key_values
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc

                if token_id in stop_token_ids:
                    return completion_ids, "stop", prompt_length, len(completion_ids)

                completion_ids.append(token_id)
                repetition_history, repetition_history_ids = self._append_repetition_penalty_token(
                    history_tensor=repetition_history,
                    history_ids=repetition_history_ids,
                    next_token=next_token,
                )

                input_ids = next_token.view(1, 1)
                attention_mask = None
                mm_token_type_ids = None
                pixel_values = None
                image_grid_thw = None
                pixel_values_videos = None
                video_grid_thw = None

                if step_idx + 1 >= config.max_new_tokens:
                    break

            return completion_ids, "length", prompt_length, len(completion_ids)
        finally:
            if past_key_values is not None:
                past_key_values.release()

    @torch.inference_mode()
    def _iter_generation(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> Iterator[tuple[str, bool, str | None, int, int]]:
        prompt_ids, prompt_length = self._validate_generation_request(prepared, config=config)
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

        input_ids = prepared.input_ids
        attention_mask = prepared.attention_mask
        mm_token_type_ids = prepared.mm_token_type_ids
        pixel_values = prepared.pixel_values
        image_grid_thw = prepared.image_grid_thw
        pixel_values_videos = prepared.pixel_values_videos
        video_grid_thw = prepared.video_grid_thw
        past_key_values = None

        try:
            for step_idx in range(config.max_new_tokens):
                try:
                    with self.execution_lock:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            pixel_values=pixel_values,
                            pixel_values_videos=pixel_values_videos,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            mm_token_type_ids=mm_token_type_ids,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                    logits = outputs.logits[0, -1]
                    next_token = sample_next_token(
                        logits,
                        generated_ids=repetition_history,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                    )
                    token_id = int(next_token.item())
                    past_key_values = outputs.past_key_values
                except RuntimeError as exc:
                    raise self._handle_runtime_failure(exc) from exc

                if token_id in stop_token_ids:
                    tail, _ = text_assembler.flush()
                    if tail:
                        yield tail, False, None, prompt_length, len(completion_ids)
                    yield "", True, "stop", prompt_length, len(completion_ids)
                    return

                completion_ids.append(token_id)
                repetition_history, repetition_history_ids = self._append_repetition_penalty_token(
                    history_tensor=repetition_history,
                    history_ids=repetition_history_ids,
                    next_token=next_token,
                )
                delta, hit_stop_string = text_assembler.feed_token(token_id)

                input_ids = next_token.view(1, 1)
                attention_mask = None
                mm_token_type_ids = None
                pixel_values = None
                image_grid_thw = None
                pixel_values_videos = None
                video_grid_thw = None

                if delta:
                    yield delta, False, None, prompt_length, len(completion_ids)

                if hit_stop_string:
                    yield "", True, "stop", prompt_length, len(completion_ids)
                    return

                if step_idx + 1 >= config.max_new_tokens:
                    break

            tail, _ = text_assembler.flush()
            if tail:
                yield tail, False, None, prompt_length, len(completion_ids)
            yield "", True, "length", prompt_length, len(completion_ids)
        finally:
            if past_key_values is not None:
                past_key_values.release()
