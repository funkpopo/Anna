from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import torch

from anna.mm.processor import PreparedInputs, Qwen3MultimodalProcessor
from anna.model.quantization import convert_module_linears_to_xpu_int4, estimate_module_xpu_int4_bytes
from anna.model.qwen import Qwen3ForConditionalGeneration
from anna.model.ops import Qwen3PageAllocator, Qwen3SparseMoeBlock
from anna.runtime.device import DeviceContext, RuntimeSafetyPolicy
from anna.runtime.streaming import IncrementalTextAssembler, strip_unstable_replacement_suffix
from anna.sampling.sampler import sample_next_token
from anna.weights.loader import build_model, estimate_model_weight_bytes, load_model_config, load_model_weights
from anna.weights.tokenizer import QwenTokenizer

logger = logging.getLogger(__name__)

ReasoningFormat = Literal["none", "deepseek"]
_REASONING_FORMAT_VALUES = frozenset({"none", "deepseek"})
_DEFAULT_REASONING_FORMAT: ReasoningFormat = "deepseek"


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


def normalize_reasoning_format(value: str | None) -> ReasoningFormat:
    if value is None:
        return _DEFAULT_REASONING_FORMAT
    normalized = value.strip().lower()
    if normalized not in _REASONING_FORMAT_VALUES:
        allowed = ", ".join(sorted(_REASONING_FORMAT_VALUES))
        raise ValueError(f"Unsupported reasoning format: {value}. Expected one of: {allowed}.")
    return cast(ReasoningFormat, normalized)


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
        hold_back = 0
        max_suffix = min(len(self.buffer), len(self.CLOSE_TAG) - 1)
        for suffix_length in range(max_suffix, 0, -1):
            if self.buffer.endswith(self.CLOSE_TAG[:suffix_length]):
                hold_back = suffix_length
                break
        safe_length = max(0, len(self.buffer) - hold_back)
        if safe_length <= 0:
            return []
        reasoning = self.buffer[:safe_length]
        self.buffer = self.buffer[safe_length:]
        return [("reasoning", reasoning)]

    def feed(self, text: str) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        self.buffer += text

        while True:
            if self.state == "content":
                stripped = self.buffer.lstrip()
                if stripped.startswith("<think>"):
                    self.buffer = stripped
                    self.state = "reasoning"
                    continue

            if self.state == "reasoning":
                self.buffer = _strip_think_open_tag(self.buffer)
                close_index = self.buffer.find(self.CLOSE_TAG)
                if close_index == -1:
                    outputs.extend(self._emit_reasoning_prefix())
                    break
                reasoning = self.buffer[:close_index].rstrip("\r\n")
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
            reasoning = self.buffer.rstrip("\r\n")
            if reasoning:
                outputs.append(("reasoning", reasoning))
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
    max_new_tokens: int | None = None
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
        default_max_completion_tokens: int | None = None,
        default_enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str = _DEFAULT_REASONING_FORMAT,
        offload_mode: str = "none",
        offload_vision: bool = False,
        expert_quant: str = "none",
        weight_quant: str = "none",
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
        resolved_offload_vision = cls._resolve_offload_vision(
            requested_offload_vision=offload_vision,
            resolved_offload_mode=resolved_offload_mode,
            config=config,
            device_context=device_context,
        )
        resolved_expert_quant = cls._resolve_expert_quant(
            requested_quant=expert_quant,
            resolved_offload_mode=resolved_offload_mode,
            device_context=device_context,
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
        try:
            model, model_quantized_replacements = build_model(
                config,
                device=model_device,
                dtype=device_context.dtype,
            )
            report = load_model_weights(model, model_path)
            runtime_weight_quantized_replacements = 0
            if resolved_weight_quant == "int4":
                runtime_weight_quantized_replacements = cls._apply_runtime_weight_quantization(
                    model=model,
                    device=device_context.device,
                    compute_dtype=device_context.dtype,
                )
            total_quantized_replacements = model_quantized_replacements + runtime_weight_quantized_replacements
            report.quantized_replacements = total_quantized_replacements
            auto_resident_indices = resolved_resident_expert_layer_indices is None
            auto_cached_experts_per_layer = resolved_cached_experts_per_layer is None
            initial_resident_expert_layer_indices = () if auto_resident_indices else resolved_resident_expert_layer_indices
            initial_cached_experts_per_layer = 0 if auto_cached_experts_per_layer else resolved_cached_experts_per_layer

            model.configure_runtime(
                device_context.device,
                offload_experts=resolved_offload_mode == "experts",
                offload_vision=resolved_offload_vision,
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
                    offload_vision=resolved_offload_vision,
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
                    offload_vision=resolved_offload_vision,
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
        resolved_default_max_completion_tokens = (
            config.default_max_completion_tokens
            if default_max_completion_tokens is None
            else default_max_completion_tokens
        )
        if resolved_default_max_completion_tokens is not None:
            resolved_default_max_completion_tokens = max(1, int(resolved_default_max_completion_tokens))

        logger.info(
            "Loaded model %s on %s (compute=%s, requested=%s, default_max_completion_tokens=%s, default_enable_thinking=%s, reasoning_format=%s, offload=%s, offload_vision=%s, expert_quant=%s, weight_quant=%s, resident_expert_layers=%s, resident_expert_layer_indices=%s, cached_experts_per_layer=%s, weights=%s); tensors loaded=%s skipped=%s quantized=%s",
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
    def _resolve_offload_vision(
        *,
        requested_offload_vision: bool,
        resolved_offload_mode: str,
        config: object,
        device_context: DeviceContext,
    ) -> bool:
        if device_context.device.type != "xpu":
            return False
        if getattr(config, "vision_config", None) is None:
            return False
        return bool(requested_offload_vision or resolved_offload_mode == "experts")

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
        if device_context.device.type != "xpu":
            return "none"
        if resolved_offload_mode == "experts" or config.text_config.is_moe_model:
            return "none"
        if normalized != "auto":
            return normalized
        if getattr(config, "quantization_config", None) is not None and config.quantization_config.is_enabled:
            return "none"

        memory_info = device_context.get_memory_info()
        if memory_info is None:
            return "none"

        weight_bytes = estimate_model_weight_bytes(model_path)
        if weight_bytes > int(memory_info.total_bytes * 0.85):
            return "int4"
        return "none"

    @classmethod
    def _apply_runtime_weight_quantization(
        cls,
        *,
        model: Qwen3ForConditionalGeneration,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> int:
        text_model = cls._text_model(model)
        if text_model is None:
            return 0
        replacements = convert_module_linears_to_xpu_int4(
            text_model,
            compute_dtype=compute_dtype,
            device=device,
        )
        logger.info(
            "Runtime dense text int4 quantization: replacements=%s device=%s compute_dtype=%s",
            replacements,
            device,
            compute_dtype,
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
            "default_max_completion_tokens": self.default_max_completion_tokens,
            "default_enable_thinking": self.default_enable_thinking,
            "reasoning_format": self.reasoning_format,
            "float8_available": self.device_context.float8_available,
            "quantization": quant_method,
            "weight_quant": self.weight_quant,
            "quantized_replacements": self.quantized_replacements,
            "offload_mode": self.offload_mode,
            "offload_vision": self.offload_vision,
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
        prepared = self.processor.encode_text(
            prompt,
            tensor_device=self.device_context.device,
        )
        return self._generate(prepared, config=config)

    def stream_text(self, prompt: str, *, config: GenerationConfig) -> Iterator[StreamEvent]:
        prepared = self.processor.encode_text(
            prompt,
            tensor_device=self.device_context.device,
        )
        yield from self._stream(prepared, config=config)

    def generate_chat(
        self,
        messages: list[object],
        *,
        config: GenerationConfig,
        enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str | None = None,
    ) -> TextGenerationResult:
        prepared = self._prepare_messages(messages, enable_thinking=enable_thinking)
        raw = self._generate(prepared, config=config)
        text, reasoning_text = self._project_chat_output(
            raw_text=raw.text,
            raw_reasoning_text=raw.reasoning_text,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
        )
        return TextGenerationResult(
            text=text,
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
        enable_thinking: bool = True,
        reasoning_format: ReasoningFormat | str | None = None,
    ) -> Iterator[StreamEvent]:
        prepared = self._prepare_messages(messages, enable_thinking=enable_thinking)
        resolved_reasoning_format = self._resolve_reasoning_format(reasoning_format)
        if resolved_reasoning_format == "none":
            yield from self._stream(prepared, config=config)
            return

        parser = ThinkingStreamParser(enable_thinking=enable_thinking)
        for event in self._stream(prepared, config=config):
            parsed_chunks = self._parse_chat_stream_event(parser, event)
            if not parsed_chunks:
                yield StreamEvent(
                    text="",
                    reasoning_text=event.reasoning_text,
                    finish_reason=event.finish_reason,
                )
                continue

            last_index = len(parsed_chunks) - 1
            for index, (kind, chunk) in enumerate(parsed_chunks):
                yield StreamEvent(
                    text=chunk if kind == "content" else "",
                    reasoning_text=chunk if kind == "reasoning" else None,
                    finish_reason=event.finish_reason if index == last_index else None,
                )

    def _prepare_messages(self, messages: list[object], *, enable_thinking: bool = True) -> PreparedInputs:
        try:
            return self.processor.prepare_messages(
                messages,
                enable_thinking=enable_thinking,
                tensor_device=self.device_context.device,
                tensor_dtype=self.device_context.dtype,
            )
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

    @staticmethod
    def _has_multimodal_inputs(prepared: PreparedInputs) -> bool:
        return prepared.pixel_values is not None or prepared.pixel_values_videos is not None

    def _forward_generation_model(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ):
        if pixel_values is None and pixel_values_videos is None and hasattr(self.model, "forward_text_only"):
            return self.model.forward_text_only(
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
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
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
        if memory_info.free_bytes < policy.min_free_bytes:
            raise AnnaEngineError(
                f"Insufficient free XPU memory before generation: free={_format_bytes(memory_info.free_bytes)}, "
                f"required reserve={_format_bytes(policy.min_free_bytes)}. Reduce workload or restart the service.",
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

        probe = GenerationConfig(max_new_tokens=1)
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

        one_token_config = GenerationConfig(max_new_tokens=1)
        estimated_bytes = self._estimate_generation_memory_bytes(prepared, config=one_token_config)
        raise AnnaEngineError(
            f"Request rejected by memory guard: estimated={_format_bytes(estimated_bytes)}, "
            f"free={_format_bytes(memory_info.free_bytes)}, reserve={_format_bytes(policy.reserve_margin_bytes)}. "
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
    ) -> tuple[str, str | None]:
        resolved_reasoning_format = self._resolve_reasoning_format(reasoning_format)
        if resolved_reasoning_format == "none":
            return raw_text, None

        parsed_reasoning, parsed_content = self._split_chat_output(
            raw_text,
            enable_thinking=enable_thinking,
        )
        reasoning_text = raw_reasoning_text if raw_reasoning_text is not None else parsed_reasoning
        if reasoning_text is not None:
            return parsed_content, reasoning_text
        return raw_text, None

    def _parse_chat_stream_event(
        self,
        parser: ThinkingStreamParser,
        event: StreamEvent,
    ) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if event.text:
            outputs.extend(parser.feed(event.text))
        if event.finish_reason is not None:
            outputs.extend(parser.flush())
        return outputs

    def _split_chat_output(self, raw_text: str, *, enable_thinking: bool) -> tuple[str | None, str]:
        normalized = raw_text
        explicit_open = normalized.startswith("<think>")
        if explicit_open:
            normalized = normalized[len("<think>") :].lstrip("\r\n")
            enable_thinking = True

        closing_tag = "</think>"
        close_index = normalized.find(closing_tag)
        if close_index != -1:
            reasoning = normalized[:close_index].strip()
            content = normalized[close_index + len(closing_tag) :].lstrip("\r\n")
            return (reasoning or None), content
        if explicit_open or enable_thinking:
            reasoning = normalized.strip()
            return (reasoning or None), ""
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

    @torch.inference_mode()
    def _generate_token_ids(
        self,
        prepared: PreparedInputs,
        config: GenerationConfig,
    ) -> tuple[list[int], str, int, int]:
        prompt_ids, prompt_length, config = self._validate_generation_request(prepared, config=config)
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
                        outputs = self._forward_generation_model(
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
                        outputs = self._forward_generation_model(
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
