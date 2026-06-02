from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anna.core.hotpath_events import record_attention_fallback
from anna.model.fused_ops import fused_op_health_report
from anna.model import ops as model_ops
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, GenerationConfig, TextGenerationResult
from anna.runtime.slot_model_runner import (
    SlotDecodeModelInputs,
    SlotModelRunner,
    SlotModelRunnerConfig,
    resolve_slot_model_runner_config,
)
from anna.sampling.capabilities import sampler_capability_report
from anna.sampling.params import SamplingBatchParams, SamplingBatchParamsCache
from anna.vllm_compat.outputs import RequestOutput, request_output_from_result
from anna.vllm_compat.sampling import SamplingParams, sampling_params_to_generation_config


@dataclass(frozen=True, slots=True)
class AnnaXPUAttentionBackend:
    name: str
    paged_decode: bool
    prefill: str
    fallback: str | None = None
    prefill_xpu_kernel_ready: bool = False
    prefill_xpu_kernel_reason: str = "custom_prefill_attention_kernel_not_implemented"
    prefill_fallback_reason: str = "anna_vllm_xpu_torch_sdpa_prefill_attention"


@dataclass(frozen=True, slots=True)
class AnnaXPUKVCacheConfig:
    block_size: int
    max_slots: int
    total_blocks: int
    max_blocks_per_seq: int
    quantization: str = "none"
    quant_bits: int = 4


@dataclass(frozen=True, slots=True)
class AnnaXPUPlatformCapabilities:
    device_type: str = "xpu"
    device_name: str = "Intel Arc / Xe-HPG"
    supported_dtypes: tuple[str, ...] = ("torch.bfloat16", "torch.float16", "torch.float32")
    attention_backend: AnnaXPUAttentionBackend = field(
        default_factory=lambda: AnnaXPUAttentionBackend(
            name="anna.paged_gqa",
            paged_decode=False,
            prefill="torch_scaled_dot_product_attention",
            fallback="missing_fused_ops",
        )
    )
    fused_ops: Mapping[str, bool] = field(default_factory=dict)
    kv_cache: AnnaXPUKVCacheConfig | None = None


@dataclass(frozen=True, slots=True)
class AnnaVLLMPluginSpec:
    """vLLM plugin discovery metadata without importing vLLM."""

    name: str = "anna_xpu"
    platform_entry_point_group: str = "vllm.platform_plugins"
    platform_entry_point: str = "anna_xpu = anna_vllm_xpu:register_platform"
    platform_class: str | None = None
    worker_class: str | None = None
    attention_backend_registry_class: str = "anna_vllm_xpu.adapter.AnnaXPUAttentionBackendRegistry"
    runtime_adapter_class: str = "anna_vllm_xpu.adapter.AnnaVLLMXPURuntimeAdapter"
    kv_cache_connector_class: str = "anna_vllm_xpu.adapter.AnnaXPUKVCacheConnector"
    integrated_vllm_worker: bool = False


def build_vllm_plugin_spec() -> AnnaVLLMPluginSpec:
    return AnnaVLLMPluginSpec()


def register_platform() -> str | None:
    """vLLM ``vllm.platform_plugins`` entry point.

    The current package exposes the discovery hook but intentionally returns
    ``None`` until an integrated vLLM Platform/Worker pair is implemented.
    Returning ``None`` keeps vLLM discovery harmless in environments where Anna
    is installed next to vLLM before the full worker plugin is enabled.
    """

    return build_vllm_plugin_spec().platform_class


class AnnaXPUAttentionBackendRegistry:
    """No-dependency attention backend boundary for a future vLLM plugin."""

    name = "anna.paged_gqa"
    paged_decode_entrypoint = "anna.model.ops.paged_kv_slot_batch_decode_attention"
    prefill_entrypoint = "torch.nn.functional.scaled_dot_product_attention"

    def __init__(
        self,
        capabilities: AnnaXPUPlatformCapabilities | None = None,
        *,
        allow_cpu_tensors_for_tests: bool = False,
    ) -> None:
        self.capabilities = capabilities
        self.allow_cpu_tensors_for_tests = bool(allow_cpu_tensors_for_tests)

    @property
    def paged_decode_supported(self) -> bool:
        if self.capabilities is None:
            return True
        return bool(self.capabilities.attention_backend.paged_decode)

    @property
    def prefill_backend(self) -> str:
        if self.capabilities is None:
            return "torch_scaled_dot_product_attention"
        return self.capabilities.attention_backend.prefill

    @property
    def prefill_xpu_kernel_ready(self) -> bool:
        if self.capabilities is None:
            return False
        return bool(self.capabilities.attention_backend.prefill_xpu_kernel_ready)

    @property
    def prefill_xpu_kernel_reason(self) -> str:
        if self.prefill_xpu_kernel_ready:
            return "ready"
        if self.capabilities is None:
            return "custom_prefill_attention_kernel_not_implemented"
        return self.capabilities.attention_backend.prefill_xpu_kernel_reason

    @property
    def prefill_fallback_reason(self) -> str | None:
        if self.prefill_xpu_kernel_ready:
            return None
        if self.capabilities is None:
            return "anna_vllm_xpu_torch_sdpa_prefill_attention"
        return self.capabilities.attention_backend.prefill_fallback_reason

    def health(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paged_decode_entrypoint": self.paged_decode_entrypoint,
            "paged_decode_supported": self.paged_decode_supported,
            "prefill_entrypoint": self.prefill_entrypoint,
            "prefill_backend": self.prefill_backend,
            "prefill_xpu_kernel_ready": self.prefill_xpu_kernel_ready,
            "prefill_xpu_kernel_reason": self.prefill_xpu_kernel_reason,
            "prefill_records_attention_fallback": not self.prefill_xpu_kernel_ready,
            "prefill_fallback_reason": self.prefill_fallback_reason,
        }

    def paged_decode(
        self,
        *,
        query_states: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling: float,
        slot_ids: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.paged_decode_supported:
            fallback = None if self.capabilities is None else self.capabilities.attention_backend.fallback
            detail = f" ({fallback})" if fallback else ""
            raise RuntimeError(
                "Anna XPU paged decode attention backend is not available; "
                f"build/load the native paged_gqa_decode_fused op before using this vLLM path{detail}."
            )
        self._require_xpu_tensors(
            query_states=query_states,
            key_pages=key_pages,
            value_pages=value_pages,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )
        optional_tensors = {}
        if slot_ids is not None:
            optional_tensors["slot_ids"] = slot_ids
        if gate is not None:
            optional_tensors["gate"] = gate
        self._require_xpu_tensors(**optional_tensors)
        return model_ops.paged_kv_slot_batch_decode_attention(
            query_states,
            key_pages,
            value_pages,
            block_tables,
            seq_lens,
            scaling=scaling,
            slot_ids=slot_ids,
            gate=gate,
        )

    def prefill(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        self._require_xpu_tensors(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
        if self.prefill_backend != "anna.prefill_attention":
            record_attention_fallback("anna_vllm_xpu_torch_sdpa_prefill_attention")
        enable_gqa = query_states.shape[1] != key_states.shape[1]
        try:
            return F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=float(dropout_p),
                is_causal=bool(causal),
                enable_gqa=enable_gqa,
            )
        except TypeError:
            if enable_gqa:
                groups = query_states.shape[1] // key_states.shape[1]
                key_states = model_ops.repeat_kv(key_states, groups)
                value_states = model_ops.repeat_kv(value_states, groups)
            return F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=float(dropout_p),
                is_causal=bool(causal),
            )

    def _require_xpu_tensors(self, **named_tensors: torch.Tensor) -> None:
        if self.allow_cpu_tensors_for_tests:
            return
        for name, tensor in named_tensors.items():
            if tensor.device.type != "xpu":
                raise RuntimeError(
                    "Anna XPU attention backend expects XPU tensors; "
                    f"{name} is on {tensor.device}. Refusing to run this vLLM backend path on CPU."
                )


class AnnaXPUKVCacheConnector:
    """No-dependency KV metadata bridge for vLLM-style block tables.

    vLLM owns its KV allocator in a real plugin. This connector mirrors the
    external slot/block-table tensors into Anna's metadata tensors and returns
    the same ``SlotDecodeModelInputs`` shape that Anna's slot runner emits.
    """

    def __init__(self, runner: SlotModelRunner) -> None:
        self.runner = runner
        self._sampling_batch_params_cache = SamplingBatchParamsCache(max_entries=64)

    @property
    def device(self) -> torch.device:
        return self.runner.config.device

    def import_decode_batch(
        self,
        *,
        request_ids: Sequence[str] | None = None,
        input_ids: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
        slot_ids: torch.Tensor | Sequence[int],
        block_tables: torch.Tensor | Sequence[Sequence[int]],
        seq_lens: torch.Tensor | Sequence[int],
        epochs: torch.Tensor | Sequence[int] | None = None,
        positions: torch.Tensor | Sequence[int] | None = None,
        sampling_params: Sequence[Any | None] | None = None,
        physical_block_tables: bool = True,
    ) -> SlotDecodeModelInputs:
        slot_ids_tensor = torch.as_tensor(slot_ids, dtype=torch.long, device=self.device).reshape(-1)
        batch_size = int(slot_ids_tensor.numel())
        if batch_size <= 0:
            raise ValueError("external KV decode batch must contain at least one slot.")
        if batch_size > self.runner.config.max_batch_size:
            raise ValueError(
                f"external KV decode batch size {batch_size} exceeds "
                f"runner max_batch_size={self.runner.config.max_batch_size}."
            )

        input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if input_ids_tensor.ndim == 1:
            if input_ids_tensor.numel() != batch_size:
                raise ValueError("1D input_ids must contain one token id per slot.")
            input_ids_tensor = input_ids_tensor.view(batch_size, 1)
        if input_ids_tensor.ndim != 2 or input_ids_tensor.shape[0] != batch_size:
            raise ValueError("input_ids must be [batch] or [batch, tokens] with one row per slot.")
        if input_ids_tensor.shape[1] <= 0:
            raise ValueError("input_ids must include at least one token per slot.")
        if self.device.type == "cpu" and bool(torch.any(input_ids_tensor < 0)):
            raise ValueError("input_ids must be non-negative.")

        seq_lens_tensor = torch.as_tensor(seq_lens, dtype=torch.long, device=self.device).reshape(-1)
        if seq_lens_tensor.numel() != batch_size:
            raise ValueError("seq_lens must contain one value per slot.")
        if positions is None:
            positions_tensor = seq_lens_tensor.clone()
        else:
            positions_tensor = torch.as_tensor(positions, dtype=torch.long, device=self.device).reshape(-1)
            if positions_tensor.numel() != batch_size:
                raise ValueError("positions must contain one value per slot.")
            if self.device.type == "cpu" and bool(torch.any(positions_tensor < 0)):
                raise ValueError("positions must be non-negative.")
        if epochs is None:
            epochs_tensor = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        else:
            epochs_tensor = torch.as_tensor(epochs, dtype=torch.long, device=self.device).reshape(-1)
            if epochs_tensor.numel() != batch_size:
                raise ValueError("epochs must contain one value per slot.")
            if self.device.type == "cpu" and bool(torch.any(epochs_tensor < 0)):
                raise ValueError("epochs must be non-negative.")

        block_tables_tensor = torch.as_tensor(block_tables, dtype=torch.int32, device=self.device)
        if block_tables_tensor.ndim != 2 or block_tables_tensor.shape[0] != batch_size:
            raise ValueError("block_tables must be [batch, max_blocks] with one row per slot.")

        if request_ids is None:
            resolved_request_ids = tuple(f"external-row-{row_idx}" for row_idx in range(batch_size))
        elif isinstance(request_ids, str):
            resolved_request_ids = (request_ids,)
        else:
            resolved_request_ids = tuple(str(request_id) for request_id in request_ids)
        if len(resolved_request_ids) != batch_size:
            raise ValueError("request_ids must contain one value per slot.")
        if any(not request_id for request_id in resolved_request_ids):
            raise ValueError("request_ids must be non-empty.")
        if len(set(resolved_request_ids)) != len(resolved_request_ids):
            raise ValueError("request_ids must be unique within an external decode batch.")

        resolved_sampling_params = (
            tuple(sampling_params)
            if sampling_params is not None
            else tuple(None for _ in range(batch_size))
        )
        if len(resolved_sampling_params) != batch_size:
            raise ValueError("sampling_params must contain one value per slot.")

        self.runner.kv_manager.mirror_external_slot_tensors(
            slot_ids=slot_ids_tensor,
            block_tables=block_tables_tensor,
            seq_lens=seq_lens_tensor,
            epochs=epochs_tensor,
            append_tokens=int(input_ids_tensor.shape[1]),
        )
        return SlotDecodeModelInputs(
            request_ids=resolved_request_ids,
            input_ids=input_ids_tensor,
            slot_ids=slot_ids_tensor.to(dtype=torch.int32),
            epochs=epochs_tensor,
            positions=positions_tensor,
            positions_are_global=False,
            seq_lens=seq_lens_tensor,
            seq_lens_are_global=False,
            block_tables=block_tables_tensor,
            block_tables_are_global=False,
            physical_block_tables=bool(physical_block_tables),
            sampling_params=resolved_sampling_params,
            sampling_batch_params=self._sampling_batch_params_for_external_batch(resolved_sampling_params),
        )

    def _sampling_batch_params_for_external_batch(
        self,
        sampling_params: Sequence[Any | None],
    ) -> SamplingBatchParams:
        return self._sampling_batch_params_cache.get(tuple(sampling_params), device=self.device)

    def health(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "mirrors_external_metadata": True,
            "owns_physical_kv_pages": False,
            "physical_block_tables_default": True,
            "block_table_ownership": "external_physical_or_compatible",
            "sampling_batch_params_cache": self._sampling_batch_params_cache.stats(),
        }


def _dtype_name(dtype: torch.dtype | str) -> str:
    return str(dtype)


def _attention_backend_from_health(health_report: Mapping[str, object]) -> AnnaXPUAttentionBackend:
    available = health_report.get("available", {})
    if not isinstance(available, Mapping):
        available = {}
    paged_decode = bool(available.get("paged_gqa_decode_fused"))
    prefill = "torch_scaled_dot_product_attention"
    fallback = None if paged_decode else "missing_paged_gqa_decode_fused"
    return AnnaXPUAttentionBackend(
        name="anna.paged_gqa",
        paged_decode=paged_decode,
        prefill=prefill,
        fallback=fallback,
        prefill_xpu_kernel_ready=False,
        prefill_xpu_kernel_reason="custom_prefill_attention_kernel_not_implemented",
        prefill_fallback_reason="anna_vllm_xpu_torch_sdpa_prefill_attention",
    )


def _get_field(obj: object, name: str, default: object | None = None) -> object | None:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_request_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _coerce_sampling_params(value: object) -> tuple[object | None, ...] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def extract_execute_model_request_ids(execute_model_request: object) -> tuple[str, ...]:
    """Extract request ids from vLLM-like execute_model inputs.

    The real vLLM classes are intentionally not imported here. This adapter
    accepts the stable shapes we need for an integration boundary: either an
    explicit ``request_ids`` field or a ``seq_group_metadata_list`` whose items
    expose ``request_id``.
    """

    direct_ids = _coerce_request_ids(_get_field(execute_model_request, "request_ids"))
    if direct_ids:
        return direct_ids

    seq_groups = _get_field(execute_model_request, "seq_group_metadata_list")
    if seq_groups is None:
        seq_groups = _get_field(execute_model_request, "seq_groups")
    if seq_groups is None or isinstance(seq_groups, str) or not isinstance(seq_groups, Sequence):
        return ()

    request_ids: list[str] = []
    for seq_group in seq_groups:
        request_id = _get_field(seq_group, "request_id")
        if request_id is None:
            continue
        request_ids.append(str(request_id))
    return tuple(request_ids)


def extract_execute_model_sampling_params(execute_model_request: object) -> tuple[object | None, ...] | None:
    """Extract per-row sampling params from vLLM-like execute_model inputs."""

    direct = _coerce_sampling_params(
        extract_execute_model_field(execute_model_request, "sampling_params", "sampling_params_list")
    )
    if direct is not None:
        return direct

    seq_groups = _get_field(execute_model_request, "seq_group_metadata_list")
    if seq_groups is None:
        seq_groups = _get_field(execute_model_request, "seq_groups")
    if seq_groups is None or isinstance(seq_groups, str) or not isinstance(seq_groups, Sequence):
        return None

    params: list[object | None] = []
    found_any = False
    for seq_group in seq_groups:
        value = _get_field(seq_group, "sampling_params")
        if value is not None:
            found_any = True
        params.append(value)
    return tuple(params) if found_any else None


def extract_execute_model_field(execute_model_request: object, *names: str) -> object | None:
    """Return the first present field from a vLLM-like execute_model input."""

    for name in names:
        value = _get_field(execute_model_request, name)
        if value is not None:
            return value
    return None


def build_platform_capabilities(
    *,
    device_name: str = "Intel Arc / Xe-HPG",
    dtype: torch.dtype | str = torch.bfloat16,
    kv_cache: AnnaXPUKVCacheConfig | None = None,
    health_report: Mapping[str, object] | None = None,
) -> AnnaXPUPlatformCapabilities:
    report = fused_op_health_report() if health_report is None else health_report
    available = report.get("available", {})
    fused_ops = dict(available) if isinstance(available, Mapping) else {}
    supported = tuple(
        dict.fromkeys(
            (
                _dtype_name(dtype),
                "torch.bfloat16",
                "torch.float16",
                "torch.float32",
            )
        )
    )
    return AnnaXPUPlatformCapabilities(
        device_name=device_name,
        supported_dtypes=supported,
        attention_backend=_attention_backend_from_health(report),
        fused_ops=fused_ops,
        kv_cache=kv_cache,
    )


class AnnaVLLMXPURuntimeAdapter:
    """Level 3 vLLM runtime adapter boundary for Anna's XPU backend.

    This class is intentionally a narrow bridge: it exposes the model runner,
    KV-cache, sampling, and output-conversion contracts that a real vLLM worker
    plugin will call, without importing vLLM as a hard dependency.
    """

    def __init__(
        self,
        *,
        engine: AnnaQwen3_5TextEngine | None = None,
        model: str | Path | None = None,
        model_id: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        if engine is None:
            if model is None:
                raise ValueError("Either engine or model must be provided.")
            engine = AnnaQwen3_5TextEngine.from_model_dir(model, model_id=model_id, **engine_kwargs)
        self.engine = engine
        self._kv_cache_connector: AnnaXPUKVCacheConnector | None = None
        self._kv_cache_connector_runner: SlotModelRunner | None = None

    @property
    def model_id(self) -> str:
        return str(getattr(self.engine, "default_model_id", "anna"))

    @property
    def slot_model_runner(self) -> SlotModelRunner | None:
        return getattr(self.engine, "slot_model_runner", None)

    def platform_capabilities(self) -> AnnaXPUPlatformCapabilities:
        kv_cache = self.kv_cache_config()
        memory_info = self.engine.device_context.get_memory_info()
        device_name = "Intel Arc / Xe-HPG"
        if memory_info is not None and getattr(memory_info, "device_name", None):
            device_name = str(memory_info.device_name)
        return build_platform_capabilities(
            device_name=device_name,
            dtype=self.engine.device_context.dtype,
            kv_cache=kv_cache,
        )

    def kv_cache_config(self) -> AnnaXPUKVCacheConfig | None:
        runner = self.slot_model_runner
        if runner is None:
            return None
        config: SlotModelRunnerConfig = runner.config
        return AnnaXPUKVCacheConfig(
            block_size=config.block_size,
            max_slots=config.max_slots,
            total_blocks=config.total_blocks,
            max_blocks_per_seq=config.max_blocks_per_seq,
            quantization=self.engine.optimization_config.kv_cache_quantization,
            quant_bits=self.engine.optimization_config.kv_cache_quant_bits,
        )

    def kv_cache_connector(self) -> AnnaXPUKVCacheConnector:
        runner = self.slot_model_runner
        if runner is None:
            raise RuntimeError("Anna slot model runner is not enabled; KV cache connector cannot be created.")
        if self._kv_cache_connector is None or self._kv_cache_connector_runner is not runner:
            self._kv_cache_connector = AnnaXPUKVCacheConnector(runner)
            self._kv_cache_connector_runner = runner
        return self._kv_cache_connector

    def attention_backend_registry(self) -> AnnaXPUAttentionBackendRegistry:
        return AnnaXPUAttentionBackendRegistry(self.platform_capabilities())

    def import_external_kv_decode_batch(
        self,
        *,
        request_ids: Sequence[str] | None = None,
        input_ids: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
        slot_ids: torch.Tensor | Sequence[int],
        block_tables: torch.Tensor | Sequence[Sequence[int]],
        seq_lens: torch.Tensor | Sequence[int],
        epochs: torch.Tensor | Sequence[int] | None = None,
        positions: torch.Tensor | Sequence[int] | None = None,
        sampling_params: Sequence[Any | None] | None = None,
        physical_block_tables: bool = True,
    ) -> SlotDecodeModelInputs:
        return self.kv_cache_connector().import_decode_batch(
            request_ids=request_ids,
            input_ids=input_ids,
            slot_ids=slot_ids,
            block_tables=block_tables,
            seq_lens=seq_lens,
            epochs=epochs,
            positions=positions,
            sampling_params=sampling_params,
            physical_block_tables=physical_block_tables,
        )

    def build_model_runner_inputs(
        self,
        *,
        request_ids: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> object:
        runner = self.slot_model_runner
        if runner is None:
            raise RuntimeError("Anna slot model runner is not enabled; vLLM runtime batches cannot be built.")
        return runner.build_decode_inputs(request_ids=request_ids, limit=limit)

    def build_model_runner_inputs_from_execute_model(self, execute_model_request: object) -> object:
        request_ids = extract_execute_model_request_ids(execute_model_request)
        if not request_ids:
            raise ValueError("vLLM execute_model request did not expose any request ids.")
        if any(not request_id for request_id in request_ids):
            raise ValueError("vLLM execute_model request ids must be non-empty.")
        return self.build_model_runner_inputs(request_ids=request_ids)

    def import_external_kv_decode_batch_from_execute_model(
        self,
        execute_model_request: object,
        *,
        sampling_params: Sequence[Any | None] | None = None,
        physical_block_tables: bool = True,
    ) -> SlotDecodeModelInputs:
        input_ids = extract_execute_model_field(execute_model_request, "input_ids", "token_ids")
        slot_ids = extract_execute_model_field(execute_model_request, "slot_ids", "slots")
        block_tables = extract_execute_model_field(execute_model_request, "block_tables", "block_table")
        seq_lens = extract_execute_model_field(execute_model_request, "seq_lens", "sequence_lengths")
        epochs = extract_execute_model_field(execute_model_request, "epochs", "slot_epochs")
        positions = extract_execute_model_field(execute_model_request, "positions")
        missing = [
            name
            for name, value in (
                ("input_ids", input_ids),
                ("slot_ids", slot_ids),
                ("block_tables", block_tables),
                ("seq_lens", seq_lens),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "vLLM execute_model request is missing external KV decode fields: "
                + ", ".join(missing)
            )
        if sampling_params is None:
            sampling_params = extract_execute_model_sampling_params(execute_model_request)
        request_ids = extract_execute_model_request_ids(execute_model_request)
        return self.import_external_kv_decode_batch(
            request_ids=request_ids or None,
            input_ids=input_ids,  # type: ignore[arg-type]
            slot_ids=slot_ids,  # type: ignore[arg-type]
            block_tables=block_tables,  # type: ignore[arg-type]
            seq_lens=seq_lens,  # type: ignore[arg-type]
            epochs=epochs,  # type: ignore[arg-type]
            positions=positions,  # type: ignore[arg-type]
            sampling_params=sampling_params,
            physical_block_tables=physical_block_tables,
        )

    @staticmethod
    def sampling_params_to_generation_config(params: SamplingParams | None) -> GenerationConfig:
        return sampling_params_to_generation_config(params)

    def generate_one(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str = "anna-vllm-xpu",
    ) -> RequestOutput:
        config = self.sampling_params_to_generation_config(sampling_params)
        result = self.engine.generate_text(prompt, config=config)
        return self.request_output_from_result(prompt, result, request_id=request_id)

    @staticmethod
    def request_output_from_result(prompt: str, result: TextGenerationResult, *, request_id: str) -> RequestOutput:
        return request_output_from_result(prompt, result, request_id=request_id)

    def health(self) -> dict[str, object]:
        capabilities = self.platform_capabilities()
        plugin_spec = build_vllm_plugin_spec()
        return {
            "runtime_adapter": "anna_vllm_xpu",
            "level": 3,
            "integrated_vllm_worker": False,
            "platform_plugin_entry_point": plugin_spec.platform_entry_point,
            "execute_model_batch_adapter": True,
            "kv_cache_connector": self.slot_model_runner is not None,
            "kv_cache_connector_health": (
                None
                if self._kv_cache_connector is None
                else self._kv_cache_connector.health()
            ),
            "attention_backend_registry": True,
            "attention_backend_boundary": self.attention_backend_registry().health(),
            "sampler": sampler_capability_report(),
            "model": self.model_id,
            "platform": capabilities,
            "slot_model_runner_enabled": self.slot_model_runner is not None,
        }
