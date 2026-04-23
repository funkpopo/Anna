from __future__ import annotations

import copy
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from anna.model.prefix_block_cache import PrefixBlockPool
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.model.turboquant import TurboQuantKVRow
from anna.model.fused_ops import (
    run_causal_conv1d_decode_fused,
    run_causal_conv1d_prefill_fused,
    run_gated_delta_decode_fused,
    run_gated_delta_prefill_fused,
    run_gqa_decode_fused,
    run_paged_gqa_decode_fused,
    run_moe_dispatch_fused,
    run_moe_grouped_int4_mlp_fused,
    run_moe_router_fused,
    run_moe_scatter_fused,
    run_qk_norm_rotary_fused,
    run_rmsnorm_gated_fused,
    run_rmsnorm_fused,
)
from anna.model.quantization import AWQLinear, AutoRoundGPTQLinear, XPUInt4Linear, convert_module_linears_to_xpu_int4
from anna.model.xpu_decode_profile import xpu_profile_region

logger = logging.getLogger(__name__)


def _compiler_disable(fn: Callable[..., object]) -> Callable[..., object]:
    """Avoid torch.compile / Dynamo tracing through cache-mutating Python state.

    KV and linear-attention caches use Python lists and custom objects; guards on
    those values force recompilation every decode step and hit recompile_limit.
    """
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        return fn
    disable = getattr(dynamo, "disable", None)
    return disable(fn) if callable(disable) else fn


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


@dataclass(slots=True)
class _GroupedInt4ExpertBank:
    capacity: int
    hidden_dim: int
    intermediate_dim: int
    group_size: int
    gate_qweight: torch.Tensor
    gate_qscale: torch.Tensor
    gate_qzeros: torch.Tensor
    up_qweight: torch.Tensor
    up_qscale: torch.Tensor
    up_qzeros: torch.Tensor
    down_qweight: torch.Tensor
    down_qscale: torch.Tensor
    down_qzeros: torch.Tensor
    slot_to_expert: list[int | None]
    expert_to_slot: dict[int, int]


class Qwen3PagedLayerAllocator:
    def __init__(
        self,
        block_size: int,
        *,
        on_page_freed: Callable[[int], None] | None = None,
        on_trim: Callable[[], None] | None = None,
    ) -> None:
        self.block_size = block_size
        self._on_page_freed = on_page_freed
        self._on_trim = on_trim
        self.key_pages: torch.Tensor | None = None
        self.value_pages: torch.Tensor | None = None
        self.free_pages: list[int] = []
        self._refcount: dict[int, int] = {}

    def allocate(
        self,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        page_ids: list[int] = []
        while self.free_pages and len(page_ids) < num_pages:
            page_id = self.free_pages.pop()
            if self._refcount.get(page_id, 0) != 0:
                raise RuntimeError(f"Invariant violated: free list page {page_id} has non-zero refcount.")
            self._refcount[page_id] = 1
            page_ids.append(page_id)

        missing = num_pages - len(page_ids)
        if missing > 0:
            new_ids = self._grow(missing, key_template=key_template, value_template=value_template)
            for page_id in new_ids:
                self._refcount[page_id] = 1
            page_ids.extend(new_ids)
        return page_ids

    def retain_page(self, page_id: int) -> None:
        cap = self.capacity()
        if page_id < 0 or (cap > 0 and page_id >= cap):
            raise ValueError(f"retain_page: invalid page_id={page_id} capacity={cap}.")
        self._refcount[page_id] = self._refcount.get(page_id, 0) + 1

    def release_page(self, page_id: int) -> None:
        rc = self._refcount.get(page_id, 0)
        if rc <= 0:
            return
        rc -= 1
        if rc == 0:
            self._refcount.pop(page_id, None)
            self.free_pages.append(page_id)
            if self._on_page_freed is not None:
                self._on_page_freed(page_id)
        else:
            self._refcount[page_id] = rc

    def release_pages(self, page_ids: list[int]) -> None:
        for page_id in reversed(page_ids):
            self.release_page(page_id)

    def free(self, page_ids: list[int]) -> None:
        self.release_pages(page_ids)

    def capacity(self) -> int:
        if self.key_pages is None:
            return 0
        return int(self.key_pages.shape[0])

    def used_pages(self) -> int:
        capacity = self.capacity()
        return max(0, min(capacity, capacity - len(self.free_pages)))

    def ensure_capacity(
        self,
        required_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> int:
        normalized_required = max(0, int(required_pages))
        current_capacity = self.capacity()
        if normalized_required <= current_capacity:
            return current_capacity
        newly_added = self._grow(
            normalized_required - current_capacity,
            key_template=key_template,
            value_template=value_template,
        )
        for page_id in reversed(newly_added):
            self.free_pages.append(page_id)
        return self.capacity()

    def trim(self) -> int:
        capacity = self.capacity()
        if capacity <= 0 or self.used_pages() > 0:
            return 0
        if self._on_trim is not None:
            self._on_trim()
        self._refcount.clear()
        self.key_pages = None
        self.value_pages = None
        self.free_pages.clear()
        return capacity

    def _grow(
        self,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        old_capacity = 0 if self.key_pages is None else int(self.key_pages.shape[0])
        grow_by = max(num_pages, old_capacity or 16)
        new_capacity = old_capacity + grow_by
        key_shape = (new_capacity, key_template.shape[1], self.block_size, key_template.shape[3])
        value_shape = (new_capacity, value_template.shape[1], self.block_size, value_template.shape[3])
        new_key_pages = key_template.new_empty(key_shape)
        new_value_pages = value_template.new_empty(value_shape)
        if self.key_pages is not None:
            new_key_pages[:old_capacity].copy_(self.key_pages)
        if self.value_pages is not None:
            new_value_pages[:old_capacity].copy_(self.value_pages)
        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

        new_page_ids = list(range(old_capacity, new_capacity))
        allocated = new_page_ids[:num_pages]
        retained_free = new_page_ids[num_pages:]
        for page_id in reversed(retained_free):
            self.free_pages.append(page_id)
        return allocated

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if self.key_pages is not None:
            if dtype is not None and self.key_pages.is_floating_point():
                kwargs["dtype"] = dtype
            self.key_pages = self.key_pages.to(**kwargs)
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if self.value_pages is not None:
            if dtype is not None and self.value_pages.is_floating_point():
                kwargs["dtype"] = dtype
            self.value_pages = self.value_pages.to(**kwargs)


class Qwen3PageAllocator:
    """Paged KV storage per layer. Block size matches ``config.cache_block_size``.

    vLLM-style *prefix block reuse* across requests needs refcounted physical pages and
    coordination in :class:`Qwen3DynamicCache.update` (see ``prefix_block_cache`` helpers).
    """

    def __init__(self, config: Qwen3_5TextConfig, *, maintain_full_attention_mirror: bool = True):
        self.config = config
        self.block_size = max(1, int(config.cache_block_size))
        self.prefix_block_pool = PrefixBlockPool()
        self.layers = [
            Qwen3PagedLayerAllocator(
                self.block_size,
                on_page_freed=lambda pid, li=layer_idx: self.prefix_block_pool.discard_page(li, pid),
                on_trim=lambda li=layer_idx: self.prefix_block_pool.clear_layer(li),
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.maintain_full_attention_mirror = bool(maintain_full_attention_mirror)
        self.full_attention_layer_indices = tuple(
            layer_idx for layer_idx, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"
        )
        self.mirrored_full_attention_layer_indices = (
            self.full_attention_layer_indices if self.maintain_full_attention_mirror else ()
        )
        self._mirrored_full_attention_layer_index_set = set(self.mirrored_full_attention_layer_indices)

    def allocate(
        self,
        layer_idx: int,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        return self.layers[layer_idx].allocate(
            num_pages,
            key_template=key_template,
            value_template=value_template,
        )

    def release_pages(self, layer_idx: int, page_ids: list[int]) -> None:
        if page_ids:
            self.layers[layer_idx].release_pages(page_ids)

    def free(self, layer_idx: int, page_ids: list[int]) -> None:
        self.release_pages(layer_idx, page_ids)

    def retain_shared_page(self, layer_idx: int, page_id: int) -> None:
        self.layers[layer_idx].retain_page(page_id)

    def ensure_capacity(
        self,
        layer_idx: int,
        required_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> int:
        return self.layers[layer_idx].ensure_capacity(
            required_pages,
            key_template=key_template,
            value_template=value_template,
        )

    def trim(self) -> int:
        return sum(layer.trim() for layer in self.layers)

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3PageAllocator":
        for layer in self.layers:
            layer.to(device=device, dtype=dtype)
        return self

    def uses_contiguous_full_attention_mirror(self, layer_idx: int) -> bool:
        return layer_idx in self._mirrored_full_attention_layer_index_set


class Qwen3DynamicCache:
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        allocator: Qwen3PageAllocator | None = None,
        batch_size: int = 0,
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
    ):
        self.config = config
        self.allocator = allocator or Qwen3PageAllocator(config)
        self.kv_cache_quantization = kv_cache_quantization.strip().lower()
        if self.kv_cache_quantization not in {"none", "turboquant"}:
            raise ValueError(f"Unsupported Qwen3 KV-cache quantization mode: {kv_cache_quantization}")
        self.kv_cache_quant_bits = int(kv_cache_quant_bits)
        if self.kv_cache_quant_bits not in {3, 4}:
            raise ValueError(f"Unsupported Qwen3 TurboQuant bit-width: {kv_cache_quant_bits}")
        self.kv_cache_residual_len = max(1, int(kv_cache_residual_len))
        self._turboquant_layer_index_set = {
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if self.kv_cache_quantization == "turboquant" and layer_type == "full_attention"
        }
        self.conv_states: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.block_size = max(1, int(config.cache_block_size))
        self.layer_lengths: list[list[int]] = [
            [0 for _ in range(batch_size)]
            for _ in range(config.num_hidden_layers)
        ]
        self.page_tables: list[list[list[int]]] = [
            [[] for _ in range(batch_size)]
            for _ in range(config.num_hidden_layers)
        ]
        self.turboquant_rows: list[list[TurboQuantKVRow | None]] = [
            [None for _ in range(batch_size)]
            for _ in range(config.num_hidden_layers)
        ]
        self.visible_key_caches: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.visible_value_caches: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.visible_cache_capacities: list[int] = [0 for _ in range(config.num_hidden_layers)]
        self.page_table_caches: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.page_table_capacities: list[int] = [0 for _ in range(config.num_hidden_layers)]
        self.seen_tokens = 0
        self.rope_deltas: torch.Tensor | None = None
        self.reserved_seq_capacity = 0
        self._released = False
        self._prefill_input_ids: torch.LongTensor | None = None
        self._prompt_token_ids: list[int] | None = None
        self._prefix_kv_share = os.environ.get("ANNA_PREFIX_KV_SHARE", "1") != "0"

    def attach_prefill_input_ids(self, input_ids: torch.LongTensor | None) -> None:
        self._prefill_input_ids = input_ids

    def detach_prefill_input_ids(self) -> None:
        self._prefill_input_ids = None

    def set_prompt_token_ids(self, input_ids: torch.LongTensor | None) -> None:
        if input_ids is None or not self._prefix_kv_share:
            return
        if self._prompt_token_ids is not None:
            return
        if int(input_ids.shape[0]) != 1:
            return
        if self.get_seq_length() != 0:
            return
        seq = int(input_ids.shape[1])
        if seq <= 0:
            return
        self._prompt_token_ids = [int(t) for t in input_ids[0].tolist()]

    def reserve_sequence_capacity(self, seq_length: int) -> None:
        self.reserved_seq_capacity = max(0, int(seq_length))

    @property
    def has_previous_state(self) -> bool:
        return any(state is not None for state in self.conv_states) or any(state is not None for state in self.recurrent_states)

    def _ensure_batch_size(self, batch_size: int) -> None:
        if not self.layer_lengths or not self.layer_lengths[0]:
            self.layer_lengths = [[0 for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            self.page_tables = [[[] for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            self.turboquant_rows = [[None for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            return
        if len(self.layer_lengths[0]) != batch_size:
            raise ValueError(f"Cache batch size mismatch: expected {len(self.layer_lengths[0])}, got {batch_size}")

    def uses_turboquant_for_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._turboquant_layer_index_set

    def _uses_contiguous_full_attention_mirror(self, layer_idx: int) -> bool:
        if self.uses_turboquant_for_layer(layer_idx):
            return False
        return self.allocator.uses_contiguous_full_attention_mirror(layer_idx)

    @staticmethod
    def _pad_cache_rows(rows: list[torch.Tensor], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not rows:
            raise ValueError("Qwen3DynamicCache expected at least one cache row.")
        max_length = max(int(row.shape[1]) for row in rows)
        head_count = int(rows[0].shape[0])
        head_dim = int(rows[0].shape[2])
        padded = torch.zeros((len(rows), head_count, max_length, head_dim), device=device, dtype=dtype)
        for batch_idx, row in enumerate(rows):
            row_length = int(row.shape[1])
            if row_length > 0:
                padded[batch_idx, :, :row_length, :].copy_(row.to(device=device, dtype=dtype))
        return padded

    def _next_visible_cache_capacity(self, layer_idx: int, required_length: int) -> int:
        current_capacity = self.visible_cache_capacities[layer_idx]
        if current_capacity >= required_length:
            return current_capacity
        growth_target = max(self.block_size, current_capacity * 2)
        return max(required_length, growth_target)

    def _next_page_table_capacity(self, layer_idx: int, required_blocks: int) -> int:
        current_capacity = self.page_table_capacities[layer_idx]
        if current_capacity >= required_blocks:
            return current_capacity
        growth_target = max(1, current_capacity * 2)
        return max(required_blocks, growth_target)

    def _ensure_visible_layer_buffers(
        self,
        layer_idx: int,
        *,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_size: int,
        required_length: int,
        previous_max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_buffer = self.visible_key_caches[layer_idx]
        value_buffer = self.visible_value_caches[layer_idx]
        required_capacity = self._next_visible_cache_capacity(
            layer_idx,
            max(required_length, self.reserved_seq_capacity),
        )
        compatible = (
            key_buffer is not None
            and value_buffer is not None
            and key_buffer.shape[0] == batch_size
            and value_buffer.shape[0] == batch_size
            and key_buffer.shape[1] == key_states.shape[1]
            and value_buffer.shape[1] == value_states.shape[1]
            and key_buffer.shape[3] == key_states.shape[3]
            and value_buffer.shape[3] == value_states.shape[3]
            and key_buffer.device == key_states.device
            and value_buffer.device == value_states.device
            and key_buffer.dtype == key_states.dtype
            and value_buffer.dtype == value_states.dtype
            and key_buffer.shape[2] >= required_capacity
            and value_buffer.shape[2] >= required_capacity
        )
        if compatible:
            return key_buffer, value_buffer

        new_key_buffer = key_states.new_empty((batch_size, key_states.shape[1], required_capacity, key_states.shape[3]))
        new_value_buffer = value_states.new_empty(
            (batch_size, value_states.shape[1], required_capacity, value_states.shape[3])
        )
        can_copy_existing = (
            key_buffer is not None
            and value_buffer is not None
            and key_buffer.shape[0] == batch_size
            and value_buffer.shape[0] == batch_size
            and key_buffer.shape[1] == key_states.shape[1]
            and value_buffer.shape[1] == value_states.shape[1]
            and key_buffer.shape[3] == key_states.shape[3]
            and value_buffer.shape[3] == value_states.shape[3]
            and key_buffer.device == key_states.device
            and value_buffer.device == value_states.device
            and key_buffer.dtype == key_states.dtype
            and value_buffer.dtype == value_states.dtype
        )
        if can_copy_existing and previous_max_length > 0:
            copy_length = min(previous_max_length, key_buffer.shape[2], value_buffer.shape[2])
            new_key_buffer[:, :, :copy_length, :].copy_(key_buffer[:, :, :copy_length, :])
            new_value_buffer[:, :, :copy_length, :].copy_(value_buffer[:, :, :copy_length, :])
        self.visible_key_caches[layer_idx] = new_key_buffer
        self.visible_value_caches[layer_idx] = new_value_buffer
        self.visible_cache_capacities[layer_idx] = required_capacity
        return new_key_buffer, new_value_buffer

    def _ensure_page_table_layer_buffer(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        required_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        page_table_buffer = self.page_table_caches[layer_idx]
        required_capacity = self._next_page_table_capacity(layer_idx, required_blocks)
        compatible = (
            page_table_buffer is not None
            and page_table_buffer.shape[0] == batch_size
            and page_table_buffer.device == device
            and page_table_buffer.shape[1] >= required_capacity
        )
        if compatible:
            return page_table_buffer

        new_page_table = torch.full((batch_size, required_capacity), -1, device=device, dtype=torch.int32)
        can_copy_existing = (
            page_table_buffer is not None
            and page_table_buffer.shape[0] == batch_size
            and page_table_buffer.device == device
        )
        if can_copy_existing and page_table_buffer.shape[1] > 0:
            copy_blocks = min(page_table_buffer.shape[1], new_page_table.shape[1])
            new_page_table[:, :copy_blocks].copy_(page_table_buffer[:, :copy_blocks])
        self.page_table_caches[layer_idx] = new_page_table
        self.page_table_capacities[layer_idx] = required_capacity
        return new_page_table

    def _sync_page_table_layer_buffer(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        required_blocks = max((len(page_ids) for page_ids in self.page_tables[layer_idx]), default=0)
        if required_blocks <= 0:
            self.page_table_caches[layer_idx] = None
            self.page_table_capacities[layer_idx] = 0
            return None

        page_table_buffer = self._ensure_page_table_layer_buffer(
            layer_idx,
            batch_size=batch_size,
            required_blocks=required_blocks,
            device=device,
        )
        for batch_idx, page_ids in enumerate(self.page_tables[layer_idx]):
            if not page_ids:
                continue
            page_ids_tensor = torch.tensor(page_ids, device=device, dtype=page_table_buffer.dtype)
            page_table_buffer[batch_idx, : page_ids_tensor.shape[0]].copy_(page_ids_tensor)
        return page_table_buffer[:, :required_blocks]

    def _update_visible_layer_cache(
        self,
        layer_idx: int,
        *,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible_lengths = self.layer_lengths[layer_idx]
        max_length = max(visible_lengths, default=0)
        previous_max_length = int(past_lengths.max().item()) if past_lengths.numel() > 0 else 0
        key_buffer, value_buffer = self._ensure_visible_layer_buffers(
            layer_idx,
            key_states=key_states,
            value_states=value_states,
            batch_size=key_states.shape[0],
            required_length=max_length,
            previous_max_length=previous_max_length,
        )

        for batch_idx, start_position in enumerate(past_lengths.tolist()):
            append_length = key_states.shape[2]
            end_position = start_position + append_length
            key_buffer[batch_idx, :, start_position:end_position, :].copy_(key_states[batch_idx])
            value_buffer[batch_idx, :, start_position:end_position, :].copy_(value_states[batch_idx])

        return key_buffer[:, :, :max_length, :], value_buffer[:, :, :max_length, :]

    def _update_turboquant_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *,
        require_dense_cache: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        batch_size = int(key_states.shape[0])
        self._ensure_batch_size(batch_size)
        past_lengths = torch.tensor(self.layer_lengths[layer_idx], device=key_states.device, dtype=torch.long)
        materialized_keys: list[torch.Tensor] = []
        materialized_values: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            row = self.turboquant_rows[layer_idx][batch_idx]
            if row is None:
                row = TurboQuantKVRow(
                    bits=self.kv_cache_quant_bits,
                    residual_len=self.kv_cache_residual_len,
                )
                self.turboquant_rows[layer_idx][batch_idx] = row
            row.append(key_states[batch_idx], value_states[batch_idx])
            self.layer_lengths[layer_idx][batch_idx] = row.length
            if require_dense_cache:
                row_key, row_value = row.materialize(
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                materialized_keys.append(row_key)
                materialized_values.append(row_value.to(device=value_states.device, dtype=value_states.dtype))

        self.visible_key_caches[layer_idx] = None
        self.visible_value_caches[layer_idx] = None
        self.visible_cache_capacities[layer_idx] = 0
        self.page_table_caches[layer_idx] = None
        self.page_table_capacities[layer_idx] = 0
        self.seen_tokens = max(self.get_seq_lengths().tolist(), default=0)
        if not require_dense_cache:
            return None, None, past_lengths
        padded_key = self._pad_cache_rows(materialized_keys, device=key_states.device, dtype=key_states.dtype)
        padded_value = self._pad_cache_rows(materialized_values, device=value_states.device, dtype=value_states.dtype)
        return padded_key, padded_value, past_lengths

    @_compiler_disable
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *,
        require_dense_cache: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.uses_turboquant_for_layer(layer_idx):
            return self._update_turboquant_layer(
                key_states,
                value_states,
                layer_idx,
                require_dense_cache=require_dense_cache,
            )
        batch_size, _, append_length, _ = key_states.shape
        self._ensure_batch_size(batch_size)
        past_lengths = torch.tensor(self.layer_lengths[layer_idx], device=key_states.device, dtype=torch.long)
        reserved_length = max(append_length, self.reserved_seq_capacity)
        required_blocks_hint = (reserved_length + self.block_size - 1) // self.block_size
        if required_blocks_hint > 0:
            self.allocator.ensure_capacity(
                layer_idx,
                required_blocks_hint * batch_size,
                key_template=key_states,
                value_template=value_states,
            )

        use_prefix_share = (
            self._prefix_kv_share
            and batch_size == 1
            and self._prompt_token_ids is not None
        )
        for batch_idx in range(batch_size):
            if use_prefix_share:
                self._update_paged_row_with_prefix_sharing(
                    key_states,
                    value_states,
                    layer_idx,
                    batch_idx,
                    append_length,
                )
                continue

            current_length = self.layer_lengths[layer_idx][batch_idx]
            required_length = current_length + append_length
            reserved_required_length = max(required_length, self.reserved_seq_capacity)
            required_blocks = (reserved_required_length + self.block_size - 1) // self.block_size
            page_table = self.page_tables[layer_idx][batch_idx]
            if len(page_table) < required_blocks:
                page_table.extend(
                    self.allocator.allocate(
                        layer_idx,
                        required_blocks - len(page_table),
                        key_template=key_states,
                        value_template=value_states,
                    )
                )

            self._write_pages(
                layer_idx=layer_idx,
                page_ids=page_table,
                start_position=current_length,
                key_states=key_states[batch_idx],
                value_states=value_states[batch_idx],
            )
            self.layer_lengths[layer_idx][batch_idx] = required_length

        self.seen_tokens = max(self.get_seq_lengths().tolist(), default=0)
        self._sync_page_table_layer_buffer(layer_idx, batch_size=batch_size, device=key_states.device)
        if self._uses_contiguous_full_attention_mirror(layer_idx):
            mirrored_key, mirrored_value = self._update_visible_layer_cache(
                layer_idx,
                key_states=key_states,
                value_states=value_states,
                past_lengths=past_lengths,
            )
            if require_dense_cache:
                return mirrored_key, mirrored_value, past_lengths
            return None, None, past_lengths
        if not require_dense_cache:
            return None, None, past_lengths
        gathered_key, gathered_value, _ = self._gather_layer_cache(layer_idx)
        return gathered_key, gathered_value, past_lengths

    def _write_pages(
        self,
        *,
        layer_idx: int,
        page_ids: list[int],
        start_position: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        layer = self.allocator.layers[layer_idx]
        if layer.key_pages is None or layer.value_pages is None:
            raise RuntimeError(f"Layer {layer_idx} pages are not initialized.")

        remaining = int(key_states.shape[1])
        src_offset = 0
        current_position = start_position
        while remaining > 0:
            block_idx = current_position // self.block_size
            block_offset = current_position % self.block_size
            take = min(remaining, self.block_size - block_offset)
            page_id = page_ids[block_idx]
            layer.key_pages[page_id, :, block_offset : block_offset + take, :].copy_(
                key_states[:, src_offset : src_offset + take, :]
            )
            layer.value_pages[page_id, :, block_offset : block_offset + take, :].copy_(
                value_states[:, src_offset : src_offset + take, :]
            )
            current_position += take
            src_offset += take
            remaining -= take

    def _update_paged_row_with_prefix_sharing(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_idx: int,
        append_length: int,
    ) -> None:
        prompt = self._prompt_token_ids
        if prompt is None:
            raise RuntimeError("prefix sharing requires prompt token ids.")
        current_length = self.layer_lengths[layer_idx][batch_idx]
        required_length = current_length + append_length
        page_table = self.page_tables[layer_idx][batch_idx]
        pool = self.allocator.prefix_block_pool
        bs = self.block_size

        pos = current_length
        while pos < required_length:
            b = pos // bs
            glob_lo = pos
            glob_hi = min(required_length, (b + 1) * bs)
            ntok = glob_hi - glob_lo
            full_block = ntok == bs and glob_lo == b * bs

            first_touch = len(page_table) <= b
            hit: int | None = None
            if first_touch and full_block and glob_hi <= len(prompt):
                token_block = tuple(prompt[glob_lo:glob_hi])
                hit = pool.lookup(layer_idx, token_block)
            else:
                token_block = None

            while len(page_table) <= b:
                if hit is not None:
                    self.allocator.retain_shared_page(layer_idx, hit)
                    page_table.append(hit)
                else:
                    page_table.append(
                        self.allocator.allocate(
                            layer_idx,
                            1,
                            key_template=key_states,
                            value_template=value_states,
                        )[0]
                    )
                break

            pid = page_table[b]
            skipped_shared = full_block and hit is not None and pid == hit

            if not skipped_shared:
                loc_lo = glob_lo - current_length
                loc_hi = glob_hi - current_length
                self._write_pages(
                    layer_idx=layer_idx,
                    page_ids=page_table,
                    start_position=glob_lo,
                    key_states=key_states[batch_idx, :, loc_lo:loc_hi, :],
                    value_states=value_states[batch_idx, :, loc_lo:loc_hi, :],
                )
                if full_block and first_touch and hit is None and glob_hi <= len(prompt):
                    pool.register(layer_idx, tuple(prompt[glob_lo:glob_hi]), pid)

            pos = glob_hi

        self.layer_lengths[layer_idx][batch_idx] = required_length

        reserved_required_length = max(required_length, self.reserved_seq_capacity)
        required_blocks_reserved = (reserved_required_length + self.block_size - 1) // self.block_size
        while len(page_table) < required_blocks_reserved:
            page_table.append(
                self.allocator.allocate(
                    layer_idx,
                    1,
                    key_template=key_states,
                    value_template=value_states,
                )[0]
            )

    def get_batch_size(self) -> int:
        if self.layer_lengths and self.layer_lengths[0]:
            return len(self.layer_lengths[0])
        for state_group in (self.conv_states, self.recurrent_states):
            for state in state_group:
                if state is not None:
                    return int(state.shape[0])
        if self.rope_deltas is not None:
            return int(self.rope_deltas.shape[0])
        return 0

    def get_seq_length(self, batch_idx: int | None = None) -> int:
        if batch_idx is not None:
            return self._request_seq_length(batch_idx)
        batch_size = self.get_batch_size()
        if batch_size == 0:
            return 0
        if batch_size == 1:
            return self._request_seq_length(0)
        return max(self._request_seq_length(idx) for idx in range(batch_size))

    def get_seq_lengths(self, *, device: torch.device | None = None) -> torch.Tensor:
        lengths = [self._request_seq_length(idx) for idx in range(self.get_batch_size())]
        return torch.tensor(lengths, dtype=torch.long, device=device)

    def _request_seq_length(self, batch_idx: int) -> int:
        for layer_idx in range(self.config.num_hidden_layers):
            length = self.layer_lengths[layer_idx][batch_idx]
            if length > 0:
                return length
        return 0

    def visible_key_cache(self, layer_idx: int) -> torch.Tensor | None:
        key_tensor, _, _ = self._gather_layer_cache(layer_idx)
        return key_tensor

    def visible_value_cache(self, layer_idx: int) -> torch.Tensor | None:
        _, value_tensor, _ = self._gather_layer_cache(layer_idx)
        return value_tensor

    def paged_attention_state(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.uses_turboquant_for_layer(layer_idx):
            return None
        batch_size = self.get_batch_size()
        if batch_size == 0:
            return None

        layer = self.allocator.layers[layer_idx]
        if layer.key_pages is None or layer.value_pages is None:
            return None

        visible_lengths = torch.tensor(self.layer_lengths[layer_idx], dtype=torch.long, device=layer.key_pages.device)
        if visible_lengths.numel() == 0:
            return None

        required_blocks = max((len(page_ids) for page_ids in self.page_tables[layer_idx]), default=0)
        layer_seq_max = max(self.layer_lengths[layer_idx], default=0)
        if required_blocks <= 0 or layer_seq_max <= 0:
            return None

        page_table = self._sync_page_table_layer_buffer(
            layer_idx,
            batch_size=batch_size,
            device=layer.key_pages.device,
        )
        if page_table is None:
            return None
        return layer.key_pages, layer.value_pages, page_table, visible_lengths

    def _gather_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.uses_turboquant_for_layer(layer_idx):
            batch_size = self.get_batch_size()
            if batch_size == 0:
                empty_lengths = torch.zeros(0, dtype=torch.long)
                return None, None, empty_lengths
            visible_lengths = torch.tensor(self.layer_lengths[layer_idx], dtype=torch.long)
            max_length = max(self.layer_lengths[layer_idx], default=0)
            if max_length <= 0:
                return None, None, visible_lengths
            materialized_keys: list[torch.Tensor] = []
            materialized_values: list[torch.Tensor] = []
            key_device: torch.device | None = None
            key_dtype: torch.dtype | None = None
            value_device: torch.device | None = None
            value_dtype: torch.dtype | None = None
            for batch_idx in range(batch_size):
                row = self.turboquant_rows[layer_idx][batch_idx]
                if row is None:
                    raise RuntimeError(f"TurboQuant cache row for layer {layer_idx} batch {batch_idx} is missing.")
                row_key, row_value = row.materialize()
                if key_device is None:
                    key_device = row_key.device
                    key_dtype = row_key.dtype
                    value_device = row_value.device
                    value_dtype = row_value.dtype
                materialized_keys.append(row_key)
                materialized_values.append(row_value)
            assert key_device is not None
            assert key_dtype is not None
            assert value_device is not None
            assert value_dtype is not None
            visible_lengths = visible_lengths.to(device=key_device)
            return (
                self._pad_cache_rows(materialized_keys, device=key_device, dtype=key_dtype),
                self._pad_cache_rows(materialized_values, device=value_device, dtype=value_dtype),
                visible_lengths,
            )
        batch_size = self.get_batch_size()
        if batch_size == 0:
            empty_lengths = torch.zeros(0, dtype=torch.long)
            return None, None, empty_lengths

        key_buffer = self.visible_key_caches[layer_idx]
        value_buffer = self.visible_value_caches[layer_idx]
        visible_lengths_device = None if key_buffer is None else key_buffer.device
        visible_lengths = torch.tensor(self.layer_lengths[layer_idx], dtype=torch.long, device=visible_lengths_device)
        max_length = max(self.layer_lengths[layer_idx], default=0)
        if max_length <= 0:
            return None, None, visible_lengths

        if key_buffer is not None and value_buffer is not None:
            return key_buffer[:, :, :max_length, :], value_buffer[:, :, :max_length, :], visible_lengths

        layer = self.allocator.layers[layer_idx]
        if layer.key_pages is None or layer.value_pages is None:
            return None, None, visible_lengths

        key_batch = layer.key_pages.new_zeros(
            (batch_size, layer.key_pages.shape[1], max_length, layer.key_pages.shape[3])
        )
        value_batch = layer.value_pages.new_zeros(
            (batch_size, layer.value_pages.shape[1], max_length, layer.value_pages.shape[3])
        )

        for batch_idx, length in enumerate(self.layer_lengths[layer_idx]):
            copied = 0
            for page_id in self.page_tables[layer_idx][batch_idx]:
                if copied >= length:
                    break
                take = min(self.block_size, length - copied)
                key_batch[batch_idx, :, copied : copied + take, :].copy_(layer.key_pages[page_id, :, :take, :])
                value_batch[batch_idx, :, copied : copied + take, :].copy_(layer.value_pages[page_id, :, :take, :])
                copied += take

        return key_batch, value_batch, visible_lengths

    def _visible_layer_state(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if self.uses_turboquant_for_layer(layer_idx):
            key_tensor, value_tensor, _ = self._gather_layer_cache(layer_idx)
            if key_tensor is None or value_tensor is None:
                return None, None, 0
            return key_tensor, value_tensor, int(key_tensor.shape[2])
        key_buffer = self.visible_key_caches[layer_idx]
        value_buffer = self.visible_value_caches[layer_idx]
        if key_buffer is not None and value_buffer is not None:
            return key_buffer, value_buffer, max(self.visible_cache_capacities[layer_idx], int(key_buffer.shape[2]))

        key_tensor, value_tensor, _ = self._gather_layer_cache(layer_idx)
        if key_tensor is None or value_tensor is None:
            return None, None, 0
        return key_tensor, value_tensor, int(key_tensor.shape[2])

    def turboquant_single_token_decode_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        *,
        scaling: float,
        num_key_value_groups: int,
    ) -> torch.Tensor:
        if not self.uses_turboquant_for_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} does not use TurboQuant-backed KV storage.")
        if query_states.ndim != 4 or query_states.shape[2] != 1:
            raise ValueError(
                "TurboQuant decode attention expects query_states shaped [batch, query_heads, 1, head_dim]."
            )
        batch = int(query_states.shape[0])
        if batch == 1:
            row = self.turboquant_rows[layer_idx][0]
            if row is None or row.length <= 0:
                return query_states.new_zeros((1, int(query_states.shape[1]), 1, int(query_states.shape[3])))
            return row.decode_attention(
                query_states[0],
                scaling=scaling,
                num_key_value_groups=num_key_value_groups,
            ).unsqueeze(0)
        n_head = int(query_states.shape[1])
        head_dim = int(query_states.shape[3])
        out = query_states.new_empty((batch, n_head, 1, head_dim))
        for batch_idx in range(batch):
            row = self.turboquant_rows[layer_idx][batch_idx]
            if row is None or row.length <= 0:
                out[batch_idx].zero_()
                continue
            out[batch_idx].copy_(
                row.decode_attention(
                    query_states[batch_idx],
                    scaling=scaling,
                    num_key_value_groups=num_key_value_groups,
                )
            )
        return out

    @classmethod
    def stack(cls, caches: list["Qwen3DynamicCache"], config: Qwen3_5TextConfig) -> "Qwen3DynamicCache":
        if not caches:
            return cls(config)

        if any(cache.allocator is not caches[0].allocator for cache in caches):
            raise ValueError("Cannot stack caches from different allocators.")

        rows: list[Qwen3DynamicCache] = []
        for cache in caches:
            if cache.get_batch_size() <= 1:
                rows.append(cache)
            else:
                rows.extend(cache.split_batch())

        prototype = rows[0]
        stacked = cls(
            config,
            allocator=prototype.allocator,
            batch_size=len(rows),
            kv_cache_quantization=prototype.kv_cache_quantization,
            kv_cache_quant_bits=prototype.kv_cache_quant_bits,
            kv_cache_residual_len=prototype.kv_cache_residual_len,
        )
        for layer_idx in range(config.num_hidden_layers):
            stacked.layer_lengths[layer_idx] = [row.layer_lengths[layer_idx][0] for row in rows]
        stacked.seen_tokens = max(stacked.get_seq_lengths().tolist(), default=0)
        stacked.rope_deltas = None if rows[0].rope_deltas is None else torch.cat([row.rope_deltas for row in rows], dim=0)
        stacked.reserved_seq_capacity = max((row.reserved_seq_capacity for row in rows), default=0)

        for layer_idx in range(config.num_hidden_layers):
            if stacked.uses_turboquant_for_layer(layer_idx):
                for batch_idx, row in enumerate(rows):
                    turboquant_row = row.turboquant_rows[layer_idx][0]
                    stacked.turboquant_rows[layer_idx][batch_idx] = None if turboquant_row is None else turboquant_row.clone()
                continue
            stacked.page_tables[layer_idx] = [list(row.page_tables[layer_idx][0]) for row in rows]

            conv_states = [row.conv_states[layer_idx] for row in rows]
            if all(state is None for state in conv_states):
                stacked.conv_states[layer_idx] = None
            elif any(state is None for state in conv_states):
                raise ValueError(f"Cannot stack partially initialized conv states at layer {layer_idx}")
            else:
                stacked.conv_states[layer_idx] = torch.cat(conv_states, dim=0)

            recurrent_states = [row.recurrent_states[layer_idx] for row in rows]
            if all(state is None for state in recurrent_states):
                stacked.recurrent_states[layer_idx] = None
            elif any(state is None for state in recurrent_states):
                raise ValueError(f"Cannot stack partially initialized recurrent states at layer {layer_idx}")
            else:
                stacked.recurrent_states[layer_idx] = torch.cat(recurrent_states, dim=0)

            if stacked._uses_contiguous_full_attention_mirror(layer_idx):
                visible_rows: list[tuple[torch.Tensor | None, torch.Tensor | None, int]] = []
                visible_capacity = 0
                ref_key: torch.Tensor | None = None
                ref_value: torch.Tensor | None = None
                for row in rows:
                    key_buffer, value_buffer, capacity = row._visible_layer_state(layer_idx)
                    visible_rows.append((key_buffer, value_buffer, capacity))
                    if key_buffer is None or value_buffer is None:
                        continue
                    if ref_key is None or ref_value is None:
                        ref_key = key_buffer
                        ref_value = value_buffer
                    visible_capacity = max(visible_capacity, capacity)

                if ref_key is not None and ref_value is not None:
                    stacked_key = ref_key.new_empty((len(rows), ref_key.shape[1], visible_capacity, ref_key.shape[3]))
                    stacked_value = ref_value.new_empty((len(rows), ref_value.shape[1], visible_capacity, ref_value.shape[3]))
                    for batch_idx, (key_buffer, value_buffer, _) in enumerate(visible_rows):
                        if key_buffer is None or value_buffer is None:
                            continue
                        key_copy_length = min(visible_capacity, int(key_buffer.shape[2]))
                        value_copy_length = min(visible_capacity, int(value_buffer.shape[2]))
                        if key_copy_length > 0:
                            stacked_key[batch_idx, :, :key_copy_length, :].copy_(key_buffer[0, :, :key_copy_length, :])
                        if value_copy_length > 0:
                            stacked_value[batch_idx, :, :value_copy_length, :].copy_(
                                value_buffer[0, :, :value_copy_length, :]
                            )
                    stacked.visible_key_caches[layer_idx] = stacked_key
                    stacked.visible_value_caches[layer_idx] = stacked_value
                    stacked.visible_cache_capacities[layer_idx] = visible_capacity

        return stacked

    def split_batch(self) -> list["Qwen3DynamicCache"]:
        batch_size = self.get_batch_size()
        if batch_size <= 1:
            return [self]

        outputs = [
            Qwen3DynamicCache(
                self.config,
                allocator=self.allocator,
                batch_size=1,
                kv_cache_quantization=self.kv_cache_quantization,
                kv_cache_quant_bits=self.kv_cache_quant_bits,
                kv_cache_residual_len=self.kv_cache_residual_len,
            )
            for _ in range(batch_size)
        ]
        for batch_idx, cache in enumerate(outputs):
            for layer_idx in range(self.config.num_hidden_layers):
                cache.layer_lengths[layer_idx][0] = self.layer_lengths[layer_idx][batch_idx]
            cache.seen_tokens = cache.get_seq_length()
            cache.reserved_seq_capacity = self.reserved_seq_capacity

        if self.rope_deltas is not None:
            rope_chunks = self.rope_deltas.split(1, dim=0)
            for idx, chunk in enumerate(rope_chunks):
                outputs[idx].rope_deltas = chunk

        for layer_idx in range(self.config.num_hidden_layers):
            for idx in range(batch_size):
                outputs[idx].page_tables[layer_idx][0] = list(self.page_tables[layer_idx][idx])
                turboquant_row = self.turboquant_rows[layer_idx][idx]
                if turboquant_row is not None:
                    outputs[idx].turboquant_rows[layer_idx][0] = turboquant_row.clone()
            if self.conv_states[layer_idx] is not None:
                for idx, chunk in enumerate(self.conv_states[layer_idx].split(1, dim=0)):
                    outputs[idx].conv_states[layer_idx] = chunk.clone()
            if self.recurrent_states[layer_idx] is not None:
                for idx, chunk in enumerate(self.recurrent_states[layer_idx].split(1, dim=0)):
                    outputs[idx].recurrent_states[layer_idx] = chunk.clone()
            if self.visible_key_caches[layer_idx] is not None and self.visible_value_caches[layer_idx] is not None:
                for idx, cache in enumerate(outputs):
                    cache.visible_key_caches[layer_idx] = self.visible_key_caches[layer_idx][idx : idx + 1].clone()
                    cache.visible_value_caches[layer_idx] = self.visible_value_caches[layer_idx][idx : idx + 1].clone()
                    cache.visible_cache_capacities[layer_idx] = self.visible_cache_capacities[layer_idx]
        return outputs

    def clone(self) -> "Qwen3DynamicCache":
        batch_size = self.get_batch_size()
        cloned = Qwen3DynamicCache(
            self.config,
            allocator=self.allocator,
            batch_size=batch_size,
            kv_cache_quantization=self.kv_cache_quantization,
            kv_cache_quant_bits=self.kv_cache_quant_bits,
            kv_cache_residual_len=self.kv_cache_residual_len,
        )
        for layer_idx in range(self.config.num_hidden_layers):
            cloned.layer_lengths[layer_idx] = list(self.layer_lengths[layer_idx])
            cloned.visible_cache_capacities[layer_idx] = self.visible_cache_capacities[layer_idx]

        cloned.seen_tokens = self.seen_tokens
        cloned.reserved_seq_capacity = self.reserved_seq_capacity
        if self.rope_deltas is not None:
            cloned.rope_deltas = self.rope_deltas.clone()

        for layer_idx in range(self.config.num_hidden_layers):
            if self.conv_states[layer_idx] is not None:
                cloned.conv_states[layer_idx] = self.conv_states[layer_idx].clone()
            if self.recurrent_states[layer_idx] is not None:
                cloned.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].clone()
            if self.visible_key_caches[layer_idx] is not None:
                cloned.visible_key_caches[layer_idx] = self.visible_key_caches[layer_idx].clone()
            if self.visible_value_caches[layer_idx] is not None:
                cloned.visible_value_caches[layer_idx] = self.visible_value_caches[layer_idx].clone()
            for batch_idx, row in enumerate(self.turboquant_rows[layer_idx]):
                if row is not None:
                    cloned.turboquant_rows[layer_idx][batch_idx] = row.clone()

            if self.uses_turboquant_for_layer(layer_idx):
                continue

            layer = self.allocator.layers[layer_idx]
            if layer.key_pages is None or layer.value_pages is None:
                continue

            key_template = layer.key_pages[:1]
            value_template = layer.value_pages[:1]
            for batch_idx, page_ids in enumerate(self.page_tables[layer_idx]):
                if not page_ids:
                    continue
                new_page_ids = self.allocator.allocate(
                    layer_idx,
                    len(page_ids),
                    key_template=key_template,
                    value_template=value_template,
                )
                for old_page_id, new_page_id in zip(page_ids, new_page_ids):
                    layer.key_pages[new_page_id].copy_(layer.key_pages[old_page_id])
                    layer.value_pages[new_page_id].copy_(layer.value_pages[old_page_id])
                cloned.page_tables[layer_idx][batch_idx] = list(new_page_ids)

        cloned._prefill_input_ids = None
        if self._prompt_token_ids is not None:
            cloned._prompt_token_ids = list(self._prompt_token_ids)
        return cloned

    def release(self) -> None:
        if self._released:
            return

        self._prefill_input_ids = None
        self._prompt_token_ids = None
        for layer_idx in range(self.config.num_hidden_layers):
            if self.uses_turboquant_for_layer(layer_idx):
                for batch_idx in range(len(self.turboquant_rows[layer_idx])):
                    self.turboquant_rows[layer_idx][batch_idx] = None
            else:
                for page_ids in self.page_tables[layer_idx]:
                    self.allocator.release_pages(layer_idx, page_ids)
                    page_ids.clear()
            for batch_idx in range(len(self.layer_lengths[layer_idx])):
                self.layer_lengths[layer_idx][batch_idx] = 0
            self.conv_states[layer_idx] = None
            self.recurrent_states[layer_idx] = None
            self.visible_key_caches[layer_idx] = None
            self.visible_value_caches[layer_idx] = None
            self.visible_cache_capacities[layer_idx] = 0
            self.page_table_caches[layer_idx] = None
            self.page_table_capacities[layer_idx] = 0

        self.seen_tokens = 0
        self.rope_deltas = None
        self.reserved_seq_capacity = 0
        self._released = True

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3DynamicCache":
        def _move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and tensor.is_floating_point():
                kwargs["dtype"] = dtype
            return tensor.to(**kwargs)

        self.conv_states = [_move_tensor(tensor) for tensor in self.conv_states]
        self.recurrent_states = [_move_tensor(tensor) for tensor in self.recurrent_states]
        self.visible_key_caches = [_move_tensor(tensor) for tensor in self.visible_key_caches]
        self.visible_value_caches = [_move_tensor(tensor) for tensor in self.visible_value_caches]
        self.page_table_caches = [_move_tensor(tensor) for tensor in self.page_table_caches]
        self.rope_deltas = _move_tensor(self.rope_deltas)
        for layer_idx in range(self.config.num_hidden_layers):
            for batch_idx, row in enumerate(self.turboquant_rows[layer_idx]):
                if row is not None:
                    self.turboquant_rows[layer_idx][batch_idx] = row.to(device=device, dtype=dtype)
        self.allocator.to(device=device, dtype=dtype)
        return self


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "xpu":
            return run_rmsnorm_fused(input=x, weight=self.weight, eps=self.eps)
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(dtype=x.dtype)


class Qwen3RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type == "xpu":
            return run_rmsnorm_gated_fused(
                input=hidden_states,
                gate=gate,
                weight=self.weight,
                eps=self.eps,
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        hidden_states = self.weight * hidden_states.to(dtype=input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(dtype=input_dtype)


class Qwen3TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        inv_freq, attention_scaling = self.compute_default_rope_parameters(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = attention_scaling
        self.mrope_section = tuple(config.rope_parameters.mrope_section)

    @staticmethod
    def compute_default_rope_parameters(config: Qwen3_5TextConfig) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters.rope_theta
        dim = int(config.head_dim * config.rope_parameters.partial_rotary_factor)
        dim = max(2, dim - (dim % 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        last_dim = freqs_t.shape[-1]
        for dim, offset in enumerate((1, 2), start=1):
            length = min(last_dim, self.mrope_section[dim] * 3)
            if length <= offset:
                continue
            idx = torch.arange(offset, length, 3, device=freqs.device)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = ((q_rot * cos) + (rotate_half(q_rot) * sin)).to(dtype=q.dtype)
    k_embed = ((k_rot * cos) + (rotate_half(k_rot) * sin)).to(dtype=k.dtype)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def grouped_query_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    scaling: float,
    causal_mask: torch.Tensor | None = None,
    visible_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, num_heads, query_len, _ = query_states.shape
    num_key_value_heads = key_states.shape[1]
    grouped_query_states = query_states.unflatten(1, (num_key_value_heads, num_heads // num_key_value_heads))
    attn_scores = torch.matmul(grouped_query_states, key_states.unsqueeze(2).transpose(-1, -2)) * scaling
    if causal_mask is not None:
        attn_scores = attn_scores.masked_fill(causal_mask[:, None, None, :, :], float("-inf"))
    if visible_mask is not None:
        attn_scores = attn_scores.masked_fill(~visible_mask[:, None, None, None, :], float("-inf"))
    if key_padding_mask is not None:
        attn_scores = attn_scores.masked_fill(~key_padding_mask[:, None, None, None, :], float("-inf"))
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states.unsqueeze(2))
    return attn_output.reshape(batch_size, num_heads, query_len, -1)


def materialized_kv_single_token_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    scaling: float,
    num_key_value_groups: int,
    visible_lengths: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if query_states.device.type == "xpu":
        if query_states.shape[2] != 1:
            raise RuntimeError("fused materialized GQA decode expects a single-token query.")
        if query_states.shape[1] != key_states.shape[1] * num_key_value_groups:
            raise RuntimeError(
                "fused materialized GQA decode head grouping mismatch: "
                f"query_heads={query_states.shape[1]} kv_heads={key_states.shape[1]} groups={num_key_value_groups}"
            )
        return run_gqa_decode_fused(
            query=query_states,
            key=key_states,
            value=value_states,
            visible_lengths=visible_lengths,
            scaling=scaling,
            gate=gate,
        )

    key_positions = torch.arange(key_states.shape[-2], device=query_states.device)[None, :]
    visible_mask = key_positions < visible_lengths[:, None]
    if not bool(torch.all(visible_mask).item()):
        key_states = key_states.masked_fill(~visible_mask[:, None, :, None], 0)
        value_states = value_states.masked_fill(~visible_mask[:, None, :, None], 0)

    repeated_key_states = repeat_kv(key_states, num_key_value_groups)
    repeated_value_states = repeat_kv(value_states, num_key_value_groups)
    attn_scores = torch.matmul(query_states, repeated_key_states.transpose(-1, -2)) * scaling
    attn_scores = attn_scores.masked_fill(~visible_mask[:, None, None, :], float("-inf"))
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
    return torch.matmul(attn_probs, repeated_value_states)


def paged_kv_single_token_decode_attention(
    query_states: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    page_table: torch.Tensor,
    *,
    scaling: float,
    visible_lengths: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    return run_paged_gqa_decode_fused(
        query=query_states,
        key_pages=key_pages,
        value_pages=value_pages,
        page_table=page_table,
        visible_lengths=visible_lengths,
        scaling=scaling,
        gate=gate,
    )


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1)
    if hidden_states_new.dtype != weight.dtype:
        hidden_states_new = hidden_states_new.to(dtype=weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    if seq_len == 1:
        window = hidden_states_new[:, :, -weight.shape[-1] :]
        out = (window * weight.unsqueeze(0)).sum(dim=-1)
        if bias is not None:
            out = out + bias.unsqueeze(0)
        out = F.silu(out).unsqueeze(-1)
        return out.to(dtype=hidden_states.dtype)
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(dtype=hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    query = query * (query.shape[-1] ** -0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_head_dim, value_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(dtype=value.dtype)
    )
    core_attn_out = torch.zeros_like(value)
    inter_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(inter_mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, value_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(dtype=initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    query = query * (query.shape[-1] ** -0.5)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_head_dim, value_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(dtype=value.dtype)
    )

    if sequence_length == 1:
        q_t = query[:, :, 0]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(2)
        if not output_final_state:
            last_recurrent_state = None
        return core_attn_out.transpose(1, 2).contiguous().to(dtype=initial_dtype), last_recurrent_state

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, value_head_dim, device=value.device, dtype=value.dtype)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out.transpose(1, 2).contiguous().to(dtype=initial_dtype), last_recurrent_state


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.profile_runtime = False
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if query_len == 1:
            return None
        q_positions = past_lengths[:, None] + torch.arange(query_len, device=device)[None, :]
        k_positions = torch.arange(key_len, device=device)[None, None, :]
        return k_positions > q_positions[:, :, None]

    @_compiler_disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        if self.profile_runtime:
            with xpu_profile_region("attention"):
                return self._forward_attention(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
        return self._forward_attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if hidden_states.device.type == "xpu":
            query_states, key_states = run_qk_norm_rotary_fused(
                query=query_states,
                key=key_states,
                query_norm_weight=self.q_norm.weight,
                key_norm_weight=self.k_norm.weight,
                cos=cos,
                sin=sin,
                query_norm_eps=self.q_norm.eps,
                key_norm_eps=self.k_norm.eps,
            )
        else:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_lengths = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.long)
        use_turboquant_cache = past_key_values is not None and past_key_values.uses_turboquant_for_layer(self.layer_idx)
        prefer_paged_decode = (
            hidden_states.device.type == "xpu"
            and seq_len == 1
            and attention_mask is None
            and past_key_values is not None
            and not use_turboquant_cache
        )
        if past_key_values is not None:
            require_dense_cache = not prefer_paged_decode
            if use_turboquant_cache and attention_mask is None and seq_len == 1:
                require_dense_cache = False
            key_states, value_states, past_lengths = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                require_dense_cache=require_dense_cache,
            )

        if past_key_values is None and attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                is_causal=seq_len > 1,
                enable_gqa=self.num_key_value_groups > 1,
            )
        else:
            causal_mask = None
            visible_mask = None
            key_padding_mask = None
            if use_turboquant_cache and attention_mask is None and seq_len == 1 and past_key_values is not None:
                attn_output = past_key_values.turboquant_single_token_decode_attention(
                    self.layer_idx,
                    query_states,
                    scaling=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                )
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                return self.o_proj(attn_output)
            if prefer_paged_decode and past_key_values is not None:
                visible_lengths = past_lengths + seq_len
                paged_state = past_key_values.paged_attention_state(self.layer_idx)
                if paged_state is not None:
                    key_pages, value_pages, page_table, _ = paged_state
                    gate_fused = gate.contiguous()
                    attn_output = paged_kv_single_token_decode_attention(
                        query_states,
                        key_pages,
                        value_pages,
                        page_table,
                        scaling=self.scaling,
                        visible_lengths=visible_lengths,
                        gate=gate_fused,
                    )
                    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
                    return self.o_proj(attn_output)
            if attention_mask is None and past_key_values is not None and seq_len == 1 and hidden_states.device.type == "xpu":
                if key_states is None or value_states is None:
                    key_states, value_states, _ = past_key_values._gather_layer_cache(self.layer_idx)
                    if key_states is None or value_states is None:
                        raise RuntimeError("Failed to materialize KV cache for Qwen single-token decode.")
                visible_lengths = past_lengths + seq_len
                gate_fused = gate.contiguous()
                attn_output = materialized_kv_single_token_decode_attention(
                    query_states,
                    key_states,
                    value_states,
                    scaling=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    visible_lengths=visible_lengths,
                    gate=gate_fused,
                )
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
                return self.o_proj(attn_output)
            # Multi-token decode and masked full-attention stay on the explicit grouped path.
            if not (batch_size == 1 and seq_len == 1 and attention_mask is None and past_key_values is not None):
                visible_lengths = past_lengths + seq_len

                causal_mask = self._causal_mask(seq_len, key_states.shape[-2], past_lengths, hidden_states.device)
                key_positions = torch.arange(key_states.shape[-2], device=hidden_states.device)[None, :]
                visible_mask = key_positions < visible_lengths[:, None]
                if attention_mask is not None and past_key_values is None:
                    key_padding_mask = attention_mask.to(dtype=torch.bool)

            attn_output = grouped_query_attention(
                query_states,
                key_states,
                value_states,
                scaling=self.scaling,
                causal_mask=causal_mask,
                visible_mask=visible_mask,
                key_padding_mask=key_padding_mask,
            )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.profile_runtime = False
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads must be divisible by linear_num_key_heads, "
                f"got {self.num_v_heads} and {self.num_k_heads}"
            )
        self.recurrent_head_repeat = self.num_v_heads // self.num_k_heads
        self.recurrent_num_heads = self.num_v_heads
        self.recurrent_state_shape = (self.recurrent_num_heads, self.head_k_dim, self.head_v_dim)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.norm = Qwen3RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self._cached_a_log_decay: torch.Tensor | None = None
        self._cached_a_log_decay_key: tuple[int, int, torch.device, torch.dtype] | None = None

    @staticmethod
    def _require_tensor_contract(
        name: str,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        contiguous: bool = False,
    ) -> None:
        if shape is not None and tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        if dtype is not None and tensor.dtype != dtype:
            raise RuntimeError(f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}")
        if device is not None and tensor.device != device:
            raise RuntimeError(f"{name} device mismatch: expected {device}, got {tensor.device}")
        if contiguous and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    def _project_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2).contiguous()
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim).contiguous()
        b = self.in_proj_b(hidden_states).contiguous()
        a = self.in_proj_a(hidden_states).contiguous()
        return mixed_qkv, z, b, a

    def _conv_weight_and_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.conv1d.weight.squeeze(1).contiguous()
        bias = None if self.conv1d.bias is None else self.conv1d.bias.contiguous()
        return weight, bias

    def _a_log_decay(self, device: torch.device) -> torch.Tensor:
        if self.A_log.device != device:
            raise RuntimeError(f"A_log device mismatch: expected {device}, got {self.A_log.device}")
        cache_key = (self.A_log.data_ptr(), self.A_log._version, self.A_log.device, self.A_log.dtype)
        if self._cached_a_log_decay is None or self._cached_a_log_decay_key != cache_key:
            self._cached_a_log_decay = (-self.A_log.float().exp()).contiguous()
            self._cached_a_log_decay_key = cache_key
        self._require_tensor_contract(
            "a_log_decay",
            self._cached_a_log_decay,
            shape=(self.num_v_heads,),
            dtype=torch.float32,
            device=device,
            contiguous=True,
        )
        return self._cached_a_log_decay

    def _prepare_conv_inputs(
        self,
        mixed_qkv: torch.Tensor,
        cache_params: Qwen3DynamicCache | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        conv_state = None if cache_params is None else cache_params.conv_states[self.layer_idx]
        expected_state_shape = (batch_size, self.conv_dim, self.conv_kernel_size)
        if conv_state is None:
            conv_state = mixed_qkv.new_zeros(expected_state_shape)
        else:
            self._require_tensor_contract(
                "conv_state",
                conv_state,
                shape=expected_state_shape,
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
                contiguous=True,
            )
        self._require_tensor_contract("mixed_qkv", mixed_qkv, contiguous=True)
        weight, bias = self._conv_weight_and_bias()
        self._require_tensor_contract(
            "conv_weight",
            weight,
            shape=(self.conv_dim, self.conv_kernel_size),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
            contiguous=True,
        )
        if bias is not None:
            self._require_tensor_contract(
                "conv_bias",
                bias,
                shape=(self.conv_dim,),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
                contiguous=True,
            )
        return mixed_qkv, conv_state, weight, bias

    def _run_conv_prefill(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(mixed_qkv.shape[-1])
        if mixed_qkv.device.type == "xpu" and seq_len > 1:
            return run_causal_conv1d_prefill_fused(
                hidden_states=mixed_qkv,
                conv_state=conv_state,
                weight=weight,
                bias=bias,
            )
        return torch_causal_conv1d_update(mixed_qkv, conv_state, weight, bias), conv_state

    def _run_conv_decode(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mixed_qkv.device.type == "xpu":
            mixed_qkv = run_causal_conv1d_decode_fused(
                hidden_states=mixed_qkv,
                conv_state=conv_state,
                weight=weight,
                bias=bias,
            )
            return mixed_qkv, conv_state
        return torch_causal_conv1d_update(mixed_qkv, conv_state, weight, bias), conv_state

    def _split_qkv(
        self,
        mixed_qkv: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).contiguous()
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim).contiguous()
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim).contiguous()
        return query, key, value

    def _validate_conv_outputs(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._require_tensor_contract(
            "mixed_qkv_after_conv",
            mixed_qkv,
            shape=(batch_size, self.conv_dim, seq_len),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "conv_state_after_conv",
            conv_state,
            shape=(batch_size, self.conv_dim, self.conv_kernel_size),
            dtype=dtype,
            device=device,
            contiguous=True,
        )

    def _prepare_recurrent_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.recurrent_head_repeat > 1:
            query = query.repeat_interleave(self.recurrent_head_repeat, dim=2)
            key = key.repeat_interleave(self.recurrent_head_repeat, dim=2)
        beta = b.sigmoid().to(dtype=torch.float32).contiguous()
        g = (self._a_log_decay(a.device) * F.softplus(a.float() + self.dt_bias)).contiguous()
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        z = z.contiguous()
        return query, key, value, z, beta, g

    def _validate_recurrent_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        z: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._require_tensor_contract(
            "query",
            query,
            shape=(batch_size, seq_len, self.recurrent_num_heads, self.head_k_dim),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "key",
            key,
            shape=(batch_size, seq_len, self.recurrent_num_heads, self.head_k_dim),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "value",
            value,
            shape=(batch_size, seq_len, self.recurrent_num_heads, self.head_v_dim),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "z",
            z,
            shape=(batch_size, seq_len, self.recurrent_num_heads, self.head_v_dim),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "beta",
            beta,
            shape=(batch_size, seq_len, self.recurrent_num_heads),
            dtype=torch.float32,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "g",
            g,
            shape=(batch_size, seq_len, self.recurrent_num_heads),
            dtype=torch.float32,
            device=device,
            contiguous=True,
        )

    def _get_recurrent_state(
        self,
        cache_params: Qwen3DynamicCache | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        recurrent_state = None if cache_params is None else cache_params.recurrent_states[self.layer_idx]
        expected_shape = (batch_size, *self.recurrent_state_shape)
        if recurrent_state is None:
            recurrent_state = torch.zeros(expected_shape, device=device, dtype=torch.float32)
        else:
            self._require_tensor_contract(
                "recurrent_state",
                recurrent_state,
                shape=expected_shape,
                dtype=torch.float32,
                device=device,
                contiguous=True,
            )
        return recurrent_state

    def _validate_recurrent_outputs(
        self,
        core_attn_out: torch.Tensor,
        recurrent_state: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._require_tensor_contract(
            "core_attn_out",
            core_attn_out,
            shape=(batch_size, seq_len, self.recurrent_num_heads, self.head_v_dim),
            dtype=dtype,
            device=device,
            contiguous=True,
        )
        self._require_tensor_contract(
            "recurrent_state_after_gated_delta",
            recurrent_state,
            shape=(batch_size, *self.recurrent_state_shape),
            dtype=torch.float32,
            device=device,
            contiguous=True,
        )

    def _use_precomputed_states(
        self,
        cache_params: Qwen3DynamicCache | None,
        seq_len: int,
    ) -> bool:
        return seq_len == 1 and cache_params is not None and cache_params.has_previous_state

    def _run_chunk_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(query.shape[1])
        if query.device.type == "xpu" and seq_len > 1:
            state = initial_state
            if state is None:
                state = torch.zeros(
                    (query.shape[0], self.recurrent_num_heads, self.head_k_dim, self.head_v_dim),
                    device=query.device,
                    dtype=torch.float32,
                )
            return run_gated_delta_prefill_fused(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                state=state,
            )
        core_attn_out, recurrent_state = torch_chunk_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
        )
        if recurrent_state is None:
            raise RuntimeError("chunk_gated_delta_rule must return the final recurrent state for prefill.")
        return core_attn_out, recurrent_state

    def _run_recurrent_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type == "xpu":
            core_attn_out = run_gated_delta_decode_fused(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                state=recurrent_state,
            )
            return core_attn_out, recurrent_state
        return torch_recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
        )

    def _write_cache(
        self,
        cache_params: Qwen3DynamicCache | None,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        if cache_params is None:
            return
        cache_params.conv_states[self.layer_idx] = conv_state
        cache_params.recurrent_states[self.layer_idx] = recurrent_state

    def _finalize_output(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, self.value_dim).contiguous()
        return self.out_proj(core_attn_out)

    @_compiler_disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = self._use_precomputed_states(cache_params, seq_len)
        mixed_qkv, z_proj, b, a = self._project_inputs(hidden_states)
        mixed_qkv, conv_state, conv_weight, conv_bias = self._prepare_conv_inputs(mixed_qkv, cache_params, batch_size)

        if self.profile_runtime:
            with xpu_profile_region("conv"):
                if use_precomputed_states:
                    mixed_qkv, conv_state = self._run_conv_decode(mixed_qkv, conv_state, conv_weight, conv_bias)
                else:
                    mixed_qkv, conv_state = self._run_conv_prefill(mixed_qkv, conv_state, conv_weight, conv_bias)
        elif use_precomputed_states:
            mixed_qkv, conv_state = self._run_conv_decode(mixed_qkv, conv_state, conv_weight, conv_bias)
        else:
            mixed_qkv, conv_state = self._run_conv_prefill(mixed_qkv, conv_state, conv_weight, conv_bias)
        self._validate_conv_outputs(mixed_qkv, conv_state, batch_size, seq_len, hidden_states.dtype, hidden_states.device)

        def _gated_delta_body() -> torch.Tensor:
            query, key, value = self._split_qkv(mixed_qkv, batch_size, seq_len)
            query, key, value, z, beta, g = self._prepare_recurrent_inputs(query, key, value, z_proj, b, a)
            self._validate_recurrent_inputs(
                query,
                key,
                value,
                z,
                beta,
                g,
                batch_size,
                seq_len,
                hidden_states.dtype,
                hidden_states.device,
            )

            recurrent_state = None
            if cache_params is not None and cache_params.recurrent_states[self.layer_idx] is not None:
                recurrent_state = self._get_recurrent_state(cache_params, batch_size, query.device)
            if use_precomputed_states:
                if recurrent_state is None:
                    recurrent_state = self._get_recurrent_state(cache_params, batch_size, query.device)
                core_attn_out, recurrent_state = self._run_recurrent_decode(query, key, value, g, beta, recurrent_state)
            else:
                core_attn_out, recurrent_state = self._run_chunk_prefill(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                )
            self._validate_recurrent_outputs(
                core_attn_out,
                recurrent_state,
                batch_size,
                seq_len,
                hidden_states.dtype,
                hidden_states.device,
            )

            self._write_cache(cache_params, conv_state, recurrent_state)
            return self._finalize_output(core_attn_out, z, batch_size, seq_len)

        if self.profile_runtime:
            with xpu_profile_region("gated_delta"):
                return _gated_delta_body()
        return _gated_delta_body()


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int | None = None):
        super().__init__()
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self._config = config
        self._expert_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.offload_experts = False
        self.resident_experts = False
        self.execution_device: torch.device | None = None
        self.expert_quant: str = "none"
        self.expert_quant_group_size: int = 128
        self.cached_experts_per_layer: int = max(self.top_k, 8)
        self.profile_runtime = False
        self.host_experts_pinned = False
        self.host_prepacked_experts_pinned = False
        self._expert_cache: OrderedDict[int, Qwen3MLP] = OrderedDict()
        self._host_prepacked_expert_cache: OrderedDict[int, Qwen3MLP] = OrderedDict()
        self._grouped_int4_lru: OrderedDict[int, None] = OrderedDict()
        self._grouped_int4_bank: _GroupedInt4ExpertBank | None = None
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen3MLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool,
        resident_experts: bool = False,
        expert_quant: str = "none",
        expert_quant_group_size: int = 128,
        cached_experts_per_layer: int | None = None,
    ) -> None:
        self.execution_device = execution_device
        self.resident_experts = resident_experts
        self.offload_experts = offload_experts and not resident_experts
        self.expert_quant = expert_quant
        self.expert_quant_group_size = expert_quant_group_size
        if cached_experts_per_layer is not None:
            self.cached_experts_per_layer = max(0, min(int(cached_experts_per_layer), self.num_experts))
        self._expert_cache.clear()
        self._host_prepacked_expert_cache.clear()
        self._grouped_int4_lru.clear()
        self._grouped_int4_bank = None
        self.host_experts_pinned = False
        self.host_prepacked_experts_pinned = False

        if resident_experts and self._should_use_xpu_int4():
            for expert in self.experts:
                convert_module_linears_to_xpu_int4(
                    expert,
                    group_size=self.expert_quant_group_size,
                    device=execution_device,
                )
        else:
            expert_device = execution_device if resident_experts or not offload_experts else torch.device("cpu")
            for expert in self.experts:
                expert.to(expert_device)
                if (
                    self.offload_experts
                    and expert_device.type == "cpu"
                    and not self._has_quantized_linear_payload(expert)
                ):
                    self.host_experts_pinned = self._pin_module_host_memory(expert) or self.host_experts_pinned

    def _should_use_xpu_int4(self) -> bool:
        return (
            self.execution_device is not None
            and self.execution_device.type == "xpu"
            and self.expert_quant == "int4"
        )

    @staticmethod
    def _pin_module_host_memory(module: nn.Module) -> bool:
        pinned_any = False
        try:
            for child in module.modules():
                for name, parameter in list(child._parameters.items()):
                    if parameter is None or parameter.device.type != "cpu":
                        continue
                    child._parameters[name].data = parameter.detach().contiguous().pin_memory()
                    pinned_any = True
                for name, buffer in list(child._buffers.items()):
                    if buffer is None or buffer.device.type != "cpu":
                        continue
                    child._buffers[name] = buffer.detach().contiguous().pin_memory()
                    pinned_any = True
        except RuntimeError:
            logger.debug("Pinned host memory is unavailable for MoE expert staging.", exc_info=True)
            return False
        return pinned_any

    def _new_expert_module(self) -> Qwen3MLP:
        return Qwen3MLP(self._config, intermediate_size=self._expert_intermediate_size)

    def _new_meta_expert_module(self) -> Qwen3MLP:
        with torch.device("meta"):
            return self._new_expert_module()

    @staticmethod
    def _require_grouped_int4_linear(
        module: nn.Module,
        *,
        linear_name: str,
        require_xpu: bool = True,
    ) -> XPUInt4Linear:
        if not isinstance(module, XPUInt4Linear):
            raise RuntimeError(
                f"Grouped int4 expert GEMM requires {linear_name} to be XPUInt4Linear, got {type(module)!r}"
            )
        if module.bias is not None:
            raise RuntimeError(f"Grouped int4 expert GEMM does not support bias on {linear_name}.")
        if require_xpu and module.qweight.device.type != "xpu":
            raise RuntimeError(f"Grouped int4 expert GEMM requires {linear_name} weights on XPU.")
        return module

    def _resolve_grouped_int4_bank_from_linears(
        self,
        *,
        gate_proj: XPUInt4Linear,
        up_proj: XPUInt4Linear,
        down_proj: XPUInt4Linear,
    ) -> _GroupedInt4ExpertBank:
        if self.execution_device is None or self.execution_device.type != "xpu":
            raise RuntimeError("Grouped int4 expert GEMM requires an XPU execution device.")
        if self.cached_experts_per_layer <= 0:
            raise RuntimeError("Grouped int4 expert GEMM requires cached_experts_per_layer > 0.")

        hidden_dim = int(gate_proj.in_features)
        intermediate_dim = int(gate_proj.out_features)
        if up_proj.in_features != hidden_dim or up_proj.out_features != intermediate_dim:
            raise RuntimeError("Grouped int4 expert GEMM requires gate_proj/up_proj to share the same shape.")
        if down_proj.in_features != intermediate_dim or down_proj.out_features != hidden_dim:
            raise RuntimeError("Grouped int4 expert GEMM requires down_proj to invert gate_proj dimensions.")
        if gate_proj.group_size != up_proj.group_size or gate_proj.group_size != down_proj.group_size:
            raise RuntimeError("Grouped int4 expert GEMM requires a shared int4 group size across all expert projections.")

        bank = self._grouped_int4_bank
        if bank is not None:
            bank_device = bank.gate_qweight.device
            execution_device_matches = bank_device.type == self.execution_device.type and (
                self.execution_device.index is None or bank_device.index == self.execution_device.index
            )
            if (
                bank.capacity != self.cached_experts_per_layer
                or bank.hidden_dim != hidden_dim
                or bank.intermediate_dim != intermediate_dim
                or bank.group_size != gate_proj.group_size
                or not execution_device_matches
            ):
                raise RuntimeError(
                    "Grouped int4 expert bank layout does not match the currently prepared MoE experts."
                )
            return bank

        bank = _GroupedInt4ExpertBank(
            capacity=self.cached_experts_per_layer,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            group_size=gate_proj.group_size,
            gate_qweight=torch.empty((self.cached_experts_per_layer, *gate_proj.qweight.shape), dtype=gate_proj.qweight.dtype, device=self.execution_device),
            gate_qscale=torch.empty((self.cached_experts_per_layer, *gate_proj.qscale.shape), dtype=gate_proj.qscale.dtype, device=self.execution_device),
            gate_qzeros=torch.empty((self.cached_experts_per_layer, *gate_proj.qzeros.shape), dtype=gate_proj.qzeros.dtype, device=self.execution_device),
            up_qweight=torch.empty((self.cached_experts_per_layer, *up_proj.qweight.shape), dtype=up_proj.qweight.dtype, device=self.execution_device),
            up_qscale=torch.empty((self.cached_experts_per_layer, *up_proj.qscale.shape), dtype=up_proj.qscale.dtype, device=self.execution_device),
            up_qzeros=torch.empty((self.cached_experts_per_layer, *up_proj.qzeros.shape), dtype=up_proj.qzeros.dtype, device=self.execution_device),
            down_qweight=torch.empty((self.cached_experts_per_layer, *down_proj.qweight.shape), dtype=down_proj.qweight.dtype, device=self.execution_device),
            down_qscale=torch.empty((self.cached_experts_per_layer, *down_proj.qscale.shape), dtype=down_proj.qscale.dtype, device=self.execution_device),
            down_qzeros=torch.empty((self.cached_experts_per_layer, *down_proj.qzeros.shape), dtype=down_proj.qzeros.dtype, device=self.execution_device),
            slot_to_expert=[None] * self.cached_experts_per_layer,
            expert_to_slot={},
        )
        self._grouped_int4_bank = bank
        return bank

    def _resolve_grouped_int4_bank(self, expert_layer: Qwen3MLP) -> _GroupedInt4ExpertBank:
        gate_proj = self._require_grouped_int4_linear(expert_layer.gate_proj, linear_name="gate_proj")
        up_proj = self._require_grouped_int4_linear(expert_layer.up_proj, linear_name="up_proj")
        down_proj = self._require_grouped_int4_linear(expert_layer.down_proj, linear_name="down_proj")
        return self._resolve_grouped_int4_bank_from_linears(
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
        )

    @staticmethod
    def _copy_grouped_int4_linear_to_bank_(
        *,
        linear: XPUInt4Linear,
        qweight_bank: torch.Tensor,
        qscale_bank: torch.Tensor,
        qzeros_bank: torch.Tensor,
        slot: int,
        non_blocking: bool,
    ) -> None:
        with torch.no_grad():
            qweight_bank[slot].copy_(linear.qweight.to(device=qweight_bank.device, non_blocking=non_blocking))
            qscale_bank[slot].copy_(linear.qscale.to(device=qscale_bank.device, non_blocking=non_blocking))
            qzeros_bank[slot].copy_(linear.qzeros.to(device=qzeros_bank.device, non_blocking=non_blocking))

    def _release_grouped_int4_bank_slot(self, expert_idx: int) -> None:
        bank = self._grouped_int4_bank
        if bank is None:
            return
        self._grouped_int4_lru.pop(int(expert_idx), None)
        slot = bank.expert_to_slot.pop(int(expert_idx), None)
        if slot is None:
            return
        if slot < 0 or slot >= bank.capacity:
            raise RuntimeError(f"Grouped int4 expert bank slot {slot} is out of range.")
        mapped_expert = bank.slot_to_expert[slot]
        if mapped_expert != int(expert_idx):
            raise RuntimeError(
                f"Grouped int4 expert bank bookkeeping mismatch for slot {slot}: expected {expert_idx}, found {mapped_expert}."
            )
        bank.slot_to_expert[slot] = None

    def _copy_grouped_int4_expert_to_bank_slot_(
        self,
        *,
        expert_idx: int,
        expert_layer: Qwen3MLP,
        slot: int,
    ) -> None:
        gate_proj = self._require_grouped_int4_linear(expert_layer.gate_proj, linear_name="gate_proj", require_xpu=False)
        up_proj = self._require_grouped_int4_linear(expert_layer.up_proj, linear_name="up_proj", require_xpu=False)
        down_proj = self._require_grouped_int4_linear(expert_layer.down_proj, linear_name="down_proj", require_xpu=False)
        bank = self._resolve_grouped_int4_bank_from_linears(
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
        )
        if slot < 0 or slot >= bank.capacity:
            raise RuntimeError(f"Grouped int4 expert bank slot {slot} is out of range.")
        previous_expert = bank.slot_to_expert[slot]
        if previous_expert is not None:
            bank.expert_to_slot.pop(previous_expert, None)
            self._grouped_int4_lru.pop(previous_expert, None)

        self._copy_grouped_int4_linear_to_bank_(
            linear=gate_proj,
            qweight_bank=bank.gate_qweight,
            qscale_bank=bank.gate_qscale,
            qzeros_bank=bank.gate_qzeros,
            slot=slot,
            non_blocking=self.host_prepacked_experts_pinned,
        )
        self._copy_grouped_int4_linear_to_bank_(
            linear=up_proj,
            qweight_bank=bank.up_qweight,
            qscale_bank=bank.up_qscale,
            qzeros_bank=bank.up_qzeros,
            slot=slot,
            non_blocking=self.host_prepacked_experts_pinned,
        )
        self._copy_grouped_int4_linear_to_bank_(
            linear=down_proj,
            qweight_bank=bank.down_qweight,
            qscale_bank=bank.down_qscale,
            qzeros_bank=bank.down_qzeros,
            slot=slot,
            non_blocking=self.host_prepacked_experts_pinned,
        )
        bank.slot_to_expert[slot] = int(expert_idx)
        bank.expert_to_slot[int(expert_idx)] = slot
        self._grouped_int4_lru[int(expert_idx)] = None

    def _ensure_grouped_int4_bank_slot(self, expert_idx: int, expert_layer: Qwen3MLP) -> int:
        bank = self._resolve_grouped_int4_bank(expert_layer)
        existing_slot = bank.expert_to_slot.get(int(expert_idx))
        if existing_slot is not None:
            mapped_expert = bank.slot_to_expert[existing_slot]
            if mapped_expert != int(expert_idx):
                raise RuntimeError(
                    f"Grouped int4 expert bank bookkeeping mismatch for slot {existing_slot}: expected {expert_idx}, found {mapped_expert}."
                )
            self._grouped_int4_lru.move_to_end(int(expert_idx))
            return existing_slot

        try:
            slot = next(slot_idx for slot_idx, mapped_expert in enumerate(bank.slot_to_expert) if mapped_expert is None)
        except StopIteration as exc:
            raise RuntimeError("Grouped int4 expert bank is full while staging a new cached expert.") from exc

        self._copy_grouped_int4_expert_to_bank_slot_(expert_idx=expert_idx, expert_layer=expert_layer, slot=slot)
        return slot

    def _ensure_grouped_int4_bank_slots(
        self,
        *,
        prepared: dict[int, Qwen3MLP],
        expert_indices: list[int],
    ) -> list[int]:
        slots: list[int] = []
        for expert_idx in expert_indices:
            expert_layer = prepared.get(expert_idx)
            if expert_layer is None:
                raise RuntimeError(f"Grouped int4 expert GEMM expected expert {expert_idx} to be prepared on XPU.")
            slots.append(self._ensure_grouped_int4_bank_slot(expert_idx, expert_layer))
        return slots

    def _prepare_grouped_int4_bank_slots(self, required_expert_indices: list[int]) -> list[int]:
        if self.execution_device is None or self.execution_device.type != "xpu":
            raise RuntimeError("Grouped int4 expert bank preparation requires an XPU execution device.")
        if self.cached_experts_per_layer <= 0:
            raise RuntimeError("Grouped int4 expert bank preparation requires cached_experts_per_layer > 0.")
        if len(required_expert_indices) > self.cached_experts_per_layer:
            raise RuntimeError(
                "Grouped int4 expert bank preparation requires cache capacity >= active expert wave size: "
                f"required={len(required_expert_indices)} capacity={self.cached_experts_per_layer}"
            )

        started_at = time.perf_counter()
        protected = {int(expert_idx) for expert_idx in required_expert_indices}
        bank = self._grouped_int4_bank
        slots_by_expert: dict[int, int] = {}
        cache_hits = 0
        staged = 0
        evicted = 0

        if bank is not None:
            for expert_idx in required_expert_indices:
                slot = bank.expert_to_slot.get(int(expert_idx))
                if slot is None:
                    continue
                if bank.slot_to_expert[slot] != int(expert_idx):
                    raise RuntimeError(
                        f"Grouped int4 expert bank bookkeeping mismatch for slot {slot}: "
                        f"expected {expert_idx}, found {bank.slot_to_expert[slot]}."
                    )
                self._grouped_int4_lru.move_to_end(int(expert_idx))
                slots_by_expert[int(expert_idx)] = slot
                cache_hits += 1

        missing = [int(expert_idx) for expert_idx in required_expert_indices if int(expert_idx) not in slots_by_expert]
        for expert_idx in missing:
            prepacked = self._get_host_prepacked_expert(expert_idx)
            if prepacked is None:
                raise RuntimeError("Grouped int4 expert bank preparation requires a host prepacked expert.")
            gate_proj = self._require_grouped_int4_linear(prepacked.gate_proj, linear_name="gate_proj", require_xpu=False)
            up_proj = self._require_grouped_int4_linear(prepacked.up_proj, linear_name="up_proj", require_xpu=False)
            down_proj = self._require_grouped_int4_linear(prepacked.down_proj, linear_name="down_proj", require_xpu=False)
            bank = self._resolve_grouped_int4_bank_from_linears(
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
            )

            free_slot = next((idx for idx, mapped in enumerate(bank.slot_to_expert) if mapped is None), None)
            if free_slot is None:
                victim_expert = next(
                    (candidate for candidate in self._grouped_int4_lru.keys() if candidate not in protected),
                    None,
                )
                if victim_expert is None:
                    raise RuntimeError(
                        "Grouped int4 expert bank has no evictable slot for the requested expert wave."
                    )
                free_slot = bank.expert_to_slot.get(victim_expert)
                if free_slot is None:
                    raise RuntimeError(
                        f"Grouped int4 expert bank LRU referenced expert {victim_expert} without a slot."
                    )
                self._release_grouped_int4_bank_slot(victim_expert)
                evicted += 1

            self._copy_grouped_int4_expert_to_bank_slot_(
                expert_idx=expert_idx,
                expert_layer=prepacked,
                slot=free_slot,
            )
            slots_by_expert[expert_idx] = free_slot
            staged += 1

        if self.profile_runtime and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
            logger.info(
                "MoE grouped int4 bank prepared: requested=%s hits=%s staged=%s evicted=%s cache_size=%s capacity=%s elapsed=%.4f",
                len(required_expert_indices),
                cache_hits,
                staged,
                evicted,
                0 if self._grouped_int4_bank is None else len(self._grouped_int4_bank.expert_to_slot),
                self.cached_experts_per_layer,
                time.perf_counter() - started_at,
            )

        return [slots_by_expert[int(expert_idx)] for expert_idx in required_expert_indices]

    @staticmethod
    def _has_quantized_linear_payload(module: nn.Module) -> bool:
        return any(isinstance(child, (AutoRoundGPTQLinear, AWQLinear)) for child in module.modules())

    def _host_prepacked_cache_capacity(self) -> int:
        if not self.offload_experts or not self._should_use_xpu_int4() or self.cached_experts_per_layer <= 0:
            return 0
        # Keep CPU-side prepacked experts bounded by the XPU bank capacity. The
        # original AutoRound tensors already live in RAM, so this cache must not
        # grow to another near-complete expert copy on 32GB systems.
        return min(self.num_experts, max(self.cached_experts_per_layer, self.top_k * 4))

    def _build_host_prepacked_expert(self, expert_idx: int) -> Qwen3MLP:
        source = self.experts[expert_idx]
        prepacked = self._new_meta_expert_module()
        for linear_name in ("gate_proj", "up_proj", "down_proj"):
            source_linear = getattr(source, linear_name)
            if isinstance(source_linear, XPUInt4Linear):
                packed_linear = XPUInt4Linear(
                    source_linear.in_features,
                    source_linear.out_features,
                    group_size=source_linear.group_size,
                    bias=source_linear.bias is not None,
                    compute_dtype=source_linear.compute_dtype,
                    device=torch.device("cpu"),
                    padded_in_features=source_linear.padded_in_features,
                )
                with torch.no_grad():
                    packed_linear.qweight.copy_(source_linear.qweight.to(device=packed_linear.qweight.device))
                    packed_linear.qscale.copy_(source_linear.qscale.to(device=packed_linear.qscale.device))
                    packed_linear.qzeros.copy_(source_linear.qzeros.to(device=packed_linear.qzeros.device))
                    if source_linear.bias is not None and packed_linear.bias is not None:
                        packed_linear.bias.copy_(
                            source_linear.bias.to(
                                device=packed_linear.bias.device,
                                dtype=packed_linear.bias.dtype,
                            )
                        )
            else:
                packed_linear = XPUInt4Linear.from_linear(source_linear, device=torch.device("cpu"))
            setattr(prepacked, linear_name, packed_linear)
        self.host_prepacked_experts_pinned = self._pin_module_host_memory(prepacked) or self.host_prepacked_experts_pinned
        return prepacked

    def _get_host_prepacked_expert(self, expert_idx: int) -> Qwen3MLP | None:
        capacity = self._host_prepacked_cache_capacity()
        if capacity <= 0:
            return None
        cached = self._host_prepacked_expert_cache.get(expert_idx)
        if cached is not None:
            self._host_prepacked_expert_cache.move_to_end(expert_idx)
            return cached

        if len(self._host_prepacked_expert_cache) >= capacity:
            self._host_prepacked_expert_cache.popitem(last=False)
        cached = self._build_host_prepacked_expert(expert_idx)
        self._host_prepacked_expert_cache[expert_idx] = cached
        return cached

    def _clone_prepacked_linear_to_execution_device(self, source: XPUInt4Linear) -> XPUInt4Linear:
        if self.execution_device is None:
            raise RuntimeError("Cannot clone a prepacked expert linear without an execution device.")
        cloned = XPUInt4Linear(
            source.in_features,
            source.out_features,
            group_size=source.group_size,
            bias=source.bias is not None,
            compute_dtype=source.compute_dtype,
            device=self.execution_device,
            padded_in_features=source.padded_in_features,
        )
        with torch.no_grad():
            cloned.qweight.copy_(
                source.qweight.to(device=cloned.qweight.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            cloned.qscale.copy_(
                source.qscale.to(device=cloned.qscale.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            cloned.qzeros.copy_(
                source.qzeros.to(device=cloned.qzeros.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            if source.bias is not None and cloned.bias is not None:
                cloned.bias.copy_(
                    source.bias.to(
                        device=cloned.bias.device,
                        dtype=cloned.bias.dtype,
                        non_blocking=self.host_prepacked_experts_pinned,
                    )
                )
        return cloned

    def _copy_prepacked_linear_to_execution_device_(self, source: XPUInt4Linear, target: XPUInt4Linear) -> None:
        if (
            target.in_features != source.in_features
            or target.out_features != source.out_features
            or target.group_size != source.group_size
            or target.padded_in_features != source.padded_in_features
        ):
            raise RuntimeError(
                "Prepacked expert linear layout mismatch during XPU cache slot reuse: "
                f"source=({source.in_features}, {source.out_features}, group={source.group_size}, padded={source.padded_in_features}) "
                f"target=({target.in_features}, {target.out_features}, group={target.group_size}, padded={target.padded_in_features})"
            )
        with torch.no_grad():
            target.qweight.copy_(
                source.qweight.to(device=target.qweight.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            target.qscale.copy_(
                source.qscale.to(device=target.qscale.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            target.qzeros.copy_(
                source.qzeros.to(device=target.qzeros.device, non_blocking=self.host_prepacked_experts_pinned)
            )
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(
                    source.bias.to(
                        device=target.bias.device,
                        dtype=target.bias.dtype,
                        non_blocking=self.host_prepacked_experts_pinned,
                    )
                )

    def _materialize_prepacked_expert(
        self,
        expert_idx: int,
        prepacked: Qwen3MLP,
        *,
        reuse_target: Qwen3MLP | None = None,
    ) -> Qwen3MLP:
        cached = self._new_meta_expert_module() if reuse_target is None else reuse_target
        for linear_name in ("gate_proj", "up_proj", "down_proj"):
            source_linear = getattr(prepacked, linear_name)
            if not isinstance(source_linear, XPUInt4Linear):
                raise RuntimeError(
                    f"Expected prepacked expert linear {linear_name} to be XPUInt4Linear, got {type(source_linear)!r}"
                )
            if reuse_target is None:
                setattr(cached, linear_name, self._clone_prepacked_linear_to_execution_device(source_linear))
                continue
            target_linear = getattr(cached, linear_name)
            if not isinstance(target_linear, XPUInt4Linear):
                raise RuntimeError(
                    f"Expected cached expert linear {linear_name} to be XPUInt4Linear during slot reuse, got {type(target_linear)!r}"
                )
            self._copy_prepacked_linear_to_execution_device_(source_linear, target_linear)
        return cached

    @staticmethod
    def _copy_module_state_(
        source: nn.Module,
        target: nn.Module,
        *,
        non_blocking: bool,
    ) -> None:
        with torch.no_grad():
            target_parameters = dict(target.named_parameters())
            for name, source_parameter in source.named_parameters():
                target_parameter = target_parameters[name]
                copied = source_parameter.to(
                    device=target_parameter.device,
                    dtype=target_parameter.dtype,
                    non_blocking=non_blocking,
                )
                target_parameter.copy_(copied)

            target_buffers = dict(target.named_buffers())
            for name, source_buffer in source.named_buffers():
                target_buffer = target_buffers[name]
                copied = source_buffer.to(
                    device=target_buffer.device,
                    dtype=target_buffer.dtype if source_buffer.is_floating_point() else source_buffer.dtype,
                    non_blocking=non_blocking,
                )
                target_buffer.copy_(copied)

    def _materialize_cached_expert(self, expert_idx: int, *, reuse_target: Qwen3MLP | None = None) -> Qwen3MLP:
        if self.execution_device is None:
            raise RuntimeError("Cannot materialize cached expert without an execution device.")

        source = self.experts[expert_idx]
        started_at = time.perf_counter()
        if self._should_use_xpu_int4():
            prepacked = self._get_host_prepacked_expert(expert_idx)
            if prepacked is None:
                raise RuntimeError("XPU int4 expert staging requires a host prepacked expert cache entry.")
            cached = self._materialize_prepacked_expert(expert_idx, prepacked, reuse_target=reuse_target)
        elif self._has_quantized_linear_payload(source):
            if reuse_target is None:
                cached = copy.deepcopy(source)
                cached = cached.to(
                    device=self.execution_device,
                    dtype=_module_dtype(source),
                )
            else:
                cached = reuse_target
                self._copy_module_state_(source, cached, non_blocking=self.host_experts_pinned)
        else:
            cached = reuse_target
            if cached is None:
                cached = self._new_expert_module().to(
                    device=self.execution_device,
                    dtype=_module_dtype(source),
                )
            self._copy_module_state_(source, cached, non_blocking=self.host_experts_pinned)

        if self.profile_runtime and self.execution_device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
            logger.info(
                "MoE expert staged to XPU: expert=%s quant=%s pinned_host=%s pinned_prepacked=%s reused_slot=%s elapsed=%.4f",
                expert_idx,
                self.expert_quant,
                self.host_experts_pinned,
                self.host_prepacked_experts_pinned,
                reuse_target is not None,
                time.perf_counter() - started_at,
            )
        return cached

    def _get_cached_expert(self, expert_idx: int) -> Qwen3MLP | None:
        if not self.offload_experts or self.execution_device is None or self.cached_experts_per_layer <= 0:
            return None
        cached = self._expert_cache.get(expert_idx)
        if cached is not None:
            self._expert_cache.move_to_end(expert_idx)
            return cached

        reuse_target = None
        if len(self._expert_cache) >= self.cached_experts_per_layer:
            evicted_expert_idx, reuse_target = self._expert_cache.popitem(last=False)
            self._release_grouped_int4_bank_slot(evicted_expert_idx)
        cached = self._materialize_cached_expert(expert_idx, reuse_target=reuse_target)
        self._expert_cache[expert_idx] = cached
        return cached

    def _prepare_cached_experts(self, required_expert_indices: list[int]) -> dict[int, Qwen3MLP]:
        if (
            not self.offload_experts
            or self.execution_device is None
            or self.cached_experts_per_layer <= 0
            or not required_expert_indices
        ):
            return {}

        started_at = time.perf_counter()
        prepared: dict[int, Qwen3MLP] = {}
        protected = {int(expert_idx) for expert_idx in required_expert_indices}
        cache_hits = 0
        staged = 0
        for expert_idx in required_expert_indices:
            cached = self._expert_cache.get(expert_idx)
            if cached is None:
                continue
            self._expert_cache.move_to_end(expert_idx)
            prepared[expert_idx] = cached
            cache_hits += 1

        missing = [expert_idx for expert_idx in required_expert_indices if expert_idx not in prepared]
        if missing:
            free_slots = max(0, self.cached_experts_per_layer - len(self._expert_cache))
            evictions_needed = max(0, len(missing) - free_slots)
            recycled_slots: list[Qwen3MLP] = []
            if evictions_needed > 0:
                eviction_candidates = [expert_idx for expert_idx in self._expert_cache.keys() if expert_idx not in protected]
                if len(eviction_candidates) < evictions_needed:
                    evictions_needed = len(eviction_candidates)
                for expert_idx in eviction_candidates[:evictions_needed]:
                    evicted = self._expert_cache.pop(expert_idx, None)
                    if evicted is not None:
                        self._release_grouped_int4_bank_slot(expert_idx)
                        recycled_slots.append(evicted)

            for expert_idx in missing:
                reuse_target = recycled_slots.pop() if recycled_slots else None
                if len(self._expert_cache) >= self.cached_experts_per_layer:
                    if reuse_target is None:
                        evicted_expert_idx, reuse_target = self._expert_cache.popitem(last=False)
                        self._release_grouped_int4_bank_slot(evicted_expert_idx)
                    else:
                        raise RuntimeError(
                            "MoE expert cache bookkeeping is inconsistent: a recycled XPU slot was available "
                            "while the cache still reported itself as full."
                        )
                cached = self._materialize_cached_expert(expert_idx, reuse_target=reuse_target)
                self._expert_cache[expert_idx] = cached
                prepared[expert_idx] = cached
                staged += 1

        if self.profile_runtime and self.execution_device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
            logger.info(
                "MoE expert cache prepared: requested=%s hits=%s staged=%s unresolved=%s cache_size=%s capacity=%s elapsed=%.4f",
                len(required_expert_indices),
                cache_hits,
                staged,
                max(0, len(required_expert_indices) - len(prepared)),
                len(self._expert_cache),
                self.cached_experts_per_layer,
                time.perf_counter() - started_at,
            )
        return prepared

    def _route_tokens(
        self,
        router_logits: torch.Tensor,
        *,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if router_logits.device.type == "xpu":
            routing_weights, selected_experts, usage = run_moe_router_fused(
                router_logits=router_logits,
                top_k=self.top_k,
                normalize_topk_prob=self.norm_topk_prob,
            )
        else:
            routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            usage = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
        return routing_weights.to(dtype=hidden_dtype), selected_experts, usage

    def _sorted_routing_assignments(
        self,
        selected_experts: torch.Tensor,
        usage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = selected_experts.shape[0]
        flat_selected_experts = selected_experts.reshape(-1)
        flat_token_idx = torch.arange(num_tokens, device=selected_experts.device, dtype=torch.long).repeat_interleave(
            self.top_k
        )
        flat_route_idx = torch.arange(self.top_k, device=selected_experts.device, dtype=torch.long).repeat(num_tokens)
        sort_key = (
            flat_selected_experts.to(dtype=torch.long) * (num_tokens * self.top_k)
            + flat_token_idx * self.top_k
            + flat_route_idx
        )
        sorted_order = torch.argsort(sort_key)
        sorted_token_idx = flat_token_idx.index_select(0, sorted_order)
        sorted_route_idx = flat_route_idx.index_select(0, sorted_order)
        expert_offsets = torch.zeros(self.num_experts + 1, device=selected_experts.device, dtype=torch.long)
        expert_offsets[1:] = usage.to(dtype=torch.long).cumsum(dim=0)
        return sorted_token_idx, sorted_route_idx, expert_offsets

    @staticmethod
    def _compact_routing_inputs(
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        sorted_token_idx: torch.Tensor,
        sorted_route_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compact_hidden_states = hidden_states.index_select(0, sorted_token_idx)
        compact_routing_weights = routing_weights.index_select(0, sorted_token_idx).gather(1, sorted_route_idx[:, None])
        return compact_hidden_states, compact_routing_weights

    def _dispatch_routing_inputs(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        usage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.device.type == "xpu":
            return run_moe_dispatch_fused(
                hidden_states=hidden_states,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                expert_usage=usage,
            )

        sorted_token_idx, sorted_route_idx, expert_offsets = self._sorted_routing_assignments(selected_experts, usage)
        compact_hidden_states, compact_routing_weights = self._compact_routing_inputs(
            hidden_states,
            routing_weights,
            sorted_token_idx,
            sorted_route_idx,
        )
        return sorted_token_idx, expert_offsets, compact_hidden_states, compact_routing_weights

    @staticmethod
    def _scatter_compact_outputs(
        compact_outputs: torch.Tensor,
        sorted_token_idx: torch.Tensor,
        *,
        num_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        if compact_outputs.device.type == "xpu":
            return run_moe_scatter_fused(
                compact_outputs=compact_outputs,
                sorted_token_idx=sorted_token_idx,
                num_tokens=num_tokens,
            )

        final_hidden_states = compact_outputs.new_zeros((num_tokens, hidden_dim))
        if compact_outputs.numel() > 0:
            final_hidden_states.index_add_(0, sorted_token_idx, compact_outputs)
        return final_hidden_states

    def _execute_expert(
        self,
        *,
        expert_layer: Qwen3MLP,
        start_idx: int,
        end_idx: int,
        compact_hidden_states: torch.Tensor,
        compact_routing_weights: torch.Tensor,
        compact_outputs: torch.Tensor,
        execution_device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> None:
        if end_idx <= start_idx:
            return

        expert_device = _module_device(expert_layer)
        current_state = compact_hidden_states[start_idx:end_idx]
        current_routing_weights = compact_routing_weights[start_idx:end_idx]
        if current_state.device != expert_device:
            current_state = current_state.to(device=expert_device)
        if current_routing_weights.device != expert_device:
            current_routing_weights = current_routing_weights.to(device=expert_device)
        current_hidden_states = expert_layer(current_state) * current_routing_weights
        if current_hidden_states.device != execution_device:
            current_hidden_states = current_hidden_states.to(device=execution_device, dtype=hidden_dtype)
        compact_outputs[start_idx:end_idx] = current_hidden_states.to(dtype=hidden_dtype)

    def _execute_grouped_int4_experts(
        self,
        *,
        wave_indices: list[int],
        wave_slots: list[int],
        offsets_np: np.ndarray,
        expert_offsets_device: torch.Tensor,
        compact_hidden_states: torch.Tensor,
        compact_routing_weights: torch.Tensor,
        compact_outputs: torch.Tensor,
    ) -> None:
        bank = self._grouped_int4_bank
        if bank is None:
            raise RuntimeError("Grouped int4 expert GEMM requires an initialized expert bank.")
        if len(wave_indices) != len(wave_slots):
            raise RuntimeError("Grouped int4 expert GEMM requires wave_indices and wave_slots to have the same length.")

        max_routes_per_expert = 0
        for expert_idx, slot in zip(wave_indices, wave_slots, strict=True):
            if expert_idx < 0 or expert_idx + 1 >= int(expert_offsets_device.numel()):
                raise RuntimeError(f"Grouped int4 expert GEMM received an out-of-range expert index: {expert_idx}")
            mapped_slot = bank.expert_to_slot.get(int(expert_idx))
            if mapped_slot != int(slot):
                raise RuntimeError(
                    f"Grouped int4 expert GEMM slot mapping mismatch for expert {expert_idx}: expected {mapped_slot}, got {slot}."
                )
            start_idx = int(offsets_np[expert_idx])
            end_idx = int(offsets_np[expert_idx + 1])
            max_routes_per_expert = max(max_routes_per_expert, end_idx - start_idx)
        if max_routes_per_expert <= 0:
            raise RuntimeError("Grouped int4 expert GEMM received an empty expert wave.")

        active_experts = torch.tensor(wave_indices, device=compact_hidden_states.device, dtype=torch.long)
        active_slots = torch.tensor(wave_slots, device=compact_hidden_states.device, dtype=torch.long)
        run_moe_grouped_int4_mlp_fused(
            compact_hidden_states=compact_hidden_states,
            compact_routing_weights=compact_routing_weights,
            compact_outputs=compact_outputs,
            expert_offsets=expert_offsets_device,
            active_experts=active_experts,
            active_slots=active_slots,
            gate_qweight=bank.gate_qweight,
            gate_qscale=bank.gate_qscale,
            gate_qzeros=bank.gate_qzeros,
            up_qweight=bank.up_qweight,
            up_qscale=bank.up_qscale,
            up_qzeros=bank.up_qzeros,
            down_qweight=bank.down_qweight,
            down_qscale=bank.down_qscale,
            down_qzeros=bank.down_qzeros,
            group_size=bank.group_size,
            max_routes_per_expert=max_routes_per_expert,
        )

    def _execute_offloaded_experts(
        self,
        *,
        hit_experts: list[int],
        expert_offsets: torch.Tensor,
        expert_offsets_host: torch.Tensor,
        compact_hidden_states: torch.Tensor,
        compact_routing_weights: torch.Tensor,
        compact_outputs: torch.Tensor,
        execution_device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> None:
        if self.execution_device is None:
            raise RuntimeError("Offloaded MoE execution requires an execution device.")
        if self.cached_experts_per_layer <= 0:
            raise RuntimeError("Offloaded MoE execution requires cached_experts_per_layer > 0.")

        wave_capacity = min(self.cached_experts_per_layer, len(hit_experts))
        if wave_capacity <= 0:
            return

        offsets_np = expert_offsets_host.detach().numpy()
        for wave_start in range(0, len(hit_experts), wave_capacity):
            wave_indices = hit_experts[wave_start : wave_start + wave_capacity]
            if self._should_use_xpu_int4():
                wave_slots = self._prepare_grouped_int4_bank_slots(wave_indices)
                self._execute_grouped_int4_experts(
                    wave_indices=wave_indices,
                    wave_slots=wave_slots,
                    offsets_np=offsets_np,
                    expert_offsets_device=expert_offsets,
                    compact_hidden_states=compact_hidden_states,
                    compact_routing_weights=compact_routing_weights,
                    compact_outputs=compact_outputs,
                )
                continue

            prepared = self._prepare_cached_experts(wave_indices)
            missing = [expert_idx for expert_idx in wave_indices if expert_idx not in prepared]
            if missing:
                raise RuntimeError(
                    "Failed to stage all requested MoE experts onto the execution device: "
                    f"missing={missing} cache_capacity={self.cached_experts_per_layer}"
                )
            for expert_idx in wave_indices:
                start_idx = int(offsets_np[expert_idx])
                end_idx = int(offsets_np[expert_idx + 1])
                self._execute_expert(
                    expert_layer=prepared[expert_idx],
                    start_idx=start_idx,
                    end_idx=end_idx,
                    compact_hidden_states=compact_hidden_states,
                    compact_routing_weights=compact_routing_weights,
                    compact_outputs=compact_outputs,
                    execution_device=execution_device,
                    hidden_dtype=hidden_dtype,
                )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        execution_device = hidden_states.device
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        def _moe_body() -> tuple[torch.Tensor, torch.Tensor]:
            routing_weights, selected_experts, usage = self._route_tokens(router_logits, hidden_dtype=hidden_states.dtype)

            num_tokens = batch_size * sequence_length
            usage = usage.to(device=selected_experts.device)
            hit_experts = usage.nonzero(as_tuple=False).flatten()
            hit_expert_list = [int(expert_idx) for expert_idx in hit_experts.tolist()]

            sorted_token_idx, expert_offsets, compact_hidden_states, compact_routing_weights = self._dispatch_routing_inputs(
                hidden_states,
                routing_weights,
                selected_experts,
                usage,
            )
            compact_outputs = hidden_states.new_empty((compact_hidden_states.shape[0], hidden_dim))
            expert_offsets_host = expert_offsets.to(device="cpu") if expert_offsets.device.type == "xpu" else expert_offsets
            if self.offload_experts and hit_expert_list:
                self._execute_offloaded_experts(
                    hit_experts=hit_expert_list,
                    expert_offsets=expert_offsets,
                    expert_offsets_host=expert_offsets_host,
                    compact_hidden_states=compact_hidden_states,
                    compact_routing_weights=compact_routing_weights,
                    compact_outputs=compact_outputs,
                    execution_device=execution_device,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                offsets_np = expert_offsets_host.detach().numpy()
                for expert_idx in hit_expert_list:
                    start_idx = int(offsets_np[expert_idx])
                    end_idx = int(offsets_np[expert_idx + 1])
                    self._execute_expert(
                        expert_layer=self.experts[expert_idx],
                        start_idx=start_idx,
                        end_idx=end_idx,
                        compact_hidden_states=compact_hidden_states,
                        compact_routing_weights=compact_routing_weights,
                        compact_outputs=compact_outputs,
                        execution_device=execution_device,
                        hidden_dtype=hidden_states.dtype,
                    )

            final_hidden_states = self._scatter_compact_outputs(
                compact_outputs,
                sorted_token_idx,
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
            )

            shared_expert_device = _module_device(self.shared_expert)
            shared_gate_device = _module_device(self.shared_expert_gate)
            shared_input = hidden_states if hidden_states.device == shared_expert_device else hidden_states.to(device=shared_expert_device)
            gate_input = hidden_states if hidden_states.device == shared_gate_device else hidden_states.to(device=shared_gate_device)
            shared_expert_output = self.shared_expert(shared_input)
            shared_expert_output = torch.sigmoid(self.shared_expert_gate(gate_input).to(device=shared_expert_output.device)) * shared_expert_output
            if shared_expert_output.device != execution_device:
                shared_expert_output = shared_expert_output.to(device=execution_device, dtype=hidden_states.dtype)
            final_hidden_states = final_hidden_states + shared_expert_output
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

        if self.profile_runtime:
            with xpu_profile_region("moe"):
                return _moe_body()
        return _moe_body()


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")
        self.mlp = Qwen3SparseMoeBlock(config) if config.uses_sparse_moe(layer_idx) else Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states
