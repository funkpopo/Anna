from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.sampling.params import SamplingBatchParams
from anna.runtime.paged_kv import PagedKVManager, SlotCapacityError
from anna.runtime.slot_scheduler import DecodeBatchPlan, SequenceSlot, SlotScheduler, SlotState


@dataclass(frozen=True, slots=True)
class SlotModelRunnerConfig:
    max_slots: int
    total_blocks: int
    block_size: int
    max_blocks_per_seq: int
    max_batch_size: int
    device: torch.device
    num_layers: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0


@dataclass(frozen=True, slots=True)
class SlotKVPageBank:
    """Optional internal physical KV page storage for slot-owned KV writes.

    The slot metadata manager assigns block ids in ``[0, total_blocks)``. When
    this bank is explicitly allocated, those ids are also valid physical page
    indices into the tensors below. The bank is not allocated by default; even
    when it is allocated, decode still waits for the production forward path to
    read the same slot-owned pages safely.
    """

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    num_layers: int
    total_blocks: int
    num_key_value_heads: int
    block_size: int
    head_dim: int

    @property
    def device(self) -> torch.device:
        return self.key_pages.device

    @property
    def dtype(self) -> torch.dtype:
        return self.key_pages.dtype

    def layer_pages(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx must be in range [0, {self.num_layers}).")
        return self.key_pages[layer_idx], self.value_pages[layer_idx]

    def health(self) -> dict[str, object]:
        return {
            "allocated": True,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_layers": int(self.num_layers),
            "total_blocks": int(self.total_blocks),
            "num_key_value_heads": int(self.num_key_value_heads),
            "block_size": int(self.block_size),
            "head_dim": int(self.head_dim),
            "key_pages_shape": tuple(int(dim) for dim in self.key_pages.shape),
            "value_pages_shape": tuple(int(dim) for dim in self.value_pages.shape),
        }


@dataclass(frozen=True, slots=True)
class SlotDecodeModelInputs:
    """Tensor view passed from a slot scheduler to the future model runner.

    ``seq_lens`` and ``positions`` describe lengths before the current
    ``input_ids`` are appended. Attention paths that read KV after writing the
    current decode tokens should use ``visible_seq_lens``.

    Internal Anna slot decode plans pass ``seq_lens``, ``positions``, and
    ``block_tables`` as live global slot tensors and set the corresponding
    ``*_are_global`` flags so model code can select active rows with ``slot_ids``
    without rebuilding per-batch metadata.
    """

    request_ids: tuple[str, ...]
    input_ids: torch.Tensor
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    positions: torch.Tensor
    positions_are_global: bool
    seq_lens: torch.Tensor
    seq_lens_are_global: bool
    block_tables: torch.Tensor
    block_tables_are_global: bool
    physical_block_tables: bool
    sampling_params: tuple[Any | None, ...]
    sampling_batch_params: SamplingBatchParams
    physical_key_pages: torch.Tensor | None = None
    physical_value_pages: torch.Tensor | None = None

    @property
    def decode_token_count(self) -> int:
        if self.input_ids.ndim <= 1:
            return 1
        return int(self.input_ids.shape[1])

    @property
    def batch_size(self) -> int:
        if self.input_ids.ndim == 0:
            return 1
        return int(self.input_ids.shape[0])

    @property
    def contains_cache_objects(self) -> bool:
        return False

    @property
    def block_table_ownership(self) -> str:
        return "physical" if self.physical_block_tables else "logical_slot_metadata"

    @property
    def owns_physical_kv_pages(self) -> bool:
        return self.physical_key_pages is not None and self.physical_value_pages is not None

    @property
    def physical_kv_layer_count(self) -> int:
        if self.physical_key_pages is None:
            return 0
        return int(self.physical_key_pages.shape[0])

    def physical_pages_for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.physical_key_pages is None or self.physical_value_pages is None:
            raise RuntimeError("Slot decode inputs do not own physical KV pages.")
        if layer_idx < 0 or layer_idx >= int(self.physical_key_pages.shape[0]):
            raise IndexError(f"layer_idx must be in range [0, {int(self.physical_key_pages.shape[0])}).")
        return self.physical_key_pages[layer_idx], self.physical_value_pages[layer_idx]

    def boundary_summary(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "decode_token_count": self.decode_token_count,
            "contains_cache_objects": self.contains_cache_objects,
            "input_ids_device": str(self.input_ids.device),
            "slot_ids_device": str(self.slot_ids.device),
            "positions_are_global": bool(self.positions_are_global),
            "seq_lens_are_global": bool(self.seq_lens_are_global),
            "block_tables_are_global": bool(self.block_tables_are_global),
            "physical_block_tables": bool(self.physical_block_tables),
            "block_table_ownership": self.block_table_ownership,
            "block_tables_shape": tuple(int(dim) for dim in self.block_tables.shape),
            "seq_lens_shape": tuple(int(dim) for dim in self.seq_lens.shape),
            "positions_shape": tuple(int(dim) for dim in self.positions.shape),
            "sampling_batch_params": self.sampling_batch_params.batch_size == self.batch_size,
            "owns_physical_kv_pages": self.owns_physical_kv_pages,
            "physical_kv_layer_count": self.physical_kv_layer_count,
            "physical_key_pages_shape": None
            if self.physical_key_pages is None
            else tuple(int(dim) for dim in self.physical_key_pages.shape),
            "physical_value_pages_shape": None
            if self.physical_value_pages is None
            else tuple(int(dim) for dim in self.physical_value_pages.shape),
        }

    def _select_batch_rows(self, tensor: torch.Tensor, *, is_global: bool) -> torch.Tensor:
        if not is_global:
            return tensor
        index = self.slot_ids.to(device=tensor.device, dtype=torch.long)
        return tensor.index_select(0, index)

    @property
    def batch_positions(self) -> torch.Tensor:
        return self._select_batch_rows(self.positions, is_global=self.positions_are_global)

    @property
    def batch_seq_lens(self) -> torch.Tensor:
        return self._select_batch_rows(self.seq_lens, is_global=self.seq_lens_are_global)

    @property
    def batch_block_tables(self) -> torch.Tensor:
        return self._select_batch_rows(self.block_tables, is_global=self.block_tables_are_global)

    @property
    def visible_seq_lens(self) -> torch.Tensor:
        return self.batch_visible_seq_lens

    @property
    def batch_visible_seq_lens(self) -> torch.Tensor:
        return self.batch_seq_lens + self.decode_token_count

    @classmethod
    def from_plan(
        cls,
        plan: DecodeBatchPlan,
        *,
        physical_block_tables: bool | None = None,
        physical_key_pages: torch.Tensor | None = None,
        physical_value_pages: torch.Tensor | None = None,
    ) -> "SlotDecodeModelInputs":
        return cls(
            request_ids=plan.request_ids,
            input_ids=plan.input_ids,
            slot_ids=plan.slot_ids,
            epochs=plan.epochs,
            positions=plan.positions,
            positions_are_global=plan.positions_are_global,
            seq_lens=plan.seq_lens,
            seq_lens_are_global=plan.seq_lens_are_global,
            block_tables=plan.block_tables,
            block_tables_are_global=plan.block_tables_are_global,
            physical_block_tables=plan.physical_block_tables
            if physical_block_tables is None
            else bool(physical_block_tables),
            sampling_params=plan.sampling_params,
            sampling_batch_params=plan.sampling_batch_params,
            physical_key_pages=physical_key_pages,
            physical_value_pages=physical_value_pages,
        )


@dataclass(frozen=True, slots=True)
class SlotPrefillModelInputs:
    """Tensor view for a future slot-owned prefill chunk forward.

    ``seq_lens`` and ``positions`` describe the slot lengths before this
    chunk is written. With ``physical_block_tables=True``, Qwen attention can
    write the chunk's K/V into the internal page bank as a side effect. The
    legacy cache object is still required for the current output/continuation
    path until decode is moved onto the same slot-owned pages.
    """

    request_ids: tuple[str, ...]
    input_ids: torch.Tensor
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    positions: torch.Tensor
    positions_are_global: bool
    seq_lens: torch.Tensor
    seq_lens_are_global: bool
    block_tables: torch.Tensor
    block_tables_are_global: bool
    physical_block_tables: bool
    physical_key_pages: torch.Tensor | None = None
    physical_value_pages: torch.Tensor | None = None

    @property
    def prefill_token_count(self) -> int:
        if self.input_ids.ndim <= 1:
            return 1
        return int(self.input_ids.shape[1])

    @property
    def batch_size(self) -> int:
        if self.input_ids.ndim == 0:
            return 1
        return int(self.input_ids.shape[0])

    @property
    def contains_cache_objects(self) -> bool:
        return False

    @property
    def block_table_ownership(self) -> str:
        return "physical" if self.physical_block_tables else "logical_slot_metadata"

    @property
    def owns_physical_kv_pages(self) -> bool:
        return self.physical_key_pages is not None and self.physical_value_pages is not None

    @property
    def physical_kv_layer_count(self) -> int:
        if self.physical_key_pages is None:
            return 0
        return int(self.physical_key_pages.shape[0])

    def physical_pages_for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.physical_key_pages is None or self.physical_value_pages is None:
            raise RuntimeError("Slot prefill inputs do not own physical KV pages.")
        if layer_idx < 0 or layer_idx >= int(self.physical_key_pages.shape[0]):
            raise IndexError(f"layer_idx must be in range [0, {int(self.physical_key_pages.shape[0])}).")
        return self.physical_key_pages[layer_idx], self.physical_value_pages[layer_idx]

    def _select_batch_rows(self, tensor: torch.Tensor, *, is_global: bool) -> torch.Tensor:
        if not is_global:
            return tensor
        index = self.slot_ids.to(device=tensor.device, dtype=torch.long)
        return tensor.index_select(0, index)

    @property
    def batch_positions(self) -> torch.Tensor:
        return self._select_batch_rows(self.positions, is_global=self.positions_are_global)

    @property
    def batch_position_ids(self) -> torch.Tensor:
        base = self.batch_positions.to(dtype=torch.long).view(self.batch_size, 1)
        return base + torch.arange(
            self.prefill_token_count,
            dtype=torch.long,
            device=base.device,
        ).view(1, -1)

    @property
    def batch_seq_lens(self) -> torch.Tensor:
        return self._select_batch_rows(self.seq_lens, is_global=self.seq_lens_are_global)

    @property
    def batch_block_tables(self) -> torch.Tensor:
        return self._select_batch_rows(self.block_tables, is_global=self.block_tables_are_global)

    @property
    def visible_seq_lens(self) -> torch.Tensor:
        return self.batch_visible_seq_lens

    @property
    def batch_visible_seq_lens(self) -> torch.Tensor:
        return self.batch_seq_lens + self.prefill_token_count

    def boundary_summary(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "prefill_token_count": self.prefill_token_count,
            "contains_cache_objects": self.contains_cache_objects,
            "input_ids_device": str(self.input_ids.device),
            "slot_ids_device": str(self.slot_ids.device),
            "positions_are_global": bool(self.positions_are_global),
            "seq_lens_are_global": bool(self.seq_lens_are_global),
            "block_tables_are_global": bool(self.block_tables_are_global),
            "physical_block_tables": bool(self.physical_block_tables),
            "block_table_ownership": self.block_table_ownership,
            "block_tables_shape": tuple(int(dim) for dim in self.block_tables.shape),
            "seq_lens_shape": tuple(int(dim) for dim in self.seq_lens.shape),
            "positions_shape": tuple(int(dim) for dim in self.positions.shape),
            "owns_physical_kv_pages": self.owns_physical_kv_pages,
            "physical_kv_layer_count": self.physical_kv_layer_count,
            "physical_key_pages_shape": None
            if self.physical_key_pages is None
            else tuple(int(dim) for dim in self.physical_key_pages.shape),
            "physical_value_pages_shape": None
            if self.physical_value_pages is None
            else tuple(int(dim) for dim in self.physical_value_pages.shape),
        }


def resolve_slot_model_runner_config(
    text_config: Qwen3_5TextConfig,
    *,
    device: torch.device | str,
    max_slots: int = 0,
    total_blocks: int = 0,
    max_blocks_per_seq: int = 0,
    max_batch_size: int = 0,
) -> SlotModelRunnerConfig:
    block_size = max(1, int(text_config.cache_block_size))
    context_blocks = (max(1, int(text_config.max_position_embeddings)) + block_size - 1) // block_size
    resolved_max_blocks = max(1, int(max_blocks_per_seq) if max_blocks_per_seq > 0 else context_blocks)
    resolved_max_slots = max(1, int(max_slots) if max_slots > 0 else (int(max_batch_size) if max_batch_size > 0 else 1))
    resolved_max_batch = max(1, int(max_batch_size) if max_batch_size > 0 else resolved_max_slots)
    resolved_max_batch = min(resolved_max_batch, resolved_max_slots)
    resolved_total_blocks = (
        int(total_blocks)
        if total_blocks > 0
        else resolved_max_slots * resolved_max_blocks
    )
    resolved_total_blocks = max(resolved_max_slots, resolved_total_blocks)
    return SlotModelRunnerConfig(
        max_slots=resolved_max_slots,
        total_blocks=resolved_total_blocks,
        block_size=block_size,
        max_blocks_per_seq=resolved_max_blocks,
        max_batch_size=resolved_max_batch,
        device=torch.device(device),
        num_layers=int(text_config.num_hidden_layers),
        num_key_value_heads=int(text_config.num_key_value_heads),
        head_dim=int(text_config.head_dim),
    )


class SlotModelRunner:
    """Experimental slot-based model-runner boundary.

    This component owns slot/block metadata and emits decode tensors that match
    the planned paged-attention interface. It intentionally does not execute the
    model yet; the current ``AnnaScheduler`` remains the production path until
    prefill writes and decode forward can consume these slot views end to end.
    """

    def __init__(self, config: SlotModelRunnerConfig) -> None:
        self.config = config
        self.kv_manager = PagedKVManager(
            max_slots=config.max_slots,
            total_blocks=config.total_blocks,
            block_size=config.block_size,
            max_blocks_per_seq=config.max_blocks_per_seq,
            device=config.device,
        )
        self.scheduler = SlotScheduler(self.kv_manager, max_batch_size=config.max_batch_size)
        self.kv_page_bank: SlotKVPageBank | None = None

    @classmethod
    def from_text_config(
        cls,
        text_config: Qwen3_5TextConfig,
        *,
        device: torch.device | str,
        max_slots: int = 0,
        total_blocks: int = 0,
        max_blocks_per_seq: int = 0,
        max_batch_size: int = 0,
    ) -> "SlotModelRunner":
        return cls(
            resolve_slot_model_runner_config(
                text_config,
                device=device,
                max_slots=max_slots,
                total_blocks=total_blocks,
                max_blocks_per_seq=max_blocks_per_seq,
                max_batch_size=max_batch_size,
            )
        )

    @property
    def active_count(self) -> int:
        return self.scheduler.active_count

    def admit_prefill(
        self,
        request_id: str,
        *,
        prompt_length: int,
        max_new_tokens: int,
        sampling_params: Any | None = None,
    ) -> SequenceSlot:
        return self.scheduler.admit(
            request_id,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )

    def advance_prefill(self, request_id: str, *, token_count: int) -> SequenceSlot:
        return self.scheduler.advance_prefill(request_id, token_count=token_count)

    def mark_prefilled(
        self,
        request_id: str,
        *,
        next_input_id: int | torch.Tensor,
        next_input_host_id: int | None = None,
    ) -> SequenceSlot:
        return self.scheduler.mark_prefilled(
            request_id,
            next_input_id=next_input_id,
            next_input_host_id=next_input_host_id,
        )

    def mark_prefilled_batch(
        self,
        *,
        request_ids: Sequence[str],
        next_input_ids: torch.Tensor | Sequence[int | torch.Tensor],
        next_input_host_ids: Sequence[int | None] | None = None,
    ) -> tuple[SequenceSlot, ...]:
        return self.scheduler.mark_prefilled_batch(
            request_ids=request_ids,
            next_input_ids=next_input_ids,
            next_input_host_ids=next_input_host_ids,
        )

    def build_prefill_inputs(
        self,
        *,
        request_ids: Sequence[str],
        input_ids: torch.Tensor,
        physical_block_tables: bool = False,
    ) -> SlotPrefillModelInputs:
        request_id_tuple = tuple(str(request_id) for request_id in request_ids)
        if not request_id_tuple:
            raise ValueError("request_ids must contain at least one request.")
        if len(set(request_id_tuple)) != len(request_id_tuple):
            raise ValueError("request_ids must be unique within a prefill batch.")
        if len(request_id_tuple) > self.config.max_batch_size:
            raise ValueError(
                f"Requested prefill batch size {len(request_id_tuple)} exceeds max_batch_size={self.config.max_batch_size}."
            )
        if physical_block_tables and self.kv_page_bank is None:
            raise RuntimeError(
                "physical_block_tables=True requires SlotModelRunner.allocate_physical_kv_page_bank() first."
            )

        chunk_input_ids = input_ids.detach()
        if chunk_input_ids.ndim == 1:
            if len(request_id_tuple) != 1:
                raise ValueError("1D prefill input_ids can only be used for a single request.")
            chunk_input_ids = chunk_input_ids.view(1, -1)
        if chunk_input_ids.ndim != 2:
            raise ValueError("prefill input_ids must be shaped [batch, chunk_tokens].")
        if int(chunk_input_ids.shape[0]) != len(request_id_tuple):
            raise ValueError("prefill input_ids must contain one row per request id.")
        chunk_tokens = int(chunk_input_ids.shape[1])
        if chunk_tokens <= 0:
            raise ValueError("prefill input_ids must contain at least one token per row.")
        if chunk_input_ids.device != self.config.device:
            chunk_input_ids = chunk_input_ids.to(device=self.config.device)
        if chunk_input_ids.dtype != torch.long:
            chunk_input_ids = chunk_input_ids.to(dtype=torch.long)

        slots = tuple(self.scheduler.get(request_id) for request_id in request_id_tuple)
        target_prefilled_by_slot: list[tuple[SequenceSlot, int]] = []
        additional_blocks_needed = 0
        for slot in slots:
            if slot.status is not SlotState.WAITING_PREFILL:
                raise ValueError(f"Request {slot.request_id!r} is not waiting for prefill.")
            target_prefilled = slot.prefilled_tokens + chunk_tokens
            if target_prefilled > slot.prompt_length:
                raise ValueError(
                    f"Request {slot.request_id!r} prefill chunk would reach {target_prefilled} tokens, "
                    f"but prompt_length={slot.prompt_length}."
                )
            required_blocks = self.kv_manager.required_blocks_for_tokens(target_prefilled)
            if required_blocks > self.kv_manager.max_blocks_per_seq:
                raise SlotCapacityError(
                    f"Request {slot.request_id!r} needs {required_blocks} KV blocks, "
                    f"but max_blocks_per_seq={self.kv_manager.max_blocks_per_seq}."
                )
            current_blocks = len(self.kv_manager.slot_blocks(slot.slot_id, slot.epoch))
            additional_blocks_needed += max(0, required_blocks - current_blocks)
            target_prefilled_by_slot.append((slot, target_prefilled))
        if additional_blocks_needed > self.kv_manager.free_block_count:
            raise ValueError(
                f"Prefill chunk needs {additional_blocks_needed} additional KV blocks, "
                f"but only {self.kv_manager.free_block_count} are free."
            )

        for slot, target_prefilled in target_prefilled_by_slot:
            self.kv_manager.ensure_token_capacity(slot.slot_id, slot.epoch, target_prefilled)

        slot_ids = torch.tensor([slot.slot_id for slot in slots], dtype=torch.int32, device=self.config.device)
        epochs = torch.tensor([slot.epoch for slot in slots], dtype=torch.int64, device=self.config.device)
        bank = self.kv_page_bank if physical_block_tables else None
        return SlotPrefillModelInputs(
            request_ids=request_id_tuple,
            input_ids=chunk_input_ids,
            slot_ids=slot_ids,
            epochs=epochs,
            positions=self.kv_manager.seq_lens,
            positions_are_global=True,
            seq_lens=self.kv_manager.seq_lens,
            seq_lens_are_global=True,
            block_tables=self.kv_manager.block_tables,
            block_tables_are_global=True,
            physical_block_tables=bool(physical_block_tables),
            physical_key_pages=None if bank is None else bank.key_pages,
            physical_value_pages=None if bank is None else bank.value_pages,
        )

    def build_decode_inputs(
        self,
        *,
        request_ids: Sequence[str] | None = None,
        limit: int | None = None,
        physical_block_tables: bool = False,
    ) -> SlotDecodeModelInputs:
        if physical_block_tables and self.kv_page_bank is None:
            raise RuntimeError(
                "physical_block_tables=True requires SlotModelRunner.allocate_physical_kv_page_bank() first."
            )
        bank = self.kv_page_bank if physical_block_tables else None
        return SlotDecodeModelInputs.from_plan(
            self.scheduler.build_decode_plan(request_ids=request_ids, limit=limit),
            physical_block_tables=bool(physical_block_tables),
            physical_key_pages=None if bank is None else bank.key_pages,
            physical_value_pages=None if bank is None else bank.value_pages,
        )

    def allocate_physical_kv_page_bank(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        num_layers: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
    ) -> SlotKVPageBank:
        resolved_layers = self.config.num_layers if num_layers is None else int(num_layers)
        resolved_heads = self.config.num_key_value_heads if num_key_value_heads is None else int(num_key_value_heads)
        resolved_head_dim = self.config.head_dim if head_dim is None else int(head_dim)
        if resolved_layers <= 0:
            raise ValueError("num_layers must be positive for a physical KV page bank.")
        if resolved_heads <= 0:
            raise ValueError("num_key_value_heads must be positive for a physical KV page bank.")
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive for a physical KV page bank.")
        shape = (
            resolved_layers,
            self.config.total_blocks,
            resolved_heads,
            self.config.block_size,
            resolved_head_dim,
        )
        key_pages = torch.zeros(shape, dtype=dtype, device=self.config.device)
        value_pages = torch.zeros_like(key_pages)
        self.kv_page_bank = SlotKVPageBank(
            key_pages=key_pages,
            value_pages=value_pages,
            num_layers=resolved_layers,
            total_blocks=self.config.total_blocks,
            num_key_value_heads=resolved_heads,
            block_size=self.config.block_size,
            head_dim=resolved_head_dim,
        )
        return self.kv_page_bank

    def advance_decode(
        self,
        request_id: str,
        *,
        next_input_id: int | torch.Tensor | None = None,
        next_input_host_id: int | None = None,
        finished: bool = False,
    ) -> SequenceSlot:
        return self.scheduler.advance_decode(
            request_id,
            next_input_id=next_input_id,
            next_input_host_id=next_input_host_id,
            finished=finished,
        )

    def advance_decode_batch(
        self,
        *,
        request_ids: Sequence[str],
        next_input_ids: torch.Tensor | Sequence[int | torch.Tensor | None] | None = None,
        next_input_host_ids: Sequence[int | None] | None = None,
        finished: Sequence[bool] | None = None,
    ) -> tuple[SequenceSlot, ...]:
        return self.scheduler.advance_decode_batch(
            request_ids=request_ids,
            next_input_ids=next_input_ids,
            next_input_host_ids=next_input_host_ids,
            finished=finished,
        )

    def finish(self, request_id: str) -> SequenceSlot:
        return self.scheduler.finish(request_id)

    def cancel(self, request_id: str) -> SequenceSlot:
        return self.scheduler.cancel(request_id)

    def health(self) -> dict[str, object]:
        prefill_physical_ready = self.kv_page_bank is not None
        return {
            "enabled": True,
            "metadata_only": True,
            "integrated_generation": False,
            "slot_owned_kv_writes": False,
            "device": str(self.config.device),
            "max_slots": self.config.max_slots,
            "active_slots": self.scheduler.active_count,
            "free_slots": self.kv_manager.free_slot_count,
            "block_size": self.config.block_size,
            "total_blocks": self.config.total_blocks,
            "free_blocks": self.kv_manager.free_block_count,
            "max_blocks_per_seq": self.config.max_blocks_per_seq,
            "max_batch_size": self.config.max_batch_size,
            "metadata_tensors": {
                "block_tables": True,
                "seq_lens": True,
                "positions_alias_seq_lens": True,
                "slot_epochs": True,
                "slot_active": True,
                "block_refcounts": True,
                "device": str(self.kv_manager.device),
                "block_tables_shape": tuple(int(dim) for dim in self.kv_manager.block_tables.shape),
                "seq_lens_shape": tuple(int(dim) for dim in self.kv_manager.seq_lens.shape),
                "slot_epochs_shape": tuple(int(dim) for dim in self.kv_manager.slot_epochs.shape),
                "block_refcounts_shape": tuple(int(dim) for dim in self.kv_manager.block_refcounts.shape),
            },
            "prefill_plan": {
                "contains_cache_objects": False,
                "input_ids": True,
                "slot_ids": True,
                "epochs": True,
                "block_tables_are_global": True,
                "seq_lens_are_global": True,
                "positions_are_global": True,
                "physical_block_tables": prefill_physical_ready,
                "physical_block_tables_available": self.kv_page_bank is not None,
                "block_table_ownership": "physical" if prefill_physical_ready else "logical_slot_metadata",
                "owns_physical_kv_pages": self.kv_page_bank is not None,
                "chunk_input_contract": True,
                "commits_seq_lens": False,
                "production_forward_integrated": prefill_physical_ready,
            },
            "decode_plan": {
                "contains_cache_objects": False,
                "input_ids": True,
                "slot_ids": True,
                "epochs": True,
                "block_tables_are_global": True,
                "seq_lens_are_global": True,
                "positions_are_global": True,
                "physical_block_tables": False,
                "physical_block_tables_available": self.kv_page_bank is not None,
                "block_table_ownership": "logical_slot_metadata",
                "owns_physical_kv_pages": self.kv_page_bank is not None,
                "sampling_batch_params": True,
            },
            "kv_write_path": {
                "physical_page_write_helper": True,
                "physical_decode_page_write_helper": True,
                "physical_prefill_page_write_helper": True,
                "external_physical_block_tables_supported": True,
                "internal_block_tables_are_physical": prefill_physical_ready,
                "internal_physical_decode_ready": False,
                "internal_physical_decode_reason": (
                    "slot_owned_decode_forward_not_integrated"
                    if self.kv_page_bank is not None
                    else "physical_kv_page_bank_not_allocated"
                ),
                "prefill_slot_owned_writes": prefill_physical_ready,
                "decode_slot_owned_writes": False,
                "legacy_cache_object_required_for_forward": True,
                "tensor_bank_ready": self.kv_page_bank is not None,
            },
            "physical_kv_page_bank": (
                {"allocated": False}
                if self.kv_page_bank is None
                else self.kv_page_bank.health()
            ),
            "token_buffers": {
                "next_input_device_tensor": True,
                "slot_owned_output_token_buffer": True,
                "host_output_ids_mirror": True,
            },
            "sampling_batch_params_cache": self.scheduler.sampling_batch_params_cache_stats(),
        }
