from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.sampling.params import SamplingBatchParams
from anna.runtime.paged_kv import PagedKVManager
from anna.runtime.slot_scheduler import DecodeBatchPlan, SequenceSlot, SlotScheduler


@dataclass(frozen=True, slots=True)
class SlotModelRunnerConfig:
    max_slots: int
    total_blocks: int
    block_size: int
    max_blocks_per_seq: int
    max_batch_size: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class SlotDecodeModelInputs:
    """Tensor view passed from a slot scheduler to the future model runner."""

    request_ids: tuple[str, ...]
    input_ids: torch.Tensor
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    physical_block_tables: bool
    sampling_params: tuple[Any | None, ...]
    sampling_batch_params: SamplingBatchParams

    @classmethod
    def from_plan(cls, plan: DecodeBatchPlan) -> "SlotDecodeModelInputs":
        return cls(
            request_ids=plan.request_ids,
            input_ids=plan.input_ids,
            slot_ids=plan.slot_ids,
            epochs=plan.epochs,
            positions=plan.positions,
            seq_lens=plan.seq_lens,
            block_tables=plan.block_tables,
            physical_block_tables=plan.physical_block_tables,
            sampling_params=plan.sampling_params,
            sampling_batch_params=plan.sampling_batch_params,
        )


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

    def mark_prefilled(self, request_id: str, *, next_input_id: int) -> SequenceSlot:
        return self.scheduler.mark_prefilled(request_id, next_input_id=next_input_id)

    def build_decode_inputs(
        self,
        *,
        request_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> SlotDecodeModelInputs:
        return SlotDecodeModelInputs.from_plan(
            self.scheduler.build_decode_plan(request_ids=request_ids, limit=limit)
        )

    def advance_decode(
        self,
        request_id: str,
        *,
        next_input_id: int | None = None,
        finished: bool = False,
    ) -> SequenceSlot:
        return self.scheduler.advance_decode(
            request_id,
            next_input_id=next_input_id,
            finished=finished,
        )

    def finish(self, request_id: str) -> SequenceSlot:
        return self.scheduler.finish(request_id)

    def cancel(self, request_id: str) -> SequenceSlot:
        return self.scheduler.cancel(request_id)

    def health(self) -> dict[str, object]:
        return {
            "enabled": True,
            "metadata_only": True,
            "integrated_generation": False,
            "device": str(self.config.device),
            "max_slots": self.config.max_slots,
            "active_slots": self.scheduler.active_count,
            "free_slots": self.kv_manager.free_slot_count,
            "block_size": self.config.block_size,
            "total_blocks": self.config.total_blocks,
            "free_blocks": self.kv_manager.free_block_count,
            "max_blocks_per_seq": self.config.max_blocks_per_seq,
            "max_batch_size": self.config.max_batch_size,
        }
