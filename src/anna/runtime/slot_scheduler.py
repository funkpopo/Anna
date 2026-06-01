from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import torch

from anna.sampling.params import SamplingBatchParams
from anna.runtime.paged_kv import KVSlotHandle, PagedKVDecodePlan, PagedKVManager


class SlotState(str, Enum):
    WAITING_PREFILL = "waiting_prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SequenceSlot:
    request_id: str
    handle: KVSlotHandle
    prompt_length: int
    max_new_tokens: int
    seq_len: int
    prefilled_tokens: int = 0
    status: SlotState = SlotState.WAITING_PREFILL
    generated_tokens: int = 0
    next_input_id: int | None = None
    sampling_params: Any | None = None
    output_token_ids: list[int] = field(default_factory=list)
    admission_index: int = 0

    @property
    def slot_id(self) -> int:
        return self.handle.slot_id

    @property
    def epoch(self) -> int:
        return self.handle.epoch

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.generated_tokens)


@dataclass(frozen=True, slots=True)
class DecodeBatchPlan:
    request_ids: tuple[str, ...]
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    physical_block_tables: bool
    sampling_params: tuple[Any | None, ...]
    sampling_batch_params: SamplingBatchParams

    @classmethod
    def from_kv_plan(
        cls,
        *,
        slots: Sequence[SequenceSlot],
        input_ids: torch.Tensor,
        kv_plan: PagedKVDecodePlan,
    ) -> "DecodeBatchPlan":
        return cls(
            request_ids=tuple(slot.request_id for slot in slots),
            slot_ids=kv_plan.slot_ids,
            epochs=kv_plan.epochs,
            input_ids=input_ids,
            positions=kv_plan.positions,
            seq_lens=kv_plan.seq_lens,
            block_tables=kv_plan.block_tables,
            physical_block_tables=False,
            sampling_params=tuple(slot.sampling_params for slot in slots),
            sampling_batch_params=SamplingBatchParams.from_sampling_params(
                tuple(slot.sampling_params for slot in slots),
                device=input_ids.device,
            ),
        )


class SlotScheduler:
    """Slot-based scheduler metadata layer for the upcoming model runner.

    This is deliberately separate from ``AnnaScheduler``. It provides the slot,
    epoch, and block-table lifecycle that the XPU decode path will consume once
    the old cache stack/split implementation is replaced.
    """

    def __init__(self, kv_manager: PagedKVManager, *, max_batch_size: int = 1) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        self.kv_manager = kv_manager
        self.max_batch_size = int(max_batch_size)
        self._active_by_request: dict[str, SequenceSlot] = {}
        self._active_by_slot: dict[int, SequenceSlot] = {}
        self._completed_by_request: dict[str, SequenceSlot] = {}
        self._next_admission_index = 0

    @property
    def active_count(self) -> int:
        return len(self._active_by_request)

    def admit(
        self,
        request_id: str,
        *,
        prompt_length: int,
        max_new_tokens: int,
        sampling_params: Any | None = None,
    ) -> SequenceSlot:
        if not request_id:
            raise ValueError("request_id must be non-empty.")
        if request_id in self._active_by_request:
            raise ValueError(f"Request {request_id!r} is already active.")
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        handle = self.kv_manager.allocate_slot()
        try:
            self.kv_manager.set_seq_len(handle.slot_id, handle.epoch, 0)
        except Exception:
            self.kv_manager.release_slot(handle.slot_id, handle.epoch)
            raise

        slot = SequenceSlot(
            request_id=request_id,
            handle=handle,
            prompt_length=int(prompt_length),
            max_new_tokens=int(max_new_tokens),
            seq_len=0,
            sampling_params=sampling_params,
            admission_index=self._next_admission_index,
        )
        self._next_admission_index += 1
        self._active_by_request[request_id] = slot
        self._active_by_slot[slot.slot_id] = slot
        return slot

    def advance_prefill(self, request_id: str, *, token_count: int) -> SequenceSlot:
        slot = self._require_active(request_id)
        if slot.status is not SlotState.WAITING_PREFILL:
            raise ValueError(f"Request {request_id!r} is not waiting for prefill.")
        if token_count < 0:
            raise ValueError("token_count must be non-negative.")

        next_prefilled = slot.prefilled_tokens + int(token_count)
        if next_prefilled > slot.prompt_length:
            raise ValueError(
                f"Request {request_id!r} prefilled {next_prefilled} tokens, "
                f"but prompt_length={slot.prompt_length}."
            )
        slot.prefilled_tokens = next_prefilled
        slot.seq_len = next_prefilled
        self.kv_manager.set_seq_len(slot.slot_id, slot.epoch, next_prefilled)
        return slot

    def mark_prefilled(self, request_id: str, *, next_input_id: int) -> SequenceSlot:
        slot = self._require_active(request_id)
        if slot.status is not SlotState.WAITING_PREFILL:
            raise ValueError(f"Request {request_id!r} is not waiting for prefill.")
        if slot.prefilled_tokens < slot.prompt_length:
            self.advance_prefill(request_id, token_count=slot.prompt_length - slot.prefilled_tokens)
        slot.next_input_id = int(next_input_id)
        slot.status = SlotState.DECODING
        return slot

    def ready_decode_slots(self, *, limit: int | None = None) -> tuple[SequenceSlot, ...]:
        batch_limit = self.max_batch_size if limit is None else min(self.max_batch_size, max(0, int(limit)))
        ready = [
            slot
            for slot in self._active_by_request.values()
            if slot.status is SlotState.DECODING and slot.next_input_id is not None
        ]
        ready.sort(key=lambda slot: slot.admission_index)
        return tuple(ready[:batch_limit])

    def build_decode_plan(
        self,
        *,
        request_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> DecodeBatchPlan:
        if request_ids is None:
            slots = self.ready_decode_slots(limit=limit)
        else:
            slots = tuple(self._require_active(request_id) for request_id in request_ids)
            for slot in slots:
                if slot.status is not SlotState.DECODING or slot.next_input_id is None:
                    raise ValueError(f"Request {slot.request_id!r} is not ready for decode.")
            if len(slots) > self.max_batch_size:
                raise ValueError(f"Requested batch size {len(slots)} exceeds max_batch_size={self.max_batch_size}.")
        if not slots:
            raise ValueError("No decode-ready slots are available.")

        input_ids = torch.tensor(
            [slot.next_input_id for slot in slots],
            dtype=torch.long,
            device=self.kv_manager.device,
        ).view(len(slots), 1)
        handles = tuple(slot.handle for slot in slots)
        self.kv_manager.prepare_decode_capacity(handles, append_tokens=1)
        kv_plan = self.kv_manager.decode_plan(handles)
        return DecodeBatchPlan.from_kv_plan(slots=slots, input_ids=input_ids, kv_plan=kv_plan)

    def advance_decode(
        self,
        request_id: str,
        *,
        next_input_id: int | None = None,
        finished: bool = False,
    ) -> SequenceSlot:
        slot = self._require_active(request_id)
        if slot.status is not SlotState.DECODING or slot.next_input_id is None:
            raise ValueError(f"Request {request_id!r} is not ready for decode.")

        consumed_token_id = int(slot.next_input_id)
        slot.seq_len = self.kv_manager.commit_tokens(slot.slot_id, slot.epoch, 1)
        slot.generated_tokens += 1
        slot.output_token_ids.append(consumed_token_id)

        if finished or slot.generated_tokens >= slot.max_new_tokens:
            return self.finish(request_id)
        if next_input_id is None:
            raise ValueError("next_input_id is required when the request remains active.")
        slot.next_input_id = int(next_input_id)
        return slot

    def finish(self, request_id: str) -> SequenceSlot:
        return self._release(request_id, status=SlotState.FINISHED)

    def cancel(self, request_id: str) -> SequenceSlot:
        return self._release(request_id, status=SlotState.CANCELLED)

    def get(self, request_id: str) -> SequenceSlot:
        return self._require_active(request_id)

    def _release(self, request_id: str, *, status: SlotState) -> SequenceSlot:
        slot = self._require_active(request_id)
        self.kv_manager.release_slot(slot.slot_id, slot.epoch)
        slot.status = status
        slot.next_input_id = None
        self._active_by_request.pop(request_id, None)
        self._active_by_slot.pop(slot.slot_id, None)
        self._completed_by_request[request_id] = slot
        return slot

    def _require_active(self, request_id: str) -> SequenceSlot:
        try:
            return self._active_by_request[request_id]
        except KeyError as exc:
            raise KeyError(f"Request {request_id!r} is not active.") from exc
