from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import torch

from anna.sampling.params import SamplingBatchParams
from anna.runtime.paged_kv import KVSlotHandle, PagedKVDecodePlan, PagedKVManager, SlotCapacityError


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
    next_input_token: torch.Tensor | None = None
    sampling_params: Any | None = None
    output_token_buffer: torch.Tensor | None = None
    output_token_count: int = 0
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

    @property
    def output_tokens_view(self) -> torch.Tensor:
        if self.output_token_buffer is None:
            return torch.empty((0,), dtype=torch.long)
        return self.output_token_buffer[: self.output_token_count]


@dataclass(frozen=True, slots=True)
class DecodeBatchPlan:
    request_ids: tuple[str, ...]
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    positions_are_global: bool
    seq_lens: torch.Tensor
    seq_lens_are_global: bool
    block_tables: torch.Tensor
    block_tables_are_global: bool
    physical_block_tables: bool
    sampling_params: tuple[Any | None, ...]
    sampling_batch_params: SamplingBatchParams

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
            positions_are_global=kv_plan.positions_are_global,
            seq_lens=kv_plan.seq_lens,
            seq_lens_are_global=kv_plan.seq_lens_are_global,
            block_tables=kv_plan.block_tables,
            block_tables_are_global=kv_plan.block_tables_are_global,
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
            output_token_buffer=torch.empty(
                (max(1, int(max_new_tokens)),),
                dtype=torch.long,
                device=self.kv_manager.device,
            ),
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

    def mark_prefilled(
        self,
        request_id: str,
        *,
        next_input_id: int | torch.Tensor,
        next_input_host_id: int | None = None,
    ) -> SequenceSlot:
        slot = self._require_active(request_id)
        if slot.status is not SlotState.WAITING_PREFILL:
            raise ValueError(f"Request {request_id!r} is not waiting for prefill.")
        if slot.prefilled_tokens < slot.prompt_length:
            self.advance_prefill(request_id, token_count=slot.prompt_length - slot.prefilled_tokens)
        self._set_next_input(slot, next_input_id=next_input_id, next_input_host_id=next_input_host_id)
        slot.status = SlotState.DECODING
        return slot

    def mark_prefilled_batch(
        self,
        *,
        request_ids: Sequence[str],
        next_input_ids: torch.Tensor | Sequence[int | torch.Tensor],
        next_input_host_ids: Sequence[int | None] | None = None,
    ) -> tuple[SequenceSlot, ...]:
        request_id_tuple = tuple(str(request_id) for request_id in request_ids)
        if not request_id_tuple:
            raise ValueError("request_ids must contain at least one request.")
        if len(set(request_id_tuple)) != len(request_id_tuple):
            raise ValueError("request_ids must be unique within a prefill batch.")
        if len(request_id_tuple) > self.max_batch_size:
            raise ValueError(
                f"Requested batch size {len(request_id_tuple)} exceeds max_batch_size={self.max_batch_size}."
            )

        slots = tuple(self._require_active(request_id) for request_id in request_id_tuple)
        next_inputs = self._split_batch_next_inputs(next_input_ids, batch_size=len(slots))
        host_ids = self._split_batch_host_ids(next_input_host_ids, batch_size=len(slots))
        normalized_next_inputs: list[torch.Tensor] = []
        additional_blocks_needed = 0
        for slot, next_input in zip(slots, next_inputs, strict=True):
            if slot.status is not SlotState.WAITING_PREFILL:
                raise ValueError(f"Request {slot.request_id!r} is not waiting for prefill.")
            if next_input is None:
                raise ValueError("next_input_ids must provide one token per prefilled request.")
            normalized_next_inputs.append(self._next_input_tensor(next_input))
            required_blocks = self.kv_manager.required_blocks_for_tokens(slot.prompt_length)
            if required_blocks > self.kv_manager.max_blocks_per_seq:
                raise SlotCapacityError(
                    f"Request {slot.request_id!r} needs {required_blocks} KV blocks, "
                    f"but max_blocks_per_seq={self.kv_manager.max_blocks_per_seq}."
                )
            current_blocks = len(self.kv_manager.slot_blocks(slot.slot_id, slot.epoch))
            additional_blocks_needed += max(0, required_blocks - current_blocks)
        if additional_blocks_needed > self.kv_manager.free_block_count:
            raise ValueError(
                f"Prefill batch needs {additional_blocks_needed} additional KV blocks, "
                f"but only {self.kv_manager.free_block_count} are free."
            )

        marked: list[SequenceSlot] = []
        for slot, next_input, host_id in zip(slots, normalized_next_inputs, host_ids, strict=True):
            if slot.prefilled_tokens < slot.prompt_length:
                self.advance_prefill(slot.request_id, token_count=slot.prompt_length - slot.prefilled_tokens)
            self._set_next_input(slot, next_input_id=next_input, next_input_host_id=host_id)
            slot.status = SlotState.DECODING
            marked.append(slot)
        return tuple(marked)

    def ready_decode_slots(self, *, limit: int | None = None) -> tuple[SequenceSlot, ...]:
        batch_limit = self.max_batch_size if limit is None else min(self.max_batch_size, max(0, int(limit)))
        ready = [
            slot
            for slot in self._active_by_request.values()
            if slot.status is SlotState.DECODING and slot.next_input_token is not None and slot.remaining_tokens > 0
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
            if len(set(request_ids)) != len(request_ids):
                raise ValueError("request_ids must be unique within a decode batch.")
            slots = tuple(self._require_active(request_id) for request_id in request_ids)
            for slot in slots:
                if slot.status is not SlotState.DECODING or slot.next_input_token is None:
                    raise ValueError(f"Request {slot.request_id!r} is not ready for decode.")
                if slot.remaining_tokens <= 0:
                    raise ValueError(f"Request {slot.request_id!r} has no remaining decode tokens.")
            if len(slots) > self.max_batch_size:
                raise ValueError(f"Requested batch size {len(slots)} exceeds max_batch_size={self.max_batch_size}.")
        if not slots:
            raise ValueError("No decode-ready slots are available.")

        input_tokens = [slot.next_input_token for slot in slots]
        if any(token is None for token in input_tokens):
            raise ValueError("Decode plan requested a slot without a next input token.")
        input_ids = torch.cat([token for token in input_tokens if token is not None], dim=0)
        handles = tuple(slot.handle for slot in slots)
        self.kv_manager.prepare_decode_capacity(handles, append_tokens=1)
        kv_plan = self.kv_manager.decode_plan(handles)
        return DecodeBatchPlan.from_kv_plan(slots=slots, input_ids=input_ids, kv_plan=kv_plan)

    def advance_decode(
        self,
        request_id: str,
        *,
        next_input_id: int | torch.Tensor | None = None,
        next_input_host_id: int | None = None,
        finished: bool = False,
    ) -> SequenceSlot:
        slot = self._require_active(request_id)
        if slot.status is not SlotState.DECODING or slot.next_input_token is None:
            raise ValueError(f"Request {request_id!r} is not ready for decode.")
        if slot.remaining_tokens <= 0:
            raise ValueError(f"Request {request_id!r} has no remaining decode tokens.")

        self._append_output_token(slot)
        slot.seq_len = self.kv_manager.commit_tokens(slot.slot_id, slot.epoch, 1)
        slot.generated_tokens += 1
        if slot.next_input_id is not None:
            slot.output_token_ids.append(slot.next_input_id)
        slot.next_input_id = None
        slot.next_input_token = None

        if finished or slot.generated_tokens >= slot.max_new_tokens:
            return self.finish(request_id)
        if next_input_id is None:
            raise ValueError("next_input_id is required when the request remains active.")
        self._set_next_input(slot, next_input_id=next_input_id, next_input_host_id=next_input_host_id)
        return slot

    def advance_decode_batch(
        self,
        *,
        request_ids: Sequence[str],
        next_input_ids: torch.Tensor | Sequence[int | torch.Tensor | None] | None = None,
        next_input_host_ids: Sequence[int | None] | None = None,
        finished: Sequence[bool] | None = None,
    ) -> tuple[SequenceSlot, ...]:
        request_id_tuple = tuple(str(request_id) for request_id in request_ids)
        if not request_id_tuple:
            raise ValueError("request_ids must contain at least one request.")
        if len(set(request_id_tuple)) != len(request_id_tuple):
            raise ValueError("request_ids must be unique within a decode batch.")
        if len(request_id_tuple) > self.max_batch_size:
            raise ValueError(
                f"Requested batch size {len(request_id_tuple)} exceeds max_batch_size={self.max_batch_size}."
            )

        slots = tuple(self._require_active(request_id) for request_id in request_id_tuple)
        next_inputs = self._split_batch_next_inputs(next_input_ids, batch_size=len(slots))
        host_ids = self._split_batch_host_ids(next_input_host_ids, batch_size=len(slots))
        finished_flags = self._split_batch_finished_flags(finished, batch_size=len(slots))

        normalized_next_inputs: list[int | torch.Tensor | None] = []
        for slot, next_input, is_finished in zip(slots, next_inputs, finished_flags, strict=True):
            if slot.status is not SlotState.DECODING or slot.next_input_token is None:
                raise ValueError(f"Request {slot.request_id!r} is not ready for decode.")
            if slot.remaining_tokens <= 0:
                raise ValueError(f"Request {slot.request_id!r} has no remaining decode tokens.")
            will_finish = bool(is_finished) or slot.generated_tokens + 1 >= slot.max_new_tokens
            if next_input is None:
                if not will_finish:
                    raise ValueError(
                        "next_input_ids must provide one token per unfinished request in a decode batch."
                    )
                normalized_next_inputs.append(None)
                continue
            normalized_next_inputs.append(self._next_input_tensor(next_input))

        advanced: list[SequenceSlot] = []
        for request_id, next_input, host_id, is_finished in zip(
            request_id_tuple,
            normalized_next_inputs,
            host_ids,
            finished_flags,
            strict=True,
        ):
            advanced.append(
                self.advance_decode(
                    request_id,
                    next_input_id=next_input,
                    next_input_host_id=host_id,
                    finished=bool(is_finished),
                )
            )
        return tuple(advanced)

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
        slot.next_input_token = None
        self._active_by_request.pop(request_id, None)
        self._active_by_slot.pop(slot.slot_id, None)
        self._completed_by_request[request_id] = slot
        return slot

    def _set_next_input(
        self,
        slot: SequenceSlot,
        *,
        next_input_id: int | torch.Tensor,
        next_input_host_id: int | None,
    ) -> None:
        slot.next_input_token = self._next_input_tensor(next_input_id)
        if next_input_host_id is not None:
            slot.next_input_id = int(next_input_host_id)
        elif isinstance(next_input_id, int):
            slot.next_input_id = int(next_input_id)
        else:
            slot.next_input_id = None

    def _append_output_token(self, slot: SequenceSlot) -> None:
        if slot.next_input_token is None:
            raise ValueError(f"Request {slot.request_id!r} is missing the consumed decode token.")
        if slot.output_token_buffer is None:
            slot.output_token_buffer = torch.empty((1,), dtype=torch.long, device=self.kv_manager.device)
        if slot.output_token_count >= int(slot.output_token_buffer.shape[0]):
            new_capacity = max(1, int(slot.output_token_buffer.shape[0]) * 2)
            resized = torch.empty((new_capacity,), dtype=slot.output_token_buffer.dtype, device=slot.output_token_buffer.device)
            if slot.output_token_count > 0:
                resized[: slot.output_token_count].copy_(slot.output_token_buffer[: slot.output_token_count])
            slot.output_token_buffer = resized
        token = slot.next_input_token.reshape(())
        if token.device != slot.output_token_buffer.device:
            token = token.to(device=slot.output_token_buffer.device)
        slot.output_token_buffer[slot.output_token_count].copy_(token.to(dtype=slot.output_token_buffer.dtype))
        slot.output_token_count += 1

    def _next_input_tensor(self, next_input_id: int | torch.Tensor) -> torch.Tensor:
        if isinstance(next_input_id, torch.Tensor):
            token = next_input_id.detach()
            if token.numel() != 1:
                raise ValueError("next_input_id tensor must contain exactly one token.")
            if token.device != self.kv_manager.device:
                token = token.to(device=self.kv_manager.device)
            if token.dtype != torch.long:
                token = token.to(dtype=torch.long)
            return token.reshape(1, 1)
        return torch.tensor([[int(next_input_id)]], dtype=torch.long, device=self.kv_manager.device)

    @staticmethod
    def _split_batch_finished_flags(finished: Sequence[bool] | None, *, batch_size: int) -> tuple[bool, ...]:
        if finished is None:
            return tuple(False for _ in range(batch_size))
        flags = tuple(bool(value) for value in finished)
        if len(flags) != batch_size:
            raise ValueError("finished must contain one value per decode batch row.")
        return flags

    @staticmethod
    def _split_batch_host_ids(
        next_input_host_ids: Sequence[int | None] | None,
        *,
        batch_size: int,
    ) -> tuple[int | None, ...]:
        if next_input_host_ids is None:
            return tuple(None for _ in range(batch_size))
        host_ids = tuple(None if token_id is None else int(token_id) for token_id in next_input_host_ids)
        if len(host_ids) != batch_size:
            raise ValueError("next_input_host_ids must contain one value per decode batch row.")
        return host_ids

    @staticmethod
    def _split_batch_next_inputs(
        next_input_ids: torch.Tensor | Sequence[int | torch.Tensor | None] | None,
        *,
        batch_size: int,
    ) -> tuple[int | torch.Tensor | None, ...]:
        if next_input_ids is None:
            return tuple(None for _ in range(batch_size))
        if isinstance(next_input_ids, torch.Tensor):
            tokens = next_input_ids.detach()
            if tokens.ndim == 0:
                if batch_size != 1:
                    raise ValueError("scalar next_input_ids can only be used for a single-row decode batch.")
                return (tokens,)
            if tokens.ndim == 1:
                if int(tokens.shape[0]) != batch_size:
                    raise ValueError("1D next_input_ids must contain one token per decode batch row.")
                return tuple(tokens[row_idx : row_idx + 1] for row_idx in range(batch_size))
            if tokens.ndim == 2:
                if int(tokens.shape[0]) != batch_size:
                    raise ValueError("2D next_input_ids must contain one row per decode batch row.")
                if int(tokens.shape[1]) != 1:
                    raise ValueError("slot decode currently accepts exactly one next token per batch row.")
                return tuple(tokens[row_idx : row_idx + 1, :] for row_idx in range(batch_size))
            raise ValueError("next_input_ids tensor must be scalar, [batch], or [batch, 1].")
        values = tuple(next_input_ids)
        if len(values) != batch_size:
            raise ValueError("next_input_ids must contain one value per decode batch row.")
        return values

    def _require_active(self, request_id: str) -> SequenceSlot:
        try:
            return self._active_by_request[request_id]
        except KeyError as exc:
            raise KeyError(f"Request {request_id!r} is not active.") from exc
