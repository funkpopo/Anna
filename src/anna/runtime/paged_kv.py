from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


class PagedKVError(RuntimeError):
    """Base error for paged KV metadata lifecycle failures."""


class NoFreeSlotError(PagedKVError):
    """Raised when all scheduler slots are currently in use."""


class NoFreeBlockError(PagedKVError):
    """Raised when the KV block pool cannot satisfy an allocation."""


class SlotCapacityError(PagedKVError):
    """Raised when a sequence exceeds its configured block-table capacity."""


class StaleSlotError(PagedKVError):
    """Raised when an old request tries to mutate a reused slot."""


@dataclass(frozen=True, slots=True)
class KVSlotHandle:
    slot_id: int
    epoch: int


@dataclass(frozen=True, slots=True)
class PagedKVDecodePlan:
    slot_ids: torch.Tensor
    epochs: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor


class PagedKVManager:
    """Control-plane owner for slot-indexed paged KV metadata.

    The first implementation intentionally manages metadata only. Real K/V
    page tensors remain owned by the existing cache implementation until the
    model runner is migrated. This gives the scheduler a vLLM-like block-table
    contract without touching the current decode path.
    """

    def __init__(
        self,
        *,
        max_slots: int,
        total_blocks: int,
        block_size: int,
        max_blocks_per_seq: int,
        device: torch.device | str | None = None,
    ) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive.")
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if max_blocks_per_seq <= 0:
            raise ValueError("max_blocks_per_seq must be positive.")

        self.max_slots = int(max_slots)
        self.total_blocks = int(total_blocks)
        self.block_size = int(block_size)
        self.max_blocks_per_seq = int(max_blocks_per_seq)
        self.device = torch.device("cpu") if device is None else torch.device(device)

        self.block_tables = torch.full(
            (self.max_slots, self.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.seq_lens = torch.zeros((self.max_slots,), dtype=torch.int64, device=self.device)
        self.slot_epochs = torch.zeros((self.max_slots,), dtype=torch.int64, device=self.device)
        self.slot_active = torch.zeros((self.max_slots,), dtype=torch.bool, device=self.device)
        self.block_refcounts = torch.zeros((self.total_blocks,), dtype=torch.int32, device=self.device)

        self._free_slots: deque[int] = deque(range(self.max_slots))
        self._free_blocks: deque[int] = deque(range(self.total_blocks))
        self._active_host = [False] * self.max_slots
        self._slot_epochs_host = [0] * self.max_slots
        self._seq_lens_host = [0] * self.max_slots
        self._slot_blocks_host: list[list[int]] = [[] for _ in range(self.max_slots)]
        self._block_refcounts_host = [0] * self.total_blocks

    @property
    def free_slot_count(self) -> int:
        return len(self._free_slots)

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def allocate_slot(self, *, required_blocks: int = 0) -> KVSlotHandle:
        if required_blocks < 0:
            raise ValueError("required_blocks must be non-negative.")
        if required_blocks > self.max_blocks_per_seq:
            raise SlotCapacityError(
                f"Requested {required_blocks} blocks, but max_blocks_per_seq={self.max_blocks_per_seq}."
            )
        if not self._free_slots:
            raise NoFreeSlotError("No free sequence slots are available.")
        if len(self._free_blocks) < required_blocks:
            raise NoFreeBlockError(
                f"Requested {required_blocks} blocks, but only {len(self._free_blocks)} are free."
            )

        slot_id = self._free_slots.popleft()
        epoch = self._slot_epochs_host[slot_id] + 1
        self._slot_epochs_host[slot_id] = epoch
        self._active_host[slot_id] = True
        self._seq_lens_host[slot_id] = 0
        self._slot_blocks_host[slot_id].clear()

        self.block_tables[slot_id].fill_(-1)
        self.seq_lens[slot_id] = 0
        self.slot_epochs[slot_id] = epoch
        self.slot_active[slot_id] = True

        if required_blocks:
            self.append_blocks(slot_id, epoch, required_blocks)
        return KVSlotHandle(slot_id=slot_id, epoch=epoch)

    def release_slot(self, slot_id: int, epoch: int) -> None:
        self._validate_slot(slot_id, epoch)
        blocks = list(self._slot_blocks_host[slot_id])
        for block_id in blocks:
            self._release_block(block_id)

        self._slot_blocks_host[slot_id].clear()
        self._seq_lens_host[slot_id] = 0
        self._active_host[slot_id] = False
        self.block_tables[slot_id].fill_(-1)
        self.seq_lens[slot_id] = 0
        self.slot_active[slot_id] = False
        self._free_slots.append(slot_id)

    def required_blocks_for_tokens(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be non-negative.")
        if token_count == 0:
            return 0
        return (int(token_count) + self.block_size - 1) // self.block_size

    def ensure_token_capacity(self, slot_id: int, epoch: int, token_count: int) -> None:
        self._validate_slot(slot_id, epoch)
        required_blocks = self.required_blocks_for_tokens(token_count)
        current_blocks = len(self._slot_blocks_host[slot_id])
        if required_blocks <= current_blocks:
            return
        self.append_blocks(slot_id, epoch, required_blocks - current_blocks)

    def append_blocks(self, slot_id: int, epoch: int, count: int) -> tuple[int, ...]:
        self._validate_slot(slot_id, epoch)
        if count < 0:
            raise ValueError("count must be non-negative.")
        if count == 0:
            return ()

        current_blocks = len(self._slot_blocks_host[slot_id])
        new_count = current_blocks + count
        if new_count > self.max_blocks_per_seq:
            raise SlotCapacityError(
                f"Slot {slot_id} would need {new_count} blocks, "
                f"but max_blocks_per_seq={self.max_blocks_per_seq}."
            )
        if len(self._free_blocks) < count:
            raise NoFreeBlockError(f"Requested {count} blocks, but only {len(self._free_blocks)} are free.")

        allocated: list[int] = []
        for offset in range(count):
            block_id = self._free_blocks.popleft()
            row_idx = current_blocks + offset
            self._retain_block(block_id)
            self._slot_blocks_host[slot_id].append(block_id)
            self.block_tables[slot_id, row_idx] = block_id
            allocated.append(block_id)
        return tuple(allocated)

    def set_seq_len(self, slot_id: int, epoch: int, length: int) -> None:
        self._validate_slot(slot_id, epoch)
        if length < 0:
            raise ValueError("length must be non-negative.")
        self.ensure_token_capacity(slot_id, epoch, length)
        self._seq_lens_host[slot_id] = int(length)
        self.seq_lens[slot_id] = int(length)

    def commit_tokens(self, slot_id: int, epoch: int, count: int = 1) -> int:
        self._validate_slot(slot_id, epoch)
        if count < 0:
            raise ValueError("count must be non-negative.")
        new_length = self._seq_lens_host[slot_id] + int(count)
        self.set_seq_len(slot_id, epoch, new_length)
        return new_length

    def decode_plan(self, handles: Sequence[KVSlotHandle]) -> PagedKVDecodePlan:
        if not handles:
            raise ValueError("decode_plan requires at least one active slot.")
        slot_ids = [int(handle.slot_id) for handle in handles]
        epochs = [int(handle.epoch) for handle in handles]
        for slot_id, epoch in zip(slot_ids, epochs, strict=True):
            self._validate_slot(slot_id, epoch)

        index = torch.tensor(slot_ids, dtype=torch.long, device=self.device)
        seq_lens = self.seq_lens.index_select(0, index)
        return PagedKVDecodePlan(
            slot_ids=index.to(dtype=torch.int32),
            epochs=torch.tensor(epochs, dtype=torch.int64, device=self.device),
            seq_lens=seq_lens,
            positions=seq_lens.clone(),
            block_tables=self.block_tables.index_select(0, index),
        )

    def slot_blocks(self, slot_id: int, epoch: int) -> tuple[int, ...]:
        self._validate_slot(slot_id, epoch)
        return tuple(self._slot_blocks_host[slot_id])

    def active_handles(self) -> tuple[KVSlotHandle, ...]:
        return tuple(
            KVSlotHandle(slot_id=slot_id, epoch=self._slot_epochs_host[slot_id])
            for slot_id, active in enumerate(self._active_host)
            if active
        )

    def _retain_block(self, block_id: int) -> None:
        self._validate_block_id(block_id)
        self._block_refcounts_host[block_id] += 1
        self.block_refcounts[block_id] = self._block_refcounts_host[block_id]

    def _release_block(self, block_id: int) -> None:
        self._validate_block_id(block_id)
        refcount = self._block_refcounts_host[block_id]
        if refcount <= 0:
            raise PagedKVError(f"Block {block_id} refcount is already zero.")
        refcount -= 1
        self._block_refcounts_host[block_id] = refcount
        self.block_refcounts[block_id] = refcount
        if refcount == 0:
            self._free_blocks.append(block_id)

    def _validate_slot(self, slot_id: int, epoch: int) -> None:
        if slot_id < 0 or slot_id >= self.max_slots:
            raise IndexError(f"slot_id {slot_id} is out of range for max_slots={self.max_slots}.")
        if not self._active_host[slot_id]:
            raise StaleSlotError(f"Slot {slot_id} is not active.")
        current_epoch = self._slot_epochs_host[slot_id]
        if int(epoch) != current_epoch:
            raise StaleSlotError(f"Slot {slot_id} epoch {epoch} is stale; current epoch is {current_epoch}.")

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.total_blocks:
            raise IndexError(f"block_id {block_id} is out of range for total_blocks={self.total_blocks}.")


def handles_from_pairs(pairs: Iterable[tuple[int, int]]) -> tuple[KVSlotHandle, ...]:
    return tuple(KVSlotHandle(slot_id=int(slot_id), epoch=int(epoch)) for slot_id, epoch in pairs)
