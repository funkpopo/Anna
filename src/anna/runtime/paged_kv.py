from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

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
        self._prefix_block_pool: dict[Hashable, int] = {}
        self._block_prefix_keys: list[set[Hashable]] = [set() for _ in range(self.total_blocks)]

    @property
    def free_slot_count(self) -> int:
        return len(self._free_slots)

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def prefix_pool_size(self) -> int:
        return len(self._prefix_block_pool)

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

    def share_prefix_from_slot(
        self,
        *,
        source_slot_id: int,
        source_epoch: int,
        target_slot_id: int,
        target_epoch: int,
        token_count: int,
    ) -> tuple[int, ...]:
        """Retain a complete-block prefix from one active slot into another.

        This is metadata-only prefix reuse for the slot runner. It intentionally
        shares only full blocks; partial blocks remain owned by the request that
        produced them because decode/prefill may still append into the tail.
        """
        self._validate_slot(source_slot_id, source_epoch)
        self._validate_slot(target_slot_id, target_epoch)
        if source_slot_id == target_slot_id:
            raise ValueError("source and target slots must be different.")
        if token_count < 0:
            raise ValueError("token_count must be non-negative.")
        if self._slot_blocks_host[target_slot_id]:
            raise PagedKVError("Prefix sharing requires an empty target slot.")

        shareable_tokens = min(int(token_count), self._seq_lens_host[source_slot_id])
        full_block_count = shareable_tokens // self.block_size
        if full_block_count <= 0:
            return ()
        if full_block_count > self.max_blocks_per_seq:
            raise SlotCapacityError(
                f"Shared prefix needs {full_block_count} blocks, "
                f"but max_blocks_per_seq={self.max_blocks_per_seq}."
            )

        source_blocks = self._slot_blocks_host[source_slot_id]
        shared_blocks = tuple(source_blocks[:full_block_count])
        if len(shared_blocks) != full_block_count:
            raise PagedKVError("Source slot is missing blocks for the requested shared prefix.")

        for row_idx, block_id in enumerate(shared_blocks):
            self._retain_block(block_id)
            self._slot_blocks_host[target_slot_id].append(block_id)
            self.block_tables[target_slot_id, row_idx] = block_id

        shared_tokens = full_block_count * self.block_size
        self._seq_lens_host[target_slot_id] = shared_tokens
        self.seq_lens[target_slot_id] = shared_tokens
        return shared_blocks

    def publish_prefix_hashes(
        self,
        slot_id: int,
        epoch: int,
        block_hashes: Sequence[Hashable],
        *,
        token_count: int | None = None,
    ) -> tuple[Hashable, ...]:
        """Publish complete prefix blocks into the metadata prefix pool.

        The pool is content-addressed by caller-provided block hashes. Only
        complete blocks are published because the tail block may still be
        mutated by subsequent decode steps.
        """
        self._validate_slot(slot_id, epoch)
        visible_tokens = self._seq_lens_host[slot_id] if token_count is None else min(int(token_count), self._seq_lens_host[slot_id])
        if visible_tokens < 0:
            raise ValueError("token_count must be non-negative.")
        full_block_count = min(visible_tokens // self.block_size, len(block_hashes), len(self._slot_blocks_host[slot_id]))
        published: list[Hashable] = []
        for row_idx in range(full_block_count):
            key = block_hashes[row_idx]
            self._validate_prefix_key(key)
            block_id = self._slot_blocks_host[slot_id][row_idx]
            existing_block_id = self._prefix_block_pool.get(key)
            if existing_block_id is not None:
                if self._is_live_block(existing_block_id):
                    if existing_block_id == block_id:
                        self._block_prefix_keys[block_id].add(key)
                        published.append(key)
                    continue
                self._prefix_block_pool.pop(key, None)
            self._prefix_block_pool[key] = block_id
            self._block_prefix_keys[block_id].add(key)
            published.append(key)
        return tuple(published)

    def share_prefix_by_hashes(
        self,
        *,
        target_slot_id: int,
        target_epoch: int,
        block_hashes: Sequence[Hashable],
        token_count: int,
    ) -> tuple[int, ...]:
        """Retain leading blocks from the prefix pool into an empty target slot.

        Sharing stops at the first missing hash so callers can use this as a
        prefix probe without special-casing partial hits.
        """
        self._validate_slot(target_slot_id, target_epoch)
        if token_count < 0:
            raise ValueError("token_count must be non-negative.")
        if self._slot_blocks_host[target_slot_id]:
            raise PagedKVError("Prefix hash sharing requires an empty target slot.")

        requested_blocks = min(int(token_count) // self.block_size, len(block_hashes), self.max_blocks_per_seq)
        shared_blocks: list[int] = []
        for row_idx in range(requested_blocks):
            key = block_hashes[row_idx]
            self._validate_prefix_key(key)
            block_id = self._prefix_block_pool.get(key)
            if block_id is None:
                break
            if not self._is_live_block(block_id):
                self._prefix_block_pool.pop(key, None)
                self._block_prefix_keys[block_id].discard(key)
                break
            self._retain_block(block_id)
            self._slot_blocks_host[target_slot_id].append(block_id)
            self.block_tables[target_slot_id, row_idx] = block_id
            shared_blocks.append(block_id)

        shared_tokens = len(shared_blocks) * self.block_size
        self._seq_lens_host[target_slot_id] = shared_tokens
        self.seq_lens[target_slot_id] = shared_tokens
        return tuple(shared_blocks)

    def prefix_pool_snapshot(self) -> dict[Hashable, int]:
        return dict(self._prefix_block_pool)

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

    def prepare_decode_capacity(self, handles: Sequence[KVSlotHandle], *, append_tokens: int = 1) -> None:
        if append_tokens < 0:
            raise ValueError("append_tokens must be non-negative.")
        for handle in handles:
            self._validate_slot(handle.slot_id, handle.epoch)
            target_length = self._seq_lens_host[handle.slot_id] + int(append_tokens)
            self.ensure_token_capacity(handle.slot_id, handle.epoch, target_length)

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
            for key in tuple(self._block_prefix_keys[block_id]):
                if self._prefix_block_pool.get(key) == block_id:
                    self._prefix_block_pool.pop(key, None)
            self._block_prefix_keys[block_id].clear()
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

    def _is_live_block(self, block_id: int) -> bool:
        self._validate_block_id(block_id)
        return self._block_refcounts_host[block_id] > 0

    @staticmethod
    def _validate_prefix_key(key: Hashable) -> None:
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("prefix block hash keys must be hashable.") from exc


def handles_from_pairs(pairs: Iterable[tuple[int, int]]) -> tuple[KVSlotHandle, ...]:
    return tuple(KVSlotHandle(slot_id=int(slot_id), epoch=int(epoch)) for slot_id, epoch in pairs)
