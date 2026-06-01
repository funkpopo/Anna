from __future__ import annotations

import pytest
import torch

from anna.runtime.paged_kv import PagedKVManager, SlotCapacityError, StaleSlotError
from anna.runtime.slot_scheduler import SlotScheduler, SlotState


def test_paged_kv_manager_allocates_releases_and_reuses_slot_with_new_epoch() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)

    handle = manager.allocate_slot()
    manager.set_seq_len(handle.slot_id, handle.epoch, 5)

    assert handle.slot_id == 0
    assert handle.epoch == 1
    assert manager.free_slot_count == 0
    assert manager.free_block_count == 2
    assert manager.slot_blocks(handle.slot_id, handle.epoch) == (0, 1)

    plan = manager.decode_plan([handle])
    assert plan.slot_ids.tolist() == [0]
    assert plan.seq_lens.tolist() == [5]
    assert plan.positions.tolist() == [5]
    assert plan.block_tables.tolist() == [[0, 1, -1]]

    manager.release_slot(handle.slot_id, handle.epoch)

    assert manager.free_slot_count == 1
    assert manager.free_block_count == 4
    assert manager.block_tables.tolist() == [[-1, -1, -1]]
    assert manager.block_refcounts.tolist() == [0, 0, 0, 0]

    reused = manager.allocate_slot()
    assert reused.slot_id == handle.slot_id
    assert reused.epoch == handle.epoch + 1
    with pytest.raises(StaleSlotError):
        manager.set_seq_len(handle.slot_id, handle.epoch, 1)


def test_paged_kv_manager_enforces_block_table_capacity() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=8, block_size=4, max_blocks_per_seq=2)
    handle = manager.allocate_slot()

    with pytest.raises(SlotCapacityError):
        manager.set_seq_len(handle.slot_id, handle.epoch, 9)

    manager.release_slot(handle.slot_id, handle.epoch)
    assert manager.free_slot_count == 1
    assert manager.free_block_count == 8


def test_paged_kv_manager_shares_full_prefix_blocks_with_refcounts() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=6, block_size=4, max_blocks_per_seq=4)
    source = manager.allocate_slot()
    manager.set_seq_len(source.slot_id, source.epoch, 10)
    target = manager.allocate_slot()

    shared = manager.share_prefix_from_slot(
        source_slot_id=source.slot_id,
        source_epoch=source.epoch,
        target_slot_id=target.slot_id,
        target_epoch=target.epoch,
        token_count=9,
    )

    assert shared == (0, 1)
    assert manager.slot_blocks(target.slot_id, target.epoch) == (0, 1)
    assert manager.seq_lens.tolist() == [10, 8]
    assert manager.block_tables.tolist() == [[0, 1, 2, -1], [0, 1, -1, -1]]
    assert manager.block_refcounts.tolist() == [2, 2, 1, 0, 0, 0]
    assert manager.free_block_count == 3

    manager.release_slot(source.slot_id, source.epoch)

    assert manager.free_block_count == 4
    assert manager.block_refcounts.tolist() == [1, 1, 0, 0, 0, 0]
    assert manager.slot_blocks(target.slot_id, target.epoch) == (0, 1)
    assert manager.decode_plan([target]).block_tables.tolist() == [[0, 1, -1, -1]]

    manager.release_slot(target.slot_id, target.epoch)

    assert manager.free_slot_count == 2
    assert manager.free_block_count == 6
    assert manager.block_refcounts.tolist() == [0, 0, 0, 0, 0, 0]


def test_paged_kv_manager_share_prefix_requires_empty_target_slot() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=6, block_size=4, max_blocks_per_seq=4)
    source = manager.allocate_slot()
    target = manager.allocate_slot()
    manager.set_seq_len(source.slot_id, source.epoch, 8)
    manager.set_seq_len(target.slot_id, target.epoch, 1)

    with pytest.raises(RuntimeError, match="empty target"):
        manager.share_prefix_from_slot(
            source_slot_id=source.slot_id,
            source_epoch=source.epoch,
            target_slot_id=target.slot_id,
            target_epoch=target.epoch,
            token_count=8,
        )


def test_paged_kv_manager_shares_prefix_blocks_by_hash_pool() -> None:
    manager = PagedKVManager(max_slots=3, total_blocks=8, block_size=4, max_blocks_per_seq=4)
    source = manager.allocate_slot()
    manager.set_seq_len(source.slot_id, source.epoch, 10)

    published = manager.publish_prefix_hashes(
        source.slot_id,
        source.epoch,
        ("block-a", "block-b", "tail-c"),
    )

    assert published == ("block-a", "block-b")
    assert manager.prefix_pool_size == 2
    assert manager.prefix_pool_snapshot() == {"block-a": 0, "block-b": 1}

    target = manager.allocate_slot()
    shared = manager.share_prefix_by_hashes(
        target_slot_id=target.slot_id,
        target_epoch=target.epoch,
        block_hashes=("block-a", "block-b", "missing"),
        token_count=12,
    )

    assert shared == (0, 1)
    assert manager.slot_blocks(target.slot_id, target.epoch) == (0, 1)
    assert manager.seq_lens.tolist() == [10, 8, 0]
    assert manager.block_tables.tolist() == [[0, 1, 2, -1], [0, 1, -1, -1], [-1, -1, -1, -1]]
    assert manager.block_refcounts.tolist() == [2, 2, 1, 0, 0, 0, 0, 0]
    assert manager.free_block_count == 5

    manager.release_slot(source.slot_id, source.epoch)

    assert manager.prefix_pool_snapshot() == {"block-a": 0, "block-b": 1}
    assert manager.block_refcounts.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert manager.free_block_count == 6
    assert manager.decode_plan([target]).block_tables.tolist() == [[0, 1, -1, -1]]

    manager.release_slot(target.slot_id, target.epoch)

    assert manager.prefix_pool_snapshot() == {}
    assert manager.free_slot_count == 3
    assert manager.free_block_count == 8
    assert manager.block_refcounts.tolist() == [0, 0, 0, 0, 0, 0, 0, 0]


def test_paged_kv_manager_hash_prefix_share_stops_at_first_miss() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=6, block_size=4, max_blocks_per_seq=4)
    source = manager.allocate_slot()
    manager.set_seq_len(source.slot_id, source.epoch, 12)
    manager.publish_prefix_hashes(source.slot_id, source.epoch, ("a", "b", "c"))
    target = manager.allocate_slot()

    shared = manager.share_prefix_by_hashes(
        target_slot_id=target.slot_id,
        target_epoch=target.epoch,
        block_hashes=("a", "missing", "c"),
        token_count=12,
    )

    assert shared == (0,)
    assert manager.slot_blocks(target.slot_id, target.epoch) == (0,)
    assert manager.seq_lens.tolist() == [12, 4]
    assert manager.block_refcounts.tolist() == [2, 1, 1, 0, 0, 0]


def test_paged_kv_manager_hash_prefix_share_prunes_stale_pool_entries() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=4, block_size=4, max_blocks_per_seq=2)
    source = manager.allocate_slot()
    manager.set_seq_len(source.slot_id, source.epoch, 4)
    manager.publish_prefix_hashes(source.slot_id, source.epoch, ("a",))
    manager.release_slot(source.slot_id, source.epoch)

    target = manager.allocate_slot()
    shared = manager.share_prefix_by_hashes(
        target_slot_id=target.slot_id,
        target_epoch=target.epoch,
        block_hashes=("a",),
        token_count=4,
    )

    assert shared == ()
    assert manager.prefix_pool_snapshot() == {}
    assert manager.slot_blocks(target.slot_id, target.epoch) == ()


def test_slot_scheduler_builds_decode_plan_without_cache_objects() -> None:
    manager = PagedKVManager(max_slots=3, total_blocks=10, block_size=4, max_blocks_per_seq=4)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    slot_a = scheduler.admit(
        "req-a",
        prompt_length=3,
        max_new_tokens=4,
        sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": 1},
    )
    slot_b = scheduler.admit(
        "req-b",
        prompt_length=8,
        max_new_tokens=4,
        sampling_params={"temperature": 0.8, "top_p": 0.9, "top_k": 3},
    )
    scheduler.mark_prefilled("req-a", next_input_id=11)
    scheduler.mark_prefilled("req-b", next_input_id=22)

    plan = scheduler.build_decode_plan()

    assert plan.request_ids == ("req-a", "req-b")
    assert plan.input_ids.tolist() == [[11], [22]]
    assert plan.slot_ids.tolist() == [slot_a.slot_id, slot_b.slot_id]
    assert plan.epochs.tolist() == [slot_a.epoch, slot_b.epoch]
    assert plan.seq_lens.tolist() == [3, 8]
    assert plan.positions.tolist() == [3, 8]
    assert plan.block_tables.tolist() == [[0, -1, -1, -1], [1, 2, 3, -1]]
    assert plan.block_tables.shape == (2, 4)
    assert plan.sampling_params == (
        {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        {"temperature": 0.8, "top_p": 0.9, "top_k": 3},
    )
    assert plan.sampling_batch_params.temperature.tolist() == pytest.approx([0.0, 0.8])
    assert plan.sampling_batch_params.top_p.tolist() == pytest.approx([1.0, 0.9])
    assert plan.sampling_batch_params.top_k.tolist() == [1, 3]
    assert plan.sampling_batch_params.temperature_values == (0.0, 0.8)
    assert plan.sampling_batch_params.top_p_values == (1.0, 0.9)
    assert plan.sampling_batch_params.top_k_values == (1, 3)
    assert not hasattr(plan, "past_key_values")


def test_slot_scheduler_advances_chunked_prefill_metadata_before_decode() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    slot = scheduler.admit("req", prompt_length=5, max_new_tokens=2)

    assert slot.seq_len == 0
    assert slot.prefilled_tokens == 0
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == ()

    scheduler.advance_prefill("req", token_count=2)

    assert scheduler.get("req").seq_len == 2
    assert scheduler.get("req").prefilled_tokens == 2
    assert manager.seq_lens.tolist() == [2]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0,)

    scheduler.advance_prefill("req", token_count=3)
    scheduler.mark_prefilled("req", next_input_id=101)

    assert scheduler.get("req").seq_len == 5
    assert scheduler.get("req").prefilled_tokens == 5
    assert manager.seq_lens.tolist() == [5]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0, 1)

    plan = scheduler.build_decode_plan()
    assert plan.positions.tolist() == [5]
    assert plan.input_ids.tolist() == [[101]]


def test_slot_scheduler_advances_decode_and_appends_blocks_as_needed() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    slot = scheduler.admit("req", prompt_length=4, max_new_tokens=3)
    scheduler.mark_prefilled("req", next_input_id=101)

    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0,)

    plan = scheduler.build_decode_plan()
    assert plan.seq_lens.tolist() == [4]
    assert plan.positions.tolist() == [4]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0, 1)

    scheduler.advance_decode("req", next_input_id=102)

    assert scheduler.get("req").seq_len == 5
    assert scheduler.get("req").generated_tokens == 1
    assert scheduler.get("req").output_token_ids == [101]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0, 1)

    plan = scheduler.build_decode_plan()
    assert plan.input_ids.tolist() == [[102]]
    assert plan.positions.tolist() == [5]


def test_slot_scheduler_finish_cancel_release_and_reuse_slots() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    first = scheduler.admit("first", prompt_length=1, max_new_tokens=1)
    scheduler.mark_prefilled("first", next_input_id=7)
    finished = scheduler.advance_decode("first", finished=True)

    assert finished.status is SlotState.FINISHED
    assert finished.output_token_ids == [7]
    assert scheduler.active_count == 0
    assert manager.free_slot_count == 1
    assert manager.free_block_count == 4

    second = scheduler.admit("second", prompt_length=2, max_new_tokens=1)
    assert second.slot_id == first.slot_id
    assert second.epoch == first.epoch + 1

    cancelled = scheduler.cancel("second")
    assert cancelled.status is SlotState.CANCELLED
    assert scheduler.active_count == 0
    assert manager.free_slot_count == 1


def test_slot_scheduler_decode_plan_limit_and_explicit_request_selection() -> None:
    manager = PagedKVManager(max_slots=3, total_blocks=8, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    scheduler.admit("a", prompt_length=1, max_new_tokens=2)
    scheduler.admit("b", prompt_length=2, max_new_tokens=2)
    scheduler.admit("c", prompt_length=3, max_new_tokens=2)
    scheduler.mark_prefilled("a", next_input_id=10)
    scheduler.mark_prefilled("b", next_input_id=20)
    scheduler.mark_prefilled("c", next_input_id=30)

    limited = scheduler.build_decode_plan(limit=1)
    assert limited.request_ids == ("a",)

    explicit = scheduler.build_decode_plan(request_ids=["c", "b"])
    assert explicit.request_ids == ("c", "b")
    assert explicit.input_ids.tolist() == [[30], [20]]
    assert explicit.slot_ids.dtype == torch.int32
