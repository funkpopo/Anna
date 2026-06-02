from __future__ import annotations

import pytest
import torch

from anna.runtime.paged_kv import PagedKVManager, SlotCapacityError, StaleSlotError
from anna.runtime.slot_scheduler import SlotScheduler, SlotState
from anna.sampling.params import SamplingBatchParams


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
    assert plan.seq_lens_are_global is True
    assert plan.positions_are_global is True
    assert plan.seq_lens.data_ptr() == manager.seq_lens.data_ptr()
    assert plan.positions.data_ptr() == manager.seq_lens.data_ptr()
    assert plan.batch_seq_lens.tolist() == [5]
    assert plan.batch_positions.tolist() == [5]
    assert plan.positions.data_ptr() == plan.seq_lens.data_ptr()
    assert plan.block_tables_are_global is True
    assert plan.block_tables.tolist() == [[0, 1, -1]]
    assert plan.batch_block_tables.tolist() == [[0, 1, -1]]
    assert plan.block_tables.data_ptr() == manager.block_tables.data_ptr()

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


def test_paged_kv_manager_rejects_duplicate_decode_slot_handles() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    handle = manager.allocate_slot()
    manager.set_seq_len(handle.slot_id, handle.epoch, 3)

    with pytest.raises(ValueError, match="duplicate slot ids"):
        manager.prepare_decode_capacity([handle, handle])
    with pytest.raises(ValueError, match="duplicate slot ids"):
        manager.decode_plan([handle, handle])

    assert manager.slot_blocks(handle.slot_id, handle.epoch) == (0,)
    assert manager.free_block_count == 3


def test_paged_kv_manager_external_mirror_validates_decode_append_capacity() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=2)

    with pytest.raises(ValueError, match="missing required prefix blocks"):
        manager.mirror_external_slot_tensors(
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[4],
            append_tokens=1,
        )

    manager.mirror_external_slot_tensors(
        slot_ids=[0],
        block_tables=[[0, 1]],
        seq_lens=[4],
        append_tokens=1,
    )

    assert manager.block_tables.tolist() == [[0, 1]]
    assert manager.seq_lens.tolist() == [4]
    assert manager.slot_active.tolist() == [True]


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
    target_plan = manager.decode_plan([target])
    assert target_plan.block_tables_are_global is True
    assert target_plan.batch_block_tables.tolist() == [[0, 1, -1, -1]]

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
    target_plan = manager.decode_plan([target])
    assert target_plan.block_tables_are_global is True
    assert target_plan.batch_block_tables.tolist() == [[0, 1, -1, -1]]

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
    assert plan.seq_lens_are_global is True
    assert plan.positions_are_global is True
    assert plan.seq_lens.data_ptr() == manager.seq_lens.data_ptr()
    assert plan.positions.data_ptr() == manager.seq_lens.data_ptr()
    assert plan.batch_seq_lens.tolist() == [3, 8]
    assert plan.batch_positions.tolist() == [3, 8]
    assert plan.positions.data_ptr() == plan.seq_lens.data_ptr()
    assert plan.block_tables_are_global is True
    assert plan.block_tables.tolist() == [
        [0, -1, -1, -1],
        [1, 2, 3, -1],
        [-1, -1, -1, -1],
    ]
    assert plan.block_tables.shape == (3, 4)
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


def test_slot_scheduler_reuses_sampling_batch_params_for_same_decode_batch(monkeypatch) -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=8, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)
    scheduler.admit(
        "a",
        prompt_length=3,
        max_new_tokens=2,
        sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": 1},
    )
    scheduler.admit(
        "b",
        prompt_length=5,
        max_new_tokens=2,
        sampling_params={"temperature": 0.8, "top_p": 0.9, "top_k": 3},
    )
    scheduler.mark_prefilled("a", next_input_id=11)
    scheduler.mark_prefilled("b", next_input_id=22)
    original_from_sampling_params = SamplingBatchParams.from_sampling_params
    calls: list[int] = []

    def _counting_from_sampling_params(sampling_params, *, device):
        calls.append(len(tuple(sampling_params)))
        return original_from_sampling_params(sampling_params, device=device)

    monkeypatch.setattr(
        SamplingBatchParams,
        "from_sampling_params",
        staticmethod(_counting_from_sampling_params),
    )

    first = scheduler.build_decode_plan(request_ids=["a", "b"])
    second = scheduler.build_decode_plan(request_ids=["a", "b"])

    assert first.sampling_batch_params is second.sampling_batch_params
    assert first.sampling_batch_params.top_k.tolist() == [1, 3]
    assert scheduler.sampling_batch_params_cache_stats() == {
        "entries": 1,
        "max_entries": 64,
        "hits": 1,
        "misses": 1,
        "evictions": 0,
    }
    assert calls == [2]


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
    assert plan.batch_positions.tolist() == [5]
    assert plan.input_ids.tolist() == [[101]]


def test_slot_scheduler_marks_prefilled_batch_from_device_token_tensor() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=6, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    scheduler.admit("a", prompt_length=3, max_new_tokens=2)
    scheduler.admit("b", prompt_length=5, max_new_tokens=2)
    scheduler.advance_prefill("b", token_count=2)

    marked = scheduler.mark_prefilled_batch(
        request_ids=["a", "b"],
        next_input_ids=torch.tensor([11, 22], dtype=torch.int32),
        next_input_host_ids=[11, 22],
    )

    assert tuple(slot.request_id for slot in marked) == ("a", "b")
    assert all(slot.status is SlotState.DECODING for slot in marked)
    assert scheduler.get("a").seq_len == 3
    assert scheduler.get("a").prefilled_tokens == 3
    assert scheduler.get("a").next_input_id == 11
    assert scheduler.get("a").next_input_token is not None
    assert scheduler.get("a").next_input_token.dtype == torch.long
    assert scheduler.get("a").next_input_token.device == manager.device
    assert scheduler.get("b").seq_len == 5
    assert scheduler.get("b").prefilled_tokens == 5
    assert scheduler.get("b").next_input_id == 22

    plan = scheduler.build_decode_plan(request_ids=["a", "b"])
    assert plan.input_ids.tolist() == [[11], [22]]
    assert plan.seq_lens_are_global is True
    assert plan.positions_are_global is True
    assert plan.batch_seq_lens.tolist() == [3, 5]
    assert plan.batch_positions.tolist() == [3, 5]


def test_slot_scheduler_prefill_batch_validation_does_not_partially_mark() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    scheduler.admit("a", prompt_length=3, max_new_tokens=2)
    scheduler.admit("b", prompt_length=3, max_new_tokens=2)

    with pytest.raises(ValueError, match="one token per prefilled request"):
        scheduler.mark_prefilled_batch(
            request_ids=["a", "b"],
            next_input_ids=[11, None],
        )

    assert scheduler.get("a").status is SlotState.WAITING_PREFILL
    assert scheduler.get("a").seq_len == 0
    assert scheduler.get("a").prefilled_tokens == 0
    assert scheduler.get("a").next_input_token is None
    assert scheduler.get("b").status is SlotState.WAITING_PREFILL
    assert scheduler.get("b").seq_len == 0
    assert scheduler.get("b").prefilled_tokens == 0
    assert scheduler.get("b").next_input_token is None
    assert manager.seq_lens.tolist() == [0, 0]

    with pytest.raises(ValueError, match="request_ids must be unique"):
        scheduler.mark_prefilled_batch(
            request_ids=["a", "a"],
            next_input_ids=torch.tensor([11, 12]),
        )


def test_slot_scheduler_advances_decode_and_appends_blocks_as_needed() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    slot = scheduler.admit("req", prompt_length=4, max_new_tokens=3)
    scheduler.mark_prefilled("req", next_input_id=101)

    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0,)

    plan = scheduler.build_decode_plan()
    assert plan.batch_seq_lens.tolist() == [4]
    assert plan.batch_positions.tolist() == [4]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0, 1)

    scheduler.advance_decode("req", next_input_id=102)

    assert scheduler.get("req").seq_len == 5
    assert scheduler.get("req").generated_tokens == 1
    assert scheduler.get("req").output_token_ids == [101]
    assert manager.slot_blocks(slot.slot_id, slot.epoch) == (0, 1)

    plan = scheduler.build_decode_plan()
    assert plan.input_ids.tolist() == [[102]]
    assert plan.batch_positions.tolist() == [5]


def test_slot_scheduler_keeps_next_decode_token_as_device_tensor() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    slot = scheduler.admit("req", prompt_length=2, max_new_tokens=3)
    first_token = torch.tensor(101, dtype=torch.int32)

    scheduler.mark_prefilled("req", next_input_id=first_token, next_input_host_id=101)

    active = scheduler.get("req")
    assert active.next_input_id == 101
    assert active.next_input_token is not None
    assert active.next_input_token.dtype == torch.long
    assert active.next_input_token.device == manager.device

    plan = scheduler.build_decode_plan()
    assert torch.equal(plan.input_ids, active.next_input_token)
    assert plan.input_ids.tolist() == [[101]]

    next_token = torch.tensor([102], dtype=torch.long)
    scheduler.advance_decode("req", next_input_id=next_token, next_input_host_id=102)

    active = scheduler.get("req")
    assert active.output_token_ids == [101]
    assert active.output_token_buffer is not None
    assert active.output_token_buffer.shape == (3,)
    assert active.output_token_count == 1
    assert active.output_tokens_view.tolist() == [101]
    assert active.next_input_id == 102
    assert active.next_input_token is not None
    assert torch.equal(active.next_input_token, torch.tensor([[102]], dtype=torch.long))

    plan = scheduler.build_decode_plan()
    assert torch.equal(plan.input_ids, active.next_input_token)


def test_slot_scheduler_allows_unstaged_device_token_without_host_output_mirror() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    scheduler.admit("req", prompt_length=1, max_new_tokens=2)
    scheduler.mark_prefilled("req", next_input_id=torch.tensor(7, dtype=torch.long))
    plan = scheduler.build_decode_plan()

    assert plan.input_ids.tolist() == [[7]]
    assert scheduler.get("req").next_input_id is None

    scheduler.advance_decode("req", next_input_id=torch.tensor(8, dtype=torch.long))

    assert scheduler.get("req").output_token_ids == []
    assert scheduler.get("req").output_tokens_view.tolist() == [7]
    assert scheduler.get("req").next_input_id is None
    assert scheduler.build_decode_plan().input_ids.tolist() == [[8]]


def test_slot_scheduler_advances_decode_batch_from_device_token_tensor() -> None:
    manager = PagedKVManager(max_slots=3, total_blocks=8, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=3)

    scheduler.admit("a", prompt_length=2, max_new_tokens=3)
    scheduler.admit("b", prompt_length=3, max_new_tokens=1)
    scheduler.admit("c", prompt_length=4, max_new_tokens=3)
    scheduler.mark_prefilled("a", next_input_id=10, next_input_host_id=10)
    scheduler.mark_prefilled("b", next_input_id=20, next_input_host_id=20)
    scheduler.mark_prefilled("c", next_input_id=30, next_input_host_id=30)
    plan = scheduler.build_decode_plan(request_ids=["a", "b", "c"])

    assert plan.input_ids.tolist() == [[10], [20], [30]]

    advanced = scheduler.advance_decode_batch(
        request_ids=plan.request_ids,
        next_input_ids=torch.tensor([11, 0, 31], dtype=torch.int32),
        next_input_host_ids=[11, None, 31],
        finished=[False, True, False],
    )

    assert tuple(slot.request_id for slot in advanced) == ("a", "b", "c")
    assert advanced[1].status is SlotState.FINISHED
    assert scheduler.active_count == 2
    assert scheduler.get("a").seq_len == 3
    assert scheduler.get("a").generated_tokens == 1
    assert scheduler.get("a").output_token_ids == [10]
    assert scheduler.get("a").output_tokens_view.tolist() == [10]
    assert scheduler.get("a").next_input_id == 11
    assert scheduler.get("a").next_input_token is not None
    assert scheduler.get("a").next_input_token.dtype == torch.long
    assert scheduler.get("a").next_input_token.device == manager.device
    assert scheduler.get("a").next_input_token.tolist() == [[11]]
    assert scheduler.get("c").next_input_id == 31
    assert scheduler.get("c").next_input_token is not None
    assert scheduler.get("c").next_input_token.tolist() == [[31]]

    next_plan = scheduler.build_decode_plan()
    assert next_plan.request_ids == ("a", "c")
    assert next_plan.input_ids.tolist() == [[11], [31]]
    assert next_plan.batch_positions.tolist() == [3, 5]


def test_slot_scheduler_decode_batch_validation_does_not_partially_advance() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=4, block_size=4, max_blocks_per_seq=2)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    scheduler.admit("a", prompt_length=1, max_new_tokens=3)
    scheduler.admit("b", prompt_length=1, max_new_tokens=3)
    scheduler.mark_prefilled("a", next_input_id=10, next_input_host_id=10)
    scheduler.mark_prefilled("b", next_input_id=20, next_input_host_id=20)
    scheduler.build_decode_plan(request_ids=["a", "b"])

    with pytest.raises(ValueError, match="one token per unfinished request"):
        scheduler.advance_decode_batch(
            request_ids=["a", "b"],
            next_input_ids=[11, None],
        )

    assert scheduler.get("a").seq_len == 1
    assert scheduler.get("a").generated_tokens == 0
    assert scheduler.get("a").output_token_ids == []
    assert scheduler.get("a").next_input_token is not None
    assert scheduler.get("a").next_input_token.tolist() == [[10]]
    assert scheduler.get("b").seq_len == 1
    assert scheduler.get("b").generated_tokens == 0
    assert scheduler.get("b").output_token_ids == []
    assert scheduler.get("b").next_input_token is not None
    assert scheduler.get("b").next_input_token.tolist() == [[20]]

    with pytest.raises(ValueError, match="request_ids must be unique"):
        scheduler.advance_decode_batch(
            request_ids=["a", "a"],
            next_input_ids=torch.tensor([11, 12]),
        )


def test_slot_scheduler_finish_cancel_release_and_reuse_slots() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    first = scheduler.admit("first", prompt_length=1, max_new_tokens=1)
    scheduler.mark_prefilled("first", next_input_id=7)
    finished = scheduler.advance_decode("first", finished=True)

    assert finished.status is SlotState.FINISHED
    assert finished.output_token_ids == [7]
    assert finished.output_tokens_view.tolist() == [7]
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


def test_slot_scheduler_decode_plan_rejects_duplicate_explicit_requests() -> None:
    manager = PagedKVManager(max_slots=2, total_blocks=8, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)

    scheduler.admit("a", prompt_length=1, max_new_tokens=2)
    scheduler.mark_prefilled("a", next_input_id=10)

    with pytest.raises(ValueError, match="request_ids must be unique"):
        scheduler.build_decode_plan(request_ids=["a", "a"])


def test_slot_scheduler_does_not_decode_slots_with_no_remaining_tokens() -> None:
    manager = PagedKVManager(max_slots=1, total_blocks=4, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=1)

    scheduler.admit("req", prompt_length=1, max_new_tokens=0)
    scheduler.mark_prefilled("req", next_input_id=10)

    assert scheduler.ready_decode_slots() == ()
    with pytest.raises(ValueError, match="No decode-ready slots"):
        scheduler.build_decode_plan()
    with pytest.raises(ValueError, match="no remaining decode tokens"):
        scheduler.build_decode_plan(request_ids=["req"])
    with pytest.raises(ValueError, match="no remaining decode tokens"):
        scheduler.advance_decode("req", finished=True)
