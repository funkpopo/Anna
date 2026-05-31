from __future__ import annotations

import torch

from anna.model import ops as model_ops
from anna.runtime.paged_kv import PagedKVManager
from anna.runtime.slot_scheduler import SlotScheduler


def test_slot_batch_decode_attention_accepts_selected_decode_plan(
    monkeypatch,
) -> None:
    manager = PagedKVManager(max_slots=3, total_blocks=8, block_size=4, max_blocks_per_seq=3)
    scheduler = SlotScheduler(manager, max_batch_size=2)
    scheduler.admit("a", prompt_length=3, max_new_tokens=2)
    scheduler.admit("b", prompt_length=5, max_new_tokens=2)
    scheduler.mark_prefilled("a", next_input_id=10)
    scheduler.mark_prefilled("b", next_input_id=20)
    plan = scheduler.build_decode_plan()

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _stub_paged_decode(
        query_states: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        page_table: torch.Tensor,
        *,
        scaling: float,
        visible_lengths: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key_pages, value_pages, scaling, gate
        calls.append((page_table.clone(), visible_lengths.clone()))
        return torch.zeros_like(query_states)

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _stub_paged_decode)

    query = torch.randn(2, 4, 1, 8)
    key_pages = torch.randn(8, 2, 4, 8)
    value_pages = torch.randn(8, 2, 4, 8)

    output = model_ops.paged_kv_slot_batch_decode_attention(
        query,
        key_pages,
        value_pages,
        plan.block_tables,
        plan.seq_lens,
        slot_ids=plan.slot_ids,
        scaling=8**-0.5,
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    assert torch.equal(calls[0][0], plan.block_tables)
    assert torch.equal(calls[0][1], plan.seq_lens)


def test_slot_batch_decode_attention_selects_from_global_slot_tables(
    monkeypatch,
) -> None:
    observed: dict[str, torch.Tensor] = {}

    def _stub_paged_decode(
        query_states: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        page_table: torch.Tensor,
        *,
        scaling: float,
        visible_lengths: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key_pages, value_pages, scaling, gate
        observed["page_table"] = page_table.clone()
        observed["visible_lengths"] = visible_lengths.clone()
        return query_states + 1

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _stub_paged_decode)

    query = torch.zeros(2, 4, 1, 8)
    key_pages = torch.randn(16, 2, 4, 8)
    value_pages = torch.randn(16, 2, 4, 8)
    global_block_tables = torch.tensor(
        [
            [0, -1, -1],
            [1, 2, -1],
            [3, 4, 5],
            [6, -1, -1],
        ],
        dtype=torch.int32,
    )
    global_seq_lens = torch.tensor([1, 8, 11, 2], dtype=torch.long)
    slot_ids = torch.tensor([2, 1], dtype=torch.int32)

    output = model_ops.paged_kv_slot_batch_decode_attention(
        query,
        key_pages,
        value_pages,
        global_block_tables,
        global_seq_lens,
        slot_ids=slot_ids,
        scaling=8**-0.5,
    )

    assert torch.equal(output, torch.ones_like(query))
    assert torch.equal(observed["page_table"], global_block_tables.index_select(0, slot_ids.to(dtype=torch.long)))
    assert torch.equal(observed["visible_lengths"], torch.tensor([11, 8], dtype=torch.long))


def test_slot_batch_decode_attention_requires_slot_ids_for_global_tables() -> None:
    query = torch.zeros(2, 4, 1, 8)
    key_pages = torch.randn(16, 2, 4, 8)
    value_pages = torch.randn(16, 2, 4, 8)
    global_block_tables = torch.zeros(4, 3, dtype=torch.int32)
    global_seq_lens = torch.zeros(4, dtype=torch.long)

    try:
        model_ops.paged_kv_slot_batch_decode_attention(
            query,
            key_pages,
            value_pages,
            global_block_tables,
            global_seq_lens,
            scaling=8**-0.5,
        )
    except RuntimeError as exc:
        assert "slot_ids are required" in str(exc)
    else:
        raise AssertionError("Expected slot_ids validation to fail.")
