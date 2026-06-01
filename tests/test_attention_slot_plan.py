from __future__ import annotations

import torch

from anna.core.hotpath_events import hotpath_event_recorder
from anna.model import ops as model_ops
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.runtime.service_metrics import AnnaServiceMetrics
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
        block_tables_are_global=plan.block_tables_are_global,
        seq_lens_are_global=plan.seq_lens_are_global,
        scaling=8**-0.5,
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    assert torch.equal(calls[0][0], plan.batch_block_tables)
    assert torch.equal(calls[0][1], plan.batch_seq_lens)


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
        block_tables_are_global=True,
        seq_lens_are_global=True,
        scaling=8**-0.5,
    )

    assert torch.equal(output, torch.ones_like(query))
    assert torch.equal(observed["page_table"], global_block_tables.index_select(0, slot_ids.to(dtype=torch.long)))
    assert torch.equal(observed["visible_lengths"], torch.tensor([11, 8], dtype=torch.long))


def test_slot_batch_decode_attention_uses_global_flags_when_row_count_matches_batch(
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
        return query_states

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _stub_paged_decode)

    query = torch.zeros(2, 4, 1, 8)
    key_pages = torch.randn(8, 2, 4, 8)
    value_pages = torch.randn(8, 2, 4, 8)
    global_block_tables = torch.tensor(
        [
            [10, -1, -1],
            [20, 21, -1],
        ],
        dtype=torch.int32,
    )
    global_seq_lens = torch.tensor([5, 2], dtype=torch.long)
    slot_ids = torch.tensor([1, 0], dtype=torch.int32)

    model_ops.paged_kv_slot_batch_decode_attention(
        query,
        key_pages,
        value_pages,
        global_block_tables,
        global_seq_lens,
        slot_ids=slot_ids,
        block_tables_are_global=True,
        seq_lens_are_global=True,
        scaling=8**-0.5,
    )

    assert torch.equal(observed["page_table"], torch.tensor([[20, 21, -1], [10, -1, -1]], dtype=torch.int32))
    assert torch.equal(observed["visible_lengths"], torch.tensor([2, 5], dtype=torch.long))


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
            block_tables_are_global=True,
            scaling=8**-0.5,
        )
    except RuntimeError as exc:
        assert "slot_ids are required" in str(exc)
    else:
        raise AssertionError("Expected slot_ids validation to fail.")


def test_slot_decode_visible_seq_lens_prefers_explicit_model_input_contract() -> None:
    seq_lens = torch.tensor([4, 7], dtype=torch.long)
    explicit = torch.tensor([6, 9], dtype=torch.long)
    slot_inputs = type("SlotInputs", (), {"visible_seq_lens": explicit})()

    visible = model_ops._slot_decode_visible_seq_lens(
        slot_inputs,
        seq_lens=seq_lens,
        decode_token_count=1,
    )

    assert torch.equal(visible, explicit)
    assert torch.equal(
        model_ops._slot_decode_visible_seq_lens(
            object(),
            seq_lens=seq_lens,
            decode_token_count=2,
        ),
        torch.tensor([6, 9], dtype=torch.long),
    )


def test_slot_decode_visible_seq_lens_selects_global_seq_lens() -> None:
    seq_lens = torch.tensor([4, 7, 11], dtype=torch.long)
    slot_inputs = type(
        "SlotInputs",
        (),
        {
            "seq_lens_are_global": True,
            "slot_ids": torch.tensor([2, 0], dtype=torch.int32),
        },
    )()

    visible = model_ops._slot_decode_visible_seq_lens(
        slot_inputs,
        seq_lens=seq_lens,
        decode_token_count=1,
    )

    assert torch.equal(visible, torch.tensor([12, 5], dtype=torch.long))


def test_physical_slot_decode_requires_initialized_kv_pages() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        vocab_size=64,
        max_position_embeddings=16,
        cache_block_size=4,
        layer_types=["full_attention"],
    )
    cache = model_ops.Qwen3DynamicCache(config)
    metrics = AnnaServiceMetrics()

    with hotpath_event_recorder(metrics):
        try:
            model_ops._require_slot_decode_physical_pages(cache, 0)
        except RuntimeError as exc:
            assert "physical_block_tables=True requires initialized physical KV pages" in str(exc)
            assert "legacy Qwen3DynamicCache.page_tables" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("physical slot decode without pages should fail")

    assert metrics.snapshot().attention_fallback_count == 1
