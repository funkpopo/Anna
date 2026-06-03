from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anna.core.hotpath_events import hotpath_event_recorder
from anna.model.ops import (
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    grouped_query_attention,
    write_slot_decode_kv_to_pages,
    write_slot_prefill_kv_to_pages,
)
from anna.model.quantization import XPUInt4Linear
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.runtime.hotpath_guard import DEFAULT_ALLOWLIST, scan_hotpath_files, summarize_findings, unexpected_findings
from anna.runtime.paged_kv import PagedKVManager
from anna.runtime.service_metrics import AnnaServiceMetrics
from anna.runtime.slot_model_runner import SlotDecodeModelInputs, SlotModelRunner
from anna.runtime.slot_scheduler import SlotScheduler
from anna.sampling.params import SamplingBatchParams, SamplingBatchParamsCache
from anna.sampling.sampler import sample_next_token, sample_next_token_from_candidates


def _time_ms(fn: Callable[[], object], *, iters: int) -> float:
    started = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
    return (time.perf_counter() - started) * 1000.0 / max(1, iters)


def _resolve_kv_heads(num_heads: int, requested_kv_heads: int) -> int:
    if requested_kv_heads > 0:
        kv_heads = int(requested_kv_heads)
    elif num_heads % 2 == 0:
        kv_heads = max(1, num_heads // 2)
    else:
        kv_heads = max(1, num_heads)
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads must be divisible by kv_heads: heads={num_heads} kv_heads={kv_heads}")
    return kv_heads


def _repeat_kv_for_bench(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _pack_paged_kv_for_bench(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_kv_heads, key_len, head_dim = key.shape
    pages_per_batch = max(1, (key_len + block_size - 1) // block_size)
    total_pages = batch_size * pages_per_batch
    key_pages = key.new_zeros((total_pages, num_kv_heads, block_size, head_dim))
    value_pages = value.new_zeros((total_pages, num_kv_heads, block_size, head_dim))
    block_tables = torch.full((batch_size, pages_per_batch), -1, dtype=torch.int32, device=key.device)
    for batch_idx in range(batch_size):
        for block_idx in range(pages_per_batch):
            page_id = batch_idx * pages_per_batch + block_idx
            start = block_idx * block_size
            take = min(block_size, max(0, key_len - start))
            if take > 0:
                key_pages[page_id, :, :take, :].copy_(key[batch_idx, :, start : start + take, :])
                value_pages[page_id, :, :take, :].copy_(value[batch_idx, :, start : start + take, :])
            block_tables[batch_idx, block_idx] = page_id
    return key_pages, value_pages, block_tables


def _paged_decode_reference_for_bench(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = query.shape
    num_kv_heads = key_pages.shape[1]
    num_key_value_groups = max(1, num_heads // num_kv_heads)
    outputs = []
    for batch_idx in range(batch_size):
        visible = int(seq_lens[batch_idx].item())
        blocks = block_tables[batch_idx]
        key_rows = []
        value_rows = []
        remaining = visible
        for block_id in blocks.tolist():
            if remaining <= 0 or block_id < 0:
                break
            take = min(remaining, key_pages.shape[2])
            key_rows.append(key_pages[int(block_id), :, :take, :])
            value_rows.append(value_pages[int(block_id), :, :take, :])
            remaining -= take
        if key_rows:
            key_states = torch.cat(key_rows, dim=1).unsqueeze(0)
            value_states = torch.cat(value_rows, dim=1).unsqueeze(0)
        else:
            key_states = key_pages.new_zeros((1, num_kv_heads, 1, head_dim))
            value_states = value_pages.new_zeros((1, num_kv_heads, 1, head_dim))
        repeated_key = _repeat_kv_for_bench(key_states, num_key_value_groups)
        repeated_value = _repeat_kv_for_bench(value_states, num_key_value_groups)
        scores = torch.matmul(query[batch_idx : batch_idx + 1], repeated_key.transpose(-1, -2)) * scaling
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        outputs.append(torch.matmul(probs, repeated_value))
    return torch.cat(outputs, dim=0)


def _tiny_config(*, layers: int, heads: int, head_dim: int, block_size: int) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=heads * head_dim,
        intermediate_size=heads * head_dim * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        vocab_size=4096,
        eos_token_id=0,
        pad_token_id=0,
        cache_block_size=block_size,
        layer_types=["full_attention"] * layers,
    )


def _make_cache(
    *,
    config: Qwen3_5TextConfig,
    allocator: Qwen3PageAllocator,
    seq_len: int,
) -> Qwen3DynamicCache:
    cache = Qwen3DynamicCache(config, allocator=allocator, batch_size=1)
    key = torch.zeros((1, config.num_key_value_heads, seq_len, config.head_dim), dtype=torch.float32)
    value = torch.zeros_like(key)
    for layer_idx in range(config.num_hidden_layers):
        cache.update(key, value, layer_idx=layer_idx)
    return cache


def _bench_cache_stack_split(
    *,
    batch_size: int,
    seq_len: int,
    layers: int,
    heads: int,
    head_dim: int,
    block_size: int,
    iters: int,
) -> dict[str, Any]:
    config = _tiny_config(layers=layers, heads=heads, head_dim=head_dim, block_size=block_size)
    allocator = Qwen3PageAllocator(config)
    caches = [_make_cache(config=config, allocator=allocator, seq_len=seq_len) for _ in range(batch_size)]

    stacked = Qwen3DynamicCache.stack(caches, config, clone_turboquant_rows=False)
    split = stacked.split_batch(clone_turboquant_rows=False)
    assert len(split) == batch_size

    def stack_once() -> Qwen3DynamicCache:
        return Qwen3DynamicCache.stack(caches, config, clone_turboquant_rows=False)

    def split_once() -> list[Qwen3DynamicCache]:
        return stacked.split_batch(clone_turboquant_rows=False)

    stack_ms = _time_ms(stack_once, iters=iters)
    split_ms = _time_ms(split_once, iters=iters)
    for cache in caches:
        cache.release()
    stacked.release()
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "stack_ms": stack_ms,
        "split_ms": split_ms,
    }


def _bench_slot_decode_plan(
    *,
    batch_size: int,
    seq_len: int,
    block_size: int,
    max_blocks_per_seq: int,
    iters: int,
) -> dict[str, Any]:
    required_blocks = 0 if seq_len <= 0 else (seq_len + block_size - 1) // block_size
    resolved_max_blocks = max(max_blocks_per_seq, required_blocks + 1, 1)
    max_slots = batch_size + 1
    manager = PagedKVManager(
        max_slots=max_slots,
        total_blocks=max_slots * resolved_max_blocks,
        block_size=block_size,
        max_blocks_per_seq=resolved_max_blocks,
        device="cpu",
    )
    scheduler = SlotScheduler(manager, max_batch_size=batch_size)
    request_ids = tuple(f"bench-{idx}" for idx in range(batch_size))
    for idx, request_id in enumerate(request_ids):
        scheduler.admit(
            request_id,
            prompt_length=seq_len,
            max_new_tokens=4,
            sampling_params={
                "temperature": 0.0 if idx % 2 == 0 else 0.7,
                "top_p": 1.0 if idx % 2 == 0 else 0.8,
                "top_k": 1 if idx % 2 == 0 else 32,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        )
        scheduler.mark_prefilled(request_id, next_input_id=idx + 1)

    plan = scheduler.build_decode_plan(request_ids=request_ids)
    model_inputs = SlotDecodeModelInputs.from_plan(plan)
    boundary = model_inputs.boundary_summary()
    decode_key_pages = torch.zeros((manager.total_blocks, 1, manager.block_size, 8), dtype=torch.float32)
    decode_value_pages = torch.zeros_like(decode_key_pages)
    decode_key_states = torch.randn(batch_size, 1, 1, 8)
    decode_value_states = torch.randn_like(decode_key_states)
    decode_visible_seq_lens = write_slot_decode_kv_to_pages(
        decode_key_states,
        decode_value_states,
        decode_key_pages,
        decode_value_pages,
        plan.block_tables,
        plan.seq_lens,
        slot_ids=plan.slot_ids,
        block_tables_are_global=plan.block_tables_are_global,
        seq_lens_are_global=plan.seq_lens_are_global,
    )
    prefill_chunk_tokens = max(1, min(block_size + 1, max(seq_len, 1)))
    prefill_base_len = max(0, seq_len - prefill_chunk_tokens)
    prefill_seq_lens = plan.seq_lens.clone()
    prefill_seq_lens.index_fill_(0, plan.slot_ids.to(dtype=torch.long), prefill_base_len)
    prefill_key_pages = torch.zeros_like(decode_key_pages)
    prefill_value_pages = torch.zeros_like(decode_value_pages)
    prefill_key_states = torch.randn(batch_size, 1, prefill_chunk_tokens, 8)
    prefill_value_states = torch.randn_like(prefill_key_states)
    prefill_visible_seq_lens = write_slot_prefill_kv_to_pages(
        prefill_key_states,
        prefill_value_states,
        prefill_key_pages,
        prefill_value_pages,
        plan.block_tables,
        prefill_seq_lens,
        slot_ids=plan.slot_ids,
        block_tables_are_global=plan.block_tables_are_global,
        seq_lens_are_global=plan.seq_lens_are_global,
    )
    text_config = Qwen3_5TextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        vocab_size=32,
        cache_block_size=block_size,
        layer_types=["full_attention"],
    )
    physical_runner = SlotModelRunner.from_text_config(
        text_config,
        device="cpu",
        max_slots=max_slots,
        total_blocks=max_slots * resolved_max_blocks,
        max_blocks_per_seq=resolved_max_blocks,
        max_batch_size=batch_size,
    )
    physical_runner.allocate_physical_kv_page_bank(dtype=torch.float32, num_layers=1)
    physical_prompt_len = max(seq_len, prefill_chunk_tokens)
    physical_base_len = max(0, physical_prompt_len - prefill_chunk_tokens)
    for request_id in request_ids:
        physical_runner.admit_prefill(request_id, prompt_length=physical_prompt_len, max_new_tokens=4)
        if physical_base_len > 0:
            physical_runner.advance_prefill(request_id, token_count=physical_base_len)
    physical_prefill_inputs = physical_runner.build_prefill_inputs(
        request_ids=request_ids,
        input_ids=torch.zeros((batch_size, prefill_chunk_tokens), dtype=torch.long),
        physical_block_tables=True,
    )
    prefill_boundary = physical_prefill_inputs.boundary_summary()
    assert plan.block_tables_are_global is True
    assert plan.seq_lens_are_global is True
    assert plan.positions_are_global is True
    assert plan.block_tables.shape == (max_slots, resolved_max_blocks)
    assert plan.seq_lens.shape == (max_slots,)
    assert plan.positions.shape == (max_slots,)
    assert plan.batch_seq_lens.shape == (batch_size,)
    assert plan.batch_positions.shape == (batch_size,)
    assert plan.sampling_batch_params.batch_size == batch_size
    assert decode_visible_seq_lens.shape == (batch_size,)
    assert prefill_visible_seq_lens.shape == (batch_size,)

    plan_ms = _time_ms(lambda: scheduler.build_decode_plan(request_ids=request_ids), iters=iters)
    slot_decode_kv_write_ms = _time_ms(
        lambda: write_slot_decode_kv_to_pages(
            decode_key_states,
            decode_value_states,
            decode_key_pages,
            decode_value_pages,
            plan.block_tables,
            plan.seq_lens,
            slot_ids=plan.slot_ids,
            block_tables_are_global=plan.block_tables_are_global,
            seq_lens_are_global=plan.seq_lens_are_global,
        ),
        iters=iters,
    )
    slot_prefill_kv_write_ms = _time_ms(
        lambda: write_slot_prefill_kv_to_pages(
            prefill_key_states,
            prefill_value_states,
            prefill_key_pages,
            prefill_value_pages,
            plan.block_tables,
            prefill_seq_lens,
            slot_ids=plan.slot_ids,
            block_tables_are_global=plan.block_tables_are_global,
            seq_lens_are_global=plan.seq_lens_are_global,
        ),
        iters=iters,
    )
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "max_blocks_per_seq": resolved_max_blocks,
        "plan_ms": plan_ms,
        "slot_kv_write_ms": slot_decode_kv_write_ms,
        "slot_decode_kv_write_ms": slot_decode_kv_write_ms,
        "slot_prefill_kv_write_ms": slot_prefill_kv_write_ms,
        "slot_prefill_chunk_tokens": prefill_chunk_tokens,
        "active_batch_rows": batch_size,
        "block_tables_are_global": bool(plan.block_tables_are_global),
        "seq_lens_are_global": bool(plan.seq_lens_are_global),
        "positions_are_global": bool(plan.positions_are_global),
        "block_table_rows": int(plan.block_tables.shape[0]),
        "block_table_cols": int(plan.block_tables.shape[1]),
        "seq_lens_rows": int(plan.seq_lens.shape[0]),
        "positions_rows": int(plan.positions.shape[0]),
        "decode_boundary_batch_size": int(boundary["batch_size"]),
        "decode_boundary_token_count": int(boundary["decode_token_count"]),
        "decode_boundary_contains_cache_objects": bool(boundary["contains_cache_objects"]),
        "decode_boundary_sampling_batch_params": bool(boundary["sampling_batch_params"]),
        "decode_boundary_block_table_ownership": str(boundary["block_table_ownership"]),
        "decode_boundary_owns_physical_kv_pages": bool(boundary["owns_physical_kv_pages"]),
        "decode_boundary_physical_kv_layer_count": int(boundary["physical_kv_layer_count"]),
        "prefill_boundary_physical_block_tables": bool(prefill_boundary["physical_block_tables"]),
        "prefill_boundary_block_table_ownership": str(prefill_boundary["block_table_ownership"]),
        "prefill_boundary_owns_physical_kv_pages": bool(prefill_boundary["owns_physical_kv_pages"]),
        "prefill_boundary_physical_kv_layer_count": int(prefill_boundary["physical_kv_layer_count"]),
        "slot_owned_output_token_buffer": all(
            slot.output_token_buffer is not None and slot.output_token_buffer.device == manager.device
            for slot in scheduler.ready_decode_slots()
        ),
        "slot_owned_kv_write_helper": True,
        "slot_owned_decode_kv_write_helper": True,
        "slot_owned_prefill_kv_write_helper": True,
        "slot_owned_kv_page_bank_available": True,
        "slot_owned_prefill_writes": True,
        "slot_owned_decode_writes": False,
        "slot_owned_kv_writes": False,
        "legacy_cache_object_required_for_forward": True,
        "slot_kv_visible_seq_lens_shape": tuple(int(dim) for dim in decode_visible_seq_lens.shape),
        "slot_decode_kv_visible_seq_lens_shape": tuple(int(dim) for dim in decode_visible_seq_lens.shape),
        "slot_prefill_kv_visible_seq_lens_shape": tuple(int(dim) for dim in prefill_visible_seq_lens.shape),
    }


def _summarize_scheduler_kv_overhead(
    *,
    cache_stack_split: dict[str, Any],
    slot_decode_plan: dict[str, Any],
) -> dict[str, Any]:
    stack_ms = float(cache_stack_split["stack_ms"])
    split_ms = float(cache_stack_split["split_ms"])
    stack_split_ms = stack_ms + split_ms
    slot_plan_ms = float(slot_decode_plan["plan_ms"])
    ratio = None if slot_plan_ms <= 0.0 else stack_split_ms / slot_plan_ms
    batch_size = int(cache_stack_split["batch_size"])
    layers = int(cache_stack_split["layers"])
    active_rows = int(slot_decode_plan.get("active_batch_rows", batch_size))
    global_rows = int(slot_decode_plan["block_table_rows"])
    block_table_cols = int(slot_decode_plan["block_table_cols"])
    seq_lens_rows = int(slot_decode_plan.get("seq_lens_rows", active_rows))
    positions_rows = int(slot_decode_plan.get("positions_rows", active_rows))
    return {
        "batch_size": batch_size,
        "seq_len": int(cache_stack_split["seq_len"]),
        "layers": layers,
        "block_size": int(cache_stack_split["block_size"]),
        "legacy_cache_objects_per_step": batch_size,
        "legacy_layer_rows_touched_per_step": batch_size * layers,
        "slot_plan_active_rows": active_rows,
        "slot_plan_rows": global_rows,
        "slot_plan_block_tables_are_global": bool(slot_decode_plan.get("block_tables_are_global", False)),
        "slot_plan_seq_lens_are_global": bool(slot_decode_plan.get("seq_lens_are_global", False)),
        "slot_plan_positions_are_global": bool(slot_decode_plan.get("positions_are_global", False)),
        "slot_plan_block_table_cols": block_table_cols,
        "slot_plan_block_table_entries": active_rows * block_table_cols,
        "slot_plan_global_block_table_entries": global_rows * block_table_cols,
        "slot_plan_seq_lens_entries": active_rows,
        "slot_plan_global_seq_lens_entries": seq_lens_rows,
        "slot_plan_positions_entries": active_rows,
        "slot_plan_global_positions_entries": positions_rows,
        "slot_decode_boundary_contains_cache_objects": bool(
            slot_decode_plan.get("decode_boundary_contains_cache_objects", True)
        ),
        "slot_decode_boundary_sampling_batch_params": bool(
            slot_decode_plan.get("decode_boundary_sampling_batch_params", False)
        ),
        "slot_decode_boundary_block_table_ownership": str(
            slot_decode_plan.get("decode_boundary_block_table_ownership", "unknown")
        ),
        "slot_decode_boundary_owns_physical_kv_pages": bool(
            slot_decode_plan.get("decode_boundary_owns_physical_kv_pages", False)
        ),
        "slot_decode_boundary_physical_kv_layer_count": int(
            slot_decode_plan.get("decode_boundary_physical_kv_layer_count", 0)
        ),
        "slot_prefill_boundary_physical_block_tables": bool(
            slot_decode_plan.get("prefill_boundary_physical_block_tables", False)
        ),
        "slot_prefill_boundary_block_table_ownership": str(
            slot_decode_plan.get("prefill_boundary_block_table_ownership", "unknown")
        ),
        "slot_prefill_boundary_owns_physical_kv_pages": bool(
            slot_decode_plan.get("prefill_boundary_owns_physical_kv_pages", False)
        ),
        "slot_prefill_boundary_physical_kv_layer_count": int(
            slot_decode_plan.get("prefill_boundary_physical_kv_layer_count", 0)
        ),
        "slot_owned_output_token_buffer": bool(slot_decode_plan.get("slot_owned_output_token_buffer", False)),
        "slot_owned_kv_write_helper": bool(slot_decode_plan.get("slot_owned_kv_write_helper", False)),
        "slot_owned_decode_kv_write_helper": bool(
            slot_decode_plan.get(
                "slot_owned_decode_kv_write_helper",
                slot_decode_plan.get("slot_owned_kv_write_helper", False),
            )
        ),
        "slot_owned_prefill_kv_write_helper": bool(
            slot_decode_plan.get("slot_owned_prefill_kv_write_helper", False)
        ),
        "slot_owned_kv_page_bank_available": bool(slot_decode_plan.get("slot_owned_kv_page_bank_available", False)),
        "slot_owned_prefill_writes": bool(slot_decode_plan.get("slot_owned_prefill_writes", False)),
        "slot_owned_decode_writes": bool(slot_decode_plan.get("slot_owned_decode_writes", False)),
        "slot_owned_kv_writes": bool(slot_decode_plan.get("slot_owned_kv_writes", False)),
        "legacy_cache_object_required_for_forward": bool(
            slot_decode_plan.get("legacy_cache_object_required_for_forward", True)
        ),
        "stack_ms": stack_ms,
        "split_ms": split_ms,
        "stack_split_ms": stack_split_ms,
        "slot_plan_ms": slot_plan_ms,
        "slot_kv_write_ms": float(slot_decode_plan.get("slot_kv_write_ms", 0.0)),
        "slot_decode_kv_write_ms": float(
            slot_decode_plan.get("slot_decode_kv_write_ms", slot_decode_plan.get("slot_kv_write_ms", 0.0))
        ),
        "slot_prefill_kv_write_ms": float(slot_decode_plan.get("slot_prefill_kv_write_ms", 0.0)),
        "stack_split_to_slot_plan_ratio": ratio,
    }


def _bench_paged_gqa_decode(
    *,
    batch_size: int,
    seq_len: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    iters: int,
) -> dict[str, Any]:
    resolved_kv_heads = _resolve_kv_heads(heads, kv_heads)
    scaling = head_dim**-0.5
    query = torch.randn(batch_size, heads, 1, head_dim)
    key = torch.randn(batch_size, resolved_kv_heads, seq_len, head_dim)
    value = torch.randn_like(key)
    key_pages, value_pages, block_tables = _pack_paged_kv_for_bench(key, value, block_size=block_size)
    global_block_tables = torch.cat(
        [
            torch.full_like(block_tables, -1),
            block_tables,
        ],
        dim=0,
    )
    slot_ids = torch.arange(batch_size, dtype=torch.long) + batch_size
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    global_seq_lens = torch.cat([torch.zeros_like(seq_lens), seq_lens], dim=0)
    num_key_value_groups = max(1, heads // resolved_kv_heads)

    def materialized_decode() -> torch.Tensor:
        repeated_key = _repeat_kv_for_bench(key, num_key_value_groups)
        repeated_value = _repeat_kv_for_bench(value, num_key_value_groups)
        scores = torch.matmul(query, repeated_key.transpose(-1, -2)) * scaling
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        return torch.matmul(probs, repeated_value)

    def paged_reference_decode() -> torch.Tensor:
        selected_tables = global_block_tables.index_select(0, slot_ids)
        selected_lens = global_seq_lens.index_select(0, slot_ids)
        return _paged_decode_reference_for_bench(
            query,
            key_pages,
            value_pages,
            selected_tables,
            selected_lens,
            scaling=scaling,
        )

    dense_output = materialized_decode()
    paged_output = paged_reference_decode()
    error = (paged_output.float() - dense_output.float()).abs()
    materialized_ms = _time_ms(materialized_decode, iters=iters)
    paged_reference_ms = _time_ms(paged_reference_decode, iters=iters)
    return {
        "backend": "cpu_reference_smoke",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "kv_heads": resolved_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "pages": int(key_pages.shape[0]),
        "block_table_rows": int(global_block_tables.shape[0]),
        "block_table_cols": int(global_block_tables.shape[1]),
        "materialized_ms": materialized_ms,
        "paged_reference_ms": paged_reference_ms,
        "max_abs_diff": float(error.max().item()),
        "mean_abs_diff": float(error.mean().item()),
    }


def _bench_prefill_attention(
    *,
    batch_size: int,
    seq_len: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    iters: int,
) -> dict[str, Any]:
    resolved_kv_heads = _resolve_kv_heads(heads, kv_heads)
    scaling = head_dim**-0.5
    query = torch.randn(batch_size, heads, seq_len, head_dim)
    key = torch.randn(batch_size, resolved_kv_heads, seq_len, head_dim)
    value = torch.randn_like(key)
    num_key_value_groups = max(1, heads // resolved_kv_heads)
    causal_mask = torch.arange(seq_len)[:, None] < torch.arange(seq_len)[None, :]
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    def sdpa_materialized() -> torch.Tensor:
        repeated_key = _repeat_kv_for_bench(key, num_key_value_groups)
        repeated_value = _repeat_kv_for_bench(value, num_key_value_groups)
        return F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            dropout_p=0.0,
            is_causal=seq_len > 1,
        )

    def grouped_fallback() -> torch.Tensor:
        return grouped_query_attention(
            query,
            key,
            value,
            scaling=scaling,
            causal_mask=causal_mask,
        )

    sdpa_output = sdpa_materialized()
    grouped_output = grouped_fallback()
    error = (grouped_output.float() - sdpa_output.float()).abs()
    sdpa_ms = _time_ms(sdpa_materialized, iters=iters)
    grouped_ms = _time_ms(grouped_fallback, iters=iters)
    return {
        "backend": "torch_cpu_smoke",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "kv_heads": resolved_kv_heads,
        "head_dim": head_dim,
        "sdpa_materialized_ms": sdpa_ms,
        "grouped_fallback_ms": grouped_ms,
        "max_abs_diff": float(error.max().item()),
        "mean_abs_diff": float(error.mean().item()),
    }


def _bench_sampler(*, vocab_size: int, candidates: int, iters: int) -> dict[str, Any]:
    logits = torch.randn(vocab_size)
    history = torch.arange(0, min(64, vocab_size), dtype=torch.long)
    candidate_logits = torch.randn(candidates)
    candidate_token_ids = torch.arange(candidates, dtype=torch.long)
    candidate_history = torch.arange(0, max(1, candidates // 2), dtype=torch.long)
    metrics = AnnaServiceMetrics()

    def full_vocab() -> torch.Tensor:
        with hotpath_event_recorder(metrics):
            return sample_next_token(
                logits,
                generated_ids=history,
                temperature=0.7,
                top_p=0.8,
                top_k=0,
                min_p=0.0,
                presence_penalty=0.1,
                repetition_penalty=1.05,
            )

    def candidate() -> torch.Tensor:
        return sample_next_token_from_candidates(
            candidate_logits,
            candidate_token_ids,
            temperature=0.7,
            top_p=0.8,
            min_p=0.0,
        )

    def candidate_penalty() -> torch.Tensor:
        return sample_next_token_from_candidates(
            candidate_logits,
            candidate_token_ids,
            temperature=0.7,
            top_p=0.8,
            top_k=max(1, min(4, candidates)),
            min_p=0.0,
            generated_ids=candidate_history,
            presence_penalty=0.1,
            repetition_penalty=1.05,
        )

    full_vocab_ms = _time_ms(full_vocab, iters=iters)
    candidate_ms = _time_ms(candidate, iters=iters)
    candidate_penalty_ms = _time_ms(candidate_penalty, iters=iters)
    snapshot = metrics.snapshot()
    return {
        "vocab_size": vocab_size,
        "candidates": candidates,
        "full_vocab_ms": full_vocab_ms,
        "candidate_ms": candidate_ms,
        "candidate_penalty_ms": candidate_penalty_ms,
        "sampler_full_vocab_sort_count": snapshot.sampler_full_vocab_sort_count,
        "sampler_full_vocab_fallback_count": snapshot.sampler_full_vocab_fallback_count,
        "sampler_full_vocab_fallback_reasons": snapshot.sampler_full_vocab_fallback_reasons,
    }


def _bench_sampling_params_cache(*, batch_size: int, iters: int) -> dict[str, Any]:
    resolved_batch_size = max(1, int(batch_size))
    row_params = tuple(
        {
            "temperature": 0.0 if row_idx % 2 == 0 else 0.7,
            "top_p": 1.0 if row_idx % 2 == 0 else 0.8,
            "top_k": 1 if row_idx % 2 == 0 else 32,
            "min_p": 0.0,
            "presence_penalty": 0.0 if row_idx % 3 else 0.1,
            "repetition_penalty": 1.0 if row_idx % 3 else 1.05,
        }
        for row_idx in range(resolved_batch_size)
    )
    equivalent_row_params = tuple(dict(params) for params in row_params)
    cache = SamplingBatchParamsCache(max_entries=4)
    first = cache.get(row_params, device="cpu")
    normalized_reuse = cache.get(equivalent_row_params, device=torch.device("cpu")) is first

    def uncached() -> SamplingBatchParams:
        return SamplingBatchParams.from_sampling_params(row_params, device="cpu")

    def cached() -> SamplingBatchParams:
        return cache.get(row_params, device="cpu")

    uncached_ms = _time_ms(uncached, iters=iters)
    cached_ms = _time_ms(cached, iters=iters)
    cached_identity_stable = cached() is first
    cache_stats = cache.stats()
    ratio = None if cached_ms <= 0.0 else uncached_ms / cached_ms
    return {
        "batch_size": resolved_batch_size,
        "cache_entries": cache_stats["entries"],
        "cache_max_entries": cache_stats["max_entries"],
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_evictions": cache_stats["evictions"],
        "cached_batch_size": first.batch_size,
        "greedy_rows": len(first.greedy_rows),
        "sample_rows": len(first.sample_rows),
        "penalty_rows": len(first.penalty_rows),
        "normalized_key_reuse": bool(normalized_reuse),
        "cached_identity_stable": bool(cached_identity_stable),
        "uncached_ms": uncached_ms,
        "cached_ms": cached_ms,
        "uncached_to_cached_ratio": ratio,
    }


def _bench_turboquant_metadata_view(*, batch_size: int, layers: int, iters: int) -> dict[str, Any]:
    class _FakeTurboQuantRow:
        def __init__(self, length: int) -> None:
            self.length = int(length)
            self.device = torch.device("cpu")

    resolved_batch_size = max(1, int(batch_size))
    resolved_layers = max(1, int(layers))
    config = _tiny_config(layers=resolved_layers, heads=2, head_dim=8, block_size=8)
    cache = Qwen3DynamicCache(
        config,
        batch_size=resolved_batch_size,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    layer_idx = 0
    for row_idx in range(resolved_batch_size):
        row = _FakeTurboQuantRow(row_idx + 1)
        cache.turboquant_rows[layer_idx][row_idx] = row  # type: ignore[assignment]
        cache.layer_lengths[layer_idx][row_idx] = row.length
        cache._set_turboquant_row_metadata(layer_idx, row_idx, row)  # type: ignore[arg-type]

    row_ids = torch.arange(resolved_batch_size - 1, -1, -1, dtype=torch.int32)
    full_view = cache.turboquant_metadata_view(layer_idx)
    selected_view = cache.turboquant_metadata_view(layer_idx, row_ids=row_ids)
    non_turboquant_view = Qwen3DynamicCache(config).turboquant_metadata_view(layer_idx)
    if full_view is None or selected_view is None:
        raise RuntimeError("TurboQuant metadata benchmark expected initialized metadata views.")

    def full_once() -> object:
        return cache.turboquant_metadata_view(layer_idx)

    def selected_once() -> object:
        return cache.turboquant_metadata_view(layer_idx, row_ids=row_ids)

    full_view_ms = _time_ms(full_once, iters=iters)
    selected_view_ms = _time_ms(selected_once, iters=iters)
    full_summary = full_view.summary()
    selected_summary = selected_view.summary()
    runtime_boundary = cache.turboquant_runtime_boundary()
    non_turboquant_boundary = Qwen3DynamicCache(config).turboquant_runtime_boundary()
    return {
        "batch_size": resolved_batch_size,
        "layers": resolved_layers,
        "layer_idx": layer_idx,
        "metadata_device": str(full_view.device),
        "row_lengths_shape": full_summary["row_lengths_shape"],
        "row_active_shape": full_summary["row_active_shape"],
        "selected_row_lengths_shape": selected_summary["row_lengths_shape"],
        "selected_source_row_ids_shape": selected_summary["source_row_ids_shape"],
        "full_view_uses_metadata_storage": bool(
            full_view.row_lengths.data_ptr() == cache.turboquant_row_lengths[layer_idx].data_ptr()
            and full_view.row_active.data_ptr() == cache.turboquant_row_active[layer_idx].data_ptr()
        ),
        "selected_view_uses_metadata_storage": bool(
            selected_view.row_lengths.data_ptr() == cache.turboquant_row_lengths[layer_idx].data_ptr()
        ),
        "row_object_backing": bool(full_summary["row_object_backing"]),
        "tensor_bank_ready": bool(full_summary["tensor_bank_ready"]),
        "runtime_boundary_enabled": bool(runtime_boundary["enabled"]),
        "runtime_boundary_quantized_layers": int(runtime_boundary["quantized_layer_count"]),
        "runtime_boundary_tensor_metadata_layers": int(runtime_boundary["tensor_metadata_layers"]),
        "runtime_boundary_row_object_rows": int(runtime_boundary["row_object_rows"]),
        "runtime_boundary_active_row_objects": int(runtime_boundary["active_row_objects"]),
        "runtime_boundary_tensor_bank_ready": bool(runtime_boundary["tensor_bank_ready"]),
        "runtime_boundary_tensor_bank_reason": runtime_boundary["tensor_bank_reason"],
        "non_turboquant_boundary_tensor_bank_reason": non_turboquant_boundary["tensor_bank_reason"],
        "non_turboquant_view_is_none": non_turboquant_view is None,
        "full_view_ms": full_view_ms,
        "selected_view_ms": selected_view_ms,
    }


def _bench_int4_linear(
    *,
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    group_size: int,
    iters: int,
) -> dict[str, Any]:
    rows = {
        "decode_gemv": 1,
        "batch_gemm": max(1, int(batch_size)),
        "prefill_gemm": max(1, int(batch_size)) * max(1, int(seq_len)),
    }
    dense = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(
        dense,
        group_size=group_size,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    result: dict[str, Any] = {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "backend": "cpu_dequant_smoke",
        "prefill_tokens": max(1, int(batch_size)) * max(1, int(seq_len)),
    }
    for label, row_count in rows.items():
        inputs = torch.randn(row_count, in_features, dtype=torch.bfloat16)
        dense_weight = dense.weight.detach().to(dtype=torch.bfloat16)

        def dense_once() -> torch.Tensor:
            return F.linear(inputs, dense_weight)

        def int4_once() -> torch.Tensor:
            return quantized(inputs)

        dense_output = dense_once()
        int4_output = int4_once()
        error = (int4_output.float() - dense_output.float()).abs()
        dense_ms = _time_ms(dense_once, iters=iters)
        int4_ms = _time_ms(int4_once, iters=iters)
        result[f"{label}_rows"] = row_count
        result[f"{label}_dense_ms"] = dense_ms
        result[f"{label}_int4_ms"] = int4_ms
        result[f"{label}_max_abs_diff"] = float(error.max().item())
        result[f"{label}_mean_abs_diff"] = float(error.mean().item())
    return result


def _bench_hotpath_guard() -> dict[str, Any]:
    started = time.perf_counter()
    findings = scan_hotpath_files(root=Path.cwd())
    unexpected = unexpected_findings(findings)
    summary = summarize_findings(findings)
    allowlist = Counter(DEFAULT_ALLOWLIST)
    stale_allowlist = allowlist - summary
    extra_findings = summary - allowlist
    return {
        "scan_ms": (time.perf_counter() - started) * 1000.0,
        "findings": len(findings),
        "allowlist_entries": sum(allowlist.values()),
        "unexpected": len(unexpected),
        "allowlist_exact": summary == allowlist,
        "stale_allowlist": sum(stale_allowlist.values()),
        "extra_findings": sum(extra_findings.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-model Anna runtime hot-path microbenchmarks.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=0, help="KV heads for GQA attention benchmarks. Set 0 to derive.")
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--max-blocks-per-seq",
        type=int,
        default=0,
        help="Max block-table entries per sequence for the slot decode plan benchmark. Set 0 to derive from seq len.",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--int4-in-features", type=int, default=128)
    parser.add_argument("--int4-out-features", type=int, default=256)
    parser.add_argument("--int4-group-size", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_stack_split = _bench_cache_stack_split(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        iters=args.iters,
    )
    slot_decode_plan = _bench_slot_decode_plan(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        max_blocks_per_seq=args.max_blocks_per_seq,
        iters=args.iters,
    )
    result = {
        "hotpath_guard": _bench_hotpath_guard(),
        "cache_stack_split": cache_stack_split,
        "slot_decode_plan": slot_decode_plan,
        "scheduler_kv_overhead": _summarize_scheduler_kv_overhead(
            cache_stack_split=cache_stack_split,
            slot_decode_plan=slot_decode_plan,
        ),
        "paged_gqa_decode": _bench_paged_gqa_decode(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            iters=args.iters,
        ),
        "prefill_attention": _bench_prefill_attention(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            iters=args.iters,
        ),
        "sampler": _bench_sampler(
            vocab_size=args.vocab_size,
            candidates=args.candidates,
            iters=args.iters,
        ),
        "sampling_params_cache": _bench_sampling_params_cache(
            batch_size=args.batch_size,
            iters=args.iters,
        ),
        "turboquant_metadata_view": _bench_turboquant_metadata_view(
            batch_size=args.batch_size,
            layers=args.layers,
            iters=args.iters,
        ),
        "int4_linear": _bench_int4_linear(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            in_features=args.int4_in_features,
            out_features=args.int4_out_features,
            group_size=args.int4_group_size,
            iters=args.iters,
        ),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("section,metric,value")
        for section, values in result.items():
            for key, value in values.items():
                print(f"{section},{key},{value}")
    return 1 if result["hotpath_guard"]["unexpected"] or not result["hotpath_guard"]["allowlist_exact"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
