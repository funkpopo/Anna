from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_bench_module():
    module_path = Path.cwd() / "tools" / "bench_runtime_hotpath.py"
    spec = importlib.util.spec_from_file_location("bench_runtime_hotpath", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_hotpath_bench_includes_slot_decode_plan() -> None:
    bench = _load_bench_module()

    result = bench._bench_slot_decode_plan(
        batch_size=2,
        seq_len=8,
        block_size=4,
        max_blocks_per_seq=0,
        iters=1,
    )

    assert result["batch_size"] == 2
    assert result["seq_len"] == 8
    assert result["active_batch_rows"] == 2
    assert result["block_tables_are_global"] is True
    assert result["seq_lens_are_global"] is True
    assert result["positions_are_global"] is True
    assert result["block_table_rows"] == 3
    assert result["block_table_cols"] == 3
    assert result["seq_lens_rows"] == 3
    assert result["positions_rows"] == 3
    assert result["plan_ms"] >= 0.0


def test_runtime_hotpath_bench_summarizes_scheduler_kv_overhead() -> None:
    bench = _load_bench_module()

    result = bench._summarize_scheduler_kv_overhead(
        cache_stack_split={
            "batch_size": 3,
            "seq_len": 16,
            "layers": 4,
            "block_size": 8,
            "stack_ms": 2.0,
            "split_ms": 3.0,
        },
        slot_decode_plan={
            "plan_ms": 1.25,
            "block_table_rows": 3,
            "block_table_cols": 3,
            "seq_lens_rows": 3,
            "positions_rows": 3,
        },
    )

    assert result["batch_size"] == 3
    assert result["seq_len"] == 16
    assert result["layers"] == 4
    assert result["block_size"] == 8
    assert result["legacy_cache_objects_per_step"] == 3
    assert result["legacy_layer_rows_touched_per_step"] == 12
    assert result["slot_plan_active_rows"] == 3
    assert result["slot_plan_rows"] == 3
    assert result["slot_plan_block_tables_are_global"] is False
    assert result["slot_plan_seq_lens_are_global"] is False
    assert result["slot_plan_positions_are_global"] is False
    assert result["slot_plan_block_table_cols"] == 3
    assert result["slot_plan_block_table_entries"] == 9
    assert result["slot_plan_global_block_table_entries"] == 9
    assert result["slot_plan_seq_lens_entries"] == 3
    assert result["slot_plan_global_seq_lens_entries"] == 3
    assert result["slot_plan_positions_entries"] == 3
    assert result["slot_plan_global_positions_entries"] == 3
    assert result["stack_ms"] == 2.0
    assert result["split_ms"] == 3.0
    assert result["stack_split_ms"] == 5.0
    assert result["slot_plan_ms"] == 1.25
    assert result["stack_split_to_slot_plan_ratio"] == 4.0


def test_runtime_hotpath_bench_reports_exact_hotpath_guard_baseline() -> None:
    bench = _load_bench_module()

    result = bench._bench_hotpath_guard()

    assert result["scan_ms"] >= 0.0
    assert result["findings"] == result["allowlist_entries"]
    assert result["unexpected"] == 0
    assert result["allowlist_exact"] is True
    assert result["stale_allowlist"] == 0
    assert result["extra_findings"] == 0


def test_runtime_hotpath_bench_sampler_reports_candidate_and_full_vocab() -> None:
    bench = _load_bench_module()

    result = bench._bench_sampler(vocab_size=64, candidates=8, iters=1)

    assert result["vocab_size"] == 64
    assert result["candidates"] == 8
    assert result["full_vocab_ms"] >= 0.0
    assert result["candidate_ms"] >= 0.0
    assert result["candidate_penalty_ms"] >= 0.0
    assert result["sampler_full_vocab_sort_count"] == 1


def test_runtime_hotpath_bench_includes_paged_gqa_decode_shapes() -> None:
    bench = _load_bench_module()

    result = bench._bench_paged_gqa_decode(
        batch_size=2,
        seq_len=8,
        heads=4,
        kv_heads=2,
        head_dim=8,
        block_size=4,
        iters=1,
    )

    assert result["backend"] == "cpu_reference_smoke"
    assert result["batch_size"] == 2
    assert result["seq_len"] == 8
    assert result["heads"] == 4
    assert result["kv_heads"] == 2
    assert result["head_dim"] == 8
    assert result["block_size"] == 4
    assert result["pages"] == 4
    assert result["block_table_rows"] == 4
    assert result["block_table_cols"] == 2
    assert result["materialized_ms"] >= 0.0
    assert result["paged_reference_ms"] >= 0.0
    assert result["max_abs_diff"] >= 0.0


def test_runtime_hotpath_bench_includes_prefill_attention_shapes() -> None:
    bench = _load_bench_module()

    result = bench._bench_prefill_attention(
        batch_size=2,
        seq_len=8,
        heads=4,
        kv_heads=2,
        head_dim=8,
        iters=1,
    )

    assert result["backend"] == "torch_cpu_smoke"
    assert result["batch_size"] == 2
    assert result["seq_len"] == 8
    assert result["heads"] == 4
    assert result["kv_heads"] == 2
    assert result["head_dim"] == 8
    assert result["sdpa_materialized_ms"] >= 0.0
    assert result["grouped_fallback_ms"] >= 0.0
    assert result["max_abs_diff"] >= 0.0


def test_runtime_hotpath_bench_includes_int4_linear_shapes() -> None:
    bench = _load_bench_module()

    result = bench._bench_int4_linear(
        batch_size=3,
        seq_len=5,
        in_features=32,
        out_features=16,
        group_size=32,
        iters=1,
    )

    assert result["in_features"] == 32
    assert result["out_features"] == 16
    assert result["group_size"] == 32
    assert result["backend"] == "cpu_dequant_smoke"
    assert result["prefill_tokens"] == 15
    assert result["decode_gemv_rows"] == 1
    assert result["batch_gemm_rows"] == 3
    assert result["prefill_gemm_rows"] == 15
    assert result["decode_gemv_dense_ms"] >= 0.0
    assert result["decode_gemv_int4_ms"] >= 0.0
    assert result["batch_gemm_dense_ms"] >= 0.0
    assert result["batch_gemm_int4_ms"] >= 0.0
    assert result["prefill_gemm_dense_ms"] >= 0.0
    assert result["prefill_gemm_int4_ms"] >= 0.0
    assert result["decode_gemv_max_abs_diff"] >= 0.0
    assert result["batch_gemm_max_abs_diff"] >= 0.0
    assert result["prefill_gemm_max_abs_diff"] >= 0.0
