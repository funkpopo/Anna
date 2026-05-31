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
    assert result["block_table_rows"] == 2
    assert result["block_table_cols"] == 3
    assert result["plan_ms"] >= 0.0


def test_runtime_hotpath_bench_sampler_reports_candidate_and_full_vocab() -> None:
    bench = _load_bench_module()

    result = bench._bench_sampler(vocab_size=64, candidates=8, iters=1)

    assert result["vocab_size"] == 64
    assert result["candidates"] == 8
    assert result["full_vocab_ms"] >= 0.0
    assert result["candidate_ms"] >= 0.0
    assert result["sampler_full_vocab_sort_count"] == 1
