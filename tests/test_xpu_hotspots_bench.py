from __future__ import annotations

import csv
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


def _load_bench_module():
    module_path = Path.cwd() / "tools" / "bench_xpu_hotspots.py"
    spec = importlib.util.spec_from_file_location("bench_xpu_hotspots", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xpu_hotspots_bench_formats_arc_profile_rows() -> None:
    bench = _load_bench_module()
    rows: list[dict[str, str]] = []

    bench._append_arc_profile_row(
        rows,
        profile="arc_int4_profile",
        device_name="Arc Test GPU",
        strategy="auto",
        m=1,
        k=1024,
        n=4096,
        group_size=128,
        dtype=torch.bfloat16,
        baseline_ms=2.0,
        candidate_ms=0.5,
        max_abs_diff=0.125,
    )
    bench._append_arc_profile_row(
        rows,
        profile="arc_moe_grouped_int4_mlp_profile",
        device_name="Arc Test GPU",
        gate_local_size=64,
        down_local_size=128,
        tokens_per_expert=2,
        experts=8,
        hidden_size=1024,
        intermediate_size=4096,
        group_size=128,
        dtype=torch.float16,
        baseline_ms=3.0,
        candidate_ms=0.0,
        max_abs_diff=float("inf"),
    )

    assert rows[0] == {
        "profile": "arc_int4_profile",
        "device_name": "Arc Test GPU",
        "strategy": "auto",
        "local_size": "-",
        "gate_local_size": "-",
        "down_local_size": "-",
        "M": "1",
        "K": "1024",
        "N": "4096",
        "top_k": "-",
        "tokens_per_expert": "-",
        "experts": "-",
        "hidden_size": "-",
        "intermediate_size": "-",
        "group_size": "128",
        "dtype": "torch.bfloat16",
        "baseline_ms": "2.0000",
        "candidate_ms": "0.5000",
        "speedup": "4.00x",
        "max_abs_diff": "0.125000",
    }
    assert rows[1]["profile"] == "arc_moe_grouped_int4_mlp_profile"
    assert rows[1]["strategy"] == "-"
    assert rows[1]["gate_local_size"] == "64"
    assert rows[1]["down_local_size"] == "128"
    assert rows[1]["tokens_per_expert"] == "2"
    assert rows[1]["experts"] == "8"
    assert rows[1]["hidden_size"] == "1024"
    assert rows[1]["intermediate_size"] == "4096"
    assert rows[1]["dtype"] == "torch.float16"
    assert rows[1]["speedup"] == "n/a"
    assert rows[1]["max_abs_diff"] == "inf"


def test_xpu_hotspots_bench_writes_arc_profile_csv() -> None:
    bench = _load_bench_module()
    rows: list[dict[str, str]] = []
    bench._append_arc_profile_row(
        rows,
        profile="arc_lm_head_int4_topk_profile",
        device_name="Arc Test GPU",
        local_size=32,
        m=1,
        k=1024,
        n=32000,
        top_k=8,
        group_size=128,
        dtype=torch.bfloat16,
        baseline_ms=4.0,
        candidate_ms=1.0,
        max_abs_diff=0.0,
    )

    output_path = Path.cwd() / f"pytest_tmp_codex_arc_profile_{uuid.uuid4().hex}.csv"
    try:
        bench._write_arc_profile_csv(str(output_path), rows)

        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            loaded = list(reader)
            fieldnames = reader.fieldnames
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except PermissionError:
            pass

    assert fieldnames == list(bench.ARC_PROFILE_FIELDNAMES)
    assert loaded == rows


def test_xpu_hotspots_bench_validates_arc_csv_requires_arc_profile() -> None:
    bench = _load_bench_module()

    with pytest.raises(ValueError, match="--arc-int4-only requires --arc-profile"):
        bench._validate_benchmark_args(
            SimpleNamespace(arc_int4_only=True, arc_profile=False, arc_csv_output=None)
        )
    with pytest.raises(ValueError, match="--arc-csv-output requires --arc-profile"):
        bench._validate_benchmark_args(
            SimpleNamespace(arc_int4_only=False, arc_profile=False, arc_csv_output="arc.csv")
        )

    bench._validate_benchmark_args(
        SimpleNamespace(arc_int4_only=True, arc_profile=True, arc_csv_output="arc.csv")
    )
