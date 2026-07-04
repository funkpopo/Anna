from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_xpu_hotspots import GDN_DECODE_SHAPE_PRESETS  # noqa: E402
from tools.validate_arc_gdn_decode import (  # noqa: E402
    ARC_BENCH_EXPECTATIONS,
    ARC_DEFAULT_PRESET,
    ARC_LEGACY_V64_BLOCK8_PRESET,
    ARC_V64_DEFAULT_BLOCK16_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_LEGACY_V256_BLOCK4_PRESET,
    DEFAULT_BENCH_TIMING_REPEATS,
    DEFAULT_COMPARE_RATIO_DELTA,
    _bench_args_for_preset,
    _chunk_shape_cases,
    _compare_benchmark_summary_against_baseline,
    _collect_benchmark_summary,
    _collect_benchmark_summary_from_rows,
    _load_json_report,
    _parse_gdn_decode_csv_rows,
    _parse_benchmark_output_rows_and_value_blocks,
    _parse_gdn_decode_value_blocks,
    _parse_preset_names,
    _resolve_validate_bench_max_shapes_per_scan,
    _run_benchmark_for_preset,
    _validate_benchmark_config_against_baseline,
    _validate_benchmark_output,
    _validate_benchmark_rows,
    _write_json_report,
)


def test_arc_bench_expectations_match_preset_shape_counts() -> None:
    for preset_name, expectation in ARC_BENCH_EXPECTATIONS.items():
        assert expectation.expected_row_count == len(GDN_DECODE_SHAPE_PRESETS[preset_name])


def test_parse_preset_names_preserves_order() -> None:
    assert _parse_preset_names("arc-default,arc-legacy-v256-block4") == [
        ARC_DEFAULT_PRESET,
        ARC_LEGACY_V256_BLOCK4_PRESET,
    ]


def test_parse_preset_names_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        _parse_preset_names("arc-default,unknown-preset")


@pytest.mark.parametrize(
    ("preset_name", "expected_compare_flag"),
    [
        (ARC_DEFAULT_PRESET, "--gdn-decode-default-compare"),
        (ARC_V64_DEFAULT_BLOCK16_PRESET, "--gdn-decode-default-compare"),
        (ARC_LEGACY_V64_BLOCK8_PRESET, "--gdn-decode-auto-compare"),
        (ARC_LEGACY_V128_BLOCK8_PRESET, "--gdn-decode-auto-compare"),
        (ARC_LEGACY_V256_BLOCK4_PRESET, "--gdn-decode-auto-compare"),
    ],
)
def test_bench_args_for_preset_uses_expected_compare_flag(
    preset_name: str,
    expected_compare_flag: str,
) -> None:
    args = _bench_args_for_preset(
        preset_name=preset_name,
        warmup=7,
        iters=11,
        timing_repeats=DEFAULT_BENCH_TIMING_REPEATS,
        seeds_csv="20260960,20260961",
    )
    assert expected_compare_flag in args
    assert "--gdn-decode-shape-presets" in args
    assert args[args.index("--gdn-decode-shape-presets") + 1] == preset_name
    assert args[args.index("--warmup") + 1] == "7"
    assert args[args.index("--iters") + 1] == "11"
    assert args[args.index("--gdn-decode-timing-repeats") + 1] == str(DEFAULT_BENCH_TIMING_REPEATS)


def test_parse_gdn_decode_value_blocks_extracts_csv() -> None:
    output = "shape ...\ngdn_decode_value_blocks=16\n"
    assert _parse_gdn_decode_value_blocks(output) == (16,)


def test_validate_bench_chunk_helpers_auto_chunk_only_large_presets() -> None:
    assert _resolve_validate_bench_max_shapes_per_scan(None, shape_count=32) is None
    assert _resolve_validate_bench_max_shapes_per_scan(None, shape_count=33) == 16
    assert _resolve_validate_bench_max_shapes_per_scan(24, shape_count=64) == 24
    assert _chunk_shape_cases([(idx, 8, 256) for idx in range(1, 6)], max_shapes_per_scan=2) == [
        [(1, 8, 256), (2, 8, 256)],
        [(3, 8, 256), (4, 8, 256)],
        [(5, 8, 256)],
    ]


def test_parse_gdn_decode_csv_rows_extracts_header_and_rows() -> None:
    output = "\n".join(
        [
            "gdn_decode_default_compare,device_name,batch,heads,key_head_dim,value_head_dim,default_value_block,default_strategy,single_ms,tiled_ms,default_ms,best_explicit_strategy,best_explicit_ms,default_minus_best_ms,default_speed_ratio,max_abs_diff",
            "gdn_decode_default_compare,Intel Arc,1,16,128,128,16,tiled,0.3000,0.2000,0.2010,tiled,0.2000,0.0010,1.0050,0.000100",
        ]
    )
    rows = _parse_gdn_decode_csv_rows(output, "gdn_decode_default_compare")
    assert rows == [
        {
            "gdn_decode_default_compare": "gdn_decode_default_compare",
            "device_name": "Intel Arc",
            "batch": "1",
            "heads": "16",
            "key_head_dim": "128",
            "value_head_dim": "128",
            "default_value_block": "16",
            "default_strategy": "tiled",
            "single_ms": "0.3000",
            "tiled_ms": "0.2000",
            "default_ms": "0.2010",
            "best_explicit_strategy": "tiled",
            "best_explicit_ms": "0.2000",
            "default_minus_best_ms": "0.0010",
            "default_speed_ratio": "1.0050",
            "max_abs_diff": "0.000100",
        }
    ]


def _make_benchmark_output_for_preset(
    preset_name: str,
    *,
    ratio: float,
) -> str:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    lines = [f"gdn_decode_value_blocks={','.join(str(value_block) for value_block in expectation.expected_value_blocks)}"]
    if expectation.compare_prefix == "gdn_decode_default_compare":
        lines.append(
            "gdn_decode_default_compare,device_name,batch,heads,key_head_dim,value_head_dim,default_value_block,"
            "default_strategy,single_ms,tiled_ms,default_ms,best_explicit_strategy,best_explicit_ms,"
            "default_minus_best_ms,default_speed_ratio,max_abs_diff"
        )
        for _ in range(expectation.expected_row_count):
            lines.append(
                "gdn_decode_default_compare,Intel Arc,1,16,128,128,"
                f"{expectation.default_value_block},{expectation.default_strategy},"
                f"0.3000,0.2000,{0.2 * ratio:.4f},tiled,0.2000,{0.2 * ratio - 0.2:.4f},{ratio:.4f},0.000100"
            )
    else:
        lines.append(
            "gdn_decode_auto_compare,device_name,batch,heads,key_head_dim,value_head_dim,value_block,auto_strategy,"
            "single_ms,tiled_ms,auto_ms,best_explicit_strategy,best_explicit_ms,auto_minus_best_ms,"
            "auto_speed_ratio,max_abs_diff"
        )
        value_block = expectation.expected_value_blocks[0]
        for _ in range(expectation.expected_row_count):
            lines.append(
                "gdn_decode_auto_compare,Intel Arc,1,16,128,256,"
                f"{value_block},tiled,0.3000,0.2000,{0.2 * ratio:.4f},tiled,0.2000,{0.2 * ratio - 0.2:.4f},"
                f"{ratio:.4f},0.000100"
            )
    return "\n".join(lines)


def test_validate_benchmark_output_accepts_in_threshold_output() -> None:
    _validate_benchmark_output(_make_benchmark_output_for_preset(ARC_DEFAULT_PRESET, ratio=1.01), ARC_DEFAULT_PRESET)
    _validate_benchmark_output(
        _make_benchmark_output_for_preset(ARC_LEGACY_V256_BLOCK4_PRESET, ratio=1.03),
        ARC_LEGACY_V256_BLOCK4_PRESET,
    )


def test_validate_benchmark_output_rejects_ratio_regression() -> None:
    with pytest.raises(ValueError, match="exceeded default_speed_ratio threshold"):
        _validate_benchmark_output(_make_benchmark_output_for_preset(ARC_DEFAULT_PRESET, ratio=1.20), ARC_DEFAULT_PRESET)


def test_collect_benchmark_summary_reports_worst_ratio() -> None:
    summary = _collect_benchmark_summary(
        _make_benchmark_output_for_preset(ARC_LEGACY_V256_BLOCK4_PRESET, ratio=1.03),
        ARC_LEGACY_V256_BLOCK4_PRESET,
    )
    assert summary["preset"] == ARC_LEGACY_V256_BLOCK4_PRESET
    assert summary["resolved_value_blocks"] == [4]
    assert summary["observed_max_ratio"] == pytest.approx(1.03)
    assert summary["row_count"] == len(GDN_DECODE_SHAPE_PRESETS[ARC_LEGACY_V256_BLOCK4_PRESET])


def test_validate_and_collect_benchmark_rows_match_output_wrappers() -> None:
    output = _make_benchmark_output_for_preset(ARC_LEGACY_V256_BLOCK4_PRESET, ratio=1.02)
    value_blocks, rows = _parse_benchmark_output_rows_and_value_blocks(output, ARC_LEGACY_V256_BLOCK4_PRESET)
    _validate_benchmark_rows(
        preset_name=ARC_LEGACY_V256_BLOCK4_PRESET,
        value_blocks=value_blocks,
        rows=rows,
    )
    summary = _collect_benchmark_summary_from_rows(
        preset_name=ARC_LEGACY_V256_BLOCK4_PRESET,
        value_blocks=value_blocks,
        rows=rows,
    )
    assert summary == _collect_benchmark_summary(output, ARC_LEGACY_V256_BLOCK4_PRESET)


def test_run_benchmark_for_preset_chunks_large_preset_and_merges_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run_step(*, args, cwd, env, label, capture_output):
        calls.append((args, capture_output))
        if not capture_output:
            return None
        shape_cases = args[args.index("--gdn-decode-shape-cases") + 1].split(",")
        lines = [
            "gdn_decode_value_blocks=4",
            "gdn_decode_auto_compare,device_name,batch,heads,key_head_dim,value_head_dim,value_block,auto_strategy,"
            "single_ms,tiled_ms,auto_ms,best_explicit_strategy,best_explicit_ms,auto_minus_best_ms,"
            "auto_speed_ratio,max_abs_diff",
        ]
        for shape_case in shape_cases:
            batch_size, num_heads, value_head_dim = shape_case.split("x")
            lines.append(
                "gdn_decode_auto_compare,Intel Arc,"
                f"{batch_size},{num_heads},128,{value_head_dim},4,tiled,0.3000,0.2000,0.2010,tiled,0.2000,0.0010,1.0050,0.000100"
            )
        return "\n".join(lines)

    monkeypatch.setattr("tools.validate_arc_gdn_decode._run_step", fake_run_step)

    requested_args, bench_runs, value_blocks, rows, effective_max_shapes_per_scan = _run_benchmark_for_preset(
        preset_name=ARC_LEGACY_V256_BLOCK4_PRESET,
        warmup=7,
        iters=11,
        timing_repeats=DEFAULT_BENCH_TIMING_REPEATS,
        seeds_csv="20260960,20260961",
        cwd=Path.cwd(),
        env={},
        max_shapes_per_scan=None,
    )

    assert requested_args[requested_args.index("--gdn-decode-shape-presets") + 1] == ARC_LEGACY_V256_BLOCK4_PRESET
    assert len(bench_runs) == 3
    assert len(calls) == 6
    assert sum(1 for _, capture_output in calls if not capture_output) == 3
    assert sum(1 for _, capture_output in calls if capture_output) == 3
    prime_calls = [args for args, capture_output in calls if not capture_output]
    assert all(prime_args[prime_args.index("--warmup") + 1] == "1" for prime_args in prime_calls)
    assert all(prime_args[prime_args.index("--iters") + 1] == "1" for prime_args in prime_calls)
    assert all(
        prime_args[prime_args.index("--gdn-decode-timing-repeats") + 1] == "1"
        for prime_args in prime_calls
    )
    assert effective_max_shapes_per_scan == 16
    assert value_blocks == (4,)
    assert len(rows) == len(GDN_DECODE_SHAPE_PRESETS[ARC_LEGACY_V256_BLOCK4_PRESET])

def test_write_json_report_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "arc_gdn_decode.json"
    payload = {"schema_version": 1, "status": "ok"}
    _write_json_report(output_path, payload)
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_load_json_report_reads_written_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "arc_gdn_decode.json"
    payload = {"schema_version": 1, "status": "ok"}
    _write_json_report(output_path, payload)
    assert _load_json_report(output_path) == payload


def test_validate_benchmark_config_against_baseline_accepts_matching_settings() -> None:
    current_config = {
        "seeds": "20260960,20260961",
        "warmup": 20,
        "iters": 100,
        "timing_repeats": DEFAULT_BENCH_TIMING_REPEATS,
    }
    baseline_report = {
        "benchmark": {
            "seeds": "20260960,20260961",
            "warmup": 20,
            "iters": 100,
            "timing_repeats": DEFAULT_BENCH_TIMING_REPEATS,
        }
    }
    _validate_benchmark_config_against_baseline(current_config=current_config, baseline_report=baseline_report)


def test_validate_benchmark_config_against_baseline_rejects_missing_timing_repeats() -> None:
    current_config = {
        "seeds": "20260960,20260961",
        "warmup": 20,
        "iters": 100,
        "timing_repeats": DEFAULT_BENCH_TIMING_REPEATS,
    }
    baseline_report = {
        "benchmark": {
            "seeds": "20260960,20260961",
            "warmup": 20,
            "iters": 100,
        }
    }
    with pytest.raises(ValueError, match="missing benchmark.timing_repeats"):
        _validate_benchmark_config_against_baseline(current_config=current_config, baseline_report=baseline_report)


def test_compare_benchmark_summary_against_baseline_accepts_small_drift() -> None:
    baseline = _collect_benchmark_summary(
        _make_benchmark_output_for_preset(ARC_DEFAULT_PRESET, ratio=1.00),
        ARC_DEFAULT_PRESET,
    )
    current = _collect_benchmark_summary(
        _make_benchmark_output_for_preset(ARC_DEFAULT_PRESET, ratio=1.01),
        ARC_DEFAULT_PRESET,
    )
    comparison = _compare_benchmark_summary_against_baseline(
        current_summary=current,
        baseline_summary=baseline,
        max_ratio_delta=DEFAULT_COMPARE_RATIO_DELTA,
    )
    assert comparison["preset"] == ARC_DEFAULT_PRESET
    assert comparison["max_row_ratio_delta"] == pytest.approx(0.01)


def test_compare_benchmark_summary_against_baseline_rejects_large_drift() -> None:
    baseline = _collect_benchmark_summary(
        _make_benchmark_output_for_preset(ARC_LEGACY_V256_BLOCK4_PRESET, ratio=1.00),
        ARC_LEGACY_V256_BLOCK4_PRESET,
    )
    current = _collect_benchmark_summary(
        _make_benchmark_output_for_preset(ARC_LEGACY_V256_BLOCK4_PRESET, ratio=1.05),
        ARC_LEGACY_V256_BLOCK4_PRESET,
    )
    with pytest.raises(ValueError, match="exceeded ratio delta threshold"):
        _compare_benchmark_summary_against_baseline(
            current_summary=current,
            baseline_summary=baseline,
            max_ratio_delta=0.03,
        )
