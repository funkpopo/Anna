from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_xpu_hotspots import GDN_DECODE_SHAPE_PRESETS  # noqa: E402
from tools.discover_arc_gdn_decode_shapes import (  # noqa: E402
    DISCOVERY_MODES,
    _annotate_rows,
    _build_order_sensitivity_rows,
    _build_group_summaries,
    _chunk_shape_cases,
    _collect_suspicious_rows,
    _collect_order_sensitive_rows,
    _discovery_row_key,
    _format_sampled_int_ranges,
    _parse_int_spans,
    _primary_bench_args,
    _resolve_max_shapes_per_scan,
    _resolve_shape_cases,
    _select_confirmation_rows,
    _select_confirmation_shape_cases,
    _select_confirmation_value_blocks,
)


def test_parse_int_spans_supports_single_values_ranges_and_steps() -> None:
    assert _parse_int_spans("1-3,6,10-14:2") == [1, 2, 3, 6, 10, 12, 14]


def test_parse_int_spans_rejects_descending_ranges() -> None:
    with pytest.raises(ValueError, match="stop must be >= start"):
        _parse_int_spans("4-1")


def test_resolve_shape_cases_combines_presets_explicit_cases_and_grid() -> None:
    cases, preset_names = _resolve_shape_cases(
        shape_presets="arc-v64-default-block16",
        shape_cases="1x8x64,3x16x64",
        batches="1-2",
        heads="8",
        value_head_dims="64",
    )
    assert preset_names == ["arc-v64-default-block16"]
    assert cases[0] == GDN_DECODE_SHAPE_PRESETS["arc-v64-default-block16"][0]
    assert (3, 16, 64) in cases
    assert (2, 8, 64) in cases
    assert cases.count((1, 8, 64)) == 1


def test_resolve_shape_cases_requires_complete_grid_definition() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _resolve_shape_cases(
            shape_presets=None,
            shape_cases=None,
            batches="1-4",
            heads="8,16",
            value_head_dims=None,
        )


def test_format_sampled_int_ranges_compacts_constant_step_sequences() -> None:
    assert _format_sampled_int_ranges([784, 800, 816, 832, 960]) == "784..832/16,960"


def test_chunk_shape_cases_splits_large_case_lists_and_preserves_order() -> None:
    cases = [(batch_size, 8, 256) for batch_size in range(1, 8)]
    assert _chunk_shape_cases(cases, max_shapes_per_scan=None) == [cases]
    assert _chunk_shape_cases(cases, max_shapes_per_scan=0) == [cases]
    assert _chunk_shape_cases(cases, max_shapes_per_scan=3) == [
        [(1, 8, 256), (2, 8, 256), (3, 8, 256)],
        [(4, 8, 256), (5, 8, 256), (6, 8, 256)],
        [(7, 8, 256)],
    ]


def test_primary_bench_args_returns_single_run_only() -> None:
    single_run = [["python", "tools/bench_xpu_hotspots.py", "--example"]]
    assert _primary_bench_args(single_run) == single_run[0]
    assert _primary_bench_args(single_run * 2) is None


def test_resolve_max_shapes_per_scan_auto_chunks_only_large_sweeps() -> None:
    assert _resolve_max_shapes_per_scan(None, shape_count=32) is None
    assert _resolve_max_shapes_per_scan(None, shape_count=33) == 16
    assert _resolve_max_shapes_per_scan(24, shape_count=128) == 24
    assert _resolve_max_shapes_per_scan(0, shape_count=128) == 0


def test_collect_suspicious_rows_flags_mismatches_and_ratio_only_rows() -> None:
    rows = _annotate_rows(
        [
            {
                "batch": "25",
                "heads": "32",
                "key_head_dim": "128",
                "value_head_dim": "256",
                "value_block": "4",
                "auto_strategy": "tiled",
                "single_ms": "1.9500",
                "tiled_ms": "2.2900",
                "auto_ms": "2.2890",
                "best_explicit_strategy": "single",
                "best_explicit_ms": "1.9500",
                "auto_minus_best_ms": "0.3390",
                "auto_speed_ratio": "1.1738",
                "max_abs_diff": "0.009323",
            },
            {
                "batch": "4",
                "heads": "8",
                "key_head_dim": "128",
                "value_head_dim": "128",
                "value_block": "8",
                "auto_strategy": "single",
                "single_ms": "0.2780",
                "tiled_ms": "0.3070",
                "auto_ms": "0.2865",
                "best_explicit_strategy": "single",
                "best_explicit_ms": "0.2780",
                "auto_minus_best_ms": "0.0085",
                "auto_speed_ratio": "1.0306",
                "max_abs_diff": "0.004671",
            },
        ],
        DISCOVERY_MODES["auto"],
    )

    suspicious_rows = _collect_suspicious_rows(
        rows,
        ratio_threshold=1.03,
        require_strategy_mismatch=False,
    )
    assert len(suspicious_rows) == 2
    assert suspicious_rows[0]["strategy_mismatch"] is True
    assert suspicious_rows[0]["reasons"] == ["strategy_mismatch", "ratio_threshold"]
    assert suspicious_rows[1]["strategy_mismatch"] is False
    assert suspicious_rows[1]["reasons"] == ["ratio_threshold"]

    mismatch_only_rows = _collect_suspicious_rows(
        rows,
        ratio_threshold=1.03,
        require_strategy_mismatch=True,
    )
    assert len(mismatch_only_rows) == 1
    assert mismatch_only_rows[0]["batch_size"] == 25


def test_build_group_summaries_collapses_rows_into_sampled_row_ranges() -> None:
    rows = [
        {
            "value_head_dim_int": 256,
            "value_block_int": 4,
            "observed_strategy": "tiled",
            "best_explicit_strategy_str": "single",
            "recurrent_rows": 784,
            "ratio": 1.11,
            "delta_ms": 0.22,
        },
        {
            "value_head_dim_int": 256,
            "value_block_int": 4,
            "observed_strategy": "tiled",
            "best_explicit_strategy_str": "single",
            "recurrent_rows": 800,
            "ratio": 1.17,
            "delta_ms": 0.33,
        },
        {
            "value_head_dim_int": 256,
            "value_block_int": 4,
            "observed_strategy": "tiled",
            "best_explicit_strategy_str": "single",
            "recurrent_rows": 816,
            "ratio": 1.16,
            "delta_ms": 0.31,
        },
        {
            "value_head_dim_int": 128,
            "value_block_int": 8,
            "observed_strategy": "single",
            "best_explicit_strategy_str": "tiled",
            "recurrent_rows": 144,
            "ratio": 1.05,
            "delta_ms": 0.01,
        },
    ]

    summaries = _build_group_summaries(rows)
    assert len(summaries) == 2
    assert summaries[0]["value_head_dim"] == 256
    assert summaries[0]["sampled_recurrent_row_ranges"] == "784..816/16"
    assert summaries[0]["worst_ratio"] == pytest.approx(1.17)


def test_select_confirmation_shape_cases_and_value_blocks_dedupe_candidates() -> None:
    suspicious_rows = [
        {
            "batch_size": 25,
            "num_heads": 32,
            "value_head_dim_int": 256,
            "value_block_int": 4,
        },
        {
            "batch_size": 25,
            "num_heads": 32,
            "value_head_dim_int": 256,
            "value_block_int": 4,
        },
        {
            "batch_size": 49,
            "num_heads": 16,
            "value_head_dim_int": 256,
            "value_block_int": 4,
        },
        {
            "batch_size": 9,
            "num_heads": 16,
            "value_head_dim_int": 128,
            "value_block_int": 8,
        },
    ]

    assert _select_confirmation_shape_cases(suspicious_rows) == [
        (25, 32, 256),
        (49, 16, 256),
        (9, 16, 128),
    ]
    assert _select_confirmation_value_blocks(suspicious_rows, [16]) == [4, 8]
    assert _select_confirmation_value_blocks([], [8, 16]) == [8, 16]


def test_select_confirmation_rows_and_reconfirm_suspicious_candidates() -> None:
    initial_suspicious_rows = _annotate_rows(
        [
            {
                "batch": "25",
                "heads": "32",
                "key_head_dim": "128",
                "value_head_dim": "256",
                "value_block": "4",
                "auto_strategy": "tiled",
                "single_ms": "1.9500",
                "tiled_ms": "2.2900",
                "auto_ms": "2.2890",
                "best_explicit_strategy": "single",
                "best_explicit_ms": "1.9500",
                "auto_minus_best_ms": "0.3390",
                "auto_speed_ratio": "1.1738",
                "max_abs_diff": "0.009323",
            },
            {
                "batch": "17",
                "heads": "8",
                "key_head_dim": "128",
                "value_head_dim": "128",
                "value_block": "8",
                "auto_strategy": "single",
                "single_ms": "0.3260",
                "tiled_ms": "0.3259",
                "auto_ms": "0.3267",
                "best_explicit_strategy": "tiled",
                "best_explicit_ms": "0.3259",
                "auto_minus_best_ms": "0.0009",
                "auto_speed_ratio": "1.0026",
                "max_abs_diff": "0.007109",
            },
        ],
        DISCOVERY_MODES["auto"],
    )

    confirmed_rows = _annotate_rows(
        [
            {
                "batch": "17",
                "heads": "8",
                "key_head_dim": "128",
                "value_head_dim": "128",
                "value_block": "8",
                "auto_strategy": "single",
                "single_ms": "0.3184",
                "tiled_ms": "0.3198",
                "auto_ms": "0.3178",
                "best_explicit_strategy": "single",
                "best_explicit_ms": "0.3184",
                "auto_minus_best_ms": "-0.0006",
                "auto_speed_ratio": "0.9982",
                "max_abs_diff": "0.007109",
            },
            {
                "batch": "25",
                "heads": "32",
                "key_head_dim": "128",
                "value_head_dim": "256",
                "value_block": "4",
                "auto_strategy": "single",
                "single_ms": "1.9546",
                "tiled_ms": "2.2960",
                "auto_ms": "1.9499",
                "best_explicit_strategy": "single",
                "best_explicit_ms": "1.9546",
                "auto_minus_best_ms": "-0.0046",
                "auto_speed_ratio": "0.9976",
                "max_abs_diff": "0.009395",
            },
        ],
        DISCOVERY_MODES["auto"],
    )

    selected_rows = _select_confirmation_rows(initial_suspicious_rows, confirmed_rows)
    assert [_discovery_row_key(row) for row in selected_rows] == [
        _discovery_row_key(initial_suspicious_rows[0]),
        _discovery_row_key(initial_suspicious_rows[1]),
    ]

    confirmed_suspicious_rows = _collect_suspicious_rows(
        selected_rows,
        ratio_threshold=1.03,
        require_strategy_mismatch=False,
    )
    assert confirmed_suspicious_rows == []


def test_build_order_sensitivity_rows_matches_rows_by_shape_and_computes_ratios() -> None:
    forward_rows = _annotate_rows(
        [
            {
                "batch": "179",
                "heads": "8",
                "key_head_dim": "128",
                "value_head_dim": "256",
                "value_block": "4",
                "auto_strategy": "tiled",
                "single_ms": "4.1706",
                "tiled_ms": "3.5551",
                "auto_ms": "3.5462",
                "best_explicit_strategy": "tiled",
                "best_explicit_ms": "3.5551",
                "auto_minus_best_ms": "-0.0088",
                "auto_speed_ratio": "0.9975",
                "max_abs_diff": "0.011126",
            }
        ],
        DISCOVERY_MODES["auto"],
    )
    reverse_rows = _annotate_rows(
        [
            {
                "batch": "179",
                "heads": "8",
                "key_head_dim": "128",
                "value_head_dim": "256",
                "value_block": "4",
                "auto_strategy": "tiled",
                "single_ms": "47.7031",
                "tiled_ms": "47.1897",
                "auto_ms": "47.2364",
                "best_explicit_strategy": "tiled",
                "best_explicit_ms": "47.1897",
                "auto_minus_best_ms": "0.0467",
                "auto_speed_ratio": "1.0010",
                "max_abs_diff": "0.010338",
            }
        ],
        DISCOVERY_MODES["auto"],
    )

    compared_rows = _build_order_sensitivity_rows(forward_rows, reverse_rows, mode=DISCOVERY_MODES["auto"])
    assert len(compared_rows) == 1
    assert compared_rows[0]["observed_order_ratio"] == pytest.approx(47.2364 / 3.5462)
    assert compared_rows[0]["best_explicit_order_ratio"] == pytest.approx(47.1897 / 3.5551)
    assert compared_rows[0]["observed_strategy_changed"] is False


def test_collect_order_sensitive_rows_flags_threshold_and_strategy_changes() -> None:
    rows = [
        {
            "batch_size": 179,
            "num_heads": 8,
            "recurrent_rows": 1432,
            "value_head_dim_int": 256,
            "value_block_int": 4,
            "observed_strategy": "tiled",
            "best_explicit_strategy_str": "tiled",
            "reverse_observed_strategy": "tiled",
            "reverse_best_explicit_strategy": "tiled",
            "forward_observed_ms": 3.5462,
            "reverse_observed_ms": 47.2364,
            "observed_order_ratio": 13.320737,
            "max_order_ratio": 13.320737,
            "observed_strategy_changed": False,
            "best_explicit_strategy_changed": False,
        },
        {
            "batch_size": 92,
            "num_heads": 8,
            "recurrent_rows": 736,
            "value_head_dim_int": 256,
            "value_block_int": 4,
            "observed_strategy": "single",
            "best_explicit_strategy_str": "single",
            "reverse_observed_strategy": "tiled",
            "reverse_best_explicit_strategy": "single",
            "forward_observed_ms": 1.8025,
            "reverse_observed_ms": 1.8040,
            "observed_order_ratio": 1.0008,
            "max_order_ratio": 1.0008,
            "observed_strategy_changed": True,
            "best_explicit_strategy_changed": False,
        },
    ]

    sensitive_rows = _collect_order_sensitive_rows(rows, order_ratio_threshold=1.05)
    assert len(sensitive_rows) == 2
    assert sensitive_rows[0]["order_reasons"] == ["order_ratio_threshold"]
    assert sensitive_rows[1]["order_reasons"] == ["strategy_changed"]
