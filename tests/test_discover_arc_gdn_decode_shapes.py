from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_xpu_hotspots import GDN_DECODE_SHAPE_PRESETS  # noqa: E402
from tools.discover_arc_gdn_decode_shapes import (  # noqa: E402
    DISCOVERY_MODES,
    _annotate_rows,
    _build_group_summaries,
    _collect_suspicious_rows,
    _format_sampled_int_ranges,
    _parse_int_spans,
    _resolve_shape_cases,
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
