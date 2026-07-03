from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_xpu_hotspots import (
    GDN_DECODE_PRESET_VALUE_BLOCKS,
    GDN_DECODE_SHAPE_PRESETS,
    _dedupe_gdn_decode_shape_cases,
    _parse_gdn_decode_shape_presets,
    _resolve_gdn_decode_value_blocks,
)


def test_gdn_decode_shape_presets_have_matching_value_block_metadata() -> None:
    assert set(GDN_DECODE_SHAPE_PRESETS) == set(GDN_DECODE_PRESET_VALUE_BLOCKS)


def test_parse_gdn_decode_shape_presets_preserves_order() -> None:
    assert _parse_gdn_decode_shape_presets("arc-default,arc-legacy-v256-block4") == [
        "arc-default",
        "arc-legacy-v256-block4",
    ]


def test_parse_gdn_decode_shape_presets_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown --gdn-decode-shape-presets entry"):
        _parse_gdn_decode_shape_presets("arc-default,not-a-preset")


def test_dedupe_gdn_decode_shape_cases_preserves_first_occurrence() -> None:
    cases = [
        (1, 16, 128),
        (4, 16, 128),
        (1, 16, 128),
        (4, 32, 256),
        (4, 16, 128),
    ]
    assert _dedupe_gdn_decode_shape_cases(cases) == [
        (1, 16, 128),
        (4, 16, 128),
        (4, 32, 256),
    ]


def test_resolve_gdn_decode_value_blocks_uses_preset_recommendations() -> None:
    assert _resolve_gdn_decode_value_blocks(None, ["arc-default"]) == [16]
    assert _resolve_gdn_decode_value_blocks(None, ["arc-legacy-v128-block8"]) == [8]
    assert _resolve_gdn_decode_value_blocks(None, ["arc-default", "arc-legacy-v256-block4"]) == [16, 4]


def test_resolve_gdn_decode_value_blocks_prefers_explicit_csv() -> None:
    assert _resolve_gdn_decode_value_blocks("4,8,16", ["arc-default"]) == [4, 8, 16]


def test_resolve_gdn_decode_value_blocks_falls_back_to_full_sweep_without_presets() -> None:
    assert _resolve_gdn_decode_value_blocks(None, []) == [1, 2, 4, 8, 16, 32]
