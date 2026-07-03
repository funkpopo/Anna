from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_arc_gdn_decode import (  # noqa: E402
    ARC_DEFAULT_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_LEGACY_V256_BLOCK4_PRESET,
    _bench_args_for_preset,
    _parse_preset_names,
)


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
        seeds_csv="20260960,20260961",
    )
    assert expected_compare_flag in args
    assert "--gdn-decode-shape-presets" in args
    assert args[args.index("--gdn-decode-shape-presets") + 1] == preset_name
    assert args[args.index("--warmup") + 1] == "7"
    assert args[args.index("--iters") + 1] == "11"
