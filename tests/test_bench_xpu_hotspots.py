from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import bench_xpu_hotspots as bench_hotspots
from tools.bench_xpu_hotspots import (
    GDN_DECODE_PRESET_VALUE_BLOCKS,
    GDN_DECODE_SHAPE_PRESETS,
    _collect_gated_delta_decode_candidate_timings,
    _dedupe_gdn_decode_shape_cases,
    _gated_delta_decode_candidate_id,
    _parse_gdn_decode_shape_presets,
    _resolve_gdn_decode_value_blocks,
    _run_gated_delta_decode_profile_case,
)


def test_gdn_decode_shape_presets_have_matching_value_block_metadata() -> None:
    assert set(GDN_DECODE_SHAPE_PRESETS) == set(GDN_DECODE_PRESET_VALUE_BLOCKS)


def test_arc_legacy_v256_block4_preset_keeps_upper_edge_regression_shapes() -> None:
    preset = GDN_DECODE_SHAPE_PRESETS["arc-legacy-v256-block4"]
    assert (146, 8, 256) in preset
    assert (73, 16, 256) in preset
    assert (37, 32, 256) in preset


def test_arc_watch_v256_block4_preset_tracks_boundary_watch_shapes() -> None:
    assert GDN_DECODE_SHAPE_PRESETS["arc-watch-v256-block4"] == (
        (33, 8, 256),
        (144, 8, 256),
        (145, 8, 256),
        (146, 8, 256),
        (71, 16, 256),
        (72, 16, 256),
        (73, 16, 256),
        (23, 32, 256),
        (24, 32, 256),
        (25, 32, 256),
        (35, 32, 256),
        (36, 32, 256),
        (37, 32, 256),
    )


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


def test_gated_delta_decode_candidate_id_formats_default_and_explicit_blocks() -> None:
    assert _gated_delta_decode_candidate_id("auto", None) == "auto@default"
    assert _gated_delta_decode_candidate_id("tiled", 16) == "tiled@16"


def test_collect_gated_delta_decode_candidate_timings_rotates_candidates_and_uses_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    call_counts: dict[str, int] = {}
    base_ms = {"single@8": 1.0, "tiled@8": 2.0, "auto@8": 3.0}

    def fake_benchmark_gated_delta_decode_strategy(**kwargs):
        candidate_id = f"{kwargs['strategy']}@{kwargs['value_block']}"
        call_order.append(candidate_id)
        sample_idx = call_counts.get(candidate_id, 0) + 1
        call_counts[candidate_id] = sample_idx
        return base_ms[candidate_id] + 0.1 * sample_idx, 0.01 * sample_idx

    monkeypatch.setattr(
        bench_hotspots,
        "_benchmark_gated_delta_decode_strategy",
        fake_benchmark_gated_delta_decode_strategy,
    )

    measurements = _collect_gated_delta_decode_candidate_timings(
        query=None,
        key=None,
        value=None,
        g=None,
        beta=None,
        initial_state=None,
        candidates=[
            ("single", "single", 8),
            ("tiled", "tiled", 8),
            ("auto", "auto", 8),
        ],
        single_min_elements=None,
        warmup=1,
        iters=1,
        timing_repeats=3,
    )

    assert call_order == [
        "single@8",
        "tiled@8",
        "auto@8",
        "tiled@8",
        "auto@8",
        "single@8",
        "auto@8",
        "single@8",
        "tiled@8",
    ]
    assert measurements["single"]["candidate_ms"] == pytest.approx(1.2)
    assert measurements["tiled"]["candidate_ms"] == pytest.approx(2.2)
    assert measurements["auto"]["candidate_ms"] == pytest.approx(3.2)
    assert measurements["auto"]["max_abs_diff"] == pytest.approx(0.03)


def test_run_gated_delta_decode_profile_case_reuses_profile_measurements_for_auto_compare(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    single_id = _gated_delta_decode_candidate_id("single", 8)
    tiled_id = _gated_delta_decode_candidate_id("tiled", 8)
    auto_id = _gated_delta_decode_candidate_id("auto", 8)

    monkeypatch.setattr(
        bench_hotspots,
        "_make_gated_delta_decode_bench_inputs",
        lambda **kwargs: ("query", "key", "value", "g", "beta", "state"),
    )
    monkeypatch.setattr(
        bench_hotspots,
        "_collect_gated_delta_decode_candidate_timings",
        lambda **kwargs: {
            single_id: {"candidate_ms": 1.0, "max_abs_diff": 0.1},
            tiled_id: {"candidate_ms": 2.0, "max_abs_diff": 0.2},
            auto_id: {"candidate_ms": 1.5, "max_abs_diff": 0.3},
        },
    )
    monkeypatch.setattr(
        bench_hotspots,
        "_compare_gated_delta_decode_auto_against_explicit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("compare helper should not be called")),
    )
    monkeypatch.setattr(bench_hotspots, "_resolve_gated_delta_decode_auto_strategy", lambda *args, **kwargs: "single")
    monkeypatch.setattr(bench_hotspots.torch, "empty", lambda *args, **kwargs: object())

    _run_gated_delta_decode_profile_case(
        device_name="Test GPU",
        batch_size=1,
        num_heads=8,
        key_head_dim=128,
        value_head_dim=128,
        dtype=bench_hotspots.torch.bfloat16,
        value_blocks=[8],
        single_min_elements=None,
        warmup=1,
        iters=1,
        timing_repeats=3,
        seeds=[0],
        auto_compare=True,
        default_compare=False,
        default_block_compare=False,
        compare_only=False,
    )

    output_lines = capsys.readouterr().out.strip().splitlines()
    assert any(
        line
        == "gdn_decode_auto_compare,Test GPU,1,8,128,128,8,single,1.0000,2.0000,1.5000,single,1.0000,0.5000,1.5000,0.300000"
        for line in output_lines
    )
