"""Unit tests for the XPU device capability abstraction (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anna.runtime.capability import resolve_capability  # noqa: E402


def test_arc_a770_resolves_alchemist_high_tier():
    cap = resolve_capability(0, "Intel(R) Arc(TM) A770 Graphics")
    assert cap.arch_family == "alchemist"
    assert cap.arch_generation == "xe-hpg"
    assert cap.supports_xmx is True
    assert cap.supports_dp4a is True
    assert cap.eu_count == 512
    assert cap.performance_tier == "high"
    assert "acm-g10" in cap.notes


def test_arc_a750_resolves_alchemist_high_tier():
    cap = resolve_capability(0, "Intel(R) Arc(TM) A750 Graphics")
    assert cap.arch_family == "alchemist"
    assert cap.performance_tier == "high"


def test_battlemage_resolves_without_string_match_on_a_series():
    cap = resolve_capability(0, "Intel(R) Arc(TM) B580 Graphics")
    assert cap.arch_family == "battlemage"
    assert cap.arch_generation == "xe2-hpg"
    assert cap.performance_tier == "high"


def test_unknown_device_falls_back_to_generic_profile():
    cap = resolve_capability(0, "Some Future Intel GPU 9000")
    assert cap.arch_family == "generic"
    assert cap.arch_generation == "unknown"
    # Conservative tier: new devices work out of the box with default strategies.
    assert cap.performance_tier == "low"
    assert cap.eu_count is None


def test_none_name_uses_index_placeholder_and_generic_profile():
    cap = resolve_capability(2, None)
    assert cap.device_index == 2
    assert cap.name == "xpu:2"
    assert cap.arch_family == "generic"


def test_data_center_gpu_max_resolves_pvc():
    cap = resolve_capability(0, "Intel(R) Data Center GPU Max 1100")
    assert cap.arch_family == "pvc"
    assert cap.performance_tier == "high"
    assert 16 in cap.subgroup_sizes


@pytest.mark.parametrize(
    ("name", "expected_family"),
    [
        ("Intel(R) Arc(TM) A770 Graphics", "alchemist"),
        ("Intel(R) Arc(TM) B580 Graphics", "battlemage"),
        ("Intel(R) Data Center GPU Flex 170", "ats"),
        ("Totally unknown card", "generic"),
    ],
)
def test_as_log_fields_shape(name: str, expected_family: str):
    cap = resolve_capability(0, name)
    fields = cap.as_log_fields()
    assert fields["arch_family"] == expected_family
    assert set(fields) == {
        "arch_family",
        "arch_generation",
        "supports_xmx",
        "supports_dp4a",
        "eu_count",
        "performance_tier",
    }
