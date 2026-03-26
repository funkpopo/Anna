from __future__ import annotations

from pathlib import Path

from anna.cli.serve import _build_safety_policy, build_parser
from anna.core.config import ServeSettings


def test_build_safety_policy_uses_custom_serve_overrides() -> None:
    settings = ServeSettings(
        model_dir=Path("dummy"),
        min_free_memory_mib=256,
        reserve_memory_mib=128,
        max_estimated_usage_ratio=0.95,
        generation_memory_safety_factor=1.25,
    )

    policy = _build_safety_policy(settings)

    assert policy is not None
    assert policy.min_free_bytes == 256 << 20
    assert policy.reserve_margin_bytes == 128 << 20
    assert policy.max_estimated_usage_ratio == 0.95
    assert policy.generation_memory_safety_factor == 1.25


def test_serve_parser_accepts_memory_guard_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--min-free-memory-mib",
            "256",
            "--reserve-memory-mib",
            "128",
            "--max-estimated-usage-ratio",
            "0.95",
            "--generation-memory-safety-factor",
            "1.25",
        ]
    )

    assert args.min_free_memory_mib == 256
    assert args.reserve_memory_mib == 128
    assert args.max_estimated_usage_ratio == 0.95
    assert args.generation_memory_safety_factor == 1.25
