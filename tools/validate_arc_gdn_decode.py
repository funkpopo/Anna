from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ARC_DEFAULT_PRESET = "arc-default"
ARC_LEGACY_V128_BLOCK8_PRESET = "arc-legacy-v128-block8"
ARC_LEGACY_V256_BLOCK4_PRESET = "arc-legacy-v256-block4"

DEFAULT_PRESETS = (
    ARC_DEFAULT_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_LEGACY_V256_BLOCK4_PRESET,
)

DEFAULT_PYTEST_EXPR = (
    "gated_delta_decode_value_block_debug or "
    "gated_delta_decode_strategy_debug_matches_qwen35_family_lookup or "
    "gated_delta_decode_strategy_debug_matches_arc_lookup or "
    "gated_delta_decode_xpu_auto_matches_qwen35_family_shapes or "
    "gated_delta_decode_xpu_auto_matches_arc_row_cutover_shapes or "
    "gated_delta_decode_xpu_specialized_k128_shapes_match_reference"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_op_lib_path(root: Path) -> Path:
    suffix = ".pyd" if os.name == "nt" else ".so"
    return root / ".build" / "anna_gated_delta_fused" / f"anna_gated_delta_fused{suffix}"


def _pythonpath_value(root: Path) -> str:
    return os.pathsep.join((str(root / "src"), str(root)))


def _run_step(*, args: list[str], cwd: Path, env: dict[str, str], label: str) -> None:
    print(f"[{label}] {' '.join(args)}", flush=True)
    completed = subprocess.run(args, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _bench_args_for_preset(
    *,
    preset_name: str,
    warmup: int,
    iters: int,
    seeds_csv: str,
) -> list[str]:
    compare_flag = "--gdn-decode-default-compare" if preset_name == ARC_DEFAULT_PRESET else "--gdn-decode-auto-compare"
    return [
        sys.executable,
        "tools/bench_xpu_hotspots.py",
        "--gdn-decode-only",
        compare_flag,
        "--gdn-decode-compare-only",
        "--head-dim",
        "128",
        "--gdn-decode-shape-presets",
        preset_name,
        "--gdn-decode-seeds",
        seeds_csv,
        "--dtype",
        "bf16",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]


def _parse_preset_names(raw: str) -> list[str]:
    preset_names = [item.strip() for item in raw.split(",") if item.strip()]
    if not preset_names:
        raise ValueError("--presets must contain at least one preset name")
    unknown = [preset_name for preset_name in preset_names if preset_name not in DEFAULT_PRESETS]
    if unknown:
        raise ValueError(
            f"Unknown preset(s): {', '.join(unknown)}. Available presets: {', '.join(DEFAULT_PRESETS)}"
        )
    return preset_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standard Arc Gated Delta decode benchmark presets and targeted regressions."
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(DEFAULT_PRESETS),
        help=(
            "Comma-separated Arc decode presets to benchmark. "
            f"Available presets: {', '.join(DEFAULT_PRESETS)}."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="20260960,20260961",
        help="Comma-separated fixed benchmark seeds forwarded to tools/bench_xpu_hotspots.py.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark measured iterations.")
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Skip the Arc decode benchmark preset commands.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip the targeted pytest regressions.",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Rebuild the fused op before validation by running tools/build_gated_delta_fused_op.py.",
    )
    parser.add_argument(
        "--op-lib",
        type=str,
        default=None,
        help="Explicit fused-op library path. Defaults to the local .build output.",
    )
    parser.add_argument(
        "--pytest-k",
        type=str,
        default=DEFAULT_PYTEST_EXPR,
        help="Expression forwarded to pytest -k for the XPU decode regression subset.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    preset_names = _parse_preset_names(args.presets)
    root = _repo_root()
    op_lib_path = Path(args.op_lib).expanduser() if args.op_lib is not None else _default_op_lib_path(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_value(root)
    env["ANNA_GATED_DELTA_OP_LIB"] = str(op_lib_path)

    if args.build_first:
        _run_step(
            args=[sys.executable, "tools/build_gated_delta_fused_op.py"],
            cwd=root,
            env=env,
            label="build",
        )

    if not op_lib_path.exists():
        raise SystemExit(
            f"Fused-op library not found at {op_lib_path}. Build it first or pass --op-lib explicitly."
        )

    if not args.skip_bench:
        for preset_name in preset_names:
            _run_step(
                args=_bench_args_for_preset(
                    preset_name=preset_name,
                    warmup=args.warmup,
                    iters=args.iters,
                    seeds_csv=args.seeds,
                ),
                cwd=root,
                env=env,
                label=f"bench:{preset_name}",
            )

    if not args.skip_pytest:
        _run_step(
            args=[sys.executable, "-m", "pytest", "tests/test_bench_xpu_hotspots.py", "-q"],
            cwd=root,
            env=env,
            label="pytest:bench",
        )
        _run_step(
            args=[sys.executable, "-m", "pytest", "tests/test_fused_op_xpu.py", "-k", args.pytest_k, "-q"],
            cwd=root,
            env=env,
            label="pytest:xpu",
        )


if __name__ == "__main__":
    main()
