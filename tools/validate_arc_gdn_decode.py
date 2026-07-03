from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ARC_DEFAULT_PRESET = "arc-default"
ARC_LEGACY_V128_BLOCK8_PRESET = "arc-legacy-v128-block8"
ARC_LEGACY_V256_BLOCK4_PRESET = "arc-legacy-v256-block4"

DEFAULT_PRESETS = (
    ARC_DEFAULT_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_LEGACY_V256_BLOCK4_PRESET,
)


@dataclass(frozen=True)
class ArcBenchExpectation:
    compare_prefix: str
    expected_value_blocks: tuple[int, ...]
    expected_row_count: int
    ratio_field: str
    max_ratio: float
    default_value_block: int | None = None
    default_strategy: str | None = None


ARC_BENCH_EXPECTATIONS = {
    ARC_DEFAULT_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_compare",
        expected_value_blocks=(16,),
        expected_row_count=13,
        ratio_field="default_speed_ratio",
        max_ratio=1.12,
        default_value_block=16,
        default_strategy="tiled",
    ),
    ARC_LEGACY_V128_BLOCK8_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(8,),
        expected_row_count=9,
        ratio_field="auto_speed_ratio",
        max_ratio=1.08,
    ),
    ARC_LEGACY_V256_BLOCK4_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(4,),
        expected_row_count=9,
        ratio_field="auto_speed_ratio",
        max_ratio=1.10,
    ),
}

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


def _run_step(
    *,
    args: list[str],
    cwd: Path,
    env: dict[str, str],
    label: str,
    capture_output: bool = False,
) -> str | None:
    print(f"[{label}] {' '.join(args)}", flush=True)
    if not capture_output:
        completed = subprocess.run(args, cwd=cwd, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return None

    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    return "".join(output_lines)


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


def _parse_gdn_decode_value_blocks(output: str) -> tuple[int, ...]:
    for line in output.splitlines():
        if not line.startswith("gdn_decode_value_blocks="):
            continue
        return tuple(int(part) for part in line.split("=", maxsplit=1)[1].split(",") if part.strip())
    raise ValueError("Benchmark output did not include gdn_decode_value_blocks=...")


def _parse_gdn_decode_csv_rows(output: str, prefix: str) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line.startswith(prefix + ","):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) > 1 and parts[1] == "device_name":
            header = parts
            continue
        if header is None:
            raise ValueError(f"Encountered {prefix} row before header")
        if len(parts) != len(header):
            raise ValueError(f"{prefix} row had {len(parts)} columns, expected {len(header)}")
        rows.append(dict(zip(header, parts)))
    if not rows:
        raise ValueError(f"Benchmark output did not include any {prefix} rows")
    return rows


def _validate_benchmark_output(output: str, preset_name: str) -> None:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    value_blocks = _parse_gdn_decode_value_blocks(output)
    if value_blocks != expectation.expected_value_blocks:
        raise ValueError(
            f"{preset_name} resolved value blocks {value_blocks}, expected {expectation.expected_value_blocks}"
        )

    rows = _parse_gdn_decode_csv_rows(output, expectation.compare_prefix)
    if len(rows) != expectation.expected_row_count:
        raise ValueError(f"{preset_name} produced {len(rows)} compare rows, expected {expectation.expected_row_count}")

    if expectation.default_value_block is not None:
        bad_rows = [
            row for row in rows if int(row["default_value_block"]) != expectation.default_value_block
        ]
        if bad_rows:
            raise ValueError(
                f"{preset_name} emitted default_value_block values outside {expectation.default_value_block}: {bad_rows[0]}"
            )
    if expectation.default_strategy is not None:
        bad_rows = [row for row in rows if row["default_strategy"] != expectation.default_strategy]
        if bad_rows:
            raise ValueError(
                f"{preset_name} emitted default_strategy values outside {expectation.default_strategy}: {bad_rows[0]}"
            )
    if "value_block" in rows[0]:
        bad_rows = [row for row in rows if int(row["value_block"]) not in expectation.expected_value_blocks]
        if bad_rows:
            raise ValueError(
                f"{preset_name} emitted unexpected value_block outside {expectation.expected_value_blocks}: {bad_rows[0]}"
            )

    worst_row = max(rows, key=lambda row: float(row[expectation.ratio_field]))
    worst_ratio = float(worst_row[expectation.ratio_field])
    if worst_ratio > expectation.max_ratio:
        raise ValueError(
            f"{preset_name} exceeded {expectation.ratio_field} threshold {expectation.max_ratio:.3f}: "
            f"observed {worst_ratio:.4f} on batch={worst_row['batch']} heads={worst_row['heads']} "
            f"value_dim={worst_row['value_head_dim']}"
        )
    print(
        f"[validate:{preset_name}] max_{expectation.ratio_field}={worst_ratio:.4f} "
        f"rows={len(rows)} value_blocks={value_blocks}",
        flush=True,
    )


def _collect_benchmark_summary(output: str, preset_name: str) -> dict[str, object]:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    value_blocks = _parse_gdn_decode_value_blocks(output)
    rows = _parse_gdn_decode_csv_rows(output, expectation.compare_prefix)
    worst_row = max(rows, key=lambda row: float(row[expectation.ratio_field]))
    worst_ratio = float(worst_row[expectation.ratio_field])
    return {
        "preset": preset_name,
        "compare_prefix": expectation.compare_prefix,
        "expected_value_blocks": list(expectation.expected_value_blocks),
        "resolved_value_blocks": list(value_blocks),
        "ratio_field": expectation.ratio_field,
        "max_ratio_allowed": expectation.max_ratio,
        "observed_max_ratio": worst_ratio,
        "row_count": len(rows),
        "rows": rows,
        "worst_row": worst_row,
    }


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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
        "--skip-bench-gates",
        action="store_true",
        help="Skip parsing benchmark output against Arc preset performance thresholds.",
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
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write a structured Arc decode validation summary JSON.",
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
    json_report: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "op_lib_path": str(op_lib_path),
        "presets": preset_names,
        "benchmark": {
            "seeds": args.seeds,
            "warmup": args.warmup,
            "iters": args.iters,
            "skip_bench": args.skip_bench,
            "skip_bench_gates": args.skip_bench_gates,
            "presets": {},
        },
        "pytest": {
            "skip_pytest": args.skip_pytest,
            "expression": args.pytest_k,
            "bench_unit_tests": None,
            "xpu_regressions": None,
        },
    }

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
            output = _run_step(
                args=_bench_args_for_preset(
                    preset_name=preset_name,
                    warmup=args.warmup,
                    iters=args.iters,
                    seeds_csv=args.seeds,
                ),
                cwd=root,
                env=env,
                label=f"bench:{preset_name}",
                capture_output=True,
            )
            if not args.skip_bench_gates:
                assert output is not None
                _validate_benchmark_output(output, preset_name)
            assert output is not None
            benchmark_summary = _collect_benchmark_summary(output, preset_name)
            cast_benchmark = json_report["benchmark"]
            assert isinstance(cast_benchmark, dict)
            cast_presets = cast_benchmark["presets"]
            assert isinstance(cast_presets, dict)
            cast_presets[preset_name] = benchmark_summary

    if not args.skip_pytest:
        bench_pytest_args = [sys.executable, "-m", "pytest", "tests/test_bench_xpu_hotspots.py", "-q"]
        _run_step(
            args=bench_pytest_args,
            cwd=root,
            env=env,
            label="pytest:bench",
        )
        bench_pytest = json_report["pytest"]
        assert isinstance(bench_pytest, dict)
        bench_pytest["bench_unit_tests"] = {
            "args": bench_pytest_args,
            "status": "passed",
        }
        xpu_pytest_args = [sys.executable, "-m", "pytest", "tests/test_fused_op_xpu.py", "-k", args.pytest_k, "-q"]
        _run_step(
            args=xpu_pytest_args,
            cwd=root,
            env=env,
            label="pytest:xpu",
        )
        bench_pytest["xpu_regressions"] = {
            "args": xpu_pytest_args,
            "status": "passed",
        }

    if args.json_output is not None:
        _write_json_report(Path(args.json_output).expanduser(), json_report)
        print(f"[json] wrote {Path(args.json_output).expanduser()}", flush=True)


if __name__ == "__main__":
    main()
