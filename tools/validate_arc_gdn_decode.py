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
DEFAULT_BENCH_TIMING_REPEATS = 5

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
        max_ratio=1.15,
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

DEFAULT_COMPARE_RATIO_DELTA = 0.03


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
    timing_repeats: int,
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
        "--gdn-decode-timing-repeats",
        str(timing_repeats),
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


def _load_json_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_benchmark_config_against_baseline(
    *,
    current_config: dict[str, object],
    baseline_report: dict[str, object],
) -> None:
    baseline_benchmark = baseline_report.get("benchmark")
    if not isinstance(baseline_benchmark, dict):
        raise ValueError("Baseline JSON is missing benchmark section")

    for field_name in ("seeds", "warmup", "iters", "timing_repeats"):
        if field_name not in baseline_benchmark:
            raise ValueError(
                f"Baseline JSON is missing benchmark.{field_name}; refresh the baseline with the current validator."
            )
        baseline_value = baseline_benchmark[field_name]
        current_value = current_config[field_name]
        if baseline_value != current_value:
            raise ValueError(
                f"Baseline benchmark config mismatch for {field_name}: baseline={baseline_value!r} current={current_value!r}"
            )


def _benchmark_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    value_block = row.get("value_block", row.get("default_value_block", ""))
    return (
        row["batch"],
        row["heads"],
        row["key_head_dim"],
        row["value_head_dim"],
        value_block,
    )


def _compare_benchmark_summary_against_baseline(
    *,
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
    max_ratio_delta: float,
) -> dict[str, object]:
    current_prefix = str(current_summary["compare_prefix"])
    baseline_prefix = str(baseline_summary["compare_prefix"])
    if current_prefix != baseline_prefix:
        raise ValueError(
            f"Preset {current_summary['preset']} compare_prefix changed from {baseline_prefix} to {current_prefix}"
        )

    current_value_blocks = tuple(int(value) for value in current_summary["resolved_value_blocks"])
    baseline_value_blocks = tuple(int(value) for value in baseline_summary["resolved_value_blocks"])
    if current_value_blocks != baseline_value_blocks:
        raise ValueError(
            f"Preset {current_summary['preset']} resolved value blocks changed from {baseline_value_blocks} "
            f"to {current_value_blocks}"
        )

    ratio_field = str(current_summary["ratio_field"])
    current_rows = current_summary["rows"]
    baseline_rows = baseline_summary["rows"]
    if not isinstance(current_rows, list) or not isinstance(baseline_rows, list):
        raise ValueError("Benchmark summary rows must be lists")

    current_map = {_benchmark_row_key(row): row for row in current_rows if isinstance(row, dict)}
    baseline_map = {_benchmark_row_key(row): row for row in baseline_rows if isinstance(row, dict)}
    current_keys = set(current_map)
    baseline_keys = set(baseline_map)
    if current_keys != baseline_keys:
        missing = sorted(baseline_keys - current_keys)
        extra = sorted(current_keys - baseline_keys)
        raise ValueError(
            f"Preset {current_summary['preset']} row keys changed; missing={missing[:3]} extra={extra[:3]}"
        )

    max_ratio_delta_observed = float("-inf")
    worst_key: tuple[str, str, str, str, str] | None = None
    worst_current_ratio = 0.0
    worst_baseline_ratio = 0.0
    per_row_deltas: list[dict[str, object]] = []
    for key in sorted(current_keys):
        current_row = current_map[key]
        baseline_row = baseline_map[key]
        current_ratio = float(current_row[ratio_field])
        baseline_ratio = float(baseline_row[ratio_field])
        ratio_delta = current_ratio - baseline_ratio
        if ratio_delta > max_ratio_delta_observed:
            max_ratio_delta_observed = ratio_delta
            worst_key = key
            worst_current_ratio = current_ratio
            worst_baseline_ratio = baseline_ratio
        per_row_deltas.append(
            {
                "batch": key[0],
                "heads": key[1],
                "key_head_dim": key[2],
                "value_head_dim": key[3],
                "value_block": key[4],
                "baseline_ratio": baseline_ratio,
                "current_ratio": current_ratio,
                "ratio_delta": ratio_delta,
            }
        )

    if max_ratio_delta_observed > max_ratio_delta:
        assert worst_key is not None
        raise ValueError(
            f"Preset {current_summary['preset']} exceeded ratio delta threshold {max_ratio_delta:.4f}: "
            f"baseline={worst_baseline_ratio:.4f} current={worst_current_ratio:.4f} "
            f"delta={max_ratio_delta_observed:.4f} on batch={worst_key[0]} heads={worst_key[1]} "
            f"value_dim={worst_key[3]} value_block={worst_key[4]}"
        )

    observed_max_ratio_delta = float(current_summary["observed_max_ratio"]) - float(baseline_summary["observed_max_ratio"])
    summary = {
        "preset": current_summary["preset"],
        "ratio_field": ratio_field,
        "max_ratio_delta_allowed": max_ratio_delta,
        "observed_max_ratio_delta": observed_max_ratio_delta,
        "observed_max_ratio_baseline": float(baseline_summary["observed_max_ratio"]),
        "observed_max_ratio_current": float(current_summary["observed_max_ratio"]),
        "max_row_ratio_delta": max_ratio_delta_observed,
        "row_deltas": per_row_deltas,
    }
    print(
        f"[compare:{current_summary['preset']}] max_row_ratio_delta={max_ratio_delta_observed:.4f} "
        f"observed_max_ratio_delta={observed_max_ratio_delta:.4f}",
        flush=True,
    )
    return summary


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
        "--timing-repeats",
        type=int,
        default=DEFAULT_BENCH_TIMING_REPEATS,
        help="Repeated timing samples per decode candidate forwarded to tools/bench_xpu_hotspots.py.",
    )
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
    parser.add_argument(
        "--compare-json",
        type=str,
        default=None,
        help="Optional existing Arc decode validation JSON to compare the current benchmark summaries against.",
    )
    parser.add_argument(
        "--compare-ratio-delta",
        type=float,
        default=DEFAULT_COMPARE_RATIO_DELTA,
        help="Maximum allowed increase in per-row speed-ratio metrics versus --compare-json.",
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
            "timing_repeats": args.timing_repeats,
            "skip_bench": args.skip_bench,
            "skip_bench_gates": args.skip_bench_gates,
            "presets": {},
        },
        "comparison": {
            "baseline_json": args.compare_json,
            "max_ratio_delta": args.compare_ratio_delta,
            "presets": {},
        },
        "pytest": {
            "skip_pytest": args.skip_pytest,
            "expression": args.pytest_k,
            "bench_unit_tests": None,
            "xpu_regressions": None,
        },
    }
    baseline_report: dict[str, object] | None = None
    if args.compare_json is not None:
        baseline_report = _load_json_report(Path(args.compare_json).expanduser())
        _validate_benchmark_config_against_baseline(
            current_config=json_report["benchmark"],
            baseline_report=baseline_report,
        )

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
                    timing_repeats=args.timing_repeats,
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
            if baseline_report is not None:
                baseline_benchmark = baseline_report.get("benchmark")
                if not isinstance(baseline_benchmark, dict):
                    raise ValueError("Baseline JSON is missing benchmark section")
                baseline_presets = baseline_benchmark.get("presets")
                if not isinstance(baseline_presets, dict) or preset_name not in baseline_presets:
                    raise ValueError(f"Baseline JSON does not include preset {preset_name}")
                comparison_summary = _compare_benchmark_summary_against_baseline(
                    current_summary=benchmark_summary,
                    baseline_summary=baseline_presets[preset_name],
                    max_ratio_delta=args.compare_ratio_delta,
                )
                cast_comparison = json_report["comparison"]
                assert isinstance(cast_comparison, dict)
                cast_comparison_presets = cast_comparison["presets"]
                assert isinstance(cast_comparison_presets, dict)
                cast_comparison_presets[preset_name] = comparison_summary

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
