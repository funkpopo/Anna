from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARC_DEFAULT_PRESET = "arc-default"
ARC_V64_DEFAULT_BLOCK16_PRESET = "arc-v64-default-block16"
ARC_LEGACY_V64_BLOCK8_PRESET = "arc-legacy-v64-block8"
ARC_LEGACY_V128_BLOCK8_PRESET = "arc-legacy-v128-block8"
ARC_WATCH_V128_BLOCK8_PRESET = "arc-watch-v128-block8"
ARC_WATCH_V128_DEFAULT8_VS_BLOCK16_PRESET = "arc-watch-v128-default8-vs-block16"
ARC_LEGACY_V256_BLOCK4_PRESET = "arc-legacy-v256-block4"
ARC_WATCH_V256_BLOCK4_PRESET = "arc-watch-v256-block4"
ARC_WATCH_V256_DEFAULT4_VS_BLOCK8_PRESET = "arc-watch-v256-default4-vs-block8"
ARC_WATCH_V256_DEFAULT8_VS_BLOCK4_PRESET = "arc-watch-v256-default8-vs-block4"
FULL_PRESET_ALIAS = "full"
QUICK_PRESET_ALIAS = "quick"
WATCH_PRESET_ALIAS = "watch"
DEFAULT_BENCH_TIMING_REPEATS = 5

DEFAULT_PRESETS = (
    ARC_DEFAULT_PRESET,
    ARC_V64_DEFAULT_BLOCK16_PRESET,
    ARC_LEGACY_V64_BLOCK8_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_LEGACY_V256_BLOCK4_PRESET,
)
ALL_PRESETS = DEFAULT_PRESETS + (
    ARC_WATCH_V128_BLOCK8_PRESET,
    ARC_WATCH_V128_DEFAULT8_VS_BLOCK16_PRESET,
    ARC_WATCH_V256_BLOCK4_PRESET,
    ARC_WATCH_V256_DEFAULT4_VS_BLOCK8_PRESET,
    ARC_WATCH_V256_DEFAULT8_VS_BLOCK4_PRESET,
)
QUICK_PRESETS = (
    ARC_DEFAULT_PRESET,
    ARC_V64_DEFAULT_BLOCK16_PRESET,
    ARC_LEGACY_V64_BLOCK8_PRESET,
    ARC_LEGACY_V128_BLOCK8_PRESET,
    ARC_WATCH_V256_BLOCK4_PRESET,
)
PRESET_ALIASES: dict[str, tuple[str, ...]] = {
    FULL_PRESET_ALIAS: DEFAULT_PRESETS,
    QUICK_PRESET_ALIAS: QUICK_PRESETS,
    WATCH_PRESET_ALIAS: (
        ARC_WATCH_V128_BLOCK8_PRESET,
        ARC_WATCH_V128_DEFAULT8_VS_BLOCK16_PRESET,
        ARC_WATCH_V256_BLOCK4_PRESET,
        ARC_WATCH_V256_DEFAULT4_VS_BLOCK8_PRESET,
        ARC_WATCH_V256_DEFAULT8_VS_BLOCK4_PRESET,
    ),
}
ALL_PRESET_TOKENS = tuple(PRESET_ALIASES) + ALL_PRESETS


@dataclass(frozen=True)
class ArcBenchExpectation:
    compare_prefix: str
    expected_value_blocks: tuple[int, ...]
    expected_row_count: int
    ratio_field: str
    max_ratio: float
    default_compare_ratio_delta: float | None = None
    default_value_block: int | None = None
    default_strategy: str | None = None


ARC_BENCH_EXPECTATIONS = {
    ARC_DEFAULT_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_compare",
        expected_value_blocks=(4, 8),
        expected_row_count=13,
        ratio_field="default_speed_ratio",
        max_ratio=1.15,
        default_compare_ratio_delta=0.025,
    ),
    ARC_V64_DEFAULT_BLOCK16_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_compare",
        expected_value_blocks=(8,),
        expected_row_count=11,
        ratio_field="default_speed_ratio",
        max_ratio=1.03,
        default_compare_ratio_delta=0.015,
        default_value_block=8,
        default_strategy="single",
    ),
    ARC_LEGACY_V64_BLOCK8_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(8,),
        expected_row_count=10,
        ratio_field="auto_speed_ratio",
        max_ratio=1.02,
        default_compare_ratio_delta=0.02,
    ),
    ARC_LEGACY_V128_BLOCK8_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(8,),
        expected_row_count=12,
        ratio_field="auto_speed_ratio",
        max_ratio=1.08,
        default_compare_ratio_delta=0.01,
    ),
    ARC_WATCH_V128_BLOCK8_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(8,),
        expected_row_count=8,
        ratio_field="auto_speed_ratio",
        max_ratio=1.02,
        default_compare_ratio_delta=0.005,
    ),
    ARC_WATCH_V128_DEFAULT8_VS_BLOCK16_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_block_compare",
        expected_value_blocks=(16,),
        expected_row_count=10,
        ratio_field="default_speed_ratio_vs_forced",
        max_ratio=1.02,
        default_compare_ratio_delta=0.01,
        default_value_block=8,
        default_strategy="tiled",
    ),
    ARC_LEGACY_V256_BLOCK4_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(4,),
        expected_row_count=39,
        ratio_field="auto_speed_ratio",
        max_ratio=1.03,
        default_compare_ratio_delta=0.015,
    ),
    ARC_WATCH_V256_BLOCK4_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_auto_compare",
        expected_value_blocks=(4,),
        expected_row_count=13,
        ratio_field="auto_speed_ratio",
        max_ratio=1.02,
        default_compare_ratio_delta=0.005,
    ),
    ARC_WATCH_V256_DEFAULT4_VS_BLOCK8_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_block_compare",
        expected_value_blocks=(8,),
        expected_row_count=14,
        ratio_field="default_speed_ratio_vs_forced",
        max_ratio=1.02,
        default_compare_ratio_delta=0.01,
        default_value_block=4,
        default_strategy="tiled",
    ),
    ARC_WATCH_V256_DEFAULT8_VS_BLOCK4_PRESET: ArcBenchExpectation(
        compare_prefix="gdn_decode_default_block_compare",
        expected_value_blocks=(4,),
        expected_row_count=15,
        ratio_field="default_speed_ratio_vs_forced",
        max_ratio=1.02,
        default_compare_ratio_delta=0.01,
        default_value_block=8,
        default_strategy="tiled",
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
DEFAULT_VALIDATE_BENCH_CHUNK_THRESHOLD = 32
DEFAULT_VALIDATE_BENCH_MAX_SHAPES_PER_SCAN = 16
DEFAULT_VALIDATE_BENCH_PRIME_WARMUP = 1
DEFAULT_VALIDATE_BENCH_PRIME_ITERS = 1
DEFAULT_VALIDATE_BENCH_PRIME_TIMING_REPEATS = 1
DEFAULT_VALIDATE_CONFIRM_WARMUP = 40
DEFAULT_VALIDATE_CONFIRM_ITERS = 300
DEFAULT_VALIDATE_CONFIRM_TIMING_REPEATS = 9


def _repo_root() -> Path:
    return REPO_ROOT


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
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    compare_flag = {
        "gdn_decode_default_compare": "--gdn-decode-default-compare",
        "gdn_decode_auto_compare": "--gdn-decode-auto-compare",
        "gdn_decode_default_block_compare": "--gdn-decode-default-block-compare",
    }[expectation.compare_prefix]
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


def _bench_args_for_shape_chunk(
    *,
    preset_name: str,
    shape_cases: list[tuple[int, int, int]],
    warmup: int,
    iters: int,
    timing_repeats: int,
    seeds_csv: str,
) -> list[str]:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    compare_flag = {
        "gdn_decode_default_compare": "--gdn-decode-default-compare",
        "gdn_decode_auto_compare": "--gdn-decode-auto-compare",
        "gdn_decode_default_block_compare": "--gdn-decode-default-block-compare",
    }[expectation.compare_prefix]
    shape_cases_csv = ",".join(f"{batch_size}x{num_heads}x{value_head_dim}" for batch_size, num_heads, value_head_dim in shape_cases)
    args = [
        sys.executable,
        "tools/bench_xpu_hotspots.py",
        "--gdn-decode-only",
        compare_flag,
        "--gdn-decode-compare-only",
        "--head-dim",
        "128",
        "--gdn-decode-shape-cases",
        shape_cases_csv,
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
    if expectation.expected_value_blocks:
        args.extend(
            [
                "--gdn-decode-value-blocks",
                ",".join(str(value_block) for value_block in expectation.expected_value_blocks),
            ]
        )
    return args


def _resolve_validate_bench_max_shapes_per_scan(
    requested_max_shapes_per_scan: int | None,
    *,
    shape_count: int,
) -> int | None:
    if requested_max_shapes_per_scan is not None:
        return requested_max_shapes_per_scan
    if shape_count > DEFAULT_VALIDATE_BENCH_CHUNK_THRESHOLD:
        return DEFAULT_VALIDATE_BENCH_MAX_SHAPES_PER_SCAN
    return None


def _chunk_shape_cases(
    shape_cases: list[tuple[int, int, int]],
    *,
    max_shapes_per_scan: int | None,
) -> list[list[tuple[int, int, int]]]:
    if max_shapes_per_scan is None or max_shapes_per_scan <= 0 or len(shape_cases) <= max_shapes_per_scan:
        return [list(shape_cases)]
    return [
        shape_cases[start : start + max_shapes_per_scan]
        for start in range(0, len(shape_cases), max_shapes_per_scan)
    ]


def _parse_benchmark_output_rows_and_value_blocks(output: str, preset_name: str) -> tuple[tuple[int, ...], list[dict[str, str]]]:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    value_blocks = _parse_gdn_decode_value_blocks(output)
    rows = _parse_gdn_decode_csv_rows(output, expectation.compare_prefix)
    return value_blocks, rows


def _validate_benchmark_rows(
    *,
    preset_name: str,
    value_blocks: tuple[int, ...],
    rows: list[dict[str, str]],
) -> None:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    if value_blocks != expectation.expected_value_blocks:
        raise ValueError(
            f"{preset_name} resolved value blocks {value_blocks}, expected {expectation.expected_value_blocks}"
        )

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
    if rows and "value_block" in rows[0]:
        bad_rows = [row for row in rows if int(row["value_block"]) not in expectation.expected_value_blocks]
        if bad_rows:
            raise ValueError(
                f"{preset_name} emitted unexpected value_block outside {expectation.expected_value_blocks}: {bad_rows[0]}"
            )
    if rows and "forced_value_block" in rows[0]:
        bad_rows = [row for row in rows if int(row["forced_value_block"]) not in expectation.expected_value_blocks]
        if bad_rows:
            raise ValueError(
                f"{preset_name} emitted unexpected forced_value_block outside {expectation.expected_value_blocks}: "
                f"{bad_rows[0]}"
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


def _collect_benchmark_summary_from_rows(
    *,
    preset_name: str,
    value_blocks: tuple[int, ...],
    rows: list[dict[str, str]],
) -> dict[str, object]:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
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


def _run_benchmark_for_preset(
    *,
    preset_name: str,
    warmup: int,
    iters: int,
    timing_repeats: int,
    seeds_csv: str,
    cwd: Path,
    env: dict[str, str],
    max_shapes_per_scan: int | None,
) -> tuple[list[str], list[list[str]], tuple[int, ...], list[dict[str, str]], int | None]:
    from tools.bench_xpu_hotspots import GDN_DECODE_SHAPE_PRESETS

    requested_bench_args = _bench_args_for_preset(
        preset_name=preset_name,
        warmup=warmup,
        iters=iters,
        timing_repeats=timing_repeats,
        seeds_csv=seeds_csv,
    )
    shape_cases = list(GDN_DECODE_SHAPE_PRESETS[preset_name])
    effective_max_shapes_per_scan = _resolve_validate_bench_max_shapes_per_scan(
        max_shapes_per_scan,
        shape_count=len(shape_cases),
    )
    case_chunks = _chunk_shape_cases(shape_cases, max_shapes_per_scan=effective_max_shapes_per_scan)
    if len(case_chunks) > 1:
        print(
            f"[chunking:{preset_name}] shape_count={len(shape_cases)} "
            f"max_shapes_per_scan={effective_max_shapes_per_scan} chunks={len(case_chunks)}",
            flush=True,
        )

    bench_runs: list[list[str]] = []
    rows: list[dict[str, str]] = []
    resolved_value_blocks: tuple[int, ...] | None = None
    for chunk_index, case_chunk in enumerate(case_chunks, start=1):
        if len(case_chunks) > 1:
            prime_args = _bench_args_for_shape_chunk(
                preset_name=preset_name,
                shape_cases=case_chunk,
                warmup=DEFAULT_VALIDATE_BENCH_PRIME_WARMUP,
                iters=DEFAULT_VALIDATE_BENCH_PRIME_ITERS,
                timing_repeats=DEFAULT_VALIDATE_BENCH_PRIME_TIMING_REPEATS,
                seeds_csv=seeds_csv,
            )
            _run_step(
                args=prime_args,
                cwd=cwd,
                env=env,
                label=f"bench-prime:{preset_name}:{chunk_index}/{len(case_chunks)}",
                capture_output=False,
            )
        bench_args = (
            _bench_args_for_preset(
                preset_name=preset_name,
                warmup=warmup,
                iters=iters,
                timing_repeats=timing_repeats,
                seeds_csv=seeds_csv,
            )
            if len(case_chunks) == 1
            else _bench_args_for_shape_chunk(
                preset_name=preset_name,
                shape_cases=case_chunk,
                warmup=warmup,
                iters=iters,
                timing_repeats=timing_repeats,
                seeds_csv=seeds_csv,
            )
        )
        bench_runs.append(bench_args)
        label = f"bench:{preset_name}" if len(case_chunks) == 1 else f"bench:{preset_name}:{chunk_index}/{len(case_chunks)}"
        output = _run_step(
            args=bench_args,
            cwd=cwd,
            env=env,
            label=label,
            capture_output=True,
        )
        assert output is not None
        chunk_value_blocks, chunk_rows = _parse_benchmark_output_rows_and_value_blocks(output, preset_name)
        if resolved_value_blocks is None:
            resolved_value_blocks = chunk_value_blocks
        elif chunk_value_blocks != resolved_value_blocks:
            raise ValueError(
                f"{preset_name} resolved inconsistent value blocks across chunks: "
                f"{resolved_value_blocks} vs {chunk_value_blocks}"
            )
        rows.extend(chunk_rows)
    assert resolved_value_blocks is not None
    return requested_bench_args, bench_runs, resolved_value_blocks, rows, effective_max_shapes_per_scan


def _run_benchmark_for_shape_cases(
    *,
    preset_name: str,
    shape_cases: list[tuple[int, int, int]],
    warmup: int,
    iters: int,
    timing_repeats: int,
    seeds_csv: str,
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> tuple[list[str], tuple[int, ...], list[dict[str, str]]]:
    bench_args = _bench_args_for_shape_chunk(
        preset_name=preset_name,
        shape_cases=shape_cases,
        warmup=warmup,
        iters=iters,
        timing_repeats=timing_repeats,
        seeds_csv=seeds_csv,
    )
    output = _run_step(
        args=bench_args,
        cwd=cwd,
        env=env,
        label=label,
        capture_output=True,
    )
    assert output is not None
    value_blocks, rows = _parse_benchmark_output_rows_and_value_blocks(output, preset_name)
    return bench_args, value_blocks, rows


def _parse_preset_names(raw: str) -> list[str]:
    preset_tokens = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not preset_tokens:
        raise ValueError("--presets must contain at least one preset name")
    resolved_preset_names: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for preset_token in preset_tokens:
        if preset_token in PRESET_ALIASES:
            expanded_preset_names = PRESET_ALIASES[preset_token]
        elif preset_token in ALL_PRESETS:
            expanded_preset_names = (preset_token,)
        else:
            unknown.append(preset_token)
            continue
        for preset_name in expanded_preset_names:
            if preset_name in seen:
                continue
            seen.add(preset_name)
            resolved_preset_names.append(preset_name)
    if unknown:
        raise ValueError(
            f"Unknown preset token(s): {', '.join(unknown)}. "
            f"Available preset names and aliases: {', '.join(ALL_PRESET_TOKENS)}"
        )
    return resolved_preset_names


def _preset_help_text() -> str:
    alias_parts = [f"{alias}=[{', '.join(preset_names)}]" for alias, preset_names in PRESET_ALIASES.items()]
    return (
        "Comma-separated Arc decode presets or aliases to benchmark. "
        f"Preset names: {', '.join(ALL_PRESETS)}. "
        f"Aliases: {'; '.join(alias_parts)}."
    )


def _resolve_compare_ratio_delta(
    requested_compare_ratio_delta: float | None,
    *,
    preset_name: str,
) -> float:
    if requested_compare_ratio_delta is not None:
        return requested_compare_ratio_delta

    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    if expectation.default_compare_ratio_delta is not None:
        return float(expectation.default_compare_ratio_delta)
    return DEFAULT_COMPARE_RATIO_DELTA


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
    value_blocks, rows = _parse_benchmark_output_rows_and_value_blocks(output, preset_name)
    _validate_benchmark_rows(preset_name=preset_name, value_blocks=value_blocks, rows=rows)


def _collect_benchmark_summary(output: str, preset_name: str) -> dict[str, object]:
    value_blocks, rows = _parse_benchmark_output_rows_and_value_blocks(output, preset_name)
    return _collect_benchmark_summary_from_rows(
        preset_name=preset_name,
        value_blocks=value_blocks,
        rows=rows,
    )


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
    value_block = row.get("forced_value_block", row.get("value_block", row.get("default_value_block", "")))
    return (
        row["batch"],
        row["heads"],
        row["key_head_dim"],
        row["value_head_dim"],
        value_block,
    )


def _serialize_benchmark_row_key(key: tuple[str, str, str, str, str]) -> dict[str, str]:
    return {
        "batch": key[0],
        "heads": key[1],
        "key_head_dim": key[2],
        "value_head_dim": key[3],
        "value_block": key[4],
    }


def _shape_case_from_benchmark_row(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row["batch"]), int(row["heads"]), int(row["value_head_dim"]))


def _dedupe_shape_cases(shape_cases: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    deduped: list[tuple[int, int, int]] = []
    for shape_case in shape_cases:
        if shape_case in seen:
            continue
        seen.add(shape_case)
        deduped.append(shape_case)
    return deduped


def _find_benchmark_ratio_failure_rows(
    *,
    preset_name: str,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    expectation = ARC_BENCH_EXPECTATIONS[preset_name]
    return [row for row in rows if float(row[expectation.ratio_field]) > expectation.max_ratio]


def _merge_benchmark_rows(
    *,
    base_rows: list[dict[str, str]],
    override_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    override_map = {_benchmark_row_key(row): row for row in override_rows}
    return [override_map.get(_benchmark_row_key(row), row) for row in base_rows]


def _confirm_benchmark_ratio_failures(
    *,
    preset_name: str,
    rows: list[dict[str, str]],
    seeds_csv: str,
    cwd: Path,
    env: dict[str, str],
) -> tuple[dict[str, object] | None, list[dict[str, str]]]:
    failing_rows = _find_benchmark_ratio_failure_rows(preset_name=preset_name, rows=rows)
    if not failing_rows:
        return None, rows

    shape_cases = _dedupe_shape_cases([_shape_case_from_benchmark_row(row) for row in failing_rows])
    print(
        f"[confirm-threshold:{preset_name}] failing_rows={len(failing_rows)} candidate_shapes={len(shape_cases)} "
        f"warmup={DEFAULT_VALIDATE_CONFIRM_WARMUP} iters={DEFAULT_VALIDATE_CONFIRM_ITERS} "
        f"timing_repeats={DEFAULT_VALIDATE_CONFIRM_TIMING_REPEATS}",
        flush=True,
    )
    confirm_args, confirm_value_blocks, confirm_rows = _run_benchmark_for_shape_cases(
        preset_name=preset_name,
        shape_cases=shape_cases,
        warmup=DEFAULT_VALIDATE_CONFIRM_WARMUP,
        iters=DEFAULT_VALIDATE_CONFIRM_ITERS,
        timing_repeats=DEFAULT_VALIDATE_CONFIRM_TIMING_REPEATS,
        seeds_csv=seeds_csv,
        cwd=cwd,
        env=env,
        label=f"confirm-threshold:{preset_name}",
    )
    merged_rows = _merge_benchmark_rows(base_rows=rows, override_rows=confirm_rows)
    remaining_failures = _find_benchmark_ratio_failure_rows(preset_name=preset_name, rows=merged_rows)
    summary = {
        "triggered": True,
        "bench_args": confirm_args,
        "shape_cases": shape_cases,
        "resolved_value_blocks": list(confirm_value_blocks),
        "candidate_rows": failing_rows,
        "confirmed_rows": confirm_rows,
        "cleared_row_count": len(failing_rows) - len(remaining_failures),
        "remaining_failure_rows": remaining_failures,
    }
    return summary, merged_rows


def _collect_benchmark_comparison_summary(
    *,
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
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
    shared_keys = sorted(current_keys & baseline_keys)
    missing = sorted(baseline_keys - current_keys)
    extra = sorted(current_keys - baseline_keys)
    if not shared_keys:
        raise ValueError(
            f"Preset {current_summary['preset']} has no overlapping row keys; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    max_ratio_delta_observed = float("-inf")
    worst_key: tuple[str, str, str, str, str] | None = None
    worst_current_ratio = 0.0
    worst_baseline_ratio = 0.0
    per_row_deltas: list[dict[str, object]] = []
    for key in shared_keys:
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

    observed_max_ratio_delta = float(current_summary["observed_max_ratio"]) - float(baseline_summary["observed_max_ratio"])
    return {
        "preset": current_summary["preset"],
        "ratio_field": ratio_field,
        "observed_max_ratio_delta": observed_max_ratio_delta,
        "observed_max_ratio_baseline": float(baseline_summary["observed_max_ratio"]),
        "observed_max_ratio_current": float(current_summary["observed_max_ratio"]),
        "baseline_row_count": len(baseline_map),
        "current_row_count": len(current_map),
        "overlap_row_count": len(shared_keys),
        "missing_row_count": len(missing),
        "extra_row_count": len(extra),
        "missing_rows": [_serialize_benchmark_row_key(key) for key in missing],
        "extra_rows": [_serialize_benchmark_row_key(key) for key in extra],
        "max_row_ratio_delta": max_ratio_delta_observed,
        "row_deltas": per_row_deltas,
    }


def _find_benchmark_compare_ratio_failure_rows(
    *,
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
    max_ratio_delta: float,
) -> list[dict[str, object]]:
    comparison_summary = _collect_benchmark_comparison_summary(
        current_summary=current_summary,
        baseline_summary=baseline_summary,
    )
    row_deltas = comparison_summary["row_deltas"]
    assert isinstance(row_deltas, list)
    return [
        row_delta
        for row_delta in row_deltas
        if isinstance(row_delta, dict) and float(row_delta["ratio_delta"]) > max_ratio_delta
    ]


def _confirm_benchmark_compare_ratio_failures(
    *,
    preset_name: str,
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
    max_ratio_delta: float,
    seeds_csv: str,
    cwd: Path,
    env: dict[str, str],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    failing_rows = _find_benchmark_compare_ratio_failure_rows(
        current_summary=current_summary,
        baseline_summary=baseline_summary,
        max_ratio_delta=max_ratio_delta,
    )
    if not failing_rows:
        return None, current_summary

    shape_cases = _dedupe_shape_cases(
        [
            (int(row["batch"]), int(row["heads"]), int(row["value_head_dim"]))
            for row in failing_rows
        ]
    )
    print(
        f"[confirm-compare:{preset_name}] failing_rows={len(failing_rows)} candidate_shapes={len(shape_cases)} "
        f"max_ratio_delta={max_ratio_delta:.4f} warmup={DEFAULT_VALIDATE_CONFIRM_WARMUP} "
        f"iters={DEFAULT_VALIDATE_CONFIRM_ITERS} timing_repeats={DEFAULT_VALIDATE_CONFIRM_TIMING_REPEATS}",
        flush=True,
    )
    confirm_args, confirm_value_blocks, confirm_rows = _run_benchmark_for_shape_cases(
        preset_name=preset_name,
        shape_cases=shape_cases,
        warmup=DEFAULT_VALIDATE_CONFIRM_WARMUP,
        iters=DEFAULT_VALIDATE_CONFIRM_ITERS,
        timing_repeats=DEFAULT_VALIDATE_CONFIRM_TIMING_REPEATS,
        seeds_csv=seeds_csv,
        cwd=cwd,
        env=env,
        label=f"confirm-compare:{preset_name}",
    )
    current_rows = current_summary["rows"]
    if not isinstance(current_rows, list):
        raise ValueError("Benchmark summary rows must be lists")
    merged_rows = _merge_benchmark_rows(
        base_rows=[row for row in current_rows if isinstance(row, dict)],
        override_rows=confirm_rows,
    )
    current_value_blocks = tuple(int(value) for value in current_summary["resolved_value_blocks"])
    if tuple(confirm_value_blocks) != current_value_blocks:
        raise ValueError(
            f"{preset_name} compare-confirmation resolved value blocks changed from "
            f"{current_value_blocks} to {tuple(confirm_value_blocks)}"
        )
    updated_summary = _collect_benchmark_summary_from_rows(
        preset_name=preset_name,
        value_blocks=current_value_blocks,
        rows=merged_rows,
    )
    for passthrough_field in ("bench_args", "bench_runs", "effective_max_shapes_per_scan", "threshold_confirmation"):
        if passthrough_field in current_summary:
            updated_summary[passthrough_field] = current_summary[passthrough_field]
    remaining_failures = _find_benchmark_compare_ratio_failure_rows(
        current_summary=updated_summary,
        baseline_summary=baseline_summary,
        max_ratio_delta=max_ratio_delta,
    )
    summary = {
        "triggered": True,
        "bench_args": confirm_args,
        "shape_cases": shape_cases,
        "resolved_value_blocks": list(confirm_value_blocks),
        "candidate_rows": failing_rows,
        "confirmed_rows": confirm_rows,
        "cleared_row_count": len(failing_rows) - len(remaining_failures),
        "remaining_failure_rows": remaining_failures,
    }
    return summary, updated_summary


def _compare_benchmark_summary_against_baseline(
    *,
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
    max_ratio_delta: float,
) -> dict[str, object]:
    summary = _collect_benchmark_comparison_summary(
        current_summary=current_summary,
        baseline_summary=baseline_summary,
    )
    summary["max_ratio_delta_allowed"] = max_ratio_delta
    if float(summary["max_row_ratio_delta"]) > max_ratio_delta:
        worst_row = max(
            (
                row_delta
                for row_delta in summary["row_deltas"]
                if isinstance(row_delta, dict)
            ),
            key=lambda row_delta: float(row_delta["ratio_delta"]),
        )
        raise ValueError(
            f"Preset {current_summary['preset']} exceeded ratio delta threshold {max_ratio_delta:.4f}: "
            f"baseline={float(worst_row['baseline_ratio']):.4f} current={float(worst_row['current_ratio']):.4f} "
            f"delta={float(worst_row['ratio_delta']):.4f} on batch={worst_row['batch']} heads={worst_row['heads']} "
            f"value_dim={worst_row['value_head_dim']} value_block={worst_row['value_block']}"
        )
    print(
        f"[compare:{current_summary['preset']}] overlap={int(summary['overlap_row_count'])} "
        f"missing={int(summary['missing_row_count'])} extra={int(summary['extra_row_count'])} "
        f"max_row_ratio_delta={float(summary['max_row_ratio_delta']):.4f} "
        f"observed_max_ratio_delta={float(summary['observed_max_ratio_delta']):.4f}",
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
        default=FULL_PRESET_ALIAS,
        help=_preset_help_text(),
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
        "--max-shapes-per-scan",
        type=int,
        default=None,
        help=(
            "Split large benchmark presets into multiple benchmark invocations with at most this many shapes each. "
            f"When omitted, presets larger than {DEFAULT_VALIDATE_BENCH_CHUNK_THRESHOLD} shapes auto-chunk into "
            f"{DEFAULT_VALIDATE_BENCH_MAX_SHAPES_PER_SCAN}. Pass 0 to disable chunking."
        ),
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
        default=None,
        help=(
            "Maximum allowed increase in per-row speed-ratio metrics versus --compare-json. "
            "When omitted, watch presets use tighter preset-aware defaults and other presets use 0.03."
        ),
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
            "max_shapes_per_scan": args.max_shapes_per_scan,
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
            bench_args, bench_runs, value_blocks, rows, effective_max_shapes_per_scan = _run_benchmark_for_preset(
                preset_name=preset_name,
                warmup=args.warmup,
                iters=args.iters,
                timing_repeats=args.timing_repeats,
                seeds_csv=args.seeds,
                cwd=root,
                env=env,
                max_shapes_per_scan=args.max_shapes_per_scan,
            )
            threshold_confirmation: dict[str, object] | None = None
            if not args.skip_bench_gates:
                try:
                    _validate_benchmark_rows(
                        preset_name=preset_name,
                        value_blocks=value_blocks,
                        rows=rows,
                    )
                except ValueError:
                    threshold_confirmation, rows = _confirm_benchmark_ratio_failures(
                        preset_name=preset_name,
                        rows=rows,
                        seeds_csv=args.seeds,
                        cwd=root,
                        env=env,
                    )
                    if threshold_confirmation is None:
                        raise
                    _validate_benchmark_rows(
                        preset_name=preset_name,
                        value_blocks=value_blocks,
                        rows=rows,
                    )
            benchmark_summary = _collect_benchmark_summary_from_rows(
                preset_name=preset_name,
                value_blocks=value_blocks,
                rows=rows,
            )
            benchmark_summary["bench_args"] = bench_args
            benchmark_summary["bench_runs"] = bench_runs
            benchmark_summary["effective_max_shapes_per_scan"] = effective_max_shapes_per_scan
            benchmark_summary["threshold_confirmation"] = threshold_confirmation
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
                compare_ratio_delta = _resolve_compare_ratio_delta(
                    args.compare_ratio_delta,
                    preset_name=preset_name,
                )
                comparison_confirmation: dict[str, object] | None = None
                try:
                    comparison_summary = _compare_benchmark_summary_against_baseline(
                        current_summary=benchmark_summary,
                        baseline_summary=baseline_presets[preset_name],
                        max_ratio_delta=compare_ratio_delta,
                    )
                except ValueError:
                    comparison_confirmation, benchmark_summary = _confirm_benchmark_compare_ratio_failures(
                        preset_name=preset_name,
                        current_summary=benchmark_summary,
                        baseline_summary=baseline_presets[preset_name],
                        max_ratio_delta=compare_ratio_delta,
                        seeds_csv=args.seeds,
                        cwd=root,
                        env=env,
                    )
                    if comparison_confirmation is None:
                        raise
                    cast_presets[preset_name] = benchmark_summary
                    comparison_summary = _compare_benchmark_summary_against_baseline(
                        current_summary=benchmark_summary,
                        baseline_summary=baseline_presets[preset_name],
                        max_ratio_delta=compare_ratio_delta,
                    )
                comparison_summary["delta_confirmation"] = comparison_confirmation
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
