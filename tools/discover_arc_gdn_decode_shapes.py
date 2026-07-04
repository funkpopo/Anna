from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_xpu_hotspots import (  # noqa: E402
    GDN_DECODE_SHAPE_PRESETS,
    _dedupe_gdn_decode_shape_cases,
    _parse_gdn_decode_shape_cases,
    _parse_gdn_decode_shape_presets,
    _resolve_gdn_decode_value_blocks,
)
from tools.validate_arc_gdn_decode import (  # noqa: E402
    DEFAULT_BENCH_TIMING_REPEATS,
    _default_op_lib_path,
    _parse_gdn_decode_csv_rows,
    _parse_gdn_decode_value_blocks,
    _pythonpath_value,
    _repo_root,
    _run_step,
)


@dataclass(frozen=True)
class DiscoveryMode:
    name: str
    compare_flag: str
    compare_prefix: str
    strategy_field: str
    ratio_field: str
    delta_field: str


DISCOVERY_MODES: dict[str, DiscoveryMode] = {
    "auto": DiscoveryMode(
        name="auto",
        compare_flag="--gdn-decode-auto-compare",
        compare_prefix="gdn_decode_auto_compare",
        strategy_field="auto_strategy",
        ratio_field="auto_speed_ratio",
        delta_field="auto_minus_best_ms",
    ),
    "default": DiscoveryMode(
        name="default",
        compare_flag="--gdn-decode-default-compare",
        compare_prefix="gdn_decode_default_compare",
        strategy_field="default_strategy",
        ratio_field="default_speed_ratio",
        delta_field="default_minus_best_ms",
    ),
}


def _parse_int_spans(raw: str) -> list[int]:
    values: list[int] = []
    for token in (item.strip() for item in raw.split(",")):
        if not token:
            continue
        if "-" not in token:
            values.append(int(token))
            continue
        range_part, _, step_part = token.partition(":")
        start_raw, _, stop_raw = range_part.partition("-")
        if not start_raw or not stop_raw:
            raise ValueError(f"Invalid integer span {token!r}; expected N, A-B, or A-B:S")
        start = int(start_raw)
        stop = int(stop_raw)
        step = int(step_part) if step_part else 1
        if step <= 0:
            raise ValueError(f"Invalid integer span {token!r}; step must be positive")
        if stop < start:
            raise ValueError(f"Invalid integer span {token!r}; stop must be >= start")
        values.extend(range(start, stop + 1, step))
    if not values:
        raise ValueError("Expected at least one integer span")
    return values


def _resolve_shape_cases(
    *,
    shape_presets: str | None,
    shape_cases: str | None,
    batches: str | None,
    heads: str | None,
    value_head_dims: str | None,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    cases: list[tuple[int, int, int]] = []
    preset_names: list[str] = []
    if shape_presets is not None:
        preset_names = _parse_gdn_decode_shape_presets(shape_presets)
    if preset_names:
        for preset_name in preset_names:
            cases.extend(GDN_DECODE_SHAPE_PRESETS[preset_name])
    if shape_cases is not None:
        cases.extend(_parse_gdn_decode_shape_cases(shape_cases))
    if any(value is not None for value in (batches, heads, value_head_dims)):
        if not all(value is not None for value in (batches, heads, value_head_dims)):
            raise ValueError("--batches, --heads, and --value-head-dims must be provided together")
        batch_values = _parse_int_spans(str(batches))
        head_values = _parse_int_spans(str(heads))
        value_dim_values = _parse_int_spans(str(value_head_dims))
        for batch_size in batch_values:
            for num_heads in head_values:
                for value_head_dim in value_dim_values:
                    cases.append((batch_size, num_heads, value_head_dim))
    if not cases:
        raise ValueError(
            "Provide at least one of --shape-presets, --shape-cases, or the --batches/--heads/--value-head-dims grid"
        )
    return _dedupe_gdn_decode_shape_cases(cases), preset_names


def _shape_cases_csv(cases: list[tuple[int, int, int]]) -> str:
    return ",".join(f"{batch_size}x{num_heads}x{value_head_dim}" for batch_size, num_heads, value_head_dim in cases)


def _format_sampled_int_ranges(values: list[int]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return str(values[0])

    ranges: list[str] = []
    start = values[0]
    prev = values[0]
    step: int | None = None
    for value in values[1:]:
        diff = value - prev
        if step is None:
            step = diff
            prev = value
            continue
        if diff == step:
            prev = value
            continue
        ranges.append(_format_single_range(start, prev, step))
        start = value
        prev = value
        step = None
    ranges.append(_format_single_range(start, prev, step))
    return ",".join(ranges)


def _format_single_range(start: int, stop: int, step: int | None) -> str:
    if start == stop:
        return str(start)
    effective_step = 1 if step is None else step
    if effective_step == 1:
        return f"{start}..{stop}"
    return f"{start}..{stop}/{effective_step}"


def _annotate_rows(rows: list[dict[str, str]], mode: DiscoveryMode) -> list[dict[str, object]]:
    annotated_rows: list[dict[str, object]] = []
    for row in rows:
        batch_size = int(row["batch"])
        num_heads = int(row["heads"])
        value_block = int(row.get("value_block", row.get("default_value_block", "0")))
        observed_strategy = row[mode.strategy_field]
        best_explicit_strategy = row["best_explicit_strategy"]
        ratio = float(row[mode.ratio_field])
        delta_ms = float(row[mode.delta_field])
        annotated_rows.append(
            {
                **row,
                "batch_size": batch_size,
                "num_heads": num_heads,
                "key_head_dim_int": int(row["key_head_dim"]),
                "value_head_dim_int": int(row["value_head_dim"]),
                "value_block_int": value_block,
                "recurrent_rows": batch_size * num_heads,
                "observed_strategy": observed_strategy,
                "best_explicit_strategy_str": best_explicit_strategy,
                "ratio": ratio,
                "delta_ms": delta_ms,
                "strategy_mismatch": observed_strategy != best_explicit_strategy,
            }
        )
    return annotated_rows


def _collect_suspicious_rows(
    rows: list[dict[str, object]],
    *,
    ratio_threshold: float,
    require_strategy_mismatch: bool,
) -> list[dict[str, object]]:
    suspicious_rows: list[dict[str, object]] = []
    for row in rows:
        strategy_mismatch = bool(row["strategy_mismatch"])
        ratio_exceeded = float(row["ratio"]) >= ratio_threshold
        if require_strategy_mismatch:
            keep = strategy_mismatch
        else:
            keep = strategy_mismatch or ratio_exceeded
        if not keep:
            continue
        reasons: list[str] = []
        if strategy_mismatch:
            reasons.append("strategy_mismatch")
        if ratio_exceeded:
            reasons.append("ratio_threshold")
        suspicious_rows.append({**row, "reasons": reasons})
    suspicious_rows.sort(
        key=lambda row: (
            -float(row["ratio"]),
            -abs(float(row["delta_ms"])),
            int(row["recurrent_rows"]),
            int(row["value_block_int"]),
        )
    )
    return suspicious_rows


def _discovery_row_key(row: dict[str, object]) -> tuple[int, int, int, int, int]:
    return (
        int(row["batch_size"]),
        int(row["num_heads"]),
        int(row["key_head_dim_int"]),
        int(row["value_head_dim_int"]),
        int(row["value_block_int"]),
    )


def _select_confirmation_shape_cases(rows: list[dict[str, object]]) -> list[tuple[int, int, int]]:
    return _dedupe_gdn_decode_shape_cases(
        [
            (
                int(row["batch_size"]),
                int(row["num_heads"]),
                int(row["value_head_dim_int"]),
            )
            for row in rows
        ]
    )


def _select_confirmation_value_blocks(
    rows: list[dict[str, object]],
    fallback_value_blocks: list[int],
) -> list[int]:
    unique_value_blocks = sorted({int(row["value_block_int"]) for row in rows})
    return unique_value_blocks if unique_value_blocks else list(fallback_value_blocks)


def _select_confirmation_rows(
    suspicious_rows: list[dict[str, object]],
    confirmed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    confirmed_row_map = {_discovery_row_key(row): row for row in confirmed_rows}
    selected_rows: list[dict[str, object]] = []
    for row in suspicious_rows:
        confirmed_row = confirmed_row_map.get(_discovery_row_key(row))
        if confirmed_row is not None:
            selected_rows.append(confirmed_row)
    return selected_rows


def _order_ratio(first_ms: float, second_ms: float) -> float:
    smaller = min(first_ms, second_ms)
    larger = max(first_ms, second_ms)
    if smaller <= 0.0:
        return float("inf") if larger > 0.0 else 1.0
    return larger / smaller


def _build_order_sensitivity_rows(
    forward_rows: list[dict[str, object]],
    reverse_rows: list[dict[str, object]],
    *,
    mode: DiscoveryMode,
) -> list[dict[str, object]]:
    reverse_row_map = {_discovery_row_key(row): row for row in reverse_rows}
    observed_ms_field = f"{mode.name}_ms"
    compared_rows: list[dict[str, object]] = []
    for forward_row in forward_rows:
        reverse_row = reverse_row_map.get(_discovery_row_key(forward_row))
        if reverse_row is None:
            continue
        single_order_ratio = _order_ratio(float(forward_row["single_ms"]), float(reverse_row["single_ms"]))
        tiled_order_ratio = _order_ratio(float(forward_row["tiled_ms"]), float(reverse_row["tiled_ms"]))
        observed_order_ratio = _order_ratio(
            float(forward_row[observed_ms_field]),
            float(reverse_row[observed_ms_field]),
        )
        best_explicit_order_ratio = _order_ratio(
            float(forward_row["best_explicit_ms"]),
            float(reverse_row["best_explicit_ms"]),
        )
        max_order_ratio = max(
            single_order_ratio,
            tiled_order_ratio,
            observed_order_ratio,
            best_explicit_order_ratio,
        )
        compared_rows.append(
            {
                **forward_row,
                "reverse_observed_strategy": reverse_row["observed_strategy"],
                "reverse_best_explicit_strategy": reverse_row["best_explicit_strategy_str"],
                "forward_observed_ms": float(forward_row[observed_ms_field]),
                "reverse_observed_ms": float(reverse_row[observed_ms_field]),
                "forward_best_explicit_ms": float(forward_row["best_explicit_ms"]),
                "reverse_best_explicit_ms": float(reverse_row["best_explicit_ms"]),
                "single_order_ratio": single_order_ratio,
                "tiled_order_ratio": tiled_order_ratio,
                "observed_order_ratio": observed_order_ratio,
                "best_explicit_order_ratio": best_explicit_order_ratio,
                "max_order_ratio": max_order_ratio,
                "observed_strategy_changed": forward_row["observed_strategy"] != reverse_row["observed_strategy"],
                "best_explicit_strategy_changed": (
                    forward_row["best_explicit_strategy_str"] != reverse_row["best_explicit_strategy_str"]
                ),
            }
        )
    compared_rows.sort(
        key=lambda row: (
            -float(row["max_order_ratio"]),
            -float(row["observed_order_ratio"]),
            int(row["recurrent_rows"]),
        )
    )
    return compared_rows


def _collect_order_sensitive_rows(
    rows: list[dict[str, object]],
    *,
    order_ratio_threshold: float,
) -> list[dict[str, object]]:
    sensitive_rows: list[dict[str, object]] = []
    for row in rows:
        strategy_changed = bool(row["observed_strategy_changed"]) or bool(row["best_explicit_strategy_changed"])
        ratio_exceeded = float(row["max_order_ratio"]) >= order_ratio_threshold
        if not strategy_changed and not ratio_exceeded:
            continue
        reasons: list[str] = []
        if strategy_changed:
            reasons.append("strategy_changed")
        if ratio_exceeded:
            reasons.append("order_ratio_threshold")
        sensitive_rows.append({**row, "order_reasons": reasons})
    sensitive_rows.sort(
        key=lambda row: (
            -float(row["max_order_ratio"]),
            -float(row["observed_order_ratio"]),
            int(row["recurrent_rows"]),
        )
    )
    return sensitive_rows


def _build_group_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["value_head_dim_int"]),
            int(row["value_block_int"]),
            str(row["observed_strategy"]),
            str(row["best_explicit_strategy_str"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        recurrent_rows = sorted({int(row["recurrent_rows"]) for row in group_rows})
        worst_row = max(group_rows, key=lambda row: float(row["ratio"]))
        summaries.append(
            {
                "value_head_dim": key[0],
                "value_block": key[1],
                "observed_strategy": key[2],
                "best_explicit_strategy": key[3],
                "row_count": len(group_rows),
                "sampled_recurrent_rows": recurrent_rows,
                "sampled_recurrent_row_ranges": _format_sampled_int_ranges(recurrent_rows),
                "worst_ratio": float(worst_row["ratio"]),
                "worst_delta_ms": float(worst_row["delta_ms"]),
            }
        )
    summaries.sort(key=lambda summary: (-float(summary["worst_ratio"]), int(summary["value_head_dim"])))
    return summaries


def _print_row(prefix: str, row: dict[str, object]) -> None:
    print(
        f"{prefix},batch={row['batch_size']},heads={row['num_heads']},rows={row['recurrent_rows']},"
        f"key_head_dim={row['key_head_dim_int']},value_head_dim={row['value_head_dim_int']},"
        f"value_block={row['value_block_int']},observed={row['observed_strategy']},"
        f"best={row['best_explicit_strategy_str']},ratio={float(row['ratio']):.4f},"
        f"delta_ms={float(row['delta_ms']):.4f},reasons={'+'.join(str(reason) for reason in row['reasons'])}"
    )


def _print_group_summary(summary: dict[str, object]) -> None:
    print(
        "gdn_decode_suspicious_group,"
        f"value_head_dim={summary['value_head_dim']},value_block={summary['value_block']},"
        f"observed={summary['observed_strategy']},best={summary['best_explicit_strategy']},"
        f"row_count={summary['row_count']},sampled_rows={summary['sampled_recurrent_row_ranges']},"
        f"worst_ratio={float(summary['worst_ratio']):.4f},worst_delta_ms={float(summary['worst_delta_ms']):.4f}"
    )


def _print_order_sensitive_row(prefix: str, row: dict[str, object]) -> None:
    print(
        f"{prefix},batch={row['batch_size']},heads={row['num_heads']},rows={row['recurrent_rows']},"
        f"value_head_dim={row['value_head_dim_int']},value_block={row['value_block_int']},"
        f"observed={row['observed_strategy']}->{row['reverse_observed_strategy']},"
        f"best={row['best_explicit_strategy_str']}->{row['reverse_best_explicit_strategy']},"
        f"forward_observed_ms={float(row['forward_observed_ms']):.4f},"
        f"reverse_observed_ms={float(row['reverse_observed_ms']):.4f},"
        f"observed_order_ratio={float(row['observed_order_ratio']):.4f},"
        f"max_order_ratio={float(row['max_order_ratio']):.4f},"
        f"reasons={'+'.join(str(reason) for reason in row['order_reasons'])}"
    )


def _build_discovery_bench_args(
    *,
    mode: DiscoveryMode,
    shape_cases: list[tuple[int, int, int]],
    value_blocks: list[int],
    head_dim: int,
    seeds: str,
    dtype: str,
    warmup: int,
    iters: int,
    timing_repeats: int,
) -> list[str]:
    return [
        sys.executable,
        "tools/bench_xpu_hotspots.py",
        "--gdn-decode-only",
        mode.compare_flag,
        "--gdn-decode-compare-only",
        "--head-dim",
        str(head_dim),
        "--gdn-decode-shape-cases",
        _shape_cases_csv(shape_cases),
        "--gdn-decode-value-blocks",
        ",".join(str(value_block) for value_block in value_blocks),
        "--gdn-decode-seeds",
        seeds,
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--gdn-decode-timing-repeats",
        str(timing_repeats),
    ]


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


def _primary_bench_args(bench_runs: list[list[str]]) -> list[str] | None:
    return bench_runs[0] if len(bench_runs) == 1 else None


def _run_discovery_scan(
    *,
    root: Path,
    env: dict[str, str],
    mode: DiscoveryMode,
    shape_cases: list[tuple[int, int, int]],
    value_blocks: list[int],
    head_dim: int,
    seeds: str,
    dtype: str,
    warmup: int,
    iters: int,
    timing_repeats: int,
    max_shapes_per_scan: int | None,
    label: str,
) -> tuple[list[list[str]], list[dict[str, object]], tuple[int, ...]]:
    bench_runs: list[list[str]] = []
    all_rows: list[dict[str, object]] = []
    resolved_value_blocks: tuple[int, ...] | None = None
    case_chunks = _chunk_shape_cases(shape_cases, max_shapes_per_scan=max_shapes_per_scan)
    for chunk_index, case_chunk in enumerate(case_chunks, start=1):
        bench_args = _build_discovery_bench_args(
            mode=mode,
            shape_cases=case_chunk,
            value_blocks=value_blocks,
            head_dim=head_dim,
            seeds=seeds,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            timing_repeats=timing_repeats,
        )
        bench_runs.append(bench_args)
        chunk_label = label if len(case_chunks) == 1 else f"{label}:{chunk_index}/{len(case_chunks)}"
        output = _run_step(
            args=bench_args,
            cwd=root,
            env=env,
            label=chunk_label,
            capture_output=True,
        )
        assert output is not None
        chunk_rows = _annotate_rows(_parse_gdn_decode_csv_rows(output, mode.compare_prefix), mode)
        chunk_value_blocks = _parse_gdn_decode_value_blocks(output)
        if resolved_value_blocks is None:
            resolved_value_blocks = chunk_value_blocks
        elif chunk_value_blocks != resolved_value_blocks:
            raise ValueError(
                f"Chunked discovery scan resolved inconsistent value blocks: {resolved_value_blocks} vs {chunk_value_blocks}"
            )
        all_rows.extend(chunk_rows)
    assert resolved_value_blocks is not None
    return bench_runs, all_rows, resolved_value_blocks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gated Delta decode compare sweeps and surface suspicious Arc shape buckets."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(DISCOVERY_MODES),
        default="auto",
        help="Compare mode: auto checks resolver-selected strategy for an explicit value_block; default checks built-in defaults.",
    )
    parser.add_argument(
        "--shape-presets",
        type=str,
        default=None,
        help="Comma-separated Gated Delta decode shape presets from tools/bench_xpu_hotspots.py.",
    )
    parser.add_argument(
        "--shape-cases",
        type=str,
        default=None,
        help="Comma-separated explicit BxHxV cases, for example 1x16x64,4x32x256.",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help="Batch span list for grid generation, for example 1-8,16,24-32:2.",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Head-count span list for grid generation, for example 8,16,32.",
    )
    parser.add_argument(
        "--value-head-dims",
        type=str,
        default=None,
        help="Value-head-dim span list for grid generation, for example 64,128,256.",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Key head dim forwarded to the benchmark.")
    parser.add_argument(
        "--value-blocks",
        type=str,
        default=None,
        help=(
            "Comma-separated value blocks forwarded to the benchmark. "
            "Defaults to preset recommendations when presets are provided, otherwise the full bench sweep."
        ),
    )
    parser.add_argument("--seeds", type=str, default="20260960,20260961", help="Fixed benchmark seeds.")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark measured iterations.")
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_BENCH_TIMING_REPEATS,
        help="Repeated timing samples per candidate.",
    )
    parser.add_argument(
        "--max-shapes-per-scan",
        type=int,
        default=None,
        help=(
            "Optionally split large shape lists into multiple benchmark invocations with at most this many "
            "shape cases each. Useful when long sweeps show driver or thermal drift."
        ),
    )
    parser.add_argument("--dtype", type=str, default="bf16", help="Input dtype forwarded to the benchmark.")
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.03,
        help="Report rows whose speed ratio is at or above this threshold, even if the observed strategy matches best_explicit.",
    )
    parser.add_argument(
        "--require-strategy-mismatch",
        action="store_true",
        help="Only report rows where the observed strategy differs from best_explicit_strategy.",
    )
    parser.add_argument(
        "--confirm-suspicious",
        action="store_true",
        help="Re-run only suspicious rows with a higher-sampling confirmation pass before deciding they need a bucket change.",
    )
    parser.add_argument(
        "--compare-reverse-order",
        action="store_true",
        help="Re-run the same shape cases in reverse order and report rows whose timings are sensitive to sweep order.",
    )
    parser.add_argument(
        "--order-ratio-threshold",
        type=float,
        default=1.05,
        help="Report rows whose forward/reverse max timing ratio reaches this threshold during reverse-order comparison.",
    )
    parser.add_argument(
        "--confirm-warmup",
        type=int,
        default=None,
        help="Warmup iterations for the confirmation pass. Defaults to max(--warmup, 40).",
    )
    parser.add_argument(
        "--confirm-iters",
        type=int,
        default=None,
        help="Measured iterations for the confirmation pass. Defaults to max(--iters, 300).",
    )
    parser.add_argument(
        "--confirm-timing-repeats",
        type=int,
        default=None,
        help="Timing repeats for the confirmation pass. Defaults to max(--timing-repeats, 9).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Always print the top-N worst rows by speed ratio for inspection.",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Rebuild the fused op before scanning by running tools/build_gated_delta_fused_op.py.",
    )
    parser.add_argument(
        "--op-lib",
        type=str,
        default=None,
        help="Explicit fused-op library path. Defaults to the local .build output.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write the structured suspicious-shape report as JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    mode = DISCOVERY_MODES[args.mode]
    cases, preset_names = _resolve_shape_cases(
        shape_presets=args.shape_presets,
        shape_cases=args.shape_cases,
        batches=args.batches,
        heads=args.heads,
        value_head_dims=args.value_head_dims,
    )
    resolved_value_blocks = _resolve_gdn_decode_value_blocks(args.value_blocks, preset_names)

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

    bench_runs, rows, resolved_blocks_from_output = _run_discovery_scan(
        root=root,
        env=env,
        mode=mode,
        shape_cases=cases,
        value_blocks=resolved_value_blocks,
        head_dim=args.head_dim,
        seeds=args.seeds,
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        timing_repeats=args.timing_repeats,
        max_shapes_per_scan=args.max_shapes_per_scan,
        label=f"discover:{mode.name}",
    )
    suspicious_rows = _collect_suspicious_rows(
        rows,
        ratio_threshold=args.ratio_threshold,
        require_strategy_mismatch=args.require_strategy_mismatch,
    )
    suspicious_groups = _build_group_summaries(suspicious_rows)
    top_rows = sorted(
        rows,
        key=lambda row: (-float(row["ratio"]), -abs(float(row["delta_ms"])), int(row["recurrent_rows"])),
    )[: max(0, args.top)]

    print(
        f"[discover] mode={mode.name} shapes={len(cases)} compare_rows={len(rows)} "
        f"value_blocks={resolved_blocks_from_output} suspicious_rows={len(suspicious_rows)} "
        f"ratio_threshold={args.ratio_threshold:.4f}",
        flush=True,
    )
    for row in suspicious_rows:
        _print_row("gdn_decode_suspicious", row)
    for summary in suspicious_groups:
        _print_group_summary(summary)
    if top_rows:
        print(f"[top] showing {len(top_rows)} worst rows by {mode.ratio_field}", flush=True)
        for row in top_rows:
            print(
                "gdn_decode_top_row,"
                f"batch={row['batch_size']},heads={row['num_heads']},rows={row['recurrent_rows']},"
                f"value_head_dim={row['value_head_dim_int']},value_block={row['value_block_int']},"
                f"observed={row['observed_strategy']},best={row['best_explicit_strategy_str']},"
                f"ratio={float(row['ratio']):.4f},delta_ms={float(row['delta_ms']):.4f}"
            )

    reverse_bench_runs: list[list[str]] = []
    reverse_rows: list[dict[str, object]] = []
    order_sensitive_rows: list[dict[str, object]] = []
    if args.compare_reverse_order:
        reverse_cases = list(reversed(cases))
        reverse_bench_runs, reverse_rows, _ = _run_discovery_scan(
            root=root,
            env=env,
            mode=mode,
            shape_cases=reverse_cases,
            value_blocks=resolved_value_blocks,
            head_dim=args.head_dim,
            seeds=args.seeds,
            dtype=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
            timing_repeats=args.timing_repeats,
            max_shapes_per_scan=args.max_shapes_per_scan,
            label=f"discover-reverse:{mode.name}",
        )
        order_rows = _build_order_sensitivity_rows(rows, reverse_rows, mode=mode)
        order_sensitive_rows = _collect_order_sensitive_rows(
            order_rows,
            order_ratio_threshold=args.order_ratio_threshold,
        )
        print(
            f"[reverse-order] mode={mode.name} shapes={len(reverse_rows)} "
            f"order_sensitive_rows={len(order_sensitive_rows)} "
            f"order_ratio_threshold={args.order_ratio_threshold:.4f}",
            flush=True,
        )
        for row in order_sensitive_rows[: max(0, args.top)]:
            _print_order_sensitive_row("gdn_decode_order_sensitive", row)

    confirm_bench_runs: list[list[str]] = []
    confirm_cases: list[tuple[int, int, int]] = []
    confirm_value_blocks: list[int] = []
    confirm_rows: list[dict[str, object]] = []
    confirmed_suspicious_rows: list[dict[str, object]] = []
    confirmed_groups: list[dict[str, object]] = []
    if args.confirm_suspicious and suspicious_rows:
        confirm_cases = _select_confirmation_shape_cases(suspicious_rows)
        confirm_value_blocks = _select_confirmation_value_blocks(suspicious_rows, resolved_value_blocks)
        confirm_warmup = args.confirm_warmup if args.confirm_warmup is not None else max(args.warmup, 40)
        confirm_iters = args.confirm_iters if args.confirm_iters is not None else max(args.iters, 300)
        confirm_timing_repeats = (
            args.confirm_timing_repeats if args.confirm_timing_repeats is not None else max(args.timing_repeats, 9)
        )
        confirm_bench_runs, raw_confirm_rows, _ = _run_discovery_scan(
            root=root,
            env=env,
            mode=mode,
            shape_cases=confirm_cases,
            value_blocks=confirm_value_blocks,
            head_dim=args.head_dim,
            seeds=args.seeds,
            dtype=args.dtype,
            warmup=confirm_warmup,
            iters=confirm_iters,
            timing_repeats=confirm_timing_repeats,
            max_shapes_per_scan=args.max_shapes_per_scan,
            label=f"confirm:{mode.name}",
        )
        confirm_rows = _select_confirmation_rows(suspicious_rows, raw_confirm_rows)
        confirmed_suspicious_rows = _collect_suspicious_rows(
            confirm_rows,
            ratio_threshold=args.ratio_threshold,
            require_strategy_mismatch=args.require_strategy_mismatch,
        )
        confirmed_groups = _build_group_summaries(confirmed_suspicious_rows)
        print(
            f"[confirm] mode={mode.name} candidate_shapes={len(confirm_cases)} compare_rows={len(confirm_rows)} "
            f"confirmed_suspicious_rows={len(confirmed_suspicious_rows)} "
            f"cleared_candidates={len(suspicious_rows) - len(confirmed_suspicious_rows)} "
            f"warmup={confirm_warmup} iters={confirm_iters} timing_repeats={confirm_timing_repeats}",
            flush=True,
        )
        for row in confirmed_suspicious_rows:
            _print_row("gdn_decode_confirmed_suspicious", row)
        for summary in confirmed_groups:
            print(
                "gdn_decode_confirmed_suspicious_group,"
                f"value_head_dim={summary['value_head_dim']},value_block={summary['value_block']},"
                f"observed={summary['observed_strategy']},best={summary['best_explicit_strategy']},"
                f"row_count={summary['row_count']},sampled_rows={summary['sampled_recurrent_row_ranges']},"
                f"worst_ratio={float(summary['worst_ratio']):.4f},worst_delta_ms={float(summary['worst_delta_ms']):.4f}"
            )

    if args.json_output is not None:
        report = {
            "schema_version": 1,
            "mode": mode.name,
            "bench_args": _primary_bench_args(bench_runs),
            "bench_runs": bench_runs,
            "shape_count": len(cases),
            "shape_cases": cases,
            "resolved_value_blocks": list(resolved_blocks_from_output),
            "ratio_threshold": args.ratio_threshold,
            "require_strategy_mismatch": args.require_strategy_mismatch,
            "suspicious_rows": suspicious_rows,
            "suspicious_groups": suspicious_groups,
            "top_rows": top_rows,
            "reverse_order_comparison": {
                "enabled": args.compare_reverse_order,
                "bench_args": _primary_bench_args(reverse_bench_runs),
                "bench_runs": reverse_bench_runs,
                "rows": order_sensitive_rows,
            },
            "confirmation": {
                "enabled": args.confirm_suspicious,
                "bench_args": _primary_bench_args(confirm_bench_runs),
                "bench_runs": confirm_bench_runs,
                "shape_cases": confirm_cases,
                "value_blocks": confirm_value_blocks,
                "candidate_rows": confirm_rows,
                "confirmed_suspicious_rows": confirmed_suspicious_rows,
                "confirmed_suspicious_groups": confirmed_groups,
            },
        }
        output_path = Path(args.json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[json] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
