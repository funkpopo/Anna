from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.ops import apply_rotary_pos_emb, grouped_query_attention, repeat_kv, torch_recurrent_gated_delta_rule
from anna.model.quantization import XPUInt4Linear
from anna.runtime.device import inspect_xpu_device


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


def _decode_gate_query_layout(gate: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    batch_size, seq_len, flat_dim = gate.shape
    if flat_dim != num_heads * head_dim:
        raise ValueError(f"gate last dim {flat_dim} must equal num_heads * head_dim ({num_heads * head_dim})")
    return gate.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _pack_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_kv_heads, key_len, head_dim = key.shape
    pages_per_batch = (key_len + block_size - 1) // block_size
    total_pages = batch_size * pages_per_batch
    key_pages = key.new_zeros((total_pages, num_kv_heads, block_size, head_dim))
    value_pages = value.new_zeros((total_pages, num_kv_heads, block_size, head_dim))
    page_table = torch.full((batch_size, pages_per_batch), -1, device=key.device, dtype=torch.int32)

    for batch_idx in range(batch_size):
        for block_idx in range(pages_per_batch):
            page_id = batch_idx * pages_per_batch + block_idx
            start = block_idx * block_size
            take = min(block_size, key_len - start)
            if take <= 0:
                continue
            key_pages[page_id, :, :take, :].copy_(key[batch_idx, :, start : start + take, :])
            value_pages[page_id, :, :take, :].copy_(value[batch_idx, :, start : start + take, :])
            page_table[batch_idx, block_idx] = page_id

    return key_pages, value_pages, page_table


def _time_op(fn, *, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.xpu.synchronize()
        started_at = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
    return (time.perf_counter() - started_at) * 1000.0 / max(1, iters)


def _benchmark_flashqla_gdn_prefill(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float, float]:
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device="xpu", dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, device="xpu", dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, device="xpu", dtype=dtype)
    g = torch.randn(batch_size, seq_len, num_heads, device="xpu", dtype=torch.float32) * -0.1
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, device="xpu", dtype=torch.float32))
    initial_state = torch.randn(batch_size, num_heads, head_dim, head_dim, device="xpu", dtype=torch.float32)

    current = lambda: torch.ops.anna.gated_delta_prefill(query, key, value, g, beta, initial_state)
    flashqla = lambda: torch.ops.anna.flashqla_gated_delta_prefill(query, key, value, g, beta, initial_state)
    reference = lambda: torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )

    current_ms = _time_op(current, warmup=warmup, iters=iters)
    flashqla_output, flashqla_state = flashqla()
    current_output, current_state = current()
    reference_output, reference_state = reference()
    flashqla_ms = _time_op(flashqla, warmup=warmup, iters=iters)
    reference_ms = _time_op(reference, warmup=max(1, warmup // 10), iters=max(1, iters // 10))

    output_diff = float((flashqla_output.float() - reference_output.float()).abs().max().item())
    current_diff = float((current_output.float() - reference_output.float()).abs().max().item())
    state_diff = float("inf") if reference_state is None else float((flashqla_state.float() - reference_state.float()).abs().max().item())
    return current_ms, flashqla_ms, reference_ms, max(output_diff, current_diff, state_diff)


def _benchmark_gated_delta_decode_strategy(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    strategy: str,
    value_block: int | None,
    single_min_elements: int | None,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    anna_ops = getattr(torch.ops, "anna", None)
    if anna_ops is not None and hasattr(anna_ops, "gated_delta_decode_benchmark"):
        strategy_codes = {"auto": -1, "single": 0, "tiled": 1}
        if strategy not in strategy_codes:
            raise ValueError(f"Unsupported Gated Delta decode strategy benchmark mode: {strategy}")

        benchmark_strategy = strategy
        benchmark_value_block = value_block
        if benchmark_value_block is None:
            benchmark_value_block = _resolve_gated_delta_decode_default_value_block(
                query,
                value_head_dim=int(value.shape[-1]),
            )
        if strategy == "auto" and single_min_elements is None and benchmark_value_block is not None:
            resolved_auto_strategy = _resolve_gated_delta_decode_auto_strategy(
                query,
                value_head_dim=int(value.shape[-1]),
                value_block=benchmark_value_block,
            )
            if resolved_auto_strategy in {"single", "tiled"}:
                benchmark_strategy = resolved_auto_strategy

        state_for_correctness = initial_state.clone()
        candidate_output = anna_ops.gated_delta_decode_benchmark(
            query,
            key,
            value,
            g,
            beta,
            state_for_correctness,
            strategy_codes[benchmark_strategy],
            0 if benchmark_value_block is None else benchmark_value_block,
            0 if single_min_elements is None else single_min_elements,
        )
        reference_output, reference_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
        )
        torch.xpu.synchronize()
        max_abs_diff = float((candidate_output.float() - reference_output.float()).abs().max().item())
        if reference_state is not None:
            max_abs_diff = max(max_abs_diff, float((state_for_correctness.float() - reference_state.float()).abs().max().item()))

        state_for_timing = initial_state.clone()

        def candidate() -> torch.Tensor:
            return anna_ops.gated_delta_decode_benchmark(
                query,
                key,
                value,
                g,
                beta,
                state_for_timing,
                strategy_codes[benchmark_strategy],
                0 if benchmark_value_block is None else benchmark_value_block,
                0 if single_min_elements is None else single_min_elements,
            )

        candidate_ms = _time_op(candidate, warmup=warmup, iters=iters)
        return candidate_ms, max_abs_diff

    previous_strategy = os.environ.get("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY")
    previous_value_block = os.environ.get("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK")
    previous_single_min_elements = os.environ.get("ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS")
    os.environ["ANNA_XPU_GATED_DELTA_DECODE_STRATEGY"] = strategy
    if value_block is None:
        os.environ.pop("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", None)
    else:
        os.environ["ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK"] = str(value_block)
    if single_min_elements is None:
        os.environ.pop("ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS", None)
    else:
        os.environ["ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS"] = str(single_min_elements)
    try:
        state_for_correctness = initial_state.clone()
        candidate_output = torch.ops.anna.gated_delta_decode(
            query,
            key,
            value,
            g,
            beta,
            state_for_correctness,
        )
        reference_output, reference_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
        )
        torch.xpu.synchronize()
        max_abs_diff = float((candidate_output.float() - reference_output.float()).abs().max().item())
        if reference_state is not None:
            max_abs_diff = max(max_abs_diff, float((state_for_correctness.float() - reference_state.float()).abs().max().item()))

        state_for_timing = initial_state.clone()

        def candidate() -> torch.Tensor:
            return torch.ops.anna.gated_delta_decode(
                query,
                key,
                value,
                g,
                beta,
                state_for_timing,
            )

        candidate_ms = _time_op(candidate, warmup=warmup, iters=iters)
        return candidate_ms, max_abs_diff
    finally:
        _restore_env("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", previous_strategy)
        _restore_env("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", previous_value_block)
        _restore_env("ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS", previous_single_min_elements)


def _gated_delta_decode_candidate_id(strategy: str, value_block: int | None) -> str:
    return f"{strategy}@{'default' if value_block is None else value_block}"


def _collect_gated_delta_decode_candidate_timings(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    candidates: list[tuple[str, str, int | None]],
    single_min_elements: int | None,
    warmup: int,
    iters: int,
    timing_repeats: int,
) -> dict[str, dict[str, float]]:
    timing_samples_ms: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    max_abs_diffs: dict[str, float] = {name: 0.0 for name, _, _ in candidates}
    total_candidates = len(candidates)
    if total_candidates == 0:
        return {}

    for repeat_idx in range(max(1, timing_repeats)):
        order_start = repeat_idx % total_candidates
        ordered_candidates = candidates[order_start:] + candidates[:order_start]
        for candidate_name, candidate_strategy, candidate_value_block in ordered_candidates:
            candidate_ms, diff = _benchmark_gated_delta_decode_strategy(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                strategy=candidate_strategy,
                value_block=candidate_value_block,
                single_min_elements=single_min_elements,
                warmup=warmup,
                iters=iters,
            )
            timing_samples_ms[candidate_name].append(candidate_ms)
            max_abs_diffs[candidate_name] = max(max_abs_diffs[candidate_name], diff)

    return {
        candidate_name: {
            "candidate_ms": statistics.median(samples_ms),
            "max_abs_diff": max_abs_diffs[candidate_name],
        }
        for candidate_name, samples_ms in timing_samples_ms.items()
    }


def _make_gated_delta_decode_bench_inputs(
    *,
    batch_size: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device="xpu", dtype=dtype)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device="xpu", dtype=dtype)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device="xpu", dtype=dtype)
    g = -0.1 * torch.rand(batch_size, 1, num_heads, device="xpu", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device="xpu", dtype=torch.float32))
    initial_state = torch.randn(batch_size, num_heads, key_head_dim, value_head_dim, device="xpu", dtype=torch.float32)
    return query, key, value, g, beta, initial_state


def _parse_int_csv(arg_name: str, raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{arg_name} must contain at least one integer")
    return values


def _parse_gdn_decode_batch_head_cases(raw: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(
                "--gdn-decode-batch-head-cases entries must use the form BxH, for example 1x16 or 4x32"
            )
        batch_size_raw, num_heads_raw = token.split("x", maxsplit=1)
        batch_size = int(batch_size_raw)
        num_heads = int(num_heads_raw)
        if batch_size <= 0 or num_heads <= 0:
            raise ValueError("--gdn-decode-batch-head-cases entries must have positive batch and head counts")
        cases.append((batch_size, num_heads))
    if not cases:
        raise ValueError("--gdn-decode-batch-head-cases must contain at least one BxH case")
    return cases


def _parse_gdn_decode_shape_cases(raw: str) -> list[tuple[int, int, int]]:
    cases: list[tuple[int, int, int]] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 3:
            raise ValueError(
                "--gdn-decode-shape-cases entries must use the form BxHxV, for example 1x16x64 or 4x32x128"
            )
        batch_size, num_heads, value_head_dim = (int(part) for part in parts)
        if batch_size <= 0 or num_heads <= 0 or value_head_dim <= 0:
            raise ValueError("--gdn-decode-shape-cases entries must have positive batch, head, and value-dim counts")
        cases.append((batch_size, num_heads, value_head_dim))
    if not cases:
        raise ValueError("--gdn-decode-shape-cases must contain at least one BxHxV case")
    return cases


GDN_DECODE_SHAPE_PRESETS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "arc-default": (
        (1, 8, 128),
        (4, 8, 128),
        (1, 16, 64),
        (4, 16, 64),
        (1, 16, 128),
        (4, 16, 128),
        (1, 16, 256),
        (4, 16, 256),
        (18, 16, 256),
        (1, 32, 128),
        (4, 32, 128),
        (1, 32, 256),
        (4, 32, 256),
    ),
    # Historical preset name retained; the current Arc default value block for
    # these V=64 shapes is 8 after the tiled subgroup-span remap.
    "arc-v64-default-block16": (
        (1, 8, 64),
        (4, 8, 64),
        (8, 8, 64),
        (16, 8, 64),
        (1, 16, 64),
        (2, 16, 64),
        (4, 16, 64),
        (8, 16, 64),
        (1, 32, 64),
        (2, 32, 64),
        (4, 32, 64),
    ),
    "arc-legacy-v64-block8": (
        (1, 8, 64),
        (4, 8, 64),
        (8, 8, 64),
        (1, 16, 64),
        (2, 16, 64),
        (4, 16, 64),
        (8, 16, 64),
        (1, 32, 64),
        (2, 32, 64),
        (4, 32, 64),
    ),
    # Fast A770 regression watchlist for the forced V=64/value_block=16 path,
    # which now also prefers the single-group decode strategy.
    "arc-watch-v64-block16": (
        (1, 8, 64),
        (8, 8, 64),
        (32, 8, 64),
        (128, 8, 64),
        (2, 16, 64),
        (8, 16, 64),
        (32, 16, 64),
        (2, 32, 64),
        (8, 32, 64),
        (32, 32, 64),
    ),
    "arc-legacy-v128-block8": (
        (4, 8, 128),
        (8, 8, 128),
        (16, 8, 128),
        (18, 8, 128),
        (19, 8, 128),
        (4, 16, 128),
        (8, 16, 128),
        (9, 16, 128),
        (10, 16, 128),
        (4, 32, 128),
        (5, 32, 128),
        (8, 32, 128),
    ),
    # Fast A770 regression watchlist for the V=128/value_block=8 low-row shapes
    # that flipped from single-group to tiled after the tiled subgroup-span remap.
    "arc-watch-v128-block8": (
        (16, 8, 128),
        (18, 8, 128),
        (19, 8, 128),
        (8, 16, 128),
        (9, 16, 128),
        (10, 16, 128),
        (4, 32, 128),
        (5, 32, 128),
    ),
    # Boundary watchlist for Arc's V=128 default block, which should stay on 8
    # instead of regressing to the slower tiled block=16 variant.
    "arc-watch-v128-default8-vs-block16": (
        (4, 8, 128),
        (16, 8, 128),
        (24, 8, 128),
        (4, 16, 128),
        (8, 16, 128),
        (12, 16, 128),
        (1, 32, 128),
        (4, 32, 128),
        (6, 32, 128),
        (8, 32, 128),
    ),
    "arc-legacy-v256-block4": (
        (33, 8, 256),
        (65, 8, 256),
        (91, 8, 256),
        (97, 8, 256),
        (145, 8, 256),
        (146, 8, 256),
        (193, 8, 256),
        (251, 8, 256),
        (10, 16, 256),
        (17, 16, 256),
        (18, 16, 256),
        (29, 16, 256),
        (30, 16, 256),
        (33, 16, 256),
        (34, 16, 256),
        (45, 16, 256),
        (46, 16, 256),
        (49, 16, 256),
        (59, 16, 256),
        (60, 16, 256),
        (64, 16, 256),
        (65, 16, 256),
        (71, 16, 256),
        (72, 16, 256),
        (73, 16, 256),
        (5, 32, 256),
        (9, 32, 256),
        (15, 32, 256),
        (17, 32, 256),
        (23, 32, 256),
        (24, 32, 256),
        (25, 32, 256),
        (29, 32, 256),
        (30, 32, 256),
        (32, 32, 256),
        (33, 32, 256),
        (35, 32, 256),
        (36, 32, 256),
        (37, 32, 256),
    ),
    # Fast A770 regression watchlist for the V=256/value_block=4 low-row shapes
    # that flipped from single-group to tiled after the tiled subgroup-span remap.
    "arc-watch-v256-block4": (
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
    ),
    # Boundary shapes where Arc's V=256 default block should stay on 4 even when
    # compared directly against forced block=8, including both sides of the
    # retained 264..304 block=8 band and the old second band that now falls
    # back to block=4 after the tiled full-tile specialization.
    "arc-watch-v256-default4-vs-block8": (
        (32, 8, 256),
        (39, 8, 256),
        (40, 8, 256),
        (56, 8, 256),
        (57, 8, 256),
        (58, 8, 256),
        (60, 8, 256),
        (62, 8, 256),
        (63, 8, 256),
        (64, 8, 256),
        (16, 16, 256),
        (20, 16, 256),
        (28, 16, 256),
        (29, 16, 256),
        (30, 16, 256),
        (31, 16, 256),
        (32, 16, 256),
        (14, 32, 256),
        (15, 32, 256),
        (16, 32, 256),
    ),
    # Boundary shapes where Arc's V=256 default block should stay on 8 inside
    # the retained 264..304 row band when compared directly against forced
    # block=4.
    "arc-watch-v256-default8-vs-block4": (
        (33, 8, 256),
        (37, 8, 256),
        (38, 8, 256),
        (17, 16, 256),
        (18, 16, 256),
        (19, 16, 256),
        (9, 32, 256),
    ),
}

GDN_DECODE_PRESET_VALUE_BLOCKS: dict[str, tuple[int, ...]] = {
    "arc-default": (4, 8),
    "arc-v64-default-block16": (8,),
    "arc-legacy-v64-block8": (8,),
    "arc-watch-v64-block16": (16,),
    "arc-legacy-v128-block8": (8,),
    "arc-watch-v128-block8": (8,),
    "arc-watch-v128-default8-vs-block16": (16,),
    "arc-legacy-v256-block4": (4,),
    "arc-watch-v256-block4": (4,),
    "arc-watch-v256-default4-vs-block8": (8,),
    "arc-watch-v256-default8-vs-block4": (4,),
}


def _parse_gdn_decode_shape_presets(raw: str) -> list[str]:
    preset_names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not preset_names:
        raise ValueError("--gdn-decode-shape-presets must contain at least one preset name")
    unknown_presets = [preset_name for preset_name in preset_names if preset_name not in GDN_DECODE_SHAPE_PRESETS]
    if unknown_presets:
        available = ", ".join(sorted(GDN_DECODE_SHAPE_PRESETS))
        raise ValueError(
            f"Unknown --gdn-decode-shape-presets entry(s): {', '.join(unknown_presets)}. "
            f"Available presets: {available}"
        )
    return preset_names


def _dedupe_gdn_decode_shape_cases(cases: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    deduped_cases: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for case in cases:
        if case in seen:
            continue
        seen.add(case)
        deduped_cases.append(case)
    return deduped_cases


def _resolve_gdn_decode_value_blocks(
    raw: str | None,
    preset_names: list[str],
) -> list[int]:
    if raw is not None:
        return _parse_int_csv("--gdn-decode-value-blocks", raw)

    if preset_names:
        value_blocks: list[int] = []
        for preset_name in preset_names:
            for value_block in GDN_DECODE_PRESET_VALUE_BLOCKS[preset_name]:
                if value_block not in value_blocks:
                    value_blocks.append(value_block)
        if value_blocks:
            return value_blocks

    return [1, 2, 4, 8, 16, 32]


def _parse_gdn_decode_seeds(seed: int | None, seeds_csv: str | None) -> list[int | None]:
    if seeds_csv is None:
        return [seed]
    parsed_seeds = _parse_int_csv("--gdn-decode-seeds", seeds_csv)
    return [None if parsed_seed < 0 else parsed_seed for parsed_seed in parsed_seeds]


def _resolve_gated_delta_decode_auto_strategy(
    query: torch.Tensor,
    *,
    value_head_dim: int,
    value_block: int,
) -> str:
    anna_ops = getattr(torch.ops, "anna", None)
    if anna_ops is None or not hasattr(anna_ops, "gated_delta_decode_strategy_debug"):
        return "unknown"

    strategy_code = int(anna_ops.gated_delta_decode_strategy_debug(query, value_head_dim, value_block))
    if strategy_code == 0:
        return "single"
    if strategy_code == 1:
        return "tiled"
    return f"unknown({strategy_code})"


def _resolve_gated_delta_decode_default_value_block(
    query: torch.Tensor,
    *,
    value_head_dim: int,
) -> int | None:
    anna_ops = getattr(torch.ops, "anna", None)
    if anna_ops is None or not hasattr(anna_ops, "gated_delta_decode_value_block_debug"):
        return None
    return int(anna_ops.gated_delta_decode_value_block_debug(query, value_head_dim))


def _compare_gated_delta_decode_auto_against_explicit(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    value_block: int,
    single_min_elements: int | None,
    warmup: int,
    iters: int,
    timing_repeats: int,
) -> dict[str, float | str]:
    candidate_measurements = _collect_gated_delta_decode_candidate_timings(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        candidates=[
            ("single", "single", value_block),
            ("tiled", "tiled", value_block),
            ("auto", "auto", value_block),
        ],
        single_min_elements=single_min_elements,
        warmup=warmup,
        iters=iters,
        timing_repeats=timing_repeats,
    )
    single_ms = float(candidate_measurements["single"]["candidate_ms"])
    tiled_ms = float(candidate_measurements["tiled"]["candidate_ms"])
    auto_ms = float(candidate_measurements["auto"]["candidate_ms"])
    auto_strategy = _resolve_gated_delta_decode_auto_strategy(
        query,
        value_head_dim=int(value.shape[-1]),
        value_block=value_block,
    )
    best_explicit_strategy = "single" if single_ms <= tiled_ms else "tiled"
    best_explicit_ms = min(single_ms, tiled_ms)
    auto_minus_best_ms = auto_ms - best_explicit_ms
    auto_speed_ratio = auto_ms / best_explicit_ms if best_explicit_ms > 0 else float("inf")
    max_abs_diff = max(float(candidate["max_abs_diff"]) for candidate in candidate_measurements.values())
    return {
        "auto_strategy": auto_strategy,
        "single_ms": single_ms,
        "tiled_ms": tiled_ms,
        "auto_ms": auto_ms,
        "best_explicit_strategy": best_explicit_strategy,
        "best_explicit_ms": best_explicit_ms,
        "auto_minus_best_ms": auto_minus_best_ms,
        "auto_speed_ratio": auto_speed_ratio,
        "max_abs_diff": max_abs_diff,
    }


def _compare_gated_delta_decode_default_against_forced_block(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    forced_value_block: int,
    single_min_elements: int | None,
    warmup: int,
    iters: int,
    timing_repeats: int,
) -> dict[str, float | str | int]:
    default_value_block = _resolve_gated_delta_decode_default_value_block(
        query,
        value_head_dim=int(value.shape[-1]),
    )
    default_strategy = (
        "unknown"
        if default_value_block is None
        else _resolve_gated_delta_decode_auto_strategy(
            query,
            value_head_dim=int(value.shape[-1]),
            value_block=default_value_block,
        )
    )
    forced_strategy = _resolve_gated_delta_decode_auto_strategy(
        query,
        value_head_dim=int(value.shape[-1]),
        value_block=forced_value_block,
    )

    candidate_measurements = _collect_gated_delta_decode_candidate_timings(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        candidates=[
            ("default", "auto", None),
            ("forced", "auto", forced_value_block),
        ],
        single_min_elements=single_min_elements,
        warmup=warmup,
        iters=iters,
        timing_repeats=timing_repeats,
    )
    default_ms = float(candidate_measurements["default"]["candidate_ms"])
    forced_ms = float(candidate_measurements["forced"]["candidate_ms"])
    default_best_ms = min(default_ms, forced_ms)
    forced_minus_default_ms = forced_ms - default_ms
    forced_over_default_ratio = forced_ms / default_ms if default_ms > 0 else float("inf")
    default_speed_ratio_vs_forced = default_ms / default_best_ms if default_best_ms > 0 else float("inf")
    max_abs_diff = max(float(candidate["max_abs_diff"]) for candidate in candidate_measurements.values())
    return {
        "default_value_block": -1 if default_value_block is None else default_value_block,
        "default_strategy": default_strategy,
        "forced_value_block": forced_value_block,
        "forced_strategy": forced_strategy,
        "default_ms": default_ms,
        "forced_ms": forced_ms,
        "default_best_ms": default_best_ms,
        "forced_minus_default_ms": forced_minus_default_ms,
        "forced_over_default_ratio": forced_over_default_ratio,
        "default_speed_ratio_vs_forced": default_speed_ratio_vs_forced,
        "max_abs_diff": max_abs_diff,
    }


def _compare_gated_delta_decode_default_against_explicit(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    single_min_elements: int | None,
    warmup: int,
    iters: int,
    timing_repeats: int,
) -> dict[str, float | str | int]:
    default_value_block = _resolve_gated_delta_decode_default_value_block(
        query,
        value_head_dim=int(value.shape[-1]),
    )
    default_strategy = (
        "unknown"
        if default_value_block is None
        else _resolve_gated_delta_decode_auto_strategy(
            query,
            value_head_dim=int(value.shape[-1]),
            value_block=default_value_block,
        )
    )

    candidate_measurements = _collect_gated_delta_decode_candidate_timings(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        candidates=[
            ("default", "auto", None),
            ("single", "single", default_value_block),
            ("tiled", "tiled", default_value_block),
        ],
        single_min_elements=single_min_elements,
        warmup=warmup,
        iters=iters,
        timing_repeats=timing_repeats,
    )
    default_ms = float(candidate_measurements["default"]["candidate_ms"])
    single_ms = float(candidate_measurements["single"]["candidate_ms"])
    tiled_ms = float(candidate_measurements["tiled"]["candidate_ms"])
    best_explicit_strategy = "single" if single_ms <= tiled_ms else "tiled"
    best_explicit_ms = min(single_ms, tiled_ms)
    default_minus_best_ms = default_ms - best_explicit_ms
    default_speed_ratio = default_ms / best_explicit_ms if best_explicit_ms > 0 else float("inf")
    max_abs_diff = max(float(candidate["max_abs_diff"]) for candidate in candidate_measurements.values())
    return {
        "default_value_block": -1 if default_value_block is None else default_value_block,
        "default_strategy": default_strategy,
        "single_ms": single_ms,
        "tiled_ms": tiled_ms,
        "default_ms": default_ms,
        "best_explicit_strategy": best_explicit_strategy,
        "best_explicit_ms": best_explicit_ms,
        "default_minus_best_ms": default_minus_best_ms,
        "default_speed_ratio": default_speed_ratio,
        "max_abs_diff": max_abs_diff,
    }


def _run_gated_delta_decode_profile_case(
    *,
    device_name: str,
    batch_size: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
    value_blocks: list[int],
    single_min_elements: int | None,
    warmup: int,
    iters: int,
    timing_repeats: int,
    seeds: list[int | None],
    auto_compare: bool,
    default_compare: bool,
    default_block_compare: bool,
    compare_only: bool,
) -> None:
    profile_measurements_by_seed: list[dict[str, dict[str, float]]] = []
    single_profile_value_block = value_blocks[0] if value_blocks else None
    if not compare_only:
        candidates: list[dict[str, object]] = []
        for strategy in ("single", "tiled", "auto"):
            for value_block in value_blocks:
                if strategy == "single" and value_block != value_blocks[0]:
                    continue
                candidates.append(
                    {
                        "candidate_id": _gated_delta_decode_candidate_id(strategy, value_block),
                        "strategy": strategy,
                        "value_block": value_block,
                        "samples_ms": [],
                        "max_abs_diff": 0.0,
                    }
                )
        for seed in seeds:
            query, key, value, g, beta, initial_state = _make_gated_delta_decode_bench_inputs(
                batch_size=batch_size,
                num_heads=num_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                dtype=dtype,
                seed=seed,
            )
            seed_measurements = _collect_gated_delta_decode_candidate_timings(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                candidates=[
                    (
                        str(candidate["candidate_id"]),
                        str(candidate["strategy"]),
                        int(candidate["value_block"]),
                    )
                    for candidate in candidates
                ],
                single_min_elements=single_min_elements,
                warmup=warmup,
                iters=iters,
                timing_repeats=timing_repeats,
            )
            profile_measurements_by_seed.append(seed_measurements)
            for candidate in candidates:
                measurement = seed_measurements[str(candidate["candidate_id"])]
                candidate["samples_ms"].append(float(measurement["candidate_ms"]))
                candidate["max_abs_diff"] = max(float(candidate["max_abs_diff"]), float(measurement["max_abs_diff"]))
        single_min_elements_label = "default" if single_min_elements is None else str(single_min_elements)
        for candidate in candidates:
            candidate_ms = statistics.median(candidate["samples_ms"])
            print(
                f"gdn_decode_profile,{device_name},{candidate['strategy']},{batch_size},{num_heads},"
                f"{key_head_dim},{value_head_dim},{candidate['value_block']},{single_min_elements_label},"
                f"{dtype},{timing_repeats},{candidate_ms:.4f},{float(candidate['max_abs_diff']):.6f}"
            )
    if auto_compare:
        for value_block in value_blocks:
            compare_results: list[dict[str, float | str]] = []
            if profile_measurements_by_seed:
                debug_query = torch.empty(batch_size, 1, num_heads, key_head_dim, device="xpu", dtype=dtype)
                auto_strategy = _resolve_gated_delta_decode_auto_strategy(
                    debug_query,
                    value_head_dim=value_head_dim,
                    value_block=value_block,
                )
                single_candidate_id = _gated_delta_decode_candidate_id("single", single_profile_value_block)
                tiled_candidate_id = _gated_delta_decode_candidate_id("tiled", value_block)
                auto_candidate_id = _gated_delta_decode_candidate_id("auto", value_block)
                for seed_measurements in profile_measurements_by_seed:
                    single_ms = float(seed_measurements[single_candidate_id]["candidate_ms"])
                    tiled_ms = float(seed_measurements[tiled_candidate_id]["candidate_ms"])
                    auto_ms = float(seed_measurements[auto_candidate_id]["candidate_ms"])
                    best_explicit_strategy = "single" if single_ms <= tiled_ms else "tiled"
                    best_explicit_ms = min(single_ms, tiled_ms)
                    compare_results.append(
                        {
                            "auto_strategy": auto_strategy,
                            "single_ms": single_ms,
                            "tiled_ms": tiled_ms,
                            "auto_ms": auto_ms,
                            "best_explicit_strategy": best_explicit_strategy,
                            "best_explicit_ms": best_explicit_ms,
                            "auto_minus_best_ms": auto_ms - best_explicit_ms,
                            "auto_speed_ratio": auto_ms / best_explicit_ms if best_explicit_ms > 0 else float("inf"),
                            "max_abs_diff": max(
                                float(seed_measurements[single_candidate_id]["max_abs_diff"]),
                                float(seed_measurements[tiled_candidate_id]["max_abs_diff"]),
                                float(seed_measurements[auto_candidate_id]["max_abs_diff"]),
                            ),
                        }
                    )
            else:
                for seed in seeds:
                    query, key, value, g, beta, initial_state = _make_gated_delta_decode_bench_inputs(
                        batch_size=batch_size,
                        num_heads=num_heads,
                        key_head_dim=key_head_dim,
                        value_head_dim=value_head_dim,
                        dtype=dtype,
                        seed=seed,
                    )
                    compare_results.append(
                        _compare_gated_delta_decode_auto_against_explicit(
                            query=query,
                            key=key,
                            value=value,
                            g=g,
                            beta=beta,
                            initial_state=initial_state,
                            value_block=value_block,
                            single_min_elements=single_min_elements,
                            warmup=warmup,
                            iters=iters,
                            timing_repeats=timing_repeats,
                        )
                    )
            single_ms = statistics.median(float(result["single_ms"]) for result in compare_results)
            tiled_ms = statistics.median(float(result["tiled_ms"]) for result in compare_results)
            auto_ms = statistics.median(float(result["auto_ms"]) for result in compare_results)
            auto_strategy_labels = {str(result["auto_strategy"]) for result in compare_results}
            auto_strategy = auto_strategy_labels.pop() if len(auto_strategy_labels) == 1 else "inconsistent"
            best_explicit_strategy = "single" if single_ms <= tiled_ms else "tiled"
            best_explicit_ms = min(single_ms, tiled_ms)
            auto_minus_best_ms = auto_ms - best_explicit_ms
            auto_speed_ratio = auto_ms / best_explicit_ms if best_explicit_ms > 0 else float("inf")
            max_abs_diff = max(float(result["max_abs_diff"]) for result in compare_results)
            print(
                f"gdn_decode_auto_compare,{device_name},{batch_size},{num_heads},"
                f"{key_head_dim},{value_head_dim},{value_block},{auto_strategy},{single_ms:.4f},"
                f"{tiled_ms:.4f},{auto_ms:.4f},{best_explicit_strategy},{best_explicit_ms:.4f},"
                f"{auto_minus_best_ms:.4f},{auto_speed_ratio:.4f},{max_abs_diff:.6f}"
            )
    if default_compare:
        compare_results: list[dict[str, float | str | int]] = []
        for seed in seeds:
            query, key, value, g, beta, initial_state = _make_gated_delta_decode_bench_inputs(
                batch_size=batch_size,
                num_heads=num_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                dtype=dtype,
                seed=seed,
            )
            compare_results.append(
                _compare_gated_delta_decode_default_against_explicit(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    single_min_elements=single_min_elements,
                    warmup=warmup,
                    iters=iters,
                    timing_repeats=timing_repeats,
                )
            )
        default_value_block_labels = {int(result["default_value_block"]) for result in compare_results}
        default_value_block = default_value_block_labels.pop() if len(default_value_block_labels) == 1 else -1
        default_strategy_labels = {str(result["default_strategy"]) for result in compare_results}
        default_strategy = default_strategy_labels.pop() if len(default_strategy_labels) == 1 else "inconsistent"
        single_ms = statistics.median(float(result["single_ms"]) for result in compare_results)
        tiled_ms = statistics.median(float(result["tiled_ms"]) for result in compare_results)
        default_ms = statistics.median(float(result["default_ms"]) for result in compare_results)
        best_explicit_strategy = "single" if single_ms <= tiled_ms else "tiled"
        best_explicit_ms = min(single_ms, tiled_ms)
        default_minus_best_ms = default_ms - best_explicit_ms
        default_speed_ratio = default_ms / best_explicit_ms if best_explicit_ms > 0 else float("inf")
        max_abs_diff = max(float(result["max_abs_diff"]) for result in compare_results)
        print(
            f"gdn_decode_default_compare,{device_name},{batch_size},{num_heads},"
            f"{key_head_dim},{value_head_dim},{default_value_block},{default_strategy},{single_ms:.4f},"
            f"{tiled_ms:.4f},{default_ms:.4f},{best_explicit_strategy},{best_explicit_ms:.4f},"
            f"{default_minus_best_ms:.4f},{default_speed_ratio:.4f},{max_abs_diff:.6f}"
        )
    if default_block_compare:
        for forced_value_block in value_blocks:
            compare_results: list[dict[str, float | str | int]] = []
            for seed in seeds:
                query, key, value, g, beta, initial_state = _make_gated_delta_decode_bench_inputs(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    dtype=dtype,
                    seed=seed,
                )
                compare_results.append(
                    _compare_gated_delta_decode_default_against_forced_block(
                        query=query,
                        key=key,
                        value=value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        forced_value_block=forced_value_block,
                        single_min_elements=single_min_elements,
                        warmup=warmup,
                        iters=iters,
                        timing_repeats=timing_repeats,
                    )
                )
            default_value_block_labels = {int(result["default_value_block"]) for result in compare_results}
            default_value_block = default_value_block_labels.pop() if len(default_value_block_labels) == 1 else -1
            default_strategy_labels = {str(result["default_strategy"]) for result in compare_results}
            default_strategy = default_strategy_labels.pop() if len(default_strategy_labels) == 1 else "inconsistent"
            forced_strategy_labels = {str(result["forced_strategy"]) for result in compare_results}
            forced_strategy = forced_strategy_labels.pop() if len(forced_strategy_labels) == 1 else "inconsistent"
            default_ms = statistics.median(float(result["default_ms"]) for result in compare_results)
            forced_ms = statistics.median(float(result["forced_ms"]) for result in compare_results)
            default_best_ms = min(default_ms, forced_ms)
            forced_minus_default_ms = forced_ms - default_ms
            forced_over_default_ratio = forced_ms / default_ms if default_ms > 0 else float("inf")
            default_speed_ratio_vs_forced = default_ms / default_best_ms if default_best_ms > 0 else float("inf")
            max_abs_diff = max(float(result["max_abs_diff"]) for result in compare_results)
            print(
                f"gdn_decode_default_block_compare,{device_name},{batch_size},{num_heads},"
                f"{key_head_dim},{value_head_dim},{default_value_block},{default_strategy},{forced_value_block},"
                f"{forced_strategy},{default_ms:.4f},{forced_ms:.4f},{default_best_ms:.4f},"
                f"{forced_minus_default_ms:.4f},{forced_over_default_ratio:.4f},"
                f"{default_speed_ratio_vs_forced:.4f},{max_abs_diff:.6f}"
            )


def _benchmark_xpu_int4_linear(
    *,
    tokens: int,
    in_features: int,
    out_features: int,
    group_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    strategy: str = "auto",
) -> tuple[float, float, float]:
    dense = torch.nn.Linear(in_features, out_features, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=group_size, compute_dtype=dtype, device="xpu")
    hidden_states = torch.randn(tokens, in_features, device="xpu", dtype=dtype)
    dense_weight = dense.weight.detach().to(device="xpu", dtype=dtype)

    baseline = lambda: F.linear(hidden_states, dense_weight)
    previous_strategy = os.environ.get("ANNA_XPU_INT4_MATMUL")
    try:
        os.environ["ANNA_XPU_INT4_MATMUL"] = strategy
        candidate = lambda: quantized(hidden_states)
        baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
        baseline_output = baseline()
        try:
            candidate_output = candidate()
            candidate_ms = _time_op(candidate, warmup=warmup, iters=iters)
            max_abs_diff = float((candidate_output.float() - baseline_output.float()).abs().max().item())
        except Exception as exc:
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
            print(
                "arc_int4_profile_error,"
                f"strategy={strategy},M={tokens},K={in_features},N={out_features},"
                f"group_size={group_size},error={type(exc).__name__}:{exc}"
            )
            candidate_ms = float("inf")
            max_abs_diff = float("inf")
        return baseline_ms, candidate_ms, max_abs_diff
    finally:
        if previous_strategy is None:
            os.environ.pop("ANNA_XPU_INT4_MATMUL", None)
        else:
            os.environ["ANNA_XPU_INT4_MATMUL"] = previous_strategy


def _benchmark_lm_head_int4_topk(
    *,
    tokens: int,
    in_features: int,
    vocab_size: int,
    top_k: int,
    group_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    local_size: int | None = None,
) -> tuple[float, float, float]:
    dense = torch.nn.Linear(in_features, vocab_size, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=group_size, compute_dtype=dtype, device="xpu")
    hidden_states = torch.randn(tokens, in_features, device="xpu", dtype=dtype)
    previous_local_size = os.environ.get("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE")
    if local_size is None:
        os.environ.pop("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE", None)
    else:
        os.environ["ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"] = str(local_size)

    baseline = lambda: torch.topk(quantized(hidden_states), k=top_k, dim=-1)
    candidate = lambda: torch.ops.anna.lm_head_int4_topk_fused(
        hidden_states,
        quantized.qweight,
        quantized.qscale,
        quantized.qzeros,
        quantized.group_size,
        quantized.in_features,
        top_k,
    )
    try:
        baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
        candidate_values, candidate_indices = candidate()
        baseline_values, baseline_indices = baseline()
        candidate_ms = _time_op(candidate, warmup=warmup, iters=iters)
    finally:
        if previous_local_size is None:
            os.environ.pop("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE", None)
        else:
            os.environ["ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"] = previous_local_size

    if not torch.equal(candidate_indices.cpu(), baseline_indices.cpu()):
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float((candidate_values.float() - baseline_values.float()).abs().max().item())
    return baseline_ms, candidate_ms, max_abs_diff


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _benchmark_moe_grouped_int4_mlp(
    *,
    tokens_per_expert: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    gate_local_size: int | None = None,
    down_local_size: int | None = None,
) -> tuple[float, float, float]:
    gate_layers: list[XPUInt4Linear] = []
    up_layers: list[XPUInt4Linear] = []
    down_layers: list[XPUInt4Linear] = []
    for _ in range(num_experts):
        gate_dense = torch.nn.Linear(hidden_size, intermediate_size, bias=False, device="xpu", dtype=torch.float32)
        up_dense = torch.nn.Linear(hidden_size, intermediate_size, bias=False, device="xpu", dtype=torch.float32)
        down_dense = torch.nn.Linear(intermediate_size, hidden_size, bias=False, device="xpu", dtype=torch.float32)
        gate_layers.append(XPUInt4Linear.from_linear(gate_dense, group_size=group_size, compute_dtype=dtype, device="xpu"))
        up_layers.append(XPUInt4Linear.from_linear(up_dense, group_size=group_size, compute_dtype=dtype, device="xpu"))
        down_layers.append(XPUInt4Linear.from_linear(down_dense, group_size=group_size, compute_dtype=dtype, device="xpu"))

    total_routes = max(1, tokens_per_expert) * max(1, num_experts)
    compact_hidden_states = torch.randn(total_routes, hidden_size, device="xpu", dtype=dtype)
    compact_routing_weights = torch.rand(total_routes, 1, device="xpu", dtype=dtype)
    compact_outputs = torch.empty(total_routes, hidden_size, device="xpu", dtype=dtype)
    offsets = torch.arange(0, total_routes + 1, max(1, tokens_per_expert), device="xpu", dtype=torch.long)
    active_experts = torch.arange(num_experts, device="xpu", dtype=torch.long)
    active_slots = torch.arange(num_experts, device="xpu", dtype=torch.long)

    gate_qweight = torch.stack([layer.qweight for layer in gate_layers], dim=0)
    gate_qscale = torch.stack([layer.qscale for layer in gate_layers], dim=0)
    gate_qzeros = torch.stack([layer.qzeros for layer in gate_layers], dim=0)
    up_qweight = torch.stack([layer.qweight for layer in up_layers], dim=0)
    up_qscale = torch.stack([layer.qscale for layer in up_layers], dim=0)
    up_qzeros = torch.stack([layer.qzeros for layer in up_layers], dim=0)
    down_qweight = torch.stack([layer.qweight for layer in down_layers], dim=0)
    down_qscale = torch.stack([layer.qscale for layer in down_layers], dim=0)
    down_qzeros = torch.stack([layer.qzeros for layer in down_layers], dim=0)

    def baseline() -> torch.Tensor:
        output = torch.empty_like(compact_outputs)
        for expert_idx in range(num_experts):
            start = expert_idx * max(1, tokens_per_expert)
            end = start + max(1, tokens_per_expert)
            hidden = compact_hidden_states[start:end]
            routed = down_layers[expert_idx](F.silu(gate_layers[expert_idx](hidden)) * up_layers[expert_idx](hidden))
            output[start:end] = routed * compact_routing_weights[start:end]
        return output

    def candidate() -> torch.Tensor:
        output = compact_outputs.zero_()
        return torch.ops.anna.moe_grouped_int4_mlp_fused(
            compact_hidden_states,
            compact_routing_weights,
            output,
            offsets,
            active_experts,
            active_slots,
            gate_qweight,
            gate_qscale,
            gate_qzeros,
            up_qweight,
            up_qscale,
            up_qzeros,
            down_qweight,
            down_qscale,
            down_qzeros,
            group_size,
            max(1, tokens_per_expert),
        )

    previous_gate = os.environ.get("ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE")
    previous_down = os.environ.get("ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE")
    if gate_local_size is None:
        os.environ.pop("ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE", None)
    else:
        os.environ["ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE"] = str(gate_local_size)
    if down_local_size is None:
        os.environ.pop("ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE", None)
    else:
        os.environ["ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE"] = str(down_local_size)

    try:
        baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
        candidate_output = candidate()
        baseline_output = baseline()
        candidate_ms = _time_op(candidate, warmup=warmup, iters=iters)
    finally:
        _restore_env("ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE", previous_gate)
        _restore_env("ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE", previous_down)

    max_abs_diff = float((candidate_output.float() - baseline_output.float()).abs().max().item())
    return baseline_ms, candidate_ms, max_abs_diff


def _benchmark_rmsnorm(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="xpu", dtype=dtype)
    weight = torch.randn(hidden_size, device="xpu", dtype=torch.float32)
    eps = 1e-6

    baseline = lambda: _reference_rmsnorm(hidden_states, weight, eps)
    fused = lambda: torch.ops.anna.rmsnorm_fused(hidden_states, weight, eps)

    baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
    fused_output = fused()
    baseline_output = baseline()
    max_abs_diff = float((fused_output.float() - baseline_output.float()).abs().max().item())
    fused_ms = _time_op(fused, warmup=warmup, iters=iters)
    return baseline_ms, fused_ms, max_abs_diff


def _benchmark_qk_norm_rotary(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    query_norm_weight = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    key_norm_weight = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    angles = torch.randn(batch_size, seq_len, rotary_dim, device="xpu", dtype=torch.float32)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    def baseline():
        normalized_query = _reference_rmsnorm(query, query_norm_weight, 1e-6)
        normalized_key = _reference_rmsnorm(key, key_norm_weight, 1e-6)
        return apply_rotary_pos_emb(normalized_query, normalized_key, cos, sin)

    fused = lambda: torch.ops.anna.qk_norm_rotary_fused(
        query,
        key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        1e-6,
        1e-6,
    )

    baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
    fused_query, fused_key = fused()
    baseline_query, baseline_key = baseline()
    max_abs_diff = max(
        float((fused_query.float() - baseline_query.float()).abs().max().item()),
        float((fused_key.float() - baseline_key.float()).abs().max().item()),
    )
    fused_ms = _time_op(fused, warmup=warmup, iters=iters)
    return baseline_ms, fused_ms, max_abs_diff


def _benchmark_repeat_kv(
    *,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> float:
    hidden_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    return _time_op(lambda: repeat_kv(hidden_states, num_key_value_groups), warmup=warmup, iters=iters)


def _benchmark_sdpa_gqa_prefill(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="xpu", dtype=dtype)
    num_key_value_groups = max(1, num_heads // max(1, num_kv_heads))

    def materialized():
        repeated_key_states = repeat_kv(key_states, num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, num_key_value_groups)
        return F.scaled_dot_product_attention(
            query_states,
            repeated_key_states,
            repeated_value_states,
            dropout_p=0.0,
            is_causal=seq_len > 1,
        )

    gqa = lambda: F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0,
        is_causal=seq_len > 1,
        enable_gqa=True,
    )

    baseline_ms = _time_op(materialized, warmup=warmup, iters=iters)
    grouped_output = gqa()
    baseline_output = materialized()
    max_abs_diff = float((grouped_output.float() - baseline_output.float()).abs().max().item())
    grouped_ms = _time_op(gqa, warmup=warmup, iters=iters)
    return baseline_ms, grouped_ms, max_abs_diff


def _benchmark_grouped_attention(
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, query_len, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    num_key_value_groups = max(1, num_heads // max(1, num_kv_heads))
    scaling = head_dim**-0.5

    def materialized():
        repeated_key_states = repeat_kv(key_states, num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, num_key_value_groups)
        attn_scores = torch.matmul(query_states, repeated_key_states.transpose(-1, -2)) * scaling
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
        return torch.matmul(attn_probs, repeated_value_states)

    grouped = lambda: grouped_query_attention(
        query_states,
        key_states,
        value_states,
        scaling=scaling,
    )

    baseline_ms = _time_op(materialized, warmup=warmup, iters=iters)
    grouped_output = grouped()
    baseline_output = materialized()
    max_abs_diff = float((grouped_output.float() - baseline_output.float()).abs().max().item())
    grouped_ms = _time_op(grouped, warmup=warmup, iters=iters)
    return baseline_ms, grouped_ms, max_abs_diff


def _benchmark_gqa_decode_fused(
    *,
    batch_size: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, 1, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    num_key_value_groups = max(1, num_heads // max(1, num_kv_heads))
    visible_lengths = torch.randint(max(1, key_len // 2), key_len + 1, (batch_size,), device="xpu", dtype=torch.long)
    key_positions = torch.arange(key_len, device="xpu")[None, :]
    visible_mask = key_positions < visible_lengths[:, None]
    scaling = head_dim**-0.5

    def materialized():
        repeated_key_states = repeat_kv(key_states, num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, num_key_value_groups)
        attn_scores = torch.matmul(query_states, repeated_key_states.transpose(-1, -2)) * scaling
        attn_scores = attn_scores.masked_fill(~visible_mask[:, None, None, :], float("-inf"))
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
        return torch.matmul(attn_probs, repeated_value_states)

    gqa = lambda: torch.ops.anna.gqa_decode_fused(
        query_states,
        key_states,
        value_states,
        visible_lengths,
        scaling,
    )

    baseline_ms = _time_op(materialized, warmup=warmup, iters=iters)
    gqa_output = gqa()
    baseline_output = materialized()
    max_abs_diff = float((gqa_output.float() - baseline_output.float()).abs().max().item())
    gqa_ms = _time_op(gqa, warmup=warmup, iters=iters)
    return baseline_ms, gqa_ms, max_abs_diff


def _benchmark_gqa_decode_gate_layouts(
    *,
    batch_size: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, 1, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    visible_lengths = torch.randint(max(1, key_len // 2), key_len + 1, (batch_size,), device="xpu", dtype=torch.long)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device="xpu", dtype=dtype)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scaling = head_dim**-0.5

    gated_3d = lambda: torch.ops.anna.gqa_decode_fused(
        query_states,
        key_states,
        value_states,
        visible_lengths,
        scaling,
        gate_3d,
    )
    gated_4d = lambda: torch.ops.anna.gqa_decode_fused(
        query_states,
        key_states,
        value_states,
        visible_lengths,
        scaling,
        gate_4d,
    )

    gate_3d_output = gated_3d()
    gate_4d_output = gated_4d()
    max_abs_diff = float((gate_3d_output.float() - gate_4d_output.float()).abs().max().item())
    gate_3d_ms = _time_op(gated_3d, warmup=warmup, iters=iters)
    gate_4d_ms = _time_op(gated_4d, warmup=warmup, iters=iters)
    return gate_3d_ms, gate_4d_ms, max_abs_diff


def _benchmark_paged_gqa_decode_gate_layouts(
    *,
    batch_size: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, 1, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    key_pages, value_pages, page_table = _pack_paged_kv(key_states, value_states, block_size=block_size)
    visible_lengths = torch.randint(max(1, key_len // 2), key_len + 1, (batch_size,), device="xpu", dtype=torch.long)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device="xpu", dtype=dtype)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scaling = head_dim**-0.5

    gated_3d = lambda: torch.ops.anna.paged_gqa_decode_fused(
        query_states,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        scaling,
        gate_3d,
    )
    gated_4d = lambda: torch.ops.anna.paged_gqa_decode_fused(
        query_states,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        scaling,
        gate_4d,
    )

    gate_3d_output = gated_3d()
    gate_4d_output = gated_4d()
    max_abs_diff = float((gate_3d_output.float() - gate_4d_output.float()).abs().max().item())
    gate_3d_ms = _time_op(gated_3d, warmup=warmup, iters=iters)
    gate_4d_ms = _time_op(gated_4d, warmup=warmup, iters=iters)
    return gate_3d_ms, gate_4d_ms, max_abs_diff


def _benchmark_sdpa_gqa_decode_full_visible(
    *,
    batch_size: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, 1, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    num_key_value_groups = max(1, num_heads // max(1, num_kv_heads))
    scaling = head_dim**-0.5

    def materialized():
        repeated_key_states = repeat_kv(key_states, num_key_value_groups)
        repeated_value_states = repeat_kv(value_states, num_key_value_groups)
        attn_scores = torch.matmul(query_states, repeated_key_states.transpose(-1, -2)) * scaling
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
        return torch.matmul(attn_probs, repeated_value_states)

    gqa = lambda: F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )

    baseline_ms = _time_op(materialized, warmup=warmup, iters=iters)
    gqa_output = gqa()
    baseline_output = materialized()
    max_abs_diff = float((gqa_output.float() - baseline_output.float()).abs().max().item())
    gqa_ms = _time_op(gqa, warmup=warmup, iters=iters)
    return baseline_ms, gqa_ms, max_abs_diff


def _benchmark_sdpa_gqa_decode_variable_visible(
    *,
    batch_size: int,
    key_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    query_states = torch.randn(batch_size, num_heads, 1, head_dim, device="xpu", dtype=dtype)
    key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device="xpu", dtype=dtype)
    num_key_value_groups = max(1, num_heads // max(1, num_kv_heads))
    visible_lengths = torch.randint(max(1, key_len // 2), key_len + 1, (batch_size,), device="xpu", dtype=torch.long)
    visible_mask = torch.arange(key_len, device="xpu")[None, :] < visible_lengths[:, None]
    scaling = head_dim**-0.5

    def grouped():
        return grouped_query_attention(
            query_states,
            key_states,
            value_states,
            scaling=scaling,
            visible_mask=visible_mask,
        )

    gqa = lambda: F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=visible_mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=num_key_value_groups > 1,
    )

    baseline_ms = _time_op(grouped, warmup=warmup, iters=iters)
    gqa_output = gqa()
    baseline_output = grouped()
    max_abs_diff = float((gqa_output.float() - baseline_output.float()).abs().max().item())
    gqa_ms = _time_op(gqa, warmup=warmup, iters=iters)
    return baseline_ms, gqa_ms, max_abs_diff


def _reference_moe_router(
    *,
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    normalize_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if normalize_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    usage = expert_mask.sum(dim=(-1, -2))
    return routing_weights, selected_experts, usage


def _benchmark_router(
    *,
    tokens: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    router_logits = torch.randn(tokens, num_experts, device="xpu", dtype=dtype)

    baseline = lambda: _reference_moe_router(
        router_logits=router_logits,
        num_experts=num_experts,
        top_k=top_k,
        normalize_topk_prob=True,
    )
    fused = lambda: torch.ops.anna.moe_router_fused(router_logits, top_k, True)

    baseline_ms = _time_op(baseline, warmup=warmup, iters=iters)
    fused_weights, fused_selected, fused_usage = fused()
    baseline_weights, baseline_selected, baseline_usage = baseline()
    max_abs_diff = max(
        float((fused_weights.float() - baseline_weights.float()).abs().max().item()),
        float((fused_selected.float() - baseline_selected.float()).abs().max().item()),
        float((fused_usage.float() - baseline_usage.float()).abs().max().item()),
    )
    fused_ms = _time_op(fused, warmup=warmup, iters=iters)
    return baseline_ms, fused_ms, max_abs_diff


def _format_speedup(baseline_ms: float, fused_ms: float) -> str:
    if fused_ms <= 0.0:
        return "n/a"
    return f"{baseline_ms / fused_ms:.2f}x"


def _append_benchmark_row(
    rows: list[dict[str, str]],
    *,
    op: str,
    baseline_ms: float | None,
    fused_ms: float | None,
    speedup: str,
    max_abs_diff: float | None,
) -> None:
    rows.append(
        {
            "op": op,
            "baseline_ms": "-" if baseline_ms is None else f"{baseline_ms:.4f}",
            "fused_ms": "-" if fused_ms is None else f"{fused_ms:.4f}",
            "speedup": speedup,
            "max_abs_diff": "-" if max_abs_diff is None else f"{max_abs_diff:.6f}",
        }
    )


def _write_benchmark_csv(path: str, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("op", "baseline_ms", "fused_ms", "speedup", "max_abs_diff"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark current XPU hotspot ops and fused SYCL kernels.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--rotary-fraction", type=float, default=0.25)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--int4-m", type=int, default=1, help="Token rows for XPU int4 linear benchmark.")
    parser.add_argument("--int4-k", type=int, default=None, help="Input features for XPU int4 linear benchmark.")
    parser.add_argument("--int4-n", type=int, default=None, help="Output features for XPU int4 linear benchmark.")
    parser.add_argument(
        "--lm-head-vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for lm_head_int4_topk_fused arc-profile rows. Defaults to --int4-n.",
    )
    parser.add_argument("--int4-group-size", type=int, default=128)
    parser.add_argument(
        "--arc-profile",
        action="store_true",
        help="Run additional Arc A770/A750-oriented int4 linear shapes for decode small batches.",
    )
    parser.add_argument(
        "--arc-int4-only",
        action="store_true",
        help="Only run the Arc int4 profile rows. Requires --arc-profile and skips the general hotspot suite.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Write the general hotspot benchmark rows to this CSV file.",
    )
    parser.add_argument(
        "--gdn-decode-profile",
        action="store_true",
        help="Run Gated Delta single-token decode strategy sweep rows for Qwen3.5 recurrent shapes.",
    )
    parser.add_argument(
        "--gdn-decode-only",
        action="store_true",
        help="Only run the Gated Delta decode strategy sweep.",
    )
    parser.add_argument(
        "--gdn-value-head-dim",
        type=int,
        default=None,
        help="Value head dim for Gated Delta decode profiling. Defaults to --head-dim.",
    )
    parser.add_argument(
        "--gdn-value-head-dims",
        type=str,
        default=None,
        help="Comma-separated value head dims for multi-shape Gated Delta decode profiling. Overrides --gdn-value-head-dim.",
    )
    parser.add_argument(
        "--gdn-decode-value-blocks",
        type=str,
        default=None,
        help=(
            "Comma-separated ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK values for the Gated Delta decode sweep. "
            "Defaults to preset-specific recommended blocks when --gdn-decode-shape-presets is set, "
            "otherwise 1,2,4,8,16,32."
        ),
    )
    parser.add_argument(
        "--gdn-decode-single-min-elements",
        type=int,
        default=None,
        help=(
            "Override ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS for auto strategy profiling. "
            "Defaults to the fused op's built-in threshold."
        ),
    )
    parser.add_argument(
        "--gdn-decode-seed",
        type=int,
        default=0,
        help=(
            "Seed for Gated Delta decode profile inputs. "
            "Use a negative value to keep per-run random inputs."
        ),
    )
    parser.add_argument(
        "--gdn-decode-seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seed list for decode profiling. "
            "When set, aggregate each case across multiple fixed inputs and override --gdn-decode-seed."
        ),
    )
    parser.add_argument(
        "--gdn-decode-timing-repeats",
        type=int,
        default=3,
        help="Repeated timing samples per Gated Delta decode candidate. The reported candidate_ms is the median.",
    )
    parser.add_argument(
        "--gdn-decode-auto-compare",
        action="store_true",
        help="Print per-value-block auto-vs-explicit summary rows after the decode sweep.",
    )
    parser.add_argument(
        "--gdn-decode-default-compare",
        action="store_true",
        help="Print default-path vs explicit single/tiled summary rows after the decode sweep.",
    )
    parser.add_argument(
        "--gdn-decode-default-block-compare",
        action="store_true",
        help="Print default-path vs forced-value-block summary rows after the decode sweep.",
    )
    parser.add_argument(
        "--gdn-decode-compare-only",
        action="store_true",
        help="Skip the full strategy sweep rows and only print decode compare summaries.",
    )
    parser.add_argument(
        "--gdn-decode-batch-head-cases",
        type=str,
        default=None,
        help="Comma-separated batch/head cases for multi-shape Gated Delta decode profiling, for example 1x16,1x32,4x32.",
    )
    parser.add_argument(
        "--gdn-decode-shape-cases",
        type=str,
        default=None,
        help=(
            "Comma-separated exact Gated Delta decode shape cases in BxHxV form, for example "
            "1x16x64,4x16x64,1x32x128. Overrides --gdn-decode-batch-head-cases and --gdn-value-head-dims."
        ),
    )
    parser.add_argument(
        "--gdn-decode-shape-presets",
        type=str,
        default=None,
        help=(
            "Comma-separated named Gated Delta decode shape presets. Available presets: "
            + ", ".join(sorted(GDN_DECODE_SHAPE_PRESETS))
            + ". Presets append to --gdn-decode-shape-cases when both are set."
        ),
    )
    args = parser.parse_args()
    if args.gdn_decode_only:
        args.gdn_decode_profile = True

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is unavailable in the active environment.")
    if not maybe_load_gated_delta_library():
        raise RuntimeError("Anna fused-op library is not available. Build it first with tools/build_gated_delta_fused_op.py.")

    dtype = _resolve_dtype(args.dtype)
    rotary_dim = int(args.head_dim * args.rotary_fraction)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
    tokens = args.batch_size * args.seq_len
    kv_len = args.seq_len if args.kv_len is None else int(args.kv_len)
    num_key_value_groups = max(1, args.num_heads // max(1, args.num_kv_heads))
    xpu_info = inspect_xpu_device(torch.device("xpu"))
    effective_top_k = max(1, min(args.top_k, args.experts))

    int4_k = args.hidden_size if args.int4_k is None else args.int4_k
    int4_n = args.hidden_size if args.int4_n is None else args.int4_n
    lm_head_vocab_size = int4_n if args.lm_head_vocab_size is None else args.lm_head_vocab_size

    multi_shape_gdn_profile = args.gdn_decode_profile and (
        args.gdn_decode_shape_presets is not None
        or args.gdn_decode_shape_cases is not None
        or args.gdn_decode_batch_head_cases is not None
        or args.gdn_value_head_dims is not None
    )
    if multi_shape_gdn_profile:
        shape_label = (
            "<gdn-shapes>"
            if args.gdn_decode_shape_cases is not None or args.gdn_decode_shape_presets is not None
            else "<gdn-matrix>"
        )
        print(
            f"shape batch={shape_label} seq=1 hidden={args.hidden_size} "
            f"heads={shape_label}/{args.num_kv_heads} head_dim={args.head_dim} rotary_dim={rotary_dim} kv_len=1 "
            f"dtype={dtype}"
        )
        if args.gdn_decode_shape_presets is not None:
            print(f"gdn_decode_shape_presets={args.gdn_decode_shape_presets}")
        if args.gdn_decode_shape_cases is not None:
            print(f"gdn_decode_shape_cases={args.gdn_decode_shape_cases}")
        if args.gdn_decode_batch_head_cases is not None:
            print(f"gdn_decode_batch_head_cases={args.gdn_decode_batch_head_cases}")
        if args.gdn_value_head_dims is not None:
            print(f"gdn_value_head_dims={args.gdn_value_head_dims}")
        if args.gdn_decode_seeds is not None:
            print(f"gdn_decode_seeds={args.gdn_decode_seeds}")
    else:
        print(
            f"shape batch={args.batch_size} seq={args.seq_len} hidden={args.hidden_size} "
            f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} rotary_dim={rotary_dim} kv_len={kv_len} "
            f"dtype={dtype}"
        )
        if args.gdn_decode_seeds is not None:
            print(f"gdn_decode_seeds={args.gdn_decode_seeds}")
    if xpu_info is not None:
        print(f"device_name={xpu_info.name}")
        print(f"device_index={xpu_info.device_index}")
    if args.arc_int4_only and not args.arc_profile:
        raise ValueError("--arc-int4-only requires --arc-profile")
    if args.gdn_decode_profile:
        gdn_decode_shape_preset_names: list[str] = []
        gdn_decode_shape_cases: list[tuple[int, int, int]]
        if args.gdn_decode_shape_presets is not None or args.gdn_decode_shape_cases is not None:
            gdn_decode_shape_cases = []
            if args.gdn_decode_shape_presets is not None:
                gdn_decode_shape_preset_names = _parse_gdn_decode_shape_presets(args.gdn_decode_shape_presets)
                for preset_name in gdn_decode_shape_preset_names:
                    gdn_decode_shape_cases.extend(GDN_DECODE_SHAPE_PRESETS[preset_name])
            if args.gdn_decode_shape_cases is not None:
                gdn_decode_shape_cases.extend(_parse_gdn_decode_shape_cases(args.gdn_decode_shape_cases))
            gdn_decode_shape_cases = _dedupe_gdn_decode_shape_cases(gdn_decode_shape_cases)
        else:
            gdn_value_head_dims = (
                _parse_int_csv("--gdn-value-head-dims", args.gdn_value_head_dims)
                if args.gdn_value_head_dims is not None
                else [args.head_dim if args.gdn_value_head_dim is None else args.gdn_value_head_dim]
            )
            batch_head_cases = (
                _parse_gdn_decode_batch_head_cases(args.gdn_decode_batch_head_cases)
                if args.gdn_decode_batch_head_cases is not None
                else [(args.batch_size, args.num_heads)]
            )
            gdn_decode_shape_cases = [
                (batch_size, num_heads, value_head_dim)
                for batch_size, num_heads in batch_head_cases
                for value_head_dim in gdn_value_head_dims
            ]
        gdn_decode_seed = None if args.gdn_decode_seed < 0 else args.gdn_decode_seed
        gdn_decode_seeds = _parse_gdn_decode_seeds(gdn_decode_seed, args.gdn_decode_seeds)
        value_blocks = _resolve_gdn_decode_value_blocks(args.gdn_decode_value_blocks, gdn_decode_shape_preset_names)
        print(f"gdn_decode_value_blocks={','.join(str(value_block) for value_block in value_blocks)}")
        if not args.gdn_decode_compare_only:
            print(
                "gdn_decode_profile,device_name,strategy,batch,heads,key_head_dim,value_head_dim,value_block,"
                "single_min_elements,dtype,timing_repeats,candidate_ms,max_abs_diff"
            )
        device_name = "" if xpu_info is None else xpu_info.name
        if args.gdn_decode_auto_compare:
            print(
                "gdn_decode_auto_compare,device_name,batch,heads,key_head_dim,value_head_dim,value_block,"
                "auto_strategy,single_ms,tiled_ms,auto_ms,best_explicit_strategy,best_explicit_ms,auto_minus_best_ms,"
                "auto_speed_ratio,max_abs_diff"
            )
        if args.gdn_decode_default_compare:
            print(
                "gdn_decode_default_compare,device_name,batch,heads,key_head_dim,value_head_dim,default_value_block,"
                "default_strategy,single_ms,tiled_ms,default_ms,best_explicit_strategy,best_explicit_ms,"
                "default_minus_best_ms,default_speed_ratio,max_abs_diff"
            )
        if args.gdn_decode_default_block_compare:
            print(
                "gdn_decode_default_block_compare,device_name,batch,heads,key_head_dim,value_head_dim,"
                "default_value_block,default_strategy,forced_value_block,forced_strategy,default_ms,forced_ms,"
                "default_best_ms,forced_minus_default_ms,forced_over_default_ratio,default_speed_ratio_vs_forced,"
                "max_abs_diff"
            )
        for case_idx, (batch_size, num_heads, value_head_dim) in enumerate(gdn_decode_shape_cases):
            case_seeds = [
                None if seed is None else seed + case_idx * 1000
                for seed in gdn_decode_seeds
            ]
            _run_gated_delta_decode_profile_case(
                device_name=device_name,
                batch_size=batch_size,
                num_heads=num_heads,
                key_head_dim=args.head_dim,
                value_head_dim=value_head_dim,
                dtype=dtype,
                value_blocks=value_blocks,
                single_min_elements=args.gdn_decode_single_min_elements,
                warmup=args.warmup,
                iters=args.iters,
                timing_repeats=args.gdn_decode_timing_repeats,
                seeds=case_seeds,
                auto_compare=args.gdn_decode_auto_compare,
                default_compare=args.gdn_decode_default_compare,
                default_block_compare=args.gdn_decode_default_block_compare,
                compare_only=args.gdn_decode_compare_only,
            )
    if not args.arc_int4_only and not args.gdn_decode_only:
        rmsnorm_baseline_ms, rmsnorm_fused_ms, rmsnorm_diff = _benchmark_rmsnorm(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        qk_baseline_ms, qk_fused_ms, qk_diff = _benchmark_qk_norm_rotary(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            rotary_dim=rotary_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        repeat_kv_ms = _benchmark_repeat_kv(
            batch_size=args.batch_size,
            seq_len=kv_len,
            num_kv_heads=args.num_kv_heads,
            num_key_value_groups=num_key_value_groups,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        sdpa_baseline_ms, sdpa_gqa_ms, sdpa_diff = _benchmark_sdpa_gqa_prefill(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        gqa_baseline_ms, gqa_grouped_ms, gqa_diff = _benchmark_grouped_attention(
            batch_size=args.batch_size,
            query_len=args.seq_len,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        decode_baseline_ms, decode_gqa_ms, decode_gqa_diff = _benchmark_gqa_decode_fused(
            batch_size=args.batch_size,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        decode_gate_3d_ms, decode_gate_4d_ms, decode_gate_diff = _benchmark_gqa_decode_gate_layouts(
            batch_size=args.batch_size,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        paged_decode_gate_3d_ms, paged_decode_gate_4d_ms, paged_decode_gate_diff = _benchmark_paged_gqa_decode_gate_layouts(
            batch_size=args.batch_size,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=32,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        decode_sdpa_baseline_ms, decode_sdpa_gqa_ms, decode_sdpa_gqa_diff = _benchmark_sdpa_gqa_decode_full_visible(
            batch_size=args.batch_size,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        decode_variable_baseline_ms, decode_variable_gqa_ms, decode_variable_gqa_diff = _benchmark_sdpa_gqa_decode_variable_visible(
            batch_size=args.batch_size,
            key_len=kv_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        router_baseline_ms, router_fused_ms, router_diff = _benchmark_router(
            tokens=tokens,
            num_experts=args.experts,
            top_k=effective_top_k,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        int4_baseline_ms, int4_candidate_ms, int4_diff = _benchmark_xpu_int4_linear(
            tokens=args.int4_m,
            in_features=int4_k,
            out_features=int4_n,
            group_size=args.int4_group_size,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        flashqla_gdn_profile: tuple[float, float, float, float] | None = None
        if args.seq_len > 1 and args.seq_len % 64 == 0 and hasattr(torch.ops.anna, "flashqla_gated_delta_prefill"):
            flashqla_gdn_profile = _benchmark_flashqla_gdn_prefill(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )

        benchmark_rows: list[dict[str, str]] = []
        _append_benchmark_row(
            benchmark_rows,
            op="rmsnorm",
            baseline_ms=rmsnorm_baseline_ms,
            fused_ms=rmsnorm_fused_ms,
            speedup=_format_speedup(rmsnorm_baseline_ms, rmsnorm_fused_ms),
            max_abs_diff=rmsnorm_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="qk_norm_rotary",
            baseline_ms=qk_baseline_ms,
            fused_ms=qk_fused_ms,
            speedup=_format_speedup(qk_baseline_ms, qk_fused_ms),
            max_abs_diff=qk_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="repeat_kv_materialize",
            baseline_ms=None,
            fused_ms=repeat_kv_ms,
            speedup="-",
            max_abs_diff=None,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="sdpa_gqa_prefill",
            baseline_ms=sdpa_baseline_ms,
            fused_ms=sdpa_gqa_ms,
            speedup=_format_speedup(sdpa_baseline_ms, sdpa_gqa_ms),
            max_abs_diff=sdpa_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="grouped_query_attention_decode",
            baseline_ms=gqa_baseline_ms,
            fused_ms=gqa_grouped_ms,
            speedup=_format_speedup(gqa_baseline_ms, gqa_grouped_ms),
            max_abs_diff=gqa_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="sdpa_gqa_decode_full_visible",
            baseline_ms=decode_sdpa_baseline_ms,
            fused_ms=decode_sdpa_gqa_ms,
            speedup=_format_speedup(decode_sdpa_baseline_ms, decode_sdpa_gqa_ms),
            max_abs_diff=decode_sdpa_gqa_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="sdpa_gqa_decode_variable_visible",
            baseline_ms=decode_variable_baseline_ms,
            fused_ms=decode_variable_gqa_ms,
            speedup=_format_speedup(decode_variable_baseline_ms, decode_variable_gqa_ms),
            max_abs_diff=decode_variable_gqa_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="gqa_decode_fused_proto",
            baseline_ms=decode_baseline_ms,
            fused_ms=decode_gqa_ms,
            speedup=_format_speedup(decode_baseline_ms, decode_gqa_ms),
            max_abs_diff=decode_gqa_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="gqa_decode_gate_3d_contiguous_vs_4d_query_layout",
            baseline_ms=decode_gate_3d_ms,
            fused_ms=decode_gate_4d_ms,
            speedup=_format_speedup(decode_gate_3d_ms, decode_gate_4d_ms),
            max_abs_diff=decode_gate_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="paged_gqa_decode_gate_3d_contiguous_vs_4d_query_layout",
            baseline_ms=paged_decode_gate_3d_ms,
            fused_ms=paged_decode_gate_4d_ms,
            speedup=_format_speedup(paged_decode_gate_3d_ms, paged_decode_gate_4d_ms),
            max_abs_diff=paged_decode_gate_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op="moe_router",
            baseline_ms=router_baseline_ms,
            fused_ms=router_fused_ms,
            speedup=_format_speedup(router_baseline_ms, router_fused_ms),
            max_abs_diff=router_diff,
        )
        _append_benchmark_row(
            benchmark_rows,
            op=f"xpu_int4_linear_m{args.int4_m}_k{int4_k}_n{int4_n}_g{args.int4_group_size}",
            baseline_ms=int4_baseline_ms,
            fused_ms=int4_candidate_ms,
            speedup=_format_speedup(int4_baseline_ms, int4_candidate_ms),
            max_abs_diff=int4_diff,
        )
        if flashqla_gdn_profile is not None:
            current_ms, flashqla_ms, reference_ms, diff = flashqla_gdn_profile
            _append_benchmark_row(
                benchmark_rows,
                op="xpu_gdn_prefill_current_vs_public_entry",
                baseline_ms=current_ms,
                fused_ms=flashqla_ms,
                speedup=_format_speedup(current_ms, flashqla_ms),
                max_abs_diff=diff,
            )
            _append_benchmark_row(
                benchmark_rows,
                op="flashqla_gdn_prefill_reference",
                baseline_ms=None,
                fused_ms=reference_ms,
                speedup="-",
                max_abs_diff=None,
            )

        print("op,baseline_ms,fused_ms,speedup,max_abs_diff")
        print(
            f"rmsnorm,{rmsnorm_baseline_ms:.4f},{rmsnorm_fused_ms:.4f},"
            f"{_format_speedup(rmsnorm_baseline_ms, rmsnorm_fused_ms)},{rmsnorm_diff:.6f}"
        )
        print(
            f"qk_norm_rotary,{qk_baseline_ms:.4f},{qk_fused_ms:.4f},"
            f"{_format_speedup(qk_baseline_ms, qk_fused_ms)},{qk_diff:.6f}"
        )
        print(f"repeat_kv_materialize,-,{repeat_kv_ms:.4f},-,-")
        print(
            f"sdpa_gqa_prefill,{sdpa_baseline_ms:.4f},{sdpa_gqa_ms:.4f},"
            f"{_format_speedup(sdpa_baseline_ms, sdpa_gqa_ms)},{sdpa_diff:.6f}"
        )
        print(
            f"grouped_query_attention_decode,{gqa_baseline_ms:.4f},{gqa_grouped_ms:.4f},"
            f"{_format_speedup(gqa_baseline_ms, gqa_grouped_ms)},{gqa_diff:.6f}"
        )
        print(
            f"sdpa_gqa_decode_full_visible,{decode_sdpa_baseline_ms:.4f},{decode_sdpa_gqa_ms:.4f},"
            f"{_format_speedup(decode_sdpa_baseline_ms, decode_sdpa_gqa_ms)},{decode_sdpa_gqa_diff:.6f}"
        )
        print(
            f"sdpa_gqa_decode_variable_visible,{decode_variable_baseline_ms:.4f},{decode_variable_gqa_ms:.4f},"
            f"{_format_speedup(decode_variable_baseline_ms, decode_variable_gqa_ms)},{decode_variable_gqa_diff:.6f}"
        )
        print(
            f"gqa_decode_fused_proto,{decode_baseline_ms:.4f},{decode_gqa_ms:.4f},"
            f"{_format_speedup(decode_baseline_ms, decode_gqa_ms)},{decode_gqa_diff:.6f}"
        )
        print(
            f"gqa_decode_gate_3d_contiguous_vs_4d_query_layout,{decode_gate_3d_ms:.4f},{decode_gate_4d_ms:.4f},"
            f"{_format_speedup(decode_gate_3d_ms, decode_gate_4d_ms)},{decode_gate_diff:.6f}"
        )
        print(
            f"paged_gqa_decode_gate_3d_contiguous_vs_4d_query_layout,{paged_decode_gate_3d_ms:.4f},{paged_decode_gate_4d_ms:.4f},"
            f"{_format_speedup(paged_decode_gate_3d_ms, paged_decode_gate_4d_ms)},{paged_decode_gate_diff:.6f}"
        )
        print(
            f"moe_router,{router_baseline_ms:.4f},{router_fused_ms:.4f},"
            f"{_format_speedup(router_baseline_ms, router_fused_ms)},{router_diff:.6f}"
        )
        print(
            f"xpu_int4_linear_m{args.int4_m}_k{int4_k}_n{int4_n}_g{args.int4_group_size},"
            f"{int4_baseline_ms:.4f},{int4_candidate_ms:.4f},"
            f"{_format_speedup(int4_baseline_ms, int4_candidate_ms)},{int4_diff:.6f}"
        )
        if flashqla_gdn_profile is not None:
            current_ms, flashqla_ms, reference_ms, diff = flashqla_gdn_profile
            print(
                f"xpu_gdn_prefill_current_vs_public_entry,{current_ms:.4f},{flashqla_ms:.4f},"
                f"{_format_speedup(current_ms, flashqla_ms)},{diff:.6f}"
            )
            print(f"flashqla_gdn_prefill_reference,-,{reference_ms:.4f},-,-")
        if args.csv_output is not None:
            _write_benchmark_csv(args.csv_output, benchmark_rows)
            print(f"csv_output={args.csv_output}")
    if args.arc_profile:
        print(
            "arc_int4_profile,device_name,strategy,M,K,N,group_size,dtype,"
            "baseline_ms,candidate_ms,speedup,max_abs_diff"
        )
        arc_shapes = [
            (1, args.hidden_size, args.hidden_size),
            (2, args.hidden_size, args.hidden_size),
            (4, args.hidden_size, args.hidden_size),
            (8, args.hidden_size, args.hidden_size),
            (1, args.hidden_size, args.hidden_size * 4),
            (1, args.hidden_size * 4, args.hidden_size),
        ]
        for m, k, n in arc_shapes:
            baseline_ms, candidate_ms, diff = _benchmark_xpu_int4_linear(
                tokens=m,
                in_features=k,
                out_features=n,
                group_size=args.int4_group_size,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                strategy="auto",
            )
            device_name = "" if xpu_info is None else xpu_info.name
            print(
                f"arc_int4_profile,{device_name},auto,{m},{k},{n},{args.int4_group_size},{dtype},"
                f"{baseline_ms:.4f},{candidate_ms:.4f},{_format_speedup(baseline_ms, candidate_ms)},{diff:.6f}"
            )
        print("arc_lm_head_int4_topk_profile,device_name,local_size,M,K,N,top_k,group_size,dtype,baseline_ms,candidate_ms,speedup,max_abs_diff")
        for local_size in (8, 16, 32, 64):
            baseline_ms, candidate_ms, diff = _benchmark_lm_head_int4_topk(
                tokens=args.int4_m,
                in_features=int4_k,
                vocab_size=lm_head_vocab_size,
                top_k=max(1, args.top_k),
                group_size=args.int4_group_size,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                local_size=local_size,
            )
            device_name = "" if xpu_info is None else xpu_info.name
            print(
                f"arc_lm_head_int4_topk_profile,{device_name},{local_size},{args.int4_m},{int4_k},{lm_head_vocab_size},"
                f"{max(1, args.top_k)},{args.int4_group_size},{dtype},"
                f"{baseline_ms:.4f},{candidate_ms:.4f},{_format_speedup(baseline_ms, candidate_ms)},{diff:.6f}"
            )
        print(
            "arc_moe_grouped_int4_mlp_profile,device_name,gate_local_size,down_local_size,"
            "tokens_per_expert,experts,hidden_size,intermediate_size,group_size,dtype,"
            "baseline_ms,candidate_ms,speedup,max_abs_diff"
        )
        moe_hidden_size = int4_k
        moe_intermediate_size = int4_n if int4_n != int4_k else int4_k * 2
        for gate_local_size in (64, 128, 256):
            for down_local_size in (64, 128, 256):
                baseline_ms, candidate_ms, diff = _benchmark_moe_grouped_int4_mlp(
                    tokens_per_expert=max(1, args.int4_m),
                    num_experts=max(1, min(args.experts, 8)),
                    hidden_size=moe_hidden_size,
                    intermediate_size=moe_intermediate_size,
                    group_size=args.int4_group_size,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    gate_local_size=gate_local_size,
                    down_local_size=down_local_size,
                )
                device_name = "" if xpu_info is None else xpu_info.name
                print(
                    f"arc_moe_grouped_int4_mlp_profile,{device_name},{gate_local_size},{down_local_size},"
                    f"{max(1, args.int4_m)},{max(1, min(args.experts, 8))},{moe_hidden_size},{moe_intermediate_size},"
                    f"{args.int4_group_size},{dtype},{baseline_ms:.4f},{candidate_ms:.4f},"
                    f"{_format_speedup(baseline_ms, candidate_ms)},{diff:.6f}"
                )

    print("next_paths")
    print("single-token variable-visible decode now maps to native masked GQA; multi-token masked decode remains the main full-attention gap.")
    print("expert execution still pays per-expert launches after assignment compaction and router fusion.")
    print("MoE assignment compaction is in place; the next material win is batched expert execution or packed expert weights.")


if __name__ == "__main__":
    main()
