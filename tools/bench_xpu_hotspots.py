from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.ops import apply_rotary_pos_emb, grouped_query_attention, repeat_kv


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
    args = parser.parse_args()

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
    router_baseline_ms, router_fused_ms, router_diff = _benchmark_router(
        tokens=tokens,
        num_experts=args.experts,
        top_k=args.top_k,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        f"shape batch={args.batch_size} seq={args.seq_len} hidden={args.hidden_size} "
        f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} rotary_dim={rotary_dim} kv_len={kv_len} "
        f"dtype={dtype}"
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
        f"moe_router,{router_baseline_ms:.4f},{router_fused_ms:.4f},"
        f"{_format_speedup(router_baseline_ms, router_fused_ms)},{router_diff:.6f}"
    )

    print("next_paths")
    print("visible-cache gather and mask application are the next full-attention costs after repeat_kv removal.")
    print("expert execution still pays per-expert launches and per-expert index gathering after router fusion.")
    print("full-attention score/softmax remains matmul-dominated; pursue it after routing and cache-layout work.")


if __name__ == "__main__":
    main()
