from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.ops import apply_rotary_pos_emb, repeat_kv


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


def _benchmark_router(
    *,
    tokens: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> float:
    router_logits = torch.randn(tokens, num_experts, device="xpu", dtype=dtype)

    def route():
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        return torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

    return _time_op(route, warmup=warmup, iters=iters)


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
        seq_len=args.seq_len,
        num_kv_heads=args.num_kv_heads,
        num_key_value_groups=num_key_value_groups,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
    )
    router_ms = _benchmark_router(
        tokens=tokens,
        num_experts=args.experts,
        top_k=args.top_k,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        f"shape batch={args.batch_size} seq={args.seq_len} hidden={args.hidden_size} "
        f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} rotary_dim={rotary_dim} "
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
    print(f"repeat_kv,-,{repeat_kv_ms:.4f},-,-")
    print(f"moe_router_select,-,{router_ms:.4f},-,-")

    print("next_paths")
    print("repeat_kv grows with cache length and still pays a full materialization cost per call.")
    print("moe_router_select is the strongest non-matmul decode candidate once sparse MoE is enabled.")
    print("full-attention score/softmax remains matmul-dominated; pursue it after routing and cache-layout work.")


if __name__ == "__main__":
    main()
