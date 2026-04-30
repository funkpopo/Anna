from __future__ import annotations

import argparse
import csv
import contextlib
import os
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


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
        "--arc-int4-gemv-local-sizes",
        type=_parse_int_list,
        default=(32, 64, 128),
        help="Comma-separated ANNA_XPU_INT4_GEMV_LOCAL_SIZE values for Arc int4 sycl profile rows.",
    )
    parser.add_argument(
        "--arc-int4-gemv-output-tiles",
        type=_parse_int_list,
        default=(1,),
        help="Comma-separated ANNA_XPU_INT4_GEMV_OUTPUT_TILE values for M=1,K=N=4096 sycl profile rows.",
    )
    parser.add_argument(
        "--arc-int4-gemv-kernels",
        type=str,
        default="wg",
        help="Comma-separated experimental GEMV kernel modes for sycl rows: wg or subgroup.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Write the general hotspot benchmark rows to this CSV file.",
    )
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
    xpu_info = inspect_xpu_device(torch.device("xpu"))
    effective_top_k = max(1, min(args.top_k, args.experts))

    int4_k = args.hidden_size if args.int4_k is None else args.int4_k
    int4_n = args.hidden_size if args.int4_n is None else args.int4_n
    lm_head_vocab_size = int4_n if args.lm_head_vocab_size is None else args.lm_head_vocab_size

    print(
        f"shape batch={args.batch_size} seq={args.seq_len} hidden={args.hidden_size} "
        f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} rotary_dim={rotary_dim} kv_len={kv_len} "
        f"dtype={dtype}"
    )
    if xpu_info is not None:
        print(f"device_name={xpu_info.name}")
        print(f"device_index={xpu_info.device_index}")
    if args.arc_int4_only and not args.arc_profile:
        raise ValueError("--arc-int4-only requires --arc-profile")
    arc_int4_gemv_kernels = tuple(item.strip() for item in args.arc_int4_gemv_kernels.split(",") if item.strip())
    if not arc_int4_gemv_kernels:
        raise ValueError("--arc-int4-gemv-kernels must contain at least one mode")
    unsupported_kernels = sorted(set(arc_int4_gemv_kernels) - {"wg", "subgroup"})
    if unsupported_kernels:
        raise ValueError(f"Unsupported --arc-int4-gemv-kernels values: {unsupported_kernels}")
    if not args.arc_int4_only:
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
                op="flashqla_gdn_prefill_current_vs_intel_flashqla",
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
                f"flashqla_gdn_prefill_current_vs_intel_flashqla,{current_ms:.4f},{flashqla_ms:.4f},"
                f"{_format_speedup(current_ms, flashqla_ms)},{diff:.6f}"
            )
            print(f"flashqla_gdn_prefill_reference,-,{reference_ms:.4f},-,-")
        if args.csv_output is not None:
            _write_benchmark_csv(args.csv_output, benchmark_rows)
            print(f"csv_output={args.csv_output}")
    if args.arc_profile:
        print(
            "arc_int4_profile,device_name,strategy,M,K,N,group_size,dtype,gemv_kernel,gemv_local_size,gemv_output_tile,"
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
            strategies = ("auto", "sycl") if m <= 4 else ("auto",)
            for strategy in strategies:
                gemv_kernels = arc_int4_gemv_kernels if strategy == "sycl" and m == 1 else ("wg",)
                gemv_local_sizes = args.arc_int4_gemv_local_sizes if strategy == "sycl" else (0,)
                for gemv_kernel in gemv_kernels:
                    gemv_output_tiles = (
                        args.arc_int4_gemv_output_tiles
                        if strategy == "sycl" and m == 1 and k == 4096 and n == 4096
                        else (1,)
                    )
                    for gemv_local_size in gemv_local_sizes:
                        for gemv_output_tile in gemv_output_tiles:
                            if gemv_kernel == "subgroup" and gemv_output_tile > 4:
                                continue
                            env_updates = {}
                            if strategy == "sycl":
                                env_updates = {
                                    "ANNA_XPU_INT4_GEMV_KERNEL": gemv_kernel,
                                    "ANNA_XPU_INT4_GEMV_LOCAL_SIZE": str(gemv_local_size),
                                    "ANNA_XPU_INT4_GEMV_OUTPUT_TILE": str(gemv_output_tile),
                                }
                            with _temporary_env(env_updates):
                                baseline_ms, candidate_ms, diff = _benchmark_xpu_int4_linear(
                                    tokens=m,
                                    in_features=k,
                                    out_features=n,
                                    group_size=args.int4_group_size,
                                    dtype=dtype,
                                    warmup=args.warmup,
                                    iters=args.iters,
                                    strategy=strategy,
                                )
                            device_name = "" if xpu_info is None else xpu_info.name
                            print(
                                f"arc_int4_profile,{device_name},{strategy},{m},{k},{n},{args.int4_group_size},{dtype},"
                                f"{gemv_kernel},{gemv_local_size},{gemv_output_tile},"
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
