import pytest
import torch
import os
import warnings

import anna.model.ops as model_ops
import anna.model.fused_ops as fused_ops
from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.gemma4_config import Gemma4RopeParameters, Gemma4TextConfig
from anna.model.gemma4_text_model import Gemma4TextRotaryEmbedding
from anna.model.ops import (
    Qwen3Attention,
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    Qwen3TextRotaryEmbedding,
    apply_rotary_pos_emb,
    grouped_query_attention,
    rotate_half,
    torch_causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)
from anna.model.quantization import XPUInt4Linear
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.model.turboquant import dequantize_turboquant_values, quantize_turboquant_values


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


def test_int4_fused_op_availability_respects_disable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK", "1")
    monkeypatch.setattr(fused_ops, "_lm_head_int4_topk_op", lambda: object())
    assert fused_ops.lm_head_int4_topk_fused_is_available() is False

    monkeypatch.setenv("ANNA_XPU_DISABLE_MOE_GROUPED_INT4", "true")
    monkeypatch.setattr(fused_ops, "_moe_grouped_int4_mlp_op", lambda: object())
    assert fused_ops.moe_grouped_int4_mlp_fused_is_available() is False


def test_flashqla_gated_delta_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_ops, "_flashqla_gated_delta_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    query = torch.empty(1, 64, 1, 8, dtype=torch.float16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    g = torch.empty(1, 64, 1, dtype=torch.float32)
    beta = torch.empty_like(g)
    state = torch.empty(1, 1, 8, 8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_gated_delta_prefill(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            state=state,
        )


def _reference_rmsnorm_ex(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    add_unit_offset: bool,
) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    scale = weight.float() + (1.0 if add_unit_offset else 0.0)
    output = output * scale
    return output.to(dtype=x.dtype)


def _apply_rotary_only(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


def _reference_rope_cos_sin(
    *,
    batch_size: int,
    seq_len: int,
    rotary_dim: int,
    device: str,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    sin = emb.sin().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def test_apply_rotary_pos_emb_preserves_input_dtype_with_float_trig() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 5, 16, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 5, 16, dtype=torch.bfloat16)
    cos = torch.randn(2, 5, 8, dtype=torch.float32)
    sin = torch.randn(2, 5, 8, dtype=torch.float32)

    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

    rotary_dim = cos.shape[-1]
    reference_query = torch.cat(
        [
            ((query[..., :rotary_dim] * cos.unsqueeze(1)) + (rotate_half(query[..., :rotary_dim]) * sin.unsqueeze(1))).to(
                dtype=query.dtype
            ),
            query[..., rotary_dim:],
        ],
        dim=-1,
    )
    reference_key = torch.cat(
        [
            ((key[..., :rotary_dim] * cos.unsqueeze(1)) + (rotate_half(key[..., :rotary_dim]) * sin.unsqueeze(1))).to(
                dtype=key.dtype
            ),
            key[..., rotary_dim:],
        ],
        dim=-1,
    )

    assert rotated_query.dtype == query.dtype
    assert rotated_key.dtype == key.dtype
    assert torch.equal(rotated_query, reference_query)
    assert torch.equal(rotated_key, reference_key)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the rotary dtype test")
def test_qwen3_rotary_embedding_xpu_keeps_trig_float32() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["full_attention"],
    )
    rotary = Qwen3TextRotaryEmbedding(config).to("xpu")
    hidden_states = torch.randn(2, 5, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    position_ids = torch.arange(hidden_states.shape[1], device="xpu").view(1, -1).expand(hidden_states.shape[0], -1)

    cos, sin = rotary(hidden_states, position_ids)

    assert cos.device.type == "xpu"
    assert sin.device.type == "xpu"
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the rotary dtype test")
def test_gemma4_rotary_embedding_xpu_keeps_trig_float32() -> None:
    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": Gemma4RopeParameters(rope_type="default", rope_theta=10_000.0, partial_rotary_factor=1.0),
            "full_attention": Gemma4RopeParameters(rope_type="default", rope_theta=1_000_000.0, partial_rotary_factor=0.25),
        },
    )
    rotary = Gemma4TextRotaryEmbedding(config).to("xpu")
    hidden_states = torch.randn(2, 5, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    position_ids = torch.arange(hidden_states.shape[1], device="xpu").view(1, -1).expand(hidden_states.shape[0], -1)

    for layer_type in ("sliding_attention", "full_attention"):
        cos, sin = rotary(hidden_states, position_ids, layer_type)
        assert cos.device.type == "xpu"
        assert sin.device.type == "xpu"
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32


def _reference_moe_router(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    normalize_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if normalize_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    usage = torch.bincount(selected_experts.reshape(-1), minlength=router_logits.shape[-1])
    return routing_weights, selected_experts, usage


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_lm_head_topk_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "lm_head_topk_fused"):
        pytest.skip("Anna fused-op library is not built with lm_head_topk_fused")

    torch.manual_seed(0)
    device = "xpu"
    hidden_states = torch.randn(2, 3, 16, device=device, dtype=torch.float16)
    weight = torch.randn(31, 16, device=device, dtype=torch.float16)

    values, indices = torch.ops.anna.lm_head_topk_fused(hidden_states, weight, 5)
    reference_values, reference_indices = torch.topk(hidden_states @ weight.t(), k=5, dim=-1)

    assert values.shape == (2, 3, 5)
    assert indices.shape == (2, 3, 5)
    assert indices.dtype == torch.long
    assert torch.equal(indices.cpu(), reference_indices.cpu())
    assert torch.allclose(values.cpu(), reference_values.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_lm_head_int4_topk_fused_xpu_matches_quantized_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "lm_head_int4_topk_fused"):
        pytest.skip("Anna fused-op library is not built with lm_head_int4_topk_fused")

    torch.manual_seed(0)
    dense = torch.nn.Linear(16, 31, bias=False, dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.float16, device="xpu")
    hidden_states = torch.randn(2, 3, 16, device="xpu", dtype=torch.float16)

    values, indices = torch.ops.anna.lm_head_int4_topk_fused(
        hidden_states,
        quantized.qweight,
        quantized.qscale,
        quantized.qzeros,
        quantized.group_size,
        quantized.in_features,
        5,
    )
    reference_values, reference_indices = torch.topk(quantized(hidden_states), k=5, dim=-1)

    assert values.shape == (2, 3, 5)
    assert indices.shape == (2, 3, 5)
    assert indices.dtype == torch.long
    assert torch.equal(indices.cpu(), reference_indices.cpu())
    assert torch.allclose(values.cpu(), reference_values.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_lm_head_int4_topk_fused_xpu_accepts_lm_head_topk_layout() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "lm_head_int4_topk_fused"):
        pytest.skip("Anna fused-op library is not built with lm_head_int4_topk_fused")

    torch.manual_seed(8)
    dense = torch.nn.Linear(64, 96, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.float16, device="xpu")
    quantized.prepare_lm_head_topk_layout()
    hidden_states = torch.randn(2, 64, device="xpu", dtype=torch.float16)

    values, indices = torch.ops.anna.lm_head_int4_topk_fused(
        hidden_states,
        quantized.lm_head_qweight,
        quantized.lm_head_qscale,
        quantized.lm_head_qzeros,
        quantized.group_size,
        quantized.in_features,
        5,
    )
    reference_values, reference_indices = torch.topk(quantized(hidden_states), k=5, dim=-1)

    torch.xpu.synchronize()
    assert torch.equal(indices.cpu(), reference_indices.cpu())
    assert torch.allclose(values.cpu(), reference_values.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_lm_head_int4_topk_fused_xpu_respects_local_size_override() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "lm_head_int4_topk_fused"):
        pytest.skip("Anna fused-op library is not built with lm_head_int4_topk_fused")

    previous = os.environ.get("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE")
    os.environ["ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"] = "16"
    try:
        torch.manual_seed(0)
        dense = torch.nn.Linear(16, 31, bias=False, dtype=torch.float32)
        quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.float16, device="xpu")
        hidden_states = torch.randn(2, 3, 16, device="xpu", dtype=torch.float16)

        values, indices = torch.ops.anna.lm_head_int4_topk_fused(
            hidden_states,
            quantized.qweight,
            quantized.qscale,
            quantized.qzeros,
            quantized.group_size,
            quantized.in_features,
            5,
        )
        reference_values, reference_indices = torch.topk(quantized(hidden_states), k=5, dim=-1)
    finally:
        if previous is None:
            os.environ.pop("ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE", None)
        else:
            os.environ["ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE"] = previous

    assert torch.equal(indices.cpu(), reference_indices.cpu())
    assert torch.allclose(values.cpu(), reference_values.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_lm_head_int4_topk_fused_xpu_blocked_path_matches_quantized_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "lm_head_int4_topk_fused"):
        pytest.skip("Anna fused-op library is not built with lm_head_int4_topk_fused")

    torch.manual_seed(5)
    monkeypatch.setenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_TOPK_THRESHOLD", "64")
    monkeypatch.setenv("ANNA_XPU_INT4_LM_HEAD_BLOCK_SIZE", "64")
    dense = torch.nn.Linear(96, 192, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.float16, device="xpu")
    hidden_states = torch.randn(1, 96, device="xpu", dtype=torch.float16)

    values, indices = torch.ops.anna.lm_head_int4_topk_fused(
        hidden_states,
        quantized.qweight,
        quantized.qscale,
        quantized.qzeros,
        quantized.group_size,
        quantized.in_features,
        5,
    )
    reference_values, reference_indices = torch.topk(quantized(hidden_states), k=5, dim=-1)

    torch.xpu.synchronize()
    assert torch.equal(indices.cpu(), reference_indices.cpu())
    assert torch.allclose(values.cpu(), reference_values.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_xpu_int4_linear_auto_strategy_does_not_use_gemv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(6)
    dense = torch.nn.Linear(4096, 4096, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=128, compute_dtype=torch.bfloat16, device="xpu")
    hidden_states = torch.randn(1, 4096, device="xpu", dtype=torch.bfloat16)

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "torch")
    reference = quantized(hidden_states)

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    output = quantized(hidden_states)

    torch.xpu.synchronize()
    assert torch.allclose(output.cpu(), reference.cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_xpu_int4_linear_auto_strategy_propagates_torch_int4pack_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    dense = torch.nn.Linear(32, 32, bias=False, device="xpu", dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.bfloat16, device="xpu")
    hidden_states = torch.randn(1, 32, device="xpu", dtype=torch.bfloat16)

    def _raise_int4pack_error(_x_padded: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("int4pack failed")

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    monkeypatch.setattr(quantized, "_forward_torch_xpu_int4", _raise_int4pack_error)

    with pytest.raises(RuntimeError, match="int4pack failed"):
        quantized(hidden_states)


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


def _decode_gate_query_layout(gate: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    batch_size, seq_len, flat_dim = gate.shape
    if flat_dim != num_heads * head_dim:
        raise ValueError(f"gate last dim {flat_dim} must equal num_heads * head_dim ({num_heads * head_dim})")
    return gate.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_causal_conv1d_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "causal_conv1d_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    hidden_states = torch.randn(2, 6, 7, device=device, dtype=torch.float16)
    conv_state = torch.randn(2, 6, 4, device=device, dtype=torch.float16)
    weight = torch.randn(6, 4, device=device, dtype=torch.float16)
    bias = torch.randn(6, device=device, dtype=torch.float16)

    output, fused_state = torch.ops.anna.causal_conv1d_fused(hidden_states, conv_state, weight, bias)

    reference_state = conv_state.clone()
    reference = torch_causal_conv1d_update(hidden_states, reference_state, weight, bias)

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(fused_state.float().cpu(), reference_state.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(2, 8, 1, 32, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 19, 32, device=device, dtype=torch.bfloat16)
    value = torch.randn(2, 2, 19, 32, device=device, dtype=torch.bfloat16)
    visible_lengths = torch.tensor([19, 13], device=device, dtype=torch.long)

    fused_output = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, 32**-0.5)
    visible_mask = torch.arange(key.shape[2], device=device)[None, :] < visible_lengths[:, None]
    reference = grouped_query_attention(
        query,
        key,
        value,
        scaling=32**-0.5,
        visible_mask=visible_mask,
    )

    torch.xpu.synchronize()
    assert torch.allclose(fused_output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_fused_xpu_long_qwen35_shape_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    head_dim = 256
    key_len = 4096
    query = torch.randn(1, 16, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    visible_lengths = torch.tensor([key_len], device=device, dtype=torch.long)

    fused_output = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, head_dim**-0.5)
    visible_mask = torch.arange(key.shape[2], device=device)[None, :] < visible_lengths[:, None]
    reference = grouped_query_attention(
        query,
        key,
        value,
        scaling=head_dim**-0.5,
        visible_mask=visible_mask,
    )

    torch.xpu.synchronize()
    assert torch.allclose(fused_output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_splitkv_fused_xpu_long_qwen35_shape_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_splitkv_fused"):
        pytest.skip("Anna fused-op library is not built with gqa_decode_splitkv_fused")

    torch.manual_seed(19)
    device = "xpu"
    head_dim = 256
    key_len = 4096
    query = torch.randn(1, 16, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)

    fused_output = torch.ops.anna.gqa_decode_splitkv_fused(query, key, value, head_dim**-0.5, 256)
    reference = grouped_query_attention(query, key, value, scaling=head_dim**-0.5)

    torch.xpu.synchronize()
    assert torch.allclose(fused_output.float().cpu(), reference.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_splitkv_fused_xpu_accepts_strided_kv_and_fuses_gate() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_splitkv_fused"):
        pytest.skip("Anna fused-op library is not built with gqa_decode_splitkv_fused")

    torch.manual_seed(20)
    device = "xpu"
    batch_size = 1
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    key_len = 257
    capacity = 512
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    key_storage = torch.empty(batch_size, num_kv_heads, capacity, head_dim, device=device, dtype=torch.bfloat16)
    value_storage = torch.empty_like(key_storage)
    key_storage[:, :, :key_len, :].normal_()
    value_storage[:, :, :key_len, :].normal_()
    key = key_storage[:, :, :key_len, :]
    value = value_storage[:, :, :key_len, :]
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scale = head_dim**-0.5

    assert not key.is_contiguous()
    assert not value.is_contiguous()
    fused = torch.ops.anna.gqa_decode_splitkv_fused(query, key, value, scale, 64, gate_4d)
    reference = grouped_query_attention(query, key, value, scaling=scale) * torch.sigmoid(gate_4d)

    torch.xpu.synchronize()
    assert torch.allclose(fused.float().cpu(), reference.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_splitkv_fused_out_xpu_reuses_workspace_and_accepts_strided_query() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_splitkv_fused_out"):
        pytest.skip("Anna fused-op library is not built with gqa_decode_splitkv_fused_out")

    torch.manual_seed(22)
    device = "xpu"
    batch_size = 1
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    key_len = 193
    block_size = 64
    blocks = (key_len + block_size - 1) // block_size
    query_storage = torch.empty(batch_size, num_heads, 1, head_dim * 2, device=device, dtype=torch.bfloat16)
    query_storage.normal_()
    query = query_storage[..., ::2]
    key_storage = torch.empty(batch_size, num_kv_heads, 256, head_dim, device=device, dtype=torch.bfloat16)
    value_storage = torch.empty_like(key_storage)
    key_storage[:, :, :key_len, :].normal_()
    value_storage[:, :, :key_len, :].normal_()
    key = key_storage[:, :, :key_len, :]
    value = value_storage[:, :, :key_len, :]
    output = torch.empty_like(query)
    partial_stats = torch.empty(batch_size, num_heads, blocks + 2, 2, device=device, dtype=torch.float32)
    partial_values = torch.empty(batch_size, num_heads, blocks + 2, head_dim, device=device, dtype=torch.float32)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scale = head_dim**-0.5

    assert not query.is_contiguous()
    assert not key.is_contiguous()
    returned = torch.ops.anna.gqa_decode_splitkv_fused_out(
        query,
        key,
        value,
        output,
        partial_stats,
        partial_values,
        scale,
        block_size,
        gate_4d,
    )
    reference = grouped_query_attention(query, key, value, scaling=scale) * torch.sigmoid(gate_4d)

    torch.xpu.synchronize()
    assert returned.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("value_bits", [2, 3, 4])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_splitkv_turboquant_fused_out_xpu_matches_dequantized_reference(value_bits: int) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_splitkv_turboquant_fused_out"):
        pytest.skip("Anna fused-op library is not built with gqa_decode_splitkv_turboquant_fused_out")

    torch.manual_seed(24 + value_bits)
    device = torch.device("xpu")
    batch_size = 1
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    quantized_len = 77
    residual_len = 5
    block_size = 64
    total_len = quantized_len + residual_len
    blocks = (total_len + block_size - 1) // block_size
    query_storage = torch.empty(batch_size, num_heads, 1, head_dim * 2, device=device, dtype=torch.bfloat16)
    query_storage.normal_()
    query = query_storage[..., ::2]
    quantized_key = torch.randn(batch_size, num_kv_heads, quantized_len, head_dim, device=device, dtype=torch.bfloat16)
    dense_quantized_value = torch.randn(num_kv_heads, quantized_len, head_dim, device=device, dtype=torch.bfloat16)
    value_state = quantize_turboquant_values(dense_quantized_value, bits=value_bits, group_size=16)
    residual_key = torch.randn(batch_size, num_kv_heads, residual_len, head_dim, device=device, dtype=torch.bfloat16)
    residual_value = torch.randn(batch_size, num_kv_heads, residual_len, head_dim, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    partial_stats = torch.empty(batch_size, num_heads, blocks + 1, 2, device=device, dtype=torch.float32)
    partial_values = torch.empty(batch_size, num_heads, blocks + 1, head_dim, device=device, dtype=torch.float32)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scale = head_dim**-0.5

    returned = torch.ops.anna.gqa_decode_splitkv_turboquant_fused_out(
        query,
        quantized_key,
        value_state.data.unsqueeze(0),
        value_state.scales.unsqueeze(0),
        value_state.zeros.unsqueeze(0),
        residual_key,
        residual_value,
        output,
        partial_stats,
        partial_values,
        scale,
        value_bits,
        value_state.group_size,
        block_size,
        gate_4d,
    )
    dequantized_value = dequantize_turboquant_values(value_state, device=device, dtype=torch.bfloat16).unsqueeze(0)
    reference_key = torch.cat((quantized_key, residual_key), dim=2)
    reference_value = torch.cat((dequantized_value, residual_value), dim=2)
    reference = grouped_query_attention(query, reference_key, reference_value, scaling=scale) * torch.sigmoid(gate_4d)

    torch.xpu.synchronize()
    assert returned.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_fused_xpu_with_gate_matches_sigmoid_fusion() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(1)
    device = "xpu"
    query = torch.randn(2, 8, 1, 32, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 19, 32, device=device, dtype=torch.bfloat16)
    value = torch.randn(2, 2, 19, 32, device=device, dtype=torch.bfloat16)
    visible_lengths = torch.tensor([19, 13], device=device, dtype=torch.long)
    gate = torch.randn(2, 1, 8 * 32, device=device, dtype=torch.bfloat16)

    fused = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, 32**-0.5, gate)
    base = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, 32**-0.5)
    expected = base * torch.sigmoid(gate).view(2, 8, 1, 32)

    torch.xpu.synchronize()
    assert torch.allclose(fused.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_fused_xpu_gate_accepts_equivalent_3d_and_4d_layouts() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(11)
    batch_size = 2
    num_heads = 8
    head_dim = 32
    device = "xpu"
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 2, 23, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 2, 23, head_dim, device=device, dtype=torch.bfloat16)
    visible_lengths = torch.tensor([23, 17], device=device, dtype=torch.long)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)

    assert gate_4d.shape == (batch_size, num_heads, 1, head_dim)
    assert gate_4d.stride(0) == num_heads * head_dim
    assert gate_4d.stride(1) == head_dim
    assert gate_4d.stride(3) == 1

    base = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, head_dim**-0.5)
    fused_3d = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, head_dim**-0.5, gate_3d)
    fused_4d = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, head_dim**-0.5, gate_4d)
    expected = base * torch.sigmoid(gate_4d)

    torch.xpu.synchronize()
    assert torch.allclose(fused_3d.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(fused_4d.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(fused_3d.float().cpu(), fused_4d.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_paged_gqa_decode_fused_xpu_long_qwen35_shape_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "paged_gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    head_dim = 256
    key_len = 4096
    block_size = 32
    query = torch.randn(1, 16, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    key_pages, value_pages, page_table = _pack_paged_kv(key, value, block_size=block_size)
    visible_lengths = torch.tensor([key_len], device=device, dtype=torch.long)

    fused_output = torch.ops.anna.paged_gqa_decode_fused(
        query,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        head_dim**-0.5,
    )
    visible_mask = torch.arange(key.shape[2], device=device)[None, :] < visible_lengths[:, None]
    reference = grouped_query_attention(
        query,
        key,
        value,
        scaling=head_dim**-0.5,
        visible_mask=visible_mask,
    )

    torch.xpu.synchronize()
    assert torch.allclose(fused_output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_paged_gqa_decode_fused_xpu_with_gate_matches_sigmoid_fusion() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "paged_gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(2)
    device = "xpu"
    head_dim = 128
    key_len = 512
    block_size = 32
    query = torch.randn(1, 8, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    key_pages, value_pages, page_table = _pack_paged_kv(key, value, block_size=block_size)
    visible_lengths = torch.tensor([key_len], device=device, dtype=torch.long)
    gate = torch.randn(1, 1, 8 * head_dim, device=device, dtype=torch.bfloat16)
    scale = head_dim**-0.5

    fused = torch.ops.anna.paged_gqa_decode_fused(
        query, key_pages, value_pages, page_table, visible_lengths, scale, gate
    )
    base = torch.ops.anna.paged_gqa_decode_fused(query, key_pages, value_pages, page_table, visible_lengths, scale)
    expected = base * torch.sigmoid(gate).view(1, 8, 1, head_dim)

    torch.xpu.synchronize()
    assert torch.allclose(fused.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_paged_gqa_decode_fused_xpu_gate_accepts_equivalent_3d_and_4d_layouts() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "paged_gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(12)
    batch_size = 2
    num_heads = 8
    head_dim = 64
    key_len = 257
    block_size = 32
    device = "xpu"
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    key_pages, value_pages, page_table = _pack_paged_kv(key, value, block_size=block_size)
    visible_lengths = torch.tensor([key_len, 211], device=device, dtype=torch.long)
    gate_3d = torch.randn(batch_size, 1, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate_4d = _decode_gate_query_layout(gate_3d, num_heads=num_heads, head_dim=head_dim)
    scale = head_dim**-0.5

    assert gate_4d.shape == (batch_size, num_heads, 1, head_dim)
    assert gate_4d.stride(0) == num_heads * head_dim
    assert gate_4d.stride(1) == head_dim
    assert gate_4d.stride(3) == 1

    base = torch.ops.anna.paged_gqa_decode_fused(
        query,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        scale,
    )
    fused_3d = torch.ops.anna.paged_gqa_decode_fused(
        query,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        scale,
        gate_3d,
    )
    fused_4d = torch.ops.anna.paged_gqa_decode_fused(
        query,
        key_pages,
        value_pages,
        page_table,
        visible_lengths,
        scale,
        gate_4d,
    )
    expected = base * torch.sigmoid(gate_4d)

    torch.xpu.synchronize()
    assert torch.allclose(fused_3d.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(fused_4d.float().cpu(), expected.float().cpu(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(fused_3d.float().cpu(), fused_4d.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gqa_decode_fused_xpu_gemma4_full_head_dim_512_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gqa_decode_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    head_dim = 512
    rotary_dim = 128
    key_len = 1024

    # Gemma4 full-attention decode uses RMS-normalized q/k with scaling=1.0.
    # These constants match the full-attention q_norm/k_norm weights in gemma-4-E4B-it.
    query = torch.randn(1, 8, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 2, key_len, head_dim, device=device, dtype=torch.bfloat16)
    query = _reference_rmsnorm_ex(
        query,
        torch.full((head_dim,), 1.0234375, device=device, dtype=torch.float32),
        1e-6,
        add_unit_offset=False,
    )
    key = _reference_rmsnorm_ex(
        key,
        torch.full((head_dim,), 0.06103515625, device=device, dtype=torch.float32),
        1e-6,
        add_unit_offset=False,
    )
    query_cos, query_sin = _reference_rope_cos_sin(
        batch_size=1,
        seq_len=1,
        rotary_dim=rotary_dim,
        device=device,
        rope_theta=1_000_000.0,
        dtype=query.dtype,
    )
    key_cos, key_sin = _reference_rope_cos_sin(
        batch_size=1,
        seq_len=key_len,
        rotary_dim=rotary_dim,
        device=device,
        rope_theta=1_000_000.0,
        dtype=key.dtype,
    )
    query = _apply_rotary_only(query, query_cos, query_sin)
    key = _apply_rotary_only(key, key_cos, key_sin)
    visible_lengths = torch.tensor([key_len], device=device, dtype=torch.long)

    fused_output = torch.ops.anna.gqa_decode_fused(query, key, value, visible_lengths, 1.0)
    visible_mask = torch.arange(key.shape[2], device=device)[None, :] < visible_lengths[:, None]
    reference = grouped_query_attention(
        query,
        key,
        value,
        scaling=1.0,
        visible_mask=visible_mask,
    )

    torch.xpu.synchronize()
    assert torch.allclose(fused_output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_moe_router_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "moe_router_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    router_logits = torch.randn(11, 16, device=device, dtype=torch.bfloat16)

    fused_weights, fused_selected, fused_usage = torch.ops.anna.moe_router_fused(router_logits, 4, True)
    ref_weights, ref_selected, ref_usage = _reference_moe_router(
        router_logits,
        top_k=4,
        normalize_topk_prob=True,
    )

    torch.xpu.synchronize()
    assert torch.allclose(fused_weights.float().cpu(), ref_weights.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.equal(fused_selected.cpu(), ref_selected.cpu())
    assert torch.equal(fused_usage.cpu().to(dtype=torch.long), ref_usage.cpu().to(dtype=torch.long))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_rmsnorm_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "rmsnorm_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    hidden_states = torch.randn(3, 5, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randn(32, device=device, dtype=torch.float32)
    eps = 1e-6

    output = torch.ops.anna.rmsnorm_fused(hidden_states, weight, eps)
    reference = _reference_rmsnorm(hidden_states, weight, eps)

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_fused_ops_register_autograd_fallthrough_for_inference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "rmsnorm_fused"):
        pytest.skip("Anna fused-op library is not built")

    assert torch._C._dispatch_has_kernel_for_dispatch_key("anna::rmsnorm_fused", "Autograd")

    hidden_states = torch.randn(2, 3, 32, device="xpu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, device="xpu", dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        output = torch.ops.anna.rmsnorm_fused(hidden_states, weight, 1e-6)
        torch.xpu.synchronize()

    assert output.requires_grad is False
    assert not any("autograd kernel was not registered" in str(item.message) for item in caught)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qk_norm_rotary_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "qk_norm_rotary_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(2, 8, 5, 32, device=device, dtype=torch.float16)
    key = torch.randn(2, 2, 5, 32, device=device, dtype=torch.float16)
    query_norm_weight = torch.randn(32, device=device, dtype=torch.float32)
    key_norm_weight = torch.randn(32, device=device, dtype=torch.float32)
    cos = torch.randn(2, 5, 16, device=device, dtype=torch.float32)
    sin = torch.randn(2, 5, 16, device=device, dtype=torch.float32)

    fused_query, fused_key = torch.ops.anna.qk_norm_rotary_fused(
        query,
        key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        1e-6,
        1e-5,
    )

    reference_query = _reference_rmsnorm(query, query_norm_weight, 1e-6)
    reference_key = _reference_rmsnorm(key, key_norm_weight, 1e-5)
    reference_query, reference_key = apply_rotary_pos_emb(reference_query, reference_key, cos, sin)

    torch.xpu.synchronize()
    assert torch.allclose(fused_query.float().cpu(), reference_query.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(fused_key.float().cpu(), reference_key.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qk_norm_rotary_fused_ex_xpu_matches_reference_without_unit_offset() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "qk_norm_rotary_fused_ex"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(2, 8, 5, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 5, 128, device=device, dtype=torch.bfloat16)
    query_norm_weight = torch.randn(128, device=device, dtype=torch.float32)
    key_norm_weight = torch.randn(128, device=device, dtype=torch.float32)
    cos = torch.randn(2, 5, 32, device=device, dtype=torch.float32)
    sin = torch.randn(2, 5, 32, device=device, dtype=torch.float32)

    fused_query, fused_key = torch.ops.anna.qk_norm_rotary_fused_ex(
        query,
        key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        1e-6,
        1e-6,
        False,
    )

    reference_query = _reference_rmsnorm_ex(query, query_norm_weight, 1e-6, add_unit_offset=False)
    reference_key = _reference_rmsnorm_ex(key, key_norm_weight, 1e-6, add_unit_offset=False)
    reference_query, reference_key = apply_rotary_pos_emb(reference_query, reference_key, cos, sin)

    torch.xpu.synchronize()
    assert torch.allclose(fused_query.float().cpu(), reference_query.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(fused_key.float().cpu(), reference_key.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qwen3_attention_xpu_fused_norm_rotary_matches_cpu_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "qk_norm_rotary_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["full_attention"],
    )
    attention_cpu = Qwen3Attention(config, 0).eval()
    attention_xpu = Qwen3Attention(config, 0).eval().to("xpu", dtype=torch.bfloat16)
    attention_xpu.load_state_dict(attention_cpu.state_dict())

    rotary_cpu = Qwen3TextRotaryEmbedding(config)
    rotary_xpu = Qwen3TextRotaryEmbedding(config).to("xpu")
    rotary_xpu.load_state_dict(rotary_cpu.state_dict())

    hidden_states_cpu = torch.randn(1, 16, config.hidden_size, dtype=torch.float32)
    hidden_states_xpu = hidden_states_cpu.to("xpu", dtype=torch.bfloat16)
    position_ids_cpu = torch.arange(hidden_states_cpu.shape[1]).view(1, -1)
    position_ids_xpu = position_ids_cpu.to("xpu")

    with torch.no_grad():
        cpu_position_embeddings = rotary_cpu(hidden_states_cpu, position_ids_cpu)
        xpu_position_embeddings = rotary_xpu(hidden_states_xpu, position_ids_xpu)
        output_cpu = attention_cpu(hidden_states_cpu, cpu_position_embeddings)
        output_xpu = attention_xpu(hidden_states_xpu, xpu_position_embeddings)

    torch.xpu.synchronize()
    assert torch.allclose(output_xpu.float().cpu(), output_cpu.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qwen3_attention_xpu_single_token_decode_matches_full_forward() -> None:
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["full_attention"],
    )
    attention = Qwen3Attention(config, 0).eval().to("xpu", dtype=torch.bfloat16)
    rotary = Qwen3TextRotaryEmbedding(config).to("xpu")

    prompt_states = torch.randn(2, 6, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    append_states = torch.randn(2, 1, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    full_states = torch.cat([prompt_states, append_states], dim=1)

    with torch.no_grad():
        full_position_ids = torch.arange(full_states.shape[1], device="xpu").view(1, -1).expand(full_states.shape[0], -1)
        full_embeddings = rotary(full_states, full_position_ids)
        full_output = attention(full_states, full_embeddings)

        cache = Qwen3DynamicCache(config)
        prompt_position_ids = torch.arange(prompt_states.shape[1], device="xpu").view(1, -1).expand(prompt_states.shape[0], -1)
        prompt_embeddings = rotary(prompt_states, prompt_position_ids)
        _ = attention(prompt_states, prompt_embeddings, past_key_values=cache)

        append_position_ids = torch.full((append_states.shape[0], 1), prompt_states.shape[1], device="xpu", dtype=torch.long)
        append_embeddings = rotary(append_states, append_position_ids)
        append_output = attention(append_states, append_embeddings, past_key_values=cache)

    torch.xpu.synchronize()
    assert torch.allclose(append_output.float().cpu(), full_output[:, -1:].float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_materialized_kv_single_token_decode_attention_matches_reference_on_long_qwen35_shape() -> None:
    torch.manual_seed(0)
    device = "xpu"
    head_dim = 256
    key_len = 4096
    query = torch.randn(1, 16, 1, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 4, key_len, head_dim, device=device, dtype=torch.bfloat16)
    visible_lengths = torch.tensor([key_len], device=device, dtype=torch.long)

    output = model_ops.materialized_kv_single_token_decode_attention(
        query,
        key,
        value,
        scaling=head_dim**-0.5,
        num_key_value_groups=4,
        visible_lengths=visible_lengths,
    )
    repeated_key = model_ops.repeat_kv(key, 4)
    repeated_value = model_ops.repeat_kv(value, 4)
    attn_scores = torch.matmul(query, repeated_key.transpose(-1, -2)) * (head_dim**-0.5)
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query.dtype)
    reference = torch.matmul(attn_probs, repeated_value)

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_rmsnorm_fused_ex_xpu_matches_reference_without_unit_offset() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "rmsnorm_fused_ex"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    hidden_states = torch.randn(3, 5, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.float32)
    eps = 1e-6

    output = torch.ops.anna.rmsnorm_fused_ex(hidden_states, weight, eps, False)
    reference = _reference_rmsnorm_ex(hidden_states, weight, eps, add_unit_offset=False)

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), reference.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qwen3_attention_xpu_single_token_decode_uses_paged_kv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    if (
        not maybe_load_gated_delta_library()
        or not hasattr(torch.ops.anna, "qk_norm_rotary_fused")
        or not hasattr(torch.ops.anna, "paged_gqa_decode_fused")
    ):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    calls: list[tuple[torch.Size, torch.Size, torch.Size, tuple[int, ...]]] = []
    gate_layouts: list[tuple[torch.Size, tuple[int, ...]]] = []
    paged_impl = model_ops.paged_kv_single_token_decode_attention

    def _stub_paged_decode(
        query: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        page_table: torch.Tensor,
        *,
        scaling: float,
        visible_lengths: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls.append(
            (
                query.shape,
                key_pages.shape,
                page_table.shape,
                tuple(int(item) for item in visible_lengths.cpu().tolist()),
            )
        )
        if gate is not None:
            gate_layouts.append((gate.shape, tuple(int(stride) for stride in gate.stride())))
        return paged_impl(
            query,
            key_pages,
            value_pages,
            page_table,
            scaling=scaling,
            visible_lengths=visible_lengths,
            gate=gate,
        )

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _stub_paged_decode)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["full_attention"],
    )
    attention = Qwen3Attention(config, 0).eval().to("xpu", dtype=torch.bfloat16)
    rotary = Qwen3TextRotaryEmbedding(config).to("xpu")
    cache = Qwen3DynamicCache(config)
    prompt_states = torch.randn(1, 6, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    append_states = torch.randn(1, 1, config.hidden_size, device="xpu", dtype=torch.bfloat16)

    with torch.no_grad():
        prompt_position_ids = torch.arange(prompt_states.shape[1], device="xpu").view(1, -1)
        _ = attention(prompt_states, rotary(prompt_states, prompt_position_ids), past_key_values=cache)
        append_position_ids = torch.full((1, 1), prompt_states.shape[1], device="xpu", dtype=torch.long)
        _ = attention(append_states, rotary(append_states, append_position_ids), past_key_values=cache)

    torch.xpu.synchronize()
    assert calls == [
        (
            torch.Size([1, config.num_attention_heads, 1, config.head_dim]),
            torch.Size([16, config.num_key_value_heads, config.cache_block_size, config.head_dim]),
            torch.Size([1, 1]),
            (7,),
        )
    ]
    assert gate_layouts == [
        (
            torch.Size([1, config.num_attention_heads, 1, config.head_dim]),
            (config.num_attention_heads * config.head_dim, config.head_dim, config.num_attention_heads * config.head_dim, 1),
        )
    ]


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_qwen3_attention_xpu_mixed_length_single_token_decode_matches_full_forward() -> None:
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["full_attention"],
    )
    allocator = Qwen3PageAllocator(config)
    attention = Qwen3Attention(config, 0).eval().to("xpu", dtype=torch.bfloat16)
    rotary = Qwen3TextRotaryEmbedding(config).to("xpu")

    prompt_a = torch.randn(1, 6, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    prompt_b = torch.randn(1, 9, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    append_a = torch.randn(1, 1, config.hidden_size, device="xpu", dtype=torch.bfloat16)
    append_b = torch.randn(1, 1, config.hidden_size, device="xpu", dtype=torch.bfloat16)

    with torch.no_grad():
        full_a = torch.cat([prompt_a, append_a], dim=1)
        full_b = torch.cat([prompt_b, append_b], dim=1)
        full_positions_a = torch.arange(full_a.shape[1], device="xpu").view(1, -1)
        full_positions_b = torch.arange(full_b.shape[1], device="xpu").view(1, -1)
        full_output_a = attention(full_a, rotary(full_a, full_positions_a))
        full_output_b = attention(full_b, rotary(full_b, full_positions_b))

        cache_a = Qwen3DynamicCache(config, allocator=allocator)
        cache_b = Qwen3DynamicCache(config, allocator=allocator)
        prompt_positions_a = torch.arange(prompt_a.shape[1], device="xpu").view(1, -1)
        prompt_positions_b = torch.arange(prompt_b.shape[1], device="xpu").view(1, -1)
        _ = attention(prompt_a, rotary(prompt_a, prompt_positions_a), past_key_values=cache_a)
        _ = attention(prompt_b, rotary(prompt_b, prompt_positions_b), past_key_values=cache_b)

        stacked_cache = Qwen3DynamicCache.stack([cache_a, cache_b], config)
        append_states = torch.cat([append_a, append_b], dim=0)
        append_positions = torch.tensor([[prompt_a.shape[1]], [prompt_b.shape[1]]], device="xpu", dtype=torch.long)
        append_output = attention(append_states, rotary(append_states, append_positions), past_key_values=stacked_cache)

    torch.xpu.synchronize()
    assert torch.allclose(append_output[0:1].float().cpu(), full_output_a[:, -1:].float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(append_output[1:2].float().cpu(), full_output_b[:, -1:].float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    key = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    value = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    g = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(2, 3, 4, device=device, dtype=torch.float32))
    initial_state = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float32)

    output, final_state = torch.ops.anna.gated_delta_prefill(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=2e-2, rtol=2e-2)
    assert final_state is not None
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_flashqla_gated_delta_prefill_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_gated_delta_prefill"):
        pytest.skip("Anna fused-op library is not built with flashqla_gated_delta_prefill")

    torch.manual_seed(11)
    device = "xpu"
    query = torch.randn(1, 64, 2, 16, device=device, dtype=torch.float16)
    key = torch.randn(1, 64, 2, 16, device=device, dtype=torch.float16)
    value = torch.randn(1, 64, 2, 16, device=device, dtype=torch.float16)
    g = torch.randn(1, 64, 2, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(1, 64, 2, device=device, dtype=torch.float32))
    initial_state = torch.randn(1, 2, 16, 16, device=device, dtype=torch.float32)

    output, final_state = torch.ops.anna.flashqla_gated_delta_prefill(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_fused_xpu_matches_reference_large_head_dim() -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(1, 2, 4, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 4, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 4, 128, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, 2, 4, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(1, 2, 4, device=device, dtype=torch.float32))
    initial_state = torch.randn(1, 4, 128, 128, device=device, dtype=torch.float32)

    output, final_state = torch.ops.anna.gated_delta_prefill(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert final_state is not None
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_updates_explicit_state_buffer() -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(1, 1, 4, 32, device=device, dtype=torch.float16)
    key = torch.randn(1, 1, 4, 32, device=device, dtype=torch.float16)
    value = torch.randn(1, 1, 4, 32, device=device, dtype=torch.float16)
    g = torch.randn(1, 1, 4, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(1, 1, 4, device=device, dtype=torch.float32))
    initial_state = torch.randn(1, 4, 32, 32, device=device, dtype=torch.float32)
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("strategy", ["single", "tiled", "auto"])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_strategy_matches_qwen35_shape(
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", strategy)
    torch.manual_seed(0)
    device = "xpu"
    batch_size = 1
    num_heads = 32
    head_dim = 128
    query = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device, dtype=torch.float32)
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("value_head_dim", [64, 256])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_single_strategy_matches_mixed_kv_widths(
    value_head_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", "single")
    torch.manual_seed(1200 + value_head_dim)
    device = "xpu"
    batch_size = 2
    num_heads = 16
    key_head_dim = 128
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(("value_head_dim", "value_block"), [(64, 8), (256, 16)])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_tiled_strategy_matches_mixed_kv_widths(
    value_head_dim: int,
    value_block: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", "tiled")
    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", str(value_block))
    torch.manual_seed(3200 + value_head_dim + value_block)
    device = "xpu"
    batch_size = 2
    num_heads = 16
    key_head_dim = 128
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("strategy", ["single", "tiled", "auto"])
@pytest.mark.parametrize("value_head_dim", [128, 256])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_specialized_k128_shapes_match_reference(
    strategy: str,
    value_head_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", strategy)
    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", "16")
    torch.manual_seed(5200 + value_head_dim + len(strategy))
    device = "xpu"
    batch_size = 2
    num_heads = 16
    key_head_dim = 128
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "head_dim"),
    [
        (1, 16, 64),
        (4, 16, 64),
        (1, 32, 128),
        (4, 32, 128),
        (1, 16, 256),
        (4, 16, 256),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_auto_matches_qwen35_family_shapes(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", "auto")
    torch.manual_seed(batch_size * 1000 + num_heads * 10 + head_dim)
    device = "xpu"
    query = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device, dtype=torch.float32)
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "value_head_dim", "value_block", "expected_strategy_code"),
    [
        (1, 16, 64, 4, 0),
        (4, 16, 64, 4, 0),
        (1, 32, 128, 4, 0),
        (4, 32, 128, 4, 0),
        (1, 16, 256, 4, 0),
        (4, 16, 256, 4, 0),
        (2, 16, 64, 16, 1),
        (2, 16, 128, 16, 1),
        (2, 16, 256, 16, 1),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_strategy_debug_matches_qwen35_family_lookup(
    batch_size: int,
    num_heads: int,
    value_head_dim: int,
    value_block: int,
    expected_strategy_code: int,
) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gated_delta_decode_strategy_debug"):
        pytest.skip("Anna fused-op library is not built")

    query = torch.empty(batch_size, 1, num_heads, 128, device="xpu", dtype=torch.bfloat16)
    strategy_code = torch.ops.anna.gated_delta_decode_strategy_debug(query, value_head_dim, value_block)
    assert strategy_code == expected_strategy_code


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "value_head_dim", "expected_value_block", "expected_strategy_code"),
    [
        (1, 16, 64, 16, 1),
        (4, 16, 64, 16, 1),
        (1, 8, 128, 16, 1),
        (4, 8, 128, 16, 1),
        (1, 32, 128, 16, 1),
        (4, 32, 128, 16, 1),
        (1, 16, 256, 16, 1),
        (4, 16, 256, 16, 1),
        (18, 16, 256, 16, 1),
        (1, 32, 256, 16, 1),
        (4, 32, 256, 16, 1),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_value_block_debug_matches_arc_default_lookup(
    batch_size: int,
    num_heads: int,
    value_head_dim: int,
    expected_value_block: int,
    expected_strategy_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if (
        not maybe_load_gated_delta_library()
        or not hasattr(torch.ops.anna, "gated_delta_decode_value_block_debug")
        or not hasattr(torch.ops.anna, "gated_delta_decode_strategy_debug")
    ):
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.delenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", raising=False)
    query = torch.empty(batch_size, 1, num_heads, 128, device="xpu", dtype=torch.bfloat16)
    value_block = torch.ops.anna.gated_delta_decode_value_block_debug(query, value_head_dim)
    strategy_code = torch.ops.anna.gated_delta_decode_strategy_debug(query, value_head_dim, value_block)
    assert value_block == expected_value_block
    assert strategy_code == expected_strategy_code


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_value_block_debug_respects_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gated_delta_decode_value_block_debug"):
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", "8")
    query = torch.empty(1, 1, 16, 128, device="xpu", dtype=torch.bfloat16)
    value_block = torch.ops.anna.gated_delta_decode_value_block_debug(query, 256)
    assert value_block == 8


@pytest.mark.parametrize(
    ("strategy", "strategy_code", "value_block"),
    [
        ("auto", -1, 0),
        ("single", 0, 8),
        ("tiled", 1, 8),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_benchmark_matches_env_forced_path(
    strategy: str,
    strategy_code: int,
    value_block: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gated_delta_decode_benchmark"):
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", strategy)
    if value_block > 0:
        monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", str(value_block))
    else:
        monkeypatch.delenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", raising=False)

    torch.manual_seed(9100 + strategy_code + max(0, value_block))
    batch_size = 2
    num_heads = 16
    key_head_dim = 128
    value_head_dim = 128
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device="xpu", dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device="xpu", dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device="xpu", dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device="xpu", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device="xpu", dtype=torch.float32))
    initial_state = torch.randn(batch_size, num_heads, key_head_dim, value_head_dim, device="xpu", dtype=torch.float32)

    state_env = initial_state.clone()
    output_env = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_env,
    )

    state_bench = initial_state.clone()
    output_bench = torch.ops.anna.gated_delta_decode_benchmark(
        query,
        key,
        value,
        g,
        beta,
        state_bench,
        strategy_code,
        value_block,
        0,
    )

    torch.xpu.synchronize()
    assert torch.allclose(output_bench.float().cpu(), output_env.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_bench.float().cpu(), state_env.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "value_head_dim", "value_block"),
    [
        (1, 32, 64, 4),
        (4, 32, 64, 8),
        (8, 32, 64, 16),
        (4, 16, 64, 8),
        (8, 8, 64, 4),
        (8, 16, 64, 16),
        (4, 16, 128, 8),
        (18, 8, 128, 8),
        (19, 8, 128, 8),
        (9, 16, 128, 8),
        (10, 16, 128, 8),
        (10, 16, 256, 4),
        (17, 16, 256, 4),
        (18, 16, 256, 4),
        (29, 16, 256, 4),
        (30, 16, 256, 4),
        (33, 16, 256, 4),
        (34, 16, 256, 4),
        (45, 16, 256, 4),
        (60, 16, 256, 4),
        (5, 32, 128, 8),
        (6, 32, 128, 8),
        (5, 32, 256, 4),
        (6, 32, 256, 4),
        (9, 32, 256, 4),
        (15, 32, 256, 4),
        (17, 32, 256, 4),
        (24, 32, 256, 4),
        (25, 32, 256, 4),
        (29, 32, 256, 4),
        (30, 32, 256, 4),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_xpu_auto_matches_arc_row_cutover_shapes(
    batch_size: int,
    num_heads: int,
    value_head_dim: int,
    value_block: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK", str(value_block))
    key_head_dim = 128
    torch.manual_seed(batch_size * 1000 + num_heads * 10 + value_head_dim + value_block)
    device = "xpu"
    query = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, 1, num_heads, key_head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, 1, num_heads, value_head_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_heads, device=device, dtype=torch.float32))
    initial_state = torch.randn(
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    state_buffer = initial_state.clone()

    output = torch.ops.anna.gated_delta_decode(
        query,
        key,
        value,
        g,
        beta,
        state_buffer,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_core.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(state_buffer.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "value_head_dim", "value_block", "expected_strategy_code"),
    [
        (1, 32, 64, 4, 0),
        (4, 32, 64, 8, 0),
        (8, 32, 64, 16, 1),
        (4, 16, 64, 8, 0),
        (8, 8, 64, 4, 0),
        (8, 16, 64, 16, 1),
        (4, 16, 128, 8, 0),
        (18, 8, 128, 8, 1),
        (19, 8, 128, 8, 1),
        (9, 16, 128, 8, 1),
        (10, 16, 128, 8, 1),
        (10, 16, 256, 4, 1),
        (17, 16, 256, 4, 0),
        (18, 16, 256, 4, 0),
        (29, 16, 256, 4, 0),
        (30, 16, 256, 4, 1),
        (33, 16, 256, 4, 0),
        (34, 16, 256, 4, 0),
        (45, 16, 256, 4, 0),
        (49, 16, 256, 4, 0),
        (59, 16, 256, 4, 0),
        (60, 16, 256, 4, 1),
        (4, 32, 128, 8, 0),
        (5, 32, 128, 8, 1),
        (4, 32, 256, 4, 0),
        (5, 32, 256, 4, 1),
        (9, 32, 256, 4, 0),
        (15, 32, 256, 4, 1),
        (17, 32, 256, 4, 0),
        (24, 32, 256, 4, 1),
        (25, 32, 256, 4, 0),
        (29, 32, 256, 4, 0),
        (30, 32, 256, 4, 1),
    ],
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_decode_strategy_debug_matches_arc_lookup(
    batch_size: int,
    num_heads: int,
    value_head_dim: int,
    value_block: int,
    expected_strategy_code: int,
) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "gated_delta_decode_strategy_debug"):
        pytest.skip("Anna fused-op library is not built")

    query = torch.empty(batch_size, 1, num_heads, 128, device="xpu", dtype=torch.bfloat16)
    strategy_code = torch.ops.anna.gated_delta_decode_strategy_debug(query, value_head_dim, value_block)
    assert strategy_code == expected_strategy_code
