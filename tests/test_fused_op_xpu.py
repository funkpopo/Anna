import pytest
import torch

import anna.model.ops as model_ops
from anna.model.fused_ops import maybe_load_gated_delta_library
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
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


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
def test_qwen3_attention_xpu_single_token_decode_uses_materialized_kv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "qk_norm_rotary_fused"):
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    calls: list[tuple[torch.Size, torch.Size, int, tuple[int, ...]]] = []
    materialized_impl = model_ops.materialized_kv_single_token_decode_attention

    def _stub_materialized_decode(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scaling: float,
        num_key_value_groups: int,
        visible_lengths: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(
            (
                query.shape,
                key.shape,
                num_key_value_groups,
                tuple(int(item) for item in visible_lengths.cpu().tolist()),
            )
        )
        return materialized_impl(
            query,
            key,
            value,
            scaling=scaling,
            num_key_value_groups=num_key_value_groups,
            visible_lengths=visible_lengths,
        )

    monkeypatch.setattr(model_ops, "materialized_kv_single_token_decode_attention", _stub_materialized_decode)
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
            torch.Size([1, config.num_key_value_heads, 7, config.head_dim]),
            config.num_attention_heads // config.num_key_value_heads,
            (7,),
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
    beta = torch.sigmoid(torch.randn(2, 3, 4, device=device, dtype=torch.float16))
    z = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    norm_weight = torch.randn(8, device=device, dtype=torch.float32)
    initial_state = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float32)

    output, final_state = torch.ops.anna.gated_delta_fused(
        query,
        key,
        value,
        g,
        beta,
        z,
        norm_weight,
        1e-6,
        initial_state,
        True,
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
    ref_hidden = ref_core.reshape(-1, value.shape[-1]).float()
    ref_z = z.reshape(-1, value.shape[-1]).float()
    ref_hidden = ref_hidden * torch.rsqrt(ref_hidden.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref_hidden = norm_weight * ref_hidden.to(dtype=ref_core.dtype)
    ref_output = (ref_hidden * torch.nn.functional.silu(ref_z)).to(dtype=ref_core.dtype).reshape(
        query.shape[0],
        query.shape[1],
        -1,
    )

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_output.float().cpu(), atol=2e-2, rtol=2e-2)
    assert final_state is not None
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=2e-2, rtol=2e-2)
