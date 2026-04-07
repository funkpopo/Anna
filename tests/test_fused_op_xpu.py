import pytest
import torch

from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.ops import (
    Qwen3Attention,
    Qwen3TextRotaryEmbedding,
    apply_rotary_pos_emb,
    torch_causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(dtype=x.dtype)


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
