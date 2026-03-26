from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import Qwen3TextConfig
from anna.model.linear_backend import linear_with_backend, project_linear
from anna.model.onednn_custom import custom_linear_int4_weight_only, custom_linear_pointwise
from anna.model.ops import Qwen3MLP
from anna.model.quantization import AWQLinear, DenseLinear


def test_linear_with_backend_matches_silu_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    weight = torch.randn(16, 8)
    bias = torch.randn(16)

    actual = linear_with_backend(x, weight, bias, activation="swish")
    expected = F.silu(F.linear(x, weight, bias))

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_with_backend_matches_binary_add_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    residual = torch.randn(2, 3, 8)

    actual = linear_with_backend(x, weight, bias, other=residual, binary="add")
    expected = F.linear(x, weight, bias) + residual

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_project_linear_supports_dense_linear_modules() -> None:
    torch.manual_seed(0)
    baseline = nn.Linear(8, 12, bias=True)
    dense = DenseLinear(8, 12, bias=True)
    with torch.no_grad():
        dense.weight.copy_(baseline.weight)
        dense.bias.copy_(baseline.bias)

    x = torch.randn(4, 2, 8)
    actual = project_linear(dense, x, activation="relu")
    expected = F.relu(baseline(x))

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_qwen3_mlp_residual_fusion_matches_reference() -> None:
    torch.manual_seed(0)
    config = Qwen3TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=128,
        layer_types=["full_attention"],
    )
    mlp = Qwen3MLP(config)
    x = torch.randn(2, 3, 16)
    residual = torch.randn(2, 3, 16)

    actual = mlp(x, residual=residual)
    expected = F.linear(
        F.silu(F.linear(x, mlp.gate_proj.weight, None)) * F.linear(x, mlp.up_proj.weight, None),
        mlp.down_proj.weight,
        None,
    ) + residual

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_awq_weight_only_repack_matches_xpu_kernel_layout() -> None:
    torch.manual_seed(0)
    in_features = 128
    out_features = 64
    group_size = 32
    group_count = in_features // group_size
    weight_int = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zero_int = torch.randint(0, 16, (group_count, out_features), dtype=torch.int32)
    scales = torch.randn(group_count, out_features, dtype=torch.float16).abs() + 0.01

    layer = AWQLinear(
        in_features,
        out_features,
        group_size=group_size,
        bias=False,
        compute_dtype=torch.float16,
    )
    layer.qweight = AWQLinear._pack_int4(weight_int)
    layer.qzeros = AWQLinear._pack_int4(zero_int)
    layer.scales = scales

    packed_weight, packed_scales, packed_zeros = layer._prepare_awq_weight_only_tensors(
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    expected_weight = AWQLinear._pack_int4(weight_int.transpose(0, 1).contiguous())
    assert torch.equal(packed_weight, expected_weight)
    assert torch.equal(packed_scales, scales)
    assert torch.equal(packed_zeros, zero_int.to(torch.int8))


def test_custom_linear_pointwise_xpu_repeated_execution_matches_reference() -> None:
    if not torch.xpu.is_available():
        return

    torch.manual_seed(0)
    x = torch.randn(8, 128, device="xpu", dtype=torch.float16)
    weight = torch.randn(256, 128, device="xpu", dtype=torch.float16)
    bias = torch.randn(256, device="xpu", dtype=torch.float16)

    output = None
    for _ in range(16):
        output = custom_linear_pointwise(x, weight, bias, activation="swish")

    assert output is not None
    expected = F.silu(F.linear(x, weight, bias))
    assert torch.allclose(output, expected, atol=3e-3, rtol=3e-3)


def test_awq_linear_weight_only_xpu_matches_reference() -> None:
    if not torch.xpu.is_available():
        return

    torch.manual_seed(0)
    in_features = 128
    out_features = 64
    group_size = 32
    group_count = in_features // group_size
    layer = AWQLinear(
        in_features,
        out_features,
        group_size=group_size,
        bias=True,
        compute_dtype=torch.float16,
    ).to(device="xpu")

    weight_int = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32, device="xpu")
    zero_int = torch.randint(0, 16, (group_count, out_features), dtype=torch.int32, device="xpu")
    scales = (torch.rand(group_count, out_features, dtype=torch.float16, device="xpu") * 0.2) + 0.01
    layer.qweight = AWQLinear._pack_int4(weight_int)
    layer.qzeros = AWQLinear._pack_int4(zero_int)
    layer.scales = scales
    with torch.no_grad():
        assert layer.bias is not None
        layer.bias.copy_(torch.randn(out_features, device="xpu", dtype=torch.float16))

    x = torch.randn(2, 3, in_features, device="xpu", dtype=torch.float16)
    reference_weight = layer._dequantize_awq().to(dtype=torch.float16)
    reference = F.silu(F.linear(x, reference_weight, layer.bias))

    packed_weight, packed_scales, packed_zeros = layer._prepare_awq_weight_only_tensors(
        device=torch.device("xpu"),
        dtype=torch.float16,
    )
    direct = custom_linear_int4_weight_only(
        x.reshape(-1, in_features),
        packed_weight,
        packed_scales,
        packed_zeros,
        group_size=group_size,
    )
    assert direct is not None

    def _unexpected_dense_fallback() -> torch.Tensor:
        raise AssertionError("AWQLinear fell back to dense dequantization instead of the custom INT4 XPU path.")

    layer._dequantize_awq = _unexpected_dense_fallback  # type: ignore[method-assign]
    actual = layer.project_linear(x, activation="swish")

    assert torch.allclose(actual, reference, atol=3e-3, rtol=3e-3)
