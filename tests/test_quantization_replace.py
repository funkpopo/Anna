from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import QuantizationConfig
from anna.model.quantization import AWQLinear, FP8Linear, XPUInt4Linear, convert_module_linears_to_xpu_int4, replace_linear_modules


class _TinyQuantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.block = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
        )


def _pack_int4_matrix(values: torch.Tensor) -> torch.Tensor:
    reshaped = values.to(torch.int32).reshape(values.shape[0], values.shape[1] // 8, 8)
    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
    return torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32)


def _dequantize_xpu_int4_state(
    packed_weight: torch.Tensor,
    qscale: torch.Tensor,
    qzeros: torch.Tensor,
    *,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
    unpacked = torch.bitwise_right_shift(packed_weight.unsqueeze(-1).to(torch.int32), shifts) & 0xF
    unpacked = unpacked.reshape(packed_weight.shape[0], packed_weight.shape[1] * 8)[:, :in_features].to(torch.float32)
    group_ids = torch.arange(in_features, dtype=torch.long) // group_size
    scales = qscale[group_ids].transpose(0, 1).to(torch.float32)
    zeros = qzeros[group_ids].transpose(0, 1).to(torch.float32)
    return (unpacked - zeros) * scales


def test_replace_linear_modules_fp8() -> None:
    model = _TinyQuantNet()
    specs = replace_linear_modules(
        model,
        QuantizationConfig(
            quant_method="fp8",
            weight_block_size=(128, 128),
        ),
        compute_dtype=torch.bfloat16,
    )

    assert len(specs) == 3
    assert isinstance(model.proj, FP8Linear)
    assert isinstance(model.block[0], FP8Linear)
    assert isinstance(model.block[2], FP8Linear)


def test_replace_linear_modules_awq_respects_skip_list() -> None:
    model = _TinyQuantNet()
    specs = replace_linear_modules(
        model,
        QuantizationConfig(
            quant_method="awq",
            bits=4,
            group_size=128,
            zero_point=True,
            modules_to_not_convert=["proj"],
        ),
        compute_dtype=torch.float16,
    )

    assert len(specs) == 2
    assert isinstance(model.proj, nn.Linear)
    assert isinstance(model.block[0], AWQLinear)
    assert isinstance(model.block[2], AWQLinear)


def test_convert_module_linears_to_xpu_int4_replaces_supported_linears() -> None:
    model = _TinyQuantNet()
    converted = convert_module_linears_to_xpu_int4(model, group_size=32, device=torch.device("cpu"))

    assert converted == 3
    assert isinstance(model.proj, XPUInt4Linear)
    assert isinstance(model.block[0], XPUInt4Linear)
    assert isinstance(model.block[2], XPUInt4Linear)


def test_xpu_int4_linear_cpu_fallback_matches_dense_linear_closely() -> None:
    linear = nn.Linear(32, 16, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-1.0, 1.0, steps=16 * 32).reshape(16, 32))

    quantized = XPUInt4Linear.from_linear(
        linear,
        group_size=32,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    inputs = torch.linspace(-1.0, 1.0, steps=4 * 32, dtype=torch.float32).reshape(4, 32).to(dtype=torch.bfloat16)

    reference = linear(inputs.to(dtype=torch.float32)).to(dtype=torch.bfloat16)
    actual = quantized(inputs)
    error = (actual.to(dtype=torch.float32) - reference.to(dtype=torch.float32)).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    assert max_error < 0.15
    assert mean_error < 0.05


def test_awq_linear_builds_cached_xpu_pack_state_without_redequantizing_each_forward() -> None:
    layer = AWQLinear(
        48,
        8,
        group_size=32,
        zero_point=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    weight_int = (torch.arange(48 * 8, dtype=torch.int32).reshape(48, 8) % 11) + 2
    zero_points = torch.tensor(
        [
            [8, 7, 9, 6, 8, 10, 7, 8],
            [7, 8, 9, 8, 6, 7, 8, 9],
        ],
        dtype=torch.int32,
    )
    scales = torch.linspace(0.125, 1.0, steps=16, dtype=torch.float32).reshape(2, 8)

    layer.qweight = _pack_int4_matrix(weight_int)
    layer.qzeros = _pack_int4_matrix(zero_points)
    layer.scales = scales.to(torch.float16)

    packed_weight, packed_scales, packed_zeros, padded_in_features = layer._build_xpu_fast_path_state()
    reference = layer._dequantize_awq()
    actual = _dequantize_xpu_int4_state(
        packed_weight,
        packed_scales,
        packed_zeros,
        in_features=layer.in_features,
        group_size=layer.group_size,
    )

    assert padded_in_features == 64
    assert packed_weight.shape == (8, 8)
    assert packed_scales.shape == (2, 8)
    assert packed_zeros.shape == (2, 8)
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="requires torch.xpu")
def test_awq_linear_xpu_fast_path_matches_dequantized_reference() -> None:
    layer = AWQLinear(
        32,
        8,
        group_size=32,
        zero_point=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("xpu"),
    )
    weight_int = (torch.arange(32 * 8, dtype=torch.int32).reshape(32, 8) % 13) + 1
    zero_points = torch.tensor([[8, 7, 9, 6, 8, 10, 7, 8]], dtype=torch.int32)
    scales = torch.linspace(0.125, 1.0, steps=8, dtype=torch.float32).reshape(1, 8)

    layer.qweight = _pack_int4_matrix(weight_int).to(device="xpu")
    layer.qzeros = _pack_int4_matrix(zero_points).to(device="xpu")
    layer.scales = scales.to(dtype=torch.float16, device="xpu")

    inputs = torch.randn(4, 32, device="xpu", dtype=torch.bfloat16)
    actual = layer(inputs)
    reference = F.linear(
        inputs.to(dtype=torch.bfloat16),
        layer._dequantize_awq().to(device="xpu", dtype=torch.bfloat16),
        None,
    ).to(dtype=inputs.dtype)

    assert torch.allclose(actual, reference, atol=5e-2, rtol=5e-2)
