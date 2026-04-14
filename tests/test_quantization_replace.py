from __future__ import annotations

import torch
from torch import nn

from anna.model.qwen3_5_text_config import QuantizationConfig
from anna.model.quantization import (
    AWQLinear,
    AutoRoundGPTQLinear,
    FP8Linear,
    XPUInt4Linear,
    convert_module_linears_to_xpu_int4,
    extract_linear_weight_bias_cpu,
    replace_linear_modules,
)


class _TinyQuantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.block = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
        )


def _pack_int4_first_dim(values: torch.Tensor) -> torch.Tensor:
    reshaped = values.to(torch.int32).reshape(values.shape[0] // 8, 8, values.shape[1])
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
    return torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=1, dtype=torch.int32)


def _pack_int4_last_dim(values: torch.Tensor) -> torch.Tensor:
    reshaped = values.to(torch.int32).reshape(values.shape[0], values.shape[1] // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 1, 8)
    return torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32)


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


def test_autoround_linear_preallocates_packed_buffers() -> None:
    linear = AutoRoundGPTQLinear(
        2048,
        512,
        group_size=128,
        zero_point=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        bias=False,
    )

    assert tuple(linear.qweight.shape) == (256, 512)
    assert tuple(linear.qzeros.shape) == (16, 64)
    assert tuple(linear.scales.shape) == (16, 512)


def test_convert_module_linears_to_xpu_int4_replaces_supported_linears() -> None:
    model = _TinyQuantNet()
    converted = convert_module_linears_to_xpu_int4(model, group_size=32, device=torch.device("cpu"))

    assert converted == 3
    assert isinstance(model.proj, XPUInt4Linear)
    assert isinstance(model.block[0], XPUInt4Linear)
    assert isinstance(model.block[2], XPUInt4Linear)


def test_convert_module_linears_to_xpu_int4_respects_include_predicate() -> None:
    model = _TinyQuantNet()
    converted = convert_module_linears_to_xpu_int4(
        model,
        group_size=32,
        device=torch.device("cpu"),
        include_predicate=lambda module_name, _module: module_name.startswith("block."),
    )

    assert converted == 2
    assert isinstance(model.proj, nn.Linear)
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


def test_xpu_int4_linear_from_autoround_matches_autoround_cpu_path() -> None:
    linear = AutoRoundGPTQLinear(
        32,
        16,
        group_size=32,
        zero_point=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        bias=False,
    )
    weight_int = (torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16) % 15) + 1
    zero_points = torch.full((1, 16), 8, dtype=torch.int32)
    scales = torch.linspace(0.05, 0.20, steps=16, dtype=torch.float32).reshape(1, 16)
    with torch.no_grad():
        linear.qweight.copy_(_pack_int4_first_dim(weight_int))
        linear.qzeros.copy_(_pack_int4_last_dim(zero_points))
        linear.scales.copy_(scales.to(dtype=torch.float16))

    inputs = torch.linspace(-1.0, 1.0, steps=3 * 32, dtype=torch.float32).reshape(3, 32).to(dtype=torch.bfloat16)
    reference = linear(inputs)
    converted = XPUInt4Linear.from_linear(
        linear,
        group_size=32,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    actual = converted(inputs)

    assert torch.allclose(
        actual.to(dtype=torch.float32),
        reference.to(dtype=torch.float32),
        atol=1e-4,
        rtol=1e-4,
    )


def test_extract_linear_weight_bias_cpu_matches_autoround_dequantization() -> None:
    linear = AutoRoundGPTQLinear(
        32,
        16,
        group_size=32,
        zero_point=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        bias=False,
    )
    weight_int = (torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16) % 15) + 1
    zero_points = torch.full((1, 16), 8, dtype=torch.int32)
    scales = torch.linspace(0.05, 0.20, steps=16, dtype=torch.float32).reshape(1, 16)
    with torch.no_grad():
        linear.qweight.copy_(_pack_int4_first_dim(weight_int))
        linear.qzeros.copy_(_pack_int4_last_dim(zero_points))
        linear.scales.copy_(scales.to(dtype=torch.float16))

    extracted_weight, extracted_bias = extract_linear_weight_bias_cpu(linear)
    reference_weight = linear._dequantize_autoround()

    assert extracted_bias is None
    assert extracted_weight.device.type == "cpu"
    assert torch.allclose(extracted_weight, reference_weight, atol=1e-4, rtol=1e-4)
