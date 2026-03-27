from __future__ import annotations

import torch
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
