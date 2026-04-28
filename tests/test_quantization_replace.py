from __future__ import annotations

import torch
import pytest
from torch import nn

from anna.model.qwen3_5_text_config import QuantizationConfig
from anna.model.quantization import (
    AWQLinear,
    AutoRoundGPTQLinear,
    XPUInt4Linear,
    convert_module_linears_to_xpu_int4,
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
    reshaped = values.reshape(values.shape[0] // 8, 8, values.shape[1]).permute(0, 2, 1).contiguous()
    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
    return torch.bitwise_left_shift(reshaped.to(torch.int32) & 0xF, shifts).sum(dim=-1, dtype=torch.int32)


def _pack_int4_last_dim(values: torch.Tensor) -> torch.Tensor:
    reshaped = values.reshape(values.shape[0], values.shape[1] // 8, 8).contiguous()
    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
    return torch.bitwise_left_shift(reshaped.to(torch.int32) & 0xF, shifts).sum(dim=-1, dtype=torch.int32)


def test_replace_linear_modules_autoround_respects_block_names_and_extra_config() -> None:
    model = _TinyQuantNet()
    specs = replace_linear_modules(
        model,
        QuantizationConfig(
            quant_method="auto-round",
            bits=4,
            group_size=128,
            data_type="int",
            sym=True,
            packing_format="auto_round:auto_gptq",
            block_name_to_quantize=("proj", "block"),
            extra_config={"block.0": {"bits": 16, "data_type": "fp"}},
        ),
        compute_dtype=torch.bfloat16,
    )

    assert len(specs) == 2
    assert isinstance(model.proj, AutoRoundGPTQLinear)
    assert isinstance(model.block[0], nn.Linear)
    assert isinstance(model.block[2], AutoRoundGPTQLinear)


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


def test_replace_linear_modules_rejects_fp8_models() -> None:
    model = _TinyQuantNet()
    try:
        replace_linear_modules(
            model,
            QuantizationConfig(quant_method="fp8"),
            compute_dtype=torch.bfloat16,
        )
    except ValueError as exc:
        assert "Unsupported quantization method" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("FP8 compatibility should be removed from replace_linear_modules().")


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


def test_convert_module_linears_to_xpu_int4_supports_autoround_payloads() -> None:
    autoround = AutoRoundGPTQLinear(
        32,
        8,
        bits=4,
        group_size=32,
        sym=True,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    weight_int = (torch.arange(32 * 8, dtype=torch.int32).reshape(32, 8) % 15) + 1
    qzeros = torch.full((1, 8), 8, dtype=torch.int32)
    scales = torch.tensor(
        [
            [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        ],
        dtype=torch.float16,
    )
    with torch.no_grad():
        autoround.qweight.copy_(_pack_int4_first_dim(weight_int))
        autoround.qzeros.copy_(_pack_int4_last_dim(qzeros))
        autoround.scales.copy_(scales)

    container = nn.Sequential(autoround)
    converted = convert_module_linears_to_xpu_int4(container, device=torch.device("cpu"))
    assert converted == 1
    assert isinstance(container[0], XPUInt4Linear)

    inputs = torch.linspace(-1.0, 1.0, steps=2 * 32, dtype=torch.float32).reshape(2, 32).to(dtype=torch.bfloat16)
    reference = autoround(inputs)
    actual = container(inputs)
    assert torch.allclose(actual.float(), reference.float(), atol=1e-4, rtol=1e-4)


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


def test_xpu_int4_linear_strategy_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "dequant")
    assert XPUInt4Linear._matmul_strategy() == "dequant"

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "invalid")
    assert XPUInt4Linear._matmul_strategy() == "auto"
