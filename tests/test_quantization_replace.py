from __future__ import annotations

import torch
from torch import nn

from anna.model.config import QuantizationConfig
from anna.model.quantization import AWQLinear, FP8Linear, replace_linear_modules


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
