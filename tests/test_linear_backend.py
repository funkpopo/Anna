from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from anna.model import xpu_fusion
from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.ops import Qwen3MLP
from anna.model.xpu_fusion import _load_extension_if_available, apply_linear_pointwise, describe_xpu_fusion_status


def _tiny_text_config() -> Qwen3TextConfig:
    return Qwen3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
    )


def test_apply_linear_pointwise_cpu_fallback_matches_reference() -> None:
    linear = nn.Linear(8, 8, bias=True)
    inputs = torch.randn(2, 3, 8)
    residual = torch.randn(2, 3, 8)

    expected = F.silu(linear(inputs)) + residual
    actual = apply_linear_pointwise(linear, inputs, activation="silu", residual=residual)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_qwen3_mlp_residual_fused_entry_matches_reference() -> None:
    mlp = Qwen3MLP(_tiny_text_config())
    inputs = torch.randn(2, 3, 64)
    residual = torch.randn(2, 3, 64)

    expected = residual + mlp.down_proj(F.silu(mlp.gate_proj(inputs)) * mlp.up_proj(inputs))
    actual = mlp(inputs, residual=residual)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_xpu_fusion_status_reports_shell_toolchain_requirements() -> None:
    status = describe_xpu_fusion_status()

    assert status.reason in {
        "ready",
        "disabled_by_env",
        "xpu_unavailable",
        "native_sources_missing",
        "oneapi_compiler_missing",
        "msvc_build_tools_missing",
        "extension_load_failed",
    }
    assert isinstance(status.extension_loaded, bool)


def test_apply_linear_pointwise_falls_back_cleanly_when_extension_is_unavailable(monkeypatch) -> None:
    linear = nn.Linear(8, 8, bias=True)
    inputs = torch.randn(2, 3, 8)

    monkeypatch.setattr(xpu_fusion, "_LOAD_ATTEMPTED", True)
    monkeypatch.setattr(xpu_fusion, "_LOAD_ERROR", "synthetic failure")

    expected = linear(inputs)
    actual = apply_linear_pointwise(linear, inputs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="requires torch.xpu")
def test_linear_pointwise_xpu_custom_op_matches_reference() -> None:
    status = describe_xpu_fusion_status()
    if not status.enabled:
        pytest.skip(f"requires XPU fusion toolchain readiness, got status={status.reason}")

    assert _load_extension_if_available() is True
    assert _load_extension_if_available() is True

    inputs = torch.randn(2, 3, 8, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(8, 8, device="xpu", dtype=torch.bfloat16)
    bias = torch.randn(8, device="xpu", dtype=torch.bfloat16)
    residual = torch.randn(2, 3, 8, device="xpu", dtype=torch.bfloat16)

    actual = xpu_fusion.linear_pointwise(
        inputs,
        weight,
        bias,
        activation="silu",
        residual=residual,
    )
    reference = F.silu(F.linear(inputs, weight, bias)) + residual

    assert torch.allclose(actual, reference, atol=5e-2, rtol=5e-2)
