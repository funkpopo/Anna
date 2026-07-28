from __future__ import annotations

import pytest

from anna.model.turboquant import (
    TURBOQUANT_PRESETS,
    recommend_turboquant_preset,
    resolve_turboquant_runtime_settings,
    turboquant_tier_for_weight_bytes,
)


def test_turboquant_tiers_cover_common_model_sizes() -> None:
    assert turboquant_tier_for_weight_bytes(2 * (1 << 30)) == "small"
    assert turboquant_tier_for_weight_bytes(12 * (1 << 30)) == "medium"
    assert turboquant_tier_for_weight_bytes(40 * (1 << 30)) == "large"
    assert turboquant_tier_for_weight_bytes(100 * (1 << 30)) == "xlarge"


def test_recommend_turboquant_preset_bits_and_residual() -> None:
    small = recommend_turboquant_preset(1 << 30)
    medium = recommend_turboquant_preset(10 * (1 << 30))
    large = recommend_turboquant_preset(40 * (1 << 30))
    xlarge = recommend_turboquant_preset(90 * (1 << 30))

    assert small == TURBOQUANT_PRESETS["small"]
    assert small.bits == 4 and small.residual_len == 128
    assert medium.bits == 3 and medium.residual_len == 128
    assert large.bits == 2 and large.residual_len == 96
    assert xlarge.bits == 2 and xlarge.residual_len == 64


def test_resolve_turboquant_auto_falls_back_when_dependency_missing() -> None:
    mode, bits, residual, tier = resolve_turboquant_runtime_settings(
        requested_mode="auto",
        requested_bits=4,
        requested_residual_len=128,
        weight_bytes=10 * (1 << 30),
        turboquant_available=False,
    )
    assert mode == "none"
    assert bits == 4
    assert residual == 128
    assert tier is None


def test_resolve_turboquant_auto_applies_size_preset() -> None:
    mode, bits, residual, tier = resolve_turboquant_runtime_settings(
        requested_mode="auto",
        requested_bits=4,
        requested_residual_len=128,
        weight_bytes=40 * (1 << 30),
        bits_explicit=False,
        residual_explicit=False,
        turboquant_available=True,
    )
    assert mode == "turboquant"
    assert bits == 2
    assert residual == 96
    assert tier == "large"


def test_resolve_turboquant_keeps_explicit_bits_and_residual() -> None:
    mode, bits, residual, tier = resolve_turboquant_runtime_settings(
        requested_mode="turboquant",
        requested_bits=3,
        requested_residual_len=64,
        weight_bytes=40 * (1 << 30),
        bits_explicit=True,
        residual_explicit=True,
        turboquant_available=True,
    )
    assert mode == "turboquant"
    assert bits == 3
    assert residual == 64
    assert tier is None  # no preset fill when both explicit


def test_resolve_turboquant_rejects_missing_dependency_for_explicit_mode() -> None:
    with pytest.raises(RuntimeError, match="turboquant"):
        resolve_turboquant_runtime_settings(
            requested_mode="turboquant",
            requested_bits=4,
            requested_residual_len=128,
            turboquant_available=False,
        )


def test_turboquant_value_pack_roundtrip_golden_logits_sample() -> None:
    """Lightweight accuracy regression for the value-side TurboQuant pack path.

    Full-model perplexity needs a loaded checkpoint + XPU. This samples
    ``quantize_turboquant_values`` / ``dequantize_turboquant_values`` which back
    residual windows and compressed older entries.
    """
    import torch

    from anna.model.turboquant import dequantize_turboquant_values, quantize_turboquant_values

    torch.manual_seed(0)
    # Layout: [heads, tokens, dim]
    values = torch.randn(2, 32, 64, dtype=torch.float32)
    state = quantize_turboquant_values(values, bits=4, group_size=32)
    restored = dequantize_turboquant_values(state, dtype=torch.float32, device=values.device)
    # 4-bit group quant should stay within a loose MSE bound on unit-variance noise.
    mse = float(torch.mean((restored - values) ** 2).item())
    assert mse < 0.25
    # Spot-check a few positions stay correlated (not pure noise after dequant).
    corr = torch.corrcoef(torch.stack([values.reshape(-1), restored.reshape(-1)]))[0, 1].item()
    assert corr > 0.85
