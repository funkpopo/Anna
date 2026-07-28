"""P2-2.10: quantization path golden — bf16 vs int4 vs TurboQuant agreement bounds."""

from __future__ import annotations

import torch
from torch import nn

from anna.model.quantization import XPUInt4Linear
from anna.model.turboquant import dequantize_turboquant_values, quantize_turboquant_values


def _top_k_agreement(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, *, k: int = 5) -> float:
    """Fraction of rows where top-k token sets intersect (Jaccard-style hit rate)."""
    ref_top = set(int(i) for i in reference_logits.topk(k).indices.tolist())
    cand_top = set(int(i) for i in candidate_logits.topk(k).indices.tolist())
    if not ref_top and not cand_top:
        return 1.0
    return len(ref_top & cand_top) / max(1, len(ref_top | cand_top))


def test_int4_dequant_vs_bf16_linear_token_agreement() -> None:
    """Dense bf16 matmul vs XPU int4 dequant path: logits stay within allowed error and top-k."""
    torch.manual_seed(7)
    linear = nn.Linear(64, 128, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(128, 64) * 0.05)

    quantized = XPUInt4Linear.from_linear(
        linear,
        group_size=32,
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    inputs = torch.randn(8, 64, dtype=torch.bfloat16)

    reference = linear(inputs.float()).float()
    # Force dequant strategy for a deterministic CPU path.
    import os

    previous = os.environ.get("ANNA_XPU_INT4_MATMUL")
    os.environ["ANNA_XPU_INT4_MATMUL"] = "dequant"
    try:
        actual = quantized(inputs).float()
    finally:
        if previous is None:
            os.environ.pop("ANNA_XPU_INT4_MATMUL", None)
        else:
            os.environ["ANNA_XPU_INT4_MATMUL"] = previous

    error = (actual - reference).abs().detach()
    assert float(error.mean()) < 0.08
    assert float(error.max()) < 0.35

    # Per-row top-k token agreement across the vocab-like output dim.
    agreements = [
        _top_k_agreement(reference[row], actual[row], k=8)
        for row in range(reference.shape[0])
    ]
    mean_agreement = sum(agreements) / len(agreements)
    assert mean_agreement >= 0.5, f"int4 top-k agreement too low: {mean_agreement:.3f}"


def test_turboquant_values_vs_bf16_roundtrip_agreement() -> None:
    """TurboQuant value pack/unpack vs bf16 reference: MSE + correlation bounds by bits."""
    torch.manual_seed(11)
    values = torch.randn(4, 48, 64, dtype=torch.float32)

    for bits, max_mse, min_corr in (
        (4, 0.30, 0.80),
        (3, 0.55, 0.70),
        (2, 1.20, 0.50),
    ):
        state = quantize_turboquant_values(values, bits=bits, group_size=32)
        restored = dequantize_turboquant_values(state, dtype=torch.float32, device=values.device)
        mse = float(torch.mean((restored - values) ** 2).item())
        corr = float(
            torch.corrcoef(torch.stack([values.reshape(-1), restored.reshape(-1)]))[0, 1].item()
        )
        assert mse < max_mse, f"bits={bits} mse={mse} exceeded {max_mse}"
        assert corr > min_corr, f"bits={bits} corr={corr} below {min_corr}"


def test_int4_and_turboquant_combined_noise_budget() -> None:
    """Stack int4 weight noise + TurboQuant value noise stays within a loose combined budget.

    Full-model token consistency needs a loaded checkpoint; this golden checks that
    the two dominant compression paths do not each blow the per-path bounds so badly
    that a combined residual would be unusable.
    """
    torch.manual_seed(3)
    linear = nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(32, 32) * 0.08)

    quantized = XPUInt4Linear.from_linear(
        linear,
        group_size=32,
        compute_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    x = torch.randn(4, 32, dtype=torch.float32)
    dense_out = linear(x)
    import os

    previous = os.environ.get("ANNA_XPU_INT4_MATMUL")
    os.environ["ANNA_XPU_INT4_MATMUL"] = "dequant"
    try:
        int4_out = quantized(x)
    finally:
        if previous is None:
            os.environ.pop("ANNA_XPU_INT4_MATMUL", None)
        else:
            os.environ["ANNA_XPU_INT4_MATMUL"] = previous

    # Treat the linear output rows as faux value heads for TurboQuant.
    values = dense_out.unsqueeze(0).expand(2, -1, -1).contiguous()  # [heads, tokens, dim]
    state = quantize_turboquant_values(values, bits=4, group_size=16)
    tq_out = dequantize_turboquant_values(state, dtype=torch.float32, device=values.device)

    int4_rel = float(
        ((int4_out - dense_out).abs().mean() / (dense_out.abs().mean() + 1e-6)).detach()
    )
    tq_rel = float(((tq_out - values).abs().mean() / (values.abs().mean() + 1e-6)).detach())
    # Combined relative noise should stay under a product-style budget.
    assert int4_rel < 0.25
    assert tq_rel < 0.35
    assert (int4_rel + tq_rel) < 0.55
