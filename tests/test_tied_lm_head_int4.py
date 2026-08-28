"""Regression tests: tied lm_head must be filled when it is a direct-int4 placeholder.

Bug context: with ``tie_word_embeddings=true`` checkpoints store no ``lm_head.weight``.
Low-memory loading replaces lm_head with a zero-initialized XPUInt4Linear placeholder,
and ``tie_weights()`` only handled ``nn.Linear`` — the placeholder stayed all-zero and
the model produced zero logits. This test pins the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anna.model.quantization import XPUInt4Linear  # noqa: E402
from anna.model.qwen3_5_text_model import _fill_xpu_int4_lm_head_from_embedding  # noqa: E402


def _make_placeholder(out_features: int, in_features: int, group_size: int = 32) -> XPUInt4Linear:
    return XPUInt4Linear(
        in_features,
        out_features,
        group_size=group_size,
        bias=False,
        compute_dtype=torch.bfloat16,
        device="cpu",
    )


def test_fill_int4_lm_head_from_embedding_produces_correct_payload():
    torch.manual_seed(0)
    out_features, in_features = 64, 128
    lm_head = _make_placeholder(out_features, in_features)
    # Simulate an unfilled low-memory placeholder (buffers are torch.empty).
    torch.nn.init.zeros_(lm_head.qweight)
    torch.nn.init.zeros_(lm_head.qscale)

    embed_weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    assert _fill_xpu_int4_lm_head_from_embedding(lm_head, embed_weight) is True

    # Payload must be non-zero now and match the quantization of the source weight.
    assert int(lm_head.qweight.abs().max()) > 0
    assert float(lm_head.qscale.abs().max()) > 0
    x = torch.randn(1, in_features, dtype=torch.bfloat16)
    got = lm_head.forward(x).float()
    ref = torch.nn.functional.linear(x, embed_weight.to(torch.bfloat16)).float()
    denom = ref.abs().max().clamp_min(1e-3)
    rel = ((got - ref).abs().max() / denom).item()
    assert rel < 0.1, f"int4 lm_head output diverges from tied embedding: rel={rel}"


def test_fill_rejects_shape_mismatch_without_touching_payload():
    lm_head = _make_placeholder(64, 128)
    good_weight = torch.randn(64, 128, dtype=torch.bfloat16)
    assert _fill_xpu_int4_lm_head_from_embedding(lm_head, good_weight) is True
    snapshot = lm_head.qweight.clone()

    bad_weight = torch.randn(32, 64, dtype=torch.bfloat16)
    assert _fill_xpu_int4_lm_head_from_embedding(lm_head, bad_weight) is False
    assert torch.equal(lm_head.qweight, snapshot)


def test_fill_prepares_topk_layout():
    lm_head = _make_placeholder(64, 128)
    assert getattr(lm_head, "lm_head_qweight", None) is None
    embed_weight = torch.randn(64, 128, dtype=torch.bfloat16)
    _fill_xpu_int4_lm_head_from_embedding(lm_head, embed_weight)
    assert getattr(lm_head, "lm_head_qweight", None) is not None
