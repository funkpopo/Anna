"""Phase 3: chunked flash prefill attention parity + int4 auto-strategy tests."""

from __future__ import annotations

import pytest
import torch

from anna.model.ops import chunked_flash_gqa_attention, grouped_query_attention
from anna.model.quantization import XPUInt4Linear


def _reference_attention(query, key, value, *, scaling, causal_mask, visible_mask, key_padding_mask):
    """fp32 materialized reference (grouped_query_attention semantics)."""
    batch_size, num_heads, query_len, _ = query.shape
    num_key_value_heads = key.shape[1]
    grouped_q = query.unflatten(1, (num_key_value_heads, num_heads // num_key_value_heads))
    scores = torch.matmul(grouped_q, key.unsqueeze(2).transpose(-1, -2)) * scaling
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask[:, None, None, :, :], float("-inf"))
    if visible_mask is not None:
        scores = scores.masked_fill(~visible_mask[:, None, None, None, :], float("-inf"))
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, value.unsqueeze(2))
    return out.reshape(batch_size, num_heads, query_len, -1)


@pytest.mark.parametrize("kv_chunk_size", (64, 256, 512))
@pytest.mark.parametrize("batch_size,num_heads,num_kv_heads,query_len,kv_len", [
    (1, 4, 2, 8, 128),
    (2, 4, 1, 16, 300),
    (1, 2, 2, 32, 1024),
])
def test_flash_chunked_parity_with_materialized_path(
    batch_size, num_heads, num_kv_heads, query_len, kv_len, kv_chunk_size
) -> None:
    torch.manual_seed(7)
    head_dim = 16
    query = torch.randn(batch_size, num_heads, query_len, head_dim)
    key = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)
    value = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)
    scaling = head_dim**-0.5

    # Cached-prefill style masks: causal over (past + current) region.
    past_lengths = torch.full((batch_size,), kv_len - query_len, dtype=torch.long)
    q_positions = past_lengths[:, None] + torch.arange(query_len)[None, :]
    k_positions = torch.arange(kv_len)[None, None, :]
    causal_mask = k_positions > q_positions[:, :, None]  # True = masked
    visible_mask = k_positions.reshape(1, kv_len) < (past_lengths + query_len)[:, None]
    key_padding_mask = None

    reference = _reference_attention(
        query, key, value, scaling=scaling, causal_mask=causal_mask,
        visible_mask=visible_mask, key_padding_mask=key_padding_mask,
    )
    flash = chunked_flash_gqa_attention(
        query, key, value, scaling=scaling, causal_mask=causal_mask,
        visible_mask=visible_mask, key_padding_mask=key_padding_mask,
        kv_chunk_size=kv_chunk_size,
    )
    # fp32 end-to-end: online softmax must match the materialized softmax tightly.
    assert torch.allclose(reference, flash, atol=1e-5, rtol=1e-4), \
        (reference - flash).abs().max().item()


def test_flash_chunked_padding_mask_parity() -> None:
    torch.manual_seed(11)
    batch_size, num_heads, num_kv_heads, query_len, kv_len = 2, 2, 1, 4, 64
    head_dim = 8
    query = torch.randn(batch_size, num_heads, query_len, head_dim)
    key = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)
    value = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)
    scaling = head_dim**-0.5

    key_padding = torch.ones(batch_size, kv_len, dtype=torch.bool)
    key_padding[0, 40:] = False  # left padding mask on row 0
    key_padding[1, 55:] = False

    reference = _reference_attention(
        query, key, value, scaling=scaling, causal_mask=None,
        visible_mask=None, key_padding_mask=key_padding,
    )
    flash = chunked_flash_gqa_attention(
        query, key, value, scaling=scaling, causal_mask=None,
        visible_mask=None, key_padding_mask=key_padding,
        kv_chunk_size=16,
    )
    assert torch.allclose(reference, flash, atol=1e-5, rtol=1e-4)


def test_flash_chunked_dtype_preserved() -> None:
    query = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 130, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 130, 8, dtype=torch.bfloat16)
    out = chunked_flash_gqa_attention(query, key, value, scaling=0.3, kv_chunk_size=32)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 2, 3, 8)


# ---------------------------------------------------------------------------
# Phase 3: int4 M-aware auto backend selection
# ---------------------------------------------------------------------------


def test_int4_auto_backend_m_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    monkeypatch.delenv("ANNA_XPU_INT4_GEMV_M_THRESHOLD", raising=False)

    # Default threshold is 0: legacy Arc policy (auto = torch int4pack) until
    # the GEMV kernel is validated end-to-end, regardless of op availability.
    assert XPUInt4Linear.resolve_matmul_backend(rows=1) == "torch"
    assert XPUInt4Linear.resolve_matmul_backend(rows=4096) == "torch"


def test_int4_auto_backend_selects_gemv_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import torch as _torch

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    # Phase 3 capability switch: opt in to the decode-shaped GEMV routing.
    monkeypatch.setenv("ANNA_XPU_INT4_GEMV_M_THRESHOLD", "8")

    fake_namespace = types.SimpleNamespace(xpu_int4_gemv=lambda *args, **kwargs: None)
    original_ops = _torch.ops
    monkeypatch.setattr(_torch, "ops", types.SimpleNamespace(anna=fake_namespace), raising=True)
    try:
        # Decode-shaped rows route to the SYCL GEMV.
        assert XPUInt4Linear.resolve_matmul_backend(rows=1) == "gemv"
        assert XPUInt4Linear.resolve_matmul_backend(rows=8) == "gemv"
        # Prefill-shaped rows keep the XMX batched GEMM.
        assert XPUInt4Linear.resolve_matmul_backend(rows=9) == "torch"
        assert XPUInt4Linear.resolve_matmul_backend(rows=512) == "torch"
        # rows=None (no shape info) keeps the conservative torch path.
        assert XPUInt4Linear.resolve_matmul_backend() == "torch"
    finally:
        monkeypatch.undo()
        assert _torch.ops is original_ops


def test_int4_auto_gemv_skipped_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import torch as _torch

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    monkeypatch.setenv("ANNA_XPU_INT4_GEMV_M_THRESHOLD", "8")
    # Op namespace present but gemv op not registered -> keep torch path.
    monkeypatch.setattr(_torch, "ops", types.SimpleNamespace(anna=types.SimpleNamespace()), raising=True)
    assert XPUInt4Linear.resolve_matmul_backend(rows=1) == "torch"


def test_int4_gemv_m_threshold_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import torch as _torch

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    monkeypatch.setenv("ANNA_XPU_INT4_GEMV_M_THRESHOLD", "32")
    fake_namespace = types.SimpleNamespace(xpu_int4_gemv=lambda *args, **kwargs: None)
    monkeypatch.setattr(_torch, "ops", types.SimpleNamespace(anna=fake_namespace), raising=True)
    try:
        assert XPUInt4Linear.resolve_matmul_backend(rows=32) == "gemv"
        assert XPUInt4Linear.resolve_matmul_backend(rows=33) == "torch"
    finally:
        monkeypatch.undo()


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the padded-layout guard test")
def test_int4_auto_gemv_guarded_for_padded_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SYCL GEMV kernel rejects padded inputs; padded layouts keep int4pack."""
    torch.manual_seed(3)
    # in_features=16 pads to 32 (group_size=32) -> GEMV-incompatible layout.
    dense = torch.nn.Linear(16, 31, bias=False, dtype=torch.float32)
    quantized = XPUInt4Linear.from_linear(dense, group_size=32, compute_dtype=torch.float16, device="xpu")
    assert quantized.in_features != quantized.padded_in_features

    def _fail_gemv(*_args, **_kwargs):
        raise AssertionError("gemv must not be selected for padded layouts")

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "auto")
    monkeypatch.setenv("ANNA_XPU_INT4_GEMV_M_THRESHOLD", "8")
    monkeypatch.setattr(quantized, "_forward_xpu_int4_gemv", _fail_gemv)
    hidden = torch.randn(1, 1, 16, device="xpu", dtype=torch.float16)
    out = quantized(hidden)
    assert out.shape == (1, 1, 31)


def test_int4_explicit_strategy_bypasses_m_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "dequant")
    assert XPUInt4Linear.resolve_matmul_backend(rows=1) == "dequant"
    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "torch")
    assert XPUInt4Linear.resolve_matmul_backend(rows=1) == "torch"
