import pytest
import torch
import torch.nn.functional as F

import anna.model.ops as model_ops
from anna.model.fused_ops import (
    maybe_load_gated_delta_library,
    run_flashqla_chunk_local_cumsum,
    run_flashqla_chunk_gdr_fwd,
    run_flashqla_cumsum_kkt_build,
    run_flashqla_kkt_build,
    run_flashqla_kkt_solve,
    run_flashqla_solve_wu_build,
    run_flashqla_wu_build,
)
from anna.model.ops import Qwen3GatedDeltaNet, l2norm, torch_recurrent_gated_delta_rule
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig


def _pad_and_chunk(x: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, int]:
    seq_len = int(x.shape[1])
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
    batch_size, padded_len = int(x.shape[0]), int(x.shape[1])
    return x.reshape(batch_size, padded_len // chunk_size, chunk_size, *x.shape[2:]), seq_len


def _flashqla_reference_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)

    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    g_f = g.float()
    beta_f = beta.float()
    batch_size, seq_len, num_k_heads, head_dim = key_f.shape
    num_v_heads = value_f.shape[2]
    value_dim = value_f.shape[3]
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    if num_k_heads != num_v_heads:
        repeat = num_v_heads // num_k_heads
        query_f = query_f.repeat_interleave(repeat, dim=2)
        key_f = key_f.repeat_interleave(repeat, dim=2)
    scale = head_dim**-0.5

    query_chunks, _ = _pad_and_chunk(query_f, chunk_size=chunk_size)
    key_chunks, _ = _pad_and_chunk(key_f, chunk_size=chunk_size)
    value_chunks, _ = _pad_and_chunk(value_f, chunk_size=chunk_size)
    g_chunks, _ = _pad_and_chunk(g_f.unsqueeze(-1), chunk_size=chunk_size)
    beta_chunks, _ = _pad_and_chunk(beta_f.unsqueeze(-1), chunk_size=chunk_size)
    g_chunks = g_chunks.squeeze(-1).cumsum(dim=2)
    beta_chunks = beta_chunks.squeeze(-1)

    num_chunks = int(query_chunks.shape[1])
    state = (
        torch.zeros(batch_size, num_v_heads, head_dim, value_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    output_chunks = torch.empty(
        batch_size,
        num_chunks,
        chunk_size,
        num_v_heads,
        value_dim,
        device=value.device,
        dtype=torch.float32,
    )

    lower_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=value.device, dtype=torch.bool))
    strict_upper_mask = torch.triu(torch.ones(chunk_size, chunk_size, device=value.device, dtype=torch.bool), diagonal=1)
    identity = torch.eye(chunk_size, device=value.device, dtype=torch.float32)

    for chunk_idx in range(num_chunks):
        q_i = query_chunks[:, chunk_idx]
        k_i = key_chunks[:, chunk_idx]
        v_i = value_chunks[:, chunk_idx]
        g_i = g_chunks[:, chunk_idx]
        beta_i = beta_chunks[:, chunk_idx]

        decay = torch.exp(g_i[:, :, None, :] - g_i[:, None, :, :]).masked_fill(
            ~lower_mask[None, :, :, None],
            0.0,
        )
        a_matrix = torch.einsum("bshk,bthk->bsth", k_i * beta_i.unsqueeze(-1), k_i) * decay
        a_matrix = -a_matrix.permute(0, 3, 1, 2).contiguous()
        a_matrix = a_matrix.masked_fill(torch.triu(torch.ones_like(a_matrix, dtype=torch.bool)), 0.0)
        for row_idx in range(1, chunk_size):
            row = a_matrix[..., row_idx, :row_idx].clone()
            sub = a_matrix[..., :row_idx, :row_idx].clone()
            a_matrix[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)
        a_matrix = a_matrix + identity

        v_beta = v_i * beta_i.unsqueeze(-1)
        v_local = torch.einsum("bhst,bthv->bshv", a_matrix, v_beta)
        k_cumdecay = torch.einsum("bhst,bthk->bshk", a_matrix, k_i * beta_i.unsqueeze(-1) * g_i.exp().unsqueeze(-1))
        v_prime = torch.einsum("bshk,bhkv->bshv", k_cumdecay, state)
        v_new = v_local - v_prime

        local_attn = torch.einsum("bshk,bthk->bsth", q_i * scale, k_i) * decay
        local_attn = local_attn.masked_fill(strict_upper_mask[None, :, :, None], 0.0)
        recurrent_out = torch.einsum("bshk,bhkv->bshv", q_i * scale * g_i.exp().unsqueeze(-1), state)
        output_chunks[:, chunk_idx] = recurrent_out + torch.einsum("bsth,bthv->bshv", local_attn, v_new)

        g_last = g_i[:, -1]
        state = state * g_last.exp().unsqueeze(-1).unsqueeze(-1)
        state = state + torch.einsum(
            "bshk,bshv->bhkv",
            k_i * (g_last[:, None, :, None] - g_i.unsqueeze(-1)).exp(),
            v_new,
        )

    output = output_chunks.reshape(batch_size, -1, num_v_heads, value_dim)[:, :seq_len]
    return output.to(dtype=initial_dtype), state


def _reference_chunk_local_cumsum(g: torch.Tensor, *, chunk_size: int = 64) -> torch.Tensor:
    chunks, seq_len = _pad_and_chunk(g.unsqueeze(-1), chunk_size=chunk_size)
    output = chunks.squeeze(-1).cumsum(dim=2).reshape(g.shape[0], -1, g.shape[2])
    return output[:, :seq_len].contiguous()


def _reference_kkt_build(
    key: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    key_chunks, seq_len = _pad_and_chunk(key.float(), chunk_size=chunk_size)
    beta_chunks, _ = _pad_and_chunk(beta.float().unsqueeze(-1), chunk_size=chunk_size)
    g_chunks, _ = _pad_and_chunk(g_cumsum.float().unsqueeze(-1), chunk_size=chunk_size)
    beta_chunks = beta_chunks.squeeze(-1)
    g_chunks = g_chunks.squeeze(-1)

    batch_size, num_chunks, _, num_heads, _ = key_chunks.shape
    output = torch.zeros(
        batch_size,
        num_chunks,
        chunk_size,
        num_heads,
        chunk_size,
        device=key.device,
        dtype=torch.float32,
    )
    lower_strict = torch.tril(torch.ones(chunk_size, chunk_size, device=key.device, dtype=torch.bool), diagonal=-1)
    for chunk_idx in range(num_chunks):
        k_i = key_chunks[:, chunk_idx]
        beta_i = beta_chunks[:, chunk_idx]
        g_i = g_chunks[:, chunk_idx]
        decay = torch.exp(g_i[:, :, None, :] - g_i[:, None, :, :]).masked_fill(
            ~lower_strict[None, :, :, None],
            0.0,
        )
        output[:, chunk_idx] = (torch.einsum("bshk,bthk->bsth", k_i * beta_i.unsqueeze(-1), k_i) * decay).permute(
            0,
            1,
            3,
            2,
        )
    return output.reshape(batch_size, -1, num_heads, chunk_size)[:, :seq_len].contiguous()


def _reference_kkt_solve(kkt: torch.Tensor, *, chunk_size: int = 64) -> torch.Tensor:
    kkt_chunks, seq_len = _pad_and_chunk(kkt, chunk_size=chunk_size)
    matrix = -kkt_chunks.permute(0, 1, 3, 2, 4).contiguous()
    for row_idx in range(1, chunk_size):
        row = matrix[..., row_idx, :row_idx].clone()
        sub = matrix[..., :row_idx, :row_idx].clone()
        matrix[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)
    matrix = matrix + torch.eye(chunk_size, device=kkt.device, dtype=torch.float32)
    return matrix.permute(0, 1, 3, 2, 4).reshape(kkt.shape[0], -1, kkt.shape[2], chunk_size)[:, :seq_len].contiguous()


def _reference_wu_build(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    a: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_chunks, seq_len = _pad_and_chunk(key.float(), chunk_size=chunk_size)
    value_chunks, _ = _pad_and_chunk(value.float(), chunk_size=chunk_size)
    beta_chunks, _ = _pad_and_chunk(beta.float().unsqueeze(-1), chunk_size=chunk_size)
    g_chunks, _ = _pad_and_chunk(g_cumsum.float().unsqueeze(-1), chunk_size=chunk_size)
    a_chunks, _ = _pad_and_chunk(a.float(), chunk_size=chunk_size)
    beta_chunks = beta_chunks.squeeze(-1)
    g_chunks = g_chunks.squeeze(-1)

    w = torch.einsum(
        "bnshc,bnchk->bnshk",
        a_chunks,
        key_chunks * beta_chunks.unsqueeze(-1) * g_chunks.exp().unsqueeze(-1),
    )
    u = torch.einsum("bnshc,bnchv->bnshv", a_chunks, value_chunks * beta_chunks.unsqueeze(-1))
    return (
        w.reshape(key.shape[0], -1, key.shape[2], key.shape[3])[:, :seq_len].contiguous(),
        u.reshape(value.shape[0], -1, value.shape[2], value.shape[3])[:, :seq_len].contiguous(),
    )


def _reference_chunk_gdr_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    g_cumsum: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Uses the same decomposed equations as FlashQLA after W/U have been built.
    query_f = query.float()
    key_f = key.float()
    batch_size, seq_len, num_heads, key_dim = key_f.shape
    value_dim = u.shape[-1]
    state_f = state.float().clone()
    output = torch.empty(batch_size, seq_len, num_heads, value_dim, device=query.device, dtype=torch.float32)
    scale = key_dim**-0.5
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(seq_len, chunk_start + chunk_size)
        v_new = torch.empty(batch_size, chunk_end - chunk_start, num_heads, value_dim, device=query.device, dtype=torch.float32)
        for t in range(chunk_start, chunk_end):
            row = t - chunk_start
            v_new[:, row] = u[:, t] - torch.einsum("bhk,bhkv->bhv", w[:, t], state_f)
            recurrent = torch.einsum("bhk,bhkv->bhv", query_f[:, t] * scale * g_cumsum[:, t].exp().unsqueeze(-1), state_f)
            local = torch.zeros_like(recurrent)
            for col in range(row + 1):
                source_t = chunk_start + col
                attn = (
                    (query_f[:, t] * key_f[:, source_t]).sum(dim=-1)
                    * scale
                    * torch.exp(g_cumsum[:, t] - g_cumsum[:, source_t])
                )
                local = local + attn.unsqueeze(-1) * v_new[:, col]
            output[:, t] = recurrent + local
        g_last = g_cumsum[:, chunk_end - 1]
        state_f = state_f * g_last.exp().unsqueeze(-1).unsqueeze(-1)
        for row in range(chunk_end - chunk_start):
            source_t = chunk_start + row
            state_f = state_f + torch.einsum(
                "bhk,bhv->bhkv",
                key_f[:, source_t] * torch.exp(g_last - g_cumsum[:, source_t]).unsqueeze(-1),
                v_new[:, row],
            )
    return output.to(dtype=query.dtype), state_f


def test_flashqla_chunk_local_cumsum_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_chunk_local_cumsum_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_chunk_local_cumsum(g=torch.empty(1, 64, 2), chunk_size=64)


def test_flashqla_kkt_build_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_kkt_build_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    key = torch.empty(1, 64, 2, 8)
    beta = torch.empty(1, 64, 2)
    g_cumsum = torch.empty(1, 64, 2)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=g_cumsum, chunk_size=64)


def test_flashqla_cumsum_kkt_build_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_cumsum_kkt_build_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    key = torch.empty(1, 64, 2, 8)
    beta = torch.empty(1, 64, 2)
    g = torch.empty(1, 64, 2)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_cumsum_kkt_build(key=key, beta=beta, g=g, chunk_size=64)


def test_flashqla_kkt_solve_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_kkt_solve_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_kkt_solve(kkt=torch.empty(1, 64, 2, 64), chunk_size=64)


def test_flashqla_wu_build_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_wu_build_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    key = torch.empty(1, 64, 2, 8)
    value = torch.empty(1, 64, 2, 8)
    beta = torch.empty(1, 64, 2)
    g_cumsum = torch.empty(1, 64, 2)
    a = torch.empty(1, 64, 2, 64)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_wu_build(key=key, value=value, beta=beta, g_cumsum=g_cumsum, a=a, chunk_size=64)


def test_flashqla_solve_wu_build_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_solve_wu_build_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    key = torch.empty(1, 64, 2, 8)
    value = torch.empty(1, 64, 2, 8)
    beta = torch.empty(1, 64, 2)
    g_cumsum = torch.empty(1, 64, 2)
    kkt = torch.empty(1, 64, 2, 64)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_solve_wu_build(key=key, value=value, beta=beta, g_cumsum=g_cumsum, kkt=kkt, chunk_size=64)


def test_flashqla_chunk_gdr_fwd_wrapper_raises_without_registered_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import anna.model.fused_ops as fused_ops

    monkeypatch.setattr(fused_ops, "_flashqla_chunk_gdr_fwd_op", lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda: False)
    query = torch.empty(1, 64, 2, 8)
    g_cumsum = torch.empty(1, 64, 2)
    a = torch.empty(1, 64, 2, 64)
    u = torch.empty(1, 64, 2, 8)
    state = torch.empty(1, 2, 8, 8)
    with pytest.raises(RuntimeError, match="does not fall back"):
        fused_ops.run_flashqla_chunk_gdr_fwd(
            query=query,
            key=query,
            g_cumsum=g_cumsum,
            a=a,
            w=query.float(),
            u=u,
            state=state,
            chunk_size=64,
        )


def test_flashqla_decomposed_reference_matches_recurrent_rule() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 64, 2, 16, dtype=torch.float32)
    key = torch.randn(1, 64, 2, 16, dtype=torch.float32)
    value = torch.randn(1, 64, 2, 16, dtype=torch.float32)
    g = torch.randn(1, 64, 2, dtype=torch.float32) * -0.1
    beta = torch.sigmoid(torch.randn(1, 64, 2, dtype=torch.float32))
    initial_state = torch.randn(1, 2, 16, 16, dtype=torch.float32)

    output, final_state = _flashqla_reference_forward(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
    )
    ref_output, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )

    assert ref_state is not None
    assert torch.allclose(output, ref_output, atol=3e-4, rtol=3e-4)
    assert torch.allclose(final_state, ref_state, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_chunk_local_cumsum_xpu_matches_reference(seq_len: int) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_chunk_local_cumsum"):
        pytest.skip("Anna fused-op library is not built with flashqla_chunk_local_cumsum")

    torch.manual_seed(seq_len)
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32)
    output = run_flashqla_chunk_local_cumsum(g=g, chunk_size=64)
    reference = _reference_chunk_local_cumsum(g, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(output.cpu(), reference.cpu(), atol=1e-4, rtol=1e-6)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_kkt_build_xpu_matches_reference(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_kkt_build"):
        pytest.skip("Anna fused-op library is not built with flashqla_kkt_build")

    torch.manual_seed(seq_len + (0 if dtype is torch.float16 else 1000))
    key = l2norm(torch.randn(2, seq_len, 3, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    beta = torch.sigmoid(torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32))
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32) * -0.1
    g_cumsum = _reference_chunk_local_cumsum(g, chunk_size=64)

    output = run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=g_cumsum, chunk_size=64)
    reference = _reference_kkt_build(key, beta, g_cumsum, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(output.cpu(), reference.cpu(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_cumsum_kkt_build_xpu_matches_separate_ops(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_cumsum_kkt_build"):
        pytest.skip("Anna fused-op library is not built with flashqla_cumsum_kkt_build")

    torch.manual_seed(seq_len + (9000 if dtype is torch.float16 else 10000))
    key = l2norm(torch.randn(2, seq_len, 3, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    beta = torch.sigmoid(torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32))
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32) * -0.1

    fused_g, fused_kkt = run_flashqla_cumsum_kkt_build(key=key, beta=beta, g=g, chunk_size=64)
    ref_g = run_flashqla_chunk_local_cumsum(g=g, chunk_size=64)
    ref_kkt = run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=ref_g, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(fused_g.cpu(), ref_g.cpu(), atol=1e-4, rtol=1e-6)
    assert torch.allclose(fused_kkt.cpu(), ref_kkt.cpu(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_kkt_solve_xpu_matches_reference(seq_len: int) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_kkt_solve_64"):
        pytest.skip("Anna fused-op library is not built with flashqla_kkt_solve_64")

    torch.manual_seed(seq_len + 2000)
    key = l2norm(torch.randn(2, seq_len, 3, 8, device="xpu", dtype=torch.float16), dim=-1).contiguous()
    beta = torch.sigmoid(torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32))
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32) * -0.1
    g_cumsum = _reference_chunk_local_cumsum(g, chunk_size=64)
    kkt = run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=g_cumsum, chunk_size=64)

    output = run_flashqla_kkt_solve(kkt=kkt, chunk_size=64)
    reference = _reference_kkt_solve(kkt, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(output.cpu(), reference.cpu(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_wu_build_xpu_matches_reference(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_wu_build"):
        pytest.skip("Anna fused-op library is not built with flashqla_wu_build")

    torch.manual_seed(seq_len + (3000 if dtype is torch.float16 else 4000))
    key = l2norm(torch.randn(2, seq_len, 3, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    value = torch.randn(2, seq_len, 3, 10, device="xpu", dtype=dtype)
    beta = torch.sigmoid(torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32))
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32) * -0.1
    g_cumsum = _reference_chunk_local_cumsum(g, chunk_size=64)
    kkt = run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=g_cumsum, chunk_size=64)
    a = run_flashqla_kkt_solve(kkt=kkt, chunk_size=64)

    w, u = run_flashqla_wu_build(key=key, value=value, beta=beta, g_cumsum=g_cumsum, a=a, chunk_size=64)
    ref_w, ref_u = _reference_wu_build(key, value, beta, g_cumsum, a, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(w.cpu(), ref_w.cpu(), atol=3e-3, rtol=3e-3)
    assert torch.allclose(u.cpu(), ref_u.cpu(), atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_solve_wu_build_xpu_matches_separate_ops(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_solve_wu_build"):
        pytest.skip("Anna fused-op library is not built with flashqla_solve_wu_build")

    torch.manual_seed(seq_len + (11000 if dtype is torch.float16 else 12000))
    key = l2norm(torch.randn(2, seq_len, 3, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    value = torch.randn(2, seq_len, 3, 10, device="xpu", dtype=dtype)
    beta = torch.sigmoid(torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32))
    g = torch.randn(2, seq_len, 3, device="xpu", dtype=torch.float32) * -0.1
    g_cumsum, kkt = run_flashqla_cumsum_kkt_build(key=key, beta=beta, g=g, chunk_size=64)

    fused_a, fused_w, fused_u = run_flashqla_solve_wu_build(
        key=key,
        value=value,
        beta=beta,
        g_cumsum=g_cumsum,
        kkt=kkt,
        chunk_size=64,
    )
    ref_a = run_flashqla_kkt_solve(kkt=kkt, chunk_size=64)
    ref_w, ref_u = run_flashqla_wu_build(key=key, value=value, beta=beta, g_cumsum=g_cumsum, a=ref_a, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(fused_a.cpu(), ref_a.cpu(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(fused_w.cpu(), ref_w.cpu(), atol=3e-3, rtol=3e-3)
    assert torch.allclose(fused_u.cpu(), ref_u.cpu(), atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_chunk_gdr_fwd_xpu_matches_reference(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_chunk_gdr_fwd"):
        pytest.skip("Anna fused-op library is not built with flashqla_chunk_gdr_fwd")

    torch.manual_seed(seq_len + (5000 if dtype is torch.float16 else 6000))
    query = l2norm(torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    key = l2norm(torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype), dim=-1).contiguous()
    value = torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype)
    beta = torch.sigmoid(torch.randn(1, seq_len, 2, device="xpu", dtype=torch.float32))
    g = torch.randn(1, seq_len, 2, device="xpu", dtype=torch.float32) * -0.1
    state = torch.randn(1, 2, 8, 8, device="xpu", dtype=torch.float32)
    g_cumsum = _reference_chunk_local_cumsum(g, chunk_size=64)
    kkt = run_flashqla_kkt_build(key=key, beta=beta, g_cumsum=g_cumsum, chunk_size=64)
    a = run_flashqla_kkt_solve(kkt=kkt, chunk_size=64)
    w, u = run_flashqla_wu_build(key=key, value=value, beta=beta, g_cumsum=g_cumsum, a=a, chunk_size=64)

    output, final_state = run_flashqla_chunk_gdr_fwd(
        query=query,
        key=key,
        g_cumsum=g_cumsum,
        a=a,
        w=w,
        u=u,
        state=state,
        chunk_size=64,
    )
    ref_output, ref_state = _reference_chunk_gdr_fwd(query, key, g_cumsum, a, w, u, state, chunk_size=64)

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_output.float().cpu(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(final_state.cpu(), ref_state.cpu(), atol=3e-2, rtol=3e-2)


def test_flashqla_gdn_prefill_opt_in_raises_on_cpu_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    layer = Qwen3GatedDeltaNet(config, 0)
    query = torch.randn(1, 64, 2, 8)
    key = torch.randn(1, 64, 2, 8)
    value = torch.randn(1, 64, 2, 8)
    g = torch.randn(1, 64, 2)
    beta = torch.sigmoid(torch.randn(1, 64, 2))
    monkeypatch.setenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", "1")

    with pytest.raises(RuntimeError, match="does not fall back"):
        layer._run_chunk_prefill(query, key, value, g, beta, None)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_flashqla_gdn_prefill_opt_in_uses_flashqla_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    layer = Qwen3GatedDeltaNet(config, 0).to("xpu", dtype=torch.float16)
    query = torch.randn(1, 64, 2, 8, device="xpu", dtype=torch.float16)
    key = torch.randn(1, 64, 2, 8, device="xpu", dtype=torch.float16)
    value = torch.randn(1, 64, 2, 8, device="xpu", dtype=torch.float16)
    g = torch.randn(1, 64, 2, device="xpu")
    beta = torch.sigmoid(torch.randn(1, 64, 2, device="xpu"))
    called = {"value": False}

    def _stub_flashqla(**kwargs):
        called["value"] = True
        return torch.empty_like(kwargs["value"]), kwargs["state"]

    monkeypatch.setenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", "1")
    monkeypatch.setattr(model_ops, "run_flashqla_gated_delta_prefill", _stub_flashqla)

    output, final_state = layer._run_chunk_prefill(query, key, value, g, beta, None)

    assert called["value"] is True
    assert output.shape == value.shape
    assert final_state.shape == (1, 2, 8, 8)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [64, 65, 128])
def test_flashqla_gated_delta_prefill_xpu_matches_decomposed_reference(seq_len: int, dtype: torch.dtype) -> None:
    if not maybe_load_gated_delta_library() or not hasattr(torch.ops.anna, "flashqla_gated_delta_prefill"):
        pytest.skip("Anna fused-op library is not built with flashqla_gated_delta_prefill")

    torch.manual_seed(seq_len + (7000 if dtype is torch.float16 else 8000))
    query = torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype)
    key = torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype)
    value = torch.randn(1, seq_len, 2, 8, device="xpu", dtype=dtype)
    g = torch.randn(1, seq_len, 2, device="xpu", dtype=torch.float32) * -0.1
    beta = torch.sigmoid(torch.randn(1, seq_len, 2, device="xpu", dtype=torch.float32))
    initial_state = torch.randn(1, 2, 8, 8, device="xpu", dtype=torch.float32)

    output, final_state = torch.ops.anna.flashqla_gated_delta_prefill(query, key, value, g, beta, initial_state)
    ref_output, ref_state = _flashqla_reference_forward(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
    )

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_output.float().cpu(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=5e-2, rtol=5e-2)
