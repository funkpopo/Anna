import pytest
import torch

from anna.model.fused_ops import maybe_load_gated_delta_library
from anna.model.ops import torch_recurrent_gated_delta_rule


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required for the SYCL custom op test")
def test_gated_delta_fused_xpu_matches_reference() -> None:
    if not maybe_load_gated_delta_library():
        pytest.skip("Anna fused-op library is not built")

    torch.manual_seed(0)
    device = "xpu"
    query = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    key = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    value = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    g = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(2, 3, 4, device=device, dtype=torch.float16))
    z = torch.randn(2, 3, 4, 8, device=device, dtype=torch.float16)
    norm_weight = torch.randn(8, device=device, dtype=torch.float16)
    initial_state = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float32)

    output, final_state = torch.ops.anna.gated_delta_fused(
        query,
        key,
        value,
        g,
        beta,
        z,
        norm_weight,
        1e-6,
        initial_state,
        True,
    )

    ref_core, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    ref_hidden = ref_core.reshape(-1, value.shape[-1]).float()
    ref_z = z.reshape(-1, value.shape[-1]).float()
    ref_hidden = ref_hidden * torch.rsqrt(ref_hidden.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref_hidden = norm_weight * ref_hidden.to(dtype=ref_core.dtype)
    ref_output = (ref_hidden * torch.nn.functional.silu(ref_z)).to(dtype=ref_core.dtype).reshape(
        query.shape[0],
        query.shape[1],
        -1,
    )

    torch.xpu.synchronize()
    assert torch.allclose(output.float().cpu(), ref_output.float().cpu(), atol=2e-2, rtol=2e-2)
    assert final_state is not None
    assert torch.allclose(final_state.float().cpu(), ref_state.float().cpu(), atol=2e-2, rtol=2e-2)
