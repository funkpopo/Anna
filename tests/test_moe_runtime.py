from __future__ import annotations

import torch

from anna.model.ops import Qwen3SparseMoeBlock
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig


class _FakeXPUDevice:
    type = "xpu"


def _tiny_moe_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )


def _large_arc_moe_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=128,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
    )


def test_sparse_moe_reuses_staging_module_slots() -> None:
    torch.manual_seed(0)
    block = Qwen3SparseMoeBlock(_tiny_moe_config())
    block.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_experts=False,
        expert_quant="none",
        cached_experts_per_layer=1,
    )

    first = block._get_cached_expert(0)
    second = block._get_cached_expert(1)

    assert first is not None
    assert second is not None
    assert id(first) == id(second)
    assert list(block._expert_cache.keys()) == [1]
    assert torch.allclose(second.gate_proj.weight, block.experts[1].gate_proj.weight)


def test_sparse_moe_offload_cpu_path_matches_baseline_output() -> None:
    torch.manual_seed(0)
    baseline = Qwen3SparseMoeBlock(_tiny_moe_config())
    offloaded = Qwen3SparseMoeBlock(_tiny_moe_config())
    offloaded.load_state_dict(baseline.state_dict())

    baseline.configure_runtime(
        torch.device("cpu"),
        offload_experts=False,
        resident_experts=False,
        expert_quant="none",
        cached_experts_per_layer=0,
    )
    offloaded.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_experts=False,
        expert_quant="none",
        cached_experts_per_layer=1,
    )

    hidden_states = torch.randn(2, 3, 16)
    expected, expected_logits = baseline(hidden_states)
    actual, actual_logits = offloaded(hidden_states)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.allclose(actual_logits, expected_logits, atol=1e-5, rtol=1e-4)


def test_sparse_moe_large_arc_budget_prefers_resident_experts(monkeypatch) -> None:
    block = Qwen3SparseMoeBlock(_large_arc_moe_config())
    with torch.no_grad():
        block.gate.weight.copy_(torch.arange(1, block.num_experts + 1, dtype=torch.float32).unsqueeze(1).repeat(1, 16))

    resident_materialized: list[int] = []
    staged_materialized: list[int] = []
    staged_slots: list[object] = []
    monkeypatch.setattr(block, "_pin_module_host_memory", lambda _module: False)
    monkeypatch.setattr(block, "_materialize_resident_expert", lambda expert_idx: resident_materialized.append(expert_idx))
    monkeypatch.setattr(
        block,
        "_allocate_staging_module",
        lambda _source: staged_slots.append(object()) or staged_slots[-1],
    )
    monkeypatch.setattr(
        block,
        "_materialize_cached_expert",
        lambda expert_idx, reuse_module=None: staged_materialized.append(expert_idx) or reuse_module or block.experts[expert_idx],
    )

    block.configure_runtime(
        _FakeXPUDevice(),  # type: ignore[arg-type]
        offload_experts=True,
        resident_experts=False,
        expert_quant="int4",
        cached_experts_per_layer=40,
    )

    assert block.resident_experts_per_layer == 24
    assert block.staged_experts_per_layer == 16
    assert set(block._resident_expert_indices) == set(range(104, 128))
    assert resident_materialized == list(range(127, 103, -1))
    assert staged_materialized == []
    assert len(block._staging_modules) == 16
    assert len(block._free_staging_modules) == 16
    assert list(block._expert_cache.keys()) == []

    cached = block._get_cached_expert(103)

    assert cached is staged_slots[-1]
    assert staged_materialized == [103]
    assert list(block._expert_cache.keys()) == [103]
    assert len(block._free_staging_modules) == 15
