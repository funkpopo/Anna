from __future__ import annotations

from types import SimpleNamespace

import torch

from anna.mm.processor import PreparedInputs
from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.quantization import estimate_module_xpu_int4_bytes
from anna.model.qwen import Qwen3ForCausalLM
from anna.runtime.device import DeviceMemoryInfo, RuntimeSafetyPolicy
from anna.runtime.engine import AnnaEngine, AnnaEngineError, GenerationConfig


def test_guard_generation_memory_rejects_oversized_request() -> None:
    engine = object.__new__(AnnaEngine)
    engine.config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            linear_num_key_heads=8,
            linear_key_head_dim=128,
            linear_num_value_heads=8,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            layer_types=["full_attention"] * 16 + ["linear_attention"] * 16,
        )
    )
    engine.device_context = SimpleNamespace(
        dtype=torch.bfloat16,
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=128 << 20,
            reserve_margin_bytes=64 << 20,
            max_estimated_usage_ratio=0.9,
            generation_memory_safety_factor=2.0,
        ),
        element_size=lambda dtype=None: 2,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=256 << 20,
            total_bytes=1024 << 20,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 4096), dtype=torch.long),
        attention_mask=torch.ones((1, 4096), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 4096), dtype=torch.int32),
    )

    try:
        engine._guard_generation_memory(prepared, config=GenerationConfig(max_new_tokens=2048))
    except AnnaEngineError as exc:
        assert exc.code == "estimated_device_oom"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected memory guard to reject the oversized request.")


def test_auto_resident_expert_estimation_uses_conservative_free_memory_budget() -> None:
    config = Qwen3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
        decoder_sparse_step=1,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[3],
    )
    model = Qwen3ForCausalLM(config)
    layer_bytes = AnnaEngine._module_nbytes(model.model.layers[0].mlp.experts)
    budget_bytes = (layer_bytes * 5) // 2
    free_bytes = budget_bytes * 2

    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=0,
            reserve_margin_bytes=0,
            max_estimated_usage_ratio=1.0,
            generation_memory_safety_factor=2.0,
        ),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=free_bytes,
            total_bytes=free_bytes + 1024,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    selected = AnnaEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=fake_device_context,
        expert_quant="none",
    )

    assert budget_bytes >= layer_bytes > 0
    assert selected == (0, 1)


def test_auto_resident_expert_estimation_accounts_for_int4_expert_storage() -> None:
    config = Qwen3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
        decoder_sparse_step=1,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[3],
    )
    model = Qwen3ForCausalLM(config)
    dense_layer_bytes = AnnaEngine._module_nbytes(model.model.layers[0].mlp.experts)
    int4_layer_bytes = estimate_module_xpu_int4_bytes(model.model.layers[0].mlp.experts)
    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=0,
            reserve_margin_bytes=0,
            max_estimated_usage_ratio=1.0,
            generation_memory_safety_factor=2.0,
        ),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=dense_layer_bytes * 2,
            total_bytes=dense_layer_bytes * 2 + 1024,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    selected_dense = AnnaEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=fake_device_context,
        expert_quant="none",
    )
    selected_int4 = AnnaEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=fake_device_context,
        expert_quant="int4",
    )

    assert dense_layer_bytes > int4_layer_bytes > 0
    assert selected_dense == (0,)
    assert len(selected_int4) >= 2


def test_auto_cached_experts_per_layer_scales_with_available_xpu_budget() -> None:
    config = Qwen3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
        decoder_sparse_step=1,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[3],
    )
    model = Qwen3ForCausalLM(config)
    model.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_expert_layer_indices=(1,),
        expert_quant="int4",
        cached_experts_per_layer=0,
    )

    exemplar_expert_bytes = estimate_module_xpu_int4_bytes(model.model.layers[0].mlp.experts[0])
    offloaded_layer_count = 2
    target_free_bytes = 768 << 20
    desired_cached = 3
    cache_budget_bytes = exemplar_expert_bytes * offloaded_layer_count * desired_cached
    free_bytes = int(cache_budget_bytes / 0.65) + target_free_bytes

    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=free_bytes,
            total_bytes=max(free_bytes + 1024, 16 << 20),
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    estimated = AnnaEngine._estimate_cached_experts_per_layer(
        model=model,
        device_context=fake_device_context,
        expert_quant="int4",
    )

    assert estimated >= desired_cached
    assert estimated <= config.num_experts
