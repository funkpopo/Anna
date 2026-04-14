from __future__ import annotations

from types import SimpleNamespace

import torch

from anna.mm.qwen3_5_text_processor import PreparedInputs
from anna.model.qwen3_5_text_config import QuantizationConfig, Qwen3_5TextConfig, RopeParameters
from anna.model.quantization import estimate_module_xpu_int4_bytes
from anna.model.qwen3_5_text_model import Qwen3_5TextForCausalLM
from anna.model.ops import Qwen3SparseMoeBlock
from anna.runtime.device import DeviceMemoryInfo, RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import AnnaEngineError, AnnaQwen3_5TextEngine, GenerationConfig


def test_guard_generation_memory_rejects_oversized_request() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
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


def test_guard_generation_memory_rejects_when_projected_total_exceeds_usage_ratio() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.device_context = SimpleNamespace(
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=64 << 20,
            reserve_margin_bytes=32 << 20,
            max_estimated_usage_ratio=0.9,
            generation_memory_safety_factor=1.0,
        ),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=300 << 20,
            total_bytes=1024 << 20,
            allocated_bytes=724 << 20,
            reserved_bytes=0,
        ),
    )
    engine._estimate_generation_memory_bytes = lambda prepared, config: 250 << 20

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 8), dtype=torch.long),
        attention_mask=torch.ones((1, 8), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 8), dtype=torch.int32),
    )

    try:
        engine._guard_generation_memory(prepared, config=GenerationConfig(max_new_tokens=16))
    except AnnaEngineError as exc:
        assert exc.code == "estimated_device_oom"
        assert "projected_total" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected projected-total memory guard to reject the request.")


def test_validate_generation_request_uses_remaining_context_when_no_memory_info_is_available() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.config = SimpleNamespace(
        text_config=SimpleNamespace(
            max_position_embeddings=32,
        )
    )
    engine.device_context = SimpleNamespace(get_memory_info=lambda: None)

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 8), dtype=torch.long),
        attention_mask=torch.ones((1, 8), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 8), dtype=torch.int32),
    )

    prompt_ids, prompt_length, resolved = engine._validate_generation_request(
        prepared,
        config=GenerationConfig(),
    )

    assert len(prompt_ids) == 8
    assert prompt_length == 8
    assert resolved.max_new_tokens == 24


def test_move_prepared_for_generation_prunes_trivial_attention_mask_before_device_transfer() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    seen_attention_masks: list[torch.Tensor | None] = []
    engine._guard_generation_memory = lambda prepared, config: None
    engine.device_context = SimpleNamespace(
        move_prepared_inputs=lambda prepared: seen_attention_masks.append(prepared.attention_mask) or prepared,
    )

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 4), dtype=torch.int32),
    )

    moved = engine._move_prepared_for_generation(prepared, config=GenerationConfig(max_new_tokens=8))

    assert moved.attention_mask is None
    assert seen_attention_masks == [None]


def test_validate_generation_request_auto_resolves_memory_bounded_limit() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
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
            max_position_embeddings=8192,
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
            free_bytes=1536 << 20,
            total_bytes=2048 << 20,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 1024), dtype=torch.long),
        attention_mask=torch.ones((1, 1024), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 1024), dtype=torch.int32),
    )

    _, prompt_length, resolved = engine._validate_generation_request(
        prepared,
        config=GenerationConfig(),
    )

    assert prompt_length == 1024
    assert resolved.max_new_tokens is not None
    assert 1 <= resolved.max_new_tokens < (8192 - prompt_length)


def test_auto_resident_expert_estimation_uses_conservative_free_memory_budget() -> None:
    config = Qwen3_5TextConfig(
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
    model = Qwen3_5TextForCausalLM(config)
    layer_bytes = AnnaQwen3_5TextEngine._module_nbytes(model.model.layers[0].mlp.experts)
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

    selected = AnnaQwen3_5TextEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=fake_device_context,
        expert_quant="none",
    )

    assert budget_bytes >= layer_bytes > 0
    assert selected == (0, 1)


def test_auto_resident_expert_estimation_accounts_for_int4_expert_storage() -> None:
    config = Qwen3_5TextConfig(
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
    model = Qwen3_5TextForCausalLM(config)
    dense_layer_bytes = AnnaQwen3_5TextEngine._module_nbytes(model.model.layers[0].mlp.experts)
    int4_layer_bytes = estimate_module_xpu_int4_bytes(model.model.layers[0].mlp.experts)
    dense_device_context = SimpleNamespace(
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
    int4_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=0,
            reserve_margin_bytes=0,
            max_estimated_usage_ratio=1.0,
            generation_memory_safety_factor=2.0,
        ),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=(2304 << 20) + int4_layer_bytes * 3,
            total_bytes=(2304 << 20) + int4_layer_bytes * 3 + 1024,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    selected_dense = AnnaQwen3_5TextEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=dense_device_context,
        expert_quant="none",
    )
    selected_int4 = AnnaQwen3_5TextEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=int4_device_context,
        expert_quant="int4",
    )

    assert dense_layer_bytes > int4_layer_bytes > 0
    assert selected_dense == (0,)
    assert len(selected_int4) >= 2


def test_auto_resident_expert_estimation_disables_large_arc_moe_working_sets() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention"],
        decoder_sparse_step=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=128,
        num_experts_per_tok=8,
    )
    model = Qwen3_5TextForCausalLM(config)
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
            free_bytes=8 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    selected = AnnaQwen3_5TextEngine._estimate_resident_expert_layer_indices(
        model=model,
        device_context=fake_device_context,
        expert_quant="int4",
    )

    assert selected == ()


def test_auto_cached_experts_per_layer_scales_with_available_xpu_budget() -> None:
    config = Qwen3_5TextConfig(
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
    model = Qwen3_5TextForCausalLM(config)
    model.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_expert_layer_indices=(1,),
        expert_quant="int4",
        cached_experts_per_layer=0,
    )

    exemplar_expert_bytes = estimate_module_xpu_int4_bytes(model.model.layers[0].mlp.experts[0])
    offloaded_layer_count = 2
    total_bytes = 16 << 30
    target_free_bytes = max(
        1024 << 20,
        512 << 20,
        int(total_bytes * 0.1),
        1536 << 20,
        int(total_bytes * 0.12),
    )
    desired_cached = 3
    cache_budget_bytes = exemplar_expert_bytes * offloaded_layer_count * desired_cached
    free_bytes = int((cache_budget_bytes + exemplar_expert_bytes * offloaded_layer_count) / 0.25) + target_free_bytes

    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    estimated = AnnaQwen3_5TextEngine._estimate_cached_experts_per_layer(
        model=model,
        device_context=fake_device_context,
        expert_quant="int4",
    )

    assert estimated >= desired_cached
    assert estimated <= config.num_experts


def test_auto_cached_experts_per_layer_caps_large_moe_working_set() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=2048,
        intermediate_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        vocab_size=256,
        max_position_embeddings=4096,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        num_experts_per_tok=8,
    )
    model = Qwen3_5TextForCausalLM(config)
    model.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_expert_layer_indices=(),
        expert_quant="int4",
        cached_experts_per_layer=0,
    )

    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=256 << 20,
            reserve_margin_bytes=128 << 20,
            max_estimated_usage_ratio=0.95,
            generation_memory_safety_factor=1.25,
        ),
        synchronize=lambda: None,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=12 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    estimated = AnnaQwen3_5TextEngine._estimate_cached_experts_per_layer(
        model=model,
        device_context=fake_device_context,
        expert_quant="int4",
    )

    assert estimated == 96


def test_auto_weight_quantization_promotes_oversized_dense_xpu_models(monkeypatch) -> None:
    config = SimpleNamespace(
        text_config=Qwen3_5TextConfig(
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
        ),
        quantization_config=QuantizationConfig(),
        vision_config=None,
    )
    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=16 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )
    monkeypatch.setattr("anna.runtime.qwen3_5_text_engine.estimate_qwen3_5_text_model_weight_bytes", lambda _model_path: 15 << 30)

    resolved = AnnaQwen3_5TextEngine._resolve_weight_quant(
        requested_quant="auto",
        resolved_offload_mode="none",
        model_path=SimpleNamespace(),
        config=config,
        device_context=fake_device_context,
    )

    assert resolved == "int4"


def test_auto_weight_quantization_can_promote_oversized_expert_offload_models(monkeypatch) -> None:
    config = SimpleNamespace(
        text_config=Qwen3_5TextConfig(
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
        ),
        quantization_config=QuantizationConfig(),
        vision_config=None,
    )
    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=16 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )
    monkeypatch.setattr("anna.runtime.qwen3_5_text_engine.estimate_qwen3_5_text_model_weight_bytes", lambda _model_path: 14 << 30)

    resolved = AnnaQwen3_5TextEngine._resolve_weight_quant(
        requested_quant="auto",
        resolved_offload_mode="experts",
        model_path=SimpleNamespace(),
        config=config,
        device_context=fake_device_context,
    )

    assert resolved == "int4"


def test_prequantized_arc_moe_models_use_runtime_xpu_int4_backend_adaptation() -> None:
    config = SimpleNamespace(
        text_config=Qwen3_5TextConfig(
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
            decoder_sparse_step=1,
            moe_intermediate_size=96,
            shared_expert_intermediate_size=128,
            num_experts=128,
            num_experts_per_tok=8,
        ),
        quantization_config=QuantizationConfig(
            quant_method="auto-round",
            bits=4,
            group_size=128,
        ),
        vision_config=None,
    )
    fake_device_context = SimpleNamespace(
        device=torch.device("xpu"),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=16 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    resolved = AnnaQwen3_5TextEngine._resolve_weight_quant(
        requested_quant="auto",
        resolved_offload_mode="experts",
        model_path=SimpleNamespace(),
        config=config,
        device_context=fake_device_context,
    )

    assert resolved == "int4"


def test_large_arc_moe_runtime_enables_token_io_offload() -> None:
    config = SimpleNamespace(
        text_config=Qwen3_5TextConfig(
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
            decoder_sparse_step=1,
            moe_intermediate_size=96,
            shared_expert_intermediate_size=128,
            num_experts=128,
            num_experts_per_tok=8,
        ),
        quantization_config=QuantizationConfig(
            quant_method="auto-round",
            bits=4,
            group_size=128,
        ),
        vision_config=None,
    )
    fake_device_context = SimpleNamespace(device=torch.device("xpu"))

    resolved = AnnaQwen3_5TextEngine._resolve_offload_token_io(
        config=config,
        resolved_offload_mode="experts",
        resolved_weight_quant="int4",
        device_context=fake_device_context,
    )

    assert resolved is True


def test_large_arc_moe_auto_prefill_chunking_is_capped_for_hybrid_execution() -> None:
    block_config = Qwen3_5TextConfig(
        hidden_size=4096,
        intermediate_size=8192,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        vocab_size=256,
        max_position_embeddings=4096,
        layer_types=["linear_attention"],
        decoder_sparse_step=1,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
        num_experts=128,
        num_experts_per_tok=8,
    )
    block = Qwen3SparseMoeBlock(block_config)
    block.offload_experts = True
    block.execution_device = torch.device("xpu")
    block.expert_quant = "int4"
    block.resident_experts_per_layer = 80
    block.staged_experts_per_layer = 16

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]))
    engine.config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=40,
            num_key_value_heads=8,
            head_dim=128,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_num_value_heads=16,
            linear_value_head_dim=128,
            layer_types=["full_attention"] * 10 + ["linear_attention"] * 30,
        )
    )
    engine.full_attention_cache_mirror = False
    engine.device_context = SimpleNamespace(
        device=torch.device("xpu"),
        dtype=torch.bfloat16,
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=256 << 20,
            reserve_margin_bytes=128 << 20,
            max_estimated_usage_ratio=0.95,
            generation_memory_safety_factor=1.25,
        ),
        element_size=lambda dtype=None: 2,
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=13 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    resolved = engine._resolve_prefill_chunk_size(0)

    assert resolved == 512
