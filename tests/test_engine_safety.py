from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from anna.mm.processor import PreparedInputs
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


def test_validate_generation_request_respects_configured_max_model_len() -> None:
    engine = object.__new__(AnnaEngine)
    engine.config = SimpleNamespace(
        text_config=SimpleNamespace(
            max_position_embeddings=128,
        )
    )
    engine.max_model_len = 32

    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 24), dtype=torch.long),
        attention_mask=torch.ones((1, 24), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 24), dtype=torch.int32),
    )

    with pytest.raises(AnnaEngineError) as exc_info:
        engine._validate_generation_request(prepared, config=GenerationConfig(max_new_tokens=16))

    assert exc_info.value.code == "context_length_exceeded"
    assert "configured context limit 32" in str(exc_info.value)
