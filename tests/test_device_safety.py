from __future__ import annotations

import pytest
import torch

import anna.runtime.device as runtime_device
from anna.mm.gemma4_text_processor import PreparedInputs as GemmaPreparedInputs
from anna.runtime.device import DeviceContext
from anna.mm.qwen3_5_text_processor import PreparedInputs


def test_classify_runtime_error_out_of_memory() -> None:
    exc = RuntimeError("XPU out of memory while trying to allocate tensor")
    assert DeviceContext.classify_runtime_error(exc) == "out_of_memory"


def test_classify_runtime_error_device_lost() -> None:
    exc = RuntimeError("level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)")
    assert DeviceContext.classify_runtime_error(exc) == "device_lost"


def test_element_size_matches_dtype() -> None:
    context = DeviceContext.resolve(device="cpu", dtype="fp16", model_dtype="float16")
    assert context.element_size(torch.float16) == 2


def test_move_prepared_inputs_returns_same_object_when_tensors_already_match_target() -> None:
    context = DeviceContext.resolve(device="cpu", dtype="fp32", model_dtype="float32")
    prepared = PreparedInputs(
        prompt="test",
        input_ids=torch.ones((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 4), dtype=torch.int32),
    )

    moved = context.move_prepared_inputs(prepared)

    assert moved is prepared


def test_move_prepared_inputs_preserves_gemma_prepared_input_type() -> None:
    context = DeviceContext.resolve(device="cpu", dtype="fp32", model_dtype="float32")
    prepared = GemmaPreparedInputs(
        prompt="gemma",
        input_ids=torch.ones((1, 3), dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 3), dtype=torch.int32),
        input_features=torch.randn((1, 2, 4), dtype=torch.float32),
        input_features_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    moved = context.move_prepared_inputs(prepared)

    assert isinstance(moved, GemmaPreparedInputs)
    assert moved.__class__.__module__ == "anna.mm.gemma4_text_processor"


def test_device_context_rejects_fp8_dtype_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype alias"):
        DeviceContext.resolve(device="cpu", dtype="fp8", model_dtype="bfloat16")


def test_device_context_rejects_float8_model_dtype_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "has_xpu", lambda: True)
    with pytest.raises(ValueError, match="Unsupported dtype alias"):
        DeviceContext.resolve(device="xpu", dtype="auto", model_dtype="float8_e4m3fn")
