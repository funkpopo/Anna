from __future__ import annotations

import pytest
import torch

import anna.runtime.device as runtime_device
from anna.mm.prepared_inputs import PreparedInputs
from anna.runtime.device import DeviceContext


def test_classify_runtime_error_out_of_memory() -> None:
    exc = RuntimeError("XPU out of memory while trying to allocate tensor")
    assert DeviceContext.classify_runtime_error(exc) == "out_of_memory"


def test_classify_runtime_error_device_lost() -> None:
    exc = RuntimeError("level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)")
    assert DeviceContext.classify_runtime_error(exc) == "device_lost"


def test_element_size_matches_dtype() -> None:
    context = DeviceContext.resolve(device="cpu", dtype="fp16", model_dtype="float16")
    assert context.element_size(torch.float16) == 2
    assert context.xpu_info is None


def test_inspect_xpu_device_classifies_arc_a770(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeXPU:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Intel(R) Arc(TM) A770 Graphics"

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return 8 << 30, 16 << 30

        @staticmethod
        def memory_allocated(device: torch.device) -> int:
            assert device.type == "xpu"
            return 123

        @staticmethod
        def memory_reserved(device: torch.device) -> int:
            assert device.type == "xpu"
            return 456

    monkeypatch.setattr(runtime_device.torch, "xpu", FakeXPU, raising=False)

    info = runtime_device.inspect_xpu_device(torch.device("xpu"))

    assert info is not None
    assert info.name == "Intel(R) Arc(TM) A770 Graphics"
    assert info.total_memory == 16 << 30
    assert info.free_memory == 8 << 30
    assert info.allocated_memory == 123
    assert info.reserved_memory == 456
    assert info.is_arc_alchemist is True
    assert info.is_acm_g10 is True
    assert info.is_arc_a770_or_a750 is True


def test_configure_xpu_environment_sets_level_zero_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", raising=False)
    monkeypatch.delenv("ZES_ENABLE_SYSMAN", raising=False)
    monkeypatch.delenv("ONEAPI_DEVICE_SELECTOR", raising=False)

    configured = runtime_device.configure_xpu_environment(device_index=1, set_selector=True)

    assert configured["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] == "1"
    assert configured["ZES_ENABLE_SYSMAN"] == "1"
    assert configured["ONEAPI_DEVICE_SELECTOR"] == "level_zero:1"


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


def test_move_prepared_inputs_preserves_prepared_input_type() -> None:
    context = DeviceContext.resolve(device="cpu", dtype="fp32", model_dtype="float32")
    prepared = PreparedInputs(
        prompt="gemma",
        input_ids=torch.ones((1, 3), dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 3), dtype=torch.int32),
        input_features=torch.randn((1, 2, 4), dtype=torch.float32),
        input_features_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    moved = context.move_prepared_inputs(prepared)

    assert isinstance(moved, PreparedInputs)
    assert moved.__class__.__module__ == "anna.mm.prepared_inputs"


def test_device_context_rejects_fp8_dtype_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype alias"):
        DeviceContext.resolve(device="cpu", dtype="fp8", model_dtype="bfloat16")


def test_device_context_rejects_float8_model_dtype_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_device, "has_xpu", lambda: True)
    with pytest.raises(ValueError, match="Unsupported dtype alias"):
        DeviceContext.resolve(device="xpu", dtype="auto", model_dtype="float8_e4m3fn")
