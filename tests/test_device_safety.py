from __future__ import annotations

import torch

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
