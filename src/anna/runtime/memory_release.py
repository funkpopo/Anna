from __future__ import annotations

import gc

import torch


def release_conversion_artifacts(device: torch.device) -> None:
    for _ in range(2):
        gc.collect()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
