from __future__ import annotations

from collections.abc import Mapping

import torch


def safetensors_pt_device_str(tensor_targets: Mapping[str, torch.Tensor | None]) -> str:
    devices: set[torch.device] = set()
    for tensor in tensor_targets.values():
        if isinstance(tensor, torch.Tensor):
            devices.add(tensor.device)
    if len(devices) != 1:
        return "cpu"
    device = next(iter(devices))
    if device.type in {"cuda", "xpu"}:
        return str(device)
    return "cpu"
