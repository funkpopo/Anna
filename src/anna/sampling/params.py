from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


def _param_value(params: Any | None, name: str, default: float | int) -> float | int:
    if params is None:
        return default
    if isinstance(params, dict):
        value = params.get(name, default)
    else:
        value = getattr(params, name, default)
    return default if value is None else value


@dataclass(frozen=True, slots=True)
class SamplingBatchParams:
    """Per-row sampling parameter tensors for slot decode batches.

    The tensor fields are the model-runner contract used by future GPU sampler
    kernels. The tuple mirrors keep the current Python fallback free from scalar
    tensor reads while that kernel path is still being built.
    """

    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    min_p: torch.Tensor
    presence_penalty: torch.Tensor
    repetition_penalty: torch.Tensor
    temperature_values: tuple[float, ...]
    top_p_values: tuple[float, ...]
    top_k_values: tuple[int, ...]
    min_p_values: tuple[float, ...]
    presence_penalty_values: tuple[float, ...]
    repetition_penalty_values: tuple[float, ...]

    @property
    def batch_size(self) -> int:
        return len(self.temperature_values)

    @classmethod
    def from_sampling_params(
        cls,
        sampling_params: Sequence[Any | None],
        *,
        device: torch.device | str,
    ) -> "SamplingBatchParams":
        temperatures = tuple(float(_param_value(params, "temperature", 0.7)) for params in sampling_params)
        top_ps = tuple(float(_param_value(params, "top_p", 0.8)) for params in sampling_params)
        top_ks = tuple(max(0, int(_param_value(params, "top_k", 20))) for params in sampling_params)
        min_ps = tuple(float(_param_value(params, "min_p", 0.0)) for params in sampling_params)
        presence_penalties = tuple(float(_param_value(params, "presence_penalty", 1.5)) for params in sampling_params)
        repetition_penalties = tuple(float(_param_value(params, "repetition_penalty", 1.0)) for params in sampling_params)
        resolved_device = torch.device(device)
        return cls(
            temperature=torch.tensor(temperatures, dtype=torch.float32, device=resolved_device),
            top_p=torch.tensor(top_ps, dtype=torch.float32, device=resolved_device),
            top_k=torch.tensor(top_ks, dtype=torch.int32, device=resolved_device),
            min_p=torch.tensor(min_ps, dtype=torch.float32, device=resolved_device),
            presence_penalty=torch.tensor(presence_penalties, dtype=torch.float32, device=resolved_device),
            repetition_penalty=torch.tensor(repetition_penalties, dtype=torch.float32, device=resolved_device),
            temperature_values=temperatures,
            top_p_values=top_ps,
            top_k_values=top_ks,
            min_p_values=min_ps,
            presence_penalty_values=presence_penalties,
            repetition_penalty_values=repetition_penalties,
        )
