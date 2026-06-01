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
    greedy_rows: tuple[int, ...]
    sample_rows: tuple[int, ...]
    top_p_rows: tuple[int, ...]
    top1_rows: tuple[int, ...]
    topk_rows: tuple[int, ...]
    topk_plain_rows: tuple[int, ...]
    topk_top_p_rows: tuple[int, ...]
    full_plain_rows: tuple[int, ...]
    full_top_p_rows: tuple[int, ...]
    candidate_plain_rows: tuple[int, ...]
    candidate_top_p_rows: tuple[int, ...]
    penalty_rows: tuple[int, ...]
    greedy_indices: torch.Tensor
    top1_indices: torch.Tensor
    topk_plain_indices: torch.Tensor
    topk_top_p_indices: torch.Tensor
    full_plain_indices: torch.Tensor
    full_top_p_indices: torch.Tensor
    candidate_plain_indices: torch.Tensor
    candidate_top_p_indices: torch.Tensor
    penalty_indices: torch.Tensor

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
        greedy_rows = tuple(idx for idx, temperature in enumerate(temperatures) if temperature <= 0.0)
        sample_rows = tuple(idx for idx, temperature in enumerate(temperatures) if temperature > 0.0)
        top_p_rows = tuple(idx for idx in sample_rows if top_ps[idx] < 1.0)
        top1_rows = tuple(idx for idx in sample_rows if top_ks[idx] == 1)
        topk_rows = tuple(idx for idx in sample_rows if top_ks[idx] > 1)
        topk_plain_rows = tuple(idx for idx in topk_rows if top_ps[idx] >= 1.0)
        topk_top_p_rows = tuple(idx for idx in topk_rows if top_ps[idx] < 1.0)
        full_plain_rows = tuple(idx for idx in sample_rows if top_ks[idx] <= 0 and top_ps[idx] >= 1.0)
        full_top_p_rows = tuple(idx for idx in sample_rows if top_ks[idx] <= 0 and top_ps[idx] < 1.0)
        candidate_plain_rows = tuple(idx for idx in sample_rows if top_ks[idx] != 1 and top_ps[idx] >= 1.0)
        candidate_top_p_rows = tuple(idx for idx in sample_rows if top_ks[idx] != 1 and top_ps[idx] < 1.0)
        penalty_rows = tuple(
            idx
            for idx, (presence_penalty, repetition_penalty) in enumerate(
                zip(presence_penalties, repetition_penalties, strict=True)
            )
            if presence_penalty != 0.0 or repetition_penalty != 1.0
        )
        resolved_device = torch.device(device)

        def _indices(rows: tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(rows, dtype=torch.long, device=resolved_device)

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
            greedy_rows=greedy_rows,
            sample_rows=sample_rows,
            top_p_rows=top_p_rows,
            top1_rows=top1_rows,
            topk_rows=topk_rows,
            topk_plain_rows=topk_plain_rows,
            topk_top_p_rows=topk_top_p_rows,
            full_plain_rows=full_plain_rows,
            full_top_p_rows=full_top_p_rows,
            candidate_plain_rows=candidate_plain_rows,
            candidate_top_p_rows=candidate_top_p_rows,
            penalty_rows=penalty_rows,
            greedy_indices=_indices(greedy_rows),
            top1_indices=_indices(top1_rows),
            topk_plain_indices=_indices(topk_plain_rows),
            topk_top_p_indices=_indices(topk_top_p_rows),
            full_plain_indices=_indices(full_plain_rows),
            full_top_p_indices=_indices(full_top_p_rows),
            candidate_plain_indices=_indices(candidate_plain_rows),
            candidate_top_p_indices=_indices(candidate_top_p_rows),
            penalty_indices=_indices(penalty_rows),
        )
