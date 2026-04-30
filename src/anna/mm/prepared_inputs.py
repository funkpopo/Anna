from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, TypeVar, runtime_checkable

import torch


@dataclass(slots=True)
class PreparedInputs:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_position_ids: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_position_ids: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    input_features: torch.Tensor | None = None
    input_features_mask: torch.Tensor | None = None


@runtime_checkable
class PreparedInputsLike(Protocol):
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    mm_token_type_ids: torch.Tensor | None
    pixel_values: torch.Tensor | None
    image_position_ids: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    video_position_ids: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    input_features: torch.Tensor | None
    input_features_mask: torch.Tensor | None


PreparedInputsT = TypeVar("PreparedInputsT", bound=PreparedInputsLike)


def replace_prepared_inputs(prepared: PreparedInputsT, /, **changes) -> PreparedInputsT:
    return replace(prepared, **changes)
