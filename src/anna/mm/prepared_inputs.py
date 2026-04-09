from __future__ import annotations

from dataclasses import replace
from typing import Protocol, TypeVar, runtime_checkable

import torch


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
