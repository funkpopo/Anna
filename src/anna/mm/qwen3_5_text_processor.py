from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from anna.mm.media_io import collect_message_media_refs, load_image_pil, load_video_frames
from anna.mm.prepared_inputs import PreparedInputs
from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig, VisionPreprocessorConfig
from anna.weights.qwen3_5_text_tokenizer import Qwen3_5TextTokenizer


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen3_5TextMultimodalProcessor:
    _load_image = staticmethod(load_image_pil)
    _load_video = staticmethod(load_video_frames)

    def __init__(self, config: Qwen3_5TextModelConfig, tokenizer: Qwen3_5TextTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.preprocessor = config.preprocessor_config

    def encode_text(
        self,
        prompt: str,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> PreparedInputs:
        return self._build_prepared_inputs(
            prompt=prompt,
            tensor_device=tensor_device,
        )

    def prepare_messages(
        self,
        messages: list[Any],
        *,
        enable_thinking: bool = True,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        tensor_device: torch.device | str | None = None,
        tensor_dtype: torch.dtype | None = None,
    ) -> PreparedInputs:
        render_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools is not None or tool_choice is not None or parallel_tool_calls is not None:
            render_kwargs.update(
                {
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls,
                }
            )
        prompt = self.tokenizer.render_messages(messages, **render_kwargs)
        images = collect_message_media_refs(messages, "image_url")
        videos = collect_message_media_refs(messages, "video_url")
        pixel_values = image_grid_thw = None
        pixel_values_videos = video_grid_thw = None
        video_fps: list[float] = []

        if images:
            pil_images = [load_image_pil(media_ref) for media_ref in images]
            pixel_values, image_grid_thw = self.preprocess_images(pil_images)
            prompt = self._expand_image_placeholders(prompt, image_grid_thw)

        if videos:
            decoded = [load_video_frames(media_ref) for media_ref in videos]
            video_frames = [frames for frames, _ in decoded]
            video_fps = [fps for _, fps in decoded]
            pixel_values_videos, video_grid_thw = self.preprocess_videos(video_frames)
            prompt = self._expand_video_placeholders(prompt, video_grid_thw, video_fps)

        return self._build_prepared_inputs(
            prompt=prompt,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            tensor_device=tensor_device,
            tensor_dtype=tensor_dtype,
        )

    @staticmethod
    def _resolve_tensor_device(tensor_device: torch.device | str | None) -> torch.device | None:
        if tensor_device is None:
            return None
        return tensor_device if isinstance(tensor_device, torch.device) else torch.device(tensor_device)

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor | None,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if tensor is None or device is None:
            return tensor
        if tensor.device == device and (dtype is None or tensor.dtype == dtype):
            return tensor
        kwargs: dict[str, object] = {"device": device, "non_blocking": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    def _build_prepared_inputs(
        self,
        *,
        prompt: str,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        tensor_device: torch.device | str | None = None,
        tensor_dtype: torch.dtype | None = None,
    ) -> PreparedInputs:
        resolved_device = self._resolve_tensor_device(tensor_device)
        tensor_kwargs = {} if resolved_device is None else {"device": resolved_device}
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, **tensor_kwargs)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        mm_token_type_ids = self._create_mm_token_type_ids(input_ids)
        return PreparedInputs(
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=self._move_tensor(pixel_values, device=resolved_device, dtype=tensor_dtype),
            image_grid_thw=self._move_tensor(image_grid_thw, device=resolved_device, dtype=torch.long),
            pixel_values_videos=self._move_tensor(pixel_values_videos, device=resolved_device, dtype=tensor_dtype),
            video_grid_thw=self._move_tensor(video_grid_thw, device=resolved_device, dtype=torch.long),
        )

    def _resize_pil(self, image: Image.Image, resized_height: int, resized_width: int) -> Image.Image:
        return image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(self.preprocessor.image_mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.preprocessor.image_std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def preprocess_images(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        processed: list[torch.Tensor] = []
        grids: list[list[int]] = []
        patch_size = self.preprocessor.patch_size
        temporal_patch_size = self.preprocessor.temporal_patch_size
        merge_size = self.preprocessor.merge_size
        factor = patch_size * merge_size

        for image in images:
            resized_height, resized_width = smart_resize(
                image.height,
                image.width,
                factor=factor,
                min_pixels=self.preprocessor.shortest_edge,
                max_pixels=self.preprocessor.longest_edge,
            )
            image_tensor = self._image_to_tensor(self._resize_pil(image, resized_height, resized_width))
            patches = image_tensor.unsqueeze(0).unsqueeze(0)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            processed_item, grid = self._flatten_media_patches(patches)
            processed.append(processed_item)
            grids.append(grid)

        return torch.cat(processed, dim=0), torch.tensor(grids, dtype=torch.long)

    def preprocess_videos(self, videos: list[list[Image.Image]]) -> tuple[torch.Tensor, torch.Tensor]:
        processed: list[torch.Tensor] = []
        grids: list[list[int]] = []
        patch_size = self.preprocessor.patch_size
        merge_size = self.preprocessor.merge_size
        factor = patch_size * merge_size

        for frames in videos:
            first = frames[0]
            resized_height, resized_width = smart_resize(
                first.height,
                first.width,
                factor=factor,
                min_pixels=self.preprocessor.shortest_edge,
                max_pixels=self.preprocessor.longest_edge,
            )
            frame_tensors = [self._image_to_tensor(self._resize_pil(frame, resized_height, resized_width)) for frame in frames]
            patches = torch.stack(frame_tensors, dim=0).unsqueeze(0)
            processed_item, grid = self._flatten_media_patches(patches)
            processed.append(processed_item)
            grids.append(grid)

        return torch.cat(processed, dim=0), torch.tensor(grids, dtype=torch.long)

    def _flatten_media_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        temporal_patch_size = self.preprocessor.temporal_patch_size
        patch_size = self.preprocessor.patch_size
        merge_size = self.preprocessor.merge_size

        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)

        batch_size, grid_t, channel, resized_height, resized_width = patches.shape
        grid_t = grid_t // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches.squeeze(0), [grid_t, grid_h, grid_w]

    def _expand_image_placeholders(self, text: str, image_grid_thw: torch.Tensor) -> str:
        merge_length = self.preprocessor.merge_size**2
        index = 0
        while self.tokenizer.image_token in text:
            num_image_tokens = int(image_grid_thw[index].prod().item() // merge_length)
            text = text.replace(self.tokenizer.image_token, "<|placeholder|>" * num_image_tokens, 1)
            index += 1
        return text.replace("<|placeholder|>", self.tokenizer.image_token)

    def _expand_video_placeholders(self, text: str, video_grid_thw: torch.Tensor, video_fps: list[float]) -> str:
        merge_length = self.preprocessor.merge_size**2
        temporal_patch_size = self.preprocessor.temporal_patch_size
        index = 0
        while self.tokenizer.video_token in text:
            frame_seqlen = int(video_grid_thw[index][1:].prod().item() // merge_length)
            grid_t = int(video_grid_thw[index][0].item())
            fps = video_fps[index] if index < len(video_fps) else 24.0
            placeholder = []
            for frame_idx in range(grid_t):
                curr_time = (frame_idx * temporal_patch_size) / max(fps, 1e-6)
                placeholder.append(f"<{curr_time:.1f} seconds>")
                placeholder.append(
                    self.tokenizer.vision_start_token
                    + ("<|placeholder|>" * frame_seqlen)
                    + self.tokenizer.vision_end_token
                )
            compound = "".join(placeholder)
            target = f"{self.tokenizer.vision_start_token}{self.tokenizer.video_token}{self.tokenizer.vision_end_token}"
            if target in text:
                text = text.replace(target, compound, 1)
            else:
                text = text.replace(self.tokenizer.video_token, compound, 1)
            index += 1
        return text.replace("<|placeholder|>", self.tokenizer.video_token)

    def _create_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        if self.tokenizer.image_token_id is not None:
            mm_token_type_ids = torch.where(
                input_ids == self.tokenizer.image_token_id,
                torch.ones_like(mm_token_type_ids),
                mm_token_type_ids,
            )
        if self.tokenizer.video_token_id is not None:
            mm_token_type_ids = torch.where(
                input_ids == self.tokenizer.video_token_id,
                torch.full_like(mm_token_type_ids, 2),
                mm_token_type_ids,
            )
        return mm_token_type_ids
