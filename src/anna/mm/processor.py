from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from anna.model.config import Qwen3Config, VisionPreprocessorConfig
from anna.weights.tokenizer import QwenTokenizer


@dataclass(slots=True)
class PreparedInputs:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None


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


class Qwen3MultimodalProcessor:
    def __init__(self, config: Qwen3Config, tokenizer: QwenTokenizer):
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
        tensor_device: torch.device | str | None = None,
        tensor_dtype: torch.dtype | None = None,
    ) -> PreparedInputs:
        prompt = self.tokenizer.render_messages(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        images = self._collect_media(messages, "image_url")
        videos = self._collect_media(messages, "video_url")
        pixel_values = image_grid_thw = None
        pixel_values_videos = video_grid_thw = None
        video_fps: list[float] = []

        if images:
            pil_images = [self._load_image(media_ref) for media_ref in images]
            pixel_values, image_grid_thw = self.preprocess_images(pil_images)
            prompt = self._expand_image_placeholders(prompt, image_grid_thw)

        if videos:
            decoded = [self._load_video(media_ref) for media_ref in videos]
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

    def _collect_media(self, messages: list[Any], part_type: str) -> list[Any]:
        refs: list[Any] = []
        for message in messages:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if hasattr(item, "type"):
                    item_type = getattr(item, "type")
                    value = getattr(item, part_type, None)
                else:
                    item_type = item.get("type")
                    value = item.get(part_type)
                if item_type == part_type:
                    refs.append(value)
        return refs

    def _resolve_media_url(self, ref: Any) -> str:
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            url = ref.get("url")
            if not url:
                raise ValueError("Media URL object is missing the 'url' field.")
            return url
        raise ValueError("Unsupported media reference format.")

    def _read_bytes(self, media_ref: Any) -> bytes:
        url = self._resolve_media_url(media_ref)
        if url.startswith("data:"):
            _, payload = url.split(",", 1)
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urllib.request.urlopen(url) as response:
                return response.read()
        return Path(url).read_bytes()

    def _load_image(self, media_ref: Any) -> Image.Image:
        return Image.open(io.BytesIO(self._read_bytes(media_ref))).convert("RGB")

    def _load_video(self, media_ref: Any) -> tuple[list[Image.Image], float]:
        try:
            import imageio
        except Exception as exc:  # pragma: no cover - dependency availability is environment-specific
            raise RuntimeError("Video input requires imageio with ffmpeg support installed.") from exc

        url = self._resolve_media_url(media_ref)
        temp_path: str | None = None
        if url.startswith(("http://", "https://", "data:")):
            suffix = Path(urllib.parse.urlparse(url).path).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(self._read_bytes(media_ref))
                temp_path = handle.name
            video_path = temp_path
        else:
            video_path = url

        reader = None
        try:
            reader = imageio.get_reader(video_path, format="FFMPEG")
            metadata = reader.get_meta_data()
            fps = float(metadata.get("fps") or 24.0)
            frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
        finally:
            if reader is not None:
                reader.close()
            if temp_path is not None:
                os.unlink(temp_path)
        if not frames:
            raise ValueError("Decoded video contained zero frames.")
        return frames, fps

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
