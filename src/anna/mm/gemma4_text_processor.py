from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from anna.mm.media_io import collect_message_media_refs, load_image_pil, load_video_frames, read_media_bytes
from anna.mm.prepared_inputs import PreparedInputs
from anna.weights.gemma4_tokenizer import Gemma4Tokenizer


_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def _unfold_audio(array: np.ndarray, size: int, step: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError("Audio unfold expects a 2D array with shape [batch, time].")
    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1
    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)
    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])
    return np.lib.stride_tricks.as_strided(array, shape=output_shape, strides=output_strides)


def _periodic_hann_window(length: int) -> np.ndarray:
    indices = np.arange(length, dtype=np.float32)
    return 0.5 - (0.5 * np.cos((2.0 * np.pi * indices) / float(length)))


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + (np.asarray(frequency_hz) / 700.0))


def _mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel_value) / 2595.0) - 1.0)


def _build_mel_filter_bank(
    *,
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    mel_min = float(_hz_to_mel(min_frequency))
    mel_max = float(_hz_to_mel(max_frequency))
    mel_points = np.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)
    fft_bins = np.floor(((num_frequency_bins - 1) * hz_points) / (sampling_rate / 2.0)).astype(np.int64)
    fft_bins = np.clip(fft_bins, 0, num_frequency_bins - 1)

    filters = np.zeros((num_frequency_bins, num_mel_filters), dtype=np.float64)
    for mel_idx in range(num_mel_filters):
        left = int(fft_bins[mel_idx])
        center = int(fft_bins[mel_idx + 1])
        right = int(fft_bins[mel_idx + 2])
        if center == left:
            center = min(center + 1, num_frequency_bins - 1)
        if right == center:
            right = min(right + 1, num_frequency_bins)
        if center > left:
            filters[left:center, mel_idx] = (
                np.arange(left, center, dtype=np.float64) - float(left)
            ) / float(center - left)
        if right > center:
            filters[center:right, mel_idx] = (
                float(right) - np.arange(center, right, dtype=np.float64)
            ) / float(right - center)
    return filters.astype(np.float32)


def _get_aspect_ratio_preserving_size(
    *,
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_multiple = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_multiple)) * side_multiple
    target_width = int(math.floor(ideal_width / side_multiple)) * side_multiple

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0x0 image. Check patch_size and pooling_kernel_size against the media size."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if target_height == 0:
        target_height = side_multiple
        target_width = min(int(math.floor(width / height)) * side_multiple, max_side_length)
    elif target_width == 0:
        target_width = side_multiple
        target_height = min(int(math.floor(height / width)) * side_multiple, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds the patch budget."
        )
    return target_height, target_width


def _convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


def _pad_image_patches(
    patches: np.ndarray,
    positions: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_length = patches.shape[0]
    padding_length = target_length - current_length
    if padding_length <= 0:
        return patches, positions
    patch_paddings = [(0, padding_length)] + [(0, 0)] * (patches.ndim - 1)
    pos_paddings = [(0, padding_length), (0, 0)]
    return (
        np.pad(patches, patch_paddings, mode="constant", constant_values=0),
        np.pad(positions, pos_paddings, mode="constant", constant_values=-1),
    )


def _convert_video_to_patches(video: np.ndarray, patch_size: int) -> np.ndarray:
    num_frames, num_channels, height, width = video.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    patched_video = video.reshape(
        num_frames,
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched_video = patched_video.transpose(0, 2, 4, 3, 5, 1)
    return patched_video.reshape(num_frames, num_patches_height * num_patches_width, -1)


def _pad_video_patches(
    patches: np.ndarray,
    positions: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_length = patches.shape[1]
    padding_length = target_length - current_length
    if padding_length <= 0:
        return patches, positions
    patch_paddings = [(0, 0), (0, padding_length)] + [(0, 0)] * (patches.ndim - 2)
    pos_paddings = [(0, 0), (0, padding_length), (0, 0)]
    return (
        np.pad(patches, patch_paddings, mode="constant", constant_values=0),
        np.pad(positions, pos_paddings, mode="constant", constant_values=-1),
    )


@dataclass(slots=True)
class _Gemma4ImageSettings:
    patch_size: int = 16
    max_soft_tokens: int = 280
    pooling_kernel_size: int = 3
    rescale_factor: float = 1.0 / 255.0
    image_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "_Gemma4ImageSettings":
        source = data or {}
        max_soft_tokens = int(source.get("max_soft_tokens", source.get("image_seq_length", 280)))
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"Unsupported Gemma4 image max_soft_tokens: {max_soft_tokens}")
        return cls(
            patch_size=int(source.get("patch_size", 16)),
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=int(source.get("pooling_kernel_size", 3)),
            rescale_factor=float(source.get("rescale_factor", 1.0 / 255.0)),
            image_mean=tuple(float(value) for value in source.get("image_mean", [0.0, 0.0, 0.0])),
            image_std=tuple(float(value) for value in source.get("image_std", [1.0, 1.0, 1.0])),
        )


@dataclass(slots=True)
class _Gemma4VideoSettings:
    patch_size: int = 16
    max_soft_tokens: int = 70
    pooling_kernel_size: int = 3
    num_frames: int = 32
    do_sample_frames: bool = True
    rescale_factor: float = 1.0 / 255.0
    image_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "_Gemma4VideoSettings":
        source = data or {}
        max_soft_tokens = int(source.get("max_soft_tokens", 70))
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"Unsupported Gemma4 video max_soft_tokens: {max_soft_tokens}")
        return cls(
            patch_size=int(source.get("patch_size", 16)),
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=int(source.get("pooling_kernel_size", 3)),
            num_frames=int(source.get("num_frames", 32)),
            do_sample_frames=bool(source.get("do_sample_frames", True)),
            rescale_factor=float(source.get("rescale_factor", 1.0 / 255.0)),
            image_mean=tuple(float(value) for value in source.get("image_mean", [0.0, 0.0, 0.0])),
            image_std=tuple(float(value) for value in source.get("image_std", [1.0, 1.0, 1.0])),
        )


@dataclass(slots=True)
class _Gemma4AudioSettings:
    feature_size: int = 128
    sampling_rate: int = 16_000
    frame_length_ms: float = 20.0
    hop_length_ms: float = 10.0
    min_frequency: float = 0.0
    max_frequency: float = 8_000.0
    preemphasis: float = 0.0
    preemphasis_htk_flavor: bool = True
    fft_overdrive: bool = False
    dither: float = 0.0
    input_scale_factor: float = 1.0
    mel_floor: float = 1e-3
    audio_seq_length: int = 750
    audio_ms_per_token: int = 40

    @classmethod
    def from_dict(
        cls,
        processor_config: dict[str, Any] | None,
        feature_extractor_config: dict[str, Any] | None,
    ) -> "_Gemma4AudioSettings":
        processor_source = processor_config or {}
        source = feature_extractor_config or {}
        return cls(
            feature_size=int(source.get("feature_size", 128)),
            sampling_rate=int(source.get("sampling_rate", 16_000)),
            frame_length_ms=float(source.get("frame_length", 320)) / 16.0,
            hop_length_ms=float(source.get("hop_length", 160)) / 16.0,
            min_frequency=float(source.get("min_frequency", 0.0)),
            max_frequency=float(source.get("max_frequency", 8_000.0)),
            preemphasis=float(source.get("preemphasis", 0.0)),
            preemphasis_htk_flavor=bool(source.get("preemphasis_htk_flavor", True)),
            fft_overdrive=bool(source.get("fft_overdrive", False)),
            dither=float(source.get("dither", 0.0)),
            input_scale_factor=float(source.get("input_scale_factor", 1.0)),
            mel_floor=float(source.get("mel_floor", 1e-3)),
            audio_seq_length=int(processor_source.get("audio_seq_length", 750)),
            audio_ms_per_token=int(processor_source.get("audio_ms_per_token", 40)),
        )

    @property
    def frame_length(self) -> int:
        return int(round(self.sampling_rate * self.frame_length_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sampling_rate * self.hop_length_ms / 1000.0))


class _Gemma4AudioFeatureExtractor:
    def __init__(self, config: _Gemma4AudioSettings):
        self.config = config
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if config.fft_overdrive:
            self.fft_length *= 2
        self.window = _periodic_hann_window(self.frame_length)
        self.mel_filters = _build_mel_filter_bank(
            num_frequency_bins=(self.fft_length // 2) + 1,
            num_mel_filters=config.feature_size,
            min_frequency=config.min_frequency,
            max_frequency=config.max_frequency,
            sampling_rate=config.sampling_rate,
        )

    def _extract_spectrogram(
        self,
        waveform: np.ndarray,
        attention_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if waveform.ndim == 1:
            waveform = waveform[None, :]

        working = waveform.astype(np.float32)
        if self.config.dither > 0.0:
            working = working + (self.config.dither * np.random.randn(*working.shape).astype(np.float32))
        if self.config.input_scale_factor != 1.0:
            working = working * self.config.input_scale_factor

        pad_left = self.frame_length // 2
        working = np.pad(working, ((0, 0), (pad_left, 0)), mode="constant")
        padded_mask = np.pad(attention_mask.astype(np.int32), (pad_left, 0), mode="constant")

        frame_size_for_unfold = self.frame_length + 1
        frames = _unfold_audio(working, frame_size_for_unfold, self.hop_length)
        if frames.shape[1] == 0:
            return (
                np.zeros((0, self.config.feature_size), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        if self.config.preemphasis > 0.0:
            if self.config.preemphasis_htk_flavor:
                first_in_frame = frames[..., :1] * (1.0 - self.config.preemphasis)
                rest_in_frame = frames[..., 1:-1] - (self.config.preemphasis * frames[..., :-2])
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames[..., 1:] - (self.config.preemphasis * frames[..., :-1])
        else:
            frames = frames[..., :-1]

        frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)
        magnitude_spec = np.abs(stft)
        mel_spec = np.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = np.log(mel_spec + self.config.mel_floor)

        mel_spectrogram = log_mel_spec.squeeze(0).astype(np.float32)
        frame_end_indices = np.arange(mel_spectrogram.shape[0]) * self.hop_length + frame_size_for_unfold - 1
        valid_mask = padded_mask[np.clip(frame_end_indices, 0, padded_mask.shape[0] - 1)].astype(bool)
        return mel_spectrogram, valid_mask

    def __call__(
        self,
        waveforms: list[np.ndarray],
        *,
        max_length: int = 480_000,
        pad_to_multiple_of: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_waveforms: list[np.ndarray] = []
        waveform_masks: list[np.ndarray] = []
        target_length = 0
        for waveform in waveforms:
            trimmed = np.asarray(waveform, dtype=np.float32).reshape(-1)[:max_length]
            clipped_waveforms.append(trimmed)
            waveform_masks.append(np.ones((trimmed.shape[0],), dtype=np.int32))
            target_length = max(target_length, trimmed.shape[0])

        if pad_to_multiple_of > 0 and target_length > 0:
            target_length = int(math.ceil(target_length / pad_to_multiple_of) * pad_to_multiple_of)

        spectrograms: list[np.ndarray] = []
        spectrogram_masks: list[np.ndarray] = []
        max_frames = 0
        for waveform, waveform_mask in zip(clipped_waveforms, waveform_masks):
            padded_waveform = np.pad(
                waveform,
                (0, max(0, target_length - waveform.shape[0])),
                mode="constant",
                constant_values=0.0,
            )
            padded_mask = np.pad(
                waveform_mask,
                (0, max(0, target_length - waveform_mask.shape[0])),
                mode="constant",
                constant_values=0,
            )
            spectrogram, spectrogram_mask = self._extract_spectrogram(padded_waveform, padded_mask)
            spectrograms.append(spectrogram)
            spectrogram_masks.append(spectrogram_mask)
            max_frames = max(max_frames, spectrogram.shape[0])

        padded_spectrograms: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        feature_size = self.config.feature_size
        for spectrogram, spectrogram_mask in zip(spectrograms, spectrogram_masks):
            padded_spec = np.zeros((max_frames, feature_size), dtype=np.float32)
            padded_spec[: spectrogram.shape[0], :] = spectrogram
            padded_mask = np.zeros((max_frames,), dtype=bool)
            padded_mask[: spectrogram_mask.shape[0]] = spectrogram_mask
            padded_spectrograms.append(torch.from_numpy(padded_spec))
            padded_masks.append(torch.from_numpy(padded_mask))

        if not padded_spectrograms:
            return (
                torch.zeros((0, 0, feature_size), dtype=torch.float32),
                torch.zeros((0, 0), dtype=torch.bool),
            )
        return torch.stack(padded_spectrograms, dim=0), torch.stack(padded_masks, dim=0)


class Gemma4TextProcessor:
    _load_image = staticmethod(load_image_pil)
    _load_video = staticmethod(load_video_frames)

    def __init__(
        self,
        tokenizer: Gemma4Tokenizer,
        *,
        image_settings: _Gemma4ImageSettings | None = None,
        video_settings: _Gemma4VideoSettings | None = None,
        audio_settings: _Gemma4AudioSettings | None = None,
    ):
        self.tokenizer = tokenizer
        self.image_settings = image_settings or _Gemma4ImageSettings()
        self.video_settings = video_settings or _Gemma4VideoSettings()
        self.audio_settings = audio_settings or _Gemma4AudioSettings()
        self.audio_feature_extractor = _Gemma4AudioFeatureExtractor(self.audio_settings)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        tokenizer: Gemma4Tokenizer | None = None,
    ) -> "Gemma4TextProcessor":
        model_path = Path(model_dir)
        processor_config_path = model_path / "processor_config.json"
        processor_config = (
            json.loads(processor_config_path.read_text(encoding="utf-8"))
            if processor_config_path.exists()
            else {}
        )
        return cls(
            tokenizer=tokenizer or Gemma4Tokenizer.from_model_dir(model_path),
            image_settings=_Gemma4ImageSettings.from_dict(processor_config.get("image_processor")),
            video_settings=_Gemma4VideoSettings.from_dict(processor_config.get("video_processor")),
            audio_settings=_Gemma4AudioSettings.from_dict(
                processor_config,
                processor_config.get("feature_extractor"),
            ),
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
        image_position_ids: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
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
            image_position_ids=self._move_tensor(image_position_ids, device=resolved_device, dtype=torch.long),
            pixel_values_videos=self._move_tensor(pixel_values_videos, device=resolved_device, dtype=tensor_dtype),
            video_position_ids=self._move_tensor(video_position_ids, device=resolved_device, dtype=torch.long),
            input_features=self._move_tensor(input_features, device=resolved_device, dtype=tensor_dtype),
            input_features_mask=self._move_tensor(input_features_mask, device=resolved_device, dtype=torch.bool),
        )

    def encode_text(
        self,
        prompt: str,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> PreparedInputs:
        return self._build_prepared_inputs(prompt=prompt, tensor_device=tensor_device)

    def prepare_messages(
        self,
        messages: list[Any],
        *,
        enable_thinking: bool = False,
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
        audios = collect_message_media_refs(messages, "audio_url")

        pixel_values = image_position_ids = None
        pixel_values_videos = video_position_ids = None
        input_features = input_features_mask = None

        if images:
            loaded_images = [load_image_pil(reference) for reference in images]
            pixel_values, image_position_ids, num_soft_tokens = self.preprocess_images(loaded_images)
            prompt = self._expand_image_placeholders(prompt, num_soft_tokens)

        if videos:
            loaded_videos = [load_video_frames(reference) for reference in videos]
            sampled_frames: list[list[Image.Image]] = []
            sampled_timestamps: list[list[float]] = []
            for frames, fps in loaded_videos:
                frames_for_model, timestamps = self._sample_video_frames(frames, fps)
                sampled_frames.append(frames_for_model)
                sampled_timestamps.append(timestamps)
            pixel_values_videos, video_position_ids, num_video_tokens = self.preprocess_videos(sampled_frames)
            prompt = self._expand_video_placeholders(prompt, num_video_tokens, sampled_timestamps)

        if audios:
            loaded_audio = [self._load_audio(reference) for reference in audios]
            waveforms: list[np.ndarray] = []
            for waveform, sampling_rate in loaded_audio:
                if sampling_rate != self.audio_settings.sampling_rate:
                    waveform = self._resample_audio(
                        waveform,
                        source_rate=sampling_rate,
                        target_rate=self.audio_settings.sampling_rate,
                    )
                waveforms.append(waveform)
            input_features, input_features_mask, num_audio_tokens = self.preprocess_audio(waveforms)
            prompt = self._expand_audio_placeholders(prompt, num_audio_tokens)

        return self._build_prepared_inputs(
            prompt=prompt,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            pixel_values_videos=pixel_values_videos,
            video_position_ids=video_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            tensor_device=tensor_device,
            tensor_dtype=tensor_dtype,
        )

    def _load_audio(self, media_ref: Any) -> tuple[np.ndarray, int]:
        audio_bytes = read_media_bytes(media_ref)
        waveform, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform.reshape(-1), int(sampling_rate)

    def _resample_audio(
        self,
        waveform: np.ndarray,
        *,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate == target_rate or waveform.size == 0:
            return waveform.astype(np.float32, copy=False)
        target_length = max(1, int(round((waveform.shape[0] * target_rate) / float(source_rate))))
        source_positions = np.linspace(0.0, 1.0, waveform.shape[0], endpoint=False, dtype=np.float64)
        target_positions = np.linspace(0.0, 1.0, target_length, endpoint=False, dtype=np.float64)
        resampled = np.interp(target_positions, source_positions, waveform.astype(np.float64))
        return resampled.astype(np.float32)

    @staticmethod
    def _resize_pil(image: Image.Image, resized_height: int, resized_width: int) -> Image.Image:
        return image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    @staticmethod
    def _image_to_chw_array(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

    def _rescale_image(
        self,
        image_array: np.ndarray,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        rescale_factor: float,
    ) -> np.ndarray:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        return ((image_array * rescale_factor) - mean_arr) / std_arr

    def preprocess_images(
        self,
        images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        settings = self.image_settings
        max_patches = settings.max_soft_tokens * settings.pooling_kernel_size**2
        pixel_values: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        num_soft_tokens_per_image: list[int] = []

        for image in images:
            target_height, target_width = _get_aspect_ratio_preserving_size(
                height=image.height,
                width=image.width,
                patch_size=settings.patch_size,
                max_patches=max_patches,
                pooling_kernel_size=settings.pooling_kernel_size,
            )
            resized = self._resize_pil(image, target_height, target_width)
            image_array = self._rescale_image(
                self._image_to_chw_array(resized),
                settings.image_mean,
                settings.image_std,
                settings.rescale_factor,
            )
            patches = _convert_image_to_patches(image_array, settings.patch_size)
            patch_height = image_array.shape[-2] // settings.patch_size
            patch_width = image_array.shape[-1] // settings.patch_size
            num_soft_tokens_per_image.append(patches.shape[0] // settings.pooling_kernel_size**2)

            grid_x, grid_y = np.meshgrid(
                np.arange(patch_width, dtype=np.int64),
                np.arange(patch_height, dtype=np.int64),
                indexing="xy",
            )
            positions = np.stack([grid_x, grid_y], axis=-1).reshape(patches.shape[0], 2)
            patches, positions = _pad_image_patches(patches, positions, max_patches)
            pixel_values.append(torch.from_numpy(patches.astype(np.float32)))
            position_ids.append(torch.from_numpy(positions.astype(np.int64)))

        return (
            torch.stack(pixel_values, dim=0),
            torch.stack(position_ids, dim=0),
            num_soft_tokens_per_image,
        )

    def _sample_video_frames(
        self,
        frames: list[Image.Image],
        fps: float,
    ) -> tuple[list[Image.Image], list[float]]:
        if not frames:
            raise ValueError("Decoded video contained zero frames.")
        if not self.video_settings.do_sample_frames:
            timestamps = [index / max(fps, 1e-6) for index in range(len(frames))]
            return frames, timestamps

        target_frames = max(1, int(self.video_settings.num_frames))
        if len(frames) >= target_frames:
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=np.int64).tolist()
            sampled_frames = [frames[index] for index in indices]
            timestamps = [index / max(fps, 1e-6) for index in indices]
            return sampled_frames, timestamps

        sampled_frames = list(frames)
        timestamps = [index / max(fps, 1e-6) for index in range(len(frames))]
        while len(sampled_frames) < target_frames:
            sampled_frames.append(sampled_frames[-1])
            timestamps.append(timestamps[-1] if timestamps else 0.0)
        return sampled_frames, timestamps

    def preprocess_videos(
        self,
        videos: list[list[Image.Image]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        settings = self.video_settings
        max_patches = settings.max_soft_tokens * settings.pooling_kernel_size**2
        pixel_values: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        num_soft_tokens_per_video: list[int] = []

        for frames in videos:
            first = frames[0]
            target_height, target_width = _get_aspect_ratio_preserving_size(
                height=first.height,
                width=first.width,
                patch_size=settings.patch_size,
                max_patches=max_patches,
                pooling_kernel_size=settings.pooling_kernel_size,
            )
            frame_arrays = [
                self._rescale_image(
                    self._image_to_chw_array(self._resize_pil(frame, target_height, target_width)),
                    settings.image_mean,
                    settings.image_std,
                    settings.rescale_factor,
                )
                for frame in frames
            ]
            video_array = np.stack(frame_arrays, axis=0)
            patches = _convert_video_to_patches(video_array, settings.patch_size)
            patch_height = video_array.shape[-2] // settings.patch_size
            patch_width = video_array.shape[-1] // settings.patch_size
            num_soft_tokens_per_video.append(patches.shape[1] // settings.pooling_kernel_size**2)

            grid_x, grid_y = np.meshgrid(
                np.arange(patch_width, dtype=np.int64),
                np.arange(patch_height, dtype=np.int64),
                indexing="xy",
            )
            frame_positions = np.stack([grid_x, grid_y], axis=-1).reshape(patches.shape[1], 2)
            positions = np.repeat(frame_positions[None, :, :], patches.shape[0], axis=0)
            patches, positions = _pad_video_patches(patches, positions, max_patches)
            pixel_values.append(torch.from_numpy(patches.astype(np.float32)))
            position_ids.append(torch.from_numpy(positions.astype(np.int64)))

        return (
            torch.stack(pixel_values, dim=0),
            torch.stack(position_ids, dim=0),
            num_soft_tokens_per_video,
        )

    def preprocess_audio(
        self,
        waveforms: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        input_features, input_features_mask = self.audio_feature_extractor(waveforms)
        num_audio_tokens = [
            self._compute_audio_num_tokens(waveform, self.audio_settings.sampling_rate)
            for waveform in waveforms
        ]
        return input_features, input_features_mask, num_audio_tokens

    def _compute_audio_num_tokens(self, waveform: np.ndarray, sampling_rate: int) -> int:
        num_samples = int(np.asarray(waveform).reshape(-1).shape[0])
        frame_length = int(round(sampling_rate * 20.0 / 1000.0))
        hop_length = int(round(sampling_rate * 10.0 / 1000.0))
        frame_size_for_unfold = frame_length + 1
        pad_left = frame_length // 2
        padded_samples = num_samples + pad_left
        num_mel_frames = (padded_samples - frame_size_for_unfold) // hop_length + 1
        if num_mel_frames <= 0:
            return 0
        token_frames = num_mel_frames
        for _ in range(2):
            token_frames = ((token_frames + 2 - 3) // 2) + 1
        return min(token_frames, self.audio_settings.audio_seq_length)

    def _expand_image_placeholders(self, text: str, num_soft_tokens_per_image: list[int]) -> str:
        for token_count in num_soft_tokens_per_image:
            replacement = f"{self.tokenizer.boi_token}{self.tokenizer.image_token * token_count}{self.tokenizer.eoi_token}"
            text = text.replace(self.tokenizer.image_token, replacement, 1)
        return text

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        whole = int(seconds)
        return f"{whole // 60:02d}:{whole % 60:02d}"

    def _expand_video_placeholders(
        self,
        text: str,
        num_soft_tokens_per_video: list[int],
        video_timestamps: list[list[float]],
    ) -> str:
        for token_count, timestamps in zip(num_soft_tokens_per_video, video_timestamps):
            repeated_video_tokens = self.tokenizer.video_token * token_count
            frames = [
                f"{self._format_timestamp(timestamp)} {self.tokenizer.boi_token}{repeated_video_tokens}{self.tokenizer.eoi_token}"
                for timestamp in timestamps
            ]
            text = text.replace(self.tokenizer.video_token, " ".join(frames), 1)
        return text

    def _expand_audio_placeholders(self, text: str, num_audio_tokens: list[int]) -> str:
        for token_count in num_audio_tokens:
            replacement = f"{self.tokenizer.boa_token}{self.tokenizer.audio_token * token_count}{self.tokenizer.eoa_token}"
            text = text.replace(self.tokenizer.audio_token, replacement, 1)
        return text

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
        if self.tokenizer.audio_token_id is not None:
            mm_token_type_ids = torch.where(
                input_ids == self.tokenizer.audio_token_id,
                torch.full_like(mm_token_type_ids, 3),
                mm_token_type_ids,
            )
        return mm_token_type_ids
