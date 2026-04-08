from __future__ import annotations

import math

import imageio.v3 as iio
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from anna.mm.gemma4_text_processor import (
    Gemma4TextProcessor,
    _Gemma4AudioSettings,
    _Gemma4ImageSettings,
    _Gemma4VideoSettings,
)
from anna.model.gemma4_config import Gemma4Config
from anna.model.gemma4_text_model import Gemma4ForConditionalGeneration


class _FakeGemmaTokenizer:
    def __init__(self) -> None:
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.sot_token = "<|turn>"
        self.eot_token = "<turn|>"
        self.soc_token = "<|channel>"
        self.eoc_token = "<channel|>"
        self.think_token = "<|think|>"
        self.boi_token = "<|image>"
        self.eoi_token = "<image|>"
        self.image_token = "<|image|>"
        self.boa_token = "<|audio>"
        self.eoa_token = "<audio|>"
        self.audio_token = "<|audio|>"
        self.video_token = "<|video|>"
        self._ids = {
            self.bos_token: 1,
            self.eos_token: 2,
            self.sot_token: 3,
            self.eot_token: 4,
            self.soc_token: 5,
            self.eoc_token: 6,
            self.think_token: 7,
            self.boi_token: 11,
            self.eoi_token: 12,
            self.image_token: 13,
            self.boa_token: 14,
            self.eoa_token: 15,
            self.audio_token: 16,
            self.video_token: 17,
            "<gen>": 18,
        }
        self._next_id = 100
        self._special_tokens = sorted(self._ids, key=len, reverse=True)

    @property
    def image_token_id(self) -> int:
        return self._ids[self.image_token]

    @property
    def video_token_id(self) -> int:
        return self._ids[self.video_token]

    @property
    def audio_token_id(self) -> int:
        return self._ids[self.audio_token]

    def _lookup_or_create(self, token: str) -> int:
        existing = self._ids.get(token)
        if existing is not None:
            return existing
        token_id = self._next_id
        self._next_id += 1
        self._ids[token] = token_id
        self._special_tokens = sorted(self._ids, key=len, reverse=True)
        return token_id

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue

            matched = False
            for token in self._special_tokens:
                if text.startswith(token, cursor):
                    token_ids.append(self._ids[token])
                    cursor += len(token)
                    matched = True
                    break
            if matched:
                continue

            next_special = len(text)
            for token in self._special_tokens:
                index = text.find(token, cursor)
                if index != -1:
                    next_special = min(next_special, index)
            next_whitespace = cursor
            while next_whitespace < len(text) and not text[next_whitespace].isspace():
                next_whitespace += 1
            end = min(next_special, next_whitespace)
            if end <= cursor:
                end = cursor + 1
            token_ids.append(self._lookup_or_create(text[cursor:end]))
            cursor = end
        return token_ids

    def render_messages(
        self,
        messages: list[dict[str, object]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        del enable_thinking
        parts: list[str] = [self.bos_token]
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
                continue
            for item in content:
                item_type = item["type"]
                if item_type == "text":
                    parts.append(str(item["text"]))
                elif item_type == "image_url":
                    parts.append(self.image_token)
                elif item_type == "video_url":
                    parts.append(self.video_token)
                elif item_type == "audio_url":
                    parts.append(self.audio_token)
                else:  # pragma: no cover - test helper should only receive supported types
                    raise ValueError(f"Unsupported content type: {item_type}")
        if add_generation_prompt:
            parts.append("<gen>")
        return " ".join(parts)


def _tiny_multimodal_config(tokenizer: _FakeGemmaTokenizer) -> Gemma4Config:
    return Gemma4Config.from_dict(
        {
            "model_type": "gemma4",
            "image_token_id": tokenizer.image_token_id,
            "video_token_id": tokenizer.video_token_id,
            "audio_token_id": tokenizer.audio_token_id,
            "boi_token_id": tokenizer._ids[tokenizer.boi_token],
            "eoi_token_id": tokenizer._ids[tokenizer.eoi_token],
            "boa_token_id": tokenizer._ids[tokenizer.boa_token],
            "eoa_token_id": tokenizer._ids[tokenizer.eoa_token],
            "vision_soft_tokens_per_image": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 512,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 512,
                "sliding_window": 16,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "original_max_position_embeddings": 16,
                        "factor": 2.0,
                    },
                },
            },
            "vision_config": {
                "model_type": "gemma4_vision",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 4,
                "global_head_dim": 4,
                "patch_size": 2,
                "pooling_kernel_size": 1,
                "position_embedding_size": 32,
                "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
                "standardize": False,
                "use_clipped_linears": True,
            },
            "audio_config": {
                "model_type": "gemma4_audio",
                "hidden_size": 16,
                "hidden_act": "silu",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "attention_chunk_size": 4,
                "attention_context_left": 5,
                "attention_context_right": 0,
                "attention_invalid_logits_value": -1000000000.0,
                "attention_logit_cap": 50.0,
                "conv_kernel_size": 3,
                "gradient_clipping": 10000000000.0,
                "output_proj_dims": 16,
                "residual_weight": 0.5,
                "rms_norm_eps": 1e-6,
                "subsampling_conv_channels": [8, 4],
                "use_clipped_linears": True,
            },
        }
    )


def test_gemma4_processor_prepares_image_video_audio_inputs(tmp_path) -> None:
    tokenizer = _FakeGemmaTokenizer()
    processor = Gemma4TextProcessor(
        tokenizer,
        image_settings=_Gemma4ImageSettings(patch_size=2, max_soft_tokens=4, pooling_kernel_size=1),
        video_settings=_Gemma4VideoSettings(
            patch_size=2,
            max_soft_tokens=4,
            pooling_kernel_size=1,
            num_frames=2,
            do_sample_frames=True,
        ),
        audio_settings=_Gemma4AudioSettings(feature_size=8, audio_seq_length=64),
    )

    image_path = tmp_path / "tiny-image.png"
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_path)

    video_path = tmp_path / "tiny-video.mp4"
    frames = np.stack(
        [
            np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8),
            np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8),
        ]
    )
    iio.imwrite(video_path, frames, fps=2)

    audio_path = tmp_path / "tiny-audio.wav"
    sampling_rate = processor.audio_settings.sampling_rate
    samples = np.arange(int(0.25 * sampling_rate), dtype=np.float32)
    waveform = 0.2 * np.sin((2.0 * math.pi * 440.0 * samples) / sampling_rate)
    sf.write(audio_path, waveform, sampling_rate)

    prepared = processor.prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe everything"},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                    {"type": "video_url", "video_url": {"url": str(video_path)}},
                    {"type": "audio_url", "audio_url": {"url": str(audio_path)}},
                ],
            }
        ],
        enable_thinking=False,
    )

    assert prepared.pixel_values is not None
    assert prepared.image_position_ids is not None
    assert prepared.pixel_values_videos is not None
    assert prepared.video_position_ids is not None
    assert prepared.input_features is not None
    assert prepared.input_features_mask is not None
    assert (prepared.input_ids == tokenizer.image_token_id).any()
    assert (prepared.input_ids == tokenizer.video_token_id).any()
    assert (prepared.input_ids == tokenizer.audio_token_id).any()
    assert (prepared.mm_token_type_ids == 1).any()
    assert (prepared.mm_token_type_ids == 2).any()
    assert (prepared.mm_token_type_ids == 3).any()


def test_gemma4_model_forward_accepts_image_video_audio_inputs(tmp_path) -> None:
    tokenizer = _FakeGemmaTokenizer()
    processor = Gemma4TextProcessor(
        tokenizer,
        image_settings=_Gemma4ImageSettings(patch_size=2, max_soft_tokens=4, pooling_kernel_size=1),
        video_settings=_Gemma4VideoSettings(
            patch_size=2,
            max_soft_tokens=4,
            pooling_kernel_size=1,
            num_frames=2,
            do_sample_frames=True,
        ),
        audio_settings=_Gemma4AudioSettings(feature_size=8, audio_seq_length=64),
    )

    image_path = tmp_path / "forward-image.png"
    Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8)).save(image_path)

    video_path = tmp_path / "forward-video.mp4"
    video_frames = np.stack(
        [
            np.full((4, 4, 3), 32, dtype=np.uint8),
            np.full((4, 4, 3), 96, dtype=np.uint8),
        ]
    )
    iio.imwrite(video_path, video_frames, fps=2)

    audio_path = tmp_path / "forward-audio.wav"
    sampling_rate = processor.audio_settings.sampling_rate
    samples = np.arange(int(0.25 * sampling_rate), dtype=np.float32)
    waveform = 0.15 * np.sin((2.0 * math.pi * 220.0 * samples) / sampling_rate)
    sf.write(audio_path, waveform, sampling_rate)

    prepared = processor.prepare_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "multimodal test"},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                    {"type": "video_url", "video_url": {"url": str(video_path)}},
                    {"type": "audio_url", "audio_url": {"url": str(audio_path)}},
                ],
            }
        ],
        enable_thinking=False,
    )

    model = Gemma4ForConditionalGeneration(_tiny_multimodal_config(tokenizer)).eval()
    model.tie_weights()
    model.configure_runtime(torch.device("cpu"), offload_vision=False)

    with torch.no_grad():
        outputs = model(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            mm_token_type_ids=prepared.mm_token_type_ids,
            pixel_values=prepared.pixel_values,
            image_position_ids=prepared.image_position_ids,
            pixel_values_videos=prepared.pixel_values_videos,
            video_position_ids=prepared.video_position_ids,
            input_features=prepared.input_features,
            input_features_mask=prepared.input_features_mask,
            use_cache=False,
        )

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == prepared.input_ids.shape[1]
    assert outputs.logits.shape[2] == model.config.text_config.vocab_size
