from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import torch

from anna.mm.qwen3_5_text_processor import Qwen3_5TextMultimodalProcessor
from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig, Qwen3_5TextConfig, Qwen3_5TextVisionConfig, VisionPreprocessorConfig
from anna.weights.qwen3_5_text_tokenizer import Qwen3_5TextTokenizer


class _FakeBackend:
    def __init__(self) -> None:
        self._token_ids = {
            "<|image_pad|>": 101,
            "<|video_pad|>": 102,
            "<|vision_start|>": 103,
            "<|vision_end|>": 104,
            "<|im_end|>": 105,
            "<|endoftext|>": 106,
        }

    def token_to_id(self, token: str) -> int | None:
        return self._token_ids.get(token)

    def encode(self, text: str):  # pragma: no cover - this helper is only for tokenizer formatting tests
        raise NotImplementedError

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:  # pragma: no cover
        raise NotImplementedError


def _tokenizer() -> Qwen3_5TextTokenizer:
    return Qwen3_5TextTokenizer(
        _FakeBackend(),
        metadata={
            "extra_special_tokens": {
                "image_token": "<|image_pad|>",
                "video_token": "<|video_pad|>",
                "vision_bos_token": "<|vision_start|>",
                "vision_eos_token": "<|vision_end|>",
            }
        },
    )


def _config() -> Qwen3_5TextModelConfig:
    return Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            vocab_size=256,
            layer_types=["linear_attention", "full_attention"],
        ),
        vision_config=Qwen3_5TextVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=64,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        preprocessor_config=VisionPreprocessorConfig(
            shortest_edge=16,
            longest_edge=16 * 16,
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
        ),
    )


def test_qwen3_config_keeps_multimodal_fields() -> None:
    config = Qwen3_5TextModelConfig.from_dict(
        {
            "model_type": "qwen3_5_vl",
            "max_completion_tokens": 1536,
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 4,
                "vocab_size": 256,
                "layer_types": ["linear_attention", "full_attention"],
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 64,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "version": "gemm",
            },
        },
        preprocessor_data={
            "min_pixels": 16,
            "max_pixels": 256,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
    )

    assert config.vision_config is not None
    assert config.vision_config.temporal_patch_size == 2
    assert config.quantization_config.quant_method == "awq"
    assert config.quantization_config.bits == 4
    assert config.default_max_completion_tokens == 1536
    assert config.preprocessor_config.merge_size == 2


def test_qwen3_config_leaves_default_max_completion_tokens_unset_when_missing() -> None:
    config = Qwen3_5TextModelConfig.from_dict(
        {
            "model_type": "qwen3_5_vl",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 4,
                "vocab_size": 256,
                "layer_types": ["linear_attention", "full_attention"],
            },
        }
    )

    assert config.default_max_completion_tokens is None


def test_qwen3_config_uses_top_level_pad_token_when_text_config_value_is_null() -> None:
    config = Qwen3_5TextModelConfig.from_dict(
        {
            "pad_token_id": 248055,
            "eos_token_id": 248046,
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 4,
                "vocab_size": 256,
                "layer_types": ["linear_attention", "full_attention"],
                "eos_token_id": 248044,
                "pad_token_id": None,
            },
        }
    )

    assert config.text_config.eos_token_id == 248044
    assert config.text_config.pad_token_id == 248055


def test_tokenizer_renders_native_multimodal_placeholders() -> None:
    tokenizer = _tokenizer()
    rendered = tokenizer.render_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "local-image.png"}},
                    {"type": "video_url", "video_url": {"url": "local-video.mp4"}},
                ],
            }
        ],
        enable_thinking=False,
    )

    assert "<|vision_start|><|image_pad|><|vision_end|>" in rendered
    assert "<|vision_start|><|video_pad|><|vision_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_tokenizer_renders_open_think_prompt_when_enabled() -> None:
    tokenizer = _tokenizer()
    rendered = tokenizer.render_messages(
        [{"role": "user", "content": "你好"}],
        enable_thinking=True,
    )

    assert rendered.endswith("<|im_start|>assistant\n<think>\n")


def test_tokenizer_renders_assistant_reasoning_history() -> None:
    tokenizer = _tokenizer()
    rendered = tokenizer.render_messages(
        [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "答案是2。", "reasoning_content": "我先做加法。"},
        ],
        add_generation_prompt=False,
    )

    assert "<think>\n我先做加法。\n</think>\n\n答案是2。" in rendered


def test_tokenizer_renders_raw_assistant_output_with_reasoning_content_without_duplication() -> None:
    tokenizer = _tokenizer()
    rendered = tokenizer.render_messages(
        [
            {"role": "user", "content": "夏天怎么样？"},
            {
                "role": "assistant",
                "content": "先写夏天的氛围。</think>\n\n夏天有风，也有晚霞。",
                "reasoning_content": "先写夏天的氛围。",
            },
        ],
        add_generation_prompt=False,
    )

    assert "<think>\n先写夏天的氛围。\n</think>\n\n夏天有风，也有晚霞。" in rendered
    assert rendered.count("先写夏天的氛围。") == 1


def test_processor_expands_qwen3_native_placeholders() -> None:
    processor = Qwen3_5TextMultimodalProcessor(_config(), _tokenizer())
    image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    video_prompt = "<|vision_start|><|video_pad|><|vision_end|>"

    expanded_image = processor._expand_image_placeholders(image_prompt, torch.tensor([[1, 4, 4]], dtype=torch.long))
    expanded_video = processor._expand_video_placeholders(
        video_prompt,
        torch.tensor([[2, 4, 4]], dtype=torch.long),
        [24.0],
    )

    assert expanded_image.count(processor.tokenizer.image_token) == 4
    assert expanded_video.count(processor.tokenizer.video_token) == 8
    assert "<0.0 seconds>" in expanded_video
    assert "<0.1 seconds>" in expanded_video


def test_processor_loads_local_video_file(tmp_path) -> None:
    processor = Qwen3_5TextMultimodalProcessor(_config(), _tokenizer())
    video_path = tmp_path / "colors.mp4"
    frames = []
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:] = color
        frames.append(frame)
    iio.imwrite(video_path, np.stack(frames), fps=2)

    decoded_frames, fps = processor._load_video(str(video_path))

    assert len(decoded_frames) == 3
    assert round(fps, 2) == 2.0
    assert decoded_frames[0].size == (64, 64)
