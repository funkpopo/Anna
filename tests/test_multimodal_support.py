from __future__ import annotations

import inspect
import json

import imageio.v3 as iio
import numpy as np
import torch

from anna.mm.qwen3_5_text_processor import Qwen3_5TextMultimodalProcessor
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
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


def test_qwen3_model_config_prefers_quantization_config_json_when_present(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
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
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "data_type": "int",
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "block_name_to_quantize": "model.language_model.layers",
            }
        ),
        encoding="utf-8",
    )

    config = Qwen3_5TextModelConfig.from_model_dir(tmp_path)

    assert config.quantization_config.quant_method == "auto-round"
    assert config.quantization_config.packing_format == "auto_round:auto_gptq"
    assert config.quantization_config.block_name_to_quantize == ("model.language_model.layers",)


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


def test_tokenizer_renders_qwen_function_calling_prompt_and_history() -> None:
    tokenizer = _tokenizer()
    rendered = tokenizer.render_messages(
        [
            {"role": "system", "content": "You are a tool-using assistant."},
            {"role": "user", "content": "查一下上海天气。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Shanghai\",\"unit\":\"celsius\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "{\"temperature\":28}",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        parallel_tool_calls=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    assert "# Tools" in rendered
    assert '"name": "get_weather"' in rendered
    assert "You are a tool-using assistant." in rendered
    assert "<tool_call>\n<function=get_weather>\n<parameter=location>\nShanghai\n</parameter>" in rendered
    assert "<parameter=unit>\ncelsius\n</parameter>" in rendered
    assert "<tool_response>\n{\"temperature\":28}\n</tool_response>" in rendered


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


def test_qwen3_conditional_generation_signature_uses_grid_metadata_only() -> None:
    signature = inspect.signature(Qwen3_5TextForConditionalGeneration.forward)

    assert "image_grid_thw" in signature.parameters
    assert "video_grid_thw" in signature.parameters
    assert "image_position_ids" not in signature.parameters
    assert "video_position_ids" not in signature.parameters
    assert "input_features" not in signature.parameters
    assert "input_features_mask" not in signature.parameters


def test_qwen3_conditional_generation_uses_qwen_multimodal_grid_metadata() -> None:
    config = _config()
    config.text_config.layer_types = ["full_attention"] * config.text_config.num_hidden_layers
    config.image_token_id = 101
    config.video_token_id = 102
    config.vision_start_token_id = 103
    config.vision_end_token_id = 104
    model = Qwen3_5TextForConditionalGeneration(config).eval()
    model.tie_weights()
    model.configure_runtime(torch.device("cpu"), offload_vision=False)

    input_ids = torch.tensor([[11, 12, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 13]])
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_token_type_ids[:, 2:6] = 1
    pixel_values = torch.randn(16, 24)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )

    assert outputs.logits.shape == (1, input_ids.shape[1], config.text_config.vocab_size)
