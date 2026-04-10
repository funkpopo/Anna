from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from anna.mm.gemma4_text_processor import PreparedInputs
from anna.runtime.model_runtime_loader import load_model_runtime_from_model_dir
from anna.runtime.gemma4_text_engine import AnnaGemma4TextEngine
from anna.model.gemma4_config import Gemma4Config
from anna.model.gemma4_text_model import Gemma4ForConditionalGeneration
from anna.weights.gemma4_tokenizer import Gemma4Tokenizer
from anna.weights.gemma4_text_weight_loader import build_gemma4_text_model


def test_gemma4_config_parses_text_defaults_and_generation_limit(tmp_path) -> None:
    model_dir = tmp_path / "gemma4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "eos_token_id": [1, 106],
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2560,
                    "intermediate_size": 10240,
                    "num_hidden_layers": 42,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "hidden_size_per_layer_input": 256,
                    "vocab_size_per_layer_input": 262144,
                    "layer_types": ["sliding_attention"] * 41 + ["full_attention"],
                    "rope_parameters": {
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                        "full_attention": {
                            "rope_type": "proportional",
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(json.dumps({"max_new_tokens": 321}), encoding="utf-8")

    config = Gemma4Config.from_model_dir(model_dir)

    assert config.model_type == "gemma4"
    assert config.default_max_completion_tokens == 321
    assert config.text_config.eos_token_ids == (1, 106)
    assert config.text_config.rope_parameters["full_attention"].rope_type == "proportional"
    assert config.text_config.rope_parameters["full_attention"].original_max_position_embeddings == 8192
    assert config.text_config.rope_parameters["full_attention"].factor == 16.0


def test_gemma4_tokenizer_renders_chat_and_eos_ids() -> None:
    vocab = {
        "<bos>": 2,
        "<eos>": 1,
        "<|turn>system": 200,
        "<|turn>user": 201,
        "<|turn>model": 202,
        "<|turn>": 105,
        "<turn|>": 106,
        "<|channel>": 100,
        "<channel|>": 101,
        "<|think|>": 98,
        "hello": 10,
        "world": 11,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<eos>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Gemma4Tokenizer(
        backend,
        metadata={
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "sot_token": "<|turn>",
            "eot_token": "<turn|>",
            "soc_token": "<|channel>",
            "think_token": "<|think|>",
        },
    )

    rendered = tokenizer.render_messages(
        [
            {"role": "system", "content": "hello"},
            {"role": "user", "content": "world"},
        ],
        enable_thinking=True,
    )

    assert rendered.startswith("<bos><|turn>system\n<|think|>hello<turn|>\n")
    assert rendered.endswith("<|turn>model\n")
    assert tokenizer.eos_token_ids == {1, 106}


def test_gemma4_tokenizer_renders_and_parses_function_calling() -> None:
    vocab = {
        "<bos>": 2,
        "<eos>": 1,
        "<|turn>": 105,
        "<turn|>": 106,
        "<|channel>": 100,
        "<channel|>": 101,
        "<|think|>": 98,
        "<|tool_call>": 48,
        "<tool_call|>": 49,
        "<|tool_response>": 50,
        "<tool_response|>": 51,
        "<|tool>": 52,
        "<tool|>": 53,
        "hello": 10,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<eos>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Gemma4Tokenizer(
        backend,
        metadata={
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "sot_token": "<|turn>",
            "eot_token": "<turn|>",
            "soc_token": "<|channel>",
            "eoc_token": "<channel|>",
            "think_token": "<|think|>",
            "stc_token": "<|tool_call>",
            "etc_token": "<tool_call|>",
            "str_token": "<|tool_response>",
            "etr_token": "<tool_response|>",
            "std_token": "<|tool>",
            "etd_token": "<tool|>",
        },
    )

    rendered = tokenizer.render_messages(
        [
            {"role": "system", "content": "hello"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Shanghai\",\"days\":3}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
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
                            "days": {"type": "integer"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        parallel_tool_calls=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )

    assert rendered.startswith("<bos><|turn>system\n<|think|>hello")
    assert "<|tool>declaration:get_weather{" in rendered
    assert "<|tool_call>call:get_weather{days:3,location:<|\"|>Shanghai<|\"|>}<tool_call|>" in rendered
    assert "<|tool_response>response:get_weather{temperature:28}<tool_response|>" in rendered

    reasoning, content = tokenizer.split_assistant_reasoning(
        "<|channel>thought\n先分析一下。<channel|>最终答案。",
        enable_thinking=True,
    )
    assert reasoning == "先分析一下。"
    assert content == "最终答案。"

    cleaned, tool_calls = tokenizer.extract_tool_calls(
        "先查天气。<|tool_call>call:get_weather{location:<|\"|>Shanghai<|\"|>,days:3}<tool_call|>"
    )
    assert cleaned == "先查天气。"
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0].to_openai_dict()["function"]["arguments"]) == {
        "location": "Shanghai",
        "days": 3,
    }


def test_gemma4_text_model_decode_cache_matches_full_forward() -> None:
    config = Gemma4Config.from_dict(
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 128,
                "num_kv_shared_layers": 2,
                "sliding_window": 8,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "original_max_position_embeddings": 8,
                        "factor": 2.0,
                    },
                },
            },
        }
    )
    model = Gemma4ForConditionalGeneration(config).eval()
    model.tie_weights()

    input_ids = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
    prompt_ids = input_ids[:, :-1]
    append_ids = input_ids[:, -1:]

    with torch.no_grad():
        full_output = model.forward_text_only(input_ids=input_ids, use_cache=False)
        cache_output = model.forward_text_only(input_ids=prompt_ids, use_cache=True)
        assert cache_output.past_key_values is not None
        assert cache_output.past_key_values.get_seq_length() == prompt_ids.shape[1]
        decode_output = model.forward_text_only(
            input_ids=append_ids,
            past_key_values=cache_output.past_key_values,
            use_cache=True,
        )

    assert decode_output.past_key_values is not None
    assert decode_output.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(
        decode_output.logits[:, -1].float(),
        full_output.logits[:, -1].float(),
        atol=1e-4,
        rtol=1e-4,
    )


def test_gemma4_build_restores_runtime_buffers_after_to_empty() -> None:
    config = Gemma4Config.from_dict(
        {
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 128,
                "num_kv_shared_layers": 2,
                "sliding_window": 8,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "original_max_position_embeddings": 8,
                        "factor": 2.0,
                    },
                },
            },
        }
    )

    model, _ = build_gemma4_text_model(config, device=torch.device("cpu"), dtype=torch.bfloat16)
    text_model = model.model.language_model

    assert model.lm_head is None
    assert float(text_model.embed_tokens.embed_scale.cpu()) == pytest.approx(config.text_config.hidden_size**0.5)
    assert float(text_model.embed_tokens_per_layer.embed_scale.cpu()) == pytest.approx(
        config.text_config.hidden_size_per_layer_input**0.5
    )
    assert torch.allclose(text_model.layers[0].layer_scalar.float().cpu(), torch.ones(1))
    assert float(getattr(text_model.rotary_emb, "full_attention_inv_freq").float().abs().sum()) > 0.0


def test_gemma4_runtime_loader_builds_standalone_multimodal_engine(tmp_path) -> None:
    model_dir = tmp_path / "gemma4-runtime"
    model_dir.mkdir()

    config_dict = {
        "model_type": "gemma4",
        "image_token_id": 13,
        "video_token_id": 17,
        "audio_token_id": 16,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "boa_token_id": 14,
        "eoa_token_id": 15,
        "vision_soft_tokens_per_image": 70,
        "text_config": {
            "model_type": "gemma4_text",
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 256,
            "sliding_window": 16,
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
            "num_hidden_layers": 1,
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
            "num_hidden_layers": 1,
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
    (model_dir / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    (model_dir / "processor_config.json").write_text(
        json.dumps(
            {
                "audio_ms_per_token": 40,
                "audio_seq_length": 64,
                "image_processor": {
                    "patch_size": 2,
                        "max_soft_tokens": 70,
                    "pooling_kernel_size": 1,
                    "rescale_factor": 1.0 / 255.0,
                    "image_mean": [0.0, 0.0, 0.0],
                    "image_std": [1.0, 1.0, 1.0],
                },
                "video_processor": {
                    "patch_size": 2,
                        "max_soft_tokens": 70,
                    "pooling_kernel_size": 1,
                    "num_frames": 2,
                    "do_sample_frames": True,
                    "rescale_factor": 1.0 / 255.0,
                    "image_mean": [0.0, 0.0, 0.0],
                    "image_std": [1.0, 1.0, 1.0],
                },
                "feature_extractor": {
                    "feature_size": 8,
                    "sampling_rate": 16000,
                    "frame_length": 320,
                    "hop_length": 160,
                    "min_frequency": 0.0,
                    "max_frequency": 8000.0,
                    "preemphasis": 0.0,
                    "preemphasis_htk_flavor": True,
                    "fft_overdrive": False,
                    "dither": 0.0,
                    "input_scale_factor": 1.0,
                    "mel_floor": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )

    backend = Tokenizer(
        WordLevel(
            vocab={
                "<bos>": 1,
                "<eos>": 2,
                "<|turn>": 3,
                "<turn|>": 4,
                "<|channel>": 5,
                "<channel|>": 6,
                "<|think|>": 7,
                "<|image>": 11,
                "<image|>": 12,
                "<|image|>": 13,
                "<|audio>": 14,
                "<audio|>": 15,
                "<|audio|>": 16,
                "<|video|>": 17,
                "hello": 18,
            },
            unk_token="<eos>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "sot_token": "<|turn>",
                "eot_token": "<turn|>",
                "soc_token": "<|channel>",
                "eoc_token": "<channel|>",
                "think_token": "<|think|>",
                "boi_token": "<|image>",
                "eoi_token": "<image|>",
                "image_token": "<|image|>",
                "boa_token": "<|audio>",
                "eoa_token": "<audio|>",
                "audio_token": "<|audio|>",
            }
        ),
        encoding="utf-8",
    )

    model = Gemma4ForConditionalGeneration(Gemma4Config.from_dict(config_dict)).eval()
    model.tie_weights()
    save_file(model.state_dict(), str(model_dir / "model.safetensors"))

    engine = load_model_runtime_from_model_dir(model_dir, model_id="gemma4-test", device="cpu", dtype="float32")
    health = engine.health()

    assert engine.model_family == "gemma4"
    assert health["model_family"] == "gemma4"
    assert health["vision_enabled"] is True
    assert health["audio_enabled"] is True
    assert health["weight_quant"] == "none"


def test_gemma4_runtime_forwards_gemma_only_multimodal_kwargs() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.kwargs: dict[str, object] | None = None

        def forward_text_only(self, **_kwargs):
            self.calls.append("text")
            return object()

        def __call__(self, **kwargs):
            self.calls.append("full")
            self.kwargs = kwargs
            return object()

    engine = object.__new__(AnnaGemma4TextEngine)
    engine.model = _FakeModel()

    prepared = PreparedInputs(
        prompt="gemma multimodal",
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 1, 1]], dtype=torch.int32),
        pixel_values=torch.randn(4, 8),
        image_position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        pixel_values_videos=torch.randn(2, 8),
        video_position_ids=torch.tensor([[0]], dtype=torch.long),
        input_features=torch.randn(1, 2, 4),
        input_features_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    model_kwargs = engine._build_prefill_model_kwargs(prepared, include_media=True)

    assert "image_position_ids" in model_kwargs
    assert "video_position_ids" in model_kwargs
    assert "input_features" in model_kwargs
    assert "input_features_mask" in model_kwargs
    assert "image_grid_thw" not in model_kwargs
    assert "video_grid_thw" not in model_kwargs

    engine._forward_generation_model(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        past_key_values=None,
        model_kwargs=model_kwargs,
        use_cache=True,
        logits_to_keep=1,
    )

    assert engine.model.calls == ["full"]
    assert engine.model.kwargs is not None
    assert "image_position_ids" in engine.model.kwargs
    assert "video_position_ids" in engine.model.kwargs
    assert "input_features" in engine.model.kwargs
    assert "input_features_mask" in engine.model.kwargs
    assert "image_grid_thw" not in engine.model.kwargs
    assert "video_grid_thw" not in engine.model.kwargs
