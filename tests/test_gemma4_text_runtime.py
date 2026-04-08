from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

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
