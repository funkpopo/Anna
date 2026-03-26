from __future__ import annotations

import torch

from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.qwen import Qwen3ForCausalLM


def _tiny_config() -> Qwen3TextConfig:
    return Qwen3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        rope_parameters=RopeParameters(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            mrope_section=(1, 1, 0),
        ),
    )


def test_qwen3_forward_shapes() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_tiny_config())
    input_ids = torch.randint(0, 256, (1, 6))
    outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.logits.shape == (1, 6, 256)
    assert outputs.past_key_values is not None
    assert outputs.past_key_values.get_seq_length() == 6


def test_qwen3_incremental_decode() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_tiny_config())
    prompt_ids = torch.randint(0, 256, (1, 6))
    first = model(input_ids=prompt_ids, use_cache=True)
    next_token = torch.randint(0, 256, (1, 1))
    second = model(
        input_ids=next_token,
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    assert second.logits.shape == (1, 1, 256)
    assert second.past_key_values is not None
    assert second.past_key_values.get_seq_length() == 7


def test_qwen3_allows_out_of_range_padding_idx_for_tiny_configs() -> None:
    model = Qwen3ForCausalLM(_tiny_config())
    assert model.model.embed_tokens.padding_idx is None
