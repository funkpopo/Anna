from __future__ import annotations

import torch

from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.ops import (
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    torch_causal_conv1d_update,
    torch_causal_conv1d_update_one_token,
    torch_recurrent_gated_delta_rule_one_token,
)
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


def test_qwen3_prefill_can_project_only_last_logit() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_tiny_config())
    input_ids = torch.randint(0, 256, (1, 6))
    outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    assert outputs.logits.shape == (1, 1, 256)
    assert outputs.past_key_values is not None
    assert outputs.past_key_values.get_seq_length() == 6


def test_dynamic_cache_appends_without_reallocating_visible_prefix() -> None:
    cache = Qwen3DynamicCache(_tiny_config())
    key_a = torch.randn(1, 2, 3, 16)
    value_a = torch.randn(1, 2, 3, 16)
    key_b = torch.randn(1, 2, 2, 16)
    value_b = torch.randn(1, 2, 2, 16)

    combined_key_a, combined_value_a, _ = cache.update(key_a, value_a, layer_idx=1)
    combined_key_b, combined_value_b, _ = cache.update(key_b, value_b, layer_idx=1)

    assert combined_key_a.shape == (1, 2, 3, 16)
    assert combined_value_a.shape == (1, 2, 3, 16)
    assert combined_key_b.shape == (1, 2, 5, 16)
    assert combined_value_b.shape == (1, 2, 5, 16)
    assert torch.equal(combined_key_b[:, :, :3, :], key_a)
    assert torch.equal(combined_value_b[:, :, :3, :], value_a)
    assert torch.equal(combined_key_b[:, :, 3:, :], key_b)
    assert torch.equal(combined_value_b[:, :, 3:, :], value_b)
    assert cache.get_seq_length() == 5


def test_dynamic_cache_stack_and_split_round_trips_batches() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache_a = Qwen3DynamicCache(config, allocator=allocator)
    cache_b = Qwen3DynamicCache(config, allocator=allocator)
    key_a = torch.randn(1, 2, 3, 16)
    value_a = torch.randn(1, 2, 3, 16)
    key_b = torch.randn(1, 2, 3, 16)
    value_b = torch.randn(1, 2, 3, 16)
    cache_a.update(key_a, value_a, layer_idx=1)
    cache_b.update(key_b, value_b, layer_idx=1)

    stacked = Qwen3DynamicCache.stack([cache_a, cache_b], config)
    split = stacked.split_batch()

    assert len(split) == 2
    assert torch.equal(split[0].visible_key_cache(1), key_a)
    assert torch.equal(split[0].visible_value_cache(1), value_a)
    assert torch.equal(split[1].visible_key_cache(1), key_b)
    assert torch.equal(split[1].visible_value_cache(1), value_b)


def test_dynamic_cache_release_reuses_freed_pages() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache_a = Qwen3DynamicCache(config, allocator=allocator)
    cache_b = Qwen3DynamicCache(config, allocator=allocator)
    key = torch.randn(1, 2, config.cache_block_size, 16)
    value = torch.randn(1, 2, config.cache_block_size, 16)

    cache_a.update(key, value, layer_idx=1)
    first_page_ids = list(cache_a.page_tables[1][0])
    cache_a.release()
    cache_b.update(key, value, layer_idx=1)

    assert cache_b.page_tables[1][0] == first_page_ids


def test_single_token_conv_update_matches_depthwise_conv() -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 6, 1, dtype=torch.bfloat16)
    conv_state = torch.randn(2, 6, 4, dtype=torch.bfloat16)
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    bias = torch.randn(6, dtype=torch.bfloat16)

    expected_state = conv_state.clone()
    actual_state = conv_state.clone()
    expected = torch_causal_conv1d_update(hidden_states, expected_state, weight, bias)
    actual = torch_causal_conv1d_update_one_token(hidden_states, actual_state, weight, bias)

    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.allclose(actual_state, expected_state, atol=1e-2, rtol=1e-2)


def test_single_token_recurrent_rule_matches_manual_update() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 1, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 1, 4, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 1, 4, 6, dtype=torch.bfloat16)
    g = torch.randn(2, 1, 4, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(2, 1, 4, dtype=torch.bfloat16))
    initial_state = torch.randn(2, 4, 8, 6, dtype=torch.float32)

    actual_out, actual_state = torch_recurrent_gated_delta_rule_one_token(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )

    q_t = query[:, 0]
    q_t = q_t * torch.rsqrt((q_t * q_t).sum(dim=-1, keepdim=True) + 1e-6)
    q_t = q_t.float()
    q_t = q_t * (q_t.shape[-1] ** -0.5)
    k_t = key[:, 0]
    k_t = k_t * torch.rsqrt((k_t * k_t).sum(dim=-1, keepdim=True) + 1e-6)
    k_t = k_t.float()
    v_t = value[:, 0].float()
    beta_t = beta[:, 0].float().unsqueeze(-1)
    g_t = g[:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)
    expected_state = initial_state * g_t
    kv_mem = torch.matmul(k_t.unsqueeze(-2), expected_state).squeeze(-2)
    delta = (v_t - kv_mem) * beta_t
    expected_state = expected_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    expected_out = torch.matmul(q_t.unsqueeze(-2), expected_state).squeeze(-2).unsqueeze(1).to(dtype=query.dtype)

    assert actual_state is not None
    assert torch.allclose(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(actual_state, expected_state, atol=1e-4, rtol=1e-4)


def test_qwen3_allows_out_of_range_padding_idx_for_tiny_configs() -> None:
    model = Qwen3ForCausalLM(_tiny_config())
    assert model.model.embed_tokens.padding_idx is None
