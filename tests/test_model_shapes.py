from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import anna.model.ops as model_ops
from anna.model.quantization import AutoRoundGPTQLinear, XPUInt4Linear, replace_linear_modules
from anna.model.qwen3_5_text_config import QuantizationConfig, Qwen3_5TextConfig, RopeParameters
from anna.model.turboquant import TurboQuantKVRow
from anna.model.ops import (
    Qwen3Attention,
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    Qwen3SparseMoeBlock,
    Qwen3TextRotaryEmbedding,
    grouped_query_attention,
    repeat_kv,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from anna.model.qwen3_5_text_model import Qwen3_5TextForCausalLM
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine


@contextmanager
def _temporary_torch_seed(seed: int):
    previous_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(previous_state)


def _stub_run_gated_delta_fused(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    state_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if initial_state is None:
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=output_final_state,
        )
    else:
        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    batch_size, seq_len, _, head_v_dim = value.shape
    core_attn_out = core_attn_out.reshape(-1, head_v_dim)
    z = z.reshape(-1, head_v_dim)
    input_dtype = core_attn_out.dtype
    hidden_states = core_attn_out.float()
    hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    hidden_states = norm_weight * hidden_states.to(dtype=input_dtype)
    hidden_states = hidden_states * torch.nn.functional.silu(z.float())
    return hidden_states.to(dtype=input_dtype).reshape(batch_size, seq_len, -1), last_recurrent_state


@pytest.fixture(autouse=True)
def _stub_gated_delta_fused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_ops, "run_gated_delta_fused", _stub_run_gated_delta_fused)


def _tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
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


def _tiny_moe_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
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
        decoder_sparse_step=1,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[3],
    )


def test_qwen3_forward_shapes() -> None:
    with _temporary_torch_seed(0):
        model = Qwen3_5TextForCausalLM(_tiny_config())
        input_ids = torch.randint(0, 256, (1, 6))
        outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.logits.shape == (1, 6, 256)
    assert outputs.past_key_values is not None
    assert outputs.past_key_values.get_seq_length() == 6


def test_qwen_text_model_decode_cache_with_turboquant_stays_numerically_close() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_config()).eval()
    model.tie_weights()
    model.configure_runtime(
        torch.device("cpu"),
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    input_ids = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
    prompt_ids = input_ids[:, :-1]
    append_ids = input_ids[:, -1:]

    with torch.no_grad():
        full_output = model(input_ids=input_ids, use_cache=False)
        cache_output = model(input_ids=prompt_ids, use_cache=True)
        assert cache_output.past_key_values is not None
        assert isinstance(cache_output.past_key_values.turboquant_rows[1][0], TurboQuantKVRow)
        decode_output = model(
            input_ids=append_ids,
            past_key_values=cache_output.past_key_values,
            use_cache=True,
        )

    assert decode_output.past_key_values is not None
    assert decode_output.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.isfinite(decode_output.logits).all()
    diff = (decode_output.logits[:, -1].float() - full_output.logits[:, -1].float()).abs().max().item()
    assert diff < 1.5


def test_qwen3_incremental_decode() -> None:
    with _temporary_torch_seed(0):
        model = Qwen3_5TextForCausalLM(_tiny_config())
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


def test_qwen3_incremental_multi_token_decode_matches_full_forward() -> None:
    with _temporary_torch_seed(0):
        model = Qwen3_5TextForCausalLM(_tiny_config())
        prompt_ids = torch.randint(0, 256, (1, 6))
        append_ids = torch.randint(0, 256, (1, 3))

        prefix = model(input_ids=prompt_ids, use_cache=True)
        continued = model(
            input_ids=append_ids,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        full = model(input_ids=torch.cat([prompt_ids, append_ids], dim=1), use_cache=True)

    assert continued.logits.shape == (1, 3, 256)
    assert continued.past_key_values is not None
    assert continued.past_key_values.get_seq_length() == 9
    assert torch.allclose(continued.logits, full.logits[:, -append_ids.shape[1] :, :], atol=1e-5, rtol=1e-4)


def test_qwen3_prefill_can_project_only_last_logit() -> None:
    with _temporary_torch_seed(0):
        model = Qwen3_5TextForCausalLM(_tiny_config())
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
    key_storage_ptr = combined_key_a.untyped_storage().data_ptr()
    value_storage_ptr = combined_value_a.untyped_storage().data_ptr()
    combined_key_b, combined_value_b, _ = cache.update(key_b, value_b, layer_idx=1)

    assert combined_key_a.shape == (1, 2, 3, 16)
    assert combined_value_a.shape == (1, 2, 3, 16)
    assert combined_key_b.shape == (1, 2, 5, 16)
    assert combined_value_b.shape == (1, 2, 5, 16)
    assert combined_key_b.untyped_storage().data_ptr() == key_storage_ptr
    assert combined_value_b.untyped_storage().data_ptr() == value_storage_ptr
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


def test_dynamic_cache_stack_preserves_visible_prefix_for_batched_decode_updates() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache_a = Qwen3DynamicCache(config, allocator=allocator)
    cache_b = Qwen3DynamicCache(config, allocator=allocator)
    key_a = torch.randn(1, 2, 3, 16)
    value_a = torch.randn(1, 2, 3, 16)
    key_b = torch.randn(1, 2, 5, 16)
    value_b = torch.randn(1, 2, 5, 16)
    append_key = torch.randn(2, 2, 1, 16)
    append_value = torch.randn(2, 2, 1, 16)

    cache_a.update(key_a, value_a, layer_idx=1)
    cache_b.update(key_b, value_b, layer_idx=1)

    stacked = Qwen3DynamicCache.stack([cache_a, cache_b], config)
    combined_key, combined_value, past_lengths = stacked.update(append_key, append_value, layer_idx=1)

    assert torch.equal(past_lengths, torch.tensor([3, 5], dtype=torch.long))
    assert torch.equal(combined_key[0:1, :, :3, :], key_a)
    assert torch.equal(combined_value[0:1, :, :3, :], value_a)
    assert torch.equal(combined_key[0:1, :, 3:4, :], append_key[0:1])
    assert torch.equal(combined_value[0:1, :, 3:4, :], append_value[0:1])
    assert torch.equal(combined_key[1:2, :, :5, :], key_b)
    assert torch.equal(combined_value[1:2, :, :5, :], value_b)
    assert torch.equal(combined_key[1:2, :, 5:6, :], append_key[1:2])
    assert torch.equal(combined_value[1:2, :, 5:6, :], append_value[1:2])


def test_dynamic_cache_can_store_full_attention_rows_with_turboquant() -> None:
    config = _tiny_config()
    cache = Qwen3DynamicCache(
        config,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    key_a = torch.randn(1, 2, 3, 16)
    value_a = torch.randn(1, 2, 3, 16)
    key_b = torch.randn(1, 2, 1, 16)
    value_b = torch.randn(1, 2, 1, 16)

    first_key, first_value, _ = cache.update(key_a, value_a, layer_idx=1)
    second_key, second_value, past_lengths = cache.update(key_b, value_b, layer_idx=1)

    assert isinstance(cache.turboquant_rows[1][0], TurboQuantKVRow)
    assert cache.paged_attention_state(1) is None
    assert first_key is not None
    assert first_value is not None
    assert second_key is not None
    assert second_value is not None
    assert torch.equal(past_lengths, torch.tensor([3], dtype=torch.long))
    assert first_key.shape == (1, 2, 3, 16)
    assert first_value.shape == (1, 2, 3, 16)
    assert second_key.shape == (1, 2, 4, 16)
    assert second_value.shape == (1, 2, 4, 16)
    assert torch.allclose(second_key[:, :, -1:, :], key_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(second_value[:, :, -1:, :], value_b, atol=1e-6, rtol=1e-6)
    assert ((second_key[:, :, :3, :] - key_a).float() ** 2).mean().item() < 1.0
    assert ((second_value[:, :, :3, :] - value_a).float() ** 2).mean().item() < 1.0


def test_dynamic_cache_stack_and_split_round_trips_turboquant_batches() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache_a = Qwen3DynamicCache(
        config,
        allocator=allocator,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    cache_b = Qwen3DynamicCache(
        config,
        allocator=allocator,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    key_a = torch.randn(1, 2, 3, 16)
    value_a = torch.randn(1, 2, 3, 16)
    key_b = torch.randn(1, 2, 4, 16)
    value_b = torch.randn(1, 2, 4, 16)
    cache_a.update(key_a, value_a, layer_idx=1)
    cache_b.update(key_b, value_b, layer_idx=1)

    stacked = Qwen3DynamicCache.stack([cache_a, cache_b], config)
    split = stacked.split_batch()

    assert len(split) == 2
    assert isinstance(split[0].turboquant_rows[1][0], TurboQuantKVRow)
    assert isinstance(split[1].turboquant_rows[1][0], TurboQuantKVRow)
    assert ((split[0].visible_key_cache(1) - key_a).float() ** 2).mean().item() < 1.0
    assert ((split[0].visible_value_cache(1) - value_a).float() ** 2).mean().item() < 1.0
    assert ((split[1].visible_key_cache(1) - key_b).float() ** 2).mean().item() < 1.0
    assert ((split[1].visible_value_cache(1) - value_b).float() ** 2).mean().item() < 1.0


def test_dynamic_cache_clone_preserves_contents_with_distinct_page_tables() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache = Qwen3DynamicCache(config, allocator=allocator)
    key = torch.randn(1, 2, 3, 16)
    value = torch.randn(1, 2, 3, 16)
    cache.update(key, value, layer_idx=1)

    cloned = cache.clone()

    assert torch.equal(cloned.visible_key_cache(1), key)
    assert torch.equal(cloned.visible_value_cache(1), value)
    assert cloned.page_tables[1][0] != cache.page_tables[1][0]


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


def test_dynamic_cache_clone_preserves_turboquant_rows() -> None:
    config = _tiny_config()
    cache = Qwen3DynamicCache(
        config,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=2,
    )
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)
    cache.update(key, value, layer_idx=1)

    cloned = cache.clone()

    assert isinstance(cloned.turboquant_rows[1][0], TurboQuantKVRow)
    assert ((cloned.visible_key_cache(1) - key).float() ** 2).mean().item() < 1.0
    assert ((cloned.visible_value_cache(1) - value).float() ** 2).mean().item() < 1.0


def test_dynamic_cache_reserve_sequence_capacity_preallocates_pages_and_visible_buffers() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache = Qwen3DynamicCache(config, allocator=allocator)
    cache.reserve_sequence_capacity(config.cache_block_size * 20)
    key = torch.randn(1, 2, 1, 16)
    value = torch.randn(1, 2, 1, 16)

    visible_key, visible_value, _ = cache.update(key, value, layer_idx=1)

    assert allocator.layers[1].capacity() >= 20
    assert cache.visible_cache_capacities[1] >= config.cache_block_size * 20
    assert visible_key.shape[-2] == 1
    assert visible_value.shape[-2] == 1


def test_dynamic_cache_update_can_skip_dense_full_attention_materialization() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config, maintain_full_attention_mirror=False)
    cache = Qwen3DynamicCache(config, allocator=allocator)
    key = torch.randn(1, 2, 5, 16)
    value = torch.randn(1, 2, 5, 16)

    visible_key, visible_value, past_lengths = cache.update(key, value, layer_idx=1, require_dense_cache=False)
    paged_state = cache.paged_attention_state(1)

    assert visible_key is None
    assert visible_value is None
    assert tuple(int(length) for length in past_lengths.tolist()) == (0,)
    assert paged_state is not None
    key_pages, value_pages, page_table, visible_lengths = paged_state
    assert key_pages.shape[1:] == (2, config.cache_block_size, 16)
    assert value_pages.shape[1:] == (2, config.cache_block_size, 16)
    assert page_table.shape == (1, 1)
    assert page_table.dtype == torch.int32
    assert tuple(int(length) for length in visible_lengths.tolist()) == (5,)
    assert torch.equal(cache.visible_key_cache(1), key)
    assert torch.equal(cache.visible_value_cache(1), value)


def test_page_allocator_trim_releases_idle_page_storage() -> None:
    config = _tiny_config()
    allocator = Qwen3PageAllocator(config)
    cache = Qwen3DynamicCache(config, allocator=allocator)
    key = torch.randn(1, 2, config.cache_block_size, 16)
    value = torch.randn(1, 2, config.cache_block_size, 16)

    cache.update(key, value, layer_idx=1)
    cache.release()
    trimmed_pages = allocator.trim()

    assert trimmed_pages > 0
    assert allocator.layers[1].key_pages is None
    assert allocator.layers[1].value_pages is None
    assert allocator.layers[1].free_pages == []


def test_recurrent_gated_delta_rule_single_token_fast_path_matches_general_path() -> None:
    with _temporary_torch_seed(0):
        query = torch.randn(2, 1, 3, 4)
        key = torch.randn(2, 1, 3, 4)
        value = torch.randn(2, 1, 3, 5)
        g = torch.randn(2, 1, 3)
        beta = torch.sigmoid(torch.randn(2, 1, 3))
        initial_state = torch.randn(2, 3, 4, 5)

    output, final_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )

    initial_dtype = query.dtype
    q = query.transpose(1, 2).contiguous().to(torch.float32)
    k = key.transpose(1, 2).contiguous().to(torch.float32)
    v = value.transpose(1, 2).contiguous().to(torch.float32)
    b = beta.transpose(1, 2).contiguous().to(torch.float32)
    gg = g.transpose(1, 2).contiguous().to(torch.float32)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)

    state = initial_state.to(torch.float32)
    g_t = gg[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = b[:, :, 0].unsqueeze(-1)
    state = state * g_t
    kv_mem = (state * k[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (v[:, :, 0] - kv_mem) * beta_t
    state = state + k[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    expected = (state * q[:, :, 0].unsqueeze(-1)).sum(dim=-2).unsqueeze(1).to(initial_dtype)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-4)
    assert final_state is not None
    assert torch.allclose(final_state, state, atol=1e-5, rtol=1e-4)


def test_causal_conv1d_update_single_token_fast_path_matches_grouped_conv() -> None:
    with _temporary_torch_seed(0):
        hidden_states = torch.randn(2, 6, 1)
        conv_state = torch.randn(2, 6, 4)
        weight = torch.randn(6, 4)
        bias = torch.randn(6)

    fast_state = conv_state.clone()
    fast = torch_causal_conv1d_update(hidden_states, fast_state, weight, bias)

    baseline_state = conv_state.clone()
    hidden_states_new = torch.cat([baseline_state, hidden_states], dim=-1).to(dtype=weight.dtype)
    baseline_state.copy_(hidden_states_new[:, :, -baseline_state.shape[-1] :])
    expected = torch.nn.functional.conv1d(
        hidden_states_new,
        weight.unsqueeze(1),
        bias=bias,
        padding=0,
        groups=hidden_states.shape[1],
    )
    expected = torch.nn.functional.silu(expected[:, :, -1:]).to(dtype=hidden_states.dtype)

    assert torch.allclose(fast, expected, atol=1e-5, rtol=1e-4)
    assert torch.allclose(fast_state, baseline_state, atol=1e-5, rtol=1e-4)


def test_causal_conv1d_update_multi_token_path_matches_grouped_conv() -> None:
    with _temporary_torch_seed(0):
        hidden_states = torch.randn(2, 6, 7)
        conv_state = torch.randn(2, 6, 4)
        weight = torch.randn(6, 4)
        bias = torch.randn(6)

    fast_state = conv_state.clone()
    fast = torch_causal_conv1d_update(hidden_states, fast_state, weight, bias)

    baseline_state = conv_state.clone()
    hidden_states_new = torch.cat([baseline_state, hidden_states], dim=-1).to(dtype=weight.dtype)
    baseline_state.copy_(hidden_states_new[:, :, -baseline_state.shape[-1] :])
    expected = torch.nn.functional.conv1d(
        hidden_states_new,
        weight.unsqueeze(1),
        bias=bias,
        padding=0,
        groups=hidden_states.shape[1],
    )
    expected = torch.nn.functional.silu(expected[:, :, -hidden_states.shape[-1] :]).to(dtype=hidden_states.dtype)

    assert torch.allclose(fast, expected, atol=1e-5, rtol=1e-4)
    assert torch.allclose(fast_state, baseline_state, atol=1e-5, rtol=1e-4)


def test_grouped_query_attention_matches_materialized_kv_reference() -> None:
    with _temporary_torch_seed(0):
        query_states = torch.randn(2, 4, 3, 16)
        key_states = torch.randn(2, 2, 5, 16)
        value_states = torch.randn(2, 2, 5, 16)

    q_positions = torch.tensor([[2, 3, 4], [1, 2, 3]])
    causal_mask = torch.arange(5).view(1, 1, -1) > q_positions[:, :, None]
    visible_mask = torch.arange(5).view(1, -1) < torch.tensor([[5], [4]])
    key_padding_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )

    grouped_output = grouped_query_attention(
        query_states,
        key_states,
        value_states,
        scaling=16**-0.5,
        causal_mask=causal_mask,
        visible_mask=visible_mask,
        key_padding_mask=key_padding_mask,
    )

    materialized_key_states = repeat_kv(key_states, 2)
    materialized_value_states = repeat_kv(value_states, 2)
    attn_scores = torch.matmul(query_states, materialized_key_states.transpose(-1, -2)) * (16**-0.5)
    attn_scores = attn_scores.masked_fill(causal_mask[:, None, :, :], float("-inf"))
    attn_scores = attn_scores.masked_fill(~visible_mask[:, None, None, :], float("-inf"))
    attn_scores = attn_scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
    expected = torch.matmul(attn_probs, materialized_value_states)

    assert torch.allclose(grouped_output, expected, atol=1e-5, rtol=1e-4)


def test_qwen3_attention_incremental_cache_matches_full_forward_without_repeat_kv_materialization() -> None:
    with _temporary_torch_seed(0):
        config = _tiny_config()
        attention = Qwen3Attention(config, layer_idx=1).eval()
        rotary = Qwen3TextRotaryEmbedding(config)
        prompt_states = torch.randn(1, 5, config.hidden_size)
        append_states = torch.randn(1, 2, config.hidden_size)
        full_states = torch.cat([prompt_states, append_states], dim=1)

    with torch.no_grad():
        full_position_ids = torch.arange(full_states.shape[1]).view(1, -1)
        full_embeddings = rotary(full_states, full_position_ids)
        full_output = attention(full_states, full_embeddings)

        cache = Qwen3DynamicCache(config)
        prompt_position_ids = torch.arange(prompt_states.shape[1]).view(1, -1)
        prompt_embeddings = rotary(prompt_states, prompt_position_ids)
        _ = attention(prompt_states, prompt_embeddings, past_key_values=cache)

        append_position_ids = torch.arange(prompt_states.shape[1], full_states.shape[1]).view(1, -1)
        append_embeddings = rotary(append_states, append_position_ids)
        append_output = attention(append_states, append_embeddings, past_key_values=cache)

    assert torch.allclose(append_output, full_output[:, -append_states.shape[1] :], atol=1e-5, rtol=1e-4)


def test_sparse_moe_routing_assignments_match_reference_without_expert_mask_materialization() -> None:
    with _temporary_torch_seed(0):
        block = Qwen3SparseMoeBlock(_tiny_moe_config())
        hidden_states = torch.randn(2, 3, block._config.hidden_size)

    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_logits = block.gate(flat_hidden_states)
    routing_weights, selected_experts, usage = block._route_tokens(router_logits, hidden_dtype=flat_hidden_states.dtype)
    sorted_token_idx, sorted_route_idx, expert_offsets = block._sorted_routing_assignments(selected_experts, usage)

    reference_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    reference_weights, reference_selected = torch.topk(reference_weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        reference_weights = reference_weights / reference_weights.sum(dim=-1, keepdim=True)
    reference_usage = torch.bincount(reference_selected.reshape(-1), minlength=block.num_experts)
    reference_mask = torch.nn.functional.one_hot(reference_selected, num_classes=block.num_experts).permute(2, 1, 0)

    assert torch.allclose(routing_weights.float(), reference_weights.float(), atol=1e-5, rtol=1e-4)
    assert torch.equal(selected_experts, reference_selected)
    assert torch.equal(usage.to(dtype=torch.long), reference_usage.to(dtype=torch.long))

    for expert_idx in range(block.num_experts):
        start_idx = int(expert_offsets[expert_idx].item())
        end_idx = int(expert_offsets[expert_idx + 1].item())
        got_pairs = torch.stack([sorted_token_idx[start_idx:end_idx], sorted_route_idx[start_idx:end_idx]], dim=-1)
        ref_route_idx, ref_token_idx = torch.where(reference_mask[expert_idx])
        ref_pairs = torch.stack([ref_token_idx, ref_route_idx], dim=-1)
        if got_pairs.numel() == 0 and ref_pairs.numel() == 0:
            continue
        got_order = torch.argsort(got_pairs[:, 0] * block.top_k + got_pairs[:, 1])
        ref_order = torch.argsort(ref_pairs[:, 0] * block.top_k + ref_pairs[:, 1])
        assert torch.equal(got_pairs.index_select(0, got_order), ref_pairs.index_select(0, ref_order))


def test_sparse_moe_compacted_execution_matches_reference() -> None:
    with _temporary_torch_seed(0):
        block = Qwen3SparseMoeBlock(_tiny_moe_config()).eval()
        hidden_states = torch.randn(2, 3, block._config.hidden_size)

    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_logits = block.gate(flat_hidden_states)
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(dtype=flat_hidden_states.dtype)

    reference_hidden_states = flat_hidden_states.new_zeros(flat_hidden_states.shape)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
    for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten().tolist():
        route_idx, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat_hidden_states.index_select(0, token_idx)
        current_routing_weights = routing_weights[token_idx, route_idx, None]
        current_hidden_states = block.experts[expert_idx](current_state) * current_routing_weights
        reference_hidden_states.index_add_(0, token_idx, current_hidden_states.to(dtype=flat_hidden_states.dtype))

    shared_expert_output = torch.sigmoid(block.shared_expert_gate(flat_hidden_states)) * block.shared_expert(flat_hidden_states)
    reference_hidden_states = reference_hidden_states + shared_expert_output

    with torch.no_grad():
        compacted_hidden_states, compacted_router_logits = block(hidden_states)

    assert torch.allclose(compacted_hidden_states, reference_hidden_states.reshape_as(hidden_states), atol=1e-5, rtol=1e-4)
    assert torch.allclose(compacted_router_logits, router_logits, atol=1e-5, rtol=1e-4)


def test_qwen3_allows_out_of_range_padding_idx_for_tiny_configs() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_config())
    assert model.model.embed_tokens.padding_idx is None


def test_qwen3_runtime_can_pin_first_sparse_moe_layers() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_moe_config())

    model.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_expert_layers=1,
    )

    sparse_layers = [layer.mlp for layer in model.model.layers if isinstance(layer.mlp, Qwen3SparseMoeBlock)]
    assert len(sparse_layers) == 3
    assert sparse_layers[0].resident_experts is True
    assert sparse_layers[0].offload_experts is False
    assert all(layer.resident_experts is False for layer in sparse_layers[1:])
    assert all(layer.offload_experts is True for layer in sparse_layers[1:])


def test_qwen3_runtime_can_pin_sparse_moe_layers_by_decoder_index() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_moe_config())

    model.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        resident_expert_layer_indices=(1,),
        cached_experts_per_layer=3,
    )

    resident_layer_indices = [
        layer_idx
        for layer_idx, layer in enumerate(model.model.layers)
        if isinstance(layer.mlp, Qwen3SparseMoeBlock) and layer.mlp.resident_experts
    ]
    assert resident_layer_indices == [1]
    offloaded_sparse_layers = [layer.mlp for layer in model.model.layers if isinstance(layer.mlp, Qwen3SparseMoeBlock) and layer.mlp.offload_experts]
    assert all(layer.cached_experts_per_layer == 3 for layer in offloaded_sparse_layers)


def test_sparse_moe_cached_expert_materialization_copies_weights_without_aliasing_source() -> None:
    block = Qwen3SparseMoeBlock(_tiny_moe_config())
    block.configure_runtime(
        torch.device("cpu"),
        offload_experts=True,
        cached_experts_per_layer=1,
    )

    source = block.experts[0]
    cached = block._get_cached_expert(0)

    assert cached is not None
    assert cached is not source
    for source_parameter, cached_parameter in zip(source.parameters(), cached.parameters()):
        assert torch.equal(source_parameter, cached_parameter)
        assert source_parameter.data_ptr() != cached_parameter.data_ptr()


@pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU is required for AutoRound MoE staging")
def test_sparse_moe_cached_expert_materialization_supports_autoround_payloads() -> None:
    block = Qwen3SparseMoeBlock(_tiny_moe_config())
    replace_linear_modules(
        block,
        QuantizationConfig(
            quant_method="auto-round",
            bits=4,
            group_size=128,
            data_type="int",
            sym=True,
            packing_format="auto_round:auto_gptq",
            block_name_to_quantize=("experts",),
        ),
        compute_dtype=torch.bfloat16,
    )
    block.configure_runtime(
        torch.device("xpu"),
        offload_experts=True,
        expert_quant="int4",
        cached_experts_per_layer=1,
    )

    cached = block._get_cached_expert(0)

    assert cached is not None
    assert isinstance(block.experts[0].gate_proj, AutoRoundGPTQLinear)
    assert isinstance(cached.gate_proj, XPUInt4Linear)
    assert isinstance(cached.up_proj, XPUInt4Linear)
    assert isinstance(cached.down_proj, XPUInt4Linear)


def test_engine_resolves_explicit_resident_expert_indices() -> None:
    resolved = AnnaQwen3_5TextEngine._resolve_resident_expert_layer_indices(
        requested_layers=None,
        requested_indices=(2, 0, 2),
        config=type("ConfigBox", (), {"text_config": _tiny_moe_config()})(),
        resolved_offload_mode="experts",
    )

    assert resolved == (0, 2)


def test_engine_leaves_resident_expert_indices_for_auto_estimation_when_unspecified() -> None:
    resolved = AnnaQwen3_5TextEngine._resolve_resident_expert_layer_indices(
        requested_layers=None,
        requested_indices=None,
        config=type("ConfigBox", (), {"text_config": _tiny_moe_config()})(),
        resolved_offload_mode="experts",
    )

    assert resolved is None


def test_engine_runtime_weight_quantization_converts_dense_text_linears() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_config())
    before_bytes = AnnaQwen3_5TextEngine._module_nbytes(model)

    replacements = AnnaQwen3_5TextEngine._apply_runtime_weight_quantization(
        model=model,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    after_bytes = AnnaQwen3_5TextEngine._module_nbytes(model)

    assert replacements > 0
    assert after_bytes < before_bytes
    assert isinstance(model.model.layers[0].linear_attn.in_proj_qkv, XPUInt4Linear)
    assert isinstance(model.lm_head, XPUInt4Linear)


def test_engine_runtime_weight_quantization_skips_sparse_expert_modules() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_moe_config())

    AnnaQwen3_5TextEngine._apply_runtime_weight_quantization(
        model=model,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )

    assert isinstance(model.model.layers[0].mlp.shared_expert.gate_proj, XPUInt4Linear)
    assert isinstance(model.model.layers[0].mlp.experts[0].gate_proj, torch.nn.Linear)
    assert isinstance(model.lm_head, XPUInt4Linear)


def test_runtime_weight_quantized_lm_head_preserves_forward_shapes() -> None:
    model = Qwen3_5TextForCausalLM(_tiny_config())
    model.configure_runtime(torch.device("cpu"))
    AnnaQwen3_5TextEngine._apply_runtime_weight_quantization(
        model=model,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        use_cache=True,
        logits_to_keep=1,
    )

    assert outputs.logits.shape == (1, 1, model.config.vocab_size)


def test_qwen3_5_lm_head_topk_matches_full_logits() -> None:
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
    )
    model = Qwen3_5TextForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    full = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits
    topk = model.forward_topk(input_ids=input_ids, use_cache=False, logits_to_keep=1, top_k=5)
    reference_values, reference_indices = torch.topk(full, k=5, dim=-1)

    assert torch.allclose(topk.candidate_logits, reference_values)
    assert torch.equal(topk.candidate_token_ids, reference_indices)
