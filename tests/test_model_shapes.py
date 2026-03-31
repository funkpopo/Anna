from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import anna.model.ops as model_ops
from anna.model.quantization import XPUInt4Linear
from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.ops import (
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    Qwen3SparseMoeBlock,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from anna.model.qwen import Qwen3ForCausalLM
from anna.runtime.engine import AnnaEngine


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


def _tiny_moe_config() -> Qwen3TextConfig:
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
        decoder_sparse_step=1,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[3],
    )


def test_qwen3_forward_shapes() -> None:
    with _temporary_torch_seed(0):
        model = Qwen3ForCausalLM(_tiny_config())
        input_ids = torch.randint(0, 256, (1, 6))
        outputs = model(input_ids=input_ids, use_cache=True)
    assert outputs.logits.shape == (1, 6, 256)
    assert outputs.past_key_values is not None
    assert outputs.past_key_values.get_seq_length() == 6


def test_qwen3_incremental_decode() -> None:
    with _temporary_torch_seed(0):
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
    with _temporary_torch_seed(0):
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


def test_qwen3_allows_out_of_range_padding_idx_for_tiny_configs() -> None:
    model = Qwen3ForCausalLM(_tiny_config())
    assert model.model.embed_tokens.padding_idx is None


def test_qwen3_runtime_can_pin_first_sparse_moe_layers() -> None:
    model = Qwen3ForCausalLM(_tiny_moe_config())

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
    model = Qwen3ForCausalLM(_tiny_moe_config())

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


def test_engine_resolves_explicit_resident_expert_indices() -> None:
    resolved = AnnaEngine._resolve_resident_expert_layer_indices(
        requested_layers=None,
        requested_indices=(2, 0, 2),
        config=type("ConfigBox", (), {"text_config": _tiny_moe_config()})(),
        resolved_offload_mode="experts",
    )

    assert resolved == (0, 2)


def test_engine_leaves_resident_expert_indices_for_auto_estimation_when_unspecified() -> None:
    resolved = AnnaEngine._resolve_resident_expert_layer_indices(
        requested_layers=None,
        requested_indices=None,
        config=type("ConfigBox", (), {"text_config": _tiny_moe_config()})(),
        resolved_offload_mode="experts",
    )

    assert resolved is None


def test_engine_runtime_weight_quantization_converts_dense_text_linears() -> None:
    model = Qwen3ForCausalLM(_tiny_config())
    before_bytes = AnnaEngine._module_nbytes(model)

    replacements = AnnaEngine._apply_runtime_weight_quantization(
        model=model,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    after_bytes = AnnaEngine._module_nbytes(model)

    assert replacements > 0
    assert after_bytes < before_bytes
    assert isinstance(model.model.layers[0].linear_attn.in_proj_qkv, XPUInt4Linear)
    assert isinstance(model.lm_head, XPUInt4Linear)


def test_engine_runtime_weight_quantization_skips_sparse_expert_modules() -> None:
    model = Qwen3ForCausalLM(_tiny_moe_config())

    AnnaEngine._apply_runtime_weight_quantization(
        model=model,
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )

    assert isinstance(model.model.layers[0].mlp.shared_expert.gate_proj, XPUInt4Linear)
    assert isinstance(model.model.layers[0].mlp.experts[0].gate_proj, torch.nn.Linear)
    assert isinstance(model.lm_head, XPUInt4Linear)


def test_runtime_weight_quantized_lm_head_preserves_forward_shapes() -> None:
    model = Qwen3ForCausalLM(_tiny_config())
    model.configure_runtime(torch.device("cpu"))
    AnnaEngine._apply_runtime_weight_quantization(
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
