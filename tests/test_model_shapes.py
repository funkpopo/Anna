from __future__ import annotations

import torch

from anna.model.config import Qwen3TextConfig, RopeParameters
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator, Qwen3SparseMoeBlock
from anna.model.qwen import Qwen3ForCausalLM
from anna.runtime.engine import AnnaEngine


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
