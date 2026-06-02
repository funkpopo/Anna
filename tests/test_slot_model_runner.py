from __future__ import annotations

import pytest
import torch

from anna.model import ops as model_ops
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig, Qwen3_5TextModelConfig
from anna.runtime.device import DeviceContext
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, EngineOptimizationConfig
from anna.runtime.slot_model_runner import SlotModelRunner, resolve_slot_model_runner_config


class _FakeModel:
    def __init__(self, config: Qwen3_5TextModelConfig) -> None:
        self.config = config


def _text_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        vocab_size=64,
        max_position_embeddings=16,
        cache_block_size=4,
        layer_types=["full_attention"],
    )


def _engine(*, optimization_config: EngineOptimizationConfig | None = None) -> AnnaQwen3_5TextEngine:
    config = Qwen3_5TextModelConfig(text_config=_text_config())
    return AnnaQwen3_5TextEngine(
        model=_FakeModel(config),
        tokenizer=object(),
        processor=object(),
        model_id="slot-runner-test",
        device_context=DeviceContext.resolve(device="cpu", dtype="float32"),
        optimization_config=optimization_config,
    )


def test_slot_model_runner_resolves_config_from_text_config() -> None:
    resolved = resolve_slot_model_runner_config(
        _text_config(),
        device="cpu",
        max_slots=3,
        total_blocks=0,
        max_blocks_per_seq=0,
        max_batch_size=2,
    )

    assert resolved.max_slots == 3
    assert resolved.block_size == 4
    assert resolved.max_blocks_per_seq == 4
    assert resolved.total_blocks == 12
    assert resolved.max_batch_size == 2
    assert resolved.device == torch.device("cpu")
    assert resolved.num_layers == 1
    assert resolved.num_key_value_heads == 1
    assert resolved.head_dim == 8


def test_slot_model_runner_builds_decode_model_inputs_without_cache_objects() -> None:
    runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )

    slot_a = runner.admit_prefill(
        "req-a",
        prompt_length=3,
        max_new_tokens=4,
        sampling_params={"temperature": 0.0, "top_k": 1},
    )
    slot_b = runner.admit_prefill(
        "req-b",
        prompt_length=5,
        max_new_tokens=4,
        sampling_params={"temperature": 0.8, "top_k": 2},
    )
    runner.mark_prefilled("req-a", next_input_id=11)
    runner.mark_prefilled("req-b", next_input_id=22)

    model_inputs = runner.build_decode_inputs()

    assert model_inputs.request_ids == ("req-a", "req-b")
    assert model_inputs.input_ids.tolist() == [[11], [22]]
    assert model_inputs.slot_ids.tolist() == [slot_a.slot_id, slot_b.slot_id]
    assert model_inputs.epochs.tolist() == [slot_a.epoch, slot_b.epoch]
    assert model_inputs.positions_are_global is True
    assert model_inputs.seq_lens_are_global is True
    assert model_inputs.positions.data_ptr() == runner.kv_manager.seq_lens.data_ptr()
    assert model_inputs.seq_lens.data_ptr() == runner.kv_manager.seq_lens.data_ptr()
    assert model_inputs.batch_positions.tolist() == [3, 5]
    assert model_inputs.batch_seq_lens.tolist() == [3, 5]
    assert model_inputs.decode_token_count == 1
    assert model_inputs.visible_seq_lens.tolist() == [4, 6]
    assert model_inputs.block_tables_are_global is True
    assert model_inputs.block_tables.shape == (2, 4)
    assert model_inputs.sampling_params == (
        {"temperature": 0.0, "top_k": 1},
        {"temperature": 0.8, "top_k": 2},
    )
    assert model_inputs.sampling_batch_params.temperature.tolist() == pytest.approx([0.0, 0.8])
    assert model_inputs.sampling_batch_params.top_k.tolist() == [1, 2]
    assert not hasattr(model_inputs, "past_key_values")
    assert model_inputs.contains_cache_objects is False
    assert model_inputs.boundary_summary() == {
        "batch_size": 2,
        "decode_token_count": 1,
        "contains_cache_objects": False,
        "input_ids_device": "cpu",
        "slot_ids_device": "cpu",
        "positions_are_global": True,
        "seq_lens_are_global": True,
        "block_tables_are_global": True,
        "physical_block_tables": False,
        "block_table_ownership": "logical_slot_metadata",
        "block_tables_shape": (2, 4),
        "seq_lens_shape": (2,),
        "positions_shape": (2,),
        "sampling_batch_params": True,
        "owns_physical_kv_pages": False,
        "physical_kv_layer_count": 0,
        "physical_key_pages_shape": None,
        "physical_value_pages_shape": None,
    }

    repeated = runner.build_decode_inputs(request_ids=["req-a", "req-b"])
    assert repeated.sampling_batch_params is model_inputs.sampling_batch_params
    assert runner.health()["sampling_batch_params_cache"] == {
        "entries": 1,
        "max_entries": 64,
        "hits": 1,
        "misses": 1,
        "evictions": 0,
    }

    runner.advance_decode("req-a", next_input_id=12)
    advanced = runner.build_decode_inputs(request_ids=["req-a"])
    assert advanced.batch_positions.tolist() == [4]
    assert advanced.batch_seq_lens.tolist() == [4]
    assert advanced.visible_seq_lens.tolist() == [5]
    assert advanced.input_ids.tolist() == [[12]]


def test_slot_model_runner_marks_prefilled_batch_from_sampler_tensor() -> None:
    runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    runner.admit_prefill("req-a", prompt_length=3, max_new_tokens=2)
    runner.admit_prefill("req-b", prompt_length=4, max_new_tokens=2)

    marked = runner.mark_prefilled_batch(
        request_ids=["req-a", "req-b"],
        next_input_ids=torch.tensor([[11], [22]], dtype=torch.int32),
        next_input_host_ids=[11, 22],
    )

    assert tuple(slot.request_id for slot in marked) == ("req-a", "req-b")
    model_inputs = runner.build_decode_inputs()
    assert model_inputs.request_ids == ("req-a", "req-b")
    assert model_inputs.input_ids.tolist() == [[11], [22]]
    assert model_inputs.positions_are_global is True
    assert model_inputs.seq_lens_are_global is True
    assert model_inputs.batch_positions.tolist() == [3, 4]
    assert model_inputs.batch_seq_lens.tolist() == [3, 4]
    assert model_inputs.visible_seq_lens.tolist() == [4, 5]
    assert model_inputs.block_tables_are_global is True


def test_slot_model_runner_advances_decode_batch_from_sampler_tensor() -> None:
    runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    runner.admit_prefill("req-a", prompt_length=3, max_new_tokens=3)
    runner.admit_prefill("req-b", prompt_length=4, max_new_tokens=1)
    runner.mark_prefilled("req-a", next_input_id=11, next_input_host_id=11)
    runner.mark_prefilled("req-b", next_input_id=22, next_input_host_id=22)
    decode_inputs = runner.build_decode_inputs()

    advanced = runner.advance_decode_batch(
        request_ids=decode_inputs.request_ids,
        next_input_ids=torch.tensor([[12], [0]], dtype=torch.int32),
        next_input_host_ids=[12, None],
        finished=[False, True],
    )

    assert tuple(slot.request_id for slot in advanced) == ("req-a", "req-b")
    assert advanced[1].status.value == "finished"
    assert runner.active_count == 1
    next_inputs = runner.build_decode_inputs()
    assert next_inputs.request_ids == ("req-a",)
    assert next_inputs.input_ids.tolist() == [[12]]
    assert next_inputs.batch_positions.tolist() == [4]
    assert next_inputs.visible_seq_lens.tolist() == [5]


def test_slot_model_runner_can_emit_explicit_internal_physical_kv_page_bank() -> None:
    text_config = _text_config()
    runner = SlotModelRunner.from_text_config(
        text_config,
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    runner.admit_prefill("req-a", prompt_length=3, max_new_tokens=2)
    runner.admit_prefill("req-b", prompt_length=5, max_new_tokens=2)
    runner.mark_prefilled("req-a", next_input_id=11)
    runner.mark_prefilled("req-b", next_input_id=22)

    with pytest.raises(RuntimeError, match="allocate_physical_kv_page_bank"):
        runner.build_decode_inputs(physical_block_tables=True)

    bank = runner.allocate_physical_kv_page_bank(dtype=torch.float32, num_layers=2)
    assert bank.health() == {
        "allocated": True,
        "device": "cpu",
        "dtype": "torch.float32",
        "num_layers": 2,
        "total_blocks": 8,
        "num_key_value_heads": text_config.num_key_value_heads,
        "block_size": text_config.cache_block_size,
        "head_dim": text_config.head_dim,
        "key_pages_shape": (
            2,
            8,
            text_config.num_key_value_heads,
            text_config.cache_block_size,
            text_config.head_dim,
        ),
        "value_pages_shape": (
            2,
            8,
            text_config.num_key_value_heads,
            text_config.cache_block_size,
            text_config.head_dim,
        ),
    }

    model_inputs = runner.build_decode_inputs(physical_block_tables=True)
    assert model_inputs.physical_block_tables is True
    assert model_inputs.block_table_ownership == "physical"
    assert model_inputs.owns_physical_kv_pages is True
    assert model_inputs.physical_kv_layer_count == 2
    assert model_inputs.boundary_summary()["physical_key_pages_shape"] == bank.health()["key_pages_shape"]

    key_pages, value_pages = model_inputs.physical_pages_for_layer(0)
    key_states = torch.tensor(
        [
            [[[101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]]],
            [[[201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0]]],
        ]
    )
    value_states = key_states + 1000.0

    visible = model_ops.write_slot_decode_kv_to_pages(
        key_states,
        value_states,
        key_pages,
        value_pages,
        model_inputs.block_tables,
        model_inputs.seq_lens,
        slot_ids=model_inputs.slot_ids,
        block_tables_are_global=model_inputs.block_tables_are_global,
        seq_lens_are_global=model_inputs.seq_lens_are_global,
    )

    assert torch.equal(visible, torch.tensor([4, 6], dtype=torch.long))
    assert torch.count_nonzero(key_pages) == 16
    assert torch.count_nonzero(value_pages) == 16
    assert torch.count_nonzero(bank.key_pages[1]) == 0
    assert runner.health()["decode_plan"]["physical_block_tables_available"] is True
    assert runner.health()["kv_write_path"]["tensor_bank_ready"] is True
    assert runner.health()["slot_owned_kv_writes"] is False


def test_qwen_engine_slot_runner_is_disabled_by_default_in_health() -> None:
    engine = _engine(optimization_config=EngineOptimizationConfig(prefill_chunk_size=4))

    engine_health = engine.health()
    health = engine_health["runtime_optimizations"]
    compatibility = engine_health["compatibility"]

    assert engine.slot_model_runner is None
    assert health["slot_runner"] == {"enabled": False}
    assert health["moe_runtime_boundary"] == {
        "layer_count": 0,
        "resident_layers": 0,
        "offload_layers": 0,
        "host_wave_planning_required_layers": 0,
        "xpu_host_planning_blocked_layers": 0,
        "resident_grouped_int4_ready_layers": 0,
        "layers": [],
    }
    assert health["xpu_int4_kernels"]["module_count"] == 0
    assert health["xpu_int4_kernels"]["backend"] == "inactive"
    assert "int4pack_available" in health["xpu_int4_kernels"]
    assert health["sampler"] == {
        "backend": "torch_tensor_fallback",
        "custom_xpu_kernel": False,
        "xpu_kernel_ready": False,
        "xpu_kernel_reason": "custom_xpu_sampler_kernel_not_implemented",
        "batch_params": True,
        "batch_params_cache": True,
        "batch_params_cache_benchmark": "sampling_params_cache",
        "candidate_sampler": True,
        "candidate_sampler_coverage": {
            "top_k": True,
            "top_k_one_deterministic": True,
            "top_k_top_p_min_p": True,
            "positive_penalty_overfetch": True,
        },
        "candidate_penalty_overfetch": True,
        "candidate_penalty_overfetch_requires": {
            "top_k_gt": 0,
            "presence_penalty_gte": 0.0,
            "repetition_penalty_gte": 1.0,
        },
        "direct_prefill_candidates": True,
        "full_vocab_fallback_metric": "sampler_full_vocab_fallback_count",
        "legacy_full_vocab_sort_metric": "sampler_full_vocab_sort_count",
        "full_vocab_fallback_reasons": (
            "top_p_full_logits_sort",
            "min_p_full_logits_softmax",
            "plain_full_logits_multinomial",
        ),
        "full_vocab_fallback_requires_xpu_kernel": (
            "top_p_full_logits_sort",
            "min_p_full_logits_softmax",
            "plain_full_logits_multinomial",
        ),
    }
    assert health["kernel_backends"]["xpu_int4_linear"] == "inactive"
    assert "paged_gqa_decode" in health["kernel_backends"]
    assert compatibility["openai_api"]["enabled"] is True
    assert compatibility["openai_api"]["vllm_runtime_compatible"] is False
    assert compatibility["vllm_offline_shim"]["module"] == "anna.vllm_compat"
    assert compatibility["vllm_runtime_adapter"]["module"] == "anna_vllm_xpu"
    assert compatibility["vllm_runtime_adapter"]["integrated_vllm_worker"] is False
    assert compatibility["vllm_runtime_adapter"]["platform_plugin_entry_point"] == (
        "anna_xpu = anna_vllm_xpu:register_platform"
    )
    assert compatibility["vllm_runtime_adapter"]["integrated_platform_class"] is None
    assert compatibility["vllm_runtime_adapter"]["slot_model_runner_enabled"] is False


def test_qwen_engine_can_enable_experimental_slot_runner_metadata_path() -> None:
    engine = _engine(
        optimization_config=EngineOptimizationConfig(
            prefill_chunk_size=4,
            slot_runner_enabled=True,
            slot_runner_max_slots=2,
            slot_runner_total_blocks=8,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=2,
        )
    )

    assert engine.slot_model_runner is not None
    engine_health = engine.health()
    health = engine_health["runtime_optimizations"]["slot_runner"]
    assert health["enabled"] is True
    assert health["metadata_only"] is True
    assert health["integrated_generation"] is False
    assert health["slot_owned_kv_writes"] is False
    assert health["max_slots"] == 2
    assert health["total_blocks"] == 8
    assert health["max_batch_size"] == 2
    assert health["metadata_tensors"] == {
        "block_tables": True,
        "seq_lens": True,
        "positions_alias_seq_lens": True,
        "slot_epochs": True,
        "slot_active": True,
        "block_refcounts": True,
        "device": "cpu",
        "block_tables_shape": (2, 4),
        "seq_lens_shape": (2,),
        "slot_epochs_shape": (2,),
        "block_refcounts_shape": (8,),
    }
    assert health["decode_plan"] == {
        "contains_cache_objects": False,
        "input_ids": True,
        "slot_ids": True,
        "epochs": True,
        "block_tables_are_global": True,
        "seq_lens_are_global": True,
        "positions_are_global": True,
        "physical_block_tables": False,
        "physical_block_tables_available": False,
        "block_table_ownership": "logical_slot_metadata",
        "owns_physical_kv_pages": False,
        "sampling_batch_params": True,
    }
    assert health["kv_write_path"] == {
        "physical_page_write_helper": True,
        "physical_decode_page_write_helper": True,
        "physical_prefill_page_write_helper": True,
        "external_physical_block_tables_supported": True,
        "internal_block_tables_are_physical": False,
        "prefill_slot_owned_writes": False,
        "decode_slot_owned_writes": False,
        "legacy_cache_object_required_for_forward": True,
        "tensor_bank_ready": False,
    }
    assert health["physical_kv_page_bank"] == {"allocated": False}
    assert health["token_buffers"] == {
        "next_input_device_tensor": True,
        "slot_owned_output_token_buffer": True,
        "host_output_ids_mirror": True,
    }
    assert health["sampling_batch_params_cache"] == {
        "entries": 0,
        "max_entries": 64,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
    }
    assert engine_health["compatibility"]["vllm_runtime_adapter"]["slot_model_runner_enabled"] is True


def test_qwen_engine_rejects_physical_kv_page_bank_without_slot_runner() -> None:
    with pytest.raises(ValueError, match="slot_runner_physical_kv_page_bank=True requires slot_runner_enabled=True"):
        _engine(
            optimization_config=EngineOptimizationConfig(
                slot_runner_physical_kv_page_bank=True,
            )
        )


def test_qwen_engine_can_preallocate_experimental_slot_runner_physical_kv_page_bank() -> None:
    engine = _engine(
        optimization_config=EngineOptimizationConfig(
            prefill_chunk_size=4,
            slot_runner_enabled=True,
            slot_runner_max_slots=2,
            slot_runner_total_blocks=8,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=2,
            slot_runner_physical_kv_page_bank=True,
        )
    )

    assert engine.slot_model_runner is not None
    health = engine.health()["runtime_optimizations"]["slot_runner"]
    assert health["slot_owned_kv_writes"] is False
    assert health["decode_plan"]["physical_block_tables"] is False
    assert health["decode_plan"]["physical_block_tables_available"] is True
    assert health["decode_plan"]["block_table_ownership"] == "logical_slot_metadata"
    assert health["decode_plan"]["owns_physical_kv_pages"] is True
    assert health["kv_write_path"]["internal_block_tables_are_physical"] is False
    assert health["kv_write_path"]["prefill_slot_owned_writes"] is False
    assert health["kv_write_path"]["decode_slot_owned_writes"] is False
    assert health["kv_write_path"]["legacy_cache_object_required_for_forward"] is True
    assert health["kv_write_path"]["tensor_bank_ready"] is True
    assert health["physical_kv_page_bank"] == {
        "allocated": True,
        "device": "cpu",
        "dtype": "torch.float32",
        "num_layers": 1,
        "total_blocks": 8,
        "num_key_value_heads": 1,
        "block_size": 4,
        "head_dim": 8,
        "key_pages_shape": (1, 8, 1, 4, 8),
        "value_pages_shape": (1, 8, 1, 4, 8),
    }
