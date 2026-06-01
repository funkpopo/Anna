from __future__ import annotations

import pytest
import torch

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


def test_qwen_engine_slot_runner_is_disabled_by_default_in_health() -> None:
    engine = _engine(optimization_config=EngineOptimizationConfig(prefill_chunk_size=4))

    engine_health = engine.health()
    health = engine_health["runtime_optimizations"]
    compatibility = engine_health["compatibility"]

    assert engine.slot_model_runner is None
    assert health["slot_runner"] == {"enabled": False}
    assert health["xpu_int4_kernels"]["module_count"] == 0
    assert health["xpu_int4_kernels"]["backend"] == "inactive"
    assert "int4pack_available" in health["xpu_int4_kernels"]
    assert health["sampler"] == {
        "backend": "torch_tensor_fallback",
        "custom_xpu_kernel": False,
        "batch_params": True,
        "candidate_sampler": True,
        "candidate_penalty_overfetch": True,
        "candidate_penalty_overfetch_requires": {
            "top_k_gt": 0,
            "presence_penalty_gte": 0.0,
            "repetition_penalty_gte": 1.0,
        },
        "direct_prefill_candidates": True,
        "full_vocab_fallback_metric": "sampler_full_vocab_sort_count",
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
    assert health["max_slots"] == 2
    assert health["total_blocks"] == 8
    assert health["max_batch_size"] == 2
    assert engine_health["compatibility"]["vllm_runtime_adapter"]["slot_model_runner_enabled"] is True
