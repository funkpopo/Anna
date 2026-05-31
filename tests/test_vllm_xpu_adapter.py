from __future__ import annotations

from types import SimpleNamespace

import torch

from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.runtime.qwen3_5_text_engine import EngineOptimizationConfig, GenerationConfig, TextGenerationResult
from anna.runtime.slot_model_runner import SlotModelRunner
from anna.vllm_compat import RequestOutput, SamplingParams
from anna_vllm_xpu import AnnaVLLMXPURuntimeAdapter, build_platform_capabilities


class _FakeEngine:
    default_model_id = "fake-xpu-model"

    def __init__(self) -> None:
        self.device_context = SimpleNamespace(
            dtype=torch.bfloat16,
            get_memory_info=lambda: SimpleNamespace(device_name="Arc Test GPU"),
        )
        self.optimization_config = EngineOptimizationConfig(kv_cache_quantization="turboquant", kv_cache_quant_bits=4)
        self.slot_model_runner = None
        self.calls: list[tuple[str, GenerationConfig]] = []

    def generate_text(self, prompt: str, *, config: GenerationConfig) -> TextGenerationResult:
        self.calls.append((prompt, config))
        return TextGenerationResult(
            text=f"out:{prompt}",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


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


def test_build_platform_capabilities_reports_attention_backend_from_fused_health() -> None:
    capabilities = build_platform_capabilities(
        dtype=torch.float16,
        health_report={
            "available": {
                "paged_gqa_decode_fused": True,
                "lm_head_int4_topk_fused": True,
                "moe_grouped_int4_mlp_fused": False,
            }
        },
    )

    assert capabilities.device_type == "xpu"
    assert capabilities.attention_backend.name == "anna.paged_gqa"
    assert capabilities.attention_backend.paged_decode is True
    assert capabilities.attention_backend.fallback is None
    assert capabilities.fused_ops["lm_head_int4_topk_fused"] is True
    assert capabilities.supported_dtypes[0] == "torch.float16"


def test_adapter_converts_sampling_and_outputs_without_vllm_dependency() -> None:
    engine = _FakeEngine()
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    output = adapter.generate_one(
        "hello",
        SamplingParams(max_tokens=7, temperature=0.0, top_k=1),
        request_id="req-1",
    )

    assert isinstance(output, RequestOutput)
    assert output.request_id == "req-1"
    assert output.prompt == "hello"
    assert output.outputs[0].text == "out:hello"
    assert output.outputs[0].finish_reason == "stop"
    assert engine.calls[0][1].max_new_tokens == 7
    assert engine.calls[0][1].temperature == 0.0
    assert engine.calls[0][1].top_k == 1

    health = adapter.health()
    assert health["runtime_adapter"] == "anna_vllm_xpu"
    assert health["level"] == 3
    assert health["integrated_vllm_worker"] is False
    assert health["slot_model_runner_enabled"] is False


def test_adapter_exposes_slot_model_runner_decode_inputs_and_kv_config() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    slot = engine.slot_model_runner.admit_prefill(
        "req-a",
        prompt_length=4,
        max_new_tokens=2,
        sampling_params={"temperature": 0.0, "top_k": 1},
    )
    engine.slot_model_runner.mark_prefilled("req-a", next_input_id=101)
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    kv_config = adapter.kv_cache_config()
    assert kv_config is not None
    assert kv_config.block_size == 4
    assert kv_config.max_slots == 2
    assert kv_config.total_blocks == 8
    assert kv_config.quantization == "turboquant"

    inputs = adapter.build_model_runner_inputs(request_ids=["req-a"])
    assert inputs.request_ids == ("req-a",)
    assert inputs.input_ids.tolist() == [[101]]
    assert inputs.slot_ids.tolist() == [slot.slot_id]
    assert inputs.seq_lens.tolist() == [4]

    capabilities = adapter.platform_capabilities()
    assert capabilities.kv_cache == kv_config
    assert capabilities.device_name == "Arc Test GPU"
