from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import anna_vllm_xpu.adapter as adapter_mod
from anna.core.hotpath_events import hotpath_event_recorder
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.runtime.paged_kv import SlotCapacityError
from anna.runtime.qwen3_5_text_engine import EngineOptimizationConfig, GenerationConfig, TextGenerationResult
from anna.runtime.service_metrics import AnnaServiceMetrics
from anna.runtime.slot_model_runner import SlotModelRunner
from anna.vllm_compat import RequestOutput, SamplingParams
from anna_vllm_xpu import (
    AnnaVLLMXPURuntimeAdapter,
    AnnaXPUKVCacheConnector,
    AnnaXPUAttentionBackendRegistry,
    build_platform_capabilities,
    build_vllm_plugin_spec,
    extract_execute_model_field,
    extract_execute_model_request_ids,
    extract_execute_model_sampling_params,
    register_platform,
)


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
            prompt_token_ids=[11],
            completion_token_ids=[22],
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
                "flashqla_gated_delta_fused": True,
                "lm_head_int4_topk_fused": True,
                "moe_grouped_int4_mlp_fused": False,
            }
        },
    )

    assert capabilities.device_type == "xpu"
    assert capabilities.attention_backend.name == "anna.paged_gqa"
    assert capabilities.attention_backend.paged_decode is True
    assert capabilities.attention_backend.prefill == "torch_scaled_dot_product_attention"
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
    assert output.prompt_token_ids == [11]
    assert output.outputs[0].text == "out:hello"
    assert output.outputs[0].token_ids == [22]
    assert output.outputs[0].finish_reason == "stop"
    assert engine.calls[0][1].max_new_tokens == 7
    assert engine.calls[0][1].temperature == 0.0
    assert engine.calls[0][1].top_k == 1

    health = adapter.health()
    assert health["runtime_adapter"] == "anna_vllm_xpu"
    assert health["level"] == 3
    assert health["integrated_vllm_worker"] is False
    assert health["slot_model_runner_enabled"] is False
    assert health["platform_plugin_entry_point"] == "anna_xpu = anna_vllm_xpu:register_platform"
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


def test_vllm_plugin_spec_is_discoverable_without_integrated_worker() -> None:
    spec = build_vllm_plugin_spec()

    assert spec.name == "anna_xpu"
    assert spec.platform_entry_point_group == "vllm.platform_plugins"
    assert spec.platform_entry_point == "anna_xpu = anna_vllm_xpu:register_platform"
    assert spec.platform_class is None
    assert spec.worker_class is None
    assert spec.integrated_vllm_worker is False
    assert spec.runtime_adapter_class.endswith("AnnaVLLMXPURuntimeAdapter")
    assert register_platform() is None


def test_pyproject_registers_vllm_platform_plugin_entry_point() -> None:
    spec = build_vllm_plugin_spec()
    pyproject = tomllib.loads((Path.cwd() / "pyproject.toml").read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"][spec.platform_entry_point_group]

    assert entry_points == {"anna_xpu": "anna_vllm_xpu:register_platform"}


def test_adapter_output_mapper_preserves_optional_token_ids_and_logprobs() -> None:
    rich_result = SimpleNamespace(
        text="decoded",
        finish_reason="length",
        prompt_token_ids=[1, 2],
        token_ids=[3, 4],
        cumulative_logprob=-2.5,
        logprobs=[{"3": -1.0}, {"4": -1.5}],
        perf={"total_ms": 4.0},
    )

    output = AnnaVLLMXPURuntimeAdapter.request_output_from_result(
        "prompt",
        rich_result,  # type: ignore[arg-type]
        request_id="req-rich",
    )

    assert output.request_id == "req-rich"
    assert output.prompt_token_ids == [1, 2]
    assert output.outputs[0].text == "decoded"
    assert output.outputs[0].token_ids == [3, 4]
    assert output.outputs[0].cumulative_logprob == -2.5
    assert output.outputs[0].logprobs == [{"3": -1.0}, {"4": -1.5}]
    assert output.outputs[0].finish_reason == "length"
    assert output.outputs[0].stop_reason == "length"
    assert output.metrics == {"total_ms": 4.0}


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
    assert inputs.positions_are_global is True
    assert inputs.seq_lens_are_global is True
    assert inputs.batch_seq_lens.tolist() == [4]

    capabilities = adapter.platform_capabilities()
    assert capabilities.kv_cache == kv_config
    assert capabilities.device_name == "Arc Test GPU"


def test_adapter_imports_external_kv_decode_batch_into_slot_metadata() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=4,
        total_blocks=16,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    connector = adapter.kv_cache_connector()
    assert isinstance(connector, AnnaXPUKVCacheConnector)

    inputs = adapter.import_external_kv_decode_batch(
        request_ids=("req-b", "req-a"),
        input_ids=[202, 101],
        slot_ids=[2, 0],
        block_tables=[
            [5, 6, -1, -1],
            [1, -1, -1, -1],
        ],
        seq_lens=[7, 2],
        epochs=[11, 9],
        sampling_params=(
            {"temperature": 0.7, "top_p": 0.9, "top_k": 8},
            {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
        ),
    )

    assert inputs.request_ids == ("req-b", "req-a")
    assert inputs.input_ids.tolist() == [[202], [101]]
    assert inputs.slot_ids.tolist() == [2, 0]
    assert inputs.epochs.tolist() == [11, 9]
    assert inputs.positions.tolist() == [7, 2]
    assert inputs.positions_are_global is False
    assert inputs.seq_lens.tolist() == [7, 2]
    assert inputs.seq_lens_are_global is False
    assert inputs.decode_token_count == 1
    assert inputs.visible_seq_lens.tolist() == [8, 3]
    assert inputs.block_tables.tolist() == [[5, 6, -1, -1], [1, -1, -1, -1]]
    assert inputs.block_tables_are_global is False
    assert inputs.physical_block_tables is True
    assert inputs.sampling_batch_params.temperature.tolist() == pytest.approx([0.7, 0.0])
    assert inputs.sampling_batch_params.top_k.tolist() == [8, 1]

    manager = engine.slot_model_runner.kv_manager
    assert manager.block_tables.tolist() == [
        [1, -1, -1, -1],
        [-1, -1, -1, -1],
        [5, 6, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert manager.seq_lens.tolist() == [2, 0, 7, 0]
    assert manager.slot_epochs.tolist() == [9, 0, 11, 0]
    assert manager.slot_active.tolist() == [True, False, True, False]
    assert manager.free_slot_count == 4
    assert manager.block_refcounts.tolist() == [0] * 16

    health = adapter.health()
    assert health["kv_cache_connector"] is True


def test_adapter_attention_backend_registry_exposes_decode_and_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = build_platform_capabilities(
        health_report={"available": {"paged_gqa_decode_fused": True}},
    )
    registry = AnnaXPUAttentionBackendRegistry(capabilities, allow_cpu_tensors_for_tests=True)

    assert isinstance(registry, AnnaXPUAttentionBackendRegistry)
    assert registry.name == "anna.paged_gqa"
    assert registry.paged_decode_entrypoint == "anna.model.ops.paged_kv_slot_batch_decode_attention"
    assert registry.prefill_entrypoint == "torch.nn.functional.scaled_dot_product_attention"
    assert registry.prefill_backend == "torch_scaled_dot_product_attention"

    observed: dict[str, torch.Tensor] = {}

    def _fake_paged_decode(
        query_states: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        scaling: float,
        slot_ids: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key_pages, value_pages, scaling, gate
        observed["block_tables"] = block_tables.clone()
        observed["seq_lens"] = seq_lens.clone()
        observed["slot_ids"] = torch.empty(0, dtype=torch.int32) if slot_ids is None else slot_ids.clone()
        return query_states + 1

    monkeypatch.setattr(adapter_mod.model_ops, "paged_kv_slot_batch_decode_attention", _fake_paged_decode)

    query = torch.zeros(1, 2, 1, 8)
    decoded = registry.paged_decode(
        query_states=query,
        key_pages=torch.zeros(4, 1, 4, 8),
        value_pages=torch.zeros(4, 1, 4, 8),
        block_tables=torch.tensor([[0, -1]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.long),
        slot_ids=torch.tensor([0], dtype=torch.int32),
        scaling=8**-0.5,
    )

    assert torch.equal(decoded, torch.ones_like(query))
    assert observed["block_tables"].tolist() == [[0, -1]]
    assert observed["seq_lens"].tolist() == [1]
    assert observed["slot_ids"].tolist() == [0]

    metrics = AnnaServiceMetrics()
    with hotpath_event_recorder(metrics):
        prefill = registry.prefill(
            query_states=torch.randn(1, 2, 4, 8),
            key_states=torch.randn(1, 2, 4, 8),
            value_states=torch.randn(1, 2, 4, 8),
            causal=True,
        )
    assert prefill.shape == (1, 2, 4, 8)
    assert metrics.snapshot().attention_fallback_count == 1

    adapter = AnnaVLLMXPURuntimeAdapter(engine=_FakeEngine())
    assert adapter.health()["attention_backend_registry"] is True


def test_adapter_attention_backend_rejects_paged_decode_when_fused_op_missing() -> None:
    capabilities = build_platform_capabilities(
        health_report={"available": {"paged_gqa_decode_fused": False}},
    )
    registry = AnnaXPUAttentionBackendRegistry(capabilities)

    with pytest.raises(RuntimeError, match="paged decode attention backend is not available"):
        registry.paged_decode(
            query_states=torch.zeros(1, 2, 1, 8),
            key_pages=torch.zeros(4, 1, 4, 8),
            value_pages=torch.zeros(4, 1, 4, 8),
            block_tables=torch.tensor([[0, -1]], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.long),
            scaling=8**-0.5,
        )


def test_adapter_attention_backend_rejects_cpu_tensors_by_default() -> None:
    capabilities = build_platform_capabilities(
        health_report={"available": {"paged_gqa_decode_fused": True}},
    )
    registry = AnnaXPUAttentionBackendRegistry(capabilities)

    with pytest.raises(RuntimeError, match="expects XPU tensors"):
        registry.paged_decode(
            query_states=torch.zeros(1, 2, 1, 8),
            key_pages=torch.zeros(4, 1, 4, 8),
            value_pages=torch.zeros(4, 1, 4, 8),
            block_tables=torch.tensor([[0, -1]], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.long),
            scaling=8**-0.5,
        )

    with pytest.raises(RuntimeError, match="expects XPU tensors"):
        registry.prefill(
            query_states=torch.randn(1, 2, 4, 8),
            key_states=torch.randn(1, 2, 4, 8),
            value_states=torch.randn(1, 2, 4, 8),
            causal=True,
        )


def test_adapter_attention_backend_checks_paged_decode_metadata_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    capabilities = build_platform_capabilities(
        health_report={"available": {"paged_gqa_decode_fused": True}},
    )
    registry = AnnaXPUAttentionBackendRegistry(capabilities)
    checked_fields: list[tuple[str, ...]] = []

    def _record_xpu_tensor_check(**named_tensors: torch.Tensor) -> None:
        checked_fields.append(tuple(named_tensors))

    monkeypatch.setattr(registry, "_require_xpu_tensors", _record_xpu_tensor_check)
    monkeypatch.setattr(
        adapter_mod.model_ops,
        "paged_kv_slot_batch_decode_attention",
        lambda query_states, *args, **kwargs: query_states,
    )

    query = torch.zeros(1, 2, 1, 8)
    registry.paged_decode(
        query_states=query,
        key_pages=torch.zeros(4, 1, 4, 8),
        value_pages=torch.zeros(4, 1, 4, 8),
        block_tables=torch.tensor([[0, -1]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.long),
        slot_ids=torch.tensor([0], dtype=torch.int32),
        gate=torch.ones(1, 2, 1, 1),
        scaling=8**-0.5,
    )

    assert checked_fields == [
        ("query_states", "key_pages", "value_pages", "block_tables", "seq_lens"),
        ("slot_ids", "gate"),
    ]


def test_adapter_external_kv_connector_validates_batch_shapes() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=4,
        max_blocks_per_seq=2,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    try:
        adapter.import_external_kv_decode_batch(
            input_ids=[1, 2],
            slot_ids=[0, 1],
            block_tables=[[0, 1]],
            seq_lens=[2, 2],
        )
    except ValueError as exc:
        assert "one row per slot" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched external block table rows should fail")

    with pytest.raises(ValueError, match="1D input_ids must contain one token id per slot"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1, 2],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
        )

    with pytest.raises(ValueError, match="at least one token"):
        adapter.import_external_kv_decode_batch(
            input_ids=torch.empty((1, 0), dtype=torch.long),
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
        )

    with pytest.raises(ValueError, match="input_ids must be non-negative"):
        adapter.import_external_kv_decode_batch(
            input_ids=[-1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
        )

    try:
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, 4]],
            seq_lens=[2],
        )
    except IndexError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("out-of-range external block id should fail")

    with pytest.raises(ValueError, match="positions must contain one value per slot"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
            positions=[0, 1],
        )

    single_request = adapter.import_external_kv_decode_batch(
        request_ids="single-req",
        input_ids=[1],
        slot_ids=[0],
        block_tables=[[0, -1]],
        seq_lens=[1],
    )
    assert single_request.request_ids == ("single-req",)

    with pytest.raises(ValueError, match="request_ids must be non-empty"):
        adapter.import_external_kv_decode_batch(
            request_ids=[""],
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
        )

    with pytest.raises(ValueError, match="request_ids must be unique"):
        adapter.import_external_kv_decode_batch(
            request_ids=["dup", "dup"],
            input_ids=[1, 2],
            slot_ids=[0, 1],
            block_tables=[[0, -1], [1, -1]],
            seq_lens=[1, 1],
        )


def test_adapter_external_kv_connector_respects_runner_batch_capacity() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=3,
        total_blocks=8,
        max_blocks_per_seq=2,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    with pytest.raises(ValueError, match="exceeds runner max_batch_size=2"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1, 2, 3],
            slot_ids=[0, 1, 2],
            block_tables=[[0, -1], [1, -1], [2, -1]],
            seq_lens=[1, 1, 1],
        )


def test_adapter_external_kv_connector_rejects_inconsistent_decode_metadata() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=2,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    with pytest.raises(ValueError, match="unique"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1, 2],
            slot_ids=[0, 0],
            block_tables=[[0, -1], [1, -1]],
            seq_lens=[1, 1],
        )

    with pytest.raises(ValueError, match="missing required prefix blocks"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[5],
        )

    with pytest.raises(SlotCapacityError, match="max_blocks_per_seq"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, 1]],
            seq_lens=[9],
        )

    with pytest.raises(ValueError, match="positions must be non-negative"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
            positions=[-1],
        )

    with pytest.raises(ValueError, match="epochs must be non-negative"):
        adapter.import_external_kv_decode_batch(
            input_ids=[1],
            slot_ids=[0],
            block_tables=[[0, -1]],
            seq_lens=[1],
            epochs=[-1],
        )


def test_adapter_extracts_vllm_execute_model_request_ids_without_vllm_dependency() -> None:
    request = SimpleNamespace(
        seq_group_metadata_list=[
            SimpleNamespace(request_id="req-a"),
            SimpleNamespace(request_id="req-b"),
        ]
    )

    assert extract_execute_model_request_ids(request) == ("req-a", "req-b")
    assert extract_execute_model_request_ids({"request_ids": ["req-c", "req-d"]}) == ("req-c", "req-d")
    assert extract_execute_model_field({"token_ids": [1, 2]}, "input_ids", "token_ids") == [1, 2]


def test_adapter_extracts_vllm_execute_model_sampling_params_without_vllm_dependency() -> None:
    assert extract_execute_model_sampling_params(
        {"sampling_params_list": [{"temperature": 0.3}, {"top_k": 2}]}
    ) == ({"temperature": 0.3}, {"top_k": 2})
    assert extract_execute_model_sampling_params({"sampling_params": {"temperature": 0.5}}) == (
        {"temperature": 0.5},
    )

    request = SimpleNamespace(
        seq_group_metadata_list=[
            SimpleNamespace(request_id="req-a", sampling_params={"temperature": 0.4, "top_k": 5}),
            SimpleNamespace(request_id="req-b", sampling_params={"temperature": 0.0, "top_k": 1}),
        ]
    )

    assert extract_execute_model_sampling_params(request) == (
        {"temperature": 0.4, "top_k": 5},
        {"temperature": 0.0, "top_k": 1},
    )


def test_adapter_builds_decode_inputs_from_vllm_execute_model_shape() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    engine.slot_model_runner.admit_prefill("req-a", prompt_length=4, max_new_tokens=2)
    engine.slot_model_runner.admit_prefill("req-b", prompt_length=2, max_new_tokens=2)
    engine.slot_model_runner.mark_prefilled("req-a", next_input_id=101)
    engine.slot_model_runner.mark_prefilled("req-b", next_input_id=102)
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    inputs = adapter.build_model_runner_inputs_from_execute_model(
        SimpleNamespace(
            seq_group_metadata_list=[
                SimpleNamespace(request_id="req-b"),
                SimpleNamespace(request_id="req-a"),
            ]
        )
    )

    assert inputs.request_ids == ("req-b", "req-a")
    assert inputs.input_ids.tolist() == [[102], [101]]
    assert inputs.positions_are_global is True
    assert inputs.seq_lens_are_global is True
    assert inputs.batch_positions.tolist() == [2, 4]


def test_adapter_imports_external_kv_decode_batch_from_vllm_execute_model_shape() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=4,
        total_blocks=16,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    inputs = adapter.import_external_kv_decode_batch_from_execute_model(
        {
            "request_ids": ("req-x", "req-y"),
            "token_ids": [31, 32],
            "slots": [1, 3],
            "block_table": [[2, 4, 6, -1], [5, -1, -1, -1]],
            "sequence_lengths": [8, 3],
            "slot_epochs": [12, 14],
            "positions": [7, 2],
            "sampling_params_list": (
                {"temperature": 0.7, "top_k": 4},
                {"temperature": 0.0, "top_k": 1},
            ),
        }
    )

    assert inputs.request_ids == ("req-x", "req-y")
    assert inputs.input_ids.tolist() == [[31], [32]]
    assert inputs.slot_ids.tolist() == [1, 3]
    assert inputs.epochs.tolist() == [12, 14]
    assert inputs.positions.tolist() == [7, 2]
    assert inputs.positions_are_global is False
    assert inputs.seq_lens.tolist() == [8, 3]
    assert inputs.seq_lens_are_global is False
    assert inputs.decode_token_count == 1
    assert inputs.visible_seq_lens.tolist() == [9, 4]
    assert inputs.block_tables.tolist() == [[2, 4, 6, -1], [5, -1, -1, -1]]
    assert inputs.block_tables_are_global is False
    assert inputs.physical_block_tables is True
    assert inputs.sampling_batch_params.temperature.tolist() == pytest.approx([0.7, 0.0])
    assert inputs.sampling_batch_params.top_k.tolist() == [4, 1]
    assert inputs.sampling_batch_params.greedy_rows == (1,)
    assert inputs.sampling_batch_params.top1_rows == ()
    assert inputs.sampling_batch_params.topk_rows == (0,)

    manager = engine.slot_model_runner.kv_manager
    assert manager.block_tables.tolist() == [
        [-1, -1, -1, -1],
        [2, 4, 6, -1],
        [-1, -1, -1, -1],
        [5, -1, -1, -1],
    ]
    assert manager.seq_lens.tolist() == [0, 8, 0, 3]
    assert manager.slot_epochs.tolist() == [0, 12, 0, 14]


def test_adapter_rejects_execute_model_external_kv_shape_without_required_fields() -> None:
    engine = _FakeEngine()
    engine.slot_model_runner = SlotModelRunner.from_text_config(
        _text_config(),
        device="cpu",
        max_slots=2,
        total_blocks=8,
        max_blocks_per_seq=4,
        max_batch_size=2,
    )
    adapter = AnnaVLLMXPURuntimeAdapter(engine=engine)

    with pytest.raises(ValueError, match="missing external KV decode fields"):
        adapter.import_external_kv_decode_batch_from_execute_model(
            {
                "request_ids": ("req-missing",),
                "input_ids": [1],
                "slot_ids": [0],
                "seq_lens": [1],
            }
        )


def test_adapter_rejects_execute_model_shape_without_request_ids() -> None:
    adapter = AnnaVLLMXPURuntimeAdapter(engine=_FakeEngine())

    try:
        adapter.build_model_runner_inputs_from_execute_model(SimpleNamespace(seq_group_metadata_list=[]))
    except ValueError as exc:
        assert "request ids" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing request ids should fail")


def test_adapter_rejects_execute_model_shape_with_empty_request_id() -> None:
    adapter = AnnaVLLMXPURuntimeAdapter(engine=_FakeEngine())

    with pytest.raises(ValueError, match="request ids must be non-empty"):
        adapter.build_model_runner_inputs_from_execute_model({"request_ids": [""]})
