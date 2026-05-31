from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from anna.model.fused_ops import fused_op_health_report
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, GenerationConfig, TextGenerationResult
from anna.runtime.slot_model_runner import SlotModelRunner, SlotModelRunnerConfig, resolve_slot_model_runner_config
from anna.vllm_compat.outputs import CompletionOutput, RequestOutput
from anna.vllm_compat.sampling import SamplingParams, sampling_params_to_generation_config


@dataclass(frozen=True, slots=True)
class AnnaXPUAttentionBackend:
    name: str
    paged_decode: bool
    prefill: str
    fallback: str | None = None


@dataclass(frozen=True, slots=True)
class AnnaXPUKVCacheConfig:
    block_size: int
    max_slots: int
    total_blocks: int
    max_blocks_per_seq: int
    quantization: str = "none"
    quant_bits: int = 4


@dataclass(frozen=True, slots=True)
class AnnaXPUPlatformCapabilities:
    device_type: str = "xpu"
    device_name: str = "Intel Arc / Xe-HPG"
    supported_dtypes: tuple[str, ...] = ("torch.bfloat16", "torch.float16", "torch.float32")
    attention_backend: AnnaXPUAttentionBackend = field(
        default_factory=lambda: AnnaXPUAttentionBackend(
            name="anna.paged_gqa",
            paged_decode=False,
            prefill="torch",
            fallback="missing_fused_ops",
        )
    )
    fused_ops: Mapping[str, bool] = field(default_factory=dict)
    kv_cache: AnnaXPUKVCacheConfig | None = None


def _dtype_name(dtype: torch.dtype | str) -> str:
    return str(dtype)


def _attention_backend_from_health(health_report: Mapping[str, object]) -> AnnaXPUAttentionBackend:
    available = health_report.get("available", {})
    if not isinstance(available, Mapping):
        available = {}
    paged_decode = bool(available.get("paged_gqa_decode_fused"))
    prefill = "anna.prefill_attention" if bool(available.get("flashqla_gated_delta_fused")) else "torch"
    fallback = None if paged_decode else "missing_paged_gqa_decode_fused"
    return AnnaXPUAttentionBackend(
        name="anna.paged_gqa",
        paged_decode=paged_decode,
        prefill=prefill,
        fallback=fallback,
    )


def build_platform_capabilities(
    *,
    device_name: str = "Intel Arc / Xe-HPG",
    dtype: torch.dtype | str = torch.bfloat16,
    kv_cache: AnnaXPUKVCacheConfig | None = None,
    health_report: Mapping[str, object] | None = None,
) -> AnnaXPUPlatformCapabilities:
    report = fused_op_health_report() if health_report is None else health_report
    available = report.get("available", {})
    fused_ops = dict(available) if isinstance(available, Mapping) else {}
    supported = tuple(
        dict.fromkeys(
            (
                _dtype_name(dtype),
                "torch.bfloat16",
                "torch.float16",
                "torch.float32",
            )
        )
    )
    return AnnaXPUPlatformCapabilities(
        device_name=device_name,
        supported_dtypes=supported,
        attention_backend=_attention_backend_from_health(report),
        fused_ops=fused_ops,
        kv_cache=kv_cache,
    )


class AnnaVLLMXPURuntimeAdapter:
    """Level 3 vLLM runtime adapter boundary for Anna's XPU backend.

    This class is intentionally a narrow bridge: it exposes the model runner,
    KV-cache, sampling, and output-conversion contracts that a real vLLM worker
    plugin will call, without importing vLLM as a hard dependency.
    """

    def __init__(
        self,
        *,
        engine: AnnaQwen3_5TextEngine | None = None,
        model: str | Path | None = None,
        model_id: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        if engine is None:
            if model is None:
                raise ValueError("Either engine or model must be provided.")
            engine = AnnaQwen3_5TextEngine.from_model_dir(model, model_id=model_id, **engine_kwargs)
        self.engine = engine

    @property
    def model_id(self) -> str:
        return str(getattr(self.engine, "default_model_id", "anna"))

    @property
    def slot_model_runner(self) -> SlotModelRunner | None:
        return getattr(self.engine, "slot_model_runner", None)

    def platform_capabilities(self) -> AnnaXPUPlatformCapabilities:
        kv_cache = self.kv_cache_config()
        memory_info = self.engine.device_context.get_memory_info()
        device_name = "Intel Arc / Xe-HPG"
        if memory_info is not None and getattr(memory_info, "device_name", None):
            device_name = str(memory_info.device_name)
        return build_platform_capabilities(
            device_name=device_name,
            dtype=self.engine.device_context.dtype,
            kv_cache=kv_cache,
        )

    def kv_cache_config(self) -> AnnaXPUKVCacheConfig | None:
        runner = self.slot_model_runner
        if runner is None:
            return None
        config: SlotModelRunnerConfig = runner.config
        return AnnaXPUKVCacheConfig(
            block_size=config.block_size,
            max_slots=config.max_slots,
            total_blocks=config.total_blocks,
            max_blocks_per_seq=config.max_blocks_per_seq,
            quantization=self.engine.optimization_config.kv_cache_quantization,
            quant_bits=self.engine.optimization_config.kv_cache_quant_bits,
        )

    def build_model_runner_inputs(
        self,
        *,
        request_ids: list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> object:
        runner = self.slot_model_runner
        if runner is None:
            raise RuntimeError("Anna slot model runner is not enabled; vLLM runtime batches cannot be built.")
        return runner.build_decode_inputs(request_ids=request_ids, limit=limit)

    @staticmethod
    def sampling_params_to_generation_config(params: SamplingParams | None) -> GenerationConfig:
        return sampling_params_to_generation_config(params)

    def generate_one(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str = "anna-vllm-xpu",
    ) -> RequestOutput:
        config = self.sampling_params_to_generation_config(sampling_params)
        result = self.engine.generate_text(prompt, config=config)
        return self.request_output_from_result(prompt, result, request_id=request_id)

    @staticmethod
    def request_output_from_result(prompt: str, result: TextGenerationResult, *, request_id: str) -> RequestOutput:
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[],
            outputs=[
                CompletionOutput(
                    index=0,
                    text=result.text,
                    token_ids=[],
                    finish_reason=result.finish_reason,
                    stop_reason=result.finish_reason,
                )
            ],
            finished=True,
            metrics=result.perf,
        )

    def health(self) -> dict[str, object]:
        capabilities = self.platform_capabilities()
        return {
            "runtime_adapter": "anna_vllm_xpu",
            "level": 3,
            "integrated_vllm_worker": False,
            "model": self.model_id,
            "platform": capabilities,
            "slot_model_runner_enabled": self.slot_model_runner is not None,
        }
