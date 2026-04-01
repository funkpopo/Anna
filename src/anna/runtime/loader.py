from __future__ import annotations

from pathlib import Path

from anna.core.model_family import inspect_model_family
from anna.runtime.engine import AnnaEngine
from anna.runtime.tts_engine import AnnaTTSEngine


def load_engine_from_model_dir(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    compile_mode: str = "none",
    compile_fullgraph: bool = False,
    prefill_chunk_size: int = 0,
    prompt_cache_size: int = 0,
    profile_runtime: bool = False,
    safety_policy=None,
    default_max_completion_tokens: int | None = None,
    default_enable_thinking: bool = True,
    reasoning_format: str = "deepseek",
    offload_mode: str = "auto",
    offload_vision: bool = False,
    expert_quant: str = "auto",
    weight_quant: str = "auto",
    resident_expert_layers: int | None = None,
    resident_expert_layer_indices: tuple[int, ...] | None = None,
    cached_experts_per_layer: int | None = None,
):
    family_info = inspect_model_family(model_dir)
    if family_info.family == "tts":
        return AnnaTTSEngine.from_model_dir(
            model_dir,
            model_id=model_id,
            device=device,
            dtype=dtype,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            prefill_chunk_size=prefill_chunk_size,
            prompt_cache_size=prompt_cache_size,
            profile_runtime=profile_runtime,
            safety_policy=safety_policy,
            default_max_completion_tokens=default_max_completion_tokens,
            default_enable_thinking=default_enable_thinking,
            reasoning_format=reasoning_format,
            offload_mode=offload_mode,
            offload_vision=offload_vision,
            expert_quant=expert_quant,
            weight_quant=weight_quant,
            resident_expert_layers=resident_expert_layers,
            resident_expert_layer_indices=resident_expert_layer_indices,
            cached_experts_per_layer=cached_experts_per_layer,
        )

    return AnnaEngine.from_model_dir(
        model_dir,
        model_id=model_id,
        device=device,
        dtype=dtype,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        prefill_chunk_size=prefill_chunk_size,
        prompt_cache_size=prompt_cache_size,
        profile_runtime=profile_runtime,
        safety_policy=safety_policy,
        default_max_completion_tokens=default_max_completion_tokens,
        default_enable_thinking=default_enable_thinking,
        reasoning_format=reasoning_format,
        offload_mode=offload_mode,
        offload_vision=offload_vision,
        expert_quant=expert_quant,
        weight_quant=weight_quant,
        resident_expert_layers=resident_expert_layers,
        resident_expert_layer_indices=resident_expert_layer_indices,
        cached_experts_per_layer=cached_experts_per_layer,
    )
