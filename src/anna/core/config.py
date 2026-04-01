from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def parse_resident_expert_layer_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return ()

    indices: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        indices.append(int(token))
    return tuple(indices)


@dataclass(slots=True)
class ServeSettings:
    model_dir: Path
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    compile_mode: str = "none"
    compile_fullgraph: bool = False
    prefill_chunk_size: int = 0
    prompt_cache_size: int = 0
    profile_runtime: bool = False
    default_max_completion_tokens: int | None = None
    default_enable_thinking: bool = True
    reasoning_format: str = "deepseek"
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    min_free_memory_mib: int | None = None
    reserve_memory_mib: int | None = None
    max_estimated_usage_ratio: float | None = None
    generation_memory_safety_factor: float | None = None
    scheduler_max_batch_size: int = 1
    scheduler_batch_wait_ms: float = 2.0
    metrics_log_interval_seconds: float = 10.0
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


@dataclass(slots=True)
class GenerateSettings:
    model_dir: Path
    prompt: str
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    compile_mode: str = "none"
    compile_fullgraph: bool = False
    prefill_chunk_size: int = 0
    prompt_cache_size: int = 0
    profile_runtime: bool = False
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass(slots=True)
class SpeakSettings:
    model_dir: Path
    text: str
    output: Path
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    language: str | None = None
    speaker: str | None = None
    instruct: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool = False
    response_format: str = "wav"
    max_new_tokens: int | None = None
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50
    non_streaming_mode: bool = True


@dataclass(slots=True)
class BenchmarkSettings:
    model_dir: Path
    prompt: str
    model_id: str | None = None
    image: str | None = None
    video: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    compile_mode: str = "none"
    compile_fullgraph: bool = False
    prefill_chunk_size: int = 0
    prompt_cache_size: int = 0
    profile_runtime: bool = False
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    warmup: int = 1
    runs: int = 3
