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
    prompt_cache_max_tokens: int = 0
    profile_runtime: bool = False
    kv_cache_quantization: str = "none"
    kv_cache_quant_bits: int = 4
    kv_cache_residual_len: int = 128
    default_max_completion_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    default_enable_thinking: bool = True
    reasoning_format: str = "deepseek"
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    decode_executor: str = "auto"
    min_free_memory_mib: int | None = None
    reserve_memory_mib: int | None = None
    max_estimated_usage_ratio: float | None = None
    generation_memory_safety_factor: float | None = None
    scheduler_profile: str = "none"
    scheduler_max_batch_size: int = 1
    scheduler_batch_wait_ms: float = 2.0
    scheduler_prefill_interval_steps: int = 1
    scheduler_max_prefill_tokens: int = 0
    scheduler_max_decode_tokens: int = 0
    scheduler_max_waiting_requests: int = 0
    scheduler_dynamic_token_budget: bool = False
    scheduler_skip_batch_wait_when_idle: bool = True
    scheduler_max_queue_wait_ms: float = 0.0
    scheduler_event_driven_prefill_insert: bool = False
    warmup_prefill_tokens: int | None = None
    warmup_decode_steps: int = 8
    warmup_batch_size: int | None = None
    # Performance tuning knobs. ``None`` means "use the built-in default" (or the
    # matching ANNA_* environment variable when explicitly set); a concrete value
    # always overrides both. See docs/tuning.md for the full table.
    xpu_int4_gemv_m_threshold: int | None = None
    xpu_int4_cache_load_workers: int | None = None
    weight_load_pipeline_workers: int | None = None
    torchinductor_cache_dir: Path | None = None
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
    prompt_cache_max_tokens: int = 0
    profile_runtime: bool = False
    kv_cache_quantization: str = "none"
    kv_cache_quant_bits: int = 4
    kv_cache_residual_len: int = 128
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    decode_executor: str = "auto"
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
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
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 20
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
    prompt_cache_max_tokens: int = 0
    profile_runtime: bool = False
    kv_cache_quantization: str = "none"
    kv_cache_quant_bits: int = 4
    kv_cache_residual_len: int = 128
    offload_mode: str = "auto"
    offload_vision: bool = False
    expert_quant: str = "auto"
    weight_quant: str = "auto"
    resident_expert_layers: int | None = None
    resident_expert_layer_indices: tuple[int, ...] | None = None
    cached_experts_per_layer: int | None = None
    decode_executor: str = "auto"
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    warmup: int = 1
    runs: int = 3
