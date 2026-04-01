from __future__ import annotations

import argparse
import statistics
import time

from anna.core.config import BenchmarkSettings, parse_resident_expert_layer_indices
from anna.core.logging import setup_logging
from anna.core.model_family import inspect_model_family
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.engine import AnnaEngine, GenerationConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Anna on a local model directory.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name used in benchmark output and logs.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Optional local image path.")
    parser.add_argument("--video", default=None, help="Optional local video path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
        help="Optional torch.compile mode for the text generation path.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Request fullgraph capture for torch.compile when --compile-mode is enabled.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=_non_negative_int,
        default=0,
        help="Split long text-only prefills into token chunks. Set 0 to let Anna auto-size the chunk on XPU.",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=_non_negative_int,
        default=0,
        help="Keep up to N text-only prompt KV caches resident for exact prompt reuse. Set 0 to disable.",
    )
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Log synchronized XPU forward timings and memory stats for prefill/decode profiling.",
    )
    parser.add_argument("--offload-mode", choices=("auto", "none", "experts"), default="auto")
    parser.add_argument(
        "--offload-vision",
        action="store_true",
        help="Keep the vision tower on CPU even when the execution device is XPU. Useful for text-only benchmarking on tight memory budgets.",
    )
    parser.add_argument(
        "--expert-quant",
        choices=("auto", "none", "int4"),
        default="auto",
        help="Quantization used for expert weights executed on XPU. 'auto' enables int4 for experts offload on XPU.",
    )
    parser.add_argument(
        "--weight-quant",
        choices=("auto", "none", "int4"),
        default="auto",
        help="Quantization used for dense language-model linear weights executed on XPU. 'auto' enables int4 when the model is oversized for available XPU memory.",
    )
    parser.add_argument(
        "--resident-expert-layers",
        type=int,
        default=None,
        help="Keep the first N sparse MoE layers fully resident on the execution device. Omit to auto-estimate in experts offload mode.",
    )
    parser.add_argument(
        "--resident-expert-layer-indices",
        default=None,
        help="Comma-separated 0-based decoder layer indices to keep fully resident on the execution device. Overrides --resident-expert-layers.",
    )
    parser.add_argument(
        "--cached-experts-per-layer",
        type=int,
        default=None,
        help="Max number of offloaded experts to keep cached on XPU per sparse MoE layer. Omit to auto-estimate; set 0 to disable.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=None,
        help="Completion token limit. Defaults to the model config/generation_config value when present; otherwise Anna auto-estimates a safe limit.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--log-level", default="info")
    return parser


def _build_messages(settings: BenchmarkSettings) -> list[dict]:
    if settings.image is None and settings.video is None:
        return [{"role": "user", "content": settings.prompt}]

    content: list[dict[str, object]] = [{"type": "text", "text": settings.prompt}]
    if settings.image is not None:
        content.append({"type": "image_url", "image_url": {"url": settings.image}})
    if settings.video is not None:
        content.append({"type": "video_url", "video_url": {"url": settings.video}})
    return [{"role": "user", "content": content}]


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    family_info = inspect_model_family(model_dir)
    if family_info.family == "tts":
        raise SystemExit("The selected model is a Qwen3-TTS model. Benchmark support is currently text-only; use anna-speak or anna-serve.")
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    settings = BenchmarkSettings(
        model_dir=model_dir,
        prompt=args.prompt,
        image=args.image,
        video=args.video,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        prefill_chunk_size=args.prefill_chunk_size,
        prompt_cache_size=args.prompt_cache_size,
        profile_runtime=args.profile_runtime,
        offload_mode=args.offload_mode,
        offload_vision=args.offload_vision,
        expert_quant=args.expert_quant,
        weight_quant=args.weight_quant,
        resident_expert_layers=args.resident_expert_layers,
        resident_expert_layer_indices=parse_resident_expert_layer_indices(args.resident_expert_layer_indices),
        cached_experts_per_layer=args.cached_experts_per_layer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        runs=args.runs,
    )

    setup_logging(args.log_level)
    engine = AnnaEngine.from_model_dir(
        settings.model_dir,
        model_id=settings.model_id,
        device=settings.device,
        dtype=settings.dtype,
        compile_mode=settings.compile_mode,
        compile_fullgraph=settings.compile_fullgraph,
        prefill_chunk_size=settings.prefill_chunk_size,
        prompt_cache_size=settings.prompt_cache_size,
        profile_runtime=settings.profile_runtime,
        offload_mode=settings.offload_mode,
        offload_vision=settings.offload_vision,
        expert_quant=settings.expert_quant,
        weight_quant=settings.weight_quant,
        resident_expert_layers=settings.resident_expert_layers,
        resident_expert_layer_indices=settings.resident_expert_layer_indices,
        cached_experts_per_layer=settings.cached_experts_per_layer,
    )
    generation = GenerationConfig(
        max_new_tokens=(
            settings.max_new_tokens
            if settings.max_new_tokens is not None
            else engine.default_max_completion_tokens
        ),
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        repetition_penalty=settings.repetition_penalty,
    )

    messages = _build_messages(settings)
    latencies: list[float] = []
    completion_tokens: list[int] = []
    prefill_seconds: list[float] = []
    prefill_tokens_per_second: list[float] = []
    ttft_seconds: list[float] = []
    decode_tokens_per_second: list[float] = []
    total_tokens_per_second: list[float] = []

    for _ in range(max(settings.warmup, 0)):
        if settings.image is None and settings.video is None:
            engine.generate_text(settings.prompt, config=generation)
        else:
            engine.generate_chat(messages, config=generation)

    for _ in range(max(settings.runs, 1)):
        start = time.perf_counter()
        if settings.image is None and settings.video is None:
            result = engine.generate_text(settings.prompt, config=generation)
        else:
            result = engine.generate_chat(messages, config=generation)
        latency = time.perf_counter() - start
        latencies.append(latency)
        completion_tokens.append(result.completion_tokens)
        if result.perf is not None:
            prefill_seconds.append(result.perf.prefill_seconds)
            prefill_tokens_per_second.append(result.perf.prefill_tokens_per_second)
            ttft_seconds.append(result.perf.ttft_seconds)
            decode_tokens_per_second.append(result.perf.decode_tokens_per_second)
            total_tokens_per_second.append(result.perf.total_tokens_per_second)

    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(completion_tokens)
    tokens_per_second = 0.0 if avg_latency <= 0 else avg_tokens / avg_latency
    mode = "multimodal" if settings.image or settings.video else "text"

    print(f"mode={mode}")
    print(f"device={engine.device_context.device}")
    print(f"compute_dtype={engine.device_context.dtype}")
    print(f"offload_mode={engine.offload_mode}")
    print(f"offload_vision={engine.offload_vision}")
    print(f"compile_mode={engine.optimization_config.compile_mode}")
    print(f"compile_fullgraph={engine.optimization_config.compile_fullgraph}")
    print(f"prefill_chunk_size={engine.optimization_config.prefill_chunk_size}")
    print(f"prompt_cache_size={engine.optimization_config.prompt_cache_size}")
    print(f"profile_runtime={engine.optimization_config.profile_runtime}")
    print(f"expert_quant={engine.expert_quant}")
    print(f"weight_quant={engine.weight_quant}")
    print(f"resident_expert_layers={engine.resident_expert_layers}")
    print(f"resident_expert_layer_indices={','.join(str(idx) for idx in engine.resident_expert_layer_indices)}")
    print(f"cached_experts_per_layer={engine.cached_experts_per_layer}")
    print(f"runs={len(latencies)}")
    print(f"avg_latency_seconds={avg_latency:.4f}")
    print(f"min_latency_seconds={min(latencies):.4f}")
    print(f"max_latency_seconds={max(latencies):.4f}")
    print(f"avg_completion_tokens={avg_tokens:.2f}")
    print(f"avg_tokens_per_second={tokens_per_second:.2f}")
    if prefill_seconds:
        print(f"avg_prefill_seconds={statistics.mean(prefill_seconds):.4f}")
        print(f"avg_ttft_seconds={statistics.mean(ttft_seconds):.4f}")
        print(f"avg_prefill_tokens_per_second={statistics.mean(prefill_tokens_per_second):.2f}")
        print(f"avg_decode_tokens_per_second={statistics.mean(decode_tokens_per_second):.2f}")
        print(f"avg_total_tokens_per_second={statistics.mean(total_tokens_per_second):.2f}")


if __name__ == "__main__":
    main()
