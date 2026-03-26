from __future__ import annotations

import argparse
import statistics
import time

from anna.core.config import BenchmarkSettings
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.engine import AnnaEngine, GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Anna on a local model directory.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name used in benchmark output and logs.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Optional local image path.")
    parser.add_argument("--video", default=None, help="Optional local video path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
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
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    settings = BenchmarkSettings(
        model_dir=model_dir,
        prompt=args.prompt,
        image=args.image,
        video=args.video,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
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
    )
    generation = GenerationConfig(
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        repetition_penalty=settings.repetition_penalty,
    )

    messages = _build_messages(settings)
    latencies: list[float] = []
    completion_tokens: list[int] = []
    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    decode_tokens_per_second: list[float] = []
    end_to_end_tokens_per_second: list[float] = []

    for _ in range(max(settings.warmup, 0)):
        if settings.image is None and settings.video is None:
            engine.profile_text(settings.prompt, config=generation)
        else:
            engine.profile_chat(messages, config=generation)

    for _ in range(max(settings.runs, 1)):
        start = time.perf_counter()
        if settings.image is None and settings.video is None:
            result = engine.profile_text(settings.prompt, config=generation)
        else:
            result = engine.profile_chat(messages, config=generation)
        latency = time.perf_counter() - start
        latencies.append(latency)
        completion_tokens.append(result.completion_tokens)
        if result.prefill_seconds is not None:
            prefill_latencies.append(result.prefill_seconds)
        if result.decode_seconds is not None:
            decode_latencies.append(result.decode_seconds)
        if result.decode_tokens_per_second is not None:
            decode_tokens_per_second.append(result.decode_tokens_per_second)
        if result.end_to_end_tokens_per_second is not None:
            end_to_end_tokens_per_second.append(result.end_to_end_tokens_per_second)

    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(completion_tokens)
    tokens_per_second = 0.0 if avg_latency <= 0 else avg_tokens / avg_latency
    mode = "multimodal" if settings.image or settings.video else "text"

    print(f"mode={mode}")
    print(f"device={engine.device_context.device}")
    print(f"compute_dtype={engine.device_context.dtype}")
    print(f"runs={len(latencies)}")
    print(f"avg_latency_seconds={avg_latency:.4f}")
    print(f"min_latency_seconds={min(latencies):.4f}")
    print(f"max_latency_seconds={max(latencies):.4f}")
    print(f"avg_completion_tokens={avg_tokens:.2f}")
    print(f"avg_tokens_per_second={tokens_per_second:.2f}")
    if prefill_latencies:
        print(f"avg_prefill_seconds={statistics.mean(prefill_latencies):.4f}")
    if decode_latencies:
        print(f"avg_decode_seconds={statistics.mean(decode_latencies):.4f}")
    if decode_tokens_per_second:
        print(f"avg_decode_tokens_per_second={statistics.mean(decode_tokens_per_second):.2f}")
    if end_to_end_tokens_per_second:
        print(f"avg_end_to_end_tokens_per_second={statistics.mean(end_to_end_tokens_per_second):.2f}")


if __name__ == "__main__":
    main()
