from __future__ import annotations

import argparse
import statistics
import time

from anna.core.config import BenchmarkSettings
from anna.core.logging import setup_logging
from anna.runtime.engine import AnnaEngine, GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Anna on a local model directory.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Optional local image path.")
    parser.add_argument("--video", default=None, help="Optional local video path.")
    parser.add_argument("--model-id", default=None)
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
    settings = BenchmarkSettings(
        model_dir=args.model_dir,
        prompt=args.prompt,
        image=args.image,
        video=args.video,
        model_id=args.model_id,
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


if __name__ == "__main__":
    main()
