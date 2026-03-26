from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn

from anna.api.app import create_app
from anna.core.config import ServeSettings
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.engine import AnnaEngine
from anna.runtime.scheduler import AnnaScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Anna with an OpenAI-compatible API.")
    parser.add_argument("model", nargs="?", help="Local model directory. Equivalent to --model-dir.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-name", default=None, help="Model name exposed through the API.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--scheduler-max-batch-size", type=int, default=4)
    parser.add_argument("--scheduler-max-batched-tokens", type=int, default=None)
    parser.add_argument("--scheduler-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--api-key", default=None, help="Require this API key for /v1/* endpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def _resolve_model_dir_argument(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.model is None and args.model_dir is None:
        parser.error("Provide a local model directory either positionally or via --model-dir.")

    if args.model is not None and args.model_dir is not None:
        positional = Path(args.model).expanduser().resolve()
        optional = Path(args.model_dir).expanduser().resolve()
        if positional != optional:
            parser.error("Positional MODEL and --model-dir point to different locations; provide only one or make them match.")

    return args.model_dir or args.model


def _require_positive(parser: argparse.ArgumentParser, value: int | None, flag_name: str) -> None:
    if value is not None and value <= 0:
        parser.error(f"{flag_name} must be greater than 0.")


def parse_serve_settings(argv: Sequence[str] | None = None) -> ServeSettings:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_dir_arg = _resolve_model_dir_argument(args, parser)
    model_dir = resolve_model_dir(model_dir_arg)
    if args.api_key is not None and not args.api_key.strip():
        parser.error("--api-key must not be empty.")
    _require_positive(parser, args.scheduler_max_batch_size, "--scheduler-max-batch-size")
    _require_positive(parser, args.scheduler_max_batched_tokens, "--scheduler-max-batched-tokens")
    _require_positive(parser, args.max_model_len, "--max-model-len")
    if args.scheduler_batch_wait_ms < 0:
        parser.error("--scheduler-batch-wait-ms must be greater than or equal to 0.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be within (0, 1].")

    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    return ServeSettings(
        model_dir=model_dir,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
        scheduler_max_batch_size=args.scheduler_max_batch_size,
        scheduler_max_batched_tokens=args.scheduler_max_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        api_key=args.api_key,
        scheduler_batch_wait_ms=args.scheduler_batch_wait_ms,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def main() -> None:
    try:
        settings = parse_serve_settings()
        setup_logging(settings.log_level)
        engine = AnnaEngine.from_model_dir(
            settings.model_dir,
            model_id=settings.model_id,
            device=settings.device,
            dtype=settings.dtype,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
        )
        scheduler = AnnaScheduler(
            engine,
            max_batch_size=settings.scheduler_max_batch_size,
            max_batched_tokens=settings.scheduler_max_batched_tokens,
            batch_wait_ms=settings.scheduler_batch_wait_ms,
        )
        engine.set_scheduler(scheduler)
        app = create_app(engine, scheduler=scheduler, api_key=settings.api_key)
    except SystemExit:
        raise
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Failed to start Anna: {exc}") from exc

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
