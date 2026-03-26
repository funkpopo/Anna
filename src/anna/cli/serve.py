from __future__ import annotations

import argparse

import uvicorn

from anna.api.app import create_app
from anna.core.config import ServeSettings
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.engine import AnnaEngine
from anna.runtime.scheduler import AnnaScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Anna with an OpenAI-compatible API.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name exposed through the API.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--scheduler-max-batch-size", type=int, default=4)
    parser.add_argument("--scheduler-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    settings = ServeSettings(
        model_dir=model_dir,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
        scheduler_max_batch_size=args.scheduler_max_batch_size,
        scheduler_batch_wait_ms=args.scheduler_batch_wait_ms,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )

    setup_logging(settings.log_level)
    engine = AnnaEngine.from_model_dir(
        settings.model_dir,
        model_id=settings.model_id,
        device=settings.device,
        dtype=settings.dtype,
    )
    scheduler = AnnaScheduler(
        engine,
        max_batch_size=settings.scheduler_max_batch_size,
        batch_wait_ms=settings.scheduler_batch_wait_ms,
    )
    engine.set_scheduler(scheduler)
    app = create_app(engine, scheduler=scheduler)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
