from __future__ import annotations

import argparse

import uvicorn

from anna.api.app import create_app
from anna.core.config import ServeSettings
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.engine import AnnaEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Anna with an OpenAI-compatible API.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name exposed through the API.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
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
    app = create_app(engine)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
