from __future__ import annotations

import argparse

import uvicorn

from anna.api.app import create_app
from anna.core.config import ServeSettings, parse_resident_expert_layer_indices
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
    parser.add_argument("--offload-mode", choices=("auto", "none", "experts"), default="auto")
    parser.add_argument(
        "--expert-quant",
        choices=("auto", "none", "int4"),
        default="auto",
        help="Quantization used for expert weights executed on XPU. 'auto' enables int4 for experts offload on XPU.",
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
        offload_mode=args.offload_mode,
        expert_quant=args.expert_quant,
        resident_expert_layers=args.resident_expert_layers,
        resident_expert_layer_indices=parse_resident_expert_layer_indices(args.resident_expert_layer_indices),
        cached_experts_per_layer=args.cached_experts_per_layer,
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
        offload_mode=settings.offload_mode,
        expert_quant=settings.expert_quant,
        resident_expert_layers=settings.resident_expert_layers,
        resident_expert_layer_indices=settings.resident_expert_layer_indices,
        cached_experts_per_layer=settings.cached_experts_per_layer,
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
