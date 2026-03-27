from __future__ import annotations

import argparse

import uvicorn

from anna.api.app import create_app
from anna.core.config import ServeSettings, parse_resident_expert_layer_indices
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.device import RuntimeSafetyPolicy
from anna.runtime.engine import AnnaEngine
from anna.runtime.scheduler import AnnaScheduler


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in the range (0, 1]")
    return parsed


def _safety_factor(value: str) -> float:
    parsed = float(value)
    if parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be >= 1.0")
    return parsed


def _build_safety_policy(settings: ServeSettings) -> RuntimeSafetyPolicy | None:
    if (
        settings.min_free_memory_mib is None
        and settings.reserve_memory_mib is None
        and settings.max_estimated_usage_ratio is None
        and settings.generation_memory_safety_factor is None
    ):
        return None

    defaults = RuntimeSafetyPolicy()
    min_free_bytes = (
        defaults.min_free_bytes
        if settings.min_free_memory_mib is None
        else settings.min_free_memory_mib << 20
    )
    reserve_margin_bytes = (
        defaults.reserve_margin_bytes
        if settings.reserve_memory_mib is None
        else settings.reserve_memory_mib << 20
    )
    max_estimated_usage_ratio = (
        defaults.max_estimated_usage_ratio
        if settings.max_estimated_usage_ratio is None
        else settings.max_estimated_usage_ratio
    )
    generation_memory_safety_factor = (
        defaults.generation_memory_safety_factor
        if settings.generation_memory_safety_factor is None
        else settings.generation_memory_safety_factor
    )
    return RuntimeSafetyPolicy(
        min_free_bytes=min_free_bytes,
        reserve_margin_bytes=reserve_margin_bytes,
        max_estimated_usage_ratio=max_estimated_usage_ratio,
        generation_memory_safety_factor=generation_memory_safety_factor,
    )


def _build_scheduler(engine: AnnaEngine, settings: ServeSettings) -> AnnaScheduler | None:
    if settings.scheduler_max_batch_size <= 1:
        engine.set_scheduler(None)
        return None

    scheduler = AnnaScheduler(
        engine,
        max_batch_size=settings.scheduler_max_batch_size,
        batch_wait_ms=settings.scheduler_batch_wait_ms,
    )
    engine.set_scheduler(scheduler)
    return scheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Anna with an OpenAI-compatible API.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name exposed through the API.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="default_enable_thinking",
        action="store_true",
        default=True,
        help="Enable thinking by default for chat requests that omit enable_thinking/chat_template_kwargs.enable_thinking.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="default_enable_thinking",
        action="store_false",
        help="Disable thinking by default for chat requests that omit enable_thinking/chat_template_kwargs.enable_thinking.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=_positive_int,
        default=None,
        help="Default completion token limit for API requests that omit max_completion_tokens/max_tokens. Defaults to the model config/generation_config value when present; otherwise Anna auto-estimates a safe per-request limit.",
    )
    parser.add_argument(
        "--reasoning-format",
        choices=("none", "deepseek"),
        default="deepseek",
        help="Reasoning output format for chat completions. 'none' keeps <think> tags in content only, 'deepseek' returns answer in content and reasoning in reasoning_content.",
    )
    parser.add_argument("--offload-mode", choices=("auto", "none", "experts"), default="auto")
    parser.add_argument(
        "--offload-vision",
        action="store_true",
        help="Keep the vision tower on CPU even when the execution device is XPU. Useful for text-only serving on tight memory budgets.",
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
        "--min-free-memory-mib",
        type=_non_negative_int,
        default=None,
        help="Minimum free XPU memory required before generation starts. Defaults to 1024 MiB.",
    )
    parser.add_argument(
        "--reserve-memory-mib",
        type=_non_negative_int,
        default=None,
        help="Extra XPU memory margin preserved during request admission. Defaults to 512 MiB.",
    )
    parser.add_argument(
        "--max-estimated-usage-ratio",
        type=_ratio,
        default=None,
        help="Reject requests whose estimated usage exceeds this fraction of total XPU memory. Defaults to 0.9.",
    )
    parser.add_argument(
        "--generation-memory-safety-factor",
        type=_safety_factor,
        default=None,
        help="Multiplier applied to estimated generation memory. Defaults to 2.0.",
    )
    parser.add_argument(
        "--scheduler-max-batch-size",
        type=_positive_int,
        default=1,
        help="Enable continuous batching only when set above 1. Defaults to 1, which serves requests directly.",
    )
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
        default_max_completion_tokens=args.max_completion_tokens,
        default_enable_thinking=args.default_enable_thinking,
        reasoning_format=args.reasoning_format,
        offload_mode=args.offload_mode,
        offload_vision=args.offload_vision,
        expert_quant=args.expert_quant,
        weight_quant=args.weight_quant,
        resident_expert_layers=args.resident_expert_layers,
        resident_expert_layer_indices=parse_resident_expert_layer_indices(args.resident_expert_layer_indices),
        cached_experts_per_layer=args.cached_experts_per_layer,
        min_free_memory_mib=args.min_free_memory_mib,
        reserve_memory_mib=args.reserve_memory_mib,
        max_estimated_usage_ratio=args.max_estimated_usage_ratio,
        generation_memory_safety_factor=args.generation_memory_safety_factor,
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
        safety_policy=_build_safety_policy(settings),
        default_max_completion_tokens=settings.default_max_completion_tokens,
        default_enable_thinking=settings.default_enable_thinking,
        reasoning_format=settings.reasoning_format,
        offload_mode=settings.offload_mode,
        offload_vision=settings.offload_vision,
        expert_quant=settings.expert_quant,
        weight_quant=settings.weight_quant,
        resident_expert_layers=settings.resident_expert_layers,
        resident_expert_layer_indices=settings.resident_expert_layer_indices,
        cached_experts_per_layer=settings.cached_experts_per_layer,
    )
    scheduler = _build_scheduler(engine, settings)
    app = create_app(engine, scheduler=scheduler)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
