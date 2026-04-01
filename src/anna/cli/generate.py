from __future__ import annotations

import argparse

from anna.core.config import GenerateSettings, parse_resident_expert_layer_indices
from anna.core.logging import setup_logging
from anna.core.qwen_model_family import inspect_qwen_model_family
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, GenerationConfig


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
    parser = argparse.ArgumentParser(description="Generate text with Anna.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name used in logs and API-compatible output.")
    parser.add_argument("--prompt", required=True)
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
        help="Keep the vision tower on CPU even when the execution device is XPU. Useful for text-only generation on tight memory budgets.",
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    qwen_model_family_info = inspect_qwen_model_family(model_dir)
    if qwen_model_family_info.qwen_model_family == "qwen3_tts":
        raise SystemExit("The selected model belongs to the qwen3_tts family. Use anna-speak instead of anna-generate.")
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    settings = GenerateSettings(
        model_dir=model_dir,
        prompt=args.prompt,
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
    )

    setup_logging(args.log_level)
    engine = AnnaQwen3_5TextEngine.from_model_dir(
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
    result = engine.generate_text(
        settings.prompt,
        config=GenerationConfig(
            max_new_tokens=(
                settings.max_new_tokens
                if settings.max_new_tokens is not None
                else engine.default_max_completion_tokens
            ),
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
        ),
    )
    print(result.text)


if __name__ == "__main__":
    main()
