from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from anna.core.gguf_model import has_gguf_model
from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir
from anna.runtime.device import DeviceContext, DeviceMemoryInfo, RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine
from anna.weights.qwen3_5_text_weight_loader import (
    estimate_qwen3_5_text_model_weight_bytes,
    load_qwen3_5_text_model_config,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect whether a Qwen3.5 model will use Anna's runtime XPU int4 layout cache."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--weight-quant",
        choices=("auto", "none", "int4"),
        default="auto",
        help="Weight quantization mode to evaluate. Matches anna-serve/anna-bench semantics.",
    )
    parser.add_argument(
        "--offload-mode",
        choices=("auto", "none", "experts"),
        default="none",
        help="Resolved offload mode used for the auto threshold check.",
    )
    parser.add_argument(
        "--xpu-total-memory-gib",
        type=_positive_float,
        default=None,
        help="Optional total XPU memory override for --weight-quant auto checks.",
    )
    parser.add_argument("--log-level", default="warning")
    return parser


def _device_context_for_memory(total_memory_gib: float | None) -> DeviceContext | SimpleNamespace:
    if total_memory_gib is None:
        return DeviceContext.resolve(device="auto", dtype="auto", model_dtype="bfloat16")
    total_bytes = int(total_memory_gib * (1 << 30))
    return SimpleNamespace(
        device=torch.device("xpu"),
        safety_policy=RuntimeSafetyPolicy(),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=total_bytes,
            total_bytes=total_bytes,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    model_dir = resolve_model_dir(args.model_dir)
    config = load_qwen3_5_text_model_config(model_dir)
    uses_gguf = has_gguf_model(model_dir)
    safetensors_index = model_dir / "model.safetensors.index.json"
    direct_safetensors = model_dir / "model.safetensors"
    sharded_safetensors = tuple(sorted(model_dir.glob("*.safetensors")))
    weight_bytes = estimate_qwen3_5_text_model_weight_bytes(model_dir)
    device_context = _device_context_for_memory(args.xpu_total_memory_gib)
    resolved_weight_quant = AnnaQwen3_5TextEngine._resolve_weight_quant(
        requested_quant=args.weight_quant,
        resolved_offload_mode=args.offload_mode,
        model_path=model_dir,
        config=config,
        device_context=device_context,  # type: ignore[arg-type]
    )
    cache_dir = model_dir / ".anna" / "xpu_int4_cache"
    quant_config = getattr(config, "quantization_config", None)
    quant_config_enabled = bool(getattr(quant_config, "is_enabled", False))

    print(f"model_dir={model_dir}")
    print(f"uses_gguf={uses_gguf}")
    print(f"uses_safetensors={bool(safetensors_index.exists() or direct_safetensors.exists() or sharded_safetensors)}")
    print(f"safetensors_shards={len(sharded_safetensors)}")
    print(f"weight_bytes={weight_bytes}")
    print(f"quantization_config_enabled={quant_config_enabled}")
    print(f"requested_weight_quant={args.weight_quant}")
    print(f"resolved_weight_quant={resolved_weight_quant}")
    print(f"xpu_int4_cache_enabled={resolved_weight_quant == 'int4'}")
    print(f"xpu_int4_cache_dir={cache_dir}")
    print(f"xpu_int4_cache_exists={cache_dir.exists()}")
    print(f"xpu_int4_cache_files={len(tuple(cache_dir.glob('*.pt'))) if cache_dir.exists() else 0}")


if __name__ == "__main__":
    main()
