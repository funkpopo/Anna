from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
from anna.model.quantization import replace_linear_modules

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WeightLoadReport:
    loaded: int
    skipped: int
    quantized_replacements: int = 0


def load_qwen3_5_text_model_config(model_dir: str | Path) -> Qwen3_5TextModelConfig:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return Qwen3_5TextModelConfig.from_model_dir(model_path)


def _iter_weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        unique_paths = sorted(set(index_data["weight_map"].values()))
        return [model_dir / path for path in unique_paths]

    direct_file = model_dir / "model.safetensors"
    if direct_file.exists():
        return [direct_file]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    return files


def estimate_qwen3_5_text_model_weight_bytes(model_dir: str | Path) -> int:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = index_data.get("metadata", {})
        total_size = metadata.get("total_size")
        if total_size is not None:
            return int(total_size)
    return sum(weight_file.stat().st_size for weight_file in _iter_weight_files(model_path))


def build_qwen3_5_text_model(
    config: Qwen3_5TextModelConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3_5TextForConditionalGeneration, int]:
    default_dtype = torch.get_default_dtype()
    build_dtype = dtype if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float32
    try:
        torch.set_default_dtype(build_dtype)
        logger.info("Constructing Qwen3.5 model skeleton on meta device.")
        with torch.device("meta"):
            model = Qwen3_5TextForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)

    logger.info("Replacing Qwen3.5 linear layers with quantized placeholders.")
    quantized_specs = replace_linear_modules(model, config.quantization_config, compute_dtype=dtype)
    if not hasattr(model, "to_empty"):
        raise RuntimeError("The current torch build does not support nn.Module.to_empty, which is required for low-memory model loading.")
    logger.info("Allocating empty Qwen3.5 tensors on %s.", device)
    model.to_empty(device=device)
    logger.info(
        "Constructed Qwen3.5 model skeleton: quantized_placeholders=%s target_device=%s",
        len(quantized_specs),
        device,
    )
    return model, len(quantized_specs)


def load_qwen3_5_text_model_weights(model: Qwen3_5TextForConditionalGeneration, model_dir: str | Path) -> WeightLoadReport:
    model_path = Path(model_dir)
    tensor_targets = {name: tensor for name, tensor in model.named_parameters()}
    tensor_targets.update({name: tensor for name, tensor in model.named_buffers()})
    loaded = 0
    skipped = 0
    weight_files = _iter_weight_files(model_path)
    total_shards = len(weight_files)
    total_bytes = sum(weight_file.stat().st_size for weight_file in weight_files)
    loaded_bytes = 0

    logger.info(
        "Loading Qwen3.5 weights from %s shard(s), total=%s bytes, model_dir=%s",
        total_shards,
        total_bytes,
        model_path,
    )

    for shard_idx, weight_file in enumerate(weight_files, start=1):
        shard_size = weight_file.stat().st_size
        logger.info(
            "Loading Qwen3.5 weight shard %s/%s: %s (%s bytes)",
            shard_idx,
            total_shards,
            weight_file.name,
            shard_size,
        )
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in tensor_targets:
                    skipped += 1
                    continue

                source = handle.get_tensor(key)
                target = tensor_targets[key]
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Shape mismatch for {key}: expected {tuple(target.shape)}, got {tuple(source.shape)}"
                    )

                with torch.no_grad():
                    if source.device == target.device and source.dtype == target.dtype:
                        target.copy_(source)
                    else:
                        target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded += 1
        loaded_bytes += shard_size
        logger.info(
            "Loaded Qwen3.5 weight shard %s/%s: %s (cumulative_bytes=%s/%s, tensors_loaded=%s, tensors_skipped=%s)",
            shard_idx,
            total_shards,
            weight_file.name,
            loaded_bytes,
            total_bytes,
            loaded,
            skipped,
        )

    model.tie_weights()
    return WeightLoadReport(loaded=loaded, skipped=skipped)
