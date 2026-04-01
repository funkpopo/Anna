from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
from anna.model.quantization import replace_linear_modules


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
        with torch.device("meta"):
            model = Qwen3_5TextForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)

    quantized_specs = replace_linear_modules(model, config.quantization_config, compute_dtype=dtype)
    if not hasattr(model, "to_empty"):
        raise RuntimeError("The current torch build does not support nn.Module.to_empty, which is required for low-memory model loading.")
    model.to_empty(device=device)
    return model, len(quantized_specs)


def load_qwen3_5_text_model_weights(model: Qwen3_5TextForConditionalGeneration, model_dir: str | Path) -> WeightLoadReport:
    model_path = Path(model_dir)
    tensor_targets = {name: tensor for name, tensor in model.named_parameters()}
    tensor_targets.update({name: tensor for name, tensor in model.named_buffers()})
    loaded = 0
    skipped = 0

    for weight_file in _iter_weight_files(model_path):
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
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded += 1

    model.tie_weights()
    return WeightLoadReport(loaded=loaded, skipped=skipped)
