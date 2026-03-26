from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from anna.model.config import Qwen3Config
from anna.model.qwen import Qwen3ForConditionalGeneration
from anna.model.quantization import replace_linear_modules


@dataclass(slots=True)
class WeightLoadReport:
    loaded: int
    skipped: int
    quantized_replacements: int = 0


def load_model_config(model_dir: str | Path) -> Qwen3Config:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return Qwen3Config.from_model_dir(model_path)


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


def build_model(
    config: Qwen3Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3ForConditionalGeneration, int]:
    model = Qwen3ForConditionalGeneration(config)
    model.to(device=device, dtype=dtype)
    quantized_specs = replace_linear_modules(model, config.quantization_config, compute_dtype=dtype)
    model.to(device=device)
    return model, len(quantized_specs)


def load_model_weights(model: Qwen3ForConditionalGeneration, model_dir: str | Path) -> WeightLoadReport:
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
