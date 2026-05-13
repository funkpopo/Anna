from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from anna.core.native import SafetensorsShardPlan, inspect_safetensors_load_plan, inspect_safetensors_manifest
from anna.model.gemma4_config import Gemma4Config
from anna.model.gemma4_text_model import Gemma4ForConditionalGeneration
from anna.model.quantization import _release_cpu_memory_caches
from anna.weights.safetensors_device import safetensors_pt_device_str


@dataclass(slots=True)
class WeightLoadReport:
    loaded: int
    skipped: int


def load_gemma4_text_model_config(model_dir: str | Path) -> Gemma4Config:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return Gemma4Config.from_model_dir(model_path)


def _iter_weight_files(model_dir: Path) -> list[Path]:
    return inspect_safetensors_manifest(model_dir)[0]


def estimate_gemma4_text_model_weight_bytes(model_dir: str | Path) -> int:
    model_path = Path(model_dir)
    return inspect_safetensors_manifest(model_path)[1]


def _load_plan(model_dir: Path) -> tuple[list[SafetensorsShardPlan], int]:
    return inspect_safetensors_load_plan(model_dir)


def build_gemma4_text_model(
    config: Gemma4Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Gemma4ForConditionalGeneration, int]:
    default_dtype = torch.get_default_dtype()
    build_dtype = dtype if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float32
    try:
        torch.set_default_dtype(build_dtype)
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)

    if not hasattr(model, "to_empty"):
        raise RuntimeError(
            "The current torch build does not support nn.Module.to_empty, which is required for low-memory model loading."
        )
    model.to_empty(device=device)
    model.reset_runtime_buffers()
    return model, 0


def load_gemma4_text_model_weights(
    model: Gemma4ForConditionalGeneration,
    model_dir: str | Path,
) -> WeightLoadReport:
    model_path = Path(model_dir)
    tensor_targets = {name: tensor for name, tensor in model.named_parameters()}
    tensor_targets.update({name: tensor for name, tensor in model.named_buffers()})
    loaded = 0
    skipped = 0
    st_device = safetensors_pt_device_str(tensor_targets)
    load_plan, _total_bytes = _load_plan(model_path)

    for shard_plan in load_plan:
        weight_file = shard_plan.path
        with safe_open(str(weight_file), framework="pt", device=st_device) as handle:
            for key in shard_plan.keys:
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
                del source
                loaded += 1
        _release_cpu_memory_caches()

    model.tie_weights()
    del tensor_targets
    _release_cpu_memory_caches()
    return WeightLoadReport(loaded=loaded, skipped=skipped)
