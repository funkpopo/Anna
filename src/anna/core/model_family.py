from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SupportedModelFamily = Literal["qwen3_5_text", "qwen3_tts", "gemma4"]


@dataclass(slots=True)
class ModelFamilyInfo:
    model_family: SupportedModelFamily
    model_type: str
    architectures: tuple[str, ...] = ()


def inspect_model_family(model_dir: str | Path) -> ModelFamilyInfo:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config_data.get("model_type", "")).strip()
    architectures = tuple(str(value) for value in config_data.get("architectures", []))
    if model_type == "qwen3_tts":
        model_family: SupportedModelFamily = "qwen3_tts"
    elif model_type == "gemma4":
        model_family = "gemma4"
    else:
        model_family = "qwen3_5_text"
    return ModelFamilyInfo(
        model_family=model_family,
        model_type=model_type,
        architectures=architectures,
    )
