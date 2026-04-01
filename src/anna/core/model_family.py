from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModelFamily = Literal["text", "tts"]


@dataclass(slots=True)
class ModelFamilyInfo:
    family: ModelFamily
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
    family: ModelFamily = "tts" if model_type == "qwen3_tts" else "text"
    return ModelFamilyInfo(
        family=family,
        model_type=model_type,
        architectures=architectures,
    )
