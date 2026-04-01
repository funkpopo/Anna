from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SupportedQwenModelFamily = Literal["qwen3_5_text", "qwen3_tts"]


@dataclass(slots=True)
class QwenModelFamilyInfo:
    qwen_model_family: SupportedQwenModelFamily
    model_type: str
    architectures: tuple[str, ...] = ()


def inspect_qwen_model_family(model_dir: str | Path) -> QwenModelFamilyInfo:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config_data.get("model_type", "")).strip()
    architectures = tuple(str(value) for value in config_data.get("architectures", []))
    qwen_model_family: SupportedQwenModelFamily = "qwen3_tts" if model_type == "qwen3_tts" else "qwen3_5_text"
    return QwenModelFamilyInfo(
        qwen_model_family=qwen_model_family,
        model_type=model_type,
        architectures=architectures,
    )
