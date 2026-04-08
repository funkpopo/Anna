from __future__ import annotations

import json
from pathlib import Path

from anna.core.model_family import inspect_model_family


def test_inspect_model_family_detects_qwen3_5_text_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    info = inspect_model_family(model_dir)

    assert info.model_family == "qwen3_5_text"
    assert info.model_type == "qwen3_5"


def test_inspect_model_family_detects_qwen3_tts_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "tts-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts",
                "architectures": ["Qwen3TTSForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )

    info = inspect_model_family(model_dir)

    assert info.model_family == "qwen3_tts"
    assert info.model_type == "qwen3_tts"
    assert info.architectures == ("Qwen3TTSForConditionalGeneration",)


def test_inspect_model_family_detects_gemma4_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "gemma4-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )

    info = inspect_model_family(model_dir)

    assert info.model_family == "gemma4"
    assert info.model_type == "gemma4"
    assert info.architectures == ("Gemma4ForConditionalGeneration",)
