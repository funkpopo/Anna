from __future__ import annotations

import json
from pathlib import Path

from anna.core.model_family import inspect_model_family


def test_inspect_model_family_detects_text_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    info = inspect_model_family(model_dir)

    assert info.family == "text"
    assert info.model_type == "qwen3_5"


def test_inspect_model_family_detects_qwen3_tts(tmp_path: Path) -> None:
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

    assert info.family == "tts"
    assert info.model_type == "qwen3_tts"
    assert info.architectures == ("Qwen3TTSForConditionalGeneration",)
