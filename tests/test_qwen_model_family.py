from __future__ import annotations

import json
from pathlib import Path

from anna.core.qwen_model_family import inspect_qwen_model_family


def test_inspect_qwen_model_family_detects_qwen3_5_text_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "text-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    info = inspect_qwen_model_family(model_dir)

    assert info.qwen_model_family == "qwen3_5_text"
    assert info.model_type == "qwen3_5"


def test_inspect_qwen_model_family_detects_qwen3_tts_model(tmp_path: Path) -> None:
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

    info = inspect_qwen_model_family(model_dir)

    assert info.qwen_model_family == "qwen3_tts"
    assert info.model_type == "qwen3_tts"
    assert info.architectures == ("Qwen3TTSForConditionalGeneration",)
