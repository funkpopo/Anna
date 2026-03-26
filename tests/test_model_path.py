from __future__ import annotations

from pathlib import Path

from anna.core.model_path import resolve_model_dir, resolve_model_name


def test_resolve_model_dir_keeps_explicit_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "Qwen" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    resolved = resolve_model_dir(model_dir)

    assert resolved == model_dir.resolve()


def test_resolve_model_name_prefers_explicit_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "Qwen" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    resolved = resolve_model_name(model_name="qwen3.5", model_dir=model_dir)

    assert resolved == "qwen3.5"


def test_resolve_model_name_defaults_to_full_model_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "Qwen" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    resolved = resolve_model_name(model_name=None, model_dir=model_dir)

    assert resolved == str(model_dir.resolve())
