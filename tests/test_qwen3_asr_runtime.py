from __future__ import annotations

from pathlib import Path

from anna.runtime.qwen3_asr_engine import AnnaQwen3ASREngine, Qwen3ASRTranscriptionConfig


def test_qwen3_asr_normalizes_dict_transcription_result() -> None:
    engine = object.__new__(AnnaQwen3ASREngine)

    result = engine._normalize_transcription_result(
        [{"text": "hello", "language": "English", "timestamps": [{"text": "hello", "start": 0.0, "end": 0.5}]}],
        config=Qwen3ASRTranscriptionConfig(language=None, return_timestamps=True),
        total_seconds=0.25,
    )

    assert result.text == "hello"
    assert result.language == "English"
    assert result.timestamps == [{"text": "hello", "start": 0.0, "end": 0.5}]
    assert result.total_seconds == 0.25


def test_qwen3_asr_reads_supported_languages(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"support_languages":["Chinese","English"]}', encoding="utf-8")

    assert AnnaQwen3ASREngine._read_supported_languages(tmp_path) == ["Chinese", "English"]


def test_runtime_loader_dispatches_qwen3_asr_model(tmp_path: Path, monkeypatch) -> None:
    from anna.runtime.model_runtime_loader import load_model_runtime_from_model_dir

    model_dir = tmp_path / "Qwen3-ASR-0.6B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3_asr"}', encoding="utf-8")
    sentinel = object()
    captured = {}

    def _from_model_dir(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(AnnaQwen3ASREngine, "from_model_dir", _from_model_dir)

    engine = load_model_runtime_from_model_dir(model_dir, model_id="asr", device="xpu")

    assert engine is sentinel
    assert captured["path"] == model_dir
    assert captured["kwargs"]["model_id"] == "asr"
    assert captured["kwargs"]["device"] == "xpu"


def test_qwen3_asr_rejects_non_xpu_device(monkeypatch, tmp_path: Path) -> None:
    import pytest

    class _DeviceContext:
        device = type("_Device", (), {"type": "cpu"})()

    monkeypatch.setattr("anna.runtime.qwen3_asr_engine.DeviceContext.resolve", lambda **_kwargs: _DeviceContext())

    with pytest.raises(RuntimeError, match="requires Intel XPU execution"):
        AnnaQwen3ASREngine.from_model_dir(tmp_path, device="cpu")
