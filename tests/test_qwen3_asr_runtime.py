from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from anna.runtime.device_execution import DeviceExecutionGate
from anna.runtime.qwen3_asr_engine import AnnaEngineError, AnnaQwen3ASREngine, Qwen3ASRTranscriptionConfig


def _prime_asr_engine(engine: AnnaQwen3ASREngine, *, model_id: str = "asr-test") -> AnnaQwen3ASREngine:
    engine.device_gate = DeviceExecutionGate(owner=f"asr:{model_id}")
    engine.max_inference_batch_size = 1
    engine.max_new_tokens = 512
    engine.batch_wait_seconds = 0.0
    engine.max_waiting_requests = 0
    engine._batch_waves_total = 0
    engine._batch_requests_total = 0
    engine._batch_requests_max = 1
    engine._queue_rejections_total = 0
    engine._pending = __import__("collections").deque()
    engine._queue_condition = __import__("threading").Condition()
    engine._batch_leader = None
    engine.execution_lock = __import__("threading").Lock()
    return engine


class _FakeDeviceContext:
    def __init__(self) -> None:
        self.device = torch.device("xpu")
        self.dtype = torch.bfloat16
        self.requested_dtype = "auto"
        self.reported_dtype = "bfloat16"
        self.xpu_info = None

    def synchronize(self) -> None:
        pass

    def release_unused_memory(self) -> None:
        pass

    def get_memory_info(self):
        return None

    @staticmethod
    def classify_runtime_error(_exc: BaseException) -> str:
        return "runtime_error"


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


def test_qwen3_asr_normalizes_object_transcription_result() -> None:
    class _Result:
        text = "hello from object"
        language = "English"
        time_stamps = [{"text": "hello", "start": 0.0, "end": 0.5}]

    engine = object.__new__(AnnaQwen3ASREngine)

    result = engine._normalize_transcription_result(
        [_Result()],
        config=Qwen3ASRTranscriptionConfig(language=None, return_timestamps=True),
        total_seconds=0.25,
    )

    assert result.text == "hello from object"
    assert result.language == "English"
    assert result.timestamps == [{"text": "hello", "start": 0.0, "end": 0.5}]


def test_qwen3_asr_normalizes_string_transcription_result() -> None:
    engine = object.__new__(AnnaQwen3ASREngine)

    result = engine._normalize_transcription_result(
        "hello as string",
        config=Qwen3ASRTranscriptionConfig(language="English"),
        total_seconds=0.25,
    )

    assert result.text == "hello as string"
    assert result.language == "English"
    assert result.timestamps is None


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


def test_qwen3_asr_forwards_xpu_load_options(monkeypatch, tmp_path: Path) -> None:
    class _DeviceContext(_FakeDeviceContext):
        pass

    class _Qwen3ASRModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "model"

    captured = {}
    monkeypatch.setattr("anna.runtime.qwen3_asr_engine.DeviceContext.resolve", lambda **_kwargs: _DeviceContext())
    monkeypatch.setattr("anna.runtime.qwen3_asr_engine.AnnaQwen3ASREngine._validate_xpu_only_model", lambda *_args: (123, {"xpu:0": 123}))
    monkeypatch.setattr("anna.runtime.qwen3_asr_engine.AnnaQwen3ASREngine._install_xpu_execution_guards", lambda _self: [])
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr", type("_Module", (), {"Qwen3ASRModel": _Qwen3ASRModel})())

    engine = AnnaQwen3ASREngine.from_model_dir(
        tmp_path,
        device="xpu",
        asr_max_inference_batch_size=3,
        asr_max_new_tokens=42,
    )

    assert engine.model == "model"
    assert captured["args"] == (str(tmp_path),)
    assert captured["kwargs"]["device_map"] == "xpu"
    assert captured["kwargs"]["dtype"] is torch.bfloat16
    assert captured["kwargs"]["attn_implementation"] == "eager"
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["max_inference_batch_size"] == 3
    assert captured["kwargs"]["max_new_tokens"] == 42


def test_qwen3_asr_rejects_invalid_asr_load_limits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("anna.runtime.qwen3_asr_engine.DeviceContext.resolve", lambda **_kwargs: _FakeDeviceContext())

    with pytest.raises(ValueError, match="asr_max_inference_batch_size"):
        AnnaQwen3ASREngine.from_model_dir(tmp_path, device="xpu", asr_max_inference_batch_size=0)

    with pytest.raises(ValueError, match="asr_max_new_tokens"):
        AnnaQwen3ASREngine.from_model_dir(tmp_path, device="xpu", asr_max_new_tokens=0)


def test_qwen3_asr_rejects_non_transformers_backend() -> None:
    model = type("_Model", (), {"backend": "vllm", "device": torch.device("xpu"), "model": nn.Linear(1, 1)})()

    with pytest.raises(RuntimeError, match="transformers backend"):
        AnnaQwen3ASREngine._validate_xpu_only_model(model, torch.device("xpu"))


def test_qwen3_asr_rejects_cpu_tensors_after_load() -> None:
    model = type("_Model", (), {"backend": "transformers", "device": torch.device("xpu"), "model": nn.Linear(1, 1)})()

    with pytest.raises(RuntimeError, match="tensors outside XPU"):
        AnnaQwen3ASREngine._validate_xpu_only_model(model, torch.device("xpu"))


def test_qwen3_asr_runtime_guard_rejects_cpu_inputs() -> None:
    engine = object.__new__(AnnaQwen3ASREngine)

    with pytest.raises(RuntimeError, match="non-XPU tensors"):
        engine._assert_tensor_tree_on_xpu({"input_ids": torch.ones((1, 2), dtype=torch.long)}, context="test")


def test_qwen3_asr_transcribe_passes_context_and_requires_xpu_forward(monkeypatch, tmp_path: Path) -> None:
    class _Result:
        text = "hello"
        language = "English"
        time_stamps = None

    class _Model:
        def __init__(self) -> None:
            self.calls = []

        def transcribe(self, **kwargs):
            self.calls.append(kwargs)
            engine._xpu_forward_observed_count += 1
            return [_Result()]

    engine = _prime_asr_engine(object.__new__(AnnaQwen3ASREngine))
    engine.model = _Model()
    engine.device_context = _FakeDeviceContext()
    engine._xpu_forward_observed_count = 0
    engine._last_xpu_elapsed_ms = None
    monkeypatch.setattr(AnnaQwen3ASREngine, "_start_xpu_execution_probe", lambda _self: (None, None))

    result = engine._transcribe_locked(
        str(tmp_path / "audio.wav"),
        config=Qwen3ASRTranscriptionConfig(language="English", context="names: Anna", return_timestamps=True),
        filename=None,
    )

    assert result.text == "hello"
    assert engine.model.calls == [
        {
            "audio": str(tmp_path / "audio.wav"),
            "context": "names: Anna",
            "language": "English",
            "return_time_stamps": True,
        }
    ]


def test_qwen3_asr_transcribe_rejects_missing_xpu_forward(monkeypatch, tmp_path: Path) -> None:
    class _Model:
        def transcribe(self, **_kwargs):
            return [{"text": "hello"}]

    engine = _prime_asr_engine(object.__new__(AnnaQwen3ASREngine))
    engine.model = _Model()
    engine.device_context = _FakeDeviceContext()
    engine._xpu_forward_observed_count = 0
    engine._last_xpu_elapsed_ms = None
    monkeypatch.setattr(AnnaQwen3ASREngine, "_start_xpu_execution_probe", lambda _self: (None, None))

    with pytest.raises(AnnaEngineError, match="without observing an XPU model forward pass"):
        engine._transcribe_locked(
            str(tmp_path / "audio.wav"),
            config=Qwen3ASRTranscriptionConfig(),
            filename=None,
        )


def test_qwen3_asr_continuous_batch_coalesces_compatible_requests(monkeypatch, tmp_path: Path) -> None:
    class _Result:
        def __init__(self, text: str) -> None:
            self.text = text
            self.language = "English"
            self.time_stamps = None

    class _Model:
        def __init__(self) -> None:
            self.calls = []

        def transcribe(self, **kwargs):
            self.calls.append(kwargs)
            engine._xpu_forward_observed_count += 1
            audio = kwargs["audio"]
            if isinstance(audio, list):
                return [_Result(f"t{idx}") for idx, _ in enumerate(audio)]
            return [_Result("solo")]

    from anna.runtime.service_metrics import AnnaServiceMetrics

    engine = _prime_asr_engine(object.__new__(AnnaQwen3ASREngine))
    engine.model = _Model()
    engine.device_context = _FakeDeviceContext()
    engine.metrics = AnnaServiceMetrics()
    engine.default_model_id = "asr-test"
    engine.max_inference_batch_size = 2
    engine.batch_wait_seconds = 0.02
    engine._xpu_forward_observed_count = 0
    engine._last_xpu_elapsed_ms = None
    monkeypatch.setattr(AnnaQwen3ASREngine, "_start_xpu_execution_probe", lambda _self: (None, None))
    monkeypatch.setattr(AnnaQwen3ASREngine, "_finish_xpu_execution_probe", lambda *_args, **_kwargs: None)

    audio_a = str(tmp_path / "a.wav")
    audio_b = str(tmp_path / "b.wav")
    Path(audio_a).write_bytes(b"RIFF")
    Path(audio_b).write_bytes(b"RIFF")
    config = Qwen3ASRTranscriptionConfig(language="English")

    results: list = []
    errors: list = []

    def _worker(path: str) -> None:
        try:
            results.append(engine.transcribe_qwen3_asr_audio(path, config=config))
        except Exception as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    import threading

    threads = [
        threading.Thread(target=_worker, args=(audio_a,)),
        threading.Thread(target=_worker, args=(audio_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 2
    assert {result.text for result in results} == {"t0", "t1"}
    assert engine._batch_waves_total >= 1
    assert engine._batch_requests_max >= 2
    assert any(isinstance(call.get("audio"), list) for call in engine.model.calls)
