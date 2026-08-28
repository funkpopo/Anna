from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from anna.api.app import create_app
from anna.cli.serve import (
    _build_metrics_logger,
    _build_safety_policy,
    _build_scheduler,
    _log_available_routes,
    _resolve_serve_scheduler_knobs,
    build_parser,
    configure_flashqla_environment,
    configure_int4_kernel_environment,
)
from anna.core.config import ServeSettings
from anna.runtime.service_metrics import AnnaServiceMetrics, AnnaServiceMetricsLogger


@pytest.fixture(autouse=True)
def _isolate_flashqla_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)
    yield
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)


def test_build_safety_policy_uses_custom_serve_overrides() -> None:
    settings = ServeSettings(
        model_dir=Path("dummy"),
        min_free_memory_mib=256,
        reserve_memory_mib=128,
        max_estimated_usage_ratio=0.95,
        generation_memory_safety_factor=1.25,
    )

    policy = _build_safety_policy(settings)

    assert policy is not None
    assert policy.min_free_bytes == 256 << 20
    assert policy.reserve_margin_bytes == 128 << 20
    assert policy.max_estimated_usage_ratio == 0.95
    assert policy.generation_memory_safety_factor == 1.25


def test_serve_parser_accepts_memory_guard_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--disable-thinking",
            "--max-completion-tokens",
            "1024",
            "--temperature",
            "1.0",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--min-p",
            "0.0",
            "--presence-penalty",
            "1.5",
            "--repetition-penalty",
            "1.0",
            "--reasoning-format",
            "deepseek",
            "--offload-vision",
            "--weight-quant",
            "int4",
            "--enable-flashqla-gdn-prefill",
            "--min-free-memory-mib",
            "256",
            "--reserve-memory-mib",
            "128",
            "--max-estimated-usage-ratio",
            "0.95",
            "--generation-memory-safety-factor",
            "1.25",
            "--kv-cache-quantization",
            "turboquant",
            "--kv-cache-quant-bits",
            "4",
            "--kv-cache-residual-len",
            "96",
            "--warmup-prefill-tokens",
            "512",
            "--warmup-decode-steps",
            "8",
            "--warmup-batch-size",
            "4",
            "--scheduler-max-batch-size",
            "4",
            "--scheduler-batch-wait-ms",
            "2.0",
            "--scheduler-prefill-interval-steps",
            "3",
            "--scheduler-max-prefill-tokens",
            "2048",
            "--scheduler-max-decode-tokens",
            "8192",
            "--metrics-log-interval-seconds",
            "3.5",
        ]
    )

    assert args.default_enable_thinking is False
    assert args.max_completion_tokens == 1024
    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.top_k == 20
    assert args.min_p == 0.0
    assert args.presence_penalty == 1.5
    assert args.repetition_penalty == 1.0
    assert args.reasoning_format == "deepseek"
    assert args.offload_vision is True
    assert args.weight_quant == "int4"
    assert args.xpu_int4_matmul is None
    assert args.enable_flashqla_gdn_prefill is True
    assert args.min_free_memory_mib == 256
    assert args.reserve_memory_mib == 128
    assert args.max_estimated_usage_ratio == 0.95
    assert args.generation_memory_safety_factor == 1.25
    assert args.kv_cache_quantization == "turboquant"
    assert args.kv_cache_quant_bits == 4
    assert args.kv_cache_residual_len == 96
    assert args.warmup_prefill_tokens == 512
    assert args.warmup_decode_steps == 8
    assert args.warmup_batch_size == 4
    assert args.scheduler_prefill_interval_steps == 3
    assert args.scheduler_max_prefill_tokens == 2048
    assert args.scheduler_max_decode_tokens == 8192
    assert args.metrics_log_interval_seconds == 3.5


def test_configure_int4_kernel_environment_applies_matmul_override(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--xpu-int4-matmul",
            "dequant",
        ]
    )

    monkeypatch.delenv("ANNA_XPU_INT4_MATMUL", raising=False)

    configure_int4_kernel_environment(args)

    assert os.environ["ANNA_XPU_INT4_MATMUL"] == "dequant"


def test_configure_int4_kernel_environment_preserves_runtime_default(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-dir", "model"])

    monkeypatch.setenv("ANNA_XPU_INT4_MATMUL", "dequant")

    configure_int4_kernel_environment(args)

    assert os.environ["ANNA_XPU_INT4_MATMUL"] == "dequant"


def test_configure_int4_kernel_environment_applies_new_load_overrides(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--xpu-int4-cache-load-workers",
            "4",
            "--weight-load-pipeline-workers",
            "1",
        ]
    )

    monkeypatch.delenv("ANNA_XPU_INT4_CACHE_LOAD_WORKERS", raising=False)
    monkeypatch.delenv("ANNA_WEIGHT_LOAD_PIPELINE_WORKERS", raising=False)

    configure_int4_kernel_environment(args)

    assert os.environ["ANNA_XPU_INT4_CACHE_LOAD_WORKERS"] == "4"
    assert os.environ["ANNA_WEIGHT_LOAD_PIPELINE_WORKERS"] == "1"

    # CLI omitted -> env untouched (built-in defaults stay in charge).
    args = parser.parse_args(["--model-dir", "model"])
    configure_int4_kernel_environment(args)
    assert os.environ["ANNA_XPU_INT4_CACHE_LOAD_WORKERS"] == "4"
    assert os.environ["ANNA_WEIGHT_LOAD_PIPELINE_WORKERS"] == "1"


def test_configure_compile_cache_environment_overrides_torch_ephemeral_default(monkeypatch, tmp_path) -> None:
    """--torchinductor-cache-dir must win over torch's import-time temp-dir injection."""
    from anna.cli.serve import configure_compile_cache_environment

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    configure_compile_cache_environment(cache_dir=tmp_path / "inductor")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")

    # Simulate torch injecting its ephemeral temp-dir default after import:
    # configure must replace it with the persistent location.
    from anna.cli import serve as serve_module

    ephemeral_dir = tmp_path / "torch-ephemeral"
    monkeypatch.setattr(serve_module, "_torch_ephemeral_inductor_cache_dir", lambda: str(ephemeral_dir))

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(ephemeral_dir))
    configure_compile_cache_environment(cache_dir=tmp_path / "inductor")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")

    # Same, but falling back to the built-in Anna default dir.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(ephemeral_dir))
    configure_compile_cache_environment(cache_dir=None)
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == Path.home() / ".anna" / "cache" / "torchinductor"

    # A genuinely different pre-set value is respected.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "custom"))
    configure_compile_cache_environment(cache_dir=None)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "custom")


def test_serve_settings_tuning_fields_default_to_none() -> None:
    settings = ServeSettings(model_dir=Path("model"))

    assert settings.xpu_int4_gemv_m_threshold is None
    assert settings.xpu_int4_cache_load_workers is None
    assert settings.weight_load_pipeline_workers is None
    assert settings.torchinductor_cache_dir is None

    resolved = ServeSettings(
        model_dir=Path("model"),
        xpu_int4_gemv_m_threshold=4,
        xpu_int4_cache_load_workers=2,
        weight_load_pipeline_workers=1,
        torchinductor_cache_dir=Path("cache"),
    )
    assert resolved.xpu_int4_gemv_m_threshold == 4
    assert resolved.xpu_int4_cache_load_workers == 2
    assert resolved.weight_load_pipeline_workers == 1
    assert resolved.torchinductor_cache_dir == Path("cache")


def test_weight_load_pipeline_workers_env_override(monkeypatch) -> None:
    from anna.weights.qwen3_5_text_weight_loader import (
        _WEIGHT_LOAD_PIPELINE_WORKERS_DEFAULT,
        _weight_load_pipeline_workers,
    )

    monkeypatch.delenv("ANNA_WEIGHT_LOAD_PIPELINE_WORKERS", raising=False)
    assert _weight_load_pipeline_workers() == _WEIGHT_LOAD_PIPELINE_WORKERS_DEFAULT == 2

    monkeypatch.setenv("ANNA_WEIGHT_LOAD_PIPELINE_WORKERS", "4")
    assert _weight_load_pipeline_workers() == 4

    monkeypatch.setenv("ANNA_WEIGHT_LOAD_PIPELINE_WORKERS", "bogus")
    assert _weight_load_pipeline_workers() == _WEIGHT_LOAD_PIPELINE_WORKERS_DEFAULT


def test_configure_flashqla_environment_applies_cli_override(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-dir", "model", "--enable-flashqla-gdn-prefill"])
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)

    configure_flashqla_environment(args)

    assert os.environ["ANNA_XPU_FLASHQLA_GDN_PREFILL"] == "strict"
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)


def test_configure_flashqla_environment_applies_prefer_mode(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-dir", "model", "--flashqla-gdn-prefill-mode", "prefer"])
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)

    configure_flashqla_environment(args)

    assert os.environ["ANNA_XPU_FLASHQLA_GDN_PREFILL"] == "prefer"
    monkeypatch.delenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", raising=False)


def test_configure_flashqla_environment_preserves_existing_env_when_cli_omitted(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-dir", "model"])
    monkeypatch.setenv("ANNA_XPU_FLASHQLA_GDN_PREFILL", "1")

    configure_flashqla_environment(args)

    assert os.environ["ANNA_XPU_FLASHQLA_GDN_PREFILL"] == "1"


def test_serve_parser_defaults_to_direct_generation() -> None:
    parser = build_parser()

    args = parser.parse_args(["--model-dir", "model"])

    # Scheduler knobs default to None and are resolved via profile/hard defaults.
    assert args.scheduler_profile == "none"
    assert args.scheduler_max_batch_size is None
    assert args.scheduler_prefill_interval_steps is None
    assert args.scheduler_max_prefill_tokens is None
    assert args.scheduler_max_decode_tokens is None
    assert args.kv_cache_quant_bits is None
    assert args.kv_cache_residual_len is None
    assert args.metrics_log_interval_seconds == 10.0

    knobs = _resolve_serve_scheduler_knobs(args)
    assert knobs["max_batch_size"] == 1
    assert knobs["prefill_interval_steps"] == 1
    assert knobs["max_prefill_tokens"] == 0
    assert knobs["max_decode_tokens"] == 0


def test_serve_parser_scheduler_profile_interactive() -> None:
    parser = build_parser()
    args = parser.parse_args(["--model-dir", "model", "--scheduler-profile", "interactive"])
    knobs = _resolve_serve_scheduler_knobs(args)
    assert knobs["profile"] == "interactive"
    assert knobs["max_batch_size"] == 2
    assert knobs["batch_wait_ms"] == 0.5
    assert knobs["dynamic_token_budget"] is True
    assert knobs["skip_batch_wait_when_idle"] is True
    assert knobs["max_waiting_requests"] == 32


def test_serve_parser_scheduler_profile_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--scheduler-profile",
            "throughput",
            "--scheduler-max-batch-size",
            "3",
            "--no-scheduler-dynamic-token-budget",
        ]
    )
    knobs = _resolve_serve_scheduler_knobs(args)
    assert knobs["profile"] == "throughput"
    assert knobs["max_batch_size"] == 3
    assert knobs["batch_wait_ms"] == 8.0
    assert knobs["dynamic_token_budget"] is False


def test_serve_parser_accepts_kv_cache_quant_bits_two() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--kv-cache-quantization",
            "turboquant",
            "--kv-cache-quant-bits",
            "2",
        ]
    )
    assert args.kv_cache_quant_bits == 2


class _FakeEngine:
    def __init__(self) -> None:
        self.scheduler = "sentinel"

    def set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler


def test_build_scheduler_skips_continuous_batching_when_disabled() -> None:
    engine = _FakeEngine()
    settings = ServeSettings(model_dir=Path("dummy"), scheduler_max_batch_size=1)

    scheduler = _build_scheduler(engine, settings)

    assert scheduler is None
    assert engine.scheduler is None


def test_build_scheduler_passes_prefill_interval_to_scheduler() -> None:
    engine = _FakeEngine()
    settings = ServeSettings(
        model_dir=Path("dummy"),
        scheduler_profile="interactive",
        scheduler_max_batch_size=4,
        scheduler_prefill_interval_steps=3,
        scheduler_max_prefill_tokens=1024,
        scheduler_max_decode_tokens=4096,
        scheduler_max_waiting_requests=16,
        scheduler_dynamic_token_budget=True,
        scheduler_skip_batch_wait_when_idle=True,
        scheduler_max_queue_wait_ms=50.0,
    )

    scheduler = _build_scheduler(engine, settings)

    try:
        assert scheduler is not None
        assert scheduler.prefill_interval_steps == 3
        assert scheduler.max_prefill_tokens == 1024
        assert scheduler.max_decode_tokens == 4096
        assert scheduler.max_waiting_requests == 16
        assert scheduler.dynamic_token_budget is True
        assert scheduler.skip_batch_wait_when_idle is True
        assert scheduler.profile == "interactive"
        assert engine.scheduler is scheduler
    finally:
        if scheduler is not None:
            scheduler.shutdown()


def test_build_metrics_logger_can_be_disabled() -> None:
    engine = _FakeEngine()
    engine.service_metrics_snapshot = lambda: None  # type: ignore[assignment]
    settings = ServeSettings(model_dir=Path("dummy"), metrics_log_interval_seconds=0.0)

    metrics_logger = _build_metrics_logger(engine, settings)

    assert metrics_logger is None


def test_build_metrics_logger_uses_engine_snapshot_provider() -> None:
    engine = _FakeEngine()
    engine.service_metrics_snapshot = lambda: None  # type: ignore[assignment]
    engine.metrics = AnnaServiceMetrics()
    settings = ServeSettings(model_dir=Path("dummy"), metrics_log_interval_seconds=5.0)

    metrics_logger = _build_metrics_logger(engine, settings)

    assert isinstance(metrics_logger, AnnaServiceMetricsLogger)
    assert metrics_logger.interval_seconds == 5.0
    assert metrics_logger.activity_event is engine.metrics.activity_event


def test_log_available_routes_reports_server_address_and_paths(caplog) -> None:
    app = create_app(_FakeEngine())

    with caplog.at_level(logging.INFO):
        _log_available_routes(app, host="127.0.0.1", port=8000)

    assert "Starting Anna server on http://127.0.0.1:8000" in caplog.text
    assert "Available routes are:" in caplog.text
    assert "Route: /healthz, Methods: GET" in caplog.text
    assert "Route: /v1/chat/completions, Methods: POST" in caplog.text
    assert "Route: /v1/audio/speech, Methods: POST" in caplog.text
