from __future__ import annotations

import logging
from pathlib import Path

from anna.api.app import create_app
from anna.cli.serve import _build_metrics_logger, _build_safety_policy, _build_scheduler, _log_available_routes, build_parser
from anna.core.config import ServeSettings
from anna.runtime.service_metrics import AnnaServiceMetrics, AnnaServiceMetricsLogger


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
            "--reasoning-format",
            "deepseek",
            "--offload-vision",
            "--weight-quant",
            "int4",
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
            "--metrics-log-interval-seconds",
            "3.5",
        ]
    )

    assert args.default_enable_thinking is False
    assert args.max_completion_tokens == 1024
    assert args.reasoning_format == "deepseek"
    assert args.offload_vision is True
    assert args.weight_quant == "int4"
    assert args.min_free_memory_mib == 256
    assert args.reserve_memory_mib == 128
    assert args.max_estimated_usage_ratio == 0.95
    assert args.generation_memory_safety_factor == 1.25
    assert args.kv_cache_quantization == "turboquant"
    assert args.kv_cache_quant_bits == 4
    assert args.kv_cache_residual_len == 96
    assert args.metrics_log_interval_seconds == 3.5


def test_serve_parser_defaults_to_direct_generation() -> None:
    parser = build_parser()

    args = parser.parse_args(["--model-dir", "model"])

    assert args.scheduler_max_batch_size == 1
    assert args.metrics_log_interval_seconds == 10.0


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
