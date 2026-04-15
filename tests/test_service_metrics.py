from __future__ import annotations

from anna.runtime.service_metrics import AnnaServiceMetrics, AnnaServiceMetricsLogger, ServiceMetricsSnapshot


def test_service_metrics_tracks_request_queueing_and_counters() -> None:
    metrics = AnnaServiceMetrics()
    assert metrics.activity_event.is_set() is False

    metrics.record_request_submitted(waiting=True)
    metrics.record_request_submitted(waiting=False)
    metrics.record_requests_started_from_queue(1)
    metrics.record_prompt_tokens(12)
    metrics.record_generation_tokens(5)
    metrics.record_prompt_cache_lookup(hit=True)
    metrics.record_prompt_cache_lookup(hit=False)
    metrics.record_request_finished(success=True)
    metrics.record_request_finished(success=False)

    snapshot = metrics.snapshot()

    assert snapshot.requests_started_total == 2
    assert snapshot.requests_completed_total == 1
    assert snapshot.requests_failed_total == 1
    assert snapshot.prompt_tokens_total == 12
    assert snapshot.generation_tokens_total == 5
    assert snapshot.prompt_cache_queries_total == 2
    assert snapshot.prompt_cache_hits_total == 1
    assert snapshot.running_requests == 0
    assert snapshot.waiting_requests == 0
    assert metrics.activity_event.is_set() is True


def test_service_metrics_logger_formats_interval_rates() -> None:
    previous = ServiceMetricsSnapshot(
        timestamp=10.0,
        prompt_tokens_total=8,
        generation_tokens_total=4,
        prompt_cache_queries_total=1,
        prompt_cache_hits_total=0,
    )
    current = ServiceMetricsSnapshot(
        timestamp=12.0,
        prompt_tokens_total=24,
        generation_tokens_total=14,
        prompt_cache_queries_total=5,
        prompt_cache_hits_total=3,
        prompt_cache_enabled=True,
        running_requests=2,
        waiting_requests=1,
        kv_cache_used_pages=6,
        kv_cache_total_pages=12,
    )

    line = AnnaServiceMetricsLogger.format_interval(previous, current)

    assert "Avg prompt throughput: 8.0 tokens/s" in line
    assert "Avg generation throughput: 5.0 tokens/s" in line
    assert "Running: 2 reqs" in line
    assert "Waiting: 1 reqs" in line
    assert "GPU KV cache usage: 50.0% (6/12 pages)" in line
    assert "Prompt cache hit rate: 75.0% (3/4 lookups)" in line


def test_service_metrics_logger_formats_turboquant_usage_and_disabled_prompt_cache() -> None:
    previous = ServiceMetricsSnapshot(
        timestamp=10.0,
        kv_cache_mode="turboquant",
    )
    current = ServiceMetricsSnapshot(
        timestamp=12.0,
        prompt_tokens_total=32,
        generation_tokens_total=8,
        running_requests=1,
        kv_cache_mode="turboquant",
        kv_cache_used_bytes=16 << 20,
        kv_cache_dense_equivalent_bytes=64 << 20,
        kv_cache_quantized_bytes=12 << 20,
        kv_cache_residual_bytes=4 << 20,
        kv_cache_quantized_tokens=4096,
        kv_cache_residual_tokens=128,
    )

    line = AnnaServiceMetricsLogger.format_interval(previous, current)

    assert "Avg prompt throughput: 16.0 tokens/s" in line
    assert "Avg generation throughput: 4.0 tokens/s" in line
    assert "TurboQuant KV cache: used=16.00 MiB" in line
    assert "dense=64.00 MiB" in line
    assert "compression=4.00x" in line
    assert "quantized=4096 tokens" in line
    assert "residual=128 tokens" in line
    assert "Prompt cache hit rate: disabled" in line


def test_service_metrics_logger_skips_idle_intervals_without_changes() -> None:
    previous = ServiceMetricsSnapshot(timestamp=10.0, kv_cache_total_pages=128)
    current = ServiceMetricsSnapshot(timestamp=20.0, kv_cache_total_pages=128)

    assert AnnaServiceMetricsLogger.should_log_interval(previous, current) is False


def test_service_metrics_logger_logs_idle_interval_after_completed_work() -> None:
    previous = ServiceMetricsSnapshot(timestamp=10.0)
    current = ServiceMetricsSnapshot(
        timestamp=20.0,
        requests_started_total=1,
        requests_completed_total=1,
        prompt_tokens_total=32,
        generation_tokens_total=8,
        kv_cache_total_pages=128,
    )

    assert AnnaServiceMetricsLogger.should_log_interval(previous, current) is True
