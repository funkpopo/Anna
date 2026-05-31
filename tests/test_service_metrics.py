from __future__ import annotations

from anna.runtime.service_metrics import AnnaServiceMetrics, AnnaServiceMetricsLogger, ServiceMetricsSnapshot


def test_service_metrics_tracks_request_queueing_and_counters() -> None:
    metrics = AnnaServiceMetrics()
    assert metrics.activity_event.is_set() is False

    metrics.record_request_submitted(waiting=True)
    metrics.record_request_submitted(waiting=False)
    metrics.record_requests_started_from_queue(1)
    metrics.record_queue_wait(0.25)
    metrics.record_prefill_step(0.5)
    metrics.record_decode_step(0.125)
    metrics.record_cache_stack(0.03125)
    metrics.record_cache_split(0.0625)
    metrics.record_slot_decode_plan(0.015625)
    metrics.record_cpu_sync(reason="token_id_cpu_staging", count=2)
    metrics.record_attention_fallback(reason="grouped_attention")
    metrics.record_paged_cache_materialization(reason="gather_layer_cache")
    metrics.record_sampler_full_vocab_sort(reason="top_p_full_logits_sort")
    metrics.record_moe_host_offset(reason="expert_offsets_cpu")
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
    assert snapshot.queue_wait_count == 1
    assert snapshot.queue_wait_seconds_total == 0.25
    assert snapshot.queue_wait_seconds_max == 0.25
    assert snapshot.prefill_step_count == 1
    assert snapshot.prefill_step_seconds_total == 0.5
    assert snapshot.prefill_step_seconds_max == 0.5
    assert snapshot.prefill_step_recent_seconds == (0.5,)
    assert snapshot.decode_step_count == 1
    assert snapshot.decode_step_seconds_total == 0.125
    assert snapshot.decode_step_seconds_max == 0.125
    assert snapshot.decode_step_recent_seconds == (0.125,)
    assert snapshot.cache_stack_count == 1
    assert snapshot.cache_stack_seconds_total == 0.03125
    assert snapshot.cache_stack_seconds_max == 0.03125
    assert snapshot.cache_split_count == 1
    assert snapshot.cache_split_seconds_total == 0.0625
    assert snapshot.cache_split_seconds_max == 0.0625
    assert snapshot.slot_decode_plan_count == 1
    assert snapshot.slot_decode_plan_seconds_total == 0.015625
    assert snapshot.slot_decode_plan_seconds_max == 0.015625
    assert snapshot.cpu_sync_count == 2
    assert snapshot.attention_fallback_count == 1
    assert snapshot.paged_cache_materialize_count == 1
    assert snapshot.sampler_full_vocab_sort_count == 1
    assert snapshot.moe_host_offset_count == 1
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
        running_requests=2,
        waiting_requests=1,
        kv_cache_used_pages=6,
        kv_cache_total_pages=12,
        queue_wait_seconds_total=0.25,
        queue_wait_count=1,
        queue_wait_seconds_max=0.25,
        prefill_step_seconds_total=0.5,
        prefill_step_count=2,
        prefill_step_seconds_max=0.4,
        prefill_step_recent_seconds=(0.1, 0.4),
        decode_step_seconds_total=0.125,
        decode_step_count=5,
        decode_step_seconds_max=0.05,
        decode_step_recent_seconds=(0.005, 0.01, 0.02, 0.04, 0.05),
        cache_stack_seconds_total=0.06,
        cache_stack_count=3,
        cache_stack_seconds_max=0.03,
        cache_split_seconds_total=0.08,
        cache_split_count=4,
        cache_split_seconds_max=0.04,
        slot_decode_plan_seconds_total=0.02,
        slot_decode_plan_count=2,
        slot_decode_plan_seconds_max=0.015,
        cpu_sync_count=3,
        attention_fallback_count=4,
        paged_cache_materialize_count=5,
        sampler_full_vocab_sort_count=6,
        moe_host_offset_count=7,
    )

    line = AnnaServiceMetricsLogger.format_interval(previous, current)

    assert "Interval prompt: 8.0 tok/s" in line
    assert "Interval generation: 5.0 tok/s" in line
    assert "Running: 2 reqs" in line
    assert "Queue wait avg/max: 250.0/250.0 ms" in line
    assert "Prefill step avg/max: 250.0/400.0 ms" in line
    assert "Prefill step p50/p95/p99: 100.0/400.0/400.0 ms" in line
    assert "Decode step avg/max: 25.0/50.0 ms" in line
    assert "Decode step p50/p95/p99: 20.0/50.0/50.0 ms" in line
    assert "Cache stack avg/max: 20.0/30.0 ms" in line
    assert "Cache split avg/max: 20.0/40.0 ms" in line
    assert "Slot decode plan avg/max: 10.0/15.0 ms" in line
    assert "Hot path events: cpu_sync=3, attention_fallback=4, paged_cache_materialize=5, sampler_full_vocab_sort=6, moe_host_offset=7" in line
    assert "Waiting: 1 reqs" in line
    assert "GPU KV cache usage: 50.0% (6/12 pages)" in line
    assert "Prompt cache hit rate: 75.0%" in line


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
        cpu_sync_count=1,
        kv_cache_total_pages=128,
    )

    assert AnnaServiceMetricsLogger.should_log_interval(previous, current) is True
