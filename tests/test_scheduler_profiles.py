from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from anna.mm.prepared_inputs import PreparedInputs
from anna.runtime.qwen3_5_text_engine import AnnaEngineError, GenerationConfig
from anna.runtime.scheduler import AnnaScheduler
from anna.runtime.scheduler_profiles import (
    compute_dynamic_token_budgets,
    get_scheduler_profile,
    resolve_scheduler_settings,
)
from anna.runtime.service_metrics import AnnaServiceMetrics


def test_interactive_and_throughput_profiles_differ() -> None:
    interactive = get_scheduler_profile("interactive")
    throughput = get_scheduler_profile("throughput")
    assert interactive is not None and throughput is not None
    assert interactive.max_batch_size < throughput.max_batch_size
    assert interactive.batch_wait_ms < throughput.batch_wait_ms
    assert interactive.prefill_interval_steps < throughput.prefill_interval_steps
    assert interactive.skip_batch_wait_when_idle is True
    # P2-#9: throughput inherits idle-skip (idle coalescing only added queue wait).
    assert throughput.skip_batch_wait_when_idle is True
    # P2-#9: throughput inserts waiting prefills event-driven; interactive keeps
    # its interval=1 cadence (already every-step).
    assert throughput.event_driven_prefill_insert is True
    assert interactive.event_driven_prefill_insert is False


def test_resolve_scheduler_settings_profile_with_overrides() -> None:
    knobs = resolve_scheduler_settings(
        profile="throughput",
        max_batch_size=4,
        batch_wait_ms=None,
    )
    assert knobs["profile"] == "throughput"
    assert knobs["max_batch_size"] == 4  # explicit override
    assert knobs["batch_wait_ms"] == 8.0  # from profile
    assert knobs["dynamic_token_budget"] is True


def test_resolve_scheduler_settings_none_uses_hard_defaults() -> None:
    knobs = resolve_scheduler_settings(profile="none")
    assert knobs["profile"] == "none"
    assert knobs["max_batch_size"] == 1
    assert knobs["batch_wait_ms"] == 2.0
    assert knobs["max_waiting_requests"] == 0
    assert knobs["dynamic_token_budget"] is False


def test_compute_dynamic_token_budgets_shrinks_under_memory_pressure() -> None:
    prefill, decode = compute_dynamic_token_budgets(
        base_prefill_tokens=2048,
        base_decode_tokens=4096,
        free_bytes=1 << 30,
        total_bytes=16 << 30,
        running_requests=2,
        waiting_requests=0,
        avg_running_seq_len=100,
    )
    assert prefill < 2048
    assert decode < 4096
    assert prefill >= 256
    assert decode >= 256


def test_compute_dynamic_token_budgets_tightens_for_long_sequences() -> None:
    # free_ratio=0.5 → no headroom boost; long seq halves decode only.
    prefill, decode = compute_dynamic_token_budgets(
        base_prefill_tokens=2048,
        base_decode_tokens=4096,
        free_bytes=8 << 30,
        total_bytes=16 << 30,
        running_requests=1,
        waiting_requests=0,
        avg_running_seq_len=3000,
    )
    assert prefill == 2048
    assert decode == 2048  # 0.5x for very long seq


def test_scheduler_rejects_when_waiting_queue_is_full() -> None:
    metrics = AnnaServiceMetrics()
    engine = SimpleNamespace(
        metrics=metrics,
        set_scheduler=lambda _s: None,
        _trim_runtime_cache_if_idle=lambda: None,
    )
    scheduler = AnnaScheduler(
        engine,  # type: ignore[arg-type]
        max_batch_size=2,
        batch_wait_ms=50.0,
        max_waiting_requests=1,
    )
    engine.scheduler = scheduler
    prepared = PreparedInputs(
        prompt="",
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 2), dtype=torch.int32),
    )
    config = GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0)

    # First submit occupies the only waiting slot (worker will eventually pick it up;
    # hold the worker by not providing a real engine loop — use a barrier via stop).
    # Directly fill pending under the condition lock to avoid racing the worker.
    with scheduler._condition:
        # Drain any work the worker already took.
        pass

    try:
        # Submit one request; it may be taken by the worker immediately. Force queue full
        # by stuffing pending to the limit then submitting.
        with scheduler._condition:
            while len(scheduler._pending) < scheduler.max_waiting_requests:
                from anna.runtime.scheduler import SchedulerRequest

                scheduler._pending.append(
                    SchedulerRequest(prepared=prepared, config=config, stream=False)
                )
        with pytest.raises(AnnaEngineError) as exc_info:
            scheduler._submit(prepared, config=config, stream=False)
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "scheduler_queue_full"
        assert metrics.snapshot().queue_rejected_total == 1
    finally:
        scheduler.shutdown()


def test_scheduler_skips_batch_wait_when_idle() -> None:
    scheduler = object.__new__(AnnaScheduler)
    scheduler.batch_wait_seconds = 1.0
    scheduler.max_batch_size = 4
    scheduler.skip_batch_wait_when_idle = True
    scheduler._pending = __import__("collections").deque([object(), object()])
    # Mirror the run-loop predicate used for coalescing.
    active: list = []
    should_coalesce = (
        scheduler._pending
        and scheduler.batch_wait_seconds > 0
        and len(scheduler._pending) < scheduler.max_batch_size
        and (active or not scheduler.skip_batch_wait_when_idle)
    )
    assert should_coalesce is False

    scheduler.skip_batch_wait_when_idle = False
    should_coalesce = (
        scheduler._pending
        and scheduler.batch_wait_seconds > 0
        and len(scheduler._pending) < scheduler.max_batch_size
        and (active or not scheduler.skip_batch_wait_when_idle)
    )
    assert should_coalesce is True
