from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

from anna.model.kernel_metrics import kernel_strategy_snapshot

logger = logging.getLogger(__name__)

_LATENCY_SAMPLE_LIMIT = 4096


def _quantile(samples: tuple[float, ...], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _latency_histogram(samples: tuple[float, ...]) -> dict[str, float | int]:
    """p50/p95/p99 histogram summary in seconds (plus count/max)."""
    return {
        "count": len(samples),
        "p50_seconds": _quantile(samples, 0.50),
        "p95_seconds": _quantile(samples, 0.95),
        "p99_seconds": _quantile(samples, 0.99),
        "max_seconds": 0.0 if not samples else max(samples),
    }


@dataclass(slots=True)
class ServiceMetricsSnapshot:
    timestamp: float
    requests_started_total: int = 0
    requests_completed_total: int = 0
    requests_failed_total: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    prompt_cache_queries_total: int = 0
    prompt_cache_hits_total: int = 0
    prefix_block_lookups_total: int = 0
    prefix_block_hits_total: int = 0
    prefix_block_misses_total: int = 0
    prefix_block_registers_total: int = 0
    prefix_block_entries: int = 0
    running_requests: int = 0
    waiting_requests: int = 0
    kv_cache_used_pages: int = 0
    kv_cache_total_pages: int = 0
    kv_cache_turboquant_rows: int = 0
    kv_cache_turboquant_tokens: int = 0
    prompt_cache_entries: int = 0
    queue_rejected_total: int = 0
    queue_wait_seconds_total: float = 0.0
    queue_wait_count: int = 0
    queue_wait_seconds_max: float = 0.0
    prefill_step_seconds_total: float = 0.0
    prefill_step_count: int = 0
    prefill_step_seconds_max: float = 0.0
    prefill_step_recent_seconds: tuple[float, ...] = ()
    decode_step_seconds_total: float = 0.0
    decode_step_count: int = 0
    decode_step_seconds_max: float = 0.0
    decode_step_recent_seconds: tuple[float, ...] = ()
    # P0-#3: per-step classification (pure_decode / prefill_insert / budget_recompute mixes).
    decode_step_class_counts: dict[str, int] = field(default_factory=dict)
    # P0-#1: host-side decode phases not covered by GPU component timers (sampling, etc).
    decode_phase_seconds_total: dict[str, float] = field(default_factory=dict)
    ttft_seconds_total: float = 0.0
    ttft_count: int = 0
    ttft_seconds_max: float = 0.0
    ttft_recent_seconds: tuple[float, ...] = ()
    itl_seconds_total: float = 0.0
    itl_count: int = 0
    itl_seconds_max: float = 0.0
    itl_recent_seconds: tuple[float, ...] = ()
    cache_stack_seconds_total: float = 0.0
    cache_stack_count: int = 0
    cache_stack_seconds_max: float = 0.0
    cache_split_seconds_total: float = 0.0
    cache_split_count: int = 0
    cache_split_seconds_max: float = 0.0
    cache_compact_seconds_total: float = 0.0
    cache_compact_count: int = 0
    cache_compact_seconds_max: float = 0.0
    scheduler_prefill_admitted_requests_total: int = 0
    scheduler_prefill_deferred_requests_total: int = 0
    scheduler_prefill_admitted_tokens_total: int = 0
    scheduler_prefill_admission_count: int = 0
    scheduler_prefill_admitted_tokens_max: int = 0
    scheduler_decode_batch_count: int = 0
    scheduler_decode_batch_requests_total: int = 0
    scheduler_decode_batch_requests_max: int = 0
    scheduler_decode_batch_tokens_total: int = 0
    scheduler_decode_batch_tokens_max: int = 0
    kernel_strategy_hits: dict[str, int] = field(default_factory=dict)

    @property
    def kv_cache_usage_ratio(self) -> float:
        if self.kv_cache_total_pages <= 0:
            return 0.0
        return self.kv_cache_used_pages / self.kv_cache_total_pages

    @property
    def scheduler_queue_depth(self) -> int:
        return max(0, int(self.waiting_requests))

    def ttft_histogram(self) -> dict[str, float | int]:
        return _latency_histogram(self.ttft_recent_seconds)

    def itl_histogram(self) -> dict[str, float | int]:
        return _latency_histogram(self.itl_recent_seconds)


class AnnaServiceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._activity_event = threading.Event()
        self._requests_started_total = 0
        self._requests_completed_total = 0
        self._requests_failed_total = 0
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._prompt_cache_queries_total = 0
        self._prompt_cache_hits_total = 0
        self._prefix_block_lookups_total = 0
        self._prefix_block_hits_total = 0
        self._prefix_block_misses_total = 0
        self._prefix_block_registers_total = 0
        self._prefix_block_entries = 0
        self._running_requests = 0
        self._waiting_requests = 0
        self._queue_rejected_total = 0
        self._queue_wait_seconds_total = 0.0
        self._queue_wait_count = 0
        self._queue_wait_seconds_max = 0.0
        self._prefill_step_seconds_total = 0.0
        self._prefill_step_count = 0
        self._prefill_step_seconds_max = 0.0
        self._prefill_step_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._decode_step_seconds_total = 0.0
        self._decode_step_count = 0
        self._decode_step_seconds_max = 0.0
        self._decode_step_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._decode_step_class_counts: dict[str, int] = defaultdict(int)
        self._decode_step_ewma_seconds = 0.0
        self._decode_phase_seconds_total: dict[str, float] = defaultdict(float)
        self._ttft_seconds_total = 0.0
        self._ttft_count = 0
        self._ttft_seconds_max = 0.0
        self._ttft_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._itl_seconds_total = 0.0
        self._itl_count = 0
        self._itl_seconds_max = 0.0
        self._itl_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._cache_stack_seconds_total = 0.0
        self._cache_stack_count = 0
        self._cache_stack_seconds_max = 0.0
        self._cache_split_seconds_total = 0.0
        self._cache_split_count = 0
        self._cache_split_seconds_max = 0.0
        self._cache_compact_seconds_total = 0.0
        self._cache_compact_count = 0
        self._cache_compact_seconds_max = 0.0
        self._scheduler_prefill_admitted_requests_total = 0
        self._scheduler_prefill_deferred_requests_total = 0
        self._scheduler_prefill_admitted_tokens_total = 0
        self._scheduler_prefill_admission_count = 0
        self._scheduler_prefill_admitted_tokens_max = 0
        self._scheduler_decode_batch_count = 0
        self._scheduler_decode_batch_requests_total = 0
        self._scheduler_decode_batch_requests_max = 0
        self._scheduler_decode_batch_tokens_total = 0
        self._scheduler_decode_batch_tokens_max = 0

    def record_request_submitted(self, *, waiting: bool) -> None:
        with self._lock:
            self._requests_started_total += 1
            if waiting:
                self._waiting_requests += 1
            else:
                self._running_requests += 1
        self._activity_event.set()

    def record_requests_started_from_queue(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._waiting_requests = max(0, self._waiting_requests - normalized)
            self._running_requests += normalized
        self._activity_event.set()

    def record_queue_wait(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._queue_wait_seconds_total += normalized
            self._queue_wait_count += 1
            self._queue_wait_seconds_max = max(self._queue_wait_seconds_max, normalized)
        self._activity_event.set()

    def record_prefill_step(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._prefill_step_seconds_total += normalized
            self._prefill_step_count += 1
            self._prefill_step_seconds_max = max(self._prefill_step_seconds_max, normalized)
            self._prefill_step_recent_seconds.append(normalized)
        self._activity_event.set()

    _DECODE_SPIKE_EWMA_ALPHA = 0.05
    _DECODE_SPIKE_WARMUP_STEPS = 64
    _DECODE_SPIKE_MULTIPLE = 3.0

    def record_decode_step(self, seconds: float, *, classification: str = "pure_decode") -> None:
        normalized = max(0.0, float(seconds))
        spike = False
        with self._lock:
            self._decode_step_seconds_total += normalized
            self._decode_step_count += 1
            self._decode_step_seconds_max = max(self._decode_step_seconds_max, normalized)
            self._decode_step_recent_seconds.append(normalized)
            self._decode_step_class_counts[str(classification) or "unknown"] += 1
            # P0-#3 spike attribution: EWMA baseline; report steps far above the
            # rolling average together with their step classification.
            if self._decode_step_ewma_seconds <= 0.0:
                self._decode_step_ewma_seconds = normalized
            else:
                self._decode_step_ewma_seconds += (
                    self._DECODE_SPIKE_EWMA_ALPHA * (normalized - self._decode_step_ewma_seconds)
                )
            spike = (
                self._decode_step_count > self._DECODE_SPIKE_WARMUP_STEPS
                and normalized > self._DECODE_SPIKE_MULTIPLE * self._decode_step_ewma_seconds
            )
        if spike:
            logger.warning(
                "decode step spike: %.1f ms (>%.1fx rolling avg %.1f ms) classification=%s",
                normalized * 1000.0,
                self._DECODE_SPIKE_MULTIPLE,
                self._decode_step_ewma_seconds * 1000.0,
                classification,
            )
        self._activity_event.set()

    def record_decode_phase(self, category: str, seconds: float) -> None:
        """Accumulate a host-side decode phase (sampling, cache ops, ...) in seconds."""
        normalized = max(0.0, float(seconds))
        if not category or normalized <= 0.0:
            return
        with self._lock:
            self._decode_phase_seconds_total[str(category)] += normalized
        self._activity_event.set()

    def record_ttft(self, seconds: float) -> None:
        """Record time-to-first-token for a completed request."""
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._ttft_seconds_total += normalized
            self._ttft_count += 1
            self._ttft_seconds_max = max(self._ttft_seconds_max, normalized)
            self._ttft_recent_seconds.append(normalized)
        self._activity_event.set()

    def record_itl(self, seconds: float) -> None:
        """Record inter-token latency sample (per decode token or mean ITL)."""
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._itl_seconds_total += normalized
            self._itl_count += 1
            self._itl_seconds_max = max(self._itl_seconds_max, normalized)
            self._itl_recent_seconds.append(normalized)
        self._activity_event.set()

    def record_generation_latency(self, *, ttft_seconds: float, decode_seconds: float, decode_tokens: int) -> None:
        """Record request-level TTFT and mean ITL from generation perf stats."""
        self.record_ttft(ttft_seconds)
        tokens = max(0, int(decode_tokens))
        if tokens > 0:
            self.record_itl(max(0.0, float(decode_seconds)) / tokens)

    def record_cache_stack(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._cache_stack_seconds_total += normalized
            self._cache_stack_count += 1
            self._cache_stack_seconds_max = max(self._cache_stack_seconds_max, normalized)
        self._activity_event.set()

    def record_cache_split(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._cache_split_seconds_total += normalized
            self._cache_split_count += 1
            self._cache_split_seconds_max = max(self._cache_split_seconds_max, normalized)
        self._activity_event.set()

    def record_cache_compact(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._cache_compact_seconds_total += normalized
            self._cache_compact_count += 1
            self._cache_compact_seconds_max = max(self._cache_compact_seconds_max, normalized)
        self._activity_event.set()

    def record_prefill_admission(self, *, admitted_requests: int, deferred_requests: int, admitted_tokens: int) -> None:
        admitted = max(0, int(admitted_requests))
        deferred = max(0, int(deferred_requests))
        tokens = max(0, int(admitted_tokens))
        with self._lock:
            self._scheduler_prefill_admitted_requests_total += admitted
            self._scheduler_prefill_deferred_requests_total += deferred
            self._scheduler_prefill_admitted_tokens_total += tokens
            self._scheduler_prefill_admission_count += 1
            self._scheduler_prefill_admitted_tokens_max = max(self._scheduler_prefill_admitted_tokens_max, tokens)
        self._activity_event.set()

    def record_decode_batch(self, *, requests: int, token_cost: int) -> None:
        normalized_requests = max(0, int(requests))
        normalized_tokens = max(0, int(token_cost))
        if normalized_requests <= 0:
            return
        with self._lock:
            self._scheduler_decode_batch_count += 1
            self._scheduler_decode_batch_requests_total += normalized_requests
            self._scheduler_decode_batch_requests_max = max(
                self._scheduler_decode_batch_requests_max,
                normalized_requests,
            )
            self._scheduler_decode_batch_tokens_total += normalized_tokens
            self._scheduler_decode_batch_tokens_max = max(self._scheduler_decode_batch_tokens_max, normalized_tokens)
        self._activity_event.set()

    def record_request_finished(self, *, success: bool) -> None:
        with self._lock:
            self._running_requests = max(0, self._running_requests - 1)
            if success:
                self._requests_completed_total += 1
            else:
                self._requests_failed_total += 1
        self._activity_event.set()

    def record_prompt_tokens(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._prompt_tokens_total += normalized
        self._activity_event.set()

    def record_generation_tokens(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._generation_tokens_total += normalized
        self._activity_event.set()

    def record_prompt_cache_lookup(self, *, hit: bool) -> None:
        with self._lock:
            self._prompt_cache_queries_total += 1
            if hit:
                self._prompt_cache_hits_total += 1
        self._activity_event.set()

    def record_queue_rejected(self, count: int = 1) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._queue_rejected_total += normalized
        self._activity_event.set()

    def set_prefix_block_stats(
        self,
        *,
        lookups_total: int,
        hits_total: int,
        misses_total: int,
        registers_total: int,
        entries: int,
    ) -> None:
        """Refresh absolute prefix-block pool counters (sourced from PrefixBlockPool.stats)."""
        with self._lock:
            self._prefix_block_lookups_total = max(0, int(lookups_total))
            self._prefix_block_hits_total = max(0, int(hits_total))
            self._prefix_block_misses_total = max(0, int(misses_total))
            self._prefix_block_registers_total = max(0, int(registers_total))
            self._prefix_block_entries = max(0, int(entries))
        self._activity_event.set()

    @property
    def activity_event(self) -> threading.Event:
        return self._activity_event

    def snapshot(self) -> ServiceMetricsSnapshot:
        with self._lock:
            return ServiceMetricsSnapshot(
                timestamp=time.perf_counter(),
                requests_started_total=self._requests_started_total,
                requests_completed_total=self._requests_completed_total,
                requests_failed_total=self._requests_failed_total,
                prompt_tokens_total=self._prompt_tokens_total,
                generation_tokens_total=self._generation_tokens_total,
                prompt_cache_queries_total=self._prompt_cache_queries_total,
                prompt_cache_hits_total=self._prompt_cache_hits_total,
                prefix_block_lookups_total=self._prefix_block_lookups_total,
                prefix_block_hits_total=self._prefix_block_hits_total,
                prefix_block_misses_total=self._prefix_block_misses_total,
                prefix_block_registers_total=self._prefix_block_registers_total,
                prefix_block_entries=self._prefix_block_entries,
                running_requests=self._running_requests,
                waiting_requests=self._waiting_requests,
                queue_rejected_total=self._queue_rejected_total,
                queue_wait_seconds_total=self._queue_wait_seconds_total,
                queue_wait_count=self._queue_wait_count,
                queue_wait_seconds_max=self._queue_wait_seconds_max,
                prefill_step_seconds_total=self._prefill_step_seconds_total,
                prefill_step_count=self._prefill_step_count,
                prefill_step_seconds_max=self._prefill_step_seconds_max,
                prefill_step_recent_seconds=tuple(self._prefill_step_recent_seconds),
                decode_step_seconds_total=self._decode_step_seconds_total,
                decode_step_count=self._decode_step_count,
                decode_step_seconds_max=self._decode_step_seconds_max,
                decode_step_recent_seconds=tuple(self._decode_step_recent_seconds),
                decode_step_class_counts=dict(sorted(self._decode_step_class_counts.items())),
                decode_phase_seconds_total=dict(sorted(self._decode_phase_seconds_total.items())),
                ttft_seconds_total=self._ttft_seconds_total,
                ttft_count=self._ttft_count,
                ttft_seconds_max=self._ttft_seconds_max,
                ttft_recent_seconds=tuple(self._ttft_recent_seconds),
                itl_seconds_total=self._itl_seconds_total,
                itl_count=self._itl_count,
                itl_seconds_max=self._itl_seconds_max,
                itl_recent_seconds=tuple(self._itl_recent_seconds),
                cache_stack_seconds_total=self._cache_stack_seconds_total,
                cache_stack_count=self._cache_stack_count,
                cache_stack_seconds_max=self._cache_stack_seconds_max,
                cache_split_seconds_total=self._cache_split_seconds_total,
                cache_split_count=self._cache_split_count,
                cache_split_seconds_max=self._cache_split_seconds_max,
                cache_compact_seconds_total=self._cache_compact_seconds_total,
                cache_compact_count=self._cache_compact_count,
                cache_compact_seconds_max=self._cache_compact_seconds_max,
                scheduler_prefill_admitted_requests_total=self._scheduler_prefill_admitted_requests_total,
                scheduler_prefill_deferred_requests_total=self._scheduler_prefill_deferred_requests_total,
                scheduler_prefill_admitted_tokens_total=self._scheduler_prefill_admitted_tokens_total,
                scheduler_prefill_admission_count=self._scheduler_prefill_admission_count,
                scheduler_prefill_admitted_tokens_max=self._scheduler_prefill_admitted_tokens_max,
                scheduler_decode_batch_count=self._scheduler_decode_batch_count,
                scheduler_decode_batch_requests_total=self._scheduler_decode_batch_requests_total,
                scheduler_decode_batch_requests_max=self._scheduler_decode_batch_requests_max,
                scheduler_decode_batch_tokens_total=self._scheduler_decode_batch_tokens_total,
                scheduler_decode_batch_tokens_max=self._scheduler_decode_batch_tokens_max,
                kernel_strategy_hits=kernel_strategy_snapshot(),
            )


class AnnaServiceMetricsLogger:
    def __init__(
        self,
        snapshot_provider: Callable[[], ServiceMetricsSnapshot],
        *,
        interval_seconds: float = 10.0,
        activity_event: threading.Event | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.activity_event = activity_event
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="anna-service-metrics", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self.activity_event is not None:
            self.activity_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
            self._thread = None

    @staticmethod
    def format_interval(previous: ServiceMetricsSnapshot, current: ServiceMetricsSnapshot) -> str:
        elapsed = max(1e-9, current.timestamp - previous.timestamp)
        prompt_tokens = max(0, current.prompt_tokens_total - previous.prompt_tokens_total)
        generation_tokens = max(0, current.generation_tokens_total - previous.generation_tokens_total)
        cache_queries = max(0, current.prompt_cache_queries_total - previous.prompt_cache_queries_total)
        cache_hits = max(0, current.prompt_cache_hits_total - previous.prompt_cache_hits_total)
        prefix_lookups = max(0, current.prefix_block_lookups_total - previous.prefix_block_lookups_total)
        prefix_hits = max(0, current.prefix_block_hits_total - previous.prefix_block_hits_total)
        queue_rejected = max(0, current.queue_rejected_total - previous.queue_rejected_total)
        queue_wait_total = max(0.0, current.queue_wait_seconds_total - previous.queue_wait_seconds_total)
        queue_wait_count = max(0, current.queue_wait_count - previous.queue_wait_count)
        prefill_step_total = max(0.0, current.prefill_step_seconds_total - previous.prefill_step_seconds_total)
        prefill_step_count = max(0, current.prefill_step_count - previous.prefill_step_count)
        decode_step_total = max(0.0, current.decode_step_seconds_total - previous.decode_step_seconds_total)
        decode_step_count = max(0, current.decode_step_count - previous.decode_step_count)
        cache_stack_total = max(0.0, current.cache_stack_seconds_total - previous.cache_stack_seconds_total)
        cache_stack_count = max(0, current.cache_stack_count - previous.cache_stack_count)
        cache_split_total = max(0.0, current.cache_split_seconds_total - previous.cache_split_seconds_total)
        cache_split_count = max(0, current.cache_split_count - previous.cache_split_count)
        cache_compact_total = max(0.0, current.cache_compact_seconds_total - previous.cache_compact_seconds_total)
        cache_compact_count = max(0, current.cache_compact_count - previous.cache_compact_count)
        prefill_admitted_requests = max(
            0,
            current.scheduler_prefill_admitted_requests_total
            - previous.scheduler_prefill_admitted_requests_total,
        )
        prefill_deferred_requests = max(
            0,
            current.scheduler_prefill_deferred_requests_total
            - previous.scheduler_prefill_deferred_requests_total,
        )
        prefill_admitted_tokens = max(
            0,
            current.scheduler_prefill_admitted_tokens_total - previous.scheduler_prefill_admitted_tokens_total,
        )
        prefill_admission_count = max(
            0,
            current.scheduler_prefill_admission_count - previous.scheduler_prefill_admission_count,
        )
        decode_batch_count = max(0, current.scheduler_decode_batch_count - previous.scheduler_decode_batch_count)
        decode_batch_requests = max(
            0,
            current.scheduler_decode_batch_requests_total - previous.scheduler_decode_batch_requests_total,
        )
        decode_batch_tokens = max(
            0,
            current.scheduler_decode_batch_tokens_total - previous.scheduler_decode_batch_tokens_total,
        )
        prompt_tokens_per_second = prompt_tokens / elapsed
        generation_tokens_per_second = generation_tokens / elapsed
        prompt_cache_hit_rate = 0.0 if cache_queries <= 0 else (cache_hits / cache_queries) * 100.0
        prefix_block_hit_rate = 0.0 if prefix_lookups <= 0 else (prefix_hits / prefix_lookups) * 100.0
        kv_cache_usage = current.kv_cache_usage_ratio * 100.0
        # P2-#7: under TurboQuant KV the paged storage is unused by design; report the
        # live turboquant row usage instead of a misleading 0/0 pages.
        if current.kv_cache_total_pages > 0:
            kv_cache_summary = (
                f"GPU KV cache usage: {kv_cache_usage:.1f}% "
                f"({current.kv_cache_used_pages}/{current.kv_cache_total_pages} pages)"
            )
        else:
            kv_cache_summary = (
                "KV cache: paged storage idle "
                f"(turboquant rows={current.kv_cache_turboquant_rows}, "
                f"cached tokens={current.kv_cache_turboquant_tokens})"
            )
        queue_wait_avg_ms = 0.0 if queue_wait_count <= 0 else (queue_wait_total / queue_wait_count) * 1000.0
        prefill_step_avg_ms = 0.0 if prefill_step_count <= 0 else (prefill_step_total / prefill_step_count) * 1000.0
        decode_step_avg_ms = 0.0 if decode_step_count <= 0 else (decode_step_total / decode_step_count) * 1000.0
        cache_stack_avg_ms = 0.0 if cache_stack_count <= 0 else (cache_stack_total / cache_stack_count) * 1000.0
        cache_split_avg_ms = 0.0 if cache_split_count <= 0 else (cache_split_total / cache_split_count) * 1000.0
        cache_compact_avg_ms = 0.0 if cache_compact_count <= 0 else (cache_compact_total / cache_compact_count) * 1000.0
        prefill_admitted_tokens_avg = (
            0.0 if prefill_admission_count <= 0 else prefill_admitted_tokens / prefill_admission_count
        )
        decode_batch_requests_avg = 0.0 if decode_batch_count <= 0 else decode_batch_requests / decode_batch_count
        decode_batch_tokens_avg = 0.0 if decode_batch_count <= 0 else decode_batch_tokens / decode_batch_count
        prefill_recent = current.prefill_step_recent_seconds
        decode_recent = current.decode_step_recent_seconds
        prefill_p50_ms = _quantile(prefill_recent, 0.50) * 1000.0
        prefill_p95_ms = _quantile(prefill_recent, 0.95) * 1000.0
        prefill_p99_ms = _quantile(prefill_recent, 0.99) * 1000.0
        decode_p50_ms = _quantile(decode_recent, 0.50) * 1000.0
        decode_p95_ms = _quantile(decode_recent, 0.95) * 1000.0
        decode_p99_ms = _quantile(decode_recent, 0.99) * 1000.0
        # P0-#3: step classification mix over the process lifetime so spike
        # attribution (prefill-insert steps vs budget recompute) is visible in
        # the periodic metrics line.
        class_mix = current.decode_step_class_counts
        class_summary = ",".join(f"{name}={count}" for name, count in class_mix.items()) or "none"
        # P0-#1: host-side decode phases (sampling etc) accumulated this interval.
        phase_deltas = {
            name: max(0.0, value - previous.decode_phase_seconds_total.get(name, 0.0))
            for name, value in current.decode_phase_seconds_total.items()
        }
        phase_summary = ",".join(
            f"{name}={ms:.1f}ms" for name, ms in sorted(phase_deltas.items()) if ms > 0.0
        ) or "none"
        ttft_hist = current.ttft_histogram()
        itl_hist = current.itl_histogram()
        kernel_hits = current.kernel_strategy_hits
        kernel_summary = ",".join(f"{name}={count}" for name, count in list(kernel_hits.items())[:8]) or "none"
        return (
            "Engine metrics: Interval prompt: "
            f"{prompt_tokens_per_second:.1f} tok/s, Interval generation: "
            f"{generation_tokens_per_second:.1f} tok/s, Running: {current.running_requests} reqs, "
            f"Queue depth: {current.scheduler_queue_depth}, "
            f"Queue wait avg/max: {queue_wait_avg_ms:.1f}/{current.queue_wait_seconds_max * 1000.0:.1f} ms, "
            f"Prefill step avg/max: {prefill_step_avg_ms:.1f}/{current.prefill_step_seconds_max * 1000.0:.1f} ms, "
            f"Prefill step p50/p95/p99: {prefill_p50_ms:.1f}/{prefill_p95_ms:.1f}/{prefill_p99_ms:.1f} ms, "
            f"Decode step avg/max: {decode_step_avg_ms:.1f}/{current.decode_step_seconds_max * 1000.0:.1f} ms, "
            f"Decode step p50/p95/p99: {decode_p50_ms:.1f}/{decode_p95_ms:.1f}/{decode_p99_ms:.1f} ms, "
            f"Decode step classes: {class_summary}, "
            f"Decode phases (interval): {phase_summary}, "
            f"TTFT p50/p95/p99: {float(ttft_hist['p50_seconds']) * 1000.0:.1f}/"
            f"{float(ttft_hist['p95_seconds']) * 1000.0:.1f}/{float(ttft_hist['p99_seconds']) * 1000.0:.1f} ms, "
            f"ITL p50/p95/p99: {float(itl_hist['p50_seconds']) * 1000.0:.1f}/"
            f"{float(itl_hist['p95_seconds']) * 1000.0:.1f}/{float(itl_hist['p99_seconds']) * 1000.0:.1f} ms, "
            f"Cache stack avg/max: {cache_stack_avg_ms:.1f}/{current.cache_stack_seconds_max * 1000.0:.1f} ms, "
            f"Cache split avg/max: {cache_split_avg_ms:.1f}/{current.cache_split_seconds_max * 1000.0:.1f} ms, "
            f"Cache compact avg/max: {cache_compact_avg_ms:.1f}/{current.cache_compact_seconds_max * 1000.0:.1f} ms, "
            f"Prefill admission reqs admitted/deferred: {prefill_admitted_requests}/{prefill_deferred_requests}, "
            f"Prefill admission tokens avg/max: {prefill_admitted_tokens_avg:.1f}/{current.scheduler_prefill_admitted_tokens_max}, "
            f"Decode batch reqs avg/max: {decode_batch_requests_avg:.1f}/{current.scheduler_decode_batch_requests_max}, "
            f"Decode batch tokens avg/max: {decode_batch_tokens_avg:.1f}/{current.scheduler_decode_batch_tokens_max}, "
            f"Waiting: {current.waiting_requests} reqs, Queue rejected: {queue_rejected}, "
            f"{kv_cache_summary}, "
            f"Prompt cache hit rate: {prompt_cache_hit_rate:.1f}%, "
            f"Prefix block hit rate: {prefix_block_hit_rate:.1f}% "
            f"({current.prefix_block_hits_total}/{current.prefix_block_lookups_total}, "
            f"entries={current.prefix_block_entries}), "
            f"Kernel strategy hits: {kernel_summary}"
        )

    @staticmethod
    def should_log_interval(previous: ServiceMetricsSnapshot, current: ServiceMetricsSnapshot) -> bool:
        if current.running_requests > 0 or current.waiting_requests > 0:
            return True
        deltas = (
            current.requests_started_total - previous.requests_started_total,
            current.requests_completed_total - previous.requests_completed_total,
            current.requests_failed_total - previous.requests_failed_total,
            current.prompt_tokens_total - previous.prompt_tokens_total,
            current.generation_tokens_total - previous.generation_tokens_total,
            current.prompt_cache_queries_total - previous.prompt_cache_queries_total,
            current.prompt_cache_hits_total - previous.prompt_cache_hits_total,
            current.prefix_block_lookups_total - previous.prefix_block_lookups_total,
            current.prefix_block_hits_total - previous.prefix_block_hits_total,
            current.prefix_block_registers_total - previous.prefix_block_registers_total,
            current.prefix_block_entries - previous.prefix_block_entries,
            current.kv_cache_used_pages - previous.kv_cache_used_pages,
            current.kv_cache_total_pages - previous.kv_cache_total_pages,
            current.kv_cache_turboquant_rows - previous.kv_cache_turboquant_rows,
            current.prompt_cache_entries - previous.prompt_cache_entries,
            current.queue_rejected_total - previous.queue_rejected_total,
            current.queue_wait_count - previous.queue_wait_count,
            current.prefill_step_count - previous.prefill_step_count,
            current.decode_step_count - previous.decode_step_count,
            sum(current.decode_step_class_counts.values()) - sum(previous.decode_step_class_counts.values()),
            sum(current.decode_phase_seconds_total.values()) - sum(previous.decode_phase_seconds_total.values()),
            current.ttft_count - previous.ttft_count,
            current.itl_count - previous.itl_count,
            current.cache_stack_count - previous.cache_stack_count,
            current.cache_split_count - previous.cache_split_count,
            current.cache_compact_count - previous.cache_compact_count,
            current.scheduler_prefill_admitted_requests_total - previous.scheduler_prefill_admitted_requests_total,
            current.scheduler_prefill_deferred_requests_total - previous.scheduler_prefill_deferred_requests_total,
            current.scheduler_prefill_admitted_tokens_total - previous.scheduler_prefill_admitted_tokens_total,
            current.scheduler_prefill_admission_count - previous.scheduler_prefill_admission_count,
            current.scheduler_decode_batch_count - previous.scheduler_decode_batch_count,
            current.scheduler_decode_batch_requests_total - previous.scheduler_decode_batch_requests_total,
            current.scheduler_decode_batch_tokens_total - previous.scheduler_decode_batch_tokens_total,
        )
        return any(delta != 0 for delta in deltas)

    @staticmethod
    def _is_idle(snapshot: ServiceMetricsSnapshot) -> bool:
        return snapshot.running_requests <= 0 and snapshot.waiting_requests <= 0

    def _run_loop(self) -> None:
        previous = self.snapshot_provider()
        while not self._stop_event.is_set():
            if self.activity_event is not None and self._is_idle(previous):
                self.activity_event.wait()
                if self._stop_event.is_set():
                    return
                self.activity_event.clear()
            if self._stop_event.wait(self.interval_seconds):
                return
            current = self.snapshot_provider()
            if self.should_log_interval(previous, current):
                logger.info(self.format_interval(previous, current))
            previous = current
