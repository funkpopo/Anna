from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


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
    running_requests: int = 0
    waiting_requests: int = 0
    kv_cache_used_pages: int = 0
    kv_cache_total_pages: int = 0
    prompt_cache_entries: int = 0
    queue_wait_seconds_total: float = 0.0
    queue_wait_count: int = 0
    queue_wait_seconds_max: float = 0.0
    prefill_step_seconds_total: float = 0.0
    prefill_step_count: int = 0
    prefill_step_seconds_max: float = 0.0
    decode_step_seconds_total: float = 0.0
    decode_step_count: int = 0
    decode_step_seconds_max: float = 0.0

    @property
    def kv_cache_usage_ratio(self) -> float:
        if self.kv_cache_total_pages <= 0:
            return 0.0
        return self.kv_cache_used_pages / self.kv_cache_total_pages


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
        self._running_requests = 0
        self._waiting_requests = 0
        self._queue_wait_seconds_total = 0.0
        self._queue_wait_count = 0
        self._queue_wait_seconds_max = 0.0
        self._prefill_step_seconds_total = 0.0
        self._prefill_step_count = 0
        self._prefill_step_seconds_max = 0.0
        self._decode_step_seconds_total = 0.0
        self._decode_step_count = 0
        self._decode_step_seconds_max = 0.0

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
        self._activity_event.set()

    def record_decode_step(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._decode_step_seconds_total += normalized
            self._decode_step_count += 1
            self._decode_step_seconds_max = max(self._decode_step_seconds_max, normalized)
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
                running_requests=self._running_requests,
                waiting_requests=self._waiting_requests,
                queue_wait_seconds_total=self._queue_wait_seconds_total,
                queue_wait_count=self._queue_wait_count,
                queue_wait_seconds_max=self._queue_wait_seconds_max,
                prefill_step_seconds_total=self._prefill_step_seconds_total,
                prefill_step_count=self._prefill_step_count,
                prefill_step_seconds_max=self._prefill_step_seconds_max,
                decode_step_seconds_total=self._decode_step_seconds_total,
                decode_step_count=self._decode_step_count,
                decode_step_seconds_max=self._decode_step_seconds_max,
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
        queue_wait_total = max(0.0, current.queue_wait_seconds_total - previous.queue_wait_seconds_total)
        queue_wait_count = max(0, current.queue_wait_count - previous.queue_wait_count)
        prefill_step_total = max(0.0, current.prefill_step_seconds_total - previous.prefill_step_seconds_total)
        prefill_step_count = max(0, current.prefill_step_count - previous.prefill_step_count)
        decode_step_total = max(0.0, current.decode_step_seconds_total - previous.decode_step_seconds_total)
        decode_step_count = max(0, current.decode_step_count - previous.decode_step_count)
        prompt_tokens_per_second = prompt_tokens / elapsed
        generation_tokens_per_second = generation_tokens / elapsed
        prompt_cache_hit_rate = 0.0 if cache_queries <= 0 else (cache_hits / cache_queries) * 100.0
        kv_cache_usage = current.kv_cache_usage_ratio * 100.0
        queue_wait_avg_ms = 0.0 if queue_wait_count <= 0 else (queue_wait_total / queue_wait_count) * 1000.0
        prefill_step_avg_ms = 0.0 if prefill_step_count <= 0 else (prefill_step_total / prefill_step_count) * 1000.0
        decode_step_avg_ms = 0.0 if decode_step_count <= 0 else (decode_step_total / decode_step_count) * 1000.0
        return (
            "Engine metrics: Interval prompt: "
            f"{prompt_tokens_per_second:.1f} tok/s, Interval generation: "
            f"{generation_tokens_per_second:.1f} tok/s, Running: {current.running_requests} reqs, "
            f"Queue wait avg/max: {queue_wait_avg_ms:.1f}/{current.queue_wait_seconds_max * 1000.0:.1f} ms, "
            f"Prefill step avg/max: {prefill_step_avg_ms:.1f}/{current.prefill_step_seconds_max * 1000.0:.1f} ms, "
            f"Decode step avg/max: {decode_step_avg_ms:.1f}/{current.decode_step_seconds_max * 1000.0:.1f} ms, "
            f"Waiting: {current.waiting_requests} reqs, GPU KV cache usage: {kv_cache_usage:.1f}% "
            f"({current.kv_cache_used_pages}/{current.kv_cache_total_pages} pages), "
            f"Prompt cache hit rate: {prompt_cache_hit_rate:.1f}%"
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
            current.kv_cache_used_pages - previous.kv_cache_used_pages,
            current.kv_cache_total_pages - previous.kv_cache_total_pages,
            current.prompt_cache_entries - previous.prompt_cache_entries,
            current.queue_wait_count - previous.queue_wait_count,
            current.prefill_step_count - previous.prefill_step_count,
            current.decode_step_count - previous.decode_step_count,
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
