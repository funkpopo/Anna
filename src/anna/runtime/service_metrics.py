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

    @property
    def kv_cache_usage_ratio(self) -> float:
        if self.kv_cache_total_pages <= 0:
            return 0.0
        return self.kv_cache_used_pages / self.kv_cache_total_pages


class AnnaServiceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_started_total = 0
        self._requests_completed_total = 0
        self._requests_failed_total = 0
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._prompt_cache_queries_total = 0
        self._prompt_cache_hits_total = 0
        self._running_requests = 0
        self._waiting_requests = 0

    def record_request_submitted(self, *, waiting: bool) -> None:
        with self._lock:
            self._requests_started_total += 1
            if waiting:
                self._waiting_requests += 1
            else:
                self._running_requests += 1

    def record_requests_started_from_queue(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._waiting_requests = max(0, self._waiting_requests - normalized)
            self._running_requests += normalized

    def record_request_finished(self, *, success: bool) -> None:
        with self._lock:
            self._running_requests = max(0, self._running_requests - 1)
            if success:
                self._requests_completed_total += 1
            else:
                self._requests_failed_total += 1

    def record_prompt_tokens(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._prompt_tokens_total += normalized

    def record_generation_tokens(self, count: int) -> None:
        normalized = max(0, int(count))
        if normalized <= 0:
            return
        with self._lock:
            self._generation_tokens_total += normalized

    def record_prompt_cache_lookup(self, *, hit: bool) -> None:
        with self._lock:
            self._prompt_cache_queries_total += 1
            if hit:
                self._prompt_cache_hits_total += 1

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
            )


class AnnaServiceMetricsLogger:
    def __init__(
        self,
        snapshot_provider: Callable[[], ServiceMetricsSnapshot],
        *,
        interval_seconds: float = 10.0,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.interval_seconds = max(0.0, float(interval_seconds))
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
        prompt_tokens_per_second = prompt_tokens / elapsed
        generation_tokens_per_second = generation_tokens / elapsed
        prompt_cache_hit_rate = 0.0 if cache_queries <= 0 else (cache_hits / cache_queries) * 100.0
        kv_cache_usage = current.kv_cache_usage_ratio * 100.0
        return (
            "Engine metrics: Avg prompt throughput: "
            f"{prompt_tokens_per_second:.1f} tokens/s, Avg generation throughput: "
            f"{generation_tokens_per_second:.1f} tokens/s, Running: {current.running_requests} reqs, "
            f"Waiting: {current.waiting_requests} reqs, GPU KV cache usage: {kv_cache_usage:.1f}% "
            f"({current.kv_cache_used_pages}/{current.kv_cache_total_pages} pages), "
            f"Prompt cache hit rate: {prompt_cache_hit_rate:.1f}%"
        )

    def _run_loop(self) -> None:
        previous = self.snapshot_provider()
        while not self._stop_event.wait(self.interval_seconds):
            current = self.snapshot_provider()
            logger.info(self.format_interval(previous, current))
            previous = current
