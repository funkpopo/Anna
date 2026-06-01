from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

_LATENCY_SAMPLE_LIMIT = 4096


def _quantile(samples: tuple[float, ...], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


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
    prefill_step_recent_seconds: tuple[float, ...] = ()
    decode_step_seconds_total: float = 0.0
    decode_step_count: int = 0
    decode_step_seconds_max: float = 0.0
    decode_step_recent_seconds: tuple[float, ...] = ()
    cache_stack_seconds_total: float = 0.0
    cache_stack_count: int = 0
    cache_stack_seconds_max: float = 0.0
    cache_split_seconds_total: float = 0.0
    cache_split_count: int = 0
    cache_split_seconds_max: float = 0.0
    slot_decode_plan_seconds_total: float = 0.0
    slot_decode_plan_count: int = 0
    slot_decode_plan_seconds_max: float = 0.0
    cpu_sync_count: int = 0
    attention_fallback_count: int = 0
    paged_cache_materialize_count: int = 0
    sampler_full_vocab_sort_count: int = 0
    moe_host_offset_count: int = 0
    moe_router_seconds_total: float = 0.0
    moe_router_count: int = 0
    moe_router_seconds_max: float = 0.0
    moe_dispatch_seconds_total: float = 0.0
    moe_dispatch_count: int = 0
    moe_dispatch_seconds_max: float = 0.0
    moe_expert_gemm_seconds_total: float = 0.0
    moe_expert_gemm_count: int = 0
    moe_expert_gemm_seconds_max: float = 0.0
    moe_scatter_seconds_total: float = 0.0
    moe_scatter_count: int = 0
    moe_scatter_seconds_max: float = 0.0
    moe_staging_seconds_total: float = 0.0
    moe_staging_count: int = 0
    moe_staging_seconds_max: float = 0.0
    moe_cpu_sync_seconds_total: float = 0.0
    moe_cpu_sync_count: int = 0
    moe_cpu_sync_seconds_max: float = 0.0

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
        self._prefill_step_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._decode_step_seconds_total = 0.0
        self._decode_step_count = 0
        self._decode_step_seconds_max = 0.0
        self._decode_step_recent_seconds: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
        self._cache_stack_seconds_total = 0.0
        self._cache_stack_count = 0
        self._cache_stack_seconds_max = 0.0
        self._cache_split_seconds_total = 0.0
        self._cache_split_count = 0
        self._cache_split_seconds_max = 0.0
        self._slot_decode_plan_seconds_total = 0.0
        self._slot_decode_plan_count = 0
        self._slot_decode_plan_seconds_max = 0.0
        self._cpu_sync_count = 0
        self._attention_fallback_count = 0
        self._paged_cache_materialize_count = 0
        self._sampler_full_vocab_sort_count = 0
        self._moe_host_offset_count = 0
        self._moe_router_seconds_total = 0.0
        self._moe_router_count = 0
        self._moe_router_seconds_max = 0.0
        self._moe_dispatch_seconds_total = 0.0
        self._moe_dispatch_count = 0
        self._moe_dispatch_seconds_max = 0.0
        self._moe_expert_gemm_seconds_total = 0.0
        self._moe_expert_gemm_count = 0
        self._moe_expert_gemm_seconds_max = 0.0
        self._moe_scatter_seconds_total = 0.0
        self._moe_scatter_count = 0
        self._moe_scatter_seconds_max = 0.0
        self._moe_staging_seconds_total = 0.0
        self._moe_staging_count = 0
        self._moe_staging_seconds_max = 0.0
        self._moe_cpu_sync_seconds_total = 0.0
        self._moe_cpu_sync_count = 0
        self._moe_cpu_sync_seconds_max = 0.0

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

    def record_decode_step(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._decode_step_seconds_total += normalized
            self._decode_step_count += 1
            self._decode_step_seconds_max = max(self._decode_step_seconds_max, normalized)
            self._decode_step_recent_seconds.append(normalized)
        self._activity_event.set()

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

    def record_slot_decode_plan(self, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        with self._lock:
            self._slot_decode_plan_seconds_total += normalized
            self._slot_decode_plan_count += 1
            self._slot_decode_plan_seconds_max = max(self._slot_decode_plan_seconds_max, normalized)
        self._activity_event.set()

    @staticmethod
    def _normalize_count(count: int) -> int:
        return max(0, int(count))

    def record_cpu_sync(self, *, reason: str = "", count: int = 1) -> None:
        del reason
        normalized = self._normalize_count(count)
        if normalized <= 0:
            return
        with self._lock:
            self._cpu_sync_count += normalized
        self._activity_event.set()

    def record_attention_fallback(self, *, reason: str = "", count: int = 1) -> None:
        del reason
        normalized = self._normalize_count(count)
        if normalized <= 0:
            return
        with self._lock:
            self._attention_fallback_count += normalized
        self._activity_event.set()

    def record_paged_cache_materialization(self, *, reason: str = "", count: int = 1) -> None:
        del reason
        normalized = self._normalize_count(count)
        if normalized <= 0:
            return
        with self._lock:
            self._paged_cache_materialize_count += normalized
        self._activity_event.set()

    def record_sampler_full_vocab_sort(self, *, reason: str = "", count: int = 1) -> None:
        del reason
        normalized = self._normalize_count(count)
        if normalized <= 0:
            return
        with self._lock:
            self._sampler_full_vocab_sort_count += normalized
        self._activity_event.set()

    def record_moe_host_offset(self, *, reason: str = "", count: int = 1) -> None:
        del reason
        normalized = self._normalize_count(count)
        if normalized <= 0:
            return
        with self._lock:
            self._moe_host_offset_count += normalized
        self._activity_event.set()

    def record_moe_stage(self, *, stage: str, seconds: float) -> None:
        normalized = max(0.0, float(seconds))
        stage_name = str(stage).strip().lower()
        mapping = {
            "router": (
                "_moe_router_seconds_total",
                "_moe_router_count",
                "_moe_router_seconds_max",
            ),
            "dispatch": (
                "_moe_dispatch_seconds_total",
                "_moe_dispatch_count",
                "_moe_dispatch_seconds_max",
            ),
            "expert_gemm": (
                "_moe_expert_gemm_seconds_total",
                "_moe_expert_gemm_count",
                "_moe_expert_gemm_seconds_max",
            ),
            "scatter": (
                "_moe_scatter_seconds_total",
                "_moe_scatter_count",
                "_moe_scatter_seconds_max",
            ),
            "staging": (
                "_moe_staging_seconds_total",
                "_moe_staging_count",
                "_moe_staging_seconds_max",
            ),
            "cpu_sync": (
                "_moe_cpu_sync_seconds_total",
                "_moe_cpu_sync_count",
                "_moe_cpu_sync_seconds_max",
            ),
        }
        fields = mapping.get(stage_name)
        if fields is None:
            return
        total_name, count_name, max_name = fields
        with self._lock:
            setattr(self, total_name, getattr(self, total_name) + normalized)
            setattr(self, count_name, getattr(self, count_name) + 1)
            setattr(self, max_name, max(getattr(self, max_name), normalized))
        self._activity_event.set()

    def record_request_finished(self, *, success: bool, from_waiting: bool = False) -> None:
        with self._lock:
            if from_waiting:
                self._waiting_requests = max(0, self._waiting_requests - 1)
            else:
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
                prefill_step_recent_seconds=tuple(self._prefill_step_recent_seconds),
                decode_step_seconds_total=self._decode_step_seconds_total,
                decode_step_count=self._decode_step_count,
                decode_step_seconds_max=self._decode_step_seconds_max,
                decode_step_recent_seconds=tuple(self._decode_step_recent_seconds),
                cache_stack_seconds_total=self._cache_stack_seconds_total,
                cache_stack_count=self._cache_stack_count,
                cache_stack_seconds_max=self._cache_stack_seconds_max,
                cache_split_seconds_total=self._cache_split_seconds_total,
                cache_split_count=self._cache_split_count,
                cache_split_seconds_max=self._cache_split_seconds_max,
                slot_decode_plan_seconds_total=self._slot_decode_plan_seconds_total,
                slot_decode_plan_count=self._slot_decode_plan_count,
                slot_decode_plan_seconds_max=self._slot_decode_plan_seconds_max,
                cpu_sync_count=self._cpu_sync_count,
                attention_fallback_count=self._attention_fallback_count,
                paged_cache_materialize_count=self._paged_cache_materialize_count,
                sampler_full_vocab_sort_count=self._sampler_full_vocab_sort_count,
                moe_host_offset_count=self._moe_host_offset_count,
                moe_router_seconds_total=self._moe_router_seconds_total,
                moe_router_count=self._moe_router_count,
                moe_router_seconds_max=self._moe_router_seconds_max,
                moe_dispatch_seconds_total=self._moe_dispatch_seconds_total,
                moe_dispatch_count=self._moe_dispatch_count,
                moe_dispatch_seconds_max=self._moe_dispatch_seconds_max,
                moe_expert_gemm_seconds_total=self._moe_expert_gemm_seconds_total,
                moe_expert_gemm_count=self._moe_expert_gemm_count,
                moe_expert_gemm_seconds_max=self._moe_expert_gemm_seconds_max,
                moe_scatter_seconds_total=self._moe_scatter_seconds_total,
                moe_scatter_count=self._moe_scatter_count,
                moe_scatter_seconds_max=self._moe_scatter_seconds_max,
                moe_staging_seconds_total=self._moe_staging_seconds_total,
                moe_staging_count=self._moe_staging_count,
                moe_staging_seconds_max=self._moe_staging_seconds_max,
                moe_cpu_sync_seconds_total=self._moe_cpu_sync_seconds_total,
                moe_cpu_sync_count=self._moe_cpu_sync_count,
                moe_cpu_sync_seconds_max=self._moe_cpu_sync_seconds_max,
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
        cache_stack_total = max(0.0, current.cache_stack_seconds_total - previous.cache_stack_seconds_total)
        cache_stack_count = max(0, current.cache_stack_count - previous.cache_stack_count)
        cache_split_total = max(0.0, current.cache_split_seconds_total - previous.cache_split_seconds_total)
        cache_split_count = max(0, current.cache_split_count - previous.cache_split_count)
        slot_decode_plan_total = max(
            0.0,
            current.slot_decode_plan_seconds_total - previous.slot_decode_plan_seconds_total,
        )
        slot_decode_plan_count = max(0, current.slot_decode_plan_count - previous.slot_decode_plan_count)
        cpu_sync_count = max(0, current.cpu_sync_count - previous.cpu_sync_count)
        attention_fallback_count = max(0, current.attention_fallback_count - previous.attention_fallback_count)
        paged_cache_materialize_count = max(0, current.paged_cache_materialize_count - previous.paged_cache_materialize_count)
        sampler_full_vocab_sort_count = max(0, current.sampler_full_vocab_sort_count - previous.sampler_full_vocab_sort_count)
        moe_host_offset_count = max(0, current.moe_host_offset_count - previous.moe_host_offset_count)
        moe_router_total = max(0.0, current.moe_router_seconds_total - previous.moe_router_seconds_total)
        moe_router_count = max(0, current.moe_router_count - previous.moe_router_count)
        moe_dispatch_total = max(0.0, current.moe_dispatch_seconds_total - previous.moe_dispatch_seconds_total)
        moe_dispatch_count = max(0, current.moe_dispatch_count - previous.moe_dispatch_count)
        moe_expert_gemm_total = max(0.0, current.moe_expert_gemm_seconds_total - previous.moe_expert_gemm_seconds_total)
        moe_expert_gemm_count = max(0, current.moe_expert_gemm_count - previous.moe_expert_gemm_count)
        moe_scatter_total = max(0.0, current.moe_scatter_seconds_total - previous.moe_scatter_seconds_total)
        moe_scatter_count = max(0, current.moe_scatter_count - previous.moe_scatter_count)
        moe_staging_total = max(0.0, current.moe_staging_seconds_total - previous.moe_staging_seconds_total)
        moe_staging_count = max(0, current.moe_staging_count - previous.moe_staging_count)
        moe_cpu_sync_total = max(0.0, current.moe_cpu_sync_seconds_total - previous.moe_cpu_sync_seconds_total)
        moe_cpu_sync_count = max(0, current.moe_cpu_sync_count - previous.moe_cpu_sync_count)
        prompt_tokens_per_second = prompt_tokens / elapsed
        generation_tokens_per_second = generation_tokens / elapsed
        prompt_cache_hit_rate = 0.0 if cache_queries <= 0 else (cache_hits / cache_queries) * 100.0
        kv_cache_usage = current.kv_cache_usage_ratio * 100.0
        queue_wait_avg_ms = 0.0 if queue_wait_count <= 0 else (queue_wait_total / queue_wait_count) * 1000.0
        prefill_step_avg_ms = 0.0 if prefill_step_count <= 0 else (prefill_step_total / prefill_step_count) * 1000.0
        decode_step_avg_ms = 0.0 if decode_step_count <= 0 else (decode_step_total / decode_step_count) * 1000.0
        cache_stack_avg_ms = 0.0 if cache_stack_count <= 0 else (cache_stack_total / cache_stack_count) * 1000.0
        cache_split_avg_ms = 0.0 if cache_split_count <= 0 else (cache_split_total / cache_split_count) * 1000.0
        slot_decode_plan_avg_ms = (
            0.0 if slot_decode_plan_count <= 0 else (slot_decode_plan_total / slot_decode_plan_count) * 1000.0
        )
        moe_router_avg_ms = 0.0 if moe_router_count <= 0 else (moe_router_total / moe_router_count) * 1000.0
        moe_dispatch_avg_ms = 0.0 if moe_dispatch_count <= 0 else (moe_dispatch_total / moe_dispatch_count) * 1000.0
        moe_expert_gemm_avg_ms = (
            0.0 if moe_expert_gemm_count <= 0 else (moe_expert_gemm_total / moe_expert_gemm_count) * 1000.0
        )
        moe_scatter_avg_ms = 0.0 if moe_scatter_count <= 0 else (moe_scatter_total / moe_scatter_count) * 1000.0
        moe_staging_avg_ms = 0.0 if moe_staging_count <= 0 else (moe_staging_total / moe_staging_count) * 1000.0
        moe_cpu_sync_avg_ms = 0.0 if moe_cpu_sync_count <= 0 else (moe_cpu_sync_total / moe_cpu_sync_count) * 1000.0
        prefill_recent = current.prefill_step_recent_seconds
        decode_recent = current.decode_step_recent_seconds
        prefill_p50_ms = _quantile(prefill_recent, 0.50) * 1000.0
        prefill_p95_ms = _quantile(prefill_recent, 0.95) * 1000.0
        prefill_p99_ms = _quantile(prefill_recent, 0.99) * 1000.0
        decode_p50_ms = _quantile(decode_recent, 0.50) * 1000.0
        decode_p95_ms = _quantile(decode_recent, 0.95) * 1000.0
        decode_p99_ms = _quantile(decode_recent, 0.99) * 1000.0
        return (
            "Engine metrics: Interval prompt: "
            f"{prompt_tokens_per_second:.1f} tok/s, Interval generation: "
            f"{generation_tokens_per_second:.1f} tok/s, Running: {current.running_requests} reqs, "
            f"Queue wait avg/max: {queue_wait_avg_ms:.1f}/{current.queue_wait_seconds_max * 1000.0:.1f} ms, "
            f"Prefill step avg/max: {prefill_step_avg_ms:.1f}/{current.prefill_step_seconds_max * 1000.0:.1f} ms, "
            f"Prefill step p50/p95/p99: {prefill_p50_ms:.1f}/{prefill_p95_ms:.1f}/{prefill_p99_ms:.1f} ms, "
            f"Decode step avg/max: {decode_step_avg_ms:.1f}/{current.decode_step_seconds_max * 1000.0:.1f} ms, "
            f"Decode step p50/p95/p99: {decode_p50_ms:.1f}/{decode_p95_ms:.1f}/{decode_p99_ms:.1f} ms, "
            f"Cache stack avg/max: {cache_stack_avg_ms:.1f}/{current.cache_stack_seconds_max * 1000.0:.1f} ms, "
            f"Cache split avg/max: {cache_split_avg_ms:.1f}/{current.cache_split_seconds_max * 1000.0:.1f} ms, "
            f"Slot decode plan avg/max: {slot_decode_plan_avg_ms:.1f}/"
            f"{current.slot_decode_plan_seconds_max * 1000.0:.1f} ms, "
            f"Hot path events: cpu_sync={cpu_sync_count}, attention_fallback={attention_fallback_count}, "
            f"paged_cache_materialize={paged_cache_materialize_count}, "
            f"sampler_full_vocab_sort={sampler_full_vocab_sort_count}, moe_host_offset={moe_host_offset_count}, "
            f"MoE stage avg/max ms: router={moe_router_avg_ms:.3f}/"
            f"{current.moe_router_seconds_max * 1000.0:.3f}, dispatch={moe_dispatch_avg_ms:.3f}/"
            f"{current.moe_dispatch_seconds_max * 1000.0:.3f}, expert_gemm={moe_expert_gemm_avg_ms:.3f}/"
            f"{current.moe_expert_gemm_seconds_max * 1000.0:.3f}, scatter={moe_scatter_avg_ms:.3f}/"
            f"{current.moe_scatter_seconds_max * 1000.0:.3f}, staging={moe_staging_avg_ms:.3f}/"
            f"{current.moe_staging_seconds_max * 1000.0:.3f}, cpu_sync={moe_cpu_sync_avg_ms:.3f}/"
            f"{current.moe_cpu_sync_seconds_max * 1000.0:.3f}, "
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
            current.cache_stack_count - previous.cache_stack_count,
            current.cache_split_count - previous.cache_split_count,
            current.slot_decode_plan_count - previous.slot_decode_plan_count,
            current.cpu_sync_count - previous.cpu_sync_count,
            current.attention_fallback_count - previous.attention_fallback_count,
            current.paged_cache_materialize_count - previous.paged_cache_materialize_count,
            current.sampler_full_vocab_sort_count - previous.sampler_full_vocab_sort_count,
            current.moe_host_offset_count - previous.moe_host_offset_count,
            current.moe_router_count - previous.moe_router_count,
            current.moe_dispatch_count - previous.moe_dispatch_count,
            current.moe_expert_gemm_count - previous.moe_expert_gemm_count,
            current.moe_scatter_count - previous.moe_scatter_count,
            current.moe_staging_count - previous.moe_staging_count,
            current.moe_cpu_sync_count - previous.moe_cpu_sync_count,
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
