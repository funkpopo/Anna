from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import torch

from anna.mm.prepared_inputs import PreparedInputsLike
from anna.runtime.streaming import IncrementalTextAssembler
from anna.sampling.sampler import sample_next_token, sample_next_token_from_candidates

if TYPE_CHECKING:
    from anna.runtime.qwen3_5_text_engine import (
        AnnaEngineError,
        AnnaQwen3_5TextEngine,
        GenerationConfig,
        StreamEvent,
        TextGenerationResult,
    )


_DONE = object()


@dataclass(slots=True)
class SchedulerRequest:
    prepared: PreparedInputsLike
    config: "GenerationConfig"
    stream: bool
    prompt_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0
    completion_ids: list[int] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    input_ids: torch.Tensor | None = None
    past_key_values: object | None = None
    repetition_history: torch.Tensor | None = None
    repetition_history_ids: set[int] | None = None
    assembler: IncrementalTextAssembler | None = None
    result: "TextGenerationResult | None" = None
    error: "AnnaEngineError | None" = None
    done: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue[object] = field(default_factory=queue.Queue)
    generation_started_at: float | None = None
    first_token_at: float | None = None
    queued_at: float = field(default_factory=time.perf_counter)
    cancelled: bool = False


@dataclass(slots=True)
class SchedulerPrefillGroup:
    requests: list[SchedulerRequest]
    batched: PreparedInputsLike
    past_key_values: object | None
    next_start_idx: int
    prompt_tokens_recorded: int = 0


@dataclass(slots=True)
class SchedulerDecodeGroup:
    requests: list[SchedulerRequest]
    input_ids: torch.Tensor
    past_key_values: object


class AnnaScheduler:
    def __init__(
        self,
        engine: "AnnaQwen3_5TextEngine",
        *,
        max_batch_size: int = 4,
        batch_wait_ms: float = 2.0,
        prefill_interval_steps: int = 1,
        max_prefill_tokens: int = 0,
        max_decode_tokens: int = 0,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_seconds = max(0.0, batch_wait_ms) / 1000.0
        self.prefill_interval_steps = max(1, int(prefill_interval_steps))
        self.max_prefill_tokens = max(0, int(max_prefill_tokens))
        self.max_decode_tokens = max(0, int(max_decode_tokens))
        self._decode_steps_since_prefill = 0
        self._pending: deque[SchedulerRequest] = deque()
        self._condition = threading.Condition()
        self._stop = False
        self._fatal_error: AnnaEngineError | None = None
        self._worker = threading.Thread(target=self._run_loop, name="anna-scheduler", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def generate(self, prepared: PreparedInputsLike, *, config: "GenerationConfig") -> "TextGenerationResult":
        request = self._submit(prepared, config=config, stream=False)
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Scheduler request completed without a result.")
        return request.result

    def stream(self, prepared: PreparedInputsLike, *, config: "GenerationConfig") -> Iterator["StreamEvent"]:
        request = self._submit(prepared, config=config, stream=True)
        try:
            while True:
                item = request.events.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not request.done.is_set():
                self.cancel(request)

    def cancel(self, request: SchedulerRequest) -> None:
        metrics = getattr(self.engine, "metrics", None)
        with self._condition:
            if request.done.is_set() or request.cancelled:
                return
            request.cancelled = True
            if self._pending:
                self._pending = deque(candidate for candidate in self._pending if candidate is not request)
            self._condition.notify_all()
        request.error = self._cancelled_error()
        self._release_request_cache(request)
        request.events.put(_DONE)
        request.done.set()
        if metrics is not None:
            metrics.record_request_finished(success=False)
        self.engine._trim_runtime_cache_if_idle()

    def _submit(self, prepared: PreparedInputsLike, *, config: "GenerationConfig", stream: bool) -> SchedulerRequest:
        if self._fatal_error is not None:
            raise self._fatal_error
        request = SchedulerRequest(prepared=prepared, config=config, stream=stream)
        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_request_submitted(waiting=True)
        with self._condition:
            self._pending.append(request)
            self._condition.notify()
        return request

    def _run_loop(self) -> None:
        active: list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup] = []
        try:
            while True:
                with self._condition:
                    while not self._stop and not self._pending and not active:
                        self._condition.wait()
                    if self._stop and not self._pending and not active:
                        return
                    if not active and self._pending and self.batch_wait_seconds > 0:
                        self._condition.wait(timeout=self.batch_wait_seconds)

                    pending_batch: list[SchedulerRequest] = []
                    should_admit_pending = not active or self._should_admit_prefill(active)
                    if should_admit_pending:
                        while self._pending and len(pending_batch) < self.max_batch_size:
                            request = self._pending.popleft()
                            if self._is_request_cancelled(request):
                                self._drop_cancelled_request(request)
                            else:
                                pending_batch.append(request)

                if pending_batch:
                    try:
                        active.extend(self._prefill_batch(pending_batch))
                    except Exception as exc:  # pragma: no cover - worker-level best effort
                        self._fail_requests(pending_batch, self._normalize_error(exc))

                if not active:
                    continue

                next_active: list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup] = []
                active_requests = [item for item in active if isinstance(item, SchedulerRequest)]
                prefill_groups = [item for item in active if isinstance(item, SchedulerPrefillGroup)]
                decode_groups = [item for item in active if isinstance(item, SchedulerDecodeGroup)]
                active_requests = [request for request in active_requests if not self._drop_cancelled_request(request)]
                prefill_groups = [group for group in prefill_groups if self._filter_prefill_group(group)]
                ready = [
                    request
                    for request in active_requests
                    if request.past_key_values is not None and request.input_ids is not None
                ]
                pending_decode = [
                    request
                    for request in active_requests
                    if not (request.past_key_values is not None and request.input_ids is not None)
                ]
                for request in pending_decode:
                    self._fail_request(request, RuntimeError("Missing decode cache for active request."))

                decoded_any = False
                if decode_groups:
                    for group in decode_groups:
                        try:
                            next_active.extend(self._decode_group(group))
                            self._decode_steps_since_prefill += 1
                            decoded_any = True
                        except Exception as exc:  # pragma: no cover - worker-level best effort
                            self._release_decode_group_cache(group)
                            self._fail_requests(group.requests, self._normalize_error(exc))

                if ready:
                    for chunk in self._iter_decode_chunks(ready):
                        try:
                            next_active.extend(self._decode_batch(chunk))
                            self._decode_steps_since_prefill += 1
                            decoded_any = True
                        except Exception as exc:  # pragma: no cover - worker-level best effort
                            self._fail_requests(chunk, self._normalize_error(exc))
                    if prefill_groups and self._should_run_prefill_step():
                        group = prefill_groups[0]
                        try:
                            next_active.extend(self._prefill_group_step(group))
                            self._decode_steps_since_prefill = 0
                        except Exception as exc:  # pragma: no cover - worker-level best effort
                            self._fail_requests(group.requests, self._normalize_error(exc))
                        next_active.extend(prefill_groups[1:])
                    else:
                        next_active.extend(prefill_groups)
                elif decoded_any:
                    if prefill_groups and self._should_run_prefill_step():
                        group = prefill_groups[0]
                        try:
                            next_active.extend(self._prefill_group_step(group))
                            self._decode_steps_since_prefill = 0
                        except Exception as exc:  # pragma: no cover - worker-level best effort
                            self._fail_requests(group.requests, self._normalize_error(exc))
                        next_active.extend(prefill_groups[1:])
                    else:
                        next_active.extend(prefill_groups)
                elif prefill_groups:
                    group = prefill_groups[0]
                    try:
                        next_active.extend(self._prefill_group_step(group))
                        self._decode_steps_since_prefill = 0
                    except Exception as exc:  # pragma: no cover - worker-level best effort
                        self._fail_requests(group.requests, self._normalize_error(exc))
                    next_active.extend(prefill_groups[1:])

                active = next_active
        except BaseException as exc:  # pragma: no cover - catastrophic worker failure
            pending: list[SchedulerRequest] = []
            with self._condition:
                while self._pending:
                    pending.append(self._pending.popleft())
            fatal = self._normalize_worker_crash(exc)
            self._fatal_error = fatal
            failed_active: list[SchedulerRequest] = []
            for item in active:
                if isinstance(item, SchedulerPrefillGroup):
                    failed_active.extend(item.requests)
                    release = getattr(item.past_key_values, "release", None)
                    if callable(release):
                        release()
                elif isinstance(item, SchedulerDecodeGroup):
                    failed_active.extend(item.requests)
                    self._release_decode_group_cache(item)
                else:
                    failed_active.append(item)
            self._fail_requests(failed_active + pending, fatal)

    def _prefill_batch(
        self,
        requests: list[SchedulerRequest],
    ) -> list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup]:
        for request in requests:
            if self._is_request_cancelled(request):
                self._drop_cancelled_request(request)
        requests = [request for request in requests if not self._is_request_cancelled(request)]
        if not requests:
            return []
        for request in requests:
            self._ensure_request_initialized(request)

        requests, deferred = self._select_prefill_admission(requests)
        if deferred:
            self._requeue_front(deferred)
        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_prefill_admission(
                admitted_requests=len(requests),
                deferred_requests=len(deferred),
                admitted_tokens=sum(max(1, int(request.prompt_length)) for request in requests),
            )
        if not requests:
            return []

        if metrics is not None:
            metrics.record_requests_started_from_queue(len(requests))
        for request in requests:
            request.generation_started_at = time.perf_counter()
            if metrics is not None:
                metrics.record_queue_wait(request.generation_started_at - request.queued_at)

        active: list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup] = []
        groups: dict[int, list[SchedulerRequest]] = defaultdict(list)
        for request in requests:
            groups[request.prompt_length].append(request)

        for _, group in sorted(groups.items()):
            for chunk in self._iter_prefill_group_chunks(group):
                active.extend(self._prefill_same_length_group(chunk))
        return active

    def _ensure_request_initialized(self, request: SchedulerRequest) -> None:
        if request.prompt_length > 0:
            return
        request.prompt_ids, request.prompt_length, request.config = self.engine._validate_generation_request(
            request.prepared,
            config=request.config,
        )
        request.repetition_history, request.repetition_history_ids = self.engine._init_repetition_penalty_state(
            request.prompt_ids,
            request.config.repetition_penalty,
            request.config.presence_penalty,
        )
        request.assembler = IncrementalTextAssembler(
            tokenizer=self.engine.tokenizer,
            stop_strings=request.config.stop_strings,
        )

    def _select_prefill_admission(
        self,
        requests: list[SchedulerRequest],
    ) -> tuple[list[SchedulerRequest], list[SchedulerRequest]]:
        if self.max_prefill_tokens <= 0:
            return requests, []
        accepted: list[SchedulerRequest] = []
        deferred: list[SchedulerRequest] = []
        used_tokens = 0
        for request in requests:
            prompt_tokens = max(1, int(request.prompt_length))
            if accepted and used_tokens + prompt_tokens > self.max_prefill_tokens:
                deferred.append(request)
                continue
            accepted.append(request)
            used_tokens += prompt_tokens
        return accepted, deferred

    def _requeue_front(self, requests: list[SchedulerRequest]) -> None:
        if not requests:
            return
        with self._condition:
            for request in reversed(requests):
                self._pending.appendleft(request)
            self._condition.notify_all()

    def _iter_prefill_group_chunks(self, requests: list[SchedulerRequest]) -> Iterator[list[SchedulerRequest]]:
        if self.max_prefill_tokens <= 0:
            yield requests
            return
        chunk: list[SchedulerRequest] = []
        chunk_tokens = 0
        for request in requests:
            prompt_tokens = max(1, int(request.prompt_length))
            if chunk and chunk_tokens + prompt_tokens > self.max_prefill_tokens:
                yield chunk
                chunk = []
                chunk_tokens = 0
            chunk.append(request)
            chunk_tokens += prompt_tokens
        if chunk:
            yield chunk

    def _should_admit_prefill(
        self,
        active: list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup],
    ) -> bool:
        if not self._pending:
            return False
        if any(isinstance(item, SchedulerPrefillGroup) for item in active):
            return False
        return self._decode_steps_since_prefill >= self.prefill_interval_steps

    def _should_run_prefill_step(self) -> bool:
        return self._decode_steps_since_prefill >= self.prefill_interval_steps

    def _prefill_same_length_group(
        self,
        requests: list[SchedulerRequest],
    ) -> list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup]:
        for request in requests:
            if self._is_request_cancelled(request):
                self._drop_cancelled_request(request)
        requests = [request for request in requests if not self._is_request_cancelled(request)]
        if not requests:
            return []
        batched = self._batch_text_inputs(requests)
        self._guard_batch_memory(requests)
        prompt_length = int(batched.input_ids.shape[1])
        configured_chunk_size = int(getattr(self.engine.optimization_config, "prefill_chunk_size", 0))
        past_key_values = (
            self.engine._reserve_prefill_cache(batched)
            if configured_chunk_size > 0 and prompt_length > configured_chunk_size
            else None
        )
        return self._prefill_group_step(
            SchedulerPrefillGroup(
                requests=requests,
                batched=batched,
                past_key_values=past_key_values,
                next_start_idx=0,
            )
        )

    def _prefill_group_step(
        self,
        group: SchedulerPrefillGroup,
    ) -> list[SchedulerRequest | SchedulerPrefillGroup | SchedulerDecodeGroup]:
        requests = [request for request in group.requests if not self._drop_cancelled_request(request)]
        group.requests = requests
        if not requests:
            release = getattr(group.past_key_values, "release", None)
            if callable(release):
                release()
            return []
        prompt_length = int(group.batched.input_ids.shape[1])
        configured_chunk_size = int(getattr(self.engine.optimization_config, "prefill_chunk_size", 0))
        chunk_size = prompt_length if configured_chunk_size <= 0 else min(configured_chunk_size, prompt_length)
        start_idx = group.next_start_idx
        end_idx = min(prompt_length, start_idx + chunk_size)
        is_final_chunk = end_idx >= prompt_length
        chunk = type(group.batched)(
            prompt="",
            input_ids=group.batched.input_ids[:, start_idx:end_idx],
            attention_mask=group.batched.attention_mask[:, :end_idx] if start_idx == 0 else None,
            mm_token_type_ids=group.batched.mm_token_type_ids[:, start_idx:end_idx],
        )
        chunk = self.engine.device_context.move_prepared_inputs(chunk)
        try:
            started_at = time.perf_counter()
            with self.engine.execution_lock:
                with torch.inference_mode():
                    if is_final_chunk:
                        outputs = self._forward_batch_maybe_topk(
                            stage="scheduler_prefill" if start_idx == 0 else f"scheduler_prefill[{start_idx}:{end_idx}]",
                            requests=requests,
                            input_ids=chunk.input_ids,
                            attention_mask=chunk.attention_mask,
                            past_key_values=group.past_key_values,
                            model_kwargs=self.engine._build_prefill_model_kwargs(chunk, include_media=start_idx == 0),
                        )
                    else:
                        outputs = self.engine._profiled_forward_generation_model(
                            stage=f"scheduler_prefill[{start_idx}:{end_idx}]",
                            input_ids=chunk.input_ids,
                            attention_mask=chunk.attention_mask,
                            past_key_values=group.past_key_values,
                            model_kwargs=self.engine._build_prefill_model_kwargs(chunk, include_media=start_idx == 0),
                            use_cache=True,
                            logits_to_keep=1,
                        )
        except RuntimeError as exc:
            release = getattr(group.past_key_values, "release", None)
            if callable(release):
                release()
            raise self.engine._handle_runtime_failure(exc) from exc

        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_prefill_step(time.perf_counter() - started_at)
        group.past_key_values = outputs.past_key_values
        if metrics is not None:
            chunk_tokens = max(0, end_idx - start_idx) * len(requests)
            if chunk_tokens > 0:
                metrics.record_prompt_tokens(chunk_tokens)
                group.prompt_tokens_recorded += chunk_tokens
        if not is_final_chunk:
            group.next_start_idx = end_idx
            return [group]

        total_prompt_tokens = sum(request.prompt_length for request in requests)
        if metrics is not None and group.prompt_tokens_recorded < total_prompt_tokens:
            metrics.record_prompt_tokens(total_prompt_tokens - group.prompt_tokens_recorded)
        return list(
            self._consume_batch_outputs(
                requests,
                outputs,
                batch_cache=outputs.past_key_values,
                keep_batched_cache=len(requests) > 1,
                avoid_turboquant_clone=False,
            )
        )

    def _decode_batch(self, requests: list[SchedulerRequest]) -> list[SchedulerRequest | SchedulerDecodeGroup]:
        for request in requests:
            if self._is_request_cancelled(request):
                self._drop_cancelled_request(request)
        requests = [request for request in requests if not self._is_request_cancelled(request)]
        if not requests:
            return []
        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_decode_batch(
                requests=len(requests),
                token_cost=sum(self._decode_request_token_cost(request) for request in requests),
            )
        if len(requests) == 1:
            return self._decode_single(requests[0])
        input_ids = torch.cat([request.input_ids for request in requests if request.input_ids is not None], dim=0)
        caches = [request.past_key_values for request in requests if request.past_key_values is not None]
        if not caches:
            raise RuntimeError("Scheduler decode batch is missing cache state.")
        cache_type = type(caches[0])
        stack = getattr(cache_type, "stack", None)
        if not callable(stack):
            raise RuntimeError(f"Cache type {cache_type.__name__} does not support scheduler batching.")
        metrics = getattr(self.engine, "metrics", None)
        stack_started_at = time.perf_counter()
        stack_kwargs = {"clone_turboquant_rows": False} if cache_type.__name__ == "Qwen3DynamicCache" else {}
        batch_cache = stack(caches, self.engine.config.text_config, **stack_kwargs)
        for request in requests:
            request.past_key_values = None
        if metrics is not None:
            metrics.record_cache_stack(time.perf_counter() - stack_started_at)

        try:
            started_at = time.perf_counter()
            with self.engine.execution_lock:
                with torch.inference_mode():
                    outputs = self._forward_batch_maybe_topk(
                        stage="scheduler_decode",
                        requests=requests,
                        input_ids=input_ids,
                        past_key_values=batch_cache,
                    )
        except RuntimeError as exc:
            release = getattr(batch_cache, "release", None)
            if callable(release):
                release()
            raise self.engine._handle_runtime_failure(exc) from exc

        if metrics is not None:
            metrics.record_decode_step(time.perf_counter() - started_at)

        batch_cache = outputs.past_key_values if outputs.past_key_values is not None else batch_cache
        return self._consume_batch_outputs(
            requests,
            outputs,
            batch_cache=batch_cache,
            keep_batched_cache=len(requests) > 1,
            avoid_turboquant_clone=True,
        )

    def _decode_single(self, request: SchedulerRequest) -> list[SchedulerRequest | SchedulerDecodeGroup]:
        if request.input_ids is None or request.past_key_values is None:
            raise RuntimeError("Scheduler single decode is missing input ids or cache state.")
        metrics = getattr(self.engine, "metrics", None)
        try:
            started_at = time.perf_counter()
            with self.engine.execution_lock:
                with torch.inference_mode():
                    outputs = self._forward_batch_maybe_topk(
                        stage="scheduler_decode",
                        requests=[request],
                        input_ids=request.input_ids,
                        past_key_values=request.past_key_values,
                    )
        except RuntimeError as exc:
            raise self.engine._handle_runtime_failure(exc) from exc

        if metrics is not None:
            metrics.record_decode_step(time.perf_counter() - started_at)

        batch_cache = outputs.past_key_values if outputs.past_key_values is not None else request.past_key_values
        request.past_key_values = None
        return self._consume_batch_outputs(
            [request],
            outputs,
            batch_cache=batch_cache,
            keep_batched_cache=False,
            avoid_turboquant_clone=True,
        )

    def _decode_group(self, group: SchedulerDecodeGroup) -> list[SchedulerRequest | SchedulerDecodeGroup]:
        if not group.requests:
            self._release_decode_group_cache(group)
            return []

        if any(self._is_request_cancelled(request) for request in group.requests):
            remaining = self._compact_decode_group_for_membership_change(group)
            if not remaining:
                return []
            if len(remaining) == 1:
                return remaining
            if all(request.past_key_values is None for request in remaining):
                return [self._make_decode_group_from_batched_cache(remaining, group.past_key_values)]
            return [self._make_decode_group_from_requests(remaining)]

        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_decode_batch(
                requests=len(group.requests),
                token_cost=self._decode_group_token_cost(group),
            )
        try:
            started_at = time.perf_counter()
            with self.engine.execution_lock:
                with torch.inference_mode():
                    outputs = self._forward_batch_maybe_topk(
                        stage="scheduler_decode",
                        requests=group.requests,
                        input_ids=group.input_ids,
                        past_key_values=group.past_key_values,
                    )
        except RuntimeError as exc:
            raise self.engine._handle_runtime_failure(exc) from exc

        if metrics is not None:
            metrics.record_decode_step(time.perf_counter() - started_at)

        batch_cache = outputs.past_key_values if outputs.past_key_values is not None else group.past_key_values
        return self._consume_batch_outputs(
            group.requests,
            outputs,
            batch_cache=batch_cache,
            keep_batched_cache=len(group.requests) > 1,
            avoid_turboquant_clone=True,
        )

    def _split_output_caches(self, outputs, request_count: int, *, avoid_turboquant_clone: bool) -> list:
        if outputs.past_key_values is None:
            return [None] * request_count
        return self._split_cache(outputs.past_key_values, request_count, avoid_turboquant_clone=avoid_turboquant_clone)

    def _split_cache(self, cache: object, request_count: int, *, avoid_turboquant_clone: bool) -> list:
        metrics = getattr(self.engine, "metrics", None)
        split_started_at = time.perf_counter()
        split_kwargs = (
            {"clone_turboquant_rows": False}
            if avoid_turboquant_clone and type(cache).__name__ == "Qwen3DynamicCache"
            else {}
        )
        split_batch = getattr(cache, "split_batch", None)
        if not callable(split_batch):
            raise RuntimeError(f"Cache type {type(cache).__name__} does not support scheduler split.")
        split_caches = split_batch(**split_kwargs)
        if metrics is not None:
            metrics.record_cache_split(time.perf_counter() - split_started_at)
        if len(split_caches) != request_count:
            raise RuntimeError(
                f"Scheduler cache split returned {len(split_caches)} rows for {request_count} requests."
            )
        return split_caches

    def _consume_batch_outputs(
        self,
        requests: list[SchedulerRequest],
        outputs,
        *,
        batch_cache: object | None,
        keep_batched_cache: bool,
        avoid_turboquant_clone: bool,
    ) -> list[SchedulerRequest | SchedulerDecodeGroup]:
        """Sample one token per request from batched outputs; return requests that stay active.

        When every row remains active, the returned SchedulerDecodeGroup keeps the batched
        KV cache intact across decode steps. This avoids the stack/split churn that otherwise
        dominates small-token continuous batches.
        """
        metrics = getattr(self.engine, "metrics", None)
        stop_token_ids = self.engine._stop_token_ids()
        next_active: list[SchedulerRequest] = []
        next_input_ids: list[torch.Tensor] = []
        continuing_by_row: dict[int, SchedulerRequest] = {}
        for row_idx, request in enumerate(requests):
            if self._is_request_cancelled(request):
                self._drop_cancelled_request(request)
                continue
            next_token = self._sample_next_token_from_outputs(outputs, row_idx=row_idx, request=request)
            token_id = self.engine._token_id_from_tensor(next_token)
            if request.first_token_at is None:
                request.first_token_at = time.perf_counter()
            if token_id in stop_token_ids:
                self._finish_request(request, finish_reason="stop")
                continue

            request.completion_ids.append(token_id)
            if metrics is not None:
                metrics.record_generation_tokens(1)
            request.repetition_history, request.repetition_history_ids = self.engine._append_repetition_penalty_token(
                history_tensor=request.repetition_history,
                history_ids=request.repetition_history_ids,
                next_token=next_token,
                token_id=token_id,
            )
            delta, hit_stop = request.assembler.feed_token(token_id) if request.assembler is not None else ("", False)
            if delta:
                self._emit_text(request, delta)
            if hit_stop:
                self._finish_request(request, finish_reason="stop")
                continue

            if len(request.completion_ids) >= request.config.max_new_tokens:
                self._finish_request(request, finish_reason="length")
                continue

            request.input_ids = next_token.view(1, 1)
            next_input_ids.append(request.input_ids)
            continuing_by_row[row_idx] = request
            next_active.append(request)

        if batch_cache is None:
            return next_active

        if keep_batched_cache and next_active and len(next_active) == len(requests):
            for request in next_active:
                request.past_key_values = None
            return [
                SchedulerDecodeGroup(
                    requests=next_active,
                    input_ids=torch.cat(next_input_ids, dim=0),
                    past_key_values=batch_cache,
                )
            ]

        if not next_active:
            release = getattr(batch_cache, "release", None)
            if callable(release):
                release()
            return []

        selected_rows = list(continuing_by_row)
        compacted_cache = self._compact_cache_rows(batch_cache, selected_rows)
        if compacted_cache is not None:
            if keep_batched_cache and len(next_active) > 1:
                for request in next_active:
                    request.past_key_values = None
                return [
                    SchedulerDecodeGroup(
                        requests=next_active,
                        input_ids=torch.cat(next_input_ids, dim=0),
                        past_key_values=compacted_cache,
                    )
                ]
            next_active[0].past_key_values = compacted_cache
            return next_active

        split_caches = self._split_cache(
            batch_cache,
            len(requests),
            avoid_turboquant_clone=avoid_turboquant_clone,
        )
        for row_idx, cache in enumerate(split_caches):
            request = continuing_by_row.get(row_idx)
            if request is None:
                release = getattr(cache, "release", None)
                if callable(release):
                    release()
                continue
            request.past_key_values = cache

        if keep_batched_cache and len(next_active) > 1:
            return [self._make_decode_group_from_requests(next_active)]
        return next_active

    def _iter_decode_chunks(self, requests: list[SchedulerRequest]) -> Iterator[list[SchedulerRequest]]:
        chunk: list[SchedulerRequest] = []
        chunk_tokens = 0
        for request in requests:
            cost = self._decode_request_token_cost(request)
            batch_full = len(chunk) >= self.max_batch_size
            token_full = self.max_decode_tokens > 0 and chunk and chunk_tokens + cost > self.max_decode_tokens
            if batch_full or token_full:
                yield chunk
                chunk = []
                chunk_tokens = 0
            chunk.append(request)
            chunk_tokens += cost
        if chunk:
            yield chunk

    @staticmethod
    def _decode_request_token_cost(request: SchedulerRequest) -> int:
        cache = request.past_key_values
        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return max(1, int(get_seq_length()))
            except Exception:
                pass
        input_ids = request.input_ids
        return 1 if input_ids is None else max(1, int(input_ids.numel()))

    @staticmethod
    def _decode_group_token_cost(group: SchedulerDecodeGroup) -> int:
        cache = group.past_key_values
        get_seq_lengths = getattr(cache, "get_seq_lengths", None)
        if callable(get_seq_lengths):
            try:
                lengths = get_seq_lengths()
                return sum(max(1, int(length)) for length in lengths.tolist())
            except Exception:
                pass
        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return max(1, int(get_seq_length())) * len(group.requests)
            except Exception:
                pass
        return max(1, int(group.input_ids.numel()))

    def _compact_cache_rows(self, cache: object, row_indices: list[int]) -> object | None:
        compact = getattr(cache, "compact_batch_rows", None)
        if not callable(compact):
            return None
        metrics = getattr(self.engine, "metrics", None)
        started_at = time.perf_counter()
        compacted = compact(row_indices, release_unselected=True)
        if metrics is not None:
            metrics.record_cache_compact(time.perf_counter() - started_at)
        return compacted

    def _make_decode_group_from_requests(self, requests: list[SchedulerRequest]) -> SchedulerDecodeGroup:
        if len(requests) <= 1:
            raise ValueError("A decode group requires at least two requests.")
        input_rows = [request.input_ids for request in requests if request.input_ids is not None]
        caches = [request.past_key_values for request in requests if request.past_key_values is not None]
        if len(input_rows) != len(requests) or len(caches) != len(requests):
            raise RuntimeError("Cannot create decode group from requests with missing input ids or caches.")
        input_ids = torch.cat(input_rows, dim=0)
        cache_type = type(caches[0])
        stack = getattr(cache_type, "stack", None)
        if not callable(stack):
            raise RuntimeError(f"Cache type {cache_type.__name__} does not support scheduler batching.")
        metrics = getattr(self.engine, "metrics", None)
        stack_started_at = time.perf_counter()
        stack_kwargs = {"clone_turboquant_rows": False} if cache_type.__name__ == "Qwen3DynamicCache" else {}
        batch_cache = stack(caches, self.engine.config.text_config, **stack_kwargs)
        if metrics is not None:
            metrics.record_cache_stack(time.perf_counter() - stack_started_at)
        for request in requests:
            request.past_key_values = None
        return SchedulerDecodeGroup(requests=list(requests), input_ids=input_ids, past_key_values=batch_cache)

    @staticmethod
    def _make_decode_group_from_batched_cache(
        requests: list[SchedulerRequest],
        batch_cache: object,
    ) -> SchedulerDecodeGroup:
        if len(requests) <= 1:
            raise ValueError("A decode group requires at least two requests.")
        input_rows = [request.input_ids for request in requests if request.input_ids is not None]
        if len(input_rows) != len(requests):
            raise RuntimeError("Cannot create decode group from requests with missing input ids.")
        for request in requests:
            request.past_key_values = None
        return SchedulerDecodeGroup(
            requests=list(requests),
            input_ids=torch.cat(input_rows, dim=0),
            past_key_values=batch_cache,
        )

    def _compact_decode_group_for_membership_change(self, group: SchedulerDecodeGroup) -> list[SchedulerRequest]:
        remaining: list[SchedulerRequest] = []
        selected_rows: list[int] = []
        for row_idx, request in enumerate(group.requests):
            if self._is_request_cancelled(request):
                self._drop_cancelled_request(request)
                continue
            selected_rows.append(row_idx)
            remaining.append(request)

        if not remaining:
            self._release_decode_group_cache(group)
            return []

        compacted_cache = self._compact_cache_rows(group.past_key_values, selected_rows)
        if compacted_cache is not None:
            if len(remaining) == 1:
                remaining[0].past_key_values = compacted_cache
            else:
                for request in remaining:
                    request.past_key_values = None
                group.past_key_values = compacted_cache
            return remaining

        split_caches = self._split_cache(
            group.past_key_values,
            len(group.requests),
            avoid_turboquant_clone=True,
        )
        remaining = []
        for request, cache in zip(group.requests, split_caches):
            if self._is_request_cancelled(request):
                release = getattr(cache, "release", None)
                if callable(release):
                    release()
                continue
            request.past_key_values = cache
            remaining.append(request)
        return remaining

    @staticmethod
    def _release_decode_group_cache(group: SchedulerDecodeGroup) -> None:
        release = getattr(group.past_key_values, "release", None)
        if callable(release):
            release()

    def _shared_fused_lm_head_candidate_count(self, requests: list[SchedulerRequest]) -> int | None:
        candidate_counts = [self.engine._fused_lm_head_candidate_count(request.config) for request in requests]
        first = candidate_counts[0] if candidate_counts else None
        if first is None:
            return None
        if any(candidate != first for candidate in candidate_counts):
            return None
        return int(first)

    def _forward_batch_maybe_topk(
        self,
        *,
        stage: str,
        requests: list[SchedulerRequest],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        model_kwargs: dict[str, object] | None = None,
    ):
        candidate_count = self._shared_fused_lm_head_candidate_count(requests)
        outputs = (
            self.engine._forward_generation_model_topk(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                model_kwargs=model_kwargs,
                use_cache=True,
                logits_to_keep=1,
                top_k=candidate_count,
            )
            if candidate_count is not None
            else None
        )
        if outputs is not None:
            return outputs
        return self.engine._profiled_forward_generation_model(
            stage=stage,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            model_kwargs=model_kwargs,
            use_cache=True,
            logits_to_keep=1,
        )

    def _sample_next_token_from_outputs(self, outputs, *, row_idx: int, request: SchedulerRequest) -> torch.Tensor:
        if hasattr(outputs, "candidate_logits") and hasattr(outputs, "candidate_token_ids"):
            return sample_next_token_from_candidates(
                outputs.candidate_logits[row_idx, -1],
                outputs.candidate_token_ids[row_idx, -1],
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                min_p=request.config.min_p,
            )
        return sample_next_token(
            outputs.logits[row_idx, -1],
            generated_ids=request.repetition_history,
            temperature=request.config.temperature,
            top_p=request.config.top_p,
            top_k=request.config.top_k,
            min_p=request.config.min_p,
            presence_penalty=request.config.presence_penalty,
            repetition_penalty=request.config.repetition_penalty,
        )

    def _batch_text_inputs(self, requests: list[SchedulerRequest]) -> PreparedInputsLike:
        max_prompt_length = max(request.prompt_length for request in requests)
        text_config = self.engine.config.text_config
        pad_token_id = text_config.pad_token_id if 0 <= text_config.pad_token_id < text_config.vocab_size else 0
        input_device = requests[0].prepared.input_ids.device
        prepared_type = type(requests[0].prepared)
        input_ids = torch.full((len(requests), max_prompt_length), pad_token_id, dtype=torch.long, device=input_device)
        attention_mask = torch.zeros((len(requests), max_prompt_length), dtype=torch.long, device=input_device)
        mm_token_type_ids = torch.zeros((len(requests), max_prompt_length), dtype=torch.int32, device=input_device)

        for batch_idx, request in enumerate(requests):
            length = request.prompt_length
            input_ids[batch_idx, :length] = request.prepared.input_ids[0, :length]
            attention_mask[batch_idx, :length] = request.prepared.attention_mask[0, :length]
            mm_token_type_ids[batch_idx, :length] = request.prepared.mm_token_type_ids[0, :length]

        return prepared_type(
            prompt="",
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

    def _guard_batch_memory(self, requests: list[SchedulerRequest]) -> None:
        memory_info = self.engine.device_context.get_memory_info()
        if memory_info is None:
            return

        estimated_bytes = sum(
            self.engine._estimate_generation_memory_bytes(request.prepared, config=request.config)
            for request in requests
        )
        policy = self.engine.device_context.safety_policy
        available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
        max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)

        if (
            memory_info.free_bytes < policy.min_free_bytes
            or estimated_bytes > available_budget
            or estimated_bytes > max_allowed
        ):
            reclaim = getattr(self.engine, "_reclaim_runtime_memory_for_admission", None)
            if callable(reclaim) and reclaim():
                memory_info = self.engine.device_context.get_memory_info()
                if memory_info is None:
                    return
                available_budget = max(0, memory_info.free_bytes - policy.reserve_margin_bytes)
                max_allowed = int(memory_info.total_bytes * policy.max_estimated_usage_ratio)

        if memory_info.free_bytes < policy.min_free_bytes:
            raise self.engine._handle_runtime_failure(RuntimeError("out of memory"))
        if estimated_bytes > available_budget or estimated_bytes > max_allowed:
            from anna.runtime.qwen3_5_text_engine import AnnaEngineError

            raise AnnaEngineError(
                "Scheduler batch rejected by memory guard. Reduce prompt length, batch size, or max_completion_tokens.",
                status_code=400,
                error_type="invalid_request_error",
                code="estimated_device_oom",
            )

    def _emit_text(self, request: SchedulerRequest, text: str) -> None:
        if self._is_request_cancelled(request):
            return
        request.text_parts.append(text)
        if request.stream:
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            request.events.put(StreamEvent(text=text, finish_reason=None))

    def _finish_request(self, request: SchedulerRequest, *, finish_reason: str) -> None:
        if self._is_request_cancelled(request):
            self._release_request_cache(request)
            return
        metrics = getattr(self.engine, "metrics", None)
        if request.assembler is not None:
            tail, _ = request.assembler.flush()
            if tail:
                self._emit_text(request, tail)
        if request.past_key_values is not None:
            request.past_key_values.release()
            request.past_key_values = None

        perf = self._build_perf_stats(request)
        if request.stream:
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            request.events.put(
                StreamEvent(
                    text="",
                    finish_reason=finish_reason,
                    prompt_tokens=request.prompt_length,
                    completion_tokens=len(request.completion_ids),
                    perf=perf,
                )
            )
            request.events.put(_DONE)
            request.done.set()
            if metrics is not None:
                metrics.record_request_finished(success=True)
            self.engine._trim_runtime_cache_if_idle()
            return

        from anna.runtime.qwen3_5_text_engine import TextGenerationResult

        request.result = TextGenerationResult(
            text="".join(request.text_parts),
            finish_reason=finish_reason,
            prompt_tokens=request.prompt_length,
            completion_tokens=len(request.completion_ids),
            perf=perf,
        )
        request.done.set()
        if metrics is not None:
            metrics.record_request_finished(success=True)
        self.engine._trim_runtime_cache_if_idle()

    def _release_request_cache(self, request: SchedulerRequest) -> None:
        if request.past_key_values is not None:
            request.past_key_values.release()
            request.past_key_values = None

    def _drop_cancelled_request(self, request: SchedulerRequest) -> bool:
        if not self._is_request_cancelled(request):
            return False
        request.cancelled = True
        self._release_request_cache(request)
        if not request.done.is_set():
            request.error = self._cancelled_error()
            request.events.put(_DONE)
            request.done.set()
            metrics = getattr(self.engine, "metrics", None)
            if metrics is not None:
                metrics.record_request_finished(success=False)
        return True

    def _filter_prefill_group(self, group: SchedulerPrefillGroup) -> bool:
        group.requests = [request for request in group.requests if not self._drop_cancelled_request(request)]
        if group.requests:
            return True
        release = getattr(group.past_key_values, "release", None)
        if callable(release):
            release()
        group.past_key_values = None
        return False

    def _is_request_cancelled(self, request: SchedulerRequest) -> bool:
        return request.cancelled or (
            request.config.cancellation_event is not None and request.config.cancellation_event.is_set()
        )

    def _cancelled_error(self) -> "AnnaEngineError":
        from anna.runtime.qwen3_5_text_engine import AnnaEngineError

        return AnnaEngineError(
            "Generation cancelled because the client disconnected.",
            status_code=499,
            error_type="server_error",
            code="client_disconnected",
        )

    def _build_perf_stats(self, request: SchedulerRequest):
        started_at = request.generation_started_at
        if started_at is None:
            return None
        finished_at = time.perf_counter()
        first_token_at = request.first_token_at if request.first_token_at is not None else finished_at
        total_seconds = max(0.0, finished_at - started_at)
        prefill_seconds = max(0.0, first_token_at - started_at)
        decode_seconds = max(0.0, finished_at - first_token_at)
        return self.engine._build_generation_perf_stats(
            prompt_tokens=request.prompt_length,
            completion_tokens=len(request.completion_ids),
            total_seconds=total_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
        )

    def _normalize_error(self, exc: Exception) -> "AnnaEngineError":
        from anna.runtime.qwen3_5_text_engine import AnnaEngineError

        if isinstance(exc, AnnaEngineError):
            return exc
        if isinstance(exc, RuntimeError):
            return self.engine._handle_runtime_failure(exc)
        return AnnaEngineError(str(exc), status_code=500, error_type="server_error", code="scheduler_failed")

    def _normalize_worker_crash(self, exc: BaseException) -> "AnnaEngineError":
        from anna.runtime.qwen3_5_text_engine import AnnaEngineError

        if isinstance(exc, Exception):
            return self._normalize_error(exc)
        return AnnaEngineError(
            f"Scheduler worker crashed: {exc}",
            status_code=500,
            error_type="server_error",
            code="scheduler_worker_failed",
        )

    def _fail_requests(self, requests: list[SchedulerRequest], exc: "AnnaEngineError") -> None:
        for request in requests:
            self._fail_request(request, exc)

    def _fail_request(self, request: SchedulerRequest, exc: Exception) -> None:
        from anna.runtime.qwen3_5_text_engine import AnnaEngineError

        normalized = exc if isinstance(exc, AnnaEngineError) else self._normalize_error(exc)
        metrics = getattr(self.engine, "metrics", None)
        request.error = normalized
        self._release_request_cache(request)
        if request.stream:
            request.events.put(normalized)
            request.events.put(_DONE)
        request.done.set()
        if metrics is not None:
            metrics.record_request_finished(success=False)
        self.engine._trim_runtime_cache_if_idle()
