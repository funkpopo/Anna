from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import torch

from anna.mm.processor import PreparedInputs
from anna.model.ops import Qwen3DynamicCache
from anna.runtime.streaming import IncrementalTextAssembler
from anna.sampling.sampler import sample_next_token

if TYPE_CHECKING:
    from anna.runtime.engine import AnnaEngine, AnnaEngineError, GenerationConfig, StreamEvent, TextGenerationResult


_DONE = object()


@dataclass(slots=True)
class SchedulerRequest:
    prepared: PreparedInputs
    config: "GenerationConfig"
    stream: bool
    prompt_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0
    completion_ids: list[int] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    input_ids: torch.Tensor | None = None
    past_key_values: Qwen3DynamicCache | None = None
    repetition_history: torch.Tensor | None = None
    repetition_history_ids: set[int] | None = None
    assembler: IncrementalTextAssembler | None = None
    result: "TextGenerationResult | None" = None
    error: "AnnaEngineError | None" = None
    done: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue[object] = field(default_factory=queue.Queue)
    generation_started_at: float | None = None
    first_token_at: float | None = None


class AnnaScheduler:
    def __init__(
        self,
        engine: "AnnaEngine",
        *,
        max_batch_size: int = 4,
        batch_wait_ms: float = 2.0,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_seconds = max(0.0, batch_wait_ms) / 1000.0
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

    def generate(self, prepared: PreparedInputs, *, config: "GenerationConfig") -> "TextGenerationResult":
        request = self._submit(prepared, config=config, stream=False)
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("Scheduler request completed without a result.")
        return request.result

    def stream(self, prepared: PreparedInputs, *, config: "GenerationConfig") -> Iterator["StreamEvent"]:
        request = self._submit(prepared, config=config, stream=True)
        while True:
            item = request.events.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _submit(self, prepared: PreparedInputs, *, config: "GenerationConfig", stream: bool) -> SchedulerRequest:
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
        active: list[SchedulerRequest] = []
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
                    while self._pending and len(pending_batch) < self.max_batch_size:
                        pending_batch.append(self._pending.popleft())

                if pending_batch:
                    try:
                        active.extend(self._prefill_batch(pending_batch))
                    except Exception as exc:  # pragma: no cover - worker-level best effort
                        self._fail_requests(pending_batch, self._normalize_error(exc))

                if not active:
                    continue

                next_active: list[SchedulerRequest] = []
                ready = [request for request in active if request.past_key_values is not None]
                for request in active:
                    if request.past_key_values is None:
                        self._fail_request(request, RuntimeError("Missing decode cache for active request."))

                for chunk_start in range(0, len(ready), self.max_batch_size):
                    chunk = ready[chunk_start : chunk_start + self.max_batch_size]
                    try:
                        next_active.extend(self._decode_batch(chunk))
                    except Exception as exc:  # pragma: no cover - worker-level best effort
                        self._fail_requests(chunk, self._normalize_error(exc))

                active = next_active
        except BaseException as exc:  # pragma: no cover - catastrophic worker failure
            pending: list[SchedulerRequest] = []
            with self._condition:
                while self._pending:
                    pending.append(self._pending.popleft())
            fatal = self._normalize_worker_crash(exc)
            self._fatal_error = fatal
            self._fail_requests(active + pending, fatal)

    def _prefill_batch(self, requests: list[SchedulerRequest]) -> list[SchedulerRequest]:
        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None:
            metrics.record_requests_started_from_queue(len(requests))
        for request in requests:
            request.generation_started_at = time.perf_counter()
            request.prompt_ids, request.prompt_length, request.config = self.engine._validate_generation_request(
                request.prepared,
                config=request.config,
            )
            request.repetition_history, request.repetition_history_ids = self.engine._init_repetition_penalty_state(
                request.prompt_ids,
                request.config.repetition_penalty,
            )
            request.assembler = IncrementalTextAssembler(
                tokenizer=self.engine.tokenizer,
                stop_strings=request.config.stop_strings,
            )

        active: list[SchedulerRequest] = []
        groups: dict[int, list[SchedulerRequest]] = defaultdict(list)
        for request in requests:
            groups[request.prompt_length].append(request)

        for _, group in sorted(groups.items()):
            active.extend(self._prefill_same_length_group(group))
        return active

    def _prefill_same_length_group(self, requests: list[SchedulerRequest]) -> list[SchedulerRequest]:
        batched = self._batch_text_inputs(requests)
        self._guard_batch_memory(requests)
        prompt_length = int(batched.input_ids.shape[1])
        configured_chunk_size = int(getattr(self.engine.optimization_config, "prefill_chunk_size", 0))
        outputs = None
        past_key_values = None
        prompt_tokens_recorded = 0
        metrics = getattr(self.engine, "metrics", None)

        try:
            if configured_chunk_size > 0 and prompt_length > configured_chunk_size:
                past_key_values = self.engine._reserve_prefill_cache(batched)
                for start_idx in range(0, prompt_length, configured_chunk_size):
                    end_idx = min(prompt_length, start_idx + configured_chunk_size)
                    chunk = PreparedInputs(
                        prompt="",
                        input_ids=batched.input_ids[:, start_idx:end_idx],
                        attention_mask=batched.attention_mask[:, :end_idx] if start_idx == 0 else None,
                        mm_token_type_ids=batched.mm_token_type_ids[:, start_idx:end_idx],
                    )
                    chunk = self.engine.device_context.move_prepared_inputs(chunk)
                    with self.engine.execution_lock:
                        outputs = self.engine._profiled_forward_generation_model(
                            stage=f"scheduler_prefill[{start_idx}:{end_idx}]",
                            input_ids=chunk.input_ids,
                            attention_mask=chunk.attention_mask,
                            past_key_values=past_key_values,
                            mm_token_type_ids=chunk.mm_token_type_ids,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                    past_key_values = outputs.past_key_values
                    if metrics is not None:
                        chunk_tokens = (end_idx - start_idx) * len(requests)
                        if chunk_tokens > 0:
                            metrics.record_prompt_tokens(chunk_tokens)
                            prompt_tokens_recorded += chunk_tokens
            else:
                batched = self.engine.device_context.move_prepared_inputs(batched)
                with self.engine.execution_lock:
                    outputs = self.engine._profiled_forward_generation_model(
                        stage="scheduler_prefill",
                        input_ids=batched.input_ids,
                        attention_mask=batched.attention_mask,
                        mm_token_type_ids=batched.mm_token_type_ids,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past_key_values = outputs.past_key_values
        except RuntimeError as exc:
            release = getattr(past_key_values, "release", None)
            if callable(release):
                release()
            raise self.engine._handle_runtime_failure(exc) from exc

        if outputs is None:
            raise RuntimeError("Scheduler prefill did not produce model outputs.")

        total_prompt_tokens = sum(request.prompt_length for request in requests)
        if metrics is not None and prompt_tokens_recorded < total_prompt_tokens:
            metrics.record_prompt_tokens(total_prompt_tokens - prompt_tokens_recorded)
        split_caches = outputs.past_key_values.split_batch() if outputs.past_key_values is not None else [None] * len(requests)
        stop_token_ids = set(self.engine.tokenizer.eos_token_ids)
        active: list[SchedulerRequest] = []
        for row_idx, request in enumerate(requests):
            request.past_key_values = split_caches[row_idx]
            next_token = sample_next_token(
                outputs.logits[row_idx, -1],
                generated_ids=request.repetition_history,
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                top_k=request.config.top_k,
                repetition_penalty=request.config.repetition_penalty,
            )
            token_id = int(next_token.item())
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
            active.append(request)
        return active

    def _decode_batch(self, requests: list[SchedulerRequest]) -> list[SchedulerRequest]:
        input_ids = torch.cat([request.input_ids for request in requests if request.input_ids is not None], dim=0)
        batch_cache = Qwen3DynamicCache.stack(
            [request.past_key_values for request in requests if request.past_key_values is not None],
            self.engine.config.text_config,
        )

        try:
            with self.engine.execution_lock:
                outputs = self.engine._profiled_forward_generation_model(
                    stage="scheduler_decode",
                    input_ids=input_ids,
                    past_key_values=batch_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
        except RuntimeError as exc:
            raise self.engine._handle_runtime_failure(exc) from exc

        split_caches = outputs.past_key_values.split_batch() if outputs.past_key_values is not None else [None] * len(requests)
        next_active: list[SchedulerRequest] = []
        stop_token_ids = set(self.engine.tokenizer.eos_token_ids)
        metrics = getattr(self.engine, "metrics", None)
        for row_idx, request in enumerate(requests):
            request.past_key_values = split_caches[row_idx]
            next_token = sample_next_token(
                outputs.logits[row_idx, -1],
                generated_ids=request.repetition_history,
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                top_k=request.config.top_k,
                repetition_penalty=request.config.repetition_penalty,
            )
            token_id = int(next_token.item())
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
            next_active.append(request)
        return next_active

    def _batch_text_inputs(self, requests: list[SchedulerRequest]) -> PreparedInputs:
        max_prompt_length = max(request.prompt_length for request in requests)
        text_config = self.engine.config.text_config
        pad_token_id = text_config.pad_token_id if 0 <= text_config.pad_token_id < text_config.vocab_size else 0
        input_device = requests[0].prepared.input_ids.device
        input_ids = torch.full((len(requests), max_prompt_length), pad_token_id, dtype=torch.long, device=input_device)
        attention_mask = torch.zeros((len(requests), max_prompt_length), dtype=torch.long, device=input_device)
        mm_token_type_ids = torch.zeros((len(requests), max_prompt_length), dtype=torch.int32, device=input_device)

        for batch_idx, request in enumerate(requests):
            length = request.prompt_length
            input_ids[batch_idx, :length] = request.prepared.input_ids[0, :length]
            attention_mask[batch_idx, :length] = request.prepared.attention_mask[0, :length]
            mm_token_type_ids[batch_idx, :length] = request.prepared.mm_token_type_ids[0, :length]

        return PreparedInputs(
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

        if memory_info.free_bytes < policy.min_free_bytes:
            raise self.engine._handle_runtime_failure(RuntimeError("out of memory"))
        if estimated_bytes > available_budget or estimated_bytes > max_allowed:
            from anna.runtime.engine import AnnaEngineError

            raise AnnaEngineError(
                "Scheduler batch rejected by memory guard. Reduce prompt length, batch size, or max_completion_tokens.",
                status_code=400,
                error_type="invalid_request_error",
                code="estimated_device_oom",
            )

    def _emit_text(self, request: SchedulerRequest, text: str) -> None:
        request.text_parts.append(text)
        if request.stream:
            from anna.runtime.engine import StreamEvent

            request.events.put(StreamEvent(text=text, finish_reason=None))

    def _finish_request(self, request: SchedulerRequest, *, finish_reason: str) -> None:
        metrics = getattr(self.engine, "metrics", None)
        if request.assembler is not None:
            tail, _ = request.assembler.flush()
            if tail:
                self._emit_text(request, tail)
        if request.past_key_values is not None:
            request.past_key_values.release()
            request.past_key_values = None

        if request.stream:
            from anna.runtime.engine import StreamEvent

            request.events.put(StreamEvent(text="", finish_reason=finish_reason))
            request.events.put(_DONE)
            request.done.set()
            if metrics is not None:
                metrics.record_request_finished(success=True)
            self.engine._trim_runtime_cache_if_idle()
            return

        from anna.runtime.engine import TextGenerationResult

        request.result = TextGenerationResult(
            text="".join(request.text_parts),
            finish_reason=finish_reason,
            prompt_tokens=request.prompt_length,
            completion_tokens=len(request.completion_ids),
            perf=self._build_perf_stats(request),
        )
        request.done.set()
        if metrics is not None:
            metrics.record_request_finished(success=True)
        self.engine._trim_runtime_cache_if_idle()

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
        from anna.runtime.engine import AnnaEngineError

        if isinstance(exc, AnnaEngineError):
            return exc
        if isinstance(exc, RuntimeError):
            return self.engine._handle_runtime_failure(exc)
        return AnnaEngineError(str(exc), status_code=500, error_type="server_error", code="scheduler_failed")

    def _normalize_worker_crash(self, exc: BaseException) -> "AnnaEngineError":
        from anna.runtime.engine import AnnaEngineError

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
        from anna.runtime.engine import AnnaEngineError

        normalized = exc if isinstance(exc, AnnaEngineError) else self._normalize_error(exc)
        metrics = getattr(self.engine, "metrics", None)
        request.error = normalized
        if request.past_key_values is not None:
            request.past_key_values.release()
            request.past_key_values = None
        if request.stream:
            request.events.put(normalized)
            request.events.put(_DONE)
        request.done.set()
        if metrics is not None:
            metrics.record_request_finished(success=False)
        self.engine._trim_runtime_cache_if_idle()
