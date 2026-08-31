from __future__ import annotations

import torch

from anna.mm.prepared_inputs import PreparedInputs
from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig, Qwen3_5TextConfig
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator
from anna.runtime.device import RuntimeSafetyPolicy
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, EngineOptimizationConfig, GenerationConfig
from anna.runtime.scheduler import AnnaScheduler, SchedulerRequest


class _FakeTokenizer:
    def __init__(self) -> None:
        self._pieces = {
            1: "A",
            2: "B",
            9: "",
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)

    @property
    def eos_token_ids(self) -> set[int]:
        return {9}


class _FakeDeviceContext:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.safety_policy = RuntimeSafetyPolicy()

    def get_memory_info(self):
        return None

    def element_size(self, dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def move_prepared_inputs(self, prepared: PreparedInputs) -> PreparedInputs:
        return prepared

    def move_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids


class _FakeLMHead:
    def __init__(self, owner: "_FakeModel") -> None:
        self.owner = owner

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        self.owner.prefill_batch_sizes.append(batch_size)
        logits = torch.full((batch_size, self.owner.config.text_config.vocab_size), -1000.0)
        planned = [1, 2]
        for idx in range(batch_size):
            logits[idx, planned[idx]] = 1000.0
        return logits


class _FakePrefillRunner:
    def __init__(self, owner: "_FakeModel") -> None:
        self.owner = owner

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros((batch_size, seq_len, self.owner.config.text_config.hidden_size))
        return type(
            "PrefillOutput",
            (),
            {
                "last_hidden_state": hidden,
                "past_key_values": self.owner._make_cache(batch_size=batch_size, seq_len=seq_len),
            },
        )()


class _FakeModel:
    def __init__(self, config: Qwen3_5TextModelConfig) -> None:
        self.config = config
        self.cache_allocator = Qwen3PageAllocator(config.text_config)
        self.prefill_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []
        self.text_prefill_batch_sizes: list[int] = []
        self.text_decode_batch_sizes: list[int] = []
        self.text_prefill_topk_batch_sizes: list[int] = []
        self.text_decode_topk_batch_sizes: list[int] = []
        self.text_prefill_chunk_lengths: list[int] = []
        self.decode_non_eos_steps = 0
        self.model = _FakePrefillRunner(self)
        self.lm_head = _FakeLMHead(self)

    def _make_cache(self, *, batch_size: int, seq_len: int) -> Qwen3DynamicCache:
        cache = Qwen3DynamicCache(self.config.text_config, allocator=self.cache_allocator, batch_size=batch_size)
        key = torch.zeros((batch_size, self.config.text_config.num_key_value_heads, seq_len, self.config.text_config.head_dim))
        value = torch.zeros_like(key)
        cache.update(key, value, layer_idx=0)
        return cache

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Qwen3DynamicCache | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
        **_: object,
    ):
        batch_size = input_ids.shape[0]
        self.decode_batch_sizes.append(batch_size)
        logits = torch.full((batch_size, 1, self.config.text_config.vocab_size), -1000.0)
        logits[:, 0, 9] = 1000.0
        return type(
            "DecodeOutput",
            (),
            {
                "logits": logits,
                "past_key_values": past_key_values if past_key_values is not None else self._make_cache(batch_size=batch_size, seq_len=1),
            },
        )()

    def forward_text_only(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ):
        del attention_mask, use_cache, logits_to_keep
        batch_size = input_ids.shape[0]
        if past_key_values is None or input_ids.shape[1] > 1:
            seq_len = input_ids.shape[1]
            self.text_prefill_batch_sizes.append(batch_size)
            self.text_prefill_chunk_lengths.append(seq_len)
            logits = torch.full((batch_size, 1, self.config.text_config.vocab_size), -1000.0)
            planned = [1, 2]
            for idx in range(batch_size):
                logits[idx, 0, planned[idx]] = 1000.0
            return type(
                "TextPrefillOutput",
                (),
                {
                    "logits": logits,
                    "past_key_values": self._make_cache(batch_size=batch_size, seq_len=seq_len),
                },
            )()

        self.text_decode_batch_sizes.append(batch_size)
        logits = torch.full((batch_size, 1, self.config.text_config.vocab_size), -1000.0)
        if len(self.text_decode_batch_sizes) <= self.decode_non_eos_steps:
            planned = [1, 2]
            for idx in range(batch_size):
                logits[idx, 0, planned[idx]] = 1000.0
        else:
            logits[:, 0, 9] = 1000.0
        return type(
            "TextDecodeOutput",
            (),
            {
                "logits": logits,
                "past_key_values": past_key_values,
            },
        )()

    def forward_text_only_topk(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
        top_k: int = 1,
    ):
        del attention_mask, use_cache, logits_to_keep
        batch_size = input_ids.shape[0]
        candidate_logits = torch.full((batch_size, 1, top_k), -1000.0)
        candidate_token_ids = torch.zeros((batch_size, 1, top_k), dtype=torch.long)
        if past_key_values is None or input_ids.shape[1] > 1:
            seq_len = input_ids.shape[1]
            self.text_prefill_topk_batch_sizes.append(batch_size)
            planned = [1, 2]
            for idx in range(batch_size):
                candidate_logits[idx, 0, 0] = 1000.0
                candidate_token_ids[idx, 0, 0] = planned[idx]
            cache = self._make_cache(batch_size=batch_size, seq_len=seq_len)
        else:
            self.text_decode_topk_batch_sizes.append(batch_size)
            candidate_logits[:, 0, 0] = 1000.0
            candidate_token_ids[:, 0, 0] = 9
            cache = past_key_values
        return type(
            "TextTopKOutput",
            (),
            {
                "candidate_logits": candidate_logits,
                "candidate_token_ids": candidate_token_ids,
                "past_key_values": cache,
            },
        )()


def _prepared(prompt_tokens: list[int], *, multimodal: bool = False) -> PreparedInputs:
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    return PreparedInputs(
        prompt="",
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        mm_token_type_ids=torch.zeros_like(input_ids, dtype=torch.int32),
        pixel_values=torch.randn(4, 8) if multimodal else None,
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long) if multimodal else None,
    )


def test_scheduler_batches_same_length_requests() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(_prepared([4, 5]), config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1), stream=False)
        request_b = scheduler._submit(_prepared([6, 7]), config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1), stream=False)

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert request_a.result is not None
        assert request_b.result is not None
        assert request_a.result.text == "A"
        assert request_b.result.text == "B"
        assert fake_model.text_prefill_batch_sizes == [2]
        assert fake_model.text_decode_batch_sizes == [2]
        assert fake_model.prefill_batch_sizes == []
        assert fake_model.decode_batch_sizes == []
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.requests_started_total == 2
        assert snapshot.requests_completed_total == 2
        assert snapshot.requests_failed_total == 0
        assert snapshot.prompt_tokens_total == 4
        assert snapshot.generation_tokens_total == 2
        assert snapshot.running_requests == 0
        assert snapshot.waiting_requests == 0
    finally:
        scheduler.shutdown()


def test_scheduler_chunks_long_same_length_prefills() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
        optimization_config=EngineOptimizationConfig(prefill_chunk_size=2),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5, 6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([8, 9, 10, 11]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert fake_model.text_prefill_batch_sizes == [2, 2]
        assert fake_model.text_prefill_chunk_lengths == [2, 2]
        assert fake_model.text_decode_batch_sizes == [2]
    finally:
        scheduler.shutdown()


def test_scheduler_batches_mixed_length_requests_during_decode() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7, 8]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert fake_model.text_prefill_batch_sizes == [1, 1]
        assert fake_model.text_decode_batch_sizes == [2]
        assert fake_model.prefill_batch_sizes == []
        assert fake_model.decode_batch_sizes == []
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.requests_started_total == 2
        assert snapshot.requests_completed_total == 2
        assert snapshot.requests_failed_total == 0
        assert snapshot.prompt_tokens_total == 5
        assert snapshot.generation_tokens_total == 2
        assert snapshot.running_requests == 0
        assert snapshot.waiting_requests == 0
    finally:
        scheduler.shutdown()


def test_scheduler_uses_topk_forward_for_eligible_batches() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=1, presence_penalty=0.0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=1, presence_penalty=0.0),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert fake_model.text_prefill_topk_batch_sizes == [2]
        assert fake_model.text_decode_topk_batch_sizes == [2]
        assert fake_model.text_prefill_batch_sizes == []
        assert fake_model.text_decode_batch_sizes == []
    finally:
        scheduler.shutdown()


def test_scheduler_keeps_stable_decode_batches_without_cache_resplit() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    fake_model.decode_non_eos_steps = 2
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert request_a.result is not None
        assert request_b.result is not None
        assert request_a.result.completion_tokens == 3
        assert request_b.result.completion_tokens == 3
        assert fake_model.text_prefill_batch_sizes == [2]
        assert fake_model.text_decode_batch_sizes == [2, 2, 2]
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
        assert snapshot.scheduler_decode_batch_count == 3
        assert snapshot.scheduler_decode_batch_requests_total == 6
        assert snapshot.scheduler_decode_batch_requests_max == 2
    finally:
        scheduler.shutdown()


def test_scheduler_marks_multimodal_requests_and_preserves_media() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=0.0)
    engine.set_scheduler(scheduler)
    try:
        mm_prepared = _prepared([4, 5], multimodal=True)
        text_prepared = _prepared([6, 7])
        mm_request = SchedulerRequest(prepared=mm_prepared, config=GenerationConfig(max_new_tokens=1), stream=False, is_multimodal=True)
        text_request = SchedulerRequest(prepared=text_prepared, config=GenerationConfig(max_new_tokens=1), stream=False)
        mm_request.prompt_length = 2
        text_request.prompt_length = 2

        batched_mm = scheduler._batch_text_inputs([mm_request])
        assert batched_mm.pixel_values is not None
        assert batched_mm.image_grid_thw is not None

        accepted, deferred = scheduler._select_prefill_admission([mm_request, text_request, SchedulerRequest(
            prepared=_prepared([8, 9], multimodal=True),
            config=GenerationConfig(max_new_tokens=1),
            stream=False,
            is_multimodal=True,
            prompt_length=2,
        )])
        assert len(accepted) == 2  # one multimodal + text
        assert sum(1 for req in accepted if req.is_multimodal) == 1
        assert len(deferred) == 1
        assert deferred[0].is_multimodal
    finally:
        scheduler.shutdown()


def test_scheduler_defers_multimodal_prefill_while_text_decode_is_hot() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(
        engine,
        max_batch_size=4,
        batch_wait_ms=0.0,
        prefill_interval_steps=1,
        max_queue_wait_ms=0.0,
    )
    engine.set_scheduler(scheduler)
    try:
        text_decode = SchedulerRequest(
            prepared=_prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=4),
            stream=False,
            prompt_length=2,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            past_key_values=object(),
        )
        mm_pending = SchedulerRequest(
            prepared=_prepared([6, 7], multimodal=True),
            config=GenerationConfig(max_new_tokens=1),
            stream=False,
            is_multimodal=True,
            prompt_length=2,
        )
        with scheduler._condition:
            scheduler._pending.append(mm_pending)
        scheduler._decode_steps_since_prefill = 1
        assert scheduler._should_admit_prefill([text_decode]) is False

        # Fairness timeout can still force VL admission.
        scheduler.max_queue_wait_seconds = 0.0
        mm_pending.queued_at = 0.0
        scheduler.max_queue_wait_seconds = 0.001
        import time as _time

        mm_pending.queued_at = _time.perf_counter() - 1.0
        assert scheduler._should_admit_prefill([text_decode]) is True
    finally:
        scheduler.shutdown()


def test_scheduler_compacts_decode_group_when_some_rows_finish() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    fake_model.decode_non_eos_steps = 1
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert request_a.result is not None
        assert request_b.result is not None
        assert request_a.result.completion_tokens == 2
        assert request_b.result.completion_tokens == 2
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
        assert snapshot.cache_compact_count >= 1
        assert snapshot.scheduler_decode_batch_requests_max == 2
    finally:
        scheduler.shutdown()


def test_scheduler_prefill_admission_respects_token_budget() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0, max_prefill_tokens=4)
    engine.set_scheduler(scheduler)

    try:
        requests = [
            scheduler._submit(
                _prepared([4, 5]),
                config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
                stream=False,
            )
            for _ in range(3)
        ]

        for request in requests:
            assert request.done.wait(timeout=2.0)
            assert request.error is None
        assert fake_model.text_prefill_batch_sizes[:2] == [2, 1]
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.scheduler_prefill_admitted_requests_total == 3
        assert snapshot.scheduler_prefill_deferred_requests_total in (0, 1)
        assert snapshot.scheduler_prefill_admitted_tokens_total == 6
        assert snapshot.scheduler_prefill_admission_count >= 2
        assert snapshot.scheduler_prefill_admitted_tokens_max == 4
    finally:
        scheduler.shutdown()


def test_scheduler_decode_packing_respects_token_budget() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0, max_decode_tokens=4)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7, 8]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert fake_model.text_decode_batch_sizes == [1, 1]
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.scheduler_decode_batch_count == 2
        assert snapshot.scheduler_decode_batch_requests_total == 2
        assert snapshot.scheduler_decode_batch_requests_max == 1
        assert snapshot.scheduler_decode_batch_tokens_total == 5
        assert snapshot.scheduler_decode_batch_tokens_max == 3
    finally:
        scheduler.shutdown()


def test_scheduler_streaming_final_event_includes_usage_stats() -> None:
    config = Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        events = list(
            scheduler.stream(
                _prepared([4, 5]),
                config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.1),
            )
        )

        assert [event.text for event in events] == ["A", ""]
        assert events[-1].finish_reason == "stop"
        assert events[-1].prompt_tokens == 2
        assert events[-1].completion_tokens == 1
        assert events[-1].perf is not None
    finally:
        scheduler.shutdown()


def test_scheduler_samples_decode_batch_in_one_vectorized_pass() -> None:
    """P2 hot-loop: batch sampling returns one device tensor + single host sync path."""
    from types import MethodType

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._token_id_from_tensor = staticmethod(AnnaQwen3_5TextEngine._token_id_from_tensor).__func__  # type: ignore[attr-defined]
    engine._append_repetition_penalty_token = MethodType(
        AnnaQwen3_5TextEngine._append_repetition_penalty_token,
        engine,
    )
    engine._stop_token_ids = MethodType(lambda self: {9}, engine)

    scheduler = object.__new__(AnnaScheduler)
    scheduler.engine = engine

    outputs = type(
        "Outputs",
        (),
        {
            "logits": torch.tensor(
                [
                    [[0.0, 5.0, 1.0, -1.0]],
                    [[3.0, 0.0, 2.0, 1.0]],
                ]
            ),
        },
    )()
    requests = [
        SchedulerRequest(
            prepared=_prepared([1]),
            config=GenerationConfig(
                max_new_tokens=4,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            ),
            stream=False,
            prompt_length=1,
            assembler=None,
        ),
        SchedulerRequest(
            prepared=_prepared([1]),
            config=GenerationConfig(
                max_new_tokens=4,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            ),
            stream=False,
            prompt_length=1,
            assembler=None,
        ),
    ]

    sampled = scheduler._sample_next_tokens_from_outputs(outputs, requests=requests, row_indices=[0, 1])
    assert sampled.tolist() == [1, 0]

    finished: list[str] = []

    def _finish(request, *, finish_reason: str) -> None:
        finished.append(finish_reason)
        request.done.set()

    scheduler._finish_request = _finish  # type: ignore[method-assign]
    scheduler._emit_text = lambda request, text: None  # type: ignore[method-assign]
    scheduler._is_request_cancelled = lambda request: False  # type: ignore[method-assign]
    scheduler._drop_cancelled_request = lambda request: False  # type: ignore[method-assign]
    scheduler._compact_cache_rows = lambda cache, rows: None  # type: ignore[method-assign]
    scheduler._split_cache = lambda cache, n, avoid_turboquant_clone=False: [None] * n  # type: ignore[method-assign]
    scheduler._make_decode_group_from_requests = lambda reqs: reqs  # type: ignore[method-assign]
    # Legacy per-step pull for this test (object.__new__ default is already 1).
    scheduler._deferred_pull_interval = 1
    scheduler._deferred_pull_counter = 0

    active = scheduler._consume_batch_outputs(
        requests,
        outputs,
        batch_cache=None,
        keep_batched_cache=False,
        avoid_turboquant_clone=True,
    )

    assert [request.completion_ids[-1] for request in active] == [1, 0]
    assert all(request.input_ids is not None and request.input_ids.shape == (1, 1) for request in active)
    assert finished == []


def test_scheduler_defers_non_streaming_token_pull_until_interval() -> None:
    """P0-#12: non-streaming rows keep tokens on device and flush in one bulk pull."""
    from types import MethodType

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._append_repetition_penalty_token_device = staticmethod(
        AnnaQwen3_5TextEngine._append_repetition_penalty_token_device
    ).__func__  # type: ignore[attr-defined]
    engine._stop_token_ids = MethodType(lambda self: {9}, engine)

    scheduler = object.__new__(AnnaScheduler)
    scheduler.engine = engine
    scheduler._deferred_pull_interval = 2
    scheduler._deferred_pull_counter = 0

    def _outputs(row_tokens: list[int]):
        vocab = 16
        logits = torch.full((2, 1, vocab), -1000.0)
        for row, token in enumerate(row_tokens):
            logits[row, 0, token] = 1000.0
        return type("Outputs", (), {"logits": logits})()

    def _request() -> SchedulerRequest:
        return SchedulerRequest(
            prepared=_prepared([1]),
            config=GenerationConfig(
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            ),
            stream=False,
            prompt_length=1,
            assembler=None,
        )

    request_a = _request()
    request_b = _request()
    requests = [request_a, request_b]

    finished: list[tuple[int, str]] = []

    def _finish(request, *, finish_reason: str) -> None:
        finished.append((request.completion_ids and request.completion_ids[-1] or -1, finish_reason))
        request.done.set()

    scheduler._finish_request = _finish  # type: ignore[method-assign]
    scheduler._emit_text = lambda request, text: None  # type: ignore[method-assign]
    scheduler._is_request_cancelled = lambda request: False  # type: ignore[method-assign]
    scheduler._drop_cancelled_request = lambda request: False  # type: ignore[method-assign]
    scheduler._compact_cache_rows = lambda cache, rows: None  # type: ignore[method-assign]
    scheduler._split_cache = lambda cache, n, avoid_turboquant_clone=False: [None] * n  # type: ignore[method-assign]
    scheduler._make_decode_group_from_requests = lambda reqs: reqs  # type: ignore[method-assign]

    # Step 1 (window 1/2): tokens stay on device, no host ids yet.
    active = scheduler._consume_batch_outputs(
        requests,
        _outputs([5, 7]),
        batch_cache=None,
        keep_batched_cache=False,
        avoid_turboquant_clone=True,
    )
    assert active == [request_a, request_b]
    assert request_a.completion_ids == [] and request_b.completion_ids == []
    assert len(request_a.device_pending_tokens) == 1
    assert len(request_b.device_pending_tokens) == 1
    assert scheduler._deferred_pull_counter == 1
    assert finished == []

    # Step 2 (window 2/2): bulk flush pulls both rows with one transfer; row_b hits
    # EOS (id 9) on its second token, so it finishes "stop" with the token kept.
    active = scheduler._consume_batch_outputs(
        [request_a, request_b],
        _outputs([6, 9]),
        batch_cache=None,
        keep_batched_cache=False,
        avoid_turboquant_clone=True,
    )
    assert request_a.completion_ids == [5, 6]
    assert request_b.completion_ids == [7]
    assert request_a.device_pending_tokens == []
    assert request_b.device_pending_tokens == []
    assert scheduler._deferred_pull_counter == 0
    assert finished == [(7, "stop")]
    assert active == [request_a]


def test_scheduler_batching_preserves_prepared_input_dataclass_type() -> None:
    scheduler = object.__new__(AnnaScheduler)
    scheduler.engine = type(
        "Engine",
        (),
        {
            "config": type(
                "Config",
                (),
                {
                    "text_config": type("TextConfig", (), {"pad_token_id": 0, "vocab_size": 32})(),
                },
            )(),
        },
    )()

    prepared = PreparedInputs(
        prompt="",
        input_ids=torch.tensor([[4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 3), dtype=torch.int32),
    )
    request = SchedulerRequest(prepared=prepared, config=None, stream=False, prompt_length=3)

    batched = scheduler._batch_text_inputs([request])

    assert isinstance(batched, PreparedInputs)
    assert batched.__class__.__module__ == "anna.mm.prepared_inputs"
