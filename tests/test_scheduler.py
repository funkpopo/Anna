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
        self.decode_calls: list[list[int]] = []

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        self.decode_calls.append(list(token_ids))
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
        self.text_prefill_topk_values: list[int] = []
        self.text_decode_topk_values: list[int] = []
        self.text_decode_slot_input_request_ids: list[tuple[str, ...]] = []
        self.text_decode_slot_input_batch_positions: list[list[int]] = []
        self.text_decode_slot_input_batch_seq_lens: list[list[int]] = []
        self.text_prefill_chunk_lengths: list[int] = []
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
        slot_decode_inputs: object | None = None,
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
        if slot_decode_inputs is not None:
            self.text_decode_slot_input_request_ids.append(tuple(slot_decode_inputs.request_ids))
            self.text_decode_slot_input_batch_positions.append(slot_decode_inputs.batch_positions.tolist())
            self.text_decode_slot_input_batch_seq_lens.append(slot_decode_inputs.batch_seq_lens.tolist())
        logits = torch.full((batch_size, 1, self.config.text_config.vocab_size), -1000.0)
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
        slot_decode_inputs: object | None = None,
    ):
        del attention_mask, use_cache, logits_to_keep
        batch_size = input_ids.shape[0]
        candidate_logits = torch.full((batch_size, 1, top_k), -1000.0)
        candidate_token_ids = torch.zeros((batch_size, 1, top_k), dtype=torch.long)
        if past_key_values is None or input_ids.shape[1] > 1:
            seq_len = input_ids.shape[1]
            self.text_prefill_topk_batch_sizes.append(batch_size)
            self.text_prefill_topk_values.append(top_k)
            planned = [1, 2]
            for idx in range(batch_size):
                candidate_logits[idx, 0, 0] = 1000.0
                candidate_token_ids[idx, 0, 0] = planned[idx]
            cache = self._make_cache(batch_size=batch_size, seq_len=seq_len)
        else:
            self.text_decode_topk_batch_sizes.append(batch_size)
            self.text_decode_topk_values.append(top_k)
            if slot_decode_inputs is not None:
                self.text_decode_slot_input_request_ids.append(tuple(slot_decode_inputs.request_ids))
                self.text_decode_slot_input_batch_positions.append(slot_decode_inputs.batch_positions.tolist())
                self.text_decode_slot_input_batch_seq_lens.append(slot_decode_inputs.batch_seq_lens.tolist())
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


def _prepared(prompt_tokens: list[int]) -> PreparedInputs:
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    return PreparedInputs(
        prompt="",
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        mm_token_type_ids=torch.zeros_like(input_ids, dtype=torch.int32),
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
    tokenizer = _FakeTokenizer()
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=tokenizer,
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
        assert request_a.result.prompt_token_ids == [4, 5]
        assert request_b.result.prompt_token_ids == [6, 7]
        assert request_a.result.completion_token_ids == [1]
        assert request_b.result.completion_token_ids == [2]
        assert request_a.assembler is None
        assert request_b.assembler is None
        assert request_a.text_parts == []
        assert request_b.text_parts == []
        assert tokenizer.decode_calls == [[1], [2]]
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
        assert snapshot.cpu_sync_count == 2
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
        assert snapshot.running_requests == 0
        assert snapshot.waiting_requests == 0
    finally:
        scheduler.shutdown()


def test_scheduler_single_request_decode_avoids_cache_stack_split_metrics() -> None:
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
        request = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert request.done.wait(timeout=2.0)
        assert request.error is None
        assert fake_model.text_prefill_batch_sizes == [1]
        assert fake_model.text_decode_batch_sizes == [1]
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
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
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
    finally:
        scheduler.shutdown()


def test_scheduler_advances_slot_metadata_for_chunked_prefill() -> None:
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
            max_position_embeddings=16,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    tokenizer = _FakeTokenizer()
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=tokenizer,
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
        optimization_config=EngineOptimizationConfig(
            prefill_chunk_size=2,
            slot_runner_enabled=True,
            slot_runner_max_slots=2,
            slot_runner_total_blocks=8,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=2,
        ),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=2, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    assert engine.slot_model_runner is not None
    original_advance_prefill = engine.slot_model_runner.advance_prefill
    prefill_updates: list[tuple[str, int, int, int, tuple[int, ...]]] = []

    def _record_advance_prefill(request_id: str, *, token_count: int):
        slot = original_advance_prefill(request_id, token_count=token_count)
        assert engine.slot_model_runner is not None
        blocks = engine.slot_model_runner.kv_manager.slot_blocks(slot.slot_id, slot.epoch)
        prefill_updates.append((request_id, token_count, slot.prefilled_tokens, slot.seq_len, blocks))
        return slot

    engine.slot_model_runner.advance_prefill = _record_advance_prefill

    try:
        request_a = scheduler._submit(
            _prepared([4, 5, 6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([8, 10, 11, 12]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None

        assert [(request_id, token_count, prefilled, seq_len) for request_id, token_count, prefilled, seq_len, _ in prefill_updates] == [
            ("scheduler-0", 2, 2, 2),
            ("scheduler-1", 2, 2, 2),
            ("scheduler-0", 2, 4, 4),
            ("scheduler-1", 2, 4, 4),
        ]
        assert [len(blocks) for *_, blocks in prefill_updates] == [1, 1, 2, 2]
        assert fake_model.text_prefill_chunk_lengths == [2, 2]
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
        assert snapshot.cpu_sync_count == 3
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
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
        assert fake_model.text_prefill_topk_values == [1]
        assert fake_model.text_decode_topk_values == [1]
        assert fake_model.text_prefill_batch_sizes == []
        assert fake_model.text_decode_batch_sizes == []
    finally:
        scheduler.shutdown()


def test_scheduler_uses_max_topk_forward_for_mixed_candidate_batches() -> None:
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
            config=GenerationConfig(max_new_tokens=2, temperature=0.8, top_p=1.0, top_k=1, presence_penalty=0.0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.7, top_p=1.0, top_k=3, presence_penalty=0.0),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert fake_model.text_prefill_topk_batch_sizes == [2]
        assert fake_model.text_decode_topk_batch_sizes == [2]
        assert fake_model.text_prefill_topk_values == [3]
        assert fake_model.text_decode_topk_values == [3]
        assert fake_model.text_prefill_batch_sizes == []
        assert fake_model.text_decode_batch_sizes == []
    finally:
        scheduler.shutdown()


def test_scheduler_populates_experimental_slot_decode_inputs() -> None:
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
            max_position_embeddings=16,
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
        optimization_config=EngineOptimizationConfig(
            slot_runner_enabled=True,
            slot_runner_max_slots=2,
            slot_runner_total_blocks=8,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=2,
        ),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=2, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    assert engine.slot_model_runner is not None
    original_mark_prefilled_batch = engine.slot_model_runner.mark_prefilled_batch
    prefill_batch_marks: list[tuple[tuple[str, ...], tuple[int | None, ...] | None]] = []

    def _record_mark_prefilled_batch(
        *,
        request_ids,
        next_input_ids,
        next_input_host_ids=None,
    ):
        prefill_batch_marks.append(
            (
                tuple(str(request_id) for request_id in request_ids),
                None if next_input_host_ids is None else tuple(next_input_host_ids),
            )
        )
        return original_mark_prefilled_batch(
            request_ids=request_ids,
            next_input_ids=next_input_ids,
            next_input_host_ids=next_input_host_ids,
        )

    engine.slot_model_runner.mark_prefilled_batch = _record_mark_prefilled_batch

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None

        slot_inputs = scheduler._last_slot_decode_inputs
        assert slot_inputs is not None
        assert slot_inputs.request_ids == ("scheduler-0", "scheduler-1")
        assert slot_inputs.input_ids.tolist() == [[1], [2]]
        assert slot_inputs.positions_are_global is True
        assert slot_inputs.seq_lens_are_global is True
        assert slot_inputs.block_tables.shape == (2, 4)
        assert slot_inputs.block_tables_are_global is True
        assert slot_inputs.physical_block_tables is False
        assert prefill_batch_marks == [(("scheduler-0", "scheduler-1"), (1, 2))]
        assert fake_model.text_decode_slot_input_request_ids == [("scheduler-0", "scheduler-1")]
        assert fake_model.text_decode_slot_input_batch_positions == [[2, 2]]
        assert fake_model.text_decode_slot_input_batch_seq_lens == [[2, 2]]

        snapshot = engine.service_metrics_snapshot()
        assert snapshot.slot_decode_plan_count == 1
        assert snapshot.slot_decode_plan_seconds_total >= 0.0
        assert engine.slot_model_runner.active_count == 0
        assert engine.slot_model_runner.kv_manager.free_slot_count == 2
    finally:
        scheduler.shutdown()


def test_scheduler_limits_pending_admission_to_slot_runner_capacity() -> None:
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
            max_position_embeddings=16,
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
        optimization_config=EngineOptimizationConfig(
            slot_runner_enabled=True,
            slot_runner_max_slots=2,
            slot_runner_total_blocks=8,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=1,
        ),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert request_a.done.wait(timeout=2.0)
        assert request_b.done.wait(timeout=2.0)
        assert request_a.error is None
        assert request_b.error is None
        assert request_a.result is not None
        assert request_b.result is not None
        assert fake_model.text_prefill_batch_sizes == [1, 1]
        assert fake_model.text_decode_batch_sizes == [1, 1]
        assert fake_model.text_decode_slot_input_request_ids == [("scheduler-0",), ("scheduler-1",)]

        snapshot = engine.service_metrics_snapshot()
        assert snapshot.requests_started_total == 2
        assert snapshot.requests_completed_total == 2
        assert snapshot.requests_failed_total == 0
        assert snapshot.cache_stack_count == 0
        assert snapshot.cache_split_count == 0
        assert engine.slot_model_runner is not None
        assert engine.slot_model_runner.active_count == 0
        assert engine.slot_model_runner.kv_manager.free_slot_count == 2
    finally:
        scheduler.shutdown()


def test_scheduler_waits_when_slot_runner_capacity_is_temporarily_full() -> None:
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
            max_position_embeddings=16,
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
        optimization_config=EngineOptimizationConfig(
            slot_runner_enabled=True,
            slot_runner_max_slots=1,
            slot_runner_total_blocks=4,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=1,
        ),
    )
    assert engine.slot_model_runner is not None
    engine.slot_model_runner.admit_prefill(
        "external",
        prompt_length=1,
        max_new_tokens=1,
        sampling_params=GenerationConfig(max_new_tokens=1),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=0.0)
    engine.set_scheduler(scheduler)

    try:
        request = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert not request.done.wait(timeout=0.05)
        engine.slot_model_runner.cancel("external")

        assert request.done.wait(timeout=2.0)
        assert request.error is None
        assert request.result is not None
        assert fake_model.text_prefill_batch_sizes == [1]
        assert fake_model.text_decode_batch_sizes == [1]
        assert fake_model.text_decode_slot_input_request_ids == [("scheduler-0",)]
        assert engine.slot_model_runner.active_count == 0
        assert engine.slot_model_runner.kv_manager.free_slot_count == 1

        snapshot = engine.service_metrics_snapshot()
        assert snapshot.requests_started_total == 1
        assert snapshot.requests_completed_total == 1
        assert snapshot.requests_failed_total == 0
    finally:
        if engine.slot_model_runner is not None:
            try:
                engine.slot_model_runner.cancel("external")
            except KeyError:
                pass
        scheduler.shutdown()


def test_scheduler_shutdown_fails_pending_request_waiting_for_slot_runner_capacity() -> None:
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
            max_position_embeddings=16,
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
        optimization_config=EngineOptimizationConfig(
            slot_runner_enabled=True,
            slot_runner_max_slots=1,
            slot_runner_total_blocks=4,
            slot_runner_max_blocks_per_seq=4,
            slot_runner_max_batch_size=1,
        ),
    )
    assert engine.slot_model_runner is not None
    engine.slot_model_runner.admit_prefill(
        "external",
        prompt_length=1,
        max_new_tokens=1,
        sampling_params=GenerationConfig(max_new_tokens=1),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=0.0)
    engine.set_scheduler(scheduler)
    shutdown_called = False

    try:
        request = scheduler._submit(
            _prepared([4, 5]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )

        assert not request.done.wait(timeout=0.05)
        scheduler.shutdown()
        shutdown_called = True

        assert request.done.is_set()
        assert request.result is None
        assert request.error is not None
        assert request.error.code == "scheduler_shutdown"
        assert not scheduler._worker.is_alive()

        snapshot = engine.service_metrics_snapshot()
        assert snapshot.requests_started_total == 1
        assert snapshot.requests_completed_total == 0
        assert snapshot.requests_failed_total == 1
        assert snapshot.running_requests == 0
        assert snapshot.waiting_requests == 0
        assert fake_model.text_prefill_batch_sizes == []
        assert fake_model.text_decode_batch_sizes == []
    finally:
        if not shutdown_called:
            scheduler.shutdown()
        if engine.slot_model_runner is not None:
            try:
                engine.slot_model_runner.cancel("external")
            except KeyError:
                pass


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
        prompt_token_ids=[4, 5, 6],
    )
    request = SchedulerRequest(
        prepared=prepared,
        config=None,
        stream=False,
        prompt_ids=[4, 5, 6],
        prompt_length=3,
    )

    batched = scheduler._batch_text_inputs([request])

    assert isinstance(batched, PreparedInputs)
    assert batched.__class__.__module__ == "anna.mm.prepared_inputs"
    assert batched.prompt_token_ids == [4, 5, 6]
