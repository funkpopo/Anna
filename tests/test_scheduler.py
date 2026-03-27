from __future__ import annotations

import torch

from anna.mm.processor import PreparedInputs
from anna.model.config import Qwen3Config, Qwen3TextConfig
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator
from anna.runtime.device import RuntimeSafetyPolicy
from anna.runtime.engine import AnnaEngine, GenerationConfig
from anna.runtime.scheduler import AnnaScheduler


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
        self.safety_policy = RuntimeSafetyPolicy()

    def get_memory_info(self):
        return None

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
    def __init__(self, config: Qwen3Config) -> None:
        self.config = config
        self.cache_allocator = Qwen3PageAllocator(config.text_config)
        self.prefill_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []
        self.text_prefill_batch_sizes: list[int] = []
        self.text_decode_batch_sizes: list[int] = []
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
        if past_key_values is None:
            seq_len = input_ids.shape[1]
            self.text_prefill_batch_sizes.append(batch_size)
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
        logits[:, 0, 9] = 1000.0
        return type(
            "TextDecodeOutput",
            (),
            {
                "logits": logits,
                "past_key_values": past_key_values,
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
    config = Qwen3Config(
        text_config=Qwen3TextConfig(
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
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=4, batch_wait_ms=20.0)
    engine.set_scheduler(scheduler)

    try:
        request_a = scheduler._submit(_prepared([4, 5]), config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0), stream=False)
        request_b = scheduler._submit(_prepared([6, 7]), config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0), stream=False)

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
    finally:
        scheduler.shutdown()


def test_scheduler_batches_mixed_length_requests_during_decode() -> None:
    config = Qwen3Config(
        text_config=Qwen3TextConfig(
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
            layer_types=["full_attention"],
        )
    )
    fake_model = _FakeModel(config)
    engine = AnnaEngine(
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
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
            stream=False,
        )
        request_b = scheduler._submit(
            _prepared([6, 7, 8]),
            config=GenerationConfig(max_new_tokens=2, temperature=0.0, top_p=1.0, top_k=0),
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
    finally:
        scheduler.shutdown()
