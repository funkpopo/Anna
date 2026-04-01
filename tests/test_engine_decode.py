from __future__ import annotations

from collections import OrderedDict
from types import MethodType, SimpleNamespace

import torch

from anna.mm.qwen3_5_text_processor import PreparedInputs
from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, GenerationConfig, ThinkingStreamParser
from anna.runtime.qwen3_5_text_engine import EngineOptimizationConfig, StreamEvent, TextGenerationResult
from anna.runtime.streaming import IncrementalTextAssembler


def test_stable_decode_delta_avoids_repeated_prefix_output() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    emitted = ""
    parts: list[str] = []

    transitions = [
        ("", "�"),
        ("�", "🌞"),
        ("🌞", "🌞 夏"),
        ("🌞 夏", "🌞 夏天"),
    ]

    for previous_text, current_text in transitions:
        delta, emitted = engine._stable_decode_delta(
            previous_text=previous_text,
            current_text=current_text,
            emitted_text=emitted,
        )
        if delta:
            parts.append(delta)

    tail, emitted = engine._flush_decode_tail(
        current_text="🌞 夏天",
        emitted_text=emitted,
    )
    if tail:
        parts.append(tail)

    assert "".join(parts) == "🌞 夏天"


def test_flush_decode_tail_returns_remaining_buffer() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    tail, emitted = engine._flush_decode_tail(
        current_text="冰镇西瓜🍉",
        emitted_text="冰镇",
    )

    assert tail == "西瓜🍉"
    assert emitted == "冰镇西瓜🍉"


def test_stable_decode_skips_unstable_replacement_suffix() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    emitted = ""
    previous = ""
    outputs: list[str] = []

    for current in ["夏天。", "夏天。�", "夏天。�", "夏天。🌞", "夏天。🌞�", "夏天。🌞🍃"]:
        delta, emitted = engine._stable_decode_delta(
            previous_text=previous,
            current_text=current,
            emitted_text=emitted,
        )
        if delta:
            outputs.append(delta)
        previous = current

    tail, emitted = engine._flush_decode_tail(current_text=previous, emitted_text=emitted)
    if tail:
        outputs.append(tail)

    assert "".join(outputs) == "夏天。🌞🍃"


def test_split_chat_output_separates_reasoning_and_content() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    reasoning, content = engine._split_chat_output(
        "先分析问题。</think>\n\n最终答案。",
        enable_thinking=True,
    )

    assert reasoning == "先分析问题。"
    assert content == "最终答案。"


def test_thinking_stream_parser_splits_reasoning_chunks() -> None:
    parser = ThinkingStreamParser(enable_thinking=True)
    outputs = []
    outputs.extend(parser.feed("先分析"))
    outputs.extend(parser.feed("问题。</thi"))
    outputs.extend(parser.feed("nk>\n\n最终"))
    outputs.extend(parser.feed("答案。"))
    outputs.extend(parser.flush())

    reasoning = "".join(chunk for kind, chunk in outputs if kind == "reasoning")
    content = "".join(chunk for kind, chunk in outputs if kind == "content")

    assert reasoning == "先分析问题。"
    assert content == "最终答案。"


def test_generate_without_streaming_overhead_decodes_once() -> None:
    class DummyTokenizer:
        def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
            assert skip_special_tokens is False
            return ",".join(str(token_id) for token_id in token_ids)

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.tokenizer = DummyTokenizer()
    engine._generate_token_ids = lambda prepared, config: ([3, 4, 5], "stop", 7, 3, None)

    result = engine._generate_without_streaming_overhead(object(), config=GenerationConfig())

    assert result.text == "3,4,5"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3


def test_generate_text_prepares_prompt_on_preprocess_device() -> None:
    class DummyProcessor:
        def __init__(self) -> None:
            self.tensor_device = None

        def encode_text(self, prompt: str, *, tensor_device=None):
            del prompt
            self.tensor_device = tensor_device
            return object()

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.processor = DummyProcessor()
    engine.device_context = SimpleNamespace(
        device=torch.device("xpu"),
        dtype=torch.bfloat16,
        migration_policy=SimpleNamespace(preprocess_device=torch.device("cpu")),
    )
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="ok",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
        engine,
    )

    result = engine.generate_text("hello", config=GenerationConfig())

    assert result.text == "ok"
    assert engine.processor.tensor_device == torch.device("cpu")


def test_prepare_messages_uses_preprocess_device_before_xpu_transfer() -> None:
    class DummyProcessor:
        def __init__(self) -> None:
            self.tensor_device = None
            self.tensor_dtype = None

        def prepare_messages(self, messages, *, enable_thinking: bool, tensor_device=None, tensor_dtype=None):
            del messages, enable_thinking
            self.tensor_device = tensor_device
            self.tensor_dtype = tensor_dtype
            return object()

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.processor = DummyProcessor()
    engine.device_context = SimpleNamespace(
        device=torch.device("xpu"),
        dtype=torch.bfloat16,
        migration_policy=SimpleNamespace(preprocess_device=torch.device("cpu")),
    )

    prepared = engine._prepare_messages([{"role": "user", "content": "hi"}], enable_thinking=True)

    assert prepared is not None
    assert engine.processor.tensor_device == torch.device("cpu")
    assert engine.processor.tensor_dtype == torch.bfloat16


def test_trim_runtime_cache_if_idle_releases_allocator_pages() -> None:
    class DummyAllocator:
        def __init__(self) -> None:
            self.trim_calls = 0

        def trim(self) -> int:
            self.trim_calls += 1
            return 32

    class DummyDeviceContext:
        def __init__(self) -> None:
            self.release_calls = 0

        def release_unused_memory(self) -> None:
            self.release_calls += 1

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.cache_allocator = DummyAllocator()
    engine.device_context = DummyDeviceContext()
    engine.metrics = SimpleNamespace(snapshot=lambda: SimpleNamespace(running_requests=0, waiting_requests=0))

    engine._trim_runtime_cache_if_idle()

    assert engine.cache_allocator.trim_calls == 1
    assert engine.device_context.release_calls == 1


def test_trim_runtime_cache_if_idle_skips_when_requests_are_active() -> None:
    class DummyAllocator:
        def __init__(self) -> None:
            self.trim_calls = 0

        def trim(self) -> int:
            self.trim_calls += 1
            return 32

    class DummyDeviceContext:
        def __init__(self) -> None:
            self.release_calls = 0

        def release_unused_memory(self) -> None:
            self.release_calls += 1

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.cache_allocator = DummyAllocator()
    engine.device_context = DummyDeviceContext()
    engine.metrics = SimpleNamespace(snapshot=lambda: SimpleNamespace(running_requests=1, waiting_requests=0))

    engine._trim_runtime_cache_if_idle()

    assert engine.cache_allocator.trim_calls == 0
    assert engine.device_context.release_calls == 0


def test_generate_chat_keeps_raw_think_tags_when_reasoning_format_is_none() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="先分析问题。\n</think>\n\n最终答案。",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
        ),
        engine,
    )

    result = engine.generate_chat(
        [{"role": "user", "content": "你好"}],
        config=GenerationConfig(),
        reasoning_format="none",
    )

    assert result.text == "先分析问题。\n</think>\n\n最终答案。"
    assert result.reasoning_text is None


def test_generate_chat_projects_reasoning_into_deepseek_format() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="先分析问题。\n</think>\n\n最终答案。",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
        ),
        engine,
    )

    result = engine.generate_chat(
        [{"role": "user", "content": "你好"}],
        config=GenerationConfig(),
        reasoning_format="deepseek",
    )

    assert result.text == "最终答案。"
    assert result.reasoning_text == "先分析问题。"


def test_generate_chat_leaves_thoughts_inline_when_reasoning_format_is_none() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="先分析问题。\n</think>\n\n最终答案。",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
        ),
        engine,
    )

    result = engine.generate_chat(
        [{"role": "user", "content": "你好"}],
        config=GenerationConfig(),
        reasoning_format="none",
    )

    assert result.text == "先分析问题。\n</think>\n\n最终答案。"
    assert result.reasoning_text is None


def test_stream_chat_keeps_inline_think_chunks_when_reasoning_format_is_none() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._stream = MethodType(
        lambda self, prepared, *, config: iter(
            [
                StreamEvent(text="先分析", finish_reason=None),
                StreamEvent(text="问题。</think>\n\n最终答案。", finish_reason=None),
                StreamEvent(text="", finish_reason="stop"),
            ]
        ),
        engine,
    )

    events = list(
        engine.stream_chat(
            [{"role": "user", "content": "你好"}],
            config=GenerationConfig(),
            reasoning_format="none",
        )
    )

    assert [event.text for event in events] == [
        "先分析",
        "问题。</think>\n\n最终答案。",
        "",
    ]
    assert [event.reasoning_text for event in events] == [None, None, None]


def test_stream_chat_separates_reasoning_and_content_in_deepseek_format() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._stream = MethodType(
        lambda self, prepared, *, config: iter(
            [
                StreamEvent(text="先分析", finish_reason=None),
                StreamEvent(text="问题。</think>\n\n最终答案。", finish_reason=None),
                StreamEvent(text="", finish_reason="stop"),
            ]
        ),
        engine,
    )

    events = list(
        engine.stream_chat(
            [{"role": "user", "content": "你好"}],
            config=GenerationConfig(),
            reasoning_format="deepseek",
        )
    )

    assert [(event.text, event.reasoning_text, event.finish_reason) for event in events] == [
        ("", "先分析", None),
        ("", "问题。", None),
        ("最终答案。", None, None),
        ("", None, "stop"),
    ]


def test_generate_chat_keeps_incomplete_think_block_when_length_limited() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="用户希望我写一段关于夏天的帖子。",
            reasoning_text=None,
            finish_reason="length",
            prompt_tokens=10,
            completion_tokens=4,
        ),
        engine,
    )

    result = engine.generate_chat(
        [{"role": "user", "content": "你好"}],
        config=GenerationConfig(),
        reasoning_format="deepseek",
    )

    assert result.text == ""
    assert result.reasoning_text == "用户希望我写一段关于夏天的帖子。"
    assert result.finish_reason == "length"


def test_generate_chat_keeps_incomplete_think_block_inline_when_reasoning_format_is_none() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine._prepare_messages = MethodType(lambda self, messages, *, enable_thinking: object(), engine)
    engine._generate = MethodType(
        lambda self, prepared, *, config: TextGenerationResult(
            text="用户希望我写一段关于夏天的帖子。",
            reasoning_text=None,
            finish_reason="length",
            prompt_tokens=10,
            completion_tokens=4,
        ),
        engine,
    )

    result = engine.generate_chat(
        [{"role": "user", "content": "你好"}],
        config=GenerationConfig(),
        reasoning_format="none",
    )

    assert result.text == "用户希望我写一段关于夏天的帖子。"
    assert result.reasoning_text is None
    assert result.finish_reason == "length"


def test_forward_generation_model_uses_text_fast_path_for_text_only_requests() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def forward_text_only(self, **_kwargs):
            self.calls.append("text")
            return object()

        def __call__(self, **_kwargs):
            self.calls.append("full")
            return object()

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.model = _FakeModel()

    prepared = type(
        "Prepared",
        (),
        {
            "pixel_values": None,
            "pixel_values_videos": None,
        },
    )()

    engine._forward_generation_model(
        input_ids=object(),
        attention_mask=None,
        past_key_values=None,
        pixel_values=prepared.pixel_values,
        pixel_values_videos=prepared.pixel_values_videos,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        use_cache=True,
        logits_to_keep=1,
    )

    assert engine.model.calls == ["text"]


def test_forward_generation_model_prefers_compiled_text_fast_path_when_available() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def forward_text_only(self, **_kwargs):
            self.calls.append("text")
            return object()

        def __call__(self, **_kwargs):
            self.calls.append("full")
            return object()

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.model = _FakeModel()
    compiled_calls: list[str] = []
    engine._compiled_text_forward = lambda **_kwargs: compiled_calls.append("compiled") or object()

    engine._forward_generation_model(
        input_ids=object(),
        attention_mask=None,
        past_key_values=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        use_cache=True,
        logits_to_keep=1,
    )

    assert compiled_calls == ["compiled"]
    assert engine.model.calls == []


def test_forward_generation_model_keeps_full_path_for_multimodal_requests() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def forward_text_only(self, **_kwargs):
            self.calls.append("text")
            return object()

        def __call__(self, **_kwargs):
            self.calls.append("full")
            return object()

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.model = _FakeModel()

    engine._forward_generation_model(
        input_ids=object(),
        attention_mask=None,
        past_key_values=None,
        pixel_values=object(),
        pixel_values_videos=None,
        image_grid_thw=object(),
        video_grid_thw=None,
        mm_token_type_ids=object(),
        use_cache=True,
        logits_to_keep=1,
    )

    assert engine.model.calls == ["full"]


def test_prefill_generation_chunks_long_text_prompts() -> None:
    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.optimization_config = EngineOptimizationConfig(prefill_chunk_size=3)
    engine._prompt_cache = OrderedDict()
    engine._compiled_text_forward = None
    calls: list[tuple[torch.Tensor, torch.Tensor | None, object | None]] = []

    def _fake_forward(self, **kwargs):
        calls.append((kwargs["input_ids"].clone(), kwargs["attention_mask"], kwargs["past_key_values"]))
        return SimpleNamespace(
            logits=torch.tensor([[[float(kwargs["input_ids"][0, -1].item())]]]),
            past_key_values=SimpleNamespace(tag=f"cache-{len(calls)}"),
        )

    engine._forward_generation_model = MethodType(_fake_forward, engine)

    prepared = PreparedInputs(
        prompt="chunk me",
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long),
        attention_mask=torch.ones((1, 7), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 7), dtype=torch.int32),
    )

    result = engine._prefill_generation_prompt(prepared)

    assert [chunk.tolist() for chunk, _, _ in calls] == [[[1, 2, 3]], [[4, 5, 6]], [[7]]]
    assert calls[0][1] is not None
    assert calls[1][1] is None
    assert calls[2][1] is None
    assert result.logits.item() == 7.0
    assert result.prompt_cache_hit is False


def test_prefill_generation_reuses_prompt_cache_for_exact_prompt_matches() -> None:
    class _FakeCache:
        def __init__(self, label: str) -> None:
            self.label = label
            self.released = False

        def clone(self):
            return _FakeCache(self.label)

        def release(self) -> None:
            self.released = True

    engine = object.__new__(AnnaQwen3_5TextEngine)
    engine.optimization_config = EngineOptimizationConfig(prompt_cache_size=1)
    engine._prompt_cache = OrderedDict()
    engine._compiled_text_forward = None
    forward_calls: list[int] = []

    def _fake_forward(self, **_kwargs):
        forward_calls.append(1)
        return SimpleNamespace(
            logits=torch.tensor([[[11.0]]]),
            past_key_values=_FakeCache("prefill"),
        )

    engine._forward_generation_model = MethodType(_fake_forward, engine)

    prepared = PreparedInputs(
        prompt="cache me",
        input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        mm_token_type_ids=torch.zeros((1, 3), dtype=torch.int32),
    )

    first = engine._prefill_generation_prompt(prepared)
    second = engine._prefill_generation_prompt(prepared)

    assert len(forward_calls) == 1
    assert first.prompt_cache_hit is False
    assert second.prompt_cache_hit is True
    assert second.logits.item() == 11.0


def test_incremental_text_assembler_handles_unstable_unicode_suffix() -> None:
    class DummyTokenizer:
        mapping = {
            (1,): "\ufffd",
            (1, 2): "🌞",
            (3,): " 夏",
            (4,): "天",
        }

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
            return self.mapping[tuple(token_ids)]

    assembler = IncrementalTextAssembler(tokenizer=DummyTokenizer(), stop_strings=[])
    outputs = []
    for token_id in [1, 2, 3, 4]:
        delta, stopped = assembler.feed_token(token_id)
        assert stopped is False
        if delta:
            outputs.append(delta)
    tail, stopped = assembler.flush()
    assert stopped is False
    if tail:
        outputs.append(tail)

    assert "".join(outputs) == "🌞 夏天"


def test_incremental_text_assembler_uses_suffix_window_for_stop_strings() -> None:
    class DummyTokenizer:
        mapping = {
            (10,): "ABE",
            (11,): "N",
            (12,): "D!",
        }

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
            return self.mapping[tuple(token_ids)]

    assembler = IncrementalTextAssembler(tokenizer=DummyTokenizer(), stop_strings=["END"])
    outputs = []
    stopped = False
    for token_id in [10, 11, 12]:
        delta, stopped = assembler.feed_token(token_id)
        if delta:
            outputs.append(delta)
        if stopped:
            break

    assert stopped is True
    assert "".join(outputs) == "AB"
