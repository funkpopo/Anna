from __future__ import annotations

from types import MethodType

from anna.runtime.engine import AnnaEngine, GenerationConfig, ThinkingStreamParser
from anna.runtime.engine import StreamEvent, TextGenerationResult
from anna.runtime.streaming import IncrementalTextAssembler


def test_stable_decode_delta_avoids_repeated_prefix_output() -> None:
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
    tail, emitted = engine._flush_decode_tail(
        current_text="冰镇西瓜🍉",
        emitted_text="冰镇",
    )

    assert tail == "西瓜🍉"
    assert emitted == "冰镇西瓜🍉"


def test_stable_decode_skips_unstable_replacement_suffix() -> None:
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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

    engine = object.__new__(AnnaEngine)
    engine.tokenizer = DummyTokenizer()
    engine._generate_token_ids = lambda prepared, config: ([3, 4, 5], "stop", 7, 3)

    result = engine._generate_without_streaming_overhead(object(), config=GenerationConfig())

    assert result.text == "3,4,5"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3


def test_generate_chat_preserves_raw_think_tags() -> None:
    engine = object.__new__(AnnaEngine)
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
    assert result.reasoning_text == "先分析问题。"


def test_generate_chat_projects_reasoning_into_deepseek_format() -> None:
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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
    engine = object.__new__(AnnaEngine)
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
