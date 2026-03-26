from __future__ import annotations

from anna.runtime.engine import AnnaEngine, GenerationConfig, ThinkingStreamParser, TokenGenerationRun
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
    engine._generate_token_ids = lambda prepared, config, collect_timing=False: TokenGenerationRun(
        completion_ids=[3, 4, 5],
        finish_reason="stop",
        prompt_tokens=7,
        completion_tokens=3,
        prefill_seconds=0.12 if collect_timing else None,
        decode_seconds=0.34 if collect_timing else None,
    )

    result = engine._generate_without_streaming_overhead(object(), config=GenerationConfig())

    assert result.text == "3,4,5"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3


def test_generate_without_streaming_overhead_preserves_profile_metrics() -> None:
    class DummyTokenizer:
        def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
            return "done"

    engine = object.__new__(AnnaEngine)
    engine.tokenizer = DummyTokenizer()
    engine._generate_token_ids = lambda prepared, config, collect_timing=False: TokenGenerationRun(
        completion_ids=[1, 2],
        finish_reason="length",
        prompt_tokens=5,
        completion_tokens=2,
        prefill_seconds=0.5 if collect_timing else None,
        decode_seconds=0.25 if collect_timing else None,
    )

    result = engine._generate_without_streaming_overhead(
        object(),
        config=GenerationConfig(),
        collect_timing=True,
    )

    assert result.prefill_seconds == 0.5
    assert result.decode_seconds == 0.25
    assert result.decode_tokens_per_second == 8.0
    assert result.end_to_end_tokens_per_second == 2 / 0.75


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
