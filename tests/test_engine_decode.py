from __future__ import annotations

from anna.runtime.engine import AnnaEngine, ThinkingStreamParser


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
