from __future__ import annotations

from anna.cli.speak import build_parser


def test_speak_parser_accepts_voice_clone_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--input",
            "Hello from Anna.",
            "--output",
            "out.wav",
            "--ref-audio",
            "ref.wav",
            "--ref-text",
            "Reference line.",
            "--x-vector-only-mode",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--response-format",
            "flac",
        ]
    )

    assert args.model_dir == "model"
    assert args.input == "Hello from Anna."
    assert args.output == "out.wav"
    assert args.ref_audio == "ref.wav"
    assert args.ref_text == "Reference line."
    assert args.x_vector_only_mode is True
    assert args.temperature == 0.8
    assert args.top_p == 0.95
    assert args.response_format == "flac"


def test_speak_parser_accepts_custom_voice_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--input",
            "Hello from Anna.",
            "--output",
            "out.wav",
            "--speaker",
            "Vivian",
            "--instruct",
            "Speak with energy.",
            "--language",
            "English",
            "--streaming-style-input",
            "--no-do-sample",
            "--no-subtalker-do-sample",
        ]
    )

    assert args.speaker == "Vivian"
    assert args.instruct == "Speak with energy."
    assert args.language == "English"
    assert args.non_streaming_mode is False
    assert args.do_sample is False
    assert args.subtalker_do_sample is False
