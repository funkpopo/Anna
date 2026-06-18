from __future__ import annotations

from anna.cli.qwen3_asr_transcribe import build_parser


def test_transcribe_parser_accepts_asr_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--audio",
            "input.wav",
            "--device",
            "xpu",
            "--language",
            "English",
            "--return-timestamps",
        ]
    )

    assert args.model_dir == "model"
    assert args.audio == "input.wav"
    assert args.device == "xpu"
    assert args.language == "English"
    assert args.return_timestamps is True
