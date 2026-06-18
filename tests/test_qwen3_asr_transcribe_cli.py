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
            "--context",
            "project vocabulary",
            "--return-timestamps",
            "--asr-max-inference-batch-size",
            "2",
            "--asr-max-new-tokens",
            "64",
            "--xpu-device-index",
            "0",
        ]
    )

    assert args.model_dir == "model"
    assert args.audio == "input.wav"
    assert args.device == "xpu"
    assert args.language == "English"
    assert args.context == "project vocabulary"
    assert args.return_timestamps is True
    assert args.asr_max_inference_batch_size == 2
    assert args.asr_max_new_tokens == 64
    assert args.xpu_device_index == 0
