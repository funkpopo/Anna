from __future__ import annotations

import argparse
from pathlib import Path

from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.model_runtime_loader import load_model_runtime_from_model_dir
from anna.runtime.qwen3_asr_engine import Qwen3ASRTranscriptionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with qwen3_asr in Anna.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name used in logs and API-compatible output.")
    parser.add_argument("--audio", required=True, help="Input audio file path.")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--language", default=None, help="Language name such as English or Chinese. Omit for auto detection.")
    parser.add_argument("--return-timestamps", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    model_dir = resolve_model_dir(args.model_dir)
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    engine = load_model_runtime_from_model_dir(
        model_dir,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
    )
    if not getattr(engine, "supports_audio_transcriptions", False):
        raise SystemExit(
            f"The selected model belongs to the {getattr(engine, 'model_family', 'unknown')} family and does not support audio transcription."
        )

    result = engine.transcribe_qwen3_asr_audio(
        str(audio_path),
        config=Qwen3ASRTranscriptionConfig(
            language=args.language,
            return_timestamps=args.return_timestamps,
        ),
    )
    print(result.text)
    if result.language is not None:
        print(f"language={result.language}")
    print(f"total_seconds={result.total_seconds:.3f}")
