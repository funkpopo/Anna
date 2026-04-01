from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from anna.core.logging import setup_logging
from anna.core.model_path import resolve_model_dir, resolve_model_name
from anna.runtime.loader import load_engine_from_model_dir
from anna.runtime.tts_engine import SpeechSynthesisConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize speech with Anna.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Model name used in logs and API-compatible output.")
    parser.add_argument("--input", required=True, help="Text to synthesize.")
    parser.add_argument("--output", required=True, help="Output audio file path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--language", default=None)
    parser.add_argument("--speaker", default=None, help="Speaker name for CustomVoice models.")
    parser.add_argument("--instruct", default=None, help="Instruction text for VoiceDesign/CustomVoice models.")
    parser.add_argument("--ref-audio", default=None, help="Reference audio for Base voice-clone models.")
    parser.add_argument("--ref-text", default=None, help="Reference transcript for Base voice-clone models.")
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="For Base voice clone models, use speaker embedding only and skip reference transcript conditioning.",
    )
    parser.add_argument(
        "--response-format",
        choices=("wav", "flac"),
        default="wav",
        help="Container format used for the output file.",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=None)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument(
        "--subtalker-do-sample",
        dest="subtalker_do_sample",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-subtalker-do-sample",
        dest="subtalker_do_sample",
        action="store_false",
    )
    parser.add_argument(
        "--non-streaming-mode",
        action="store_true",
        default=True,
        help="Use the non-streaming text input path for synthesis. Enabled by default.",
    )
    parser.add_argument(
        "--streaming-style-input",
        dest="non_streaming_mode",
        action="store_false",
        help="Simulate streaming text input during synthesis.",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    model_name = resolve_model_name(model_name=args.model_name, model_dir=model_dir)
    output_path = Path(args.output).expanduser().resolve()

    setup_logging(args.log_level)
    engine = load_engine_from_model_dir(
        model_dir,
        model_id=model_name,
        device=args.device,
        dtype=args.dtype,
    )
    if not getattr(engine, "supports_speech_synthesis", False):
        raise SystemExit("The selected model does not support speech synthesis. Use anna-generate for text models.")

    result = engine.synthesize_speech(
        args.input,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only_mode,
        config=SpeechSynthesisConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            subtalker_do_sample=args.subtalker_do_sample,
            subtalker_temperature=args.subtalker_temperature,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_top_k=args.subtalker_top_k,
            non_streaming_mode=args.non_streaming_mode,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_name = "FLAC" if args.response_format == "flac" else "WAV"
    sf.write(str(output_path), result.audio, result.sample_rate, format=format_name, subtype="PCM_16")
    print(f"saved={output_path}")
    print(f"sample_rate={result.sample_rate}")
    print(f"duration_seconds={result.duration_seconds:.3f}")
    print(f"total_seconds={result.total_seconds:.3f}")


if __name__ == "__main__":
    main()
