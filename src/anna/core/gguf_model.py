from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GGUFModelFiles:
    model_file: Path
    mmproj_file: Path | None
    available_mmproj_files: tuple[Path, ...]


def is_gguf_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".gguf"


def is_gguf_mmproj_file(path: str | Path) -> bool:
    candidate = Path(path)
    return is_gguf_file(candidate) and candidate.name.lower().startswith("mmproj")


def _root_dir(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _resolve_under_root(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def list_gguf_model_files(model_dir: str | Path) -> tuple[Path, ...]:
    model_path = Path(model_dir).expanduser().resolve()
    if model_path.is_file():
        if is_gguf_file(model_path) and not is_gguf_mmproj_file(model_path):
            return (model_path,)
        return ()
    if not model_path.is_dir():
        return ()
    return tuple(
        sorted(
            path.resolve()
            for path in model_path.glob("*.gguf")
            if path.is_file() and not is_gguf_mmproj_file(path)
        )
    )


def list_gguf_mmproj_files(model_dir: str | Path) -> tuple[Path, ...]:
    model_path = Path(model_dir).expanduser().resolve()
    root = _root_dir(model_path)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path.resolve()
            for path in root.glob("*.gguf")
            if path.is_file() and is_gguf_mmproj_file(path)
        )
    )


def has_gguf_model(model_dir: str | Path) -> bool:
    return bool(list_gguf_model_files(model_dir))


def resolve_gguf_model_files(
    model_dir: str | Path,
    *,
    model_file: str | Path | None = None,
    mmproj_file: str | Path | None = None,
) -> GGUFModelFiles:
    model_path = Path(model_dir).expanduser().resolve()
    root = _root_dir(model_path)
    available_mmproj_files = list_gguf_mmproj_files(model_path)

    if model_file is not None:
        resolved_model_file = _resolve_under_root(model_file, root)
        if not resolved_model_file.is_file():
            raise FileNotFoundError(f"GGUF model file does not exist: {resolved_model_file}")
        if not is_gguf_file(resolved_model_file):
            raise ValueError(f"GGUF model file must end with .gguf: {resolved_model_file}")
        if is_gguf_mmproj_file(resolved_model_file):
            raise ValueError(f"GGUF model file must not be an mmproj projection file: {resolved_model_file}")
    else:
        candidates = list_gguf_model_files(model_path)
        if not candidates:
            raise FileNotFoundError(f"No GGUF model file found in {model_path}")
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Multiple GGUF model files found in {model_path}; pass --gguf-model-file. Candidates: {names}")
        resolved_model_file = candidates[0]

    resolved_mmproj_file: Path | None = None
    if mmproj_file is not None:
        resolved_mmproj_file = _resolve_under_root(mmproj_file, root)
        if not resolved_mmproj_file.is_file():
            raise FileNotFoundError(f"GGUF mmproj file does not exist: {resolved_mmproj_file}")
        if not is_gguf_mmproj_file(resolved_mmproj_file):
            raise ValueError(f"GGUF mmproj file must be an mmproj .gguf file: {resolved_mmproj_file}")

    return GGUFModelFiles(
        model_file=resolved_model_file,
        mmproj_file=resolved_mmproj_file,
        available_mmproj_files=available_mmproj_files,
    )
