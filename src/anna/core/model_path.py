from __future__ import annotations

from pathlib import Path


def resolve_model_dir(model_dir: str | Path) -> Path:
    resolved = Path(model_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    return resolved


def resolve_model_name(
    *,
    model_name: str | None,
    model_dir: str | Path,
) -> str:
    if model_name:
        return model_name
    return str(Path(model_dir).expanduser().resolve())
