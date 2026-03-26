from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ServeSettings:
    model_dir: Path
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    scheduler_max_batch_size: int = 4
    scheduler_batch_wait_ms: float = 2.0
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


@dataclass(slots=True)
class GenerateSettings:
    model_dir: Path
    prompt: str
    model_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass(slots=True)
class BenchmarkSettings:
    model_dir: Path
    prompt: str
    model_id: str | None = None
    image: str | None = None
    video: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    warmup: int = 1
    runs: int = 3
