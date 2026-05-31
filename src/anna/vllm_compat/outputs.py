from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int] = field(default_factory=list)
    cumulative_logprob: float | None = None
    logprobs: Any | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = True
    metrics: Any | None = None
