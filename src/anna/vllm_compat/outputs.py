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


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).tolist()
    return [int(item) for item in value]


def _first_present_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def request_output_from_result(prompt: str, result: Any, *, request_id: str, index: int = 0) -> RequestOutput:
    """Build a vLLM-like request output from Anna result-shaped objects.

    Native ``TextGenerationResult`` currently exposes text and token counts.
    Scheduler/runtime paths can attach richer fields over time; this adapter
    consumes them when present while preserving the existing empty-list shape.
    """

    finish_reason = getattr(result, "finish_reason", None)
    prompt_token_ids = _coerce_int_list(
        _first_present_attr(result, ("prompt_token_ids", "prompt_ids"))
    )
    completion_token_ids = _coerce_int_list(
        _first_present_attr(result, ("token_ids", "completion_token_ids", "completion_ids", "output_token_ids"))
    )
    return RequestOutput(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        outputs=[
            CompletionOutput(
                index=index,
                text=str(getattr(result, "text", "")),
                token_ids=completion_token_ids,
                cumulative_logprob=getattr(result, "cumulative_logprob", None),
                logprobs=getattr(result, "logprobs", None),
                finish_reason=finish_reason,
                stop_reason=getattr(result, "stop_reason", finish_reason),
            )
        ],
        finished=finish_reason is not None,
        metrics=getattr(result, "perf", None),
    )
