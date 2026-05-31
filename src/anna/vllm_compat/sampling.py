from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from anna.runtime.qwen3_5_text_engine import GenerationConfig


@dataclass(slots=True)
class SamplingParams:
    n: int = 1
    best_of: int | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    extra_args: dict[str, object] = field(default_factory=dict)

    def __init__(
        self,
        *,
        n: int = 1,
        best_of: int | None = None,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        include_stop_str_in_output: bool = False,
        **kwargs: object,
    ) -> None:
        if n < 1:
            raise ValueError("n must be at least 1.")
        if best_of is not None and best_of < n:
            raise ValueError("best_of must be greater than or equal to n.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1].")
        if top_k == 0 or top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer.")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError("min_p must be in the range [0, 1].")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive.")

        self.n = int(n)
        self.best_of = None if best_of is None else int(best_of)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.presence_penalty = float(presence_penalty)
        self.repetition_penalty = float(repetition_penalty)
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = bool(ignore_eos)
        self.skip_special_tokens = bool(skip_special_tokens)
        self.spaces_between_special_tokens = bool(spaces_between_special_tokens)
        self.include_stop_str_in_output = bool(include_stop_str_in_output)
        self.extra_args = dict(kwargs)


def normalize_stop_strings(stop: str | Iterable[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [str(item) for item in stop]


def sampling_params_to_generation_config(params: SamplingParams | None) -> GenerationConfig:
    params = SamplingParams() if params is None else params
    if params.n != 1:
        raise NotImplementedError("Anna vLLM compatibility currently supports n=1.")
    if params.best_of is not None and params.best_of != 1:
        raise NotImplementedError("Anna vLLM compatibility currently supports best_of=None or 1.")
    if params.ignore_eos:
        raise NotImplementedError("Anna vLLM compatibility does not yet support ignore_eos=True.")
    if params.stop_token_ids:
        raise NotImplementedError("Anna vLLM compatibility does not yet support stop_token_ids.")
    if params.include_stop_str_in_output:
        raise NotImplementedError("Anna vLLM compatibility does not yet support include_stop_str_in_output=True.")

    return GenerationConfig(
        max_new_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=0 if params.top_k == -1 else params.top_k,
        min_p=params.min_p,
        presence_penalty=params.presence_penalty,
        repetition_penalty=params.repetition_penalty,
        stop_strings=normalize_stop_strings(params.stop),
    )
