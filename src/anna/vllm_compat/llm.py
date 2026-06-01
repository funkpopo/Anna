from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from anna.runtime.qwen3_5_text_engine import AnnaQwen3_5TextEngine, GenerationConfig, TextGenerationResult
from anna.vllm_compat.outputs import RequestOutput, request_output_from_result
from anna.vllm_compat.sampling import SamplingParams, sampling_params_to_generation_config


class AnnaLLM:
    """Offline vLLM-style facade backed by Anna's native runtime.

    This Level 2 compatibility layer intentionally does not pretend to be a
    vLLM worker/plugin. It mirrors the common ``LLM.generate`` shape so callers
    can use Anna in vLLM-like offline code while the runtime adapter is built.
    """

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        engine: object | None = None,
        model_id: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        if engine is None:
            if model is None:
                raise ValueError("Either model or engine must be provided.")
            engine = AnnaQwen3_5TextEngine.from_model_dir(model, model_id=model_id, **engine_kwargs)
        self.engine = engine
        self.model = model_id or getattr(engine, "default_model_id", None) or (Path(model).name if model is not None else "anna")

    def generate(
        self,
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool | None = None,
        request_id_prefix: str = "anna",
        **_: Any,
    ) -> list[RequestOutput]:
        del use_tqdm
        prompt_list = [prompts] if isinstance(prompts, str) else [str(prompt) for prompt in prompts]
        generation_config = sampling_params_to_generation_config(sampling_params)
        outputs: list[RequestOutput] = []
        for index, prompt in enumerate(prompt_list):
            result = self.engine.generate_text(prompt, config=generation_config)
            outputs.append(self._request_output_from_result(prompt, result, request_id=f"{request_id_prefix}-{index}-{uuid4().hex}"))
        return outputs

    @staticmethod
    def _request_output_from_result(prompt: str, result: TextGenerationResult, *, request_id: str) -> RequestOutput:
        return request_output_from_result(prompt, result, request_id=request_id)


def generation_config_from_sampling_params(params: SamplingParams | None) -> GenerationConfig:
    return sampling_params_to_generation_config(params)
