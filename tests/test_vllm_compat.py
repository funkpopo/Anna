from __future__ import annotations

from dataclasses import dataclass

import pytest

from anna.runtime.qwen3_5_text_engine import GenerationConfig, TextGenerationResult
from anna.vllm_compat import AnnaLLM, CompletionOutput, RequestOutput, SamplingParams, sampling_params_to_generation_config


def test_sampling_params_maps_to_generation_config() -> None:
    params = SamplingParams(
        max_tokens=32,
        temperature=0.0,
        top_p=0.75,
        top_k=40,
        min_p=0.05,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        stop=["END"],
    )

    config = sampling_params_to_generation_config(params)

    assert isinstance(config, GenerationConfig)
    assert config.max_new_tokens == 32
    assert config.temperature == 0.0
    assert config.top_p == 0.75
    assert config.top_k == 40
    assert config.min_p == 0.05
    assert config.presence_penalty == 0.2
    assert config.repetition_penalty == 1.1
    assert config.stop_strings == ["END"]


def test_sampling_params_treats_vllm_top_k_minus_one_as_disabled() -> None:
    config = sampling_params_to_generation_config(SamplingParams(top_k=-1))

    assert config.top_k == 0


def test_sampling_params_rejects_unsupported_multi_output() -> None:
    with pytest.raises(NotImplementedError, match="n=1"):
        sampling_params_to_generation_config(SamplingParams(n=2))


@dataclass
class _FakeEngine:
    default_model_id: str = "fake-model"
    calls: list[tuple[str, GenerationConfig]] | None = None

    def generate_text(self, prompt: str, *, config: GenerationConfig) -> TextGenerationResult:
        if self.calls is None:
            self.calls = []
        self.calls.append((prompt, config))
        return TextGenerationResult(
            text=f"out:{prompt}",
            finish_reason="stop",
            prompt_tokens=len(prompt),
            completion_tokens=1,
        )


def test_anna_llm_generate_returns_vllm_like_request_outputs() -> None:
    engine = _FakeEngine()
    llm = AnnaLLM(engine=engine)

    outputs = llm.generate(["a", "bb"], SamplingParams(max_tokens=3, temperature=0.0), request_id_prefix="req")

    assert len(outputs) == 2
    assert all(isinstance(output, RequestOutput) for output in outputs)
    assert all(isinstance(output.outputs[0], CompletionOutput) for output in outputs)
    assert [output.prompt for output in outputs] == ["a", "bb"]
    assert [output.outputs[0].text for output in outputs] == ["out:a", "out:bb"]
    assert [output.outputs[0].finish_reason for output in outputs] == ["stop", "stop"]
    assert outputs[0].request_id.startswith("req-0-")
    assert engine.calls is not None
    assert [call[1].max_new_tokens for call in engine.calls] == [3, 3]
