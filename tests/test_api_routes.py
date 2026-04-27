from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from anna.api.app import create_app, list_app_routes
from anna.runtime.qwen3_5_text_engine import AnnaEngineError, GenerationPerfStats, TextGenerationResult


class _FailingStreamEngine:
    default_model_id = "fake-model"
    default_max_completion_tokens = 256
    model_family = "qwen3_5_text"

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def stream_chat(self, *_args, **_kwargs):
        def _generator():
            raise AnnaEngineError(
                "Insufficient free XPU memory before generation.",
                status_code=503,
                error_type="server_error",
                code="insufficient_device_memory",
            )
            yield  # pragma: no cover - unreachable sentinel for generator form

        return _generator()


class _CapturingEngine:
    default_model_id = "fake-model"
    model_family = "qwen3_5_text"

    def __init__(
        self,
        *,
        default_max_completion_tokens: int | None = 768,
        default_enable_thinking: bool = True,
        reasoning_format: str = "deepseek",
    ) -> None:
        self.default_max_completion_tokens = default_max_completion_tokens
        self.default_enable_thinking = default_enable_thinking
        self.reasoning_format = reasoning_format
        self.last_chat_config = None
        self.last_completion_config = None
        self.last_enable_thinking = None
        self.last_reasoning_format = None
        self.last_tools = None
        self.last_tool_choice = None
        self.last_parallel_tool_calls = None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def generate_chat(
        self,
        _messages,
        *,
        config,
        enable_thinking: bool = False,
        reasoning_format: str | None = None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    ):
        self.last_chat_config = config
        self.last_enable_thinking = enable_thinking
        self.last_reasoning_format = reasoning_format
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        self.last_parallel_tool_calls = parallel_tool_calls
        return TextGenerationResult(
            text="ok",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def generate_text(self, _prompt, *, config):
        self.last_completion_config = config
        return TextGenerationResult(
            text="ok",
            reasoning_text=None,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_list_app_routes_includes_openapi_and_completion_endpoints() -> None:
    routes = list_app_routes(create_app(_CapturingEngine()))

    assert ("/openapi.json", "HEAD, GET") in routes
    assert ("/docs", "HEAD, GET") in routes
    assert ("/healthz", "GET") in routes
    assert ("/v1/chat/completions", "POST") in routes
    assert ("/v1/completions", "POST") in routes
    assert ("/v1/audio/speech", "POST") in routes

def test_streaming_chat_returns_sse_error_frame_for_engine_failures() -> None:
    client = TestClient(create_app(_FailingStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"role": "assistant"' in response.text
    assert "event: error" in response.text
    assert '"code": "insufficient_device_memory"' in response.text
    assert "data: [DONE]" in response.text


def test_chat_completion_uses_engine_default_max_completion_tokens_when_request_omits_it() -> None:
    engine = _CapturingEngine(default_max_completion_tokens=1024)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert engine.last_chat_config is not None
    assert engine.last_chat_config.max_new_tokens == 1024


def test_chat_completion_request_max_completion_tokens_overrides_engine_default() -> None:
    engine = _CapturingEngine(default_max_completion_tokens=1024)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 64,
        },
    )

    assert response.status_code == 200
    assert engine.last_chat_config is not None
    assert engine.last_chat_config.max_new_tokens == 64


def test_chat_completion_forwards_function_calling_request_fields() -> None:
    engine = _CapturingEngine()
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": False,
        },
    )

    assert response.status_code == 200
    assert engine.last_tools is not None
    assert engine.last_tools[0].function.name == "get_weather"
    assert engine.last_tool_choice is not None
    assert engine.last_tool_choice.function.name == "get_weather"
    assert engine.last_parallel_tool_calls is False


def test_chat_completion_uses_engine_default_enable_thinking_when_request_omits_it() -> None:
    engine = _CapturingEngine(default_enable_thinking=False)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert engine.last_enable_thinking is False


def test_chat_completion_chat_template_kwargs_enable_thinking_overrides_engine_default() -> None:
    engine = _CapturingEngine(default_enable_thinking=True)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert response.status_code == 200
    assert engine.last_enable_thinking is False


def test_chat_completion_top_level_enable_thinking_overrides_chat_template_kwargs() -> None:
    engine = _CapturingEngine(default_enable_thinking=True)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert response.status_code == 200
    assert engine.last_enable_thinking is True


def test_chat_completion_leaves_max_completion_tokens_unset_when_engine_default_is_none() -> None:
    engine = _CapturingEngine(default_max_completion_tokens=None)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert engine.last_chat_config is not None
    assert engine.last_chat_config.max_new_tokens is None


def test_completion_uses_engine_default_max_completion_tokens_when_request_omits_it() -> None:
    engine = _CapturingEngine(default_max_completion_tokens=384)
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
        },
    )

    assert response.status_code == 200
    assert engine.last_completion_config is not None
    assert engine.last_completion_config.max_new_tokens == 384


def test_chat_completion_preflight_allows_cors() -> None:
    client = TestClient(create_app(_CapturingEngine()))

    response = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_chat_completion_response_includes_cors_header_for_browser_requests() -> None:
    client = TestClient(create_app(_CapturingEngine()))

    response = client.post(
        "/v1/chat/completions",
        headers={"Origin": "http://localhost:3000"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_streaming_chat_emits_fragmented_reasoning_deltas_immediately() -> None:
    class _ChunkedStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(text="", reasoning_text="用户", finish_reason=None)
            yield StreamEvent(text="", reasoning_text="要求", finish_reason=None)
            yield StreamEvent(text="", reasoning_text="我写。", finish_reason=None)
            yield StreamEvent(text="", reasoning_text=None, finish_reason="stop")

    client = TestClient(create_app(_ChunkedStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.text.count('"reasoning_content"') == 3
    assert '"reasoning_content": "用户"' in response.text
    assert '"reasoning_content": "要求"' in response.text
    assert '"reasoning_content": "我写。"' in response.text
    assert '"content": "用户要求我写。"' not in response.text


def test_streaming_chat_flushes_reasoning_before_content_switch() -> None:
    class _ChunkedStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None
        reasoning_format = "deepseek"

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(text="", reasoning_text="先分析", finish_reason=None)
            yield StreamEvent(text="最终答案。", reasoning_text=None, finish_reason="stop")

    client = TestClient(create_app(_ChunkedStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"reasoning_content": "先分析"' in response.text
    assert '"content": "最终答案。"' in response.text
    assert '"reasoning_content": "先分析", "content": "最终答案。"' not in response.text


def test_streaming_chat_emits_content_deltas_immediately() -> None:
    class _ChunkedStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None
        reasoning_format = "deepseek"

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(text="夏天", reasoning_text=None, finish_reason=None)
            yield StreamEvent(text="到了。", reasoning_text=None, finish_reason="stop")

    client = TestClient(create_app(_ChunkedStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.text.count('"content"') == 2
    assert '"content": "夏天"' in response.text
    assert '"content": "到了。"' in response.text


def test_streaming_chat_include_usage_emits_openai_final_usage_chunk() -> None:
    class _ChunkedStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None
        reasoning_format = "deepseek"

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(text="夏天", finish_reason=None)
            yield StreamEvent(text="", finish_reason="stop", prompt_tokens=16, completion_tokens=4)

    client = TestClient(create_app(_ChunkedStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks[0]["usage"] is None
    assert chunks[1]["usage"] is None
    assert chunks[2]["usage"] is None
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 16,
        "completion_tokens": 4,
        "total_tokens": 20,
    }


def test_streaming_completion_include_usage_accepts_vllm_flat_field() -> None:
    class _ChunkedCompletionEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_text(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(text="Hello", finish_reason=None)
            yield StreamEvent(text="", finish_reason="stop", prompt_tokens=8, completion_tokens=2)

    client = TestClient(create_app(_ChunkedCompletionEngine()))

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "stream": True,
            "stream_include_usage": True,
        },
    )

    assert response.status_code == 200
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks[0]["usage"] is None
    assert chunks[1]["usage"] is None
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }


def test_chat_completion_uses_engine_default_reasoning_format_when_request_omits_it() -> None:
    engine = _CapturingEngine(reasoning_format="deepseek")
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert engine.last_reasoning_format == "deepseek"


def test_chat_completion_request_reasoning_format_overrides_engine_default() -> None:
    engine = _CapturingEngine(reasoning_format="deepseek")
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_format": "deepseek",
        },
    )

    assert response.status_code == 200
    assert engine.last_reasoning_format == "deepseek"


def test_chat_completion_returns_openai_tool_calls_payload() -> None:
    class _ToolCallingEngine(_CapturingEngine):
        def generate_chat(
            self,
            _messages,
            *,
            config,
            enable_thinking: bool = False,
            reasoning_format: str | None = None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
        ):
            self.last_chat_config = config
            self.last_enable_thinking = enable_thinking
            self.last_reasoning_format = reasoning_format
            self.last_tools = tools
            self.last_tool_choice = tool_choice
            self.last_parallel_tool_calls = parallel_tool_calls
            return TextGenerationResult(
                text="",
                reasoning_text="先查天气。",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Shanghai\"}",
                        },
                    }
                ],
                finish_reason="tool_calls",
                prompt_tokens=4,
                completion_tokens=2,
            )

    client = TestClient(create_app(_ToolCallingEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == "先查天气。"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_streaming_chat_emits_tool_call_deltas_and_tool_calls_finish_reason() -> None:
    class _ToolCallingStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None
        reasoning_format = "deepseek"

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.core.function_calling import ToolCallDelta
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(
                text="",
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="call_123",
                        name="get_weather",
                        arguments="{\"location\":\"Shanghai\"}",
                    )
                ],
                finish_reason=None,
            )
            yield StreamEvent(text="", finish_reason="tool_calls")

    client = TestClient(create_app(_ToolCallingStreamEngine()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"tool_calls": [{"index": 0, "id": "call_123", "type": "function"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text


def test_chat_completion_logs_prefill_and_decode_metrics(caplog) -> None:
    class _ProfilingEngine(_CapturingEngine):
        def generate_chat(
            self,
            _messages,
            *,
            config,
            enable_thinking: bool = False,
            reasoning_format: str | None = None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
        ):
            self.last_chat_config = config
            self.last_enable_thinking = enable_thinking
            self.last_reasoning_format = reasoning_format
            self.last_tools = tools
            self.last_tool_choice = tool_choice
            self.last_parallel_tool_calls = parallel_tool_calls
            return TextGenerationResult(
                text="ok",
                reasoning_text=None,
                finish_reason="stop",
                prompt_tokens=50,
                completion_tokens=5,
                perf=GenerationPerfStats(
                    total_seconds=0.5,
                    prefill_seconds=0.25,
                    ttft_seconds=0.25,
                    decode_seconds=0.25,
                    prompt_tokens=50,
                    completion_tokens=5,
                    prefill_tokens_per_second=200.0,
                    decode_tokens=4,
                    decode_tokens_per_second=16.0,
                    total_tokens_per_second=10.0,
                ),
            )

    client = TestClient(create_app(_ProfilingEngine()))

    with caplog.at_level(logging.INFO, logger="anna.api.routes"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    log_lines = [record.message for record in caplog.records if record.name == "anna.api.routes"]
    assert any("prefill_seconds=0.250" in line for line in log_lines)
    assert any("decode_tokens_per_second=16.00" in line for line in log_lines)


def test_audio_speech_returns_wav_bytes_and_forwards_request_fields() -> None:
    class _SpeechEngine:
        default_model_id = "fake-tts-model"
        model_family = "qwen3_tts"

        def __init__(self) -> None:
            self.last_request = None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def synthesize_qwen3_tts_speech(
            self,
            text,
            *,
            config,
            language=None,
            speaker=None,
            instruct=None,
            ref_audio=None,
            ref_text=None,
            x_vector_only_mode=False,
        ):
            from anna.runtime.qwen3_tts_engine import Qwen3TTSSynthesisResult
            import numpy as np

            self.last_request = {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
                "config": config,
            }
            return Qwen3TTSSynthesisResult(
                audio=np.zeros(2400, dtype=np.float32),
                sample_rate=24000,
                duration_seconds=0.1,
                total_seconds=0.2,
            )

    engine = _SpeechEngine()
    client = TestClient(create_app(engine))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "fake-tts-model",
            "input": "Hello from Anna.",
            "voice": "Vivian",
            "language": "English",
            "instruct": "Speak warmly.",
            "reference_audio": "ref.wav",
            "reference_text": "Reference line.",
            "x_vector_only_mode": True,
            "response_format": "wav",
            "temperature": 0.8,
            "top_p": 0.95,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.content[:4] == b"RIFF"
    assert engine.last_request is not None
    assert engine.last_request["text"] == "Hello from Anna."
    assert engine.last_request["speaker"] == "Vivian"
    assert engine.last_request["language"] == "English"
    assert engine.last_request["instruct"] == "Speak warmly."
    assert engine.last_request["ref_audio"] == "ref.wav"
    assert engine.last_request["ref_text"] == "Reference line."
    assert engine.last_request["x_vector_only_mode"] is True
    assert engine.last_request["config"].temperature == 0.8
    assert engine.last_request["config"].top_p == 0.95


def test_audio_speech_rejects_models_without_speech_support() -> None:
    client = TestClient(create_app(_CapturingEngine()))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "fake-model",
            "input": "Hello from Anna.",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "The loaded qwen3_5_text model family does not support speech synthesis."
