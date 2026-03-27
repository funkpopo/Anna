from __future__ import annotations

from fastapi.testclient import TestClient

from anna.api.app import create_app
from anna.runtime.engine import AnnaEngineError, TextGenerationResult


class _FailingStreamEngine:
    default_model_id = "fake-model"
    default_max_completion_tokens = 256

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

    def __init__(
        self,
        *,
        default_max_completion_tokens: int | None = 768,
        reasoning_format: str = "deepseek",
    ) -> None:
        self.default_max_completion_tokens = default_max_completion_tokens
        self.reasoning_format = reasoning_format
        self.last_chat_config = None
        self.last_completion_config = None
        self.last_reasoning_format = None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def generate_chat(self, _messages, *, config, enable_thinking: bool = False, reasoning_format: str | None = None):
        self.last_chat_config = config
        self.last_reasoning_format = reasoning_format
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


def test_streaming_chat_coalesces_fragmented_reasoning_deltas() -> None:
    class _ChunkedStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None

        def health(self) -> dict[str, str]:
            return {"status": "ok"}

        def list_models(self) -> list[str]:
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.engine import StreamEvent

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
    assert response.text.count('"reasoning_content"') == 1
    assert '"reasoning_content": "用户要求我写。"' in response.text
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
            from anna.runtime.engine import StreamEvent

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
