from __future__ import annotations

from fastapi.testclient import TestClient

from anna.api.app import create_app
from anna.runtime.engine import AnnaEngineError


class _FailingStreamEngine:
    default_model_id = "fake-model"

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
