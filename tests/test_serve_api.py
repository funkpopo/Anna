from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from anna.api.app import create_app
from anna.runtime.engine import StreamEvent, TextGenerationResult


class _StubEngine:
    def __init__(self) -> None:
        self.default_model_id = "anna-test"

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def generate_text(self, prompt: str, *, config) -> TextGenerationResult:
        return TextGenerationResult(
            text=f"echo:{prompt}",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def generate_chat(self, messages: list[object], *, config, enable_thinking: bool = False) -> TextGenerationResult:
        return TextGenerationResult(
            text="hello",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def stream_text(self, prompt: str, *, config) -> Iterator[StreamEvent]:
        yield StreamEvent(text="echo", finish_reason=None)
        yield StreamEvent(text="", finish_reason="stop")

    def stream_chat(self, messages: list[object], *, config, enable_thinking: bool = False) -> Iterator[StreamEvent]:
        yield StreamEvent(text="hello", finish_reason=None)
        yield StreamEvent(text="", finish_reason="stop")


def _client(*, api_key: str | None = "secret") -> TestClient:
    return TestClient(create_app(_StubEngine(), api_key=api_key))


def test_healthz_is_open_without_api_key() -> None:
    with _client(api_key="secret") as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_v1_models_rejects_missing_api_key() -> None:
    with _client(api_key="secret") as client:
        response = client.get("/v1/models")

    assert response.status_code == 401
    payload = response.json()["error"]
    assert payload["type"] == "authentication_error"
    assert payload["code"] == "missing_api_key"


def test_v1_models_accepts_x_api_key_header() -> None:
    with _client(api_key="secret") as client:
        response = client.get("/v1/models", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "anna-test"


def test_completions_returns_model_not_found_error() -> None:
    with _client(api_key="secret") as client:
        response = client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "missing-model", "prompt": "hi"},
        )

    assert response.status_code == 404
    payload = response.json()["error"]
    assert payload["type"] == "invalid_request_error"
    assert payload["code"] == "model_not_found"
    assert payload["param"] == "model"


def test_completions_wraps_request_validation_errors() -> None:
    with _client(api_key="secret") as client:
        response = client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "anna-test", "prompt": "hi", "top_p": 1.5},
        )

    assert response.status_code == 400
    payload = response.json()["error"]
    assert payload["type"] == "invalid_request_error"
    assert payload["code"] == "invalid_request"
    assert payload["param"] == "top_p"
    assert "top_p" in payload["message"]
