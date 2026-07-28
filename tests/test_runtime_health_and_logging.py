from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from anna.api.app import create_app
from anna.core.function_calling import ToolCallDelta
from anna.core.logging import JsonLogFormatter, clear_trace_id, get_trace_id, set_trace_id, setup_logging
from anna.model.kernel_metrics import record_kernel_strategy, reset_kernel_strategy_hits, kernel_strategy_snapshot
from anna.runtime.qwen3_5_text_engine import AnnaEngineError, TextGenerationResult
from anna.runtime.runtime_health import PROCESS_ADMISSION_GATE, RuntimeAdmissionGate
from anna.runtime.service_metrics import AnnaServiceMetrics


class _OkEngine:
    default_model_id = "fake-model"
    default_max_completion_tokens = 64
    model_family = "qwen3_5_text"

    def __init__(self) -> None:
        self.admission_gate = RuntimeAdmissionGate()
        self.metrics = AnnaServiceMetrics()

    def health(self) -> dict:
        admission = self.admission_gate.to_health_dict()
        snap = self.metrics.snapshot()
        return {
            "status": admission["status"],
            "accepting_requests": admission["accepting_requests"],
            "runtime_admission": admission,
            "service_metrics": {
                "prefix_block_hit_rate": 0.0,
                "scheduler_queue_depth": snap.scheduler_queue_depth,
                "ttft_histogram": snap.ttft_histogram(),
                "itl_histogram": snap.itl_histogram(),
                "kernel_strategy_hits": snap.kernel_strategy_hits,
                "waiting_requests": snap.waiting_requests,
                "running_requests": snap.running_requests,
            },
        }

    def list_models(self) -> list[str]:
        return [self.default_model_id]

    def ensure_accepting_requests(self) -> None:
        snap = self.admission_gate.snapshot()
        if not snap.accepting_requests:
            raise AnnaEngineError(
                snap.degradation_reason or "degraded",
                status_code=503,
                error_type="server_error",
                code="runtime_degraded",
            )

    def generate_chat(self, *_args, **_kwargs):
        self.ensure_accepting_requests()
        return TextGenerationResult(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_admission_gate_rejects_new_requests_when_degraded() -> None:
    engine = _OkEngine()
    client = TestClient(create_app(engine))

    ok = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok.status_code == 200

    engine.admission_gate.enter_degraded(category="device_lost", reason="device lost for test")
    health = client.get("/healthz").json()
    assert health["status"] == "degraded"
    assert health["accepting_requests"] is False

    blocked = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert blocked.status_code == 503


def test_request_trace_id_header_roundtrip() -> None:
    engine = _OkEngine()
    client = TestClient(create_app(engine))

    response = client.get("/healthz", headers={"X-Request-Id": "trace-abc-123"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "trace-abc-123"
    assert response.json().get("trace_id") == "trace-abc-123"


def test_json_log_formatter_includes_trace_id() -> None:
    set_trace_id("tid-1")
    try:
        record = logging.LogRecord(
            name="anna.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        line = JsonLogFormatter().format(record)
        payload = json.loads(line)
        assert payload["message"] == "hello world"
        assert payload["trace_id"] == "tid-1"
        assert payload["level"] == "INFO"
    finally:
        clear_trace_id()


def test_setup_logging_json_mode_does_not_raise() -> None:
    setup_logging("INFO", log_format="json")
    logging.getLogger("anna.test").info("structured")
    setup_logging("INFO", log_format="text")


def test_kernel_strategy_hits_recorded() -> None:
    reset_kernel_strategy_hits()
    record_kernel_strategy("gqa_decode", "paged")
    record_kernel_strategy("gqa_decode", "paged")
    record_kernel_strategy("gdn_decode", "fused")
    hits = kernel_strategy_snapshot()
    assert hits["gqa_decode:paged"] == 2
    assert hits["gdn_decode:fused"] == 1

    metrics = AnnaServiceMetrics()
    snap = metrics.snapshot()
    assert snap.kernel_strategy_hits["gqa_decode:paged"] == 2


def test_tool_call_delta_streams_openai_multi_phase() -> None:
    delta = ToolCallDelta(
        index=0,
        id="call_1",
        name="get_weather",
        arguments='{"location":"Shanghai","unit":"c"}',
    )
    phases = delta.to_openai_stream_dicts()
    assert phases[0]["id"] == "call_1"
    assert phases[0]["function"]["name"] == "get_weather"
    assert phases[0]["function"]["arguments"] == ""
    joined = "".join(str(p["function"]["arguments"]) for p in phases[1:])
    assert joined == '{"location":"Shanghai","unit":"c"}'


def test_streaming_chat_emits_multi_phase_tool_call_deltas() -> None:
    class _ToolStreamEngine:
        default_model_id = "fake-model"
        default_max_completion_tokens = None
        reasoning_format = "deepseek"

        def health(self):
            return {"status": "ok"}

        def list_models(self):
            return [self.default_model_id]

        def stream_chat(self, *_args, **_kwargs):
            from anna.runtime.qwen3_5_text_engine import StreamEvent

            yield StreamEvent(
                text="",
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="call_xyz",
                        name="lookup",
                        arguments='{"q":"anna"}',
                    )
                ],
            )
            yield StreamEvent(
                text="",
                finish_reason="tool_calls",
                prompt_tokens=2,
                completion_tokens=3,
            )

    client = TestClient(create_app(_ToolStreamEngine()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200
    body = response.text
    # Header phase announces name with empty arguments.
    assert '"name": "lookup"' in body
    assert '"arguments": ""' in body
    # Argument fragments appear in later deltas.
    assert '"arguments": "{\\"q\\":\\"anna\\"}"' in body or '"arguments": "{"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert "data: [DONE]" in body


def test_process_admission_gate_clear() -> None:
    try:
        PROCESS_ADMISSION_GATE.enter_degraded(category="out_of_memory", reason="test")
        assert PROCESS_ADMISSION_GATE.accepting_requests is False
        PROCESS_ADMISSION_GATE.clear_degraded()
        assert PROCESS_ADMISSION_GATE.accepting_requests is True
    finally:
        PROCESS_ADMISSION_GATE.clear_degraded()
