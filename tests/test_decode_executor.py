"""Phase 2: decode step graph executor tests (CPU-safe unless CUDA exists)."""

from __future__ import annotations

import pytest
import torch

from anna.runtime.decode_executor import (
    DecodeGraphRunner,
    DecodeGraphUnavailable,
    detect_graph_backend,
    normalize_decode_executor,
    resolve_decode_executor_mode,
)


def test_normalize_decode_executor() -> None:
    assert normalize_decode_executor(None) == "auto"
    assert normalize_decode_executor(" EAGER ") == "eager"
    assert normalize_decode_executor("Graph") == "graph"
    assert normalize_decode_executor("auto") == "auto"
    with pytest.raises(ValueError):
        normalize_decode_executor("cudagraph")


def test_detect_graph_backend_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANNA_DECODE_GRAPH_BACKEND", raising=False)
    assert detect_graph_backend("cpu") is None


def test_detect_graph_backend_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_DECODE_GRAPH_BACKEND", "off")
    assert detect_graph_backend("cuda") is None
    monkeypatch.setenv("ANNA_DECODE_GRAPH_BACKEND", "cuda")
    if not torch.cuda.is_available():
        assert detect_graph_backend("cuda") is None


def test_resolve_mode_auto_falls_back_to_eager_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANNA_DECODE_GRAPH_BACKEND", raising=False)
    mode, backend = resolve_decode_executor_mode("auto", "cpu")
    assert mode == "eager"
    assert backend is None
    mode, backend = resolve_decode_executor_mode("eager", "cpu")
    assert mode == "eager"
    if not torch.cuda.is_available():
        mode, backend = resolve_decode_executor_mode("graph", "cpu")
        assert mode == "eager"
        assert backend is None


def test_runner_replay_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity: graph replay must produce identical outputs to eager execution.

    Uses a fixed-step toy "model" whose state advances per call, mirroring the
    KV-cache-update-in-forward contract of the real decode step.
    """
    monkeypatch.setenv("ANNA_DECODE_GRAPH_BACKEND", "off")
    torch.manual_seed(0)
    weight = torch.randn(4, 4)

    state = {"step": 0}

    def step_fn(token_ids: torch.Tensor) -> torch.Tensor:
        state["step"] += 1
        return token_ids.float() @ weight + state["step"]

    # Static-shape decode inputs: [1, 4] token embedding row, one-hot per step.
    def tokens_for(step: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(step % 4), 4).float().unsqueeze(0)

    eager_outputs = []
    eager_state = {"step": 0}
    for step in range(5):
        tokens = tokens_for(step)
        eager_state["step"] += 1
        eager_outputs.append(tokens @ weight + eager_state["step"])

    graph_state = {"step": 0}

    def graph_step_fn(token_ids: torch.Tensor) -> torch.Tensor:
        graph_state["step"] += 1
        return token_ids.float() @ weight + graph_state["step"]

    runner = FakeBackendRunner()
    graph_outputs = []
    for step in range(5):
        tokens = tokens_for(step)
        graph_outputs.append(runner.step(graph_step_fn, tokens))

    for eager, graphed in zip(eager_outputs, graph_outputs, strict=True):
        assert torch.allclose(eager, graphed)


class FakeBackendRunner(DecodeGraphRunner):
    """Deterministic no-GPU stand-in for the device graph replay path."""

    def __init__(self) -> None:
        super().__init__("cpu", backend="fake")
        self._fake_output: torch.Tensor | None = None
        self._fake_step_fn = None
        self.replays = 0

    def _build_graph(self, step_fn, static_input: torch.Tensor) -> object:
        # Record the closure; "replay" re-executes it on the static buffer,
        # which is exactly what a device graph replay does semantically.
        self._fake_step_fn = step_fn
        return object()

    def _replay(self, input_ids: torch.Tensor) -> object:
        if self._static_input is None:
            raise DecodeGraphUnavailable("Decode graph was never captured.")
        self._static_input.copy_(input_ids)
        output = self._fake_step_fn(self._static_input)
        self._static_output = output
        self.replays += 1
        self.stats.replays += 1
        return output


def test_fake_backend_replay_shape_mismatch_falls_back() -> None:
    runner = FakeBackendRunner()
    first = runner.step(lambda ids: ids * 2, torch.tensor([[1]]))
    assert first.shape == (1, 1)
    with pytest.raises(DecodeGraphUnavailable):
        runner.step(lambda ids: ids * 2, torch.tensor([[1, 2]]))
    assert "shape_mismatch" in runner.stats.fallback_reasons


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires a CUDA device")
def test_cuda_graph_parity_against_eager() -> None:
    """Real capture/replay parity on CUDA (ops parity regression)."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    weight = torch.randn(16, 16, device=device)
    bias = torch.randn(16, device=device)

    # Eager reference: stateless per-step linear + step offset.
    eager = []
    for step in range(4):
        tokens = torch.tensor([[step]], device=device)
        eager.append(torch.nn.functional.linear(tokens.float(), weight, bias) + step)

    runner = DecodeGraphRunner(device, backend="cuda")
    graphed = []
    for step in range(4):
        tokens = torch.tensor([[step]], device=device)
        graphed.append(
            runner.step(
                lambda ids, s=step: torch.nn.functional.linear(ids.float(), weight, bias) + s,
                tokens,
            )
        )

    assert runner.captured
    assert runner.stats.replays >= 4
    for eager_out, graph_out in zip(eager, graphed, strict=True):
        assert torch.allclose(eager_out, graph_out)
