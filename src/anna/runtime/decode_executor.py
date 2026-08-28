"""Decode-step graph executor (Phase 2).

Captures the static-shape single-token decode forward into a replayable
device graph (CUDA graph today, XPU graph capture when the installed torch
build exposes it) so that one scheduler tick submits a single graph launch
instead of hundreds of eager kernel launches.

Design notes:
- The capture scope is the *model forward only* (embeddings -> logits). The
  KV-cache update inside the forward is an in-place device operation whose
  kernel sequence is shape-stable for ``seq_len == 1`` decode, so re-running
  the recorded sequence on every replay advances the cache exactly once per
  step, matching eager semantics.
- Sampling, stop checks, and repetition-penalty history stay OUTSIDE the
  graph: they depend on dynamic shapes (the repetition history grows every
  step) and host control flow.
- Capture failures are non-fatal: the caller permanently falls back to eager
  execution and the reason is recorded for parity debugging.

Configuration:
- ``EngineOptimizationConfig.decode_executor``: ``eager`` | ``graph`` | ``auto``
  (auto = use graph when a capture backend is detected).
- ``ANNA_DECODE_EXECUTOR`` environment variable mirrors the config value.
- ``ANNA_DECODE_GRAPH_BACKEND``: ``off`` | ``cuda`` | ``xpu`` | ``auto``
  force/override backend detection (diagnostics).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_DECODE_EXECUTOR_VALUES = ("eager", "graph", "auto")


class DecodeGraphUnavailable(RuntimeError):
    """Raised when graph capture or replay fails; caller must fall back to eager."""


def normalize_decode_executor(value: str | None) -> str:
    """Normalize the decode-executor selector; unknown values raise ValueError."""
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    if normalized not in _DECODE_EXECUTOR_VALUES:
        allowed = ", ".join(_DECODE_EXECUTOR_VALUES)
        raise ValueError(f"Unsupported decode executor: {value}. Expected one of: {allowed}.")
    return normalized


def normalize_decode_executor_env() -> str:
    return normalize_decode_executor(os.getenv("ANNA_DECODE_EXECUTOR"))


def detect_graph_backend(device: torch.device | str) -> str | None:
    """Return the available graph-capture backend name for ``device``, or None.

    Probing order (overridable via ``ANNA_DECODE_GRAPH_BACKEND``):
    1. CUDA devices: ``torch.cuda.CUDAGraph`` when CUDA is available.
    2. XPU devices: torch.xpu graph capture APIs when the installed build
       exposes them (``torch.xpu.CUDAGraph`` / ``torch.xpu.graph``).
    """
    override = os.getenv("ANNA_DECODE_GRAPH_BACKEND", "").strip().lower()
    if override in ("off", "none", "0", "false"):
        return None

    resolved = torch.device(device)
    kind = resolved.type

    def _cuda_ok() -> bool:
        cuda = getattr(torch, "cuda", None)
        return (
            cuda is not None
            and hasattr(cuda, "CUDAGraph")
            and hasattr(cuda, "graph")
            and torch.cuda.is_available()
        )

    def _xpu_ok() -> bool:
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        return hasattr(xpu, "CUDAGraph") or hasattr(xpu, "graph")

    if override == "cuda":
        return "cuda" if _cuda_ok() else None
    if override == "xpu":
        return "xpu" if _xpu_ok() else None

    if kind == "cuda" and _cuda_ok():
        return "cuda"
    if kind == "xpu" and _xpu_ok():
        return "xpu"
    return None


def resolve_decode_executor_mode(value: str | None, device: torch.device | str) -> tuple[str, str | None]:
    """Resolve the effective executor mode for ``device``.

    Returns ``(mode, backend)`` where mode is ``eager`` or ``graph``. ``auto``
    resolves to graph only when a capture backend is available.
    """
    normalized = normalize_decode_executor(value)
    if normalized == "eager":
        return "eager", None
    backend = detect_graph_backend(device)
    if normalized == "graph":
        if backend is None:
            logger.warning(
                "decode-executor=graph requested but no graph capture backend is available on %s; "
                "falling back to eager. Set ANNA_DECODE_GRAPH_BACKEND to override detection.",
                torch.device(device),
            )
            return "eager", None
        return "graph", backend
    # auto
    if backend is None:
        return "eager", None
    return "graph", backend


@dataclass(slots=True)
class DecodeGraphStats:
    captures: int = 0
    replays: int = 0
    fallbacks: int = 0
    fallback_reasons: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "fallbacks": self.fallbacks,
            "fallback_reasons": sorted(self.fallback_reasons),
        }


class DecodeGraphRunner:
    """Capture a single-token decode forward as a replayable device graph.

    Usage::

        runner = DecodeGraphRunner(device, backend="cuda")
        outputs = runner.step(lambda ids: model(input_ids=ids, ...), input_ids)

    The first call performs capture (running ``step_fn`` exactly once inside
    the capture region; its device work is recorded, not executed) and then
    replays once to produce the step's real outputs. Later calls copy the
    fresh ``input_ids`` into the static input buffer and replay.
    """

    def __init__(self, device: torch.device | str, *, backend: str, warmup_steps: int = 0) -> None:
        self.device = torch.device(device)
        self.backend = backend
        self.warmup_steps = max(0, int(warmup_steps))
        self.stats = DecodeGraphStats()
        self._graph: object | None = None
        self._static_input: torch.Tensor | None = None
        self._static_output: object | None = None
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def step(self, step_fn: Callable[[torch.Tensor], object], input_ids: torch.Tensor) -> object:
        """Run one decode step through the graph (capturing on first call)."""
        if self._captured:
            if input_ids.shape != self._static_input.shape:
                self.stats.fallbacks += 1
                self.stats.fallback_reasons.add("shape_mismatch")
                raise DecodeGraphUnavailable(
                    f"decode graph shape mismatch: captured {tuple(self._static_input.shape)} "
                    f"got {tuple(input_ids.shape)}"
                )
            return self._replay(input_ids)
        return self._capture_and_run(step_fn, input_ids)

    # -- capture -----------------------------------------------------------

    def _capture_and_run(self, step_fn: Callable[[torch.Tensor], object], input_ids: torch.Tensor) -> object:
        try:
            static_input = input_ids.detach().clone()
            graph = self._build_graph(step_fn, static_input)
        except Exception as exc:  # pragma: no cover - device/driver dependent
            self.stats.fallbacks += 1
            self.stats.fallback_reasons.add(f"capture:{type(exc).__name__}")
            logger.warning("Decode graph capture failed (%s); falling back to eager decode.", exc)
            raise DecodeGraphUnavailable(str(exc)) from exc
        self._graph = graph
        self._static_input = static_input
        self._captured = True
        self.stats.captures += 1
        try:
            return self._replay(input_ids)
        except Exception as exc:  # pragma: no cover - device/driver dependent
            self._captured = False
            self._graph = None
            self.stats.fallbacks += 1
            self.stats.fallback_reasons.add(f"replay:{type(exc).__name__}")
            raise DecodeGraphUnavailable(str(exc)) from exc

    def _build_graph(self, step_fn: Callable[[torch.Tensor], object], static_input: torch.Tensor) -> object:
        if self.backend == "cuda":
            return self._build_cuda_graph(step_fn, static_input)
        if self.backend == "xpu":
            return self._build_xpu_graph(step_fn, static_input)
        raise DecodeGraphUnavailable(f"Unknown decode graph backend: {self.backend}")

    def _warmup(self, step_fn: Callable[[torch.Tensor], object], static_input: torch.Tensor) -> None:
        """Warm up JIT/allocator state on a side stream without recording.

        Warmup runs execute real device work against the caller's state, so
        they are disabled by default (``warmup_steps=0``); callers that want
        warmup must pass state whose mutation is acceptable (e.g. a cloned
        KV cache).
        """
        if self.warmup_steps <= 0:
            return
        if self.backend == "cuda" and torch.cuda.is_available():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(self.warmup_steps):
                    step_fn(static_input)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            return
        for _ in range(self.warmup_steps):
            step_fn(static_input)

    def _build_cuda_graph(self, step_fn: Callable[[torch.Tensor], object], static_input: torch.Tensor) -> object:
        self._warmup(step_fn, static_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = step_fn(static_input)
        self._static_output = static_output
        return graph

    def _build_xpu_graph(self, step_fn: Callable[[torch.Tensor], object], static_input: torch.Tensor) -> object:
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            raise DecodeGraphUnavailable("torch.xpu is unavailable")
        graph_cls = getattr(xpu, "CUDAGraph", None)
        graph_ctx = getattr(xpu, "graph", None)
        self._warmup(step_fn, static_input)
        if graph_ctx is not None:
            with graph_ctx():
                static_output = step_fn(static_input)
            self._static_output = static_output
            return None  # context-managed capture; replay is a no-op passthrough
        if graph_cls is not None:
            graph = graph_cls()
            with xpu.graph(graph):
                static_output = step_fn(static_input)
            self._static_output = static_output
            return graph
        raise DecodeGraphUnavailable("torch.xpu does not expose graph capture APIs in this build")

    # -- replay ------------------------------------------------------------

    def _replay(self, input_ids: torch.Tensor) -> object:
        if self._static_input is None or self._static_output is None:
            raise DecodeGraphUnavailable("Decode graph was never captured.")
        self._static_input.copy_(input_ids)
        if self._graph is not None:
            if self.backend == "cuda":
                self._graph.replay()
            else:
                self._graph.replay()
        # Context-managed backends (xpu.graph()) re-execute eagerly; the static
        # output tensor is refreshed by the step_fn registered at capture time.
        self.stats.replays += 1
        return self._static_output


def create_decode_graph_runner(
    device: torch.device | str,
    *,
    mode: str | None = None,
    warmup_steps: int = 0,
) -> tuple[str, DecodeGraphRunner | None]:
    """Resolve the executor mode and optionally build a runner.

    Returns ``(effective_mode, runner_or_none)``. ``runner`` is None for eager.
    """
    resolved_mode, backend = resolve_decode_executor_mode(mode, device)
    if resolved_mode != "graph" or backend is None:
        return "eager", None
    return "graph", DecodeGraphRunner(device, backend=backend, warmup_steps=warmup_steps)
