"""Built-in continuous-batching presets and dynamic token-budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SchedulerProfile:
    """Resolved continuous-batching knobs for one named serving profile."""

    name: str
    max_batch_size: int
    batch_wait_ms: float
    prefill_interval_steps: int
    max_prefill_tokens: int
    max_decode_tokens: int
    max_waiting_requests: int
    dynamic_token_budget: bool
    # TTFT: skip coalescing wait when the GPU has no active work.
    skip_batch_wait_when_idle: bool
    # Fairness: force prefill admission after this many consecutive decode steps
    # even if prefill_interval_steps has not been reached, when the oldest waiter
    # has waited longer than max_queue_wait_ms (0 disables age-based force).
    max_queue_wait_ms: float
    # P2-#9: event-driven prefill insertion. When True, a waiting request is admitted
    # after the current decode step instead of waiting out prefill_interval_steps
    # (throughput profile: 4 steps ≈ ~320ms added TTFT per wave at 79ms/step).
    event_driven_prefill_insert: bool
    description: str


# Named profiles used by ``anna-serve --scheduler-profile``.
# Explicit CLI scheduler knobs always override profile defaults after resolution.
SCHEDULER_PROFILES: dict[str, SchedulerProfile] = {
    "interactive": SchedulerProfile(
        name="interactive",
        max_batch_size=2,
        batch_wait_ms=0.5,
        prefill_interval_steps=1,
        max_prefill_tokens=1024,
        max_decode_tokens=2048,
        max_waiting_requests=32,
        dynamic_token_budget=True,
        skip_batch_wait_when_idle=True,
        max_queue_wait_ms=50.0,
        event_driven_prefill_insert=False,
        description="Latency-first: small batches, short coalesce wait, frequent prefill inserts, TTFT-friendly idle admit.",
    ),
    "throughput": SchedulerProfile(
        name="throughput",
        max_batch_size=8,
        batch_wait_ms=8.0,
        prefill_interval_steps=4,
        max_prefill_tokens=2048,
        max_decode_tokens=4096,
        max_waiting_requests=128,
        dynamic_token_budget=True,
        # P2-#9: idle-skip inherited from interactive — waiting 8ms for coalescing
        # when the GPU has no active work only adds queue wait (measured max 18.5ms).
        skip_batch_wait_when_idle=True,
        max_queue_wait_ms=500.0,
        # P2-#9: insert waiting prefills immediately instead of every 4th decode step.
        event_driven_prefill_insert=True,
        description="Throughput-first: larger batches, higher queue depth, idle-skip coalescing, event-driven prefill inserts.",
    ),
}


def normalize_scheduler_profile(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "off", "manual"}:
        return "none"
    if normalized not in SCHEDULER_PROFILES:
        allowed = ", ".join(["none", *sorted(SCHEDULER_PROFILES)])
        raise ValueError(f"Unsupported scheduler profile: {value}. Expected one of: {allowed}.")
    return normalized


def get_scheduler_profile(name: str | None) -> SchedulerProfile | None:
    key = normalize_scheduler_profile(name)
    if key == "none":
        return None
    return SCHEDULER_PROFILES[key]


def resolve_scheduler_settings(
    *,
    profile: str | None = None,
    max_batch_size: int | None = None,
    batch_wait_ms: float | None = None,
    prefill_interval_steps: int | None = None,
    max_prefill_tokens: int | None = None,
    max_decode_tokens: int | None = None,
    max_waiting_requests: int | None = None,
    dynamic_token_budget: bool | None = None,
    skip_batch_wait_when_idle: bool | None = None,
    max_queue_wait_ms: float | None = None,
    event_driven_prefill_insert: bool | None = None,
) -> dict[str, Any]:
    """Merge an optional named profile with explicit overrides.

    ``None`` on an explicit knob means "use profile default, else hard default".
    """
    preset = get_scheduler_profile(profile)
    hard = {
        "max_batch_size": 1,
        "batch_wait_ms": 2.0,
        "prefill_interval_steps": 1,
        "max_prefill_tokens": 0,
        "max_decode_tokens": 0,
        "max_waiting_requests": 0,
        "dynamic_token_budget": False,
        "skip_batch_wait_when_idle": False,
        "max_queue_wait_ms": 0.0,
        "event_driven_prefill_insert": False,
    }
    base: dict[str, Any] = dict(hard)
    if preset is not None:
        base.update(
            {
                "max_batch_size": preset.max_batch_size,
                "batch_wait_ms": preset.batch_wait_ms,
                "prefill_interval_steps": preset.prefill_interval_steps,
                "max_prefill_tokens": preset.max_prefill_tokens,
                "max_decode_tokens": preset.max_decode_tokens,
                "max_waiting_requests": preset.max_waiting_requests,
                "dynamic_token_budget": preset.dynamic_token_budget,
                "skip_batch_wait_when_idle": preset.skip_batch_wait_when_idle,
                "max_queue_wait_ms": preset.max_queue_wait_ms,
                "event_driven_prefill_insert": preset.event_driven_prefill_insert,
            }
        )
    overrides: Mapping[str, Any] = {
        "max_batch_size": max_batch_size,
        "batch_wait_ms": batch_wait_ms,
        "prefill_interval_steps": prefill_interval_steps,
        "max_prefill_tokens": max_prefill_tokens,
        "max_decode_tokens": max_decode_tokens,
        "max_waiting_requests": max_waiting_requests,
        "dynamic_token_budget": dynamic_token_budget,
        "skip_batch_wait_when_idle": skip_batch_wait_when_idle,
        "max_queue_wait_ms": max_queue_wait_ms,
        "event_driven_prefill_insert": event_driven_prefill_insert,
    }
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    base["profile"] = preset.name if preset is not None else "none"
    base["max_batch_size"] = max(1, int(base["max_batch_size"]))
    base["batch_wait_ms"] = max(0.0, float(base["batch_wait_ms"]))
    base["prefill_interval_steps"] = max(1, int(base["prefill_interval_steps"]))
    base["max_prefill_tokens"] = max(0, int(base["max_prefill_tokens"]))
    base["max_decode_tokens"] = max(0, int(base["max_decode_tokens"]))
    base["max_waiting_requests"] = max(0, int(base["max_waiting_requests"]))
    base["dynamic_token_budget"] = bool(base["dynamic_token_budget"])
    base["skip_batch_wait_when_idle"] = bool(base["skip_batch_wait_when_idle"])
    base["max_queue_wait_ms"] = max(0.0, float(base["max_queue_wait_ms"]))
    base["event_driven_prefill_insert"] = bool(base["event_driven_prefill_insert"])
    return base


def compute_dynamic_token_budgets(
    *,
    base_prefill_tokens: int,
    base_decode_tokens: int,
    free_bytes: int | None,
    total_bytes: int | None,
    running_requests: int,
    waiting_requests: int,
    avg_running_seq_len: float,
) -> tuple[int, int]:
    """Scale configured token budgets from free memory and live load.

    Returns ``(max_prefill_tokens, max_decode_tokens)``. A base of ``0`` means
    "derive a soft default from free memory" rather than unlimited; callers that
    want unlimited should disable dynamic budgets.
    """
    free = max(0, int(free_bytes or 0))
    total = max(0, int(total_bytes or 0))
    free_ratio = 1.0 if total <= 0 else min(1.0, free / max(1, total))

    # Soft defaults when the operator left budgets at 0 (unlimited) but asked for dynamic.
    prefill = int(base_prefill_tokens) if base_prefill_tokens > 0 else 2048
    decode = int(base_decode_tokens) if base_decode_tokens > 0 else 4096

    # Memory pressure: shrink budgets as free ratio drops below 40%.
    if free_ratio < 0.40:
        scale = max(0.25, free_ratio / 0.40)
        prefill = max(256, int(prefill * scale))
        decode = max(256, int(decode * scale))
    elif free_ratio > 0.70 and running_requests <= 1 and waiting_requests <= 1:
        # Plenty of headroom and light load: allow slightly larger waves.
        prefill = int(prefill * 1.25)
        decode = int(decode * 1.25)

    # Long running sequences consume more KV; tighten decode packing.
    if avg_running_seq_len > 2048:
        decode = max(256, int(decode * 0.5))
    elif avg_running_seq_len > 1024:
        decode = max(256, int(decode * 0.75))

    # Many waiters: prefer larger prefill waves to clear the queue.
    if waiting_requests >= 4:
        prefill = int(prefill * 1.25)

    return max(0, prefill), max(0, decode)


# P1-#4: warmup shape table. Real chat requests arrive at ~13 prompt tokens; the
# legacy warmup (prefill_tokens=2, decode_steps=1) left the first request paying
# recompile / lazy-kernel-init stalls (0.88s TTFT). Cover the shapes that
# actually occur in serving, derived from the scheduler profile / chunk size
# instead of fixed constants.
_WARMUP_PREFILL_LADDER = (13, 64, 256, 2048)
_WARMUP_DECODE_STEPS = 8


def derive_warmup_shape_table(
    *,
    profile: str | None = None,
    max_batch_size: int = 1,
    chunk_size: int = 0,
    prefill_tokens: int | None = None,
    decode_steps: int | None = None,
) -> dict[str, object]:
    """Derive the post-load warmup shape table from scheduler profile / chunk size.

    Returns ``{"prefill_lengths": [...], "batch_sizes": [...], "decode_steps": int}``.

    - Prefill ladder: ``{13, 64, 256, 2048}`` (real chat shapes) always included;
      the configured prefill chunk size is added when chunked prefill is enabled;
      an explicit ``prefill_tokens`` override is always kept.
    - Batch ladder: powers of two up to ``max_batch_size`` (plus 1) so batched
      decode shapes are captured without warming shapes the scheduler never runs.
    - Decode steps: default 8 (exercises multi-step steady-state decode, turboquant
      dequant, and the fused LM-head top-k path).
    """
    chunk = max(0, int(chunk_size or 0))
    lengths = set(_WARMUP_PREFILL_LADDER)
    if chunk > 1:
        lengths.add(chunk)
    if prefill_tokens is not None and int(prefill_tokens) > 1:
        lengths.add(int(prefill_tokens))
    lengths = {max(2, int(length)) for length in lengths}
    # Keep the full ladder: long-prompt (2048) coverage trades a few warmup
    # seconds for eliminating the first long-prompt recompile stall.
    prefill_lengths = sorted(lengths)

    max_batch = max(1, int(max_batch_size or 1))
    batch_sizes = {1}
    power = 2
    while power <= max_batch:
        batch_sizes.add(power)
        power *= 2
    if max_batch > 1:
        batch_sizes.add(max_batch)

    resolved_decode_steps = _WARMUP_DECODE_STEPS if decode_steps is None else max(1, int(decode_steps))
    return {
        "prefill_lengths": prefill_lengths,
        "batch_sizes": sorted(batch_sizes),
        "decode_steps": resolved_decode_steps,
    }
