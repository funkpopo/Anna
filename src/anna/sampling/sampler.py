from __future__ import annotations

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor | None,
    penalty: float,
) -> torch.Tensor:
    if generated_ids is None or generated_ids.numel() == 0 or penalty == 1.0:
        return logits

    output = logits.clone()
    if generated_ids.device != output.device:
        generated_ids = generated_ids.to(device=output.device)
    unique_ids = torch.unique(generated_ids)
    if output.ndim == 1:
        values = output.index_select(0, unique_ids)
        adjusted = torch.where(values < 0, values * penalty, values / penalty)
        output.index_copy_(0, unique_ids, adjusted)
        return output

    # Batched [B, V]: apply the same history set to every row.
    values = output.index_select(-1, unique_ids)
    adjusted = torch.where(values < 0, values * penalty, values / penalty)
    output.index_copy_(-1, unique_ids, adjusted)
    return output


def apply_presence_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor | None,
    penalty: float,
) -> torch.Tensor:
    if generated_ids is None or generated_ids.numel() == 0 or penalty == 0.0:
        return logits

    output = logits.clone()
    if generated_ids.device != output.device:
        generated_ids = generated_ids.to(device=output.device)
    unique_ids = torch.unique(generated_ids)
    if output.ndim == 1:
        values = output.index_select(0, unique_ids) - penalty
        output.index_copy_(0, unique_ids, values)
        return output

    values = output.index_select(-1, unique_ids) - penalty
    output.index_copy_(-1, unique_ids, values)
    return output


def apply_repetition_penalty_rows(
    logits: torch.Tensor,
    generated_ids_rows: list[torch.Tensor | None],
    penalties: list[float] | float,
) -> torch.Tensor:
    """Apply per-row repetition penalty to batched logits ``[B, V]``."""
    if logits.ndim != 2:
        raise ValueError("apply_repetition_penalty_rows expects logits with shape [B, V]")
    batch = logits.shape[0]
    if len(generated_ids_rows) != batch:
        raise ValueError("generated_ids_rows length must match batch size")
    if isinstance(penalties, float):
        penalty_values = [penalties] * batch
    else:
        if len(penalties) != batch:
            raise ValueError("penalties length must match batch size")
        penalty_values = list(penalties)

    if all(p == 1.0 or ids is None or (isinstance(ids, torch.Tensor) and ids.numel() == 0) for p, ids in zip(penalty_values, generated_ids_rows)):
        return logits

    output = logits.clone()
    for row_idx, (generated_ids, penalty) in enumerate(zip(generated_ids_rows, penalty_values)):
        if generated_ids is None or generated_ids.numel() == 0 or penalty == 1.0:
            continue
        row = apply_repetition_penalty(output[row_idx], generated_ids, penalty)
        output[row_idx].copy_(row)
    return output


def apply_presence_penalty_rows(
    logits: torch.Tensor,
    generated_ids_rows: list[torch.Tensor | None],
    penalties: list[float] | float,
) -> torch.Tensor:
    """Apply per-row presence penalty to batched logits ``[B, V]``."""
    if logits.ndim != 2:
        raise ValueError("apply_presence_penalty_rows expects logits with shape [B, V]")
    batch = logits.shape[0]
    if len(generated_ids_rows) != batch:
        raise ValueError("generated_ids_rows length must match batch size")
    if isinstance(penalties, float):
        penalty_values = [penalties] * batch
    else:
        if len(penalties) != batch:
            raise ValueError("penalties length must match batch size")
        penalty_values = list(penalties)

    if all(p == 0.0 or ids is None or (isinstance(ids, torch.Tensor) and ids.numel() == 0) for p, ids in zip(penalty_values, generated_ids_rows)):
        return logits

    output = logits.clone()
    for row_idx, (generated_ids, penalty) in enumerate(zip(generated_ids_rows, penalty_values)):
        if generated_ids is None or generated_ids.numel() == 0 or penalty == 0.0:
            continue
        row = apply_presence_penalty(output[row_idx], generated_ids, penalty)
        output[row_idx].copy_(row)
    return output


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k=top_k)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    max_prob = torch.max(probs, dim=-1, keepdim=True).values
    keep = probs >= max_prob * min_p
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))


def _normalize_logits_batch(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 1 or 2, got shape {tuple(logits.shape)}")
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    generated_ids: torch.Tensor | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Sample one token from rank-1 logits, or a batch from rank-2 ``[B, V]`` logits.

    When ``logits`` is ``[B, V]``, ``generated_ids`` (if provided) is treated as a shared
    history applied to every row. Prefer :func:`sample_next_tokens` for per-row histories
    or heterogeneous sampling hyper-parameters.
    """
    batched = _normalize_logits_batch(logits)
    next_logits = apply_repetition_penalty(batched, generated_ids, repetition_penalty)
    next_logits = apply_presence_penalty(next_logits, generated_ids, presence_penalty)

    if temperature <= 0.0:
        sampled = torch.argmax(next_logits, dim=-1)
    else:
        next_logits = next_logits / temperature
        next_logits = apply_top_k(next_logits, top_k)
        next_logits = apply_top_p(next_logits, top_p)
        next_logits = apply_min_p(next_logits, min_p)
        probs = torch.softmax(next_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if logits.ndim == 1:
        return sampled.reshape(())
    return sampled


def sample_next_token_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    min_p: float = 0.0,
) -> torch.Tensor:
    if candidate_logits.shape != candidate_token_ids.shape:
        raise ValueError("candidate_logits and candidate_token_ids must have the same shape")
    if candidate_logits.numel() == 0:
        raise ValueError("candidate logits must not be empty")

    batched_logits = _normalize_logits_batch(candidate_logits)
    batched_ids = _normalize_logits_batch(candidate_token_ids.to(dtype=torch.long))

    if temperature <= 0.0:
        selected = torch.argmax(batched_logits, dim=-1)
        sampled = batched_ids.gather(dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
    else:
        next_logits = batched_logits / temperature
        next_logits = apply_top_p(next_logits, top_p)
        next_logits = apply_min_p(next_logits, min_p)
        probs = torch.softmax(next_logits, dim=-1)
        selected = torch.multinomial(probs, num_samples=1)
        sampled = batched_ids.gather(dim=-1, index=selected).squeeze(-1)

    if candidate_logits.ndim == 1:
        return sampled.reshape(())
    return sampled


def sample_next_tokens(
    logits: torch.Tensor,
    *,
    generated_ids_rows: list[torch.Tensor | None] | None = None,
    temperatures: list[float] | float = 0.7,
    top_ps: list[float] | float = 0.8,
    top_ks: list[int] | int = 20,
    min_ps: list[float] | float = 0.0,
    presence_penalties: list[float] | float = 0.0,
    repetition_penalties: list[float] | float = 1.0,
) -> torch.Tensor:
    """Batched next-token sampling for continuous decode.

    ``logits`` must be ``[B, V]``. Sampling hyper-parameters may be scalars (shared) or
    per-row lists. When parameters are homogeneous the top-k / top-p / multinomial path
    stays fully vectorized; heterogeneous rows are grouped so each group still batches.
    """
    batched = _normalize_logits_batch(logits)
    batch = batched.shape[0]
    if batch == 0:
        return torch.empty((0,), dtype=torch.long, device=batched.device)

    def _as_list(value: list | float | int, cast):
        if isinstance(value, list):
            if len(value) != batch:
                raise ValueError("per-row sampling parameter length must match batch size")
            return [cast(v) for v in value]
        return [cast(value)] * batch

    temps = _as_list(temperatures, float)
    top_p_values = _as_list(top_ps, float)
    top_k_values = _as_list(top_ks, int)
    min_p_values = _as_list(min_ps, float)
    presence_values = _as_list(presence_penalties, float)
    repetition_values = _as_list(repetition_penalties, float)
    history_rows = generated_ids_rows if generated_ids_rows is not None else [None] * batch
    if len(history_rows) != batch:
        raise ValueError("generated_ids_rows length must match batch size")

    next_logits = apply_repetition_penalty_rows(batched, history_rows, repetition_values)
    next_logits = apply_presence_penalty_rows(next_logits, history_rows, presence_values)

    result = torch.empty((batch,), dtype=torch.long, device=batched.device)

    # Partition rows by sampling signature so each partition stays vectorized.
    groups: dict[tuple[float, float, int, float], list[int]] = {}
    for row_idx in range(batch):
        key = (temps[row_idx], top_p_values[row_idx], top_k_values[row_idx], min_p_values[row_idx])
        groups.setdefault(key, []).append(row_idx)

    for (temperature, top_p, top_k, min_p), rows in groups.items():
        row_tensor = torch.tensor(rows, dtype=torch.long, device=batched.device)
        group_logits = next_logits.index_select(0, row_tensor)
        if temperature <= 0.0:
            sampled = torch.argmax(group_logits, dim=-1)
        else:
            scaled = group_logits / temperature
            scaled = apply_top_k(scaled, top_k)
            scaled = apply_top_p(scaled, top_p)
            scaled = apply_min_p(scaled, min_p)
            probs = torch.softmax(scaled, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        result.index_copy_(0, row_tensor, sampled.to(dtype=torch.long))

    return result


def sample_next_tokens_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperatures: list[float] | float = 0.7,
    top_ps: list[float] | float = 0.8,
    min_ps: list[float] | float = 0.0,
) -> torch.Tensor:
    """Batched candidate sampling for fused LM-head top-k outputs ``[B, K]``."""
    if candidate_logits.shape != candidate_token_ids.shape:
        raise ValueError("candidate_logits and candidate_token_ids must have the same shape")
    batched_logits = _normalize_logits_batch(candidate_logits)
    batched_ids = _normalize_logits_batch(candidate_token_ids.to(dtype=torch.long))
    batch = batched_logits.shape[0]
    if batch == 0:
        return torch.empty((0,), dtype=torch.long, device=batched_logits.device)

    def _as_list(value: list | float, cast):
        if isinstance(value, list):
            if len(value) != batch:
                raise ValueError("per-row sampling parameter length must match batch size")
            return [cast(v) for v in value]
        return [cast(value)] * batch

    temps = _as_list(temperatures, float)
    top_p_values = _as_list(top_ps, float)
    min_p_values = _as_list(min_ps, float)

    result = torch.empty((batch,), dtype=torch.long, device=batched_logits.device)
    groups: dict[tuple[float, float, float], list[int]] = {}
    for row_idx in range(batch):
        key = (temps[row_idx], top_p_values[row_idx], min_p_values[row_idx])
        groups.setdefault(key, []).append(row_idx)

    for (temperature, top_p, min_p), rows in groups.items():
        row_tensor = torch.tensor(rows, dtype=torch.long, device=batched_logits.device)
        group_logits = batched_logits.index_select(0, row_tensor)
        group_ids = batched_ids.index_select(0, row_tensor)
        if temperature <= 0.0:
            selected = torch.argmax(group_logits, dim=-1)
            sampled = group_ids.gather(dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
        else:
            scaled = group_logits / temperature
            scaled = apply_top_p(scaled, top_p)
            scaled = apply_min_p(scaled, min_p)
            probs = torch.softmax(scaled, dim=-1)
            selected = torch.multinomial(probs, num_samples=1)
            sampled = group_ids.gather(dim=-1, index=selected).squeeze(-1)
        result.index_copy_(0, row_tensor, sampled.to(dtype=torch.long))

    return result


def token_ids_to_host(token_tensor: torch.Tensor) -> list[int]:
    """Single host sync for one or more device token ids."""
    flat = token_tensor.detach().reshape(-1)
    if flat.device.type == "cpu":
        return [int(value) for value in flat.tolist()]
    return [int(value) for value in flat.to(device="cpu").tolist()]
