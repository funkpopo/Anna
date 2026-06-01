from __future__ import annotations

from collections.abc import Sequence

import torch

from anna.core.hotpath_events import record_sampler_full_vocab_sort
from anna.sampling.params import SamplingBatchParams


def _row_index_tensor(
    rows: Sequence[int],
    *,
    device: torch.device | str,
    cached: torch.Tensor | None = None,
) -> torch.Tensor:
    if cached is not None:
        return cached.to(device=device, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long, device=device)


def _candidate_positions(candidate_count: int, *, device: torch.device | str) -> torch.Tensor:
    return torch.arange(candidate_count, dtype=torch.long, device=device).reshape(1, candidate_count)


def _partition_bounded_topk_rows(
    rows: tuple[int, ...],
    indices: torch.Tensor,
    params: SamplingBatchParams,
    *,
    candidate_count: int,
) -> tuple[tuple[int, ...], torch.Tensor | None, tuple[int, ...], torch.Tensor | None]:
    if not rows:
        return (), indices, (), indices
    if all(params.top_k_values[idx] < candidate_count for idx in rows):
        return rows, indices, (), None
    if all(params.top_k_values[idx] >= candidate_count for idx in rows):
        return (), None, rows, indices
    bounded = tuple(idx for idx in rows if params.top_k_values[idx] < candidate_count)
    unbounded = tuple(idx for idx in rows if params.top_k_values[idx] >= candidate_count)
    return bounded, None, unbounded, None


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
    values = output.index_select(0, unique_ids)
    adjusted = torch.where(values < 0, values * penalty, values / penalty)
    output.index_copy_(0, unique_ids, adjusted)
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
    values = output.index_select(0, unique_ids) - penalty
    output.index_copy_(0, unique_ids, values)
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
    return sample_next_token_batch(
        logits,
        generated_ids_batch=None if generated_ids is None else (generated_ids,),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )


def sample_next_token_batch(
    logits: torch.Tensor,
    *,
    generated_ids_batch: Sequence[torch.Tensor | None] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    if logits.ndim == 0:
        raise ValueError("logits must include a vocab dimension")

    output_shape = logits.shape[:-1]
    vocab_size = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab_size)
    row_count = int(flat_logits.shape[0])
    next_logits = flat_logits

    needs_penalty = repetition_penalty != 1.0 or presence_penalty != 0.0
    if needs_penalty:
        histories: Sequence[torch.Tensor | None]
        if generated_ids_batch is None:
            histories = (None,) * row_count
        else:
            histories = generated_ids_batch
        if len(histories) != row_count:
            raise ValueError(
                f"generated_ids_batch length {len(histories)} does not match flattened batch size {row_count}."
            )
        rows: list[torch.Tensor] = []
        for row_logits, generated_ids in zip(flat_logits.unbind(0), histories, strict=True):
            row_logits = apply_repetition_penalty(row_logits, generated_ids, repetition_penalty)
            row_logits = apply_presence_penalty(row_logits, generated_ids, presence_penalty)
            rows.append(row_logits)
        next_logits = torch.stack(rows, dim=0)

    if temperature <= 0.0:
        return torch.argmax(next_logits, dim=-1).reshape(output_shape)
    if top_k == 1:
        return torch.argmax(next_logits, dim=-1).reshape(output_shape)

    next_logits = next_logits / temperature
    if 0 < top_k < vocab_size:
        candidate_logits, candidate_token_ids = torch.topk(next_logits, k=top_k, dim=-1)
        return sample_next_token_batch_from_candidates(
            candidate_logits,
            candidate_token_ids,
            temperature=1.0,
            top_p=top_p,
            min_p=min_p,
            candidates_are_sorted=True,
        ).reshape(output_shape)

    next_logits = apply_top_k(next_logits, top_k)
    if top_p < 1.0:
        record_sampler_full_vocab_sort("top_p_full_logits_sort", count=row_count)
    next_logits = apply_top_p(next_logits, top_p)
    next_logits = apply_min_p(next_logits, min_p)
    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).reshape(output_shape)


def sample_next_token_batch_with_params(
    logits: torch.Tensor,
    params: SamplingBatchParams,
    *,
    generated_ids_batch: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    if logits.ndim == 0:
        raise ValueError("logits must include a vocab dimension")

    output_shape = logits.shape[:-1]
    vocab_size = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab_size)
    row_count = int(flat_logits.shape[0])
    if params.batch_size != row_count:
        raise ValueError(f"sampling params batch size {params.batch_size} does not match row count {row_count}.")
    if generated_ids_batch is not None and len(generated_ids_batch) != row_count:
        raise ValueError(
            f"generated_ids_batch length {len(generated_ids_batch)} does not match flattened batch size {row_count}."
        )

    histories = generated_ids_batch
    next_logits = _apply_batched_penalties_with_params(flat_logits, histories, params)
    temperatures = params.temperature.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count)
    top_p = params.top_p.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count).clamp(min=0.0, max=1.0)
    top_k = params.top_k.to(device=flat_logits.device, dtype=torch.long).reshape(row_count)
    min_p = params.min_p.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count).clamp(min=0.0, max=1.0)

    selected_tokens = torch.empty((row_count,), dtype=torch.long, device=flat_logits.device)
    greedy_rows = params.greedy_rows
    if greedy_rows:
        greedy_indices = _row_index_tensor(greedy_rows, device=flat_logits.device, cached=params.greedy_indices)
        greedy_selected = torch.argmax(next_logits, dim=-1)
        selected_tokens.index_copy_(0, greedy_indices, greedy_selected.index_select(0, greedy_indices))

    top1_rows = params.top1_rows

    if top1_rows:
        top1_indices = _row_index_tensor(top1_rows, device=flat_logits.device, cached=params.top1_indices)
        top1_selected = torch.argmax(next_logits, dim=-1)
        selected_tokens.index_copy_(0, top1_indices, top1_selected.index_select(0, top1_indices))

    topk_plain_rows, topk_plain_indices, full_plain_from_topk, full_plain_from_topk_indices = _partition_bounded_topk_rows(
        params.topk_plain_rows,
        params.topk_plain_indices,
        params,
        candidate_count=vocab_size,
    )
    topk_top_p_rows, topk_top_p_indices, full_top_p_from_topk, full_top_p_from_topk_indices = _partition_bounded_topk_rows(
        params.topk_top_p_rows,
        params.topk_top_p_indices,
        params,
        candidate_count=vocab_size,
    )
    full_plain_rows = params.full_plain_rows + full_plain_from_topk
    full_top_p_rows = params.full_top_p_rows + full_top_p_from_topk

    def _sample_topk_rows(rows: tuple[int, ...], *, indices: torch.Tensor | None, apply_top_p: bool) -> None:
        if not rows:
            return
        topk_indices = _row_index_tensor(rows, device=flat_logits.device, cached=indices)
        max_top_k = max(params.top_k_values[idx] for idx in rows)
        row_logits = next_logits.index_select(0, topk_indices)
        candidate_logits, candidate_token_ids = torch.topk(row_logits, k=max_top_k, dim=-1)
        active_top_k = top_k.index_select(0, topk_indices).clamp(min=1, max=max_top_k)
        keep = _candidate_positions(max_top_k, device=flat_logits.device) < active_top_k.reshape(-1, 1)
        candidate_logits = torch.where(keep, candidate_logits, torch.full_like(candidate_logits, float("-inf")))
        sampled_tokens = _sample_candidate_logits_with_tensors(
            candidate_logits,
            candidate_token_ids,
            temperatures=temperatures.index_select(0, topk_indices),
            top_p=top_p.index_select(0, topk_indices),
            min_p=min_p.index_select(0, topk_indices),
            apply_top_p=apply_top_p,
            candidates_are_sorted=True,
        )
        selected_tokens.index_copy_(0, topk_indices, sampled_tokens)

    _sample_topk_rows(topk_plain_rows, indices=topk_plain_indices, apply_top_p=False)
    _sample_topk_rows(topk_top_p_rows, indices=topk_top_p_indices, apply_top_p=True)

    def _sample_full_rows(rows: tuple[int, ...], *, indices: torch.Tensor | None, apply_top_p: bool) -> None:
        if not rows:
            return
        full_indices = _row_index_tensor(rows, device=flat_logits.device, cached=indices)
        full_logits = next_logits.index_select(0, full_indices)
        full_token_ids = torch.arange(vocab_size, dtype=torch.long, device=flat_logits.device).reshape(1, vocab_size)
        full_token_ids = full_token_ids.expand(len(rows), vocab_size)
        if apply_top_p:
            record_sampler_full_vocab_sort("top_p_full_logits_sort", count=len(rows))
        sampled_tokens = _sample_candidate_logits_with_tensors(
            full_logits,
            full_token_ids,
            temperatures=temperatures.index_select(0, full_indices),
            top_p=top_p.index_select(0, full_indices),
            min_p=min_p.index_select(0, full_indices),
            apply_top_p=apply_top_p,
            candidates_are_sorted=False,
        )
        selected_tokens.index_copy_(0, full_indices, sampled_tokens)

    full_plain_indices = (
        params.full_plain_indices
        if not full_plain_from_topk
        else None if params.full_plain_rows else full_plain_from_topk_indices
    )
    full_top_p_indices = (
        params.full_top_p_indices
        if not full_top_p_from_topk
        else None if params.full_top_p_rows else full_top_p_from_topk_indices
    )
    _sample_full_rows(full_plain_rows, indices=full_plain_indices, apply_top_p=False)
    _sample_full_rows(full_top_p_rows, indices=full_top_p_indices, apply_top_p=True)

    return selected_tokens.reshape(output_shape)


def _apply_batched_penalties_with_params(
    flat_logits: torch.Tensor,
    histories: Sequence[torch.Tensor | None] | None,
    params: SamplingBatchParams,
) -> torch.Tensor:
    if histories is None or not params.penalty_rows:
        return flat_logits

    output: torch.Tensor | None = None
    for row_idx in params.penalty_rows:
        generated_ids = histories[row_idx]
        repetition_penalty = params.repetition_penalty_values[row_idx]
        presence_penalty = params.presence_penalty_values[row_idx]
        if generated_ids is None or generated_ids.numel() == 0:
            continue
        row_logits = flat_logits[row_idx]
        adjusted = row_logits
        if repetition_penalty != 1.0:
            adjusted = apply_repetition_penalty(adjusted, generated_ids, repetition_penalty)
        if presence_penalty != 0.0:
            adjusted = apply_presence_penalty(adjusted, generated_ids, presence_penalty)
        if adjusted is row_logits:
            continue
        if output is None:
            output = flat_logits.clone()
        output[row_idx].copy_(adjusted)
    return flat_logits if output is None else output


def _sample_candidate_logits_with_tensors(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    apply_top_p: bool,
    candidates_are_sorted: bool,
) -> torch.Tensor:
    if candidate_logits.shape != candidate_token_ids.shape:
        raise ValueError("candidate_logits and candidate_token_ids must have the same shape")
    row_count = int(candidate_logits.shape[0])
    if row_count == 0:
        return torch.empty((0,), dtype=candidate_token_ids.dtype, device=candidate_token_ids.device)
    if int(candidate_logits.shape[-1]) == 1:
        return candidate_token_ids.reshape(row_count, 1)[:, 0]

    eps = torch.finfo(candidate_logits.dtype).eps
    scaled_logits = candidate_logits / temperatures.clamp_min(eps).reshape(row_count, 1)
    row_top_p = top_p.reshape(row_count, 1)
    row_min_p = min_p.reshape(row_count, 1)

    if apply_top_p:
        if candidates_are_sorted:
            sorted_logits = scaled_logits
            sorted_token_ids = candidate_token_ids
        else:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, dim=-1, descending=True)
            sorted_token_ids = candidate_token_ids.gather(dim=-1, index=sorted_indices)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > row_top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        cutoff = cutoff & (row_top_p < 1.0)
        filtered_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        filtered_token_ids = sorted_token_ids
    else:
        filtered_logits = scaled_logits
        filtered_token_ids = candidate_token_ids

    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    max_prob = torch.max(filtered_probs, dim=-1, keepdim=True).values
    keep = filtered_probs >= max_prob * row_min_p
    filtered_logits = torch.where(keep, filtered_logits, torch.full_like(filtered_logits, float("-inf")))

    probs = torch.softmax(filtered_logits, dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1)
    return filtered_token_ids.gather(dim=-1, index=sampled_indices).squeeze(-1)


def sample_next_token_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    min_p: float = 0.0,
    candidates_are_sorted: bool = False,
) -> torch.Tensor:
    return sample_next_token_batch_from_candidates(
        candidate_logits,
        candidate_token_ids,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        candidates_are_sorted=candidates_are_sorted,
    )


def sample_next_token_batch_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    min_p: float = 0.0,
    candidates_are_sorted: bool = False,
) -> torch.Tensor:
    if candidate_logits.shape != candidate_token_ids.shape:
        raise ValueError("candidate_logits and candidate_token_ids must have the same shape")
    if candidate_logits.numel() == 0:
        raise ValueError("candidate logits must not be empty")
    if candidate_logits.ndim == 0:
        raise ValueError("candidate logits must include a candidate dimension")

    output_shape = candidate_logits.shape[:-1]
    candidate_count = candidate_logits.shape[-1]
    logits = candidate_logits.reshape(-1, candidate_count)
    token_ids = candidate_token_ids.reshape(-1, candidate_count)
    if temperature <= 0.0:
        selected = torch.argmax(logits, dim=-1, keepdim=True)
        return token_ids.gather(dim=-1, index=selected).squeeze(-1).reshape(output_shape)

    row_count = int(logits.shape[0])
    sampled = _sample_candidate_logits_with_tensors(
        logits,
        token_ids,
        temperatures=torch.full((row_count,), float(temperature), dtype=logits.dtype, device=logits.device),
        top_p=torch.full((row_count,), float(top_p), dtype=logits.dtype, device=logits.device).clamp(
            min=0.0,
            max=1.0,
        ),
        min_p=torch.full((row_count,), float(min_p), dtype=logits.dtype, device=logits.device).clamp(
            min=0.0,
            max=1.0,
        ),
        apply_top_p=top_p < 1.0,
        candidates_are_sorted=candidates_are_sorted,
    )
    return sampled.reshape(output_shape)


def sample_next_token_batch_from_candidates_with_params(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    params: SamplingBatchParams,
    *,
    candidates_are_sorted: bool = False,
) -> torch.Tensor:
    if candidate_logits.shape != candidate_token_ids.shape:
        raise ValueError("candidate_logits and candidate_token_ids must have the same shape")
    if candidate_logits.numel() == 0:
        raise ValueError("candidate logits must not be empty")
    if candidate_logits.ndim == 0:
        raise ValueError("candidate logits must include a candidate dimension")

    output_shape = candidate_logits.shape[:-1]
    candidate_count = int(candidate_logits.shape[-1])
    flat_logits = candidate_logits.reshape(-1, candidate_count)
    flat_token_ids = candidate_token_ids.reshape(-1, candidate_count)
    row_count = int(flat_logits.shape[0])
    if params.batch_size != row_count:
        raise ValueError(f"sampling params batch size {params.batch_size} does not match row count {row_count}.")

    temperatures = params.temperature.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count)
    top_p = params.top_p.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count)
    min_p = params.min_p.to(device=flat_logits.device, dtype=flat_logits.dtype).reshape(row_count)

    selected_tokens = torch.empty((row_count,), dtype=flat_token_ids.dtype, device=flat_token_ids.device)
    greedy_rows = params.greedy_rows
    if greedy_rows:
        greedy_indices = _row_index_tensor(greedy_rows, device=flat_logits.device, cached=params.greedy_indices)
        greedy_selected = torch.argmax(flat_logits, dim=-1, keepdim=True)
        greedy_tokens = flat_token_ids.gather(dim=-1, index=greedy_selected).squeeze(-1)
        selected_tokens.index_copy_(0, greedy_indices, greedy_tokens.index_select(0, greedy_indices))

    top1_rows = params.top1_rows
    if top1_rows:
        top1_indices = _row_index_tensor(top1_rows, device=flat_logits.device, cached=params.top1_indices)
        top1_logits = flat_logits.index_select(0, top1_indices)
        top1_token_ids = flat_token_ids.index_select(0, top1_indices)
        top1_selected = torch.argmax(top1_logits, dim=-1, keepdim=True)
        top1_tokens = top1_token_ids.gather(dim=-1, index=top1_selected).squeeze(-1)
        selected_tokens.index_copy_(0, top1_indices, top1_tokens)

    def _sample_candidate_rows(rows: tuple[int, ...], *, indices: torch.Tensor, apply_top_p: bool) -> None:
        if not rows:
            return
        row_indices = _row_index_tensor(rows, device=flat_logits.device, cached=indices)
        row_logits = flat_logits.index_select(0, row_indices)
        row_token_ids = flat_token_ids.index_select(0, row_indices)
        row_temperatures = temperatures.index_select(0, row_indices).clamp_min(torch.finfo(flat_logits.dtype).eps)
        row_top_p = top_p.index_select(0, row_indices).clamp(min=0.0, max=1.0)
        row_min_p = min_p.index_select(0, row_indices).clamp(min=0.0, max=1.0)

        bounded_top_k = any(0 < params.top_k_values[idx] < candidate_count for idx in rows)
        rows_are_sorted = candidates_are_sorted
        if bounded_top_k:
            if not rows_are_sorted:
                sorted_logits, sorted_indices = torch.sort(row_logits, dim=-1, descending=True)
                row_token_ids = row_token_ids.gather(dim=-1, index=sorted_indices)
                row_logits = sorted_logits
                rows_are_sorted = True
            row_top_k = params.top_k.to(device=flat_logits.device, dtype=torch.long).index_select(0, row_indices)
            effective_top_k = torch.where(
                (row_top_k > 0) & (row_top_k < candidate_count),
                row_top_k,
                torch.full_like(row_top_k, candidate_count),
            )
            keep = _candidate_positions(candidate_count, device=flat_logits.device) < effective_top_k.reshape(-1, 1)
            row_logits = torch.where(keep, row_logits, torch.full_like(row_logits, float("-inf")))

        sampled_tokens = _sample_candidate_logits_with_tensors(
            row_logits,
            row_token_ids,
            temperatures=row_temperatures,
            top_p=row_top_p,
            min_p=row_min_p,
            apply_top_p=apply_top_p,
            candidates_are_sorted=rows_are_sorted,
        )
        selected_tokens.index_copy_(0, row_indices, sampled_tokens)

    _sample_candidate_rows(params.candidate_plain_rows, indices=params.candidate_plain_indices, apply_top_p=False)
    _sample_candidate_rows(params.candidate_top_p_rows, indices=params.candidate_top_p_indices, apply_top_p=True)

    return selected_tokens.reshape(output_shape)
