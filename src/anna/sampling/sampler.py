from __future__ import annotations

from collections.abc import Sequence

import torch

from anna.core.hotpath_events import record_sampler_full_vocab_sort
from anna.sampling.params import SamplingBatchParams


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

    next_logits = next_logits / temperature
    if 0 < top_k < vocab_size:
        candidate_logits, candidate_token_ids = torch.topk(next_logits, k=top_k, dim=-1)
        return sample_next_token_batch_from_candidates(
            candidate_logits,
            candidate_token_ids,
            temperature=1.0,
            top_p=top_p,
            min_p=min_p,
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

    histories = (None,) * row_count if generated_ids_batch is None else generated_ids_batch
    tokens = [
        sample_next_token_batch(
            row_logits,
            generated_ids_batch=None if generated_ids is None else (generated_ids,),
            temperature=params.temperature_values[row_idx],
            top_p=params.top_p_values[row_idx],
            top_k=params.top_k_values[row_idx],
            min_p=params.min_p_values[row_idx],
            presence_penalty=params.presence_penalty_values[row_idx],
            repetition_penalty=params.repetition_penalty_values[row_idx],
        ).reshape(())
        for row_idx, (row_logits, generated_ids) in enumerate(zip(flat_logits.unbind(0), histories, strict=True))
    ]
    return torch.stack(tokens, dim=0).reshape(output_shape)


def sample_next_token_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    min_p: float = 0.0,
) -> torch.Tensor:
    return sample_next_token_batch_from_candidates(
        candidate_logits,
        candidate_token_ids,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
    )


def sample_next_token_batch_from_candidates(
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
    if candidate_logits.ndim == 0:
        raise ValueError("candidate logits must include a candidate dimension")

    output_shape = candidate_logits.shape[:-1]
    candidate_count = candidate_logits.shape[-1]
    logits = candidate_logits.reshape(-1, candidate_count)
    token_ids = candidate_token_ids.reshape(-1, candidate_count)
    if temperature <= 0.0:
        selected = torch.argmax(logits, dim=-1, keepdim=True)
        return token_ids.gather(dim=-1, index=selected).squeeze(-1).reshape(output_shape)

    next_logits = logits / temperature
    next_logits = apply_top_p(next_logits, top_p)
    next_logits = apply_min_p(next_logits, min_p)
    probs = torch.softmax(next_logits, dim=-1)
    selected = torch.multinomial(probs, num_samples=1)
    return token_ids.gather(dim=-1, index=selected).squeeze(-1).reshape(output_shape)


def sample_next_token_batch_from_candidates_with_params(
    candidate_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    params: SamplingBatchParams,
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
    greedy_indices = torch.nonzero(temperatures <= 0.0, as_tuple=False).flatten()
    if greedy_indices.numel() > 0:
        greedy_selected = torch.argmax(flat_logits, dim=-1, keepdim=True)
        greedy_tokens = flat_token_ids.gather(dim=-1, index=greedy_selected).squeeze(-1)
        selected_tokens.index_copy_(0, greedy_indices, greedy_tokens.index_select(0, greedy_indices))

    sample_indices = torch.nonzero(temperatures > 0.0, as_tuple=False).flatten()
    if sample_indices.numel() > 0:
        row_logits = flat_logits.index_select(0, sample_indices)
        row_token_ids = flat_token_ids.index_select(0, sample_indices)
        row_temperatures = temperatures.index_select(0, sample_indices).clamp_min(torch.finfo(flat_logits.dtype).eps)
        row_top_p = top_p.index_select(0, sample_indices).clamp(min=0.0, max=1.0)
        row_min_p = min_p.index_select(0, sample_indices).clamp(min=0.0, max=1.0)

        sorted_logits, sorted_indices = torch.sort(row_logits / row_temperatures[:, None], dim=-1, descending=True)
        sorted_token_ids = row_token_ids.gather(dim=-1, index=sorted_indices)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > row_top_p[:, None]
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        cutoff = cutoff & (row_top_p[:, None] < 1.0)
        filtered_logits = sorted_logits.masked_fill(cutoff, float("-inf"))

        filtered_probs = torch.softmax(filtered_logits, dim=-1)
        max_prob = torch.max(filtered_probs, dim=-1, keepdim=True).values
        keep = filtered_probs >= max_prob * row_min_p[:, None]
        filtered_logits = torch.where(keep, filtered_logits, torch.full_like(filtered_logits, float("-inf")))

        probs = torch.softmax(filtered_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)
        sampled_tokens = sorted_token_ids.gather(dim=-1, index=sampled_indices).squeeze(-1)
        selected_tokens.index_copy_(0, sample_indices, sampled_tokens)

    return selected_tokens.reshape(output_shape)
