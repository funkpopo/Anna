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
    unique_ids = torch.unique(generated_ids)
    values = output.index_select(0, unique_ids)
    adjusted = torch.where(values < 0, values * penalty, values / penalty)
    output.index_copy_(0, unique_ids, adjusted)
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


def sample_next_token(
    logits: torch.Tensor,
    *,
    generated_ids: torch.Tensor | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    next_logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature <= 0.0:
        return torch.argmax(next_logits, dim=-1)

    next_logits = next_logits / temperature
    next_logits = apply_top_k(next_logits, top_k)
    next_logits = apply_top_p(next_logits, top_p)
    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
