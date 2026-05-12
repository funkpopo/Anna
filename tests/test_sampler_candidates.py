import torch

from anna.sampling.sampler import apply_min_p, apply_presence_penalty, sample_next_token_from_candidates


def test_sample_next_token_from_candidates_greedy_maps_back_to_token_id() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    token_ids = torch.tensor([[10, 20, 30]])

    next_token = sample_next_token_from_candidates(logits, token_ids, temperature=0.0, top_p=1.0)

    assert next_token.item() == 20


def test_apply_presence_penalty_subtracts_from_seen_tokens() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0])
    generated_ids = torch.tensor([1, 1, 2])

    adjusted = apply_presence_penalty(logits, generated_ids, penalty=0.5)

    assert torch.equal(adjusted, torch.tensor([1.0, 1.5, 2.5]))


def test_apply_min_p_keeps_tokens_above_scaled_max_probability() -> None:
    logits = torch.tensor([4.0, 3.0, 0.0])

    filtered = apply_min_p(logits, min_p=0.25)

    assert torch.isfinite(filtered[0])
    assert torch.isfinite(filtered[1])
    assert torch.isneginf(filtered[2])
