import torch

from anna.sampling.sampler import sample_next_token_from_candidates


def test_sample_next_token_from_candidates_greedy_maps_back_to_token_id() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    token_ids = torch.tensor([[10, 20, 30]])

    next_token = sample_next_token_from_candidates(logits, token_ids, temperature=0.0, top_p=1.0)

    assert next_token.item() == 20
