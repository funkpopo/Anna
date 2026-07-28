import torch

from anna.sampling.sampler import (
    apply_min_p,
    apply_presence_penalty,
    sample_next_token,
    sample_next_token_from_candidates,
    sample_next_tokens,
    sample_next_tokens_from_candidates,
    token_ids_to_host,
)


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


def test_sample_next_tokens_batched_greedy_homogeneous() -> None:
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, -1.0],
            [3.0, 0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 9.0],
        ]
    )

    sampled = sample_next_tokens(logits, temperatures=0.0, top_ps=1.0, top_ks=0, presence_penalties=0.0, repetition_penalties=1.0)

    assert sampled.tolist() == [1, 0, 3]


def test_sample_next_tokens_batched_heterogeneous_temperature_groups() -> None:
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0],
            [0.0, 5.0, 1.0],
            [9.0, 0.0, 0.0],
        ]
    )

    sampled = sample_next_tokens(
        logits,
        temperatures=[0.0, 0.0, 0.0],
        top_ps=[1.0, 1.0, 1.0],
        top_ks=[0, 0, 0],
        presence_penalties=0.0,
        repetition_penalties=1.0,
    )

    assert sampled.tolist() == [1, 1, 0]


def test_sample_next_tokens_per_row_repetition_penalty() -> None:
    logits = torch.tensor(
        [
            [2.0, 3.0, 1.0],
            [2.0, 3.0, 1.0],
        ]
    )
    # Row 0 penalizes token 1 heavily so argmax flips to token 0.
    history = [torch.tensor([1]), None]

    sampled = sample_next_tokens(
        logits,
        generated_ids_rows=history,
        temperatures=0.0,
        top_ps=1.0,
        top_ks=0,
        presence_penalties=0.0,
        repetition_penalties=[10.0, 1.0],
    )

    assert sampled.tolist() == [0, 1]


def test_sample_next_tokens_from_candidates_batched_greedy() -> None:
    candidate_logits = torch.tensor(
        [
            [1.0, 4.0, 2.0],
            [5.0, 0.0, 1.0],
        ]
    )
    candidate_ids = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
        ]
    )

    sampled = sample_next_tokens_from_candidates(
        candidate_logits,
        candidate_ids,
        temperatures=0.0,
        top_ps=1.0,
    )

    assert sampled.tolist() == [20, 40]


def test_sample_next_token_rank1_and_rank2_agree_greedy() -> None:
    row = torch.tensor([0.5, 2.0, 1.0, -3.0])
    batch = row.unsqueeze(0)

    single = sample_next_token(row, temperature=0.0, presence_penalty=0.0, repetition_penalty=1.0)
    batched = sample_next_token(batch, temperature=0.0, presence_penalty=0.0, repetition_penalty=1.0)

    assert int(single.item()) == int(batched[0].item()) == 1


def test_token_ids_to_host_single_sync_for_batch() -> None:
    tokens = torch.tensor([3, 7, 11], dtype=torch.long)

    assert token_ids_to_host(tokens) == [3, 7, 11]
    assert token_ids_to_host(tokens[1]) == [7]
