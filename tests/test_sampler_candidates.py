import torch
import pytest

from anna.sampling.sampler import (
    apply_min_p,
    apply_presence_penalty,
    apply_repetition_penalty,
    sample_next_token_batch,
    sample_next_token_batch_from_candidates,
    sample_next_token_batch_from_candidates_with_params,
    sample_next_token_batch_with_params,
    sample_next_token_from_candidates,
)
from anna.sampling.params import SamplingBatchParams
from anna.core.hotpath_events import hotpath_event_recorder
from anna.runtime.service_metrics import AnnaServiceMetrics


def _reference_filtered_logits(
    logits: torch.Tensor,
    *,
    generated_ids_batch: tuple[torch.Tensor | None, ...] | None = None,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if logits.ndim == 1:
        flat = logits.unsqueeze(0)
        output_shape = logits.shape
    else:
        flat = logits.reshape(-1, logits.shape[-1])
        output_shape = logits.shape

    rows: list[torch.Tensor] = []
    histories = generated_ids_batch or (None,) * int(flat.shape[0])
    for row, generated_ids in zip(flat.unbind(0), histories, strict=True):
        adjusted = apply_repetition_penalty(row, generated_ids, repetition_penalty)
        adjusted = apply_presence_penalty(adjusted, generated_ids, presence_penalty)
        rows.append(adjusted)
    filtered = torch.stack(rows, dim=0)

    if temperature <= 0.0:
        return filtered.reshape(output_shape)
    filtered = filtered / temperature
    if 0 < top_k < filtered.shape[-1]:
        kth = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = torch.where(filtered < kth, torch.full_like(filtered, float("-inf")), filtered)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        next_filtered = torch.full_like(filtered, float("-inf"))
        next_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        filtered = next_filtered
    if min_p > 0.0:
        probs = torch.softmax(filtered, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True).values
        keep = probs >= max_prob * min_p
        filtered = torch.where(keep, filtered, torch.full_like(filtered, float("-inf")))
    return filtered.reshape(output_shape)


def test_sample_next_token_from_candidates_greedy_maps_back_to_token_id() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    token_ids = torch.tensor([[10, 20, 30]])

    next_token = sample_next_token_from_candidates(logits, token_ids, temperature=0.0, top_p=1.0)

    assert next_token.item() == 20


def test_sample_next_token_batch_from_candidates_greedy_maps_each_row_back_to_token_id() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])

    next_tokens = sample_next_token_batch_from_candidates(logits, token_ids, temperature=0.0, top_p=1.0)

    assert torch.equal(next_tokens, torch.tensor([20, 60]))


def test_sample_next_token_batch_from_candidates_validates_shapes() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    token_ids = torch.tensor([[10, 20]])

    with pytest.raises(ValueError, match="same shape"):
        sample_next_token_batch_from_candidates(logits, token_ids)


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


def test_top_k_sampling_avoids_full_vocab_top_p_sort_metric() -> None:
    metrics = AnnaServiceMetrics()
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    with hotpath_event_recorder(metrics):
        token = sample_next_token_batch(logits, temperature=1.0, top_k=2, top_p=0.9, min_p=0.0)

    assert int(token.item()) in {0, 1}
    assert metrics.snapshot().sampler_full_vocab_sort_count == 0


def test_top_k_one_sampling_is_deterministic_without_rng_or_full_vocab_sort() -> None:
    metrics = AnnaServiceMetrics()
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0], [5.0, 2.0, 4.0, 3.0]])

    torch.manual_seed(0)
    with hotpath_event_recorder(metrics):
        first = sample_next_token_batch(logits, temperature=0.9, top_k=1, top_p=0.3, min_p=0.9)
    torch.manual_seed(1234)
    second = sample_next_token_batch(logits, temperature=0.9, top_k=1, top_p=0.3, min_p=0.9)

    assert torch.equal(first, torch.tensor([1, 0]))
    assert torch.equal(second, first)
    assert metrics.snapshot().sampler_full_vocab_sort_count == 0


def test_top_p_without_top_k_records_full_vocab_sort_metric() -> None:
    metrics = AnnaServiceMetrics()
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    with hotpath_event_recorder(metrics):
        sample_next_token_batch(logits, temperature=1.0, top_k=0, top_p=0.9, min_p=0.0)

    assert metrics.snapshot().sampler_full_vocab_sort_count == 1


def test_sample_next_token_batch_greedy_matches_reference_with_penalties() -> None:
    logits = torch.tensor(
        [
            [2.0, 9.0, 7.0, 1.0],
            [8.0, 6.0, 4.0, 3.0],
        ]
    )
    histories = (torch.tensor([1, 2]), torch.tensor([0]))

    next_tokens = sample_next_token_batch(
        logits,
        generated_ids_batch=histories,
        temperature=0.0,
        top_k=2,
        top_p=0.75,
        min_p=0.1,
        presence_penalty=1.5,
        repetition_penalty=2.0,
    )
    reference = torch.argmax(
        _reference_filtered_logits(
            logits,
            generated_ids_batch=histories,
            temperature=0.0,
            top_k=2,
            top_p=0.75,
            min_p=0.1,
            presence_penalty=1.5,
            repetition_penalty=2.0,
        ),
        dim=-1,
    )

    assert torch.equal(next_tokens, reference)


def test_top_k_candidate_sampler_support_matches_full_vocab_reference() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 5.0, 4.0, 3.0, 2.0],
        ]
    )
    reference = _reference_filtered_logits(
        logits,
        temperature=1.0,
        top_k=3,
        top_p=0.72,
        min_p=0.05,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    allowed = torch.isfinite(reference)

    for seed in range(20):
        torch.manual_seed(seed)
        sampled = sample_next_token_batch(
            logits,
            temperature=1.0,
            top_k=3,
            top_p=0.72,
            min_p=0.05,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )

        assert allowed[0, int(sampled[0].item())]
        assert allowed[1, int(sampled[1].item())]


def test_full_vocab_sampler_support_matches_reference_with_top_p_min_p_and_penalties() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.5, 2.0, 1.0, -1.0],
            [1.0, 3.0, 4.0, 2.5, 0.5],
        ]
    )
    histories = (torch.tensor([0, 1]), torch.tensor([2, 3, 3]))
    reference = _reference_filtered_logits(
        logits,
        generated_ids_batch=histories,
        temperature=0.8,
        top_k=0,
        top_p=0.82,
        min_p=0.08,
        presence_penalty=0.4,
        repetition_penalty=1.2,
    )
    allowed = torch.isfinite(reference)

    for seed in range(20):
        torch.manual_seed(seed)
        sampled = sample_next_token_batch(
            logits,
            generated_ids_batch=histories,
            temperature=0.8,
            top_k=0,
            top_p=0.82,
            min_p=0.08,
            presence_penalty=0.4,
            repetition_penalty=1.2,
        )

        assert allowed[0, int(sampled[0].item())]
        assert allowed[1, int(sampled[1].item())]


def test_sample_next_token_batch_with_params_supports_per_row_sampling_config() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 5.0, 2.0, 0.5],
        ]
    )
    histories = (torch.tensor([0, 1]), torch.tensor([2]))
    params = SamplingBatchParams.from_sampling_params(
        (
            {
                "temperature": 0.0,
                "top_k": 2,
                "top_p": 0.7,
                "min_p": 0.1,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
            },
            {
                "temperature": 0.9,
                "top_k": 0,
                "top_p": 0.82,
                "min_p": 0.05,
                "presence_penalty": 0.2,
                "repetition_penalty": 1.3,
            },
        ),
        device="cpu",
    )
    assert params.greedy_rows == (0,)
    assert params.sample_rows == (1,)
    assert params.top_p_rows == (1,)
    assert params.top1_rows == ()
    assert params.topk_rows == ()
    assert params.penalty_rows == (0, 1)
    assert params.topk_plain_rows == ()
    assert params.topk_top_p_rows == ()
    assert params.full_plain_rows == ()
    assert params.full_top_p_rows == (1,)
    assert params.candidate_plain_rows == ()
    assert params.candidate_top_p_rows == (1,)
    assert params.penalty_indices.tolist() == [0, 1]
    first_reference = torch.argmax(
        _reference_filtered_logits(
            logits[0],
            generated_ids_batch=(histories[0],),
            temperature=0.0,
            top_k=2,
            top_p=0.7,
            min_p=0.1,
            presence_penalty=0.3,
            repetition_penalty=1.1,
        )
    )
    second_allowed = torch.isfinite(
        _reference_filtered_logits(
            logits[1],
            generated_ids_batch=(histories[1],),
            temperature=0.9,
            top_k=0,
            top_p=0.82,
            min_p=0.05,
            presence_penalty=0.2,
            repetition_penalty=1.3,
        )
    )

    for seed in range(20):
        torch.manual_seed(seed)
        sampled = sample_next_token_batch_with_params(
            logits,
            params,
            generated_ids_batch=histories,
        )

        assert sampled[0] == first_reference
        assert second_allowed[int(sampled[1].item())]


def test_sample_next_token_batch_with_params_treats_top_k_one_sample_rows_as_top1() -> None:
    metrics = AnnaServiceMetrics()
    logits = torch.tensor(
        [
            [1.0, 6.0, 5.0, 4.0],
            [3.0, 2.0, 7.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    params = SamplingBatchParams.from_sampling_params(
        (
            {"temperature": 0.8, "top_k": 1, "top_p": 0.2, "min_p": 0.9},
            {"temperature": 0.7, "top_k": 2, "top_p": 1.0, "min_p": 0.0},
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
        ),
        device="cpu",
    )
    assert params.greedy_rows == (2,)
    assert params.sample_rows == (0, 1)
    assert params.top_p_rows == (0,)
    assert params.top1_rows == (0,)
    assert params.topk_rows == (1,)
    assert params.penalty_rows == (0, 1, 2)
    assert params.topk_plain_rows == (1,)
    assert params.topk_top_p_rows == ()
    assert params.full_plain_rows == ()
    assert params.full_top_p_rows == ()
    assert params.candidate_plain_rows == (1,)
    assert params.candidate_top_p_rows == ()
    assert params.greedy_indices.tolist() == [2]
    assert params.top1_indices.tolist() == [0]
    assert params.topk_plain_indices.tolist() == [1]
    assert params.topk_top_p_indices.tolist() == []
    assert params.full_plain_indices.tolist() == []
    assert params.full_top_p_indices.tolist() == []
    assert params.candidate_plain_indices.tolist() == [1]
    assert params.candidate_top_p_indices.tolist() == []
    assert params.penalty_indices.tolist() == [0, 1, 2]

    torch.manual_seed(1)
    with hotpath_event_recorder(metrics):
        sampled = sample_next_token_batch_with_params(logits, params)

    assert sampled[0] == 1
    assert sampled[1] in (0, 2)
    assert sampled[2] == 0
    assert metrics.snapshot().sampler_full_vocab_sort_count == 0


def test_sample_next_token_batch_with_params_counts_only_full_vocab_top_p_rows() -> None:
    metrics = AnnaServiceMetrics()
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 5.0, 4.0, 3.0],
            [4.0, 1.0, 3.0, 2.0],
        ]
    )
    params = SamplingBatchParams.from_sampling_params(
        (
            {"temperature": 0.8, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
            {"temperature": 0.9, "top_k": 0, "top_p": 0.7, "min_p": 0.0},
            {"temperature": 0.7, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
        ),
        device="cpu",
    )
    assert params.full_plain_rows == (0, 2)
    assert params.full_top_p_rows == (1,)
    assert params.candidate_plain_rows == (0, 2)
    assert params.candidate_top_p_rows == (1,)
    assert params.penalty_rows == (0, 1, 2)
    assert params.full_plain_indices.tolist() == [0, 2]
    assert params.full_top_p_indices.tolist() == [1]
    assert params.candidate_plain_indices.tolist() == [0, 2]
    assert params.candidate_top_p_indices.tolist() == [1]

    torch.manual_seed(2)
    with hotpath_event_recorder(metrics):
        sampled = sample_next_token_batch_with_params(logits, params)

    assert sampled.shape == (3,)
    assert metrics.snapshot().sampler_full_vocab_sort_count == 1


def test_candidate_sampler_with_params_supports_per_row_sampling_config() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0], [1.0, 5.0, 4.0]])
    token_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    params = SamplingBatchParams.from_sampling_params(
        (
            {"temperature": 0.0, "top_p": 1.0, "min_p": 0.0},
            {"temperature": 0.8, "top_p": 0.7, "min_p": 0.0},
        ),
        device="cpu",
    )
    assert params.greedy_rows == (0,)
    assert params.sample_rows == (1,)
    assert params.top_p_rows == (1,)
    assert params.top1_rows == ()
    assert params.topk_rows == (1,)
    assert params.penalty_rows == (0, 1)
    assert params.topk_plain_rows == ()
    assert params.topk_top_p_rows == (1,)
    assert params.candidate_plain_rows == ()
    assert params.candidate_top_p_rows == (1,)
    assert params.penalty_indices.tolist() == [0, 1]
    second_allowed = torch.isfinite(
        _reference_filtered_logits(
            logits[1],
            temperature=0.8,
            top_k=0,
            top_p=0.7,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )
    )

    for seed in range(20):
        torch.manual_seed(seed)
        sampled = sample_next_token_batch_from_candidates_with_params(logits, token_ids, params)

        assert sampled[0] == 10
        assert second_allowed[int((token_ids[1] == sampled[1]).nonzero(as_tuple=True)[0].item())]


def test_candidate_sampler_with_params_respects_per_row_top_k_on_wide_candidates() -> None:
    logits = torch.tensor(
        [
            [1.0, 6.0, 5.0],
            [10.0, 9.0, -10.0],
        ]
    )
    token_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ]
    )
    params = SamplingBatchParams.from_sampling_params(
        (
            {"temperature": 0.8, "top_k": 1, "top_p": 0.2, "min_p": 0.9},
            {"temperature": 0.8, "top_k": 2, "top_p": 1.0, "min_p": 0.0},
        ),
        device="cpu",
    )

    for seed in range(20):
        torch.manual_seed(seed)
        sampled = sample_next_token_batch_from_candidates_with_params(
            logits,
            token_ids,
            params,
            candidates_are_sorted=False,
        )

        assert sampled[0] == 11
        assert int(sampled[1].item()) in {20, 21}


def test_candidate_sampler_top1_returns_only_candidate_without_sampling() -> None:
    logits = torch.tensor([[2.0], [9.0]])
    token_ids = torch.tensor([[42], [77]])
    params = SamplingBatchParams.from_sampling_params(
        (
            {"temperature": 0.8, "top_p": 0.1, "min_p": 1.0},
            {"temperature": 0.6, "top_p": 0.2, "min_p": 0.9},
        ),
        device="cpu",
    )

    torch.manual_seed(0)
    first = sample_next_token_batch_from_candidates_with_params(logits, token_ids, params)
    torch.manual_seed(123)
    second = sample_next_token_batch_from_candidates_with_params(logits, token_ids, params)

    assert torch.equal(first, torch.tensor([42, 77]))
    assert torch.equal(second, first)
