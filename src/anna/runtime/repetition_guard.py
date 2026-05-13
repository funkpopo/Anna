from __future__ import annotations


def repeated_suffix_trim_index(
    token_ids: list[int],
    *,
    min_ngram_size: int = 4,
    max_ngram_size: int = 64,
    min_repeats: int = 3,
) -> int | None:
    if min_ngram_size <= 0:
        raise ValueError("min_ngram_size must be positive")
    if max_ngram_size < min_ngram_size:
        raise ValueError("max_ngram_size must be >= min_ngram_size")
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")

    token_count = len(token_ids)
    max_size = min(max_ngram_size, token_count // min_repeats)
    for ngram_size in range(min_ngram_size, max_size + 1):
        repeated_token_count = ngram_size * min_repeats
        suffix = token_ids[-repeated_token_count:]
        block = suffix[:ngram_size]
        if all(suffix[start : start + ngram_size] == block for start in range(ngram_size, repeated_token_count, ngram_size)):
            return token_count - repeated_token_count + ngram_size
    return None
