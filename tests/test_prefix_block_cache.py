from __future__ import annotations

import pytest

from anna.model.prefix_block_cache import PrefixBlockPool, prompt_token_blocks


def test_prompt_token_blocks_aligns_to_block_size() -> None:
    blocks = prompt_token_blocks([1, 2, 3, 4, 5], block_size=2)
    assert blocks == [(1, 2), (3, 4), (5,)]


def test_prefix_block_pool_tracks_hit_miss_and_register() -> None:
    pool = PrefixBlockPool()
    assert pool.lookup(0, (1, 2, 3, 4)) is None
    pool.register(0, (1, 2, 3, 4), page_id=7)
    assert pool.lookup(0, (1, 2, 3, 4)) == 7
    assert pool.lookup(0, (9, 9, 9, 9)) is None

    stats = pool.stats()
    assert stats.lookups_total == 3
    assert stats.hits_total == 1
    assert stats.misses_total == 2
    assert stats.registers_total == 1
    assert stats.entries == 1
    assert stats.hit_rate == pytest.approx(1.0 / 3.0)


def test_prefix_block_pool_discard_page_removes_stale_keys() -> None:
    pool = PrefixBlockPool()
    pool.register(1, (1, 2), page_id=3)
    pool.discard_page(1, 3)
    assert pool.lookup(1, (1, 2)) is None
    assert pool.stats().entries == 0
