"""Block-aligned prefix identity and cross-request KV page registry (paged path only)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def prompt_token_blocks(tokens: Sequence[int], *, block_size: int) -> list[tuple[int, ...]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if not tokens:
        return []
    out: list[tuple[int, ...]] = []
    for start in range(0, len(tokens), block_size):
        out.append(tuple(int(t) for t in tokens[start : start + block_size]))
    return out


class PrefixBlockPool:
    """Maps (layer_idx, token_block) -> physical page_id for prefix reuse.

    When a page is recycled (refcount hits zero), :meth:`discard_page` removes
    all keys that pointed at that page so stale mappings are not reused.
    """

    def __init__(self) -> None:
        self._key_to_page: dict[tuple[int, tuple[int, ...]], int] = {}
        self._page_to_keys: dict[tuple[int, int], set[tuple[int, tuple[int, ...]]]] = defaultdict(set)

    def lookup(self, layer_idx: int, token_block: tuple[int, ...]) -> int | None:
        return self._key_to_page.get((layer_idx, token_block))

    def register(self, layer_idx: int, token_block: tuple[int, ...], page_id: int) -> None:
        key = (layer_idx, token_block)
        if key in self._key_to_page:
            return
        self._key_to_page[key] = page_id
        self._page_to_keys[(layer_idx, page_id)].add(key)

    def discard_page(self, layer_idx: int, page_id: int) -> None:
        keys = self._page_to_keys.pop((layer_idx, page_id), None)
        if not keys:
            return
        for key in keys:
            self._key_to_page.pop(key, None)

    def clear_layer(self, layer_idx: int) -> None:
        dead = [k for k in self._key_to_page if k[0] == layer_idx]
        for key in dead:
            pid = self._key_to_page.pop(key, None)
            if pid is None:
                continue
            bucket = self._page_to_keys.get((layer_idx, pid))
            if bucket:
                bucket.discard(key)
                if not bucket:
                    del self._page_to_keys[(layer_idx, pid)]

    def clear(self) -> None:
        self._key_to_page.clear()
        self._page_to_keys.clear()
