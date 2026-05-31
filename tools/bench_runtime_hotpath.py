from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anna.core.hotpath_events import hotpath_event_recorder
from anna.model.ops import Qwen3DynamicCache, Qwen3PageAllocator
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig
from anna.runtime.hotpath_guard import scan_hotpath_files, unexpected_findings
from anna.runtime.paged_kv import PagedKVManager
from anna.runtime.service_metrics import AnnaServiceMetrics
from anna.runtime.slot_scheduler import SlotScheduler
from anna.sampling.sampler import sample_next_token, sample_next_token_from_candidates


def _time_ms(fn: Callable[[], object], *, iters: int) -> float:
    started = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
    return (time.perf_counter() - started) * 1000.0 / max(1, iters)


def _tiny_config(*, layers: int, heads: int, head_dim: int, block_size: int) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=heads * head_dim,
        intermediate_size=heads * head_dim * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        vocab_size=4096,
        eos_token_id=0,
        pad_token_id=0,
        cache_block_size=block_size,
        layer_types=["full_attention"] * layers,
    )


def _make_cache(
    *,
    config: Qwen3_5TextConfig,
    allocator: Qwen3PageAllocator,
    seq_len: int,
) -> Qwen3DynamicCache:
    cache = Qwen3DynamicCache(config, allocator=allocator, batch_size=1)
    key = torch.zeros((1, config.num_key_value_heads, seq_len, config.head_dim), dtype=torch.float32)
    value = torch.zeros_like(key)
    for layer_idx in range(config.num_hidden_layers):
        cache.update(key, value, layer_idx=layer_idx)
    return cache


def _bench_cache_stack_split(
    *,
    batch_size: int,
    seq_len: int,
    layers: int,
    heads: int,
    head_dim: int,
    block_size: int,
    iters: int,
) -> dict[str, Any]:
    config = _tiny_config(layers=layers, heads=heads, head_dim=head_dim, block_size=block_size)
    allocator = Qwen3PageAllocator(config)
    caches = [_make_cache(config=config, allocator=allocator, seq_len=seq_len) for _ in range(batch_size)]

    stacked = Qwen3DynamicCache.stack(caches, config, clone_turboquant_rows=False)
    split = stacked.split_batch(clone_turboquant_rows=False)
    assert len(split) == batch_size

    def stack_once() -> Qwen3DynamicCache:
        return Qwen3DynamicCache.stack(caches, config, clone_turboquant_rows=False)

    def split_once() -> list[Qwen3DynamicCache]:
        return stacked.split_batch(clone_turboquant_rows=False)

    stack_ms = _time_ms(stack_once, iters=iters)
    split_ms = _time_ms(split_once, iters=iters)
    for cache in caches:
        cache.release()
    stacked.release()
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "stack_ms": stack_ms,
        "split_ms": split_ms,
    }


def _bench_slot_decode_plan(
    *,
    batch_size: int,
    seq_len: int,
    block_size: int,
    max_blocks_per_seq: int,
    iters: int,
) -> dict[str, Any]:
    required_blocks = 0 if seq_len <= 0 else (seq_len + block_size - 1) // block_size
    resolved_max_blocks = max(max_blocks_per_seq, required_blocks + 1, 1)
    manager = PagedKVManager(
        max_slots=batch_size,
        total_blocks=batch_size * resolved_max_blocks,
        block_size=block_size,
        max_blocks_per_seq=resolved_max_blocks,
        device="cpu",
    )
    scheduler = SlotScheduler(manager, max_batch_size=batch_size)
    request_ids = tuple(f"bench-{idx}" for idx in range(batch_size))
    for idx, request_id in enumerate(request_ids):
        scheduler.admit(
            request_id,
            prompt_length=seq_len,
            max_new_tokens=4,
            sampling_params={
                "temperature": 0.0 if idx % 2 == 0 else 0.7,
                "top_p": 1.0 if idx % 2 == 0 else 0.8,
                "top_k": 1 if idx % 2 == 0 else 32,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        )
        scheduler.mark_prefilled(request_id, next_input_id=idx + 1)

    plan = scheduler.build_decode_plan(request_ids=request_ids)
    assert plan.block_tables.shape == (batch_size, resolved_max_blocks)
    assert plan.sampling_batch_params.batch_size == batch_size

    plan_ms = _time_ms(lambda: scheduler.build_decode_plan(request_ids=request_ids), iters=iters)
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "max_blocks_per_seq": resolved_max_blocks,
        "plan_ms": plan_ms,
        "block_table_rows": int(plan.block_tables.shape[0]),
        "block_table_cols": int(plan.block_tables.shape[1]),
    }


def _bench_sampler(*, vocab_size: int, candidates: int, iters: int) -> dict[str, Any]:
    logits = torch.randn(vocab_size)
    history = torch.arange(0, min(64, vocab_size), dtype=torch.long)
    candidate_logits = torch.randn(candidates)
    candidate_token_ids = torch.arange(candidates, dtype=torch.long)
    metrics = AnnaServiceMetrics()

    def full_vocab() -> torch.Tensor:
        with hotpath_event_recorder(metrics):
            return sample_next_token(
                logits,
                generated_ids=history,
                temperature=0.7,
                top_p=0.8,
                top_k=0,
                min_p=0.0,
                presence_penalty=0.1,
                repetition_penalty=1.05,
            )

    def candidate() -> torch.Tensor:
        return sample_next_token_from_candidates(
            candidate_logits,
            candidate_token_ids,
            temperature=0.7,
            top_p=0.8,
            min_p=0.0,
        )

    full_vocab_ms = _time_ms(full_vocab, iters=iters)
    candidate_ms = _time_ms(candidate, iters=iters)
    snapshot = metrics.snapshot()
    return {
        "vocab_size": vocab_size,
        "candidates": candidates,
        "full_vocab_ms": full_vocab_ms,
        "candidate_ms": candidate_ms,
        "sampler_full_vocab_sort_count": snapshot.sampler_full_vocab_sort_count,
    }


def _bench_hotpath_guard() -> dict[str, Any]:
    started = time.perf_counter()
    findings = scan_hotpath_files(root=Path.cwd())
    unexpected = unexpected_findings(findings)
    return {
        "scan_ms": (time.perf_counter() - started) * 1000.0,
        "findings": len(findings),
        "unexpected": len(unexpected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-model Anna runtime hot-path microbenchmarks.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--max-blocks-per-seq",
        type=int,
        default=0,
        help="Max block-table entries per sequence for the slot decode plan benchmark. Set 0 to derive from seq len.",
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = {
        "hotpath_guard": _bench_hotpath_guard(),
        "cache_stack_split": _bench_cache_stack_split(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            layers=args.layers,
            heads=args.heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            iters=args.iters,
        ),
        "slot_decode_plan": _bench_slot_decode_plan(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            block_size=args.block_size,
            max_blocks_per_seq=args.max_blocks_per_seq,
            iters=args.iters,
        ),
        "sampler": _bench_sampler(
            vocab_size=args.vocab_size,
            candidates=args.candidates,
            iters=args.iters,
        ),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("section,metric,value")
        for section, values in result.items():
            for key, value in values.items():
                print(f"{section},{key},{value}")
    return 1 if result["hotpath_guard"]["unexpected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
