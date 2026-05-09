from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequestResult:
    ok: bool
    status: int | None
    error: str | None
    latency_seconds: float
    ttft_seconds: float | None
    output_tokens: int
    input_tokens: int
    inter_token_seconds: tuple[float, ...]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> tuple[int, bytes]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), response.read()


def _get_json(url: str, *, timeout: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return input_tokens, output_tokens


def _estimate_completion_tokens(payload: dict[str, Any]) -> int:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return 0
    total = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            total += max(1, len(text.split()))
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                total += max(1, len(content.split()))
    return total


def _read_sse_response(response, *, started_at: float) -> tuple[float | None, int, tuple[float, ...], dict[str, Any] | None]:
    first_token_at: float | None = None
    previous_token_at: float | None = None
    inter_token_seconds: list[float] = []
    output_tokens = 0
    final_usage: dict[str, Any] | None = None
    event_lines: list[str] = []

    def consume_event(lines: list[str]) -> None:
        nonlocal first_token_at, previous_token_at, output_tokens, final_usage
        data_parts: list[str] = []
        for line in lines:
            if line.startswith("data:"):
                data_parts.append(line[5:].strip())
        if not data_parts:
            return
        data = "\n".join(data_parts)
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            final_usage = usage
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        emitted = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            text = choice.get("text")
            content = None
            if isinstance(delta, dict):
                content = delta.get("content") or delta.get("reasoning_content")
            elif isinstance(text, str):
                content = text
            if isinstance(content, str) and content:
                emitted = True
        if not emitted:
            return
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        if previous_token_at is not None:
            inter_token_seconds.append(now - previous_token_at)
        previous_token_at = now
        output_tokens += 1

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            consume_event(event_lines)
            event_lines = []
        else:
            event_lines.append(line)
    if event_lines:
        consume_event(event_lines)
    if final_usage is not None:
        output_tokens = max(output_tokens, int(final_usage.get("completion_tokens") or 0))
    ttft = None if first_token_at is None else first_token_at - started_at
    return ttft, output_tokens, tuple(inter_token_seconds), final_usage


def _run_one(
    *,
    base_url: str,
    route: str,
    model: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    stream: bool,
    timeout: float,
) -> RequestResult:
    url = base_url.rstrip("/") + route
    if route.endswith("/v1/completions"):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    else:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    if top_k is not None:
        payload["top_k"] = top_k
    if stream:
        payload["stream_options"] = {"include_usage": True}

    started_at = time.perf_counter()
    try:
        if stream:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                ttft, output_tokens, itl, usage = _read_sse_response(response, started_at=started_at)
                latency = time.perf_counter() - started_at
                input_tokens = 0 if usage is None else int(usage.get("prompt_tokens") or 0)
                return RequestResult(True, status, None, latency, ttft, output_tokens, input_tokens, itl)

        status, body = _post_json(url, payload, timeout=timeout)
        latency = time.perf_counter() - started_at
        response_payload = json.loads(body.decode("utf-8"))
        input_tokens, output_tokens = _extract_usage(response_payload)
        if output_tokens <= 0:
            output_tokens = _estimate_completion_tokens(response_payload)
        return RequestResult(True, status, None, latency, None, output_tokens, input_tokens, ())
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - started_at
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return RequestResult(False, int(exc.code), detail, latency, None, 0, 0, ())
    except Exception as exc:
        latency = time.perf_counter() - started_at
        return RequestResult(False, None, repr(exc), latency, None, 0, 0, ())


def _long_prompt() -> str:
    paragraph = (
        "Anna runtime 正在优化 Qwen3.5 在 Intel Arc XPU 上的推理服务。"
        "目标是让 prefill 阶段和 decode 阶段尽量分离，降低长 prompt 插入时对已在生成请求的影响。"
        "当前配置使用 bf16、PyTorch int4pack dense 权重、TurboQuant KV cache、scheduler decode batching、"
        "FlashQLA-compatible GDN prefill、LM-head top-k fused，并通过 runtime profile 观察 attention、MoE、"
        "gated delta、TurboQuant dequant 和 LM-head 的耗时。"
    )
    return "\n".join([paragraph for _ in range(24)])


def _prompt_for_request(base_prompt: str, *, scenario: str, index: int) -> tuple[str, str | None]:
    shared_system = (
        "你是一个严谨的系统性能分析助手。回答必须短、直接，并优先给出可验证的工程判断。"
    )
    if scenario == "custom":
        return base_prompt, None
    if scenario == "concurrent-short":
        return f"{base_prompt}\n请求编号 {index}：用三句话回答。", None
    if scenario == "single-long":
        return _long_prompt() + "\n\n请总结其中的关键性能风险。", None
    if scenario == "mixed":
        if index % 4 == 0:
            return _long_prompt() + "\n\n请给出 5 条优化建议。", None
        return f"{base_prompt}\n请求编号 {index}：只输出 3 个要点。", None
    if scenario == "repeated-system":
        return f"{base_prompt}\n请求编号 {index}：保持回答在 60 字以内。", shared_system
    raise ValueError(f"Unsupported scenario: {scenario}")


def _summarize(results: list[RequestResult], *, wall_seconds: float) -> dict[str, Any]:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_seconds for result in successes]
    ttfts = [result.ttft_seconds for result in successes if result.ttft_seconds is not None]
    itls = [value for result in successes for value in result.inter_token_seconds]
    output_tokens = sum(result.output_tokens for result in successes)
    input_tokens = sum(result.input_tokens for result in successes)
    return {
        "requests": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(successes) / wall_seconds if wall_seconds > 0 else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "latency_ms": {
            "avg": _mean(latencies) * 1000.0,
            "p50": _percentile(latencies, 0.50) * 1000.0,
            "p95": _percentile(latencies, 0.95) * 1000.0,
            "p99": _percentile(latencies, 0.99) * 1000.0,
            "max": max(latencies, default=0.0) * 1000.0,
        },
        "ttft_ms": {
            "avg": _mean(ttfts) * 1000.0,
            "p50": _percentile(ttfts, 0.50) * 1000.0,
            "p95": _percentile(ttfts, 0.95) * 1000.0,
            "p99": _percentile(ttfts, 0.99) * 1000.0,
            "max": max(ttfts, default=0.0) * 1000.0,
        },
        "itl_ms": {
            "avg": _mean(itls) * 1000.0,
            "p50": _percentile(itls, 0.50) * 1000.0,
            "p95": _percentile(itls, 0.95) * 1000.0,
            "p99": _percentile(itls, 0.99) * 1000.0,
            "max": max(itls, default=0.0) * 1000.0,
        },
        "failure_samples": [
            {"status": result.status, "error": result.error[:500] if result.error else None}
            for result in failures[:5]
        ],
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "requests={requests} successes={successes} failures={failures} wall={wall_seconds:.2f}s "
        "rps={requests_per_second:.2f} out_tps={output_tokens_per_second:.2f}".format(**summary)
    )
    for name in ("latency_ms", "ttft_ms", "itl_ms"):
        values = summary[name]
        print(
            "{name}: avg={avg:.1f} p50={p50:.1f} p95={p95:.1f} p99={p99:.1f} max={max:.1f}".format(
                name=name,
                **values,
            )
        )
    if summary["failure_samples"]:
        print("failure_samples:")
        for sample in summary["failure_samples"]:
            print(f"  status={sample['status']} error={sample['error']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Anna OpenAI-compatible HTTP generation concurrency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--route", choices=("/v1/chat/completions", "/v1/completions"), default="/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--prompt", default="写一段 80 字以内的技术总结，说明 prefill 和 decode 分离为什么能降低尾延迟。")
    parser.add_argument("--prompt-file", default=None, help="Read prompt text from a UTF-8 file.")
    parser.add_argument(
        "--scenario",
        choices=("custom", "concurrent-short", "single-long", "mixed", "repeated-system"),
        default="custom",
        help="Built-in prompt scenario. repeated-system is useful with --prompt-cache-size on the server.",
    )
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1, help="Use <=16 to exercise fused LM-head top-k when enabled.")
    parser.add_argument("--no-top-k", action="store_true", help="Omit top_k from request payload.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--healthz", action="store_true", help="Fetch /healthz before and after the run.")
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.requests <= 0:
        raise SystemExit("--requests must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    prompt = args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompt = handle.read()
    health_before = _get_json(args.base_url.rstrip("/") + "/healthz", timeout=args.timeout) if args.healthz else None

    started_at = time.perf_counter()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        request_prompts = [
            _prompt_for_request(prompt, scenario=args.scenario, index=idx)
            for idx in range(args.requests)
        ]
        futures = [
            executor.submit(
                _run_one,
                base_url=args.base_url,
                route=args.route,
                model=args.model,
                prompt=request_prompt,
                system_prompt=system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=None if args.no_top_k else args.top_k,
                stream=args.stream,
                timeout=args.timeout,
            )
            for request_prompt, system_prompt in request_prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            done = len(results)
            if not args.json:
                print(f"completed {done}/{args.requests}", file=sys.stderr)
    wall_seconds = time.perf_counter() - started_at
    health_after = _get_json(args.base_url.rstrip("/") + "/healthz", timeout=args.timeout) if args.healthz else None
    summary = _summarize(results, wall_seconds=wall_seconds)
    summary["config"] = {
        "base_url": args.base_url,
        "route": args.route,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": None if args.no_top_k else args.top_k,
        "stream": args.stream,
        "scenario": args.scenario,
    }
    if args.healthz:
        summary["health_before"] = health_before
        summary["health_after"] = health_after
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_summary(summary)
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
