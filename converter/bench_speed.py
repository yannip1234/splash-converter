"""Streaming speed benchmark against any OpenAI-compatible endpoint.

Used to compare the Splash package built here against the same Swift weights
served by oMLX. Measures, per prompt:

- TTFT: wall time from request to the first token of the stream
- decode rate: (tokens - 1) / (total - ttft), i.e. excluding prefill
- end-to-end rate: tokens / total

Token counts are produced by tokenizing the generated text with the model's own
tokenizer, identically for every engine. Counting streamed deltas is NOT safe:
Splash emits one token per SSE delta while oMLX batches about eight, which
undercounts oMLX roughly 8x. Each server's self-reported usage is recorded
alongside for cross-checking, but the tokenizer count is what the rates use.

Usage:
  python -m converter.bench_speed --base-url http://127.0.0.1:8127/v1 \
      --model local/Swift-Qwen3.8-27B-Splash --label splash --out report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that reverses a linked list.",
    "List three causes of the French Revolution.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Translate to French: 'The weather is pleasant today.'",
    "What are the tradeoffs between TCP and UDP?",
    "Write a haiku about winter mornings.",
]


_TOKENIZER = None


def load_tokenizer(path):
    """The model's own tokenizer, so every engine is counted identically."""
    global _TOKENIZER
    from tokenizers import Tokenizer
    _TOKENIZER = Tokenizer.from_file(path)
    return _TOKENIZER


def count_tokens(text):
    return len(_TOKENIZER.encode(text, add_special_tokens=False).ids)


def stream_once(base_url, model, prompt, max_tokens, api_key, timeout, tokenizer=_TOKENIZER):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/chat/completions",
                                     json.dumps(body).encode(), headers)
    started = time.perf_counter()
    ttft = None
    counted = 0
    usage = None
    finish = None
    text_parts = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    counted += 1
                    text_parts.append(text)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.perf_counter() - started
    generated = "".join(text_parts)
    reported = (usage or {}).get("completion_tokens")
    tokens = count_tokens(generated) if tokenizer is not None else (reported or counted)
    if ttft is None:
        ttft = total
    decode = total - ttft
    return dict(prompt=prompt, tokens=tokens, total_s=round(total, 3),
                ttft_s=round(ttft, 3),
                decode_tps=round((tokens - 1) / decode, 2) if decode > 0 and tokens > 1 else None,
                end_to_end_tps=round(tokens / total, 2) if total > 0 else None,
                finish=finish, counted_deltas=counted,
                server_reported_tokens=reported, characters=len(generated))


def run(base_url, model, max_tokens, api_key, timeout, warmup=True, log=print):
    if warmup:
        log("  warming up (loads the model if cold) …")
        w = stream_once(base_url, model, "Say hello.", 32, api_key, timeout, _TOKENIZER)
        log(f"  warmup: {w['tokens']} tok, {w['total_s']}s")
    results = []
    for prompt in PROMPTS:
        r = stream_once(base_url, model, prompt, max_tokens, api_key, timeout, _TOKENIZER)
        results.append(r)
        log(f"  {r['tokens']:5d} tok  TTFT {r['ttft_s']:6.2f}s  "
            f"decode {str(r['decode_tps']):>7}  total {r['total_s']:6.2f}s  {prompt[:40]}")
    tokens = sum(r["tokens"] for r in results)
    total = sum(r["total_s"] for r in results)
    decodes = [r["decode_tps"] for r in results if r["decode_tps"]]
    return dict(model=model, base_url=base_url, prompts=len(results),
                total_tokens=tokens, total_seconds=round(total, 2),
                aggregate_tps=round(tokens / total, 2),
                median_decode_tps=round(statistics.median(decodes), 2) if decodes else None,
                mean_decode_tps=round(statistics.fmean(decodes), 2) if decodes else None,
                median_ttft_s=round(statistics.median(r["ttft_s"] for r in results), 3),
                results=results)


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_speed")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-file", default=None,
                   help="read the key from this file (keeps it off the command line)")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--tokenizer",
                   default="output/swift-splash/tokenizer/tokenizer.json",
                   help="tokenizer used to count tokens for EVERY engine")
    args = p.parse_args(argv)
    key = args.api_key
    if args.api_key_file:
        with open(args.api_key_file) as f:
            key = f.read().strip()
    load_tokenizer(args.tokenizer)
    print(f"benchmarking {args.label}: {args.model} @ {args.base_url}")
    print(f"  counting tokens with {args.tokenizer}")
    result = run(args.base_url, args.model, args.max_tokens, key, args.timeout,
                 warmup=not args.no_warmup)
    result["label"] = args.label
    print(f"\n{args.label}: {result['total_tokens']} tok in {result['total_seconds']}s | "
          f"aggregate {result['aggregate_tps']} tok/s | median decode "
          f"{result['median_decode_tps']} tok/s | median TTFT {result['median_ttft_s']}s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
