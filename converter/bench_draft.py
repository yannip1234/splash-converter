"""Measure DFlash2 draft acceptance against a served package (PLAN T6.3).

Runs a fixed prompt set through the OpenAI-compatible endpoint and reads the
deltas of `splash_drafted_tokens_total` / `splash_accepted_draft_tokens_total`
from /metrics, so the figure covers exactly this run.

Usage:  python -m converter.bench_draft --port 8127 --label swift
"""

from __future__ import annotations

import argparse
import json
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


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def metrics(port):
    out = {}
    for line in _get(f"http://127.0.0.1:{port}/metrics").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, _, value = line.rpartition(" ")
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def complete(port, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    usage = payload["usage"]
    return dict(seconds=time.time() - started, completion_tokens=usage["completion_tokens"],
                finish=payload["choices"][0]["finish_reason"])


def run(port, model, max_tokens, timeout, log=print):
    before = metrics(port)
    results = []
    for prompt in PROMPTS:
        r = complete(port, model, prompt, max_tokens, timeout)
        results.append(r)
        log(f"  {r['completion_tokens']:5d} tok in {r['seconds']:6.1f}s "
            f"({r['completion_tokens'] / r['seconds']:6.1f} tok/s)  {r['finish']:6s}  {prompt[:44]}")
    after = metrics(port)
    drafted = after["splash_drafted_tokens_total"] - before["splash_drafted_tokens_total"]
    accepted = after["splash_accepted_draft_tokens_total"] - before["splash_accepted_draft_tokens_total"]
    tokens = sum(r["completion_tokens"] for r in results)
    seconds = sum(r["seconds"] for r in results)
    return dict(prompts=len(PROMPTS), completion_tokens=tokens, seconds=round(seconds, 1),
                tokens_per_second=round(tokens / seconds, 2),
                drafted_tokens=int(drafted), accepted_draft_tokens=int(accepted),
                acceptance=round(accepted / drafted, 4) if drafted else None,
                cumulative_acceptance=after.get("splash_draft_acceptance_ratio"),
                results=results)


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_draft")
    p.add_argument("--port", type=int, default=8127)
    p.add_argument("--model", default=None, help="defaults to the served model id")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    model = args.model
    if model is None:
        model = json.loads(_get(f"http://127.0.0.1:{args.port}/v1/models"))["data"][0]["id"]
    print(f"benchmarking {model} on port {args.port} ({len(PROMPTS)} prompts)")
    result = run(args.port, model, args.max_tokens, args.timeout)
    result["label"] = args.label
    result["model"] = model
    print(f"\n{args.label}: {result['completion_tokens']} tokens, "
          f"{result['tokens_per_second']} tok/s, drafted {result['drafted_tokens']}, "
          f"accepted {result['accepted_draft_tokens']}, acceptance {result['acceptance']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
