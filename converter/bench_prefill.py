"""Prefill / time-to-first-token versus prompt length, across engines.

Decode rate is only half the story for agent workloads: a coding session sends
tens of thousands of tokens of context on every step, so prefill dominates the
latency the user feels. This measures TTFT at several prompt sizes, with the
prompt built to an exact token length using the model's own tokenizer.

Usage:
  python -m converter.bench_prefill --base-url http://127.0.0.1:8127/v1 \
      --model local/Swift-Qwen3.8-27B-Splash --label splash --sizes 2000,8000,32000
"""

from __future__ import annotations

import argparse
import json
import time

from .bench_speed import count_tokens, load_tokenizer, stream_once

FILLER = (
    "The archivist catalogued the seventh crate before noon, noting the serial number, "
    "the provenance, and the condition of each plate. A later revision of the ledger "
    "corrected two entries and appended a remark about the humidity in the east wing. "
)


def build_prompt(target_tokens):
    """A prompt of (approximately) exactly `target_tokens` tokens."""
    text = FILLER
    while count_tokens(text) < target_tokens * 1.1:
        text += FILLER
    ids = load_tokenizer.tokenizer.encode(text, add_special_tokens=False).ids[:target_tokens]
    return load_tokenizer.tokenizer.decode(ids)


def run(base_url, model, sizes, api_key, timeout, log=print):
    rows = []
    for size in sizes:
        body = build_prompt(size)
        prompt = body + "\n\nIn one short sentence, what is this passage about?"
        actual = count_tokens(prompt)
        r = stream_once(base_url, model, prompt, 24, api_key, timeout, True)
        rows.append(dict(requested_tokens=size, prompt_tokens=actual, ttft_s=r["ttft_s"],
                         total_s=r["total_s"], output_tokens=r["tokens"],
                         prefill_tokens_per_second=round(actual / r["ttft_s"], 1)))
        log(f"  prompt {actual:7,} tok → TTFT {r['ttft_s']:7.2f}s  "
            f"({actual / r['ttft_s']:8,.0f} tok/s prefill)  total {r['total_s']:6.2f}s")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_prefill")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--sizes", default="2000,8000,32000")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    p.add_argument("--tokenizer", default="output/swift-splash/tokenizer/tokenizer.json")
    args = p.parse_args(argv)
    key = None
    if args.api_key_file:
        with open(args.api_key_file) as f:
            key = f.read().strip()
    load_tokenizer.tokenizer = load_tokenizer(args.tokenizer)
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"prefill benchmark {args.label}: {args.model}")
    rows = run(args.base_url, args.model, sizes, key, args.timeout)
    result = dict(label=args.label, model=args.model, base_url=args.base_url, rows=rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
