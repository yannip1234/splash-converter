"""Extensive performance suite: prefill, decode and concurrency vs context depth.

Reports, per context size:
  prefill tok/s   prompt tokens / time-to-first-token
  TTFT            wall time to the first streamed token
  decode tok/s    (tokens - 1) / (total - ttft), i.e. generation excluding prefill
  acceptance      DFlash draft acceptance for that request (Splash only, from /metrics)

Decode rate at depth is the number an agent workload actually feels: the KV cache
grows with context, so throughput measured on a short prompt is optimistic.

Tokens are counted with the model's own tokenizer for every engine, because
Splash streams one token per SSE delta while oMLX batches several — counting
deltas undercounts oMLX roughly 8x.

Every prompt carries a unique random prefix. Without one, a sweep built from the
same filler makes each prompt a prefix of the next, and Splash's prefix cache
serves half of it for free: a 32,831-token prompt logged `cached 16,416` and
reported 1,462 tok/s where the true cold figure is ~880. Salting the *start* of
the prompt defeats prefix matching for all engines alike.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .bench_speed import count_tokens, load_tokenizer, stream_once

FILLER = (
    "The archivist catalogued the seventh crate before noon, noting the serial number, "
    "the provenance, and the condition of each plate. A later revision of the ledger "
    "corrected two entries and appended a remark about the humidity in the east wing. "
)
QUESTION = "\n\nIn two sentences, summarise the passage above."


def metrics(port, api_key=None):
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/metrics")
        with urllib.request.urlopen(request, timeout=20) as response:
            out = {}
            for line in response.read().decode().splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                key, _, value = line.rpartition(" ")
                try:
                    out[key] = float(value)
                except ValueError:
                    pass
            return out
    except Exception:
        return {}


def build_prompt(target_tokens, salt=None):
    """A prompt of ~target_tokens, unique at the front so no prefix cache applies."""
    salt = salt or secrets.token_hex(16)
    head = f"Reference {salt}. "
    text = head + FILLER
    while count_tokens(text) < target_tokens * 1.15:
        text += FILLER
    ids = load_tokenizer.tokenizer.encode(text, add_special_tokens=False).ids[:target_tokens]
    return load_tokenizer.tokenizer.decode(ids)


def one_point(base_url, model, size, api_key, max_tokens, timeout, port):
    prompt = build_prompt(size) + QUESTION
    actual = count_tokens(prompt)
    before = metrics(port) if port else {}
    r = stream_once(base_url, model, prompt, max_tokens, api_key, timeout, True)
    after = metrics(port) if port else {}
    drafted = after.get("splash_drafted_tokens_total", 0) - before.get("splash_drafted_tokens_total", 0)
    accepted = after.get("splash_accepted_draft_tokens_total", 0) - before.get("splash_accepted_draft_tokens_total", 0)
    return dict(
        requested=size, prompt_tokens=actual, ttft_s=r["ttft_s"], total_s=r["total_s"],
        output_tokens=r["tokens"], finish=r["finish"],
        prefill_tps=round(actual / r["ttft_s"], 1) if r["ttft_s"] else None,
        decode_tps=r["decode_tps"],
        acceptance=round(accepted / drafted, 4) if drafted else None,
    )


def concurrency(base_url, model, api_key, timeout, size, streams, max_tokens):
    def run(index):
        prompt = build_prompt(size) + QUESTION
        return stream_once(base_url, model, prompt, max_tokens, api_key, timeout, True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=streams) as pool:
        results = list(pool.map(run, range(streams)))
    elapsed = time.perf_counter() - started
    tokens = sum(r["tokens"] for r in results)
    return dict(streams=streams, wall_s=round(elapsed, 2), total_tokens=tokens,
                aggregate_tps=round(tokens / elapsed, 2),
                per_stream_tps=round(statistics.fmean(r["decode_tps"] for r in results
                                                     if r["decode_tps"]), 2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_suite")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", default="512,1024,2048,4096,8192,16384,32768,65536")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--metrics-port", type=int, default=None,
                   help="port exposing /metrics for draft acceptance (Splash only)")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--concurrency-size", type=int, default=2048)
    p.add_argument("--tokenizer", default="output/swift-splash/tokenizer/tokenizer.json")
    args = p.parse_args(argv)
    key = None
    if args.api_key_file:
        with open(args.api_key_file) as f:
            key = f.read().strip()
    load_tokenizer.tokenizer = load_tokenizer(args.tokenizer)

    print(f"=== {args.label}: {args.model}")
    print(f"{'prompt':>9} {'TTFT':>8} {'prefill t/s':>12} {'decode t/s':>11} "
          f"{'accept':>7} {'out':>6}")
    rows = []
    print("  (each prompt uniquely salted; cache hits would otherwise double these)")
    for size in [int(s) for s in args.sizes.split(",")]:
        r = one_point(args.base_url, args.model, size, key, args.max_tokens,
                      args.timeout, args.metrics_port)
        rows.append(r)
        print(f"{r['prompt_tokens']:>9,} {r['ttft_s']:>7.2f}s {r['prefill_tps']:>12,.0f} "
              f"{str(r['decode_tps']):>11} {str(r['acceptance']):>7} {r['output_tokens']:>6}",
              flush=True)

    conc = []
    if args.concurrency:
        print(f"\nconcurrency at {args.concurrency_size} tokens:")
        print(f"{'streams':>8} {'wall':>8} {'aggregate t/s':>14} {'per-stream t/s':>15}")
        for streams in [int(s) for s in args.concurrency.split(",")]:
            c = concurrency(args.base_url, args.model, key, args.timeout,
                            args.concurrency_size, streams, args.max_tokens)
            conc.append(c)
            print(f"{c['streams']:>8} {c['wall_s']:>7.2f}s {c['aggregate_tps']:>14} "
                  f"{c['per_stream_tps']:>15}", flush=True)

    json.dump({"label": args.label, "model": args.model, "depth": rows,
               "concurrency": conc}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
