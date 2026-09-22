"""Long-form greedy agreement: the version of the test that discriminates.

Short canonical answers ("4", "Paris") are identical across every healthy
configuration, so agreement on them measures correctness, not numerics — the
base-model package agreed with this one exactly as much as an independent
quantization of the *same* weights did. Long free-form continuations at
temperature 0 diverge as soon as two configurations disagree about any logit,
so the divergence point is a real measure of how alike their numerics are.

Reported per pair: exact-match rate, and the mean number of identical leading
tokens before the first difference.
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Write a detailed paragraph explaining how a hash table works, including collisions.",
    "Describe the water cycle in detail, covering evaporation, condensation and precipitation.",
    "Explain the causes of the First World War in a detailed paragraph.",
    "Write a short story, about 150 words, about a lighthouse keeper who finds a message.",
    "Explain how a transformer neural network processes a sequence, in detail.",
    "Describe how vaccines train the immune system, in a detailed paragraph.",
    "Explain the difference between TCP and UDP in detail, with examples of each.",
    "Write a detailed explanation of how photosynthesis converts light into chemical energy.",
    "Describe the plot of Hamlet in detail, act by act.",
    "Explain how an internal combustion engine works through its four strokes.",
]


def generate(base_url, model, prompt, api_key, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/chat/completions", body, headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    message = payload["choices"][0]["message"]
    return (message.get("content") or "").strip()


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_longform")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args(argv)
    key = None
    if args.api_key_file:
        with open(args.api_key_file) as f:
            key = f.read().strip()
    rows = []
    for prompt in PROMPTS:
        text = generate(args.base_url, args.model, prompt, key, args.max_tokens, args.timeout)
        rows.append({"prompt": prompt, "text": text})
        print(f"  {len(text.split()):4d} words  {prompt[:52]}", flush=True)
    json.dump({"label": args.label, "model": args.model, "rows": rows},
              open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
