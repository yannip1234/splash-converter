"""Cross-engine output agreement on identical prompts.

Task accuracy saturates once a configuration is healthy — every one of ours scores
95/95 — so it establishes parity but cannot rank. Agreement is finer: with greedy
decoding, two configurations that represent the same weights the same way tend to
produce the same words, and diverge as their numerics diverge.

The pairing that matters is this package against an independent 4-bit group-64
affine quantization of the same checkpoint (mlx_lm). The control pairing is against
the base-model package: different weights, so agreement must drop. If it does not,
the metric is measuring nothing.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from converter.bench_quality import normalise


def load(path):
    d = json.load(open(path))
    answers = {r["prompt"]: normalise(r["answer"]) for r in d["task_rows"]}
    answers.update({r["prompt"]: normalise(r["answer"]) for r in d.get("agreement_rows", [])})
    return d["label"], answers


def token_overlap(a, b):
    """Longest common prefix of words, as a fraction of the shorter answer."""
    x, y = a.split(), b.split()
    n = 0
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]:
            break
        n += 1
    return n / max(1, min(len(x), len(y)))


def main():
    reports = sorted(glob.glob("output/reports/hard-*.json"))
    data = dict(load(p) for p in reports)
    labels = sorted(data)
    print(f"prompts compared: {len(set.intersection(*(set(v) for v in data.values())))}\n")
    print(f"{'pair':44s} {'identical':>10s} {'mean prefix':>12s}")
    seen = set()
    for a in labels:
        for b in labels:
            if a >= b or (a, b) in seen:
                continue
            seen.add((a, b))
            shared = set(data[a]) & set(data[b])
            same = sum(data[a][p] == data[b][p] for p in shared)
            prefix = sum(token_overlap(data[a][p], data[b][p]) for p in shared) / len(shared)
            print(f"  {a} vs {b:28s} {same / len(shared):9.3f} {prefix:12.3f}")


if __name__ == "__main__":
    main()
