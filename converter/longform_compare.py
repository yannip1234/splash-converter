"""Compare long-form outputs pairwise: exact match and divergence point."""
import glob
import json
import sys


def prefix_words(a, b):
    x, y = a.split(), b.split()
    n = 0
    for i in range(min(len(x), len(y))):
        if x[i] != y[i]:
            break
        n += 1
    return n, min(len(x), len(y))


def main():
    data = {}
    for path in sorted(glob.glob("output/reports/longform-*.json")):
        d = json.load(open(path))
        data[d["label"]] = {r["prompt"]: r["text"] for r in d["rows"]}
    labels = sorted(data)
    print(f"{'pair':52s} {'identical':>10s} {'mean common prefix':>20s}")
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = set(data[a]) & set(data[b])
            same = sum(data[a][p] == data[b][p] for p in shared)
            fracs, words = [], []
            for p in shared:
                n, total = prefix_words(data[a][p], data[b][p])
                fracs.append(n / max(1, total))
                words.append(n)
            print(f"  {a} vs {b:32s} {same}/{len(shared):<6} "
                  f"{sum(fracs) / len(fracs):6.3f}  ({sum(words) / len(words):5.0f} words)")


if __name__ == "__main__":
    main()
