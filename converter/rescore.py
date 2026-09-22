"""Re-score saved bench_quality reports with the current matcher.

Scoring rules have been tightened twice (units, then numeric equality). Re-scoring
the stored answers keeps every configuration on identical rules without paying to
re-run the models, and without the temptation to re-run only the ones that improve.
"""
import json, sys, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from converter.bench_quality import answer_matches

for path in sorted(sys.argv[1:] or glob.glob("output/reports/hard-*.json")):
    d = json.load(open(path))
    correct = 0
    for r in d["task_rows"]:
        ok = answer_matches(r["answer"], r["accepted"])
        r["correct"] = ok
        correct += ok
    d["correct"] = correct
    d["accuracy"] = round(correct / d["tasks"], 4)
    json.dump(d, open(path, "w"), indent=1)
    print(f"  {d['label']:16s} {correct:3d}/{d['tasks']}  {d['accuracy']*100:5.1f}%")
