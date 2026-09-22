"""T6 (Gate G5) — the package in the real, unmodified Splash runtime.

Validates the evidence recorded by an actual run against
`/opt/homebrew/opt/splash/libexec/engine/splash` via the stock `server/server.py`,
with the package served in place (the installed Splash package is untouched).

Regenerate the evidence by serving the package and re-running the capture:
    $P/python/bin/python3 -u $P/server/server.py output/swift-splash/target \
        output/swift-splash/draft --tokenizer output/swift-splash/tokenizer \
        --model local/Swift-Qwen3.8-27B-Splash --binary $P/engine/splash --port 8127 --no-webui
    python -m converter.bench_draft --label swift --out output/reports/draft-swift.json

Run:  .venv/bin/python tests/test_t6_runtime_g5.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import package  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPORT = os.path.join(ROOT, "output/reports/g5-runtime.json")
PACKAGE = os.path.join(ROOT, "output/swift-splash")

have_report = os.path.exists(REPORT)


@unittest.skipUnless(have_report, "no G5 runtime evidence recorded yet")
class RuntimeG5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPORT) as f:
            cls.report = json.load(f)

    def test_package_loaded_in_the_unmodified_runtime(self):
        """T6.1 — the stock engine binary accepted the package."""
        self.assertTrue(self.report["loaded"])
        self.assertEqual(self.report["maximum_context_tokens"], 262144)
        self.assertTrue(self.report["runtime"]["binary"].endswith("engine/splash"))

    def test_engine_layout_hash_matches_our_computed_fingerprint(self):
        """The engine's own weight-record hash equals converter/package.py's."""
        self.assertTrue(self.report["layout_sha_matches_computed"])
        self.assertEqual(self.report["loaded_model_layout_sha256"],
                         self.report["computed_manifest_fingerprint"])
        if os.path.isdir(PACKAGE):
            recomputed = package.manifest_fingerprint(package.weight_records(PACKAGE))
            self.assertEqual(recomputed, self.report["loaded_model_layout_sha256"])

    def test_generated_text_is_coherent_and_correct(self):
        """T6.2 — real answers, cleanly terminated."""
        answers = {g["prompt"]: (g["content"] or "").strip() for g in self.report["generations"]}
        for g in self.report["generations"]:
            with self.subTest(g["prompt"][:32]):
                self.assertEqual(g["finish_reason"], "stop")
                self.assertTrue((g["content"] or "").strip(), "empty completion")
        self.assertEqual(answers["What is 2+2? Reply with just the number."], "4")
        self.assertIn("Paris", answers["Name the capital of France."])
        sky = answers["Explain in two sentences why the sky is blue."].lower()
        self.assertIn("scatter", sky)

    def test_draft_acceptance_is_measured_for_both_packages(self):
        """T6.3 — the drafter figure, and the base-model control it is judged against."""
        bench = self.report["draft_benchmark"]
        for label in ("swift", "base_reference"):
            self.assertIn(label, bench)
            self.assertGreater(bench[label]["drafted_tokens"], 0)
            self.assertIsNotNone(bench[label]["acceptance"])
        self.assertGreater(bench["base_reference"]["acceptance"], bench["swift"]["acceptance"],
                           "expected the drafter to do better on the model it was trained for")

    def test_swift_benchmark_is_reproducible(self):
        bench = self.report["draft_benchmark"]
        if "swift_run2" not in bench:
            self.skipTest("only one Swift run recorded")
        a, b = bench["swift"], bench["swift_run2"]
        self.assertEqual(a["completion_tokens"], b["completion_tokens"])
        self.assertEqual(a["drafted_tokens"], b["drafted_tokens"])
        self.assertEqual(a["accepted_draft_tokens"], b["accepted_draft_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
