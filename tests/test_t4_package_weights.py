"""T4 — all 64 target layers plus head and embedding.

The artifacts are produced by `python -m converter.build` (12.8 GB); this suite
does not rebuild them. It re-validates a representative sample directly, and
checks the build report covers all 64 layers within the SPEC §5 item 3 gate.

Sampled layers deliberately include 59 and 63 — the heavy-tailed late layers that
sit at the format's practical limit and that a threshold fitted to layer 0 alone
would reject.

Run:  .venv/bin/python tests/test_t4_package_weights.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import build, layer, verify  # noqa: E402
from converter.checkpoint import Checkpoint  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SWIFT = os.path.join(ROOT, "refs/swift-bf16")
TARGET = os.path.join(ROOT, "output/swift-splash/target")
REPORT = os.path.join(ROOT, "output/reports/target-layers.json")
SAMPLE_LAYERS = [0, 3, 31, 59, 63]

have_artifacts = os.path.isdir(TARGET) and os.path.exists(os.path.join(TARGET, "layer-63.bin"))


@unittest.skipUnless(have_artifacts and os.path.isdir(SWIFT),
                     "run `python -m converter.build layers|head|embedding` first")
class PackageWeightsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck = Checkpoint(SWIFT)

    def test_all_64_layer_files_exist_with_exact_sizes(self):
        for n in range(64):
            path = os.path.join(TARGET, f"layer-{n}.bin")
            with self.subTest(layer=n):
                self.assertTrue(os.path.exists(path), f"missing layer-{n}.bin")
                expected = (layer.ATTENTION_LAYER_BYTES if layer.is_full_attention(n)
                            else layer.GDN_LAYER_BYTES)
                self.assertEqual(os.path.getsize(path), expected)

    def test_every_layer_passes_the_loader_walk(self):
        """Structure only — cheap, so it runs over all 64."""
        for n in range(64):
            with self.subTest(layer=n):
                verify.walk_target_layer(os.path.join(TARGET, f"layer-{n}.bin"), n)

    def test_sampled_layers_pass_the_full_battery(self):
        for n in SAMPLE_LAYERS:
            with self.subTest(layer=n):
                entry = build.validate_target_layer(
                    self.ck, os.path.join(TARGET, f"layer-{n}.bin"), n)
                self.assertEqual(entry["layer"], n)
                for label, m in entry["sections"].items():
                    self.assertGreaterEqual(m["cosine"], build.COSINE_MIN, f"{label}")
                    self.assertLessEqual(m["rel_rmse"], build.REL_RMSE_MAX, f"{label}")

    def test_head_and_embedding(self):
        head = build.validate_head(self.ck, os.path.join(TARGET, "head.bin"))
        self.assertEqual(head["size"], layer.HEAD_BYTES)
        emb = build.validate_embedding(self.ck, os.path.join(TARGET, "embedding.bin"))
        self.assertEqual(emb["size"], layer.EMBEDDING_BYTES)
        for entry in (head, emb):
            m = list(entry["sections"].values())[0]
            self.assertGreaterEqual(m["cosine"], build.COSINE_MIN)
            self.assertLessEqual(m["rel_rmse"], build.REL_RMSE_MAX)

    def test_report_covers_all_64_layers_within_the_gate(self):
        self.assertTrue(os.path.exists(REPORT), "build report missing")
        with open(REPORT) as f:
            report = json.load(f)
        layers = report["layers"]
        self.assertEqual([e["layer"] for e in layers], list(range(64)))
        sections = [(e["layer"], label, m) for e in layers for label, m in e["sections"].items()]
        self.assertEqual(len(sections), 320, "64 layers x 5 Q4 sections")
        for n, label, m in sections:
            self.assertGreaterEqual(m["cosine"], build.COSINE_MIN, f"layer {n} {label}")
            self.assertLessEqual(m["rel_rmse"], build.REL_RMSE_MAX, f"layer {n} {label}")
        kinds = [e["kind"] for e in layers]
        self.assertEqual(kinds.count("attention"), 16)
        self.assertEqual(kinds.count("gdn"), 48)

    def test_layer_kind_matches_the_geometry_rule(self):
        for n in range(64):
            with self.subTest(layer=n):
                self.assertEqual(layer.is_full_attention(n), (n + 1) % 4 == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
