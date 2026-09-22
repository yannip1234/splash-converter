"""T7.1 — Definition-of-Done review against SPEC.md §5, item by item.

Each test maps to one numbered DoD item and asserts it from the artifacts on
disk, so "done" is checkable rather than asserted in prose.

Set SWIFT_SPLASH_FULL_HASH=1 to re-hash all 18 source shards (≈52 GB) in item 7
instead of a three-shard sample.

Run:  .venv/bin/python tests/test_t7_done.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import build, layer, package  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "output/swift-splash")
REFERENCE = os.path.join(ROOT, "refs/qwen38-splash")
SWIFT = os.path.join(ROOT, "refs/swift-bf16")
LAYER_REPORT = os.path.join(ROOT, "output/reports/target-layers.json")
G5_REPORT = os.path.join(ROOT, "output/reports/g5-runtime.json")

have_package = os.path.exists(os.path.join(OUT, "manifest.json"))


@unittest.skipUnless(have_package, "package not built")
class DefinitionOfDoneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(OUT, "manifest.json")) as f:
            cls.manifest = json.load(f)
        with open(os.path.join(REFERENCE, "manifest.json")) as f:
            cls.reference = json.load(f)

    # DoD 1
    def test_item1_package_is_complete_and_target_comes_from_swift(self):
        paths = {a["path"] for a in self.manifest["artifacts"]}
        weights = {p for p in paths if p.endswith(".bin")}
        self.assertEqual(len(weights), 73)
        self.assertEqual(len(paths - weights), 5, "five tokenizer files")
        self.assertTrue(os.path.exists(os.path.join(OUT, "manifest.json")))
        # every target file must DIFFER from the reference package: no base reuse
        ours = {a["path"]: a["sha256"] for a in self.manifest["artifacts"]}
        theirs = {a["path"]: a["sha256"] for a in self.reference["artifacts"]}
        target = [p for p in ours if p.startswith("target/")]
        self.assertEqual(len(target), 66)
        for p in target:
            with self.subTest(p):
                self.assertNotEqual(ours[p], theirs[p], f"{p} was reused from base")

    # DoD 2
    def test_item2_vision_and_draft_are_reused_with_hashes_recorded(self):
        ours = {a["path"]: a["sha256"] for a in self.manifest["artifacts"]}
        theirs = {a["path"]: a["sha256"] for a in self.reference["artifacts"]}
        reused = ["vision/model.bin", "draft/model.bin"] + [f"draft/layer-{n}.bin"
                                                            for n in range(5)]
        for p in reused:
            with self.subTest(p):
                self.assertEqual(ours[p], theirs[p])
                self.assertEqual(len(ours[p]), 64, "sha256 recorded in the manifest")

    # DoD 3
    def test_item3_every_q4_section_validated_within_threshold(self):
        with open(LAYER_REPORT) as f:
            report = json.load(f)
        self.assertEqual([e["layer"] for e in report["layers"]], list(range(64)))
        sections = [(e["layer"], k, m) for e in report["layers"]
                    for k, m in e["sections"].items()]
        self.assertEqual(len(sections), 320)
        for n, label, m in sections:
            with self.subTest(layer=n, section=label):
                self.assertGreaterEqual(m["cosine"], build.COSINE_MIN)
                self.assertLessEqual(m["rel_rmse"], build.REL_RMSE_MAX)
                for key in ("rmse", "max_abs", "cosine", "rel_rmse"):
                    self.assertIn(key, m)

    # DoD 4
    def test_item4_package_passes_the_loader_equivalent_harness(self):
        result = package.verify_package(OUT, full_hash=True, log=lambda *a: None)
        self.assertEqual(result["artifacts"], 78)
        self.assertEqual(result["weight_files"], 73)
        self.assertEqual(self.manifest["format"], package.FORMAT)
        self.assertEqual(self.manifest["execution_geometry"],
                         self.reference["execution_geometry"])
        # both fingerprints computed, and the manifest hash is self-consistent
        self.assertEqual(result["packageManifestSha256"],
                         package.sha256_file(os.path.join(OUT, "manifest.json")))
        self.assertEqual(len(result["manifestFingerprintSha256"]), 64)

    # DoD 5
    def test_item5_runtime_generated_coherent_text(self):
        with open(G5_REPORT) as f:
            g5 = json.load(f)
        self.assertTrue(g5["loaded"])
        answers = {g["prompt"]: (g["content"] or "").strip() for g in g5["generations"]}
        self.assertEqual(answers["What is 2+2? Reply with just the number."], "4")
        self.assertIn("Paris", answers["Name the capital of France."])

    # DoD 6
    def test_item6_drafter_benchmarked_and_unmodified(self):
        with open(G5_REPORT) as f:
            g5 = json.load(f)
        bench = g5["draft_benchmark"]
        self.assertIsNotNone(bench["swift"]["acceptance"])
        self.assertIsNotNone(bench["base_reference"]["acceptance"])
        for n in range(5):
            rel = f"draft/layer-{n}.bin"
            self.assertEqual(package.sha256_file(os.path.join(OUT, rel)),
                             package.sha256_file(os.path.join(REFERENCE, rel)))
        self.assertEqual(package.sha256_file(os.path.join(OUT, "draft/model.bin")),
                         package.sha256_file(os.path.join(REFERENCE, "draft/model.bin")))

    # DoD 7
    def test_item7_originals_unmodified(self):
        meta = os.path.join(SWIFT, ".cache/huggingface/download")
        shards = sorted(n for n in os.listdir(SWIFT) if n.endswith(".safetensors"))
        self.assertEqual(len(shards), 18)
        if not os.environ.get("SWIFT_SPLASH_FULL_HASH"):
            shards = shards[:1] + shards[9:10] + shards[-1:]
        for name in shards:
            with self.subTest(name):
                path = os.path.join(meta, name + ".metadata")
                self.assertTrue(os.path.exists(path), "LFS metadata missing")
                with open(path) as f:
                    tokens = f.read().split()
                expected = next(t for t in tokens if len(t) == 64
                                and all(c in "0123456789abcdef" for c in t))
                self.assertEqual(package.sha256_file(os.path.join(SWIFT, name)), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
