"""T5 — the assembled package: reuse, tokenizer, manifest, whole-package harness.

Runs against `output/swift-splash/`, produced by:
    python -m converter.build layers|head|embedding
    python -m converter.package assemble

Run:  .venv/bin/python tests/test_t5_package.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import package  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "output/swift-splash")
REFERENCE = os.path.join(ROOT, "refs/qwen38-splash")
SWIFT = os.path.join(ROOT, "refs/swift-bf16")

have_package = os.path.exists(os.path.join(OUT, "manifest.json"))


@unittest.skipUnless(have_package, "run `python -m converter.package assemble` first")
class PackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(OUT, "manifest.json")) as f:
            cls.manifest = json.load(f)
        with open(os.path.join(REFERENCE, "manifest.json")) as f:
            cls.reference = json.load(f)

    def test_whole_package_harness_passes(self):
        """T5.5 — loader- and installer-equivalent checks over every file."""
        result = package.verify_package(OUT, full_hash=True, log=lambda *a: None)
        self.assertEqual(result["artifacts"], 78)
        self.assertEqual(result["weight_files"], 73)
        self.assertEqual(len(result["packageManifestSha256"]), 64)
        self.assertEqual(len(result["manifestFingerprintSha256"]), 64)
        # Files present but not manifest artifacts. The installer permits these and
        # the reference package ships its own; the check exists to catch strays
        # (.DS_Store, stray logs), so it allows exactly the publication set.
        allowed = {"LICENSE", "LICENSE-APACHE-2.0", "NOTICE", "README.md",
                   "figures/throughput.svg", "figures/concurrency.svg"}
        self.assertEqual(set(result["undeclared_files"]) - allowed, set(),
                         "unexpected file in the package")

    def test_manifest_fields(self):
        m = self.manifest
        self.assertEqual(m["schema_version"], 3)
        self.assertTrue(isinstance(m["model"], str) and m["model"].strip())
        self.assertEqual(m["format"], package.FORMAT)
        # execution_geometry must equal the compiled constants, i.e. the reference's
        self.assertEqual(m["execution_geometry"], self.reference["execution_geometry"])

    def test_artifact_records_are_exactly_three_keys(self):
        """install/models.py:146 rejects any extra key."""
        for record in self.manifest["artifacts"]:
            self.assertEqual(set(record), {"path", "size", "sha256"})
        paths = [r["path"] for r in self.manifest["artifacts"]]
        self.assertEqual(len(paths), len(set(paths)), "duplicate artifact paths")
        self.assertEqual(paths, sorted(paths), "artifacts should be path-sorted")

    def test_structural_fingerprint_matches_the_reference_package(self):
        """Same file set, sizes, magics, layer/type — so the loader sees the same records."""
        ours = package.manifest_fingerprint(package.weight_records(OUT))
        theirs = package.manifest_fingerprint(package.weight_records(REFERENCE))
        self.assertEqual(ours, theirs)

    def test_draft_and_vision_are_byte_reused(self):
        """T5.1 / T5.2 — reused unchanged from the reference package."""
        for rel in [f"draft/layer-{n}.bin" for n in range(5)] + ["draft/model.bin",
                                                                 "vision/model.bin"]:
            with self.subTest(rel):
                self.assertEqual(package.sha256_file(os.path.join(OUT, rel)),
                                 package.sha256_file(os.path.join(REFERENCE, rel)))

    def test_tokenizer_comes_from_swift(self):
        """T5.3 — Swift's own files (= base), not the package's MLX repack (§2.1)."""
        differing = 0
        for name in package.TOKENIZER_FILES:
            with self.subTest(name):
                ours = package.sha256_file(os.path.join(OUT, "tokenizer", name))
                self.assertEqual(ours, package.sha256_file(os.path.join(SWIFT, name)))
                if ours != package.sha256_file(os.path.join(REFERENCE, "tokenizer", name)):
                    differing += 1
        self.assertEqual(differing, 4, "only vocab.json matches the reference package")

    def test_tokenizer_loads_and_agrees_with_the_reference(self):
        """The pinned tokenizers==0.22.2 must parse Swift's legacy merges format."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers not installed")
        ours = Tokenizer.from_file(os.path.join(OUT, "tokenizer", "tokenizer.json"))
        theirs = Tokenizer.from_file(os.path.join(REFERENCE, "tokenizer", "tokenizer.json"))
        self.assertEqual(ours.get_vocab_size(), theirs.get_vocab_size())
        for text in ["Hello, world!", " leading space", "naïve café — résumé",
                     "<|im_start|>system\nYou are helpful.<|im_end|>\n",
                     "def f(x):\n    return x**2\n", "日本語のテキスト"]:
            with self.subTest(text[:20]):
                self.assertEqual(ours.encode(text).ids, theirs.encode(text).ids)

    def test_upstream_records_swift_provenance(self):
        up = self.manifest["upstream"]
        self.assertEqual(up["target"]["repo_id"], "ukisai/Swift-Qwen3.8-27b")
        self.assertEqual(up["target"]["revision"],
                         "048328f4059015b63f860a453bf94834af0db683")
        self.assertEqual(up["tokenizer"]["repo_id"], "ukisai/Swift-Qwen3.8-27b")
        # reused parts keep the reference package's provenance
        self.assertEqual(up["draft"], self.reference["upstream"]["draft"])
        self.assertEqual(up["vision"], self.reference["upstream"]["vision"])

    def test_vision_reuse_is_justified_completely_not_by_sample(self):
        """T5.1 — Swift's vision tower is byte-identical to base, all 333 tensors.

        SWIFT_COMPATIBILITY §5 justified the byte-reuse on a 7-tensor sample. All
        333 `model.visual.*` tensors happen to live in base shard 1, which is
        local, so the claim can be settled completely rather than sampled: if any
        differed, we would be shipping a base vision tower with Swift language
        weights.
        """
        from converter.checkpoint import Checkpoint
        swift = Checkpoint(SWIFT)
        base_root = os.path.join(ROOT, "refs/qwen-bf16")
        if not os.path.exists(os.path.join(base_root, "model-00001-of-00018.safetensors")):
            self.skipTest("base shard 1 not present")
        base = Checkpoint(base_root)
        names = sorted(k for k in swift.weight_map if "visual" in k)
        self.assertEqual(len(names), 333)
        for name in names:
            self.assertIn(name, base, f"{name} absent from base")
            if swift.raw(name) != base.raw(name):
                self.fail(f"{name} differs from base — vision reuse is NOT justified")

    def test_reference_package_also_passes_the_harness(self):
        """The harness is meaningful only if it accepts a known-good package."""
        result = package.verify_package(REFERENCE, full_hash=False, log=lambda *a: None)
        self.assertEqual(result["artifacts"], 78)
        self.assertEqual(result["weight_files"], 73)


if __name__ == "__main__":
    unittest.main(verbosity=2)
