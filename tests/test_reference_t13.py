"""T1.3 — verify converter/q4.py unpack against the REAL Splash reference package.

Dequantizes `refs/qwen38-splash/target/layer-0.bin` mlp-gate section
(offset 65,798,144, 50,135,040 B, Q4(17408x5120), docs/SPLASH_FORMAT_SPEC.md §5.1)
and checks that the per-group affine statistics and row-0 dequantization
reproduce the measured values recorded in spec §3.6.

This is an end-to-end check of the unpacker against ground-truth package bytes —
independent of the C++ packSlab cross-check (T1.1).

Run:  .venv/bin/python tests/test_reference_t13.py
"""

from __future__ import annotations

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import q4  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAYER0 = os.path.join(ROOT, "refs", "qwen38-splash", "target", "layer-0.bin")

# docs/SPLASH_FORMAT_SPEC.md §5.1 (GDN layer, 216,203,264 B)
FILE_SIZE = 216_203_264
MAGIC = b"MDFL0006"
LAYER = 0
TYPE_ = 0
MLP_GATE_OFFSET = 65_798_144
MLP_GATE_BYTES = 50_135_040
OUT, IN = 17408, 5120


@unittest.skipUnless(os.path.exists(LAYER0), f"reference package missing: {LAYER0}")
class TestReferenceLayer0MlpGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(LAYER0, "rb") as f:
            header = f.read(16)
            cls.magic = header[:8]
            cls.layer, cls.type = struct.unpack("<II", header[8:16])
            f.seek(MLP_GATE_OFFSET)
            cls.section = f.read(MLP_GATE_BYTES)
        cls.codes, cls.scales, cls.biases = q4.unpack_projection(
            cls.section, OUT, IN
        )

    def test_header_and_section_geometry(self):
        self.assertEqual(os.path.getsize(LAYER0), FILE_SIZE)
        self.assertEqual(self.magic, MAGIC)
        self.assertEqual(self.layer, LAYER)
        self.assertEqual(self.type, TYPE_)
        self.assertEqual(len(self.section), MLP_GATE_BYTES)
        self.assertEqual(len(self.section), q4.q4_packed_bytes(OUT, IN))
        self.assertEqual(self.codes.shape, (OUT, IN))
        self.assertEqual(self.scales.shape, (OUT, IN // 64))
        self.assertEqual(self.biases.shape, (OUT, IN // 64))
        self.assertTrue((self.codes <= 15).all(), "codes out of 4-bit range")
        self.assertTrue(np.isfinite(self.scales).all())
        self.assertTrue(np.isfinite(self.biases).all())

    def test_group_affine_statistics_reproduce_spec_36(self):
        # spec §3.6 (corrected 2026-09-21, true bf16 decode):
        # |scale| p0.5/p50/p99.5 = 0.002106/0.003098/0.005127,
        # |bias|   p0.5/p50/p99.5 = 0.017090/0.025757/0.046631,
        # sign(scale)==-sign(bias) in 100% of groups, bias/scale mean -8.3779
        s, b = self.scales, self.biases
        self.assertEqual(int((s == 0).sum() + (b == 0).sum()), 0, "zero scale/bias present")
        sign_match = float(np.mean(np.sign(s) == -np.sign(b)))
        self.assertEqual(sign_match, 1.0, f"sign agreement {sign_match:.6f}, spec says 100%")

        ratio = float(np.mean(b / s))
        self.assertAlmostEqual(ratio, -8.3779, delta=0.05, msg=f"bias/scale mean {ratio}")

        abs_s = np.abs(s)
        abs_b = np.abs(b)
        self.assertAlmostEqual(float(np.percentile(abs_s, 50)), 0.003098, delta=0.0001,
                               msg=f"|scale| p50 {np.percentile(abs_s, 50)}")
        self.assertTrue(
            0.001 <= float(np.percentile(abs_s, 0.5)) and float(np.percentile(abs_s, 99.5)) <= 0.01,
            f"|scale| p0.5/p99.5 = {np.percentile(abs_s, 0.5):.6f}/{np.percentile(abs_s, 99.5):.6f}",
        )
        self.assertAlmostEqual(float(np.percentile(abs_b, 50)), 0.025757, delta=0.0005,
                               msg=f"|bias| p50 {np.percentile(abs_b, 50)}")
        self.assertTrue(
            0.008 <= float(np.percentile(abs_b, 0.5)) and float(np.percentile(abs_b, 99.5)) <= 0.08,
            f"|bias| p0.5/p99.5 = {np.percentile(abs_b, 0.5):.6f}/{np.percentile(abs_b, 99.5):.6f}",
        )

    def test_row0_dequantization_reproduces_spec_36(self):
        # spec §3.6 (corrected): row-0 min -0.038330, max +0.039795,
        # mean 0.000120, std 0.010972 (~1/sqrt(5120) = 0.014 scale, sane gate weights)
        w0 = q4.dequantize_projection(self.codes[:1], self.scales[:1], self.biases[:1])[0]
        self.assertTrue(np.isfinite(w0).all())
        self.assertAlmostEqual(float(w0.min()), -0.038330, delta=0.0005, msg=f"min {w0.min()}")
        self.assertAlmostEqual(float(w0.max()), 0.039795, delta=0.0005, msg=f"max {w0.max()}")
        self.assertAlmostEqual(float(w0.mean()), 0.000120, delta=0.0002, msg=f"mean {w0.mean()}")
        self.assertAlmostEqual(float(w0.std()), 0.010972, delta=0.0002, msg=f"std {w0.std()}")

    def test_code_histogram_single_dominant_peak_at_8(self):
        # spec §3.6 (corrected): single dominant peak at code 8 (10,734,108),
        # 9 close second (10,625,498); mean code 8.2386
        hist = np.bincount(self.codes.ravel(), minlength=16)
        peak = int(np.argmax(hist))
        self.assertIn(peak, (8, 9), f"peak at {peak}: {hist.tolist()}")
        self.assertTrue(int(hist[peak]) > int(hist[:6].max()), f"low tail too high: {hist.tolist()}")
        self.assertTrue(int(hist[peak]) > int(hist[peak + 2:].max()), f"high tail too high: {hist.tolist()}")
        self.assertAlmostEqual(float(self.codes.mean()), 8.2386, delta=0.01,
                               msg=f"mean code {self.codes.mean()}")

    def test_smoke_dequantize_sample_rows(self):
        # a few full rows, far from row 0, must be finite and in a sane range
        for n in (1000, 8000, 17407):
            w = q4.dequantize_projection(
                self.codes[n : n + 1], self.scales[n : n + 1], self.biases[n : n + 1]
            )[0]
            self.assertTrue(np.isfinite(w).all(), f"row {n} not finite")
            self.assertTrue(
                float(np.abs(w).max()) < 100.0, f"row {n} out of sane range: {np.abs(w).max()}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
