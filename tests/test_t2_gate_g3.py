"""T2 (Gate G3) — one Swift tensor: quantize, pack, independently unpack, measure.

Source : refs/swift-bf16 `model.language_model.layers.0.mlp.gate_proj.weight`
         [17408, 5120] BF16, read from the local shard (no network).
Rule   : converter/encoder.py, pinned in docs/SPLASH_FORMAT_SPEC.md §9.
Bar    : SPEC.md §5 item 3 — reference-encoder parity, cosine >= 0.9958 and
         RMSE <= 9.4e-4. (NOT 0.999: the shipped reference package itself only
         reaches 0.99582 against its own source.)

T2.3 requires an unpack that does not share index logic with the packer, so
`independent_element` below recomputes every byte offset from the addressing
formulae in spec §3 using scalar struct reads — no numpy reshape/transpose, no
call into converter.q4.

Run:  .venv/bin/python tests/test_t2_gate_g3.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import encoder, q4  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SWIFT = os.path.join(ROOT, "refs/swift-bf16")
TENSOR = "model.language_model.layers.0.mlp.gate_proj.weight"
OUT, IN = 17408, 5120
SECTION_BYTES = 50135040  # spec §5.1 mlp-gate
GROUP, TILE = 64, 256
COSINE_MIN, RMSE_MAX = 0.9958, 9.4e-4


def shard_for(name):
    index = json.load(open(os.path.join(SWIFT, "model.safetensors.index.json")))
    return os.path.join(SWIFT, index["weight_map"][name])


def read_bf16(path, name):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        max_end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        assert data_start + max_end == size, "data_offsets are data-relative"
        info = header[name]
        assert info["dtype"] == "BF16", info["dtype"]
        start, end = info["data_offsets"]
        f.seek(data_start + start)
        raw = f.read(end - start)
    bits = np.frombuffer(raw, dtype="<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])


def independent_element(data, out, in_, n, k):
    """Dequantize one element straight from the section bytes (spec §3).

    Deliberately scalar and offset-arithmetic only: this is the check that the
    packer's array gymnastics put every byte where the kernel expects it.
    """
    groups = in_ // GROUP
    g, j = k // GROUP, k % GROUP
    block = ((n // TILE) * groups + g) * TILE * 32 + (n % TILE) * 32
    byte = data[block + j // 2]
    code = byte & 0x0F if j % 2 == 0 else (byte >> 4) & 0x0F

    param = ((n // TILE) * groups + g) * TILE + (n % TILE)
    codes_end = out * in_ // 2
    params_len = out * in_ // 32
    (s_bits,) = struct.unpack_from("<H", data, codes_end + 2 * param)
    (b_bits,) = struct.unpack_from("<H", data, codes_end + params_len + 2 * param)
    to_f32 = lambda bits: struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    return code, code * to_f32(s_bits) + to_f32(b_bits)


@unittest.skipUnless(os.path.isdir(SWIFT), "refs/swift-bf16 not present")
class GateG3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = read_bf16(shard_for(TENSOR), TENSOR)          # T2.1
        cls.codes, cls.scales, cls.biases = encoder.quantize_projection(cls.w)
        cls.data = q4.pack_projection(cls.codes, cls.scales, cls.biases)  # T2.2
        cls.deq = q4.dequantize_projection(cls.codes, cls.scales, cls.biases)

    # ---- T2.1 ----------------------------------------------------------
    def test_source_tensor(self):
        self.assertEqual(self.w.shape, (OUT, IN))
        self.assertTrue(bool(np.isfinite(self.w).all()))

    # ---- T2.2 ----------------------------------------------------------
    def test_section_byte_count(self):
        self.assertEqual(len(self.data), SECTION_BYTES)
        self.assertEqual(len(self.data), q4.q4_packed_bytes(OUT, IN))

    def test_no_nan_or_inf(self):
        self.assertTrue(bool(np.isfinite(self.deq).all()))
        self.assertTrue(bool(np.isfinite(self.scales).all()))
        self.assertTrue(bool(np.isfinite(self.biases).all()))
        self.assertLessEqual(int(self.codes.max()), 15)
        self.assertGreaterEqual(int(self.codes.min()), 0)

    def test_fidelity_meets_spec_threshold(self):
        d = (self.deq - self.w).astype(np.float64)
        a, b = self.deq.ravel().astype(np.float64), self.w.ravel().astype(np.float64)
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rmse = float(np.sqrt((d * d).mean()))
        print(f"\n  max abs {np.abs(d).max():.8f}  mean abs {np.abs(d).mean():.8e}"
              f"  RMSE {rmse:.8e}  cosine {cosine:.8f}")
        self.assertGreaterEqual(cosine, COSINE_MIN)
        self.assertLessEqual(rmse, RMSE_MAX)

    def test_error_is_bounded_by_the_quantization_step(self):
        """In-range values miss by at most half a step of their own group.

        1.4% of elements fall OUTSIDE the representable grid and are clamped:
        k = round(15*maxabs/span) rounds up for ~60% of groups, making 15*|s|
        slightly shorter than the span, so the non-anchored extreme is clipped.
        That is a property of the rule, not of this implementation — the
        reference encoder clips 1.3909% of mlp-gate with a max error identical
        to ours to the last bit (0.0147705078). Measured 2026-09-21.
        """
        scale = np.repeat(self.scales, GROUP, axis=1)  # signed: the grid runs
        bias = np.repeat(self.biases, GROUP, axis=1)   # away from the anchor
        step = np.abs(scale)
        lo = np.minimum(bias, bias + 15 * scale)
        hi = np.maximum(bias, bias + 15 * scale)
        outside = (self.w < lo - 1e-9) | (self.w > hi + 1e-9)
        err = np.abs(self.deq - self.w)
        self.assertEqual(int((~outside & (err > step / 2 + 1e-6)).sum()), 0)
        self.assertLessEqual(float(outside.mean()), 0.015)
        overshoot = np.maximum(lo - self.w, 0) + np.maximum(self.w - hi, 0)
        self.assertAlmostEqual(float(err.max()), float(overshoot.max()), places=9)

    def test_codes_are_a_nearest_grid_point(self):
        """Given (s, b), every code minimizes |w - (b + c*s)|; ties cost nothing."""
        rng = np.random.default_rng(7)
        idx = rng.integers(0, self.w.size, 1_000_000)
        w = self.w.ravel()[idx]
        s = np.repeat(self.scales, GROUP, axis=1).ravel()[idx]
        b = np.repeat(self.biases, GROUP, axis=1).ravel()[idx]
        c = self.codes.ravel()[idx].astype(np.int32)
        cand = b[:, None] + np.arange(16)[None, :] * s[:, None]
        dist = np.abs(cand - w[:, None])
        rows = np.arange(len(c))
        extra = dist[rows, c] - dist.min(axis=1)
        self.assertEqual(float(extra.max()), 0.0, "a code was not a nearest grid point")

    # ---- T2.3: independent unpack --------------------------------------
    def test_independent_unpack_boundaries(self):
        rows = [0, 1, 255, 256, 257, 8704, 17406, 17407]
        cols = [0, 1, 62, 63, 64, 65, 2559, 5056, 5118, 5119]
        for n in rows:
            for k in cols:
                code, value = independent_element(self.data, OUT, IN, n, k)
                self.assertEqual(code, int(self.codes[n, k]), f"code at ({n},{k})")
                self.assertEqual(value, float(self.deq[n, k]), f"value at ({n},{k})")

    def test_independent_unpack_random_sample(self):
        rng = np.random.default_rng(20260921)
        ns = rng.integers(0, OUT, 4000)
        ks = rng.integers(0, IN, 4000)
        worst = 0.0
        for n, k in zip(ns.tolist(), ks.tolist()):
            code, value = independent_element(self.data, OUT, IN, n, k)
            self.assertEqual(code, int(self.codes[n, k]))
            worst = max(worst, abs(value - float(self.w[n, k])))
        # independently-read values track the ORIGINAL source, not just our arrays
        self.assertLess(worst, 0.02)

    def test_independent_unpack_whole_row(self):
        for n in (0, 9999, 17407):
            got = [independent_element(self.data, OUT, IN, n, k)[1] for k in range(IN)]
            self.assertTrue(np.array_equal(np.array(got, dtype=np.float32), self.deq[n]))

    def test_packer_round_trip(self):
        c2, s2, b2 = q4.unpack_projection(self.data, OUT, IN)
        self.assertTrue(np.array_equal(c2, self.codes))
        self.assertTrue(np.array_equal(s2, self.scales))
        self.assertTrue(np.array_equal(b2, self.biases))


if __name__ == "__main__":
    unittest.main(verbosity=2)
