"""T1.4 — verify the derived Q4 encoder rule against the REAL reference package.

The reference package was built from `Qwen/Qwen3.8-27B` BF16 (provenance proven
in PROGRESS iteration 4), so `converter/encoder.py` applied to that checkpoint
must reproduce the package's own codes/scales/biases. Two independent sections
of `refs/qwen38-splash/target/layer-0.bin` are checked: mlp-gate, mlp-up and
mlp-down. mlp-down carries the loader's `q4(5120x17408)` geometry — the HF
`down_proj` tensor exactly as stored, no transpose and no permutation. (Its byte
count cannot distinguish that from `q4(17408x5120)`, since `out*in*9/16` is
symmetric; only decoding tells them apart, and the wrong one scores 0.4% bias
agreement against 100% for the right one.)

The acceptance bar is NOT bit-identity — the reference encoder's tie-breaking at
exact .5 boundaries is not recoverable from the data, and costs nothing. The bar
is that our reconstruction is no less faithful to the source than the encoder
that produced the shipped package.

Run:  .venv/bin/python tests/test_encoder_t14.py
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
SHARD = os.path.join(ROOT, "refs/qwen-bf16/model-00001-of-00018.safetensors")
LAYER0 = os.path.join(ROOT, "refs/qwen38-splash/target/layer-0.bin")
# section (spec §5.1): offset -> (base-checkpoint tensor, out, in)
SECTIONS = {
    "mlp-gate": (65798144, "model.language_model.layers.0.mlp.gate_proj.weight", 17408, 5120),
    "mlp-up": (115933184, "model.language_model.layers.0.mlp.up_proj.weight", 17408, 5120),
    "mlp-down": (166068224, "model.language_model.layers.0.mlp.down_proj.weight", 5120, 17408),
}


def read_bf16(path, name):
    """Read one BF16 safetensors tensor as float32.

    safetensors `data_offsets` are relative to the start of the data block
    (8 + header length), not to the file. Getting this wrong reads shifted
    bytes that still look like plausible weights — the assert below is the
    cheap guard that catches it.
    """
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


def read_section(offset, out, in_):
    with open(LAYER0, "rb") as f:
        f.seek(offset)
        return f.read(q4.q4_packed_bytes(out, in_))


def fidelity(approx, exact):
    d = (approx - exact).astype(np.float64)
    a, b = approx.ravel().astype(np.float64), exact.ravel().astype(np.float64)
    return dict(
        maxabs=float(np.abs(d).max()),
        meanabs=float(np.abs(d).mean()),
        rmse=float(np.sqrt((d * d).mean())),
        cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
    )


@unittest.skipUnless(os.path.exists(SHARD) and os.path.exists(LAYER0), "refs/ not present")
class EncoderRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = {}
        for label, (offset, tensor, out, in_) in SECTIONS.items():
            w = read_bf16(SHARD, tensor)
            assert w.shape == (out, in_), (label, w.shape)
            data = read_section(offset, out, in_)
            assert len(data) == 50135040, (label, len(data))
            c_ref, s_ref, b_ref = q4.unpack_projection(data, out, in_)
            c_us, s_us, b_us = encoder.quantize_projection(w)
            _, _, k_us = encoder.group_parameters(w)
            cls.case[label] = dict(w=w, c_ref=c_ref, s_ref=s_ref, b_ref=b_ref, c_us=c_us,
                                   s_us=s_us, b_us=b_us, k_us=k_us, out=out, in_=in_)

    def k_reference(self, c):
        """k as stored: b ≈ -k*s, so k = round(-b/s) (integrality asserted below)."""
        return np.rint(np.where(c["s_ref"] != 0, -c["b_ref"] / c["s_ref"], 0)).astype(np.int32)

    def test_bias_is_the_dominant_extreme_exactly(self):
        for label, c in self.case.items():
            with self.subTest(label):
                same = q4.f32_to_bf16_u16(c["b_us"]) == q4.f32_to_bf16_u16(c["b_ref"])
                self.assertEqual(same.mean(), 1.0, "bias must be bit-exact")

    def test_scale_sign_rule(self):
        # s > 0 iff mn+mx < 0; the mn+mx == 0 tie resolves to s < 0.
        for label, c in self.case.items():
            with self.subTest(label):
                self.assertEqual(float(np.mean(np.sign(c["s_us"]) == np.sign(c["s_ref"]))), 1.0)

    def test_scale_magnitude_is_bf16_maxabs_over_k(self):
        for label, c in self.case.items():
            with self.subTest(label):
                k = self.k_reference(c)
                self.assertGreaterEqual(int(k.min()), 1)
                self.assertLessEqual(int(k.max()), 15)
                g = c["w"].reshape(c["out"], c["in_"] // 64, 64)
                maxabs = np.maximum(np.abs(g.min(axis=2)), np.abs(g.max(axis=2)))
                pred = (maxabs / k.astype(np.float32)).astype(np.float32)
                same = q4.f32_to_bf16_u16(pred) == q4.f32_to_bf16_u16(np.abs(c["s_ref"]))
                self.assertEqual(same.mean(), 1.0, "|s| = bf16(maxabs/k) must be exact")

    def test_k_rule_exact_away_from_ties(self):
        for label, c in self.case.items():
            with self.subTest(label):
                k_ref = self.k_reference(c)
                bad = c["k_us"] != k_ref
                self.assertGreaterEqual(float(np.mean(~bad)), 0.999)
                g = c["w"].reshape(c["out"], c["in_"] // 64, 64)
                mn, mx = g.min(axis=2), g.max(axis=2)
                x = 15.0 * np.maximum(np.abs(mn), np.abs(mx)) / (mx - mn)
                is_tie = np.abs(x - np.floor(x) - 0.5) < 1e-6
                # every disagreement is an exact .5 tie, and never more than one step
                self.assertTrue(bool(np.all(is_tie[bad])))
                self.assertTrue(bool(np.all(np.abs(c["k_us"][bad] - k_ref[bad]) == 1)))

    def test_codes_agree(self):
        for label, c in self.case.items():
            with self.subTest(label):
                exact = float(np.mean(c["c_us"] == c["c_ref"]))
                near = float(np.mean(np.abs(c["c_us"].astype(np.int16) - c["c_ref"].astype(np.int16)) <= 1))
                self.assertGreaterEqual(exact, 0.995)
                self.assertGreaterEqual(near, 0.9998)

    def test_no_less_faithful_than_the_reference_encoder(self):
        """The acceptance bar: our reconstruction must not lose accuracy."""
        for label, c in self.case.items():
            with self.subTest(label):
                ours = fidelity(q4.dequantize_projection(c["c_us"], c["s_us"], c["b_us"]), c["w"])
                ref = fidelity(q4.dequantize_projection(c["c_ref"], c["s_ref"], c["b_ref"]), c["w"])
                self.assertLessEqual(ours["rmse"], ref["rmse"])
                self.assertLessEqual(ours["meanabs"], ref["meanabs"])
                self.assertLessEqual(ours["maxabs"], ref["maxabs"])
                self.assertGreaterEqual(ours["cosine"], ref["cosine"])
                self.assertFalse(np.isnan(ours["rmse"]))

    def test_pack_round_trip(self):
        for label, c in self.case.items():
            with self.subTest(label):
                data = q4.pack_projection(c["c_us"], c["s_us"], c["b_us"])
                self.assertEqual(len(data), q4.q4_packed_bytes(c["out"], c["in_"]))
                c2, s2, b2 = q4.unpack_projection(data, c["out"], c["in_"])
                self.assertTrue(np.array_equal(c2, c["c_us"]))
                self.assertTrue(np.array_equal(s2, c["s_us"]))
                self.assertTrue(np.array_equal(b2, c["b_us"]))

    def test_no_nan_or_inf(self):
        for label, c in self.case.items():
            with self.subTest(label):
                deq = q4.dequantize_projection(c["c_us"], c["s_us"], c["b_us"])
                self.assertTrue(bool(np.isfinite(deq).all()))
                self.assertLessEqual(int(c["c_us"].max()), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
