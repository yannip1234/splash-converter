"""T3 (Gate G4) — two complete Swift layers: GDN layer 0 and attention layer 3.

Builds `output/layers/layer-{0,3}.bin` from `refs/swift-bf16` and applies the full
battery: loader-equivalent walk, documented section offsets, header, alignment,
zero padding, full consumption, identity sections byte-compared to the source,
and every Q4 section through the T2 checks.

Thresholds are SPEC.md §5 item 3: cosine >= 0.9955, relative RMSE <= 0.095
(reference-encoder parity, scale-free).

Run:  .venv/bin/python tests/test_t3_layer_g4.py
"""

from __future__ import annotations

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import encoder, layer, q4, verify  # noqa: E402
from converter.checkpoint import Checkpoint  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SWIFT = os.path.join(ROOT, "refs/swift-bf16")
OUTDIR = os.path.join(ROOT, "output/layers")
COSINE_MIN, REL_RMSE_MAX = 0.9955, 0.095

# Offsets as documented in spec §5.1 / §5.2 — ground truth, not derived from our writer.
DOC_OFFSETS = {
    0: [("input-norm", 16384), ("gdn-input", 32768), ("gdn-convolution", 47955968),
        ("gdn-decay", 48037888), ("gdn-time-bias", 48054272), ("gdn-norm", 48070656),
        ("gdn-output", 48087040), ("post-attention-norm", 65781760),
        ("mlp-gate", 65798144), ("mlp-up", 115933184), ("mlp-down", 166068224)],
    3: [("input-norm", 16384), ("attention-input", 32768), ("query-norm", 41320448),
        ("key-norm", 41336832), ("attention-output", 41353216),
        ("post-attention-norm", 59047936), ("mlp-gate", 59064320),
        ("mlp-up", 109199360), ("mlp-down", 159334400)],
}
FILE_BYTES = {0: 216203264, 3: 209469440}
# Q4 sections: label -> (out, in, source accessor)
Q4_SOURCES = {
    0: {"gdn-input": (16640, 5120, lambda ck, p: layer.gdn_input_matrix(ck, p)),
        "gdn-output": (5120, 6144, lambda ck, p: ck.f32(p + "linear_attn.out_proj.weight"))},
    3: {"attention-input": (14336, 5120, lambda ck, p: layer.attention_input_matrix(ck, p)),
        "attention-output": (5120, 6144, lambda ck, p: ck.f32(p + "self_attn.o_proj.weight"))},
}
MLP_SOURCES = {
    "mlp-gate": (17408, 5120, "mlp.gate_proj.weight"),
    "mlp-up": (17408, 5120, "mlp.up_proj.weight"),
    "mlp-down": (5120, 17408, "mlp.down_proj.weight"),
}


def metrics(deq, w):
    d = (deq - w).astype(np.float64)
    a, b = deq.ravel().astype(np.float64), w.ravel().astype(np.float64)
    rmse = float(np.sqrt((d * d).mean()))
    rms = float(np.sqrt((b * b).mean()))
    return dict(
        cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        rmse=rmse, rel=rmse / rms, maxabs=float(np.abs(d).max()),
        finite=bool(np.isfinite(deq).all()), maxcode=None,
    )


@unittest.skipUnless(os.path.isdir(SWIFT), "refs/swift-bf16 not present")
class LayerG4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTDIR, exist_ok=True)
        cls.ck = Checkpoint(SWIFT)
        cls.paths, cls.tables, cls.q4_metrics = {}, {}, {}
        for n in (0, 3):
            path = os.path.join(OUTDIR, f"layer-{n}.bin")
            cls.tables[n] = layer.write_layer(cls.ck, n, path)
            cls.paths[n] = path
            prefix = f"model.language_model.layers.{n}."
            data = open(path, "rb").read()
            offsets = {label: (off, nb) for label, off, nb in cls.tables[n]}
            sources = dict(Q4_SOURCES[n])
            for label, (o, i, tensor) in MLP_SOURCES.items():
                sources[label] = (o, i, (lambda t: (lambda ck, p: ck.f32(p + t)))(tensor))
            cls.q4_metrics[n] = {}
            for label, (o, i, fn) in sources.items():
                off, nb = offsets[label]
                codes, scales, biases = q4.unpack_projection(data[off:off + nb], o, i)
                w = fn(cls.ck, prefix)
                m = metrics(q4.dequantize_projection(codes, scales, biases), w)
                m["maxcode"] = int(codes.max())
                m["bytes"] = nb
                m["expected_bytes"] = q4.q4_packed_bytes(o, i)
                cls.q4_metrics[n][label] = m
                del w, codes, scales, biases
            del data

    # ---- structure -----------------------------------------------------
    def test_loader_walk_and_full_consumption(self):
        for n in (0, 3):
            with self.subTest(layer=n):
                walked = verify.walk_target_layer(self.paths[n], n)  # raises if invalid
                self.assertEqual(walked, self.tables[n])

    def test_offsets_match_the_documented_layout(self):
        for n in (0, 3):
            with self.subTest(layer=n):
                got = [(label, off) for label, off, _ in self.tables[n]]
                self.assertEqual(got, DOC_OFFSETS[n])

    def test_file_size_and_header(self):
        for n in (0, 3):
            with self.subTest(layer=n):
                self.assertEqual(os.path.getsize(self.paths[n]), FILE_BYTES[n])
                with open(self.paths[n], "rb") as f:
                    head = f.read(16)
                self.assertEqual(head[:8], b"MDFL0006")
                self.assertEqual(struct.unpack("<I", head[8:12])[0], n)
                self.assertEqual(struct.unpack("<I", head[12:16])[0],
                                 1 if layer.is_full_attention(n) else 0)

    def test_alignment_and_zero_padding(self):
        for n in (0, 3):
            with self.subTest(layer=n):
                data = open(self.paths[n], "rb").read()
                end = 16
                for label, off, nb in self.tables[n]:
                    self.assertEqual(off % verify.ALIGN, 0, f"{label} not aligned")
                    self.assertEqual(data[end:off], b"\x00" * (off - end), f"pad before {label}")
                    end = off + nb
                self.assertEqual(data[end:], b"\x00" * (len(data) - end), "tail padding")

    # ---- identity / transformed non-Q4 sections ------------------------
    def test_identity_sections_are_byte_copies_of_the_source(self):
        cases = {
            0: [("gdn-convolution", "linear_attn.conv1d.weight"),
                ("gdn-time-bias", "linear_attn.dt_bias"),
                ("gdn-norm", "linear_attn.norm.weight")],
            3: [],
        }
        for n, items in cases.items():
            data = open(self.paths[n], "rb").read()
            offsets = {label: (off, nb) for label, off, nb in self.tables[n]}
            for label, tensor in items:
                with self.subTest(layer=n, section=label):
                    off, nb = offsets[label]
                    src = self.ck.raw(f"model.language_model.layers.{n}." + tensor)
                    self.assertEqual(len(src), nb)
                    self.assertEqual(data[off:off + nb], src)

    def test_norm_sections_carry_the_unit_offset(self):
        """input-norm and post-attention-norm store bf16(γ+1); gdn-norm stores γ."""
        for n in (0, 3):
            data = open(self.paths[n], "rb").read()
            offsets = {label: (off, nb) for label, off, nb in self.tables[n]}
            prefix = f"model.language_model.layers.{n}."
            pairs = [("input-norm", "input_layernorm.weight"),
                     ("post-attention-norm", "post_attention_layernorm.weight")]
            if layer.is_full_attention(n):
                pairs += [("query-norm", "self_attn.q_norm.weight"),
                          ("key-norm", "self_attn.k_norm.weight")]
            for label, tensor in pairs:
                with self.subTest(layer=n, section=label):
                    off, nb = offsets[label]
                    gamma = self.ck.f32(prefix + tensor)
                    self.assertEqual(data[off:off + nb], layer.norm_plus_one_bytes(gamma))
                    self.assertNotEqual(data[off:off + nb], self.ck.raw(prefix + tensor))

    def test_gdn_decay_is_negative_exp(self):
        off, nb = {l: (o, n) for l, o, n in self.tables[0]}["gdn-decay"]
        data = open(self.paths[0], "rb").read()[off:off + nb]
        a_log = self.ck.f32("model.language_model.layers.0.linear_attn.A_log")
        self.assertEqual(nb, 192)
        got = np.frombuffer(data, dtype="<f4")
        self.assertTrue(np.array_equal(got, (-np.exp(a_log.astype(np.float32))).astype(np.float32)))
        self.assertTrue(bool((got < 0).all()), "decay must be negative")

    # ---- Q4 sections ---------------------------------------------------
    def test_q4_byte_counts(self):
        for n, sections in self.q4_metrics.items():
            for label, m in sections.items():
                with self.subTest(layer=n, section=label):
                    self.assertEqual(m["bytes"], m["expected_bytes"])

    def test_q4_no_nan_or_inf(self):
        for n, sections in self.q4_metrics.items():
            for label, m in sections.items():
                with self.subTest(layer=n, section=label):
                    self.assertTrue(m["finite"])
                    self.assertLessEqual(m["maxcode"], 15)

    def test_q4_fidelity_meets_spec_threshold(self):
        for n, sections in self.q4_metrics.items():
            for label, m in sections.items():
                with self.subTest(layer=n, section=label):
                    print(f"\n  L{n} {label:18s} cosine {m['cosine']:.8f}  relRMSE {m['rel']:.5f}"
                          f"  RMSE {m['rmse']:.4e}  maxabs {m['maxabs']:.6f}")
                    self.assertGreaterEqual(m["cosine"], COSINE_MIN)
                    self.assertLessEqual(m["rel"], REL_RMSE_MAX)

    def test_gdn_input_padding_rows_are_zero(self):
        """The 160 tail rows of the packed GDN row must dequantize to exactly 0."""
        off, nb = {l: (o, n) for l, o, n in self.tables[0]}["gdn-input"]
        data = open(self.paths[0], "rb").read()[off:off + nb]
        codes, scales, biases = q4.unpack_projection(data, 16640, 5120)
        pad = slice(16480, 16640)
        self.assertEqual(int(codes[pad].max()), 0)
        self.assertTrue(bool((scales[pad] == 0).all()))
        self.assertTrue(bool((biases[pad] == 0).all()))
        # and stored as +0.0, matching the reference package
        self.assertEqual(int(q4.f32_to_bf16_u16(scales[pad]).max()), 0)


class NormConventionTest(unittest.TestCase):
    """Every bf16 norm section must use the SAME convention as the reference package.

    This is the check that was missing. The old test asserted our identity sections
    matched *our own source*, which is self-consistent and proves nothing about
    whether identity was the right transform. Here each norm section of the shipped
    reference package is compared against Swift's gamma and gamma+1 — Swift is a
    light fine-tune, so whichever is closer is the convention the package uses — and
    the builder is required to agree.

    Shipping query-norm/key-norm as identity copies cost 27.5 points of task
    accuracy and was invisible to every numerical check (PROGRESS iteration 13).
    """

    REFERENCE = os.path.join(ROOT, "refs/qwen38-splash/target")

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "refs/qwen38-splash")),
                         "reference package not present")
    def test_norm_sections_match_the_reference_convention(self):
        ck = Checkpoint(SWIFT)
        cases = [
            (0, "input-norm", "input_layernorm.weight", True),
            (0, "post-attention-norm", "post_attention_layernorm.weight", True),
            (0, "gdn-norm", "linear_attn.norm.weight", False),
            (3, "input-norm", "input_layernorm.weight", True),
            (3, "post-attention-norm", "post_attention_layernorm.weight", True),
            (3, "query-norm", "self_attn.q_norm.weight", True),
            (3, "key-norm", "self_attn.k_norm.weight", True),
        ]
        for n, section, tensor, expect_offset in cases:
            with self.subTest(layer=n, section=section):
                path = os.path.join(self.REFERENCE, f"layer-{n}.bin")
                table = {l: (o, b) for l, o, b in verify.walk_target_layer(path, n)}
                off, nbytes = table[section]
                with open(path, "rb") as f:
                    f.seek(off)
                    raw = f.read(nbytes)
                reference = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                gamma = ck.f32(f"model.language_model.layers.{n}." + tensor)
                to_gamma = float(np.abs(reference - gamma).max())
                to_offset = float(np.abs(reference - (gamma + 1.0)).max())
                uses_offset = to_offset < to_gamma
                self.assertEqual(uses_offset, expect_offset,
                                 f"{section}: |ref-γ|={to_gamma:.5f} |ref-(γ+1)|={to_offset:.5f}")
                # and the builder must produce that convention
                built = (layer.norm_plus_one_bytes(gamma) if uses_offset
                         else ck.raw(f"model.language_model.layers.{n}." + tensor))
                sections = dict(layer.attention_sections(ck, f"model.language_model.layers.{n}.")
                                if layer.is_full_attention(n)
                                else layer.gdn_sections(ck, f"model.language_model.layers.{n}."))
                self.assertEqual(sections[section], built)


if __name__ == "__main__":
    unittest.main(verbosity=2)
