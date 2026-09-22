"""Acceptance tests for converter/q4.py (PLAN T1.1).

Independence: the packer is cross-checked byte-for-byte against a standalone
C++ binary that is a verbatim port of the Splash CPU reference `packSlab`
(refs/splash-src/dev/tests/engine/moe_metal_test.mm:98-142) — not a Python
re-implementation sharing indexing code. Hand-computed byte spot checks are
a second independent reference.

Run:  .venv/bin/python -m tests.test_q4
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from converter import q4  # noqa: E402

CPP_SRC = os.path.join(os.path.dirname(__file__), "cpp_ref", "pack_slab_ref.cpp")
CPP_BIN = os.path.join(os.path.dirname(__file__), "cpp_ref", "build", "pack_slab_ref")

MASK64 = (1 << 64) - 1


class LCG:
    """Python mirror of the C++ `Random` (same constants, 64-bit wraparound)."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.state >> 33

    def unit(self) -> np.float32:
        return np.float32(self.next() & 0xFFFFFF) / np.float32(8388608.0) - np.float32(1.0)


def ensure_cpp_bin() -> str:
    os.makedirs(os.path.dirname(CPP_BIN), exist_ok=True)
    if not os.path.exists(CPP_BIN):
        subprocess.run(
            ["clang++", "-O2", "-std=c++17", "-o", CPP_BIN, CPP_SRC],
            check=True,
            capture_output=True,
        )
    return CPP_BIN


def run_cpp(mode_args: list[str], outfile: str) -> None:
    subprocess.run(
        [ensure_cpp_bin(), *mode_args, outfile],
        check=True,
        capture_output=True,
    )


def param_to_logical(flat: np.ndarray, out: int, in_: int) -> np.ndarray:
    """Flat array in C++ parameter(n,g) order -> logical (out, in//64) matrix."""
    t, g = out // q4.STORAGE_N, in_ // q4.GROUP
    return flat.reshape(t, g, q4.STORAGE_N).transpose(0, 2, 1).reshape(out, g)


def logical_to_param(mat: np.ndarray, out: int, in_: int) -> np.ndarray:
    """Logical (out, in//64) matrix -> flat array in C++ parameter(n,g) order."""
    t, g = out // q4.STORAGE_N, in_ // q4.GROUP
    return mat.reshape(t, q4.STORAGE_N, g).transpose(0, 2, 1).reshape(-1)


class TestBf16(unittest.TestCase):
    def test_hand_computed_bits(self):
        # every value cross-verified against compiler __bf16 (bf16_probe, 2026-09-21)
        cases = [
            (0.0, 0x0000),
            (1.0, 0x3F80),
            (-1.0, 0xBF80),
            (-2.5, 0xC020),
            (15.0, 0x4170),
            (-15.0, 0xC170),
            (0.1, 0x3DCD),  # 0x3DCCCCCD -> RNE round-up (lower 0xCCCD > 0x8000)
            (1e38, 0x7E96),  # 0x7E967699 -> 0x7E96 (finite; bf16 max ~3.39e38)
            (3.3e38, 0x7F78),  # 0x7F7843B0 -> 0x7F78 (largest-finite side)
        ]
        for value, want in cases:
            got = int(q4.f32_to_bf16_u16(np.array([value], dtype=np.float32))[0])
            self.assertEqual(got, want, f"bf16({value}) = {got:#06x}, want {want:#06x}")

    def test_tie_rounding_round_to_even(self):
        tie_down = np.array([struct.unpack("<f", struct.pack("<I", 0x3F808000))[0]], dtype=np.float32)
        tie_up = np.array([struct.unpack("<f", struct.pack("<I", 0x3F818000))[0]], dtype=np.float32)
        self.assertEqual(int(q4.f32_to_bf16_u16(tie_down)[0]), 0x3F80)
        self.assertEqual(int(q4.f32_to_bf16_u16(tie_up)[0]), 0x3F82)

    def test_saturation_and_nan(self):
        fmax = np.float32(np.finfo(np.float32).max)  # (2-2^-23)*2^127 > bf16 max
        self.assertEqual(int(q4.f32_to_bf16_u16(np.array([fmax]))[0]), 0x7F80)
        self.assertEqual(int(q4.f32_to_bf16_u16(np.array([-fmax]))[0]), 0xFF80)
        self.assertEqual(int(q4.f32_to_bf16_u16(np.array([np.nan], dtype=np.float32))[0]), 0x7FC0)

    def test_roundtrip_exact_for_representable(self):
        rng = np.random.default_rng(7)
        # 0x7F80 and above are inf/nan patterns; only finite bit patterns round-trip exactly
        u = rng.integers(0, 0x7F80, size=10000).astype(np.uint16)
        f = q4.bf16_u16_to_f32(u)
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(f), u), "bf16 f32->bf16 must be idempotent")


class TestHandComputedPacking(unittest.TestCase):
    """T=1, G=1 corner: byte index == n*32 + j, direct formula."""

    def test_codes_and_scales_layout_256x64(self):
        out, in_ = 256, 64
        codes = ((np.arange(out)[:, None] + np.arange(in_)[None, :]) & 15).astype(np.uint8)
        scales = (np.arange(out, dtype=np.float32)[:, None] * 0.0625)  # exact in bf16
        biases = (-np.arange(out, dtype=np.float32)[:, None] * 0.0625)
        data = q4.pack_projection(codes, scales, biases)
        self.assertEqual(len(data), q4.q4_packed_bytes(out, in_))
        cb, pb = out * in_ // 2, out * in_ // 32

        def code_byte(n, j):
            return data[n * 32 + j]

        self.assertEqual(code_byte(0, 0), (0 & 15) | ((1 & 15) << 4))   # 0x10
        self.assertEqual(code_byte(1, 0), (1 & 15) | ((2 & 15) << 4))   # 0x32
        self.assertEqual(code_byte(255, 31), (317 & 15) | ((318 & 15) << 4))  # 0x65
        self.assertEqual(code_byte(255, 0), (255 & 15) | ((256 & 15) << 4))   # 0x0F

        def bf16_at(off):
            return int.from_bytes(data[off:off + 2], "little")

        self.assertEqual(bf16_at(cb + 0 * 2), 0x0000)     # 0.0
        self.assertEqual(bf16_at(cb + 16 * 2), 0x3F80)    # 1.0
        self.assertEqual(bf16_at(cb + 240 * 2), 0x4170)   # 15.0
        self.assertEqual(bf16_at(cb + pb + 240 * 2), 0xC170)  # -15.0


class TestRoundtrip(unittest.TestCase):
    SHAPES = [(256, 64), (256, 256), (512, 128), (768, 64), (256, 128)]

    def test_projection_exact(self):
        rng = np.random.default_rng(42)
        for out, in_ in self.SHAPES:
            g = in_ // 64
            codes = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
            scales = (0.01 + 0.02 * rng.random((out, g), dtype=np.float32)).astype(np.float32)
            biases = (-0.2 + 0.1 * rng.random((out, g), dtype=np.float32)).astype(np.float32)
            data = q4.pack_projection(codes, scales, biases)
            self.assertEqual(len(data), q4.q4_packed_bytes(out, in_))
            uc, us, ub = q4.unpack_projection(data, out, in_)
            self.assertTrue(np.array_equal(uc, codes), f"codes mismatch {out}x{in_}")
            # scales/biases round-trip exactly through bf16 (both sides bf16-rounded)
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(us), q4.f32_to_bf16_u16(scales)))
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(ub), q4.f32_to_bf16_u16(biases)))

    def test_embedding_exact(self):
        rng = np.random.default_rng(43)
        for out, in_ in [(1024, 5120), (512, 1024)]:
            g = in_ // 64
            codes = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
            scales = (0.01 * rng.random((out, g), dtype=np.float32)).astype(np.float32)
            biases = (rng.random((out, g), dtype=np.float32)).astype(np.float32)
            w, s, b = q4.pack_embedding_rows(codes, scales, biases)
            self.assertEqual(len(w), out * in_ // 2)
            self.assertEqual(len(s), out * in_ // 32)
            self.assertEqual(len(b), out * in_ // 32)
            uc, us, ub = q4.unpack_embedding_rows(w, s, b, out, in_)
            self.assertTrue(np.array_equal(uc, codes), f"embedding codes mismatch {out}x{in_}")
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(us), q4.f32_to_bf16_u16(scales)))
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(ub), q4.f32_to_bf16_u16(biases)))

    def test_dequantize_matches_definition(self):
        rng = np.random.default_rng(44)
        out, in_ = 256, 128
        codes = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
        scales = (0.01 * rng.random((out, in_ // 64), dtype=np.float32)).astype(np.float32)
        biases = (0.05 * rng.random((out, in_ // 64), dtype=np.float32)).astype(np.float32)
        w = q4.dequantize_projection(codes, scales, biases)
        n0, k0 = 7, 33
        self.assertAlmostEqual(
            float(w[n0, k0]), float(codes[n0, k0]) * float(scales[n0, k0 // 64]) + float(biases[n0, k0 // 64]),
            places=6,
        )


class TestCppCrossCheck(unittest.TestCase):
    """Byte-identity against the verbatim C++ packSlab reference."""

    def test_lcg_modes_small_shapes(self):
        cases = [(256, 64, 1), (256, 256, 2), (512, 128, 3), (768, 64, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            for out, in_, seed in cases:
                cpp_out = os.path.join(tmp, f"cpp_{out}x{in_}.bin")
                run_cpp([str(seed), str(out), str(in_)], cpp_out)
                with open(cpp_out, "rb") as f:
                    cpp_bytes = f.read()
                self.assertEqual(len(cpp_bytes), q4.q4_packed_bytes(out, in_))

                # identical inputs via the mirrored LCG (nibbles, scales, biases)
                r = LCG(seed)
                elements = out * in_
                params = elements // 64
                nib = np.array([r.next() & 15 for _ in range(elements)], dtype=np.uint8).reshape(out, in_)
                sc = np.empty(params, dtype=np.float32)
                bi = np.empty(params, dtype=np.float32)
                # C++ consumes one unit() per scale and one per bias, interleaved per i
                for i in range(params):
                    sc[i] = np.float32(0.01) + np.float32(0.01) * (r.unit() + np.float32(1.0))
                    bi[i] = np.float32(-0.2) + np.float32(0.05) * r.unit()

                sc_mat = param_to_logical(sc, out, in_)
                bi_mat = param_to_logical(bi, out, in_)
                py_bytes = q4.pack_projection(nib, sc_mat, bi_mat)
                self.assertEqual(
                    py_bytes, cpp_bytes,
                    f"byte mismatch vs C++ packSlab for {out}x{in_} (first diff at "
                    f"{next((i for i in range(len(py_bytes)) if py_bytes[i] != cpp_bytes[i]), 'n/a')})",
                )
                # unpack the C++ bytes: codes exact; scales/biases bf16-bit exact
                uc, us, ub = q4.unpack_projection(cpp_bytes, out, in_)
                self.assertTrue(np.array_equal(uc, nib))
                self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(us), q4.f32_to_bf16_u16(sc_mat)))
                self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(ub), q4.f32_to_bf16_u16(bi_mat)))

    def test_production_shape_file_mode(self):
        out, in_ = 17408, 5120  # mlp-gate production geometry
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(1234)
            codes = rng.integers(0, 16, size=(out, in_)).astype(np.uint8)
            g = in_ // 64
            scales = (0.01 + 0.02 * rng.random((out, g), dtype=np.float32)).astype(np.float32)
            biases = (-0.2 + 0.1 * rng.random((out, g), dtype=np.float32)).astype(np.float32)
            # C++ file mode indexes scales/biases by parameter(n,g), so the file
            # must carry them in parameter order (nibbles stay in [n][k] order).
            inp = os.path.join(tmp, "inputs.bin")
            with open(inp, "wb") as f:
                f.write(codes.tobytes())
                f.write(logical_to_param(scales, out, in_).astype("<f4").tobytes())
                f.write(logical_to_param(biases, out, in_).astype("<f4").tobytes())
            cpp_out = os.path.join(tmp, "cpp_prod.bin")
            run_cpp([f"@{inp}", str(out), str(in_)], cpp_out)
            with open(cpp_out, "rb") as f:
                cpp_bytes = f.read()
            self.assertEqual(len(cpp_bytes), q4.q4_packed_bytes(out, in_), "size 50,135,040 expected")

            py_bytes = q4.pack_projection(codes, scales, biases)
            self.assertEqual(py_bytes, cpp_bytes, "byte mismatch at production shape 17408x5120")

            uc, us, ub = q4.unpack_projection(cpp_bytes, out, in_)
            self.assertTrue(np.array_equal(uc, codes))
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(us), q4.f32_to_bf16_u16(scales)))
            self.assertTrue(np.array_equal(q4.f32_to_bf16_u16(ub), q4.f32_to_bf16_u16(biases)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
