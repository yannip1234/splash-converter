"""Format checks independent of a 69 GB BF16 checkpoint.

Full checkpoint and runtime checks are performed by the CLI's `validate`
command on the conversion host and by Splash on Apple Silicon.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from converter import encoder, q4
from converter import mlx_affine
from converter.qwen36_moe import (
    Geometry, Qwen36Checkpoint, _q4_row, _q8_row, align, expected_sizes,
    q8_bytes, q8_pack, section_offsets, section_specs,
)


class Qwen36FormatTests(unittest.TestCase):
    @staticmethod
    def geometry():
        text = {
            "model_type": "qwen3_5_moe_text", "num_hidden_layers": 40,
            "hidden_size": 2048, "vocab_size": 248320,
            "num_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "max_position_embeddings": 262144, "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4, "attn_output_gate": True,
            "hidden_act": "silu", "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10_000_000}, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "layer_types": ["full_attention" if (n + 1) % 4 == 0
                            else "linear_attention" for n in range(40)],
        }
        return Geometry.from_config({"model_type": "qwen3_5_moe", "text_config": text})

    def test_schema4_section_geometry(self):
        # Current Splash Qwen3.6 binary sizes, checked against its published
        # layout.json and WeightStore.cpp's section walks.
        g = self.geometry()
        sizes = expected_sizes(g)
        self.assertEqual(sizes["target/layer-0.bin"], 475201536)
        self.assertEqual(sizes["target/layer-3.bin"], 471285760)
        self.assertEqual(sizes["target/head.bin"], 286097408)
        self.assertEqual(sizes["target/embedding.bin"], 286081024)
        self.assertEqual(sum(sizes.values()), 19541082112)
        self.assertEqual(section_offsets(section_specs(g, 0))[0][9],
                         ("experts-gate", 19890176, 150994944))

    def test_q8_router_row_round_trip(self):
        rng = np.random.default_rng(20)
        source = rng.normal(0, 0.1, (256, 128)).astype(np.float32)
        packed = q8_pack(source)
        self.assertEqual(len(packed), q8_bytes(256, 128))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.bin"
            path.write_bytes(b"\0" * align(16) + packed)
            for row in (0, 127, 255):
                recovered = _q8_row(path, align(16), 256, 128, row)
                relative = np.linalg.norm(recovered - source[row]) / np.linalg.norm(source[row])
                self.assertLess(relative, 0.02)

    def test_q4_projection_row_round_trip(self):
        rng = np.random.default_rng(21)
        source = rng.normal(0, 0.1, (256, 128)).astype(np.float32)
        packed = encoder.quantize_and_pack(source)
        self.assertEqual(len(packed), q4.q4_packed_bytes(256, 128))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection.bin"
            path.write_bytes(b"\0" * align(16) + packed)
            for row in (0, 127, 255):
                recovered = _q4_row(path, align(16), 256, 128, row)
                relative = np.linalg.norm(recovered - source[row]) / np.linalg.norm(source[row])
                self.assertLess(relative, 0.2)

    def test_mlx_u32_nibble_order(self):
        words = np.array([0x76543210] * 8, dtype="<u4")
        codes = mlx_affine.unpack_codes(words.tobytes(), 1, 64, 4)
        self.assertEqual(codes[0].tolist(), list(range(8)) * 8)

    def test_local_mlx4_repack_preserves_values(self):
        source = Path(__file__).parents[2] / (
            "RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-"
            "Pentester-BugHunter-RATH-mlx-mlx-4Bit")
        if not source.exists():
            self.skipTest("local MLX 4-bit checkpoint unavailable")
        ck = Qwen36Checkpoint(str(source))
        ck.validate()
        self.assertEqual((ck.source_mode, len(ck._required)), ("mlx4", 1757))
        name = "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight"
        packed = ck.pack_q4(name)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repacked.bin"
            path.write_bytes(packed)
            for row in (0, 255, 511):
                self.assertTrue(np.array_equal(
                    _q4_row(path, 0, 512, 2048, row),
                    ck.f32_rows(name, row, row + 1)[0]))


if __name__ == "__main__":
    unittest.main()
