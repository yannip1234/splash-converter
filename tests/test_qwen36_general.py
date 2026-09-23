"""Small real safetensors fixtures for compatible, non-RavenX checkpoints."""

import contextlib
import copy
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from converter import q4
from converter.qwen36_moe import Qwen36Checkpoint, _q8_row, assemble, main
from tests import test_qwen36_moe as format_tests


def checkpoint(root, tensors, *, quant=None, indexed=True):
    config = format_tests.Qwen36FormatTests.config()
    if quant is not None:
        config["quantization_config"] = quant
    (root / "config.json").write_text(json.dumps(config))
    header, data = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data.extend(raw)
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    if indexed:
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    return Qwen36Checkpoint(str(root))


def quantized(name, *, bits=4):
    rows, cols = 256, 64
    codes = (np.arange(rows * cols).reshape(rows, cols) % (1 << bits)).astype(np.uint32)
    packed = np.zeros((rows, cols // (32 // bits)), dtype="<u4")
    for shift in range(32 // bits):
        packed |= codes[:, shift::32 // bits] << (shift * bits)
    base = name.removesuffix("weight")
    scales = np.full((rows, 1), 0.5, dtype=np.float32)
    biases = np.full((rows, 1), -1, dtype=np.float32)
    tensors = {
        name: ("U32", list(packed.shape), packed.tobytes()),
        base + "scales": ("BF16", [rows, 1], q4.f32_to_bf16_bytes(scales)),
        base + "biases": ("BF16", [rows, 1], q4.f32_to_bf16_bytes(biases)),
    }
    return tensors, codes.astype(np.float32) * 0.5 - 1


class GeneralCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_single_safetensors_without_index(self):
        name = "language_model.model.norm.weight"
        source = np.array([1, 2], dtype=np.float32)
        ck = checkpoint(self.root, {name: ("BF16", [2], q4.f32_to_bf16_bytes(source))}, indexed=False)
        np.testing.assert_array_equal(ck.f32(name), source)

    def test_bf16_projection_in_mixed_mlx_checkpoint(self):
        name = "language_model.model.layers.0.mlp.gate.weight"
        source = np.linspace(-1, 1, 256 * 64, dtype=np.float32).reshape(256, 64)
        ck = checkpoint(self.root, {name: ("BF16", [256, 64], q4.f32_to_bf16_bytes(source))},
                        quant={"bits": 4, "group_size": 64, "mode": "affine"})
        codes, scales, biases = ck.q8_parts(name)
        recovered = codes * scales.repeat(64, axis=1) + biases.repeat(64, axis=1)
        np.testing.assert_allclose(recovered, source, atol=0.01)
        self.assertEqual(len(ck.pack_q4(name)), q4.q4_packed_bytes(256, 64))

    def test_four_bit_router_preserves_quantized_values_in_q8(self):
        name = "language_model.model.layers.0.mlp.gate.weight"
        tensors, source = quantized(name)
        ck = checkpoint(self.root, tensors,
                        quant={"bits": 4, "group_size": 64, "mode": "affine"})
        codes, scales, biases = ck.q8_parts(name)
        np.testing.assert_array_equal(codes * scales.repeat(64, axis=1) + biases.repeat(64, axis=1), source)
        packed = ck.pack_q8(name)
        self.assertEqual(len(packed), 256 * 64 * 17 // 16)
        path = self.root / "router.bin"
        path.write_bytes(packed)
        for row in (0, 127, 255):
            np.testing.assert_array_equal(_q8_row(path, 0, 256, 64, row), source[row])

    def test_existing_eight_bit_router_preserves_values(self):
        name = "language_model.model.layers.0.mlp.gate.weight"
        tensors, source = quantized(name, bits=8)
        ck = checkpoint(self.root, tensors, quant={"bits": 4, "group_size": 64, "mode": "affine",
                         name.removesuffix(".weight"): {"bits": 8}})
        np.testing.assert_array_equal(ck.f32(name), source)

    def test_quantization_alias_and_conflicts(self):
        name = "language_model.model.layers.0.self_attn.q_proj.weight"
        tensors, _ = quantized(name)
        checkpoint(self.root, tensors, quant={"bits": 4, "group_size": 64, "mode": "affine"})
        path = self.root / "config.json"
        config = json.loads(path.read_text())
        config["quantization"] = config.pop("quantization_config")
        path.write_text(json.dumps(config))
        self.assertEqual(Qwen36Checkpoint(str(self.root)).source_mode, "mlx4")
        config["quantization_config"] = {"bits": 8, "group_size": 64, "mode": "affine"}
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "conflicting quantization"):
            Qwen36Checkpoint(str(self.root))

    def test_invalid_quantization_override_fails_during_inspection(self):
        name = "language_model.model.layers.0.self_attn.q_proj.weight"
        tensors, _ = quantized(name, bits=8)
        ck = checkpoint(self.root, tensors, quant={"bits": 4, "group_size": 64, "mode": "affine",
                         name.removesuffix(".weight"): {"bits": 8}})
        ck._required = set()
        with self.assertRaisesRegex(ValueError, "Q4 destination"):
            ck.require(name, (256, 64))

    def test_raw_huggingface_layout_rejected_with_export_guidance(self):
        ck = checkpoint(self.root, {"model.language_model.norm.weight":
                        ("BF16", [1], q4.f32_to_bf16_bytes(np.ones(1, np.float32)))})
        with self.assertRaisesRegex(ValueError, "MLX layout first"):
            ck.validate()

    def test_bf16_expert_in_mixed_mlx_checkpoint(self):
        name = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        weights = np.linspace(-1, 1, 2 * 256 * 64, dtype=np.float32).reshape(2, 256, 64)
        ck = checkpoint(self.root, {name: ("BF16", [2, 256, 64], q4.f32_to_bf16_bytes(weights))},
                        quant={"bits": 4, "group_size": 64, "mode": "affine"})
        parts = ck.q4_parts(name, expert=1)
        decoded = parts[0] * parts[1].repeat(64, axis=1) + parts[2].repeat(64, axis=1)
        np.testing.assert_allclose(decoded, weights[1], atol=0.08)

    def test_non_affine_override_rejected(self):
        name = "language_model.model.layers.0.mlp.gate.weight"
        tensors, _ = quantized(name)
        ck = checkpoint(self.root, tensors, quant={"bits": 4, "group_size": 64, "mode": "affine",
                         name.removesuffix(".weight"): {"mode": "mxfp4"}})
        ck._required = set()
        with self.assertRaisesRegex(ValueError, "unsupported quantization mode"):
            ck.require(name, (256, 64))


class GeneralPackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "another-finetune"
        self.reference = self.root / "reference"
        self.out = self.root / "package"
        for folder in (self.source, self.reference / "tokenizer", self.out):
            folder.mkdir(parents=True)
        config = format_tests.Qwen36FormatTests.config()
        config.update(architectures=["Qwen3_5MoeForConditionalGeneration"], image_token_id=2)
        tokenizer = {"model": {"vocab": {"one": 0, "two": 1}},
                     "added_tokens": [{"id": 2, "content": "<image>"}]}
        for folder in (self.source, self.reference / "tokenizer"):
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "tokenizer.json").write_text(json.dumps(tokenizer))
            (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": "source template"}))
        self.ref_manifest = {"schema_version": 4, "model": "local-reference-model",
            "format": {"name": "splash-packed-q4-moe"}, "execution_geometry": {}, "draft": {},
            "upstream": {"draft": {"repo_id": "org/draft"}, "vision": {"repo_id": "org/vision"}},
            "artifacts": []}
        (self.reference / "manifest.json").write_text(json.dumps(self.ref_manifest))
        self.ck = mock.Mock(root=str(self.source), config=config, source_mode="bf16",
                            geometry=format_tests.Qwen36FormatTests.geometry())

    def assemble(self, **kwargs):
        with mock.patch("converter.qwen36_moe.expected_sizes", return_value={}):
            assemble(self.ck, self.out, self.reference, **kwargs)
        return json.loads((self.out / "manifest.json").read_text())

    def test_local_source_is_not_mislabeled_as_ravenx(self):
        manifest = self.assemble()
        self.assertEqual(manifest["model"], "another-finetune")
        self.assertEqual(manifest["upstream"]["target"]["local_name"], "another-finetune")
        self.assertNotIn("repo_id", manifest["upstream"]["target"])
        self.assertNotIn("RavenX", json.dumps(manifest))
        self.assertEqual(manifest["converter"]["reference_package"], "local-reference-model")
        self.assertEqual(len(manifest["converter"]["reference_manifest_sha256"]), 64)

    def test_explicit_source_and_revision_reach_manifest(self):
        manifest = self.assemble(source_repo="org/another-model", source_revision="abc123", model_name="Another-Splash")
        self.assertEqual(manifest["model"], "Another-Splash")
        for part in ("target", "tokenizer"):
            self.assertEqual(manifest["upstream"][part]["repo_id"], "org/another-model")
            self.assertEqual(manifest["upstream"][part]["revision"], "abc123")
        self.assertEqual(manifest["upstream"]["draft"], self.ref_manifest["upstream"]["draft"])

    def test_template_falls_back_to_source_tokenizer_config(self):
        for template in ("my template", {"default": "my template"},
                         [{"name": "default", "template": "my template"}]):
            with self.subTest(template=template):
                (self.source / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
                self.assemble()
                self.assertEqual((self.out / "tokenizer/chat_template.jinja").read_text(), "my template")

    def test_standalone_source_template_takes_precedence(self):
        (self.source / "chat_template.jinja").write_text("standalone source")
        self.assemble()
        self.assertEqual((self.out / "tokenizer/chat_template.jinja").read_text(), "standalone source")

    def test_different_token_ids_reject_reference_before_writing(self):
        path = self.source / "tokenizer.json"
        source = json.loads(path.read_text())
        for change in ("vocab", "added"):
            with self.subTest(change=change):
                altered = copy.deepcopy(source)
                if change == "vocab":
                    altered["model"]["vocab"] = {"one": 1, "two": 0}
                else:
                    altered["added_tokens"][0]["id"] = 3
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "token.*IDs"):
                    self.assemble()
                self.assertEqual(list(self.out.iterdir()), [])

    def test_ambiguous_template_rejected(self):
        (self.source / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "tools", "template": "x"}]}))
        with self.assertRaisesRegex(ValueError, "chat template"):
            self.assemble()

    def test_revision_without_source_repo_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source-repo"):
            self.assemble(source_revision="main")

    def test_cli_rejects_invalid_reference_before_target_writes(self):
        with (mock.patch("converter.qwen36_moe.Qwen36Checkpoint", return_value=self.ck),
              mock.patch("converter.qwen36_moe.write_layer") as write):
            (self.reference / "tokenizer/tokenizer.json").write_text('{"model":{"vocab":{"other":0}}}')
            with self.assertRaisesRegex(ValueError, "token.*IDs"):
                main(["build", "--source", str(self.source), "--out", str(self.out),
                      "--reference", str(self.reference)])
            write.assert_not_called()

    def test_base_model_hint_does_not_override_actual_source_identity(self):
        path = self.source / "config.json"
        config = json.loads(path.read_text())
        config["_name_or_path"] = "org/base-model"
        path.write_text(json.dumps(config))
        manifest = self.assemble()
        self.assertNotIn("repo_id", manifest["upstream"]["target"])

    def test_reference_path_cannot_escape_package(self):
        self.ref_manifest["artifacts"] = [{"path": "draft/../../escape", "size": 0}]
        (self.reference / "manifest.json").write_text(json.dumps(self.ref_manifest))
        with self.assertRaisesRegex(ValueError, "invalid reference artifact path"):
            self.assemble()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_reference_assets_are_copied_and_rehashed(self):
        asset = b"reference draft contents"
        (self.reference / "draft").mkdir()
        (self.reference / "draft/model.bin").write_bytes(asset)
        self.ref_manifest["artifacts"] = [{"path": "draft/model.bin", "size": len(asset),
            "sha256": hashlib.sha256(asset).hexdigest()}]
        (self.reference / "manifest.json").write_text(json.dumps(self.ref_manifest))
        manifest = self.assemble()
        self.assertEqual((self.out / "draft/model.bin").read_bytes(), asset)
        for record in manifest["artifacts"]:
            content = (self.out / record["path"]).read_bytes()
            self.assertEqual(record["size"], len(content))
            self.assertEqual(record["sha256"], hashlib.sha256(content).hexdigest())

    def test_cli_provenance_arguments_reach_assembled_package(self):
        with (mock.patch("converter.qwen36_moe.Qwen36Checkpoint", return_value=self.ck),
              mock.patch("converter.qwen36_moe.expected_sizes", return_value={}),
              mock.patch("converter.qwen36_moe.verify_structure", return_value={}),
              contextlib.redirect_stdout(io.StringIO())):
            main(["assemble", "--source", str(self.source), "--out", str(self.out),
                  "--reference", str(self.reference), "--source-repo", "org/my-finetune",
                  "--source-revision", "abc123", "--model-name", "My-Splash"])
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["model"], "My-Splash")
        self.assertEqual(manifest["upstream"]["target"]["repo_id"], "org/my-finetune")
        self.assertEqual(manifest["upstream"]["target"]["revision"], "abc123")
