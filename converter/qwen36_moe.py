"""Qwen3.6 / Qwen3.5-MoE BF16 or MLX 4-bit -> Splash schema-4 converter.

The section sequence and dimensions mirror Splash's QwenTarget.hpp,
QwenTarget.cpp, Qwen3_6Moe.cpp and WeightStore.cpp. Routed expert slabs are
written one at a time, in source expert index order; no complete MoE tensor is
materialized in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import encoder, mlx_affine, q4
from .checkpoint import Checkpoint
from .layer import decay_bytes

ALIGN = 16_384
LAYER_MAGIC = b"MDFM0001"
HEAD_MAGIC = b"MDFM0002"
EMBEDDING_MAGIC = b"MDFE0001"
REFERENCE_ID = "incoai/Qwen3.6-35B-A3B-Splash"
RAVENX_ID = "deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx"
RAVENX_MLX4_ID = "Ck-TnT/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx-mlx-4Bit"


def align(value: int) -> int:
    return (value + ALIGN - 1) // ALIGN * ALIGN


def q8_bytes(rows: int, cols: int) -> int:
    if rows % 256 or cols % 64:
        raise ValueError(f"invalid Q8 geometry {rows}x{cols}")
    return rows * cols * 17 // 16


@dataclass(frozen=True)
class Geometry:
    layers: int
    hidden: int
    vocab: int
    experts: int
    active_experts: int
    expert_hidden: int
    shared_hidden: int
    attention_width: int
    key_width: int
    gdn_conv: int
    gdn_value_heads: int
    gdn_head_dim: int
    gdn_packed: int
    full_packed: int
    layer_types: tuple[str, ...]
    tied_head: bool

    @property
    def gdn_actual_width(self) -> int:
        return self.gdn_conv + self.attention_width + 2 * self.gdn_value_heads

    @classmethod
    def from_config(cls, config: dict) -> "Geometry":
        t = config.get("text_config", config)
        if config.get("model_type") != "qwen3_5_moe" or t.get("model_type") != "qwen3_5_moe_text":
            raise ValueError("expected a Qwen3.6/Qwen3.5-MoE text checkpoint")
        types = tuple(t.get("layer_types", ()))
        expected = tuple("full_attention" if (n + 1) % 4 == 0 else "linear_attention"
                         for n in range(40))
        if types != expected:
            raise ValueError("Splash requires the 3 GDN + 1 full-attention pattern over 40 layers")
        g = cls(
            layers=t["num_hidden_layers"], hidden=t["hidden_size"],
            vocab=t["vocab_size"], experts=t["num_experts"],
            active_experts=t["num_experts_per_tok"],
            expert_hidden=t["moe_intermediate_size"],
            shared_hidden=t["shared_expert_intermediate_size"],
            attention_width=t["num_attention_heads"] * t["head_dim"],
            key_width=t["num_key_value_heads"] * t["head_dim"],
            gdn_conv=(2 * t["linear_num_key_heads"] + t["linear_num_value_heads"])
                     * t["linear_key_head_dim"],
            gdn_value_heads=t["linear_num_value_heads"],
            gdn_head_dim=t["linear_key_head_dim"],
            gdn_packed=align_width((2 * t["linear_num_key_heads"] + t["linear_num_value_heads"])
                                   * t["linear_key_head_dim"]
                                   + t["num_attention_heads"] * t["head_dim"]
                                   + 2 * t["linear_num_value_heads"]),
            full_packed=2 * t["num_attention_heads"] * t["head_dim"]
                        + 2 * t["num_key_value_heads"] * t["head_dim"],
            layer_types=types, tied_head=t.get("tie_word_embeddings", False),
        )
        # Current Splash binaries compile these exact dimensions into kernels.
        required = dict(layers=40, hidden=2048, vocab=248320, experts=256,
                        active_experts=8, expert_hidden=512, shared_hidden=512,
                        attention_width=4096, key_width=512, gdn_conv=8192,
                        gdn_value_heads=32, gdn_head_dim=128,
                        gdn_packed=12544, full_packed=9216)
        for field, value in required.items():
            if getattr(g, field) != value:
                raise ValueError(f"Splash Qwen3.6 expects {field}={value}, got {getattr(g, field)}")
        if t.get("max_position_embeddings") != 262144 or t.get("full_attention_interval") != 4:
            raise ValueError("unsupported context or attention interval")
        if (t.get("linear_conv_kernel_dim") != 4 or
                t.get("linear_value_head_dim") != 128 or
                t.get("attn_output_gate") is not True or
                t.get("hidden_act") != "silu" or
                t.get("partial_rotary_factor") != 0.25 or
                t.get("rope_parameters", {}).get("rope_theta") != 10_000_000 or
                t.get("rms_norm_eps") != 1e-6):
            raise ValueError("Qwen3.6 execution settings differ from Splash's compiled model")
        return g


def align_width(width: int) -> int:
    return (width + 255) // 256 * 256


class Qwen36Checkpoint(Checkpoint):
    def __init__(self, root: str):
        super().__init__(root)
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.geometry = Geometry.from_config(self.config)
        quant = self.config.get("quantization_config")
        if quant is None:
            self.source_mode = "bf16"
        elif (quant.get("bits"), quant.get("group_size"), quant.get("mode")) == (4, 64, "affine"):
            self.source_mode = "mlx4"
        else:
            raise ValueError("supported inputs are BF16 or MLX affine 4-bit/group-64")

    def bits_for(self, name: str) -> int:
        base = name.removesuffix(".weight")
        quant = self.config.get("quantization_config") or {}
        override = quant.get(base, {})
        if override.get("group_size", 64) != 64:
            raise ValueError(f"{base}: unsupported group size")
        bits = override.get("bits", quant.get("bits", 4))
        if bits not in (4, 8):
            raise ValueError(f"{base}: unsupported {bits}-bit quantization")
        return bits

    def require(self, name: str, shape: tuple[int, ...]) -> None:
        if name not in self:
            raise ValueError(f"missing tensor: {name}")
        actual = self.shape(name)
        quantized = (self.source_mode == "mlx4" and name.endswith(".weight") and
                     name.removesuffix("weight") + "scales" in self)
        if quantized:
            bits = self.bits_for(name)
            packed = shape[:-1] + (shape[-1] // (32 // bits),)
            parameters = shape[:-1] + (shape[-1] // 64,)
            if actual != packed or self.dtype(name) != "U32":
                raise ValueError(f"{name}: expected U32 {packed}, got {self.dtype(name)} {actual}")
            for suffix in ("scales", "biases"):
                param = name.removesuffix("weight") + suffix
                if param not in self or self.shape(param) != parameters or self.dtype(param) != "BF16":
                    raise ValueError(f"{param}: expected BF16 {parameters}")
                self._required.add(param)
        elif actual != shape or self.dtype(name) != "BF16":
            raise ValueError(f"{name}: expected BF16 {shape}, got {self.dtype(name)} {actual}")
        self._required.add(name)

    def _quant_parts(self, name: str, start: int, stop: int,
                     expert: int | None = None):
        bits = self.bits_for(name)
        shape = self.shape(name)
        logical_cols = shape[-1] * (32 // bits)
        rows = stop - start
        codes = mlx_affine.unpack_codes(
            self.tensor_rows_raw(name, start, stop, expert), rows, logical_cols, bits)
        base = name.removesuffix("weight")
        parameters = []
        for suffix in ("scales", "biases"):
            raw = self.tensor_rows_raw(base + suffix, start, stop, expert)
            parameters.append(q4.bf16_bytes_to_f32(raw).reshape(rows, logical_cols // 64))
        return codes, *parameters

    def q4_parts(self, name: str, start: int = 0, stop: int | None = None,
                 expert: int | None = None):
        shape = self.shape(name)
        rows = shape[-2]
        stop = rows if stop is None else stop
        if self.source_mode == "mlx4":
            if self.bits_for(name) != 4:
                raise ValueError(f"{name}: expected MLX 4-bit")
            return self._quant_parts(name, start, stop, expert)
        w = (super().f32_expert(name, expert)[start:stop] if expert is not None
             else super().f32_rows(name, start, stop))
        return encoder.quantize_projection(w)

    def q8_parts(self, name: str, start: int = 0, stop: int | None = None):
        rows = self.shape(name)[0]
        stop = rows if stop is None else stop
        if self.source_mode == "mlx4":
            if self.bits_for(name) != 8:
                raise ValueError(f"{name}: expected MLX 8-bit")
            return self._quant_parts(name, start, stop)
        return mlx_affine.q8_quantize_parts(super().f32_rows(name, start, stop))

    def pack_q4(self, name: str, *, expert: int | None = None) -> bytes:
        return q4.pack_projection(*self.q4_parts(name, expert=expert))

    def pack_q8(self, name: str, *, pad_rows: int = 0) -> bytes:
        codes, scales, biases = self.q8_parts(name)
        if pad_rows:
            codes = np.pad(codes, ((0, pad_rows), (0, 0)))
            scales = np.pad(scales, ((0, pad_rows), (0, 0)))
            biases = np.pad(biases, ((0, pad_rows), (0, 0)))
        return mlx_affine.q8_pack_parts(codes, scales, biases)

    def f32(self, name: str) -> np.ndarray:
        if self.dtype(name) == "BF16":
            return super().f32(name)
        if len(self.shape(name)) != 2:
            raise ValueError(f"{name}: use f32_expert for rank-3 quantized tensor")
        return self.f32_rows(name, 0, self.shape(name)[0])

    def f32_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        if self.dtype(name) == "BF16":
            return super().f32_rows(name, start, stop)
        parts = (self.q8_parts(name, start, stop) if self.bits_for(name) == 8
                 else self.q4_parts(name, start, stop))
        return mlx_affine.dequantize(*parts)

    def f32_expert(self, name: str, expert: int) -> np.ndarray:
        if self.dtype(name) == "BF16":
            return super().f32_expert(name, expert)
        return mlx_affine.dequantize(*self.q4_parts(name, expert=expert))

    def validate(self) -> None:
        g = self.geometry
        self._required = set()
        self.require("language_model.model.embed_tokens.weight", (g.vocab, g.hidden))
        self.require("language_model.model.norm.weight", (g.hidden,))
        if not g.tied_head:
            self.require("language_model.lm_head.weight", (g.vocab, g.hidden))
        for n in range(g.layers):
            p = f"language_model.model.layers.{n}."
            self.require(p + "input_layernorm.weight", (g.hidden,))
            self.require(p + "post_attention_layernorm.weight", (g.hidden,))
            if g.layer_types[n] == "full_attention":
                for name, shape in {
                    "self_attn.q_proj.weight": (g.attention_width * 2, g.hidden),
                    "self_attn.k_proj.weight": (g.key_width, g.hidden),
                    "self_attn.v_proj.weight": (g.key_width, g.hidden),
                    "self_attn.o_proj.weight": (g.hidden, g.attention_width),
                    "self_attn.q_norm.weight": (256,),
                    "self_attn.k_norm.weight": (256,),
                }.items():
                    self.require(p + name, shape)
            else:
                for name, shape in {
                    "linear_attn.in_proj_qkv.weight": (g.gdn_conv, g.hidden),
                    "linear_attn.in_proj_z.weight": (g.attention_width, g.hidden),
                    "linear_attn.in_proj_b.weight": (g.gdn_value_heads, g.hidden),
                    "linear_attn.in_proj_a.weight": (g.gdn_value_heads, g.hidden),
                    "linear_attn.out_proj.weight": (g.hidden, g.attention_width),
                    "linear_attn.conv1d.weight": (g.gdn_conv, 4, 1),
                    "linear_attn.A_log": (g.gdn_value_heads,),
                    "linear_attn.dt_bias": (g.gdn_value_heads,),
                    "linear_attn.norm.weight": (g.gdn_head_dim,),
                }.items():
                    self.require(p + name, shape)
            for name, shape in {
                "mlp.gate.weight": (g.experts, g.hidden),
                "mlp.switch_mlp.gate_proj.weight": (g.experts, g.expert_hidden, g.hidden),
                "mlp.switch_mlp.up_proj.weight": (g.experts, g.expert_hidden, g.hidden),
                "mlp.switch_mlp.down_proj.weight": (g.experts, g.hidden, g.expert_hidden),
                "mlp.shared_expert.gate_proj.weight": (g.shared_hidden, g.hidden),
                "mlp.shared_expert.up_proj.weight": (g.shared_hidden, g.hidden),
                "mlp.shared_expert.down_proj.weight": (g.hidden, g.shared_hidden),
                "mlp.shared_expert_gate.weight": (1, g.hidden),
            }.items():
                self.require(p + name, shape)
        extras = set(self.weight_map) - self._required
        if extras:
            raise ValueError(f"unmapped source tensors: {sorted(extras)[:8]}")


def q8_pack(weights: np.ndarray) -> bytes:
    """Splash Q8: [tile][group][row][64 codes], then BF16 scales/biases."""
    rows, cols = weights.shape
    if rows % 256 or cols % 64:
        raise ValueError(f"invalid Q8 shape {weights.shape}")
    result = mlx_affine.q8_pack_parts(*mlx_affine.q8_quantize_parts(weights))
    assert len(result) == q8_bytes(rows, cols)
    return result


def section_specs(g: Geometry, n: int) -> list[tuple[str, int]]:
    h, a = g.hidden, g.attention_width
    result = [("input-norm", h * 2)]
    if g.layer_types[n] == "full_attention":
        result += [("attention-input", q4.q4_packed_bytes(g.full_packed, h)),
                   ("query-norm", 512), ("key-norm", 512),
                   ("attention-output", q4.q4_packed_bytes(h, a))]
    else:
        result += [("gdn-input", q4.q4_packed_bytes(g.gdn_packed, h)),
                   ("gdn-convolution", g.gdn_conv * 4 * 2),
                   ("gdn-decay", g.gdn_value_heads * 4),
                   ("gdn-time-bias", g.gdn_value_heads * 2),
                   ("gdn-norm", g.gdn_head_dim * 2),
                   ("gdn-output", q4.q4_packed_bytes(h, a))]
    result += [("post-attention-norm", h * 2),
               ("router", q8_bytes(g.experts, h))]
    for label, rows, cols in (("experts-gate", g.expert_hidden, h),
                              ("experts-up", g.expert_hidden, h),
                              ("experts-down", h, g.expert_hidden)):
        result.append((label, g.experts * q4.q4_packed_bytes(rows, cols)))
    for label, rows, cols in (("shared-expert-gate", g.shared_hidden, h),
                              ("shared-expert-up", g.shared_hidden, h),
                              ("shared-expert-down", h, g.shared_hidden)):
        result.append((label, q4.q4_packed_bytes(rows, cols)))
    result.append(("shared-expert-scalar-gate", q8_bytes(256, h)))
    return result


def section_offsets(specs: list[tuple[str, int]]) -> tuple[list[tuple[str, int, int]], int]:
    offset = 16
    table = []
    for label, size in specs:
        offset = align(offset)
        table.append((label, offset, size))
        offset += size
    return table, align(offset)


def expected_sizes(g: Geometry) -> dict[str, int]:
    result = {}
    for n in range(g.layers):
        result[f"target/layer-{n}.bin"] = section_offsets(section_specs(g, n))[1]
    result["target/head.bin"] = align(align(align(16) + g.hidden * 2)
                                      + q4.q4_packed_bytes(g.vocab, g.hidden))
    w = g.vocab * g.hidden // 2
    p = g.vocab * g.hidden // 32
    result["target/embedding.bin"] = align(align(align(align(16) + w) + p) + p)
    return result


def _pad(f):
    f.write(b"\0" * (-f.tell() % ALIGN))


def _layer_data(ck: Qwen36Checkpoint, n: int, label: str):
    g = ck.geometry
    p = f"language_model.model.layers.{n}."
    if label == "input-norm":
        return ck.raw(p + "input_layernorm.weight")
    if label == "post-attention-norm":
        return ck.raw(p + "post_attention_layernorm.weight")
    if label == "gdn-input":
        pieces = []
        for suffix in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            pieces.append(ck.q4_parts(p + f"linear_attn.{suffix}.weight"))
        codes = np.concatenate([part[0] for part in pieces], axis=0)
        scales = np.concatenate([part[1] for part in pieces], axis=0)
        biases = np.concatenate([part[2] for part in pieces], axis=0)
        assert codes.shape == (g.gdn_actual_width, g.hidden)
        pad = g.gdn_packed - g.gdn_actual_width
        codes = np.pad(codes, ((0, pad), (0, 0)))
        scales = np.pad(scales, ((0, pad), (0, 0)))
        biases = np.pad(biases, ((0, pad), (0, 0)))
        return q4.pack_projection(codes, scales, biases)
    if label == "gdn-convolution":
        return ck.raw(p + "linear_attn.conv1d.weight")
    if label == "gdn-decay":
        return decay_bytes(ck.f32(p + "linear_attn.A_log"))
    if label == "gdn-time-bias":
        return ck.raw(p + "linear_attn.dt_bias")
    if label == "gdn-norm":
        return ck.raw(p + "linear_attn.norm.weight")
    if label == "gdn-output":
        return ck.pack_q4(p + "linear_attn.out_proj.weight")
    if label == "attention-input":
        parts = [ck.q4_parts(p + f"self_attn.{part}_proj.weight")
                 for part in ("q", "k", "v")]
        codes, scales, biases = (np.concatenate([part[i] for part in parts], axis=0)
                                 for i in range(3))
        assert codes.shape == (g.full_packed, g.hidden)
        return q4.pack_projection(codes, scales, biases)
    if label in ("query-norm", "key-norm"):
        part = "q" if label == "query-norm" else "k"
        return ck.raw(p + f"self_attn.{part}_norm.weight")
    if label == "attention-output":
        return ck.pack_q4(p + "self_attn.o_proj.weight")
    if label == "router":
        return ck.pack_q8(p + "mlp.gate.weight")
    if label.startswith("shared-expert-") and label != "shared-expert-scalar-gate":
        part = label.removeprefix("shared-expert-")
        return ck.pack_q4(p + f"mlp.shared_expert.{part}_proj.weight")
    if label == "shared-expert-scalar-gate":
        return ck.pack_q8(p + "mlp.shared_expert_gate.weight", pad_rows=255)
    raise ValueError(label)


def write_layer(ck: Qwen36Checkpoint, n: int, path: str | Path) -> list[tuple[str, int, int]]:
    g = ck.geometry
    specs = section_specs(g, n)
    table, expected = section_offsets(specs)
    p = f"language_model.model.layers.{n}.mlp.switch_mlp."
    with open(path, "wb") as f:
        f.write(LAYER_MAGIC + struct.pack("<II", n, int(g.layer_types[n] == "full_attention")))
        for label, offset, size in table:
            _pad(f)
            assert f.tell() == offset
            start = f.tell()
            if label.startswith("experts-"):
                part = label.removeprefix("experts-")
                name = p + f"{part}_proj.weight"
                for expert in range(g.experts):
                    # One expert slab = [Q4 codes][BF16 scales][BF16 biases].
                    f.write(ck.pack_q4(name, expert=expert))
            else:
                f.write(_layer_data(ck, n, label))
            if f.tell() - start != size:
                raise ValueError(f"layer {n} {label}: wrote {f.tell()-start}, expected {size}")
        _pad(f)
        assert f.tell() == expected
    return table


def _tile_bytes(codes, scales, biases):
    tile = 256
    groups = codes.shape[1] // 64
    packed = codes.reshape(tile, groups, 64).transpose(1, 0, 2)
    packed = (packed[..., 0::2] | (packed[..., 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scales.T.copy().tobytes(), biases.T.copy().tobytes()


def write_head_embedding(ck: Qwen36Checkpoint, path: str | Path, *, head: bool) -> None:
    g = ck.geometry
    tensor = ("language_model.model.embed_tokens.weight" if not head or g.tied_head
              else "language_model.lm_head.weight")
    scales, biases = [], []
    with open(path, "wb") as f:
        if head:
            f.write(HEAD_MAGIC + struct.pack("<II", g.layers, 2))
            _pad(f)
            f.write(ck.raw("language_model.model.norm.weight"))
        else:
            f.write(EMBEDDING_MAGIC + struct.pack("<II", g.vocab, g.hidden))
        _pad(f)
        for row in range(0, g.vocab, 256):
            c, s, b = ck.q4_parts(tensor, row, row + 256)
            if head:
                raw, _, _ = _tile_bytes(c, s, b)
                f.write(raw)
                scales.append(s.T.copy())
                biases.append(b.T.copy())
            else:
                f.write((c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8).tobytes())
                scales.append(s)
                biases.append(b)
        # The head is one Q4 section. Embeddings expose three aligned components.
        if not head:
            _pad(f)
        f.write(q4.f32_to_bf16_bytes(np.concatenate(scales).reshape(-1)))
        if not head:
            _pad(f)
        f.write(q4.f32_to_bf16_bytes(np.concatenate(biases).reshape(-1)))
        _pad(f)
    key = "target/head.bin" if head else "target/embedding.bin"
    want = expected_sizes(g)[key]
    if os.path.getsize(path) != want:
        raise ValueError(f"{key}: wrote {os.path.getsize(path)}, expected {want}")


def inspect(ck: Qwen36Checkpoint) -> dict:
    ck.validate()
    sizes = expected_sizes(ck.geometry)
    return {
        "model_type": "qwen36-moe", "source_mode": ck.source_mode,
        "source": ck.root,
        "tensors": len(ck.weight_map), "layers": ck.geometry.layers,
        "source_tensors": {name: {"dtype": ck.dtype(name), "shape": ck.shape(name)}
                           for name in sorted(ck.weight_map)},
        "layer_0_sections": section_offsets(section_specs(ck.geometry, 0))[0],
        "layer_3_sections": section_offsets(section_specs(ck.geometry, 3))[0],
        "target_files": sizes, "target_bytes": sum(sizes.values()),
        "target_gib": round(sum(sizes.values()) / 2**30, 3),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_process_ram_bytes() -> int | None:
    """Peak RSS reported by the OS, when available."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


def assemble(ck: Qwen36Checkpoint, out: Path, reference: Path) -> None:
    """Reuse matching non-target assets; target files are always newly encoded."""
    with (reference / "manifest.json").open(encoding="utf-8") as f:
        ref = json.load(f)
    if ref.get("schema_version") != 4 or ref.get("format", {}).get("name") != "splash-packed-q4-moe":
        raise ValueError("reference must be a Splash Qwen3.6 schema-4 package")
    with (reference / "tokenizer" / "config.json").open(encoding="utf-8") as f:
        reference_config = json.load(f)
    for key in ("model_type", "architectures", "image_token_id"):
        if ck.config.get(key) != reference_config.get(key):
            raise ValueError(f"reference vision/tokenizer config differs at {key}")
    expected = expected_sizes(ck.geometry)
    recorded = {a["path"]: a["size"] for a in ref["artifacts"]}
    for name, size in expected.items():
        if recorded.get(name) != size:
            raise ValueError(f"reference size mismatch for {name}")
    for a in ref["artifacts"]:
        rel = a["path"]
        if rel.startswith(("draft/", "vision/")) or rel == "layout.json":
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference / rel, dest)
            if sha256(dest) != a["sha256"]:
                raise ValueError(f"reference asset hash mismatch: {rel}")
    tok = out / "tokenizer"
    tok.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(Path(ck.root) / name, tok / name)
    with (tok / "tokenizer.json").open(encoding="utf-8") as f:
        vocabulary = json.load(f)["model"]["vocab"]
    with (tok / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, separators=(",", ":"))
    # Keep execution/draft declarations from the compatible reference; derive
    # the target declaration and all artifact hashes from actual output.
    manifest = {key: ref[key] for key in ("execution_geometry", "format", "draft")}
    source_id = RAVENX_MLX4_ID if ck.source_mode == "mlx4" else RAVENX_ID
    manifest.update(schema_version=4, model=Path(ck.root).name,
                    target=dict(architecture="qwen3_5_moe", layers=ck.geometry.layers,
                                hidden_size=ck.geometry.hidden,
                                vocabulary_size=ck.geometry.vocab,
                                gdn_actual_width=ck.geometry.gdn_actual_width,
                                gdn_packed_width=ck.geometry.gdn_packed,
                                attention_packed_width=ck.geometry.full_packed,
                                experts=ck.geometry.experts,
                                experts_per_token=ck.geometry.active_experts,
                                moe_intermediate_size=ck.geometry.expert_hidden,
                                shared_expert_intermediate_size=ck.geometry.shared_hidden,
                                layer_types=["attention" if x == "full_attention" else "gdn"
                                             for x in ck.geometry.layer_types]),
                    upstream={"target": {"repo_id": source_id},
                              "draft": ref.get("upstream", {}).get("draft", {}),
                              "vision": ref.get("upstream", {}).get("vision", {}),
                              "tokenizer": {"repo_id": source_id}},
                    converter={"source_path": "converter/qwen36_moe.py",
                               "reference_package": REFERENCE_ID})
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest["artifacts"] = [{"path": p.relative_to(out).as_posix(),
                              "size": p.stat().st_size, "sha256": sha256(p)} for p in paths]
    # A deterministic digest of this package's artifact records. Splash's
    # installer validates each record independently; this is provenance data.
    records = "".join(f"{a['path']}\t{a['size']}\t{a['sha256']}\n"
                      for a in manifest["artifacts"])
    manifest["artifact_set_sha256"] = hashlib.sha256(records.encode()).hexdigest()
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def verify_structure(out: Path, geometry: Geometry, *, full: bool) -> dict:
    sizes = expected_sizes(geometry)
    checked = 0
    for rel, want in sizes.items():
        path = out / rel
        if not path.is_file() or path.stat().st_size != want:
            raise ValueError(f"{rel}: expected {want} bytes")
        with path.open("rb") as f:
            magic, first, second = struct.unpack("<8sII", f.read(16))
        if rel.endswith("embedding.bin"):
            expected = (EMBEDDING_MAGIC, geometry.vocab, geometry.hidden)
        elif rel.endswith("head.bin"):
            expected = (HEAD_MAGIC, geometry.layers, 2)
        else:
            n = int(Path(rel).stem.split("-")[1])
            expected = (LAYER_MAGIC, n, int(geometry.layer_types[n] == "full_attention"))
        if (magic, first, second) != expected:
            raise ValueError(f"{rel}: invalid header")
        checked += 1
    layout_path = out / "layout.json"
    if layout_path.is_file():
        with layout_path.open(encoding="utf-8") as f:
            layout = json.load(f)
        by_path = {entry["path"]: entry for entry in layout["target"]}
        for n in range(geometry.layers):
            rel = f"target/layer-{n}.bin"
            reference_sections = by_path[rel]["sections"]
            actual_sections = section_offsets(section_specs(geometry, n))[0]
            if [(s["offset"], s["bytes"]) for s in reference_sections] != [
                    (offset, size) for _, offset, size in actual_sections]:
                raise ValueError(f"{rel}: section offsets differ from reference layout")
            if by_path[rel]["bytes"] != sizes[rel]:
                raise ValueError(f"{rel}: reference layout file size mismatch")
    if full:
        with (out / "manifest.json").open(encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != 4 or manifest.get("format", {}).get("name") != "splash-packed-q4-moe":
            raise ValueError("invalid schema-4 manifest")
        artifacts = {a["path"]: a for a in manifest["artifacts"]}
        required = set(sizes) | {"draft/model.bin", "vision/model.bin", "layout.json"}
        required |= {f"draft/layer-{n}.bin" for n in range(6)}
        required |= {f"tokenizer/{name}" for name in
                     ("config.json", "tokenizer.json", "tokenizer_config.json",
                      "chat_template.jinja", "vocab.json")}
        if not required <= artifacts.keys():
            raise ValueError(f"missing package artifacts: {sorted(required-artifacts.keys())}")
        for rel, a in artifacts.items():
            path = out / rel
            if not path.is_file() or path.stat().st_size != a["size"] or sha256(path) != a["sha256"]:
                raise ValueError(f"artifact failed size/hash check: {rel}")
    return {"target_files_checked": checked, "target_bytes": sum(sizes.values()),
            "package_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}


def completed_target_file(path: Path, geometry: Geometry) -> bool:
    rel = f"target/{path.name}"
    if not path.is_file() or path.stat().st_size != expected_sizes(geometry).get(rel):
        return False
    with path.open("rb") as f:
        header = struct.unpack("<8sII", f.read(16))
    if path.name == "head.bin":
        return header == (HEAD_MAGIC, geometry.layers, 2)
    if path.name == "embedding.bin":
        return header == (EMBEDDING_MAGIC, geometry.vocab, geometry.hidden)
    n = int(path.stem.split("-")[1])
    return header == (LAYER_MAGIC, n, int(geometry.layer_types[n] == "full_attention"))


def _q4_row(path: Path, start: int, rows: int, cols: int, row: int) -> np.ndarray:
    """Independent offset decoder; reads one row without mapping a whole slab."""
    groups = cols // 64
    code_base = rows * cols // 2
    param_bytes = rows * cols // 32
    codes = np.empty(cols, np.uint8)
    scale = np.empty(groups, np.float32)
    bias = np.empty(groups, np.float32)
    with path.open("rb") as f:
        for group in range(groups):
            index = ((row // 256) * groups + group) * 256 + row % 256
            f.seek(start + index * 32)
            packed = np.frombuffer(f.read(32), np.uint8)
            codes[group * 64:group * 64 + 64:2] = packed & 15
            codes[group * 64 + 1:group * 64 + 64:2] = packed >> 4
            f.seek(start + code_base + index * 2)
            scale[group] = q4.bf16_bytes_to_f32(f.read(2))[0]
            f.seek(start + code_base + param_bytes + index * 2)
            bias[group] = q4.bf16_bytes_to_f32(f.read(2))[0]
    return codes.astype(np.float32) * np.repeat(scale, 64) + np.repeat(bias, 64)


def _q8_row(path: Path, start: int, rows: int, cols: int, row: int) -> np.ndarray:
    groups = cols // 64
    code_base = rows * cols
    param_bytes = rows * cols // 32
    values = np.empty(cols, np.uint8)
    scale = np.empty(groups, np.float32)
    bias = np.empty(groups, np.float32)
    with path.open("rb") as f:
        for group in range(groups):
            index = ((row // 256) * groups + group) * 256 + row % 256
            f.seek(start + index * 64)
            values[group * 64:group * 64 + 64] = np.frombuffer(f.read(64), np.uint8)
            f.seek(start + code_base + index * 2)
            scale[group] = q4.bf16_bytes_to_f32(f.read(2))[0]
            f.seek(start + code_base + param_bytes + index * 2)
            bias[group] = q4.bf16_bytes_to_f32(f.read(2))[0]
    return values.astype(np.float32) * np.repeat(scale, 64) + np.repeat(bias, 64)


def _embedding_row(path: Path, g: Geometry, row: int) -> np.ndarray:
    code_start = align(16)
    scale_start = align(code_start + g.vocab * g.hidden // 2)
    bias_start = align(scale_start + g.vocab * g.hidden // 32)
    with path.open("rb") as f:
        f.seek(code_start + row * g.hidden // 2)
        packed = np.frombuffer(f.read(g.hidden // 2), np.uint8)
        f.seek(scale_start + row * g.hidden // 64 * 2)
        scale = q4.bf16_bytes_to_f32(f.read(g.hidden // 64 * 2))
        f.seek(bias_start + row * g.hidden // 64 * 2)
        bias = q4.bf16_bytes_to_f32(f.read(g.hidden // 64 * 2))
    codes = np.empty(g.hidden, np.uint8)
    codes[0::2], codes[1::2] = packed & 15, packed >> 4
    return codes.astype(np.float32) * np.repeat(scale, 64) + np.repeat(bias, 64)


def validate_tensors(ck: Qwen36Checkpoint, out: Path) -> dict:
    """Numerical Q4/Q8 samples plus exact norm/decay checks across both layer types."""
    g = ck.geometry
    results = {}

    def measure(label, approx, source, *, q8=False):
        if not np.isfinite(approx).all():
            raise ValueError(f"{label}: non-finite dequantized values")
        source = source.astype(np.float32)
        delta = approx.astype(np.float64) - source.astype(np.float64)
        rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
        rel = float(np.sqrt(np.mean(delta ** 2)) / max(rms, 1e-12))
        cosine = float(np.dot(approx, source) /
                       max(np.linalg.norm(approx) * np.linalg.norm(source), 1e-12))
        results[label] = {"relative_rmse": rel, "cosine": cosine,
                          "max_absolute_error": float(np.max(np.abs(delta)))}
        if rel > (0.03 if q8 else 0.20) or cosine < (0.995 if q8 else 0.97):
            raise ValueError(f"{label}: quantization error outside expected range: {results[label]}")

    for n in (0, 3):
        path = out / "target" / f"layer-{n}.bin"
        offsets = {label: offset for label, offset, _ in section_offsets(section_specs(g, n))[0]}
        prefix = f"language_model.model.layers.{n}."
        with path.open("rb") as f:
            f.seek(offsets["input-norm"])
            if f.read(g.hidden * 2) != ck.raw(prefix + "input_layernorm.weight"):
                raise ValueError(f"layer {n}: input norm mismatch")
            f.seek(offsets["post-attention-norm"])
            if f.read(g.hidden * 2) != ck.raw(prefix + "post_attention_layernorm.weight"):
                raise ValueError(f"layer {n}: post-attention norm mismatch")
        if n == 0:
            for row, suffix, source_row in ((0, "in_proj_qkv", 0),
                                            (8192, "in_proj_z", 0),
                                            (12288, "in_proj_b", 0),
                                            (12320, "in_proj_a", 0)):
                measure(f"gdn-{suffix}", _q4_row(path, offsets["gdn-input"], g.gdn_packed, g.hidden, row),
                        ck.f32_rows(prefix + f"linear_attn.{suffix}.weight", source_row, source_row + 1)[0])
            measure("gdn-output", _q4_row(path, offsets["gdn-output"], g.hidden, g.attention_width, 0),
                    ck.f32_rows(prefix + "linear_attn.out_proj.weight", 0, 1)[0])
            with path.open("rb") as f:
                for label, source in (("gdn-convolution", "linear_attn.conv1d.weight"),
                                      ("gdn-time-bias", "linear_attn.dt_bias"),
                                      ("gdn-norm", "linear_attn.norm.weight")):
                    raw = ck.raw(prefix + source)
                    f.seek(offsets[label])
                    if f.read(len(raw)) != raw:
                        raise ValueError(f"{label}: source bytes mismatch")
                decay = decay_bytes(ck.f32(prefix + "linear_attn.A_log"))
                f.seek(offsets["gdn-decay"])
                if f.read(len(decay)) != decay:
                    raise ValueError("gdn-decay mismatch")
        else:
            for row, suffix in ((0, "q"), (8192, "k"), (8704, "v")):
                measure(f"attention-{suffix}", _q4_row(path, offsets["attention-input"], g.full_packed, g.hidden, row),
                        ck.f32_rows(prefix + f"self_attn.{suffix}_proj.weight", 0, 1)[0])
            measure("attention-output", _q4_row(path, offsets["attention-output"], g.hidden, g.attention_width, 0),
                    ck.f32_rows(prefix + "self_attn.o_proj.weight", 0, 1)[0])
            with path.open("rb") as f:
                for label, suffix in (("query-norm", "q"), ("key-norm", "k")):
                    raw = ck.raw(prefix + f"self_attn.{suffix}_norm.weight")
                    f.seek(offsets[label])
                    if f.read(len(raw)) != raw:
                        raise ValueError(f"{label}: norm mismatch")
        measure(f"router-{n}", _q8_row(path, offsets["router"], 256, g.hidden, 0),
                ck.f32_rows(prefix + "mlp.gate.weight", 0, 1)[0], q8=True)
        stride_gate = q4.q4_packed_bytes(g.expert_hidden, g.hidden)
        stride_down = q4.q4_packed_bytes(g.hidden, g.expert_hidden)
        for expert in (0, 128, 255):
            for part, rows, cols, stride in (("gate", g.expert_hidden, g.hidden, stride_gate),
                                             ("up", g.expert_hidden, g.hidden, stride_gate),
                                             ("down", g.hidden, g.expert_hidden, stride_down)):
                measure(f"expert-{n}-{expert}-{part}",
                        _q4_row(path, offsets[f"experts-{part}"] + expert * stride, rows, cols, 0),
                        ck.f32_expert(prefix + f"mlp.switch_mlp.{part}_proj.weight", expert)[0])
        for part, rows, cols in (("gate", g.shared_hidden, g.hidden),
                                 ("up", g.shared_hidden, g.hidden),
                                 ("down", g.hidden, g.shared_hidden)):
            measure(f"shared-{n}-{part}",
                    _q4_row(path, offsets[f"shared-expert-{part}"], rows, cols, 0),
                    ck.f32_rows(prefix + f"mlp.shared_expert.{part}_proj.weight", 0, 1)[0])
        measure(f"shared-gate-{n}",
                _q8_row(path, offsets["shared-expert-scalar-gate"], 256, g.hidden, 0),
                ck.f32(prefix + "mlp.shared_expert_gate.weight")[0], q8=True)

    head_name = ("language_model.model.embed_tokens.weight" if g.tied_head
                 else "language_model.lm_head.weight")
    head = out / "target" / "head.bin"
    embedding = out / "target" / "embedding.bin"
    with head.open("rb") as f:
        f.seek(align(16))
        if f.read(g.hidden * 2) != ck.raw("language_model.model.norm.weight"):
            raise ValueError("final norm mismatch")
    head_start = align(align(16) + g.hidden * 2)
    for row in (0, 12345, g.vocab - 1):
        measure(f"head-{row}", _q4_row(head, head_start, g.vocab, g.hidden, row),
                ck.f32_rows(head_name, row, row + 1)[0])
        measure(f"embedding-{row}", _embedding_row(embedding, g, row),
                ck.f32_rows("language_model.model.embed_tokens.weight", row, row + 1)[0])
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m converter.qwen36_moe")
    parser.add_argument("command", choices=("inspect", "build", "assemble", "verify", "validate"))
    parser.add_argument("--source", required=True, help="local BF16 or MLX 4-bit sharded safetensors directory")
    parser.add_argument("--out", help="output Splash package directory")
    parser.add_argument("--reference", help="local official Qwen3.6 Splash package; required for full build")
    parser.add_argument("--target-only", action="store_true", help="build target files without draft/vision")
    parser.add_argument("--resume", action="store_true", help="skip target files with expected size and header")
    parser.add_argument("--model-type", choices=("auto", "qwen36-moe"), default="auto")
    args = parser.parse_args(argv)
    ck = Qwen36Checkpoint(args.source)
    if args.command == "inspect":
        print(json.dumps(inspect(ck), indent=2))
        return 0
    if not args.out:
        parser.error("--out is required for build/verify")
    out = Path(args.out)
    if args.command in ("verify", "validate"):
        report = verify_structure(out, ck.geometry, full=not args.target_only)
        if args.command == "validate":
            report["tensor_validation"] = validate_tensors(ck, out)
        print(json.dumps(report, indent=2))
        return 0
    ck.validate()
    if args.command == "assemble":
        if not args.reference:
            parser.error("assemble requires --reference")
        verify_structure(out, ck.geometry, full=False)
        assemble(ck, out, Path(args.reference))
        print(json.dumps(verify_structure(out, ck.geometry, full=True), indent=2))
        return 0
    if not args.target_only and not args.reference:
        parser.error("full package build requires --reference; use --target-only for target weights")
    started = time.monotonic()
    target = out / "target"
    target.mkdir(parents=True, exist_ok=True)
    for n in range(ck.geometry.layers):
        name = target / f"layer-{n}.bin"
        if args.resume and completed_target_file(name, ck.geometry):
            print(f"layer {n}/39 already complete", flush=True)
            continue
        print(f"layer {n}/39 -> {name}", flush=True)
        write_layer(ck, n, name)
    if not args.resume or not completed_target_file(target / "head.bin", ck.geometry):
        write_head_embedding(ck, target / "head.bin", head=True)
    if not args.resume or not completed_target_file(target / "embedding.bin", ck.geometry):
        write_head_embedding(ck, target / "embedding.bin", head=False)
    if not args.target_only:
        assemble(ck, out, Path(args.reference))
    report = verify_structure(out, ck.geometry, full=not args.target_only)
    report["conversion_seconds"] = round(time.monotonic() - started, 2)
    report["peak_conversion_ram_bytes"] = peak_process_ram_bytes()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
