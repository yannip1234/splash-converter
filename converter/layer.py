"""Assemble Splash `target/layer-N.bin` files from a BF16 checkpoint.

Layout: docs/SPLASH_FORMAT_SPEC.md §2 (header/section framing), §4 (geometry),
§5.1 (GDN sections) and §5.2 (full-attention sections); source mapping in
docs/TENSOR_MAPPING.md §1-2, with three corrections established by decoding the
reference package (see PROGRESS iteration 6):

- `input-norm` and `post-attention-norm` store **bf16(γ + 1)**, not γ. Verified
  bit-exact against `refs/qwen38-splash/target/layer-0.bin` for both sections.
- `input-norm`, `post-attention-norm`, `query-norm`, `key-norm` and the head's
  `final-norm` all store **bf16(γ + 1)**. `gdn-norm` (`linear_attn.norm.weight`)
  is the lone exception and stores **γ unchanged** — verified byte-exact both ways
  against the reference package.
- `gdn-decay` stores **float32 `-exp(A_log)`**, not a dtype upcast of `A_log`.

Every section is raw bytes; the loader aligns each to 16384 and requires the
aligned consumed offset to equal the file size (`finish()`, WeightStore.cpp:214-221).
"""

from __future__ import annotations

import struct

import numpy as np

from . import encoder, q4
from .checkpoint import Checkpoint

ALIGN = 16384
TARGET_LAYER_MAGIC = b"MDFL0006"
HEAD_MAGIC = b"MDFL0002"
EMBEDDING_MAGIC = b"MDFE0001"
GDN_ROW_WIDTH = 16640          # spec §4.1: qkv 10240 | z 6144 | b 48 | a 48 | pad 160
ATTENTION_ROW_WIDTH = 14336    # spec §4.2
HIDDEN = 5120

GDN_LAYER_BYTES = 216203264
ATTENTION_LAYER_BYTES = 209469440
HEAD_BYTES = 715194368
EMBEDDING_BYTES = 715177984
VOCAB = 248320
TILE = q4.STORAGE_N  # 256


def is_full_attention(layer: int) -> bool:
    """spec §4: full attention iff (layer+1) % 4 == 0 (16 of 64 layers)."""
    return (layer + 1) % 4 == 0


# --- section encoders -------------------------------------------------------

def norm_plus_one_bytes(gamma: np.ndarray) -> bytes:
    """RMS γ stored with the unit offset: bf16(γ + 1)."""
    return q4.f32_to_bf16_u16(gamma.astype(np.float32) + np.float32(1.0)).astype("<u2").tobytes()


def decay_bytes(a_log: np.ndarray) -> bytes:
    """GDN decay: float32 -exp(A_log)."""
    return (-np.exp(a_log.astype(np.float32))).astype("<f4").tobytes()


def gdn_input_matrix(ck: Checkpoint, prefix: str) -> np.ndarray:
    """Fuse the four GDN input projections into the packed row (spec §4.1)."""
    qkv = ck.f32(prefix + "linear_attn.in_proj_qkv.weight")   # [10240, 5120]
    z = ck.f32(prefix + "linear_attn.in_proj_z.weight")       # [ 6144, 5120]
    b = ck.f32(prefix + "linear_attn.in_proj_b.weight")       # [   48, 5120]
    a = ck.f32(prefix + "linear_attn.in_proj_a.weight")       # [   48, 5120]
    w = np.zeros((GDN_ROW_WIDTH, HIDDEN), dtype=np.float32)   # tail 160 rows stay zero
    w[0:10240] = qkv
    w[10240:16384] = z
    w[16384:16432] = b
    w[16432:16480] = a
    return w


def attention_input_matrix(ck: Checkpoint, prefix: str) -> np.ndarray:
    """Fuse q/k/v into the packed attention row (spec §4.2); no reorder."""
    q = ck.f32(prefix + "self_attn.q_proj.weight")            # [12288, 5120]
    k = ck.f32(prefix + "self_attn.k_proj.weight")            # [ 1024, 5120]
    v = ck.f32(prefix + "self_attn.v_proj.weight")            # [ 1024, 5120]
    return np.concatenate([q, k, v], axis=0)


# --- section lists ----------------------------------------------------------

def gdn_sections(ck: Checkpoint, prefix: str):
    """(label, bytes) for a GDN layer, in file order (spec §5.1)."""
    yield "input-norm", norm_plus_one_bytes(ck.f32(prefix + "input_layernorm.weight"))
    yield "gdn-input", encoder.quantize_and_pack(gdn_input_matrix(ck, prefix))
    yield "gdn-convolution", ck.raw(prefix + "linear_attn.conv1d.weight")
    yield "gdn-decay", decay_bytes(ck.f32(prefix + "linear_attn.A_log"))
    yield "gdn-time-bias", ck.raw(prefix + "linear_attn.dt_bias")
    yield "gdn-norm", ck.raw(prefix + "linear_attn.norm.weight")
    yield "gdn-output", encoder.quantize_and_pack(ck.f32(prefix + "linear_attn.out_proj.weight"))
    yield "post-attention-norm", norm_plus_one_bytes(ck.f32(prefix + "post_attention_layernorm.weight"))
    yield from _mlp_sections(ck, prefix)


def attention_sections(ck: Checkpoint, prefix: str):
    """(label, bytes) for a full-attention layer, in file order (spec §5.2)."""
    yield "input-norm", norm_plus_one_bytes(ck.f32(prefix + "input_layernorm.weight"))
    yield "attention-input", encoder.quantize_and_pack(attention_input_matrix(ck, prefix))
    # query-norm and key-norm carry the SAME unit offset as the block norms.
    # Verified against refs/qwen38-splash/target/layer-3.bin: max|ref - (gamma+1)|
    # = 0.0039 against max|ref - gamma| = 1.0039. TENSOR_MAPPING A3/A4 called these
    # identity copies, which shipped every attention layer with q/k norms short by
    # exactly 1.0 and cost 27.5 points of task accuracy (PROGRESS iteration 13).
    yield "query-norm", norm_plus_one_bytes(ck.f32(prefix + "self_attn.q_norm.weight"))
    yield "key-norm", norm_plus_one_bytes(ck.f32(prefix + "self_attn.k_norm.weight"))
    yield "attention-output", encoder.quantize_and_pack(ck.f32(prefix + "self_attn.o_proj.weight"))
    yield "post-attention-norm", norm_plus_one_bytes(ck.f32(prefix + "post_attention_layernorm.weight"))
    yield from _mlp_sections(ck, prefix)


def _mlp_sections(ck: Checkpoint, prefix: str):
    yield "mlp-gate", encoder.quantize_and_pack(ck.f32(prefix + "mlp.gate_proj.weight"))
    yield "mlp-up", encoder.quantize_and_pack(ck.f32(prefix + "mlp.up_proj.weight"))
    yield "mlp-down", encoder.quantize_and_pack(ck.f32(prefix + "mlp.down_proj.weight"))


# --- file assembly ----------------------------------------------------------

def write_layer(ck: Checkpoint, layer: int, path: str, prefix_fmt: str =
                "model.language_model.layers.{layer}.") -> list[tuple[str, int, int]]:
    """Write one target/layer-N.bin. Returns [(label, offset, bytes)]."""
    prefix = prefix_fmt.format(layer=layer)
    attention = is_full_attention(layer)
    sections = attention_sections(ck, prefix) if attention else gdn_sections(ck, prefix)

    table: list[tuple[str, int, int]] = []
    with open(path, "wb") as f:
        f.write(TARGET_LAYER_MAGIC)
        f.write(struct.pack("<I", layer))
        f.write(struct.pack("<I", 1 if attention else 0))
        for label, data in sections:
            pad = (-f.tell()) % ALIGN
            if pad:
                f.write(b"\x00" * pad)
            table.append((label, f.tell(), len(data)))
            f.write(data)
        tail = (-f.tell()) % ALIGN
        if tail:
            f.write(b"\x00" * tail)
        size = f.tell()

    expected = ATTENTION_LAYER_BYTES if attention else GDN_LAYER_BYTES
    if size != expected:
        raise ValueError(f"layer {layer}: wrote {size} bytes, expected {expected}")
    return table


# --- head and embedding -----------------------------------------------------
#
# Both are 248320x5120 (5.1 GB as float32), so they are packed in blocks rather
# than materialized. `head` uses the ordinary tile-major projection layout
# (spec §5.3); `embedding` uses the row-major exception (spec §3.5, §5.4): three
# separate 16384-aligned sections, plain [token][dim] order.
#
# Both carry the unit-offset norm: head's `final-norm` stores bf16(gamma + 1),
# verified against the reference package (max|pkg - (gamma+1)| = 0.0078 vs
# 1.0078 for plain gamma).

def _tile_major_block(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray):
    """Code bytes and parameter order for one 256-row tile (see q4.pack_projection)."""
    rows, in_ = codes.shape
    if rows != TILE:
        raise ValueError(f"tile must be {TILE} rows, got {rows}")
    groups = in_ // q4.GROUP
    c = codes.reshape(TILE, groups, q4.GROUP).transpose(1, 0, 2)
    code_bytes = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8).tobytes()
    return code_bytes, scales.T.reshape(-1), biases.T.reshape(-1)


def write_head(ck: Checkpoint, path: str, logits_tensor: str = "lm_head.weight",
               norm_tensor: str = "model.language_model.norm.weight") -> list:
    """Write target/head.bin (spec §5.3): final-norm + tile-major Q4 logits."""
    out, in_ = ck.shape(logits_tensor)
    if (out, in_) != (VOCAB, HIDDEN):
        raise ValueError(f"{logits_tensor}: expected {(VOCAB, HIDDEN)}, got {(out, in_)}")
    table = []
    scales_all, biases_all = [], []
    with open(path, "wb") as f:
        f.write(HEAD_MAGIC + struct.pack("<I", 64) + struct.pack("<I", 2))
        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        norm = norm_plus_one_bytes(ck.f32(norm_tensor))
        table.append(("final-norm", f.tell(), len(norm)))
        f.write(norm)

        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        start = f.tell()
        for base in range(0, out, TILE):
            w = ck.f32_rows(logits_tensor, base, base + TILE)
            codes, scales, biases = encoder.quantize_projection(w)
            code_bytes, s_flat, b_flat = _tile_major_block(codes, scales, biases)
            f.write(code_bytes)
            scales_all.append(s_flat)
            biases_all.append(b_flat)
        f.write(q4.f32_to_bf16_bytes(np.concatenate(scales_all)))
        f.write(q4.f32_to_bf16_bytes(np.concatenate(biases_all)))
        table.append(("logits", start, f.tell() - start))
        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        size = f.tell()
    if size != HEAD_BYTES:
        raise ValueError(f"head: wrote {size} bytes, expected {HEAD_BYTES}")
    return table


def write_embedding(ck: Checkpoint, path: str,
                    tensor: str = "model.language_model.embed_tokens.weight",
                    block: int = 8192) -> list:
    """Write target/embedding.bin (spec §5.4): row-major codes, scales, biases."""
    out, in_ = ck.shape(tensor)
    if (out, in_) != (VOCAB, HIDDEN):
        raise ValueError(f"{tensor}: expected {(VOCAB, HIDDEN)}, got {(out, in_)}")
    table = []
    scales_all, biases_all = [], []
    with open(path, "wb") as f:
        f.write(EMBEDDING_MAGIC + struct.pack("<I", VOCAB) + struct.pack("<I", HIDDEN))
        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        start = f.tell()
        for base in range(0, out, block):
            w = ck.f32_rows(tensor, base, min(base + block, out))
            codes, scales, biases = encoder.quantize_projection(w)
            packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
            f.write(packed.tobytes())
            scales_all.append(scales.reshape(-1))
            biases_all.append(biases.reshape(-1))
        table.append(("embedding-weights", start, f.tell() - start))

        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        start = f.tell()
        f.write(q4.f32_to_bf16_bytes(np.concatenate(scales_all)))
        table.append(("embedding-scales", start, f.tell() - start))

        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        start = f.tell()
        f.write(q4.f32_to_bf16_bytes(np.concatenate(biases_all)))
        table.append(("embedding-biases", start, f.tell() - start))

        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        size = f.tell()
    if size != EMBEDDING_BYTES:
        raise ValueError(f"embedding: wrote {size} bytes, expected {EMBEDDING_BYTES}")
    return table
