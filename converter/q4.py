"""Q4 packing primitives for the Splash `splash-packed-q4` format.

Implements the container-level pack/unpack of the Q4 tensor family:

- Projection packing (all target/draft projections): tile-major codes with
  per-64-group bf16 scale/bias, one section laid out [codes | scales | biases].
- Embedding packing (exception): plain row-major [token][dim], three separate
  16384-aligned sections.

Sources of truth (see docs/SPLASH_FORMAT_SPEC.md §3):
- `readQ4Projection` / `q4PackedBytes`            refs/splash-src/runtime/model/WeightStore.cpp:57-60, 231-247
- CPU reference `packSlab` / `parameter(n, g)`    refs/splash-src/dev/tests/engine/moe_metal_test.mm:98-142
- Kernel code/scale addressing                    refs/splash-src/runtime/metal/kernels/shared/q4_mpp_tiles.h:148-174
- Embedding row-major layout                      refs/splash-src/runtime/metal/kernels/shared/embedding.metal:20-24

Dequantization (kernel epilogue, q4_mpp_tiles.h:172-174):
    w[n, k] = code(n, k) * scale(n, k // 64) + bias(n, k // 64)
Unsigned 4-bit codes, no +8 centering.

NOTE: this module does NOT implement the encoder rule (how codes/scale/bias are
derived from raw BF16 weights) — that is a separate task (PLAN T1.4) and must be
derived from evidence before use.
"""

from __future__ import annotations

import numpy as np

GROUP = 64
STORAGE_N = 256
ALIGN = 16384


def q4_packed_bytes(out: int, in_: int) -> int:
    """Total section bytes for a Q4 projection (WeightStore.cpp:57-60)."""
    _validate_q4_layout(out, in_)
    return out * in_ * 9 // 16


def _validate_q4_layout(out: int, in_: int) -> None:
    if in_ % GROUP != 0:
        raise ValueError(f"input dimension {in_} must be a multiple of {GROUP}")
    if out % STORAGE_N != 0:
        raise ValueError(f"output dimension {out} must be a multiple of {STORAGE_N}")


# ---------------------------------------------------------------------------
# bf16 (bfloat16) <-> float32, round-to-nearest-even, saturating to inf,
# matching clang __bf16 semantics.
# ---------------------------------------------------------------------------

def f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 array of bf16 bits (RNE, saturate to inf)."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    sign = bits & 0x80000000
    mag = bits & 0x7FFFFFFF
    finite = mag < 0x7F800000
    rounded = mag + 0x7FFF + ((mag >> 16) & 1)  # RNE; may overflow to inf
    out = np.where(finite, sign | ((rounded >> 16) << 16), sign | 0x7F800000)
    out = np.where(finite & (rounded >= 0x80000000), sign | 0x7F800000, out)
    out = np.where(np.isnan(x), sign | 0x7FC00000, out)
    return (out >> 16).astype(np.uint16)


def bf16_u16_to_f32(u: np.ndarray) -> np.ndarray:
    """uint16 array of bf16 bits -> float32 array (exact)."""
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bytes(x: np.ndarray) -> bytes:
    return f32_to_bf16_u16(x).astype("<u2").tobytes()


def bf16_bytes_to_f32(data: bytes) -> np.ndarray:
    return bf16_u16_to_f32(np.frombuffer(data, dtype="<u2"))


# ---------------------------------------------------------------------------
# Projection packing (tile-major).
#
# Section layout: [codes: out*in/2 B][scales: out*in/32 B][biases: out*in/32 B]
#
# Codes: tile = 256 consecutive output rows. Byte offset of the 64-nibble row
# block for (n, g) is ((n//256)*(in/64) + g)*256*32 + (n%256)*32, i.e. the
# tile-major order [tile][group][row_in_tile][32 bytes], low nibble first.
# Scales/biases: element parameter(n, g) = ((n//256)*(in/64) + g)*256 + n%256.
# ---------------------------------------------------------------------------

def pack_projection(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> bytes:
    """Pack a Q4 projection into its section bytes.

    codes:   uint8 [out, in], values 0..15 (logical row-major)
    scales:  float32 [out, in//64] (logical, group g = k//64)
    biases:  float32 [out, in//64]
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    biases = np.ascontiguousarray(biases, dtype=np.float32)
    out, in_ = codes.shape
    _validate_q4_layout(out, in_)
    if codes.max(initial=0) > 15:
        raise ValueError("codes must be 4-bit values 0..15")
    if scales.shape != (out, in_ // GROUP) or biases.shape != (out, in_ // GROUP):
        raise ValueError("scales/biases must have shape (out, in//64)")

    t, g = out // STORAGE_N, in_ // GROUP
    # (n, k) -> (tile, group, row_in_tile, col_in_group)
    c = codes.reshape(t, STORAGE_N, g, GROUP).transpose(0, 2, 1, 3)
    byte_pairs = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)
    codes_bytes = byte_pairs.reshape(-1).tobytes()
    s = scales.reshape(t, STORAGE_N, g).transpose(0, 2, 1).reshape(-1)
    b = biases.reshape(t, STORAGE_N, g).transpose(0, 2, 1).reshape(-1)
    return codes_bytes + f32_to_bf16_bytes(s) + f32_to_bf16_bytes(b)


def unpack_projection(data: bytes, out: int, in_: int):
    """Inverse of pack_projection. Returns (codes uint8 [out,in], scales f32, biases f32)."""
    _validate_q4_layout(out, in_)
    cb = out * in_ // 2
    pb = out * in_ // 32
    if len(data) != cb + 2 * pb:
        raise ValueError(f"section size {len(data)} != {cb + 2 * pb} for {out}x{in_}")
    t, g = out // STORAGE_N, in_ // GROUP

    c = np.frombuffer(data[:cb], dtype=np.uint8).reshape(t, g, STORAGE_N, 32)
    nib = np.empty((t, g, STORAGE_N, GROUP), dtype=np.uint8)
    nib[..., 0::2] = c & 0x0F
    nib[..., 1::2] = (c >> 4) & 0x0F
    codes = nib.transpose(0, 2, 1, 3).reshape(out, in_)

    s = bf16_bytes_to_f32(data[cb:cb + pb]).reshape(t, g, STORAGE_N).transpose(0, 2, 1)
    b = bf16_bytes_to_f32(data[cb + pb:]).reshape(t, g, STORAGE_N).transpose(0, 2, 1)
    return codes, s.reshape(out, g), b.reshape(out, g)


def dequantize_projection(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """w[n,k] = codes[n,k]*scales[n,k//64] + biases[n,k//64] (q4_mpp_tiles.h:172-174)."""
    out, in_ = codes.shape
    return (
        codes.astype(np.float32)
        * np.repeat(scales, GROUP, axis=1)
        + np.repeat(biases, GROUP, axis=1)
    )


# ---------------------------------------------------------------------------
# Embedding packing (row-major exception, spec §3.5): three separate sections,
# plain [token][dim] order, low nibble first.
# ---------------------------------------------------------------------------

def pack_embedding_rows(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray):
    """Pack an embedding table. Returns (weights_bytes, scales_bytes, biases_bytes),
    each a separate 16384-aligned section:
      weights: [out][in/2] bytes, row-major, low nibble first
      scales:  [out][in/64] bf16, row-major
      biases:  [out][in/64] bf16, row-major
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    biases = np.ascontiguousarray(biases, dtype=np.float32)
    out, in_ = codes.shape
    if in_ % GROUP != 0:
        raise ValueError(f"input dimension {in_} must be a multiple of {GROUP}")
    if codes.max(initial=0) > 15:
        raise ValueError("codes must be 4-bit values 0..15")
    if scales.shape != (out, in_ // GROUP) or biases.shape != (out, in_ // GROUP):
        raise ValueError("scales/biases must have shape (out, in//64)")
    w = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return (
        w.reshape(-1).tobytes(),
        f32_to_bf16_bytes(scales.reshape(-1)),
        f32_to_bf16_bytes(biases.reshape(-1)),
    )


def unpack_embedding_rows(weights: bytes, scales: bytes, biases: bytes, out: int, in_: int):
    """Inverse of pack_embedding_rows. Returns (codes uint8 [out,in], scales f32, biases f32)."""
    if in_ % GROUP != 0:
        raise ValueError(f"input dimension {in_} must be a multiple of {GROUP}")
    cb = out * in_ // 2
    pb = out * in_ // 32
    if len(weights) != cb or len(scales) != pb or len(biases) != pb:
        raise ValueError("embedding section sizes mismatch")
    w = np.frombuffer(weights, dtype=np.uint8).reshape(out, in_ // 2)
    codes = np.empty((out, in_), dtype=np.uint8)
    codes[:, 0::2] = w & 0x0F
    codes[:, 1::2] = (w >> 4) & 0x0F
    s = bf16_bytes_to_f32(scales).reshape(out, in_ // GROUP)
    b = bf16_bytes_to_f32(biases).reshape(out, in_ // GROUP)
    return codes, s, b
