"""Read MLX affine U32 codes and repack them without a second quantization.

MLX stores successive codes from low to high bits of each little-endian U32.
For group size 64 its BF16 scales and biases already have the dtype and
meaning consumed by Splash: value = unsigned_code * scale + bias.
"""

from __future__ import annotations

import numpy as np

from . import q4


def unpack_codes(raw: bytes, rows: int, logical_cols: int, bits: int) -> np.ndarray:
    if bits not in (4, 8) or logical_cols % 64:
        raise ValueError("expected MLX affine 4/8-bit, group size 64")
    per_word = 32 // bits
    packed_cols = logical_cols // per_word
    words = np.frombuffer(raw, dtype="<u4")
    if words.size != rows * packed_cols:
        raise ValueError("MLX weight byte count does not match logical shape")
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    return ((words.reshape(rows, packed_cols, 1) >> shifts) & ((1 << bits) - 1))\
        .astype(np.uint8).reshape(rows, logical_cols)


def dequantize(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) * np.repeat(scales, 64, axis=1)
            + np.repeat(biases, 64, axis=1))


def q8_quantize_parts(weights: np.ndarray):
    rows, cols = weights.shape
    if cols % 64:
        raise ValueError("Q8 input dimension must be divisible by 64")
    groups = weights.reshape(rows, cols // 64, 64)
    low, high = groups.min(2), groups.max(2)
    scale = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16((high - low) / 255))
    bias = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(low))
    with np.errstate(divide="ignore", invalid="ignore"):
        codes = np.where(scale[..., None] != 0,
                         np.floor((groups - bias[..., None]) / scale[..., None] + 0.5), 0)
    return np.clip(codes, 0, 255).astype(np.uint8).reshape(rows, cols), scale, bias


def q8_pack_parts(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> bytes:
    rows, cols = codes.shape
    if rows % 256 or cols % 64 or scales.shape != (rows, cols // 64) or biases.shape != scales.shape:
        raise ValueError("invalid Splash Q8 dimensions")
    tiles, groups = rows // 256, cols // 64
    weights = codes.reshape(tiles, 256, groups, 64).transpose(0, 2, 1, 3).tobytes()
    scales = scales.reshape(tiles, 256, groups).transpose(0, 2, 1)
    biases = biases.reshape(tiles, 256, groups).transpose(0, 2, 1)
    return weights + q4.f32_to_bf16_bytes(scales) + q4.f32_to_bf16_bytes(biases)
