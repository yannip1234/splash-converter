"""Q4 encoder rule for the Splash `splash-packed-q4` format (PLAN T1.4).

Derived from the reference package by fitting against the checkpoint it was
actually built from -- `Qwen/Qwen3.8-27B` BF16, *not* the Swift fine-tune
(provenance established in PROGRESS iteration 4). The rule is stated in
docs/SPLASH_FORMAT_SPEC.md section 9.

Per group of 64 consecutive input elements of one output row:

    mn, mx   = group min, max              (bf16-exact, the source is bf16)
    maxabs   = max(|mn|, |mx|)             magnitude of the dominant extreme
    span     = mx - mn
    k        = round_half_up(15*maxabs/span)   integer grid position of zero
    |s|      = bf16(maxabs / k)
    s        = +|s| if mn+mx < 0 else -|s|     sign points away from the anchor
    b        = mn   if mn+mx < 0 else mx       anchor = dominant extreme
    c        = clip(round_half_up((w - b)/s), 0, 15)

The 16 codes span 15 intervals across [mn, mx]; `k` is chosen so that zero
lands exactly on an integer code (hence the measured b/s ~= -k and the code
histogram peaking at 8), and `|s| = maxabs/k` makes the dominant extreme
exactly representable at code 0.

Dequantization (unchanged, kernel epilogue): w = c*s + b, so w = (c - k)*s.
"""

from __future__ import annotations

import numpy as np

from . import q4
from .q4 import GROUP


def _round_half_up(x: np.ndarray) -> np.ndarray:
    """Round to nearest, ties away from zero on the positive side (x >= 0)."""
    return np.floor(x + 0.5)


def group_parameters(w: np.ndarray):
    """Per-group (scale, bias, k) for a [out, in] float32 weight matrix.

    Returns (scales f32 [out, in//64], biases f32 [out, in//64], k int32).
    Scales/biases are already bf16-rounded, i.e. exactly the stored values.
    """
    out, in_ = w.shape
    g = w.reshape(out, in_ // GROUP, GROUP)
    mn = g.min(axis=2)
    mx = g.max(axis=2)
    span = mx - mn
    maxabs = np.maximum(np.abs(mn), np.abs(mx))

    # k = grid position of zero. Degenerate groups (span == 0) get k = 0/s = 0,
    # which stores the constant in the bias and every code at 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(span > 0, 15.0 * maxabs / span, 0.0)
    k = _round_half_up(x).astype(np.int32)
    k = np.clip(k, 1, 15)

    with np.errstate(divide="ignore", invalid="ignore"):
        s_mag = np.where(span > 0, maxabs / k.astype(np.float32), 0.0)
    s_mag = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(s_mag.astype(np.float32)))

    anchor_low = (mn + mx) < 0  # |mn| dominant -> anchor at mn, scale positive
    scales = np.where(anchor_low, s_mag, -s_mag).astype(np.float32)
    # A degenerate group (span == 0, e.g. the 160 zero-padding rows of gdn-input)
    # would otherwise store -0.0; the reference package stores +0.0 there.
    scales = np.where(s_mag == 0, np.float32(0.0), scales)
    biases = np.where(anchor_low, mn, mx).astype(np.float32)
    # The bias is stored as bf16; the source is bf16 so it is already exact,
    # but round explicitly so callers cannot smuggle an f32 bias through.
    biases = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(biases))
    return scales, biases, k


def quantize_projection(w: np.ndarray):
    """Quantize a [out, in] float32 weight matrix to (codes, scales, biases)."""
    out, in_ = w.shape
    scales, biases, _ = group_parameters(w)
    s_rep = np.repeat(scales, GROUP, axis=1)
    b_rep = np.repeat(biases, GROUP, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s_rep != 0, (w - b_rep) / s_rep, 0.0).astype(np.float32)
    codes = np.clip(_round_half_up(q), 0, 15).astype(np.uint8)
    return codes, scales, biases


# The quantizer every section writer uses. `quantize_projection` (the pinned
# reference rule) is the default; `converter.build --quantizer clip` swaps it.
ACTIVE_QUANTIZER = None   # set below, once quantize_projection_clip exists


def quantize_and_pack(w: np.ndarray) -> bytes:
    """Quantize and pack a projection straight to its section bytes."""
    codes, scales, biases = (ACTIVE_QUANTIZER or quantize_projection)(w)
    return q4.pack_projection(codes, scales, biases)


def use_quantizer(name: str):
    """Select the quantizer by name ('reference' or 'clip'); returns the callable."""
    global ACTIVE_QUANTIZER
    if name not in QUANTIZERS:
        raise ValueError(f"unknown quantizer {name!r}; choose from {sorted(QUANTIZERS)}")
    ACTIVE_QUANTIZER = QUANTIZERS[name]
    return ACTIVE_QUANTIZER


# ---------------------------------------------------------------------------
# Alternative quantizer (opt-in, NOT the pinned rule).
#
# The container fixes only "4-bit code + bf16 scale + bf16 bias per 64-group";
# the choice of scale and bias is free. Reproducing the reference encoder is
# required to *verify provenance* against the shipped package, but for Swift
# weights there is no obligation to match it.
#
# `quantize_projection_clip` shrinks each group's range by the factor that
# minimises squared error, trading harder clipping of outliers for finer steps
# on the bulk. Measured (PROGRESS iteration 11) at +4.2e-04 to +4.8e-04 cosine
# over the pinned rule -- about five times the gain available from choosing a
# better k within the pinned rule.
#
# Caveat recorded with the measurement: lower squared error is not the same as a
# better model, and transformer quantization is specifically sensitive to the
# outlier weights this trades away. Use only with a quality benchmark behind it.
# ---------------------------------------------------------------------------

CLIP_FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def quantize_projection_clip(w: np.ndarray, factors=CLIP_FACTORS):
    """Per-group range search minimising squared error. Returns (codes, scales, biases)."""
    out, in_ = w.shape
    g = w.reshape(out, in_ // GROUP, GROUP)
    mn = g.min(axis=2)
    mx = g.max(axis=2)
    middle = (mn + mx) / 2
    half = (mx - mn) / 2
    flat = w
    best_err = best_s = best_b = None
    for factor in factors:
        lo = middle - half * factor
        hi = middle + half * factor
        s = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(((hi - lo) / 15.0).astype(np.float32)))
        s = np.where(s == 0, np.float32(0.0), s).astype(np.float32)
        b = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(lo.astype(np.float32)))
        s_rep = np.repeat(s, GROUP, axis=1)
        b_rep = np.repeat(b, GROUP, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s_rep != 0, (flat - b_rep) / s_rep, 0.0)
        codes = np.clip(_round_half_up(q), 0, 15)
        err = ((codes * s_rep + b_rep - flat) ** 2).reshape(out, in_ // GROUP, GROUP).sum(axis=2)
        if best_err is None:
            best_err, best_s, best_b = err, s, b
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_s = np.where(take, s, best_s)
            best_b = np.where(take, b, best_b)
    s_rep = np.repeat(best_s, GROUP, axis=1)
    b_rep = np.repeat(best_b, GROUP, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s_rep != 0, (flat - b_rep) / s_rep, 0.0)
    codes = np.clip(_round_half_up(q), 0, 15).astype(np.uint8)
    return codes, best_s.astype(np.float32), best_b.astype(np.float32)


def quantize_projection_minmax(w: np.ndarray):
    """Plain asymmetric min/max: the grid spans [min, max] exactly, nothing clipped.

    This is what `mx.quantize` does (affine, group 64) and what every other
    quantizer in the ecosystem does. It differs from the pinned reference rule in
    one way that turns out to matter enormously: the reference rule snaps zero
    onto the grid, which shortens the representable range and clips ~1.4% of
    weights -- the largest-magnitude ones. Swift's late layers are heavy-tailed
    (kurtosis 8-9), so clipping its outliers costs real accuracy.
    """
    out, in_ = w.shape
    g = w.reshape(out, in_ // GROUP, GROUP)
    mn = g.min(axis=2)
    mx = g.max(axis=2)
    scales = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(((mx - mn) / 15.0).astype(np.float32)))
    scales = np.where(scales == 0, np.float32(0.0), scales).astype(np.float32)
    biases = q4.bf16_u16_to_f32(q4.f32_to_bf16_u16(mn.astype(np.float32)))
    s_rep = np.repeat(scales, GROUP, axis=1)
    b_rep = np.repeat(biases, GROUP, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s_rep != 0, (w - b_rep) / s_rep, 0.0).astype(np.float32)
    codes = np.clip(_round_half_up(q), 0, 15).astype(np.uint8)
    return codes, scales, biases


QUANTIZERS = {"reference": quantize_projection, "clip": quantize_projection_clip,
              "minmax": quantize_projection_minmax}
