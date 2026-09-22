"""Loader-equivalent verification of a Splash weight file.

Mirrors what `WeightStore`/`WeightFile` enforce at load time
(docs/SPLASH_FORMAT_SPEC.md §2, WeightStore.cpp:169-221):

- file size >= 16, a multiple of 16384, and >= header + one section
- exact 8-byte magic; `layer` and `type` u32 little-endian must match
- each section starts at the running offset aligned up to 16384 and consumes
  exactly its declared byte count
- `finish()`: the aligned consumed offset must equal the file size

Used by T3 per-layer checks and by the whole-package harness in T5.5.
"""

from __future__ import annotations

import os
import struct

ALIGN = 16384
HEADER_BYTES = 16

# Section byte counts in file order (spec §5.1 / §5.2).
GDN_SECTIONS = [
    ("input-norm", 10240), ("gdn-input", 47923200), ("gdn-convolution", 81920),
    ("gdn-decay", 192), ("gdn-time-bias", 96), ("gdn-norm", 256),
    ("gdn-output", 17694720), ("post-attention-norm", 10240),
    ("mlp-gate", 50135040), ("mlp-up", 50135040), ("mlp-down", 50135040),
]
ATTENTION_SECTIONS = [
    ("input-norm", 10240), ("attention-input", 41287680), ("query-norm", 512),
    ("key-norm", 512), ("attention-output", 17694720), ("post-attention-norm", 10240),
    ("mlp-gate", 50135040), ("mlp-up", 50135040), ("mlp-down", 50135040),
]


HEAD_SECTIONS = [("final-norm", 10240), ("logits", 715161600)]              # spec §5.3
EMBEDDING_SECTIONS = [("embedding-weights", 635699200),                     # spec §5.4
                      ("embedding-scales", 39731200), ("embedding-biases", 39731200)]


class VerificationError(AssertionError):
    pass


def _check(condition, message):
    if not condition:
        raise VerificationError(message)


def walk(path, magic: bytes, layer: int, type_: int, sections):
    """Walk a weight file exactly as the loader does.

    Returns [(label, offset, bytes)]. Raises VerificationError on any condition
    the loader would reject.
    """
    size = os.path.getsize(path)
    _check(size >= HEADER_BYTES, f"{path}: size {size} < header")
    _check(size % ALIGN == 0, f"{path}: size {size} is not a multiple of {ALIGN}")
    _check(size >= HEADER_BYTES + sections[0][1], f"{path}: smaller than header + first section")

    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    _check(head[:8] == magic, f"{path}: magic {head[:8]!r} != {magic!r}")
    got_layer = struct.unpack("<I", head[8:12])[0]
    got_type = struct.unpack("<I", head[12:16])[0]
    _check(got_layer == layer, f"{path}: layer {got_layer} != {layer}")
    _check(got_type == type_, f"{path}: type {got_type} != {type_}")

    table, offset = [], HEADER_BYTES
    for label, nbytes in sections:
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        _check(offset + nbytes <= size, f"{path}: section {label} overruns the file")
        table.append((label, offset, nbytes))
        offset += nbytes
    consumed = (offset + ALIGN - 1) // ALIGN * ALIGN
    _check(consumed == size, f"{path}: consumed {consumed} != file size {size} (finish())")
    return table


def walk_target_layer(path, layer: int):
    """Verify a target/layer-N.bin, choosing the section list by layer index."""
    from .layer import is_full_attention, TARGET_LAYER_MAGIC
    attention = is_full_attention(layer)
    sections = ATTENTION_SECTIONS if attention else GDN_SECTIONS
    return walk(path, TARGET_LAYER_MAGIC, layer, 1 if attention else 0, sections)


def walk_head(path):
    """Verify target/head.bin (magic MDFL0002, layer 64, type 2)."""
    from .layer import HEAD_MAGIC
    return walk(path, HEAD_MAGIC, 64, 2, HEAD_SECTIONS)


def walk_embedding(path):
    """Verify target/embedding.bin (magic MDFE0001, layer/type carry the geometry)."""
    from .layer import EMBEDDING_MAGIC, VOCAB, HIDDEN
    return walk(path, EMBEDDING_MAGIC, VOCAB, HIDDEN, EMBEDDING_SECTIONS)
