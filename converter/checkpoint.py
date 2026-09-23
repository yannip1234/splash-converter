"""Sharded safetensors reader for the Swift/Qwen BF16 checkpoints.

One tensor at a time, by name, from an 18-shard checkpoint. Nothing is cached
beyond the parsed shard headers, so a 64-layer conversion streams rather than
loading 52 GB.

`data_offsets` in a safetensors header are relative to the start of the data
block (8 + header length), NOT to the file. Reading them as file offsets yields
plausible-looking garbage — weights are homogeneous enough that the result still
correlates like a weight matrix. `_data_start` asserts the invariant on every
shard open so that mistake cannot recur.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np


class Checkpoint:
    """Random access to a sharded BF16 safetensors checkpoint."""

    def __init__(self, root: str):
        self.root = root
        index_path = os.path.join(root, "model.safetensors.index.json")
        self._headers: dict[str, tuple[dict, int]] = {}
        if os.path.isfile(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            header, _ = self._header("model.safetensors")
            self.weight_map = {name: "model.safetensors" for name in header
                               if name != "__metadata__"}

    def _header(self, shard: str):
        if shard not in self._headers:
            path = os.path.join(self.root, shard)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen))
            data_start = 8 + hlen
            max_end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            if data_start + max_end != size:
                raise ValueError(f"{shard}: data_offsets are not data-relative "
                                 f"({data_start} + {max_end} != {size})")
            self._headers[shard] = (header, data_start)
        return self._headers[shard]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def shape(self, name: str):
        shard = self.weight_map[name]
        header, _ = self._header(shard)
        return tuple(header[name]["shape"])

    def dtype(self, name: str) -> str:
        shard = self.weight_map[name]
        header, _ = self._header(shard)
        return header[name]["dtype"]

    def tensor_rows_raw(self, name: str, start: int, stop: int,
                        expert: int | None = None) -> bytes:
        """Read contiguous rows of a rank-2 or rank-3 BF16/U32 tensor."""
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        info = header[name]
        shape = tuple(info["shape"])
        if len(shape) == 2 and expert is None:
            rows, cols = shape
            outer = 0
        elif len(shape) == 3 and expert is not None and 0 <= expert < shape[0]:
            rows, cols = shape[1:]
            outer = expert * rows
        else:
            raise ValueError(f"{name}: unsupported row slice for shape {shape}")
        if not 0 <= start <= stop <= rows:
            raise ValueError(f"{name}: row range {start}:{stop} outside {rows}")
        item_bytes = {"BF16": 2, "U32": 4}.get(info["dtype"])
        if item_bytes is None:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        byte_start = data_start + info["data_offsets"][0] + (outer + start) * cols * item_bytes
        size = (stop - start) * cols * item_bytes
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(byte_start)
            raw = f.read(size)
        if len(raw) != size:
            raise ValueError(f"{name}: short row read")
        return raw

    def f32_expert(self, name: str, expert: int) -> np.ndarray:
        """Read one [out, in] slab from a [experts, out, in] BF16 tensor."""
        shape = self.shape(name)
        if len(shape) != 3 or not 0 <= expert < shape[0]:
            raise ValueError(f"{name}: invalid expert {expert} for {shape}")
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        if header[name]["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16")
        elements = shape[1] * shape[2]
        offset = data_start + header[name]["data_offsets"][0] + expert * elements * 2
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(offset)
            raw = f.read(elements * 2)
        if len(raw) != elements * 2:
            raise ValueError(f"{name}: short expert read")
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape[1:])

    def raw(self, name: str) -> bytes:
        """The tensor's stored bytes, untouched — for byte-identical copies."""
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        info = header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
        start, end = info["data_offsets"]
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(data_start + start)
            return f.read(end - start)

    def f32(self, name: str) -> np.ndarray:
        """The tensor as float32, exactly (bf16 -> f32 is lossless)."""
        shard = self.weight_map[name]
        header, _ = self._header(shard)
        shape = tuple(header[name]["shape"])
        bits = np.frombuffer(self.raw(name), dtype="<u2")
        return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)

    def f32_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) of a 2-D BF16 tensor, as float32.

        Reads only those rows off disk — `embed_tokens` and `lm_head` are
        248320x5120 (5.1 GB as float32), so the head/embedding writers stream
        them a tile at a time instead of materializing the whole matrix.
        """
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        info = header[name]
        shape = info["shape"]
        if len(shape) != 2:
            raise ValueError(f"{name}: f32_rows needs a 2-D tensor, got {shape}")
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError(f"{name}: row range {start}:{stop} outside {shape[0]}")
        cols = shape[1]
        base = data_start + info["data_offsets"][0]
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(base + start * cols * 2)
            raw = f.read((stop - start) * cols * 2)
        bits = np.frombuffer(raw, dtype="<u2")
        return (bits.astype(np.uint32) << 16).view(np.float32).reshape(stop - start, cols)
