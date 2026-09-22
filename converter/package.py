r"""Assemble and verify the complete Splash package (PLAN Phase 5).

Layout and required files: docs/SPLASH_FORMAT_SPEC.md §1, §8.

Provenance of each part:
- `target/`   converted from the Swift BF16 checkpoint (converter/build.py)
- `draft/`    byte-copied from the reference package — the existing DFlash2 drafter
              is reused unchanged (SPEC §5 item 2; no retraining in scope)
- `vision/`   byte-copied from the reference package (compatibility doc §5)
- `tokenizer/` Swift's own five files — byte-identical to the base HF repo, and
              deliberately NOT the reference package's, whose tokenizer is a separate
              MLX-style repack (SWIFT_COMPATIBILITY §2.1 documents the differences; only
              `vocab.json` is byte-identical between the two). Independently reconfirmed
              here: the two are semantically identical (same 248044-entry vocab, same
              247587 merges, same 33 added tokens), differing in merges serialization
              (pairs vs strings — the whole 7 MB), the pre-tokenizer's `\p{M}` class, and
              ByteLevel decoder flags. Under the pinned `tokenizers==0.22.2` both files
              load and produce identical token ids on every probe tried.

`verify_package` is the loader- *and* installer-equivalent harness (PLAN T5.5): it
applies every rule in install/models.py:100-190 plus the per-file checks in
converter/verify.py, re-hashes every artifact, and computes both fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from posixpath import normpath

from . import layer as layer_mod
from . import verify

ALIGNMENT = 16384
TOKENIZER_FILES = ["chat_template.jinja", "config.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json"]
DRAFT_LAYERS = 5
TARGET_LAYERS = 64

# Header expectations for the files we copy rather than build (spec §2.3).
DRAFT_LAYER_BYTES = 187449344
DRAFT_MODEL_BYTES = 328794112
VISION_BYTES = 930250752
DRAFT_MAGIC = b"MDFD0004"
VISION_MAGIC = b"MDFV0001"

FORMAT = {
    "name": "splash-packed-q4",
    "q4_bits": 4,
    "q4_group_size": 64,
    "q4_storage_n": 256,
    "section_alignment_bytes": ALIGNMENT,
    "target_layer_magic": "MDFL0006",
    "draft_layer_magic": "MDFD0004",
    "vision_magic": "MDFV0001",
}


class PackageError(AssertionError):
    pass


def sha256_file(path, chunk=1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _header(path):
    with open(path, "rb") as f:
        head = f.read(16)
    return head[:8], struct.unpack("<I", head[8:12])[0], struct.unpack("<I", head[12:16])[0]


def weight_records(root):
    """(relativePath, declaredBytes, magic, layer, type) for all 73 weight files.

    Mirrors the records the loaders build (QwenTarget.hpp:87-116, DFlashDraft.cpp,
    QwenVision.cpp); `relativePath` is package-root-relative, as there.
    """
    records = []
    for n in range(TARGET_LAYERS):
        records.append(f"target/layer-{n}.bin")
    records += ["target/head.bin", "target/embedding.bin"]
    records += [f"draft/layer-{n}.bin" for n in range(DRAFT_LAYERS)]
    records += ["draft/model.bin", "vision/model.bin"]
    out = []
    for rel in records:
        path = os.path.join(root, rel)
        magic, layer, type_ = _header(path)
        out.append((rel, os.path.getsize(path), magic.decode("ascii"), layer, type_))
    return out


def manifest_fingerprint(records) -> str:
    """WeightStore.cpp:302-333 — sorted by path, tab-separated, v1 preamble."""
    lines = ["splash-packed-manifest-v1\n"]
    for rel, declared, magic, layer, type_ in sorted(records, key=lambda r: r[0]):
        lines.append(f"{rel}\t{declared}\t{magic}\t{layer}\t{type_}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


# --- assembly ---------------------------------------------------------------

def copy_part(src_root, dst_root, relative, log=print):
    src = os.path.join(src_root, relative)
    dst = os.path.join(dst_root, relative)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        raise PackageError(f"missing source file: {src}")
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        raise PackageError(f"short copy: {relative}")
    return dst


def build_manifest(root, model: str, upstream: dict, execution_geometry: dict) -> dict:
    """Artifact list + validated fields. Records carry exactly {path, size, sha256}."""
    artifacts = []
    for rel, _, _, _, _ in weight_records(root):
        path = os.path.join(root, rel)
        artifacts.append({"path": rel, "size": os.path.getsize(path),
                          "sha256": sha256_file(path)})
    for name in TOKENIZER_FILES:
        rel = f"tokenizer/{name}"
        path = os.path.join(root, rel)
        artifacts.append({"path": rel, "size": os.path.getsize(path),
                          "sha256": sha256_file(path)})
    artifacts.sort(key=lambda a: a["path"])
    return {
        "schema_version": 3,
        "model": model,
        "format": dict(FORMAT),
        "execution_geometry": dict(execution_geometry),
        "artifacts": artifacts,
        "upstream": upstream,
    }


def write_manifest(root, manifest) -> str:
    path = os.path.join(root, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


# --- verification (T5.5) ----------------------------------------------------

def _validate_manifest_rules(manifest):
    """install/models.py:100-190, applied to a manifest dict."""
    fmt = manifest.get("format")
    if not isinstance(fmt, dict) or fmt.get("name") != "splash-packed-q4":
        raise PackageError("unsupported package format name")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 3:
        raise PackageError("schema_version must be the integer 3")
    if not isinstance(manifest.get("model"), str) or not manifest["model"].strip():
        raise PackageError("model must be a non-empty string")
    if not isinstance(manifest.get("execution_geometry"), dict):
        raise PackageError("execution_geometry must be an object")
    for key, value in FORMAT.items():
        if type(fmt.get(key)) is not type(value) or fmt[key] != value:
            raise PackageError(f"format.{key} mismatch: {fmt.get(key)!r} != {value!r}")

    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise PackageError("manifest has no artifact list")
    seen = set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}
                or not isinstance(record["path"], str)):
            raise PackageError(f"invalid artifact record: {record!r}")
        p = record["path"]
        if (p.startswith("/") or not p or ".." in p.split("/") or normpath(p) != p
                or any(c in p for c in "\\*?[]") or any(ord(c) < 32 for c in p)
                or p == "manifest.json" or type(record["size"]) is not int
                or record["size"] <= 0 or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise PackageError(f"invalid artifact record: {p!r}")
        if p in seen:
            raise PackageError(f"duplicate artifact path: {p}")
        if p.endswith(".bin") and record["size"] % ALIGNMENT:
            raise PackageError(f"unaligned packed file: {p}")
        seen.add(p)
    for name in seen:
        parts = name.split("/")
        for i in range(1, len(parts)):
            if "/".join(parts[:i]) in seen:
                raise PackageError(f"artifact paths overlap: {name}")
    required = {"target/embedding.bin", "target/head.bin", "draft/model.bin",
                "vision/model.bin",
                *(f"target/layer-{i}.bin" for i in range(TARGET_LAYERS)),
                *(f"draft/layer-{i}.bin" for i in range(DRAFT_LAYERS)),
                *(f"tokenizer/{n}" for n in TOKENIZER_FILES)}
    missing = required - seen
    if missing:
        raise PackageError("manifest is missing: " + ", ".join(sorted(missing)))
    return records


def verify_package(root, full_hash: bool = True, log=print) -> dict:
    """Whole-package harness: manifest rules, artifact hashes, per-file loader walks."""
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise PackageError("manifest.json missing")
    with open(manifest_path) as f:
        manifest = json.load(f)
    records = _validate_manifest_rules(manifest)

    # every artifact present, right size, right hash
    for record in records:
        path = os.path.join(root, record["path"])
        if not os.path.isfile(path):
            raise PackageError(f"missing artifact: {record['path']}")
        if os.path.getsize(path) != record["size"]:
            raise PackageError(f"wrong size: {record['path']}")
        if full_hash and sha256_file(path) != record["sha256"]:
            raise PackageError(f"sha256 mismatch: {record['path']}")

    # Files present but not declared. The installer does not forbid these -- the
    # reference package itself ships LICENSE, README.md and a HuggingFace cache --
    # so they are reported, not rejected. Dot-directories are ignored entirely.
    declared = {r["path"] for r in records} | {"manifest.json"}
    on_disk = set()
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in names:
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            on_disk.add(rel)
    undeclared = sorted(on_disk - declared)

    # per-file structural checks, as the loaders do them
    for n in range(TARGET_LAYERS):
        verify.walk_target_layer(os.path.join(root, f"target/layer-{n}.bin"), n)
    verify.walk_head(os.path.join(root, "target/head.bin"))
    verify.walk_embedding(os.path.join(root, "target/embedding.bin"))
    for n in range(DRAFT_LAYERS):
        path = os.path.join(root, f"draft/layer-{n}.bin")
        magic, layer, type_ = _header(path)
        if (magic, layer, type_) != (DRAFT_MAGIC, n, 0):
            raise PackageError(f"draft/layer-{n}.bin header {(magic, layer, type_)}")
        if os.path.getsize(path) != DRAFT_LAYER_BYTES:
            raise PackageError(f"draft/layer-{n}.bin size")
    magic, layer, type_ = _header(os.path.join(root, "draft/model.bin"))
    if (magic, layer, type_) != (DRAFT_MAGIC, 5, 1):
        raise PackageError(f"draft/model.bin header {(magic, layer, type_)}")
    magic, layer, type_ = _header(os.path.join(root, "vision/model.bin"))
    if (magic, layer, type_) != (VISION_MAGIC, 27, 0):
        raise PackageError(f"vision/model.bin header {(magic, layer, type_)}")

    records_for_fingerprint = weight_records(root)
    result = {
        "root": root,
        "artifacts": len(records),
        "weight_files": len(records_for_fingerprint),
        "bytes": sum(r["size"] for r in records),
        "packageManifestSha256": sha256_file(manifest_path),
        "manifestFingerprintSha256": manifest_fingerprint(records_for_fingerprint),
        "hashes_verified": full_hash,
        "undeclared_files": undeclared,
    }
    log(f"package OK: {result['artifacts']} artifacts, {result['weight_files']} weight files, "
        f"{result['bytes']:,} bytes")
    if undeclared:
        log(f"  note: {len(undeclared)} undeclared file(s) present: "
            + ", ".join(undeclared[:4]) + (" …" if len(undeclared) > 4 else ""))
    return result


# --- CLI --------------------------------------------------------------------

def assemble_package(out_root, swift_root, reference_root, model: str,
                     swift_revision: str, log=print) -> dict:
    """PLAN T5.1-T5.4. `target/` must already be built by converter.build."""
    for n in range(TARGET_LAYERS):
        if not os.path.exists(os.path.join(out_root, f"target/layer-{n}.bin")):
            raise PackageError(f"target/layer-{n}.bin not built yet — run converter.build")
    for name in ("head.bin", "embedding.bin"):
        if not os.path.exists(os.path.join(out_root, "target", name)):
            raise PackageError(f"target/{name} not built yet — run converter.build")

    copied = []
    for n in range(DRAFT_LAYERS):                                   # T5.2
        copied.append(copy_part(reference_root, out_root, f"draft/layer-{n}.bin"))
    copied.append(copy_part(reference_root, out_root, "draft/model.bin"))
    copied.append(copy_part(reference_root, out_root, "vision/model.bin"))  # T5.1
    log(f"copied {len(copied)} reused file(s) from {reference_root}")

    for name in TOKENIZER_FILES:                                    # T5.3
        src = os.path.join(swift_root, name)
        dst = os.path.join(out_root, "tokenizer", name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    log(f"copied {len(TOKENIZER_FILES)} tokenizer file(s) from {swift_root}")

    with open(os.path.join(reference_root, "manifest.json")) as f:
        reference = json.load(f)
    upstream = {                                                    # T5.4
        "target": {"repo_id": "ukisai/Swift-Qwen3.8-27b", "revision": swift_revision},
        "tokenizer": {"repo_id": "ukisai/Swift-Qwen3.8-27b", "revision": swift_revision},
        "draft": dict(reference["upstream"]["draft"]),
        "vision": dict(reference["upstream"]["vision"]),
    }
    manifest = build_manifest(out_root, model, upstream, reference["execution_geometry"])
    path = write_manifest(out_root, manifest)
    log(f"wrote {path}: {len(manifest['artifacts'])} artifacts")
    return manifest


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="converter.package")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--out", default="output/swift-splash")
    a.add_argument("--swift", default="refs/swift-bf16")
    a.add_argument("--reference", default="refs/qwen38-splash")
    a.add_argument("--model", default="Qwen3.8-27B")
    a.add_argument("--revision", default="048328f4059015b63f860a453bf94834af0db683")
    v = sub.add_parser("verify")
    v.add_argument("--root", default="output/swift-splash")
    v.add_argument("--quick", action="store_true", help="skip artifact re-hashing")
    v.add_argument("--report", default="output/reports/package.json")
    args = p.parse_args(argv)

    if args.command == "assemble":
        assemble_package(args.out, args.swift, args.reference, args.model, args.revision)
        return 0
    result = verify_package(args.root, full_hash=not args.quick)
    for key in ("artifacts", "weight_files", "bytes", "packageManifestSha256",
                "manifestFingerprintSha256"):
        print(f"  {key}: {result[key]}")
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def install_model_card(out_root, card="docs/MODEL_CARD.md", log=print):
    """Copy the tracked model card into the package as README.md.

    The package lives under output/, which is gitignored, so the card must be
    versioned in the repo and copied in — otherwise edits to it are invisible to
    review and are lost on a clean rebuild.
    """
    if not os.path.exists(card):
        raise PackageError(f"model card not found: {card}")
    shutil.copyfile(card, os.path.join(out_root, "README.md"))
    for name in ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE"):
        source = os.path.join("docs", "package", name)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(out_root, name))
    log(f"installed {card} -> {out_root}/README.md")
