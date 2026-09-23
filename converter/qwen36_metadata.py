"""Source provenance and compatibility checks for Qwen3.6 packages."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(source: Path, repo: str | None, revision: str | None) -> dict:
    if revision is not None and not repo:
        raise ValueError("--source-revision requires --source-repo")
    if repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("--source-repo must be an owner/model repository ID")
    if revision is not None and not revision.strip():
        raise ValueError("--source-revision must not be empty")
    result = {"config_sha256": sha256(source / "config.json")}
    if repo:
        result["repo_id"] = repo
    else:
        result["local_name"] = source.resolve().name
    if revision:
        result["revision"] = revision
    return result


def chat_template(source: Path) -> str:
    standalone = source / "chat_template.jinja"
    if standalone.is_file():
        template = standalone.read_text(encoding="utf-8")
    else:
        config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if isinstance(template, dict):
            template = template.get("default")
        elif isinstance(template, list):
            defaults = [item.get("template") for item in template
                        if isinstance(item, dict) and item.get("name") == "default"]
            template = defaults[0] if len(defaults) == 1 else None
    if not isinstance(template, str) or not template.strip():
        raise ValueError("source needs a nonempty chat template: chat_template.jinja or a default in tokenizer_config.json")
    return template


def token_ids(path: Path) -> dict:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError(f"{path}: expected a token-to-ID vocabulary")
    result = dict(vocabulary)
    for token in tokenizer.get("added_tokens", []):
        content, token_id = token["content"], token["id"]
        if content in result and result[content] != token_id:
            raise ValueError(f"{path}: conflicting token IDs for {content!r}")
        result[content] = token_id
    if any(not isinstance(token, str) or type(token_id) is not int or token_id < 0
           for token, token_id in result.items()) or len(set(result.values())) != len(result):
        raise ValueError(f"{path}: invalid or duplicate token IDs")
    return result


def reference_metadata(source: Path, config: dict, reference: Path, sizes: dict) -> tuple[dict, str]:
    """Validate metadata before writing target weights or borrowing any assets."""
    ref = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if ref.get("schema_version") != 4 or ref.get("format", {}).get("name") != "splash-packed-q4-moe":
        raise ValueError("reference must be a Splash Qwen3.6 schema-4 package")
    reference_config = json.loads((reference / "tokenizer/config.json").read_text(encoding="utf-8"))
    for key in ("model_type", "architectures", "image_token_id", "video_token_id",
                "vision_start_token_id", "vision_end_token_id"):
        if config.get(key) != reference_config.get(key):
            raise ValueError(f"reference vision/tokenizer config differs at {key}")
    if token_ids(source / "tokenizer.json") != token_ids(reference / "tokenizer/tokenizer.json"):
        raise ValueError("source and reference token IDs differ; the borrowed draft/vision assets require matching token IDs")
    template = chat_template(source)
    # Ensure this required source file exists even when a standalone template wins.
    json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    recorded = {}
    for artifact in ref["artifacts"]:
        rel = artifact["path"]
        parts = PurePosixPath(rel)
        if parts.is_absolute() or ".." in parts.parts or "\\" in rel or rel in recorded:
            raise ValueError(f"invalid reference artifact path: {rel}")
        recorded[rel] = artifact["size"]
    for name, size in sizes.items():
        if recorded.get(name) != size:
            raise ValueError(f"reference size mismatch for {name}")
    return ref, template
