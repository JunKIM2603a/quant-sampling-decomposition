"""Validate regenerated assets against the recorded contract before any writes."""
import json
from pathlib import Path
from .data import manifest_hash


def validate_asset_contract(manifest_path, lock_path, manifest, lock, worklists, output_dir):
    manifest_path, lock_path, output_dir = map(Path, (manifest_path, lock_path, output_dir))
    if manifest_path.exists():
        if manifest_hash(json.loads(manifest_path.read_text())) != manifest_hash(manifest):
            raise ValueError("regenerated split differs from recorded split; no files written")
    if lock_path.exists():
        recorded = json.loads(lock_path.read_text())
        # Runtime package differences are reported separately, never rewritten into the lock.
        if {k: v for k, v in recorded.items() if k != "packages"} != {
                k: v for k, v in lock.items() if k != "packages"}:
            raise ValueError("regenerated execution lock differs from recorded lock; no files written")
    for name, content in worklists.items():
        path = output_dir / f"{name}.jsonl"
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"existing {name} worklist differs; no files written")
