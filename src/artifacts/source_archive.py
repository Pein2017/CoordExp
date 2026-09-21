"""Explicit, read-only recovery of hash-bound source evidence.

This reader verifies bytes. It never imports, executes, restores, or rewrites
an archived source, and is not a fallback for current implementation identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceArchive:
    """A caller-selected manifest, indexed by the exact expected content hash."""

    def __init__(self, manifest: Path):
        self.manifest = Path(manifest).resolve(strict=True)
        value = json.loads(self.manifest.read_text())
        if value.get("schema") != "coordexp.output_source_archive.v1":
            raise ValueError("unsupported source-archive manifest")
        self.files = value["files"]
        self.by_hash: dict[str, list[dict[str, Any]]] = {}
        for entry in self.files:
            if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("invalid archived SHA-256")
            if not Path(entry["source"]).is_absolute() or not Path(entry["archive"]).is_absolute():
                raise ValueError("source archive requires explicit absolute locations")
            self.by_hash.setdefault(entry["sha256"], []).append(entry)

    def resolve(self, source: Path, expected_sha256: str) -> dict[str, Any]:
        source = Path(source)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("an exact expected SHA-256 is required")
        if source.is_file() and file_sha256(source) == expected_sha256:
            return {"source": str(source), "path": str(source),
                    "sha256": expected_sha256, "size_bytes": source.stat().st_size,
                    "resolution": "live_exact_bytes"}
        candidates = sorted(self.by_hash.get(expected_sha256, []),
                            key=lambda item: (item["source"] != str(source), item["archive"]))
        for entry in candidates:
            path = Path(entry["archive"])
            if path.is_file() and not path.is_symlink() and file_sha256(path) == expected_sha256:
                return {"source": str(source), "path": str(path),
                        "sha256": expected_sha256, "size_bytes": path.stat().st_size,
                        "resolution": "archived_exact_path" if entry["source"] == str(source)
                        else "archived_identical_bytes", "manifest": str(self.manifest)}
        raise FileNotFoundError(f"no verified source bytes: {source} sha256={expected_sha256}")

    def verify_bindings(self, bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        resolved = {}
        for key, binding in bindings.items():
            result = self.resolve(Path(binding["path"]), binding["sha256"])
            if "size_bytes" in binding and result["size_bytes"] != binding["size_bytes"]:
                raise ValueError(f"historical source size differs: {key}")
            resolved[key] = result
        return resolved

    def verify_archive(self) -> dict[str, int]:
        for entry in self.files:
            path = Path(entry["archive"])
            if path.is_symlink() or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"archived bytes changed: {path}")
        return {"verified_files": len(self.files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    archive = SourceArchive(args.manifest)
    if args.verify:
        result = archive.verify_archive()
    elif args.source is not None and args.sha256 is not None:
        result = archive.resolve(args.source, args.sha256)
    else:
        parser.error("choose --verify or both --source and --sha256")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
