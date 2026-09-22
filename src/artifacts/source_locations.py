"""Read-only location changes for retained source bytes and document ownership.

This is a file-move index, not an import hook or a current-source fallback.
Frozen manifests remain unchanged. Their exact hashes still own byte identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


INDEX = Path("manifests/documentation-layout.json")


class SourceLocations:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.index = self.root / INDEX
        self.entries: dict[str, dict] = {}
        self.directories: dict[str, str] = {}
        if self.index.is_file():
            value = json.loads(self.index.read_text())
            if value.get("schema") != "coordexp.documentation_layout.v1":
                raise ValueError("unsupported documentation location index")
            self.directories = value.get("directory_routes", {})
            for source, target in self.directories.items():
                self.local(source)
                self.local(target)
            for entry in value["files"]:
                source = entry["source"]
                self.local(source)
                if source in self.entries:
                    raise ValueError("duplicate documentation source location")
                for name in ("target", "preserved"):
                    if entry.get(name) is not None:
                        self.local(entry[name])
                if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                    raise ValueError("invalid original source hash")
                self.entries[source] = entry

    def local(self, name: str) -> Path:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\x00" in name:
            raise ValueError("source location must stay within its checkout")
        target = self.root / path
        if not target.resolve().is_relative_to(self.root):
            raise ValueError("source location escapes checkout")
        return target

    def relative(self, path: str | Path) -> str | None:
        path = Path(path)
        if path.is_absolute():
            if not path.is_relative_to(self.root):
                return None
            path = path.relative_to(self.root)
        return str(path)

    def target(self, name: str) -> str:
        """Locate the present content owner without claiming unchanged bytes."""
        seen: set[str] = set()
        while name in self.entries:
            if name in seen:
                raise ValueError("cyclic source location index")
            seen.add(name)
            target = self.entries[name].get("target")
            if target is None or target == name:
                break
            name = target
        return self.directories.get(name, name)

    def materialized(self, source: str | Path, expected: str) -> Path | None:
        """Find an existing exact-byte copy; never extract or execute an archive."""
        name = self.relative(source)
        entry = self.entries.get(name) if name is not None else None
        if entry is None or entry["sha256"] != expected:
            return None
        for key in ("preserved", "target"):
            if not entry.get(key):
                continue
            path = self.local(entry[key])
            if path.is_file() and not path.is_symlink():
                if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                    return path
        return None

    def original_bytes(self, source: str, expected: str) -> bytes:
        """Recover original documentation for checks, including pinned Git bytes."""
        entry = self.entries.get(source)
        path = self.local(source)
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == expected:
                return data
        retained = self.materialized(source, expected)
        if retained is not None:
            return retained.read_bytes()
        spec = entry.get("git_blob", "") if entry and entry["sha256"] == expected else ""
        if not re.fullmatch(r"[0-9a-f]{40}:.+", spec):
            raise FileNotFoundError(f"no exact retained source: {source}")
        data = subprocess.check_output(["git", "cat-file", "blob", spec], cwd=self.root)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"retained Git bytes differ: {source}")
        return data
