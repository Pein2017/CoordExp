"""Read-only path routing for documents moved during the research migration."""
from __future__ import annotations

import json
from pathlib import Path


INDEX = Path("manifests/documentation-layout.json")


class DocumentLocations:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.index = self.root / INDEX
        self.entries: dict[str, dict[str, str]] = {}
        self.directories: dict[str, str] = {}
        if not self.index.is_file():
            return
        value = json.loads(self.index.read_text())
        if value.get("schema") != "coordexp.documentation_layout.v1":
            raise ValueError("unsupported documentation location index")
        self.directories = value.get("directory_routes", {})
        for source, target in self.directories.items():
            self.local(source)
            self.local(target)
        for entry in value["files"]:
            source, target = entry["source"], entry.get("target")
            self.local(source)
            if source in self.entries:
                raise ValueError("duplicate documentation source location")
            if target is not None:
                self.local(target)
                self.entries[source] = {"target": target}

    def local(self, name: str) -> Path:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\x00" in name:
            raise ValueError("source location must stay within its checkout")
        target = self.root / path
        if not target.resolve().is_relative_to(self.root):
            raise ValueError("source location escapes checkout")
        return target

    def target(self, name: str) -> str:
        """Follow a retired path to its current owner without claiming byte identity."""
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
