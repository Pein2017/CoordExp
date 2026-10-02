"""Read-only output/source boundary check; never execute or delete findings."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat


FORBIDDEN_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".md", ".sh"})
FORBIDDEN_DIRECTORIES = frozenset({".venv", "venv", ".git", "__pycache__"})
PAYLOAD_MANIFEST = "inference_payload_manifest.json"
NATIVE_PAYLOAD_SCHEMA = "coordexp-infras-inference-checkpoint-payload-manifest"
LEGACY_IDENTITY_FILES = frozenset(
    {
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "special_token_embeddings/special_token_embeddings.json",
        "special_token_embeddings/special_token_embeddings.safetensors",
    }
)


def _is_regular_nonsymlink(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _is_directory_nonsymlink(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _native_adapter_readme_binding(root: Path) -> tuple[Path | None, bool]:
    """Return the bound README and whether a native manifest declares adapter/."""
    manifest_path = root / PAYLOAD_MANIFEST
    adapter_root = root / "adapter"
    if not _is_regular_nonsymlink(manifest_path):
        return None, False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != NATIVE_PAYLOAD_SCHEMA
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            return None, False
        adapter = manifest.get("adapter")
        if (
            not isinstance(adapter, dict)
            or adapter.get("relative_root") != "adapter"
        ):
            return None, False
        if (
            adapter.get("status") != "present"
            or not _is_directory_nonsymlink(adapter_root)
        ):
            return None, True
        entries = adapter.get("files")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("relative_path"), str)
            for entry in entries
        ):
            return None, True
        readme_entries = [
            entry for entry in entries if entry["relative_path"] == "README.md"
        ]
        if len(readme_entries) != 1:
            return None, True
        for filename in (
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            if not _is_regular_nonsymlink(adapter_root / filename):
                return None, True

        readme_path = adapter_root / "README.md"
        readme_bytes = readme_path.read_bytes()
        readme_entry = readme_entries[0]
        size_bytes = readme_entry.get("size_bytes")
        digest = readme_entry.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes != len(readme_bytes)
            or not isinstance(digest, str)
            or hashlib.sha256(readme_bytes).hexdigest() != digest
        ):
            return None, True
        return readme_path, True
    except (OSError, ValueError, TypeError):
        return None, False


def _legacy_identity_adapter_readme_binding(
    checkpoint_root: Path,
) -> tuple[Path | None, bool]:
    """Bind a legacy card only to its exact frozen checkpoint identity."""
    identity_path = checkpoint_root / "identity.json"
    identity_present = identity_path.exists() or identity_path.is_symlink()
    if (
        not checkpoint_root.name.startswith("checkpoint-")
        or not checkpoint_root.name.removeprefix("checkpoint-").isdigit()
        or not _is_directory_nonsymlink(checkpoint_root)
        or not identity_present
    ):
        return None, identity_present

    adapter_root = checkpoint_root / "adapter"
    token_root = checkpoint_root / "special_token_embeddings"
    readme_path = adapter_root / "README.md"
    manifest_path = checkpoint_root / PAYLOAD_MANIFEST
    model_card_path = adapter_root / "model_card.json"
    if (
        manifest_path.exists()
        or manifest_path.is_symlink()
        or model_card_path.exists()
        or model_card_path.is_symlink()
        or not _is_regular_nonsymlink(identity_path)
        or not _is_directory_nonsymlink(adapter_root)
        or not _is_directory_nonsymlink(token_root)
    ):
        return None, True

    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if not isinstance(identity, dict) or set(identity) != LEGACY_IDENTITY_FILES:
            return None, True
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in identity.values()
        ):
            return None, True
        for relative_path in LEGACY_IDENTITY_FILES:
            if not _is_regular_nonsymlink(checkpoint_root / relative_path):
                return None, True
        if hashlib.sha256(readme_path.read_bytes()).hexdigest() != identity[
            "adapter/README.md"
        ]:
            return None, True
        return readme_path, True
    except (OSError, ValueError, TypeError):
        return None, True


def scan_output_root(root: Path) -> dict:
    root = Path(root).absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"output root is missing or not a directory: {root}")
    if root.is_symlink():
        raise ValueError(f"select the resolved output root explicitly: {root.resolve()}")
    findings = []
    files = 0
    links = 0
    allowed_readmes: set[Path] = set()

    def onerror(error):
        raise error

    for directory, directories, filenames in os.walk(root, followlinks=False, onerror=onerror):
        current = Path(directory)
        allowed_readme = None
        native_adapter_declared = False
        legacy_identity_present = False
        if PAYLOAD_MANIFEST in filenames:
            allowed_readme, native_adapter_declared = _native_adapter_readme_binding(current)
            if allowed_readme is not None:
                allowed_readmes.add(allowed_readme)
        if allowed_readme is None:
            allowed_readme, legacy_identity_present = (
                _legacy_identity_adapter_readme_binding(current)
            )
            if allowed_readme is not None:
                allowed_readmes.add(allowed_readme)
        for name in sorted(directories + filenames):
            path = current / name
            if path.is_symlink():
                links += 1
                if not path.exists():
                    findings.append({"path": str(path), "reason": "broken_symlink"})
                elif native_adapter_declared and path == current / "adapter":
                    findings.append(
                        {"path": str(path), "reason": "manifest_bound_adapter_symlink"}
                    )
                elif legacy_identity_present and path == current / "adapter":
                    findings.append(
                        {"path": str(path), "reason": "identity_bound_adapter_symlink"}
                    )
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                if path not in allowed_readmes:
                    findings.append({"path": str(path), "reason": "source_or_prose_in_outputs"})
            elif name in FORBIDDEN_DIRECTORIES:
                findings.append({"path": str(path), "reason": "runtime_or_checkout_in_outputs"})
            if name in filenames and not path.is_symlink():
                files += 1
    return {"root": str(root), "files": files, "symlinks_not_followed": links,
            "findings": findings, "passed": not findings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    reports = [scan_output_root(root) for root in args.root]
    for report in reports:
        report["finding_count"] = len(report["findings"])
        report["reasons"] = dict(Counter(item["reason"] for item in report["findings"]))
        if not args.details:
            report.pop("findings")
    print(json.dumps({"passed": all(item["passed"] for item in reports), "roots": reports}, indent=2))
    raise SystemExit(0 if all(item["passed"] for item in reports) else 1)


if __name__ == "__main__":
    main()
