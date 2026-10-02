"""Check the small live-doc surface and read a closed historical snapshot.

No writes, model imports, network access or restoration are performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

SEAL = "manifests/documentation/retirement-20261003.json"
ROOT = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def safe_doc_path(value: str) -> bool:
    path = PurePosixPath(value)
    return value.startswith("docs/") and not path.is_absolute() and ".." not in path.parts


def declared_docs(root: Path) -> list[str]:
    """The current README is the only live inventory; the historical seal is frozen."""
    directory = root / "docs"
    paths = {"docs/README.md"}
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", (directory / "README.md").read_text()):
        if urlsplit(target).scheme or target.startswith("#"):
            continue
        dest = (directory / unquote(target.split("#", 1)[0])).resolve()
        if dest.is_relative_to(directory.resolve()):
            rel = dest.relative_to(root.resolve()).as_posix()
            if rel != "docs":
                paths.add(rel)
    forbidden = ("docs/history/", "docs/archive/", "docs/handoffs/", "docs/superpowers/")
    if any(p.startswith(forbidden) for p in paths):
        raise ValueError("retired documentation trees cannot be readmitted")
    return sorted(paths)


def load_seal(root: Path) -> dict:
    value = json.loads((root / SEAL).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("status") != "sealed":
        raise ValueError("unsupported documentation seal")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("source_commit", "")):
        raise ValueError("invalid historical commit")
    paths = declared_docs(root)
    value["live_docs"] = paths
    if not paths or len(paths) != len(set(paths)) or not all(safe_doc_path(p) for p in paths):
        raise ValueError("invalid live-doc inventory")
    return value


def check_docs(root: Path, seal: dict) -> list[str]:
    expected = set(seal["live_docs"])
    actual = {p.relative_to(root).as_posix() for p in (root / "docs").rglob("*")
              if p.is_file() or p.is_symlink()}
    errors = [f"unexpected doc: {p}" for p in sorted(actual - expected)]
    errors += [f"missing doc: {p}" for p in sorted(expected - actual)]
    for rel in sorted(actual & expected):
        path = root / rel
        if path.is_symlink():
            errors.append(f"document symlink: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().strip("<>")
            if not target or target.startswith("#") or urlsplit(target).scheme:
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            dest = (path.parent / target).resolve()
            if not dest.exists():
                errors.append(f"broken local link: {rel} -> {target}")
    # Runtime string literals must not use prose or retired docs as admission data.
    for path in (root / "src").rglob("*.py"):
        if path.is_symlink():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if re.search(r"(?:^|[ /])docs/[A-Za-z0-9_./-]+\.(?:md|yaml)", value):
                    errors.append(f"runtime documentation dependency: {path.relative_to(root)}:{node.lineno}")
    return errors


def read_extras(root: Path, seal: dict) -> dict[str, bytes]:
    spec = seal.get("sealed_extras")
    if spec is None:
        return {}
    rel = PurePosixPath(spec["path"])
    if rel.is_absolute() or ".." in rel.parts or not str(rel).startswith("manifests/documentation/"):
        raise ValueError("invalid sealed-extras location")
    path = root / rel
    if path.is_symlink():
        raise ValueError("sealed extras cannot be a symlink")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("sealed-extras file checksum mismatch")
    payload = json.loads(data)
    if payload.get("schema_version") != 1 or payload.get("status") != "sealed":
        raise ValueError("invalid sealed-extras schema")
    result = {}
    for row in payload["entries"]:
        name = row["path"]
        content = row["content_utf8"].encode("utf-8")
        if not safe_doc_path(name) or name in result:
            raise ValueError("invalid or duplicate original extra path")
        if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"original extra checksum mismatch: {name}")
        result[name] = content
    if len(result) != spec["count"] or len(result) != seal["untracked_file_count"]:
        raise ValueError("sealed-extras count mismatch")
    return result


def snapshot_files(root: Path, seal: dict) -> dict[str, str]:
    commit = seal["source_commit"]
    if git(root, "rev-parse", commit + ":docs").decode().strip() != seal["source_docs_tree"]:
        raise ValueError("historical docs tree mismatch")
    records = git(root, "ls-tree", "-r", "-z", commit, "--", "docs").split(b"\0")
    result = {}
    for record in records:
        if not record:
            continue
        meta, rawpath = record.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = rawpath.decode()
        if kind != "blob" or mode not in ("100644", "100755") or not safe_doc_path(name):
            raise ValueError(f"unsupported historical entry: {name}")
        result[name] = oid
    if len(result) != seal["git_file_count"]:
        raise ValueError("historical docs count mismatch")
    return result


def verify_recovery(root: Path, seal: dict) -> dict:
    files = snapshot_files(root, seal)
    # Read every blob once, checking Git's content identity without restoring files.
    objects = list(dict.fromkeys(files.values()))
    data = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=("\n".join(objects) + "\n").encode(), stderr=subprocess.PIPE,
    )
    offset = 0
    for oid in objects:
        end = data.index(b"\n", offset)
        actual, kind, size = data[offset:end].decode().split()
        size = int(size)
        body = data[end + 1:end + 1 + size]
        expected = hashlib.sha1(f"blob {size}\0".encode() + body).hexdigest()
        if actual != oid or kind != "blob" or expected != oid:
            raise ValueError(f"historical object mismatch: {oid}")
        offset = end + size + 2
    extras = read_extras(root, seal)
    if set(extras) & set(files):
        raise ValueError("extra shadows a Git original")
    retired_tests = seal.get("retired_tests", [])
    for record in retired_tests:
        name = record["path"]
        if not name.startswith("tests/") or ".." in PurePosixPath(name).parts:
            raise ValueError("invalid retired-test path")
        original = git(root, "show", record["source_commit"] + ":" + name)
        if hashlib.sha256(original).hexdigest() != record["sha256"]:
            raise ValueError(f"retired test checksum mismatch: {name}")
    return {"git_files": len(files), "extra_files": len(extras),
            "original_files": len(files) + len(extras),
            "retired_test_files": len(retired_tests), "source_commit": seal["source_commit"]}


def read_original(root: Path, seal: dict, name: str) -> bytes:
    if not safe_doc_path(name):
        raise ValueError("select an original docs/ path without traversal")
    extras = read_extras(root, seal)
    if name in extras:
        return extras[name]
    files = snapshot_files(root, seal)
    if name not in files:
        raise ValueError("path is absent from the closed snapshot")
    return git(root, "cat-file", "blob", files[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-recovery", action="store_true")
    parser.add_argument("--show-original", metavar="DOC_PATH")
    args = parser.parse_args()
    try:
        seal = load_seal(ROOT)
        if args.show_original:
            sys.stdout.buffer.write(read_original(ROOT, seal, args.show_original))
            return 0
        errors = check_docs(ROOT, seal)
        recovery = verify_recovery(ROOT, seal) if args.verify_recovery else None
        print(json.dumps({"ok": not errors, "errors": errors,
                          "live_docs": len(seal["live_docs"]), "recovery": recovery}, indent=2))
        return int(bool(errors))
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
