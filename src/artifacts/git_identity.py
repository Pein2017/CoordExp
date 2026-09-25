"""Clean Git source identities, never historical-source recovery.

The caller supplies the checkout and required source closure, not the receipt.
This binds repository bytes only; data, environment and numerical qualification
remain separate. Validation performs no writes and never repairs an old receipt.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "coordexp.clean_git_source.v1"
_HEX = re.compile(r"[0-9a-f]{40}\Z")


class SourceIdentityError(ValueError):
    """Source is historical/unsupported for continuation."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fail(reason: str) -> None:
    raise SourceIdentityError(f"historical/unsupported for continuation: {reason}")


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True, check=False, timeout=30,
    )
    if result.returncode:
        _fail("Git identity cannot be verified")
    return result.stdout


def _local_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\r\n\\"):
        _fail("invalid source path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts or str(relative) != value:
        _fail("source path must be normalized and checkout-relative")
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("symlink source is not a regular committed file")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
        _fail("required source is missing")
    return candidate


def _clean_state(root: Path) -> tuple[str, str]:
    observed_root = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if observed_root != root:
        _fail("caller did not select the exact checkout root")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        _fail("checkout is dirty; qualify a new clean commit")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    if not _HEX.fullmatch(commit) or not _HEX.fullmatch(tree):
        _fail("unsupported Git object identity")
    return commit, tree


def capture_source_identity(
    paths: Sequence[str], *, root: Path | None = None,
) -> dict[str, Any]:
    """Bind an explicit source closure to a clean commit and exact working bytes."""
    checkout = (repository_root() if root is None else Path(root)).resolve(strict=True)
    if isinstance(paths, (str, bytes)) or not paths or any(not isinstance(p, str) for p in paths):
        _fail("a nonempty explicit source list is required")
    if len(set(paths)) != len(paths):
        _fail("duplicate source paths")
    commit, tree = _clean_state(checkout)
    files = []
    for name in sorted(paths):
        file = _local_file(checkout, name)
        entry = _git(checkout, "ls-tree", "-z", commit, "--", name)
        records = [row for row in entry.split(b"\0") if row]
        if len(records) != 1:
            _fail("source is not tracked in the bound commit")
        metadata, stored_path = records[0].split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"} or stored_path.decode() != name:
            _fail("source is not a regular committed blob")
        original = _git(checkout, "cat-file", "blob", blob)
        current = file.read_bytes()
        if current != original:
            _fail("source bytes differ from the committed blob")
        files.append({"path": name, "blob": blob, "sha256": hashlib.sha256(current).hexdigest(),
                      "size_bytes": len(current)})
    if _clean_state(checkout) != (commit, tree):
        _fail("checkout changed while binding sources")
    for row in files:
        if hashlib.sha256(_local_file(checkout, row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            _fail("source changed while binding sources")
    return {"schema": SCHEMA, "commit": commit, "tree": tree, "files": files}


def verify_source_identity(
    identity: Mapping[str, Any], *, required_paths: Sequence[str], root: Path | None = None,
) -> None:
    """Reject unsupported envelopes before any continuation computation or write."""
    if not isinstance(identity, Mapping) or set(identity) != {"schema", "commit", "tree", "files"}:
        _fail("missing current source identity")
    if identity["schema"] != SCHEMA or not isinstance(identity["files"], list):
        _fail("legacy source identity schema")
    if any(not isinstance(identity.get(k), str) or not _HEX.fullmatch(identity[k]) for k in ("commit", "tree")):
        _fail("malformed commit or tree")
    rows = identity["files"]
    if not rows or any(not isinstance(row, Mapping) or set(row) != {"path", "blob", "sha256", "size_bytes"} for row in rows):
        _fail("malformed source inventory")
    paths = [row["path"] for row in rows]
    if isinstance(required_paths, (str, bytes)) or not required_paths or any(not isinstance(p, str) for p in paths):
        _fail("required source closure is absent")
    if not set(required_paths).issubset(paths):
        _fail("receipt does not cover the required source closure")
    actual = capture_source_identity(paths, root=root)
    if dict(identity) != actual:
        _fail("commit, tree, path or source identity changed")
