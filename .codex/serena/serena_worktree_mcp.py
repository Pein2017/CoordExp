#!/usr/bin/env python3
"""Run one official Serena backend per Git worktree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class RuntimeFailure(RuntimeError):
    """Raised when the shared Serena runtime cannot satisfy its contract."""


@dataclass(frozen=True)
class Slot:
    root: Path
    key: str
    path: Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    executable: str
    argv: tuple[str, ...]
    project_root: Path


def resolve_worktree(start: Path) -> Path:
    """Return the canonical Git worktree root containing *start*."""
    result = subprocess.run(
        ["/usr/bin/git", "-C", os.fspath(start), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "not inside a Git worktree"
        raise RuntimeFailure(f"cannot resolve Git worktree from {start}: {detail}")
    root = Path(result.stdout.strip())
    if not root.is_absolute():
        raise RuntimeFailure(f"Git returned a non-absolute worktree root: {root}")
    return root.resolve(strict=True)


def slot_for(root: Path, runtime_base: Path) -> Slot:
    """Map a canonical worktree root to a collision-detectable runtime slot."""
    canonical_root = root.resolve(strict=True)
    key = hashlib.sha256(os.fsencode(canonical_root)).hexdigest()[:24]
    return Slot(root=canonical_root, key=key, path=runtime_base / key)


def read_process_identity(pid: int) -> ProcessIdentity | None:
    """Read a Serena-compatible process identity from Linux procfs."""
    if pid <= 0:
        return None
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        command_end = stat_text.rfind(")")
        if command_end < 0:
            return None
        stat_fields = stat_text[command_end + 2 :].split()
        start_ticks = int(stat_fields[19])
        executable = os.path.realpath(os.readlink(proc / "exe"))
        argv = tuple(
            os.fsdecode(part)
            for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
        return None

    project_indexes = [index for index, value in enumerate(argv) if value == "--project"]
    if len(project_indexes) != 1:
        return None
    project_index = project_indexes[0]
    if project_index + 1 >= len(argv):
        return None
    try:
        project_root = Path(argv[project_index + 1]).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        executable=executable,
        argv=argv,
        project_root=project_root,
    )


def identity_matches(expected: ProcessIdentity) -> bool:
    """Return whether the exact process identity is still alive."""
    return read_process_identity(expected.pid) == expected


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Durably replace a small private JSON metadata file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(temporary_fd, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(temporary_fd, encoded[offset:])
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
