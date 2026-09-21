"""Immutable source capture outside the artifact namespace.

Callers retain their existing receipt schemas and record the returned path.
This module neither executes archived sources nor repairs historical bindings.
"""

import hashlib
import os
from pathlib import Path
import shutil


SOURCE_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "docs/history/run-sources"


def source_snapshot_path(run_root: Path, relative_name: str | Path) -> Path:
    name = Path(relative_name)
    if name.is_absolute() or ".." in name.parts or name == Path("."):
        raise ValueError("source snapshot name must be a nonempty relative path")
    run_id = hashlib.sha256(str(Path(run_root).resolve()).encode()).hexdigest()
    return SOURCE_ARCHIVE_ROOT / run_id / name


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def preserve_source(source: Path, *, run_root: Path, relative_name: str | Path) -> Path:
    """Copy and verify exact bytes, accepting only an identical existing capture."""
    source = Path(source).resolve(strict=True)
    expected = _sha256(source)
    destination = source_snapshot_path(run_root, relative_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = destination.open("xb")
    except FileExistsError:
        if destination.is_symlink() or _sha256(destination) != expected:
            raise ValueError(f"occupied source capture differs: {destination}")
        return destination
    try:
        with output, source.open("rb") as handle:
            shutil.copyfileobj(handle, output, length=8 << 20)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(source) != expected or _sha256(destination) != expected:
            raise ValueError("source changed during capture or copy verification failed")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
