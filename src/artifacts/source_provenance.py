"""Immutable source capture outside the artifact namespace.

Callers retain their existing receipt schemas and record the returned path.
This module neither executes archived sources nor repairs historical bindings.
"""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile


SOURCE_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "reference/retained-sources/runs"


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


def _verify_capture(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"occupied source capture differs: {path}")


def preserve_source(source: Path, *, run_root: Path, relative_name: str | Path) -> Path:
    """Publish verified bytes once; run paths hard-link the read-only object.

    Capture paths are immutable by contract: never edit a returned path in place.
    Existing run/name bindings are accepted only when their bytes still match.
    """
    source = Path(source).resolve(strict=True)
    expected = _sha256(source)
    destination = source_snapshot_path(run_root, relative_name)
    if destination.exists() or destination.is_symlink():
        _verify_capture(destination, expected)
        return destination
    object_path = (SOURCE_ARCHIVE_ROOT.parent / "objects" / expected[:2]
                   / (expected + Path(relative_name).suffix))
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if not object_path.exists() and not object_path.is_symlink():
        with tempfile.NamedTemporaryFile(dir=object_path.parent) as output:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=8 << 20)
            output.flush()
            os.fsync(output.fileno())
            if _sha256(source) != expected or _sha256(Path(output.name)) != expected:
                raise ValueError("source changed during capture or copy verification failed")
            os.fchmod(output.fileno(), 0o444)
            try:
                os.link(output.name, object_path)
            except FileExistsError:
                pass  # Another capture may have published the same object.
    _verify_capture(object_path, expected)
    object_path.chmod(0o444)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(object_path, destination)
    except FileExistsError:
        pass
    _verify_capture(destination, expected)
    return destination
