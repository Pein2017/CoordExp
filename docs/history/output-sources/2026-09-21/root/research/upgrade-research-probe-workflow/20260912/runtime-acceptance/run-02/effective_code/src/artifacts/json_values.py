"""Strict JSON values and crash-consistent exclusive publication helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from src.common.errors import ArtifactContractError


class FileSystemOps(Protocol):
    """Small injection seam for the durability steps of a file publication."""

    def fsync_file(self, file_descriptor: int) -> None: ...

    def publish_exclusive(self, temporary_path: Path, final_path: Path) -> None: ...

    def fsync_directory(self, directory: Path) -> None: ...


class LocalFileSystemOps:
    """POSIX implementation of the journal publication sequence."""

    def fsync_file(self, file_descriptor: int) -> None:
        os.fsync(file_descriptor)

    def publish_exclusive(self, temporary_path: Path, final_path: Path) -> None:
        # link(2) creates the final name only when it does not already exist.
        # Both paths are in the same directory, so this is an atomic publication.
        os.link(temporary_path, final_path)

    def fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


DEFAULT_FILESYSTEM_OPS = LocalFileSystemOps()


def validate_json_value(value: Any) -> None:
    """Reject values outside the explicit recursive JSON value algebra."""

    _validate_json_value(value, path="$", seen=set())


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 representation for a strict value."""

    validate_json_value(value)
    return json.dumps(
        _normalize_json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_text(value: Any, *, compact: bool = False) -> str:
    """Validate first, then render JSON in the requested presentation format."""

    validate_json_value(value)
    return json.dumps(
        _normalize_json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    """Fingerprint canonical bytes, never a caller's presentation formatting."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_canonical_json(path: Path) -> Any:
    """Load one canonical strict JSON file, rejecting altered presentation bytes."""

    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(
            "artifact file is not readable strict JSON",
            code="artifact.invalid_json_file",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    try:
        canonical = canonical_json_bytes(value)
    except ArtifactContractError as exc:
        raise ArtifactContractError(
            "artifact file contains a non-strict JSON value",
            code="artifact.invalid_json_file",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if encoded != canonical:
        raise ArtifactContractError(
            "artifact file is not canonically encoded",
            code="artifact.noncanonical_json_file",
            context={"path": str(path)},
        )
    return value


def publish_json_exclusive(
    path: Path,
    value: Any,
    *,
    filesystem: FileSystemOps = DEFAULT_FILESYSTEM_OPS,
) -> Path:
    """Durably publish canonical bytes at a final path that cannot be replaced."""

    encoded = canonical_json_bytes(value)
    if path.exists():
        raise ArtifactContractError(
            "artifact final path already exists",
            code="artifact.path_already_exists",
            context={"path": str(path)},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            filesystem.fsync_file(handle.fileno())
        filesystem.publish_exclusive(temporary_path, path)
        published = True
        temporary_path.unlink()
        filesystem.fsync_directory(path.parent)
    except FileExistsError as exc:
        raise ArtifactContractError(
            "artifact final path already exists",
            code="artifact.path_already_exists",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    except OSError as exc:
        raise ArtifactContractError(
            "artifact publication did not complete",
            code="artifact.publish_failed",
            context={"path": str(path), "published_before_failure": published},
            cause=exc,
        ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def _validate_json_value(value: Any, *, path: str, seen: set[int]) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        _raise_invalid_value(path, "non-finite float")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            _raise_invalid_value(path, "cyclic mapping")
        seen.add(identity)
        try:
            for key, nested in value.items():
                if not isinstance(key, str):
                    _raise_invalid_value(path, "mapping key is not a string")
                _validate_json_value(nested, path=f"{path}.{key}", seen=seen)
        finally:
            seen.remove(identity)
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            _raise_invalid_value(path, "cyclic list")
        seen.add(identity)
        try:
            for index, nested in enumerate(value):
                _validate_json_value(nested, path=f"{path}[{index}]", seen=seen)
        finally:
            seen.remove(identity)
        return
    _raise_invalid_value(path, type(value).__name__)


def _raise_invalid_value(path: str, value_type: str) -> None:
    raise ArtifactContractError(
        "artifact value is not recursively strict JSON",
        code="artifact.invalid_json_value",
        context={"path": path, "value_type": value_type},
    )


def _normalize_json_value(value: Any) -> Any:
    """Materialize accepted abstract mappings as built-in JSON containers."""

    if isinstance(value, Mapping):
        return {key: _normalize_json_value(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(nested) for nested in value]
    return value
