"""Byte-compatible artifact identity primitives for training-set completion.

These helpers intentionally preserve this direction's existing JSON bytes:
UTF-8, ``ensure_ascii=False``, compact sorted keys, and one trailing newline.
They are not aliases for ``src.artifacts`` and do not define publication or
loading policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": file_hash(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def literal_binding(path: Path) -> dict[str, Any]:
    """Preserve the authored path, unlike the resolved-path ``binding`` above."""
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "size_bytes": path.stat().st_size,
    }


def write_pretty_json(path: Path, value: Any) -> None:
    """Historical overwrite writer: ASCII escaping, NaN allowed, one newline.

    This is byte compatibility, not the default for a newly frozen receipt.
    The caller still owns collision/immutability and directory creation.
    """
    path.write_text(json.dumps(value, indent=2) + "\n")


def ascii_json_digest(value: Any) -> str:
    """Historical sorted ASCII JSON without a terminal newline."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
