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
