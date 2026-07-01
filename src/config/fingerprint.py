"""Stable fingerprint helpers for resolved configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.errors import ConfigContractError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except ValueError as exc:
        raise ConfigContractError(
            "JSON fingerprint payload contains non-finite values",
            code="config.fingerprint_non_finite",
            cause=exc,
        ) from exc
    return hashlib.sha256(encoded).hexdigest()
