"""Deterministic identity for model snapshots consumed by inference backends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.errors import RuntimeContractError


SNAPSHOT_MANIFEST_VERSION = "coordexp-swift-model-snapshot-v1"


def build_model_snapshot_manifest(root: str | Path) -> dict[str, Any]:
    """Hash every regular file visible in a model snapshot.

    Model loaders may acquire behavior from files beyond weight/config names,
    including tokenizer assets, processor configs, and chat templates. An
    exhaustive manifest avoids maintaining an incomplete positive allowlist.
    """

    snapshot_root = Path(root).expanduser().resolve()
    if not snapshot_root.is_dir():
        raise RuntimeContractError(
            "model snapshot root is not a directory",
            code="inference.model_snapshot_missing",
            context={"root": str(snapshot_root)},
        )
    files = tuple(
        path
        for path in sorted(snapshot_root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )
    if not files:
        raise RuntimeContractError(
            "model snapshot contains no files",
            code="inference.model_snapshot_empty",
            context={"root": str(snapshot_root)},
        )
    identities = [
        {
            "relative_path": path.relative_to(snapshot_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    fingerprint = _sha256_json(
        {
            "version": SNAPSHOT_MANIFEST_VERSION,
            "files": identities,
        }
    )
    return {
        "version": SNAPSHOT_MANIFEST_VERSION,
        "root": str(snapshot_root),
        "file_count": len(identities),
        "files": identities,
        "fingerprint": fingerprint,
    }


def validate_model_snapshot_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Rebuild and compare a snapshot manifest without trusting its path alone."""

    root = manifest.get("root")
    expected_fingerprint = manifest.get("fingerprint")
    if not isinstance(root, str) or not root:
        raise RuntimeContractError(
            "model snapshot manifest is missing its root",
            code="inference.model_snapshot_manifest_invalid",
            context={"field": "root"},
        )
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise RuntimeContractError(
            "model snapshot manifest is missing its fingerprint",
            code="inference.model_snapshot_manifest_invalid",
            context={"field": "fingerprint"},
        )
    observed = build_model_snapshot_manifest(root)
    if observed["fingerprint"] != expected_fingerprint:
        raise RuntimeContractError(
            "model snapshot no longer matches its manifest",
            code="inference.model_snapshot_identity_mismatch",
            context={
                "root": root,
                "expected_fingerprint": expected_fingerprint,
                "observed_fingerprint": observed["fingerprint"],
            },
        )
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
