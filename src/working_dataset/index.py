"""Read-only authenticated task and working-row indexes.

The store owns reader locks and generation selection. Index construction never
publishes files or repairs the journal; it validates the selected bytes and returns
row positions and hashes for bounded navigation within that generation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .codec import (
    _is_sha256,
    _parse_working_jsonl_row,
    _strict_json_loads,
    _task_id,
    sha256_file,
    sha256_json,
)
from .records import (
    SCHEMA_VERSION,
    TASK_INDEX_IDENTITY_SCHEMA_VERSION,
    ManifestDriftError,
    RecoveryError,
    StaleCommitError,
)


@dataclass(frozen=True)
class RowIndex:
    images: dict[int, int]
    object_ids: dict[int, set[int]]
    hashes: dict[int, str]
    spans: dict[int, tuple[int, int]]
    signature: tuple[int, int, int, int, int]
    working_sha256: str

    def attest_publication(
        self,
        working_path: Path,
        manifest: Mapping[str, Any],
        *,
        required: bool,
    ) -> bool:
        if self.signature != _stat_signature(working_path.stat()):
            raise RecoveryError("working file changed while building row cache")
        attested = len(self.images) == int(
            manifest.get("working_line_count", -1)
        ) and self.working_sha256 == manifest.get("working_sha256")
        if required and not attested:
            raise RecoveryError("working input hash/line-count attestation failed")
        return attested


def load_task_index(
    task_index_path: Path, manifest: Mapping[str, Any]
) -> dict[int, int]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestDriftError("schema_version")
    expected_hash = manifest.get("task_index_sha256")
    if not isinstance(expected_hash, str) or not task_index_path.is_file():
        raise ManifestDriftError("task_index_sha256")
    if sha256_file(task_index_path) != expected_hash:
        raise ManifestDriftError("task_index_sha256")
    try:
        payload = _strict_json_loads(task_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestDriftError("task_index") from exc
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("split") != manifest.get("split")
        or not isinstance(payload.get("entries"), list)
    ):
        raise ManifestDriftError("task_index")
    identity_schema = payload.get("identity_schema_version")
    if identity_schema not in {None, TASK_INDEX_IDENTITY_SCHEMA_VERSION}:
        raise ManifestDriftError("task_index.identity_schema_version")
    entries = payload["entries"]
    if len(entries) != int(manifest.get("task_count", -1)):
        raise ManifestDriftError("task_index")
    image_to_index: dict[int, int] = {}
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ManifestDriftError("task_index")
        image_id = entry.get("image_id")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or entry.get("source_row_index") != expected_index
            or entry.get("source_line") != expected_index + 1
            or entry.get("task_id") != _task_id(str(manifest["split"]), image_id)
            or image_id in image_to_index
        ):
            raise ManifestDriftError("task_index")
        if identity_schema == TASK_INDEX_IDENTITY_SCHEMA_VERSION:
            expected_fields = {
                "source_row_index",
                "source_line",
                "image_id",
                "task_id",
                "file_name",
                "width",
                "height",
                "metadata",
                "task_row_fingerprint",
                "image_sha256",
            }
            if set(entry) != expected_fields:
                raise ManifestDriftError("task_index.identity")
            if (
                not isinstance(entry.get("file_name"), str)
                or isinstance(entry.get("width"), bool)
                or not isinstance(entry.get("width"), int)
                or entry["width"] <= 0
                or isinstance(entry.get("height"), bool)
                or not isinstance(entry.get("height"), int)
                or entry["height"] <= 0
                or not isinstance(entry.get("metadata"), Mapping)
                or not _is_sha256(entry.get("task_row_fingerprint"))
                or not _is_sha256(entry.get("image_sha256"))
            ):
                raise ManifestDriftError("task_index.identity")
        image_to_index[image_id] = expected_index
    return image_to_index


def scan_row_index(
    working_path: Path, source_row_by_image: Mapping[int, int]
) -> RowIndex:
    images: dict[int, int] = {}
    object_ids: dict[int, set[int]] = {}
    hashes: dict[int, str] = {}
    spans: dict[int, tuple[int, int]] = {}
    digest = hashlib.sha256()
    offset = 0
    with working_path.open("rb") as handle:
        before_signature = _stat_signature(os.fstat(handle.fileno()))
        for source_row_index, raw in enumerate(handle):
            digest.update(raw)
            spans[source_row_index] = (offset, len(raw))
            offset += len(raw)
            row = _parse_working_jsonl_row(raw, source_row_index)
            image_id = int(row.get("image_id", -1))
            if source_row_by_image.get(image_id) != source_row_index:
                raise RecoveryError("working source row index attestation failed")
            images[source_row_index] = image_id
            object_ids[source_row_index] = {
                int(obj["coco_ann_id"]) for obj in row["objects"]
            }
            hashes[source_row_index] = sha256_json(row)
        after_signature = _stat_signature(os.fstat(handle.fileno()))
    if before_signature != after_signature:
        raise RecoveryError("working file changed while building row cache")
    return RowIndex(
        images, object_ids, hashes, spans, after_signature, digest.hexdigest()
    )


def read_indexed_rows(
    working_path: Path,
    requested_image_ids: set[int],
    source_row_by_image: Mapping[int, int],
    index: RowIndex,
) -> dict[int, dict[str, Any]]:
    """Read only requested rows while reattesting the indexed file and row hashes."""

    expected_signature = index.signature
    rows: dict[int, dict[str, Any]] = {}
    with working_path.open("rb") as handle:
        if _stat_signature(os.fstat(handle.fileno())) != expected_signature:
            raise RecoveryError("working file changed before navigation read")
        for image_id in requested_image_ids:
            source_row_index = source_row_by_image.get(image_id)
            if source_row_index is None:
                raise StaleCommitError(f"unknown image_id: {image_id}")
            span = index.spans.get(source_row_index)
            if span is None:
                raise RecoveryError("navigation row is absent from attested index")
            offset, length = span
            handle.seek(offset)
            raw = handle.read(length)
            if len(raw) != length or not raw.endswith(b"\n"):
                raise RecoveryError("navigation row span is invalid")
            row = _parse_working_jsonl_row(raw, source_row_index)
            if (
                row.get("image_id") != image_id
                or index.images.get(source_row_index) != image_id
                or sha256_json(row) != index.hashes.get(source_row_index)
            ):
                raise RecoveryError("navigation row attestation failed")
            rows[image_id] = row
        if _stat_signature(os.fstat(handle.fileno())) != expected_signature:
            raise RecoveryError("working file changed during navigation read")
    if _stat_signature(working_path.stat()) != expected_signature:
        raise RecoveryError("working file changed during navigation read")
    return rows


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
