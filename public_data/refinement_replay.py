"""Replay the retained deterministic COCO-refinement publisher serialization.

This is deliberately a narrow, offline publisher: authority over the selected
``working.norm.jsonl`` and its accompanying provenance stays with the caller.
It neither changes the source nor copies the shared May image store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReplayError(ValueError):
    """A retained working row cannot be replayed with the historical contract."""


@dataclass(frozen=True)
class ReplayResult:
    """Paths and inventory for one deterministic replay or verification."""

    norm_path: Path
    coord_path: Path
    row_count: int
    norm_sha256: str
    coord_sha256: str
    verified_existing: bool


def replay_refinement(
    *,
    working_norm: str | Path,
    destination: str | Path,
    image_root: str | Path,
    split: str,
    limit: int | None = None,
    verify_existing: bool = False,
) -> ReplayResult:
    """Write, or read-only verify, historical norm and coord JSONL siblings.

    The emitted filenames are ``{split}.norm.jsonl`` and ``{split}.coord.jsonl``.
    Normal replay refuses either pre-existing output.  Verification never makes
    a directory or writes a byte; it compares the retained serialization line
    by line against both existing siblings.
    """

    source = Path(working_norm).expanduser().resolve(strict=True)
    target_root = Path(destination).expanduser().resolve()
    shared_images = Path(image_root).expanduser().resolve(strict=True)
    _validate_options(source, target_root, shared_images, split, limit, verify_existing)
    norm_path = target_root / f"{split}.norm.jsonl"
    coord_path = target_root / f"{split}.coord.jsonl"

    if verify_existing:
        if not norm_path.is_file() or not coord_path.is_file():
            raise FileNotFoundError(
                "verification requires existing norm and coord outputs: "
                f"{norm_path}, {coord_path}"
            )
        row_count, norm_digest, coord_digest = _verify_existing(
            source=source,
            norm_path=norm_path,
            coord_path=coord_path,
            image_root=shared_images,
            target_root=target_root,
            split=split,
            limit=limit,
        )
        return ReplayResult(
            norm_path=norm_path,
            coord_path=coord_path,
            row_count=row_count,
            norm_sha256=norm_digest.hexdigest(),
            coord_sha256=coord_digest.hexdigest(),
            verified_existing=True,
        )

    if norm_path.exists() or coord_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing replay output; use --verify-existing "
            f"for read-only comparison: {norm_path}, {coord_path}"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    norm_temp, norm_handle = _temporary_output(norm_path)
    coord_temp, coord_handle = _temporary_output(coord_path)
    try:
        row_count, norm_digest, coord_digest = _write_candidates(
            source=source,
            norm_handle=norm_handle,
            coord_handle=coord_handle,
            image_root=shared_images,
            target_root=target_root,
            split=split,
            limit=limit,
        )
        norm_handle.close()
        coord_handle.close()
        os.replace(norm_temp, norm_path)
        os.replace(coord_temp, coord_path)
    finally:
        if not norm_handle.closed:
            norm_handle.close()
        if not coord_handle.closed:
            coord_handle.close()
        norm_temp.unlink(missing_ok=True)
        coord_temp.unlink(missing_ok=True)
    return ReplayResult(
        norm_path=norm_path,
        coord_path=coord_path,
        row_count=row_count,
        norm_sha256=norm_digest.hexdigest(),
        coord_sha256=coord_digest.hexdigest(),
        verified_existing=False,
    )


def _validate_options(
    source: Path,
    target_root: Path,
    image_root: Path,
    split: str,
    limit: int | None,
    verify_existing: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"working norm JSONL is not a file: {source}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"shared image root is not a directory: {image_root}")
    if split not in {"train", "val"}:
        raise ReplayError(f"split must be 'train' or 'val', got {split!r}")
    if isinstance(limit, bool) or (limit is not None and (not isinstance(limit, int) or limit <= 0)):
        raise ReplayError(f"limit must be a positive integer when given, got {limit!r}")
    if verify_existing and not target_root.is_dir():
        raise FileNotFoundError(f"verification destination is not a directory: {target_root}")


def _temporary_output(path: Path) -> tuple[Path, Any]:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(temporary), os.fdopen(descriptor, "wb")


def _write_candidates(
    *,
    source: Path,
    norm_handle: Any,
    coord_handle: Any,
    image_root: Path,
    target_root: Path,
    split: str,
    limit: int | None,
) -> tuple[int, Any, Any]:
    norm_digest = hashlib.sha256()
    coord_digest = hashlib.sha256()
    row_count = 0
    for norm_line, coord_line in _serialized_rows(
        source=source,
        image_root=image_root,
        target_root=target_root,
        split=split,
        limit=limit,
    ):
        norm_handle.write(norm_line)
        coord_handle.write(coord_line)
        norm_digest.update(norm_line)
        coord_digest.update(coord_line)
        row_count += 1
    norm_handle.flush()
    coord_handle.flush()
    os.fsync(norm_handle.fileno())
    os.fsync(coord_handle.fileno())
    return row_count, norm_digest, coord_digest


def _verify_existing(
    *,
    source: Path,
    norm_path: Path,
    coord_path: Path,
    image_root: Path,
    target_root: Path,
    split: str,
    limit: int | None,
) -> tuple[int, Any, Any]:
    norm_digest = hashlib.sha256()
    coord_digest = hashlib.sha256()
    row_count = 0
    with norm_path.open("rb") as norm_handle, coord_path.open("rb") as coord_handle:
        for row_count, (norm_line, coord_line) in enumerate(
            _serialized_rows(
                source=source,
                image_root=image_root,
                target_root=target_root,
                split=split,
                limit=limit,
            ),
            start=1,
        ):
            if norm_handle.readline() != norm_line:
                raise ReplayError(f"existing norm output does not match replay at row {row_count}")
            if coord_handle.readline() != coord_line:
                raise ReplayError(f"existing coord output does not match replay at row {row_count}")
            norm_digest.update(norm_line)
            coord_digest.update(coord_line)
        if norm_handle.read(1):
            raise ReplayError("existing norm output does not match replay: unexpected trailing bytes")
        if coord_handle.read(1):
            raise ReplayError("existing coord output does not match replay: unexpected trailing bytes")
    return row_count, norm_digest, coord_digest


def _serialized_rows(
    *,
    source: Path,
    image_root: Path,
    target_root: Path,
    split: str,
    limit: int | None,
) -> Iterator[tuple[bytes, bytes]]:
    emitted = 0
    with source.open("rb") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ReplayError(f"working JSONL contains a blank row at row {row_number}")
            try:
                payload = json.loads(raw_line, parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReplayError(f"working JSONL is not strict JSON at row {row_number}") from exc
            norm_payload = _historical_row(
                payload,
                image_root=image_root,
                target_root=target_root,
                split=split,
                row_number=row_number,
                coord_tokens=False,
            )
            coord_payload = _historical_row(
                payload,
                image_root=image_root,
                target_root=target_root,
                split=split,
                row_number=row_number,
                coord_tokens=True,
            )
            yield _jsonl_bytes(norm_payload), _jsonl_bytes(coord_payload)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _historical_row(
    payload: object,
    *,
    image_root: Path,
    target_root: Path,
    split: str,
    row_number: int,
    coord_tokens: bool,
) -> dict[str, Any]:
    row = _mapping(payload, field=f"working[{row_number}]")
    _exact_keys(
        row,
        {"file_name", "height", "image_id", "images", "metadata", "objects", "width"},
        field=f"working[{row_number}]",
    )
    image_id = _positive_int(row["image_id"], field=f"working[{row_number}].image_id")
    width = _positive_int(row["width"], field=f"working[{row_number}].width")
    height = _positive_int(row["height"], field=f"working[{row_number}].height")
    expected_locator = f"images/{split}2017/{image_id:012d}.jpg"
    if row["file_name"] != expected_locator or row["images"] != [expected_locator]:
        raise ReplayError(
            f"working row {row_number} does not use the historical {split} image locator"
        )
    metadata = _mapping(row["metadata"], field=f"working[{row_number}].metadata")
    if metadata.get("split") != split:
        raise ReplayError(f"working row {row_number} metadata split does not match {split!r}")
    raw_objects = row["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ReplayError(f"working row {row_number} objects must be a non-empty list")
    object_payloads = [
        _historical_object(value, field=f"working[{row_number}].objects[{index}]", coord_tokens=coord_tokens)
        for index, value in enumerate(raw_objects)
    ]
    ann_ids = [value["coco_ann_id"] for value in object_payloads]
    if len(set(ann_ids)) != len(ann_ids):
        raise ReplayError(f"working row {row_number} has duplicate coco_ann_id values")
    public_locator = os.path.relpath(
        image_root / f"{split}2017" / f"{image_id:012d}.jpg",
        start=target_root,
    )
    return {
        "images": [public_locator],
        "objects": object_payloads,
        "width": width,
        "height": height,
        "image_id": image_id,
        "file_name": expected_locator,
        "metadata": metadata,
    }


def _historical_object(value: object, *, field: str, coord_tokens: bool) -> dict[str, Any]:
    item = _mapping(value, field=field)
    required = {"bbox_2d", "desc", "category_id", "category_name", "coco_ann_id"}
    _exact_keys(item, required | {"metadata"}, required=required, field=field)
    bbox = item["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ReplayError(f"{field}.bbox_2d must be a four-item list")
    bins = [_coord_bin(number, field=f"{field}.bbox_2d[{index}]") for index, number in enumerate(bbox)]
    if bins[0] >= bins[2] or bins[1] >= bins[3]:
        raise ReplayError(f"{field}.bbox_2d must be non-degenerate xyxy bins")
    if not isinstance(item["desc"], str) or not isinstance(item["category_name"], str):
        raise ReplayError(f"{field} category text must be strings")
    category_id = _positive_int(item["category_id"], field=f"{field}.category_id")
    coco_ann_id = _nonzero_int(item["coco_ann_id"], field=f"{field}.coco_ann_id")
    result: dict[str, Any] = {
        "bbox_2d": [f"<|coord_{number}|>" for number in bins] if coord_tokens else bins,
        "desc": item["desc"],
        "category_id": category_id,
        "category_name": item["category_name"],
        "coco_ann_id": coco_ann_id,
    }
    if "metadata" in item:
        result["metadata"] = _mapping(item["metadata"], field=f"{field}.metadata")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ReplayError(f"{field} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    required: set[str] | None = None,
    field: str,
) -> None:
    required = allowed if required is None else required
    keys = set(value)
    if not required <= keys or not keys <= allowed:
        raise ReplayError(f"{field} has unexpected or missing fields")


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayError(f"{field} must be a positive integer")
    return value


def _nonzero_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ReplayError(f"{field} must be a non-zero integer")
    return value


def _coord_bin(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
        raise ReplayError(f"{field} must be an integer coordinate bin in [0, 999]")
    return value


def _jsonl_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _reject_json_constant(value: str) -> None:
    raise ReplayError(f"NaN and Infinity are not valid publication JSON: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-norm", required=True, type=Path, help="Authoritative ordered working.norm.jsonl.")
    parser.add_argument("--destination", required=True, type=Path, help="Explicit output directory for split norm/coord JSONL.")
    parser.add_argument("--image-root", required=True, type=Path, help="Existing shared image root; images are never copied.")
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--limit", type=int, help="Replay only the first N authoritative rows (for bounded tests).")
    parser.add_argument("--verify-existing", action="store_true", help="Read-only compare existing outputs instead of writing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = replay_refinement(**vars(args))
    print(
        json.dumps(
            {
                "coord_path": str(result.coord_path),
                "coord_sha256": result.coord_sha256,
                "norm_path": str(result.norm_path),
                "norm_sha256": result.norm_sha256,
                "row_count": result.row_count,
                "verified_existing": result.verified_existing,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
