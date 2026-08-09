"""Build the stable x-then-y view for the owner-interface unit.

The source admission panel is deliberately left untouched.  This command
creates a canonically serialized derived JSONL whose only semantic change is
the stable ordering of each row's objects by decoded ``(x1, y1,
source_index)``.  Its receipt proves canonical object-payload identity, arity,
multisets, image references, source bytes, and mappings; it does not claim the
whitespace or key ordering of an object substring is byte-preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-04-sorted-prospective-13-image-panel-admission/"
    "evaluation-inputs/human-refined-13.coord.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
DEFAULT_RECEIPT = DEFAULT_OUTPUT.with_suffix(".receipt.json")
UNIT_ID = "2026-08-05-static-dynamic-owner-interface-crossover"
COORD_RE = re.compile(r"^<\|coord_(?P<bin>[0-9]{1,3})\|>$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(obj: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(obj))


def _decoded_bbox(obj: dict[str, Any], *, row_index: int, source_index: int) -> tuple[int, ...]:
    raw = obj.get("bbox_2d")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(
            f"row {row_index} object {source_index} must have bbox_2d arity 4"
        )
    decoded: list[int] = []
    for slot, token in enumerate(raw):
        if not isinstance(token, str):
            raise ValueError(
                f"row {row_index} object {source_index} bbox_2d[{slot}] is not a token"
            )
        match = COORD_RE.fullmatch(token)
        if match is None or int(match.group("bin")) > 999:
            raise ValueError(
                f"row {row_index} object {source_index} bbox_2d[{slot}] is invalid"
            )
        decoded.append(int(match.group("bin")))
    return tuple(decoded)


def _image_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ("image_id", "file_name", "images", "width", "height")
        if key in row
    }


def _resolve_image_manifest(
    row: dict[str, Any],
    *,
    jsonl_parent: Path,
    row_index: int,
) -> list[dict[str, Any]]:
    references = row.get("images")
    if not isinstance(references, list) or not references:
        raise ValueError(f"row {row_index} must have a non-empty images list")
    manifest: list[dict[str, Any]] = []
    for image_index, declared in enumerate(references):
        if not isinstance(declared, str) or not declared:
            raise ValueError(
                f"row {row_index} images[{image_index}] must be a non-empty path"
            )
        declared_path = Path(declared).expanduser()
        candidate = (
            declared_path
            if declared_path.is_absolute()
            else jsonl_parent / declared_path
        )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(
                f"row {row_index} images[{image_index}] is not a regular file"
            )
        payload = resolved.read_bytes()
        manifest.append(
            {
                "image_index": image_index,
                "declared_path": declared,
                "resolved_path": str(resolved),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return manifest


def build_inputs(
    source_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    source_bytes = source.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError(
            "source panel hash mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )

    rows: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    source_object_hashes: list[str] = []
    derived_object_hashes: list[str] = []
    image_references_before: list[dict[str, Any]] = []
    image_references_after: list[dict[str, Any]] = []
    source_images_manifest: list[dict[str, Any]] = []
    total_objects = 0
    for row_index, line in enumerate(source_bytes.decode("utf-8").splitlines()):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("objects"), list):
            raise ValueError(f"row {row_index} must be an object with an objects list")
        objects = list(row["objects"])
        image_references_before.append(_image_reference(row))
        source_images_manifest.append(
            {
                "row_index": row_index,
                "image_id": row.get("image_id"),
                "images": _resolve_image_manifest(
                    row,
                    jsonl_parent=source.parent,
                    row_index=row_index,
                ),
            }
        )
        indexed: list[tuple[int, dict[str, Any], tuple[int, ...]]] = []
        for source_index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"row {row_index} object {source_index} is not an object")
            bbox = _decoded_bbox(obj, row_index=row_index, source_index=source_index)
            indexed.append((source_index, obj, bbox))
            source_object_hashes.append(_object_sha256(obj))
        indexed.sort(key=lambda item: (item[2][0], item[2][1], item[0]))
        derived_row = dict(row)
        derived_row["objects"] = [obj for _source_index, obj, _bbox in indexed]
        rows.append(derived_row)
        image_references_after.append(_image_reference(derived_row))
        row_mapping: list[dict[str, Any]] = []
        for derived_index, (source_index, obj, bbox) in enumerate(indexed):
            object_hash = _object_sha256(obj)
            derived_object_hashes.append(object_hash)
            row_mapping.append(
                {
                    "source_index": source_index,
                    "derived_index": derived_index,
                    "object_sha256": object_hash,
                    "decoded_xyxy_bins": list(bbox),
                }
            )
        mappings.append(
            {
                "row_index": row_index,
                "image_id": row.get("image_id"),
                "mapping": row_mapping,
            }
        )
        total_objects += len(indexed)

    if image_references_before != image_references_after:
        raise AssertionError("derived ordering changed an image reference")
    derived_bytes = b"".join(
        _canonical_bytes(row) + b"\n"
        for row in rows
    )
    mapping_sha256 = _sha256_bytes(_canonical_bytes(mappings))
    source_multiset_sha256 = _sha256_bytes(
        _canonical_bytes(sorted(source_object_hashes))
    )
    derived_multiset_sha256 = _sha256_bytes(
        _canonical_bytes(sorted(derived_object_hashes))
    )
    if source_multiset_sha256 != derived_multiset_sha256:
        raise AssertionError("derived object multiset differs from source")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    derived_images_manifest = [
        {
            "row_index": row_index,
            "image_id": row.get("image_id"),
            "images": _resolve_image_manifest(
                row,
                jsonl_parent=output.parent,
                row_index=row_index,
            ),
        }
        for row_index, row in enumerate(rows)
    ]
    if source_images_manifest != derived_images_manifest:
        raise ValueError(
            "image references do not resolve to identical files from derived location"
        )
    output.write_bytes(derived_bytes)
    result = {
        "schema_version": 1,
        "unit_id": UNIT_ID,
        "ordering": "geo_sorted_xy",
        "sort_key": ["decoded_x1", "decoded_y1", "source_index"],
        "source_path": str(source),
        "source_sha256": source_sha256,
        "derived_path": str(output),
        "derived_sha256": _sha256_bytes(derived_bytes),
        "receipt_path": str(receipt),
        "row_count": len(rows),
        "owner_count": total_objects,
        "source_owner_multiset_sha256": source_multiset_sha256,
        "derived_owner_multiset_sha256": derived_multiset_sha256,
        "object_payload_identity": "canonical_json_sha256",
        "source_coordinate_arity": 4,
        "coordinate_arity_verified": True,
        "image_references_preserved": True,
        "image_references_resolve_from_derived": True,
        "images_manifest": derived_images_manifest,
        "images_manifest_sha256": _sha256_bytes(
            _canonical_bytes(derived_images_manifest)
        ),
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
        "mapping_count": total_objects,
        "mapping_sha256": mapping_sha256,
        "image_references_sha256": _sha256_bytes(_canonical_bytes(image_references_after)),
        "source_to_derived": mappings,
    }
    receipt.write_bytes(_canonical_bytes(result) + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    result = build_inputs(
        args.source,
        args.output,
        args.receipt,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "source_to_derived"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
