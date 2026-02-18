from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_TOP_LEVEL_KEYS = ("images", "objects", "width", "height")


def _is_non_empty_desc(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _assert_geometry_shape(obj: dict, key: str, line_num: int, obj_idx: int) -> None:
    seq = obj.get(key)
    if not isinstance(seq, list):
        raise ValueError(
            f"line {line_num} obj[{obj_idx}] key '{key}' must be a list, got {type(seq).__name__}"
        )

    if key == "bbox_2d":
        if len(seq) != 4:
            raise ValueError(f"line {line_num} obj[{obj_idx}] bbox_2d must have len=4")
        return

    if key == "poly":
        if len(seq) < 6 or len(seq) % 2 != 0:
            raise ValueError(
                f"line {line_num} obj[{obj_idx}] poly must have even len >= 6, got {len(seq)}"
            )


def validate_record_structure(record: dict, line_num: int) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in record:
            raise ValueError(f"line {line_num} missing required key: {key}")

    objects = record.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"line {line_num} objects must be a list")

    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"line {line_num} obj[{obj_idx}] must be an object")

        if not _is_non_empty_desc(obj.get("desc")):
            raise ValueError(f"line {line_num} obj[{obj_idx}] desc must be a non-empty string")

        has_bbox = obj.get("bbox_2d") is not None
        has_poly = obj.get("poly") is not None
        if has_bbox == has_poly:
            raise ValueError(
                f"line {line_num} obj[{obj_idx}] must have exactly one geometry key (bbox_2d xor poly)"
            )

        if has_bbox:
            _assert_geometry_shape(obj, "bbox_2d", line_num, obj_idx)
        if has_poly:
            _assert_geometry_shape(obj, "poly", line_num, obj_idx)


def run_structural_preflight(jsonl_paths: Iterable[Path]) -> dict:
    stats = {"files": 0, "records": 0}
    for path in jsonl_paths:
        if not path.exists():
            continue
        stats["files"] += 1
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                validate_record_structure(rec, line_num)
                stats["records"] += 1
    return stats
