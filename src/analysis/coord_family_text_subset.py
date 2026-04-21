from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.common.geometry.coord_utils import denorm_and_clamp


def convert_norm_row_to_text_pixel(row: Dict[str, Any]) -> Dict[str, Any]:
    width = int(row["width"])
    height = int(row["height"])
    row_out = dict(row)
    objects_out = []
    for obj in row.get("objects", []):
        obj_out = dict(obj)
        bbox = obj_out.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            obj_out["bbox_2d"] = denorm_and_clamp(
                bbox,
                width,
                height,
                coord_mode="norm1000",
            )
        objects_out.append(obj_out)
    row_out["objects"] = objects_out
    return row_out


def materialize_text_pixel_subset(
    src_jsonl: Path,
    dst_jsonl: Path,
) -> Dict[str, Any]:
    rows_out = []
    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows_out.append(convert_norm_row_to_text_pixel(json.loads(line)))

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dst_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out),
        encoding="utf-8",
    )

    summary = {
        "src_jsonl": str(src_jsonl),
        "dst_jsonl": str(dst_jsonl),
        "row_count": len(rows_out),
        "surface": "text_pixel_from_norm1000",
    }
    meta_path = dst_jsonl.with_suffix(dst_jsonl.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["convert_norm_row_to_text_pixel", "materialize_text_pixel_subset"]
