"""Source256 annotation/prediction row projection for its TP50 reward."""

import math

from typing import Any

from src.data.geometry import coord_bins_to_pixel_xyxy

from src.eval.detection_categories import normalize_coco_category_name

def _category(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    return normalize_coco_category_name(
        obj.get("description", obj.get("desc", obj.get("label", obj.get("category", ""))))
    )

def _bbox(obj: Any) -> Any:
    return obj.get("bbox", obj.get("bbox_2d")) if isinstance(obj, dict) else None

def _pixel_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    return box

def _dimensions(row: dict[str, Any]) -> tuple[int, int] | None:
    width = row.get("image_width", row.get("width"))
    height = row.get("image_height", row.get("height"))
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    return width, height

def _gt_objects(row: dict[str, Any], *, row_id: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    objects = row.get("gt", [])
    if not isinstance(objects, list):
        raise ValueError(f"row {row_id!r} has malformed GT list")
    dimensions = _dimensions(row)
    if dimensions is None:
        raise ValueError(f"row {row_id!r} has invalid image dimensions")
    width, height = dimensions
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, obj in enumerate(objects):
        raw = _bbox(obj)
        try:
            box = coord_bins_to_pixel_xyxy(
                raw,
                image_width=width,
                image_height=height,
                field=f"gt[{index}].bbox",
            )
        except Exception as exc:
            raise ValueError(f"row {row_id!r} has invalid GT bbox at index {index}") from exc
        result.append((_category(obj), tuple(float(item) for item in box)))
    return result

def _pred_objects(row: dict[str, Any]) -> tuple[list[tuple[str, tuple[float, float, float, float]]], int]:
    objects = row.get("pred", [])
    if not isinstance(objects, list):
        return [], 1
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    invalid = 0
    for obj in objects:
        box = _pixel_box(_bbox(obj))
        if box is None or not _category(obj):
            invalid += 1
            continue
        result.append((_category(obj), box))
    return result, invalid
