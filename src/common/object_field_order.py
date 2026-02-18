from __future__ import annotations

from typing import Any, Dict, Literal

ObjectFieldOrder = Literal["desc_first", "geometry_first"]
ALLOWED_OBJECT_FIELD_ORDER: tuple[ObjectFieldOrder, ObjectFieldOrder] = (
    "desc_first",
    "geometry_first",
)


def normalize_object_field_order(
    value: Any, *, path: str = "custom.object_field_order"
) -> ObjectFieldOrder:
    normalized = str(value).strip().lower()
    if normalized not in ALLOWED_OBJECT_FIELD_ORDER:
        raise ValueError(
            f"{path} must be one of {{'desc_first', 'geometry_first'}}; got {value!r}"
        )
    return "geometry_first" if normalized == "geometry_first" else "desc_first"


def build_object_payload(
    *,
    desc: str,
    geometry_key: str,
    geometry_value: Any,
    object_field_order: ObjectFieldOrder,
) -> Dict[str, Any]:
    if geometry_key not in {"bbox_2d", "poly"}:
        raise ValueError(
            f"geometry_key must be 'bbox_2d' or 'poly', got {geometry_key!r}"
        )
    if object_field_order == "geometry_first":
        return {
            geometry_key: geometry_value,
            "desc": desc,
        }
    return {
        "desc": desc,
        geometry_key: geometry_value,
    }


__all__ = [
    "ObjectFieldOrder",
    "ALLOWED_OBJECT_FIELD_ORDER",
    "normalize_object_field_order",
    "build_object_payload",
]
