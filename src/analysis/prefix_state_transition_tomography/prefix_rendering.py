from __future__ import annotations

import re
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


_WS_RE = re.compile(r"\s+")
_COMPACT_ROWS_PATH = Path(__file__).resolve().parents[2] / "common" / "detection_compact_rows.py"
_COMPACT_ROWS_SPEC = importlib.util.spec_from_file_location(
    "_prefix_state_detection_compact_rows",
    _COMPACT_ROWS_PATH,
)
if _COMPACT_ROWS_SPEC is None or _COMPACT_ROWS_SPEC.loader is None:
    raise ImportError(f"cannot load compact row helpers from {_COMPACT_ROWS_PATH}")
_COMPACT_ROWS_MODULE = importlib.util.module_from_spec(_COMPACT_ROWS_SPEC)
sys.modules[_COMPACT_ROWS_SPEC.name] = _COMPACT_ROWS_MODULE
_COMPACT_ROWS_SPEC.loader.exec_module(_COMPACT_ROWS_MODULE)

BOX_START_TOKEN = str(_COMPACT_ROWS_MODULE.BOX_START_TOKEN)
render_compact_row = _COMPACT_ROWS_MODULE.render_compact_row


def canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def render_compact_object_row(desc: str, bbox_xyxy: Sequence[int]) -> str:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    bbox_tokens = tuple(_int_to_coord_token(int(value)) for value in bbox_xyxy)
    return render_compact_row(
        canonical_desc(desc),
        bbox_tokens,
        include_object_ref_marker=True,
        include_bbox_start_marker=True,
    )


def render_teacher_prefix(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(render_compact_object_row(_row_desc(row), _row_bbox(row)) for row in rows)


def render_boundary_assistant_text(prefix_rows: Sequence[Mapping[str, Any]]) -> str:
    return render_teacher_prefix(prefix_rows)


def render_forced_desc_pre_x1_assistant_text(
    prefix_rows: Sequence[Mapping[str, Any]],
    desc: str,
) -> str:
    return render_teacher_prefix(prefix_rows) + render_compact_row(
        canonical_desc(desc),
        (),
        include_object_ref_marker=True,
        include_bbox_start_marker=True,
    )


def image_local_desc_groups(objects: Sequence[Mapping[str, Any]]) -> list[str]:
    descs = {
        canonical_desc(str(obj.get("desc") or obj.get("desc_text") or obj.get("desc_text_canonical") or ""))
        for obj in objects
    }
    return sorted(desc for desc in descs if desc)


def _row_desc(row: Mapping[str, Any]) -> str:
    desc = row.get("desc") or row.get("desc_text") or row.get("desc_text_canonical")
    if desc is None:
        raise ValueError("prefix row is missing desc")
    return str(desc)


def _row_bbox(row: Mapping[str, Any]) -> Sequence[int]:
    bbox = row.get("bbox_xyxy")
    if bbox is None:
        raise ValueError("prefix row is missing bbox_xyxy")
    if not isinstance(bbox, Sequence) or isinstance(bbox, str):
        raise ValueError("bbox_xyxy must be a sequence")
    return [int(value) for value in bbox]


def _int_to_coord_token(value: int) -> str:
    if value < 0 or value > 999:
        raise ValueError(f"coordinate token value out of range: {value}")
    return f"<|coord_{value}|>"


__all__ = [
    "BOX_START_TOKEN",
    "canonical_desc",
    "image_local_desc_groups",
    "render_boundary_assistant_text",
    "render_compact_object_row",
    "render_forced_desc_pre_x1_assistant_text",
    "render_teacher_prefix",
]
