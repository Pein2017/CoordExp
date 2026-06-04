from __future__ import annotations

import hashlib
from typing import Sequence

from src.common.detection_compact_rows import BOX_START_TOKEN, OBJECT_REF_START_TOKEN, render_compact_row


def render_pre_x1_prefix(desc: str, previous_rows: Sequence[str] = ()) -> dict[str, str | int]:
    if previous_rows:
        return render_teacher_prefix_at_boundary(desc, previous_rows=previous_rows)
    return render_teacher_set_empty_prefix(desc)


def render_teacher_set_empty_prefix(desc: str) -> dict[str, str | int]:
    return _render_prefix_identity(
        desc,
        previous_rows=(),
        prefix_condition="teacher_set_empty_prefix",
    )


def render_teacher_prefix_at_boundary(
    desc: str,
    *,
    previous_rows: Sequence[str],
) -> dict[str, str | int]:
    return _render_prefix_identity(
        desc,
        previous_rows=previous_rows,
        prefix_condition="teacher_prefix_at_boundary",
    )


def render_self_rollout_prefix(
    desc: str,
    *,
    previous_rows: Sequence[str],
    self_subcondition: str,
) -> dict[str, str | int]:
    row = _render_prefix_identity(
        desc,
        previous_rows=previous_rows,
        prefix_condition="self_rollout_prefix",
    )
    row["self_prefix_subcondition"] = self_subcondition
    return row


def _render_prefix_identity(
    desc: str,
    *,
    previous_rows: Sequence[str],
    prefix_condition: str,
) -> dict[str, str | int]:
    prefix_text = "".join(previous_rows)
    prompt_text = f"{prefix_text}{OBJECT_REF_START_TOKEN}{desc}{BOX_START_TOKEN}"
    return {
        "prefix_condition": prefix_condition,
        "prefix_row_count": len(previous_rows),
        "prefix_text_sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "prompt_text": prompt_text,
        "prompt_template_id": "coco_80:compact_full:desc_first:xyxy",
        "object_field_order": "desc_first",
        "bbox_format": "xyxy",
        "coord_surface": "norm1000_coord_tokens",
        "normalization": "lower_strip_collapse_ws_v1",
    }


__all__ = [
    "render_compact_row",
    "render_pre_x1_prefix",
    "render_self_rollout_prefix",
    "render_teacher_prefix_at_boundary",
    "render_teacher_set_empty_prefix",
]
