from __future__ import annotations

from src.analysis.prefix_state_transition_tomography.prefix_rendering import (
    BOX_START_TOKEN,
    image_local_desc_groups,
    render_boundary_assistant_text,
    render_compact_object_row,
    render_forced_desc_pre_x1_assistant_text,
    render_teacher_prefix,
)


def test_render_compact_object_row_uses_marker_delimited_coord_tokens() -> None:
    text = render_compact_object_row(" Person ", [7, 20, 30, 40])

    assert text == "<|object_ref_start|>person<|box_start|><|coord_7|><|coord_20|><|coord_30|><|coord_40|>"
    assert "\n" not in text
    assert "<|coord_007|>" not in text


def test_teacher_prefix_concatenates_completed_rows_without_newline() -> None:
    rows = [
        {"desc": "person", "bbox_xyxy": [10, 20, 30, 40]},
        {"desc": "traffic  light", "bbox_xyxy": [50, 60, 70, 80]},
    ]

    text = render_teacher_prefix(rows)

    assert "\n" not in text
    assert text.count("<|object_ref_start|>") == 2
    assert text.count("<|box_start|>") == 2


def test_boundary_and_forced_desc_prefix_shapes() -> None:
    rows = [{"desc": "person", "bbox_xyxy": [10, 20, 30, 40]}]

    boundary = render_boundary_assistant_text(rows)
    forced = render_forced_desc_pre_x1_assistant_text(rows, "chair")

    assert boundary.endswith("<|coord_40|>")
    assert forced.endswith(BOX_START_TOKEN)
    assert forced.count("<|object_ref_start|>") == 2
    assert "<|im_end|>" not in forced


def test_image_local_desc_groups_are_canonical_and_image_local() -> None:
    objects = [
        {"desc": " Traffic   LIGHT "},
        {"desc_text": "traffic light"},
        {"desc_text_canonical": "Person"},
        {"desc": ""},
    ]

    assert image_local_desc_groups(objects) == ["person", "traffic light"]

