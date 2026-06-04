from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.prefixes import (
    render_pre_x1_prefix,
    render_self_rollout_prefix,
    render_teacher_prefix_at_boundary,
    render_teacher_set_empty_prefix,
)
from src.common.detection_compact_rows import BOX_START_TOKEN, OBJECT_REF_START_TOKEN


def test_teacher_set_empty_prefix_has_stable_identity_fields() -> None:
    prefix = render_teacher_set_empty_prefix("person")

    assert prefix["prefix_condition"] == "teacher_set_empty_prefix"
    assert prefix["prefix_row_count"] == 0
    assert prefix["prompt_text"] == f"{OBJECT_REF_START_TOKEN}person{BOX_START_TOKEN}"
    assert prefix["prompt_template_id"] == "coco_80:compact_full:desc_first:xyxy"
    assert prefix["object_field_order"] == "desc_first"
    assert prefix["bbox_format"] == "xyxy"
    assert prefix["coord_surface"] == "norm1000_coord_tokens"
    assert prefix["normalization"] == "lower_strip_collapse_ws_v1"


def test_teacher_boundary_prefix_hash_changes_with_previous_rows() -> None:
    first = render_teacher_prefix_at_boundary("person", previous_rows=("row-a",))
    second = render_teacher_prefix_at_boundary("person", previous_rows=("row-b",))

    assert first["prefix_condition"] == "teacher_prefix_at_boundary"
    assert first["prefix_row_count"] == 1
    assert first["prefix_text_sha256"] != second["prefix_text_sha256"]
    assert first["prompt_text_sha256"] != second["prompt_text_sha256"]


def test_desc_changes_prompt_hash_not_prefix_hash() -> None:
    person = render_teacher_prefix_at_boundary("person", previous_rows=("same-prefix",))
    bicycle = render_teacher_prefix_at_boundary("bicycle", previous_rows=("same-prefix",))

    assert person["prefix_text_sha256"] == bicycle["prefix_text_sha256"]
    assert person["prompt_text_sha256"] != bicycle["prompt_text_sha256"]


def test_self_rollout_prefix_records_clean_or_dirty_subcondition() -> None:
    clean = render_self_rollout_prefix("cat", previous_rows=("row-a",), self_subcondition="clean_self")
    dirty = render_self_rollout_prefix("cat", previous_rows=("row-a",), self_subcondition="dirty_self")

    assert clean["prefix_condition"] == "self_rollout_prefix"
    assert clean["self_prefix_subcondition"] == "clean_self"
    assert dirty["self_prefix_subcondition"] == "dirty_self"
    assert clean["prompt_text_sha256"] == dirty["prompt_text_sha256"]


def test_render_pre_x1_prefix_keeps_backward_compatible_empty_and_boundary_behavior() -> None:
    empty = render_pre_x1_prefix("person")
    boundary = render_pre_x1_prefix("person", previous_rows=("row-a",))

    assert empty["prefix_condition"] == "teacher_set_empty_prefix"
    assert boundary["prefix_condition"] == "teacher_prefix_at_boundary"
