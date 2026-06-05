from __future__ import annotations

from src.analysis.post_x1_instance_basin_tomography.prefix_modes import (
    build_prefix_mode_rows,
    materialize_prefix_modes,
    render_compact_prefix_rows,
    render_forced_state_prompt,
)


OBJECTS = [
    {"gt_idx": 0, "desc": "person", "bbox_coord_token_xyxy": [100, 100, 200, 300]},
    {"gt_idx": 3, "desc": "chair", "bbox_coord_token_xyxy": [400, 500, 600, 700]},
]


def test_render_prefix_uses_no_newline_separator_for_pure_ce() -> None:
    text = render_compact_prefix_rows(OBJECTS, row_separator="none")

    assert "\n" not in text
    assert text.count("<|object_ref_start|>") == 2
    assert text == (
        "<|object_ref_start|>person<|box_start|>"
        "<|coord_100|><|coord_100|><|coord_200|><|coord_300|>"
        "<|object_ref_start|>chair<|box_start|>"
        "<|coord_400|><|coord_500|><|coord_600|><|coord_700|>"
    )


def test_render_prefix_uses_newline_separator_for_et_rmp() -> None:
    text = render_compact_prefix_rows(OBJECTS, row_separator="newline")

    assert text.count("\n") == 1
    assert text.splitlines()[0].endswith("<|coord_300|>")
    assert text.splitlines()[1].startswith("<|object_ref_start|>chair")


def test_forced_partial_uses_separator_between_completed_rows_and_forced_row_only() -> None:
    text = render_forced_state_prompt(
        prefix_objects=OBJECTS,
        desc="Person",
        forced_x1=123,
        forced_state="post_x1",
        row_separator="newline",
    )

    assert text.count("\n") == 2
    assert "<|box_start|><|coord_123|>" in text
    assert text.endswith("<|coord_123|>")


def test_prefix_modes_do_not_include_target_for_recall_modes() -> None:
    case = {
        "case_id": "case-1",
        "desc": "person",
        "target_gt_idx": 1,
        "same_desc_gt_indices": [0, 1, 2],
        "competitor_gt_indices": [0, 2],
        "objects": [
            {"gt_idx": 0, "desc": "person", "bbox_coord_token_xyxy": [10, 10, 50, 80]},
            {"gt_idx": 1, "desc": "person", "bbox_coord_token_xyxy": [100, 10, 150, 90]},
            {"gt_idx": 2, "desc": "person", "bbox_coord_token_xyxy": [200, 10, 250, 90]},
        ],
    }

    rows = build_prefix_mode_rows(
        case,
        modes=("same_desc_good_prefix", "same_desc_bad_prefix"),
    )

    assert {row["prefix_mode"] for row in rows} == {
        "same_desc_good_prefix",
        "same_desc_bad_prefix",
    }
    for row in rows:
        assert row["target_leak"] is False
        assert 1 not in row["prefix_gt_indices"]
        assert "prefix_text" not in row
        assert row["prefix_objects"]


def test_materialize_prefix_modes_summarizes_rollout_skip_policy() -> None:
    case = {
        "case_id": "case-1",
        "desc": "person",
        "target_gt_idx": 1,
        "same_desc_gt_indices": [0, 1],
        "competitor_gt_indices": [0],
        "objects": [
            {"gt_idx": 0, "desc": "person", "bbox_coord_token_xyxy": [10, 10, 50, 80]},
            {"gt_idx": 1, "desc": "person", "bbox_coord_token_xyxy": [100, 10, 150, 90]},
            {"gt_idx": 2, "desc": "chair", "bbox_coord_token_xyxy": [200, 10, 250, 90]},
        ],
    }

    rows, summary = materialize_prefix_modes(
        [case],
        modes=("empty", "rollout_good_prefix", "rollout_bad_prefix"),
        rollout_prefix_rows=None,
        rollout_prefix_missing_policy="skip_with_manifest",
    )

    assert [row["prefix_mode"] for row in rows] == ["empty"]
    assert summary["prefix_modes_requested"] == [
        "empty",
        "rollout_good_prefix",
        "rollout_bad_prefix",
    ]
    assert summary["prefix_modes_materialized"] == ["empty"]
    assert summary["prefix_modes_skipped"] == [
        "rollout_bad_prefix",
        "rollout_good_prefix",
    ]
    assert summary["rollout_prefix_missing_policy"] == "skip_with_manifest"
    assert summary["skipped_mode_reasons"] == {
        "rollout_bad_prefix": "rollout_prefix_source_missing",
        "rollout_good_prefix": "rollout_prefix_source_missing",
    }
