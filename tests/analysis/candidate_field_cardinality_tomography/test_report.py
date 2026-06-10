from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.report import (
    build_probe_breakdowns,
    build_summary,
)


def test_summary_separates_validation_from_headline_eligibility() -> None:
    summary = build_summary(
        case_index_total=10,
        case_index_rows=20,
        planned=10,
        attempted=10,
        valid=2,
        assigned=2,
        control_status_by_type={"wrong_desc_same_image": "pass"},
        checkpoint_id="checkpoint-3664",
        checkpoint_path="/tmp/checkpoint-3664",
    )

    assert summary["validation_status"] == "ok"
    assert summary["checkpoint_id"] == "checkpoint-3664"
    assert summary["checkpoint_path"] == "/tmp/checkpoint-3664"
    assert summary["headline_eligibility_status"] == "ineligible_low_coverage"
    assert summary["denominator_counts"]["indexed_cases"] == 10
    assert summary["denominator_counts"]["indexed_gt_rows"] == 20


def test_probe_breakdowns_include_required_denominator_axes() -> None:
    x1_rows = [
        {
            "probe_status": "ok",
            "split": "train",
            "desc_text_canonical": "person",
            "same_desc_gt_count_annotated": 5,
            "pool_role": "headline_crowded",
            "fn_rescue_overlay_membership": False,
            "prefix_condition": "teacher_set_empty_prefix",
        },
        {
            "probe_status": "failed",
            "split": "val",
            "desc_text_canonical": "cat",
            "same_desc_gt_count_annotated": 1,
            "pool_role": "same_desc_count_1_control",
            "fn_rescue_overlay_membership": True,
            "prefix_condition": "teacher_set_empty_prefix",
        },
    ]
    taxonomy_rows = [
        {
            "split": "train",
            "desc_text_canonical": "person",
            "same_desc_gt_count_annotated": 5,
            "pool_role": "headline_crowded",
            "fn_rescue_overlay_membership": False,
            "prefix_condition": "teacher_set_empty_prefix",
            "primary_bucket": "A1_cardinality_collapse",
        }
    ]

    breakdowns = build_probe_breakdowns(x1_rows, taxonomy_rows)

    assert breakdowns["split"]["train"]["attempted"] == 1
    assert breakdowns["split"]["train"]["valid"] == 1
    assert breakdowns["split"]["val"]["attempted"] == 1
    assert breakdowns["split"]["val"]["valid"] == 0
    assert breakdowns["desc_text"]["person"]["taxonomy"] == 1
    assert breakdowns["same_desc_count_bucket"]["same_desc_4_5"]["attempted"] == 1
    assert breakdowns["same_desc_count_bucket"]["same_desc_1"]["attempted"] == 1
    assert breakdowns["pool_role"]["headline_crowded"]["valid"] == 1
    assert breakdowns["fn_rescue_overlay_membership"]["true"]["attempted"] == 1
    assert breakdowns["prefix_condition"]["teacher_set_empty_prefix"]["taxonomy"] == 1
    assert breakdowns["primary_bucket"]["A1_cardinality_collapse"]["taxonomy"] == 1
    assert breakdowns["primary_bucket"]["A1_cardinality_collapse"]["taxonomy_denominator"] == 1
    assert "attempted" not in breakdowns["primary_bucket"]["A1_cardinality_collapse"]
