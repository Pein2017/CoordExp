from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.case_index import (
    build_case_index,
    canonical_desc,
)


def test_canonical_desc_policy_lower_strip_collapse_ws() -> None:
    assert canonical_desc(" Traffic   LIGHT ") == "traffic light"


def test_case_index_materializes_headline_and_control_pool_roles(tiny_coord_jsonl) -> None:
    rows, summary = build_case_index(
        train_jsonl=tiny_coord_jsonl,
        val_jsonl=tiny_coord_jsonl,
        checkpoint_id="checkpoint-3664",
        run_id="run-test",
    )

    headline = [row for row in rows if row["pool_role"] == "headline_crowded"]
    count1 = [row for row in rows if row["pool_role"] == "same_desc_count_1_control"]
    count2 = [row for row in rows if row["pool_role"] == "same_desc_count_2_control"]

    assert {row["desc_text_canonical"] for row in headline} == {"person"}
    assert {row["desc_text_canonical"] for row in count2} == {"traffic light"}
    assert {row["desc_text_canonical"] for row in count1} == {"cat"}
    assert all(row["case_id"] and row["case_index_row_id"] for row in rows)
    assert summary["pool_role_counts"]["headline_crowded"] == 2
    assert summary["pool_role_counts"]["same_desc_count_2_control"] == 2
