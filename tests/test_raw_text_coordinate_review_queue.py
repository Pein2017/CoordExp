from src.analysis.raw_text_coordinate_review_queue import build_review_queue_rows


def test_build_review_queue_rows_exposes_notion_friendly_columns() -> None:
    rows = build_review_queue_rows(
        shortlist=[
            {
                "case_uid": "base_only:1:0:0:first_burst_onset",
                "review_bucket": "FP",
                "model_alias": "base_only",
                "selection_rank": 1,
            }
        ]
    )

    assert rows[0]["case_uid"] == "base_only:1:0:0:first_burst_onset"
    assert rows[0]["bucket"] == "FP"
    assert rows[0]["status"] == "unreviewed"
    assert rows[0]["bbox_judgment"] == ""
