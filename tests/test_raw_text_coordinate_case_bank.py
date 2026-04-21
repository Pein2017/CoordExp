from src.analysis.raw_text_coordinate_case_bank import (
    build_case_bank_rows,
    freeze_review_shortlist,
)


def test_build_case_bank_rows_emits_required_fields() -> None:
    duplicate_rows = [
        {
            "model_alias": "base_plus_adapter",
            "image_id": 11,
            "line_idx": 0,
            "record_idx": 0,
            "source_object_index": 2,
            "onset_object_index": 3,
            "selection_rank": 1,
            "serializer_surface": "pretty_inline",
        }
    ]
    fn_rows = [
        {
            "model_alias": "base_only",
            "image_id": 22,
            "line_idx": 1,
            "record_idx": 4,
            "gt_idx": 5,
            "selection_rank": 3,
            "serializer_surface": "model_native",
        }
    ]

    rows = build_case_bank_rows(
        duplicate_rows=duplicate_rows,
        fn_rows=fn_rows,
    )

    assert rows[0].bucket == "first_burst_onset"
    assert rows[0].case_uid == "base_plus_adapter:11:0:0:first_burst_onset:src2:on3"
    assert rows[1].bucket == "labeled_fn"
    assert rows[1].case_uid == "base_only:22:1:4:labeled_fn:gt5"
    assert rows[1].serializer_surface == "model_native"


def test_freeze_review_shortlist_respects_fp_and_fn_budgets() -> None:
    rows = build_case_bank_rows(
        duplicate_rows=[
            {
                "model_alias": "base_plus_adapter",
                "image_id": idx,
                "line_idx": 0,
                "record_idx": idx,
                "source_object_index": idx,
                "onset_object_index": idx + 1,
                "selection_rank": idx,
                "serializer_surface": "pretty_inline",
            }
            for idx in range(20)
        ],
        fn_rows=[
            {
                "model_alias": "base_only",
                "image_id": 100 + idx,
                "line_idx": 0,
                "record_idx": idx,
                "gt_idx": idx,
                "selection_rank": idx,
                "serializer_surface": "pretty_inline",
            }
            for idx in range(10)
        ],
    )

    shortlist = freeze_review_shortlist(rows, fp_budget=15, fn_budget=5)

    assert len(shortlist) == 20
    assert sum(row.review_bucket == "FP" for row in shortlist) == 15
    assert sum(row.review_bucket == "FN" for row in shortlist) == 5


def test_freeze_review_shortlist_budgets_unique_cases_not_serializer_rows() -> None:
    rows = build_case_bank_rows(
        duplicate_rows=[
            {
                "model_alias": "base_only",
                "image_id": 1,
                "line_idx": 0,
                "record_idx": 0,
                "source_object_index": 1,
                "onset_object_index": 2,
                "selection_rank": 1,
                "serializer_surface": "model_native",
            },
            {
                "model_alias": "base_only",
                "image_id": 1,
                "line_idx": 0,
                "record_idx": 0,
                "source_object_index": 1,
                "onset_object_index": 2,
                "selection_rank": 1,
                "serializer_surface": "pretty_inline",
            },
            {
                "model_alias": "base_plus_adapter",
                "image_id": 2,
                "line_idx": 0,
                "record_idx": 0,
                "source_object_index": 3,
                "onset_object_index": 4,
                "selection_rank": 1,
                "serializer_surface": "model_native",
            },
            {
                "model_alias": "base_plus_adapter",
                "image_id": 2,
                "line_idx": 0,
                "record_idx": 0,
                "source_object_index": 3,
                "onset_object_index": 4,
                "selection_rank": 1,
                "serializer_surface": "pretty_inline",
            },
        ],
        fn_rows=[
            {
                "model_alias": "base_only",
                "image_id": 10,
                "line_idx": 1,
                "record_idx": 1,
                "gt_idx": 7,
                "selection_rank": 1,
                "serializer_surface": "pretty_inline",
            },
            {
                "model_alias": "base_only",
                "image_id": 10,
                "line_idx": 1,
                "record_idx": 1,
                "gt_idx": 8,
                "selection_rank": 2,
                "serializer_surface": "pretty_inline",
            },
        ],
    )

    shortlist = freeze_review_shortlist(rows, fp_budget=2, fn_budget=2)

    assert len(shortlist) == 4
    assert [row.model_alias for row in shortlist[:2]] == [
        "base_only",
        "base_plus_adapter",
    ]
    assert [row.serializer_surface for row in shortlist[:2]] == [
        "model_native",
        "model_native",
    ]
    assert [row.gt_idx for row in shortlist[2:]] == [7, 8]
