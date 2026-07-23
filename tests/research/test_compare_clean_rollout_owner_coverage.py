from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.research.compare_clean_rollout_owner_coverage import _global_matches, compare_artifacts, main


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(row_id: str, pred: list[dict], **extra: object) -> dict:
    return {
        "row_id": row_id,
        "image_width": 100,
        "image_height": 200,
        "gt": [
            {"description": "person", "bbox": [0, 0, 500, 500]},
            {"description": "person", "bbox": [500, 0, 999, 500]},
        ],
        "pred": pred,
        **extra,
    }


def test_dense_same_category_matching_duplicates_and_norm1000_conversion(tmp_path: Path) -> None:
    arm_a = tmp_path / "a.jsonl"
    arm_b = tmp_path / "b.jsonl"
    a_rows = [
        _row(
            "row-1",
            [
                {"description": "person", "bbox": [0, 0, 50, 100]},
                {"description": "person", "bbox": [1, 1, 49, 99]},
                {"description": "person", "bbox": [50, 0, 100, 100]},
                {"description": "person", "bbox": [25, 0, 75, 100]},
                {"description": "person", "bbox": [10, 10, 10, 20]},
            ],
            parse_status="accepted_with_drops",
            dropped_prediction_count=1,
        ),
        _row("row-2", [], parse_status="malformed", dropped_predictions=[{"reason": "bad"}]),
    ]
    b_rows = [
        _row(
            "row-1",
            [
                {"description": "person", "bbox": [0, 0, 50, 100]},
                {"description": "person", "bbox": [50, 0, 100, 100]},
            ],
        ),
        _row("row-2", [], parse_status="empty"),
    ]
    _write(arm_a, a_rows)
    _write(arm_b, b_rows)

    result = compare_artifacts(arm_a, arm_b)
    assert result["inputs"]["arm_a"]["path"] == str(arm_a.resolve())
    assert len(result["inputs"]["arm_a"]["sha256"]) == 64
    assert result["arm_a"]["gt_count"] == 4
    assert result["arm_a"]["unique_matched_gt_owners"] == 2
    assert result["arm_a"]["owner_coverage"] == 0.5
    assert result["arm_a"]["duplicate_candidate_count"] == 1
    assert result["arm_a"]["strict_physical_owner_duplicate_candidate_count"] == 1
    assert result["arm_a"]["strict_physical_owner_duplicate_candidates"] == [
        {
            "row_id": "row-1",
            "owner_index": 0,
            "prediction_index": 1,
            "earlier_prediction_index": 0,
            "annotation_iou": pytest.approx(0.9408),
            "earlier_annotation_iou": 1.0,
            "prediction_to_prediction_iou": pytest.approx(0.9408),
        }
    ]
    assert result["arm_a"]["strict_physical_owner_ambiguous_attribution_count"] == 0
    assert result["arm_a"]["prediction_count"] == 4
    assert result["arm_a"]["invalid_prediction_count"] == 1
    assert result["arm_a"]["malformed_row_count"] == 1
    assert result["arm_a"]["dropped_row_count"] == 2
    assert result["arm_a"]["ambiguous_duplicate_candidate_count"] == 1
    assert result["arm_b"]["owner_coverage"] == 0.5
    assert result["arm_b"]["duplicate_candidate_count"] == 0
    assert result["paired_deltas"]["duplicate_candidate_count"] == -1
    assert result["arm_b"]["matched_iou"]["mean"] == 1.0
    assert result["common_owner_geometry"]["owner_count"] == 2

    stricter_prediction_overlap = compare_artifacts(
        arm_a,
        arm_b,
        strict_prediction_iou_threshold=0.95,
    )
    assert stricter_prediction_overlap["arm_a"]["duplicate_candidate_count"] == 1
    assert stricter_prediction_overlap["arm_a"]["strict_physical_owner_duplicate_candidate_count"] == 0


def test_include_row_ids_filters_repeatably_and_emits_reviewable_owner_changes(tmp_path: Path) -> None:
    arm_a = tmp_path / "a.jsonl"
    arm_b = tmp_path / "b.jsonl"
    rows_a = [
        _row("enrolled", [{"description": "person", "bbox": [0, 0, 50, 100]}]),
        _row("never-exposed", [{"description": "person", "bbox": [0, 0, 50, 100]}]),
    ]
    rows_b = [
        _row("enrolled", [{"description": "person", "bbox": [0, 0, 50, 100]}]),
        _row("never-exposed", [{"description": "person", "bbox": [50, 0, 100, 100]}]),
    ]
    _write(arm_a, rows_a)
    _write(arm_b, rows_b)

    enrolled = compare_artifacts(arm_a, arm_b, include_row_ids=["enrolled"])
    assert enrolled["inputs"]["cohort"] == {
        "selection": "explicit_include_row_ids",
        "included_row_ids": ["enrolled"],
    }
    assert enrolled["arm_a"]["row_count"] == 1
    assert enrolled["common_owner_geometry"]["arm_a_only_owner_refs"] == []
    assert enrolled["common_owner_geometry"]["arm_b_only_owner_refs"] == []

    never_exposed = compare_artifacts(arm_a, arm_b, include_row_ids=["never-exposed"])
    assert never_exposed["common_owner_geometry"]["arm_a_only_owner_refs"] == [
        {"row_id": "never-exposed", "owner_index": 0}
    ]
    assert never_exposed["common_owner_geometry"]["arm_b_only_owner_refs"] == [
        {"row_id": "never-exposed", "owner_index": 1}
    ]

    with pytest.raises(ValueError, match="absent"):
        compare_artifacts(arm_a, arm_b, include_row_ids=["missing"])


def test_cli_combines_repeated_row_ids_and_row_id_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm_a = tmp_path / "a.jsonl"
    arm_b = tmp_path / "b.jsonl"
    _write(arm_a, [_row("row-a", []), _row("row-b", [])])
    _write(arm_b, [_row("row-a", []), _row("row-b", [])])
    include_file = tmp_path / "cohort.txt"
    include_file.write_text("row-b\n\n", encoding="utf-8")
    out = tmp_path / "result.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_clean_rollout_owner_coverage.py",
            str(arm_a),
            str(arm_b),
            "--out",
            str(out),
            "--include-row-id",
            "row-a",
            "--include-row-id-file",
            str(include_file),
        ],
    )
    main()
    assert json.loads(out.read_text(encoding="utf-8"))["inputs"]["cohort"] == {
        "selection": "explicit_include_row_ids",
        "included_row_ids": ["row-a", "row-b"],
    }


def test_row_set_and_gt_mismatch_fail_fast(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    base = _row("row-1", [])
    _write(a, [base])
    _write(b, [_row("row-2", [])])
    with pytest.raises(ValueError, match="row sets"):
        compare_artifacts(a, b)

    changed = _row("row-1", [])
    changed["gt"][0]["bbox"] = [0, 0, 400, 500]
    _write(b, [changed])
    with pytest.raises(ValueError, match="GT mismatch"):
        compare_artifacts(a, b)

    changed_dimensions = _row("row-1", [])
    changed_dimensions["image_width"] = 101
    _write(b, [changed_dimensions])
    with pytest.raises(ValueError, match="GT mismatch"):
        compare_artifacts(a, b)

    identified = _row("row-1", [])
    for index, obj in enumerate(identified["gt"]):
        obj["object_id"] = f"owner-{index}"
    _write(a, [identified])
    changed_identity = _row("row-1", [])
    for index, obj in enumerate(changed_identity["gt"]):
        obj["object_id"] = f"owner-{index + 10}"
    _write(b, [changed_identity])
    with pytest.raises(ValueError, match="GT mismatch"):
        compare_artifacts(a, b)


def test_output_is_deterministic_independent_of_input_row_order(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    rows = [_row("z", []), _row("a", [])]
    _write(a, rows)
    _write(b, list(reversed(rows)))
    first = compare_artifacts(a, b)
    second = compare_artifacts(a, b)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_matching_is_cardinality_first_then_maximum_iou() -> None:
    # Highest-IoU greedy would take GT-0/pred-0 (0.5855) and strand GT-1;
    # the global assignment takes the two lower-overlap edges for two owners.
    gt = [
        ("person", (22, 31, 83, 72)),
        ("person", (29, 10, 79, 71)),
    ]
    pred = [
        ("person", (31, 27, 81, 88)),
        ("person", (27, 37, 93, 87)),
    ]
    matches = _global_matches(gt, pred, 0.50)
    assert {(gt_index, pred_index) for gt_index, pred_index, _ in matches} == {(0, 1), (1, 0)}
