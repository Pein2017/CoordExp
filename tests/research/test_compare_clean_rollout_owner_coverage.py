from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.compare_clean_rollout_owner_coverage import _global_matches, compare_artifacts


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
