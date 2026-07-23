"""Focused contracts for the physical-owner duplicate review queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.build_physical_owner_duplication_review_queue import (
    ReviewQueueError,
    build_physical_owner_duplication_review_queue,
    write_physical_owner_duplication_review_queue,
)


OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_START = 151670


def _token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("utf-8")).hexdigest()


def _row_tokens(index: int) -> list[int]:
    return [
        OBJECT_REF_START,
        100 + index,
        OBJECT_REF_END,
        BOX_START,
        COORDINATE_START + index,
        COORDINATE_START + 1 + index,
        COORDINATE_START + 20 + index,
        COORDINATE_START + 21 + index,
        BOX_END,
    ]


def _prediction(index: int, box: list[float]) -> dict[str, object]:
    return {
        "object_span_id": f"image-1:greedy:span-{index}",
        "generated_order": index,
        "description": "person",
        "bbox": box,
    }


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _rollout_document(
    rows: list[list[float]],
    *,
    mode: str = "greedy",
    seed: int | None = 0,
    trajectory_id: str | None = None,
    partial_suffix: list[int] | None = None,
) -> dict[str, object]:
    tokens = [token for index in range(len(rows)) for token in _row_tokens(index)]
    if partial_suffix:
        tokens.extend(partial_suffix)
    rollout: dict[str, object] = {
        "image_id": "1",
        "decode_mode": mode,
        "generated_token_ids": tokens,
        "generated_token_ids_sha256": _token_hash(tokens),
        "predictions": {"predictions": [_prediction(index, box) for index, box in enumerate(rows)]},
    }
    if seed is not None:
        rollout["seed"] = seed
    if trajectory_id is not None:
        rollout["trajectory_id"] = trajectory_id
    return {
        "schema_version": "current_seeded_sampled_rollouts.v1",
        "config": {"decode_mode": mode},
        "prompt_metadata": {"1": {"image_path": "/synthetic/image-1.jpg"}},
        "rollouts": [rollout],
    }


def _receipt(
    index: int,
    box: list[float],
    status: str,
    *,
    owner_id: str | None = None,
    annotation_iou: float | None = 1.0,
    trajectory_id: str = "greedy",
) -> dict[str, object]:
    result: dict[str, object] = {
        "image_id": "1",
        "trajectory_id": trajectory_id,
        "decode_mode": "greedy" if trajectory_id == "greedy" else "sampled",
        "generated_row_index": index,
        "prediction_id": f"image-1:greedy:span-{index}" if trajectory_id == "greedy" else f"image-1:{trajectory_id}:span-{index}",
        "category": "person",
        "bbox": box,
        "entity_status": status,
    }
    if owner_id is not None:
        if status in {"duplicate", "duplicate_owner"}:
            result["candidate_owner_id"] = owner_id
            result["candidate_owner_iou"] = annotation_iou
        else:
            result["owner_id"] = owner_id
            result["owner_bbox"] = [0.0, 0.0, 10.0, 10.0] if owner_id == "1:a" else [20.0, 0.0, 30.0, 10.0]
            result["owner_category"] = "person"
            result["intersection_over_union"] = annotation_iou
    return result


def _union_document(receipts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "individual_trajectory_union_support.v1",
        "image_results": [
            {
                "image_id": "1",
                "owners": [
                    {"owner_id": "1:a", "category": "person", "bbox": [0.0, 0.0, 10.0, 10.0]},
                    {"owner_id": "1:b", "category": "person", "bbox": [20.0, 0.0, 30.0, 10.0]},
                ],
                "budgets": [{"budget": 16, "row_assignment_receipts": receipts}],
            }
        ],
    }


def _run(tmp_path: Path, receipts: list[dict[str, object]], rows: list[list[float]], **kwargs: object) -> dict[str, object]:
    union_path = _write_json(tmp_path / "union.json", _union_document(receipts))
    greedy_path = _write_json(tmp_path / "greedy.json", _rollout_document(rows))
    return build_physical_owner_duplication_review_queue(
        union_support_path=union_path,
        greedy_json_paths=[greedy_path],
        **kwargs,
    )


def test_groups_reversed_receipts_preserves_first_owner_and_finds_recovery(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a"),
        _receipt(1, boxes[1], "duplicate", owner_id="1:a"),
        _receipt(2, boxes[2], "duplicate_owner", owner_id="1:a"),
        _receipt(3, boxes[3], "verified_owner", owner_id="1:b"),
    ]
    result = _run(tmp_path, list(reversed(receipts)), boxes)

    queue = result["candidate_queue"]
    assert isinstance(queue, list) and len(queue) == 1
    candidate = queue[0]
    assert candidate["earlier_accepted_owner_row"]["row_index"] == 0
    assert [item["row_index"] for item in candidate["duplicate_rows"]] == [1, 2]
    assert candidate["first_later_trusted_unseen_verified_owner_row"]["row_index"] == 3
    assert candidate["not_a_training_label"] is True


def test_ambiguous_and_unmatched_rows_stay_neutral_not_candidates(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [40.0, 0.0, 50.0, 10.0]]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a"),
        _receipt(1, boxes[1], "ambiguous_matched_review"),
        _receipt(2, boxes[2], "unmatched"),
    ]
    result = _run(tmp_path, receipts, boxes)

    assert result["candidate_queue"] == []
    census = result["census"]
    assert census["row_classification_counts"]["neutral"] == 2
    assert census["neutral_reason_counts"]["official_unmatched_or_ambiguous"] == 2


def test_no_recovery_control_and_threshold_boundaries(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a", annotation_iou=0.5),
        _receipt(1, boxes[1], "duplicate", owner_id="1:a", annotation_iou=0.5),
    ]
    result = _run(
        tmp_path,
        receipts,
        boxes,
        annotation_iou_threshold=0.5,
        prediction_iou_threshold=1.0,
    )
    queue = result["candidate_queue"]
    assert isinstance(queue, list) and len(queue) == 1
    assert queue[0]["first_later_trusted_unseen_verified_owner_row"] is None

    rejected = _run(
        tmp_path / "below",
        [
            _receipt(0, boxes[0], "verified_owner", owner_id="1:a", annotation_iou=0.5),
            _receipt(1, boxes[1], "duplicate", owner_id="1:a", annotation_iou=0.499999),
        ],
        boxes,
        annotation_iou_threshold=0.5,
        prediction_iou_threshold=1.0,
    )
    assert rejected["candidate_queue"] == []


def test_one_burst_per_owner_default_and_deterministic_written_jsonl(tmp_path: Path) -> None:
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
        [40.0, 0.0, 50.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
    ]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a"),
        _receipt(1, boxes[1], "duplicate", owner_id="1:a"),
        _receipt(2, boxes[2], "unmatched"),
        _receipt(3, boxes[3], "duplicate", owner_id="1:a"),
    ]
    first = _run(tmp_path / "input", receipts, boxes)
    second = _run(tmp_path / "input", receipts, boxes)
    assert first == second
    assert len(first["candidate_queue"]) == 1
    assert first["census"]["neutral_reason_counts"]["additional_burst_suppressed"] == 1

    first_paths = write_physical_owner_duplication_review_queue(first, tmp_path / "out-one")
    second_paths = write_physical_owner_duplication_review_queue(second, tmp_path / "out-two")
    assert first_paths["candidate_queue"].read_bytes() == second_paths["candidate_queue"].read_bytes()
    assert first_paths["census"].read_bytes() == second_paths["census"].read_bytes()
    assert first_paths["visualization_manifest"].read_bytes() == second_paths["visualization_manifest"].read_bytes()


def test_parser_dropped_partial_suffix_is_neutral_census_evidence(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a"),
        _receipt(1, boxes[1], "duplicate", owner_id="1:a"),
    ]
    union_path = _write_json(tmp_path / "union.json", _union_document(receipts))
    greedy_path = _write_json(
        tmp_path / "greedy.json",
        _rollout_document(boxes, partial_suffix=[OBJECT_REF_START, 777]),
    )
    result = build_physical_owner_duplication_review_queue(
        union_support_path=union_path,
        greedy_json_paths=[greedy_path],
    )

    assert len(result["candidate_queue"]) == 1
    assert result["census"]["neutral_exact_token_suffix_trajectory_count"] == 1
    assert result["census"]["neutral_exact_token_suffix_token_count"] == 2


def test_nonpositive_decoded_box_is_retained_as_neutral_exact_evidence(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 4.0, 10.0]]
    receipts = [
        _receipt(0, boxes[0], "verified_owner", owner_id="1:a"),
        _receipt(1, boxes[1], "unmatched"),
    ]
    result = _run(tmp_path, receipts, boxes)

    assert result["candidate_queue"] == []
    assert result["census"]["neutral_invalid_exact_prediction_row_count"] == 1
    assert result["census"]["neutral_reason_counts"]["invalid_exact_prediction_geometry"] == 1
    prediction = result["visualization_manifest"]["images"][0]["prediction_boxes"][1]
    assert prediction["prediction_bbox"] is None
    assert prediction["geometry_status"] == "neutral_invalid_exact_prediction_geometry"


def test_malformed_box_end_delimited_exact_row_is_neutral_not_fatal(tmp_path: Path) -> None:
    valid_box = [0.0, 0.0, 10.0, 10.0]
    union_path = _write_json(
        tmp_path / "union.json",
        _union_document([_receipt(0, valid_box, "verified_owner", owner_id="1:a")]),
    )
    rollout = _rollout_document([valid_box])
    row = rollout["rollouts"][0]
    tokens = list(row["generated_token_ids"])
    tokens.extend(
        [
            OBJECT_REF_START,
            555,
            OBJECT_REF_END,
            BOX_START,
            COORDINATE_START,
            COORDINATE_START + 1,
            COORDINATE_START + 2,
            COORDINATE_START + 3,
            COORDINATE_START + 4,
            BOX_END,
        ]
    )
    row["generated_token_ids"] = tokens
    row["generated_token_ids_sha256"] = _token_hash(tokens)
    greedy_path = _write_json(tmp_path / "greedy.json", rollout)

    result = build_physical_owner_duplication_review_queue(
        union_support_path=union_path,
        greedy_json_paths=[greedy_path],
    )
    assert result["census"]["neutral_invalid_exact_token_row_count"] == 1
    assert result["candidate_queue"] == []


def test_fails_closed_on_missing_exact_prediction_or_duplicate_trajectory(tmp_path: Path) -> None:
    boxes = [[0.0, 0.0, 10.0, 10.0]]
    missing = _run(tmp_path / "missing", [_receipt(0, boxes[0], "verified_owner", owner_id="1:a")], boxes)
    assert missing["candidate_queue"] == []

    union_path = _write_json(tmp_path / "dupe-union.json", _union_document([_receipt(0, boxes[0], "verified_owner", owner_id="1:a")]))
    rollout = _rollout_document(boxes)
    rollout["rollouts"].append(dict(rollout["rollouts"][0]))
    greedy_path = _write_json(tmp_path / "dupe-greedy.json", rollout)
    with pytest.raises(ReviewQueueError, match="duplicate trajectory identity"):
        build_physical_owner_duplication_review_queue(union_support_path=union_path, greedy_json_paths=[greedy_path])
