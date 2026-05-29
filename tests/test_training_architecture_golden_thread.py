from __future__ import annotations

from pathlib import Path

import pytest

from src.training.stage2.planners import Stage2PlanningObject

from helpers.training_architecture_fixture_builder import (
    _build_stage2_object,
    build_stage1_golden_thread,
    build_stage2_golden_thread,
    load_fixture,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "training_architecture"


def test_stage1_compact_full_golden_thread_matches_static_snapshot() -> None:
    source = load_fixture(FIXTURE_DIR / "compact_full_stage1_source.json")
    expected = load_fixture(FIXTURE_DIR / "compact_full_stage1_expected.json")

    actual = build_stage1_golden_thread(source).snapshot

    assert actual == expected
    assert actual["supervision_plan"]["object_ids"] == [
        "stage1-golden-compact:ann-100:src-2",
        "stage1-golden-compact:ann-101:src-4",
        "stage1-golden-compact:ann-102:src-7",
    ]
    assert actual["selected_targets"]["trie"]["target_token_ids"]
    assert actual["selected_targets"]["coordinate"]["target_token_ids"]
    assert actual["selected_targets"]["trie"]["target_token_texts"] == [
        "traffic",
        "red",
        "person",
    ]
    assert actual["spans"][0]["role"] == "free_text"
    assert actual["mapper"][0]["row_index"] == (
        actual["mapper"][0]["label_position"] - 1
    )


def test_stage2_rollout_golden_thread_matches_static_snapshot() -> None:
    source = load_fixture(FIXTURE_DIR / "stage2_rollout_source.json")
    expected = load_fixture(FIXTURE_DIR / "stage2_rollout_expected.json")

    actual = build_stage2_golden_thread(source).snapshot

    assert actual == expected
    assert actual["assignment"]["pairs"] == [
        {
            "prediction_index": 0,
            "ground_truth_index": 0,
            "prediction_id": "pred-dog-survivor",
            "ground_truth_id": "gt-dog",
            "iou": 0.960977,
            "reason": "matched",
        }
    ]
    assert actual["plan"]["object_ids"] == ["pred-dog-survivor", "gt-cat"]
    assert actual["plan"]["source_roles"] == ["accepted_rollout", "false_negative"]
    assert actual["diagnostics"]["suppressed_duplicate_count"] == 2
    assert actual["diagnostics"]["false_negative_count"] == 1
    assert actual["diagnostics"]["post_duplicate_prediction_count"] == 1
    assert actual["plan"]["metadata"]["accepted_rollout_count"] == (
        actual["diagnostics"]["predicted_count"]
    )
    assert [
        decision["reason"]
        for decision in actual["duplicate_decisions"]
        if decision["action"] == "suppressed"
    ] == ["lower_duplicate_priority", "lower_duplicate_priority"]


def test_stage2_fixture_builder_preserves_planning_object_validation() -> None:
    with pytest.raises(TypeError, match="crowd_exempt"):
        _build_stage2_object(
            {
                "object_id": "pred-invalid",
                "description": "invalid",
                "bbox": [0, 0, 10, 10],
                "crowd_exempt": "false",
            },
            default_provenance="rollout_accepted",
        )

    with pytest.raises(TypeError, match="evidence_count"):
        _build_stage2_object(
            {
                "object_id": "pred-invalid",
                "description": "invalid",
                "bbox": [0, 0, 10, 10],
                "evidence_count": 1.5,
            },
            default_provenance="rollout_accepted",
        )

    valid = _build_stage2_object(
        {
            "object_id": "pred-valid",
            "description": "valid",
            "bbox": [0, 0, 10, 10],
            "crowd_exempt": True,
            "evidence_count": 1,
        },
        default_provenance="rollout_accepted",
    )

    assert isinstance(valid, Stage2PlanningObject)
    assert valid.crowd_exempt is True
    assert valid.evidence_count == 1
