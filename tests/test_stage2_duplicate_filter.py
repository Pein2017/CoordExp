from __future__ import annotations

import pytest

from src.trainers.rollout_matching.contracts import GTObject
from src.trainers.rollout_correction.target_builder import (
    _apply_rollout_correction_duplicate_control,
)
from src.training.stage2.duplicate_filter import (
    DuplicateCandidate,
    DuplicateFilter,
    LegacyChannelBDuplicateControlAdapter,
    PassthroughDuplicateFilter,
)


def test_duplicate_filter_prefers_evidence_and_explorer_support_over_confidence() -> None:
    duplicate_filter = DuplicateFilter(iou_threshold=0.5)
    candidates = (
        DuplicateCandidate(
            object_id="high-confidence",
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.99,
        ),
        DuplicateCandidate(
            object_id="evidence-backed",
            bbox=(1.0, 1.0, 11.0, 11.0),
            confidence=0.20,
            evidence_count=1,
        ),
        DuplicateCandidate(
            object_id="explorer-backed",
            bbox=(30.0, 30.0, 40.0, 40.0),
            confidence=0.10,
            explorer_support=3,
        ),
        DuplicateCandidate(
            object_id="near-explorer",
            bbox=(31.0, 31.0, 41.0, 41.0),
            confidence=0.95,
        ),
    )

    result = duplicate_filter.filter(candidates)

    assert [candidate.object_id for candidate in result.survivors] == [
        "evidence-backed",
        "explorer-backed",
    ]
    assert result.metrics["input_count"] == 4
    assert result.metrics["survivor_count"] == 2
    assert result.metrics["suppressed_count"] == 2

    decisions = {decision.object_id: decision for decision in result.decisions}
    assert decisions["evidence-backed"].action == "kept"
    assert decisions["evidence-backed"].reason == "survivor"
    assert decisions["high-confidence"].action == "suppressed"
    assert decisions["high-confidence"].survivor_id == "evidence-backed"
    assert decisions["high-confidence"].reason == "lower_duplicate_priority"

    assert decisions["explorer-backed"].action == "kept"
    assert decisions["near-explorer"].action == "suppressed"
    assert decisions["near-explorer"].survivor_id == "explorer-backed"


def test_duplicate_filter_keeps_crowd_exempt_duplicates() -> None:
    duplicate_filter = DuplicateFilter(iou_threshold=0.5)
    candidates = (
        DuplicateCandidate(
            object_id="crowd-a",
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.10,
            crowd_exempt=True,
        ),
        DuplicateCandidate(
            object_id="crowd-b",
            bbox=(1.0, 1.0, 11.0, 11.0),
            confidence=0.90,
            crowd_exempt=True,
        ),
    )

    result = duplicate_filter.filter(candidates)

    assert [candidate.object_id for candidate in result.survivors] == [
        "crowd-a",
        "crowd-b",
    ]
    assert result.suppressed == ()
    assert {decision.reason for decision in result.decisions} == {"crowd_exempt"}


def test_duplicate_filter_uses_confidence_then_stable_input_order() -> None:
    duplicate_filter = DuplicateFilter(iou_threshold=0.5)
    candidates = (
        DuplicateCandidate(
            object_id="stable-first",
            bbox=(0.0, 0.0, 10.0, 10.0),
            confidence=0.42,
        ),
        DuplicateCandidate(
            object_id="stable-second",
            bbox=(1.0, 1.0, 11.0, 11.0),
            confidence=0.42,
        ),
        DuplicateCandidate(
            object_id="confidence-winner",
            bbox=(30.0, 30.0, 40.0, 40.0),
            confidence=0.91,
        ),
        DuplicateCandidate(
            object_id="confidence-loser",
            bbox=(31.0, 31.0, 41.0, 41.0),
            confidence=0.71,
        ),
    )

    result = duplicate_filter.filter(candidates)

    assert [candidate.object_id for candidate in result.survivors] == [
        "stable-first",
        "confidence-winner",
    ]
    assert [candidate.object_id for candidate in result.suppressed] == [
        "stable-second",
        "confidence-loser",
    ]

    decisions = {decision.object_id: decision for decision in result.decisions}
    assert decisions["stable-second"].survivor_id == "stable-first"
    assert decisions["confidence-loser"].survivor_id == "confidence-winner"
    assert all(decision.policy_id == "deterministic_duplicate_filter" for decision in result.decisions)


def test_duplicate_filter_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        DuplicateFilter(iou_threshold=1.5)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    (
        ("evidence_count", 1.5, TypeError),
        ("evidence_count", True, TypeError),
        ("evidence_count", -1, ValueError),
        ("explorer_support", 1.5, TypeError),
        ("explorer_support", True, TypeError),
        ("explorer_support", -1, ValueError),
    ),
)
def test_duplicate_filter_rejects_non_integral_support_counts(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    kwargs = {
        "object_id": "candidate",
        "bbox": (0.0, 0.0, 10.0, 10.0),
        field_name: value,
    }

    with pytest.raises(error_type, match=field_name):
        DuplicateCandidate(**kwargs)


@pytest.mark.parametrize("value", ("false", "0", 1, None))
def test_duplicate_candidate_rejects_non_boolean_crowd_exempt(value: object) -> None:
    with pytest.raises(TypeError, match="crowd_exempt"):
        DuplicateCandidate(
            object_id="candidate",
            bbox=(0.0, 0.0, 10.0, 10.0),
            crowd_exempt=value,
        )


def test_legacy_duplicate_control_adapter_matches_live_channel_b_owner() -> None:
    candidates = (
        DuplicateCandidate(
            object_id="winner",
            description="duplicate",
            bbox=(0.0, 0.0, 100.0, 100.0),
        ),
        DuplicateCandidate(
            object_id="suppressed",
            description="duplicate",
            bbox=(0.0, 0.0, 100.0, 100.0),
        ),
        DuplicateCandidate(
            object_id="independent",
            description="independent",
            bbox=(300.0, 300.0, 400.0, 400.0),
        ),
    )
    legacy_result = _apply_rollout_correction_duplicate_control(
        anchor_objects_raw=[
            GTObject(
                index=0,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="duplicate",
            ),
            GTObject(
                index=1,
                geom_type="bbox_2d",
                points_norm1000=[0, 0, 100, 100],
                desc="duplicate",
            ),
            GTObject(
                index=2,
                geom_type="bbox_2d",
                points_norm1000=[300, 300, 400, 400],
                desc="independent",
            ),
        ],
        explorer_objects_raw_by_view=[],
        duplicate_iou_threshold=0.9,
        center_radius_scale=0.0,
        unlabeled_consistent_iou_threshold=0.0,
    )
    adapter_result = LegacyChannelBDuplicateControlAdapter(
        iou_threshold=0.9,
        center_radius_scale=0.0,
        unlabeled_consistent_iou_threshold=0.0,
    ).filter(candidates)

    assert [candidate.object_id for candidate in adapter_result.survivors] == [
        candidates[int(obj.index)].object_id for obj in legacy_result.kept_anchor_objects
    ]
    assert [candidate.object_id for candidate in adapter_result.suppressed] == [
        candidates[index].object_id for index in legacy_result.suppressed_anchor_indices
    ]
    assert adapter_result.metrics["suppressed_count"] == int(
        legacy_result.counter_metrics["stage2_rollout_correction/correction/dup/N_objects_suppressed"]
    )
    assert {
        decision.policy_id for decision in adapter_result.decisions
    } == {"rollout_correction_duplicate_control"}


def test_passthrough_duplicate_filter_marks_already_filtered_legacy_outputs() -> None:
    candidates = (
        DuplicateCandidate(object_id="accepted-a", bbox=(0.0, 0.0, 10.0, 10.0)),
        DuplicateCandidate(object_id="accepted-b", bbox=(1.0, 1.0, 11.0, 11.0)),
    )

    result = PassthroughDuplicateFilter().filter(candidates)

    assert result.survivors == candidates
    assert result.suppressed == ()
    assert [decision.reason for decision in result.decisions] == [
        "passthrough_already_filtered",
        "passthrough_already_filtered",
    ]
