from __future__ import annotations

import json
import math

import pytest

from src.trainers.stage2_two_channel.ul_consensus import (
    ULConsensusCluster,
    ULConsensusResult,
    ULGeometryConfig,
    ULMember,
    ULRolloutEvidence,
    mine_ul_consensus,
    ul_cluster_artifact_rows,
)


GEOMETRY = ULGeometryConfig(
    iou_min=0.7,
    center_distance_scale_max=0.2,
    area_ratio_max=1.5,
    aspect_ratio_max=1.5,
    consumed_overlap_iou_min=0.8,
)


def make_unmatched(desc: str, bbox: tuple[float, float, float, float], local_index: int = 0) -> ULMember:
    return ULMember(
        rollout_id="",
        local_index=local_index,
        desc_id=desc,
        desc_text=desc.replace("_", " "),
        bbox_norm1000=bbox,
    )


def make_valid_rollout(rollout_id: str, members: tuple[ULMember, ...]) -> ULRolloutEvidence:
    return ULRolloutEvidence(
        rollout_id=rollout_id,
        is_valid=True,
        skip_reason=None,
        unmatched_members=tuple(
            ULMember(
                rollout_id=rollout_id,
                local_index=member.local_index,
                desc_id=member.desc_id,
                desc_text=member.desc_text,
                bbox_norm1000=member.bbox_norm1000,
            )
            for member in members
        ),
    )


def make_invalid_rollout(rollout_id: str, reason: str = "parse_error") -> ULRolloutEvidence:
    return ULRolloutEvidence(
        rollout_id=rollout_id,
        is_valid=False,
        skip_reason=reason,
        unmatched_members=(),
    )


def test_ul_consensus_uses_k_valid_denominator_and_promotes_ratio_one() -> None:
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),
        make_valid_rollout("r1", (make_unmatched("person", (102, 101, 202, 221)),)),
        make_valid_rollout("r2", (make_unmatched("person", (101, 102, 201, 222)),)),
        make_invalid_rollout("r3", "parse_error"),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
    )

    assert result.k_valid == 3
    assert result.skip_reasons == {"parse_error": 1}
    assert len(result.promoted_clusters) == 1
    promoted = result.promoted_clusters[0]
    assert promoted.support_rollout_ids == ("r0", "r1", "r2")
    assert promoted.support_ratio == 1.0
    assert promoted.decision == "promoted"
    first_pair = promoted.pairwise_geometry[0]
    assert first_pair["center_distance_scale"] == pytest.approx(
        math.sqrt(5) / ((math.sqrt(100**2 + 120**2) + math.sqrt(100**2 + 120**2)) / 2.0)
    )


def test_same_rollout_near_duplicates_contribute_one_vote() -> None:
    rollouts = (
        make_valid_rollout(
            "r0",
            (
                make_unmatched("person", (100, 100, 200, 220), local_index=0),
                make_unmatched("person", (101, 101, 201, 221), local_index=1),
            ),
        ),
        make_valid_rollout("r1", (make_unmatched("person", (102, 102, 202, 222)),)),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
    )

    assert result.duplicate_like_suppressed_count == 1
    promoted = result.promoted_clusters[0]
    assert promoted.support_rollout_ids == ("r0", "r1")
    assert tuple(member.local_index for member in promoted.members_by_rollout["r0"]) == (0,)


def test_cross_rollout_duplicate_burst_is_quarantined_by_consumed_overlap() -> None:
    consumed = (ULMember("gt", 0, "person", "person", (98, 99, 202, 223)),)
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),
        make_valid_rollout("r1", (make_unmatched("person", (102, 101, 202, 221)),)),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
        consumed_members=consumed,
    )

    assert result.promoted_clusters == ()
    assert len(result.quarantined_clusters) == 1
    assert result.quarantined_clusters[0].reason == "consumed_target_overlap"


def test_geometry_complete_link_rejects_far_box_without_promotion() -> None:
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),
        make_valid_rollout("r1", (make_unmatched("person", (102, 101, 202, 221)),)),
        make_valid_rollout("r2", (make_unmatched("person", (650, 650, 760, 770)),)),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
    )

    assert result.promoted_clusters == ()
    assert any(cluster.reason == "geometry_mismatch" for cluster in result.rejected_clusters)


def test_center_distance_uses_pair_median_diagonal_not_max_diagonal() -> None:
    geometry = ULGeometryConfig(
        iou_min=0.0,
        center_distance_scale_max=0.2,
        area_ratio_max=30.0,
        aspect_ratio_max=2.0,
        consumed_overlap_iou_min=0.8,
    )
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 200)),)),
        make_valid_rollout("r1", (make_unmatched("person", (160, 140, 180, 160)),)),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=geometry,
    )

    assert result.promoted_clusters == ()
    assert any(cluster.reason == "geometry_mismatch" for cluster in result.rejected_clusters)


def test_ul_consensus_artifacts_are_stable_under_rollout_order_permutation() -> None:
    geometry = ULGeometryConfig(
        iou_min=0.0,
        center_distance_scale_max=0.8,
        area_ratio_max=1.0,
        aspect_ratio_max=1.0,
        consumed_overlap_iou_min=0.8,
    )
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (0, 0, 100, 100)),)),
        make_valid_rollout("r1", (make_unmatched("person", (200, 0, 300, 100)),)),
        make_valid_rollout("r2", (make_unmatched("person", (100, 0, 200, 100)),)),
    )

    forward = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=geometry,
    )
    permuted = mine_ul_consensus(
        (rollouts[1], rollouts[0], rollouts[2]),
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=geometry,
    )

    assert ul_cluster_artifact_rows(forward, image_id="img-1") == ul_cluster_artifact_rows(permuted, image_id="img-1")
    assert [cluster.decision for cluster in forward.rejected_clusters] == [
        cluster.decision for cluster in permuted.rejected_clusters
    ]


def test_different_desc_ids_never_cluster_or_promote_together() -> None:
    rollouts = (
        make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),
        make_valid_rollout("r1", (make_unmatched("car", (102, 101, 202, 221)),)),
    )

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=2,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
    )

    assert result.promoted_clusters == ()
    assert {cluster.desc_id for cluster in result.rejected_clusters} == {"person", "car"}
    assert all(cluster.reason == "insufficient_support" for cluster in result.rejected_clusters)


def test_non_one_consensus_ratio_raises() -> None:
    with pytest.raises(ValueError, match="consensus_ratio"):
        mine_ul_consensus(
            (make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),),
            min_ul_valid_rollouts=1,
            consensus_ratio=0.5,
            geometry=GEOMETRY,
        )


def test_artifact_rows_include_all_decisions_and_required_fields() -> None:
    promoted = ULConsensusCluster(
        desc_id="person",
        desc_text="person",
        support_rollout_ids=("r0",),
        support_ratio=1.0,
        decision="promoted",
        reason="consensus",
        members_by_rollout={"r0": (ULMember("r0", 0, "person", "person", (1, 2, 3, 4)),)},
        pairwise_geometry=({"a": 1},),
        consumed_overlap=(),
    )
    rejected = ULConsensusCluster(
        desc_id="car",
        desc_text="car",
        support_rollout_ids=("r0",),
        support_ratio=0.5,
        decision="rejected",
        reason="insufficient_support",
        members_by_rollout={"r0": (ULMember("r0", 0, "car", "car", (5, 6, 7, 8)),)},
        pairwise_geometry=(),
        consumed_overlap=(),
    )
    quarantined = ULConsensusCluster(
        desc_id="dog",
        desc_text="dog",
        support_rollout_ids=("r0",),
        support_ratio=1.0,
        decision="quarantined",
        reason="consumed_target_overlap",
        members_by_rollout={"r0": (ULMember("r0", 0, "dog", "dog", (9, 10, 11, 12)),)},
        pairwise_geometry=(),
        consumed_overlap=({"iou": 0.9},),
    )
    result = ULConsensusResult(
        k_valid=1,
        min_ul_valid_rollouts=1,
        consensus_ratio=1.0,
        geometry=GEOMETRY,
        skip_reasons={},
        promoted_clusters=(promoted,),
        rejected_clusters=(rejected,),
        quarantined_clusters=(quarantined,),
        duplicate_like_suppressed_count=0,
    )

    rows = ul_cluster_artifact_rows(result, image_id="img-1")
    json.dumps(rows, allow_nan=False)

    assert [row["decision"] for row in rows] == ["promoted", "rejected", "quarantined"]
    for row in rows:
        assert row["image_id"] == "img-1"
        assert row["k_valid"] == 1
        assert row["min_ul_valid_rollouts"] == 1
        assert row["consensus_ratio"] == 1.0
        assert row["geometry_thresholds"] == {
            "iou_min": GEOMETRY.iou_min,
            "center_distance_scale_max": GEOMETRY.center_distance_scale_max,
            "area_ratio_max": GEOMETRY.area_ratio_max,
            "aspect_ratio_max": GEOMETRY.aspect_ratio_max,
            "consumed_overlap_iou_min": GEOMETRY.consumed_overlap_iou_min,
        }
        assert {
            "decision",
            "reason",
            "desc_id",
            "desc_text",
            "support_rollout_ids",
            "support_ratio",
            "member_boxes",
            "pairwise_geometry",
            "consumed_overlap",
        }.issubset(row)


@pytest.mark.parametrize(
    "bbox",
    (
        (0, 0, 999, 1000),
        (-1, 0, 100, 100),
        (0, 0, 100, math.inf),
        (0, 0, 100, math.nan),
        (0, 0, 100, True),
        (0, 0, 100, "101"),
        (0, 0, 0, 100),
        (0, 0, 100, 0),
        (100, 0, 99, 100),
    ),
)
def test_ul_member_rejects_bad_norm1000_boxes(bbox: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        ULMember("r0", 0, "person", "person", bbox)  # type: ignore[arg-type]


def test_ul_member_accepts_boundary_valid_norm1000_box() -> None:
    member = ULMember("r0", 0, "person", "person", (0, 0, 999, 999))

    assert member.bbox_norm1000 == (0.0, 0.0, 999.0, 999.0)


@pytest.mark.parametrize("min_ul_valid_rollouts", (False, 0, -1, 1.5, "3"))
def test_mine_ul_consensus_rejects_bad_min_ul_valid_rollouts(min_ul_valid_rollouts: object) -> None:
    with pytest.raises(ValueError, match="min_ul_valid_rollouts"):
        mine_ul_consensus(
            (make_valid_rollout("r0", (make_unmatched("person", (100, 100, 200, 220)),)),),
            min_ul_valid_rollouts=min_ul_valid_rollouts,  # type: ignore[arg-type]
            consensus_ratio=1.0,
            geometry=GEOMETRY,
        )
