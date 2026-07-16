from __future__ import annotations

import copy

import torch

from scripts.research.run_batch_coordinate_logit_invariance import summarize_logits
from scripts.research.run_single_target_visual_feature_replay_batch_four import (
    EXPECTED_CACHED_HOMOGENEOUS_VECTOR_SHA256,
    EXPECTED_CACHED_SINGLE_VECTOR_SHA256,
    NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM,
    NATURAL_SINGLE_TARGET_ARM,
    REPLAYED_SINGLE_TARGET_ARM,
    SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM,
    build_causal_summary,
    build_layouts,
    build_trust_gate,
    classify_recovery,
    split_homogeneous_feature_bundle,
)
from src.analysis.visual_support_counterfactual.intervention import FeatureBundle


def _summary(coordinate_bin: int) -> dict[str, object]:
    logits = torch.zeros(152700, dtype=torch.float32)
    logits[151670 + coordinate_bin] = 8.0
    return summarize_logits(logits)


def _path(summary: dict[str, object], vector_hash: str, *, count: int) -> dict[str, object]:
    rows = []
    for position in range(count):
        row_summary = copy.deepcopy(summary)
        row_summary["coordinate_raw_logits_float32_sha256"] = vector_hash
        rows.append(
            {
                "batch_position": position,
                "is_target_recipient": True,
                "logit_summary": row_summary,
            }
        )
    return {"rows": rows, "controller": None}


def _arm(
    summary: dict[str, object],
    vector_hash: str,
    *,
    count: int,
    replayed: bool = False,
) -> dict[str, object]:
    controller = {
        "feature_call_count": 1,
        "grid_mismatch_count": 0,
        "hook_restored": True,
    }
    cached = _path(summary, vector_hash, count=count)
    direct = _path(summary, vector_hash, count=count)
    if replayed:
        cached["controller"] = copy.deepcopy(controller)
        direct["controller"] = copy.deepcopy(controller)
    return {"cached_path": cached, "direct_path": direct}


def test_layouts_are_single_and_four_exact_target_copies() -> None:
    single, homogeneous = build_layouts()
    assert single["target_positions"] == [0]
    assert homogeneous["target_positions"] == [0, 1, 2, 3]
    assert len(set(homogeneous["condition_names"])) == 1


def test_split_homogeneous_feature_bundle_proves_per_image_equality() -> None:
    primary = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    deepstack = tuple(torch.cat([primary] * 4, dim=0) for _ in range(3))
    bundle = FeatureBundle(primary=(primary,) * 4, deepstack=deepstack)
    per_image, receipt = split_homogeneous_feature_bundle(bundle)
    assert len(per_image) == 4
    assert receipt["primary_all_equal"] is True
    assert receipt["deepstack_all_equal"] is True
    assert all(len(item.primary) == 1 and len(item.deepstack) == 3 for item in per_image)


def test_recovery_classification_distinguishes_endpoints() -> None:
    assert classify_recovery(1.0, closer_to=NATURAL_SINGLE_TARGET_ARM) == (
        "vision-output ownership"
    )
    assert classify_recovery(0.0, closer_to=NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM) == (
        "post-vision ownership"
    )
    assert classify_recovery(0.5, closer_to=NATURAL_SINGLE_TARGET_ARM) == (
        "mixed or unresolved ownership"
    )


def test_causal_summary_recovers_vision_endpoint() -> None:
    single = _summary(181)
    homogeneous = _summary(438)
    arms = {
        NATURAL_SINGLE_TARGET_ARM: _arm(single, "single", count=1),
        NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM: _arm(
            homogeneous, "homogeneous", count=4
        ),
        REPLAYED_SINGLE_TARGET_ARM: _arm(single, "single", count=1, replayed=True),
        SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM: _arm(
            single, "replayed", count=4, replayed=True
        ),
    }
    summary = build_causal_summary(arms)
    assert summary["cached_path"]["centered_root_mean_square_recovery_fraction"] == 1.0
    assert summary["cached_path"]["ownership_classification"] == (
        "vision-output ownership"
    )
    assert summary["cross_path_verdict"] == "vision-output ownership"


def test_trust_gate_requires_frozen_baselines_and_exact_noop() -> None:
    single = _summary(181)
    homogeneous = _summary(438)
    arms = {
        NATURAL_SINGLE_TARGET_ARM: _arm(
            single, EXPECTED_CACHED_SINGLE_VECTOR_SHA256, count=1
        ),
        NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM: _arm(
            homogeneous, EXPECTED_CACHED_HOMOGENEOUS_VECTOR_SHA256, count=4
        ),
        REPLAYED_SINGLE_TARGET_ARM: _arm(
            single, EXPECTED_CACHED_SINGLE_VECTOR_SHA256, count=1, replayed=True
        ),
        SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM: _arm(
            homogeneous, "causal-outcome-unconstrained", count=4, replayed=True
        ),
    }
    gate = build_trust_gate(
        arms=arms,
        homogeneous_feature_equality={
            "primary_all_equal": True,
            "deepstack_all_equal": True,
        },
    )
    assert gate["passed"] is True
    arms[REPLAYED_SINGLE_TARGET_ARM]["cached_path"]["rows"][0]["logit_summary"][
        "coordinate_raw_logits_float32_sha256"
    ] = "not-a-noop"
    assert build_trust_gate(
        arms=arms,
        homogeneous_feature_equality={
            "primary_all_equal": True,
            "deepstack_all_equal": True,
        },
    )["passed"] is False
