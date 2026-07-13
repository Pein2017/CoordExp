from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.analysis.spatial_scope_history.metrics import (
    bootstrap_named_metric_report,
    bootstrap_paired_arm_metric_report,
)
from src.common.errors import DataContractError
from src.eval.detection_categories import COCO_80_CATEGORY_NAMESPACE_SHA256


_SHA256 = "1" * 64


def _primitive(
    arm_identifier: str,
    *,
    image_id: str = "image-1",
    accepted_reference_ids: tuple[str, ...] = ("reference-1", "reference-2"),
    matched_reference_ids: tuple[str, ...] = (),
    unmatched_valid_prediction_ids: tuple[str, ...] = (),
) -> Any:
    """Build only the validated primitive surface consumed by bootstrap APIs."""

    return SimpleNamespace(
        image_id=image_id,
        metric_admission_sha256=_SHA256,
        schedule_sha256=_SHA256,
        cohort_sha256=_SHA256,
        readiness_seal_sha256=_SHA256,
        source_image_sha256=("2" if image_id == "image-1" else "3") * 64,
        image_frozen_order=0 if image_id == "image-1" else 1,
        spatial_grid_spec_sha256=_SHA256,
        arm_identifier=arm_identifier,
        source_width=128,
        source_height=128,
        ledger_scope="audit_augmented",
        category_namespace_sha256=COCO_80_CATEGORY_NAMESPACE_SHA256,
        threshold=0.5,
        accepted_reference_ids=accepted_reference_ids,
        core_interior_reference_ids=(),
        post_merge_match=SimpleNamespace(
            matched_reference_ids=matched_reference_ids
        ),
        post_merge_matched_reference_ids=matched_reference_ids,
        unmatched_valid_prediction_ids=unmatched_valid_prediction_ids,
        post_merge_strict_duplicate_components=(),
        post_merge_duplicate_excess_count=0,
        final_valid_prediction_count=(
            len(matched_reference_ids) + len(unmatched_valid_prediction_ids)
        ),
        malformed_row_count=0,
        invalid_row_count=0,
        attempted_row_count=1,
        invalid_call_count=0,
        attempted_call_count=1,
        natural_closure_count=1,
    )


@pytest.mark.parametrize(
    ("candidate_arm", "comparator_arm"),
    (
        ("MASK_RESET", "FULL_BAG_K"),
        ("MASK_RESET", "MASK_CUMULATIVE"),
        ("MASK_CUMULATIVE", "MASK_RESET"),
        ("TILE_RESET", "MASK_RESET"),
    ),
)
def test_paired_arm_bootstrap_accepts_only_declared_arm_directions(
    candidate_arm: str,
    comparator_arm: str,
) -> None:
    report = bootstrap_paired_arm_metric_report(
        (_primitive(candidate_arm),),
        (_primitive(comparator_arm),),
        (_primitive("FULL_SINGLE"),),
        metric_name="natural_closure_rate_difference",
        scope="audit-augmented paired-arm guardrail",
    )

    assert report.point_estimate.comparator == comparator_arm
    assert report.point_estimate.estimate == 0.0
    assert report.total_replicates == 10_000
    assert report.applicable_replicates == 10_000
    assert report.records == ()


def test_paired_rescue_uses_one_full_single_missed_set_and_is_deterministic() -> None:
    baseline = (
        _primitive("FULL_SINGLE", matched_reference_ids=("reference-1",)),
        _primitive(
            "FULL_SINGLE",
            image_id="image-2",
            matched_reference_ids=("reference-2",),
        ),
    )
    candidate = (
        _primitive(
            "MASK_RESET",
            matched_reference_ids=("reference-1", "reference-2"),
        ),
        _primitive(
            "MASK_RESET",
            image_id="image-2",
            matched_reference_ids=("reference-1", "reference-2"),
        ),
    )
    comparator = (
        _primitive("FULL_BAG_K", matched_reference_ids=("reference-1",)),
        _primitive(
            "FULL_BAG_K",
            image_id="image-2",
            matched_reference_ids=("reference-2",),
        ),
    )

    first = bootstrap_paired_arm_metric_report(
        candidate,
        comparator,
        baseline,
        metric_name="post_merge_local_rescue_rate_difference",
        scope="audit-augmented post-merge Local Rescue Rate difference",
    )
    second = bootstrap_paired_arm_metric_report(
        tuple(reversed(candidate)),
        tuple(reversed(comparator)),
        tuple(reversed(baseline)),
        metric_name="post_merge_local_rescue_rate_difference",
        scope="audit-augmented post-merge Local Rescue Rate difference",
    )

    assert first == second
    assert first.point_estimate.numerator == 2.0
    assert first.point_estimate.denominator == 2.0
    assert first.point_estimate.comparator_numerator == 0.0
    assert first.point_estimate.comparator_denominator == 2.0
    assert first.point_estimate.estimate == 1.0


@pytest.mark.parametrize(
    "metric_name",
    (
        "manual_precision_difference",
        "post_merge_strict_duplicate_rate_difference",
        "invalid_row_rate_difference",
        "invalid_call_rate_difference",
        "natural_closure_rate_difference",
    ),
)
def test_paired_arm_bootstrap_supports_only_declared_guardrail_differences(
    metric_name: str,
) -> None:
    report = bootstrap_paired_arm_metric_report(
        (_primitive("MASK_RESET", matched_reference_ids=("reference-1",)),),
        (_primitive("FULL_BAG_K", matched_reference_ids=("reference-1",)),),
        (_primitive("FULL_SINGLE"),),
        metric_name=metric_name,
        scope="paired safety guardrail difference",
    )
    assert report.point_estimate.estimate == 0.0

    with pytest.raises(DataContractError, match="paired_bootstrap_metric_name"):
        bootstrap_paired_arm_metric_report(
            (_primitive("MASK_RESET"),),
            (_primitive("FULL_BAG_K"),),
            (_primitive("FULL_SINGLE"),),
            metric_name="overall_retention_difference",
            scope="undeclared difference",
        )


def test_paired_arm_bootstrap_rejects_swaps_missing_images_and_wrong_baseline() -> None:
    with pytest.raises(DataContractError, match="paired_bootstrap_arm_pair"):
        bootstrap_paired_arm_metric_report(
            (_primitive("FULL_BAG_K"),),
            (_primitive("MASK_RESET"),),
            (_primitive("FULL_SINGLE"),),
            metric_name="manual_precision_difference",
            scope="forbidden swapped direction",
        )
    with pytest.raises(DataContractError, match="image_universe"):
        bootstrap_paired_arm_metric_report(
            (_primitive("MASK_RESET"),),
            (_primitive("FULL_BAG_K", image_id="image-2"),),
            (_primitive("FULL_SINGLE"),),
            metric_name="manual_precision_difference",
            scope="missing comparator image",
        )
    with pytest.raises(DataContractError, match="wrong_comparator_arm"):
        bootstrap_paired_arm_metric_report(
            (_primitive("MASK_RESET"),),
            (_primitive("FULL_BAG_K"),),
            (_primitive("TILE_RESET"),),
            metric_name="manual_precision_difference",
            scope="wrong baseline",
        )


@pytest.mark.parametrize(
    ("field_name", "altered_value"),
    (
        ("ledger_scope", "official_annotation"),
        ("threshold", 0.75),
        ("category_namespace_sha256", "9" * 64),
        ("accepted_reference_ids", ("reference-1",)),
    ),
)
def test_paired_arm_bootstrap_rejects_reference_contract_tampering(
    field_name: str,
    altered_value: object,
) -> None:
    comparator = _primitive("FULL_BAG_K")
    setattr(comparator, field_name, altered_value)

    with pytest.raises(DataContractError, match="pair_contract"):
        bootstrap_paired_arm_metric_report(
            (_primitive("MASK_RESET"),),
            (comparator,),
            (_primitive("FULL_SINGLE"),),
            metric_name="manual_precision_difference",
            scope="tampered primitive contract",
        )


def test_existing_named_bootstrap_contract_is_unchanged() -> None:
    report = bootstrap_named_metric_report(
        (_primitive("MASK_RESET"),),
        (_primitive("FULL_SINGLE"),),
        metric_name="natural_closure_rate",
        scope="existing named bootstrap",
    )

    assert report.point_estimate.estimate == 1.0
    assert report.point_estimate.comparator is None
    assert report.total_replicates == 10_000
    assert report.minimum_applicable_replicates == 9_500
    assert report.evidence_status == "metric_bearing"
