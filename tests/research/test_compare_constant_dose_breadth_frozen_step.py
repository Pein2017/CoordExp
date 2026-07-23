from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

import scripts.research.compare_constant_dose_breadth_frozen_step as comparator
from scripts.research.compare_constant_dose_breadth_frozen_step import (
    ARMS,
    SEEDS,
    FrozenStepComparisonError,
    compare_frozen_step,
)


LedgerKey = tuple[str, int]
TreatmentCount = Callable[[str, int, str, int], int]
HealthCount = Callable[[str, int, str, int], int]


@pytest.fixture(autouse=True)
def _small_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparator, "BOOTSTRAP_REPLICATE_COUNT", 80)


def _matching(
    owner_ids: list[str], *, duplicate_count: int = 0, review_count: int = 0
) -> dict[str, Any]:
    return {
        "matched_owner_ids": owner_ids,
        "matched_owner_count": len(owner_ids),
        "matched_prediction_count": len(owner_ids),
        "unmatched_prediction_ids": [
            f"review-{index}" for index in range(review_count)
        ],
        "unmatched_prediction_count": review_count,
        "review_needed_prediction_ids": [
            f"review-{index}" for index in range(review_count)
        ],
        "review_needed_prediction_count": review_count,
        "owner_matching_ineligible_prediction_ids": [],
        "owner_matching_ineligible_prediction_count": 0,
        "duplicate_prediction_ids": [
            f"duplicate-{index}" for index in range(duplicate_count)
        ],
        "duplicate_prediction_count": duplicate_count,
    }


def _health(eligible: bool = True) -> dict[str, Any]:
    return {
        "source_b16_eligible": eligible,
        "source_b16_status": (
            "accepted_natural_end" if eligible else "failed_invalid_before_budget"
        ),
        "invalid_before_budget_count": 0 if eligible else 1,
        "prediction_count": 1,
    }


def _panel_health(candidate_count: int, ineligible_count: int) -> dict[str, Any]:
    return {
        "image_count": candidate_count,
        "eligible_image_count": candidate_count - ineligible_count,
        "invalid_before_budget_count": ineligible_count,
    }


def _write_ledgers(
    root: Path,
    *,
    candidate_count: int = 8,
    treatment_count: TreatmentCount | None = None,
    duplicate_count: HealthCount | None = None,
    review_count: HealthCount | None = None,
    ineligible: set[tuple[LedgerKey, int]] | None = None,
) -> dict[LedgerKey, Path]:
    treatment_count = treatment_count or (lambda _arm, _seed, _threshold, _image_id: 1)
    duplicate_count = duplicate_count or (lambda _arm, _seed, _threshold, _image_id: 0)
    review_count = review_count or (lambda _arm, _seed, _threshold, _image_id: 0)
    ineligible = ineligible or set()
    candidate_identity = {
        "path": "/synthetic/candidates.jsonl",
        "sha256": "candidate-identity",
        "image_count": candidate_count,
    }
    source_panel_identity = {
        "root": "/synthetic/source",
        "execution_model_identity_sha256": "source-model",
        "ordered_image_ids_sha256": "source-images",
    }
    paths: dict[LedgerKey, Path] = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = (arm, seed)
            records = []
            ineligible_ids = []
            for image_id in range(candidate_count):
                source_ids = [f"{image_id}:source"]
                source_matching = {
                    threshold: _matching(source_ids, review_count=1)
                    for threshold in ("0.30", "0.50")
                }
                eligible = (key, image_id) not in ineligible
                if not eligible:
                    ineligible_ids.append(str(image_id))
                comparisons: dict[str, Any] | None = {} if eligible else None
                treatment_matching: dict[str, Any] | None = {} if eligible else None
                if (
                    eligible
                    and comparisons is not None
                    and treatment_matching is not None
                ):
                    for threshold in ("0.30", "0.50"):
                        count = treatment_count(arm, seed, threshold, image_id)
                        retained = source_ids[: min(1, count)]
                        lost = [] if retained else source_ids
                        gained = [
                            f"{image_id}:gained-{index}"
                            for index in range(max(0, count - 1))
                        ]
                        treatment_ids = retained + gained
                        comparisons[threshold] = {
                            "source_owner_ids": source_ids,
                            "treatment_owner_ids": treatment_ids,
                            "retained_source_owner_ids": retained,
                            "lost_source_owner_ids": lost,
                            "gained_annotated_owner_ids": gained,
                            "net_owner_delta": count - 1,
                        }
                        treatment_matching[threshold] = _matching(
                            treatment_ids,
                            duplicate_count=duplicate_count(
                                arm, seed, threshold, image_id
                            ),
                            review_count=review_count(arm, seed, threshold, image_id),
                        )
                records.append(
                    {
                        "image_id": str(image_id),
                        "annotated_object_count": 2
                        if image_id < candidate_count // 2
                        else 8,
                        "object_count_band": "lower_count"
                        if image_id < candidate_count // 2
                        else "higher_count",
                        "source": {
                            "health": _health(),
                            "matching_by_intersection_over_union": source_matching,
                        },
                        "treatment": {
                            "health": _health(eligible),
                            "matching_by_intersection_over_union": treatment_matching,
                        },
                        "owner_comparison": {
                            "eligible": eligible,
                            "exclusion_reason": None
                            if eligible
                            else "treatment_ineligible",
                            "by_intersection_over_union": comparisons,
                        },
                    }
                )
            treatment_name = f"{arm}-seed{seed}-frozen-step"
            aggregate = {
                "candidate_image_count": candidate_count,
                "source_ineligible_image_count": 0,
                "source_ineligible_image_ids": [],
                "treatment_ineligible_image_count": len(ineligible_ids),
                "treatment_ineligible_image_ids": ineligible_ids,
                "source_panel_health": _panel_health(candidate_count, 0),
                "treatment_panel_health": _panel_health(
                    candidate_count, len(ineligible_ids)
                ),
            }
            payload = {
                "schema_version": "source_b16_treatment_owner_ledger.v1",
                "inputs": {
                    "candidate_jsonl": candidate_identity,
                    "source_panel": source_panel_identity,
                },
                "treatments": {
                    treatment_name: {"aggregate": aggregate, "per_image": records}
                },
            }
            path = root / f"{treatment_name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[key] = path
    return paths


def test_infers_candidate_count_and_filters_one_global_denominator(
    tmp_path: Path,
) -> None:
    paths = _write_ledgers(
        tmp_path,
        candidate_count=8,
        ineligible={(("concentrated", 23), 7)},
    )
    result = compare_frozen_step(paths)

    assert result["validated_inputs"]["full_candidate_image_count"] == 8
    assert result["complete_case_cohort"]["image_count"] == 7
    assert result["complete_case_cohort"]["object_count_band_image_counts"] == {
        "higher_count": 3,
        "lower_count": 4,
    }
    threshold = result["intersection_over_union_thresholds"]["0.30"]
    assert threshold["arms"]["broad"]["seeds"]["19"]["complete_case_image_count"] == 7
    health = result["full_candidate_health_by_arm_and_seed"]["concentrated"]["23"]
    assert health["candidate_image_count"] == 8
    assert health["treatment_ineligible_image_ids"] == ["7"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate", "candidate JSONL identity mismatch"),
        ("source_panel", "Source panel identity mismatch"),
        ("source_per_image", "Source per-image surface mismatch"),
    ],
)
def test_rejects_identity_mismatch(tmp_path: Path, mutation: str, message: str) -> None:
    paths = _write_ledgers(tmp_path)
    key = ("concentrated", 23)
    payload = json.loads(paths[key].read_text(encoding="utf-8"))
    if mutation == "candidate":
        payload["inputs"]["candidate_jsonl"]["sha256"] = "different"
    elif mutation == "source_panel":
        payload["inputs"]["source_panel"]["root"] = "/different"
    else:
        treatment = next(iter(payload["treatments"].values()))
        treatment["per_image"][0]["source"]["health"]["source_b16_status"] = "different"
    paths[key].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FrozenStepComparisonError, match=message):
        compare_frozen_step(paths)


def test_reports_owner_metrics_health_differences_and_bands(tmp_path: Path) -> None:
    def counts(arm: str, seed: int, _threshold: str, _image_id: int) -> int:
        if arm == "broad" and seed == 19:
            return 3
        return 1

    result = compare_frozen_step(
        _write_ledgers(
            tmp_path,
            treatment_count=counts,
            duplicate_count=lambda arm, _seed, _threshold, _image: int(arm == "broad"),
            review_count=lambda _arm, seed, _threshold, _image: int(seed == 23),
        )
    )
    threshold = result["intersection_over_union_thresholds"]["0.30"]
    broad_19 = threshold["arms"]["broad"]["seeds"]["19"]

    assert broad_19["source_matched_annotated_owner_count"] == 8
    assert broad_19["treatment_matched_annotated_owner_count"] == 24
    assert broad_19["retained_source_owner_count"] == 8
    assert broad_19["gained_annotated_owner_count"] == 16
    assert broad_19["net_owner_delta"] == 16
    assert broad_19["source_review_needed_prediction_count"] == 8
    assert broad_19["treatment_duplicate_candidate_count"] == 8
    assert (
        broad_19["object_count_bands"]["lower_count"][
            "treatment_matched_annotated_owner_count"
        ]
        == 12
    )
    assert threshold[
        "broad_minus_concentrated_matched_annotated_owner_count_by_seed"
    ] == {"19": 16, "23": 0}
    assert threshold["mean_broad_minus_concentrated_matched_annotated_owner_count"] == 8
    assert (
        threshold["arms"]["broad"][
            "mean_seed_arm_versus_source_net_owner_delta_bootstrap"
        ]["point_estimate"]
        == 8
    )


def test_both_stratified_bootstrap_families_are_deterministic(tmp_path: Path) -> None:
    def counts(arm: str, seed: int, _threshold: str, image_id: int) -> int:
        if arm == "broad":
            return 2 if (image_id + seed) % 3 else 0
        return 1

    paths = _write_ledgers(tmp_path, treatment_count=counts)
    first = compare_frozen_step(paths)
    second = compare_frozen_step(paths)
    first_threshold = first["intersection_over_union_thresholds"]["0.50"]
    second_threshold = second["intersection_over_union_thresholds"]["0.50"]

    assert (
        first_threshold["mean_seed_broad_minus_concentrated_bootstrap"]
        == second_threshold["mean_seed_broad_minus_concentrated_bootstrap"]
    )
    assert (
        first_threshold["arms"]["broad"][
            "mean_seed_arm_versus_source_net_owner_delta_bootstrap"
        ]
        == second_threshold["arms"]["broad"][
            "mean_seed_arm_versus_source_net_owner_delta_bootstrap"
        ]
    )
    assert (
        first_threshold["mean_seed_broad_minus_concentrated_bootstrap"]["random_seed"]
        == 20260723
    )
