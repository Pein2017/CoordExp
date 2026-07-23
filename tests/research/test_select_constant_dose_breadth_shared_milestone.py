from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

import scripts.research.select_constant_dose_breadth_shared_milestone as selector
from scripts.research.select_constant_dose_breadth_shared_milestone import (
    ARMS,
    SEEDS,
    STEPS,
    SharedMilestoneSelectionError,
    select_shared_milestone,
)


LedgerKey = tuple[str, int, int]
TreatmentCount = Callable[[str, int, int, str, int], int]
DuplicateCount = Callable[[str, int, int, str, int], int]


@pytest.fixture(autouse=True)
def _small_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector, "BOOTSTRAP_REPLICATE_COUNT", 80)


def _matching(owner_ids: list[str], duplicate_count: int = 0) -> dict[str, Any]:
    return {
        "matched_owner_ids": owner_ids,
        "matched_owner_count": len(owner_ids),
        "matched_prediction_count": len(owner_ids),
        "unmatched_prediction_ids": [],
        "unmatched_prediction_count": 0,
        "review_needed_prediction_ids": [],
        "review_needed_prediction_count": 0,
        "owner_matching_ineligible_prediction_ids": [],
        "owner_matching_ineligible_prediction_count": 0,
        "duplicate_prediction_ids": [
            f"duplicate-{index}" for index in range(duplicate_count)
        ],
        "duplicate_prediction_count": duplicate_count,
    }


def _health(*, eligible: bool = True) -> dict[str, Any]:
    return {
        "source_b16_eligible": eligible,
        "source_b16_status": "accepted_natural_end"
        if eligible
        else "failed_invalid_before_budget",
        "invalid_before_budget_count": 0 if eligible else 1,
        "token_limit_before_budget_count": 0,
        "natural_end_before_budget_count": 1 if eligible else 0,
        "raw_stop_reason": "im_end",
        "prediction_count": 1,
        "owner_matching_eligible_prediction_count": 1,
        "owner_matching_ineligible_prediction_count": 0,
        "dropped_prediction_count": 0 if eligible else 1,
        "malformed_row_count": 0 if eligible else 1,
        "truncation_count": 0,
    }


def _panel_health(ineligible_count: int) -> dict[str, Any]:
    return {
        "image_count": 256,
        "source_b16_status_counts": {
            "accepted_natural_end": 256 - ineligible_count,
            "failed_invalid_before_budget": ineligible_count,
        },
        "eligible_image_count": 256 - ineligible_count,
        "invalid_before_budget_count": ineligible_count,
        "token_limit_before_budget_count": 0,
        "natural_end_before_budget_count": 256 - ineligible_count,
        "raw_stop_reason_counts": {"im_end": 256},
        "prediction_count": 256,
        "owner_matching_eligible_prediction_count": 256,
        "owner_matching_ineligible_prediction_count": 0,
        "dropped_prediction_count": ineligible_count,
        "malformed_row_count": ineligible_count,
        "truncation_count": 0,
    }


def _write_ledgers(
    root: Path,
    *,
    treatment_count: TreatmentCount | None = None,
    duplicate_count: DuplicateCount | None = None,
    ineligible: set[tuple[LedgerKey, int]] | None = None,
) -> dict[LedgerKey, Path]:
    treatment_count = treatment_count or (
        lambda _arm, _seed, _step, _threshold, _image: 1
    )
    duplicate_count = duplicate_count or (
        lambda _arm, _seed, _step, _threshold, _image: 0
    )
    ineligible = ineligible or set()
    paths: dict[LedgerKey, Path] = {}
    candidate_identity = {
        "path": "/synthetic/development.jsonl",
        "sha256": "candidate-sha256",
        "image_count": 256,
    }
    source_panel_identity = {
        "root": "/synthetic/source-panel",
        "ordered_image_ids_sha256": "source-image-order",
        "execution_model_identity_sha256": "source-model",
    }
    bands = ("one_to_three", "four_to_six", "seven_to_nine", "ten_or_more")
    for arm in ARMS:
        for seed in SEEDS:
            for step in STEPS:
                key = (arm, seed, step)
                records = []
                treatment_ineligible_ids = []
                for image_id in range(256):
                    source_ids = [f"{image_id}:source"]
                    source_matching = {
                        threshold: _matching(source_ids)
                        for threshold in ("0.30", "0.50")
                    }
                    eligible = (key, image_id) not in ineligible
                    if not eligible:
                        treatment_ineligible_ids.append(str(image_id))
                    comparisons: dict[str, Any] | None = {} if eligible else None
                    treatment_matching: dict[str, Any] | None = {} if eligible else None
                    if (
                        eligible
                        and comparisons is not None
                        and treatment_matching is not None
                    ):
                        for threshold in ("0.30", "0.50"):
                            count = treatment_count(
                                arm, seed, step, threshold, image_id
                            )
                            retained = source_ids[: min(count, 1)]
                            gained = [
                                f"{image_id}:gained-{index}"
                                for index in range(max(0, count - 1))
                            ]
                            treatment_ids = retained + gained
                            lost = [] if retained else source_ids
                            comparisons[threshold] = {
                                "source_owner_ids": source_ids,
                                "treatment_owner_ids": treatment_ids,
                                "retained_source_owner_ids": retained,
                                "lost_source_owner_ids": lost,
                                "gained_annotated_owner_ids": gained,
                                "retained_source_owner_count": len(retained),
                                "lost_source_owner_count": len(lost),
                                "gained_annotated_owner_count": len(gained),
                                "net_owner_delta": count - 1,
                            }
                            treatment_matching[threshold] = _matching(
                                treatment_ids,
                                duplicate_count(arm, seed, step, threshold, image_id),
                            )
                    band = bands[image_id // 64]
                    records.append(
                        {
                            "image_id": str(image_id),
                            "annotated_object_count": image_id // 64 + 1,
                            "object_count_band": band,
                            "source": {
                                "health": _health(),
                                "matching_by_intersection_over_union": source_matching,
                            },
                            "treatment": {
                                "health": _health(eligible=eligible),
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
                treatment_name = f"{arm}-seed{seed}-step{step}"
                aggregate = {
                    "candidate_image_count": 256,
                    "source_ineligible_image_count": 0,
                    "source_ineligible_image_ids": [],
                    "treatment_ineligible_image_count": len(treatment_ineligible_ids),
                    "treatment_ineligible_image_ids": treatment_ineligible_ids,
                    "source_panel_health": _panel_health(0),
                    "treatment_panel_health": _panel_health(
                        len(treatment_ineligible_ids)
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


def test_rejects_image_denominator_mismatch(tmp_path: Path) -> None:
    paths = _write_ledgers(tmp_path)
    key = ("broad", 19, 20)
    payload = json.loads(paths[key].read_text(encoding="utf-8"))
    payload["treatments"]["broad-seed19-step20"]["per_image"].pop()
    paths[key].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SharedMilestoneSelectionError, match="image denominator mismatch"
    ):
        select_shared_milestone(paths)


def test_selects_lexicographic_winner_and_uses_one_complete_case_cohort(
    tmp_path: Path,
) -> None:
    def counts(arm: str, seed: int, step: int, threshold: str, _image: int) -> int:
        if arm == "broad" and step == 20:
            return 3 if threshold == "0.30" else 2
        if arm == "broad" and step == 30:
            return 2
        return 1

    excluded_key = (("concentrated", 23, 31), 255)
    result = select_shared_milestone(
        _write_ledgers(tmp_path, treatment_count=counts, ineligible={excluded_key})
    )

    assert result["selection"]["status"] == "selected"
    assert result["selection"]["selected_step"] == 20
    cohort = result["complete_case_development_cohort"]
    assert cohort["image_count"] == 255
    assert cohort["object_count_band_image_counts"]["ten_or_more"] == 63
    step = result["milestone_summaries"]["20"]["intersection_over_union_thresholds"][
        "0.30"
    ]
    assert step["seeds"]["19"]["arms"]["broad"]["complete_case_image_count"] == 255
    assert len(step["seeds"]["19"]["arms"]["broad"]["object_count_bands"]) == 4
    assert step["seeds"]["19"][
        "object_count_band_broad_minus_concentrated_matched_annotated_owner_counts"
    ] == {
        "one_to_three": 128,
        "four_to_six": 128,
        "seven_to_nine": 128,
        "ten_or_more": 126,
    }
    full_health = result["full_development_health_by_step_seed_and_arm"]["31"]["23"][
        "concentrated"
    ]
    assert full_health["candidate_image_count"] == 256
    assert full_health["treatment_ineligible_image_ids"] == ["255"]


def test_retains_seed_sign_disagreement(tmp_path: Path) -> None:
    def counts(arm: str, seed: int, step: int, _threshold: str, _image: int) -> int:
        if step == 10 and arm == "broad":
            return 2 if seed == 19 else 0
        return 1

    result = select_shared_milestone(_write_ledgers(tmp_path, treatment_count=counts))
    summary = result["milestone_summaries"]["10"]["intersection_over_union_thresholds"][
        "0.30"
    ]

    assert (
        summary["seeds"]["19"]["broad_minus_concentrated_matched_annotated_owner_count"]
        == 256
    )
    assert (
        summary["seeds"]["23"]["broad_minus_concentrated_matched_annotated_owner_count"]
        == -256
    )
    assert summary["mean_broad_minus_concentrated_matched_annotated_owner_count"] == 0


def test_stops_when_mean_average_precision_is_required_for_tie(tmp_path: Path) -> None:
    result = select_shared_milestone(_write_ledgers(tmp_path))

    assert result["selection"] == {
        "status": "mean_average_precision_required_tie",
        "selected_step": None,
        "mean_average_precision_required_tied_steps": [10, 20, 30, 31],
        "lexicographic_policy": result["selection"]["lexicographic_policy"],
        "step_selection_keys": result["selection"]["step_selection_keys"],
    }


def test_paired_stratified_bootstrap_is_deterministic(tmp_path: Path) -> None:
    def counts(arm: str, _seed: int, step: int, _threshold: str, image: int) -> int:
        if arm == "broad" and step == 30:
            return 2 if image % 3 else 0
        return 1

    paths = _write_ledgers(tmp_path, treatment_count=counts)
    first = select_shared_milestone(paths)
    second = select_shared_milestone(paths)
    first_bootstrap = first["milestone_summaries"]["30"][
        "intersection_over_union_thresholds"
    ]["0.30"]["paired_image_stratified_bootstrap"]
    second_bootstrap = second["milestone_summaries"]["30"][
        "intersection_over_union_thresholds"
    ]["0.30"]["paired_image_stratified_bootstrap"]

    assert first_bootstrap == second_bootstrap
    assert first_bootstrap["random_seed"] == 20260723
    assert (
        first_bootstrap["percentile_interval_lower"] < first_bootstrap["point_estimate"]
    )
    assert (
        first_bootstrap["point_estimate"] < first_bootstrap["percentile_interval_upper"]
    )
