#!/usr/bin/env python3
"""Select one shared constant-dose breadth-screen training milestone."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


ARMS = ("broad", "concentrated")
SEEDS = (19, 23)
STEPS = (10, 20, 30, 31)
INTERSECTION_OVER_UNION_THRESHOLDS = ("0.30", "0.50")
EXPECTED_ANALYZER_SCHEMA_VERSION = "source_b16_treatment_owner_ledger.v1"
OUTPUT_SCHEMA_VERSION = "constant_dose_breadth_shared_milestone_selection.v1"
EXPECTED_DEVELOPMENT_IMAGE_COUNT = 256
BOOTSTRAP_RANDOM_SEED = 20260723
BOOTSTRAP_REPLICATE_COUNT = 2000

LedgerKey = tuple[str, int, int]


class SharedMilestoneSelectionError(ValueError):
    """Raised when ledgers cannot support one shared milestone selection."""


def _expected_keys() -> set[LedgerKey]:
    return {(arm, seed, step) for arm in ARMS for seed in SEEDS for step in STEPS}


def _load_ledgers(
    paths: Mapping[LedgerKey, str | Path],
) -> dict[LedgerKey, dict[str, Any]]:
    observed = set(paths)
    expected = _expected_keys()
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise SharedMilestoneSelectionError(
            f"ledger matrix mismatch; missing={missing}, unexpected={unexpected}"
        )
    loaded: dict[LedgerKey, dict[str, Any]] = {}
    for key in sorted(paths):
        path = Path(paths[key]).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SharedMilestoneSelectionError(
                f"cannot load ledger {path}: {error}"
            ) from error
        if payload.get("schema_version") != EXPECTED_ANALYZER_SCHEMA_VERSION:
            raise SharedMilestoneSelectionError(
                f"unsupported analyzer schema for {key}: {payload.get('schema_version')!r}"
            )
        arm, seed, step = key
        expected_treatment_name = f"{arm}-seed{seed}-step{step}"
        treatments = payload.get("treatments")
        if not isinstance(treatments, dict) or set(treatments) != {
            expected_treatment_name
        }:
            raise SharedMilestoneSelectionError(
                f"ledger {key} must contain treatment {expected_treatment_name!r} only"
            )
        payload["_selection_input_path"] = str(path)
        payload["_selection_treatment_name"] = expected_treatment_name
        loaded[key] = payload
    return loaded


def _treatment(ledger: Mapping[str, Any]) -> Mapping[str, Any]:
    return ledger["treatments"][ledger["_selection_treatment_name"]]


def _index_records(
    ledger: Mapping[str, Any], key: LedgerKey
) -> dict[str, Mapping[str, Any]]:
    records = _treatment(ledger).get("per_image")
    if not isinstance(records, list):
        raise SharedMilestoneSelectionError(f"ledger {key} lacks per_image records")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        if image_id in indexed:
            raise SharedMilestoneSelectionError(
                f"duplicate image {image_id} in ledger {key}"
            )
        indexed[image_id] = record
    return indexed


def _validate_shared_inputs(
    ledgers: Mapping[LedgerKey, Mapping[str, Any]],
) -> tuple[dict[LedgerKey, dict[str, Mapping[str, Any]]], list[str]]:
    first_key = sorted(ledgers)[0]
    first = ledgers[first_key]
    candidate_identity = first.get("inputs", {}).get("candidate_jsonl")
    source_panel_identity = first.get("inputs", {}).get("source_panel")
    if not isinstance(candidate_identity, dict) or not isinstance(
        source_panel_identity, dict
    ):
        raise SharedMilestoneSelectionError(
            "ledger inputs lack candidate or Source panel identity"
        )

    indexed = {key: _index_records(ledger, key) for key, ledger in ledgers.items()}
    first_image_ids = set(indexed[first_key])
    if (
        candidate_identity.get("image_count") != EXPECTED_DEVELOPMENT_IMAGE_COUNT
        or len(first_image_ids) != EXPECTED_DEVELOPMENT_IMAGE_COUNT
    ):
        raise SharedMilestoneSelectionError(
            "development ledger must contain exactly 256 candidate images"
        )
    for key, ledger in ledgers.items():
        inputs = ledger.get("inputs", {})
        if inputs.get("candidate_jsonl") != candidate_identity:
            raise SharedMilestoneSelectionError(
                f"candidate JSONL identity mismatch for {key}"
            )
        if inputs.get("source_panel") != source_panel_identity:
            raise SharedMilestoneSelectionError(
                f"Source panel identity mismatch for {key}"
            )
        if set(indexed[key]) != first_image_ids:
            raise SharedMilestoneSelectionError(f"image denominator mismatch for {key}")

    for image_id in sorted(first_image_ids):
        reference = indexed[first_key][image_id]
        reference_source = reference.get("source")
        reference_annotation = {
            "annotated_object_count": reference.get("annotated_object_count"),
            "object_count_band": reference.get("object_count_band"),
        }
        for key in sorted(ledgers):
            record = indexed[key][image_id]
            if record.get("source") != reference_source:
                raise SharedMilestoneSelectionError(
                    f"Source per-image health or owners mismatch for image {image_id} in {key}"
                )
            observed_annotation = {
                "annotated_object_count": record.get("annotated_object_count"),
                "object_count_band": record.get("object_count_band"),
            }
            if observed_annotation != reference_annotation:
                raise SharedMilestoneSelectionError(
                    f"annotation band identity mismatch for image {image_id} in {key}"
                )
    return indexed, sorted(first_image_ids)


def _complete_case_cohort(
    indexed: Mapping[LedgerKey, Mapping[str, Mapping[str, Any]]],
    image_ids: Sequence[str],
) -> list[str]:
    return [
        image_id
        for image_id in image_ids
        if all(
            bool(indexed[key][image_id]["owner_comparison"]["eligible"])
            for key in sorted(indexed)
        )
    ]


def _sum_records(
    records: Sequence[Mapping[str, Any]], threshold: str
) -> dict[str, int]:
    matched_owner_count = 0
    retained_source_owner_count = 0
    lost_source_owner_count = 0
    gained_annotated_owner_count = 0
    net_owner_delta = 0
    treatment_duplicate_candidate_count = 0
    for record in records:
        comparison = record["owner_comparison"]["by_intersection_over_union"][threshold]
        treatment_matching = record["treatment"]["matching_by_intersection_over_union"][
            threshold
        ]
        matched_owner_count += len(comparison["treatment_owner_ids"])
        retained_source_owner_count += len(comparison["retained_source_owner_ids"])
        lost_source_owner_count += len(comparison["lost_source_owner_ids"])
        gained_annotated_owner_count += len(comparison["gained_annotated_owner_ids"])
        net_owner_delta += int(comparison["net_owner_delta"])
        treatment_duplicate_candidate_count += int(
            treatment_matching["duplicate_prediction_count"]
        )
    return {
        "complete_case_image_count": len(records),
        "matched_annotated_owner_count": matched_owner_count,
        "retained_source_owner_count": retained_source_owner_count,
        "lost_source_owner_count": lost_source_owner_count,
        "gained_annotated_owner_count": gained_annotated_owner_count,
        "net_owner_delta": net_owner_delta,
        "treatment_duplicate_candidate_count": treatment_duplicate_candidate_count,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_stratified_bootstrap(
    *,
    indexed: Mapping[LedgerKey, Mapping[str, Mapping[str, Any]]],
    cohort: Sequence[str],
    step: int,
    threshold: str,
) -> dict[str, Any]:
    values_by_band: dict[str, list[float]] = defaultdict(list)
    for image_id in cohort:
        seed_differences = []
        for seed in SEEDS:
            broad = indexed[("broad", seed, step)][image_id]
            concentrated = indexed[("concentrated", seed, step)][image_id]
            broad_count = len(
                broad["owner_comparison"]["by_intersection_over_union"][threshold][
                    "treatment_owner_ids"
                ]
            )
            concentrated_count = len(
                concentrated["owner_comparison"]["by_intersection_over_union"][
                    threshold
                ]["treatment_owner_ids"]
            )
            seed_differences.append(broad_count - concentrated_count)
        band = str(indexed[("broad", SEEDS[0], step)][image_id]["object_count_band"])
        values_by_band[band].append(0.5 * sum(seed_differences))
    if not values_by_band:
        raise SharedMilestoneSelectionError("complete-case development cohort is empty")
    random_generator = random.Random(BOOTSTRAP_RANDOM_SEED)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATE_COUNT):
        replicate = 0.0
        for band in sorted(values_by_band):
            values = values_by_band[band]
            replicate += sum(random_generator.choice(values) for _ in values)
        replicates.append(replicate)
    return {
        "estimand": (
            "sum over complete-case development images of one half times the sum "
            "of broad-minus-concentrated matched annotated owner counts for seeds 19 and 23"
        ),
        "stratification": "object_count_band",
        "random_seed": BOOTSTRAP_RANDOM_SEED,
        "replicate_count": BOOTSTRAP_REPLICATE_COUNT,
        "point_estimate": sum(sum(values) for values in values_by_band.values()),
        "percentile_interval_probability": 0.95,
        "percentile_interval_lower": _percentile(replicates, 0.025),
        "percentile_interval_upper": _percentile(replicates, 0.975),
    }


def _full_development_health(
    ledgers: Mapping[LedgerKey, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in STEPS:
        step_result: dict[str, Any] = {}
        for seed in SEEDS:
            seed_result: dict[str, Any] = {}
            for arm in ARMS:
                key = (arm, seed, step)
                aggregate = _treatment(ledgers[key])["aggregate"]
                seed_result[arm] = {
                    "input_path": ledgers[key]["_selection_input_path"],
                    "candidate_image_count": aggregate["candidate_image_count"],
                    "source_ineligible_image_count": aggregate[
                        "source_ineligible_image_count"
                    ],
                    "source_ineligible_image_ids": aggregate[
                        "source_ineligible_image_ids"
                    ],
                    "treatment_ineligible_image_count": aggregate[
                        "treatment_ineligible_image_count"
                    ],
                    "treatment_ineligible_image_ids": aggregate[
                        "treatment_ineligible_image_ids"
                    ],
                    "source_panel_health": aggregate["source_panel_health"],
                    "treatment_panel_health": aggregate["treatment_panel_health"],
                }
            step_result[str(seed)] = seed_result
        result[str(step)] = step_result
    return result


def select_shared_milestone(
    ledger_paths: Mapping[LedgerKey, str | Path],
) -> dict[str, Any]:
    """Validate 16 owner ledgers and select one common training step."""
    ledgers = _load_ledgers(ledger_paths)
    indexed, image_ids = _validate_shared_inputs(ledgers)
    cohort = _complete_case_cohort(indexed, image_ids)
    if not cohort:
        raise SharedMilestoneSelectionError("complete-case development cohort is empty")
    reference_key = sorted(indexed)[0]
    bands = sorted(
        {
            str(indexed[reference_key][image_id]["object_count_band"])
            for image_id in image_ids
        }
    )
    cohort_by_band = {
        band: [
            image_id
            for image_id in cohort
            if indexed[reference_key][image_id]["object_count_band"] == band
        ]
        for band in bands
    }

    milestone_summaries: dict[str, Any] = {}
    selection_keys: dict[int, tuple[float, float, int]] = {}
    for step in STEPS:
        threshold_summaries: dict[str, Any] = {}
        for threshold in INTERSECTION_OVER_UNION_THRESHOLDS:
            seeds: dict[str, Any] = {}
            broad_minus_concentrated: list[int] = []
            duplicate_total = 0
            for seed in SEEDS:
                arm_summaries: dict[str, Any] = {}
                for arm in ARMS:
                    records = [
                        indexed[(arm, seed, step)][image_id] for image_id in cohort
                    ]
                    summary = _sum_records(records, threshold)
                    summary["object_count_bands"] = {
                        band: _sum_records(
                            [
                                indexed[(arm, seed, step)][image_id]
                                for image_id in band_ids
                            ],
                            threshold,
                        )
                        for band, band_ids in cohort_by_band.items()
                    }
                    arm_summaries[arm] = summary
                    duplicate_total += summary["treatment_duplicate_candidate_count"]
                difference = (
                    arm_summaries["broad"]["matched_annotated_owner_count"]
                    - arm_summaries["concentrated"]["matched_annotated_owner_count"]
                )
                band_differences = {
                    band: (
                        arm_summaries["broad"]["object_count_bands"][band][
                            "matched_annotated_owner_count"
                        ]
                        - arm_summaries["concentrated"]["object_count_bands"][band][
                            "matched_annotated_owner_count"
                        ]
                    )
                    for band in cohort_by_band
                }
                broad_minus_concentrated.append(difference)
                seeds[str(seed)] = {
                    "arms": arm_summaries,
                    "broad_minus_concentrated_matched_annotated_owner_count": difference,
                    "object_count_band_broad_minus_concentrated_matched_annotated_owner_counts": band_differences,
                }
            threshold_summaries[threshold] = {
                "seeds": seeds,
                "mean_broad_minus_concentrated_matched_annotated_owner_count": (
                    sum(broad_minus_concentrated) / len(broad_minus_concentrated)
                ),
                "total_treatment_duplicate_candidate_count_across_arms_and_seeds": duplicate_total,
                "paired_image_stratified_bootstrap": _paired_stratified_bootstrap(
                    indexed=indexed, cohort=cohort, step=step, threshold=threshold
                ),
            }
        milestone_summaries[str(step)] = {
            "intersection_over_union_thresholds": threshold_summaries
        }
        selection_keys[step] = (
            -threshold_summaries["0.30"][
                "mean_broad_minus_concentrated_matched_annotated_owner_count"
            ],
            -threshold_summaries["0.50"][
                "mean_broad_minus_concentrated_matched_annotated_owner_count"
            ],
            threshold_summaries["0.30"][
                "total_treatment_duplicate_candidate_count_across_arms_and_seeds"
            ],
        )
    best_key = min(selection_keys.values())
    tied_steps = [step for step in STEPS if selection_keys[step] == best_key]
    if len(tied_steps) == 1:
        selection: dict[str, Any] = {
            "status": "selected",
            "selected_step": tied_steps[0],
            "mean_average_precision_required_tied_steps": [],
        }
    else:
        selection = {
            "status": "mean_average_precision_required_tie",
            "selected_step": None,
            "mean_average_precision_required_tied_steps": tied_steps,
        }
    selection["lexicographic_policy"] = [
        "maximize mean broad-minus-concentrated matched annotated owner count at intersection over union 0.30",
        "maximize mean broad-minus-concentrated matched annotated owner count at intersection over union 0.50",
        "minimize total treatment duplicate candidate count at intersection over union 0.30 across both arms and seeds",
        "require mean average precision to resolve any remaining tie",
    ]
    selection["step_selection_keys"] = {
        str(step): {
            "mean_intersection_over_union_0.30_difference": -key[0],
            "mean_intersection_over_union_0.50_difference": -key[1],
            "intersection_over_union_0.30_treatment_duplicate_candidate_count": key[2],
        }
        for step, key in selection_keys.items()
    }
    first = ledgers[sorted(ledgers)[0]]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "policy": {
            "development_denominator": "one global 16-ledger complete-case cohort",
            "selected_owner_or_nonselected_owner_analysis_included": False,
            "intersection_over_union_thresholds": [0.30, 0.50],
        },
        "validated_inputs": {
            "ledger_count": len(ledgers),
            "candidate_jsonl": first["inputs"]["candidate_jsonl"],
            "source_panel": first["inputs"]["source_panel"],
            "full_development_image_count": len(image_ids),
        },
        "complete_case_development_cohort": {
            "image_count": len(cohort),
            "excluded_image_count": len(image_ids) - len(cohort),
            "image_ids": cohort,
            "object_count_band_image_counts": {
                band: len(ids) for band, ids in cohort_by_band.items()
            },
        },
        "full_development_health_by_step_seed_and_arm": _full_development_health(
            ledgers
        ),
        "milestone_summaries": milestone_summaries,
        "selection": selection,
    }


def _ledger_argument(value: str) -> tuple[LedgerKey, Path]:
    identity, separator, raw_path = value.partition("=")
    parts = [part.strip() for part in identity.split(",")]
    if not separator or len(parts) != 3 or not raw_path.strip():
        raise argparse.ArgumentTypeError("ledger must be ARM,SEED,STEP=PATH")
    try:
        key = (parts[0], int(parts[1]), int(parts[2]))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ledger seed and step must be integers"
        ) from error
    return key, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=_ledger_argument,
        action="append",
        required=True,
        metavar="ARM,SEED,STEP=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths: dict[LedgerKey, Path] = {}
    for key, path in args.ledger:
        if key in paths:
            raise SharedMilestoneSelectionError(f"duplicate ledger argument for {key}")
        paths[key] = path
    result = select_shared_milestone(paths)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
