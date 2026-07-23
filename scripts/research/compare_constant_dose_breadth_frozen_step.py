#!/usr/bin/env python3
"""Compare broad and concentrated treatments at one frozen training step."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


ARMS = ("broad", "concentrated")
SEEDS = (19, 23)
INTERSECTION_OVER_UNION_THRESHOLDS = ("0.30", "0.50")
EXPECTED_ANALYZER_SCHEMA_VERSION = "source_b16_treatment_owner_ledger.v1"
OUTPUT_SCHEMA_VERSION = "constant_dose_breadth_frozen_step_comparison.v1"
BOOTSTRAP_RANDOM_SEED = 20260723
BOOTSTRAP_REPLICATE_COUNT = 2000

LedgerKey = tuple[str, int]


class FrozenStepComparisonError(ValueError):
    """Raised when four ledgers cannot support a paired comparison."""


def _expected_keys() -> set[LedgerKey]:
    return {(arm, seed) for arm in ARMS for seed in SEEDS}


def _load_ledgers(
    paths: Mapping[LedgerKey, str | Path],
) -> dict[LedgerKey, dict[str, Any]]:
    if set(paths) != _expected_keys():
        raise FrozenStepComparisonError(
            "ledger matrix mismatch; "
            f"missing={sorted(_expected_keys() - set(paths))}, "
            f"unexpected={sorted(set(paths) - _expected_keys())}"
        )
    ledgers: dict[LedgerKey, dict[str, Any]] = {}
    for key in sorted(paths):
        path = Path(paths[key]).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FrozenStepComparisonError(
                f"cannot load ledger {path}: {error}"
            ) from error
        if payload.get("schema_version") != EXPECTED_ANALYZER_SCHEMA_VERSION:
            raise FrozenStepComparisonError(
                f"unsupported analyzer schema for {key}: {payload.get('schema_version')!r}"
            )
        treatments = payload.get("treatments")
        if not isinstance(treatments, dict) or len(treatments) != 1:
            raise FrozenStepComparisonError(
                f"ledger {key} must contain exactly one treatment"
            )
        payload["_comparison_input_path"] = str(path)
        payload["_comparison_treatment_name"] = next(iter(treatments))
        ledgers[key] = payload
    return ledgers


def _treatment(ledger: Mapping[str, Any]) -> Mapping[str, Any]:
    return ledger["treatments"][ledger["_comparison_treatment_name"]]


def _index_records(
    ledger: Mapping[str, Any], key: LedgerKey
) -> dict[str, Mapping[str, Any]]:
    records = _treatment(ledger).get("per_image")
    if not isinstance(records, list):
        raise FrozenStepComparisonError(f"ledger {key} lacks per_image records")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        image_id = str(record.get("image_id"))
        if image_id in indexed:
            raise FrozenStepComparisonError(
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
        raise FrozenStepComparisonError(
            "ledger inputs lack candidate or Source panel identity"
        )
    indexed = {key: _index_records(ledger, key) for key, ledger in ledgers.items()}
    image_ids = set(indexed[first_key])
    candidate_count = candidate_identity.get("image_count")
    if not isinstance(candidate_count, int) or candidate_count <= 0:
        raise FrozenStepComparisonError(
            "candidate image count must be a positive integer"
        )
    if len(image_ids) != candidate_count:
        raise FrozenStepComparisonError(
            "reference ledger image count does not match candidate identity"
        )
    for key, ledger in ledgers.items():
        inputs = ledger.get("inputs", {})
        if inputs.get("candidate_jsonl") != candidate_identity:
            raise FrozenStepComparisonError(
                f"candidate JSONL identity mismatch for {key}"
            )
        if inputs.get("source_panel") != source_panel_identity:
            raise FrozenStepComparisonError(f"Source panel identity mismatch for {key}")
        if set(indexed[key]) != image_ids:
            raise FrozenStepComparisonError(f"image denominator mismatch for {key}")
    for image_id in sorted(image_ids):
        reference = indexed[first_key][image_id]
        reference_annotation = (
            reference.get("annotated_object_count"),
            reference.get("object_count_band"),
        )
        reference_source = reference.get("source")
        for key in sorted(indexed):
            record = indexed[key][image_id]
            if record.get("source") != reference_source:
                raise FrozenStepComparisonError(
                    f"Source per-image surface mismatch for image {image_id} in {key}"
                )
            annotation = (
                record.get("annotated_object_count"),
                record.get("object_count_band"),
            )
            if annotation != reference_annotation:
                raise FrozenStepComparisonError(
                    f"annotation band identity mismatch for image {image_id} in {key}"
                )
    return indexed, sorted(image_ids)


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
) -> dict[str, Any]:
    counts = {
        "source_matched_annotated_owner_count": 0,
        "treatment_matched_annotated_owner_count": 0,
        "retained_source_owner_count": 0,
        "lost_source_owner_count": 0,
        "gained_annotated_owner_count": 0,
        "net_owner_delta": 0,
        "source_duplicate_candidate_count": 0,
        "source_review_needed_prediction_count": 0,
        "treatment_duplicate_candidate_count": 0,
        "treatment_review_needed_prediction_count": 0,
    }
    for record in records:
        comparison = record["owner_comparison"]["by_intersection_over_union"][threshold]
        source = record["source"]["matching_by_intersection_over_union"][threshold]
        treatment = record["treatment"]["matching_by_intersection_over_union"][
            threshold
        ]
        counts["source_matched_annotated_owner_count"] += len(
            comparison["source_owner_ids"]
        )
        counts["treatment_matched_annotated_owner_count"] += len(
            comparison["treatment_owner_ids"]
        )
        counts["retained_source_owner_count"] += len(
            comparison["retained_source_owner_ids"]
        )
        counts["lost_source_owner_count"] += len(comparison["lost_source_owner_ids"])
        counts["gained_annotated_owner_count"] += len(
            comparison["gained_annotated_owner_ids"]
        )
        counts["net_owner_delta"] += int(comparison["net_owner_delta"])
        counts["source_duplicate_candidate_count"] += int(
            source["duplicate_prediction_count"]
        )
        counts["source_review_needed_prediction_count"] += int(
            source["review_needed_prediction_count"]
        )
        counts["treatment_duplicate_candidate_count"] += int(
            treatment["duplicate_prediction_count"]
        )
        counts["treatment_review_needed_prediction_count"] += int(
            treatment["review_needed_prediction_count"]
        )
    return {"complete_case_image_count": len(records), **counts}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stratified_bootstrap(
    values_by_band: Mapping[str, Sequence[float]], *, estimand: str
) -> dict[str, Any]:
    generator = random.Random(BOOTSTRAP_RANDOM_SEED)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATE_COUNT):
        replicate = 0.0
        for band in sorted(values_by_band):
            values = values_by_band[band]
            replicate += sum(generator.choice(values) for _ in values)
        replicates.append(replicate)
    return {
        "estimand": estimand,
        "stratification": "object_count_band",
        "random_seed": BOOTSTRAP_RANDOM_SEED,
        "replicate_count": BOOTSTRAP_REPLICATE_COUNT,
        "point_estimate": sum(sum(values) for values in values_by_band.values()),
        "percentile_interval_probability": 0.95,
        "percentile_interval_lower": _percentile(replicates, 0.025),
        "percentile_interval_upper": _percentile(replicates, 0.975),
    }


def _mean_seed_arm_net_bootstrap(
    indexed: Mapping[LedgerKey, Mapping[str, Mapping[str, Any]]],
    cohort: Sequence[str],
    arm: str,
    threshold: str,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for image_id in cohort:
        records = [indexed[(arm, seed)][image_id] for seed in SEEDS]
        per_seed = [
            int(
                record["owner_comparison"]["by_intersection_over_union"][threshold][
                    "net_owner_delta"
                ]
            )
            for record in records
        ]
        band = str(records[0]["object_count_band"])
        values[band].append(0.5 * sum(per_seed))
    return _stratified_bootstrap(
        values,
        estimand=(
            f"sum over complete-case images of the mean seed treatment-minus-Source "
            f"matched annotated owner count for the {arm} arm"
        ),
    )


def _mean_seed_arm_difference_bootstrap(
    indexed: Mapping[LedgerKey, Mapping[str, Mapping[str, Any]]],
    cohort: Sequence[str],
    threshold: str,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for image_id in cohort:
        differences = []
        reference = indexed[("broad", SEEDS[0])][image_id]
        for seed in SEEDS:
            broad = indexed[("broad", seed)][image_id]
            concentrated = indexed[("concentrated", seed)][image_id]
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
            differences.append(broad_count - concentrated_count)
        band = str(reference["object_count_band"])
        values[band].append(0.5 * sum(differences))
    return _stratified_bootstrap(
        values,
        estimand=(
            "sum over complete-case images of the mean seed broad-minus-concentrated "
            "matched annotated owner count"
        ),
    )


def compare_frozen_step(
    ledger_paths: Mapping[LedgerKey, str | Path],
) -> dict[str, Any]:
    """Validate and compare one frozen training step across arms and seeds."""
    ledgers = _load_ledgers(ledger_paths)
    indexed, image_ids = _validate_shared_inputs(ledgers)
    cohort = _complete_case_cohort(indexed, image_ids)
    if not cohort:
        raise FrozenStepComparisonError("complete-case cohort is empty")
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
    thresholds: dict[str, Any] = {}
    for threshold in INTERSECTION_OVER_UNION_THRESHOLDS:
        arms: dict[str, Any] = {}
        per_seed_differences: dict[str, int] = {}
        per_band_seed_differences: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            seed_summaries: dict[str, Any] = {}
            for seed in SEEDS:
                records = [indexed[(arm, seed)][image_id] for image_id in cohort]
                summary = _sum_records(records, threshold)
                summary["object_count_bands"] = {
                    band: _sum_records(
                        [indexed[(arm, seed)][image_id] for image_id in band_ids],
                        threshold,
                    )
                    for band, band_ids in cohort_by_band.items()
                }
                seed_summaries[str(seed)] = summary
            arms[arm] = {
                "seeds": seed_summaries,
                "mean_seed_arm_versus_source_net_owner_delta_bootstrap": (
                    _mean_seed_arm_net_bootstrap(indexed, cohort, arm, threshold)
                ),
            }
        for seed in SEEDS:
            broad = arms["broad"]["seeds"][str(seed)]
            concentrated = arms["concentrated"]["seeds"][str(seed)]
            per_seed_differences[str(seed)] = (
                broad["treatment_matched_annotated_owner_count"]
                - concentrated["treatment_matched_annotated_owner_count"]
            )
            per_band_seed_differences[str(seed)] = {
                band: (
                    broad["object_count_bands"][band][
                        "treatment_matched_annotated_owner_count"
                    ]
                    - concentrated["object_count_bands"][band][
                        "treatment_matched_annotated_owner_count"
                    ]
                )
                for band in bands
            }
        thresholds[threshold] = {
            "arms": arms,
            "broad_minus_concentrated_matched_annotated_owner_count_by_seed": per_seed_differences,
            "mean_broad_minus_concentrated_matched_annotated_owner_count": (
                sum(per_seed_differences.values()) / len(per_seed_differences)
            ),
            "object_count_band_broad_minus_concentrated_matched_annotated_owner_count_by_seed": per_band_seed_differences,
            "mean_seed_broad_minus_concentrated_bootstrap": (
                _mean_seed_arm_difference_bootstrap(indexed, cohort, threshold)
            ),
        }
    full_health = {
        arm: {
            str(seed): {
                "input_path": ledgers[(arm, seed)]["_comparison_input_path"],
                "candidate_image_count": _treatment(ledgers[(arm, seed)])["aggregate"][
                    "candidate_image_count"
                ],
                "source_ineligible_image_count": _treatment(ledgers[(arm, seed)])[
                    "aggregate"
                ]["source_ineligible_image_count"],
                "source_ineligible_image_ids": _treatment(ledgers[(arm, seed)])[
                    "aggregate"
                ]["source_ineligible_image_ids"],
                "treatment_ineligible_image_count": _treatment(ledgers[(arm, seed)])[
                    "aggregate"
                ]["treatment_ineligible_image_count"],
                "treatment_ineligible_image_ids": _treatment(ledgers[(arm, seed)])[
                    "aggregate"
                ]["treatment_ineligible_image_ids"],
                "source_panel_health": _treatment(ledgers[(arm, seed)])["aggregate"][
                    "source_panel_health"
                ],
                "treatment_panel_health": _treatment(ledgers[(arm, seed)])["aggregate"][
                    "treatment_panel_health"
                ],
            }
            for seed in SEEDS
        }
        for arm in ARMS
    }
    first = ledgers[sorted(ledgers)[0]]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "policy": {
            "comparison_denominator": "one global four-ledger complete-case cohort",
            "intersection_over_union_thresholds": [0.30, 0.50],
        },
        "validated_inputs": {
            "ledger_count": len(ledgers),
            "candidate_jsonl": first["inputs"]["candidate_jsonl"],
            "source_panel": first["inputs"]["source_panel"],
            "full_candidate_image_count": len(image_ids),
        },
        "complete_case_cohort": {
            "image_count": len(cohort),
            "excluded_image_count": len(image_ids) - len(cohort),
            "image_ids": cohort,
            "object_count_band_image_counts": {
                band: len(ids) for band, ids in cohort_by_band.items()
            },
        },
        "full_candidate_health_by_arm_and_seed": full_health,
        "intersection_over_union_thresholds": thresholds,
    }


def _ledger_argument(value: str) -> tuple[LedgerKey, Path]:
    identity, separator, raw_path = value.partition("=")
    parts = [part.strip() for part in identity.split(",")]
    if not separator or len(parts) != 2 or not raw_path.strip():
        raise argparse.ArgumentTypeError("ledger must be ARM,SEED=PATH")
    try:
        key = (parts[0], int(parts[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ledger seed must be an integer") from error
    return key, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=_ledger_argument,
        action="append",
        required=True,
        metavar="ARM,SEED=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths: dict[LedgerKey, Path] = {}
    for key, path in args.ledger:
        if key in paths:
            raise FrozenStepComparisonError(f"duplicate ledger argument for {key}")
        paths[key] = path
    result = compare_frozen_step(paths)
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
