#!/usr/bin/env python3
"""Compare the frozen RP=1.00 and RP=1.10 heldout owner ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


POLICIES = {"rp1p00": 1.0, "rp1p10": 1.1}
ARMS = ("broad", "concentrated")
SEEDS = (19, 23)
THRESHOLDS = ("0.30", "0.50")
TREATMENT_PATTERN = re.compile(r"^(broad|concentrated)-seed(19|23)-step30$")
SCHEMA = "source_b16_treatment_owner_ledger.v1"


class ComparatorError(ValueError):
    """Raised when inputs do not satisfy the frozen comparison contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparatorError(f"{context} must be an object")
    return value


def _load(policy: str, path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparatorError(f"invalid JSON: {path}") from exc
    ledger = dict(_mapping(payload, str(path)))
    if ledger.get("schema_version") != SCHEMA:
        raise ComparatorError(f"unexpected ledger schema: {path}")
    treatments = _mapping(ledger.get("treatments"), f"{path}.treatments")
    if len(treatments) != 1:
        raise ComparatorError(f"ledger must contain one treatment: {path}")
    treatment_name, treatment_value = next(iter(treatments.items()))
    match = TREATMENT_PATTERN.fullmatch(str(treatment_name))
    if match is None:
        raise ComparatorError(f"unexpected treatment identity {treatment_name}: {path}")
    arm, seed_text = match.groups()
    treatment = _mapping(treatment_value, f"{path}.treatments.{treatment_name}")
    source_panel = _mapping(
        _mapping(ledger.get("inputs"), f"{path}.inputs").get("source_panel"),
        f"{path}.inputs.source_panel",
    )
    panel = _mapping(treatment.get("panel"), f"{path}.treatment.panel")
    for label, panel_value in (("source", source_panel), ("treatment", panel)):
        config = _mapping(
            panel_value.get("config_identity"), f"{path}.{label}.config_identity"
        )
        if float(config.get("repetition_penalty", -1)) != POLICIES[policy]:
            raise ComparatorError(f"{path} is not {policy} for {label}")
    rows_value = treatment.get("per_image")
    if not isinstance(rows_value, list) or not rows_value:
        raise ComparatorError(f"{path} has no per-image rows")
    rows: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(rows_value):
        row = _mapping(value, f"{path}.per_image[{index}]")
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in rows:
            raise ComparatorError(f"invalid or duplicate image identity in {path}")
        comparison = _mapping(
            row.get("owner_comparison"), f"{path}.{image_id}.owner_comparison"
        )
        by_iou_value = comparison.get("by_intersection_over_union")
        if comparison.get("eligible") is True:
            by_iou = _mapping(
                by_iou_value, f"{path}.{image_id}.by_intersection_over_union"
            )
            if set(by_iou) != set(THRESHOLDS):
                raise ComparatorError(f"unexpected IoU thresholds for {image_id}: {path}")
        elif by_iou_value is not None:
            raise ComparatorError(
                f"ineligible image {image_id} unexpectedly has owner comparisons: {path}"
            )
        rows[image_id] = row
    candidate = _mapping(
        _mapping(ledger.get("inputs"), f"{path}.inputs").get("candidate_jsonl"),
        f"{path}.inputs.candidate_jsonl",
    )
    return {
        "policy": policy,
        "arm": arm,
        "seed": int(seed_text),
        "name": str(treatment_name),
        "path": path,
        "sha256": _sha256(path),
        "candidate_sha256": candidate.get("sha256"),
        "rows": rows,
    }


def _owner_ids(row: Mapping[str, Any], surface: str, threshold: str) -> set[str]:
    comparison = _mapping(row["owner_comparison"], "owner_comparison")
    by_iou = _mapping(comparison["by_intersection_over_union"], "by_iou")
    entry = _mapping(by_iou[threshold], f"by_iou.{threshold}")
    key = f"{surface}_owner_ids"
    values = entry.get(key)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ComparatorError(f"{key} must be a string list")
    if len(values) != len(set(values)):
        raise ComparatorError(f"{key} contains duplicates")
    return set(values)


def _identity(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("image_id")),
        int(row.get("annotated_object_count", -1)),
        str(row.get("object_count_band")),
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stratified_bootstrap(
    image_ids: Sequence[str],
    bands: Mapping[str, str],
    statistic: Callable[[Sequence[str]], float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ComparatorError("bootstrap iterations must be positive")
    by_band: dict[str, list[str]] = {}
    for image_id in image_ids:
        by_band.setdefault(bands[image_id], []).append(image_id)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample: list[str] = []
        for band in sorted(by_band):
            values = by_band[band]
            sample.extend(rng.choice(values) for _ in values)
        estimates.append(statistic(sample))
    estimates.sort()
    return {
        "estimate": statistic(image_ids),
        "confidence_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "image_within_object_count_band",
    }


def _sum_counts(
    ledger: Mapping[str, Any],
    image_ids: Sequence[str],
    surface: str,
    threshold: str,
) -> int:
    return sum(
        len(_owner_ids(ledger["rows"][image_id], surface, threshold))
        for image_id in image_ids
    )


def _owner_union(
    ledger: Mapping[str, Any],
    image_ids: Sequence[str],
    surface: str,
    threshold: str,
) -> set[str]:
    result: set[str] = set()
    for image_id in image_ids:
        result.update(_owner_ids(ledger["rows"][image_id], surface, threshold))
    return result


def _delta_summary(left: set[str], right: set[str]) -> dict[str, Any]:
    gained = sorted(right - left)
    lost = sorted(left - right)
    return {
        "owner_count_delta": len(right) - len(left),
        "gained_owner_count": len(gained),
        "gained_owner_ids": gained,
        "lost_owner_count": len(lost),
        "lost_owner_ids": lost,
    }


def compare_ledgers(
    ledger_specs: Sequence[tuple[str, str | Path]],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260723,
) -> dict[str, Any]:
    if len(ledger_specs) != 8:
        raise ComparatorError("exactly eight ledgers are required")
    loaded = [_load(policy, path) for policy, path in ledger_specs]
    matrix: dict[tuple[str, str, int], dict[str, Any]] = {}
    for ledger in loaded:
        key = (ledger["policy"], ledger["arm"], ledger["seed"])
        if key in matrix:
            raise ComparatorError(f"duplicate matrix cell: {key}")
        matrix[key] = ledger
    expected = {
        (policy, arm, seed)
        for policy in POLICIES
        for arm in ARMS
        for seed in SEEDS
    }
    if set(matrix) != expected:
        raise ComparatorError(f"matrix mismatch: expected {sorted(expected)}, got {sorted(matrix)}")

    candidate_hashes = {ledger["candidate_sha256"] for ledger in loaded}
    if len(candidate_hashes) != 1 or None in candidate_hashes:
        raise ComparatorError("candidate JSONL identity differs across ledgers")
    image_sets = [set(ledger["rows"]) for ledger in loaded]
    if any(image_set != image_sets[0] for image_set in image_sets[1:]):
        raise ComparatorError("candidate image identities differ across ledgers")
    all_image_ids = sorted(image_sets[0])
    identities = {
        image_id: {_identity(ledger["rows"][image_id]) for ledger in loaded}
        for image_id in all_image_ids
    }
    if any(len(values) != 1 for values in identities.values()):
        raise ComparatorError("candidate object-count identity differs across ledgers")

    complete_ids = [
        image_id
        for image_id in all_image_ids
        if all(
            _mapping(ledger["rows"][image_id]["owner_comparison"], "owner_comparison").get(
                "eligible"
            )
            is True
            for ledger in loaded
        )
    ]
    if not complete_ids:
        raise ComparatorError("the common complete-case cohort is empty")
    bands = {
        image_id: str(loaded[0]["rows"][image_id]["object_count_band"])
        for image_id in complete_ids
    }

    # Source must be one baseline realization per policy, not arm/seed-specific.
    for policy in POLICIES:
        reference = matrix[(policy, ARMS[0], SEEDS[0])]
        for threshold in THRESHOLDS:
            for arm in ARMS:
                for seed in SEEDS:
                    other = matrix[(policy, arm, seed)]
                    for image_id in complete_ids:
                        if _owner_ids(reference["rows"][image_id], "source", threshold) != _owner_ids(
                            other["rows"][image_id], "source", threshold
                        ):
                            raise ComparatorError(
                                f"Source owner identity varies within {policy} at {threshold}"
                            )

    thresholds: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(THRESHOLDS):
        absolute: dict[str, Any] = {}
        broad_minus_concentrated: dict[str, Any] = {}
        direct_rp: dict[str, Any] = {}
        bootstrap: dict[str, Any] = {"direct_rp_effect": {}, "mean_seed_contrasts": {}}
        for policy in POLICIES:
            absolute[policy] = {}
            broad_minus_concentrated[policy] = {}
            for arm in ARMS:
                absolute[policy][arm] = {}
                for seed in SEEDS:
                    ledger = matrix[(policy, arm, seed)]
                    source = _owner_union(ledger, complete_ids, "source", threshold)
                    treatment = _owner_union(ledger, complete_ids, "treatment", threshold)
                    retained = source & treatment
                    gained = treatment - source
                    lost = source - treatment
                    absolute[policy][arm][str(seed)] = {
                        "source_matched_owner_count": len(source),
                        "treatment_matched_owner_count": len(treatment),
                        "retained_source_owner_count": len(retained),
                        "gained_annotated_owner_count": len(gained),
                        "lost_source_owner_count": len(lost),
                        "net_owner_delta": len(treatment) - len(source),
                    }
            for seed in SEEDS:
                broad = absolute[policy]["broad"][str(seed)]["treatment_matched_owner_count"]
                concentrated = absolute[policy]["concentrated"][str(seed)][
                    "treatment_matched_owner_count"
                ]
                broad_minus_concentrated[policy][str(seed)] = broad - concentrated

        rp00_source = matrix[("rp1p00", ARMS[0], SEEDS[0])]
        rp10_source = matrix[("rp1p10", ARMS[0], SEEDS[0])]
        direct_rp["source"] = _delta_summary(
            _owner_union(rp00_source, complete_ids, "source", threshold),
            _owner_union(rp10_source, complete_ids, "source", threshold),
        )
        surfaces: list[tuple[str, str, int, str]] = [("source", ARMS[0], SEEDS[0], "source")]
        for arm in ARMS:
            direct_rp[arm] = {}
            for seed in SEEDS:
                left = matrix[("rp1p00", arm, seed)]
                right = matrix[("rp1p10", arm, seed)]
                direct_rp[arm][str(seed)] = _delta_summary(
                    _owner_union(left, complete_ids, "treatment", threshold),
                    _owner_union(right, complete_ids, "treatment", threshold),
                )
                surfaces.append((f"{arm}-seed{seed}", arm, seed, "treatment"))

        for surface_index, (label, arm, seed, surface) in enumerate(surfaces):
            left = matrix[("rp1p00", arm, seed)]
            right = matrix[("rp1p10", arm, seed)]

            def direct_stat(sample: Sequence[str], *, left=left, right=right, surface=surface) -> float:
                return float(
                    _sum_counts(right, sample, surface, threshold)
                    - _sum_counts(left, sample, surface, threshold)
                )

            bootstrap["direct_rp_effect"][label] = _stratified_bootstrap(
                complete_ids,
                bands,
                direct_stat,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + threshold_index * 100 + surface_index,
            )

        for policy_index, policy in enumerate(POLICIES):

            def breadth_stat(sample: Sequence[str], *, policy=policy) -> float:
                deltas = [
                    _sum_counts(matrix[(policy, "broad", seed)], sample, "treatment", threshold)
                    - _sum_counts(
                        matrix[(policy, "concentrated", seed)], sample, "treatment", threshold
                    )
                    for seed in SEEDS
                ]
                return sum(deltas) / len(deltas)

            bootstrap["mean_seed_contrasts"][f"{policy}-broad-minus-concentrated"] = (
                _stratified_bootstrap(
                    complete_ids,
                    bands,
                    breadth_stat,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + threshold_index * 100 + 20 + policy_index,
                )
            )

        for arm_index, arm in enumerate(ARMS):

            def mean_rp_stat(sample: Sequence[str], *, arm=arm) -> float:
                deltas = [
                    _sum_counts(matrix[("rp1p10", arm, seed)], sample, "treatment", threshold)
                    - _sum_counts(matrix[("rp1p00", arm, seed)], sample, "treatment", threshold)
                    for seed in SEEDS
                ]
                return sum(deltas) / len(deltas)

            bootstrap["mean_seed_contrasts"][f"{arm}-rp1p10-minus-rp1p00"] = (
                _stratified_bootstrap(
                    complete_ids,
                    bands,
                    mean_rp_stat,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + threshold_index * 100 + 30 + arm_index,
                )
            )

        thresholds[threshold] = {
            "absolute_by_policy_arm_seed": absolute,
            "broad_minus_concentrated_treatment_owner_count_by_seed": (
                broad_minus_concentrated
            ),
            "direct_rp1p10_minus_rp1p00": direct_rp,
            "object_count_band_stratified_paired_bootstrap": bootstrap,
        }

    return {
        "schema_version": "repetition_penalty_owner_ledger_comparison.v1",
        "inputs": [
            {
                "policy": ledger["policy"],
                "arm": ledger["arm"],
                "seed": ledger["seed"],
                "path": str(ledger["path"]),
                "sha256": ledger["sha256"],
            }
            for ledger in sorted(
                loaded, key=lambda item: (item["policy"], item["arm"], item["seed"])
            )
        ],
        "candidate_jsonl_sha256": next(iter(candidate_hashes)),
        "cohort": {
            "candidate_image_count": len(all_image_ids),
            "common_complete_case_image_count": len(complete_ids),
            "excluded_image_count": len(all_image_ids) - len(complete_ids),
            "image_ids": complete_ids,
            "object_count_band_counts": {
                band: sum(value == band for value in bands.values())
                for band in sorted(set(bands.values()))
            },
        },
        "thresholds": thresholds,
    }


def _parse_ledger(value: str) -> tuple[str, str]:
    try:
        policy, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use POLICY=PATH") from exc
    if policy not in POLICIES or not path:
        raise argparse.ArgumentTypeError("POLICY must be rp1p00 or rp1p10")
    return policy, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", type=_parse_ledger, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    result = compare_ledgers(
        args.ledger,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}))


if __name__ == "__main__":
    main()
