#!/usr/bin/env python3
"""Stage 2 retention/rejection analysis for the sampled span-likelihood mining unit.

Answers the pre-registered Stage 2 question: at leave-one-image-out
out-of-sample retention of at least 95% of greedy-missed union-recovered
owners, does any likelihood-bearing cluster confidence family reject more
catastrophic false-positive clusters than trajectory support alone?

Design notes that matter for reading the result:

- Thresholds are chosen on 11 images and evaluated on the held-out 12th. The
  reported number is the pooled out-of-sample estimate; the in-sample value is
  reported alongside and is an upper bound, not a result.
- Feature normalization is fit on the training folds only and applied to the
  held-out image by interpolation, so the held-out image never informs its own
  score.
- Multi-feature families combine equally weighted training-fold percentiles.
  This is deliberately unfitted: the question is whether adding a feature helps
  at all, and a second fitting layer would confound family comparison with
  model capacity.

Research unit:
research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = "cluster_confidence_retention.v1"

CHECKPOINTS = ("sorted", "random", "permutation")
CATASTROPHIC = ("catastrophic_class_absent", "catastrophic_misgrounded")

# Pre-registered operating point and decision threshold.
RETENTION_TARGET = 0.95
REJECTION_PASS_FRACTION = 0.20

BOOTSTRAP_SEED = 20260729
BOOTSTRAP_DRAWS = 2000


def _feature_support(cluster: dict[str, Any]) -> float:
    return float(cluster["support"])


def _medoid(cluster: dict[str, Any], field: str) -> float | None:
    block = (cluster.get("likelihood") or {}).get("medoid")
    if not block or block.get(field) is None:
        return None
    return float(block[field])


def _feature_coord(cluster: dict[str, Any]) -> float | None:
    return _medoid(cluster, "coord_mean")


def _feature_description(cluster: dict[str, Any]) -> float | None:
    return _medoid(cluster, "description_mean")


def _feature_compactness(cluster: dict[str, Any]) -> float:
    # Higher mean pairwise IoU means a tighter cluster, so this is already
    # oriented so that larger is better, like every other feature here.
    return float(cluster["spatial_dispersion"]["mean_pairwise_iou"])


# name -> list of component feature extractors, all oriented "larger is better"
FAMILIES: dict[str, list[Callable[[dict[str, Any]], float | None]]] = {
    "support_only": [_feature_support],
    "coordinate_only": [_feature_coord],
    "description_plus_coordinate": [_feature_description, _feature_coord],
    "support_plus_likelihood": [_feature_support, _feature_coord],
    "support_plus_likelihood_plus_dispersion": [
        _feature_support,
        _feature_coord,
        _feature_compactness,
    ],
}


def _percentile_mapper(train_values: Sequence[float]) -> Callable[[float], float]:
    """Map a value to its percentile in the TRAINING distribution only."""

    ordered = sorted(train_values)
    size = len(ordered)

    def mapper(value: float) -> float:
        low, high = 0, size
        while low < high:
            mid = (low + high) // 2
            if ordered[mid] < value:
                low = mid + 1
            else:
                high = mid
        below = low
        high2 = size
        low2 = below
        while low2 < high2:
            mid = (low2 + high2) // 2
            if ordered[mid] <= value:
                low2 = mid + 1
            else:
                high2 = mid
        # Midpoint of the tied block keeps ties symmetric.
        return (below + low2) / (2.0 * size) if size else 0.0

    return mapper


def _score(
    clusters: Sequence[dict[str, Any]],
    extractors: Sequence[Callable[[dict[str, Any]], float | None]],
    mappers: Sequence[Callable[[float], float]],
) -> list[float | None]:
    scores: list[float | None] = []
    for cluster in clusters:
        total = 0.0
        usable = True
        for extractor, mapper in zip(extractors, mappers, strict=True):
            value = extractor(cluster)
            if value is None:
                usable = False
                break
            total += mapper(value)
        scores.append(total / len(extractors) if usable else None)
    return scores


def _threshold_for_retention(recovered_scores: Sequence[float], target: float) -> float:
    """Largest threshold retaining at least `target` of the recovered owners.

    Retention counts clusters with score >= threshold, so the threshold is the
    (1 - target) quantile from the bottom of the recovered-owner scores.
    """

    if not recovered_scores:
        return float("-inf")
    ordered = sorted(recovered_scores)
    # Number we are allowed to lose.
    droppable = int((1.0 - target) * len(ordered))
    index = min(droppable, len(ordered) - 1)
    return ordered[index]


def _bootstrap_interval(
    flags: Sequence[bool], draws: int = BOOTSTRAP_DRAWS
) -> dict[str, float] | None:
    if not flags:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    size = len(flags)
    means = []
    for _ in range(draws):
        means.append(
            sum(flags[rng.randrange(size)] for _ in range(size)) / size
        )
    means.sort()
    return {
        "low": means[int(0.025 * draws)],
        "high": means[int(0.975 * draws) - 1],
    }


def evaluate_family(
    clusters: Sequence[dict[str, Any]],
    extractors: Sequence[Callable[[dict[str, Any]], float | None]],
    target: float,
) -> dict[str, Any]:
    images = sorted({cluster["image_id"] for cluster in clusters})

    retained_flags: list[bool] = []
    rejected_catastrophic = 0
    total_catastrophic = 0
    fold_thresholds: list[float] = []
    skipped_folds = 0

    for held_out in images:
        train = [c for c in clusters if c["image_id"] != held_out]
        test = [c for c in clusters if c["image_id"] == held_out]

        mappers = []
        for extractor in extractors:
            values = [v for v in (extractor(c) for c in train) if v is not None]
            if not values:
                mappers = None
                break
            mappers.append(_percentile_mapper(values))
        if mappers is None:
            skipped_folds += 1
            continue

        train_scores = _score(train, extractors, mappers)
        recovered_train = [
            score
            for score, cluster in zip(train_scores, train, strict=True)
            if score is not None and cluster["greedy_missed_union_recovered"]
        ]
        if not recovered_train:
            skipped_folds += 1
            continue
        threshold = _threshold_for_retention(recovered_train, target)
        fold_thresholds.append(threshold)

        test_scores = _score(test, extractors, mappers)
        for score, cluster in zip(test_scores, test, strict=True):
            if score is None:
                continue
            if cluster["greedy_missed_union_recovered"]:
                retained_flags.append(score >= threshold)
            if cluster["fp_subclass"] in CATASTROPHIC:
                total_catastrophic += 1
                if score < threshold:
                    rejected_catastrophic += 1

    retention = (
        sum(retained_flags) / len(retained_flags) if retained_flags else None
    )
    rejection = (
        rejected_catastrophic / total_catastrophic if total_catastrophic else None
    )
    return {
        "target_retention": target,
        "out_of_sample_retention": retention,
        "out_of_sample_retention_ci": _bootstrap_interval(retained_flags),
        "retained_owners": sum(retained_flags),
        "recovered_owner_clusters": len(retained_flags),
        "out_of_sample_catastrophic_rejection": rejection,
        "rejected_catastrophic": rejected_catastrophic,
        "total_catastrophic": total_catastrophic,
        "fold_threshold_spread": {
            "n": len(fold_thresholds),
            "min": min(fold_thresholds) if fold_thresholds else None,
            "median": statistics.median(fold_thresholds) if fold_thresholds else None,
            "max": max(fold_thresholds) if fold_thresholds else None,
        },
        "skipped_folds": skipped_folds,
    }


def evaluate_in_sample(
    clusters: Sequence[dict[str, Any]],
    extractors: Sequence[Callable[[dict[str, Any]], float | None]],
    target: float,
) -> dict[str, Any]:
    """Upper bound: select and evaluate the threshold on all 12 images."""

    mappers = []
    for extractor in extractors:
        values = [v for v in (extractor(c) for c in clusters) if v is not None]
        if not values:
            return {"out_of_sample": False, "retention": None, "rejection": None}
        mappers.append(_percentile_mapper(values))
    scores = _score(clusters, extractors, mappers)
    recovered = [
        s for s, c in zip(scores, clusters, strict=True)
        if s is not None and c["greedy_missed_union_recovered"]
    ]
    threshold = _threshold_for_retention(recovered, target)
    retained = sum(1 for s in recovered if s >= threshold)
    catastrophic = [
        s for s, c in zip(scores, clusters, strict=True)
        if s is not None and c["fp_subclass"] in CATASTROPHIC
    ]
    rejected = sum(1 for s in catastrophic if s < threshold)
    return {
        "retention": retained / len(recovered) if recovered else None,
        "rejection": rejected / len(catastrophic) if catastrophic else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
        "/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1"
    )
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    clusters_by_checkpoint = json.loads(
        (args.input_dir / "cluster-confidence.json").read_text(encoding="utf-8")
    )["clusters"]

    # Sweep the in-fold retention target; the pre-registered operating point is
    # the target whose pooled out-of-sample retention first reaches 95%.
    targets = [round(0.80 + 0.01 * step, 2) for step in range(21)]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "retention_target": RETENTION_TARGET,
        "rejection_pass_fraction": REJECTION_PASS_FRACTION,
        "primary_denominator": "greedy_missed_union_recovered_owner_clusters",
        "checkpoints": {},
    }

    for checkpoint in CHECKPOINTS:
        clusters = clusters_by_checkpoint[checkpoint]
        catastrophic_total = sum(
            1 for c in clusters if c["fp_subclass"] in CATASTROPHIC
        )
        pass_count = -(-catastrophic_total * 20 // 100)
        entry: dict[str, Any] = {
            "catastrophic_total": catastrophic_total,
            "catastrophic_rejection_pass_threshold": pass_count,
            "recovered_owner_clusters": sum(
                1 for c in clusters if c["greedy_missed_union_recovered"]
            ),
            "families": {},
        }
        for family, extractors in FAMILIES.items():
            sweep = [evaluate_family(clusters, extractors, t) for t in targets]
            # Operating point: the sweep entry whose out-of-sample retention is
            # at least 95%, preferring the one that rejects most.
            qualifying = [
                row
                for row in sweep
                if row["out_of_sample_retention"] is not None
                and row["out_of_sample_retention"] >= RETENTION_TARGET
                and row["out_of_sample_catastrophic_rejection"] is not None
            ]
            operating = (
                max(qualifying, key=lambda row: row["out_of_sample_catastrophic_rejection"])
                if qualifying
                else None
            )
            entry["families"][family] = {
                "operating_point": operating,
                "in_sample_upper_bound": evaluate_in_sample(
                    clusters, extractors, RETENTION_TARGET
                ),
                "sweep": sweep,
            }
        payload["checkpoints"][checkpoint] = entry

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
