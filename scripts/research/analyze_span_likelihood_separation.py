#!/usr/bin/env python3
"""Stage 1 separation analysis for the sampled span-likelihood mining unit.

Answers the pre-registered Stage 1 question: within each checkpoint, does raw
model likelihood separate true-positive clusters from catastrophic false
positives, and does it do so through coordinates rather than schema grammar?

The wrapper component is the control. If schema-wrapper likelihood separates
but coordinate likelihood does not, the result is grammar confidence rather
than grounding confidence and the unit says to reject it.

Research unit:
research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "span_likelihood_separation.v1"

CHECKPOINTS = ("sorted", "random", "permutation")

# Medoid-level features. The medoid box is the canonical union prediction, so
# the medoid's own likelihood is the pre-registered primary aggregation.
FEATURES = (
    "coord_mean",
    "coord_min",
    "description_mean",
    "wrapper_mean",
    "row_logprob_mean",
)

CATASTROPHIC = ("catastrophic_class_absent", "catastrophic_misgrounded")
ORDINARY = ("ordinary_duplicate", "ordinary_loose")


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Mann-Whitney U statistic, normalized. Ties contribute 0.5.

    Returns P(score of a random positive > score of a random negative). 0.5 is
    no separation. Higher likelihood is expected for positives, so a value
    above 0.5 means likelihood ranks true positives above false positives.
    """

    if not positive or not negative:
        return None
    merged = sorted(
        [(value, 1) for value in positive] + [(value, 0) for value in negative]
    )
    # Average ranks within tied groups so ties score exactly 0.5.
    ranks: list[float] = [0.0] * len(merged)
    index = 0
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, merged) if label == 1
    )
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "mean": statistics.fmean(ordered),
    }


def _medoid_feature(cluster: dict[str, Any], feature: str) -> float | None:
    block = (cluster.get("likelihood") or {}).get("medoid")
    if not block:
        return None
    value = block.get(feature)
    return None if value is None else float(value)


def analyze_checkpoint(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "tp": [],
        "ordinary_fp": [],
        "catastrophic_fp": [],
    }
    for cluster in clusters:
        if cluster["canonical_verdict"] == "tp":
            groups["tp"].append(cluster)
        elif cluster["fp_subclass"] in CATASTROPHIC:
            groups["catastrophic_fp"].append(cluster)
        elif cluster["fp_subclass"] in ORDINARY:
            groups["ordinary_fp"].append(cluster)

    result: dict[str, Any] = {
        "group_sizes": {name: len(rows) for name, rows in groups.items()},
        "features": {},
        "support": {},
    }

    for feature in FEATURES:
        by_group = {
            name: [
                value
                for value in (_medoid_feature(row, feature) for row in rows)
                if value is not None
            ]
            for name, rows in groups.items()
        }
        result["features"][feature] = {
            "distribution": {name: _describe(values) for name, values in by_group.items()},
            "auroc_tp_vs_catastrophic": _auroc(
                by_group["tp"], by_group["catastrophic_fp"]
            ),
            "auroc_tp_vs_ordinary": _auroc(by_group["tp"], by_group["ordinary_fp"]),
            "auroc_ordinary_vs_catastrophic": _auroc(
                by_group["ordinary_fp"], by_group["catastrophic_fp"]
            ),
        }

    # Support is the consensus baseline the likelihood families must beat.
    support = {name: [float(row["support"]) for row in rows] for name, rows in groups.items()}
    result["support"] = {
        "distribution": {name: _describe(values) for name, values in support.items()},
        "auroc_tp_vs_catastrophic": _auroc(support["tp"], support["catastrophic_fp"]),
        "auroc_tp_vs_ordinary": _auroc(support["tp"], support["ordinary_fp"]),
    }
    return result


def analyze_trajectory_position(span_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Does low likelihood arrive later in a trajectory (bad-basin entry)?"""

    buckets: dict[str, list[float]] = {}
    for row in span_rows:
        total = int(row["trajectory_row_count"])
        if total < 2:
            continue
        fraction = int(row["generated_order"]) / (total - 1)
        key = "first_third" if fraction < 1 / 3 else ("middle_third" if fraction < 2 / 3 else "last_third")
        buckets.setdefault(key, []).append(float(row["coord_mean"]))
    return {key: _describe(values) for key, values in sorted(buckets.items())}


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
    span_rows = [
        json.loads(line)
        for line in (args.input_dir / "span-likelihood.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "primary_aggregation": "medoid",
        "checkpoints": {},
    }
    for checkpoint in CHECKPOINTS:
        analysis = analyze_checkpoint(clusters_by_checkpoint[checkpoint])
        analysis["trajectory_position_coord_mean"] = analyze_trajectory_position(
            [row for row in span_rows if row["checkpoint"] == checkpoint]
        )
        payload["checkpoints"][checkpoint] = analysis

    # The pre-registered control: coordinates must beat the schema wrapper.
    payload["grammar_control"] = {
        checkpoint: {
            "coord_mean_auroc": payload["checkpoints"][checkpoint]["features"]["coord_mean"][
                "auroc_tp_vs_catastrophic"
            ],
            "coord_min_auroc": payload["checkpoints"][checkpoint]["features"]["coord_min"][
                "auroc_tp_vs_catastrophic"
            ],
            "wrapper_mean_auroc": payload["checkpoints"][checkpoint]["features"]["wrapper_mean"][
                "auroc_tp_vs_catastrophic"
            ],
            "coordinates_beat_wrapper": (
                max(
                    payload["checkpoints"][checkpoint]["features"]["coord_mean"][
                        "auroc_tp_vs_catastrophic"
                    ],
                    payload["checkpoints"][checkpoint]["features"]["coord_min"][
                        "auroc_tp_vs_catastrophic"
                    ],
                )
                > payload["checkpoints"][checkpoint]["features"]["wrapper_mean"][
                    "auroc_tp_vs_catastrophic"
                ]
            ),
        }
        for checkpoint in CHECKPOINTS
    }

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
