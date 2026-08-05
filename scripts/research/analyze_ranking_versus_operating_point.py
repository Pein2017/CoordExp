#!/usr/bin/env python3
"""Separate ranking quality from usable rejection at the operating point.

Stage 1 ranked candidate signals by area under the ROC curve over ALL true
positive clusters. Stage 2 then found that the winner of that ranking (support)
rejects nothing once a retention constraint is applied. This script isolates
why, and checks whether area under the curve predicts usable rejection at all.

Two decompositions:

1. **Population.** A filter must retain greedy-missed union-recovered owners.
   It never has to retain the true positives greedy already found. Measuring
   separation over all true positives therefore scores a signal on a population
   whose retention is free. Reported per feature: area under the curve against
   the catastrophic tail for all true positives, for recovered owners only, and
   for the remaining true positives only.

2. **Discreteness.** Trajectory support is an integer in 1..16 with a large
   atom at 1. A threshold that must retain 95% of recovered owners cannot cut
   into an atom that holds a substantial share of them, no matter how well the
   feature ranks. Reported per checkpoint: the share of recovered owners and of
   catastrophic clusters sitting at support exactly 1.

Both are computed in sample and without folds. This is a mechanism decomposition
of an already-decided result, not a new estimate; the decision-bearing
out-of-sample numbers are owned by the parent unit's Stage 2 artifact.

Research unit:
research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-ranking-quality-versus-usable-rejection/unit.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = "ranking_versus_operating_point.v1"

CATASTROPHIC = ("catastrophic_class_absent", "catastrophic_misgrounded")
RETENTION_TARGET = 0.95


def _load_separation_module() -> Any:
    path = Path(__file__).with_name("analyze_span_likelihood_separation.py")
    spec = importlib.util.spec_from_file_location("separation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load separation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _medoid_coord(cluster: dict[str, Any]) -> float | None:
    block = (cluster.get("likelihood") or {}).get("medoid")
    if not block or block.get("coord_mean") is None:
        return None
    return float(block["coord_mean"])


def _support(cluster: dict[str, Any]) -> float | None:
    return float(cluster["support"])


FEATURES: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "coordinate_mean": _medoid_coord,
    "support": _support,
}


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, index))]


def analyze(clusters: list[dict[str, Any]], separation: Any) -> dict[str, Any]:
    groups = {
        "recovered": [c for c in clusters if c["greedy_missed_union_recovered"]],
        "other_tp": [
            c
            for c in clusters
            if c["canonical_verdict"] == "tp" and not c["greedy_missed_union_recovered"]
        ],
        "all_tp": [c for c in clusters if c["canonical_verdict"] == "tp"],
        "catastrophic": [c for c in clusters if c["fp_subclass"] in CATASTROPHIC],
    }

    result: dict[str, Any] = {
        "group_sizes": {name: len(rows) for name, rows in groups.items()},
        "features": {},
    }

    for name, extractor in FEATURES.items():
        values = {
            key: [v for v in (extractor(c) for c in rows) if v is not None]
            for key, rows in groups.items()
        }
        catastrophic = values["catastrophic"]
        # Threshold that retains the target share of recovered owners, then the
        # rejection it actually buys on the catastrophic tail.
        threshold = _quantile(values["recovered"], 1.0 - RETENTION_TARGET)
        rejected = sum(1 for v in catastrophic if v < threshold)
        result["features"][name] = {
            "auroc_all_tp_vs_catastrophic": separation._auroc(values["all_tp"], catastrophic),
            "auroc_recovered_vs_catastrophic": separation._auroc(
                values["recovered"], catastrophic
            ),
            "auroc_other_tp_vs_catastrophic": separation._auroc(
                values["other_tp"], catastrophic
            ),
            "threshold_at_target_retention": threshold,
            "rejected_catastrophic": rejected,
            "rejection_fraction": rejected / len(catastrophic) if catastrophic else None,
        }

    at_one = {
        "recovered_at_support_1": sum(1 for c in groups["recovered"] if c["support"] == 1),
        "recovered_total": len(groups["recovered"]),
        "catastrophic_at_support_1": sum(
            1 for c in groups["catastrophic"] if c["support"] == 1
        ),
        "catastrophic_total": len(groups["catastrophic"]),
    }
    at_one["recovered_fraction_at_support_1"] = (
        at_one["recovered_at_support_1"] / at_one["recovered_total"]
        if at_one["recovered_total"]
        else None
    )
    at_one["catastrophic_fraction_at_support_1"] = (
        at_one["catastrophic_at_support_1"] / at_one["catastrophic_total"]
        if at_one["catastrophic_total"]
        else None
    )
    result["support_atom"] = at_one
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration"
        "/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1"
    )
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    separation = _load_separation_module()
    clusters_by_checkpoint = json.loads(
        (args.input_dir / "cluster-confidence.json").read_text(encoding="utf-8")
    )["clusters"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "retention_target": RETENTION_TARGET,
        "in_sample": True,
        "checkpoints": {
            checkpoint: analyze(clusters, separation)
            for checkpoint, clusters in clusters_by_checkpoint.items()
        },
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
