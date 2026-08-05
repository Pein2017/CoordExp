#!/usr/bin/env python3
"""Compare matched sorted/random image-2299 native and accessibility evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCHEMA_VERSION = "image2299-random-sorted-accessibility-contrast.v1"
OWNER_SCHEMA_VERSION = "image2299-random-sorted-accessibility-owner-contrast.v1"
IMAGE_ID = "2299"


class ContrastError(ValueError):
    """Raised when the two evidence roots are not a matched image-2299 pair."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContrastError(f"expected JSON object at {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _localization_features(row: Mapping[str, Any]) -> dict[str, Any]:
    localization = row["localization"]
    lower = localization["generator_local_max_excluding_other_owner_strict"][
        "ambiguity_excluded_l"
    ]
    competition = row["owner_competition_l"]
    return {
        "context_id": row["context_id"],
        "boundary_index": int(row["boundary_index"]),
        "context_role": row["context_role"],
        "exact_anchor_score": localization.get("exact_anchor_score"),
        "best_local_candidate_score": lower.get("value"),
        "local_concentration": lower.get("local_concentration"),
        "peak_lift": lower.get("peak_lift"),
        "category_candidate_rank": lower.get("rank"),
        "category_candidate_population": lower.get("unique_population_size"),
        "owner_rank": competition.get("rank"),
        "owner_population": competition.get("population_size"),
        "owner_margin_to_best": competition.get("margin_to_best_owner"),
        "clears_sorted_frozen_rule": lower.get("clears_frozen_support_rule"),
        "loop_tail": row["loop_marking"]["loop_tail"],
        "continue_vs_stop_margin": row["proposal_surface"]["boundary_gate"].get(
            "continue_vs_stop_logprob_margin"
        ),
        "category_route_rank": row["proposal_surface"]["category_routing_event"].get(
            "within_context_rank"
        ),
    }


def _best_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        _localization_features(row)
        for row in rows
        if not row["loop_marking"]["loop_tail"]
    ]
    if not candidates:
        raise ContrastError("owner has no non-loop context")
    return max(
        candidates,
        key=lambda row: (
            float(row["peak_lift"]),
            float(row["local_concentration"]),
            -int(row["category_candidate_rank"]),
        ),
    )


def _free_hits(
    sidecars: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in sidecars:
        if not row.get("well_formed_box") or not isinstance(row.get("coord_bins"), list):
            continue
        bins = [int(value) for value in row["coord_bins"]]
        if len(bins) != 4:
            continue
        box = [
            round(bins[0] * 1216 / 1000),
            round(bins[1] * 736 / 1000),
            round(bins[2] * 1216 / 1000),
            round(bins[3] * 736 / 1000),
        ]
        for owner_id, owner in owners.items():
            if owner["normalized_description"] != row.get("normalized_description"):
                continue
            overlap = _iou(box, owner["bbox_pixel_xyxy"])
            previous = best.get(owner_id)
            if previous is None or overlap > previous["iou"]:
                best[owner_id] = {
                    "iou": overlap,
                    "strict_hit": overlap >= 0.5,
                    "context_id": row["context_id"],
                    "coord_bins": bins,
                    "bbox_pixel_xyxy": box,
                    "complete_box_logprob_sum": row.get("complete_box_logprob_sum"),
                }
    return best


def _load_arm(root: Path, label: str) -> dict[str, Any]:
    analysis_dir = root / "s1-analysis"
    shard_dir = root / "s1-shard"
    if label == "random":
        shard_dir = root / "s1-shard-full-b16"
    required = [
        root / "s0-native/receipt.json",
        root / "s0-native/prediction-row-ledger.jsonl",
        analysis_dir / "analysis.json",
        analysis_dir / "owner-summaries.jsonl",
        analysis_dir / "owner-context-features.jsonl",
        shard_dir / "free-decode-sidecars.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContrastError(f"{label} evidence root is incomplete: {missing}")
    summary_rows = _read_jsonl(analysis_dir / "owner-summaries.jsonl")
    summaries = {str(row["gt_owner_id"]): row for row in summary_rows}
    if len(summaries) != 46:
        raise ContrastError(f"{label} analysis does not contain 46 owners")
    contexts_by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(analysis_dir / "owner-context-features.jsonl"):
        contexts_by_owner.setdefault(str(row["gt_owner_id"]), []).append(row)
    if set(contexts_by_owner) != set(summaries):
        raise ContrastError(f"{label} context/summary owner keys disagree")
    predictions = _read_jsonl(root / "s0-native/prediction-row-ledger.jsonl")
    native_tp = {
        str(row["strict_match_gt_owner_id"])
        for row in predictions
        if row["strict_match_status"] == "matched"
    }
    sidecars = _read_jsonl(shard_dir / "free-decode-sidecars.jsonl")
    return {
        "label": label,
        "root": root,
        "analysis": _read_json(analysis_dir / "analysis.json"),
        "s0": _read_json(root / "s0-native/receipt.json"),
        "summaries": summaries,
        "contexts": contexts_by_owner,
        "predictions": predictions,
        "native_tp": native_tp,
        "free_hits": _free_hits(sidecars, summaries),
        "digests": {str(path.relative_to(root)): _sha256_file(path) for path in required},
    }


def compare(sorted_root: Path, random_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sorted_arm = _load_arm(sorted_root, "sorted")
    random_arm = _load_arm(random_root, "random")
    if set(sorted_arm["summaries"]) != set(random_arm["summaries"]):
        raise ContrastError("sorted and random owner identities disagree")
    owner_rows: list[dict[str, Any]] = []
    for owner_id in sorted(sorted_arm["summaries"]):
        sorted_summary = sorted_arm["summaries"][owner_id]
        random_summary = random_arm["summaries"][owner_id]
        if (
            sorted_summary["bbox_pixel_xyxy"] != random_summary["bbox_pixel_xyxy"]
            or sorted_summary["normalized_description"]
            != random_summary["normalized_description"]
        ):
            raise ContrastError(f"owner {owner_id} differs between arms")
        sorted_contexts = sorted_arm["contexts"][owner_id]
        random_contexts = random_arm["contexts"][owner_id]
        sorted_root_row = next(row for row in sorted_contexts if int(row["boundary_index"]) == 0)
        random_root_row = next(row for row in random_contexts if int(row["boundary_index"]) == 0)
        sorted_root_features = _localization_features(sorted_root_row)
        random_root_features = _localization_features(random_root_row)
        sorted_tp = owner_id in sorted_arm["native_tp"]
        random_tp = owner_id in random_arm["native_tp"]
        owner_rows.append(
            {
                "schema_version": OWNER_SCHEMA_VERSION,
                "gt_owner_id": owner_id,
                "normalized_description": sorted_summary["normalized_description"],
                "bbox_pixel_xyxy": sorted_summary["bbox_pixel_xyxy"],
                "native": {
                    "sorted_true_positive": sorted_tp,
                    "random_true_positive": random_tp,
                    "transition": (
                        "retained" if sorted_tp and random_tp else
                        "lost_by_random" if sorted_tp else
                        "gained_by_random" if random_tp else "missed_by_both"
                    ),
                },
                "root": {
                    "sorted": sorted_root_features,
                    "random": random_root_features,
                    "delta_random_minus_sorted": {
                        field: float(random_root_features[field]) - float(sorted_root_features[field])
                        for field in (
                            "exact_anchor_score",
                            "best_local_candidate_score",
                            "local_concentration",
                            "peak_lift",
                            "owner_margin_to_best",
                            "continue_vs_stop_margin",
                        )
                        if random_root_features[field] is not None and sorted_root_features[field] is not None
                    },
                },
                "best_native_context": {
                    "sorted": _best_context(sorted_contexts),
                    "random": _best_context(random_contexts),
                },
                "free_conditional_box": {
                    "sorted": sorted_arm["free_hits"].get(owner_id),
                    "random": random_arm["free_hits"].get(owner_id),
                },
                "sorted_frozen_rule_sensitivity": {
                    "sorted": sorted_summary["frozen_disposition_descriptive"],
                    "random": random_summary["frozen_disposition_descriptive"],
                    "classification_is_not_primary": True,
                },
            }
        )

    sorted_tp = sorted_arm["native_tp"]
    random_tp = random_arm["native_tp"]
    random_predictions = random_arm["predictions"]
    full_canvas = sum(
        list(row["bbox_xyxy"]) == [0.0, 0.0, 1215.0, 735.0]
        for row in random_predictions
    )
    transitions = Counter(row["native"]["transition"] for row in owner_rows)
    root_fields = (
        "exact_anchor_score",
        "best_local_candidate_score",
        "local_concentration",
        "peak_lift",
        "owner_margin_to_best",
        "continue_vs_stop_margin",
    )
    root_delta_medians = {
        field: _median(
            [
                row["root"]["delta_random_minus_sorted"][field]
                for row in owner_rows
                if field in row["root"]["delta_random_minus_sorted"]
            ]
        )
        for field in root_fields
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "image_id": IMAGE_ID,
        "owner_count": 46,
        "native": {
            "sorted": {
                "true_positive_owner_count": len(sorted_tp),
                "prediction_row_count": len(sorted_arm["predictions"]),
            },
            "random": {
                "true_positive_owner_count": len(random_tp),
                "prediction_row_count": len(random_predictions),
                "full_canvas_duplicate_row_count": full_canvas,
                "unmatched_prediction_row_count": sum(
                    row["strict_match_status"] == "unmatched" for row in random_predictions
                ),
            },
            "transition_counts": dict(sorted(transitions.items())),
            "retained_owner_ids": sorted(sorted_tp & random_tp),
            "gained_by_random_owner_ids": sorted(random_tp - sorted_tp),
            "lost_by_random_owner_ids": sorted(sorted_tp - random_tp),
        },
        "root_context": {
            "history_is_identical": True,
            "median_delta_random_minus_sorted": root_delta_medians,
            "random_owner_rank_1_count": sum(
                row["root"]["random"]["owner_rank"] == 1 for row in owner_rows
            ),
            "sorted_owner_rank_1_count": sum(
                row["root"]["sorted"]["owner_rank"] == 1 for row in owner_rows
            ),
            "random_category_candidate_top3_count": sum(
                row["root"]["random"]["category_candidate_rank"] <= 3 for row in owner_rows
            ),
            "sorted_category_candidate_top3_count": sum(
                row["root"]["sorted"]["category_candidate_rank"] <= 3 for row in owner_rows
            ),
        },
        "free_conditional_box": {
            "sorted_any_context_strict_owner_hit_count": sum(
                bool((row["free_conditional_box"]["sorted"] or {}).get("strict_hit"))
                for row in owner_rows
            ),
            "random_any_context_strict_owner_hit_count": sum(
                bool((row["free_conditional_box"]["random"] or {}).get("strict_hit"))
                for row in owner_rows
            ),
        },
        "calibration_transfer": {
            "sorted": sorted_arm["analysis"]["calibration_transfer"],
            "random": random_arm["analysis"]["calibration_transfer"],
            "binary_cross_checkpoint_prevalence_claimed": False,
        },
        "source_roots": {"sorted": str(sorted_root), "random": str(random_root)},
        "source_digests": {
            "sorted": sorted_arm["digests"],
            "random": random_arm["digests"],
        },
    }
    return summary, owner_rows


def _report(summary: Mapping[str, Any]) -> str:
    native = summary["native"]
    root = summary["root_context"]
    free = summary["free_conditional_box"]
    return "\n".join(
        [
            "# Image 2299 random versus sorted matched contrast",
            "",
            "## Native rollout",
            "",
            f"- Sorted: {native['sorted']['true_positive_owner_count']} TP owners from {native['sorted']['prediction_row_count']} rows.",
            f"- Random: {native['random']['true_positive_owner_count']} TP owners from {native['random']['prediction_row_count']} rows.",
            f"- Random emits {native['random']['full_canvas_duplicate_row_count']} exact full-canvas duplicate rows and {native['random']['unmatched_prediction_row_count']} unmatched rows.",
            f"- Retained/gained/lost: {native['transition_counts']}.",
            "",
            "## Identical root context",
            "",
            f"- Owner-rank 1: sorted {root['sorted_owner_rank_1_count']}/46; random {root['random_owner_rank_1_count']}/46.",
            f"- Category-candidate top 3: sorted {root['sorted_category_candidate_top3_count']}/46; random {root['random_category_candidate_top3_count']}/46.",
            f"- Median random-minus-sorted deltas: `{json.dumps(root['median_delta_random_minus_sorted'], sort_keys=True)}`.",
            "",
            "## Conditional free-box behavior",
            "",
            f"- Any-context strict owner hit: sorted {free['sorted_any_context_strict_owner_hit_count']}/46; random {free['random_any_context_strict_owner_hit_count']}/46.",
            "",
            "Binary frozen-threshold FN prevalence is not claimed across checkpoints.",
            "",
        ]
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorted-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary, owners = compare(args.sorted_root, args.random_root)
    except (OSError, json.JSONDecodeError, ContrastError) as exc:
        print(f"contrast error: {exc}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "owner-comparison.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in owners),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary["native"], indent=2, sort_keys=True))
    print(json.dumps(summary["root_context"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
