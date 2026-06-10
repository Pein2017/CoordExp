#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_ET_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/"
    "candidate_field_cardinality_tomography_representative8192"
)
DEFAULT_PURE_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_purece/"
    "candidate_field_cardinality_tomography_representative8192"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/"
    "candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/"
    "phase_a2_dual_checkpoint_analysis"
)

A1_BUCKET = "A1_cardinality_collapse"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return None if not values else sum(values) / len(values)


def pct(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def same_desc_bucket(count: int) -> str:
    if count <= 1:
        return "count1"
    if count == 2:
        return "count2"
    if count == 3:
        return "count3"
    if count <= 5:
        return "count4_5"
    return "count6_plus"


def load_run(root: Path, label: str, valid_radius: float) -> dict[str, Any]:
    x1_rows = read_jsonl(root / "x1_candidate_field_rows.jsonl")
    taxonomy_rows = read_jsonl(root / "phase_a_case_taxonomy_rows.jsonl")
    taxonomy_by_case = {row["case_id"]: row for row in taxonomy_rows}
    rows: dict[str, dict[str, Any]] = {}
    duplicate_case_ids: list[str] = []
    for row in x1_rows:
        case_id = row["case_id"]
        if case_id in rows:
            duplicate_case_ids.append(case_id)
        tax = taxonomy_by_case.get(case_id, {})
        enriched = dict(row)
        enriched["primary_bucket"] = tax.get("primary_bucket")
        enriched["is_A1"] = tax.get("primary_bucket") == A1_BUCKET
        enriched.update(peak_validity(enriched, valid_radius))
        rows[case_id] = enriched
    return {
        "label": label,
        "root": str(root),
        "rows": rows,
        "row_count": len(rows),
        "taxonomy_row_count": len(taxonomy_by_case),
        "duplicate_case_ids": duplicate_case_ids[:20],
        "duplicate_case_id_count": len(duplicate_case_ids),
    }


def peak_validity(row: dict[str, Any], valid_radius: float) -> dict[str, Any]:
    gt_x1s = [as_float(x) for x in row.get("same_desc_gt_x1_values", [])]
    peaks = row.get("merged_peaks", []) or []
    valid_peak_count = 0
    valid_peak_mass = 0.0
    unmatched_peak_count = 0
    unmatched_peak_mass = 0.0
    peak_records = []
    for idx, peak in enumerate(peaks):
        x1 = as_float(peak.get("x1"))
        mass = as_float(peak.get("mass"))
        nearest = None
        if gt_x1s:
            nearest = min(abs(x1 - gt) for gt in gt_x1s)
        is_valid = nearest is not None and nearest <= valid_radius
        if is_valid:
            valid_peak_count += 1
            valid_peak_mass += mass
        else:
            unmatched_peak_count += 1
            unmatched_peak_mass += mass
        peak_records.append(
            {
                "peak_index": idx,
                "x1": x1,
                "mass": mass,
                "nearest_gt_x1_distance": nearest,
                "annotated_valid_x1_radius": bool(is_valid),
            }
        )
    covered_gt = 0
    for gt_x1 in gt_x1s:
        if any(abs(as_float(peak.get("x1")) - gt_x1) <= valid_radius for peak in peaks):
            covered_gt += 1
    peak_count = len(peaks)
    peak_mass = valid_peak_mass + unmatched_peak_mass
    same_desc_count = as_int(row.get("same_desc_gt_count_annotated"))
    coverage_fraction = covered_gt / same_desc_count if same_desc_count else 0.0
    return {
        "peak_count": peak_count,
        "valid_peak_count": valid_peak_count,
        "valid_peak_mass": valid_peak_mass,
        "unmatched_peak_count": unmatched_peak_count,
        "unmatched_peak_mass": unmatched_peak_mass,
        "valid_peak_share": valid_peak_count / peak_count if peak_count else None,
        "valid_peak_mass_share": valid_peak_mass / peak_mass if peak_mass > 0 else None,
        "covered_gt_count_recomputed": covered_gt,
        "coverage_fraction_recomputed": coverage_fraction,
        "peak_records_for_examples": peak_records,
    }


def top32_clusters(
    row: dict[str, Any],
    *,
    merge_radius: float,
    relative_floor: float,
    absolute_floor: float,
    valid_radius: float,
) -> dict[str, Any]:
    bins = row.get("top_bins", []) or []
    if not bins:
        return {
            "peak_count": 0,
            "multi_peak": False,
            "covered_gt_count": 0,
            "coverage_fraction": 0.0,
            "valid_peak_count": 0,
            "valid_peak_mass": 0.0,
            "unmatched_peak_count": 0,
            "unmatched_peak_mass": 0.0,
        }
    top_prob = max(as_float(item.get("prob_cond")) for item in bins)
    floor = max(absolute_floor, top_prob * relative_floor)
    candidates = [
        {
            "x1": as_float(item.get("bin")),
            "mass": as_float(item.get("prob_cond")),
        }
        for item in bins
        if as_float(item.get("prob_cond")) >= floor
    ]
    candidates.sort(key=lambda item: item["mass"], reverse=True)
    clusters: list[dict[str, Any]] = []
    for candidate in candidates:
        best_idx = None
        best_dist = None
        for idx, cluster in enumerate(clusters):
            dist = abs(candidate["x1"] - cluster["center"])
            if dist <= merge_radius and (best_dist is None or dist < best_dist):
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            clusters.append(
                {
                    "center": candidate["x1"],
                    "mass": candidate["mass"],
                    "members": [candidate],
                }
            )
        else:
            cluster = clusters[best_idx]
            cluster["members"].append(candidate)
            cluster["mass"] += candidate["mass"]
            total = sum(member["mass"] for member in cluster["members"])
            cluster["center"] = sum(member["x1"] * member["mass"] for member in cluster["members"]) / total
    gt_x1s = [as_float(x) for x in row.get("same_desc_gt_x1_values", [])]
    valid_peak_count = 0
    valid_peak_mass = 0.0
    unmatched_peak_count = 0
    unmatched_peak_mass = 0.0
    covered_gt = 0
    for cluster in clusters:
        valid = any(
            abs(member["x1"] - gt_x1) <= valid_radius
            for member in cluster["members"]
            for gt_x1 in gt_x1s
        )
        if valid:
            valid_peak_count += 1
            valid_peak_mass += as_float(cluster["mass"])
        else:
            unmatched_peak_count += 1
            unmatched_peak_mass += as_float(cluster["mass"])
    for gt_x1 in gt_x1s:
        if any(abs(member["x1"] - gt_x1) <= valid_radius for cluster in clusters for member in cluster["members"]):
            covered_gt += 1
    same_desc_count = as_int(row.get("same_desc_gt_count_annotated"))
    peak_mass = valid_peak_mass + unmatched_peak_mass
    return {
        "peak_count": len(clusters),
        "multi_peak": len(clusters) >= 2,
        "covered_gt_count": covered_gt,
        "coverage_fraction": covered_gt / same_desc_count if same_desc_count else 0.0,
        "valid_peak_count": valid_peak_count,
        "valid_peak_mass": valid_peak_mass,
        "unmatched_peak_count": unmatched_peak_count,
        "unmatched_peak_mass": unmatched_peak_mass,
        "valid_peak_share": valid_peak_count / len(clusters) if clusters else None,
        "valid_peak_mass_share": valid_peak_mass / peak_mass if peak_mass > 0 else None,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    peak_counts = [as_float(row["peak_count"]) for row in rows]
    coverage = [as_float(row["coverage_fraction_recomputed"]) for row in rows]
    valid_counts = [as_float(row["valid_peak_count"]) for row in rows]
    unmatched_counts = [as_float(row["unmatched_peak_count"]) for row in rows]
    valid_share = [row["valid_peak_share"] for row in rows if row["valid_peak_share"] is not None]
    valid_mass_share = [row["valid_peak_mass_share"] for row in rows if row["valid_peak_mass_share"] is not None]
    target_ranks = [as_float(row.get("x1_target_rank"), math.nan) for row in rows]
    p_gt_cond = [as_float(row.get("p_gt_cond"), math.nan) for row in rows]
    a1 = [1.0 if row.get("is_A1") else 0.0 for row in rows]
    return {
        "row_count": len(rows),
        "A1_rate": mean(a1),
        "multi_peak_row_rate": mean([1.0 if row["peak_count"] >= 2 else 0.0 for row in rows]),
        "mean_peak_count": mean(peak_counts),
        "median_peak_count": median(peak_counts),
        "mean_valid_peak_count": mean(valid_counts),
        "mean_unmatched_peak_count": mean(unmatched_counts),
        "total_peak_count": sum(peak_counts),
        "total_valid_peak_count": sum(valid_counts),
        "total_unmatched_peak_count": sum(unmatched_counts),
        "total_valid_peak_share": sum(valid_counts) / sum(peak_counts) if sum(peak_counts) > 0 else None,
        "mean_coverage_fraction": mean(coverage),
        "median_coverage_fraction": median(coverage),
        "mean_valid_peak_share": mean(valid_share),
        "mean_valid_peak_mass_share": mean(valid_mass_share),
        "median_target_rank": pct(target_ranks, 0.5),
        "median_p_gt_cond": pct(p_gt_cond, 0.5),
    }


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"row_count": 0}
    numeric_keys = [
        "delta_peak_count",
        "delta_valid_peak_count",
        "delta_unmatched_peak_count",
        "delta_coverage_fraction",
        "delta_valid_peak_mass_share",
        "delta_target_rank",
        "delta_p_gt_cond",
    ]
    summary = {"row_count": len(pairs)}
    for key in numeric_keys:
        vals = [as_float(pair.get(key), math.nan) for pair in pairs]
        summary[f"mean_{key}"] = mean(vals)
        summary[f"median_{key}"] = pct(vals, 0.5)
        summary[f"p25_{key}"] = pct(vals, 0.25)
        summary[f"p75_{key}"] = pct(vals, 0.75)
    summary["pure_more_peaks_rate"] = mean([1.0 if pair["delta_peak_count"] > 0 else 0.0 for pair in pairs])
    summary["pure_more_valid_peaks_rate"] = mean([1.0 if pair["delta_valid_peak_count"] > 0 else 0.0 for pair in pairs])
    summary["pure_more_unmatched_peaks_rate"] = mean([1.0 if pair["delta_unmatched_peak_count"] > 0 else 0.0 for pair in pairs])
    summary["pure_higher_coverage_rate"] = mean([1.0 if pair["delta_coverage_fraction"] > 0 else 0.0 for pair in pairs])
    return summary


def build_pair(et: dict[str, Any], pure: dict[str, Any]) -> dict[str, Any]:
    count = as_int(et.get("same_desc_gt_count_annotated"))
    et_rank = as_float(et.get("x1_target_rank"), math.nan)
    pure_rank = as_float(pure.get("x1_target_rank"), math.nan)
    et_valid_mass_share = et.get("valid_peak_mass_share")
    pure_valid_mass_share = pure.get("valid_peak_mass_share")
    if et_valid_mass_share is None or pure_valid_mass_share is None:
        delta_valid_peak_mass_share = math.nan
    else:
        delta_valid_peak_mass_share = as_float(pure_valid_mass_share) - as_float(et_valid_mass_share)
    if et.get("is_A1") and pure.get("is_A1"):
        transition = "both_A1"
    elif et.get("is_A1") and not pure.get("is_A1"):
        transition = "ET_only_A1"
    elif not et.get("is_A1") and pure.get("is_A1"):
        transition = "pure_only_A1"
    else:
        transition = "neither_A1"
    return {
        "case_id": et["case_id"],
        "split": et.get("split"),
        "image_id": et.get("image_id"),
        "source_line_idx": et.get("source_line_idx"),
        "desc_text_canonical": et.get("desc_text_canonical"),
        "pool_role": et.get("pool_role"),
        "same_desc_gt_count_annotated": count,
        "same_desc_bucket": same_desc_bucket(count),
        "x1_projection_collision": bool(et.get("x1_projection_collision")),
        "transition": transition,
        "et_A1": bool(et.get("is_A1")),
        "pure_A1": bool(pure.get("is_A1")),
        "et_peak_count": et["peak_count"],
        "pure_peak_count": pure["peak_count"],
        "delta_peak_count": pure["peak_count"] - et["peak_count"],
        "et_valid_peak_count": et["valid_peak_count"],
        "pure_valid_peak_count": pure["valid_peak_count"],
        "delta_valid_peak_count": pure["valid_peak_count"] - et["valid_peak_count"],
        "et_unmatched_peak_count": et["unmatched_peak_count"],
        "pure_unmatched_peak_count": pure["unmatched_peak_count"],
        "delta_unmatched_peak_count": pure["unmatched_peak_count"] - et["unmatched_peak_count"],
        "et_valid_peak_mass_share": et_valid_mass_share,
        "pure_valid_peak_mass_share": pure_valid_mass_share,
        "delta_valid_peak_mass_share": delta_valid_peak_mass_share,
        "et_coverage_fraction": et["coverage_fraction_recomputed"],
        "pure_coverage_fraction": pure["coverage_fraction_recomputed"],
        "delta_coverage_fraction": pure["coverage_fraction_recomputed"] - et["coverage_fraction_recomputed"],
        "et_target_rank": et_rank,
        "pure_target_rank": pure_rank,
        "delta_target_rank": pure_rank - et_rank,
        "et_p_gt_cond": as_float(et.get("p_gt_cond"), math.nan),
        "pure_p_gt_cond": as_float(pure.get("p_gt_cond"), math.nan),
        "delta_p_gt_cond": as_float(pure.get("p_gt_cond"), math.nan) - as_float(et.get("p_gt_cond"), math.nan),
        "et_peaks_for_examples": et.get("peak_records_for_examples", [])[:8],
        "pure_peaks_for_examples": pure.get("peak_records_for_examples", [])[:8],
    }


def grouped_summary(pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair.get(key))].append(pair)
    return {name: summarize_pairs(rows) for name, rows in sorted(groups.items())}


def desc_delta_tables(pairs: list[dict[str, Any]], min_rows: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair["desc_text_canonical"])].append(pair)
    rows = []
    for desc, desc_pairs in groups.items():
        if len(desc_pairs) < min_rows:
            continue
        summary = summarize_pairs(desc_pairs)
        rows.append(
            {
                "desc_text_canonical": desc,
                "row_count": len(desc_pairs),
                "mean_same_desc_count": mean([as_float(pair["same_desc_gt_count_annotated"]) for pair in desc_pairs]),
                "mean_delta_peak_count": summary.get("mean_delta_peak_count"),
                "mean_delta_valid_peak_count": summary.get("mean_delta_valid_peak_count"),
                "mean_delta_unmatched_peak_count": summary.get("mean_delta_unmatched_peak_count"),
                "mean_delta_coverage_fraction": summary.get("mean_delta_coverage_fraction"),
                "pure_higher_coverage_rate": summary.get("pure_higher_coverage_rate"),
                "ET_only_A1_count": sum(1 for pair in desc_pairs if pair["transition"] == "ET_only_A1"),
                "pure_only_A1_count": sum(1 for pair in desc_pairs if pair["transition"] == "pure_only_A1"),
            }
        )
    by_valid = sorted(rows, key=lambda item: (item["mean_delta_valid_peak_count"], item["row_count"]), reverse=True)
    by_unmatched = sorted(rows, key=lambda item: (item["mean_delta_unmatched_peak_count"], item["row_count"]), reverse=True)
    by_coverage = sorted(rows, key=lambda item: (item["mean_delta_coverage_fraction"], item["row_count"]), reverse=True)
    return {
        "top_desc_by_pure_valid_peak_gain": by_valid[:20],
        "top_desc_by_pure_unmatched_peak_gain": by_unmatched[:20],
        "top_desc_by_pure_coverage_gain": by_coverage[:20],
    }


def representative_examples(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fields = [
        "case_id",
        "split",
        "image_id",
        "source_line_idx",
        "desc_text_canonical",
        "same_desc_gt_count_annotated",
        "same_desc_bucket",
        "transition",
        "et_peak_count",
        "pure_peak_count",
        "delta_peak_count",
        "et_valid_peak_count",
        "pure_valid_peak_count",
        "delta_valid_peak_count",
        "et_unmatched_peak_count",
        "pure_unmatched_peak_count",
        "delta_unmatched_peak_count",
        "et_coverage_fraction",
        "pure_coverage_fraction",
        "delta_coverage_fraction",
        "et_target_rank",
        "pure_target_rank",
        "et_peaks_for_examples",
        "pure_peaks_for_examples",
    ]

    def slim(pair: dict[str, Any]) -> dict[str, Any]:
        return {field: pair.get(field) for field in fields}

    return {
        "pure_resolves_ET_A1_by_coverage_gain": [
            slim(pair)
            for pair in sorted(
                (p for p in pairs if p["transition"] == "ET_only_A1"),
                key=lambda p: (p["delta_coverage_fraction"], p["delta_valid_peak_count"], p["same_desc_gt_count_annotated"]),
                reverse=True,
            )[:12]
        ],
        "ET_resolves_pure_A1_by_coverage_loss": [
            slim(pair)
            for pair in sorted(
                (p for p in pairs if p["transition"] == "pure_only_A1"),
                key=lambda p: (-p["delta_coverage_fraction"], p["delta_valid_peak_count"], p["same_desc_gt_count_annotated"]),
            )[:12]
        ],
        "pure_adds_valid_peaks": [
            slim(pair)
            for pair in sorted(
                pairs,
                key=lambda p: (p["delta_valid_peak_count"], p["delta_coverage_fraction"], p["same_desc_gt_count_annotated"]),
                reverse=True,
            )[:12]
        ],
        "pure_adds_unmatched_peaks": [
            slim(pair)
            for pair in sorted(
                pairs,
                key=lambda p: (p["delta_unmatched_peak_count"], p["delta_peak_count"], p["same_desc_gt_count_annotated"]),
                reverse=True,
            )[:12]
        ],
    }


def sensitivity_summary(
    runs: dict[str, dict[str, Any]],
    *,
    valid_radius: float,
) -> list[dict[str, Any]]:
    grid = []
    for radius in [12, 24, 36, 48]:
        for rel_floor in [0.05, 0.10, 0.20]:
            for abs_floor in [0.001, 0.002, 0.005]:
                for label, run in runs.items():
                    rows = []
                    for row in run["rows"].values():
                        result = top32_clusters(
                            row,
                            merge_radius=radius,
                            relative_floor=rel_floor,
                            absolute_floor=abs_floor,
                            valid_radius=valid_radius,
                        )
                        same_desc_count = as_int(row.get("same_desc_gt_count_annotated"))
                        noncollision = not bool(row.get("x1_projection_collision"))
                        rows.append(
                            {
                                **result,
                                "same_desc_count": same_desc_count,
                                "same_desc_bucket": same_desc_bucket(same_desc_count),
                                "approx_A1_noncollision_gate": bool(
                                    noncollision and result["covered_gt_count"] < same_desc_count
                                ),
                            }
                        )
                    grid.append(
                        {
                            "checkpoint": label,
                            "merge_radius": radius,
                            "relative_floor": rel_floor,
                            "absolute_floor": abs_floor,
                            "row_count": len(rows),
                            "a1_rate_top32_approx_noncollision_gate": mean(
                                [1.0 if row["approx_A1_noncollision_gate"] else 0.0 for row in rows]
                            ),
                            "multi_peak_rate_top32_approx": mean([1.0 if row["multi_peak"] else 0.0 for row in rows]),
                            "mean_peak_count_top32_approx": mean([as_float(row["peak_count"]) for row in rows]),
                            "mean_coverage_fraction_top32_approx": mean(
                                [as_float(row["coverage_fraction"]) for row in rows]
                            ),
                            "mean_valid_peak_share_top32_approx": mean(
                                [row["valid_peak_share"] for row in rows if row["valid_peak_share"] is not None]
                            ),
                            "mean_valid_peak_mass_share_top32_approx": mean(
                                [row["valid_peak_mass_share"] for row in rows if row["valid_peak_mass_share"] is not None]
                            ),
                        }
                    )
    return grid


def sensitivity_robustness(grid: list[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[tuple[float, float, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in grid:
        key = (row["merge_radius"], row["relative_floor"], row["absolute_floor"])
        paired[key][row["checkpoint"]] = row
    metrics = {
        "pure_lower_A1_rate": ("a1_rate_top32_approx_noncollision_gate", "lower"),
        "pure_higher_multi_peak_rate": ("multi_peak_rate_top32_approx", "higher"),
        "pure_higher_mean_peak_count": ("mean_peak_count_top32_approx", "higher"),
        "pure_higher_coverage_fraction": ("mean_coverage_fraction_top32_approx", "higher"),
        "pure_higher_valid_peak_share": ("mean_valid_peak_share_top32_approx", "higher"),
        "pure_higher_valid_peak_mass_share": ("mean_valid_peak_mass_share_top32_approx", "higher"),
    }
    result = {"grid_cell_count": len(paired)}
    for out_key, (metric, direction) in metrics.items():
        wins = 0
        deltas = []
        for rows in paired.values():
            if "pure_ce" not in rows or "et_rmp_ce" not in rows:
                continue
            delta = rows["pure_ce"][metric] - rows["et_rmp_ce"][metric]
            deltas.append(delta)
            if (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0):
                wins += 1
        result[out_key] = {
            "favorable_cells": wins,
            "compared_cells": len(deltas),
            "mean_delta": mean(deltas),
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
        }
    return result


def make_plots(summary: dict[str, Any], output_root: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    transitions = summary["paired_transition_counts"]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["both_A1", "ET_only_A1", "pure_only_A1", "neither_A1"]
    ax.bar(names, [transitions.get(name, 0) for name in names], color=["#7f7f7f", "#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("Rows")
    ax.set_title("Paired A1 transitions")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = plot_root / "paired_a1_transitions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["paired_a1_transitions"] = str(path)

    bucket_order = ["count1", "count2", "count3", "count4_5", "count6_plus"]
    by_bucket = summary["paired_by_same_desc_bucket"]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [by_bucket.get(bucket, {}).get("mean_delta_coverage_fraction", 0.0) for bucket in bucket_order]
    ax.bar(bucket_order, vals, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pure CE - ET-RMP-CE coverage")
    ax.set_title("Coverage delta by same-desc count")
    fig.tight_layout()
    path = plot_root / "coverage_delta_by_count.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["coverage_delta_by_count"] = str(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [by_bucket.get(bucket, {}).get("mean_delta_valid_peak_count", 0.0) for bucket in bucket_order]
    ys = [by_bucket.get(bucket, {}).get("mean_delta_unmatched_peak_count", 0.0) for bucket in bucket_order]
    ax.scatter(xs, ys, s=90, color="#e45756")
    for bucket, x, y in zip(bucket_order, xs, ys):
        ax.annotate(bucket, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean valid peak delta")
    ax.set_ylabel("Mean unmatched peak delta")
    ax.set_title("Pure CE extra peaks: valid vs unmatched")
    fig.tight_layout()
    path = plot_root / "valid_vs_unmatched_peak_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["valid_vs_unmatched_peak_delta"] = str(path)

    sensitivity = [
        row
        for row in summary["top32_approx_sensitivity"]
        if abs(row["relative_floor"] - 0.10) < 1e-9 and abs(row["absolute_floor"] - 0.002) < 1e-12
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [("et_rmp_ce", "#4c78a8"), ("pure_ce", "#f58518")]:
        rows = sorted([row for row in sensitivity if row["checkpoint"] == label], key=lambda row: row["merge_radius"])
        ax.plot(
            [row["merge_radius"] for row in rows],
            [row["a1_rate_top32_approx_noncollision_gate"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    ax.set_xlabel("Merge radius")
    ax.set_ylabel("Top-32 approx A1 rate")
    ax.set_title("A1 sensitivity at relative_floor=0.10, absolute_floor=0.002")
    ax.legend()
    fig.tight_layout()
    path = plot_root / "sensitivity_grid_a1_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["sensitivity_grid_a1_rate"] = str(path)

    write_json(plot_root / "plot_summary.json", plot_paths)
    return plot_paths


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def build_report(summary: dict[str, Any]) -> str:
    overall = summary["paired_overall"]
    by_bucket = summary["paired_by_same_desc_bucket"]
    transitions = summary["paired_transition_counts"]
    run_summary = summary["run_summaries"]["overall"]
    rows = [
        [
            metric,
            run_summary["et_rmp_ce"].get(metric),
            run_summary["pure_ce"].get(metric),
            (
                run_summary["pure_ce"].get(metric) - run_summary["et_rmp_ce"].get(metric)
                if isinstance(run_summary["et_rmp_ce"].get(metric), (int, float))
                and isinstance(run_summary["pure_ce"].get(metric), (int, float))
                else None
            ),
        ]
        for metric in [
            "A1_rate",
            "multi_peak_row_rate",
            "mean_peak_count",
            "mean_valid_peak_count",
            "mean_unmatched_peak_count",
            "total_valid_peak_share",
            "mean_coverage_fraction",
            "mean_valid_peak_share",
            "mean_valid_peak_mass_share",
            "median_target_rank",
            "median_p_gt_cond",
        ]
    ]
    bucket_rows = []
    for bucket in ["count1", "count2", "count3", "count4_5", "count6_plus"]:
        item = by_bucket.get(bucket, {})
        bucket_rows.append(
            [
                bucket,
                item.get("row_count"),
                item.get("mean_delta_peak_count"),
                item.get("mean_delta_valid_peak_count"),
                item.get("mean_delta_unmatched_peak_count"),
                item.get("mean_delta_coverage_fraction"),
                item.get("pure_higher_coverage_rate"),
            ]
        )
    trans_rows = [[name, transitions.get(name, 0)] for name in ["both_A1", "ET_only_A1", "pure_only_A1", "neither_A1"]]

    sensitivity = [
        row
        for row in summary["top32_approx_sensitivity"]
        if row["merge_radius"] == 24
        and abs(row["relative_floor"] - 0.10) < 1e-9
        and abs(row["absolute_floor"] - 0.002) < 1e-12
    ]
    sensitivity_rows = [
        [
            row["checkpoint"],
            row["a1_rate_top32_approx_noncollision_gate"],
            row["multi_peak_rate_top32_approx"],
            row["mean_peak_count_top32_approx"],
            row["mean_coverage_fraction_top32_approx"],
            row["mean_valid_peak_share_top32_approx"],
            row["mean_valid_peak_mass_share_top32_approx"],
        ]
        for row in sorted(sensitivity, key=lambda row: row["checkpoint"])
    ]
    robustness = summary["top32_approx_sensitivity_robustness"]
    desc_rows = [
        [
            row["desc_text_canonical"],
            row["row_count"],
            row["mean_same_desc_count"],
            row["mean_delta_valid_peak_count"],
            row["mean_delta_unmatched_peak_count"],
            row["mean_delta_coverage_fraction"],
        ]
        for row in summary["desc_delta_tables"]["top_desc_by_pure_valid_peak_gain"][:12]
    ]

    lines = [
        "# Phase A2 Dual-Checkpoint Candidate-Field Analysis",
        "",
        "Scope: paired offline analysis over the matched ET-RMP-CE and pure-CE `representative8192` artifacts.  "
        "This is a diagnostic result, not production training and not a fully isolated objective ablation.",
        "",
        "## Artifact Roots",
        "",
        f"- ET-RMP-CE: `{summary['roots']['et_rmp_ce']}`",
        f"- pure CE: `{summary['roots']['pure_ce']}`",
        f"- Phase A2 output: `{summary['roots']['output_root']}`",
        "",
        "## Headline Comparison",
        "",
        md_table(["Metric", "ET-RMP-CE", "pure CE", "pure - ET"], rows),
        "",
        "## Paired A1 Transitions",
        "",
        md_table(["Transition", "Rows"], trans_rows),
        "",
        "Transition semantics: `ET_only_A1` means ET-RMP-CE under-covered the row by the A1 rule while pure CE did not; "
        "`pure_only_A1` means the reverse.",
        "",
        "## Pure-minus-ET Deltas By Same-Desc Count",
        "",
        md_table(
            [
                "Bucket",
                "Rows",
                "Delta peaks",
                "Delta valid peaks",
                "Delta unmatched peaks",
                "Delta coverage",
                "Pure higher coverage rate",
            ],
            bucket_rows,
        ),
        "",
        "`Delta valid peaks` is peak-side validity: more than one peak can fall near the same annotated GT x1.  "
        "`Delta coverage` is the instance-side coverage signal and is the safer recall-proxy field.",
        "",
        "## Top-32 Approx Sensitivity Snapshot",
        "",
        "This table recomputes peaks from the stored top-32 coordinate bins with `merge_radius=24`, "
        "`relative_floor=0.10`, and `absolute_floor=0.002`.  It is labeled `top32_approx` because it does not "
        "re-run the model or use the full coordinate posterior.",
        "",
        md_table(
            [
                "Checkpoint",
                "A1 approx",
                "Multi-peak",
                "Mean peaks",
                "Mean coverage",
                "Valid peak share",
                "Valid mass share",
            ],
            sensitivity_rows,
        ),
        "",
        "Across the full top-32 approximate sensitivity grid, pure CE has lower A1, higher multi-peak rate, "
        "higher mean peak count, and higher mean coverage in "
        f"{robustness['pure_lower_A1_rate']['favorable_cells']}/"
        f"{robustness['pure_lower_A1_rate']['compared_cells']} cells; "
        f"{robustness['pure_higher_multi_peak_rate']['favorable_cells']}/"
        f"{robustness['pure_higher_multi_peak_rate']['compared_cells']} cells; "
        f"{robustness['pure_higher_mean_peak_count']['favorable_cells']}/"
        f"{robustness['pure_higher_mean_peak_count']['compared_cells']} cells; and "
        f"{robustness['pure_higher_coverage_fraction']['favorable_cells']}/"
        f"{robustness['pure_higher_coverage_fraction']['compared_cells']} cells, respectively.",
        "",
        "## Descs With Largest Pure-CE Valid-Peak Gain",
        "",
        md_table(
            ["Desc", "Rows", "Mean count", "Delta valid peaks", "Delta unmatched peaks", "Delta coverage"],
            desc_rows,
        ),
        "",
        "## Evidence-Backed Interpretations",
        "",
        "1. The single-step local-singleton explanation is weakened.  Pure CE is broader than ET-RMP-CE on the same rows: "
        f"mean peak delta is {overall['mean_delta_peak_count']:.4f}, and pure CE has more peaks on "
        f"{overall['pure_more_peaks_rate']:.4f} of rows.",
        "",
        "2. Breadth is not identical to clean annotated coverage.  Pure CE also increases unmatched peaks under the COCO "
        f"annotation universe: mean unmatched-peak delta is {overall['mean_delta_unmatched_peak_count']:.4f}.  "
        "Because COCO can miss objects, this is an ambiguity bucket rather than automatic hallucination.",
        "",
        "3. Candidate compression remains in crowded rows.  Even when pure CE is broader, high-cardinality rows still have "
        "coverage below one; the candidate field is richer, but not an exhaustive instance ledger.",
        "",
        "4. ET-RMP-CE appears to compact the pre-x1 candidate field relative to pure CE on this diagnostic surface.  "
        "This may coexist with better final rollout behavior, so candidate-field breadth and final decode stability should "
        "stay separated in future claims.",
        "",
        "5. This contrast is not an isolated objective ablation.  Normalization, state weighting, and effective batch shape "
        "also differ between the checkpoints.",
        "",
        "## Verification",
        "",
        f"- Matched row count: `{summary['matched_row_count']}`",
        f"- Sample exact order: `{summary['sample_match_exact_order']}`",
        f"- Valid x1 radius: `{summary['valid_radius']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A2 paired ET-RMP-CE vs pure-CE candidate-field analysis.")
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET_ROOT)
    parser.add_argument("--pure-root", type=Path, default=DEFAULT_PURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--valid-radius", type=float, default=24.0)
    parser.add_argument("--desc-min-rows", type=int, default=50)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    et_run = load_run(args.et_root, "et_rmp_ce", args.valid_radius)
    pure_run = load_run(args.pure_root, "pure_ce", args.valid_radius)
    et_ids = list(et_run["rows"].keys())
    pure_ids = list(pure_run["rows"].keys())
    common_ids = [case_id for case_id in et_ids if case_id in pure_run["rows"]]
    pairs = [build_pair(et_run["rows"][case_id], pure_run["rows"][case_id]) for case_id in common_ids]
    transition_counts = dict(Counter(pair["transition"] for pair in pairs))

    run_rows = {
        "et_rmp_ce": [et_run["rows"][case_id] for case_id in common_ids],
        "pure_ce": [pure_run["rows"][case_id] for case_id in common_ids],
    }
    run_summaries = {"overall": {label: summarize_rows(rows) for label, rows in run_rows.items()}}
    for group_key in ["same_desc_bucket", "pool_role"]:
        grouped = {}
        for label, rows in run_rows.items():
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                if group_key == "same_desc_bucket":
                    key = same_desc_bucket(as_int(row.get("same_desc_gt_count_annotated")))
                else:
                    key = str(row.get(group_key))
                groups[key].append(row)
            grouped[label] = {key: summarize_rows(value) for key, value in sorted(groups.items())}
        run_summaries[group_key] = grouped

    runs = {"et_rmp_ce": et_run, "pure_ce": pure_run}
    top32_sensitivity = sensitivity_summary(runs, valid_radius=args.valid_radius)
    summary = {
        "analysis_id": "phase_a2_dual_checkpoint_analysis",
        "roots": {
            "et_rmp_ce": str(args.et_root),
            "pure_ce": str(args.pure_root),
            "output_root": str(args.output_root),
        },
        "valid_radius": args.valid_radius,
        "et_row_count": et_run["row_count"],
        "pure_row_count": pure_run["row_count"],
        "matched_row_count": len(common_ids),
        "sample_match_exact_order": et_ids == pure_ids,
        "sample_match_same_ids": set(et_ids) == set(pure_ids),
        "duplicate_case_id_counts": {
            "et_rmp_ce": et_run["duplicate_case_id_count"],
            "pure_ce": pure_run["duplicate_case_id_count"],
        },
        "paired_transition_counts": transition_counts,
        "run_summaries": run_summaries,
        "paired_overall": summarize_pairs(pairs),
        "paired_by_same_desc_bucket": grouped_summary(pairs, "same_desc_bucket"),
        "paired_by_pool_role": grouped_summary(pairs, "pool_role"),
        "desc_delta_tables": desc_delta_tables(pairs, args.desc_min_rows),
        "representative_examples": representative_examples(pairs),
        "top32_approx_sensitivity": top32_sensitivity,
        "top32_approx_sensitivity_robustness": sensitivity_robustness(top32_sensitivity),
    }
    plot_paths = make_plots(summary, args.output_root)
    summary["plots"] = plot_paths
    write_json(args.output_root / "phase_a2_summary.json", summary)
    (args.output_root / "phase_a2_report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "matched_row_count": len(common_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
