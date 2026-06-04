#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/"
    "prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096"
)

ROLES = ("et_rmp_ce", "pure_ce")
QUADRANTS = (
    "boundary_bad_x1_bad",
    "boundary_bad_x1_good",
    "boundary_good_x1_bad",
    "boundary_good_x1_good",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase A3.1 prefix-state transition tomography artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-desc-rows", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.artifact_root)
    out_dir = Path(args.output_dir or root / "phase_a3_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(root / "prefix_state_sampled_rows.jsonl")
    boundary_rows = read_jsonl(root / "boundary_score_rows.jsonl")
    forced_rows = read_jsonl(root / "forced_x1_rows.jsonl")
    quadrant_rows = read_jsonl(root / "quadrant_rows.jsonl")
    sample_by_state = {str(row["prefix_state_id"]): row for row in samples}

    enriched_forced = [enrich_with_sample(row, sample_by_state) for row in forced_rows]
    enriched_quadrants = [enrich_with_sample(row, sample_by_state) for row in quadrant_rows]
    boundary_decisions = unique_boundary_decisions(boundary_rows, sample_by_state)
    enriched_boundary = [enrich_with_sample(row, sample_by_state) for row in boundary_rows]

    summary = {
        "artifact_root": str(root),
        "output_dir": str(out_dir),
        "row_counts": {
            "prefix_state_sampled_rows": len(samples),
            "boundary_score_rows": len(boundary_rows),
            "boundary_decision_rows": len(boundary_decisions),
            "forced_x1_rows": len(forced_rows),
            "quadrant_rows": len(quadrant_rows),
        },
        "sample_index": sample_index_summary(samples),
        "quadrant_by_role": quadrant_summary(enriched_quadrants, ("checkpoint_role",)),
        "quadrant_by_role_transition": quadrant_summary(enriched_quadrants, ("checkpoint_role", "transition_type")),
        "quadrant_by_role_prefix_condition": quadrant_summary(
            enriched_quadrants, ("checkpoint_role", "prefix_condition")
        ),
        "quadrant_by_role_prefix_depth": quadrant_summary(enriched_quadrants, ("checkpoint_role", "prefix_depth")),
        "quadrant_by_role_probe_desc_role": quadrant_summary(
            enriched_quadrants, ("checkpoint_role", "probe_desc_role")
        ),
        "quadrant_by_role_residual_gt_bucket": quadrant_summary(
            enriched_quadrants, ("checkpoint_role", "residual_gt_bucket")
        ),
        "forced_x1_by_role": forced_summary(enriched_forced, ("checkpoint_role",)),
        "forced_x1_by_role_transition": forced_summary(enriched_forced, ("checkpoint_role", "transition_type")),
        "forced_x1_by_role_prefix_condition": forced_summary(enriched_forced, ("checkpoint_role", "prefix_condition")),
        "forced_x1_by_role_residual_gt_bucket": forced_summary(
            enriched_forced, ("checkpoint_role", "residual_gt_bucket")
        ),
        "boundary_decision_by_role": boundary_decision_summary(boundary_decisions, ("checkpoint_role",)),
        "boundary_decision_by_role_transition": boundary_decision_summary(
            boundary_decisions, ("checkpoint_role", "transition_type")
        ),
        "boundary_desc_rank_by_role_probe_desc_role": boundary_desc_rank_summary(
            enriched_boundary, ("checkpoint_role", "probe_desc_role")
        ),
        "paired_delta_by_all": paired_delta_summary(enriched_quadrants, ()),
        "paired_delta_by_transition": paired_delta_summary(enriched_quadrants, ("transition_type",)),
        "paired_delta_by_prefix_condition": paired_delta_summary(enriched_quadrants, ("prefix_condition",)),
        "paired_delta_by_prefix_depth": paired_delta_summary(enriched_quadrants, ("prefix_depth",)),
        "paired_delta_by_probe_desc_role": paired_delta_summary(enriched_quadrants, ("probe_desc_role",)),
        "paired_delta_by_residual_gt_bucket": paired_delta_summary(enriched_quadrants, ("residual_gt_bucket",)),
        "top_probe_desc_slices": top_probe_desc_slices(enriched_quadrants, min_rows=args.min_desc_rows),
        "top_disagreement_examples": top_disagreement_examples(enriched_quadrants, limit=40),
    }

    write_json(out_dir / "phase_a3_analysis_summary.json", summary)
    write_json(out_dir / "quadrant_by_role_transition.json", summary["quadrant_by_role_transition"])
    write_json(out_dir / "forced_x1_by_role_transition.json", summary["forced_x1_by_role_transition"])
    write_json(out_dir / "paired_delta_by_transition.json", summary["paired_delta_by_transition"])
    write_report(out_dir / "phase_a3_analysis_report.md", summary)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "row_counts": summary["row_counts"]}, indent=2))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return None if not vals else sum(vals) / len(vals)


def med(values: Iterable[float]) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    return None if not vals else float(median(vals))


def pct(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def key_for(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    if not fields:
        return "ALL"
    return "|".join(str(row.get(field)) for field in fields)


def group_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_for(row, fields)].append(row)
    return dict(grouped)


def enrich_with_sample(row: Mapping[str, Any], sample_by_state: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    sample = sample_by_state.get(str(out.get("prefix_state_id")), {})
    for field in (
        "prefix_condition",
        "prefix_order_policy_id",
        "desc_count_bucket",
        "object_count_bucket",
        "emitted_object_count",
        "residual_object_count",
        "residual_target_count",
        "target_residual_desc",
        "hard_competitor_desc",
    ):
        if out.get(field) is None and sample.get(field) is not None:
            out[field] = sample.get(field)
    if out.get("probe_desc_role") is None:
        if out.get("probe_desc") == sample.get("target_residual_desc"):
            out["probe_desc_role"] = "target_residual_desc"
        elif out.get("probe_desc") == sample.get("hard_competitor_desc"):
            out["probe_desc_role"] = "hard_competitor_desc"
        else:
            out["probe_desc_role"] = "other"
    out["residual_gt_bucket"] = residual_gt_bucket(as_float(out.get("residual_gt_count", sample.get("residual_target_count", 0))))
    return out


def residual_gt_bucket(value: float) -> str:
    count = int(value)
    if count <= 0:
        return "residual0"
    if count == 1:
        return "residual1"
    if count == 2:
        return "residual2"
    if count <= 4:
        return "residual3_4"
    return "residual5_plus"


def sample_index_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(samples),
        "by_split": dict(Counter(str(row.get("split")) for row in samples)),
        "by_transition_type": dict(Counter(str(row.get("transition_type")) for row in samples)),
        "by_prefix_depth": dict(Counter(str(row.get("prefix_depth")) for row in samples)),
        "by_prefix_condition": dict(Counter(str(row.get("prefix_condition")) for row in samples)),
        "by_object_count_bucket": dict(Counter(str(row.get("object_count_bucket")) for row in samples)),
        "by_desc_count_bucket": dict(Counter(str(row.get("desc_count_bucket")) for row in samples)),
    }


def quadrant_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for group, group_rows_ in sorted(group_rows(rows, fields).items()):
        total = len(group_rows_)
        counts = Counter(str(row.get("quadrant")) for row in group_rows_)
        boundary_good = counts["boundary_good_x1_bad"] + counts["boundary_good_x1_good"]
        x1_good = counts["boundary_bad_x1_good"] + counts["boundary_good_x1_good"]
        out.append(
            {
                "group": group,
                "rows": total,
                "boundary_good_rate": rate(boundary_good, total),
                "x1_good_rate": rate(x1_good, total),
                "both_good_rate": rate(counts["boundary_good_x1_good"], total),
                "boundary_bad_x1_good_rate": rate(counts["boundary_bad_x1_good"], total),
                "boundary_good_x1_bad_rate": rate(counts["boundary_good_x1_bad"], total),
                "both_bad_rate": rate(counts["boundary_bad_x1_bad"], total),
                "quadrant_counts": {quadrant: counts[quadrant] for quadrant in QUADRANTS},
            }
        )
    return out


def forced_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for group, group_rows_ in sorted(group_rows(rows, fields).items()):
        coverages = [as_float(row.get("forced_x1_residual_coverage")) for row in group_rows_]
        ranks = [as_float(row.get("x1_target_rank")) for row in group_rows_]
        out.append(
            {
                "group": group,
                "rows": len(group_rows_),
                "mean_forced_x1_residual_coverage": mean(coverages),
                "median_forced_x1_residual_coverage": med(coverages),
                "coverage_positive_rate": rate(sum(v > 0 for v in coverages), len(coverages)),
                "coverage_full_rate": rate(sum(v >= 0.999 for v in coverages), len(coverages)),
                "mean_coord_vocab_mass": mean(as_float(row.get("coord_vocab_mass")) for row in group_rows_),
                "mean_merged_peak_count": mean(as_float(row.get("merged_peak_count")) for row in group_rows_),
                "median_x1_target_rank": med(ranks),
                "p90_x1_target_rank": pct(ranks, 0.90),
                "mean_p_gt_cond": mean(as_float(row.get("p_gt_cond")) for row in group_rows_),
                "mean_emitted_attraction_rate": mean(as_float(row.get("emitted_attraction_rate")) for row in group_rows_),
                "mean_unmatched_x1_peak_rate": mean(as_float(row.get("unmatched_x1_peak_rate")) for row in group_rows_),
                "mean_boundary_artifact_x1_peak_rate": mean(
                    as_float(row.get("boundary_artifact_x1_peak_rate")) for row in group_rows_
                ),
                "mean_residual_gt_count": mean(as_float(row.get("residual_gt_count")) for row in group_rows_),
            }
        )
    return out


def boundary_decision_id(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("checkpoint_role"),
        row.get("image_id"),
        row.get("source_line_idx"),
        row.get("prefix_state_id"),
        row.get("prefix_condition"),
        row.get("prefix_depth"),
        row.get("prefix_order_policy_id"),
    )


def unique_boundary_decisions(
    rows: Sequence[Mapping[str, Any]],
    sample_by_state: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = boundary_decision_id(row)
        if key not in seen:
            seen[key] = enrich_with_sample(row, sample_by_state)
    return list(seen.values())


def boundary_decision_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for group, group_rows_ in sorted(group_rows(rows, fields).items()):
        total = len(group_rows_)
        align = Counter(str(row.get("boundary_alignment")) for row in group_rows_)
        best_roles = Counter(str(row.get("best_role")) for row in group_rows_)
        margins = [as_float(row.get("margin_best_residual_vs_eos")) for row in group_rows_]
        out.append(
            {
                "group": group,
                "rows": total,
                "residual_favored_rate": rate(align["residual_favored"], total),
                "eos_favored_rate": rate(align["eos_favored"], total),
                "emitted_favored_rate": rate(align["emitted_favored"], total),
                "mixed_or_tied_rate": rate(align["mixed_or_tied"], total),
                "alignment_counts": dict(align),
                "best_role_counts": dict(best_roles),
                "mean_margin_best_residual_vs_eos": mean(margins),
                "median_margin_best_residual_vs_eos": med(margins),
                "margin_positive_rate": rate(sum(v > 0 for v in margins), len(margins)),
            }
        )
    return out


def boundary_desc_rank_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for group, group_rows_ in sorted(group_rows(rows, fields).items()):
        ranks = [as_float(row.get("desc_span_rank")) for row in group_rows_]
        out.append(
            {
                "group": group,
                "rows": len(group_rows_),
                "median_desc_span_rank": med(ranks),
                "rank1_rate": rate(sum(v <= 1 for v in ranks), len(ranks)),
                "rank2_rate": rate(sum(v <= 2 for v in ranks), len(ranks)),
                "mean_desc_span_score": mean(as_float(row.get("desc_span_score")) for row in group_rows_),
            }
        )
    return out


def paired_delta_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped_pair: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_key = tuple([key_for(row, fields), row.get("paired_key")])
        grouped_pair[pair_key][str(row.get("checkpoint_role"))] = row

    by_group: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for (group, _), role_rows in grouped_pair.items():
        if set(role_rows) == set(ROLES):
            by_group[str(group)].append((role_rows["et_rmp_ce"], role_rows["pure_ce"]))

    out = []
    for group, pairs in sorted(by_group.items()):
        deltas = [
            as_float(pure.get("forced_x1_residual_coverage")) - as_float(et.get("forced_x1_residual_coverage"))
            for et, pure in pairs
        ]
        pure_boundary_good = sum(is_boundary_good(pure) and not is_boundary_good(et) for et, pure in pairs)
        et_boundary_good = sum(is_boundary_good(et) and not is_boundary_good(pure) for et, pure in pairs)
        pure_x1_good = sum(is_x1_good(pure) and not is_x1_good(et) for et, pure in pairs)
        et_x1_good = sum(is_x1_good(et) and not is_x1_good(pure) for et, pure in pairs)
        out.append(
            {
                "group": group,
                "paired_rows": len(pairs),
                "mean_pure_minus_et_coverage": mean(deltas),
                "median_pure_minus_et_coverage": med(deltas),
                "pure_coverage_gt_et_rate": rate(sum(delta > 1e-12 for delta in deltas), len(deltas)),
                "et_coverage_gt_pure_rate": rate(sum(delta < -1e-12 for delta in deltas), len(deltas)),
                "coverage_tie_rate": rate(sum(abs(delta) <= 1e-12 for delta in deltas), len(deltas)),
                "pure_only_boundary_good_rate": rate(pure_boundary_good, len(pairs)),
                "et_only_boundary_good_rate": rate(et_boundary_good, len(pairs)),
                "pure_only_x1_good_rate": rate(pure_x1_good, len(pairs)),
                "et_only_x1_good_rate": rate(et_x1_good, len(pairs)),
                "quadrant_disagreement_rate": rate(sum(et.get("quadrant") != pure.get("quadrant") for et, pure in pairs), len(pairs)),
            }
        )
    return out


def is_boundary_good(row: Mapping[str, Any]) -> bool:
    return str(row.get("quadrant")).startswith("boundary_good")


def is_x1_good(row: Mapping[str, Any]) -> bool:
    return str(row.get("quadrant")).endswith("x1_good")


def top_probe_desc_slices(rows: Sequence[Mapping[str, Any]], *, min_rows: int) -> list[dict[str, Any]]:
    by_desc = group_rows(rows, ("probe_desc",))
    out = []
    for desc, desc_rows in sorted(by_desc.items()):
        if len(desc_rows) < min_rows:
            continue
        role_summary = {row["group"]: row for row in quadrant_summary(desc_rows, ("checkpoint_role",))}
        if not all(role in role_summary for role in ROLES):
            continue
        et = role_summary["et_rmp_ce"]
        pure = role_summary["pure_ce"]
        out.append(
            {
                "probe_desc": desc,
                "rows": len(desc_rows),
                "et_boundary_good_rate": et["boundary_good_rate"],
                "pure_boundary_good_rate": pure["boundary_good_rate"],
                "pure_minus_et_boundary_good_rate": pure["boundary_good_rate"] - et["boundary_good_rate"],
                "et_x1_good_rate": et["x1_good_rate"],
                "pure_x1_good_rate": pure["x1_good_rate"],
                "pure_minus_et_x1_good_rate": pure["x1_good_rate"] - et["x1_good_rate"],
                "et_both_good_rate": et["both_good_rate"],
                "pure_both_good_rate": pure["both_good_rate"],
                "pure_minus_et_both_good_rate": pure["both_good_rate"] - et["both_good_rate"],
            }
        )
    out.sort(key=lambda row: (-abs(row["pure_minus_et_both_good_rate"]), -row["rows"], row["probe_desc"]))
    return out[:50]


def top_disagreement_examples(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row.get("paired_key"))][str(row.get("checkpoint_role"))] = row
    examples = []
    for pair_key, role_rows in grouped.items():
        if set(role_rows) != set(ROLES):
            continue
        et = role_rows["et_rmp_ce"]
        pure = role_rows["pure_ce"]
        delta = as_float(pure.get("forced_x1_residual_coverage")) - as_float(et.get("forced_x1_residual_coverage"))
        if et.get("quadrant") == pure.get("quadrant") and abs(delta) <= 1e-12:
            continue
        examples.append(
            {
                "paired_key": pair_key,
                "image_id": et.get("image_id"),
                "split": et.get("split"),
                "transition_type": et.get("transition_type"),
                "prefix_depth": et.get("prefix_depth"),
                "prefix_condition": et.get("prefix_condition"),
                "probe_desc": et.get("probe_desc"),
                "probe_desc_role": et.get("probe_desc_role"),
                "et_quadrant": et.get("quadrant"),
                "pure_quadrant": pure.get("quadrant"),
                "et_forced_x1_residual_coverage": as_float(et.get("forced_x1_residual_coverage")),
                "pure_forced_x1_residual_coverage": as_float(pure.get("forced_x1_residual_coverage")),
                "pure_minus_et_coverage": delta,
            }
        )
    examples.sort(key=lambda row: (-abs(row["pure_minus_et_coverage"]), str(row["image_id"]), str(row["probe_desc"])))
    return examples[:limit]


def table_md(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, limit: int | None = None) -> list[str]:
    selected = list(rows[:limit]) if limit else list(rows)
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(format_cell(row.get(key)) for _, key in columns) + " |")
    return lines


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# Phase A3.1 Prefix-State Transition Tomography Analysis",
            "",
            "## Scope",
            "",
            f"Artifact root: `{summary['artifact_root']}`",
            "",
            "Evidence scope: 4096 sampled prefix-state rows, paired ET-RMP-CE vs pure-CE checkpoint-3664 readouts.",
            "",
            "This analysis uses boundary full-desc span rows, forced-desc pre-x1 rows, and merged quadrant rows. It does not use full attention dumps.",
            "",
            "## Row Counts",
            "",
            "```json",
            json.dumps(summary["row_counts"], indent=2, sort_keys=True),
            "```",
            "",
            "## Main Quadrant Rates By Checkpoint",
            "",
        ]
    )
    lines.extend(
        table_md(
            summary["quadrant_by_role"],
            [
                ("group", "group"),
                ("rows", "rows"),
                ("boundary_good", "boundary_good_rate"),
                ("x1_good", "x1_good_rate"),
                ("both_good", "both_good_rate"),
                ("bad_boundary_good_x1", "boundary_bad_x1_good_rate"),
                ("good_boundary_bad_x1", "boundary_good_x1_bad_rate"),
                ("both_bad", "both_bad_rate"),
            ],
        )
    )
    lines.extend(["", "## Same-Desc Vs Different-Desc", ""])
    lines.extend(
        table_md(
            summary["quadrant_by_role_transition"],
            [
                ("group", "group"),
                ("rows", "rows"),
                ("boundary_good", "boundary_good_rate"),
                ("x1_good", "x1_good_rate"),
                ("both_good", "both_good_rate"),
                ("both_bad", "both_bad_rate"),
            ],
        )
    )
    lines.extend(["", "## Forced X1 Summary By Checkpoint And Transition", ""])
    lines.extend(
        table_md(
            summary["forced_x1_by_role_transition"],
            [
                ("group", "group"),
                ("rows", "rows"),
                ("mean_cov", "mean_forced_x1_residual_coverage"),
                ("cov>0", "coverage_positive_rate"),
                ("cov=1", "coverage_full_rate"),
                ("median_rank", "median_x1_target_rank"),
                ("mean_peaks", "mean_merged_peak_count"),
                ("emitted_attr", "mean_emitted_attraction_rate"),
                ("boundary_artifact", "mean_boundary_artifact_x1_peak_rate"),
            ],
        )
    )
    lines.extend(["", "## Boundary Decision Summary", ""])
    lines.extend(
        table_md(
            summary["boundary_decision_by_role_transition"],
            [
                ("group", "group"),
                ("rows", "rows"),
                ("residual_favored", "residual_favored_rate"),
                ("eos_favored", "eos_favored_rate"),
                ("emitted_favored", "emitted_favored_rate"),
                ("mixed", "mixed_or_tied_rate"),
                ("mean_margin_resid_vs_eos", "mean_margin_best_residual_vs_eos"),
            ],
        )
    )
    lines.extend(["", "## Paired Delta Summary", ""])
    lines.extend(
        table_md(
            summary["paired_delta_by_transition"],
            [
                ("group", "group"),
                ("paired_rows", "paired_rows"),
                ("mean_pure_minus_et_cov", "mean_pure_minus_et_coverage"),
                ("pure_cov>et", "pure_coverage_gt_et_rate"),
                ("et_cov>pure", "et_coverage_gt_pure_rate"),
                ("pure_only_boundary_good", "pure_only_boundary_good_rate"),
                ("et_only_boundary_good", "et_only_boundary_good_rate"),
                ("quadrant_disagree", "quadrant_disagreement_rate"),
            ],
        )
    )
    lines.extend(["", "## Top Probe Description Slices", ""])
    lines.extend(
        table_md(
            summary["top_probe_desc_slices"],
            [
                ("desc", "probe_desc"),
                ("rows", "rows"),
                ("pure-et boundary", "pure_minus_et_boundary_good_rate"),
                ("pure-et x1", "pure_minus_et_x1_good_rate"),
                ("pure-et both", "pure_minus_et_both_good_rate"),
                ("et both", "et_both_good_rate"),
                ("pure both", "pure_both_good_rate"),
            ],
            limit=20,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- `boundary_good` means `boundary_alignment == residual_favored` at the boundary/full-desc scoring stage.",
            "- `x1_good` means forced-desc pre-x1 peaks cover at least one residual same-desc GT x1.",
            "- `boundary_bad_x1_good` is the key decoupling bucket: the model has x1 evidence under forced desc, but boundary selection does not favor residual continuation.",
            "- `boundary_good_x1_bad` is the complementary bucket: boundary continuation/desc selection is healthy, but forced x1 binding does not cover a residual instance.",
            "- This run does not prove attention-head causality or training-objective causality by itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
