#!/usr/bin/env python3
"""Robustness summary for a paired Source/treatment transition experiment.

This is deliberately an experiment-local analysis utility.  It uses the
cardinality-first global owner matcher and strict-duplicate definition from
``compare_clean_rollout_owner_coverage``; it does not turn unmatched
predictions into hallucination labels.  Token-cutoff curves use only complete
object spans whose *closing* generated token is before the requested cutoff.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _category,
    _global_matches,
    _gt_objects,
    _pred_objects,
    _read_jsonl,
    _row_metrics,
    _sha256,
    compare_artifacts,
)


SCHEMA_VERSION = "transition_phase0_robustness.v1"
PRIMARY_THRESHOLD = 0.50
THRESHOLDS = (0.30, 0.50, 0.75)
DEFAULT_CUTOFFS = (64, 128, 256, 512, 1024, 2048)


class RobustnessError(ValueError):
    """Raised when paired experiment evidence is malformed or incompatible."""


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def symmetric_trimmed_mean(values: Iterable[float], proportion: float = 0.10) -> dict[str, Any]:
    """Return a declared symmetric trimmed mean, with deterministic small-n behavior."""
    if not 0.0 <= proportion < 0.5:
        raise RobustnessError("trim proportion must be in [0, 0.5)")
    ordered = sorted(float(value) for value in values)
    trim_each_tail = math.floor(len(ordered) * proportion)
    retained = ordered[trim_each_tail : len(ordered) - trim_each_tail] if trim_each_tail else ordered
    return {
        "proportion_each_tail": proportion,
        "trim_each_tail_count": trim_each_tail,
        "retained_count": len(retained),
        "mean": sum(retained) / len(retained) if retained else None,
    }


def _owner_id(row: Mapping[str, Any], index: int) -> str:
    gt = row.get("gt", [])
    value = gt[index] if isinstance(gt, list) and index < len(gt) else {}
    if isinstance(value, Mapping) and value.get("object_id", value.get("id")) is not None:
        return str(value.get("object_id", value.get("id")))
    return str(index)


def _owner_sets(row: dict[str, Any], row_id: str, threshold: float) -> tuple[set[int], Counter[str], int, int]:
    gt = _gt_objects(row, row_id=row_id)
    pred, _ = _pred_objects(row)
    matched = {gt_index for gt_index, _, _ in _global_matches(gt, pred, threshold)}
    categories = Counter(_category(row["gt"][index]) for index in matched)
    metrics, _, _, _, _, _ = _row_metrics(
        row,
        row_id=row_id,
        match_iou_threshold=threshold,
        duplicate_iou_threshold=0.30,
        strict_annotation_iou_threshold=0.50,
        strict_prediction_iou_threshold=0.90,
    )
    return matched, categories, int(metrics["prediction_count"]), int(metrics["strict_physical_owner_duplicate_candidate_count"])


def _per_image(source: dict[str, dict[str, Any]], treatment: dict[str, dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    if set(source) != set(treatment):
        raise RobustnessError("Source/treatment row IDs do not match")
    rows: list[dict[str, Any]] = []
    for row_id in sorted(source):
        source_owners, _, source_predictions, source_strict = _owner_sets(source[row_id], row_id, threshold)
        treatment_owners, _, treatment_predictions, treatment_strict = _owner_sets(treatment[row_id], row_id, threshold)
        gt = _gt_objects(source[row_id], row_id=row_id)
        all_owners = set(range(len(gt)))
        gained = treatment_owners - source_owners
        lost = source_owners - treatment_owners
        retained = source_owners & treatment_owners
        missed_both = all_owners - (source_owners | treatment_owners)
        gt_list = source[row_id].get("gt", [])
        category = lambda index: _category(gt_list[index])  # noqa: E731
        rows.append({
            "row_id": row_id,
            "gt_owner_count": len(all_owners),
            "source_owner_count": len(source_owners),
            "treatment_owner_count": len(treatment_owners),
            "gained_owner_count": len(gained),
            "retained_owner_count": len(retained),
            "lost_owner_count": len(lost),
            "missed_by_both_owner_count": len(missed_both),
            "net_owner_delta": len(gained) - len(lost),
            "source_prediction_count": source_predictions,
            "treatment_prediction_count": treatment_predictions,
            "prediction_count_delta": treatment_predictions - source_predictions,
            "source_strict_duplicate_count": source_strict,
            "treatment_strict_duplicate_count": treatment_strict,
            "strict_duplicate_delta": treatment_strict - source_strict,
            "gained_owner_ids": [_owner_id(source[row_id], index) for index in sorted(gained)],
            "lost_owner_ids": [_owner_id(source[row_id], index) for index in sorted(lost)],
            "gained_categories": dict(sorted(Counter(category(index) for index in gained).items())),
            "lost_categories": dict(sorted(Counter(category(index) for index in lost).items())),
        })
    return rows


def _aggregate(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    total = lambda key: sum(int(row[key]) for row in per_image)  # noqa: E731
    source_owners = total("source_owner_count")
    treatment_owners = total("treatment_owner_count")
    source_predictions = total("source_prediction_count")
    treatment_predictions = total("treatment_prediction_count")
    source_strict = total("source_strict_duplicate_count")
    treatment_strict = total("treatment_strict_duplicate_count")
    return {
        "image_count": len(per_image),
        "gt_owner_count": total("gt_owner_count"),
        "gained_owner_count": total("gained_owner_count"),
        "retained_owner_count": total("retained_owner_count"),
        "lost_owner_count": total("lost_owner_count"),
        "missed_by_both_owner_count": total("missed_by_both_owner_count"),
        "net_owner_delta": treatment_owners - source_owners,
        "source": {
            "owner_count": source_owners,
            "prediction_count": source_predictions,
            "strict_duplicate_count": source_strict,
            "owner_yield": source_owners / source_predictions if source_predictions else 0.0,
            "strict_duplicate_rate": source_strict / source_predictions if source_predictions else 0.0,
        },
        "treatment": {
            "owner_count": treatment_owners,
            "prediction_count": treatment_predictions,
            "strict_duplicate_count": treatment_strict,
            "owner_yield": treatment_owners / treatment_predictions if treatment_predictions else 0.0,
            "strict_duplicate_rate": treatment_strict / treatment_predictions if treatment_predictions else 0.0,
        },
    }


def _influence(per_image: list[dict[str, Any]], trim_proportion: float) -> dict[str, Any]:
    deltas = [float(row["net_owner_delta"]) for row in per_image]
    total = sum(deltas)
    leave_one_out = [total - value for value in deltas]
    gained_categories: Counter[str] = Counter()
    lost_categories: Counter[str] = Counter()
    for row in per_image:
        gained_categories.update({str(key): int(value) for key, value in row["gained_categories"].items()})
        lost_categories.update({str(key): int(value) for key, value in row["lost_categories"].items()})
    largest = sorted(per_image, key=lambda row: (-abs(int(row["net_owner_delta"])), str(row["row_id"])))[:10]
    return {
        "median_net_owner_delta": _median(deltas),
        "symmetric_trimmed_mean_net_owner_delta": symmetric_trimmed_mean(deltas, trim_proportion),
        "leave_one_image_out_net_range": {
            "minimum": min(leave_one_out) if leave_one_out else None,
            "maximum": max(leave_one_out) if leave_one_out else None,
        },
        "largest_influences": [
            {"row_id": row["row_id"], "net_owner_delta": row["net_owner_delta"], "leave_one_out_net": total - int(row["net_owner_delta"])}
            for row in largest
        ],
        "category_concentration": {
            "gained": dict(sorted(gained_categories.items(), key=lambda item: (-item[1], item[0]))),
            "lost": dict(sorted(lost_categories.items(), key=lambda item: (-item[1], item[0]))),
        },
    }


def _paired_natural_stop_subset(source: dict[str, dict[str, Any]], treatment: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    source_stops = {row_id for row_id, row in source.items() if str(row.get("decode_stop_reason")) == "im_end"}
    treatment_stops = {row_id for row_id, row in treatment.items() if str(row.get("decode_stop_reason")) == "im_end"}
    selected = source_stops & treatment_stops
    return (
        {row_id: source[row_id] for row_id in sorted(selected)},
        {row_id: treatment[row_id] for row_id in sorted(selected)},
        {"source_natural_stop_count": len(source_stops), "treatment_natural_stop_count": len(treatment_stops), "paired_natural_stop_count": len(selected)},
    )


def _cutoff_row(scored_row: dict[str, Any], cutoff: int) -> dict[str, Any]:
    row = copy.deepcopy(scored_row)
    predictions = row.get("pred", [])
    if not isinstance(predictions, list):
        raise RobustnessError(f"scored row {row.get('row_id')!r} has malformed pred list")
    completed: list[dict[str, Any]] = []
    for prediction in predictions:
        provenance = prediction.get("pred_score_source", {}) if isinstance(prediction, dict) else {}
        steps = provenance.get("generated_step_indices") if isinstance(provenance, dict) else None
        if not isinstance(steps, list) or not steps or any(not isinstance(step, int) or isinstance(step, bool) for step in steps):
            raise RobustnessError(f"scored prediction in row {row.get('row_id')!r} lacks generated_step_indices")
        if max(steps) < cutoff:
            completed.append(prediction)
    row["pred"] = completed
    return row


def _cutoff_curve(source_scored: dict[str, dict[str, Any]], treatment_scored: dict[str, dict[str, Any]], cutoffs: Iterable[int]) -> list[dict[str, Any]]:
    if set(source_scored) != set(treatment_scored):
        raise RobustnessError("Source/treatment scored row IDs do not match")
    result: list[dict[str, Any]] = []
    for cutoff in sorted(set(cutoffs)):
        if cutoff <= 0:
            raise RobustnessError("token cutoffs must be positive")
        source = {row_id: _cutoff_row(row, cutoff) for row_id, row in source_scored.items()}
        treatment = {row_id: _cutoff_row(row, cutoff) for row_id, row in treatment_scored.items()}
        aggregate = _aggregate(_per_image(source, treatment, PRIMARY_THRESHOLD))
        result.append({"token_cutoff_exclusive": cutoff, **aggregate})
    return result


def _input_provenance(paths: Mapping[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise RobustnessError(f"missing input: {path}")
        row_count = None
        if path.suffix == ".jsonl":
            # Token traces intentionally repeat row IDs, so their provenance
            # count is a record count rather than the paired-artifact count.
            with path.open("r", encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
        result[name] = {"path": str(path.resolve()), "sha256": _sha256(path), "row_count": row_count}
    return result


def _owner_comparison_evidence(
    paths: Iterable[Path], *, source_raw: Path, treatment_raw: Path, canonical_primary: dict[str, Any]
) -> list[dict[str, Any]]:
    """Bind compatible prior owner comparisons without treating them as authority.

    A comparison whose recorded raw digests match is checked against this
    script's fresh primary-threshold calculation.  Other supplied JSON files
    remain provenance-only, which permits a caller to carry related analyses
    without accidentally mixing cohorts.
    """
    source_digest = _sha256(source_raw)
    treatment_digest = _sha256(treatment_raw)
    evidence: list[dict[str, Any]] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RobustnessError(f"owner comparison is not valid JSON: {path}") from exc
        if not isinstance(document, dict):
            raise RobustnessError(f"owner comparison must be a JSON object: {path}")
        inputs = document.get("inputs", {})
        arm_a = inputs.get("arm_a", {}) if isinstance(inputs, dict) else {}
        arm_b = inputs.get("arm_b", {}) if isinstance(inputs, dict) else {}
        bound = isinstance(arm_a, dict) and isinstance(arm_b, dict) and arm_a.get("sha256") == source_digest and arm_b.get("sha256") == treatment_digest
        record: dict[str, Any] = {
            "path": str(path.resolve()), "sha256": _sha256(path), "binds_source_treatment": bound,
            "recorded_row_count": document.get("arm_a", {}).get("row_count") if isinstance(document.get("arm_a"), dict) else None,
        }
        policy = document.get("policy", {})
        if bound and isinstance(policy, dict) and policy.get("match_iou_threshold") == PRIMARY_THRESHOLD:
            fields = ("unique_matched_gt_owners", "prediction_count", "strict_physical_owner_duplicate_candidate_count")
            for arm_name in ("arm_a", "arm_b"):
                recorded = document.get(arm_name, {})
                fresh = canonical_primary[arm_name]
                if not isinstance(recorded, dict) or any(recorded.get(field) != fresh.get(field) for field in fields):
                    raise RobustnessError(f"bound owner comparison disagrees with fresh primary metrics: {path}")
            record["fresh_primary_metric_check"] = "passed"
        else:
            record["fresh_primary_metric_check"] = "not_applicable"
        evidence.append(record)
    return evidence


def summarize(
    *,
    source_raw: Path,
    treatment_raw: Path,
    source_scored: Path,
    treatment_scored: Path,
    source_trace: Path | None = None,
    treatment_trace: Path | None = None,
    owner_comparisons: Iterable[Path] = (),
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    trim_proportion: float = 0.10,
) -> dict[str, Any]:
    owner_comparisons = tuple(owner_comparisons)
    paths: dict[str, Path] = {
        "source_raw": source_raw, "treatment_raw": treatment_raw,
        "source_scored": source_scored, "treatment_scored": treatment_scored,
    }
    if source_trace is not None:
        paths["source_trace"] = source_trace
    if treatment_trace is not None:
        paths["treatment_trace"] = treatment_trace
    for index, path in enumerate(owner_comparisons):
        paths[f"owner_comparison_{index}"] = path
    source = _read_jsonl(source_raw)
    treatment = _read_jsonl(treatment_raw)
    source_scored_rows = _read_jsonl(source_scored)
    treatment_scored_rows = _read_jsonl(treatment_scored)
    if set(source) != set(treatment) or set(source) != set(source_scored_rows) or set(source) != set(treatment_scored_rows):
        raise RobustnessError("raw and scored artifacts must carry the same paired row IDs")

    comparisons: dict[str, Any] = {}
    primary_per_image: list[dict[str, Any]] | None = None
    for threshold in THRESHOLDS:
        # The imported comparison is the canonical all-image metric calculation.
        comparison = compare_artifacts(source_raw, treatment_raw, match_iou_threshold=threshold)
        per_image = _per_image(source, treatment, threshold)
        comparisons[f"{threshold:.2f}"] = {
            "aggregate": _aggregate(per_image),
            "canonical_comparison": comparison,
        }
        if threshold == PRIMARY_THRESHOLD:
            primary_per_image = per_image
    assert primary_per_image is not None
    source_natural, treatment_natural, natural_counts = _paired_natural_stop_subset(source, treatment)
    natural_per_image = _per_image(source_natural, treatment_natural, PRIMARY_THRESHOLD)
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": _input_provenance(paths),
        "policy": {
            "primary_match_iou_threshold": PRIMARY_THRESHOLD,
            "threshold_sensitivity": list(THRESHOLDS),
            "symmetric_trim_proportion_each_tail": trim_proportion,
            "token_cutoff_policy": "include complete object spans only when max(pred_score_source.generated_step_indices) < cutoff",
            "owner_yield_denominator": "valid parsed predictions",
            "strict_duplicate_rate_denominator": "valid parsed predictions",
            "unmatched_predictions_are_not_hallucinations": True,
            "caveats": [
                "Prediction scores are policy-logprob-derived annotations, not calibrated probabilities or causal evidence.",
                "Owner matching and strict duplicates are geometry-derived under the imported global matcher; strict duplicate candidates are not human-confirmed duplicates.",
            ],
        },
        "full_cohort": {
            "primary": comparisons["0.50"]["aggregate"],
            "influence": _influence(primary_per_image, trim_proportion),
            "threshold_sensitivity": {key: value["aggregate"] for key, value in comparisons.items()},
            "canonical_comparisons": {key: value["canonical_comparison"] for key, value in comparisons.items()},
        },
        "owner_comparison_evidence": _owner_comparison_evidence(
            owner_comparisons, source_raw=source_raw, treatment_raw=treatment_raw,
            canonical_primary=comparisons["0.50"]["canonical_comparison"],
        ),
        "paired_natural_stop_subset": {"selection": natural_counts, "primary": _aggregate(natural_per_image)},
        "fixed_token_cutoff_curve": _cutoff_curve(source_scored_rows, treatment_scored_rows, cutoffs),
        "per_image": primary_per_image,
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows])


def write_outputs(result: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_tsv(output_root / "per_image.tsv", list(result["per_image"]))
    _write_tsv(output_root / "token_cutoff.tsv", list(result["fixed_token_cutoff_curve"]))
    threshold_rows = [{"match_iou_threshold": key, **value} for key, value in result["full_cohort"]["threshold_sensitivity"].items()]
    _write_tsv(output_root / "threshold_sensitivity.tsv", threshold_rows)


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--token-cutoffs must be comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--treatment-raw", type=Path, required=True)
    parser.add_argument("--source-scored", type=Path, required=True)
    parser.add_argument("--treatment-scored", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path)
    parser.add_argument("--treatment-trace", type=Path)
    parser.add_argument("--owner-comparison", type=Path, action="append", default=[])
    parser.add_argument("--token-cutoffs", type=_parse_cutoffs, default=DEFAULT_CUTOFFS)
    parser.add_argument("--trim-proportion", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        source_raw=args.source_raw, treatment_raw=args.treatment_raw,
        source_scored=args.source_scored, treatment_scored=args.treatment_scored,
        source_trace=args.source_trace, treatment_trace=args.treatment_trace,
        owner_comparisons=args.owner_comparison, cutoffs=args.token_cutoffs,
        trim_proportion=args.trim_proportion,
    )
    write_outputs(result, args.output_root)
    print(json.dumps({"image_count": result["full_cohort"]["primary"]["image_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
