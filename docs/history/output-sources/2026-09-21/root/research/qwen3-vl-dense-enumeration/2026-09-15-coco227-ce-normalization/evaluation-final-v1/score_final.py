#!/usr/bin/env python3
"""CPU-only final scoring for the frozen COCO227 CE-normalization trial.

The input must be the completed ``trial-v1/readback/result.json`` collection.
This script re-admits every endpoint through the frozen evaluator, writes one
score per endpoint, a compact JSON/CSV summary, and a non-decision figure.  It
never launches generation or training and refuses partial collections.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path("/data/CoordExp/.worktrees/research-probes")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from probes.training_set_completion import coco227_evaluation as evaluation
from probes.training_set_completion import coco227_trial as trial_module


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT = ROOT / "trial-v1/readback/result.json"
DEFAULT_PREPARATION = ROOT / "evaluation-v1/preparation-v3.json"
DEFAULT_SOURCE0_SCORE = ROOT / "evaluation-v1/source0-score-v2.json"
STEPS = (8, 16, 32, 64, 128, 256)
ARMS = ("S", "T")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return evaluation.read(path.resolve(strict=True))


def _check_binding(value: Mapping[str, Any], *, name: str) -> Path:
    require(isinstance(value, Mapping), f"{name} binding")
    path = Path(str(value.get("path", ""))).resolve(strict=True)
    require(evaluation.binding(path) == dict(value), f"{name} bytes changed")
    return path


def _publish_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists() and not path.is_symlink(), f"publication collision: {path}")
    require(rows, "CSV rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _diagnostic_counts(score: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for ledger in ("scoped227", "historical232", "current-known248"):
        totals = {"target_count": 0, "matched_count": 0, "fn_count": 0}
        for image in score["per_image"]:
            row = image["ledgers_iou_0_8_diagnostic"][ledger]
            for key in totals:
                totals[key] += int(row[key])
        result[ledger] = totals
    return result


def _error_burden(raw: Mapping[str, Any]) -> int:
    """Count output-error events for display only; it is not a composite metric."""

    fields = (
        "malformed_non_geometry",
        "geometry_invalid",
        "outside_literal_coco80",
        "duplicate_candidate_pairs",
        "confirmed_fp",
        "physical_unknown",
        "reviewed_physical_repeat",
        "reviewed_invalid_output",
        "eos_debt",
        "cap_debt",
    )
    return sum(int(raw.get(field, 0)) for field in fields)


def _compact(score: Mapping[str, Any], *, source_kind: str) -> dict[str, Any]:
    primary = score["aggregate"]["ledgers_iou_0_5"]
    micro = score["aggregate"]["annotation_relative_micro_scoped227"]
    raw = score["aggregate"]["raw_and_physical"]
    return {
        "arm": score["arm"],
        "step": int(score["step"]),
        "source_kind": source_kind,
        "joint227": {
            key: primary["scoped227"][key]
            for key in ("target_count", "matched_count", "fn_count", "fn_rate", "class_correct_count", "class_wrong_count")
        },
        "old218": {
            key: primary["old218"][key]
            for key in ("target_count", "matched_count", "fn_count", "fn_rate", "class_correct_count", "class_wrong_count")
        },
        "new9": {
            key: primary["new9"][key]
            for key in ("target_count", "matched_count", "fn_count", "fn_rate", "class_correct_count", "class_wrong_count")
        },
        "historical232": {
            key: primary["historical232"][key]
            for key in ("target_count", "matched_count", "fn_count", "fn_rate")
        },
        "current_known248": {
            key: primary["current-known248"][key]
            for key in ("target_count", "matched_count", "fn_count", "fn_rate")
        },
        "annotation_relative_micro_scoped227": {
            key: micro[key]
            for key in ("tp", "prediction_denominator_all_valid_parsed_rows", "annotation_unmatched_prediction_count", "fn_scoped227", "precision", "f1")
        },
        "raw_and_physical": dict(raw),
        "iou_0_8_diagnostic": _diagnostic_counts(score),
        "clean_completion": score["clean_completion"],
        "display_error_event_count": _error_burden(raw),
    }


def _csv_row(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item["raw_and_physical"]
    micro = item["annotation_relative_micro_scoped227"]
    diagnostic = item["iou_0_8_diagnostic"]
    return {
        "arm": item["arm"],
        "step": item["step"],
        "source_kind": item["source_kind"],
        "joint227_matched": item["joint227"]["matched_count"],
        "joint227_fn": item["joint227"]["fn_count"],
        "joint227_fn_rate": item["joint227"]["fn_rate"],
        "old218_matched": item["old218"]["matched_count"],
        "old218_fn": item["old218"]["fn_count"],
        "new9_matched": item["new9"]["matched_count"],
        "new9_fn": item["new9"]["fn_count"],
        "historical232_matched": item["historical232"]["matched_count"],
        "historical232_fn": item["historical232"]["fn_count"],
        "current_known248_matched": item["current_known248"]["matched_count"],
        "current_known248_fn": item["current_known248"]["fn_count"],
        "annotation_precision": micro["precision"],
        "annotation_f1": micro["f1"],
        "all_valid_predictions": micro["prediction_denominator_all_valid_parsed_rows"],
        "iou08_joint227_matched": diagnostic["scoped227"]["matched_count"],
        "clean_complete": item["clean_completion"]["clean_complete"],
        "clean_failure_counts": evaluation.canonical(item["clean_completion"]["failure_counts"]).decode().strip(),
        "parser_dropped": raw.get("parser_dropped_total", 0),
        "geometry_invalid": raw.get("geometry_invalid", 0),
        "outside_literal_coco80": raw.get("outside_literal_coco80", 0),
        "duplicate_candidate_pairs": raw.get("duplicate_candidate_pairs", 0),
        "confirmed_fp": raw.get("confirmed_fp", 0),
        "physical_unknown": raw.get("physical_unknown", 0),
        "reviewed_physical_repeat": raw.get("reviewed_physical_repeat", 0),
        "reviewed_invalid_output": raw.get("reviewed_invalid_output", 0),
        "eos_debt": raw.get("eos_debt", 0),
        "cap_debt": raw.get("cap_debt", 0),
        "display_error_event_count": item["display_error_event_count"],
    }


def _compact_comparison(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            key: value["per_ledger"][name][key]
            for key in ("retained_count", "gained_count", "lost_count")
        }
        for name in ("scoped227", "old218", "new9", "historical232", "current-known248")
    }


def _validate_result(
    result_path: Path, *, preparation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, int], tuple[Path, Path]]]:
    result = read(result_path)
    require(
        result.get("schema") == "training_set_completion.coco227_ce_normalization.v1.readback_result.v1"
        and result.get("status") == "completed_unscored",
        "completed trial readback collection",
    )
    trial_path = _check_binding(result.get("trial"), name="trial collection")
    trial = trial_module.validate_trial(read(trial_path))
    require(result.get("source0") == trial["readback"]["source0"], "result source0 identity")
    require(result.get("request_count") == 132 and result.get("endpoint_count") == 12, "readback collection cardinality")
    endpoints = result.get("endpoints")
    require(isinstance(endpoints, list) and len(endpoints) == 12, "readback endpoint inventory")
    index: dict[tuple[str, int], tuple[Path, Path]] = {}
    for item in endpoints:
        require(isinstance(item, Mapping), "readback endpoint entry")
        arm, step = item.get("arm"), item.get("step")
        require(arm in ARMS and step in STEPS, "readback endpoint identity")
        key = (str(arm), int(step))
        require(key not in index, "duplicate readback endpoint")
        index[key] = (
            _check_binding(item.get("rows"), name=f"{arm}/{step} endpoint"),
            _check_binding(item.get("admission"), name=f"{arm}/{step} admission"),
        )
    require(set(index) == {(arm, step) for arm in ARMS for step in STEPS}, "readback endpoint cross product")
    summary_rows = result.get("rows")
    require(isinstance(summary_rows, list) and len(summary_rows) == 132, "readback result row inventory")
    by_key = {(str(row.get("arm")), int(row.get("step", -1)), int(row.get("image_id", -1))): row for row in summary_rows}
    require(len(by_key) == 132, "readback result duplicate rows")
    for (arm, step), (endpoint_path, _) in index.items():
        endpoint = read(endpoint_path)
        require(endpoint.get("arm") == arm and endpoint.get("checkpoint_step") == step, "endpoint identity")
        endpoint_rows, endpoint_bindings = endpoint.get("rows"), endpoint.get("row_bindings")
        require(isinstance(endpoint_rows, list) and isinstance(endpoint_bindings, list) and len(endpoint_rows) == len(endpoint_bindings) == 11, "endpoint row cardinality")
        for row, row_binding in zip(endpoint_rows, endpoint_bindings):
            result_row = by_key.get((arm, step, int(row["image_id"])))
            require(isinstance(result_row, Mapping), "result row absent from endpoint")
            require(result_row.get("row") == row_binding, "result row binding")
            require(result_row.get("generated_token_ids_sha256") == row.get("generated_token_ids_sha256"), "result row token hash")
            require(result_row.get("decode_stop_reason") == row.get("decode_stop_reason"), "result row stop")
    require(result["trial"]["path"] == str(trial_path), "result/trial path")
    require(trial["teacher"] == preparation["sources"]["teacher_bank"], "trial/preparation teacher")
    return result, index


def _source0_score(path: Path, *, preparation_path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    score = read(path)
    require(
        score.get("schema") == evaluation.SCHEMA
        and score.get("status") == "saved_readback_scored_not_model_selection"
        and score.get("arm") == "source0"
        and score.get("step") == 0,
        "source0 scored baseline",
    )
    require(score.get("sources", {}).get("preparation") == evaluation.binding(preparation_path), "source0 preparation identity")
    source0_path = _check_binding(result.get("source0"), name="trial source0 admission")
    require(read(source0_path) == score["sources"]["readback_admission"], "source0 admission identity")
    return score


def _score_endpoint(
    *, preparation_path: Path, preparation: Mapping[str, Any], arm: str, step: int, endpoint_path: Path, admission_path: Path
) -> dict[str, Any]:
    admission, rows = evaluation._source_for_endpoint(admission_path, endpoint_path, preparation=preparation)
    score = evaluation.score_admitted_rows(
        preparation_path=preparation_path,
        label=f"{arm}-step-{step:05d}",
        rows=rows,
        readback_admission=admission,
    )
    require(score["arm"] == arm and score["step"] == step, "scored endpoint identity")
    return score


def _figure(path: Path, *, source0: Mapping[str, Any], endpoints: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    require(not path.exists() and not path.is_symlink(), f"publication collision: {path}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = _compact(source0, source_kind="reused_source0")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for arm, color in (("S", "#1f77b4"), ("T", "#d62728")):
        items = [baseline] + [_compact(endpoints[(arm, step)], source_kind="scientific_endpoint") for step in STEPS]
        x = [item["step"] for item in items]
        axes[0].plot(x, [item["joint227"]["matched_count"] for item in items], marker="o", color=color, label=f"{arm} joint227")
        axes[0].plot(x, [item["new9"]["matched_count"] for item in items], marker="x", linestyle="--", color=color, label=f"{arm} new9")
        axes[1].plot(x, [item["old218"]["matched_count"] for item in items], marker="o", color=color, label=f"{arm} old218")
        axes[2].plot(x, [item["display_error_event_count"] for item in items], marker="o", color=color, label=f"{arm} output errors")
    axes[0].set(title="Acquisition", xlabel="saved checkpoint", ylabel="joint matches")
    axes[0].set_ylim(-2, 230)
    axes[1].set(title="Old218 retention", xlabel="saved checkpoint", ylabel="matched owners")
    axes[1].set_ylim(-2, 220)
    axes[2].set(title="Complete-output errors", xlabel="saved checkpoint", ylabel="event count")
    for axis in axes:
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("COCO227 CE-normalization: saved readbacks; no scalar winner")
    fig.savefig(path, dpi=180, metadata={"Title": "COCO227 CE-normalization saved-readback summary"})
    plt.close(fig)


def run(*, result_path: Path, preparation_path: Path, source0_score_path: Path, output: Path) -> dict[str, Any]:
    """Score the complete collection and publish immutable CPU artifacts."""

    output = output.resolve()
    require(output == SCRIPT_ROOT, "final evaluation output root")
    preparation_path = preparation_path.resolve(strict=True)
    preparation = read(preparation_path)
    evaluation.validate_preparation(preparation)
    result, index = _validate_result(result_path.resolve(strict=True), preparation=preparation)
    source0 = _source0_score(source0_score_path.resolve(strict=True), preparation_path=preparation_path, result=result)

    scores: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for step in STEPS:
            endpoint_path, admission_path = index[(arm, step)]
            scores[(arm, step)] = _score_endpoint(
                preparation_path=preparation_path,
                preparation=preparation,
                arm=arm,
                step=step,
                endpoint_path=endpoint_path,
                admission_path=admission_path,
            )

    score_paths: dict[tuple[str, int], Path] = {}
    for key, score in scores.items():
        arm, step = key
        path = output / "scores" / f"{arm}-step-{step:05d}.json"
        evaluation.publish(path, score)
        score_paths[key] = path

    compact_source0 = _compact(source0, source_kind="reused_source0")
    compact_endpoints = [
        _compact(scores[(arm, step)], source_kind="scientific_endpoint")
        for arm in ARMS
        for step in STEPS
    ]
    comparisons: dict[str, Any] = {"source0_to_endpoint": {}}
    for arm in ARMS:
        comparisons["source0_to_endpoint"][arm] = {}
        for step in STEPS:
            comparison = evaluation.compare(baseline=source0, endpoint=scores[(arm, step)], label=f"source0-to-{arm}-step-{step:05d}")
            comparisons["source0_to_endpoint"][arm][str(step)] = _compact_comparison(comparison)
    final_comparison = evaluation.compare(baseline=scores[("S", 256)], endpoint=scores[("T", 256)], label="S-versus-T-step-00256")
    comparisons["S_to_T_final256"] = _compact_comparison(final_comparison)
    milestones = {
        arm: evaluation.sustained_clean_milestone([scores[(arm, step)] for step in STEPS], arm=arm)
        for arm in ARMS
    }

    csv_rows = [_csv_row(compact_source0)] + [_csv_row(item) for item in compact_endpoints]
    _publish_csv(output / "checkpoint-summary.csv", csv_rows)
    _figure(output / "acquisition-retention-errors.png", source0=source0, endpoints=scores)
    summary = {
        "schema": "training_set_completion.coco227_ce_normalization.final_scoring.v1",
        "status": "completed_cpu_scoring_not_model_selection",
        "sources": {
            "trial_readback_result": evaluation.binding(result_path),
            "evaluation_preparation": evaluation.binding(preparation_path),
            "source0_score": evaluation.binding(source0_score_path),
            "producer": evaluation.binding(Path(__file__)),
            "evaluator": evaluation.binding(Path(evaluation.__file__)),
        },
        "scored_endpoint_files": [
            {"arm": arm, "step": step, "score": evaluation.binding(score_paths[(arm, step)])}
            for arm in ARMS
            for step in STEPS
        ],
        "source0": compact_source0,
        "endpoints": compact_endpoints,
        "sustained_clean_milestones": milestones,
        "retention_comparisons": comparisons,
        "artifacts": {
            "checkpoint_summary_csv": evaluation.binding(output / "checkpoint-summary.csv"),
            "acquisition_retention_errors_png": evaluation.binding(output / "acquisition-retention-errors.png"),
        },
        "decision_boundary": "No scalar composite, automatic winner, visual claim, or publication decision is produced. Root assesses final256 and the full trajectory ledger.",
    }
    evaluation.publish(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--source0-score", type=Path, default=DEFAULT_SOURCE0_SCORE)
    parser.add_argument("--output", type=Path, default=SCRIPT_ROOT)
    args = parser.parse_args()
    summary = run(
        result_path=args.trial_result,
        preparation_path=args.preparation,
        source0_score_path=args.source0_score,
        output=args.output,
    )
    print({"summary": str((Path(args.output) / "summary.json").resolve()), "status": summary["status"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
