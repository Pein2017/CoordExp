#!/usr/bin/env python3
"""Render the descriptive v2 COCO227 saved-readback figure from frozen scores."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path("/data/CoordExp/.worktrees/research-probes")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from probes.training_set_completion import coco227_evaluation as evaluation


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-15-coco227-ce-normalization"
)
OUTPUT = Path(__file__).resolve().parent
SUMMARY = OUTPUT / "summary.json"
CHECKPOINTS = (0, 8, 16, 32, 64, 128, 256)
ARMS = ("S", "T")
COLORS = {"S": "#1f77b4", "T": "#d62728"}


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


def _rows(summary: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    source0 = summary.get("source0")
    endpoints = summary.get("endpoints")
    require(isinstance(source0, Mapping) and isinstance(endpoints, list), "summary rows")
    values: dict[tuple[str, int], Mapping[str, Any]] = {("source0", 0): source0}
    for row in endpoints:
        require(isinstance(row, Mapping), "endpoint compact row")
        arm, step = row.get("arm"), row.get("step")
        require(arm in ARMS and step in CHECKPOINTS[1:], "endpoint compact identity")
        key = (str(arm), int(step))
        require(key not in values, "duplicate compact endpoint")
        values[key] = row
    require(set(values) == {("source0", 0)} | {(arm, step) for arm in ARMS for step in CHECKPOINTS[1:]}, "compact checkpoint cross product")
    return values


def _verify_summary(summary_path: Path) -> tuple[dict[str, Any], dict[tuple[str, int], Mapping[str, Any]]]:
    summary = read(summary_path)
    require(summary.get("schema") == "training_set_completion.coco227_ce_normalization.final_scoring.v1", "summary schema")
    require(summary.get("status") == "completed_cpu_scoring_not_model_selection", "summary status")
    _check_binding(summary["sources"]["trial_readback_result"], name="trial readback result")
    _check_binding(summary["sources"]["evaluation_preparation"], name="evaluation preparation")
    _check_binding(summary["sources"]["source0_score"], name="source0 score")
    score_index = {(str(item["arm"]), int(item["step"])): _check_binding(item["score"], name="endpoint score") for item in summary["scored_endpoint_files"]}
    require(set(score_index) == {(arm, step) for arm in ARMS for step in CHECKPOINTS[1:]}, "scored endpoint files")
    values = _rows(summary)
    for (arm, step), path in score_index.items():
        score = read(path)
        compact = values[(arm, step)]
        require(score.get("arm") == arm and score.get("step") == step, "endpoint score identity")
        primary = score["aggregate"]["ledgers_iou_0_5"]
        raw = score["aggregate"]["raw_and_physical"]
        require(compact["new9"]["matched_count"] == primary["new9"]["matched_count"], "new9 compact identity")
        require(compact["old218"]["matched_count"] == primary["old218"]["matched_count"], "old218 compact identity")
        require(compact["raw_and_physical"]["parser_dropped_total"] == raw["parser_dropped_total"], "parser compact identity")
        require(compact["raw_and_physical"]["physical_unknown"] == raw["physical_unknown"], "unknown compact identity")
    return summary, values


def _series(values: Mapping[tuple[str, int], Mapping[str, Any]], arm: str, path: tuple[str, ...]) -> list[float]:
    selected = [values[("source0", 0)]] + [values[(arm, step)] for step in CHECKPOINTS[1:]]
    result = []
    for row in selected:
        value: Any = row
        for key in path:
            value = value[key]
        result.append(float(value))
    return result


def render(*, summary_path: Path = SUMMARY, output: Path = OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    require(output == OUTPUT, "v2 figure output root")
    figure_path = output / "acquisition-retention-parser-unknown-v2.png"
    receipt_path = output / "figure-v2-receipt.json"
    require(not figure_path.exists() and not figure_path.is_symlink(), f"publication collision: {figure_path}")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), f"publication collision: {receipt_path}")
    summary, values = _verify_summary(summary_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(CHECKPOINTS)))
    labels = ["0\nsource0", "8", "16", "32", "64", "128", "256"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for arm in ARMS:
        color = COLORS[arm]
        axes[0].plot(x, _series(values, arm, ("new9", "matched_count")), marker="o", color=color, label=f"{arm} new9")
        axes[1].plot(x, _series(values, arm, ("old218", "matched_count")), marker="o", color=color, label=f"{arm} old218")
        axes[2].plot(x, _series(values, arm, ("raw_and_physical", "parser_dropped_total")), marker="o", color=color, label=f"{arm} parser-dropped")
        axes[2].plot(x, _series(values, arm, ("raw_and_physical", "physical_unknown")), marker="x", linestyle="--", color=color, label=f"{arm} physical-unknown")
    axes[0].set(title="New9 acquisition", ylabel="matched owners", ylim=(-0.25, 9.25))
    axes[1].set(title="Old218 retention", ylabel="matched owners", ylim=(216.5, 218.5))
    axes[2].set(title="Parser drops and physical unknown", ylabel="rows")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("saved checkpoint (categorical spacing; not elapsed update distance)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("COCO227 CE-normalization saved readbacks: descriptive trajectories, no scalar winner")
    fig.savefig(figure_path, dpi=180, metadata={"Title": "COCO227 saved readback trajectories v2"})
    plt.close(fig)

    receipt = {
        "schema": "training_set_completion.coco227_ce_normalization.figure_v2.v1",
        "status": "descriptive_figure_rendered_not_model_selection",
        "figure": evaluation.binding(figure_path),
        "entrypoint": evaluation.binding(Path(__file__)),
        "summary": evaluation.binding(summary_path),
        "sources": {
            "trial_readback_result": summary["sources"]["trial_readback_result"],
            "evaluation_preparation": summary["sources"]["evaluation_preparation"],
            "source0_score": summary["sources"]["source0_score"],
            "endpoint_scores": summary["scored_endpoint_files"],
        },
        "series": {
            "panel_1": "new9 matched owners, range 0..9",
            "panel_2": "old218 matched owners",
            "panel_3": "parser_dropped_total and physical_unknown are separate labeled series and are never summed",
            "x_axis": "equally spaced saved checkpoint categories: source0, 8, 16, 32, 64, 128, 256; spacing does not represent elapsed update distance",
        },
        "non_claims": [
            "physical unknown remains unresolved, not an error or false positive",
            "no scalar composite or automatic winner",
            "no visual owner adjudication or publication claim",
        ],
    }
    evaluation.publish(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = render(summary_path=args.summary, output=args.output)
    print({"figure": receipt["figure"]["path"], "receipt": str((Path(args.output) / "figure-v2-receipt.json").resolve())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
