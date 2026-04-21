"""Reporting helpers for raw-text coordinate continuity probes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("softmax requires at least one value")
    raw = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError("softmax values must be finite")
    shifted = raw - float(np.max(raw))
    weights = np.exp(shifted)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("softmax weights must sum to a positive value")
    return [float(value) for value in (weights / weight_sum)]


def compute_basin_metrics(
    rows: Sequence[dict[str, object]],
    *,
    center_key: str,
) -> dict[str, float]:
    if not rows:
        raise ValueError("compute_basin_metrics requires at least one row")
    scores = [float(row["score"]) for row in rows]
    candidates = [int(row["candidate_value"]) for row in rows]
    center = int(rows[0][center_key])
    weights = _softmax(scores)

    def neighborhood_mass(radius: int) -> float:
        return float(
            sum(
                weight
                for candidate, weight in zip(candidates, weights)
                if abs(candidate - center) <= radius
            )
        )

    local_expected_abs_error = float(
        sum(
            abs(candidate - center) * weight
            for candidate, weight in zip(candidates, weights)
        )
    )
    peak = max(scores)
    floor = min(scores)
    half_height = peak - (peak - floor) / 2.0
    half_height_width = float(
        max(
            abs(candidate - center)
            for candidate, score in zip(candidates, scores)
            if score >= half_height
        )
    )
    return {
        "mass_at_1": neighborhood_mass(1),
        "mass_at_2": neighborhood_mass(2),
        "mass_at_4": neighborhood_mass(4),
        "mass_at_8": neighborhood_mass(8),
        "mass_at_16": neighborhood_mass(16),
        "local_expected_abs_error": local_expected_abs_error,
        "half_height_width": half_height_width,
    }


def compute_vision_lift_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        case_id = str(row["case_id"])
        slot = str(row["slot"])
        image_condition = str(row["image_condition"])
        if image_condition in grouped[(case_id, slot)]:
            raise ValueError(
                f"duplicate image_condition for case_id={case_id} slot={slot}: {image_condition}"
            )
        grouped[(case_id, slot)][image_condition] = float(row["gt_score"])
    lifted_rows = []
    for (case_id, slot), values in sorted(grouped.items()):
        if "correct" not in values or "swapped" not in values:
            continue
        lifted_rows.append(
            {
                "case_id": case_id,
                "slot": slot,
                "vision_lift": float(values["correct"] - values["swapped"]),
            }
        )
    return lifted_rows


def summarize_wrong_anchor_advantage(
    rows: Sequence[dict[str, object]],
) -> dict[str, float]:
    gt_metrics = compute_basin_metrics(rows, center_key="gt_value")
    pred_metrics = compute_basin_metrics(rows, center_key="pred_value")
    summary = {
        "gt_center_mass_at_4": gt_metrics["mass_at_4"],
        "pred_center_mass_at_4": pred_metrics["mass_at_4"],
        "wrong_anchor_advantage_at_4": (
            pred_metrics["mass_at_4"] - gt_metrics["mass_at_4"]
        ),
    }
    if all(row.get("previous_value") is not None for row in rows):
        previous_metrics = compute_basin_metrics(rows, center_key="previous_value")
        summary.update(
            {
                "previous_center_mass_at_4": previous_metrics["mass_at_4"],
                "previous_minus_gt_mass_at_4": (
                    previous_metrics["mass_at_4"] - gt_metrics["mass_at_4"]
                ),
            }
        )
    return summary


def build_xy_heatmap_grid(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    x_values = sorted({int(row["candidate_x1"]) for row in rows})
    y_values = sorted({int(row["candidate_y1"]) for row in rows})
    score_lookup: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["candidate_x1"]), int(row["candidate_y1"]))
        if key in score_lookup:
            raise ValueError(f"duplicate heatmap cell: {key}")
        score_lookup[key] = float(row["score"])
    for y_value in y_values:
        for x_value in x_values:
            if (x_value, y_value) not in score_lookup:
                raise ValueError(f"missing heatmap cell: {(x_value, y_value)}")
    return {
        "x_values": x_values,
        "y_values": y_values,
        "z_matrix": [
            [score_lookup[(x_value, y_value)] for x_value in x_values]
            for y_value in y_values
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_report_bundle(
    *,
    out_dir: Path,
    summary: dict[str, object],
    report_md: str,
    per_coord_rows: Sequence[dict[str, object]],
    hard_cases: Sequence[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "per_coord_scores.jsonl", per_coord_rows)
    _write_jsonl(out_dir / "hard_cases.jsonl", hard_cases)
