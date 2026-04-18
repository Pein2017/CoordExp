"""Reporting helpers for raw-text coordinate continuity probes."""

from __future__ import annotations

from collections import defaultdict
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
