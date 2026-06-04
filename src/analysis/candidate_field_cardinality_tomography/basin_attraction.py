from __future__ import annotations

from typing import Mapping

from .taxonomy import assign_primary_bucket


def classify_basin_attraction(row: Mapping[str, object]) -> str:
    if "target_tail_beats_competitor_tail" not in row:
        return "A3_unresolved"
    if not bool(row.get("target_tail_beats_competitor_tail")):
        return "A3b_tail_representation_failure"
    greedy_rank = int(row.get("greedy_target_iou_rank") or 0)
    greedy_iou = float(row.get("greedy_target_iou") or 0.0)
    best_competitor_iou = float(row.get("best_same_desc_competitor_iou") or 0.0)
    if greedy_rank != 1 or best_competitor_iou > greedy_iou:
        return "A3a_decode_basin_failure"
    return "basin_success"


__all__ = ["assign_primary_bucket", "classify_basin_attraction"]
