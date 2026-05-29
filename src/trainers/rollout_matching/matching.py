"""Rollout-matching: greedy matching and cost computation helpers.

This module is intentionally import-light with respect to trainers (no swift/HF
trainer imports). It provides the stable matching surface used by both
rollout-matching SFT and Stage-2 rollout-correction.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from pycocotools import mask as maskUtils

from src.common.geometry import bbox_from_points, bbox_to_quadrilateral
from src.training.stage2.assignment import AssignmentObject, GreedyIoUAssignment

from .contracts import GTObject, GeomType, MatchResult


def _bbox_xyxy_from_norm(
    points: Sequence[int], kind: GeomType
) -> Tuple[float, float, float, float]:
    if kind == "bbox_2d":
        if len(points) != 4:
            return 0.0, 0.0, 0.0, 0.0
        x1, y1, x2, y2 = [float(v) for v in points]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    x1, y1, x2, y2 = bbox_from_points([float(v) for v in points])
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _mask_iou_norm1000(
    *,
    pred_kind: GeomType,
    pred_points: Sequence[int],
    gt_kind: GeomType,
    gt_points: Sequence[int],
    resolution: int,
) -> float:
    """maskIoU in norm1000 space on a virtual RxR canvas."""
    r = int(resolution)
    if r <= 0:
        return 0.0

    def _clamp01k(values: Sequence[int]) -> List[float]:
        return [float(min(max(int(v), 0), 999)) for v in values]

    def _project(values: Sequence[float]) -> List[float]:
        # Project [0,999] -> [0,R-1] continuous coordinates.
        # Mirror ints_to_pixels_norm1000: frac=v/999, then scale by (R-1).
        denom = 999.0
        scale = float(max(r - 1, 1)) / denom
        return [float(v) * scale for v in values]

    def _as_poly(kind: GeomType, pts: Sequence[int]) -> List[float]:
        if kind == "bbox_2d":
            if len(pts) != 4:
                return []
            x1, y1, x2, y2 = [float(v) for v in pts]
            quad = bbox_to_quadrilateral(
                [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            )
            return [float(v) for v in quad]
        return [float(v) for v in pts]

    p_poly = _project(_clamp01k(_as_poly(pred_kind, pred_points)))
    g_poly = _project(_clamp01k(_as_poly(gt_kind, gt_points)))
    if len(p_poly) < 6 or len(g_poly) < 6:
        return 0.0

    try:
        rle_p = maskUtils.frPyObjects([p_poly], r, r)
        rle_g = maskUtils.frPyObjects([g_poly], r, r)
        if isinstance(rle_p, list):
            rle_p = maskUtils.merge(rle_p)
        if isinstance(rle_g, list):
            rle_g = maskUtils.merge(rle_g)
        ious = maskUtils.iou([rle_p], [rle_g], [0])
        return float(ious[0][0]) if getattr(ious, "size", 0) else 0.0
    except (IndexError, TypeError, ValueError, RuntimeError):
        return 0.0


def _assignment_object_from_gt_object(
    *,
    obj: GTObject,
    role: str,
    position: int,
) -> AssignmentObject:
    """Return a Stage-2 assignment object from a rollout-matching object."""

    return AssignmentObject(
        object_id=f"{role}:{int(position)}:{int(obj.index)}",
        bbox=_bbox_xyxy_from_norm(obj.points_norm1000, obj.geom_type),
        description=str(obj.desc),
        metadata={
            "source_index": int(position),
            "object_index": int(obj.index),
        },
    )


def greedy_match_iou(
    *,
    preds: Sequence[GTObject],
    gts: Sequence[GTObject],
    gate_threshold: float,
) -> MatchResult:
    """Return deterministic greedy IoU matches for rollout predictions and GT.

    The returned ``MatchResult`` keeps the historical field names used by
    telemetry, but the summed score is bbox IoU from the greedy assignment.
    """

    pred_n = len(preds)
    gt_n = len(gts)
    if pred_n == 0:
        return MatchResult(
            matched_pairs=[],
            fn_gt_indices=list(range(gt_n)),
            fp_pred_indices=[],
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )
    if gt_n == 0:
        return MatchResult(
            matched_pairs=[],
            fn_gt_indices=[],
            fp_pred_indices=list(range(pred_n)),
            gating_rejections=0,
            matched_maskiou_sum=0.0,
            matched_maskiou_count=0,
        )

    strategy = GreedyIoUAssignment(iou_threshold=float(gate_threshold))
    result = strategy.assign(
        predictions=[
            _assignment_object_from_gt_object(
                obj=pred,
                role="pred",
                position=pred_i,
            )
            for pred_i, pred in enumerate(preds)
        ],
        ground_truth=[
            _assignment_object_from_gt_object(
                obj=gt,
                role="gt",
                position=gt_i,
            )
            for gt_i, gt in enumerate(gts)
        ],
    )

    matched_pairs = [
        (int(pair.prediction_index), int(pair.ground_truth_index))
        for pair in result.pairs
    ]
    fp_preds = [int(item.index) for item in result.unmatched_predictions]
    fn_gts = [int(item.index) for item in result.unmatched_ground_truth]
    matched_iou_sum = float(sum(float(pair.iou) for pair in result.pairs))

    return MatchResult(
        matched_pairs=matched_pairs,
        fn_gt_indices=fn_gts,
        fp_pred_indices=fp_preds,
        gating_rejections=0,
        matched_maskiou_sum=float(matched_iou_sum),
        matched_maskiou_count=int(len(result.pairs)),
    )


def associate_one_to_one_greedy_iou(
    *,
    anchors: Sequence[GTObject],
    explorers: Sequence[GTObject],
    min_iou: float,
) -> List[Tuple[int, int]]:
    """Associate anchor/explorer objects by deterministic greedy IoU."""

    threshold = float(min_iou)
    if not math.isfinite(threshold):
        raise ValueError("min_iou must be finite")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("min_iou must be in [0, 1]")

    if not anchors or not explorers:
        return []

    anchor_boxes = [
        _bbox_xyxy_from_norm(obj.points_norm1000, obj.geom_type) for obj in anchors
    ]
    explorer_boxes = [
        _bbox_xyxy_from_norm(obj.points_norm1000, obj.geom_type) for obj in explorers
    ]

    candidates: List[Tuple[int, int, float]] = []
    for anchor_i, anchor_box in enumerate(anchor_boxes):
        for explorer_i, explorer_box in enumerate(explorer_boxes):
            iou = _bbox_iou_xyxy(anchor_box, explorer_box)
            if float(iou) < float(threshold):
                continue
            candidates.append((int(anchor_i), int(explorer_i), float(iou)))

    candidates.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))

    matched_anchors: set[int] = set()
    matched_explorers: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for anchor_i, explorer_i, _score in candidates:
        if int(anchor_i) in matched_anchors:
            continue
        if int(explorer_i) in matched_explorers:
            continue
        pairs.append((int(anchor_i), int(explorer_i)))
        matched_anchors.add(int(anchor_i))
        matched_explorers.add(int(explorer_i))

    return pairs


__all__ = ["associate_one_to_one_greedy_iou", "greedy_match_iou"]
