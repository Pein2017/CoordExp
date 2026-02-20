"""Rollout-matching: matching and cost computation helpers.

This module is intentionally import-light with respect to trainers (no swift/HF
trainer imports). It provides the stable matching surface used by both
rollout-matching SFT and Stage-2 AB.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from pycocotools import mask as maskUtils
from scipy.optimize import linear_sum_assignment

from src.common.geometry import bbox_from_points, bbox_to_quadrilateral

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

    rle_p = maskUtils.frPyObjects([p_poly], r, r)
    rle_g = maskUtils.frPyObjects([g_poly], r, r)
    if isinstance(rle_p, list):
        rle_p = maskUtils.merge(rle_p)
    if isinstance(rle_g, list):
        rle_g = maskUtils.merge(rle_g)
    ious = maskUtils.iou([rle_p], [rle_g], [0])
    return float(ious[0][0]) if getattr(ious, "size", 0) else 0.0


def hungarian_match_maskiou(
    *,
    preds: Sequence[GTObject],
    gts: Sequence[GTObject],
    top_k: int,
    gate_threshold: float,
    mask_resolution: int,
    fp_cost: float,
    fn_cost: float,
) -> MatchResult:
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

    k = max(1, int(top_k))
    gate = float(gate_threshold)
    inf = 1e6

    gt_boxes = [_bbox_xyxy_from_norm(gt.points_norm1000, gt.geom_type) for gt in gts]
    pred_boxes = [
        _bbox_xyxy_from_norm(pr.points_norm1000, pr.geom_type) for pr in preds
    ]

    # Candidate pruning per pred.
    cand: List[List[int]] = []
    for pb in pred_boxes:
        ious = [(_bbox_iou_xyxy(pb, gb), j) for j, gb in enumerate(gt_boxes)]
        ious.sort(key=lambda t: (-t[0], t[1]))
        best = [j for _, j in ious[:k]]
        if ious and ious[0][0] <= 0.0:
            # Fallback: center distance.
            pcx = 0.5 * (pb[0] + pb[2])
            pcy = 0.5 * (pb[1] + pb[3])
            dists = []
            for j, gb in enumerate(gt_boxes):
                gcx = 0.5 * (gb[0] + gb[2])
                gcy = 0.5 * (gb[1] + gb[3])
                d = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
                dists.append((float(d), j))
            dists.sort(key=lambda t: (t[0], t[1]))
            best = [j for _, j in dists[:k]]
        cand.append(best)

    gating_rejections = 0
    cost_pg = np.full((pred_n, gt_n), inf, dtype=np.float64)
    for i, pr in enumerate(preds):
        for j in cand[i]:
            iou = _mask_iou_norm1000(
                pred_kind=pr.geom_type,
                pred_points=pr.points_norm1000,
                gt_kind=gts[j].geom_type,
                gt_points=gts[j].points_norm1000,
                resolution=mask_resolution,
            )
            if iou < gate:
                gating_rejections += 1
                continue
            cost_pg[i, j] = 1.0 - float(iou)

    # Dummy-augmented square matrix.
    n = pred_n + gt_n
    cost = np.full((n, n), 0.0, dtype=np.float64)
    cost[:pred_n, :gt_n] = cost_pg
    cost[:pred_n, gt_n:] = float(fp_cost)
    cost[pred_n:, :gt_n] = float(fn_cost)
    cost[pred_n:, gt_n:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    assign = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    matched_pairs: List[Tuple[int, int]] = []
    matched_maskiou_sum = 0.0
    matched_maskiou_count = 0
    fp_preds: List[int] = []
    matched_gt: set[int] = set()

    for i in range(pred_n):
        c = assign.get(i)
        if c is None:
            fp_preds.append(i)
            continue
        if c < gt_n and cost_pg[i, c] < inf * 0.5:
            matched_pairs.append((i, c))
            matched_gt.add(c)
            # cost_pg is (1 - iou) for allowed candidates.
            iou = 1.0 - float(cost_pg[i, c])
            if iou < 0.0:
                iou = 0.0
            if iou > 1.0:
                iou = 1.0
            matched_maskiou_sum += float(iou)
            matched_maskiou_count += 1
        else:
            fp_preds.append(i)

    fn_gts = [j for j in range(gt_n) if j not in matched_gt]
    return MatchResult(
        matched_pairs=matched_pairs,
        fn_gt_indices=fn_gts,
        fp_pred_indices=fp_preds,
        gating_rejections=int(gating_rejections),
        matched_maskiou_sum=float(matched_maskiou_sum),
        matched_maskiou_count=int(matched_maskiou_count),
    )


__all__ = ["hungarian_match_maskiou"]
