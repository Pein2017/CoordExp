"""
Offline detection evaluator for CoordExp (pixel-space schema).

Features:
- Ingests standardized ``pred.jsonl`` (pixel-space ``gt`` / ``pred`` objects) or legacy GT JSONL.
- Converts geometries to COCO-format GT and prediction artifacts (bbox + segm for polygons).
- Runs COCOeval (bbox + segm) and/or a set-matching "F1-ish" metric and emits metrics plus robustness counters.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tqdm import tqdm

from src.common.geometry import (
    bbox_from_points,
    bbox_to_quadrilateral,
    coerce_point_list,
    denorm_and_clamp,
    is_degenerate_bbox,
)
from src.common.geometry.object_geometry import extract_single_geometry
from src.common.lvis_semantics import (
    LvisCategory,
    LvisImagePolicy,
    build_lvis_category_catalog,
    extract_lvis_image_policy,
)
from src.common.duplicate_control import (
    DuplicateControlConfig,
    DuplicateControlObject,
    DuplicateControlResult,
    apply_duplicate_policy,
    duplicate_control_object_from_mapping,
    validate_duplicate_control_config,
)
from src.common.prediction_parsing import GEOM_KEYS
from src.common.semantic_desc import SemanticDescEncoder, normalize_desc
from src.common.io import load_jsonl_with_diagnostics
from src.eval.artifacts import (
    build_per_image_report,
    resolve_duplicate_guard_report_path,
    resolve_guarded_prediction_artifact_path,
    resolve_matches_artifact_path,
    write_jsonl_records,
    write_outputs,
)
from src.utils import get_logger
from src.vis import materialize_gt_vs_pred_vis_resource, render_gt_vs_pred_review

logger = get_logger(__name__)

def _bbox_iou(a: List[float], b: List[float]) -> float:
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


def _object_has_poly(obj: Dict[str, Any]) -> bool:
    if obj.get("type") == "poly":
        return True
    segm = obj.get("segmentation")
    return isinstance(segm, list) and bool(segm)


def _object_segmentation(obj: Dict[str, Any]) -> List[List[float]]:
    if obj.get("type") == "poly":
        pts = obj.get("points") or []
        return [cast(List[float], pts)]
    segm = obj.get("segmentation")
    if isinstance(segm, list) and segm and isinstance(segm[0], list):
        return cast(List[List[float]], segm)
    return [bbox_to_quadrilateral(cast(List[float], obj["bbox"]))]


def _segm_iou(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> float:
    """Segmentation IoU between objects (supports bbox↔poly via rectangle segmentation)."""
    try:
        seg_a = _object_segmentation(a)
        seg_b = _object_segmentation(b)
        rle_a = maskUtils.frPyObjects(seg_a, height, width)
        rle_b = maskUtils.frPyObjects(seg_b, height, width)
        if isinstance(rle_a, list):
            rle_a = maskUtils.merge(rle_a)
        if isinstance(rle_b, list):
            rle_b = maskUtils.merge(rle_b)
        ious = maskUtils.iou([rle_a], [rle_b], [0])
        return float(ious[0][0]) if ious.size else 0.0
    except (IndexError, TypeError, ValueError, RuntimeError):
        return 0.0


def _object_iou_auto(
    pred_obj: Dict[str, Any],
    gt_obj: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> float:
    """Auto-select IoU type: segm when either side has poly, else bbox."""
    if _object_has_poly(pred_obj) or _object_has_poly(gt_obj):
        return _segm_iou(pred_obj, gt_obj, width=width, height=height)
    return _bbox_iou(
        cast(List[float], pred_obj["bbox"]), cast(List[float], gt_obj["bbox"])
    )


def _greedy_match_by_iou(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    *,
    iou_thr: float,
    width: int,
    height: int,
) -> List[Tuple[int, int, float]]:
    """Greedy 1:1 assignment by IoU with deterministic tie-breaking."""
    candidates: List[Tuple[float, int, int]] = []
    thr = float(iou_thr)
    for pred_idx, pred in enumerate(preds):
        for gt_idx, gt in enumerate(gts):
            iou = _object_iou_auto(pred, gt, width=width, height=height)
            if iou >= thr:
                candidates.append((float(iou), int(pred_idx), int(gt_idx)))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matches.append((pred_idx, gt_idx, float(iou)))
    return matches


def _project_bbox_to_source_resolution(
    bbox_xyxy: Sequence[float],
    *,
    pred_width: int,
    pred_height: int,
    source_width: int,
    source_height: int,
) -> List[float]:
    if pred_width <= 0 or pred_height <= 0:
        raise ValueError(
            "Official COCO submission export requires positive prediction width/height "
            f"for resolution rollback, got {(pred_width, pred_height)}."
        )
    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            "Official COCO submission export requires positive source width/height "
            f"for resolution rollback, got {(source_width, source_height)}."
        )
    if len(bbox_xyxy) != 4:
        raise ValueError(
            f"Expected bbox_xyxy with 4 values, got {len(bbox_xyxy)}: {bbox_xyxy!r}"
        )

    sx = float(source_width) / float(pred_width)
    sy = float(source_height) / float(pred_height)
    x1, y1, x2, y2 = bbox_xyxy
    scaled = [
        max(0.0, min(float(source_width), float(x1) * sx)),
        max(0.0, min(float(source_height), float(y1) * sy)),
        max(0.0, min(float(source_width), float(x2) * sx)),
        max(0.0, min(float(source_height), float(y2) * sy)),
    ]
    return scaled
