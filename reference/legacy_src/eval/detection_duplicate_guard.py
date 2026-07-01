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

from src.eval.detection_records import EvalOptions, _wants_official_metrics

_OFFLINE_DUPLICATE_CONTROL_CONFIG = validate_duplicate_control_config(
    iou_threshold=0.90,
    center_radius_scale=0.80,
)


_OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD = 0.50


_SUPPRESSION_REASON_DUPLICATE_CLUSTER = "duplicate_cluster_non_survivor"


def _resolve_guarded_pred_path(
    *,
    pred_path: Path,
    options: EvalOptions,
) -> Path:
    if options.guarded_pred_path is not None:
        return options.guarded_pred_path
    return resolve_guarded_prediction_artifact_path(
        out_dir=options.output_dir,
        scored_input=_wants_official_metrics(options.metrics),
    )


def _resolve_duplicate_guard_report_path_for_options(options: EvalOptions) -> Path:
    if options.duplicate_guard_report_path is not None:
        return options.duplicate_guard_report_path
    return resolve_duplicate_guard_report_path(out_dir=options.output_dir)


def _duplicate_control_objects_for_record(
    record: Mapping[str, Any],
) -> Tuple[List[DuplicateControlObject], set[int]]:
    raw_predictions = record.get("pred")
    pred_key = "pred"
    if not isinstance(raw_predictions, list):
        raw_predictions = record.get("predictions")
        pred_key = "predictions"
    if not isinstance(raw_predictions, list):
        return [], set()

    width_raw = record.get("width")
    height_raw = record.get("height")
    try:
        width = int(width_raw) if width_raw is not None else None
        height = int(height_raw) if height_raw is not None else None
    except (TypeError, ValueError):
        width = None
        height = None

    objects: List[DuplicateControlObject] = []
    controlled_indices: set[int] = set()
    for pred_index, obj in enumerate(raw_predictions):
        if not isinstance(obj, Mapping):
            continue
        candidate = duplicate_control_object_from_mapping(
            obj,
            index=pred_index,
            width=width,
            height=height,
            source=pred_key,
        )
        if candidate is None:
            continue
        objects.append(candidate)
        controlled_indices.add(int(pred_index))
    return objects, controlled_indices


def _record_identity(record: Mapping[str, Any], fallback_index: int) -> Dict[str, Any]:
    image_id = record.get("image_id", record.get("index", fallback_index))
    image = record.get("image")
    images = record.get("images")
    return {
        "record_index": int(fallback_index),
        "image_id": image_id,
        "image": image,
        "images": images if isinstance(images, list) else None,
    }


def _build_duplicate_guard_report(
    *,
    pred_records: Sequence[Mapping[str, Any]],
    results_by_record: Sequence[Tuple[DuplicateControlResult, set[int]]],
) -> Dict[str, Any]:
    total_predictions_inspected = 0
    total_predictions_suppressed = 0
    total_guarded_records_affected = 0
    suppression_reasons: Dict[str, int] = OrderedDict()
    exemption_reasons: Dict[str, int] = OrderedDict()
    counter_metrics: Dict[str, float] = OrderedDict()
    records_report: List[Dict[str, Any]] = []

    for record_idx, (record, (result, controlled_indices)) in enumerate(
        zip(pred_records, results_by_record)
    ):
        inspected = len(controlled_indices)
        suppressed = len(result.suppressed_indices)
        total_predictions_inspected += inspected
        total_predictions_suppressed += suppressed
        if suppressed > 0:
            total_guarded_records_affected += 1
            suppression_reasons[_SUPPRESSION_REASON_DUPLICATE_CLUSTER] = (
                suppression_reasons.get(_SUPPRESSION_REASON_DUPLICATE_CLUSTER, 0)
                + suppressed
            )

        for cluster in result.clusters:
            for reason in cluster.exemption_reasons:
                exemption_reasons[str(reason)] = exemption_reasons.get(str(reason), 0) + 1

        for key, value in result.counter_metrics.items():
            counter_metrics[str(key)] = counter_metrics.get(str(key), 0.0) + float(value)

        record_report = {
            **_record_identity(record, record_idx),
            "inspected_predictions": int(inspected),
            "suppressed_predictions": int(suppressed),
            "kept_indices": [int(idx) for idx in result.kept_indices],
            "suppressed_indices": [int(idx) for idx in result.suppressed_indices],
            "exempt_indices": [int(idx) for idx in result.exempt_indices],
            "survivor_indices": [int(idx) for idx in result.survivor_indices],
            "support_counts": [int(value) for value in result.support_counts],
            "support_rates": [float(value) for value in result.support_rates],
            "suppression_reasons": (
                {_SUPPRESSION_REASON_DUPLICATE_CLUSTER: int(suppressed)}
                if suppressed > 0
                else {}
            ),
            "clusters": [
                {
                    "cluster_id": int(cluster.cluster_id),
                    "member_indices": [int(idx) for idx in cluster.member_indices],
                    "survivor_index": int(cluster.survivor_index),
                    "suppressed_indices": [int(idx) for idx in cluster.suppressed_indices],
                    "is_exempt": bool(cluster.is_exempt),
                    "exemption_reasons": [str(reason) for reason in cluster.exemption_reasons],
                    "max_center_distance": float(cluster.max_center_distance),
                }
                for cluster in result.clusters
            ],
            "decisions": [
                {
                    "object_index": int(decision.object_index),
                    "cluster_id": (
                        int(decision.cluster_id) if decision.cluster_id is not None else None
                    ),
                    "action": str(decision.action),
                    "survivor_index": int(decision.survivor_index),
                    "support_count": int(decision.support_count),
                    "support_rate": float(decision.support_rate),
                    "border_saturated": bool(decision.border_saturated),
                    "is_exempt": bool(decision.is_exempt),
                    "exemption_reasons": [
                        str(reason) for reason in decision.exemption_reasons
                    ],
                }
                for decision in result.decisions
            ],
        }
        records_report.append(record_report)

    return {
        "policy": "offline_conservative",
        "config": {
            "iou_threshold": float(_OFFLINE_DUPLICATE_CONTROL_CONFIG.iou_threshold),
            "center_radius_scale": float(
                _OFFLINE_DUPLICATE_CONTROL_CONFIG.center_radius_scale
            ),
            "support_iou_threshold": float(
                _OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD
            ),
        },
        "total_records": int(len(pred_records)),
        "total_predictions_inspected": int(total_predictions_inspected),
        "total_predictions_suppressed": int(total_predictions_suppressed),
        "total_guarded_records_affected": int(total_guarded_records_affected),
        "suppression_reasons": OrderedDict(
            (str(key), int(value))
            for key, value in sorted(suppression_reasons.items(), key=lambda item: item[0])
        ),
        "exemption_reasons": OrderedDict(
            (str(key), int(value))
            for key, value in sorted(exemption_reasons.items(), key=lambda item: item[0])
        ),
        "counter_metrics": OrderedDict(
            (str(key), float(value))
            for key, value in sorted(counter_metrics.items(), key=lambda item: item[0])
        ),
        "records": records_report,
    }


def _apply_offline_duplicate_control(
    pred_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    guarded_records: List[Dict[str, Any]] = []
    results_by_record: List[Tuple[DuplicateControlResult, set[int]]] = []
    config: DuplicateControlConfig = _OFFLINE_DUPLICATE_CONTROL_CONFIG

    for record in pred_records:
        guarded_record = copy.deepcopy(dict(record))
        raw_predictions = guarded_record.get("pred")
        pred_key = "pred"
        if not isinstance(raw_predictions, list):
            raw_predictions = guarded_record.get("predictions")
            pred_key = "predictions"
        if not isinstance(raw_predictions, list):
            guarded_records.append(guarded_record)
            empty_result = apply_duplicate_policy(
                anchor_objects=(),
                explorer_objects_by_view=(),
                config=config,
                support_iou_threshold=_OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD,
            )
            results_by_record.append((empty_result, set()))
            continue

        objects, controlled_indices = _duplicate_control_objects_for_record(guarded_record)
        if objects:
            result = apply_duplicate_policy(
                anchor_objects=objects,
                explorer_objects_by_view=(),
                config=config,
                support_iou_threshold=_OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD,
            )
            kept_indices = set(int(idx) for idx in result.kept_indices)
            guarded_record[pred_key] = [
                obj
                for pred_index, obj in enumerate(raw_predictions)
                if pred_index not in controlled_indices or pred_index in kept_indices
            ]
        else:
            result = apply_duplicate_policy(
                anchor_objects=(),
                explorer_objects_by_view=(),
                config=config,
                support_iou_threshold=_OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD,
            )
        results_by_record.append((result, controlled_indices))
        guarded_records.append(guarded_record)

    report = _build_duplicate_guard_report(
        pred_records=pred_records,
        results_by_record=results_by_record,
    )
    return guarded_records, report
