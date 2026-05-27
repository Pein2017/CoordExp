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

from src.eval.detection_geometry import _project_bbox_to_source_resolution
from src.eval.detection_records import (
    EvalCounters,
    EvalOptions,
    Sample,
    _build_semantic_desc_mapping,
    _prepare_pred_objects,
    load_jsonl,
)
from src.infer.artifacts import load_comparable_artifact

def _validate_score_provenance_for_coco(
    record: Dict[str, Any], record_idx: int
) -> None:
    source = record.get("pred_score_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "COCO evaluation requires scored artifacts with score provenance. "
            f"Record index {record_idx} is missing non-empty `pred_score_source`. "
            "Run confidence post-op first and evaluate `gt_vs_pred_scored.jsonl`."
        )

    version = record.get("pred_score_version")
    if not isinstance(version, int):
        raise ValueError(
            "COCO evaluation requires scored artifacts with score provenance. "
            f"Record index {record_idx} is missing integer `pred_score_version`. "
            "Run confidence post-op first and evaluate `gt_vs_pred_scored.jsonl`."
        )


def _parse_coco_score(
    *,
    score_value: Any,
    record_idx: int,
    object_idx: int,
) -> float:
    if score_value is None:
        raise ValueError(
            "COCO score contract violation: missing `pred[*].score` at "
            f"record index {record_idx}, object index {object_idx}."
        )

    try:
        score = float(score_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "COCO score contract violation: non-numeric `pred[*].score` at "
            f"record index {record_idx}, object index {object_idx}: {score_value!r}"
        ) from exc

    if not math.isfinite(score):
        raise ValueError(
            "COCO score contract violation: non-finite `pred[*].score` at "
            f"record index {record_idx}, object index {object_idx}: {score_value!r}"
        )
    if score < 0.0 or score > 1.0:
        raise ValueError(
            "COCO score contract violation: out-of-range `pred[*].score` at "
            f"record index {record_idx}, object index {object_idx}: {score!r} "
            "(expected 0.0 <= score <= 1.0)."
        )
    return score


def _load_coco_categories_json(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("categories_json must be a list of {id, name} objects")

    categories: Dict[str, int] = {}
    ids_to_names: Dict[int, str] = {}
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"categories_json entry {idx} must be an object, got {type(item).__name__}"
            )

        raw_id = item.get("id")
        raw_name = item.get("name")
        try:
            category_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"categories_json entry {idx} has invalid `id`: {raw_id!r}"
            ) from exc

        category_name = str(raw_name or "").strip()
        if not category_name:
            raise ValueError(
                f"categories_json entry {idx} has empty `name`: {raw_name!r}"
            )

        prev_name = ids_to_names.get(category_id)
        if prev_name is not None and prev_name != category_name:
            raise ValueError(
                "categories_json contains conflicting entries for "
                f"category id {category_id}: {prev_name!r} vs {category_name!r}"
            )

        prev_id = categories.get(category_name)
        if prev_id is not None and prev_id != category_id:
            raise ValueError(
                "categories_json contains conflicting entries for "
                f"category name {category_name!r}: {prev_id} vs {category_id}"
            )

        categories[category_name] = category_id
        ids_to_names[category_id] = category_name

    if not categories:
        raise ValueError("categories_json must contain at least one category")
    return categories


def export_coco_submission(
    pred_path: Path,
    *,
    source_jsonl: Path,
    categories_json: Path,
    out_json: Path,
    options: EvalOptions,
) -> Dict[str, Any]:
    load_comparable_artifact(pred_path, require_score=True)
    counters = EvalCounters()
    pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
    source_records = load_jsonl(source_jsonl, strict=True)

    if len(source_records) != len(pred_records):
        raise ValueError(
            "Official COCO submission export requires source and prediction artifacts "
            "to have the same number of records. "
            f"source={len(source_records)} pred={len(pred_records)} "
            f"(source_jsonl={source_jsonl}, pred_jsonl={pred_path})"
        )

    categories = _load_coco_categories_json(categories_json)
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]] = []

    for record_idx, (source_record, pred_record) in enumerate(
        zip(source_records, pred_records)
    ):
        if not isinstance(source_record, dict):
            raise ValueError(
                f"Source JSONL record {record_idx} must be a JSON object, got "
                f"{type(source_record).__name__}"
            )
        if not isinstance(pred_record, dict):
            raise ValueError(
                f"Prediction JSONL record {record_idx} must be a JSON object, got "
                f"{type(pred_record).__name__}"
            )

        _validate_score_provenance_for_coco(pred_record, record_idx)

        raw_image_id = source_record.get("image_id")
        try:
            image_id = int(raw_image_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Official COCO submission export requires `image_id` in the source JSONL. "
                f"Source record index {record_idx} has invalid image_id={raw_image_id!r}."
            ) from exc

        source_images = source_record.get("images")
        if (
            not isinstance(source_images, list)
            or len(source_images) != 1
            or not isinstance(source_images[0], str)
            or not source_images[0].strip()
        ):
            raise ValueError(
                "Official COCO submission export requires source JSONL records to contain "
                f"exactly one image path. Source record index {record_idx} has "
                f"images={source_images!r}."
            )
        source_image = source_images[0]

        width_raw = source_record.get("width")
        height_raw = source_record.get("height")
        try:
            source_width = int(width_raw)
            source_height = int(height_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Official COCO submission export requires source JSONL width/height. "
                f"Source record index {record_idx} has width={width_raw!r} "
                f"height={height_raw!r}."
            ) from exc
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                "Official COCO submission export requires positive source JSONL width/height. "
                f"Source record index {record_idx} has width={source_width} height={source_height}."
            )

        pred_image = pred_record.get("image")
        if pred_image is None and isinstance(pred_record.get("images"), list):
            pred_images = pred_record["images"]
            if pred_images:
                pred_image = pred_images[0]
        if pred_image is not None and pred_image != source_image:
            raise ValueError(
                "Source/prediction record alignment mismatch while exporting COCO submission. "
                f"Record index {record_idx} has source image {source_image!r} but "
                f"prediction image {pred_image!r}."
            )

        pred_width_raw = pred_record.get("width")
        pred_height_raw = pred_record.get("height")
        try:
            pred_width = int(pred_width_raw)
            pred_height = int(pred_height_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Official COCO submission export requires prediction width/height. "
                f"Prediction record index {record_idx} has width={pred_width_raw!r} "
                f"height={pred_height_raw!r}."
            ) from exc
        if pred_width <= 0 or pred_height <= 0:
            raise ValueError(
                "Official COCO submission export requires positive prediction width/height. "
                f"Prediction record index {record_idx} has width={pred_width} height={pred_height}."
            )

        preds, _invalid = _prepare_pred_objects(
            pred_record,
            width=pred_width,
            height=pred_height,
            options=options,
            counters=counters,
        )
        if pred_width != source_width or pred_height != source_height:
            for pred in preds:
                pred["bbox"] = _project_bbox_to_source_resolution(
                    pred["bbox"],
                    pred_width=pred_width,
                    pred_height=pred_height,
                    source_width=source_width,
                    source_height=source_height,
                )
        pred_samples.append((image_id, preds))

    results = _to_coco_preds(
        pred_samples, categories, options=options, counters=counters
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    summary = {
        "pred_jsonl": str(pred_path),
        "source_jsonl": str(source_jsonl),
        "categories_json": str(categories_json),
        "submission_json": str(out_json),
        "images_total": len(source_records),
        "predictions_total": len(results),
        "categories_total": len(categories),
        "semantic_model": options.semantic_model,
        "semantic_threshold": float(options.semantic_threshold),
        "counters": counters.to_dict(),
    }
    (out_json.parent / "submission_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _to_coco_gt(
    gt_samples: List[Sample],
    categories: Dict[str, int],
    *,
    add_box_segmentation: bool = False,
) -> Dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1
    for sample in gt_samples:
        images.append(
            {
                "id": sample.image_id,
                "file_name": sample.file_name,
                "width": sample.width,
                "height": sample.height,
            }
        )
        for obj in sample.objects:
            cat = (obj.get("desc") or "").strip()
            cat_id = categories.get(cat)
            if cat_id is None:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            w = x2 - x1
            h = y2 - y1
            ann = {
                "id": ann_id,
                "image_id": sample.image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "area": max(w, 0.0) * max(h, 0.0),
                "iscrowd": 0,
            }
            if obj.get("type") == "poly":
                ann["segmentation"] = [obj["points"]]
            elif add_box_segmentation:
                ann["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            annotations.append(ann)
            ann_id += 1
    categories_list = [
        {"id": cid, "name": name}
        for name, cid in sorted(categories.items(), key=lambda kv: kv[1])
    ]
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }


def _to_coco_preds(
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    categories: Dict[str, int],
    *,
    options: EvalOptions,
    counters: EvalCounters,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    semantic_map = _build_semantic_desc_mapping(
        pred_samples, categories, options=options, counters=counters
    )
    sem_thr = float(options.semantic_threshold)

    for image_id, preds in pred_samples:
        for object_idx, pred in enumerate(preds):
            desc = (pred.get("desc") or "").strip()
            cat_id = categories.get(desc)
            if cat_id is None:
                best_name, score, _ = semantic_map.get(desc, (None, 0.0, 0))
                if best_name is not None and score >= sem_thr:
                    candidate_id = categories.get(best_name)
                    if candidate_id is not None:
                        cat_id = candidate_id
                        counters.semantic_mapped += 1
                if cat_id is None:
                    counters.semantic_unmapped += 1
                    counters.unknown_dropped += 1
                    continue
            x1, y1, x2, y2 = pred["bbox"]
            w = x2 - x1
            h = y2 - y1
            score = _parse_coco_score(
                score_value=pred.get("score"),
                record_idx=int(image_id),
                object_idx=int(object_idx),
            )
            res = {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": score,
            }
            if pred.get("segmentation") and options.use_segm:
                res["segmentation"] = pred["segmentation"]
            results.append(res)
    return results


def _run_coco_eval(
    coco_gt: COCO,
    results: List[Dict[str, Any]],
    *,
    options: EvalOptions,
    run_segm: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    metric_suffixes = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    )
    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    iou_types = ["bbox"]
    if run_segm:
        iou_types.append("segm")

    if not results:
        for iou_type in iou_types:
            for suffix in metric_suffixes:
                metrics[f"{iou_type}_{suffix}"] = 0.0
        return metrics, per_class

    coco_dt = coco_gt.loadRes(copy.deepcopy(results))

    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        if options.iou_thrs:
            coco_eval.params.iouThrs = np.array(options.iou_thrs)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # TODO: add polygon GIoU metric alongside COCOeval outputs.
        prefix = f"{iou_type}_"
        metrics.update(
            {
                f"{prefix}AP": float(coco_eval.stats[0]),
                f"{prefix}AP50": float(coco_eval.stats[1]),
                f"{prefix}AP75": float(coco_eval.stats[2]),
                f"{prefix}APs": float(coco_eval.stats[3]),
                f"{prefix}APm": float(coco_eval.stats[4]),
                f"{prefix}APl": float(coco_eval.stats[5]),
                f"{prefix}AR1": float(coco_eval.stats[6]),
                f"{prefix}AR10": float(coco_eval.stats[7]),
                f"{prefix}AR100": float(coco_eval.stats[8]),
                f"{prefix}ARs": float(coco_eval.stats[9]),
                f"{prefix}ARm": float(coco_eval.stats[10]),
                f"{prefix}ARl": float(coco_eval.stats[11]),
            }
        )
        # per-class AP (bbox only to avoid duplication)
        if iou_type == "bbox" and coco_eval.eval is not None:
            precisions = coco_eval.eval["precision"]  # shape [TxRxKxAxM]
            cat_ids = coco_gt.getCatIds()
            for idx, cat_id in enumerate(cat_ids):
                # average over IoU thresholds and area/all max dets
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = float(np.mean(precision)) if precision.size else float("nan")
                cat_name = coco_gt.loadCats(cat_id)[0]["name"]
                per_class[cat_name] = ap
    return metrics, per_class
