"""
Offline detection evaluator for CoordExp (pixel-space schema).

Features:
- Ingests standardized ``pred.jsonl`` (pixel-space ``gt`` / ``pred`` objects) or legacy GT JSONL.
- Converts geometries to COCO-format GT and prediction artifacts (bbox + segm for polygons).
- Runs COCOeval (bbox + segm) and/or a set-matching "F1-ish" metric and emits metrics plus robustness counters.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

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
    flatten_points,
    is_degenerate_bbox,
)
from src.common.geometry.object_geometry import extract_single_geometry
from src.common.prediction_parsing import GEOM_KEYS
from src.common.semantic_desc import SemanticDescEncoder, normalize_desc
from src.common.io import load_jsonl_with_diagnostics
from src.common.paths import resolve_image_path_best_effort
from src.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_desc(desc: str) -> str:
    return normalize_desc(desc)


def _fmt_iou_thr(iou_thr: float) -> str:
    return f"{float(iou_thr):.2f}"


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
    return _bbox_iou(cast(List[float], pred_obj["bbox"]), cast(List[float], gt_obj["bbox"]))


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


def _build_semantic_desc_mapping(
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    categories: Dict[str, int],
    *,
    options: "EvalOptions",
    counters: "EvalCounters",
) -> Dict[str, Tuple[Optional[str], float, int]]:
    """Return mapping: pred_desc -> (best_gt_desc|None, score, count)."""
    from collections import Counter

    unknown_counts: Counter[str] = Counter()
    for _, preds in pred_samples:
        for pred in preds:
            desc = (pred.get("desc") or "").strip()
            if not desc:
                continue
            if desc not in categories:
                unknown_counts[desc] += 1

    if not unknown_counts:
        return {}

    # Candidates are GT category strings (exclude synthetic 'unknown').
    candidate_names = [k for k in categories.keys() if k and k != "unknown"]
    if not candidate_names:
        return {}

    pred_names = list(unknown_counts.keys())
    pred_norm = [_normalize_desc(s) for s in pred_names]
    cand_norm = [_normalize_desc(s) for s in candidate_names]

    model_name = options.semantic_model or _DEFAULT_SEMANTIC_MODEL
    device = options.semantic_device or "auto"
    bs = max(1, int(options.semantic_batch_size))

    encoder = SemanticDescEncoder(model_name=str(model_name), device=str(device), batch_size=int(bs))

    try:
        pred_map = encoder.encode_norm_texts(pred_norm)
        cand_map = encoder.encode_norm_texts(cand_norm)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Description matching requires the semantic encoder "
            f"'{model_name}', but loading failed. Ensure the model exists "
            "in the local HuggingFace cache or that the runtime has network access. "
            "The evaluator no longer supports bucket/drop fallbacks."
        ) from exc

    pred_vecs: List[np.ndarray] = []
    for t in pred_norm:
        v = pred_map.get(t)
        if v is None:
            return {}
        pred_vecs.append(v)

    cand_vecs: List[np.ndarray] = []
    for t in cand_norm:
        v = cand_map.get(t)
        if v is None:
            return {}
        cand_vecs.append(v)

    if not pred_vecs or not cand_vecs:
        return {}

    pred_emb = np.stack(pred_vecs, axis=0)
    cand_emb = np.stack(cand_vecs, axis=0)

    # Cosine similarity via dot product (embeddings already normalized).
    sim = pred_emb @ cand_emb.T  # [P, C]
    best_idx = cast("np.ndarray", np.argmax(sim, axis=1))
    best_score = cast("np.ndarray", np.max(sim, axis=1))

    mapping: Dict[str, Tuple[Optional[str], float, int]] = {}
    for i, pred_desc in enumerate(pred_names):
        j = int(best_idx[i])
        score = float(best_score[i])
        best_name = candidate_names[j] if 0 <= j < len(candidate_names) else None
        mapping[pred_desc] = (best_name, score, int(unknown_counts[pred_desc]))

    # Write a small report for inspection.
    try:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "semantic_model": model_name,
            "semantic_threshold": float(options.semantic_threshold),
            "semantic_behavior": "map-or-drop",
            "unique_unknown_desc": len(pred_names),
            "unknown_total_preds": int(sum(unknown_counts.values())),
            "rows": [
                {
                    "pred_desc": d,
                    "count": c,
                    "best_gt_desc": best,
                    "score": s,
                    "mapped": bool(best is not None and s >= float(options.semantic_threshold)),
                }
                for d, (best, s, c) in sorted(
                    mapping.items(),
                    key=lambda kv: (-(kv[1][2]), -(kv[1][1])),
                )[: min(200, len(mapping))]
            ],
        }
        (options.output_dir / "semantic_desc_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        counters.semantic_report_failed += 1
        logger.warning("Failed to write semantic report: %s", exc)

    return mapping



def load_jsonl(
    path: Path,
    counters: EvalCounters | None = None,
    *,
    strict: bool = False,
    max_snippet_len: int = 200,
) -> List[Dict[str, Any]]:
    try:
        records, invalid_seen = load_jsonl_with_diagnostics(
            path,
            strict=bool(strict),
            max_snippet_len=int(max_snippet_len),
            warn_limit=5,
        )
    except ValueError:
        # In strict mode, fail fast on the first invalid record but keep counters
        # consistent with the legacy loader.
        if counters is not None:
            counters.invalid_json += 1
        raise

    if counters is not None:
        counters.invalid_json += int(invalid_seen)

    return records


def preds_to_gt_records(pred_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build minimal GT records from prediction lines that contain inline 'gt'."""

    gt_records: List[Dict[str, Any]] = []
    for rec in pred_records:
        raw_gt = rec.get("gt") or rec.get("objects") or []
        if not isinstance(raw_gt, list) or not raw_gt:
            continue
        gt_objs: List[Dict[str, Any]] = []
        for obj in raw_gt:
            if not isinstance(obj, dict):
                continue
            # If already keyed geometry, keep as-is
            gkeys = [g for g in GEOM_KEYS if g in obj and obj[g] is not None]
            if gkeys:
                gt_objs.append(obj)
                continue
            gtype = obj.get("type")
            pts = obj.get("points")
            if gtype in GEOM_KEYS and isinstance(pts, (list, tuple)):
                gt_objs.append({gtype: pts, "desc": obj.get("desc", "")})
        if not gt_objs:
            continue
        width = rec.get("width")
        height = rec.get("height")
        image = None
        if isinstance(rec.get("image"), str):
            image = rec["image"]
        elif isinstance(rec.get("images"), list) and rec["images"]:
            image = rec["images"][0]
        gt_records.append(
            {
                "images": [image] if image else [],
                "width": width,
                "height": height,
                "objects": gt_objs,
            }
        )
    return gt_records


@dataclass
class EvalCounters:
    invalid_json: int = 0
    invalid_geometry: int = 0
    invalid_coord: int = 0
    missing_size: int = 0
    size_mismatch: int = 0
    multi_image_ignored: int = 0
    degenerate: int = 0
    empty_pred: int = 0
    unknown_desc: int = 0
    unknown_dropped: int = 0
    semantic_mapped: int = 0
    semantic_unmapped: int = 0
    semantic_report_failed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


@dataclass
class EvalOptions:
    metrics: str = "f1ish"  # coco | f1ish | both
    strict_parse: bool = True
    use_segm: bool = True
    iou_types: Tuple[str, ...] = ("bbox", "segm")
    iou_thrs: Optional[List[float]] = None  # None -> COCO defaults
    f1ish_iou_thrs: List[float] = field(default_factory=lambda: [0.3, 0.5])
    # F1-ish prediction scope:
    # - "annotated": ignore predictions whose desc is not semantically close to any GT desc in the image
    # - "all": count all predictions (strict; penalizes "extra" open-vocab objects as FP)
    f1ish_pred_scope: str = "annotated"  # annotated | all
    output_dir: Path = Path("eval_out")
    overlay: bool = False
    overlay_k: int = 12
    open_vocab_recall: bool = False  # class-agnostic recall
    num_workers: int = 0  # parallelize pred parsing/denorm on CPU
    semantic_model: str = _DEFAULT_SEMANTIC_MODEL  # forced semantic matcher (unmatched descs are dropped)
    semantic_threshold: float = 0.6
    semantic_device: str = "auto"
    semantic_batch_size: int = 64

    def __post_init__(self) -> None:
        semantic_model = str(self.semantic_model or "").strip()
        if not semantic_model:
            raise ValueError(
                "semantic_model must be a non-empty HuggingFace model id. "
                "Semantic matching is mandatory; empty values are unsupported."
            )
        self.semantic_model = semantic_model


@dataclass
class Sample:
    image_id: int
    file_name: str
    width: int
    height: int
    objects: List[Dict[str, Any]] = field(default_factory=list)
    invalid: List[Dict[str, Any]] = field(default_factory=list)


def _prepare_gt_record(
    record: Dict[str, Any],
    idx: int,
    counters: EvalCounters,
    *,
    strict: bool,
) -> Optional[Sample]:
    images = record.get("images") or []
    if len(images) != 1:
        if images:
            counters.multi_image_ignored += 1
        elif strict:
            return None
    if not record.get("width") or not record.get("height"):
        counters.missing_size += 1
        return None
    width = int(record["width"])
    height = int(record["height"])
    file_name = images[0] if images else f"image_{idx}.jpg"
    objects: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    objs_in = record.get("gt") or record.get("objects") or []
    coord_mode_hint = record.get("coord_mode")
    for obj in objs_in:
        try:
            gtype, pts_raw = extract_single_geometry(
                obj,
                allow_type_and_points=True,
                allow_nested_points=False,
                path="gt",
            )
        except ValueError as exc:
            msg = str(exc)
            counters.invalid_geometry += 1
            if "type must be bbox_2d|poly" in msg:
                reason = "geometry_kind"
            elif "must contain exactly one geometry field" in msg:
                reason = "geometry_keys"
            else:
                reason = "geometry_points"
            invalid.append({"reason": reason, "raw": obj})
            continue

        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            counters.invalid_coord += 1
            invalid.append({"reason": "coord_parse", "raw": obj})
            continue
        coord_mode = "norm1000" if (had_tokens or coord_mode_hint == "norm1000") else "pixel"
        pts_px = denorm_and_clamp(points, width, height, coord_mode=coord_mode)
        x1, y1, x2, y2 = bbox_from_points(pts_px)
        if is_degenerate_bbox(x1, y1, x2, y2):
            counters.degenerate += 1
            invalid.append({"reason": "degenerate", "raw": obj})
            continue
        objects.append(
            {
                "type": gtype,
                "points": pts_px,
                "desc": obj.get("desc", ""),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return Sample(
        image_id=idx,
        file_name=file_name,
        width=width,
        height=height,
        objects=objects,
        invalid=invalid,
    )


def _prepare_pred_objects(
    record: Dict[str, Any],
    *,
    width: int,
    height: int,
    options: EvalOptions,
    counters: EvalCounters,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objs_raw: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    # Size mismatch tracking
    pred_w_raw = record.get("width")
    pred_h_raw = record.get("height")
    pred_w = None
    pred_h = None
    if pred_w_raw is not None:
        try:
            pred_w = int(pred_w_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prediction width must be int-compatible, got {pred_w_raw!r}."
            ) from exc
    if pred_h_raw is not None:
        try:
            pred_h = int(pred_h_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prediction height must be int-compatible, got {pred_h_raw!r}."
            ) from exc
    if pred_w and pred_w != width:
        counters.size_mismatch += 1
    if pred_h and pred_h != height:
        counters.size_mismatch += 1

    if isinstance(record.get("pred"), list):
        objs_raw = record["pred"]
    elif isinstance(record.get("predictions"), list):
        objs_raw = record["predictions"]

    if not objs_raw:
        counters.empty_pred += 1
        return [], invalid

    preds: List[Dict[str, Any]] = []
    coord_mode_hint = record.get("coord_mode")
    for obj in objs_raw:
        try:
            gtype, pts_raw = extract_single_geometry(
                obj,
                allow_type_and_points=True,
                allow_nested_points=False,
                path="pred",
            )
        except ValueError as exc:
            msg = str(exc)
            counters.invalid_geometry += 1
            if "type must be bbox_2d|poly" in msg:
                reason = "geometry_kind"
            elif "must contain exactly one geometry field" in msg:
                reason = "geometry_keys"
            else:
                reason = "geometry_points"
            invalid.append({"reason": reason, "raw": obj})
            continue

        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            counters.invalid_coord += 1
            invalid.append({"reason": "coord_parse", "raw": obj})
            continue

        # Pixel-ready by default; allow norm1000 if tokens or hint present.
        coord_mode = (
            "norm1000" if (had_tokens or coord_mode_hint == "norm1000") else "pixel"
        )
        pts_px = denorm_and_clamp(points, width, height, coord_mode=coord_mode)

        if gtype == "poly":
            if len(pts_px) < 6:
                counters.invalid_geometry += 1
                invalid.append({"reason": "poly_points", "raw": obj})
                continue
            x1, y1, x2, y2 = bbox_from_points(pts_px)
            gtype_export = "poly"
            segm = [pts_px]
        elif gtype == "bbox_2d":
            if len(pts_px) != 4:
                counters.invalid_geometry += 1
                invalid.append({"reason": "bbox_points", "raw": obj})
                continue
            x1, y1, x2, y2 = pts_px
            gtype_export = "bbox_2d"
            segm = None
        else:
            counters.invalid_geometry += 1
            invalid.append({"reason": "geometry_kind", "raw": obj})
            continue

        if is_degenerate_bbox(x1, y1, x2, y2):
            counters.degenerate += 1
            invalid.append({"reason": "degenerate", "raw": obj})
            continue

        desc = str(obj.get("desc", "")).strip()
        preds.append(
            {
                "type": gtype_export,
                "points": pts_px,
                "bbox": [x1, y1, x2, y2],
                "segmentation": segm,
                "desc": desc,
                "score": 1.0,  # greedy decoding; confidence not available
            }
        )
    return preds, invalid


def _prepare_pred_objects_detached(
    rec_and_size: Tuple[int, Dict[str, Any], int, int],
    options: EvalOptions,
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    image_id, record, width, height = rec_and_size
    local_counts = EvalCounters()
    preds, invalid = _prepare_pred_objects(
        record, width=width, height=height, options=options, counters=local_counts
    )
    return image_id, preds, invalid, local_counts.to_dict()


def _build_categories(gt_samples: List[Sample]) -> Dict[str, int]:
    cats: Dict[str, int] = {}
    next_id = 1
    for sample in gt_samples:
        for obj in sample.objects:
            desc = (obj.get("desc") or "").strip()
            if desc not in cats:
                cats[desc] = next_id
                next_id += 1
    return cats


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
        for pred in preds:
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
            res = {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": float(pred.get("score", 1.0)),
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
    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}

    if not results:
        return metrics, per_class

    coco_dt = coco_gt.loadRes(copy.deepcopy(results))
    iou_types = ["bbox"]
    if run_segm:
        iou_types.append("segm")

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


def _resolve_image_path(base_dir: Path, image_rel: str | None) -> Path:
    if image_rel is None:
        return base_dir / "missing.jpg"

    return resolve_image_path_best_effort(
        image_rel,
        jsonl_dir=None,
        root_image_dir=base_dir,
        env_root_var=None,
    )


def _draw_overlays(
    gt_samples: List[Sample],
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    *,
    base_dir: Path,
    out_dir: Path,
    k: int,
) -> None:
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from PIL import Image
    except (ImportError, OSError) as exc:
        logger.warning("Overlay rendering skipped (missing matplotlib/PIL): %s", exc)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_lookup = {img_id: preds for img_id, preds in pred_samples}
    for sample in gt_samples[: max(0, k)]:
        img_path = _resolve_image_path(base_dir, sample.file_name)
        if not img_path.exists():
            logger.warning("Overlay skipped: image not found %s", img_path)
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, ValueError) as exc:
            logger.warning("Overlay skipped: failed to load %s (%s)", img_path, exc)
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis("off")

        def _add_box(x1, y1, x2, y2, color, linestyle="-"):
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                linestyle=linestyle,
            )
            ax.add_patch(rect)

        colors = {"gt": "#2ca02c", "pred": "#d62728"}
        for obj in sample.objects:
            x1, y1, x2, y2 = obj["bbox"]
            _add_box(x1, y1, x2, y2, colors["gt"], "-")
            ax.text(
                x1,
                y1 - 2,
                obj.get("desc", ""),
                color=colors["gt"],
                fontsize=8,
                backgroundcolor="white",
            )

        for obj in pred_lookup.get(sample.image_id, []):
            x1, y1, x2, y2 = obj["bbox"]
            _add_box(x1, y1, x2, y2, colors["pred"], "--")
            ax.text(
                x1,
                y2 + 2,
                obj.get("desc", ""),
                color=colors["pred"],
                fontsize=8,
                backgroundcolor="white",
            )

        save_path = out_dir / f"overlay_{sample.image_id:04d}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def _prepare_all_from_records(
    gt_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    gt_samples: List[Sample] = []
    for idx, rec in enumerate(
        tqdm(gt_records, desc="GT", unit="img", disable=len(gt_records) < 10)
    ):
        sample = _prepare_gt_record(rec, idx, counters, strict=options.strict_parse)
        if sample:
            gt_samples.append(sample)
        elif options.strict_parse:
            raise ValueError(f"Failed to prepare GT for index {idx}")

    pred_map: Dict[int, Dict[str, Any]] = {}
    for i, rec in enumerate(pred_records):
        image_id = rec.get("index", i)
        pred_map[int(image_id)] = rec

    pred_samples: List[Tuple[int, List[Dict[str, Any]]]] = []
    invalid_preds: Dict[int, List[Dict[str, Any]]] = {}

    if options.num_workers and options.num_workers > 0:
        num_workers = min(options.num_workers, cpu_count())
        args_list = [
            (
                sample.image_id,
                pred_map.get(sample.image_id, {}),
                sample.width,
                sample.height,
            )
            for sample in gt_samples
        ]
        with Pool(processes=num_workers) as pool:
            for image_id, preds, invalid, local_counts in tqdm(
                pool.imap_unordered(
                    partial(_prepare_pred_objects_detached, options=options), args_list
                ),
                total=len(args_list),
                desc="Pred",
                unit="img",
                disable=len(args_list) < 10,
            ):
                pred_samples.append((image_id, preds))
                if invalid:
                    invalid_preds[image_id] = invalid
                for key, value in local_counts.items():
                    setattr(counters, key, getattr(counters, key) + value)
        pred_samples.sort(key=lambda x: x[0])
    else:
        for sample in tqdm(
            gt_samples, desc="Pred", unit="img", disable=len(gt_samples) < 10
        ):
            rec = pred_map.get(sample.image_id, {})
            preds, invalid = _prepare_pred_objects(
                rec,
                width=sample.width,
                height=sample.height,
                options=options,
                counters=counters,
            )
            pred_samples.append((sample.image_id, preds))
            if invalid:
                invalid_preds[sample.image_id] = invalid

    per_image = build_per_image_report(gt_samples, pred_samples, invalid_preds)

    if not prepare_coco:
        return (
            gt_samples,
            pred_samples,
            {},
            {},
            [],
            False,
            per_image,
        )

    categories = _build_categories(gt_samples)
    coco_gt_dict = _to_coco_gt(
        gt_samples, categories, add_box_segmentation=options.use_segm
    )
    results = _to_coco_preds(pred_samples, categories, options=options, counters=counters)
    run_segm = options.use_segm and any("segmentation" in r for r in results)
    return (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    )

def _prepare_all(
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    gt_records = preds_to_gt_records(pred_records)
    return _prepare_all_from_records(
        gt_records,
        pred_records,
        options,
        counters,
        prepare_coco=prepare_coco,
    )


def _prepare_all_separate(
    gt_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
    *,
    prepare_coco: bool,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    return _prepare_all_from_records(
        gt_records,
        pred_records,
        options,
        counters,
        prepare_coco=prepare_coco,
    )


def evaluate_detection(
    gt_path: Path,
    pred_path: Path | None = None,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    counters = EvalCounters()
    want_coco = str(options.metrics).lower() in {"coco", "both"}
    if pred_path is None:
        pred_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _,
        ) = _prepare_all(pred_records, options, counters, prepare_coco=want_coco)
    else:
        gt_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _,
        ) = _prepare_all_separate(gt_records, pred_records, options, counters, prepare_coco=want_coco)

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_coco:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()

        metrics, per_class = _run_coco_eval(
            coco_gt, results, options=options, run_segm=run_segm
        )

    summary = {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }
    return summary


def write_outputs(
    out_dir: Path,
    *,
    coco_gt: Dict[str, Any] | None,
    coco_preds: List[Dict[str, Any]] | None,
    summary: Dict[str, Any],
    per_image: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if coco_gt is not None:
        (out_dir / "coco_gt.json").write_text(
            json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8"
        )
    if coco_preds is not None:
        (out_dir / "coco_preds.json").write_text(
            json.dumps(coco_preds, ensure_ascii=False), encoding="utf-8"
        )
    metrics_payload = {
        "metrics": summary.get("metrics", {}),
        "counters": summary.get("counters", {}),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if coco_gt is not None and coco_preds is not None:
        (out_dir / "per_class.csv").write_text(
            "category,AP\n"
            + "\n".join(f"{k},{v}" for k, v in summary.get("per_class", {}).items()),
            encoding="utf-8",
        )
    (out_dir / "per_image.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_per_image_report(
    gt_samples: List[Sample],
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    invalid_preds: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    pred_lookup = {img_id: preds for img_id, preds in pred_samples}
    for sample in gt_samples:
        preds = pred_lookup.get(sample.image_id, [])
        report.append(
            {
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "gt_count": len(sample.objects),
                "pred_count": len(preds),
                "invalid_gt": sample.invalid,
                "invalid_pred": invalid_preds.get(sample.image_id, []),
            }
        )
    return report


def evaluate_and_save(
    pred_path: Path,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    counters = EvalCounters()
    metrics_mode = str(options.metrics).lower()
    want_coco = metrics_mode in {"coco", "both"}
    want_f1ish = metrics_mode in {"f1ish", "both"}
    pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    ) = _prepare_all(pred_records, options, counters, prepare_coco=want_coco)

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}

    if want_coco:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()

        metrics, per_class = _run_coco_eval(
            coco_gt, results, options=options, run_segm=run_segm
        )

    summary = {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }

    if want_f1ish:
        f1ish_summary = evaluate_f1ish(
            gt_samples,
            pred_samples,
            per_image,
            options=options,
        )
        summary["metrics"].update(f1ish_summary["metrics"])

    write_outputs(
        options.output_dir,
        coco_gt=coco_gt_dict if want_coco else None,
        coco_preds=results if want_coco else None,
        summary=summary,
        per_image=per_image,
    )

    if options.overlay:
        from src.infer.pipeline import resolve_root_image_dir_for_jsonl

        root_dir, root_source = resolve_root_image_dir_for_jsonl(pred_path)
        base_dir = root_dir if root_dir is not None else pred_path.parent
        if root_dir is not None:
            logger.info(
                "Overlay image root resolved (source=%s): %s", root_source, root_dir
            )

        overlay_dir = options.output_dir / "overlays"
        _draw_overlays(
            gt_samples,
            pred_samples,
            base_dir=base_dir,
            out_dir=overlay_dir,
            k=options.overlay_k,
        )

    return summary


def _f1ish_filter_gt_objects(sample: Sample) -> List[Dict[str, Any]]:
    return list(sample.objects)


def _compute_prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    p = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0.0 else 1.0
    r = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0.0 else 1.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
    return float(p), float(r), float(f1)


def _select_primary_f1ish_iou_thr(iou_thrs: List[float]) -> float:
    thrs = [float(t) for t in (iou_thrs or [])]
    if not thrs:
        return 0.5
    if any(abs(t - 0.5) < 1e-9 for t in thrs):
        return 0.5
    return max(thrs)


def _try_build_semantic_embeddings(
    unique_norm_texts: List[str],
    *,
    options: EvalOptions,
) -> Dict[str, np.ndarray]:
    if not unique_norm_texts:
        return {}
    model_name = str(options.semantic_model or "").strip()
    if not model_name:
        raise ValueError(
            "semantic_model must be a non-empty HuggingFace model id for F1-ish evaluation"
        )
    device = options.semantic_device or "auto"
    bs = max(1, int(options.semantic_batch_size))

    encoder = SemanticDescEncoder(model_name=model_name, device=str(device), batch_size=int(bs))

    try:
        embs = encoder.encode_norm_texts(unique_norm_texts)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "F1-ish semantic filtering requires the semantic encoder "
            f"'{model_name}', but loading/encoding failed. Ensure the model is "
            "available in the local HuggingFace cache or that downloads are allowed."
        ) from exc

    out: Dict[str, np.ndarray] = {}
    for text in unique_norm_texts:
        v = embs.get(text)
        if v is None:
            continue
        out[text] = v
    return out


def evaluate_f1ish(
    gt_samples: List[Sample],
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]],
    per_image: List[Dict[str, Any]],
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    """Compute F1-ish (set matching) metrics and emit match diagnostics.

    Notes on prediction scope:
    - In ``f1ish_pred_scope=annotated`` (default), predictions whose desc is not semantically
      close to any GT desc in the image are ignored (not counted as FP).
    - In ``f1ish_pred_scope=all``, all predictions are counted (strict; extra open-vocab
      objects become FP).
    """
    iou_thrs_in = options.f1ish_iou_thrs or [0.3, 0.5]
    iou_thrs = sorted({float(t) for t in iou_thrs_in})
    primary_thr = _select_primary_f1ish_iou_thr(iou_thrs)

    pred_scope = str(options.f1ish_pred_scope or "annotated").strip().lower()
    if pred_scope not in {"annotated", "all"}:
        logger.warning("Unknown f1ish_pred_scope='%s'; defaulting to 'annotated'", pred_scope)
        pred_scope = "annotated"

    pred_lookup = {img_id: preds for img_id, preds in pred_samples}
    per_image_lookup = {row.get("image_id"): row for row in per_image}

    unique_norm_texts: set[str] = set()

    # Collect texts for semantic matching/filtering. This is also used by the
    # per-match semantic scoring below.
    for sample in gt_samples:
        gts = _f1ish_filter_gt_objects(sample)
        for gt in gts:
            gt_desc = _normalize_desc(str(gt.get("desc", "")))
            if gt_desc:
                unique_norm_texts.add(gt_desc)
        for pred in pred_lookup.get(sample.image_id, []):
            pred_desc = _normalize_desc(str(pred.get("desc", "")))
            if pred_desc:
                unique_norm_texts.add(pred_desc)

    # Build semantic embedding cache once (if available)
    norm_texts_sorted = sorted(unique_norm_texts)
    emb = _try_build_semantic_embeddings(norm_texts_sorted, options=options)
    sem_thr = float(options.semantic_threshold)
    use_embeddings = bool(emb)

    def _pred_in_gt_label_space(pred_desc: str, gt_descs: List[str]) -> bool:
        if pred_scope == "all":
            return True
        if not pred_desc or not gt_descs:
            return False
        if pred_desc in set(gt_descs):
            return True
        if not use_embeddings:
            return False
        pred_vec = emb.get(pred_desc)
        if pred_vec is None:
            return False
        best = -1.0
        for gt_desc in gt_descs:
            gt_vec = emb.get(gt_desc)
            if gt_vec is None:
                continue
            best = max(best, float(pred_vec @ gt_vec))
        return best >= sem_thr

    # Pre-filter predictions based on the requested scope. This is independent of IoU
    # threshold and can be reused across all thresholds.
    gts_by_image: Dict[int, List[Dict[str, Any]]] = {}
    preds_eval_by_image: Dict[int, List[Dict[str, Any]]] = {}
    preds_eval_orig_idx_by_image: Dict[int, List[int]] = {}
    preds_ignored_orig_idx_by_image: Dict[int, List[int]] = {}

    for sample in gt_samples:
        gts = _f1ish_filter_gt_objects(sample)
        gts_by_image[sample.image_id] = gts

        gt_descs_set: set[str] = set()
        for gt in gts:
            d = _normalize_desc(str(gt.get("desc", "")))
            if d:
                gt_descs_set.add(d)
        gt_descs = sorted(gt_descs_set)

        preds_total = pred_lookup.get(sample.image_id, [])
        preds_eval: List[Dict[str, Any]] = []
        preds_eval_orig_idx: List[int] = []
        preds_ignored_orig_idx: List[int] = []
        for idx, pred in enumerate(preds_total):
            pred_desc = _normalize_desc(str(pred.get("desc", "")))
            if _pred_in_gt_label_space(pred_desc, gt_descs):
                preds_eval.append(pred)
                preds_eval_orig_idx.append(int(idx))
            else:
                preds_ignored_orig_idx.append(int(idx))
        preds_eval_by_image[sample.image_id] = preds_eval
        preds_eval_orig_idx_by_image[sample.image_id] = preds_eval_orig_idx
        preds_ignored_orig_idx_by_image[sample.image_id] = preds_ignored_orig_idx

    # Location matching per threshold.
    matches_by_thr: Dict[str, Dict[int, List[Tuple[int, int, float]]]] = {}
    for thr in iou_thrs:
        thr_key = _fmt_iou_thr(thr)
        matches_by_thr[thr_key] = {}

    for sample in gt_samples:
        preds_eval = preds_eval_by_image.get(sample.image_id, [])
        gts = gts_by_image.get(sample.image_id, [])
        width = int(sample.width)
        height = int(sample.height)
        for thr in iou_thrs:
            thr_key = _fmt_iou_thr(thr)
            matches_by_thr[thr_key][sample.image_id] = _greedy_match_by_iou(
                preds_eval, gts, iou_thr=thr, width=width, height=height
            )

    # Accumulators
    metrics_out: Dict[str, float] = {}

    for thr in iou_thrs:
        thr_key = _fmt_iou_thr(thr)

        sum_tp = 0
        sum_fp = 0
        sum_fn = 0
        sum_sem_ok = 0
        sum_sem_bad = 0
        sum_pred_total = 0
        sum_pred_eval = 0
        sum_pred_ignored = 0

        sum_tp_full = 0
        sum_fp_full = 0
        sum_fn_full = 0

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        precisions_full: List[float] = []
        recalls_full: List[float] = []
        f1s_full: List[float] = []

        matches_records: List[Dict[str, Any]] = []

        for sample in gt_samples:
            preds_total = pred_lookup.get(sample.image_id, [])
            preds = preds_eval_by_image.get(sample.image_id, [])
            preds_eval_orig = preds_eval_orig_idx_by_image.get(sample.image_id, [])
            preds_ignored_orig = preds_ignored_orig_idx_by_image.get(sample.image_id, [])
            gts = gts_by_image.get(sample.image_id, [])
            matches = matches_by_thr[thr_key].get(sample.image_id, [])

            matched_pred = {pred_idx for pred_idx, _, _ in matches}  # pred_idx is in *eval* space
            matched_gt = {gt_idx for _, gt_idx, _ in matches}

            tp_loc = len(matches)
            fp_loc = max(0, len(preds) - len(matched_pred))
            fn_loc = max(0, len(gts) - len(matched_gt))

            sum_pred_total += len(preds_total)
            sum_pred_eval += len(preds)
            sum_pred_ignored += len(preds_ignored_orig)

            sem_ok = 0
            sem_bad = 0
            match_rows: List[Dict[str, Any]] = []
            for pred_idx, gt_idx, iou in matches:
                pred_desc_raw = str(preds[pred_idx].get("desc", ""))
                gt_desc_raw = str(gts[gt_idx].get("desc", ""))
                pred_desc = _normalize_desc(pred_desc_raw)
                gt_desc = _normalize_desc(gt_desc_raw)
                pred_idx_orig = (
                    int(preds_eval_orig[pred_idx])
                    if 0 <= int(pred_idx) < len(preds_eval_orig)
                    else int(pred_idx)
                )

                exact_ok = bool(pred_desc) and (pred_desc == gt_desc)
                sim: float | None = None
                if use_embeddings and pred_desc in emb and gt_desc in emb:
                    sim = float(emb[pred_desc] @ emb[gt_desc])
                ok = bool(exact_ok or (sim is not None and sim >= sem_thr))
                if ok:
                    sem_ok += 1
                else:
                    sem_bad += 1

                match_rows.append(
                    {
                        "pred_idx": int(pred_idx_orig),
                        "gt_idx": int(gt_idx),
                        "iou": float(iou),
                        "pred_desc": pred_desc_raw,
                        "gt_desc": gt_desc_raw,
                        "sem_sim": float(sim) if sim is not None else None,
                        "sem_ok": bool(ok),
                        "pred_bbox": preds[pred_idx].get("bbox"),
                        "gt_bbox": gts[gt_idx].get("bbox"),
                        "pred_type": preds[pred_idx].get("type"),
                        "gt_type": gts[gt_idx].get("type"),
                    }
                )

            sum_tp += tp_loc
            sum_fp += fp_loc
            sum_fn += fn_loc
            sum_sem_ok += sem_ok
            sum_sem_bad += sem_bad

            tp_full = sem_ok
            fp_full = fp_loc + sem_bad
            fn_full = fn_loc + sem_bad

            sum_tp_full += tp_full
            sum_fp_full += fp_full
            sum_fn_full += fn_full

            p_i, r_i, f1_i = _compute_prf_from_counts(tp_loc, fp_loc, fn_loc)
            precisions.append(p_i)
            recalls.append(r_i)
            f1s.append(f1_i)

            p_f, r_f, f1_f = _compute_prf_from_counts(tp_full, fp_full, fn_full)
            precisions_full.append(p_f)
            recalls_full.append(r_f)
            f1s_full.append(f1_f)

            image_row = per_image_lookup.get(sample.image_id)
            if image_row is not None:
                f1ish_field = image_row.setdefault("f1ish", {})
                f1ish_field[thr_key] = {
                    "tp_loc": int(tp_loc),
                    "fp_loc": int(fp_loc),
                    "fn_loc": int(fn_loc),
                    "pred_count_eval": int(len(preds)),
                    "pred_count_ignored": int(len(preds_ignored_orig)),
                    "matched_sem_ok": int(sem_ok),
                    "matched_sem_bad": int(sem_bad),
                    "sem_acc_on_matched": float(sem_ok / tp_loc) if tp_loc > 0 else 1.0,
                    "tp_full": int(tp_full),
                    "fp_full": int(fp_full),
                    "fn_full": int(fn_full),
                }

            is_primary = abs(float(thr) - float(primary_thr)) < 1e-9
            if is_primary or len(iou_thrs) > 1:
                unmatched_pred_indices = [
                    int(preds_eval_orig[i])
                    for i in range(len(preds))
                    if i not in matched_pred and i < len(preds_eval_orig)
                ]
                matches_records.append(
                    {
                        "image_id": int(sample.image_id),
                        "file_name": sample.file_name,
                        "width": int(sample.width),
                        "height": int(sample.height),
                        "iou_thr": float(thr),
                        "gt_count": int(len(gts)),
                        "pred_count": int(len(preds_total)),
                        "pred_count_eval": int(len(preds)),
                        "pred_count_ignored": int(len(preds_ignored_orig)),
                        "pred_scope": str(pred_scope),
                        "tp_loc": int(tp_loc),
                        "fp_loc": int(fp_loc),
                        "fn_loc": int(fn_loc),
                        "matches": match_rows,
                        "unmatched_pred_indices": unmatched_pred_indices,
                        "ignored_pred_indices": [int(i) for i in preds_ignored_orig],
                        "unmatched_gt_indices": [
                            int(i) for i in range(len(gts)) if i not in matched_gt
                        ],
                    }
                )

        p_micro, r_micro, f1_micro = _compute_prf_from_counts(sum_tp, sum_fp, sum_fn)
        p_macro = float(np.mean(precisions)) if precisions else 0.0
        r_macro = float(np.mean(recalls)) if recalls else 0.0
        f1_macro = float(np.mean(f1s)) if f1s else 0.0

        p_full_micro, r_full_micro, f1_full_micro = _compute_prf_from_counts(
            sum_tp_full, sum_fp_full, sum_fn_full
        )
        p_full_macro = float(np.mean(precisions_full)) if precisions_full else 0.0
        r_full_macro = float(np.mean(recalls_full)) if recalls_full else 0.0
        f1_full_macro = float(np.mean(f1s_full)) if f1s_full else 0.0

        prefix = f"f1ish@{thr_key}_"
        metrics_out.update(
            {
                f"{prefix}tp_loc": float(sum_tp),
                f"{prefix}fp_loc": float(sum_fp),
                f"{prefix}fn_loc": float(sum_fn),
                f"{prefix}pred_total": float(sum_pred_total),
                f"{prefix}pred_eval": float(sum_pred_eval),
                f"{prefix}pred_ignored": float(sum_pred_ignored),
                f"{prefix}precision_loc_micro": float(p_micro),
                f"{prefix}recall_loc_micro": float(r_micro),
                f"{prefix}f1_loc_micro": float(f1_micro),
                f"{prefix}precision_loc_macro": float(p_macro),
                f"{prefix}recall_loc_macro": float(r_macro),
                f"{prefix}f1_loc_macro": float(f1_macro),
                f"{prefix}matched_sem_ok": float(sum_sem_ok),
                f"{prefix}matched_sem_bad": float(sum_sem_bad),
                f"{prefix}sem_acc_on_matched": float(sum_sem_ok / sum_tp) if sum_tp > 0 else 1.0,
                f"{prefix}tp_full": float(sum_tp_full),
                f"{prefix}fp_full": float(sum_fp_full),
                f"{prefix}fn_full": float(sum_fn_full),
                f"{prefix}precision_full_micro": float(p_full_micro),
                f"{prefix}recall_full_micro": float(r_full_micro),
                f"{prefix}f1_full_micro": float(f1_full_micro),
                f"{prefix}precision_full_macro": float(p_full_macro),
                f"{prefix}recall_full_macro": float(r_full_macro),
                f"{prefix}f1_full_macro": float(f1_full_macro),
            }
        )

        # Emit matches artifacts
        out_dir = options.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        is_primary = abs(float(thr) - float(primary_thr)) < 1e-9
        if is_primary:
            matches_path = out_dir / "matches.jsonl"
        else:
            matches_path = out_dir / f"matches@{thr_key}.jsonl"
        if is_primary or len(iou_thrs) > 1:
            with matches_path.open("w", encoding="utf-8") as f:
                for row in matches_records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "metrics": metrics_out,
    }
