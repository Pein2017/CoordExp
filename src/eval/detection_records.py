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
from src.eval.detection_eval_records import DetectionEvalRecord, ScoredDetectionEvalRecord
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

_DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_desc(desc: str) -> str:
    return normalize_desc(desc)


def _wants_official_metrics(metrics: str) -> bool:
    return str(metrics).strip().lower() in {"coco", "lvis", "both"}


def _resolve_semantic_desc_encoder() -> Any:
    import sys

    facade = sys.modules.get("src.eval.detection")
    if facade is not None:
        encoder_cls = getattr(facade, "SemanticDescEncoder", None)
        if encoder_cls is not None:
            return encoder_cls
    return SemanticDescEncoder


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

    encoder_cls = _resolve_semantic_desc_encoder()
    encoder = encoder_cls(
        model_name=str(model_name), device=str(device), batch_size=int(bs)
    )

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
                    "mapped": bool(
                        best is not None and s >= float(options.semantic_threshold)
                    ),
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

    return [
        record.to_json_record()
        for record in (DetectionEvalRecord.from_json_record(rec) for rec in records)
    ]


def preds_to_gt_records(pred_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build minimal GT records from prediction lines that contain inline 'gt'."""

    gt_records: List[Dict[str, Any]] = []
    for eval_record in (
        DetectionEvalRecord.from_json_record(rec) for rec in pred_records
    ):
        rec = eval_record.to_json_record()
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
                "image_id": rec.get("image_id"),
                "metadata": (
                    dict(rec["metadata"])
                    if isinstance(rec.get("metadata"), Mapping)
                    else {}
                ),
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
    lvis_matched_verified_positive: int = 0
    lvis_verified_negative_unmatched: int = 0
    lvis_ignored_not_exhaustive: int = 0
    lvis_ignored_unevaluable: int = 0

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


@dataclass
class EvalOptions:
    metrics: str = "f1ish"  # coco | lvis | f1ish | both
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
    semantic_model: str = (
        _DEFAULT_SEMANTIC_MODEL  # forced semantic matcher (unmatched descs are dropped)
    )
    semantic_threshold: float = 0.6
    semantic_device: str = "auto"
    semantic_batch_size: int = 64
    lvis_max_dets: int = 300
    duplicate_control_enabled: bool = False
    guarded_pred_path: Optional[Path] = None
    duplicate_guard_report_path: Optional[Path] = None
    artifact_name_suffix: str = ""

    def __post_init__(self) -> None:
        semantic_model = str(self.semantic_model or "").strip()
        if not semantic_model:
            raise ValueError(
                "semantic_model must be a non-empty HuggingFace model id. "
                "Semantic matching is mandatory; empty values are unsupported."
            )
        self.semantic_model = semantic_model
        artifact_name_suffix = str(self.artifact_name_suffix or "").strip()
        if artifact_name_suffix and not artifact_name_suffix.startswith("_"):
            artifact_name_suffix = f"_{artifact_name_suffix}"
        self.artifact_name_suffix = artifact_name_suffix


@dataclass
class Sample:
    image_id: int
    file_name: str
    width: int
    height: int
    objects: List[Dict[str, Any]] = field(default_factory=list)
    invalid: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    lvis_policy = extract_lvis_image_policy(metadata)
    objects: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    objs_in = record.get("gt") or record.get("objects") or []
    coord_mode_hint = record.get("coord_mode")
    for obj_idx, obj in enumerate(objs_in):
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
        coord_mode = (
            "norm1000" if (had_tokens or coord_mode_hint == "norm1000") else "pixel"
        )
        pts_px = denorm_and_clamp(points, width, height, coord_mode=coord_mode)
        x1, y1, x2, y2 = bbox_from_points(pts_px)
        if is_degenerate_bbox(x1, y1, x2, y2):
            counters.degenerate += 1
            invalid.append({"reason": "degenerate", "raw": obj})
            continue
        prepared_obj = {
            "type": gtype,
            "points": pts_px,
            "desc": obj.get("desc", ""),
            "bbox": [x1, y1, x2, y2],
        }
        if lvis_policy is not None and int(obj_idx) < len(lvis_policy.gt_objects):
            gt_cat = lvis_policy.gt_objects[int(obj_idx)]
            prepared_obj["category_id"] = int(gt_cat.category_id)
            prepared_obj["category_frequency"] = str(gt_cat.frequency)
        objects.append(prepared_obj)

    return Sample(
        image_id=idx,
        file_name=file_name,
        width=width,
        height=height,
        objects=objects,
        invalid=invalid,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
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
            }
        )
        if "score" in obj:
            preds[-1]["score"] = obj.get("score")
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
