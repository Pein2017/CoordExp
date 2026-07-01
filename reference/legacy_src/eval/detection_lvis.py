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

from src.eval.detection_coco import _to_coco_preds
from src.eval.detection_geometry import _greedy_match_by_iou
from src.eval.detection_records import EvalCounters, EvalOptions, Sample, _normalize_desc

def _normalize_lvis_frequency_label(value: Any) -> str:
    freq = str(value or "").strip().lower()
    if freq in {"r", "rare"}:
        return "r"
    if freq in {"c", "common"}:
        return "c"
    if freq in {"f", "frequent"}:
        return "f"
    return "unknown"


def _matches_by_record_idx(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for fallback_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        image_id = row.get("image_id")
        try:
            record_idx = int(image_id)
        except (TypeError, ValueError):
            record_idx = int(fallback_idx)
        out[int(record_idx)] = dict(row)
    return out


def _image_key_variants(image_value: str) -> List[str]:
    pure = PurePosixPath(str(image_value).replace("\\", "/"))
    parts = [part for part in pure.parts if part not in {"", "."}]
    variants: List[str] = []
    if len(parts) >= 2:
        variants.append("/".join(parts[-2:]))
    if parts:
        variants.append(parts[-1])
    if not variants:
        variants.append(str(image_value))
    out: List[str] = []
    for item in variants:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _infer_lvis_split(gt_jsonl: Path) -> str:
    text = str(gt_jsonl).lower()
    name = gt_jsonl.name.lower()
    if "val" in name or "/val" in text:
        return "val"
    return "train"


def _default_lvis_annotations_json(gt_jsonl: Path) -> Path:
    split = _infer_lvis_split(gt_jsonl)
    return Path("public_data/lvis/raw/annotations") / f"lvis_v1_{split}.json"


class _LvisLegacyMetadataIndex:
    def __init__(self, annotations_json: Path) -> None:
        if not annotations_json.is_file():
            raise FileNotFoundError(
                "LVIS eval metadata backfill requires the raw annotations JSON at "
                f"{annotations_json}"
            )
        payload = json.loads(annotations_json.read_text(encoding="utf-8"))
        categories_raw = payload.get("categories")
        images_raw = payload.get("images")
        if not isinstance(categories_raw, list) or not isinstance(images_raw, list):
            raise ValueError(f"Malformed LVIS annotations JSON: {annotations_json}")

        self.categories_by_norm_name: Dict[str, Dict[str, Any]] = {}
        self.categories_by_id: Dict[int, Dict[str, Any]] = {}
        for category in categories_raw:
            if not isinstance(category, Mapping):
                continue
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(category.get("name") or "").strip()
            if not name:
                continue
            entry = {
                "category_id": int(category_id),
                "name": name,
                "frequency": str(category.get("frequency") or "unknown"),
            }
            self.categories_by_id[int(category_id)] = dict(entry)
            norm_name = normalize_desc(name)
            if norm_name and norm_name not in self.categories_by_norm_name:
                self.categories_by_norm_name[norm_name] = dict(entry)

        self.images_by_key: Dict[str, Dict[str, Any]] = {}
        for image in images_raw:
            if not isinstance(image, Mapping):
                continue
            try:
                image_id = int(image.get("id"))
            except (TypeError, ValueError):
                continue
            coco_url = str(image.get("coco_url") or "").strip()
            key_variants = _image_key_variants(coco_url) if coco_url else []
            image_meta = {
                "image_id": int(image_id),
                "neg_category_ids": [
                    int(cat_id) for cat_id in list(image.get("neg_category_ids") or [])
                ],
                "not_exhaustive_category_ids": [
                    int(cat_id)
                    for cat_id in list(image.get("not_exhaustive_category_ids") or [])
                ],
            }
            for key in key_variants:
                self.images_by_key.setdefault(key, dict(image_meta))

    def image_meta_for(self, image_value: str) -> Optional[Dict[str, Any]]:
        for key in _image_key_variants(image_value):
            image_meta = self.images_by_key.get(key)
            if image_meta is not None:
                return dict(image_meta)
        return None

    def category_entry_for_desc(self, desc: str) -> Optional[Dict[str, Any]]:
        norm_desc = normalize_desc(str(desc or ""))
        if not norm_desc:
            return None
        entry = self.categories_by_norm_name.get(norm_desc)
        return dict(entry) if entry is not None else None

    def category_entries_from_ids(
        self, category_ids: Iterable[int]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for category_id in category_ids:
            cat_id = int(category_id)
            if cat_id in seen:
                continue
            seen.add(cat_id)
            entry = self.categories_by_id.get(cat_id)
            if entry is not None:
                out.append(dict(entry))
        return out


def _resolve_gt_jsonl_for_eval_artifact(pred_path: Path) -> Optional[Path]:
    from src.infer.pipeline import _find_resolved_config_for_jsonl

    resolved_cfg = _find_resolved_config_for_jsonl(pred_path)
    if not isinstance(resolved_cfg, Mapping):
        return None

    cfg = resolved_cfg.get("cfg")
    if not isinstance(cfg, Mapping):
        return None
    infer_cfg = cfg.get("infer")
    if not isinstance(infer_cfg, Mapping):
        return None

    gt_jsonl = infer_cfg.get("gt_jsonl")
    if not isinstance(gt_jsonl, str) or not gt_jsonl.strip():
        return None
    return Path(str(gt_jsonl))


def _maybe_backfill_lvis_metadata_for_eval(
    records: Sequence[Mapping[str, Any]],
    *,
    pred_path: Path,
    options: "EvalOptions",
) -> List[Dict[str, Any]]:
    metrics_mode = str(options.metrics or "f1ish").strip().lower()
    if metrics_mode not in {"lvis", "both"}:
        return [dict(record) for record in records]

    rows = [dict(record) for record in records]
    rows_advertise_lvis = bool(rows) and all(
        isinstance(row.get("metadata"), Mapping)
        and str(row["metadata"].get("dataset_policy") or "").strip().lower()
        == "lvis_federated"
        for row in rows
    )
    if rows_advertise_lvis:
        return rows

    gt_jsonl = _resolve_gt_jsonl_for_eval_artifact(pred_path)
    gt_jsonl_lower = str(gt_jsonl).lower() if gt_jsonl is not None else ""
    gt_jsonl_is_lvis = "/public_data/lvis/" in f"/{gt_jsonl_lower.lstrip('/')}"

    # `metrics=both` means "official dataset metric + f1-ish".
    # On COCO-like artifacts this should stay COCO + f1-ish, so we must not
    # attempt LVIS metadata backfill unless the artifact or recovered GT path
    # clearly points to LVIS federated evaluation.
    if metrics_mode == "both" and not rows_advertise_lvis and not gt_jsonl_is_lvis:
        return rows

    if gt_jsonl is None:
        if metrics_mode == "lvis":
            raise ValueError(
                "LVIS evaluation requires federated LVIS metadata on the artifact, "
                "or a recoverable infer.gt_jsonl via resolved_config.path so the "
                "offline evaluator can backfill metadata."
            )
        return rows

    annotations_path = _default_lvis_annotations_json(gt_jsonl)
    index = _LvisLegacyMetadataIndex(annotations_path)

    out: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and str(metadata.get("dataset_policy") or "").strip().lower()
            == "lvis_federated"
        ):
            out.append(dict(row))
            continue

        image_value = str(row.get("image") or "").strip()
        image_meta = index.image_meta_for(image_value)
        if image_meta is None:
            raise ValueError(
                "Unable to backfill LVIS metadata for image "
                f"{image_value!r} from {annotations_path}"
            )

        gt_objects_raw = row.get("gt")
        if not isinstance(gt_objects_raw, list):
            raise ValueError(
                "LVIS eval metadata backfill requires canonical `gt` objects in "
                f"gt_vs_pred rows for image {image_value!r}"
            )

        gt_categories: List[Dict[str, Any]] = []
        positive_category_ids: List[int] = []
        for obj in gt_objects_raw:
            if not isinstance(obj, Mapping):
                continue
            category_entry = index.category_entry_for_desc(str(obj.get("desc") or ""))
            if category_entry is None:
                raise ValueError(
                    "Unable to map GT desc to an LVIS category while backfilling "
                    f"metadata for image {image_value!r}: desc={obj.get('desc')!r}"
                )
            gt_categories.append(dict(category_entry))
            positive_category_ids.append(int(category_entry["category_id"]))

        enriched = dict(row)
        enriched["image_id"] = int(image_meta["image_id"])
        enriched["metadata"] = {
            "dataset": "lvis",
            "dataset_policy": "lvis_federated",
            "image_id": int(image_meta["image_id"]),
            "split": _infer_lvis_split(gt_jsonl),
            "lvis": {
                "gt_objects": list(gt_categories),
                "positive_categories": index.category_entries_from_ids(
                    positive_category_ids
                ),
                "neg_categories": index.category_entries_from_ids(
                    image_meta["neg_category_ids"]
                ),
                "not_exhaustive_categories": index.category_entries_from_ids(
                    image_meta["not_exhaustive_category_ids"]
                ),
            },
        }
        out.append(enriched)
    return out


def _use_lvis_backend(
    gt_samples: Sequence[Sample],
    *,
    options: EvalOptions,
) -> bool:
    metrics_mode = str(options.metrics or "f1ish").strip().lower()
    if metrics_mode == "coco":
        return False

    saw_lvis = False
    for sample in gt_samples:
        if extract_lvis_image_policy(sample.metadata) is not None:
            saw_lvis = True
            break

    if metrics_mode == "lvis" and not saw_lvis:
        raise ValueError(
            "LVIS evaluation requires federated LVIS metadata on the GT records, "
            "but no records advertised `metadata.dataset_policy = lvis_federated`."
        )
    return bool(saw_lvis and metrics_mode in {"lvis", "both"})


def _limit_dets_per_image(
    results: Sequence[Mapping[str, Any]],
    *,
    max_dets: int,
) -> List[Dict[str, Any]]:
    if int(max_dets) <= 0:
        return [dict(item) for item in results]

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for item in results:
        try:
            image_id = int(item.get("image_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        grouped.setdefault(image_id, []).append(dict(item))

    limited: List[Dict[str, Any]] = []
    for image_id in sorted(grouped):
        anns = grouped[image_id]
        anns.sort(key=lambda ann: float(ann.get("score", 0.0)), reverse=True)
        limited.extend(anns[: int(max_dets)])
    return limited


def _prepare_lvis_artifacts(
    gt_samples: Sequence[Sample],
    pred_samples: Sequence[Tuple[int, List[Dict[str, Any]]]],
    *,
    options: EvalOptions,
    counters: EvalCounters,
) -> Tuple[Dict[str, int], Dict[str, Any], List[Dict[str, Any]], bool]:
    policies: List[LvisImagePolicy] = []
    for sample in gt_samples:
        policy = extract_lvis_image_policy(sample.metadata)
        if policy is None:
            raise ValueError(
                "LVIS evaluation requires federated metadata on every GT sample; "
                f"sample image_id={sample.image_id} is missing it."
            )
        policies.append(policy)

    category_catalog = build_lvis_category_catalog(policies)
    if not category_catalog:
        raise ValueError(
            "LVIS evaluation requires category metadata, but no LVIS categories "
            "could be recovered from the GT records."
        )

    categories = {
        str(item.name): int(item.category_id)
        for item in sorted(category_catalog.values(), key=lambda cat: int(cat.category_id))
    }
    results = _to_coco_preds(
        list(pred_samples),
        categories,
        options=options,
        counters=counters,
    )
    results = _limit_dets_per_image(results, max_dets=int(options.lvis_max_dets))
    coco_gt_dict = _to_lvis_gt(
        gt_samples=list(gt_samples),
        category_catalog=category_catalog,
        add_box_segmentation=bool(options.use_segm),
    )
    run_segm = bool(options.use_segm and any("segmentation" in row for row in results))
    _accumulate_lvis_prediction_diagnostics(
        gt_samples=list(gt_samples),
        results=results,
        counters=counters,
    )
    return categories, coco_gt_dict, results, run_segm


def _to_lvis_gt(
    *,
    gt_samples: List[Sample],
    category_catalog: Mapping[int, LvisCategory],
    add_box_segmentation: bool = False,
) -> Dict[str, Any]:
    categories_by_norm_name = {
        str(item.norm_name): item for item in category_catalog.values()
    }

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for sample in gt_samples:
        policy = extract_lvis_image_policy(sample.metadata)
        if policy is None:
            raise ValueError(
                "LVIS evaluation requires federated metadata on every GT sample; "
                f"sample image_id={sample.image_id} is missing it."
            )

        negative_ids = sorted(
            int(item.category_id) for item in list(policy.neg_categories)
        )
        not_exhaustive_ids = sorted(
            int(item.category_id) for item in list(policy.not_exhaustive_categories)
        )

        images.append(
            {
                "id": int(sample.image_id),
                "file_name": sample.file_name,
                "width": int(sample.width),
                "height": int(sample.height),
                "neg_category_ids": negative_ids,
                "not_exhaustive_category_ids": not_exhaustive_ids,
            }
        )

        for obj in sample.objects:
            category_id_raw = obj.get("category_id")
            if category_id_raw is None:
                norm_desc = _normalize_desc(str(obj.get("desc", "")))
                mapped = categories_by_norm_name.get(norm_desc)
                if mapped is None:
                    raise ValueError(
                        "LVIS GT object is missing category_id and could not be "
                        f"resolved by desc={obj.get('desc', '')!r} "
                        f"(image_id={sample.image_id})."
                    )
                category_id = int(mapped.category_id)
            else:
                category_id = int(category_id_raw)

            x1, y1, x2, y2 = obj["bbox"]
            w = float(x2 - x1)
            h = float(y2 - y1)
            ann = {
                "id": int(ann_id),
                "image_id": int(sample.image_id),
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(max(w, 0.0) * max(h, 0.0)),
                "iscrowd": 0,
            }
            if obj.get("type") == "poly":
                ann["segmentation"] = [obj["points"]]
            elif add_box_segmentation:
                ann["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            annotations.append(ann)
            ann_id += 1

    categories_list = [
        {
            "id": int(item.category_id),
            "name": str(item.name),
            "frequency": _normalize_lvis_frequency_label(item.frequency),
        }
        for item in sorted(category_catalog.values(), key=lambda cat: int(cat.category_id))
    ]
    return {
        "info": {"dataset_policy": "lvis_federated"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }


def _result_to_eval_object(result: Mapping[str, Any]) -> Dict[str, Any]:
    bbox_raw = result.get("bbox")
    if not isinstance(bbox_raw, Sequence) or len(bbox_raw) != 4:
        raise ValueError(f"Malformed result bbox: {bbox_raw!r}")
    x, y, w, h = [float(v) for v in bbox_raw]
    obj = {
        "bbox": [x, y, x + w, y + h],
        "score": float(result.get("score", 0.0) or 0.0),
    }
    segmentation = result.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        obj["segmentation"] = segmentation
    return obj


def _accumulate_lvis_prediction_diagnostics(
    *,
    gt_samples: List[Sample],
    results: Sequence[Mapping[str, Any]],
    counters: EvalCounters,
) -> None:
    results_by_image_cat: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for row in results:
        try:
            image_id = int(row.get("image_id"))
            category_id = int(row.get("category_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        results_by_image_cat.setdefault(image_id, {}).setdefault(category_id, []).append(
            _result_to_eval_object(row)
        )

    for sample in gt_samples:
        policy = extract_lvis_image_policy(sample.metadata)
        if policy is None:
            continue

        gt_by_cat: Dict[int, List[Dict[str, Any]]] = {}
        for obj in sample.objects:
            category_id_raw = obj.get("category_id")
            if category_id_raw is None:
                continue
            gt_by_cat.setdefault(int(category_id_raw), []).append(dict(obj))

        positive_ids = {
            int(item.category_id) for item in list(policy.positive_categories)
        }
        negative_ids = {
            int(item.category_id) for item in list(policy.neg_categories)
        }
        not_exhaustive_ids = {
            int(item.category_id) for item in list(policy.not_exhaustive_categories)
        }

        for category_id, pred_rows in results_by_image_cat.get(int(sample.image_id), {}).items():
            pred_rows_sorted = sorted(
                list(pred_rows),
                key=lambda row: float(row.get("score", 0.0)),
                reverse=True,
            )
            gt_rows = list(gt_by_cat.get(int(category_id), []))
            matches = _greedy_match_by_iou(
                pred_rows_sorted,
                gt_rows,
                iou_thr=0.5,
                width=int(sample.width),
                height=int(sample.height),
            )
            matched_count = int(len(matches))
            unmatched_count = max(0, int(len(pred_rows_sorted)) - matched_count)

            if int(category_id) in negative_ids:
                counters.lvis_verified_negative_unmatched += int(len(pred_rows_sorted))
                continue

            if int(category_id) in positive_ids:
                counters.lvis_matched_verified_positive += int(matched_count)
                if int(category_id) in not_exhaustive_ids:
                    counters.lvis_ignored_not_exhaustive += int(unmatched_count)
                continue

            # Keep the legacy diagnostic split: predictions for categories that
            # are explicitly marked not-exhaustive are reported as such, even
            # when the official LVIS scorer filters them before accumulation.
            if int(category_id) in not_exhaustive_ids:
                counters.lvis_ignored_not_exhaustive += int(unmatched_count)
                continue

            counters.lvis_ignored_unevaluable += int(len(pred_rows_sorted))


class _OfficialLvisParams:
    def __init__(
        self,
        iou_type: str,
        *,
        iou_thrs: Optional[Sequence[float]],
        max_dets: int,
    ) -> None:
        self.img_ids: List[int] = []
        self.cat_ids: List[int] = []
        if iou_thrs:
            self.iou_thrs = np.array([float(v) for v in iou_thrs], dtype=float)
        else:
            self.iou_thrs = np.linspace(
                0.5,
                0.95,
                int(np.round((0.95 - 0.5) / 0.05)) + 1,
                endpoint=True,
            )
        self.rec_thrs = np.linspace(
            0.0,
            1.00,
            int(np.round((1.00 - 0.0) / 0.01)) + 1,
            endpoint=True,
        )
        self.max_dets = int(max_dets)
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1
        self.img_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type


def _lvis_ann_to_rle(ann: Mapping[str, Any], *, height: int, width: int) -> Any:
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, Mapping):
        return dict(segmentation)
    if isinstance(segmentation, list) and segmentation:
        rle = maskUtils.frPyObjects(segmentation, height, width)
        return maskUtils.merge(rle) if isinstance(rle, list) else rle

    bbox_raw = ann.get("bbox")
    if isinstance(bbox_raw, Sequence) and len(bbox_raw) == 4:
        x, y, w, h = [float(v) for v in bbox_raw]
        quad = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        rle = maskUtils.frPyObjects(quad, height, width)
        return maskUtils.merge(rle) if isinstance(rle, list) else rle

    raise ValueError(f"Unable to build LVIS segmentation payload from ann: {ann!r}")


class _OfficialLikeLvisEval:
    """In-repo LVIS evaluator aligned to the official lvis_eval.py flow."""

    def __init__(
        self,
        *,
        gt_dataset: Mapping[str, Any],
        detections: Sequence[Mapping[str, Any]],
        iou_type: str,
        iou_thrs: Optional[Sequence[float]],
        max_dets: int,
    ) -> None:
        if iou_type not in {"bbox", "segm"}:
            raise ValueError(f"Unsupported LVIS iou_type: {iou_type}")

        self.logger = logger
        self.gt_dataset = copy.deepcopy(dict(gt_dataset))
        self.images = {
            int(image["id"]): dict(image)
            for image in self.gt_dataset.get("images", [])
            if isinstance(image, Mapping) and image.get("id") is not None
        }
        self.categories = {
            int(cat["id"]): dict(cat)
            for cat in self.gt_dataset.get("categories", [])
            if isinstance(cat, Mapping) and cat.get("id") is not None
        }
        self.gt_annotations = [
            dict(ann)
            for ann in self.gt_dataset.get("annotations", [])
            if isinstance(ann, Mapping)
        ]
        for ann in self.gt_annotations:
            if ann.get("area") is None and isinstance(ann.get("bbox"), Sequence):
                _, _, w, h = [float(v) for v in ann["bbox"]]
                ann["area"] = float(max(w, 0.0) * max(h, 0.0))
        self.dt_annotations = []
        for idx, det in enumerate(detections, start=1):
            row = dict(det)
            row.setdefault("id", int(idx))
            if row.get("area") is None and isinstance(row.get("bbox"), Sequence):
                _, _, w, h = [float(v) for v in row["bbox"]]
                row["area"] = float(max(w, 0.0) * max(h, 0.0))
            self.dt_annotations.append(row)

        self.eval_imgs: List[Optional[Dict[str, Any]]] = []
        self.eval: Dict[str, Any] = {}
        self._gts: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self._dts: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self.params = _OfficialLvisParams(
            iou_type,
            iou_thrs=iou_thrs,
            max_dets=max_dets,
        )
        self.results: OrderedDict[str, float] = OrderedDict()
        self.ious: Dict[Tuple[int, int], Any] = {}

        self.params.img_ids = sorted(self.images.keys())
        self.params.cat_ids = sorted(self.categories.keys())
        self.img_nel: Dict[int, List[int]] = {}
        self.freq_groups: List[List[int]] = []

    def _prepare_freq_group(self) -> List[List[int]]:
        freq_groups = [[] for _ in self.params.img_count_lbl]
        for idx, cat_id in enumerate(self.params.cat_ids):
            cat = self.categories.get(int(cat_id), {})
            frequency = _normalize_lvis_frequency_label(cat.get("frequency"))
            if frequency in self.params.img_count_lbl:
                freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def _prepare(self) -> None:
        for gt in self.gt_annotations:
            gt = dict(gt)
            if "ignore" not in gt:
                gt["ignore"] = 0
            self._gts[int(gt["image_id"]), int(gt["category_id"])].append(gt)

        img_nl = {
            int(image_id): [
                int(cat_id)
                for cat_id in self.images.get(int(image_id), {}).get("neg_category_ids", [])
            ]
            for image_id in self.params.img_ids
        }
        img_pl: Dict[int, set[int]] = defaultdict(set)
        for ann in self.gt_annotations:
            img_pl[int(ann["image_id"])].add(int(ann["category_id"]))
        self.img_nel = {
            int(image_id): [
                int(cat_id)
                for cat_id in self.images.get(int(image_id), {}).get(
                    "not_exhaustive_category_ids", []
                )
            ]
            for image_id in self.params.img_ids
        }

        for dt in self.dt_annotations:
            img_id = int(dt["image_id"])
            cat_id = int(dt["category_id"])
            if cat_id not in img_nl.get(img_id, []) and cat_id not in img_pl.get(img_id, set()):
                continue
            self._dts[img_id, cat_id].append(dict(dt))

        self.freq_groups = self._prepare_freq_group()

    def _get_gt_dt(self, img_id: int, cat_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                ann
                for local_cat_id in self.params.cat_ids
                for ann in self._gts[img_id, int(local_cat_id)]
            ]
            dt = [
                ann
                for local_cat_id in self.params.cat_ids
                for ann in self._dts[img_id, int(local_cat_id)]
            ]
        return gt, dt

    def compute_iou(self, img_id: int, cat_id: int) -> Any:
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        dt_order = np.argsort([-float(d["score"]) for d in dt], kind="mergesort")
        dt = [dt[int(i)] for i in dt_order]
        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "bbox":
            gt_payload = [g["bbox"] for g in gt]
            dt_payload = [d["bbox"] for d in dt]
        else:
            image = self.images.get(int(img_id), {})
            height = int(image.get("height", 0) or 0)
            width = int(image.get("width", 0) or 0)
            gt_payload = [
                _lvis_ann_to_rle(g, height=height, width=width) for g in gt
            ]
            dt_payload = [
                _lvis_ann_to_rle(d, height=height, width=width) for d in dt
            ]

        return maskUtils.iou(dt_payload, gt_payload, iscrowd)

    def evaluate(self) -> None:
        self.logger.info("Running official-style LVIS evaluation (%s).", self.params.iou_type)

        self.params.img_ids = list(np.unique(self.params.img_ids))
        cat_ids = self.params.cat_ids if self.params.use_cats else [-1]

        self._prepare()
        self.ious = {
            (int(img_id), int(cat_id)): self.compute_iou(int(img_id), int(cat_id))
            for img_id in self.params.img_ids
            for cat_id in cat_ids
        }
        self.eval_imgs = [
            self.evaluate_img(int(img_id), int(cat_id), area_rng)
            for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def evaluate_img(
        self,
        img_id: int,
        cat_id: int,
        area_rng: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        gt, dt = self._get_gt_dt(img_id, cat_id)
        if len(gt) == 0 and len(dt) == 0:
            return None

        gt = [dict(g) for g in gt]
        dt = [dict(d) for d in dt]

        for gt_ann in gt:
            if gt_ann["ignore"] or (
                float(gt_ann["area"]) < float(area_rng[0])
                or float(gt_ann["area"]) > float(area_rng[1])
            ):
                gt_ann["_ignore"] = 1
            else:
                gt_ann["_ignore"] = 0

        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[int(i)] for i in gt_idx]

        dt_idx = np.argsort([-float(d["score"]) for d in dt], kind="mergesort")
        dt = [dt[int(i)] for i in dt_idx]

        ious = (
            self.ious[img_id, cat_id][:, gt_idx]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))
        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_local_idx, _dt in enumerate(dt):
                iou = min([float(iou_thr), 1 - 1e-10])
                match_idx = -1
                for gt_local_idx, _gt in enumerate(gt):
                    if gt_m[iou_thr_idx, gt_local_idx] > 0:
                        continue
                    if (
                        match_idx > -1
                        and gt_ig[match_idx] == 0
                        and gt_ig[gt_local_idx] == 1
                    ):
                        break
                    if ious[dt_local_idx, gt_local_idx] < iou:
                        continue
                    iou = ious[dt_local_idx, gt_local_idx]
                    match_idx = gt_local_idx

                if match_idx == -1:
                    continue

                dt_ig[iou_thr_idx, dt_local_idx] = gt_ig[match_idx]
                dt_m[iou_thr_idx, dt_local_idx] = gt[match_idx]["id"]
                gt_m[iou_thr_idx, match_idx] = _dt["id"]

        dt_ig_mask = [
            float(d["area"]) < float(area_rng[0])
            or float(d["area"]) > float(area_rng[1])
            or int(d["category_id"]) in self.img_nel.get(int(d["image_id"]), [])
            for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))

        return {
            "image_id": int(img_id),
            "category_id": int(cat_id),
            "area_rng": area_rng,
            "dt_ids": [int(d["id"]) for d in dt],
            "gt_ids": [int(g["id"]) for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [float(d["score"]) for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def accumulate(self) -> None:
        if not self.eval_imgs:
            self.logger.warning("No LVIS eval images found; run evaluate() first.")

        cat_ids = self.params.cat_ids if self.params.use_cats else [-1]
        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        precision = -np.ones((num_thrs, num_recalls, num_cats, num_area_rngs))
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))
        dt_pointers: Dict[int, Dict[int, Dict[str, Any]]] = {}

        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        for cat_idx in range(num_cats):
            nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                na = area_idx * num_imgs
                entries = [
                    self.eval_imgs[nk + na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                entries = [entry for entry in entries if entry is not None]
                if not entries:
                    continue

                dt_scores = np.concatenate([entry["dt_scores"] for entry in entries], axis=0)
                dt_ids = np.concatenate([entry["dt_ids"] for entry in entries], axis=0)
                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([entry["dt_matches"] for entry in entries], axis=1)[
                    :, dt_idx
                ]
                dt_ig = np.concatenate([entry["dt_ignore"] for entry in entries], axis=1)[
                    :, dt_idx
                ]
                gt_ig = np.concatenate([entry["gt_ignore"] for entry in entries])
                num_gt = np.count_nonzero(gt_ig == 0)
                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    pr = tp / (fp + tp + np.spacing(1))
                    pr_list = pr.tolist()
                    for precision_idx in range(num_tp - 1, 0, -1):
                        if pr_list[precision_idx] > pr_list[precision_idx - 1]:
                            pr_list[precision_idx - 1] = pr_list[precision_idx]

                    rec_insert_idx = np.searchsorted(
                        rc,
                        self.params.rec_thrs,
                        side="left",
                    )
                    pr_at_recall = [0.0] * num_recalls
                    try:
                        for recall_idx, precision_idx in enumerate(rec_insert_idx):
                            pr_at_recall[recall_idx] = pr_list[int(precision_idx)]
                    except Exception:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)

        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize(
        self,
        summary_type: str,
        iou_thr: Optional[float] = None,
        area_rng: str = "all",
        freq_group_idx: Optional[int] = None,
    ) -> float:
        aidx = [
            idx
            for idx, label in enumerate(self.params.area_rng_lbl)
            if label == area_rng
        ]

        if summary_type == "ap":
            scores = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                scores = scores[tidx]
            if freq_group_idx is not None:
                scores = scores[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                scores = scores[:, :, :, aidx]
        else:
            scores = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                scores = scores[tidx]
            scores = scores[:, :, aidx]

        valid_scores = scores[scores > -1]
        if len(valid_scores) == 0:
            return -1.0
        return float(np.mean(valid_scores))

    def summarize(self) -> None:
        if not self.eval:
            raise RuntimeError("Please run accumulate() before summarize().")

        max_dets = int(self.params.max_dets)
        self.results["AP"] = self._summarize("ap")
        self.results["AP50"] = self._summarize("ap", iou_thr=0.50)
        self.results["AP75"] = self._summarize("ap", iou_thr=0.75)
        self.results["APs"] = self._summarize("ap", area_rng="small")
        self.results["APm"] = self._summarize("ap", area_rng="medium")
        self.results["APl"] = self._summarize("ap", area_rng="large")
        self.results["APr"] = self._summarize("ap", freq_group_idx=0)
        self.results["APc"] = self._summarize("ap", freq_group_idx=1)
        self.results["APf"] = self._summarize("ap", freq_group_idx=2)
        self.results[f"AR@{max_dets}"] = self._summarize("ar")
        for area_rng in ["small", "medium", "large"]:
            self.results[f"AR{area_rng[0]}@{max_dets}"] = self._summarize(
                "ar",
                area_rng=area_rng,
            )

    def per_class_ap(self) -> Dict[int, float]:
        if not self.eval:
            return {}
        aidx = [
            idx
            for idx, label in enumerate(self.params.area_rng_lbl)
            if label == "all"
        ]
        precision = self.eval["precision"]
        out: Dict[int, float] = {}
        for cat_idx, cat_id in enumerate(self.params.cat_ids):
            scores = precision[:, :, cat_idx, aidx]
            valid_scores = scores[scores > -1]
            out[int(cat_id)] = (
                float(np.mean(valid_scores)) if len(valid_scores) else float("nan")
            )
        return out


def _run_lvis_eval(
    coco_gt: COCO,
    results: List[Dict[str, Any]],
    *,
    options: EvalOptions,
    run_segm: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    max_dets = int(options.lvis_max_dets)
    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    iou_types = ["bbox"]
    if run_segm:
        iou_types.append("segm")

    for iou_type in iou_types:
        official_eval = _OfficialLikeLvisEval(
            gt_dataset=cast(Mapping[str, Any], coco_gt.dataset),
            detections=results,
            iou_type=iou_type,
            iou_thrs=options.iou_thrs,
            max_dets=max_dets,
        )
        official_eval.evaluate()
        official_eval.accumulate()
        official_eval.summarize()
        prefix = f"{iou_type}_"
        for metric_name, value in official_eval.results.items():
            metrics[f"{prefix}{metric_name}"] = float(value)

        # Backward-compatible aliases for downstream tooling that still expects
        # the legacy key style without "@".
        metrics[f"{prefix}AR{max_dets}"] = float(official_eval.results[f"AR@{max_dets}"])
        metrics[f"{prefix}ARs{max_dets}"] = float(
            official_eval.results[f"ARs@{max_dets}"]
        )
        metrics[f"{prefix}ARm{max_dets}"] = float(
            official_eval.results[f"ARm@{max_dets}"]
        )
        metrics[f"{prefix}ARl{max_dets}"] = float(
            official_eval.results[f"ARl@{max_dets}"]
        )

        if iou_type == "bbox":
            cat_names = {
                int(cat.get("id")): str(cat.get("name", ""))
                for cat in coco_gt.dataset.get("categories", [])
                if isinstance(cat, Mapping) and cat.get("id") is not None
            }
            per_class = {
                cat_names[int(cat_id)]: float(ap)
                for cat_id, ap in official_eval.per_class_ap().items()
                if cat_names.get(int(cat_id))
            }

    return metrics, per_class
