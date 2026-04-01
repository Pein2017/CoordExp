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
from src.common.prediction_parsing import GEOM_KEYS
from src.common.semantic_desc import SemanticDescEncoder, normalize_desc
from src.common.io import load_jsonl_with_diagnostics
from src.eval.artifacts import build_per_image_report, write_outputs
from src.utils import get_logger
from src.vis import materialize_gt_vs_pred_vis_resource, render_gt_vs_pred_review

logger = get_logger(__name__)

_DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_desc(desc: str) -> str:
    return normalize_desc(desc)


def _fmt_iou_thr(iou_thr: float) -> str:
    return f"{float(iou_thr):.2f}"


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
    if rows and all(
        isinstance(row.get("metadata"), Mapping)
        and str(row["metadata"].get("dataset_policy") or "").strip().lower()
        == "lvis_federated"
        for row in rows
    ):
        return rows

    gt_jsonl = _resolve_gt_jsonl_for_eval_artifact(pred_path)
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

    encoder = SemanticDescEncoder(
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

    return records


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


def export_coco_submission(
    pred_path: Path,
    *,
    source_jsonl: Path,
    categories_json: Path,
    out_json: Path,
    options: EvalOptions,
) -> Dict[str, Any]:
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
    if prepare_coco:
        for record_idx, rec in enumerate(pred_records):
            _validate_score_provenance_for_coco(rec, record_idx)

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

    if _use_lvis_backend(gt_samples, options=options):
        categories, coco_gt_dict, results, run_segm = _prepare_lvis_artifacts(
            gt_samples,
            pred_samples,
            options=options,
            counters=counters,
        )
    else:
        categories = _build_categories(gt_samples)
        coco_gt_dict = _to_coco_gt(
            gt_samples, categories, add_box_segmentation=options.use_segm
        )
        results = _to_coco_preds(
            pred_samples, categories, options=options, counters=counters
        )
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


def compute_coco_metrics_from_records(
    pred_records: Sequence[Mapping[str, Any]],
    *,
    options: EvalOptions,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute official detection metrics from in-memory scored ``gt_vs_pred`` records.

    Returns ``(metrics, counters_dict)`` where ``metrics`` uses the same
    ``bbox_*`` / ``segm_*`` keys as offline evaluation. When ``options.metrics``
    requests LVIS and the records carry federated LVIS metadata, this dispatches
    to the LVIS-aware backend; otherwise it uses the COCO backend.
    """

    counters = EvalCounters()
    pred_records_list = [dict(r) for r in pred_records]
    (
        gt_samples,
        _pred_samples,
        _categories,
        coco_gt_dict,
        results,
        run_segm,
        _per_image,
    ) = _prepare_all(
        pred_records_list,
        options,
        counters,
        prepare_coco=True,
    )

    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(coco_gt_dict)
    coco_gt.createIndex()

    if _use_lvis_backend(gt_samples, options=options):
        metrics, _per_class = _run_lvis_eval(
            coco_gt,
            results,
            options=options,
            run_segm=run_segm,
        )
    else:
        metrics, _per_class = _run_coco_eval(
            coco_gt,
            results,
            options=options,
            run_segm=run_segm,
        )
    return metrics, counters.to_dict()


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
    metrics_mode = str(options.metrics).strip().lower()
    want_official = metrics_mode in {"coco", "lvis", "both"}

    if pred_path is None:
        pred_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            _pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _per_image,
        ) = _prepare_all(
            pred_records,
            options,
            counters,
            prepare_coco=want_official,
        )
    else:
        gt_records = load_jsonl(gt_path, counters, strict=options.strict_parse)
        pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
        (
            gt_samples,
            _pred_samples,
            categories,
            coco_gt_dict,
            results,
            run_segm,
            _per_image,
        ) = _prepare_all_separate(
            gt_records,
            pred_records,
            options,
            counters,
            prepare_coco=want_official,
        )

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_official:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        if _use_lvis_backend(gt_samples, options=options):
            metrics, per_class = _run_lvis_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )
        else:
            metrics, per_class = _run_coco_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )

    return {
        "metrics": metrics,
        "per_class": per_class,
        "counters": counters.to_dict(),
        "categories": categories,
    }


def evaluate_and_save(
    pred_path: Path,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    from src.infer.pipeline import resolve_root_image_dir_for_jsonl

    counters = EvalCounters()
    metrics_mode = str(options.metrics).strip().lower()
    want_official = metrics_mode in {"coco", "lvis", "both"}
    want_f1ish = metrics_mode in {"f1ish", "both"}

    pred_records = load_jsonl(pred_path, counters, strict=options.strict_parse)
    pred_records = _maybe_backfill_lvis_metadata_for_eval(
        pred_records,
        pred_path=pred_path,
        options=options,
    )
    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    ) = _prepare_all(
        pred_records,
        options,
        counters,
        prepare_coco=want_official,
    )

    metrics: Dict[str, float] = {}
    per_class: Dict[str, float] = {}
    if want_official:
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(coco_gt_dict)
        coco_gt.createIndex()
        if _use_lvis_backend(gt_samples, options=options):
            metrics, per_class = _run_lvis_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )
        else:
            metrics, per_class = _run_coco_eval(
                coco_gt,
                results,
                options=options,
                run_segm=run_segm,
            )

    if _use_lvis_backend(gt_samples, options=options):
        metrics.update(
            {
                "lvis_diag_matched_verified_positive": float(
                    counters.lvis_matched_verified_positive
                ),
                "lvis_diag_verified_negative_unmatched": float(
                    counters.lvis_verified_negative_unmatched
                ),
                "lvis_diag_ignored_not_exhaustive": float(
                    counters.lvis_ignored_not_exhaustive
                ),
                "lvis_diag_ignored_unevaluable": float(
                    counters.lvis_ignored_unevaluable
                ),
            }
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
    else:
        f1ish_summary = {"matches_by_thr": {}}

    vis_matches: Dict[int, Dict[str, Any]] | None = None
    if want_f1ish:
        primary_thr = _select_primary_f1ish_iou_thr(options.f1ish_iou_thrs)
        primary_key = _fmt_iou_thr(primary_thr)
        vis_matches = _matches_by_record_idx(
            f1ish_summary.get("matches_by_thr", {}).get(primary_key, [])
        )

    vis_resource_path = materialize_gt_vs_pred_vis_resource(
        pred_path,
        source_kind="detection_eval",
        external_matches=vis_matches,
        materialize_matching=True,
    )

    write_outputs(
        options.output_dir,
        coco_gt=coco_gt_dict if want_official else None,
        coco_preds=results if want_official else None,
        summary=summary,
        per_image=per_image,
    )

    if options.overlay:
        root_dir, root_source = resolve_root_image_dir_for_jsonl(pred_path)
        if root_dir is not None:
            logger.info(
                "Overlay image root resolved (source=%s): %s",
                root_source,
                root_dir,
            )

        overlay_dir = options.output_dir / "overlays"
        render_gt_vs_pred_review(
            vis_resource_path,
            out_dir=overlay_dir,
            limit=options.overlay_k,
            root_image_dir=root_dir,
            root_source=root_source,
            record_order="error_first",
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

    encoder = SemanticDescEncoder(
        model_name=model_name, device=str(device), batch_size=int(bs)
    )

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
        logger.warning(
            "Unknown f1ish_pred_scope='%s'; defaulting to 'annotated'", pred_scope
        )
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
    matches_records_by_thr: Dict[str, List[Dict[str, Any]]] = {}

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
            preds_ignored_orig = preds_ignored_orig_idx_by_image.get(
                sample.image_id, []
            )
            gts = gts_by_image.get(sample.image_id, [])
            matches = matches_by_thr[thr_key].get(sample.image_id, [])

            matched_pred = {
                pred_idx for pred_idx, _, _ in matches
            }  # pred_idx is in *eval* space
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
                f"{prefix}sem_acc_on_matched": float(sum_sem_ok / sum_tp)
                if sum_tp > 0
                else 1.0,
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
        matches_records_by_thr[thr_key] = matches_records

    return {
        "metrics": metrics_out,
        "matches_by_thr": matches_records_by_thr,
    }
