"""Canonical GT-vs-Pred visualization resources and shared review rendering.

This module centralizes three responsibilities:

1. Normalize heterogeneous scene/object payloads into one canonical review
   resource contract.
2. Materialize canonical sidecars at ``vis_resources/gt_vs_pred.jsonl``.
3. Render canonical resources with the shared 1x2 GT-vs-Pred review semantics.

The renderer is intentionally strict: canonical matching must already be present.
Workflows that start from raw artifacts should materialize the canonical sidecar
first, then render from that sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.common.geometry import bbox_from_points, coerce_point_list, denorm_and_clamp
from src.common.geometry.object_geometry import extract_single_geometry
from src.common.paths import resolve_image_path_strict

CANONICAL_SCHEMA_VERSION = 1
_MATCH_IOU_THR = 0.5
_HEADER_HEIGHT = 52
_PANEL_GAP = 12
_CANONICAL_GT_DOMAIN = "canonical_gt_index"
_CANONICAL_PRED_DOMAIN = "canonical_pred_index"

_COLOR_GT = "#2ca02c"
_COLOR_FN = "#ff8c00"
_COLOR_MATCHED = "#2ca02c"
_COLOR_FP = "#d62728"
_COLOR_IGNORED = "#7f7f7f"
_COLOR_UNMATCHED = "#1f77b4"
DEFAULT_BBOX_OUTLINE_WIDTH = 1


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_vis_resource_path(source_jsonl: Path) -> Path:
    return source_jsonl.parent / "vis_resources" / "gt_vs_pred.jsonl"


def _load_matches_by_record_idx(matches_jsonl: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for fallback_record_idx, row in enumerate(_iter_jsonl(matches_jsonl)):
        image_id = row.get("image_id")
        try:
            record_idx = int(image_id)
        except (TypeError, ValueError):
            record_idx = int(fallback_record_idx)
        out[int(record_idx)] = dict(row)
    return out


def _load_per_image_by_record_idx(per_image_json: Path) -> Dict[int, Dict[str, Any]]:
    payload = _load_json(per_image_json)
    if not isinstance(payload, list):
        raise ValueError(
            f"per-image evaluator report must be a list, got {type(payload).__name__}"
        )
    out: Dict[int, Dict[str, Any]] = {}
    for fallback_record_idx, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(
                "per-image evaluator report rows must be mappings, "
                f"got {type(row).__name__}"
            )
        image_id = row.get("image_id")
        try:
            record_idx = int(image_id)
        except (TypeError, ValueError):
            record_idx = int(fallback_record_idx)
        out[int(record_idx)] = dict(row)
    return out


def _coerce_record_idx(record: Mapping[str, Any], fallback: int) -> int:
    for key in ("record_idx", "index", "line_idx"):
        value = record.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _coerce_size(record: Mapping[str, Any], *, key: str, record_idx: int) -> int:
    raw = record.get(key)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"record {record_idx}: `{key}` must be int-compatible, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"record {record_idx}: `{key}` must be > 0, got {value}")
    return value


def _coerce_image_field(record: Mapping[str, Any], *, record_idx: int) -> str:
    image = record.get("image")
    if isinstance(image, str) and image.strip():
        return image.strip()
    images = record.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                return value.strip()
    file_name = record.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return file_name.strip()
    raise ValueError(f"record {record_idx}: missing `image`/`images[0]`/`file_name`")


def _normalize_desc(value: Any) -> str:
    return str(value or "").strip()


def _is_canonical_object(obj: Mapping[str, Any]) -> bool:
    bbox = obj.get("bbox_2d")
    return (
        isinstance(obj.get("index"), int)
        and isinstance(obj.get("desc"), str)
        and isinstance(bbox, list)
        and len(bbox) == 4
    )


def _is_canonical_matching(match: Mapping[str, Any]) -> bool:
    return (
        str(match.get("pred_index_domain") or "") == _CANONICAL_PRED_DOMAIN
        and str(match.get("gt_index_domain") or "") == _CANONICAL_GT_DOMAIN
        and isinstance(match.get("matched_pairs"), list)
        and isinstance(match.get("fn_gt_indices"), list)
        and isinstance(match.get("fp_pred_indices"), list)
    )


def _is_canonical_record(
    record: Mapping[str, Any],
    *,
    require_matching: bool = True,
) -> bool:
    gt = record.get("gt")
    pred = record.get("pred")
    matching = record.get("matching")
    base_ok = (
        int(record.get("schema_version") or 0) == CANONICAL_SCHEMA_VERSION
        and str(record.get("coord_mode") or "") == "pixel"
        and isinstance(gt, list)
        and isinstance(pred, list)
        and all(isinstance(obj, Mapping) and _is_canonical_object(obj) for obj in gt)
        and all(isinstance(obj, Mapping) and _is_canonical_object(obj) for obj in pred)
    )
    if not base_ok:
        return False
    if not require_matching:
        return True
    return isinstance(matching, Mapping) and _is_canonical_matching(matching)


def _extract_source_object_index(
    obj: Mapping[str, Any],
    *,
    fallback: int,
    path: str,
) -> int:
    raw = obj.get("index")
    if raw is None:
        return int(fallback)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: `index` must be int-compatible, got {raw!r}") from exc


def _coerce_bbox_from_object(
    obj: Mapping[str, Any],
    *,
    width: int,
    height: int,
    record_coord_mode: str,
    path: str,
) -> List[int]:
    geom_type = "bbox_2d"
    raw_points: Sequence[Any] | None = None
    coord_mode_hint = str(
        obj.get("_coord_mode") or obj.get("coord_mode") or record_coord_mode or ""
    ).strip()

    if obj.get("bbox_2d") is not None:
        raw_points = obj.get("bbox_2d")
        geom_type = "bbox_2d"
    elif obj.get("bbox") is not None:
        raw_points = obj.get("bbox")
        geom_type = "bbox_2d"
    elif obj.get("points_norm1000") is not None:
        raw_points = obj.get("points_norm1000")
        geom_type = str(obj.get("geom_type") or obj.get("type") or "bbox_2d").strip()
        coord_mode_hint = "norm1000"
    else:
        try:
            geom_type, raw_points = extract_single_geometry(
                obj,
                allow_type_and_points=True,
                allow_nested_points=True,
                path=path,
            )
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise ValueError(f"{path}: geometry must be a coordinate sequence")

    points_numeric, had_tokens = coerce_point_list(raw_points)
    if points_numeric is None:
        raise ValueError(f"{path}: failed to coerce geometry coordinates")

    coord_mode = (
        "norm1000"
        if (had_tokens or coord_mode_hint == "norm1000")
        else "pixel"
    )
    points_px = denorm_and_clamp(
        points_numeric,
        width,
        height,
        coord_mode=coord_mode,
    )
    if geom_type == "bbox_2d":
        if len(points_px) != 4:
            raise ValueError(f"{path}: bbox_2d must contain 4 values")
        x1, y1, x2, y2 = [int(v) for v in points_px]
    else:
        x1f, y1f, x2f, y2f = bbox_from_points(points_px)
        x1, y1, x2, y2 = (int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f)))

    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    return [x_lo, y_lo, x_hi, y_hi]


def _normalize_gt_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    record_coord_mode: str,
    record_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    staged: List[Dict[str, Any]] = []
    seen_source_indices: set[int] = set()
    for fallback_index, raw_obj in enumerate(objects):
        source_index = _extract_source_object_index(
            raw_obj,
            fallback=fallback_index,
            path=f"record {record_idx}.gt[{fallback_index}]",
        )
        if source_index in seen_source_indices:
            raise ValueError(
                f"record {record_idx}.gt[{fallback_index}]: duplicate source index {source_index}"
            )
        seen_source_indices.add(source_index)
        bbox = _coerce_bbox_from_object(
            raw_obj,
            width=width,
            height=height,
            record_coord_mode=record_coord_mode,
            path=f"record {record_idx}.gt[{fallback_index}]",
        )
        staged.append(
            {
                "_source_index": int(source_index),
                "desc": _normalize_desc(raw_obj.get("desc")),
                "bbox_2d": bbox,
            }
        )

    staged.sort(
        key=lambda obj: (
            int(obj["bbox_2d"][0]),
            int(obj["bbox_2d"][1]),
            int(obj["bbox_2d"][2]),
            int(obj["bbox_2d"][3]),
            str(obj["desc"]),
            int(obj["_source_index"]),
        )
    )

    source_to_canonical: Dict[int, int] = {}
    out: List[Dict[str, Any]] = []
    for canonical_index, obj in enumerate(staged):
        source_to_canonical[int(obj["_source_index"])] = int(canonical_index)
        out.append(
            {
                "index": int(canonical_index),
                "desc": str(obj["desc"]),
                "bbox_2d": [int(v) for v in obj["bbox_2d"]],
            }
        )
    return out, source_to_canonical


def _normalize_pred_objects(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    record_coord_mode: str,
    record_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    out: List[Dict[str, Any]] = []
    source_to_canonical: Dict[int, int] = {}
    seen_source_indices: set[int] = set()
    for fallback_index, raw_obj in enumerate(objects):
        source_index = _extract_source_object_index(
            raw_obj,
            fallback=fallback_index,
            path=f"record {record_idx}.pred[{fallback_index}]",
        )
        if source_index in seen_source_indices:
            raise ValueError(
                f"record {record_idx}.pred[{fallback_index}]: duplicate source index {source_index}"
            )
        seen_source_indices.add(source_index)
        bbox = _coerce_bbox_from_object(
            raw_obj,
            width=width,
            height=height,
            record_coord_mode=record_coord_mode,
            path=f"record {record_idx}.pred[{fallback_index}]",
        )
        canonical_index = int(source_index)
        source_to_canonical[int(source_index)] = int(canonical_index)
        out.append(
            {
                "index": int(canonical_index),
                "desc": _normalize_desc(raw_obj.get("desc")),
                "bbox_2d": bbox,
            }
        )
    return out, source_to_canonical


def _bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _build_greedy_matching(
    *,
    gt: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
    iou_thr: float = _MATCH_IOU_THR,
) -> Dict[str, Any]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_obj in pred:
        pred_index = int(pred_obj["index"])
        pred_box = pred_obj["bbox_2d"]
        for gt_obj in gt:
            gt_index = int(gt_obj["index"])
            iou = _bbox_iou(pred_box, gt_obj["bbox_2d"])
            if iou >= float(iou_thr):
                candidates.append((float(iou), int(pred_index), int(gt_index)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_pairs: List[List[int]] = []
    for _iou, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        matched_pairs.append([int(pred_index), int(gt_index)])

    fn_gt = sorted(
        int(obj["index"]) for obj in gt if int(obj["index"]) not in matched_gt
    )
    fp_pred = sorted(
        int(obj["index"]) for obj in pred if int(obj["index"]) not in matched_pred
    )
    return {
        "match_source": "materialized",
        "match_policy": "greedy_iou",
        "pred_index_domain": _CANONICAL_PRED_DOMAIN,
        "gt_index_domain": _CANONICAL_GT_DOMAIN,
        "matched_pairs": matched_pairs,
        "fn_gt_indices": fn_gt,
        "fp_pred_indices": fp_pred,
        "iou_thr": float(iou_thr),
        "pred_scope": "all",
        "unmatched_pred_indices": list(fp_pred),
        "unmatched_gt_indices": list(fn_gt),
    }


def _normalize_index_list(
    raw_indices: Any,
    *,
    domain: str,
    mapping: Mapping[int, int],
    allowed_indices: set[int],
    field_name: str,
) -> List[int]:
    if raw_indices is None:
        return []
    if not isinstance(raw_indices, list):
        raise ValueError(f"{field_name} must be a list[int]")
    out: List[int] = []
    for item in raw_indices:
        try:
            source_index = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain int-compatible values") from exc
        if domain == "canonical":
            canonical_index = source_index
        else:
            if source_index not in mapping:
                raise ValueError(
                    f"{field_name} contains index {source_index} missing from canonicalized objects"
                )
            canonical_index = int(mapping[source_index])
        if canonical_index not in allowed_indices:
            raise ValueError(
                f"{field_name} contains out-of-range canonical index {canonical_index}"
            )
        out.append(int(canonical_index))
    return sorted(set(out))


def _normalize_matched_pairs(
    raw_pairs: Any,
    *,
    pred_domain: str,
    gt_domain: str,
    pred_mapping: Mapping[int, int],
    gt_mapping: Mapping[int, int],
    allowed_pred_indices: set[int],
    allowed_gt_indices: set[int],
    field_name: str,
) -> List[List[int]]:
    if raw_pairs is None:
        return []
    if not isinstance(raw_pairs, list):
        raise ValueError(f"{field_name} must be a list[[pred_idx, gt_idx], ...]")
    seen: set[Tuple[int, int]] = set()
    out: List[List[int]] = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{field_name} entries must be [pred_idx, gt_idx]")
        pred_raw, gt_raw = pair
        try:
            pred_source = int(pred_raw)
            gt_source = int(gt_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} entries must be int-compatible") from exc
        pred_index = (
            pred_source
            if pred_domain == "canonical"
            else int(pred_mapping.get(pred_source, -1))
        )
        gt_index = (
            gt_source if gt_domain == "canonical" else int(gt_mapping.get(gt_source, -1))
        )
        if pred_index not in allowed_pred_indices:
            raise ValueError(f"{field_name} pred index {pred_source} is out of range")
        if gt_index not in allowed_gt_indices:
            raise ValueError(f"{field_name} gt index {gt_source} is out of range")
        key = (int(pred_index), int(gt_index))
        if key in seen:
            continue
        seen.add(key)
        out.append([int(pred_index), int(gt_index)])
    out.sort(key=lambda pair: (pair[0], pair[1]))
    return out


def _normalize_matching(
    match: Mapping[str, Any],
    *,
    pred_mapping: Mapping[int, int],
    gt_mapping: Mapping[int, int],
    pred_count: int,
    gt_count: int,
    pred_indices: set[int],
    gt_indices: set[int],
) -> Dict[str, Any]:
    if "matches" in match and "matched_pairs" not in match:
        raw_pairs = [
            [row.get("pred_idx"), row.get("gt_idx")]
            for row in (match.get("matches") or [])
            if isinstance(row, Mapping)
        ]
        raw_fn = match.get("unmatched_gt_indices")
        raw_fp = match.get("unmatched_pred_indices")
        raw_ignored = match.get("ignored_pred_indices")
        pred_scope = match.get("pred_scope")
        iou_thr = match.get("iou_thr")
        match_source = "detection_eval"
        match_policy = "f1ish_primary"
        pred_domain_name = "source"
        gt_domain_name = "source"
    else:
        raw_pairs = match.get("matched_pairs")
        raw_fn = match.get("fn_gt_indices")
        raw_fp = match.get("fp_pred_indices")
        raw_ignored = match.get("ignored_pred_indices")
        pred_scope = match.get("pred_scope")
        iou_thr = match.get("iou_thr")
        match_source = match.get("match_source") or "precomputed"
        match_policy = match.get("match_policy") or "precomputed"
        pred_domain_name = (
            "canonical"
            if str(match.get("pred_index_domain") or "") == _CANONICAL_PRED_DOMAIN
            else "source"
        )
        gt_domain_name = (
            "canonical"
            if str(match.get("gt_index_domain") or "") == _CANONICAL_GT_DOMAIN
            else "source"
        )

    matched_pairs = _normalize_matched_pairs(
        raw_pairs,
        pred_domain=pred_domain_name,
        gt_domain=gt_domain_name,
        pred_mapping=pred_mapping,
        gt_mapping=gt_mapping,
        allowed_pred_indices=set(pred_indices),
        allowed_gt_indices=set(gt_indices),
        field_name="matching.matched_pairs",
    )
    fn_gt_indices = _normalize_index_list(
        raw_fn,
        domain=gt_domain_name,
        mapping=gt_mapping,
        allowed_indices=set(gt_indices),
        field_name="matching.fn_gt_indices",
    )
    fp_pred_indices = _normalize_index_list(
        raw_fp,
        domain=pred_domain_name,
        mapping=pred_mapping,
        allowed_indices=set(pred_indices),
        field_name="matching.fp_pred_indices",
    )
    ignored_pred_indices = _normalize_index_list(
        raw_ignored,
        domain=pred_domain_name,
        mapping=pred_mapping,
        allowed_indices=set(pred_indices),
        field_name="matching.ignored_pred_indices",
    )

    matched_pred = {int(pair[0]) for pair in matched_pairs}
    matched_gt = {int(pair[1]) for pair in matched_pairs}
    unmatched_pred_indices = sorted(
        set(fp_pred_indices)
        | {
            idx
            for idx in pred_indices
            if idx not in matched_pred and idx not in ignored_pred_indices
        }
    )
    unmatched_gt_indices = sorted(
        set(fn_gt_indices) | {idx for idx in gt_indices if idx not in matched_gt}
    )

    out: Dict[str, Any] = {
        "match_source": str(match_source),
        "match_policy": str(match_policy),
        "pred_index_domain": _CANONICAL_PRED_DOMAIN,
        "gt_index_domain": _CANONICAL_GT_DOMAIN,
        "matched_pairs": matched_pairs,
        "fn_gt_indices": fn_gt_indices,
        "fp_pred_indices": fp_pred_indices,
    }
    if iou_thr is not None:
        out["iou_thr"] = float(iou_thr)
    if pred_scope is not None:
        out["pred_scope"] = str(pred_scope)
    if ignored_pred_indices:
        out["ignored_pred_indices"] = ignored_pred_indices
    if unmatched_pred_indices:
        out["unmatched_pred_indices"] = unmatched_pred_indices
    if unmatched_gt_indices:
        out["unmatched_gt_indices"] = unmatched_gt_indices
    if isinstance(match.get("gating_rejections"), list):
        out["gating_rejections"] = list(match.get("gating_rejections") or [])
    return out


def canonicalize_gt_objects_for_comparison(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    coord_mode: str = "pixel",
) -> List[Dict[str, Any]]:
    out, _ = _normalize_gt_objects(
        objects,
        width=width,
        height=height,
        record_coord_mode=coord_mode,
        record_idx=-1,
    )
    return out


def canonicalize_gt_vs_pred_record(
    record: Mapping[str, Any],
    *,
    fallback_record_idx: int,
    source_kind: str,
    explicit_matching: Mapping[str, Any] | None = None,
    materialize_matching: bool = True,
) -> Dict[str, Any]:
    return _canonicalize_record(
        record,
        fallback_record_idx=fallback_record_idx,
        source_kind=source_kind,
        explicit_matching=explicit_matching,
        materialize_matching=materialize_matching,
    )


def _canonicalize_record(
    record: Mapping[str, Any],
    *,
    fallback_record_idx: int,
    source_kind: str,
    explicit_matching: Mapping[str, Any] | None,
    materialize_matching: bool,
) -> Dict[str, Any]:
    if _is_canonical_record(record):
        return dict(record)

    record_idx = _coerce_record_idx(record, fallback_record_idx)
    width = _coerce_size(record, key="width", record_idx=record_idx)
    height = _coerce_size(record, key="height", record_idx=record_idx)
    image = _coerce_image_field(record, record_idx=record_idx)
    record_coord_mode = str(record.get("coord_mode") or "").strip()

    gt_raw = record.get("gt") or record.get("objects") or []
    pred_raw = record.get("pred") or record.get("predictions") or []
    if not isinstance(gt_raw, list):
        raise ValueError(f"record {record_idx}: `gt`/`objects` must be a list")
    if not isinstance(pred_raw, list):
        raise ValueError(f"record {record_idx}: `pred`/`predictions` must be a list")

    gt, gt_mapping = _normalize_gt_objects(
        gt_raw,
        width=width,
        height=height,
        record_coord_mode=record_coord_mode,
        record_idx=record_idx,
    )
    pred, pred_mapping = _normalize_pred_objects(
        pred_raw,
        width=width,
        height=height,
        record_coord_mode=record_coord_mode,
        record_idx=record_idx,
    )
    pred_indices = {int(obj["index"]) for obj in pred}
    gt_indices = {int(obj["index"]) for obj in gt}

    raw_matching = explicit_matching
    if raw_matching is None:
        match_payload = record.get("matching")
        if isinstance(match_payload, Mapping):
            raw_matching = match_payload
        else:
            match_payload = record.get("match")
            if isinstance(match_payload, Mapping):
                raw_matching = match_payload

    if raw_matching is not None:
        matching = _normalize_matching(
            raw_matching,
            pred_mapping=pred_mapping,
            gt_mapping=gt_mapping,
            pred_count=len(pred),
            gt_count=len(gt),
            pred_indices=pred_indices,
            gt_indices=gt_indices,
        )
    elif materialize_matching:
        matching = _build_greedy_matching(gt=gt, pred=pred)
    else:
        matching = None

    out: Dict[str, Any] = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "source_kind": str(record.get("source_kind") or source_kind),
        "record_idx": int(record_idx),
        "image": image,
        "width": int(width),
        "height": int(height),
        "coord_mode": "pixel",
        "gt": gt,
        "pred": pred,
    }
    if matching is not None:
        out["matching"] = matching

    for key in ("image_id", "file_name", "images", "stats", "provenance", "debug"):
        value = record.get(key)
        if value is not None:
            out[key] = value
    return out


def materialize_gt_vs_pred_vis_resource(
    source_jsonl: Path,
    *,
    output_path: Path | None = None,
    source_kind: str = "offline_single_run",
    external_matches: Mapping[int, Mapping[str, Any]] | None = None,
    materialize_matching: bool = True,
) -> Path:
    out_path = output_path or default_vis_resource_path(source_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for fallback_record_idx, record in enumerate(_iter_jsonl(source_jsonl)):
            record_idx = _coerce_record_idx(record, fallback_record_idx)
            explicit_matching = None
            if external_matches is not None:
                explicit_matching = external_matches.get(int(record_idx))
            canonical = _canonicalize_record(
                record,
                fallback_record_idx=fallback_record_idx,
                source_kind=source_kind,
                explicit_matching=explicit_matching,
                materialize_matching=materialize_matching,
            )
            provenance = dict(canonical.get("provenance") or {})
            provenance.setdefault("source_jsonl_dir", str(source_jsonl.parent))
            canonical["provenance"] = provenance
            handle.write(json.dumps(canonical, ensure_ascii=False) + "\n")
    return out_path


def materialize_eval_gt_vs_pred_vis_resource(
    source_jsonl: Path,
    *,
    matches_jsonl: Path | None = None,
    per_image_json: Path | None = None,
    output_path: Path | None = None,
    source_kind: str = "detection_eval",
) -> Path:
    external_matches = (
        _load_matches_by_record_idx(matches_jsonl) if matches_jsonl is not None else None
    )
    per_image_by_record = (
        _load_per_image_by_record_idx(per_image_json) if per_image_json is not None else {}
    )

    out_path = output_path or default_vis_resource_path(source_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for fallback_record_idx, raw_record in enumerate(_iter_jsonl(source_jsonl)):
            record = dict(raw_record)
            record_idx = _coerce_record_idx(record, fallback_record_idx)
            per_image_row = per_image_by_record.get(int(record_idx))
            if per_image_row is not None:
                image_id = per_image_row.get("image_id")
                file_name = per_image_row.get("file_name")
                if image_id is not None:
                    record.setdefault("image_id", image_id)
                if isinstance(file_name, str) and file_name.strip():
                    record.setdefault("file_name", file_name.strip())

            explicit_matching = None
            if external_matches is not None:
                explicit_matching = external_matches.get(int(record_idx))

            canonical = _canonicalize_record(
                record,
                fallback_record_idx=fallback_record_idx,
                source_kind=source_kind,
                explicit_matching=explicit_matching,
                materialize_matching=True,
            )
            provenance = dict(canonical.get("provenance") or {})
            provenance.setdefault("source_jsonl_dir", str(source_jsonl.parent))
            if matches_jsonl is not None:
                provenance.setdefault("matches_jsonl", str(matches_jsonl))
            if per_image_json is not None:
                provenance.setdefault("per_image_json", str(per_image_json))
            canonical["provenance"] = provenance
            handle.write(json.dumps(canonical, ensure_ascii=False) + "\n")
    return out_path


def ensure_gt_vs_pred_vis_resource(
    source_jsonl: Path,
    *,
    output_path: Path | None = None,
    source_kind: str = "offline_single_run",
    materialize_matching: bool = True,
) -> Path:
    first_record = next(_iter_jsonl(source_jsonl), None)
    if first_record is None:
        raise ValueError(f"Empty JSONL: {source_jsonl}")
    if _is_canonical_record(first_record):
        return source_jsonl
    return materialize_gt_vs_pred_vis_resource(
        source_jsonl,
        output_path=output_path,
        source_kind=source_kind,
        external_matches=None,
        materialize_matching=materialize_matching,
    )


def _load_canonical_records(
    path: Path,
    *,
    require_matching: bool = True,
) -> List[Dict[str, Any]]:
    records = list(_iter_jsonl(path))
    if not records:
        return []
    for idx, record in enumerate(records):
        if not _is_canonical_record(record, require_matching=require_matching):
            raise ValueError(
                f"record {idx} in {path} is not a canonical GT-vs-Pred visualization resource"
            )
    return records


def _review_priority(record: Mapping[str, Any]) -> Tuple[int, int]:
    matching = record.get("matching") or {}
    fn_count = len(matching.get("fn_gt_indices") or [])
    fp_count = len(matching.get("fp_pred_indices") or [])
    return (-(int(fn_count) + int(fp_count)), int(record.get("record_idx") or 0))


def _text_size(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    text: str,
) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left), int(bottom - top)


def _truncate_to_width(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    text: str,
    *,
    max_width: int,
) -> str:
    if max_width <= 0:
        return ""
    if _text_size(draw, font, text)[0] <= max_width:
        return text
    ellipsis = "..."
    lo = 0
    hi = len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _text_size(draw, font, candidate)[0] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _overlaps(existing: Sequence[Tuple[int, int, int, int]], rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    for ex1, ey1, ex2, ey2 in existing:
        if x1 >= ex2 or ex1 >= x2 or y1 >= ey2 or ey1 >= y2:
            continue
        return True
    return False


def _place_label(
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    panel_width: int,
    panel_height: int,
    bbox: Sequence[int],
    text: str,
    occupied: List[Tuple[int, int, int, int]],
) -> Tuple[str, Tuple[int, int, int, int]] | None:
    max_width = max(24, panel_width - 8)
    label = _truncate_to_width(draw, font, text, max_width=max_width)
    if not label:
        return None
    text_w, text_h = _text_size(draw, font, label)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    candidates = [
        (x1, max(0, y1 - text_h - 4)),
        (x1, min(max(0, panel_height - text_h - 2), y2 + 4)),
        (max(0, x2 - text_w), max(0, y1 - text_h - 4)),
        (max(0, x2 - text_w), min(max(0, panel_height - text_h - 2), y2 + 4)),
    ]
    for shift in range(4):
        dy = shift * (text_h + 2)
        candidates.extend(
            [
                (x1, max(0, y1 - text_h - 4 - dy)),
                (x1, min(max(0, panel_height - text_h - 2), y2 + 4 + dy)),
            ]
        )

    fallback: Tuple[int, int, int, int] | None = None
    for cand_x, cand_y in candidates:
        cand_x = max(0, min(int(cand_x), max(0, panel_width - text_w - 2)))
        cand_y = max(0, min(int(cand_y), max(0, panel_height - text_h - 2)))
        rect = (cand_x, cand_y, cand_x + text_w + 2, cand_y + text_h + 2)
        fallback = rect
        if not _overlaps(occupied, rect):
            occupied.append(rect)
            return label, rect
    if fallback is None:
        return None
    occupied.append(fallback)
    return label, fallback


def _draw_panel(
    *,
    img: Image.Image,
    objects: Sequence[Mapping[str, Any]],
    box_color_for_index: Mapping[int, str],
    label_prefix_for_index: Mapping[int, str],
) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    occupied: List[Tuple[int, int, int, int]] = []

    for obj in objects:
        obj_index = int(obj["index"])
        bbox = [int(v) for v in obj["bbox_2d"]]
        color = str(box_color_for_index.get(obj_index) or _COLOR_UNMATCHED)
        draw.rectangle(bbox, outline=color, width=DEFAULT_BBOX_OUTLINE_WIDTH)
        prefix = label_prefix_for_index.get(obj_index)
        if not prefix:
            continue
        desc = _normalize_desc(obj.get("desc"))
        label_text = prefix if not desc else f"{prefix}: {desc}"
        placed = _place_label(
            draw=draw,
            font=font,
            panel_width=out.size[0],
            panel_height=out.size[1],
            bbox=bbox,
            text=label_text,
            occupied=occupied,
        )
        if placed is None:
            continue
        text, (rx1, ry1, rx2, ry2) = placed
        draw.rectangle([rx1, ry1, rx2, ry2], fill="white")
        draw.text((rx1 + 1, ry1 + 1), text, fill=color, font=font)
    return out


def render_gt_vs_pred_review(
    jsonl_path: Path,
    *,
    out_dir: Path,
    limit: int = 20,
    root_image_dir: Path | None = None,
    root_source: str = "none",
    record_order: Literal["input", "error_first"] = "input",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_canonical_records(jsonl_path, require_matching=False)
    if record_order == "error_first":
        records = sorted(records, key=_review_priority)

    if root_image_dir is not None:
        root_image_dir = root_image_dir.resolve()

    for render_index, record in enumerate(records):
        if limit and render_index >= limit:
            break
        matching = record.get("matching")
        if not isinstance(matching, Mapping):
            raise ValueError(
                f"record {record.get('record_idx')} is missing canonical `matching`"
            )
        if not _is_canonical_matching(matching):
            raise ValueError(
                f"record {record.get('record_idx')} has non-canonical `matching`"
            )
        provenance = record.get("provenance") or {}
        source_jsonl_dir_raw = provenance.get("source_jsonl_dir")
        fallback_jsonl_dir = jsonl_path.parent
        if isinstance(source_jsonl_dir_raw, str) and source_jsonl_dir_raw.strip():
            fallback_jsonl_dir = Path(source_jsonl_dir_raw)
        image_path = resolve_image_path_strict(
            str(record.get("image") or ""),
            jsonl_dir=fallback_jsonl_dir,
            root_image_dir=root_image_dir,
        )
        if image_path is None:
            raise FileNotFoundError(
                f"review render: missing image for record {record.get('record_idx')} "
                f"(image={record.get('image')!r}, root_source={root_source})"
            )
        image = Image.open(image_path).convert("RGB")
        if image.size != (int(record["width"]), int(record["height"])):
            image = image.resize((int(record["width"]), int(record["height"])))

        gt_objects = list(record.get("gt") or [])
        pred_objects = list(record.get("pred") or [])
        fn_gt = {int(v) for v in (matching.get("fn_gt_indices") or [])}
        fp_pred = {int(v) for v in (matching.get("fp_pred_indices") or [])}
        ignored_pred = {int(v) for v in (matching.get("ignored_pred_indices") or [])}
        matched_pred = {int(pair[0]) for pair in (matching.get("matched_pairs") or [])}

        gt_colors = {
            int(obj["index"]): (_COLOR_FN if int(obj["index"]) in fn_gt else _COLOR_GT)
            for obj in gt_objects
        }
        gt_labels = {
            int(obj["index"]): "FN" for obj in gt_objects if int(obj["index"]) in fn_gt
        }
        pred_colors: Dict[int, str] = {}
        pred_labels: Dict[int, str] = {}
        for obj in pred_objects:
            pred_index = int(obj["index"])
            if pred_index in fp_pred:
                pred_colors[pred_index] = _COLOR_FP
                pred_labels[pred_index] = "FP"
            elif pred_index in matched_pred:
                pred_colors[pred_index] = _COLOR_MATCHED
            elif pred_index in ignored_pred:
                pred_colors[pred_index] = _COLOR_IGNORED
            else:
                pred_colors[pred_index] = _COLOR_UNMATCHED

        gt_panel = _draw_panel(
            img=image,
            objects=gt_objects,
            box_color_for_index=gt_colors,
            label_prefix_for_index=gt_labels,
        )
        pred_panel = _draw_panel(
            img=image,
            objects=pred_objects,
            box_color_for_index=pred_colors,
            label_prefix_for_index=pred_labels,
        )

        width = int(record["width"])
        height = int(record["height"])
        canvas = Image.new(
            "RGB",
            (width * 2 + _PANEL_GAP, height + _HEADER_HEIGHT),
            color=(255, 255, 255),
        )
        canvas.paste(gt_panel, (0, _HEADER_HEIGHT))
        canvas.paste(pred_panel, (width + _PANEL_GAP, _HEADER_HEIGHT))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        title = (
            f"record={int(record['record_idx'])} "
            f"GT={len(gt_objects)} Pred={len(pred_objects)} "
            f"FN={len(fn_gt)} FP={len(fp_pred)}"
        )
        draw.text((8, 6), title, fill="black", font=font)
        draw.text((8, 24), "GT", fill="black", font=font)
        draw.text((width + _PANEL_GAP + 8, 24), "Pred", fill="black", font=font)
        legend = [
            ("GT", _COLOR_GT),
            ("FN", _COLOR_FN),
            ("Matched", _COLOR_MATCHED),
            ("FP", _COLOR_FP),
        ]
        legend_x = max(8, canvas.size[0] - 260)
        for idx, (label, color) in enumerate(legend):
            x = legend_x + idx * 62
            draw.rectangle([x, 8, x + 10, 18], fill=color, outline=color)
            draw.text((x + 14, 5), label, fill="black", font=font)

        save_path = out_dir / f"vis_{render_index:04d}.png"
        canvas.save(save_path)
