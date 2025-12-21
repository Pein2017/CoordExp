"""
Offline detection evaluator for CoordExp (pixel-space schema).

Features:
- Ingests standardized ``pred.jsonl`` (pixel-space ``gt`` / ``pred`` objects) or legacy GT JSONL.
- Converts geometries to COCO-format GT and prediction artifacts (bbox + segm for polygons).
- Lines are tolerated structurally but excluded from metrics.
- Runs COCOeval (bbox + segm) and emits metrics plus robustness counters.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.common.geometry import (
    bbox_from_points,
    coerce_point_list,
    denorm_and_clamp,
    flatten_points,
    is_degenerate_bbox,
)
from src.eval.parsing import GEOM_KEYS
from src.utils import get_logger

logger = get_logger(__name__)


def load_jsonl(
    path: Path, counters: EvalCounters | None = None
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                if counters is not None:
                    counters.invalid_json += 1
                continue
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

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


@dataclass
class EvalOptions:
    unknown_policy: str = "bucket"  # bucket | drop
    strict_parse: bool = False
    use_segm: bool = True
    iou_types: Tuple[str, ...] = ("bbox", "segm")
    iou_thrs: Optional[List[float]] = None  # None -> COCO defaults
    output_dir: Path = Path("eval_out")
    overlay: bool = False
    overlay_k: int = 12
    open_vocab_recall: bool = False  # class-agnostic recall
    num_workers: int = 0  # parallelize pred parsing/denorm on CPU


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
    for obj in objs_in:
        gkeys = [g for g in GEOM_KEYS if g in obj and obj[g] is not None]
        if len(gkeys) != 1:
            counters.invalid_geometry += 1
            invalid.append({"reason": "geometry_keys", "raw": obj})
            continue
        gtype = gkeys[0]
        pts_raw = flatten_points(obj[gtype])
        if pts_raw is None:
            counters.invalid_geometry += 1
            invalid.append({"reason": "geometry_points", "raw": obj})
            continue
        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            counters.invalid_coord += 1
            invalid.append({"reason": "coord_parse", "raw": obj})
            continue
        coord_mode = "norm1000" if had_tokens else "pixel"
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
    try:
        pred_w = int(record.get("width")) if record.get("width") is not None else None
        pred_h = int(record.get("height")) if record.get("height") is not None else None
        if pred_w and pred_w != width:
            counters.size_mismatch += 1
        if pred_h and pred_h != height:
            counters.size_mismatch += 1
    except Exception:
        pass

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
        gkeys = (
            [g for g in GEOM_KEYS if g in obj and obj[g] is not None]
            if not obj.get("type")
            else [obj["type"]]
        )
        if len(gkeys) != 1:
            counters.invalid_geometry += 1
            invalid.append({"reason": "geometry_keys", "raw": obj})
            continue
        gtype = gkeys[0]
        pts_value = obj.get(gtype) if gtype in obj else obj.get("points")
        pts_raw = flatten_points(pts_value)
        if pts_raw is None:
            counters.invalid_geometry += 1
            invalid.append({"reason": "geometry_points", "raw": obj})
            continue
        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            counters.invalid_coord += 1
            invalid.append({"reason": "coord_parse", "raw": obj})
            continue

        if gtype == "line":
            # lines are carried through in reports but excluded from metrics
            invalid.append({"reason": "line_skipped", "raw": obj})
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


def _build_categories(
    gt_samples: List[Sample], *, include_unknown: bool
) -> Dict[str, int]:
    cats: Dict[str, int] = {}
    next_id = 1
    for sample in gt_samples:
        for obj in sample.objects:
            desc = (obj.get("desc") or "").strip()
            if desc not in cats:
                cats[desc] = next_id
                next_id += 1
    if include_unknown and "unknown" not in cats:
        cats["unknown"] = next_id
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
    include_unknown = options.unknown_policy == "bucket"
    unknown_id = categories.get("unknown")

    for image_id, preds in pred_samples:
        for pred in preds:
            desc = (pred.get("desc") or "").strip()
            cat_id = categories.get(desc)
            if cat_id is None:
                if include_unknown and unknown_id is not None:
                    cat_id = unknown_id
                    counters.unknown_desc += 1
                else:
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


def _resolve_image_path(base_dir: Path, image_rel: str) -> Path:
    if image_rel is None:
        return base_dir / "missing.jpg"
    p = Path(image_rel)
    if p.is_absolute():
        return p
    return (base_dir / image_rel).resolve()


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
    except Exception as exc:
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
        except Exception as exc:
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


def _prepare_all(
    pred_records: List[Dict[str, Any]],
    options: EvalOptions,
    counters: EvalCounters,
) -> Tuple[
    List[Sample],
    List[Tuple[int, List[Dict[str, Any]]]],
    Dict[str, int],
    Dict[str, Any],
    List[Dict[str, Any]],
    bool,
    List[Dict[str, Any]],
]:
    # Derive GT from prediction file (must be present per line)
    gt_records = preds_to_gt_records(pred_records)
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
                for k, v in local_counts.items():
                    setattr(counters, k, getattr(counters, k) + v)
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

    include_unknown = options.unknown_policy == "bucket"
    categories = _build_categories(gt_samples, include_unknown=include_unknown)
    coco_gt_dict = _to_coco_gt(
        gt_samples, categories, add_box_segmentation=options.use_segm
    )
    results = _to_coco_preds(
        pred_samples, categories, options=options, counters=counters
    )
    run_segm = options.use_segm and any("segmentation" in r for r in results)
    per_image = build_per_image_report(gt_samples, pred_samples, invalid_preds)
    return (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    )


def evaluate_detection(
    pred_path: Path,
    *,
    options: EvalOptions,
) -> Dict[str, Any]:
    counters = EvalCounters()
    pred_records = load_jsonl(pred_path, counters)
    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        _,
    ) = _prepare_all(pred_records, options, counters)

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
    coco_gt: Dict[str, Any],
    coco_preds: List[Dict[str, Any]],
    summary: Dict[str, Any],
    per_image: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco_gt.json").write_text(
        json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8"
    )
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
    pred_records = load_jsonl(pred_path, counters)
    (
        gt_samples,
        pred_samples,
        categories,
        coco_gt_dict,
        results,
        run_segm,
        per_image,
    ) = _prepare_all(pred_records, options, counters)

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

    write_outputs(
        options.output_dir,
        coco_gt=coco_gt_dict,
        coco_preds=results,
        summary=summary,
        per_image=per_image,
    )

    if options.overlay:
        root_dir_env = os.environ.get("ROOT_IMAGE_DIR")
        base_dir = Path(root_dir_env) if root_dir_env else pred_path.parent
        overlay_dir = options.output_dir / "overlays"
        _draw_overlays(
            gt_samples,
            pred_samples,
            base_dir=base_dir,
            out_dir=overlay_dir,
            k=options.overlay_k,
        )

    return summary
