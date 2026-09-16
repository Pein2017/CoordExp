"""Bounded Co-DETR corroboration for Source256 unmatched hypotheses.

The input is a JSONL file produced by the Source256 owner.  This module only
does detector corroboration and review-packet rendering.  It never reads GT,
never paints a candidate into a detector input, and never admits a teacher.

CPU preparation is intentionally separate from detector execution::

    python probes/training_set_completion/source256_codetr.py prepare CANDIDATES OUT
    conda run -n mmdet python probes/training_set_completion/source256_codetr.py infer PREPARED OUT

The detector call reuses the installed Co-DETR MMDetection API and the pinned
predecessor helper.  The config's native test pipeline supplies keep-ratio
Resize with ``img_scale=(2048, 1280)``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


MAX_VISUAL_REVIEW_CANDIDATES = 128
MAX_VISUAL_REVIEW_PER_IMAGE = 2
SCORE_THRESHOLD = 0.50
SUPPORT_IOU = 0.75
CONTEXT_MULTIPLIER = 3.0
MIN_CONTEXT_SIDE = 128
RESIZE_SCALE = (2048, 1280)

CODETR_ROOT = Path("/data/CoordExp/external/Co-DETR")
CONFIG = CODETR_ROOT / "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"
WEIGHTS = CODETR_ROOT / "models/co_dino_5scale_vit_large_coco.pth"
HELPER = CODETR_ROOT / "tools/codetr_infer_human_refined12.py"
PREDECESSOR_PROBE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-10-autonomous-unmatched-evaluator/codetr_probe.py"
)

_ID_RE = re.compile(r"[A-Za-z0-9_.:-]+\Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text()
    if path.suffix == ".json":
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else [loaded]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty candidate JSONL")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("candidate rows must be JSON objects")
    return rows


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _canonical_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    candidate_id = _first(row, "candidate_id", "case_id")
    if not isinstance(candidate_id, str) or not candidate_id or not _ID_RE.fullmatch(candidate_id):
        raise ValueError(f"invalid candidate_id at row {index}: {candidate_id!r}")
    image_id = _first(row, "image_id", "example_id")
    if image_id is None or isinstance(image_id, bool) or not str(image_id):
        raise ValueError(f"missing image_id/example_id at row {index}")
    image_path_value = _first(row, "image_path", "path")
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(f"missing image_path at row {index}")
    image_path = Path(image_path_value)
    if not image_path.is_file():
        raise ValueError(f"missing image: {image_path}")
    bbox_value = _first(row, "bbox", "bbox_xyxy", "pixel_bbox", "bbox_pixel_xyxy")
    if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
        raise ValueError(f"missing pixel bbox at row {index}")
    bbox = [_number(value, "bbox") for value in bbox_value]
    with Image.open(image_path) as image:
        actual_width, actual_height = image.size
    width = _first(row, "width", "image_width")
    height = _first(row, "height", "image_height")
    width = actual_width if width is None else int(width)
    height = actual_height if height is None else int(height)
    if (width, height) != (actual_width, actual_height):
        raise ValueError(f"declared dimensions disagree with image at row {index}")
    x1, y1, x2, y2 = bbox
    bbox_valid = 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    category = _first(
        row, "category", "category_name", "class_name", "normalized_description", "description"
    )
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"missing detector category at row {index}")
    # Preserve every producer field while adding a small canonical surface.
    canonical = dict(row)
    canonical.update({
        "candidate_id": candidate_id,
        "image_id": str(image_id),
        "image_path": str(image_path),
        "bbox": bbox,
        "width": width,
        "height": height,
        "category": category.strip().lower().replace("_", " "),
        "input_image_sha256": sha256_file(image_path),
        "bbox_valid": bbox_valid,
        "bbox_invalid_reason": None if bbox_valid else "nonpositive_or_out_of_bounds_pixel_box",
    })
    return canonical


def validate_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Validate the direct Source256 detector-corpus candidate interface."""

    rows = [_canonical_row(row, index) for index, row in enumerate(_rows_from_path(Path(path)))]
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate_id")
    crop_names = [_safe_name(candidate_id) for candidate_id in candidate_ids]
    if len(set(crop_names)) != len(crop_names):
        raise ValueError("candidate_id crop filename collision")
    return rows


def context_window(row: dict[str, Any]) -> list[int]:
    """Return the integer source-image crop window [left, top, right, bottom]."""

    x1, y1, x2, y2 = row["bbox"]
    width, height = row["width"], row["height"]
    crop_width = min(width, max(MIN_CONTEXT_SIDE, round(CONTEXT_MULTIPLIER * (x2 - x1))))
    crop_height = min(height, max(MIN_CONTEXT_SIDE, round(CONTEXT_MULTIPLIER * (y2 - y1))))
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(width - crop_width, max(0, round(center_x - crop_width / 2.0)))
    top = min(height - crop_height, max(0, round(center_y - crop_height / 2.0)))
    return [int(left), int(top), int(left + crop_width), int(top + crop_height)]


def _safe_name(candidate_id: str) -> str:
    return candidate_id.replace(":", "_")


def _prepare_row(row: dict[str, Any], crop_path: Path) -> dict[str, Any]:
    prepared = dict(row)
    if not row["bbox_valid"]:
        prepared.update({
            "context_window_xyxy": None,
            "context_path": None,
            "context_width": None,
            "context_height": None,
            "transform": {
                "kind": "not_applicable",
                "source_frame": "original_image_pixels",
                "reason": row["bbox_invalid_reason"],
            },
        })
        return prepared
    window = context_window(row)
    prepared.update({
        "context_window_xyxy": window,
        "context_path": str(crop_path),
        "context_width": window[2] - window[0],
        "context_height": window[3] - window[1],
        "transform": {
            "kind": "crop_then_native_keep_ratio_resize",
            "source_frame": "original_image_pixels",
            "context_frame": "unpainted_crop_pixels_before_detector_resize",
            "crop_xyxy_source": window,
            "resize_img_scale": list(RESIZE_SCALE),
            "resize_keep_ratio": True,
            "detector_to_source": "add crop left/top exactly once after official rescale=True inference",
        },
    })
    return prepared


def prepare(candidates_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create immutable raw context crops and a transform-bound manifest."""

    candidates_path, output_dir = Path(candidates_path), Path(output_dir)
    rows = validate_candidates(candidates_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    crop_dir = output_dir / "context-crops"
    crop_dir.mkdir()
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        crop_path = crop_dir / f"{_safe_name(row['candidate_id'])}.png"
        if row["bbox_valid"]:
            window = context_window(row)
            with Image.open(row["image_path"]) as image:
                image.convert("RGB").crop(tuple(window)).save(crop_path)
        prepared_rows.append(_prepare_row(row, crop_path))
    prepared_path = output_dir / "candidates.jsonl"
    prepared_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared_rows))
    manifest = {
        "status": "prepared",
        "scope": "Source256 Co-DETR corroboration only; no GT, labels, or model admission",
        "candidate_path": str(candidates_path.resolve()),
        "candidate_sha256": sha256_file(candidates_path),
        "candidate_count": len(prepared_rows),
        "unique_images": len({row["image_id"] for row in prepared_rows}),
        "detector_candidate_count": len(prepared_rows),
        "visual_review_budget": {
            "max_unique_candidates": MAX_VISUAL_REVIEW_CANDIDATES,
            "max_per_image": MAX_VISUAL_REVIEW_PER_IMAGE,
            "selection_stage": "after detector triage",
        },
        "context_rule": "3x candidate width/height centered, minimum 128 source pixels per side, image clipped",
        "resize_rule": {"img_scale": list(RESIZE_SCALE), "keep_ratio": True},
        "code_sha256": sha256_file(Path(__file__)),
        "detector_binding": {
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "weights": str(WEIGHTS),
            "weights_sha256": sha256_file(WEIGHTS),
            "predecessor_helper": str(HELPER),
            "predecessor_helper_sha256": sha256_file(HELPER),
            "predecessor_probe": str(PREDECESSOR_PROBE),
            "predecessor_probe_sha256": sha256_file(PREDECESSOR_PROBE),
            "backend": "MMDetection official init_detector/inference_detector",
        },
        "prepared_candidates": str(prepared_path),
        "prepared_candidates_sha256": sha256_file(prepared_path),
        "invalid_candidate_count": sum(not row["bbox_valid"] for row in prepared_rows),
        "crops": [
            {
                "candidate_id": row["candidate_id"],
                "path": row["context_path"],
                "sha256": sha256_file(row["context_path"]),
                "source_image_sha256": row["input_image_sha256"],
                "window": row["context_window_xyxy"],
            }
            for row in prepared_rows
            if row["bbox_valid"]
        ],
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return manifest


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def triage(candidate: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify detector evidence without creating an admission decision."""

    if not candidate.get("bbox_valid", True):
        return {
            "triage": "unknown",
            "teacher_admission": "not_run",
            "support_priority": False,
            "candidate_iou": None,
            "selected_same_category": None,
            "alternative_category_iou": None,
            "alternative_category_prediction": None,
            "category_conflict_flag": False,
            "score_threshold": SCORE_THRESHOLD,
            "support_iou_threshold": SUPPORT_IOU,
            "technical_status": "invalid_candidate_geometry",
            "scope": "candidate geometry invalid; no detector call and no negative label",
        }

    candidate_box = candidate["bbox"]
    valid = [prediction for prediction in predictions if prediction["score"] >= SCORE_THRESHOLD]
    same = [prediction for prediction in valid if prediction["category"] == candidate["category"]]
    other = [prediction for prediction in valid if prediction["category"] != candidate["category"]]
    selected = max(same, key=lambda p: iou_xyxy(p["bbox_xyxy_source"], candidate_box), default=None)
    alternative = max(other, key=lambda p: iou_xyxy(p["bbox_xyxy_source"], candidate_box), default=None)
    same_iou = iou_xyxy(selected["bbox_xyxy_source"], candidate_box) if selected else 0.0
    alternative_iou = iou_xyxy(alternative["bbox_xyxy_source"], candidate_box) if alternative else 0.0
    category_conflict = alternative_iou >= 0.50
    # A detector miss is explicitly unknown.  A same-class partial localization
    # or competing category is a conflict for visual prioritization, not a
    # negative label and not a rejection of the physical hypothesis.
    if same_iou >= SUPPORT_IOU:
        status = "support"
    elif same_iou >= 0.25 or category_conflict:
        status = "conflict"
    else:
        status = "unknown"
    return {
        "triage": status,
        "teacher_admission": "not_run",
        "support_priority": same_iou >= SUPPORT_IOU,
        "candidate_iou": same_iou,
        "selected_same_category": selected,
        "alternative_category_iou": alternative_iou,
        "alternative_category_prediction": alternative,
        "category_conflict_flag": category_conflict,
        "score_threshold": SCORE_THRESHOLD,
        "support_iou_threshold": SUPPORT_IOU,
        "scope": "Co-DETR support-priority signal only; detector miss is unknown; lead visual review owns admission",
    }


def select_visual_review(decisions: list[dict[str, Any]]) -> list[str]:
    """Select bounded review cases after full-corpus detector inference.

    Support and conflict are prioritized over unknown.  Within each status,
    candidates are taken in image round-robin order, up to two per image.  The
    input order remains the producer's order and is the tie-breaker.
    """

    image_order: list[str] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in decisions:
        if item["decision"].get("technical_status") == "invalid_candidate_geometry":
            continue
        image_id = item["image_id"]
        if image_id not in grouped:
            image_order.append(image_id)
            grouped[image_id] = {"support": [], "conflict": [], "unknown": []}
        grouped[image_id][item["decision"]["triage"]].append(item)
    selected: list[str] = []
    selected_per_image: Counter[str] = Counter()
    for status in ("support", "conflict", "unknown"):
        for _round in range(MAX_VISUAL_REVIEW_PER_IMAGE):
            for image_id in image_order:
                if len(selected) >= MAX_VISUAL_REVIEW_CANDIDATES:
                    return selected
                if selected_per_image[image_id] >= MAX_VISUAL_REVIEW_PER_IMAGE:
                    continue
                candidates = grouped[image_id][status]
                if not candidates:
                    continue
                item = candidates.pop(0)
                selected.append(item["candidate_id"])
                selected_per_image[image_id] += 1
    return selected


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("source256_codetr_predecessor", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load predecessor helper: {HELPER}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _cfg_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _assert_native_resize(model: Any) -> None:
    pipeline = _cfg_value(_cfg_value(model.cfg, "data"), "test")
    pipeline = _cfg_value(pipeline, "pipeline")
    if pipeline is None:
        raise RuntimeError("Co-DETR config has no test pipeline")
    resize = None
    for stage in pipeline:
        transforms = _cfg_value(stage, "transforms")
        if transforms is not None:
            for transform in transforms:
                if _cfg_value(transform, "type") == "Resize":
                    resize = transform
                    break
        if _cfg_value(stage, "type") == "Resize":
            resize = stage
    img_scale = _cfg_value(resize, "img_scale") if resize is not None else None
    keep_ratio = _cfg_value(resize, "keep_ratio") if resize is not None else None
    # The native config puts img_scale on MultiScaleFlipAug, with keep_ratio on
    # its nested Resize.  Require both pieces instead of trusting one field.
    scale_owner = next((stage for stage in pipeline if _cfg_value(stage, "img_scale") is not None), None)
    img_scale = _cfg_value(scale_owner, "img_scale", img_scale)
    if tuple(img_scale or ()) != RESIZE_SCALE or keep_ratio is not True:
        raise RuntimeError(f"unexpected detector resize: img_scale={img_scale!r}, keep_ratio={keep_ratio!r}")


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box: Iterable[float], color: tuple[int, int, int], label: str, font: Any) -> None:
    values = [round(float(value)) for value in box]
    draw.rectangle(values, outline=color, width=3)
    draw.text((values[0], max(0, values[1] - 20)), label, fill=color, font=font)


def _render_figures(row: dict[str, Any], predictions: list[dict[str, Any]], decision: dict[str, Any], figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(row["candidate_id"])
    font = _font()
    source_path = Path(row["image_path"])
    with Image.open(source_path) as source:
        full = source.convert("RGB")
        full_draw = ImageDraw.Draw(full)
        _draw_box(full_draw, row["bbox"], (245, 48, 48), f"candidate {row['category']}", font)
        for prediction in predictions:
            if prediction["score"] < SCORE_THRESHOLD:
                continue
            _draw_box(full_draw, prediction["bbox_xyxy_source"], (38, 155, 245),
                      f"det {prediction['category']} {prediction['score']:.2f}", font)
        full_path = figure_dir / f"{stem}_full.png"
        full.save(full_path)
    with Image.open(row["context_path"]) as context:
        context_image = context.convert("RGB")
        context_draw = ImageDraw.Draw(context_image)
        left, top, _, _ = row["context_window_xyxy"]
        local_candidate = [row["bbox"][0] - left, row["bbox"][1] - top,
                           row["bbox"][2] - left, row["bbox"][3] - top]
        _draw_box(context_draw, local_candidate, (245, 48, 48), f"candidate {row['category']}", font)
        for prediction in predictions:
            if prediction["score"] < SCORE_THRESHOLD:
                continue
            _draw_box(context_draw, prediction["bbox_xyxy_context"], (38, 155, 245),
                      f"det {prediction['category']} {prediction['score']:.2f}", font)
        context_path = figure_dir / f"{stem}_context.png"
        context_image.save(context_path)
    packet_path = figure_dir / f"{stem}.json"
    packet = {
        "candidate_id": row["candidate_id"],
        "image_id": row["image_id"],
        "source_image": str(source_path),
        "context_crop": row["context_path"],
        "full_figure": str(full_path),
        "context_figure": str(context_path),
        "candidate_bbox_source": row["bbox"],
        "context_window_source": row["context_window_xyxy"],
        "decision": decision,
        "figure_semantics": {
            "red": "Source unmatched candidate hypothesis",
            "blue": "Co-DETR detections at score >= 0.50, mapped from context crop",
            "detector_input": "raw unpainted context crop; overlays are rendered after inference",
            "not_ground_truth": True,
        },
    }
    _json_dump(packet_path, packet)
    return {"full": str(full_path), "context": str(context_path), "packet": str(packet_path)}


def infer(prepared_dir: str | Path, output_dir: str | Path, device: str = "cuda:0") -> dict[str, Any]:
    """Run Co-DETR on prepared context crops and write per-case evidence."""

    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    prepared_path = prepared_dir / "candidates.jsonl"
    if sha256_file(Path(manifest["candidate_path"])) != manifest["candidate_sha256"]:
        raise RuntimeError("candidate input changed after CPU preparation")
    if sha256_file(Path(__file__)) != manifest["code_sha256"]:
        raise RuntimeError("wrapper code changed after CPU preparation")
    if sha256_file(prepared_path) != manifest["prepared_candidates_sha256"]:
        raise RuntimeError("prepared candidate rows changed after CPU preparation")
    binding = manifest["detector_binding"]
    for path_key, hash_key in (("config", "config_sha256"), ("weights", "weights_sha256"),
                               ("predecessor_helper", "predecessor_helper_sha256"),
                               ("predecessor_probe", "predecessor_probe_sha256")):
        if sha256_file(Path(binding[path_key])) != binding[hash_key]:
            raise RuntimeError(f"detector binding changed after CPU preparation: {path_key}")
    rows = _rows_from_path(prepared_path)
    if len(rows) != manifest["candidate_count"]:
        raise RuntimeError("prepared candidate count mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "figures").mkdir()
    started = time.perf_counter()
    run_receipt: dict[str, Any] = {"status": "failed", "prepared_dir": str(prepared_dir.resolve())}
    try:
        import torch  # type: ignore
        import mmcv  # type: ignore
        import mmdet  # type: ignore

        helper = _load_helper()
        model = helper.init_detector(str(CONFIG), str(WEIGHTS), device=device)
        model.eval()
        _assert_native_resize(model)
        classes = tuple(model.CLASSES)
        unique_contexts: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for row in rows:
            if not row["bbox_valid"]:
                continue
            context_path = Path(row["context_path"])
            window = row["context_window_xyxy"]
            if not context_path.is_file():
                raise RuntimeError(f"missing prepared crop: {context_path}")
            key = str(context_path.resolve())
            unique_contexts.setdefault(key, {"row": row, "window": window, "path": context_path})
        # Check every crop against the preparation manifest in candidate order.
        crop_hashes = {item["candidate_id"]: item["sha256"] for item in manifest["crops"]}
        for item in unique_contexts.values():
            candidate_id = item["row"]["candidate_id"]
            if sha256_file(item["path"]) != crop_hashes.get(candidate_id):
                raise RuntimeError(f"prepared crop changed: {candidate_id}")
        predictions_by_context: dict[str, list[dict[str, Any]]] = {}
        latencies: list[float] = []
        raw_path = output_dir / "raw-detections.jsonl"
        with raw_path.open("x") as raw_handle:
            for row in rows:
                if row["bbox_valid"]:
                    continue
                raw_handle.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "source_image_sha256": row["input_image_sha256"],
                    "predictions": [],
                    "technical_status": "invalid_candidate_geometry",
                }, ensure_ascii=False) + "\n")
            for key, item in unique_contexts.items():
                row, window, context_path = item["row"], item["window"], item["path"]
                t0 = time.perf_counter()
                result = helper.inference_detector(model, str(context_path))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                predictions = helper.collect_predictions(result, classes, SCORE_THRESHOLD)
                mapped: list[dict[str, Any]] = []
                left, top = window[:2]
                for prediction in predictions:
                    context_box = [float(value) for value in prediction["bbox_xyxy"]]
                    source_box = [context_box[0] + left, context_box[1] + top,
                                  context_box[2] + left, context_box[3] + top]
                    mapped_prediction = dict(prediction)
                    mapped_prediction["bbox_xyxy_context"] = context_box
                    mapped_prediction["bbox_xyxy_source"] = source_box
                    mapped.append(mapped_prediction)
                predictions_by_context[key] = mapped
                raw_handle.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "context_path": str(context_path),
                    "context_window_xyxy": window,
                    "source_image_sha256": row["input_image_sha256"],
                    "context_sha256": sha256_file(context_path),
                    "wall_seconds": elapsed,
                    "predictions": mapped,
                }, ensure_ascii=False) + "\n")
                raw_handle.flush()
                latencies.append(elapsed)
        decisions_path = output_dir / "decisions.jsonl"
        decisions: list[dict[str, Any]] = []
        for row in rows:
            prediction_list = []
            if row["bbox_valid"]:
                key = str(Path(row["context_path"]).resolve())
                prediction_list = predictions_by_context[key]
            decision = triage(row, prediction_list)
            decision_record = dict(row)
            decision_record.update({"predictions": prediction_list, "decision": decision})
            decisions.append(decision_record)
        selected_ids = set(select_visual_review(decisions))
        for item in decisions:
            item["review_selected"] = item["candidate_id"] in selected_ids
            item["review_status"] = "selected" if item["review_selected"] else "HOLD"
            if item["review_selected"]:
                item["figures"] = _render_figures(
                    item, item["predictions"], item["decision"], output_dir / "figures"
                )
        decisions_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decisions))
        counts = Counter(item["decision"]["triage"] for item in decisions)
        review_counts = Counter(item["decision"]["triage"] for item in decisions if item["review_selected"])
        review_selection_path = output_dir / "review-selection.json"
        _json_dump(review_selection_path, {
            "scope": "bounded lead visual review selection after full detector corpus",
            "max_unique_candidates": MAX_VISUAL_REVIEW_CANDIDATES,
            "max_per_image": MAX_VISUAL_REVIEW_PER_IMAGE,
            "priority": ["support", "conflict", "unknown"],
            "round_robin": True,
            "selected_count": len(selected_ids),
            "selected_by_triage": dict(review_counts),
            "selected_candidate_ids": [item["candidate_id"] for item in decisions if item["review_selected"]],
            "held_count": len(decisions) - len(selected_ids),
        })
        run_receipt = {
            "status": "complete",
            "scope": "Co-DETR corroboration and visual packets; no automatic teacher admission and no VLM calls",
            "candidate_count": len(rows),
            "unique_images": len({row["image_id"] for row in rows}),
            "unique_context_inferences": len(unique_contexts),
            "technical_invalid_candidates": sum(not row["bbox_valid"] for row in rows),
            "triage_counts": dict(counts),
            "visual_review_selection_count": len(selected_ids),
            "visual_review_selection_counts": dict(review_counts),
            "visual_review_budget": {
                "max_unique_candidates": MAX_VISUAL_REVIEW_CANDIDATES,
                "max_per_image": MAX_VISUAL_REVIEW_PER_IMAGE,
            },
            "candidate_path": manifest["candidate_path"],
            "candidate_sha256": manifest["candidate_sha256"],
            "prepared_manifest_sha256": sha256_file(prepared_dir / "manifest.json"),
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "weights": str(WEIGHTS),
            "weights_sha256": sha256_file(WEIGHTS),
            "predecessor_helper": str(HELPER),
            "predecessor_helper_sha256": sha256_file(HELPER),
            "predecessor_probe": str(PREDECESSOR_PROBE),
            "predecessor_probe_sha256": sha256_file(PREDECESSOR_PROBE),
            "wrapper_sha256": sha256_file(Path(__file__)),
            "backend": "MMDetection official init_detector/inference_detector",
            "resize": {"img_scale": list(RESIZE_SCALE), "keep_ratio": True},
            "context_rule": "3x candidate width/height centered, minimum 128 source pixels per side, clipped",
            "score_threshold": SCORE_THRESHOLD,
            "support_iou_threshold": SUPPORT_IOU,
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "mmcv": getattr(mmcv, "__version__", "unknown"),
            "mmdet": getattr(mmdet, "__version__", "unknown"),
            "image_latencies_seconds": latencies,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "wall_seconds": time.perf_counter() - started,
            "raw_detections": str(raw_path),
            "decisions": str(decisions_path),
            "review_selection": str(review_selection_path),
            "figures_dir": str((output_dir / "figures").resolve()),
            "automatic_teacher_admission": 0,
            "unknown_semantics": "detector miss or unresolved disagreement remains unknown; no negative label",
        }
        _json_dump(output_dir / "runtime.json", run_receipt)
        return run_receipt
    except BaseException as exc:
        run_receipt.update({"error": repr(exc), "wall_seconds": time.perf_counter() - started})
        _json_dump(output_dir / "runtime.json", run_receipt)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="validate rows and create raw context crops")
    prep.add_argument("candidates")
    prep.add_argument("output_dir")
    run = subparsers.add_parser("infer", help="run installed Co-DETR on prepared crops")
    run.add_argument("prepared_dir")
    run.add_argument("output_dir")
    run.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.candidates, args.output_dir), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(infer(args.prepared_dir, args.output_dir, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
