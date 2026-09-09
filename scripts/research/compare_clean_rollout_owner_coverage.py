#!/usr/bin/env python
"""Compare owner coverage in two clean ``gt_vs_pred.jsonl`` artifacts.

The artifact contract stores GT boxes as norm1000 bins and decoded prediction
boxes as pixels.  This utility deliberately keeps entity discovery (category
and box matching) separate from interpretation of unmatched predictions:
unmatched predictions are counted, but are never labelled hallucinations.

The broad duplicate metric remains a low-threshold geometry triage signal for
backward compatibility.  The stricter physical-owner metric is also a
geometry-derived candidate signal: it requires an unambiguous annotated owner
and an earlier overlapping prediction for that owner, but still requires image
review before it can be called a confirmed duplicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

try:  # Allow ``python scripts/research/...py`` from a checkout root.
    from src.data.geometry import coord_bins_to_pixel_xyxy
    from src.eval.detection_categories import normalize_coco_category_name
    from src.data.geometry import iou_xyxy
    from src.eval.assignment import global_matches as _global_matches
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI use
    if exc.name != "src":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.geometry import coord_bins_to_pixel_xyxy
    from src.eval.detection_categories import normalize_coco_category_name
    from src.data.geometry import iou_xyxy
    from src.eval.assignment import global_matches as _global_matches


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            row_id = value.get("row_id", value.get("example_id"))
            if row_id is None:
                raise ValueError(f"{path}:{line_number} is missing row_id")
            key = str(row_id)
            if key in rows:
                raise ValueError(f"{path} contains duplicate row_id {key!r}")
            rows[key] = value
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    return normalize_coco_category_name(
        obj.get("description", obj.get("desc", obj.get("label", obj.get("category", ""))))
    )


def _bbox(obj: Any) -> Any:
    return obj.get("bbox", obj.get("bbox_2d")) if isinstance(obj, dict) else None


def _pixel_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    return box


def _dimensions(row: dict[str, Any]) -> tuple[int, int] | None:
    width = row.get("image_width", row.get("width"))
    height = row.get("image_height", row.get("height"))
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    return width, height


def _gt_objects(row: dict[str, Any], *, row_id: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    objects = row.get("gt", [])
    if not isinstance(objects, list):
        raise ValueError(f"row {row_id!r} has malformed GT list")
    dimensions = _dimensions(row)
    if dimensions is None:
        raise ValueError(f"row {row_id!r} has invalid image dimensions")
    width, height = dimensions
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, obj in enumerate(objects):
        raw = _bbox(obj)
        try:
            box = coord_bins_to_pixel_xyxy(
                raw,
                image_width=width,
                image_height=height,
                field=f"gt[{index}].bbox",
            )
        except Exception as exc:
            raise ValueError(f"row {row_id!r} has invalid GT bbox at index {index}") from exc
        result.append((_category(obj), tuple(float(item) for item in box)))
    return result


def _pred_objects(row: dict[str, Any]) -> tuple[list[tuple[str, tuple[float, float, float, float]]], int]:
    objects = row.get("pred", [])
    if not isinstance(objects, list):
        return [], 1
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    invalid = 0
    for obj in objects:
        box = _pixel_box(_bbox(obj))
        if box is None or not _category(obj):
            invalid += 1
            continue
        result.append((_category(obj), box))
    return result, invalid


def _strict_physical_owner_duplicate_candidates(
    row: dict[str, Any],
    *,
    row_id: str,
    annotation_iou_threshold: float,
    prediction_iou_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return ordered, unambiguous physical-owner duplicate candidates.

    A valid prediction is attributed only when exactly one same-category GT
    owner reaches the annotation threshold.  A later attributed prediction is
    a candidate only when it overlaps an earlier attributed prediction for the
    same owner at the prediction-to-prediction threshold.  This deliberately
    stays a geometry-derived review aid rather than a semantic label.
    """
    gt = _gt_objects(row, row_id=row_id)
    raw_predictions = row.get("pred", [])
    if not isinstance(raw_predictions, list):
        return [], 0

    attributed_by_owner: dict[int, list[tuple[int, tuple[float, float, float, float], float]]] = {}
    candidates: list[dict[str, Any]] = []
    ambiguous_attribution_count = 0
    for prediction_index, value in enumerate(raw_predictions):
        category = _category(value)
        box = _pixel_box(_bbox(value))
        if not category or box is None:
            continue
        owner_candidates = [
            (iou_xyxy(gt_box, box), owner_index)
            for owner_index, (owner_category, gt_box) in enumerate(gt)
            if owner_category == category and iou_xyxy(gt_box, box) >= annotation_iou_threshold
        ]
        if len(owner_candidates) != 1:
            ambiguous_attribution_count += int(len(owner_candidates) > 1)
            continue
        owner_iou, owner_index = owner_candidates[0]
        earlier = attributed_by_owner.get(owner_index, [])
        overlapping_earlier = [
            (iou_xyxy(earlier_box, box), earlier_index, earlier_owner_iou)
            for earlier_index, earlier_box, earlier_owner_iou in earlier
            if iou_xyxy(earlier_box, box) >= prediction_iou_threshold
        ]
        if overlapping_earlier:
            prediction_overlap, earlier_index, earlier_owner_iou = max(
                overlapping_earlier,
                key=lambda item: (item[0], -item[1]),
            )
            candidates.append(
                {
                    "row_id": row_id,
                    "owner_index": owner_index,
                    "prediction_index": prediction_index,
                    "earlier_prediction_index": earlier_index,
                    "annotation_iou": owner_iou,
                    "earlier_annotation_iou": earlier_owner_iou,
                    "prediction_to_prediction_iou": prediction_overlap,
                }
            )
        attributed_by_owner.setdefault(owner_index, []).append((prediction_index, box, owner_iou))
    return candidates, ambiguous_attribution_count


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p90": None}
    def quantile(q: float) -> float:
        return ordered[int(round((len(ordered) - 1) * q))]
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
    }


def _row_metrics(
    row: dict[str, Any],
    *,
    row_id: str,
    match_iou_threshold: float,
    duplicate_iou_threshold: float,
    strict_annotation_iou_threshold: float,
    strict_prediction_iou_threshold: float,
) -> tuple[dict[str, Any], int, int, int, int, int]:
    gt = _gt_objects(row, row_id=row_id)
    pred, invalid_predictions = _pred_objects(row)
    dimensions_valid = _dimensions(row) is not None
    status = str(row.get("parse_status", "")).lower()
    malformed = int("malformed" in status)
    invalid = int("invalid" in status or not dimensions_valid or invalid_predictions > 0)
    reported_dropped = row.get("dropped_prediction_count", 0)
    try:
        dropped_predictions = max(0, int(reported_dropped or 0))
    except (TypeError, ValueError):
        dropped_predictions = 0
    if isinstance(row.get("dropped_predictions"), list):
        dropped_predictions = max(dropped_predictions, len(row["dropped_predictions"]))
    dropped = int(dropped_predictions > 0 or "dropped" in status)

    matches = _global_matches(gt, pred, match_iou_threshold)
    used_gt = {gt_index for gt_index, _, _ in matches}
    matched_ious: list[float] = []
    center_errors: list[float] = []
    size_errors: list[float] = []
    for gt_index, pred_index, overlap in matches:
        matched_ious.append(overlap)
        gx1, gy1, gx2, gy2 = gt[gt_index][1]
        px1, py1, px2, py2 = pred[pred_index][1]
        center_errors.append(math.hypot((gx1 + gx2 - px1 - px2) / 2, (gy1 + gy2 - py1 - py2) / 2))
        size_errors.append(math.hypot((gx2 - gx1) - (px2 - px1), (gy2 - gy1) - (py2 - py1)))

    # Assign each prediction with sufficient overlap to its best physical GT owner.
    # This includes unmatched predictions, while intentionally making no claim
    # about whether such predictions are hallucinations.
    owner_hits: dict[int, int] = {}
    ambiguous_duplicate_candidate_count = 0
    for pred_index, (_, pred_box) in enumerate(pred):
        owner_candidates = [
            (iou_xyxy(gt_box, pred_box), gt_index)
            for gt_index, (gt_category, gt_box) in enumerate(gt)
            if gt_category == pred[pred_index][0]
            and iou_xyxy(gt_box, pred_box) >= duplicate_iou_threshold
        ]
        if len(owner_candidates) == 1:
            _, owner = owner_candidates[0]
            owner_hits[owner] = owner_hits.get(owner, 0) + 1
        elif len(owner_candidates) > 1:
            ambiguous_duplicate_candidate_count += 1
    duplicate_candidate_count = sum(max(0, count - 1) for count in owner_hits.values())
    strict_duplicate_candidates, strict_ambiguous_attribution_count = _strict_physical_owner_duplicate_candidates(
        row,
        row_id=row_id,
        annotation_iou_threshold=strict_annotation_iou_threshold,
        prediction_iou_threshold=strict_prediction_iou_threshold,
    )
    return (
        {
            "gt_count": len(gt),
            "prediction_count": len(pred),
            "unique_matched_gt_owners": len(used_gt),
            "matched_iou": matched_ious,
            "center_errors_px": center_errors,
            "size_errors_px": size_errors,
            "duplicate_candidate_count": duplicate_candidate_count,
            "ambiguous_duplicate_candidate_count": ambiguous_duplicate_candidate_count,
            "strict_physical_owner_duplicate_candidate_count": len(strict_duplicate_candidates),
            "strict_physical_owner_duplicate_candidates": strict_duplicate_candidates,
            "strict_physical_owner_ambiguous_attribution_count": strict_ambiguous_attribution_count,
            "dropped_prediction_count": dropped_predictions,
            "invalid_prediction_count": invalid_predictions,
        },
        malformed,
        invalid,
        dropped,
        dropped_predictions,
        invalid_predictions,
    )


def _gt_signature(row: dict[str, Any], *, row_id: str) -> tuple[Any, ...]:
    # Validate and convert through the canonical path, but compare source
    # norm1000 bins so two distinct bins that round to the same pixel box do
    # not silently pass the GT-identity gate.
    _gt_objects(row, row_id=row_id)
    objects = row.get("gt", [])
    dimensions = _dimensions(row)
    return (
        dimensions,
        tuple(
            (
                obj.get("object_id", obj.get("id", index)),
                _category(obj),
                tuple(_bbox(obj)),
            )
            for index, obj in enumerate(objects)
        ),
    )


def _matched_owner_geometry(
    row: dict[str, Any], *, row_id: str, match_iou_threshold: float
) -> dict[int, tuple[float, float, float]]:
    """Return IoU, center error, and size error keyed by annotated owner index."""
    gt = _gt_objects(row, row_id=row_id)
    pred, _ = _pred_objects(row)
    result: dict[int, tuple[float, float, float]] = {}
    for gt_index, pred_index, overlap in _global_matches(gt, pred, match_iou_threshold):
        gx1, gy1, gx2, gy2 = gt[gt_index][1]
        px1, py1, px2, py2 = pred[pred_index][1]
        center_error = math.hypot(
            (gx1 + gx2 - px1 - px2) / 2,
            (gy1 + gy2 - py1 - py2) / 2,
        )
        size_error = math.hypot(
            (gx2 - gx1) - (px2 - px1),
            (gy2 - gy1) - (py2 - py1),
        )
        result[gt_index] = (overlap, center_error, size_error)
    return result


def _summarize(
    rows: dict[str, dict[str, Any]], *, match_iou_threshold: float, duplicate_iou_threshold: float,
    strict_annotation_iou_threshold: float, strict_prediction_iou_threshold: float,
) -> dict[str, Any]:
    totals = {"gt_count": 0, "prediction_count": 0, "unique_matched_gt_owners": 0,
              "duplicate_candidate_count": 0, "ambiguous_duplicate_candidate_count": 0,
              "strict_physical_owner_duplicate_candidate_count": 0,
              "strict_physical_owner_ambiguous_attribution_count": 0,
              "malformed_row_count": 0, "invalid_row_count": 0, "dropped_row_count": 0,
              "dropped_prediction_count": 0, "invalid_prediction_count": 0}
    ious: list[float] = []
    centers: list[float] = []
    sizes: list[float] = []
    strict_duplicate_candidates: list[dict[str, Any]] = []
    for row_id in sorted(rows):
        metrics, malformed, invalid, dropped, dropped_predictions, invalid_predictions = _row_metrics(
            rows[row_id], row_id=row_id, match_iou_threshold=match_iou_threshold,
            duplicate_iou_threshold=duplicate_iou_threshold,
            strict_annotation_iou_threshold=strict_annotation_iou_threshold,
            strict_prediction_iou_threshold=strict_prediction_iou_threshold,
        )
        for key in ("gt_count", "prediction_count", "unique_matched_gt_owners",
                    "duplicate_candidate_count", "ambiguous_duplicate_candidate_count",
                    "strict_physical_owner_duplicate_candidate_count",
                    "strict_physical_owner_ambiguous_attribution_count"):
            totals[key] += metrics[key]
        totals["malformed_row_count"] += malformed
        totals["invalid_row_count"] += invalid
        totals["dropped_row_count"] += dropped
        totals["dropped_prediction_count"] += dropped_predictions
        totals["invalid_prediction_count"] += invalid_predictions
        ious.extend(metrics["matched_iou"])
        centers.extend(metrics["center_errors_px"])
        sizes.extend(metrics["size_errors_px"])
        strict_duplicate_candidates.extend(metrics["strict_physical_owner_duplicate_candidates"])
    gt_count = totals["gt_count"]
    totals.update({
        "row_count": len(rows),
        "owner_coverage": totals["unique_matched_gt_owners"] / gt_count if gt_count else 0.0,
        "owner_false_negative_rate": 1.0 - totals["unique_matched_gt_owners"] / gt_count if gt_count else 0.0,
        "matched_iou": _distribution(ious),
        "center_error_mean_px": sum(centers) / len(centers) if centers else None,
        "size_error_mean_px": sum(sizes) / len(sizes) if sizes else None,
        "strict_physical_owner_duplicate_candidates": strict_duplicate_candidates,
    })
    return totals


def _select_rows(
    rows: dict[str, dict[str, Any]], *, include_row_ids: Iterable[str] | None
) -> tuple[dict[str, dict[str, Any]], list[str] | None]:
    if include_row_ids is None:
        return rows, None
    selected_ids = sorted({str(row_id) for row_id in include_row_ids})
    if not selected_ids:
        raise ValueError("include-row-id selection is empty")
    missing = sorted(set(selected_ids) - set(rows))
    if missing:
        raise ValueError(f"include-row-id values are absent from artifacts: {missing[:5]}")
    return {row_id: rows[row_id] for row_id in selected_ids}, selected_ids


def _read_include_row_ids(
    include_row_ids: Iterable[str], include_row_id_files: Iterable[Path]
) -> list[str] | None:
    selected = {str(row_id) for row_id in include_row_ids if str(row_id)}
    for path in include_row_id_files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read include-row-id file {path}") from exc
        selected.update(line.strip() for line in lines if line.strip())
    return sorted(selected) if selected else None


def compare_artifacts(
    arm_a_path: str | Path,
    arm_b_path: str | Path,
    *,
    match_iou_threshold: float = 0.50,
    duplicate_iou_threshold: float = 0.30,
    strict_annotation_iou_threshold: float = 0.50,
    strict_prediction_iou_threshold: float = 0.90,
    include_row_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not all(
        0.0 <= threshold <= 1.0
        for threshold in (
            match_iou_threshold,
            duplicate_iou_threshold,
            strict_annotation_iou_threshold,
            strict_prediction_iou_threshold,
        )
    ):
        raise ValueError("IoU thresholds must be between 0 and 1")
    path_a = Path(arm_a_path).resolve()
    path_b = Path(arm_b_path).resolve()
    arm_a = _read_jsonl(path_a)
    arm_b = _read_jsonl(path_b)
    if set(arm_a) != set(arm_b):
        raise ValueError("artifact row sets do not match")
    for row_id in sorted(arm_a):
        if _gt_signature(arm_a[row_id], row_id=row_id) != _gt_signature(arm_b[row_id], row_id=row_id):
            raise ValueError(f"GT mismatch for row_id {row_id!r}")
    arm_a, selected_row_ids = _select_rows(arm_a, include_row_ids=include_row_ids)
    arm_b, selected_row_ids_b = _select_rows(arm_b, include_row_ids=include_row_ids)
    if selected_row_ids != selected_row_ids_b:  # Defensive: row-set equality above should guarantee this.
        raise ValueError("selected artifact row sets do not match")
    summary_a = _summarize(
        arm_a, match_iou_threshold=match_iou_threshold, duplicate_iou_threshold=duplicate_iou_threshold,
        strict_annotation_iou_threshold=strict_annotation_iou_threshold,
        strict_prediction_iou_threshold=strict_prediction_iou_threshold,
    )
    summary_b = _summarize(
        arm_b, match_iou_threshold=match_iou_threshold, duplicate_iou_threshold=duplicate_iou_threshold,
        strict_annotation_iou_threshold=strict_annotation_iou_threshold,
        strict_prediction_iou_threshold=strict_prediction_iou_threshold,
    )
    delta: dict[str, Any] = {}
    for key in ("gt_count", "prediction_count", "unique_matched_gt_owners", "owner_coverage",
                "owner_false_negative_rate", "duplicate_candidate_count",
                "ambiguous_duplicate_candidate_count", "malformed_row_count",
                "strict_physical_owner_duplicate_candidate_count",
                "strict_physical_owner_ambiguous_attribution_count",
                "invalid_row_count", "dropped_row_count", "dropped_prediction_count",
                "invalid_prediction_count", "center_error_mean_px", "size_error_mean_px"):
        left, right = summary_a[key], summary_b[key]
        delta[key] = None if left is None or right is None else right - left
    delta["matched_iou_mean"] = (
        None if summary_a["matched_iou"]["mean"] is None or summary_b["matched_iou"]["mean"] is None
        else summary_b["matched_iou"]["mean"] - summary_a["matched_iou"]["mean"]
    )
    common_iou_deltas: list[float] = []
    common_center_deltas: list[float] = []
    common_size_deltas: list[float] = []
    arm_a_only_owner_count = 0
    arm_b_only_owner_count = 0
    arm_a_only_owner_refs: list[dict[str, Any]] = []
    arm_b_only_owner_refs: list[dict[str, Any]] = []
    for row_id in sorted(arm_a):
        geometry_a = _matched_owner_geometry(
            arm_a[row_id], row_id=row_id, match_iou_threshold=match_iou_threshold
        )
        geometry_b = _matched_owner_geometry(
            arm_b[row_id], row_id=row_id, match_iou_threshold=match_iou_threshold
        )
        common = set(geometry_a) & set(geometry_b)
        arm_a_only = set(geometry_a) - set(geometry_b)
        arm_b_only = set(geometry_b) - set(geometry_a)
        arm_a_only_owner_count += len(arm_a_only)
        arm_b_only_owner_count += len(arm_b_only)
        arm_a_only_owner_refs.extend(
            {"row_id": row_id, "owner_index": owner_index}
            for owner_index in sorted(arm_a_only)
        )
        arm_b_only_owner_refs.extend(
            {"row_id": row_id, "owner_index": owner_index}
            for owner_index in sorted(arm_b_only)
        )
        for owner_index in sorted(common):
            iou_a, center_a, size_a = geometry_a[owner_index]
            iou_b, center_b, size_b = geometry_b[owner_index]
            common_iou_deltas.append(iou_b - iou_a)
            common_center_deltas.append(center_b - center_a)
            common_size_deltas.append(size_b - size_a)
    return {
        "inputs": {
            "arm_a": {"path": str(path_a), "sha256": _sha256(path_a)},
            "arm_b": {"path": str(path_b), "sha256": _sha256(path_b)},
            "cohort": {
                "selection": "all_artifact_rows" if selected_row_ids is None else "explicit_include_row_ids",
                "included_row_ids": selected_row_ids,
            },
        },
        "policy": {
            "match_iou_threshold": match_iou_threshold,
            "duplicate_iou_threshold": duplicate_iou_threshold,
            "strict_annotation_iou_threshold": strict_annotation_iou_threshold,
            "strict_prediction_iou_threshold": strict_prediction_iou_threshold,
            "delta_convention": "arm_b_minus_arm_a",
            "unmatched_predictions_are_not_hallucinations": True,
            "duplicate_policy": (
                "count only extra predictions uniquely attributable to one same-category GT owner; "
                "report ambiguous candidates separately"
            ),
            "strict_physical_owner_duplicate_policy": (
                "geometry-derived candidate only, not human-confirmed: require exactly one same-category "
                "GT owner at the annotation IoU threshold and a prior attributed prediction for that owner "
                "at the prediction-to-prediction IoU threshold"
            ),
        },
        "arm_a": summary_a,
        "arm_b": summary_b,
        "paired_deltas": delta,
        "common_owner_geometry": {
            "owner_count": len(common_iou_deltas),
            "arm_a_only_owner_count": arm_a_only_owner_count,
            "arm_b_only_owner_count": arm_b_only_owner_count,
            "arm_a_only_owner_refs": arm_a_only_owner_refs,
            "arm_b_only_owner_refs": arm_b_only_owner_refs,
            "iou_delta": _distribution(common_iou_deltas),
            "center_error_delta_px": _distribution(common_center_deltas),
            "size_error_delta_px": _distribution(common_size_deltas),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.30)
    parser.add_argument("--strict-annotation-iou-threshold", type=float, default=0.50)
    parser.add_argument("--strict-prediction-iou-threshold", type=float, default=0.90)
    parser.add_argument("--include-row-id", action="append", default=[])
    parser.add_argument("--include-row-id-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    include_row_ids = _read_include_row_ids(args.include_row_id, args.include_row_id_file)
    result = compare_artifacts(args.arm_a, args.arm_b, match_iou_threshold=args.match_iou_threshold,
                               duplicate_iou_threshold=args.duplicate_iou_threshold,
                               strict_annotation_iou_threshold=args.strict_annotation_iou_threshold,
                               strict_prediction_iou_threshold=args.strict_prediction_iou_threshold,
                               include_row_ids=include_row_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
