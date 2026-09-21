#!/usr/bin/env python3
"""Corrected paired diagnostic for dynamic HF and materialized BF16 HF detections.

This file intentionally lives beside the research receipt.  It consumes the
authoritative raw/scored artifact families and does not alter inference code.
The pair matcher is class-aware and one-to-one: same-class candidates with
pixel IoU >= 0.50 are sorted by descending pixel IoU, then dynamic and
materialized prediction index, and greedily consumed.  Thus a changed output
sequence never causes a positional zip of boxes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


RAW_NAME = "gt_vs_pred.jsonl"
SCORED_NAME = "gt_vs_pred_scored.jsonl"
PROVENANCE_NAME = "gt_vs_pred_scored.jsonl.provenance.json"
TRACE_NAME = "pred_token_trace.jsonl"
ACCEPTED_PARSE_STATUSES = {"accepted", "accepted_with_drops"}
COORD_NAMES = ("x1", "y1", "x2", "y2")
PAIRED_IOU_THRESHOLD = 0.50
ANALYSIS_VERSION = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired coordinate/prediction consistency from two "
            "coordexp-infras inference artifact directories."
        )
    )
    parser.add_argument("--dynamic-dir", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort-json",
        type=Path,
        default=None,
        help="Frozen cohort receipt used to authenticate image, GT, and dimensions.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Frozen model input JSONL; defaults to cohort_json.input_jsonl.",
    )
    parser.add_argument(
        "--expected-row-count",
        type=int,
        default=None,
        help="Optional exact row count check; rows are never filtered on failure.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Worktree containing scripts/evaluate_detection.py (defaults to cwd).",
    )
    parser.add_argument(
        "--reuse-evaluation-root",
        type=Path,
        default=None,
        help=(
            "Existing evaluator root with dynamic/ and materialized/ metrics; "
            "reuse it without launching evaluate_detection.py."
        ),
    )
    return parser


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def _read_json(path: Path, *, errors: list[dict[str, Any]], code: str) -> Any:
    if not path.is_file():
        errors.append({"code": code, "path": str(path), "message": "file is missing"})
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append({"code": code, "path": str(path), "message": str(exc)})
        return None


def _read_jsonl(path: Path, *, errors: list[dict[str, Any]], code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append({"code": code, "path": str(path), "message": "file is missing"})
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append({"code": code, "path": str(path), "message": str(exc)})
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(
                {
                    "code": "jsonl_blank_line",
                    "path": str(path),
                    "line": line_number,
                    "message": "blank JSONL lines are not accepted",
                }
            )
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "code": "jsonl_decode",
                    "path": str(path),
                    "line": line_number,
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(value, dict):
            errors.append(
                {
                    "code": "jsonl_row_shape",
                    "path": str(path),
                    "line": line_number,
                    "message": "row must be an object",
                }
            )
            continue
        rows.append(value)
    return rows


def _row_id(row: dict[str, Any]) -> str | None:
    value = row.get("row_id")
    return value if isinstance(value, str) and value.strip() else None


def _rows_by_id(
    rows: list[dict[str, Any]], *, arm: str, artifact: str, errors: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_id = _row_id(row)
        if row_id is None:
            errors.append(
                {
                    "code": "row_id_missing",
                    "arm": arm,
                    "artifact": artifact,
                    "row_position": index,
                }
            )
            continue
        if row_id in result:
            errors.append(
                {
                    "code": "duplicate_row_id",
                    "arm": arm,
                    "artifact": artifact,
                    "row_id": row_id,
                }
            )
        result[row_id] = row
    return result


def _description(obj: dict[str, Any]) -> str:
    value = obj.get("description", obj.get("desc", obj.get("category_name", "")))
    return str(value or "").strip()


def _normalized_description(value: Any) -> str:
    from src.eval.detection_categories import normalize_coco_category_name

    return normalize_coco_category_name(value)


def _bbox_value(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = obj.get("bbox", obj.get("bbox_2d"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def _coord_bins(obj: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = obj.get("coord_bins")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    parsed = tuple(int(item) for item in value)
    if any(item < 0 or item > 999 for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def _prediction_signature(obj: Any) -> str:
    if not isinstance(obj, dict):
        return _canonical({"invalid": obj})
    # Raw rows retain parser span evidence while scored rows retain score
    # provenance.  Compare the immutable decoded object payload shared by both
    # artifacts; score/span sidecars are reported independently.
    value = {
        "description": _description(obj),
        "bbox": obj.get("bbox", obj.get("bbox_2d")),
        "coord_bins": obj.get("coord_bins"),
        "generated_order": obj.get("generated_order"),
        "object_span_id": obj.get("object_span_id"),
    }
    return _canonical(value)


def _semantic_prediction_signature(obj: Any) -> tuple[Any, ...]:
    if not isinstance(obj, dict):
        return ("<invalid>", _canonical(obj))
    bins = _coord_bins(obj)
    bbox = _bbox_value(obj)
    return (
        _normalized_description(_description(obj)),
        tuple(bins) if bins is not None else None,
        tuple(round(value, 8) for value in bbox) if bbox is not None else None,
    )


def _raw_scored_identity(
    raw: dict[str, Any], scored: dict[str, Any], *, arm: str, row_id: str, errors: list[dict[str, Any]]
) -> None:
    for field in ("row_id", "row_index", "example_id", "image_path", "image_width", "image_height"):
        if raw.get(field) != scored.get(field):
            errors.append(
                {
                    "code": "raw_scored_identity_mismatch",
                    "arm": arm,
                    "row_id": row_id,
                    "field": field,
                    "raw": raw.get(field),
                    "scored": scored.get(field),
                }
            )
    if _canonical(raw.get("gt", [])) != _canonical(scored.get("gt", [])):
        errors.append(
            {
                "code": "raw_scored_gt_mismatch",
                "arm": arm,
                "row_id": row_id,
            }
        )
    raw_pred = raw.get("pred", [])
    scored_pred = scored.get("pred", [])
    if not isinstance(raw_pred, list) or not isinstance(scored_pred, list):
        errors.append(
            {
                "code": "raw_scored_pred_shape",
                "arm": arm,
                "row_id": row_id,
                "raw_type": type(raw_pred).__name__,
                "scored_type": type(scored_pred).__name__,
            }
        )
        return
    raw_signatures = [_prediction_signature(item) for item in raw_pred]
    scored_signatures = [_prediction_signature(item) for item in scored_pred]
    if raw_signatures != scored_signatures:
        errors.append(
            {
                "code": "raw_scored_pred_mismatch",
                "arm": arm,
                "row_id": row_id,
                "raw_count": len(raw_pred),
                "scored_count": len(scored_pred),
            }
        )


def _load_arm(name: str, path: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    raw_path = path / RAW_NAME
    scored_path = path / SCORED_NAME
    raw_rows = _read_jsonl(raw_path, errors=errors, code="missing_or_invalid_raw")
    scored_rows = _read_jsonl(scored_path, errors=errors, code="missing_or_invalid_scored")
    raw_by_id = _rows_by_id(raw_rows, arm=name, artifact=RAW_NAME, errors=errors)
    scored_by_id = _rows_by_id(scored_rows, arm=name, artifact=SCORED_NAME, errors=errors)
    for row_id in sorted(set(raw_by_id).intersection(scored_by_id)):
        _raw_scored_identity(
            raw_by_id[row_id], scored_by_id[row_id], arm=name, row_id=row_id, errors=errors
        )
    trace_rows = _read_jsonl(path / TRACE_NAME, errors=errors, code="missing_or_invalid_trace")
    traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trace_rows:
        row_id = _row_id(item)
        if row_id is not None and item.get("trace_type") == "generated_token":
            traces_by_id[row_id].append(item)
    for row_id in traces_by_id:
        traces_by_id[row_id].sort(key=lambda item: int(item.get("generated_step_index", 0)))
    provenance = _read_json(
        path / PROVENANCE_NAME, errors=errors, code="missing_or_invalid_provenance"
    )
    raw_sha = _sha256_file(raw_path) if raw_path.is_file() else None
    scored_sha = _sha256_file(scored_path) if scored_path.is_file() else None
    sidecar_sha256 = {
        filename: _sha256_file(path / filename)
        if (path / filename).is_file()
        else None
        for filename in (PROVENANCE_NAME, TRACE_NAME, "run_manifest.json", "summary.json")
    }
    statuses = Counter(str(row.get("parse_status", "missing")) for row in raw_rows)
    stop_reasons = Counter(str(row.get("decode_stop_reason", "")) for row in raw_rows)
    raw_pred_count = sum(len(row.get("pred", [])) for row in raw_rows if isinstance(row.get("pred"), list))
    scored_pred_count = sum(
        len(row.get("pred", [])) for row in scored_rows if isinstance(row.get("pred"), list)
    )
    score_policy = provenance.get("score_policy_fingerprint") if isinstance(provenance, dict) else None
    return {
        "name": name,
        "path": path.as_posix(),
        "raw_rows": raw_rows,
        "scored_rows": scored_rows,
        "raw_by_id": raw_by_id,
        "scored_by_id": scored_by_id,
        "traces_by_id": dict(traces_by_id),
        "provenance": provenance,
        "errors": errors,
        "raw_sha256": raw_sha,
        "scored_sha256": scored_sha,
        "sidecar_sha256": sidecar_sha256,
        "row_count": len(raw_rows),
        "scored_row_count": len(scored_rows),
        "parse_status_counts": dict(sorted(statuses.items())),
        "decode_stop_reason_counts": dict(sorted(stop_reasons.items())),
        "parser_failure_count": sum(
            count for status, count in statuses.items() if status not in ACCEPTED_PARSE_STATUSES
        ),
        "truncated_decode_count": int(stop_reasons.get("length", 0)),
        "raw_prediction_count": raw_pred_count,
        "scored_prediction_count": scored_pred_count,
        "empty_prediction_row_count": sum(
            1 for row in scored_rows if isinstance(row.get("pred"), list) and not row["pred"]
        ),
        "dropped_prediction_count": sum(
            int(row.get("dropped_prediction_count", 0) or 0)
            for row in raw_rows
            if isinstance(row.get("dropped_prediction_count", 0), (int, float))
        ),
        "score_policy_fingerprint": score_policy,
    }


def _input_gt_signature(row: dict[str, Any]) -> list[tuple[str, tuple[int, int, int, int]]]:
    # Input rows use canonical coordinate tokens under bbox_2d.  Importing the
    # project parser here keeps the conversion identical to production.
    from src.data.geometry import parse_source_bbox_tokens

    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, obj in enumerate(row.get("objects", [])):
        if not isinstance(obj, dict):
            out.append(("<invalid>", (0, 0, 0, 0)))
            continue
        try:
            bins = parse_source_bbox_tokens(obj.get("bbox_2d"), field=f"objects[{index}].bbox_2d")
        except Exception:
            bins = (0, 0, 0, 0)
        out.append((_normalized_description(obj.get("desc", obj.get("category_name", ""))), bins))
    return out


def _artifact_gt_signature(row: dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
    out: list[tuple[str, tuple[Any, ...]]] = []
    value = row.get("gt", [])
    if not isinstance(value, list):
        return out
    for obj in value:
        if not isinstance(obj, dict):
            out.append(("<invalid>", ()))
            continue
        bbox = obj.get("bbox", obj.get("bbox_2d"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            parsed: tuple[Any, ...] = ()
        else:
            parsed = tuple(bbox)
        out.append((_normalized_description(_description(obj)), parsed))
    return out


def _expected_row_id(image_id: Any, image_path: str) -> str | None:
    if isinstance(image_id, bool) or not isinstance(image_id, int):
        return None
    split = "val" if "/val2017/" in image_path or "val2017" in image_path else "train"
    return f"coco2017_{split}_{image_id:012d}"


def _load_expected(
    *, cohort_path: Path | None, input_path: Path | None, errors: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    cohort: dict[str, Any] | None = None
    if cohort_path is not None:
        value = _read_json(cohort_path, errors=errors, code="missing_or_invalid_cohort")
        if isinstance(value, dict):
            cohort = value
    if input_path is None and isinstance(cohort, dict) and isinstance(cohort.get("input_jsonl"), str):
        input_path = Path(cohort["input_jsonl"])
    if input_path is None:
        return {}
    input_rows = _read_jsonl(input_path, errors=errors, code="missing_or_invalid_input")
    if isinstance(cohort, dict):
        expected_sha = cohort.get("input_sha256")
        observed_sha = _sha256_file(input_path) if input_path.is_file() else None
        if expected_sha and observed_sha != expected_sha:
            errors.append(
                {
                    "code": "input_sha256_mismatch",
                    "path": str(input_path),
                    "expected": expected_sha,
                    "observed": observed_sha,
                }
            )
        if cohort.get("count") is not None and cohort.get("count") != len(input_rows):
            errors.append(
                {
                    "code": "cohort_input_count_mismatch",
                    "expected": cohort.get("count"),
                    "observed": len(input_rows),
                }
            )
    expected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(input_rows):
        images = row.get("images")
        image_path_value = images[0] if isinstance(images, list) and images else row.get("image")
        image_path = str(image_path_value or "")
        image_id = row.get("image_id")
        row_id = _expected_row_id(image_id, image_path)
        if row_id is None:
            errors.append({"code": "input_row_identity_missing", "row_position": index})
            continue
        expected[row_id] = {
            "row_id": row_id,
            "row_index": index,
            "image_id": image_id,
            "image_path": (input_path.parent / image_path).resolve().as_posix()
            if image_path
            else image_path,
            "image_width": row.get("width"),
            "image_height": row.get("height"),
            "gt": _input_gt_signature(row),
        }
    return expected


def _artifact_identity(row: dict[str, Any]) -> dict[str, Any]:
    path = row.get("image_path")
    return {
        "row_id": row.get("row_id"),
        "row_index": row.get("row_index"),
        "image_path": Path(path).resolve().as_posix() if isinstance(path, str) else path,
        "image_width": row.get("image_width"),
        "image_height": row.get("image_height"),
        "gt": _artifact_gt_signature(row),
    }


def _authenticate(
    dynamic: dict[str, Any],
    materialized: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    *,
    expected_row_count: int | None,
) -> tuple[list[str], dict[str, Any]]:
    errors = list(dynamic["errors"]) + list(materialized["errors"])
    dynamic_ids = set(dynamic["raw_by_id"])
    materialized_ids = set(materialized["raw_by_id"])
    if dynamic_ids != materialized_ids:
        errors.append(
            {
                "code": "paired_row_id_set_mismatch",
                "dynamic_only": sorted(dynamic_ids - materialized_ids),
                "materialized_only": sorted(materialized_ids - dynamic_ids),
            }
        )
    if expected_row_count is not None:
        for arm in (dynamic, materialized):
            if arm["row_count"] != expected_row_count:
                errors.append(
                    {
                        "code": "expected_row_count_mismatch",
                        "arm": arm["name"],
                        "expected": expected_row_count,
                        "observed": arm["row_count"],
                    }
                )
    if expected:
        for arm in (dynamic, materialized):
            observed_ids = set(arm["raw_by_id"])
            expected_ids = set(expected)
            if observed_ids != expected_ids:
                errors.append(
                    {
                        "code": "cohort_row_id_set_mismatch",
                        "arm": arm["name"],
                        "missing": sorted(expected_ids - observed_ids),
                        "unexpected": sorted(observed_ids - expected_ids),
                    }
                )
            for row_id in sorted(observed_ids.intersection(expected_ids)):
                observed = _artifact_identity(arm["raw_by_id"][row_id])
                target = expected[row_id]
                for field in ("image_path", "image_width", "image_height"):
                    if observed[field] != target[field]:
                        errors.append(
                            {
                                "code": "cohort_identity_mismatch",
                                "arm": arm["name"],
                                "row_id": row_id,
                                "field": field,
                                "observed": observed[field],
                                "expected": target[field],
                            }
                        )
                if observed["gt"] != target["gt"]:
                    errors.append(
                        {
                            "code": "cohort_gt_mismatch",
                            "arm": arm["name"],
                            "row_id": row_id,
                        }
                    )
    for row_id in sorted(dynamic_ids.intersection(materialized_ids)):
        left = _artifact_identity(dynamic["raw_by_id"][row_id])
        right = _artifact_identity(materialized["raw_by_id"][row_id])
        for field in ("image_path", "image_width", "image_height", "gt"):
            if left[field] != right[field]:
                errors.append(
                    {
                        "code": "paired_immutable_identity_mismatch",
                        "row_id": row_id,
                        "field": field,
                        "dynamic": left[field],
                        "materialized": right[field],
                    }
                )
    left_policy = dynamic.get("score_policy_fingerprint")
    right_policy = materialized.get("score_policy_fingerprint")
    if left_policy and right_policy and left_policy != right_policy:
        errors.append(
            {
                "code": "score_policy_fingerprint_mismatch",
                "dynamic": left_policy,
                "materialized": right_policy,
            }
        )
    paired_ids = sorted(dynamic_ids.intersection(materialized_ids))
    auth = {
        "status": "passed" if not errors else "failed",
        "paired_row_count": len(paired_ids),
        "dynamic_row_count": dynamic["row_count"],
        "materialized_row_count": materialized["row_count"],
        "expected_row_count": expected_row_count,
        "cohort_bound": bool(expected),
        "errors": errors,
    }
    return paired_ids, auth


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    parsed = [float(value) for value in values if math.isfinite(float(value))]
    if not parsed:
        return {"count": 0}
    return {
        "count": len(parsed),
        "mean": sum(parsed) / len(parsed),
        "min": min(parsed),
        "p50": _percentile(parsed, 50),
        "p90": _percentile(parsed, 90),
        "p95": _percentile(parsed, 95),
        "p99": _percentile(parsed, 99),
        "max": max(parsed),
    }


def _fraction_at_most(values: list[float], thresholds: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    denominator = len(values)
    return {
        "denominator": denominator,
        **{
            f"at_most_{threshold}": {
                "count": sum(value <= threshold for value in values),
                "fraction": (
                    sum(value <= threshold for value in values) / denominator
                    if denominator
                    else None
                ),
            }
            for threshold in thresholds
        },
    }


def _text_diff(left: Any, right: Any) -> dict[str, Any]:
    left_text = left if isinstance(left, str) else ""
    right_text = right if isinstance(right, str) else ""
    prefix = 0
    for a, b in zip(left_text, right_text):
        if a != b:
            break
        prefix += 1
    suffix = 0
    max_suffix = min(len(left_text) - prefix, len(right_text) - prefix)
    while suffix < max_suffix and left_text[-suffix - 1] == right_text[-suffix - 1]:
        suffix += 1
    return {
        "exact_equal": left_text == right_text,
        "dynamic_available": isinstance(left, str),
        "materialized_available": isinstance(right, str),
        "dynamic_length": len(left_text),
        "materialized_length": len(right_text),
        "length_delta": len(right_text) - len(left_text),
        "dynamic_sha256": hashlib.sha256(left_text.encode()).hexdigest(),
        "materialized_sha256": hashlib.sha256(right_text.encode()).hexdigest(),
        "common_prefix_length": prefix,
        "common_suffix_length": suffix,
    }


def _sequence_diff(left: list[Any], right: list[Any]) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    opcodes = matcher.get_opcodes()
    edit_distance = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in opcodes
        if tag != "equal"
    )
    return {
        "exact_equal": left == right,
        "dynamic_length": len(left),
        "materialized_length": len(right),
        "common_prefix_length": next(
            (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right))
        ),
        "edit_distance": edit_distance,
        "opcode_counts": dict(Counter(tag for tag, *_ in opcodes)),
    }


def _trace_signature(items: list[dict[str, Any]] | None) -> list[tuple[Any, ...]] | None:
    if items is None:
        return None
    return [
        (
            item.get("token_id"),
            item.get("token_text"),
            bool(item.get("is_stop", False)),
            bool(item.get("is_pad", False)),
        )
        for item in items
    ]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    # Reuse the consumer's closed-open pixel xyxy IoU implementation.
    from src.vis.matching import iou_xyxy

    return float(iou_xyxy(a, b))


def _valid_pixel_box(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    bbox = _bbox_value(obj)
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _pair_predictions(
    dynamic_pred: list[Any], materialized_pred: list[Any]
) -> dict[str, Any]:
    left = list(dynamic_pred)
    right = list(materialized_pred)
    left_valid_indices = [
        index for index, item in enumerate(left) if _valid_pixel_box(item)
    ]
    right_valid_indices = [
        index for index, item in enumerate(right) if _valid_pixel_box(item)
    ]
    left_valid = set(left_valid_indices)
    right_valid = set(right_valid_indices)
    candidates: list[tuple[float, int, int]] = []
    for left_index in left_valid_indices:
        left_obj = left[left_index]
        left_class = _normalized_description(_description(left_obj))
        left_box = _bbox_value(left_obj)
        if left_box is None:
            continue
        for right_index in right_valid_indices:
            right_obj = right[right_index]
            if left_class != _normalized_description(_description(right_obj)):
                continue
            right_box = _bbox_value(right_obj)
            if right_box is None:
                continue
            value = _iou(left_box, right_box)
            if value < PAIRED_IOU_THRESHOLD:
                continue
            candidates.append((value, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for value, left_index, right_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        left_obj = left[left_index]
        right_obj = right[right_index]
        matches.append(
            {
                "dynamic_index": left_index,
                "materialized_index": right_index,
                "class": _normalized_description(_description(left_obj)),
                "iou": value,
                "dynamic_coord_bins": list(_coord_bins(left_obj) or ()),
                "materialized_coord_bins": list(_coord_bins(right_obj) or ()),
                "dynamic_score": left_obj.get("score"),
                "materialized_score": right_obj.get("score"),
            }
        )
    unmatched_left = [index for index in left_valid_indices if index not in used_left]
    unmatched_right = [index for index in right_valid_indices if index not in used_right]
    class_counts_left = Counter(
        _normalized_description(_description(item)) for item in left if isinstance(item, dict)
    )
    class_counts_right = Counter(
        _normalized_description(_description(item)) for item in right if isinstance(item, dict)
    )
    class_delta = {
        key: class_counts_right.get(key, 0) - class_counts_left.get(key, 0)
        for key in sorted(set(class_counts_left) | set(class_counts_right))
        if class_counts_right.get(key, 0) != class_counts_left.get(key, 0)
    }
    return {
        "matching_algorithm": (
            "same normalized class and pixel IoU >= 0.50; candidates sorted by "
            "(-IoU, dynamic_index, materialized_index); greedy one-to-one"
        ),
        "dynamic_valid_object_count": len(left_valid_indices),
        "materialized_valid_object_count": len(right_valid_indices),
        "dynamic_invalid_object_count": len(dynamic_pred) - len(left_valid_indices),
        "materialized_invalid_object_count": len(materialized_pred) - len(right_valid_indices),
        "dynamic_invalid_object_indices": [index for index in range(len(dynamic_pred)) if index not in left_valid],
        "materialized_invalid_object_indices": [index for index in range(len(materialized_pred)) if index not in right_valid],
        "matched_count": len(matches),
        "unmatched_dynamic_indices": unmatched_left,
        "unmatched_materialized_indices": unmatched_right,
        "unmatched_dynamic_count": len(unmatched_left),
        "unmatched_materialized_count": len(unmatched_right),
        "class_counts_dynamic": dict(sorted(class_counts_left.items())),
        "class_counts_materialized": dict(sorted(class_counts_right.items())),
        "class_count_delta_materialized_minus_dynamic": class_delta,
        "matches": matches,
    }


def _matcher_sensitivity_checks() -> dict[str, Any]:
    """Exercise the threshold and sequence-order invariants deterministically."""
    reorder_dynamic = [
        {"description": "person", "bbox": [0, 0, 10, 10]},
        {"description": "dog", "bbox": [100, 100, 110, 110]},
    ]
    reorder_materialized = [
        {"description": "dog", "bbox": [100, 100, 110, 110]},
        {"description": "person", "bbox": [0, 0, 10, 10]},
    ]
    below_threshold_dynamic = [
        {"description": "person", "bbox": [0, 0, 10, 10]},
    ]
    below_threshold_materialized = [
        {"description": "person", "bbox": [100, 100, 110, 110]},
    ]
    cases = [
        {
            "name": "reordered_same_class_boxes",
            "dynamic": reorder_dynamic,
            "materialized": reorder_materialized,
            "expected_pairs": [(0, 1), (1, 0)],
            "expected_unmatched": (0, 0),
        },
        {
            "name": "same_class_below_iou50_is_unmatched",
            "dynamic": below_threshold_dynamic,
            "materialized": below_threshold_materialized,
            "expected_pairs": [],
            "expected_unmatched": (1, 1),
        },
    ]
    checks: list[dict[str, Any]] = []
    for case in cases:
        result = _pair_predictions(case["dynamic"], case["materialized"])
        observed_pairs = [
            (item["dynamic_index"], item["materialized_index"])
            for item in result["matches"]
        ]
        passed = (
            observed_pairs == case["expected_pairs"]
            and result["unmatched_dynamic_count"] == case["expected_unmatched"][0]
            and result["unmatched_materialized_count"] == case["expected_unmatched"][1]
        )
        checks.append(
            {
                "name": case["name"],
                "status": "passed" if passed else "failed",
                "matched_index_pairs": observed_pairs,
                "unmatched_dynamic_count": result["unmatched_dynamic_count"],
                "unmatched_materialized_count": result["unmatched_materialized_count"],
            }
        )
    return {
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "paired_iou_threshold": PAIRED_IOU_THRESHOLD,
        "checks": checks,
    }


def _bin_drift(matches: list[dict[str, Any]]) -> dict[str, Any]:
    drifts: dict[str, list[float]] = {name: [] for name in COORD_NAMES}
    max_abs_per_box: list[float] = []
    missing_dynamic = 0
    missing_materialized = 0
    for match in matches:
        left = match.get("dynamic_coord_bins") or []
        right = match.get("materialized_coord_bins") or []
        if len(left) != 4:
            missing_dynamic += 1
        if len(right) != 4:
            missing_materialized += 1
        if len(left) != 4 or len(right) != 4:
            continue
        max_abs_per_box.append(max(abs(right[index] - left[index]) for index in range(4)))
        for index, name in enumerate(COORD_NAMES):
            # Signed convention: materialized minus dynamic.
            drifts[name].append(float(right[index] - left[index]))
    return {
        "signed_convention": "materialized_minus_dynamic",
        "matched_pair_count": len(matches),
        "missing_dynamic_coord_bins_match_count": missing_dynamic,
        "missing_materialized_coord_bins_match_count": missing_materialized,
        "per_coordinate": {
            name: {
                "signed": _distribution(values),
                "absolute": _distribution(abs(value) for value in values),
            }
            for name, values in drifts.items()
        },
        "per_box_max_abs_delta": {
            "distribution": _distribution(max_abs_per_box),
            "fraction_at_most": _fraction_at_most(max_abs_per_box),
        },
    }


def _score_summary(pred: list[Any]) -> dict[str, Any]:
    scores = [
        float(item["score"])
        for item in pred
        if isinstance(item, dict)
        and isinstance(item.get("score"), (int, float))
        and not isinstance(item.get("score"), bool)
        and math.isfinite(float(item["score"]))
    ]
    return {
        "count": len(scores),
        "values": scores,
        "distribution": _distribution(scores),
        "descending_index_order": [
            index
            for index, item in sorted(
                enumerate(pred),
                key=lambda pair: (
                    -float(pair[1].get("score"))
                    if isinstance(pair[1], dict)
                    and isinstance(pair[1].get("score"), (int, float))
                    and not isinstance(pair[1].get("score"), bool)
                    and math.isfinite(float(pair[1]["score"]))
                    else math.inf,
                    pair[0],
                ),
            )
            if isinstance(item, dict)
            and isinstance(item.get("score"), (int, float))
            and not isinstance(item.get("score"), bool)
            and math.isfinite(float(item["score"]))
        ],
    }


def _paired_score_stats(matches: list[dict[str, Any]]) -> dict[str, Any]:
    deltas: list[float] = []
    for match in matches:
        left = match.get("dynamic_score")
        right = match.get("materialized_score")
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and math.isfinite(float(left))
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(float(right))
        ):
            delta = float(right) - float(left)
            deltas.append(delta)
    return {
        "matched_score_count": len(deltas),
        "delta_materialized_minus_dynamic": _distribution(deltas),
    }


def _gt_metrics(row: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    # This is the existing visualization consumer policy: same normalized class,
    # IoU threshold, descending IoU greedy one-to-one matching.  The consumer's
    # COCO conversion drops unknown classes and malformed prediction boxes; the
    # counts below expose those drops rather than filtering rows.
    from src.data.geometry import coord_bins_to_pixel_xyxy
    from src.eval.detection_categories import COCO_80_CATEGORY_IDS

    width = row.get("image_width")
    height = row.get("image_height")
    gt: list[tuple[str, tuple[float, float, float, float]]] = []
    invalid_gt = 0
    if isinstance(width, int) and not isinstance(width, bool) and isinstance(height, int) and not isinstance(height, bool):
        for obj in row.get("gt", []):
            if not isinstance(obj, dict):
                invalid_gt += 1
                continue
            try:
                bbox = coord_bins_to_pixel_xyxy(
                    obj.get("bbox", obj.get("bbox_2d")),
                    image_width=width,
                    image_height=height,
                    field="gt.bbox",
                )
            except Exception:
                invalid_gt += 1
                continue
            gt.append((_normalized_description(_description(obj)), tuple(float(v) for v in bbox)))
    else:
        invalid_gt = len(row.get("gt", [])) if isinstance(row.get("gt"), list) else 0
    pred: list[tuple[int, str, tuple[float, float, float, float]]] = []
    unknown_category = 0
    invalid_pred = 0
    raw_pred = row.get("pred", [])
    if not isinstance(raw_pred, list):
        raw_pred = []
    for index, obj in enumerate(raw_pred):
        if not isinstance(obj, dict):
            invalid_pred += 1
            continue
        category = _normalized_description(_description(obj))
        if category not in COCO_80_CATEGORY_IDS:
            unknown_category += 1
            continue
        bbox = _bbox_value(obj)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            invalid_pred += 1
            continue
        pred.append((index, category, bbox))
    candidates: list[tuple[float, int, int]] = []
    for gt_index, (gt_class, gt_box) in enumerate(gt):
        for pred_position, (_, pred_class, pred_box) in enumerate(pred):
            if gt_class != pred_class:
                continue
            value = _iou(gt_box, pred_box)
            if value >= threshold:
                candidates.append((value, gt_index, pred_position))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for value, gt_index, pred_position in sorted(
        candidates, key=lambda item: (-item[0], item[1], pred[item[2]][0])
    ):
        if gt_index in used_gt or pred_position in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_position)
        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred[pred_position][0],
                "class": gt[gt_index][0],
                "iou": value,
            }
        )
    return {
        "threshold": threshold,
        "matching_policy": "same normalized COCO class; IoU descending greedy one-to-one",
        "gt_count": len(gt),
        "prediction_count_input": len(raw_pred),
        "prediction_count_consumer_valid": len(pred),
        "unknown_category_pred_count": unknown_category,
        "invalid_pred_bbox_count": invalid_pred,
        "invalid_gt_count": invalid_gt,
        "tp": len(matches),
        "fp": len(pred) - len(matches),
        "fn": len(gt) - len(matches),
        "matches": matches,
    }


def _row_metric(
    row_id: str,
    dynamic: dict[str, Any],
    materialized: dict[str, Any],
) -> dict[str, Any]:
    dynamic_raw = dynamic["raw_by_id"][row_id]
    materialized_raw = materialized["raw_by_id"][row_id]
    dynamic_scored = dynamic["scored_by_id"].get(row_id, {})
    materialized_scored = materialized["scored_by_id"].get(row_id, {})
    dynamic_raw_pred = dynamic_raw.get("pred", [])
    materialized_raw_pred = materialized_raw.get("pred", [])
    dynamic_scored_pred = dynamic_scored.get("pred", [])
    materialized_scored_pred = materialized_scored.get("pred", [])
    if not isinstance(dynamic_raw_pred, list):
        dynamic_raw_pred = []
    if not isinstance(materialized_raw_pred, list):
        materialized_raw_pred = []
    if not isinstance(dynamic_scored_pred, list):
        dynamic_scored_pred = []
    if not isinstance(materialized_scored_pred, list):
        materialized_scored_pred = []
    dynamic_semantic = [_semantic_prediction_signature(item) for item in dynamic_raw_pred]
    materialized_semantic = [_semantic_prediction_signature(item) for item in materialized_raw_pred]
    pair = _pair_predictions(dynamic_scored_pred, materialized_scored_pred)
    return {
        "row_id": row_id,
        "row_index_dynamic": dynamic_raw.get("row_index"),
        "row_index_materialized": materialized_raw.get("row_index"),
        "image_path": dynamic_raw.get("image_path"),
        "image_width": dynamic_raw.get("image_width"),
        "image_height": dynamic_raw.get("image_height"),
        "gt_count": len(dynamic_raw.get("gt", [])) if isinstance(dynamic_raw.get("gt"), list) else None,
        "parser": {
            "dynamic": {
                "parse_status": dynamic_raw.get("parse_status"),
                "metric_bearing": dynamic_raw.get("metric_bearing"),
                "valid_prediction_count": dynamic_raw.get("valid_prediction_count"),
                "dropped_prediction_count": dynamic_raw.get("dropped_prediction_count"),
                "decode_stop_reason": dynamic_raw.get("decode_stop_reason"),
            },
            "materialized": {
                "parse_status": materialized_raw.get("parse_status"),
                "metric_bearing": materialized_raw.get("metric_bearing"),
                "valid_prediction_count": materialized_raw.get("valid_prediction_count"),
                "dropped_prediction_count": materialized_raw.get("dropped_prediction_count"),
                "decode_stop_reason": materialized_raw.get("decode_stop_reason"),
            },
        },
        "raw_prediction_count": {
            "dynamic": len(dynamic_raw_pred),
            "materialized": len(materialized_raw_pred),
            "delta_materialized_minus_dynamic": len(materialized_raw_pred) - len(dynamic_raw_pred),
        },
        "text": _text_diff(
            dynamic_raw.get("raw_decode_text"), materialized_raw.get("raw_decode_text")
        ),
        "object_sequence": _sequence_diff(dynamic_semantic, materialized_semantic),
        "trace_sequence": {
            "available_dynamic": row_id in dynamic["traces_by_id"],
            "available_materialized": row_id in materialized["traces_by_id"],
            "comparison": _sequence_diff(
                _trace_signature(dynamic["traces_by_id"].get(row_id)) or [],
                _trace_signature(materialized["traces_by_id"].get(row_id)) or [],
            )
            if row_id in dynamic["traces_by_id"] and row_id in materialized["traces_by_id"]
            else None,
        },
        "scores": {
            "dynamic": _score_summary(dynamic_scored_pred),
            "materialized": _score_summary(materialized_scored_pred),
            "paired": _paired_score_stats(pair["matches"]),
        },
        "paired_prediction_boxes": {
            **{key: value for key, value in pair.items() if key != "matches"},
            "matched_iou": _distribution(match["iou"] for match in pair["matches"]),
            "bin_drift": _bin_drift(pair["matches"]),
            "matches": pair["matches"],
        },
        "gt_correctness": {
            "dynamic": {
                "iou_0.50": _gt_metrics(dynamic_scored, threshold=0.50),
                "iou_0.75": _gt_metrics(dynamic_scored, threshold=0.75),
            },
            "materialized": {
                "iou_0.50": _gt_metrics(materialized_scored, threshold=0.50),
                "iou_0.75": _gt_metrics(materialized_scored, threshold=0.75),
            },
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row["paired_prediction_boxes"] for row in rows]
    matches = [match for item in paired for match in item.get("matches", [])]
    all_bin_signed: dict[str, list[float]] = {name: [] for name in COORD_NAMES}
    all_bin_absolute: dict[str, list[float]] = {name: [] for name in COORD_NAMES}
    all_box_max_abs: list[float] = []
    for item in paired:
        # Reconstruct from each matched pair so aggregate percentiles remain
        # exact and are not computed from rounded row-level summaries.
        for match in item.get("matches", []):
            left = match.get("dynamic_coord_bins") or []
            right = match.get("materialized_coord_bins") or []
            if len(left) == 4 and len(right) == 4:
                all_box_max_abs.append(max(abs(right[index] - left[index]) for index in range(4)))
                for index, name in enumerate(COORD_NAMES):
                    signed = float(right[index] - left[index])
                    all_bin_signed[name].append(signed)
                    all_bin_absolute[name].append(abs(signed))
    class_delta: Counter[str] = Counter()
    for item in paired:
        class_delta.update(item.get("class_count_delta_materialized_minus_dynamic", {}))
    parser_fields = ("parse_status", "decode_stop_reason")
    parser_differences = {
        field: sum(
            row["parser"]["dynamic"].get(field) != row["parser"]["materialized"].get(field)
            for row in rows
        )
        for field in parser_fields
    }
    count_difference_fields = (
        "raw_prediction_count",
        "valid_prediction_count",
        "dropped_prediction_count",
        "scored_prediction_count",
        "metric_bearing",
    )
    count_differences = {
        field: sum(
            (
                row["raw_prediction_count"]["dynamic"]
                != row["raw_prediction_count"]["materialized"]
                if field == "raw_prediction_count"
                else row["parser"]["dynamic"].get(field)
                != row["parser"]["materialized"].get(field)
            )
            for row in rows
        )
        for field in count_difference_fields
    }
    text_length_deltas = [row["text"]["length_delta"] for row in rows]
    sequence_length_deltas = [
        row["object_sequence"]["materialized_length"]
        - row["object_sequence"]["dynamic_length"]
        for row in rows
    ]
    native_dynamic_scores = [
        score
        for row in rows
        for score in row["scores"]["dynamic"].get("values", [])
    ]
    native_materialized_scores = [
        score
        for row in rows
        for score in row["scores"]["materialized"].get("values", [])
    ]
    paired_score_deltas = [
        float(match["materialized_score"]) - float(match["dynamic_score"])
        for match in matches
        if isinstance(match.get("dynamic_score"), (int, float))
        and not isinstance(match.get("dynamic_score"), bool)
        and math.isfinite(float(match["dynamic_score"]))
        and isinstance(match.get("materialized_score"), (int, float))
        and not isinstance(match.get("materialized_score"), bool)
        and math.isfinite(float(match["materialized_score"]))
    ]
    gt: dict[str, dict[str, int]] = {}
    for threshold in ("iou_0.50", "iou_0.75"):
        gt[threshold] = {}
        for arm in ("dynamic", "materialized"):
            counts = Counter()
            for row in rows:
                metric = row["gt_correctness"][arm][threshold]
                for key in ("tp", "fp", "fn", "unknown_category_pred_count", "invalid_pred_bbox_count"):
                    counts[key] += int(metric.get(key, 0))
            gt[threshold][arm] = dict(counts)
    return {
        "row_count": len(rows),
        "text_exact_equal_row_count": sum(row["text"]["exact_equal"] for row in rows),
        "object_sequence_exact_equal_row_count": sum(
            row["object_sequence"]["exact_equal"] for row in rows
        ),
        "trace_sequence_exact_equal_row_count": sum(
            bool(row["trace_sequence"]["comparison"] and row["trace_sequence"]["comparison"]["exact_equal"])
            for row in rows
        ),
        "parser_difference_row_counts": parser_differences,
        "parser_count_difference_row_counts": count_differences,
        "text_changed_row_count": sum(not row["text"]["exact_equal"] for row in rows),
        "text_length_delta_distribution": _distribution(text_length_deltas),
        "object_sequence_length_delta_distribution": _distribution(sequence_length_deltas),
        "object_sequence_edit_distance_distribution": _distribution(
            row["object_sequence"].get("edit_distance", 0) for row in rows
        ),
        "raw_prediction_count_dynamic": sum(row["raw_prediction_count"]["dynamic"] for row in rows),
        "raw_prediction_count_materialized": sum(
            row["raw_prediction_count"]["materialized"] for row in rows
        ),
        "paired_prediction_match_count": len(matches),
        "paired_iou_distribution": _distribution(match["iou"] for match in matches),
        "unmatched_dynamic_count": sum(
            item.get("unmatched_dynamic_count", 0) for item in paired
        ),
        "unmatched_materialized_count": sum(
            item.get("unmatched_materialized_count", 0) for item in paired
        ),
        "class_count_delta_materialized_minus_dynamic": dict(sorted(class_delta.items())),
        "bin_drift": {
            "signed_convention": "materialized_minus_dynamic",
            "pooled_absolute": _distribution(
                value for name in COORD_NAMES for value in all_bin_absolute[name]
            ),
            "per_coordinate": {
                name: {
                    "signed": _distribution(all_bin_signed[name]),
                    "absolute": _distribution(all_bin_absolute[name]),
                }
                for name in COORD_NAMES
            },
            "per_box_max_abs_delta": {
                "distribution": _distribution(all_box_max_abs),
                "fraction_at_most": _fraction_at_most(all_box_max_abs),
            },
        },
        "native_scores": {
            "dynamic": _distribution(native_dynamic_scores),
            "materialized": _distribution(native_materialized_scores),
            "paired_delta_materialized_minus_dynamic": _distribution(paired_score_deltas),
        },
        "gt_correctness": gt,
    }


def _worst_rows(rows: list[dict[str, Any]], *, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    def row_ref(row: dict[str, Any]) -> dict[str, Any]:
        pair = row["paired_prediction_boxes"]
        return {
            "row_id": row["row_id"],
            "image_path": row["image_path"],
            "matched_count": pair.get("matched_count", 0),
            "matched_iou_mean": pair.get("matched_iou", {}).get("mean"),
            "unmatched_dynamic_count": pair.get("unmatched_dynamic_count", 0),
            "unmatched_materialized_count": pair.get("unmatched_materialized_count", 0),
            "object_sequence_edit_distance": row["object_sequence"].get("edit_distance", 0),
            "text_exact_equal": row["text"].get("exact_equal", False),
            "max_abs_bin_drift": max(
                (
                    pair.get("bin_drift", {})
                    .get("per_coordinate", {})
                    .get(name, {})
                    .get("absolute", {})
                    .get("max", 0.0)
                    or 0.0
                )
                for name in COORD_NAMES
            ),
        }

    refs = [row_ref(row) for row in rows]
    return {
        "lowest_matched_iou": sorted(
            refs,
            key=lambda item: (
                item["matched_iou_mean"] is None,
                item["matched_iou_mean"] if item["matched_iou_mean"] is not None else math.inf,
                item["row_id"],
            ),
        )[:limit],
        "largest_absolute_bin_drift": sorted(
            refs, key=lambda item: (-item["max_abs_bin_drift"], item["row_id"])
        )[:limit],
        "largest_structural_difference": sorted(
            refs,
            key=lambda item: (
                -(
                    item["unmatched_dynamic_count"]
                    + item["unmatched_materialized_count"]
                    + item["object_sequence_edit_distance"]
                    + (0 if item["text_exact_equal"] else 1)
                ),
                item["row_id"],
            ),
        )[:limit],
    }


def _run_evaluator(
    *, label: str, run_dir: Path, output_dir: Path, repo_root: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "evaluator.stdout.log"
    stderr_path = output_dir / "evaluator.stderr.log"
    metrics_path = output_dir / "metrics.json"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_detection.py"),
        "--artifact-dir",
        str(run_dir),
        "--out-dir",
        str(output_dir),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        metrics = None
        if result.returncode == 0 and metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "label": label,
                    "status": "failed",
                    "returncode": result.returncode,
                    "command": command,
                    "stdout_path": stdout_path.as_posix(),
                    "stderr_path": stderr_path.as_posix(),
                    "metrics_path": metrics_path.as_posix(),
                    "error": {"code": "metrics_read", "message": str(exc)},
                }
        return {
            "label": label,
            "status": "passed" if result.returncode == 0 and metrics is not None else "failed",
            "returncode": result.returncode,
            "command": command,
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "metrics": metrics,
        }
    except OSError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
        return {
            "label": label,
            "status": "failed",
            "returncode": None,
            "command": command,
            "stdout_path": stdout_path.as_posix(),
            "stderr_path": stderr_path.as_posix(),
            "metrics_path": metrics_path.as_posix(),
            "error": {"code": "evaluator_launch", "message": str(exc)},
        }


def _reuse_evaluator(
    *, label: str, source_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Copy an already completed consumer evaluation into this receipt root."""
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_source = source_dir / "metrics.json"
    if not metrics_source.is_file():
        return {
            "label": label,
            "status": "failed",
            "returncode": None,
            "source_dir": source_dir.as_posix(),
            "error": {"code": "reused_metrics_missing", "path": metrics_source.as_posix()},
        }
    try:
        metrics = json.loads(metrics_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "label": label,
            "status": "failed",
            "returncode": None,
            "source_dir": source_dir.as_posix(),
            "error": {"code": "reused_metrics_read", "message": str(exc)},
        }
    copied: list[str] = []
    for filename in (
        "metrics.json",
        "evaluation_receipt.json",
        "coco_gt.json",
        "coco_predictions.json",
        "command.log",
    ):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)
            copied.append(filename)
    return {
        "label": label,
        "status": "reused",
        "returncode": 0,
        "source_dir": source_dir.as_posix(),
        "source_metrics_path": metrics_source.as_posix(),
        "source_evaluation_receipt_path": (source_dir / "evaluation_receipt.json").as_posix(),
        "copied_files": copied,
        "metrics_path": (output_dir / "metrics.json").as_posix(),
        "metrics": metrics,
    }


def _find_repo_root(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "scripts" / "evaluate_detection.py").is_file():
        return cwd
    for parent in cwd.parents:
        if (parent / "scripts" / "evaluate_detection.py").is_file():
            return parent
    raise RuntimeError("cannot locate worktree containing scripts/evaluate_detection.py")


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    repo_root = _find_repo_root(args.repo_root)
    sys.path.insert(0, str(repo_root))

    # Importing source modules after adding the requested worktree makes the
    # conversion/matching policy explicit in the receipt and avoids depending
    # on the output tree's ancestry.
    try:
        from src.eval.detection_categories import COCO_80_CATEGORY_IDS  # noqa: F401
    except Exception as exc:
        errors.append({"code": "repo_import", "message": str(exc)})

    matcher_sensitivity = _matcher_sensitivity_checks()
    if matcher_sensitivity["status"] != "passed":
        errors.append({"code": "matcher_sensitivity", "details": matcher_sensitivity})

    cohort_input = args.input_jsonl
    expected = _load_expected(
        cohort_path=args.cohort_json,
        input_path=cohort_input,
        errors=errors,
    )
    dynamic = _load_arm("dynamic_hf", args.dynamic_dir.resolve())
    materialized = _load_arm("materialized_bf16_hf", args.materialized_dir.resolve())
    paired_ids, authentication = _authenticate(
        dynamic,
        materialized,
        expected,
        expected_row_count=args.expected_row_count,
    )
    if errors:
        authentication["errors"].extend(errors)
        authentication["status"] = "failed"
    row_metrics = [
        _row_metric(row_id, dynamic, materialized)
        for row_id in paired_ids
    ]
    row_metrics.sort(key=lambda row: (row.get("row_index_dynamic") is None, row.get("row_index_dynamic") or 0, row["row_id"]))
    if args.reuse_evaluation_root is not None:
        evaluation_root = args.reuse_evaluation_root.resolve()
        evaluation = {
            "dynamic_hf": _reuse_evaluator(
                label="dynamic_hf",
                source_dir=evaluation_root / "dynamic",
                output_dir=output_dir / "eval" / "dynamic_hf",
            ),
            "materialized_bf16_hf": _reuse_evaluator(
                label="materialized_bf16_hf",
                source_dir=evaluation_root / "materialized",
                output_dir=output_dir / "eval" / "materialized_bf16_hf",
            ),
        }
    else:
        evaluation = {
            "dynamic_hf": _run_evaluator(
                label="dynamic_hf",
                run_dir=args.dynamic_dir.resolve(),
                output_dir=output_dir / "eval" / "dynamic_hf",
                repo_root=repo_root,
            ),
            "materialized_bf16_hf": _run_evaluator(
                label="materialized_bf16_hf",
                run_dir=args.materialized_dir.resolve(),
                output_dir=output_dir / "eval" / "materialized_bf16_hf",
                repo_root=repo_root,
            ),
        }
    evaluation_ok = all(result.get("status") in {"passed", "reused"} for result in evaluation.values())
    if authentication["status"] != "passed":
        analysis_status = "diagnostic_only_authentication_failed"
    elif not evaluation_ok:
        analysis_status = "diagnostic_only_evaluator_failed"
    else:
        analysis_status = "complete"
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": analysis_status,
        "claim_status": "diagnostic_only_small_cohort",
        "claim_boundary": (
            "32-row paired consistency and consumer-evaluator diagnostics; "
            "mAP/AP50/AP75 are small-cohort values and carry no benchmark claim"
        ),
        "matcher_sensitivity": matcher_sensitivity,
        "cohort": {
            "cohort_json": args.cohort_json.resolve().as_posix() if args.cohort_json else None,
            "input_jsonl": args.input_jsonl.resolve().as_posix() if args.input_jsonl else None,
            "expected_row_count": args.expected_row_count,
            "authenticated_expected_row_count": len(expected),
        },
        "authentication": authentication,
        "arms": {
            key: {
                field: value
                for field, value in arm.items()
                if field not in {"raw_rows", "scored_rows", "raw_by_id", "scored_by_id", "traces_by_id", "provenance", "errors"}
            }
            for key, arm in (("dynamic_hf", dynamic), ("materialized_bf16_hf", materialized))
        },
        "procedure": {
            "paired_prediction_match": (
                "same normalized class and pixel IoU >= 0.50; candidates sorted by "
                "(-IoU, dynamic_index, materialized_index); greedy one-to-one"
            ),
            "paired_iou_space": "pixel xyxy",
            "gt_geometry": "src.data.geometry.coord_bins_to_pixel_xyxy",
            "gt_correctness": "same normalized COCO class; IoU descending greedy one-to-one at 0.50 and 0.75; consumer drops unknown/invalid prediction objects and reports counts",
            "coordinate_drift": "matched coord_bins only; signed materialized_minus_dynamic; absolute p50/p90/p95/p99/max",
            "sequence_policy": "semantic prediction sequence comparison plus optional generated-token trace; never positional-box zip",
            "score_policy": "native scored values retained per arm; no calibration or score rewriting",
            "row_inclusion": "all authenticated paired rows retained; no row is filtered for parser status, truncation, count, or object validity",
        },
        "aggregate": _aggregate(row_metrics),
        "worst_changed_image_ids": _worst_rows(row_metrics),
        "evaluation": evaluation,
    }
    _write_jsonl(output_dir / "paired_rows.jsonl", row_metrics)
    _write_json(output_dir / "paired_consistency.json", payload)
    _write_json(
        output_dir / "analysis_receipt.json",
        {
            "status": payload["status"],
            "claim_status": payload["claim_status"],
            "paired_consistency_json": "paired_consistency.json",
            "paired_rows_jsonl": "paired_rows.jsonl",
            "dynamic_dir": args.dynamic_dir.resolve().as_posix(),
            "materialized_dir": args.materialized_dir.resolve().as_posix(),
            "evaluation": {
                key: {
                    field: value
                    for field, value in result.items()
                    if field in {
                        "status",
                        "returncode",
                        "metrics_path",
                        "stdout_path",
                        "stderr_path",
                        "source_dir",
                        "source_metrics_path",
                        "source_evaluation_receipt_path",
                        "copied_files",
                    }
                }
                for key, result in evaluation.items()
            },
        },
    )
    print(json.dumps({
        "status": payload["status"],
        "claim_status": payload["claim_status"],
        "paired_row_count": authentication["paired_row_count"],
        "output": str(output_dir / "paired_consistency.json"),
    }, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
