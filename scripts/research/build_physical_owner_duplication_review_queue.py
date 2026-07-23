#!/usr/bin/env python3
"""Build a conservative, experiment-local physical-owner review queue.

This tool is deliberately a review aid, not a label builder.  It joins the
budget-16 trajectory-union receipts back to the exact rollout JSON and emits
only high-confidence *candidate* repeated-owner bursts.  Ambiguous,
unresolved, and official-unmatched rows remain neutral evidence.

The three output files are deterministic and contain no training targets:

* ``census.json`` records the evidence and conservative counts;
* ``candidate_queue.jsonl`` contains one candidate burst per owner/trajectory
  by default; and
* ``visualization_manifest.json`` is a compact per-image drawing manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "physical_owner_duplication_review_queue.v1"
UNION_SUPPORT_SCHEMA_VERSION = "individual_trajectory_union_support.v1"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
DEFAULT_BUDGET = 16

OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670

_VERIFIED = "verified_owner"
_COMPARATOR_DUPLICATES = frozenset({"duplicate", "duplicate_owner"})


class ReviewQueueError(ValueError):
    """Raised when immutable rollout or receipt evidence cannot be joined."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tokens(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(token) for token in tokens], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ReviewQueueError(f"JSON document must be an object: {source}")
    return dict(value)


def _image_id(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise ReviewQueueError(f"{field} must be a non-empty image identity")
    result = str(value).strip()
    if not result:
        raise ReviewQueueError(f"{field} must be a non-empty image identity")
    return result


def _int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ReviewQueueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewQueueError(f"{field} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ReviewQueueError(f"{field} must be >= {minimum}")
    return result


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ReviewQueueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewQueueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ReviewQueueError(f"{field} must be finite")
    return result


def _box(value: Any, *, field: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ReviewQueueError(f"{field} must contain four coordinates")
    x1, y1, x2, y2 = (_number(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if x2 <= x1 or y2 <= y1:
        raise ReviewQueueError(f"{field} must have positive area")
    return (x1, y1, x2, y2)


def _json_box(box: Sequence[float]) -> list[float]:
    return [float(value) for value in box]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _normalise_category(value: Any) -> str:
    return " ".join(str(value if value is not None else "").strip().lower().split())


def _trajectory_id(row: Mapping[str, Any], *, mode: str) -> str:
    explicit = row.get("trajectory_id", row.get("request_id"))
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if mode == "greedy":
        return "greedy"
    if row.get("seed") is not None:
        return f"seed-{_int(row['seed'], field='rollout.seed')}"
    raise ReviewQueueError("sampled rollout lacks trajectory_id and seed")


def _token_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReviewQueueError(f"{field} must be a token-id sequence")
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ReviewQueueError(f"{field}[{index}] must be a non-negative token id")
        result.append(int(token))
    return result


def _exact_complete_rows(tokens: Sequence[int], *, field: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split exact canonical object rows without decoding or retokenizing."""

    rows: list[dict[str, Any]] = []
    start = 0
    for offset, token in enumerate(tokens):
        if token != BOX_END:
            continue
        row = list(tokens[start : offset + 1])
        invalid_reason: str | None = None
        if len(row) < 9 or row[0] != OBJECT_REF_START or row[-1] != BOX_END:
            invalid_reason = "not_complete_canonical_object_row"
        else:
            markers = (OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END)
            if any(row.count(marker) != 1 for marker in markers):
                invalid_reason = "crossing_or_repeated_row_markers"
            else:
                description_end = row.index(OBJECT_REF_END)
                box_start = row.index(BOX_START)
                coordinates = row[box_start + 1 : -1]
                if not (0 < description_end < box_start < len(row) - 5):
                    invalid_reason = "invalid_schema_boundaries"
                elif len(coordinates) != 4 or any(
                    token_id < COORDINATE_TOKEN_START or token_id >= COORDINATE_TOKEN_END
                    for token_id in coordinates
                ):
                    invalid_reason = "invalid_coordinate_token_arity_or_range"
        rows.append(
            {
                "row_index": len(rows),
                "start_token_offset": start,
                "end_token_offset": offset + 1,
                "row_token_ids_sha256": _sha256_tokens(row),
                "valid_exact_token_row": invalid_reason is None,
                "invalid_exact_token_reason": invalid_reason,
            }
        )
        start = offset + 1
    suffix = list(tokens[start:])
    # Current rollout artifacts can end with a parser-dropped partial object
    # span.  It has no complete exact row identity and is therefore explicitly
    # neutral evidence rather than a reason to discard the completed prefix.
    suffix_evidence = {
        "neutral_parser_dropped_suffix": bool(suffix),
        "partial_suffix_token_count": len(suffix),
        "partial_suffix_token_ids_sha256": _sha256_tokens(suffix) if suffix else None,
    }
    return rows, suffix_evidence


def _prediction_values(row: Mapping[str, Any], *, field: str) -> list[Mapping[str, Any]]:
    parser = row.get("predictions", {})
    if isinstance(parser, list):
        values = parser
    elif isinstance(parser, Mapping):
        values = parser.get("predictions", parser.get("rows", []))
    else:
        raise ReviewQueueError(f"{field}.predictions must be an object or list")
    if values is None:
        values = []
    if not isinstance(values, list):
        raise ReviewQueueError(f"{field}.predictions list is malformed")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ReviewQueueError(f"{field}.predictions[{index}] must be an object")
        result.append(item)
    return result


def _validate_raw_span(value: Mapping[str, Any], generated_text: Any, *, field: str) -> None:
    """Verify available parser span evidence without relying on re-tokenization."""

    raw_text = value.get("raw_span_text")
    raw_hash = value.get("raw_span_sha256")
    if raw_text is not None and raw_hash is not None:
        if not isinstance(raw_text, str) or not isinstance(raw_hash, str):
            raise ReviewQueueError(f"{field} raw-span evidence is malformed")
        actual = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if actual != raw_hash:
            raise ReviewQueueError(f"{field} raw-span hash mismatch")
    start = value.get("char_start")
    end = value.get("char_end")
    if start is None and end is None:
        return
    if not isinstance(generated_text, str) or not isinstance(raw_text, str):
        raise ReviewQueueError(f"{field} lacks generated-text span evidence")
    start_index = _int(start, field=f"{field}.char_start", minimum=0)
    end_index = _int(end, field=f"{field}.char_end", minimum=start_index)
    if generated_text[start_index:end_index] != raw_text:
        raise ReviewQueueError(f"{field} does not match its exact generated-text span")


def _expand_artifact_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw in values:
        text = str(raw)
        matches = [Path(item) for item in glob.glob(text, recursive=True)] if any(char in text for char in "*?[") else []
        candidates = matches or [Path(text)]
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded.is_dir():
                paths.update(item.resolve() for item in expanded.rglob("*.json") if item.is_file())
            elif expanded.is_file():
                paths.add(expanded.resolve())
            else:
                raise FileNotFoundError(expanded)
    if not paths:
        raise ReviewQueueError("no exact rollout JSON files were supplied")
    return sorted(paths, key=str)


def _prompt_metadata_for_row(artifact: Mapping[str, Any], row: Mapping[str, Any], image_id: str) -> Mapping[str, Any]:
    metadata = artifact.get("prompt_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    candidate = metadata.get(str(row.get("example_id")), metadata.get(image_id))
    return candidate if isinstance(candidate, Mapping) else {}


def _load_exact_rollouts(
    paths: Iterable[str | Path], *, expected_mode: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    if expected_mode not in {"greedy", "sampled"}:
        raise ValueError(expected_mode)
    trajectories: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in _expand_artifact_paths(paths):
        artifact = _read_json(path)
        if artifact.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            raise ReviewQueueError(f"unexpected rollout schema: {path}")
        config = artifact.get("config")
        rollouts = artifact.get("rollouts")
        if not isinstance(config, Mapping) or not isinstance(rollouts, list):
            raise ReviewQueueError(f"invalid rollout artifact contract: {path}")
        config_mode = str(config.get("decode_mode", ""))
        if config_mode != expected_mode:
            raise ReviewQueueError(f"{path} decode_mode={config_mode!r}; expected {expected_mode!r}")
        source_path = str(path)
        source_sha256 = _sha256_file(path)
        sources.append({"path": source_path, "sha256": source_sha256, "decode_mode": expected_mode})
        for ordinal, raw in enumerate(rollouts):
            if not isinstance(raw, Mapping):
                raise ReviewQueueError(f"{path}.rollouts[{ordinal}] must be an object")
            row = dict(raw)
            row_mode = str(row.get("decode_mode", config_mode))
            if row_mode != expected_mode:
                raise ReviewQueueError(f"{path}.rollouts[{ordinal}] has inconsistent decode_mode")
            image_id = _image_id(row.get("image_id"), field=f"{path}.rollouts[{ordinal}].image_id")
            trajectory_id = _trajectory_id(row, mode=expected_mode)
            key = (image_id, trajectory_id)
            if key in trajectories:
                raise ReviewQueueError(f"duplicate trajectory identity: {image_id}:{trajectory_id}")
            tokens = _token_ids(row.get("generated_token_ids"), field=f"{path}.rollouts[{ordinal}].generated_token_ids")
            declared_hash = row.get("generated_token_ids_sha256")
            if not isinstance(declared_hash, str) or not declared_hash:
                raise ReviewQueueError(f"{path}.rollouts[{ordinal}] lacks generated_token_ids_sha256")
            actual_hash = _sha256_tokens(tokens)
            if declared_hash != actual_hash:
                raise ReviewQueueError(f"{path}.rollouts[{ordinal}] generated token hash mismatch")
            exact_rows, suffix_evidence = _exact_complete_rows(
                tokens, field=f"{path}.rollouts[{ordinal}].generated_token_ids"
            )
            exact_by_index = {int(item["row_index"]): item for item in exact_rows}
            predictions: dict[tuple[int, str], dict[str, Any]] = {}
            indexes: set[int] = set()
            invalid_geometry_count = 0
            invalid_token_row_count = sum(
                not bool(item["valid_exact_token_row"]) for item in exact_rows
            )
            for prediction_ordinal, parsed in enumerate(
                _prediction_values(row, field=f"{path}.rollouts[{ordinal}")
            ):
                field = f"{path}.rollouts[{ordinal}].predictions[{prediction_ordinal}]"
                row_index = _int(parsed.get("generated_order", parsed.get("row_index", prediction_ordinal)), field=f"{field}.generated_order", minimum=0)
                if row_index not in exact_by_index:
                    raise ReviewQueueError(f"{field} does not map to an exact generated token row")
                if row_index in indexes:
                    raise ReviewQueueError(f"duplicate exact generated row index: {image_id}:{trajectory_id}:{row_index}")
                indexes.add(row_index)
                prediction_id_value = parsed.get("object_span_id", parsed.get("prediction_id"))
                prediction_id = str(prediction_id_value).strip() if prediction_id_value is not None else ""
                if not prediction_id:
                    prediction_id = f"{trajectory_id}:row-{row_index}"
                bbox_value = parsed.get("bbox", parsed.get("bbox_xyxy"))
                try:
                    box: tuple[float, float, float, float] | None = _box(bbox_value, field=f"{field}.bbox")
                    geometry_error = None
                except ReviewQueueError as exc:
                    # The token row is still exact, but a non-positive parser
                    # box cannot be a trusted owner or duplicate candidate.
                    box = None
                    geometry_error = str(exc)
                    invalid_geometry_count += 1
                category = _normalise_category(
                    parsed.get("description", parsed.get("desc", parsed.get("category_name", parsed.get("category"))))
                )
                if not category:
                    raise ReviewQueueError(f"{field} lacks a prediction category")
                _validate_raw_span(parsed, row.get("generated_text"), field=field)
                prediction_key = (row_index, prediction_id)
                if prediction_key in predictions:
                    raise ReviewQueueError(f"duplicate exact prediction identity: {image_id}:{trajectory_id}:{row_index}:{prediction_id}")
                predictions[prediction_key] = {
                    "image_id": image_id,
                    "trajectory_id": trajectory_id,
                    "decode_mode": expected_mode,
                    "seed": row.get("seed"),
                    "generated_row_index": row_index,
                    "prediction_id": prediction_id,
                    "category": category,
                    "bbox": box,
                    "valid_geometry": box is not None and bool(exact_by_index[row_index]["valid_exact_token_row"]),
                    "invalid_geometry_reason": (
                        geometry_error
                        if geometry_error is not None
                        else exact_by_index[row_index]["invalid_exact_token_reason"]
                    ),
                    "invalid_evidence_reason": (
                        "invalid_exact_token_row"
                        if not bool(exact_by_index[row_index]["valid_exact_token_row"])
                        else ("invalid_exact_prediction_geometry" if box is None else None)
                    ),
                    "row_token_ids_sha256": exact_by_index[row_index]["row_token_ids_sha256"],
                    "row_token_start": exact_by_index[row_index]["start_token_offset"],
                    "row_token_end": exact_by_index[row_index]["end_token_offset"],
                }
            metadata = _prompt_metadata_for_row(artifact, row, image_id)
            trajectories[key] = {
                "image_id": image_id,
                "trajectory_id": trajectory_id,
                "decode_mode": expected_mode,
                "seed": row.get("seed"),
                "source_path": source_path,
                "source_sha256": source_sha256,
                "generated_token_ids_sha256": actual_hash,
                "image_path": metadata.get("image_path", row.get("image_path")),
                "neutral_exact_token_suffix": suffix_evidence,
                "neutral_invalid_exact_prediction_row_count": invalid_geometry_count,
                "neutral_invalid_exact_token_row_count": invalid_token_row_count,
                "predictions": predictions,
            }
    return trajectories, sorted(sources, key=lambda item: (item["decode_mode"], item["path"]))


def _budget16_receipts(union_support_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    source = Path(union_support_path).expanduser().resolve(strict=True)
    document = _read_json(source)
    if document.get("schema_version") != UNION_SUPPORT_SCHEMA_VERSION:
        raise ReviewQueueError(f"unexpected union-support schema: {source}")
    image_results = document.get("image_results", document.get("images"))
    if not isinstance(image_results, list):
        raise ReviewQueueError("union-support document lacks image_results")
    image_records: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_receipts: set[tuple[str, str, int, str]] = set()
    for image_ordinal, raw_image in enumerate(image_results):
        if not isinstance(raw_image, Mapping):
            raise ReviewQueueError(f"image_results[{image_ordinal}] must be an object")
        image_id = _image_id(raw_image.get("image_id"), field=f"image_results[{image_ordinal}].image_id")
        if image_id in seen_images:
            raise ReviewQueueError(f"duplicate union image identity: {image_id}")
        seen_images.add(image_id)
        budgets = raw_image.get("budgets")
        if not isinstance(budgets, list):
            raise ReviewQueueError(f"union image {image_id} lacks budgets")
        matching_budgets = [item for item in budgets if isinstance(item, Mapping) and item.get("budget") == DEFAULT_BUDGET]
        if len(matching_budgets) != 1:
            raise ReviewQueueError(f"union image {image_id} must contain exactly one budget-{DEFAULT_BUDGET} block")
        budget = matching_budgets[0]
        receipts = budget.get("row_assignment_receipts")
        if not isinstance(receipts, list):
            raise ReviewQueueError(f"union image {image_id} budget-{DEFAULT_BUDGET} lacks row_assignment_receipts")
        normalized_receipts: list[dict[str, Any]] = []
        for receipt_ordinal, raw_receipt in enumerate(receipts):
            if not isinstance(raw_receipt, Mapping):
                raise ReviewQueueError(f"union receipt {image_id}[{receipt_ordinal}] must be an object")
            receipt = dict(raw_receipt)
            receipt_image = _image_id(receipt.get("image_id", image_id), field=f"union receipt {image_id}[{receipt_ordinal}].image_id")
            if receipt_image != image_id:
                raise ReviewQueueError(f"union receipt image mismatch: expected {image_id}, got {receipt_image}")
            trajectory_id = str(receipt.get("trajectory_id", "")).strip()
            prediction_id = str(receipt.get("prediction_id", "")).strip()
            row_index = _int(receipt.get("generated_row_index"), field=f"union receipt {image_id}[{receipt_ordinal}].generated_row_index", minimum=0)
            if not trajectory_id or not prediction_id:
                raise ReviewQueueError(f"union receipt {image_id}[{receipt_ordinal}] lacks trajectory or prediction identity")
            key = (image_id, trajectory_id, row_index, prediction_id)
            if key in seen_receipts:
                raise ReviewQueueError(f"duplicate union receipt identity: {image_id}:{trajectory_id}:{row_index}:{prediction_id}")
            seen_receipts.add(key)
            receipt["image_id"] = image_id
            receipt["trajectory_id"] = trajectory_id
            receipt["prediction_id"] = prediction_id
            receipt["generated_row_index"] = row_index
            normalized_receipts.append(receipt)
        image_records.append(
            {
                "image_id": image_id,
                "owners": list(raw_image.get("owners", [])) if isinstance(raw_image.get("owners", []), list) else [],
                "receipts": normalized_receipts,
            }
        )
    source_info = {"path": str(source), "sha256": _sha256_file(source)}
    return document, image_records, source_info


def _same_box(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6) for a, b in zip(left, right, strict=True))


def _joined_rows(
    image_records: Sequence[Mapping[str, Any]],
    trajectories: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    trajectory_scope: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for image in image_records:
        image_id = str(image["image_id"])
        for receipt in image["receipts"]:
            trajectory_id = str(receipt["trajectory_id"])
            # A greedy-only request intentionally needs no sampled shard.
            # Receipt decode_mode is an immutable union-support field, so it
            # can decide this scope filter before any exact sampled join.
            if trajectory_scope == "greedy" and str(receipt.get("decode_mode", "")) != "greedy":
                continue
            key = (image_id, trajectory_id)
            exact = trajectories.get(key)
            if exact is None:
                raise ReviewQueueError(f"union receipt has no exact rollout trajectory: {image_id}:{trajectory_id}")
            row_index = int(receipt["generated_row_index"])
            prediction_id = str(receipt["prediction_id"])
            prediction = exact["predictions"].get((row_index, prediction_id))
            if prediction is None:
                raise ReviewQueueError(
                    "union receipt has no exact rollout prediction: "
                    f"{image_id}:{trajectory_id}:{row_index}:{prediction_id}"
                )
            if bool(prediction.get("valid_geometry", True)):
                receipt_box = _box(receipt.get("bbox"), field=f"union receipt {image_id}:{trajectory_id}:{row_index}.bbox")
                if not _same_box(receipt_box, prediction["bbox"]):
                    raise ReviewQueueError(f"union/exact prediction geometry mismatch: {image_id}:{trajectory_id}:{row_index}")
            receipt_category = _normalise_category(receipt.get("category"))
            if receipt_category and receipt_category != prediction["category"]:
                raise ReviewQueueError(f"union/exact prediction category mismatch: {image_id}:{trajectory_id}:{row_index}")
            receipt_mode = receipt.get("decode_mode")
            if receipt_mode is not None and str(receipt_mode) != str(exact["decode_mode"]):
                raise ReviewQueueError(f"union/exact decode_mode mismatch: {image_id}:{trajectory_id}")
            grouped[key].append(
                {
                    "receipt": dict(receipt),
                    "prediction": dict(prediction),
                    "source_path": exact["source_path"],
                    "source_sha256": exact["source_sha256"],
                    "generated_token_ids_sha256": exact["generated_token_ids_sha256"],
                    "image_path": exact.get("image_path"),
                }
            )
    for key, rows in grouped.items():
        rows.sort(key=lambda item: (int(item["prediction"]["generated_row_index"]), str(item["prediction"]["prediction_id"])))
        row_indexes = [int(item["prediction"]["generated_row_index"]) for item in rows]
        if len(row_indexes) != len(set(row_indexes)):
            raise ReviewQueueError(f"duplicate union exact-row identity within trajectory: {key[0]}:{key[1]}")
    return dict(grouped)


def _status(receipt: Mapping[str, Any]) -> str:
    return str(receipt.get("entity_status", "")).strip().lower()


def _owner_id(receipt: Mapping[str, Any]) -> str | None:
    for field in ("owner_id", "candidate_owner_id"):
        value = receipt.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _receipt_owner_box(receipt: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    value = receipt.get("owner_bbox")
    if value is None:
        return None
    try:
        return _box(value, field="owner_bbox")
    except ReviewQueueError:
        return None


def _annotation_iou(receipt: Mapping[str, Any], owner_box: Sequence[float] | None, prediction_box: Sequence[float]) -> float | None:
    for field in ("intersection_over_union", "candidate_owner_iou"):
        if receipt.get(field) is not None:
            try:
                return _number(receipt[field], field=field)
            except ReviewQueueError:
                return None
    return _iou(prediction_box, owner_box) if owner_box is not None else None


def _row_reference(item: Mapping[str, Any], *, annotation_iou: float | None = None) -> dict[str, Any]:
    receipt = item["receipt"]
    prediction = item["prediction"]
    result: dict[str, Any] = {
        "row_index": int(prediction["generated_row_index"]),
        "prediction_id": str(prediction["prediction_id"]),
        "category": str(prediction["category"]),
        "prediction_bbox": _json_box(prediction["bbox"]),
        "row_token_ids_sha256": str(prediction["row_token_ids_sha256"]),
        "row_token_span": [int(prediction["row_token_start"]), int(prediction["row_token_end"])],
        "entity_status": _status(receipt),
    }
    if annotation_iou is not None:
        result["annotation_iou"] = float(annotation_iou)
    owner_id = _owner_id(receipt)
    if owner_id is not None:
        result["owner_id"] = owner_id
    owner_box = _receipt_owner_box(receipt)
    if owner_box is not None:
        result["owner_bbox"] = _json_box(owner_box)
    owner_category = _normalise_category(receipt.get("owner_category"))
    if owner_category:
        result["owner_category"] = owner_category
    return result


def _trusted_verified(item: Mapping[str, Any], *, annotation_iou_threshold: float) -> tuple[str, tuple[float, float, float, float], float] | None:
    receipt = item["receipt"]
    if not bool(item["prediction"].get("valid_geometry", True)):
        return None
    if _status(receipt) != _VERIFIED:
        return None
    owner_id = _owner_id(receipt)
    owner_box = _receipt_owner_box(receipt)
    if owner_id is None or owner_box is None:
        return None
    predicted = item["prediction"]
    annotation_iou = _annotation_iou(receipt, owner_box, predicted["bbox"])
    if annotation_iou is None or annotation_iou < annotation_iou_threshold:
        return None
    owner_category = _normalise_category(receipt.get("owner_category"))
    if owner_category and owner_category != predicted["category"]:
        return None
    return owner_id, owner_box, annotation_iou


def _candidate_duplicate(
    item: Mapping[str, Any],
    *,
    accepted: Mapping[str, Mapping[str, Any]],
    annotation_iou_threshold: float,
    prediction_iou_threshold: float,
) -> tuple[str, float, float] | None:
    receipt = item["receipt"]
    if not bool(item["prediction"].get("valid_geometry", True)):
        return None
    if _status(receipt) not in _COMPARATOR_DUPLICATES:
        return None
    owner_id = _owner_id(receipt)
    earlier = accepted.get(owner_id) if owner_id is not None else None
    if earlier is None:
        return None
    prediction = item["prediction"]
    if prediction["category"] != earlier["category"]:
        return None
    owner_category = _normalise_category(receipt.get("owner_category"))
    if owner_category and owner_category != earlier["category"]:
        return None
    owner_box = _receipt_owner_box(receipt) or earlier["owner_bbox"]
    annotation_iou = _annotation_iou(receipt, owner_box, prediction["bbox"])
    if annotation_iou is None or annotation_iou < annotation_iou_threshold:
        return None
    prediction_overlap = _iou(prediction["bbox"], earlier["prediction_bbox"])
    if prediction_overlap < prediction_iou_threshold:
        return None
    return str(owner_id), annotation_iou, prediction_overlap


def _build_trajectory_candidates(
    *,
    image_id: str,
    trajectory_id: str,
    rows: Sequence[Mapping[str, Any]],
    annotation_iou_threshold: float,
    prediction_iou_threshold: float,
    emit_separate_bursts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    """Classify one ordered trajectory without converting neutral rows to labels."""

    accepted: dict[str, dict[str, Any]] = {}
    classifications: list[dict[str, Any]] = []
    bursts: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    emitted_owner_count: Counter[str] = Counter()
    neutral_counts: Counter[str] = Counter()

    def close_active() -> None:
        nonlocal active
        if active is not None:
            bursts.append(active)
            active = None

    for item in rows:
        receipt = item["receipt"]
        prediction = item["prediction"]
        row_index = int(prediction["generated_row_index"])
        status = _status(receipt)
        if not bool(prediction.get("valid_geometry", True)):
            close_active()
            invalid_reason = str(prediction.get("invalid_evidence_reason") or "invalid_exact_prediction_geometry")
            neutral_counts[invalid_reason] += 1
            classifications.append(
                {
                    "row_index": row_index,
                    "classification": "neutral",
                    "neutral_reason": invalid_reason,
                    "entity_status": status,
                }
            )
            continue
        trusted = _trusted_verified(item, annotation_iou_threshold=annotation_iou_threshold)
        if trusted is not None:
            close_active()
            owner_id, owner_box, annotation_iou = trusted
            if owner_id not in accepted:
                accepted[owner_id] = {
                    "owner_id": owner_id,
                    "owner_bbox": owner_box,
                    "category": str(prediction["category"]),
                    "prediction_bbox": tuple(prediction["bbox"]),
                    "item": item,
                    "annotation_iou": annotation_iou,
                }
                classifications.append({"row_index": row_index, "classification": "accepted_first_occurrence", "owner_id": owner_id})
            else:
                neutral_counts["verified_owner_already_seen"] += 1
                classifications.append({"row_index": row_index, "classification": "neutral", "neutral_reason": "verified_owner_already_seen", "owner_id": owner_id})
            continue

        candidate = _candidate_duplicate(
            item,
            accepted=accepted,
            annotation_iou_threshold=annotation_iou_threshold,
            prediction_iou_threshold=prediction_iou_threshold,
        )
        if candidate is None:
            close_active()
            reason = "official_unmatched_or_ambiguous" if status not in _COMPARATOR_DUPLICATES else "duplicate_comparator_below_threshold_or_missing_owner"
            neutral_counts[reason] += 1
            classifications.append({"row_index": row_index, "classification": "neutral", "neutral_reason": reason, "entity_status": status})
            continue

        owner_id, annotation_iou, prediction_overlap = candidate
        contiguous = (
            active is not None
            and active["owner_id"] == owner_id
            and row_index == int(active["duplicate_rows"][-1]["row_index"]) + 1
        )
        if not contiguous:
            close_active()
            if emitted_owner_count[owner_id] and not emit_separate_bursts:
                neutral_counts["additional_burst_suppressed"] += 1
                classifications.append({"row_index": row_index, "classification": "neutral", "neutral_reason": "additional_burst_suppressed", "owner_id": owner_id})
                continue
            active = {
                "owner_id": owner_id,
                "earlier": accepted[owner_id],
                "duplicate_rows": [],
                "accepted_before_burst": set(accepted),
            }
            emitted_owner_count[owner_id] += 1
        assert active is not None
        active["duplicate_rows"].append(
            {
                **_row_reference(item, annotation_iou=annotation_iou),
                "prediction_to_earlier_owner_iou": float(prediction_overlap),
            }
        )
        classifications.append({"row_index": row_index, "classification": "candidate_duplicate", "owner_id": owner_id})
    close_active()

    candidates: list[dict[str, Any]] = []
    for burst_ordinal, burst in enumerate(bursts, start=1):
        duplicate_rows = list(burst["duplicate_rows"])
        final_row_index = int(duplicate_rows[-1]["row_index"])
        accepted_before = set(burst["accepted_before_burst"])
        recovery: dict[str, Any] | None = None
        for item in rows:
            prediction = item["prediction"]
            if int(prediction["generated_row_index"]) <= final_row_index:
                continue
            trusted = _trusted_verified(item, annotation_iou_threshold=annotation_iou_threshold)
            if trusted is None:
                continue
            owner_id, _owner_box, annotation_iou = trusted
            if owner_id not in accepted_before:
                recovery = _row_reference(item, annotation_iou=annotation_iou)
                recovery["owner_id"] = owner_id
                break
        earlier = burst["earlier"]
        candidate_id = (
            f"{image_id}:{trajectory_id}:{burst['owner_id']}:"
            f"rows-{duplicate_rows[0]['row_index']}-{duplicate_rows[-1]['row_index']}:burst-{burst_ordinal}"
        )
        candidates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "candidate_status": "high_confidence_candidate_repeated_physical_owner",
                "review_required": True,
                "not_a_training_label": True,
                "image_id": image_id,
                "trajectory_id": trajectory_id,
                "decode_mode": str(rows[0]["prediction"]["decode_mode"]),
                "physical_owner_id": str(burst["owner_id"]),
                "earlier_accepted_owner_row": _row_reference(earlier["item"], annotation_iou=float(earlier["annotation_iou"])),
                "duplicate_rows": duplicate_rows,
                "first_later_trusted_unseen_verified_owner_row": recovery,
                "thresholds": {
                    "annotation_iou": float(annotation_iou_threshold),
                    "prediction_to_prediction_iou": float(prediction_iou_threshold),
                },
                "exact_rollout": {
                    "source_path": str(rows[0]["source_path"]),
                    "source_sha256": str(rows[0]["source_sha256"]),
                    "generated_token_ids_sha256": str(rows[0]["generated_token_ids_sha256"]),
                },
            }
        )
    return candidates, classifications, neutral_counts


def _owners_for_manifest(image: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    owners: dict[str, dict[str, Any]] = {}
    for raw in image.get("owners", []):
        if not isinstance(raw, Mapping):
            continue
        owner_id_value = raw.get("owner_id", raw.get("physical_owner_id"))
        if owner_id_value is None:
            continue
        owner_id = str(owner_id_value)
        try:
            box = _box(raw.get("bbox", raw.get("bbox_xyxy")), field=f"owner {owner_id}.bbox")
        except ReviewQueueError:
            continue
        category = _normalise_category(raw.get("category", raw.get("category_name")))
        owners[owner_id] = {"owner_id": owner_id, "category": category or None, "owner_bbox": _json_box(box)}
    for item in rows:
        receipt = item["receipt"]
        owner_id = _owner_id(receipt)
        box = _receipt_owner_box(receipt)
        if owner_id is None or box is None or owner_id in owners:
            continue
        owners[owner_id] = {
            "owner_id": owner_id,
            "category": _normalise_category(receipt.get("owner_category")) or None,
            "owner_bbox": _json_box(box),
        }
    return [owners[key] for key in sorted(owners)]


def build_physical_owner_duplication_review_queue(
    *,
    union_support_path: str | Path,
    greedy_json_paths: Iterable[str | Path],
    sampled_json_paths: Iterable[str | Path] = (),
    trajectory_scope: str = "greedy",
    annotation_iou_threshold: float = 0.5,
    prediction_iou_threshold: float = 0.9,
    emit_separate_bursts: bool = False,
) -> dict[str, Any]:
    """Return deterministic candidate-review artifacts without writing labels."""

    if trajectory_scope not in {"greedy", "all"}:
        raise ReviewQueueError("trajectory_scope must be 'greedy' or 'all'")
    annotation_iou_threshold = _number(annotation_iou_threshold, field="annotation_iou_threshold")
    prediction_iou_threshold = _number(prediction_iou_threshold, field="prediction_iou_threshold")
    if not 0.0 <= annotation_iou_threshold <= 1.0:
        raise ReviewQueueError("annotation_iou_threshold must be in [0, 1]")
    if not 0.0 <= prediction_iou_threshold <= 1.0:
        raise ReviewQueueError("prediction_iou_threshold must be in [0, 1]")

    _union_document, image_records, union_source = _budget16_receipts(union_support_path)
    greedy, greedy_sources = _load_exact_rollouts(greedy_json_paths, expected_mode="greedy")
    sampled_paths = list(sampled_json_paths)
    sampled: dict[tuple[str, str], dict[str, Any]] = {}
    sampled_sources: list[dict[str, str]] = []
    if sampled_paths:
        sampled, sampled_sources = _load_exact_rollouts(sampled_paths, expected_mode="sampled")
    trajectories = {**greedy}
    for key, value in sampled.items():
        if key in trajectories:
            raise ReviewQueueError(f"duplicate trajectory identity across greedy/sampled inputs: {key[0]}:{key[1]}")
        trajectories[key] = value
    joined = _joined_rows(image_records, trajectories, trajectory_scope=trajectory_scope)

    candidate_rows: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    neutral_counts: Counter[str] = Counter()
    manifests: list[dict[str, Any]] = []
    image_by_id = {str(item["image_id"]): item for item in image_records}
    for (image_id, trajectory_id), rows in sorted(joined.items(), key=lambda item: (item[0][0], item[0][1])):
        candidates, classifications, trajectory_neutral_counts = _build_trajectory_candidates(
            image_id=image_id,
            trajectory_id=trajectory_id,
            rows=rows,
            annotation_iou_threshold=annotation_iou_threshold,
            prediction_iou_threshold=prediction_iou_threshold,
            emit_separate_bursts=emit_separate_bursts,
        )
        candidate_rows.extend(candidates)
        classification_counts.update(item["classification"] for item in classifications)
        neutral_counts.update(trajectory_neutral_counts)

    candidate_rows.sort(
        key=lambda item: (
            str(item["image_id"]),
            str(item["trajectory_id"]),
            str(item["physical_owner_id"]),
            int(item["duplicate_rows"][0]["row_index"]),
        )
    )
    candidate_ids_by_image: dict[str, list[str]] = defaultdict(list)
    for candidate in candidate_rows:
        candidate_ids_by_image[str(candidate["image_id"])].append(str(candidate["candidate_id"]))
    for image_id in sorted({key[0] for key in joined}):
        image = image_by_id[image_id]
        image_rows = [item for key, rows in joined.items() if key[0] == image_id for item in rows]
        image_rows.sort(key=lambda item: (str(item["prediction"]["trajectory_id"]), int(item["prediction"]["generated_row_index"])))
        source_artifacts = {
            (str(item["source_path"]), str(item["source_sha256"]), str(item["generated_token_ids_sha256"]))
            for item in image_rows
        }
        manifests.append(
            {
                "image_id": image_id,
                "image_path": next((item.get("image_path") for item in image_rows if item.get("image_path")), None),
                "owner_boxes": _owners_for_manifest(image, image_rows),
                "prediction_boxes": [
                    {
                        "trajectory_id": str(item["prediction"]["trajectory_id"]),
                        "row_index": int(item["prediction"]["generated_row_index"]),
                        "prediction_id": str(item["prediction"]["prediction_id"]),
                        "prediction_bbox": (
                            _json_box(item["prediction"]["bbox"])
                            if bool(item["prediction"].get("valid_geometry", True))
                            else None
                        ),
                        "geometry_status": (
                            "valid"
                            if bool(item["prediction"].get("valid_geometry", True))
                            else "neutral_" + str(item["prediction"].get("invalid_evidence_reason") or "invalid_exact_prediction_geometry")
                        ),
                        "row_token_ids_sha256": str(item["prediction"]["row_token_ids_sha256"]),
                    }
                    for item in image_rows
                ],
                "candidate_ids": sorted(candidate_ids_by_image.get(image_id, [])),
                "source_artifacts": [
                    {
                        "union_support_path": union_source["path"],
                        "union_support_sha256": union_source["sha256"],
                        "rollout_path": path,
                        "rollout_sha256": digest,
                        "generated_token_ids_sha256": tokens_hash,
                    }
                    for path, digest, tokens_hash in sorted(source_artifacts)
                ],
            }
        )
    census = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "human_review_queue_only_not_training_labels",
        "trajectory_scope": trajectory_scope,
        "budget": DEFAULT_BUDGET,
        "thresholds": {
            "annotation_iou": annotation_iou_threshold,
            "prediction_to_prediction_iou": prediction_iou_threshold,
        },
        "emit_separate_bursts": bool(emit_separate_bursts),
        "input_artifacts": {
            "union_support": union_source,
            "exact_rollouts": sorted(greedy_sources + sampled_sources, key=lambda item: (item["decode_mode"], item["path"])),
        },
        "image_count": len(manifests),
        "trajectory_count": len(joined),
        "candidate_burst_count": len(candidate_rows),
        "candidate_duplicate_row_count": sum(len(item["duplicate_rows"]) for item in candidate_rows),
        "candidate_with_recovery_count": sum(item["first_later_trusted_unseen_verified_owner_row"] is not None for item in candidate_rows),
        "neutral_exact_token_suffix_trajectory_count": sum(
            bool(item.get("neutral_exact_token_suffix", {}).get("neutral_parser_dropped_suffix"))
            for item in trajectories.values()
        ),
        "neutral_exact_token_suffix_token_count": sum(
            int(item.get("neutral_exact_token_suffix", {}).get("partial_suffix_token_count", 0))
            for item in trajectories.values()
        ),
        "neutral_invalid_exact_prediction_row_count": sum(
            int(item.get("neutral_invalid_exact_prediction_row_count", 0))
            for item in trajectories.values()
        ),
        "neutral_invalid_exact_token_row_count": sum(
            int(item.get("neutral_invalid_exact_token_row_count", 0))
            for item in trajectories.values()
        ),
        "row_classification_counts": {key: int(value) for key, value in sorted(classification_counts.items())},
        "neutral_reason_counts": {key: int(value) for key, value in sorted(neutral_counts.items())},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "compact_visual_review_manifest_not_training_labels",
        "images": manifests,
    }
    return {"census": census, "candidate_queue": candidate_rows, "visualization_manifest": manifest}


# Short alias for focused callers that do not need the long experiment name.
build_review_queue = build_physical_owner_duplication_review_queue


def write_physical_owner_duplication_review_queue(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write the three immutable review artifacts, refusing to overwrite them."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = {
        "census": destination / "census.json",
        "candidate_queue": destination / "candidate_queue.jsonl",
        "visualization_manifest": destination / "visualization_manifest.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite review-queue artifact(s): " + ", ".join(str(path) for path in existing))
    targets["census"].write_text(_canonical_json(result["census"]) + "\n", encoding="utf-8")
    with targets["candidate_queue"].open("w", encoding="utf-8") as handle:
        for row in result["candidate_queue"]:
            handle.write(_canonical_json(row) + "\n")
    targets["visualization_manifest"].write_text(_canonical_json(result["visualization_manifest"]) + "\n", encoding="utf-8")
    return targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-support", required=True, help="budget-16 trajectory-union-support JSON")
    parser.add_argument("--greedy-json", action="append", required=True, help="exact greedy rollout JSON (repeatable; directories/globs accepted)")
    parser.add_argument("--sampled-json", action="append", default=[], help="optional exact sampled rollout JSON/shard (repeatable)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-scope", choices=("greedy", "all"), default="greedy")
    parser.add_argument("--annotation-iou-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-iou-threshold", type=float, default=0.9)
    parser.add_argument("--emit-separate-bursts", action="store_true", help="emit later non-contiguous bursts for the same owner/trajectory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_physical_owner_duplication_review_queue(
        union_support_path=args.union_support,
        greedy_json_paths=args.greedy_json,
        sampled_json_paths=args.sampled_json,
        trajectory_scope=args.trajectory_scope,
        annotation_iou_threshold=args.annotation_iou_threshold,
        prediction_iou_threshold=args.prediction_iou_threshold,
        emit_separate_bursts=args.emit_separate_bursts,
    )
    paths = write_physical_owner_duplication_review_queue(result, args.output_dir)
    print(_canonical_json({key: str(value) for key, value in sorted(paths.items())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
