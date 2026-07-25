#!/usr/bin/env python3
"""Build an arm-blinded held-out physical-owner change review packet.

The comparison ledger is treated as a provenance claim, not as authoritative
attribution.  This script verifies its input digests and recomputes every
changed-owner reference with the comparator's cardinality-first global matcher
before materializing reviewer-visible files.

Reviewer-visible files use neutral case and view identifiers.  Checkpoint
identity, change direction, original row/owner references, and input paths are
kept under ``private/``.  Officially unmatched predictions are represented as
unresolved review items; they are never relabelled as hallucinations here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # Allow direct script execution.
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _bbox,
    _global_matches,
    _gt_objects,
    _gt_signature,
    _pred_objects,
)
from src.vis.matching import MatchPair, MatchResult  # noqa: E402
from src.vis.normalization import VisualObject, VisualRow  # noqa: E402
from src.vis.rendering import render_comparison_png  # noqa: E402


SCHEMA_VERSION = "heldout_owner_change_review_packet.v1"
DEFAULT_CROP_HALO_PX = 72
EXPECTED_PHASE_ZERO_REFERENCE_COUNT = 85
TERMINAL_TEXT = "<|im_end|>"
ENTITY_CATEGORY_OPTIONS = (
    "real_owner_change",
    "duplicate",
    "category_alias_or_disagreement",
    "unsupported_candidate",
    "uncertain",
)
GEOMETRY_OPTIONS = (
    "acceptable",
    "localization_error",
    "neighboring_instance_or_mixed_extent_error",
    "uncertain",
)
_REVIEWER_LEAK_PATTERN = re.compile(r"\b(?:source|transition|gain|loss)\b", re.IGNORECASE)


class ReviewPacketContractError(ValueError):
    """Raised when an input cannot support a provenance-safe packet."""


@dataclass(frozen=True)
class ArtifactRows:
    path: Path
    sha256: str
    ordered_row_ids: tuple[str, ...]
    rows: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class OwnerReference:
    row_id: str
    owner_index: int
    ledger_side: str

    @property
    def key(self) -> tuple[str, int]:
        return self.row_id, self.owner_index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewPacketContractError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReviewPacketContractError(f"{label} must be a list")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewPacketContractError(f"{label} must be non-empty text")
    return value.strip()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewPacketContractError(f"{label} must be a non-negative integer")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewPacketContractError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label)


def _load_artifact(input_entry: Any, label: str) -> ArtifactRows:
    entry = _mapping(input_entry, label)
    path = Path(_nonempty_text(entry.get("path"), f"{label}.path")).resolve(strict=True)
    declared_sha256 = _nonempty_text(entry.get("sha256"), f"{label}.sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != declared_sha256:
        raise ReviewPacketContractError(
            f"{label} digest mismatch: declared={declared_sha256} actual={actual_sha256}"
        )

    ordered: list[str] = []
    rows: dict[str, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReviewPacketContractError(f"cannot read {label} artifact: {path}") from exc
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row_value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewPacketContractError(
                f"{label} JSONL line {line_index + 1} is malformed"
            ) from exc
        row = _mapping(row_value, f"{label}[{line_index}]")
        row_id = _nonempty_text(row.get("row_id"), f"{label}[{line_index}].row_id")
        if row_id in rows:
            raise ReviewPacketContractError(f"{label} has duplicate row_id {row_id!r}")
        if row.get("example_id") != row_id:
            raise ReviewPacketContractError(
                f"{label} row {row_id!r} has inconsistent example_id provenance"
            )
        row_index = _nonnegative_int(row.get("row_index"), f"{label}.{row_id}.row_index")
        if row_index != len(ordered):
            raise ReviewPacketContractError(
                f"{label} row {row_id!r} has row_index={row_index}, expected {len(ordered)}"
            )
        ordered.append(row_id)
        rows[row_id] = row
    if not ordered:
        raise ReviewPacketContractError(f"{label} artifact is empty")
    return ArtifactRows(
        path=path,
        sha256=actual_sha256,
        ordered_row_ids=tuple(ordered),
        rows=rows,
    )


def _dimensions(row: Mapping[str, Any], row_id: str) -> tuple[int, int]:
    width = row.get("image_width", row.get("width"))
    height = row.get("image_height", row.get("height"))
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ReviewPacketContractError(f"row {row_id!r} has invalid image dimensions")
    return width, height


def _image_path(row: Mapping[str, Any], row_id: str) -> Path:
    value = _nonempty_text(
        row.get("image_path", row.get("image")), f"row {row_id!r}.image_path"
    )
    path = Path(value)
    if not path.is_absolute():
        raise ReviewPacketContractError(f"row {row_id!r} image path must be absolute")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ReviewPacketContractError(f"row {row_id!r} image is missing: {path}") from exc


def _validate_arm_pair(arm_a: ArtifactRows, arm_b: ArtifactRows) -> None:
    if arm_a.ordered_row_ids != arm_b.ordered_row_ids:
        raise ReviewPacketContractError(
            "arm artifacts must preserve identical ordered row provenance"
        )
    for row_id in arm_a.ordered_row_ids:
        left = arm_a.rows[row_id]
        right = arm_b.rows[row_id]
        try:
            left_signature = _gt_signature(dict(left), row_id=row_id)
            right_signature = _gt_signature(dict(right), row_id=row_id)
        except ValueError as exc:
            raise ReviewPacketContractError(
                f"row {row_id!r} has invalid GT provenance"
            ) from exc
        if left_signature != right_signature or left.get("gt") != right.get("gt"):
            raise ReviewPacketContractError(f"GT provenance mismatch for row {row_id!r}")
        if _dimensions(left, row_id) != _dimensions(right, row_id):
            raise ReviewPacketContractError(f"image dimension mismatch for row {row_id!r}")
        left_image = left.get("image_path", left.get("image"))
        right_image = right.get("image_path", right.get("image"))
        if left_image != right_image:
            raise ReviewPacketContractError(f"image provenance mismatch for row {row_id!r}")
        if left.get("row_index") != right.get("row_index"):
            raise ReviewPacketContractError(f"row-index provenance mismatch for row {row_id!r}")


def _parse_declared_refs(
    geometry: Mapping[str, Any],
    *,
    field: str,
    count_field: str,
    ledger_side: str,
) -> tuple[OwnerReference, ...]:
    entries = _list(geometry.get(field), f"common_owner_geometry.{field}")
    refs: list[OwnerReference] = []
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"common_owner_geometry.{field}[{index}]")
        refs.append(
            OwnerReference(
                row_id=_nonempty_text(entry.get("row_id"), f"{field}[{index}].row_id"),
                owner_index=_nonnegative_int(
                    entry.get("owner_index"), f"{field}[{index}].owner_index"
                ),
                ledger_side=ledger_side,
            )
        )
    if len({ref.key for ref in refs}) != len(refs):
        raise ReviewPacketContractError(f"{field} contains duplicate owner references")
    declared_count = _nonnegative_int(geometry.get(count_field), f"{count_field}")
    if declared_count != len(refs):
        raise ReviewPacketContractError(
            f"{count_field}={declared_count} does not match {field} length={len(refs)}"
        )
    return tuple(refs)


def _recompute_reference_sets(
    arm_a: ArtifactRows,
    arm_b: ArtifactRows,
    *,
    threshold: float,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    arm_a_only: set[tuple[str, int]] = set()
    arm_b_only: set[tuple[str, int]] = set()
    for row_id in arm_a.ordered_row_ids:
        left_row = dict(arm_a.rows[row_id])
        right_row = dict(arm_b.rows[row_id])
        try:
            gt = _gt_objects(left_row, row_id=row_id)
            left_pred, _ = _pred_objects(left_row)
            right_pred, _ = _pred_objects(right_row)
            left_matched = {
                gt_index for gt_index, _, _ in _global_matches(gt, left_pred, threshold)
            }
            right_matched = {
                gt_index for gt_index, _, _ in _global_matches(gt, right_pred, threshold)
            }
        except ValueError as exc:
            raise ReviewPacketContractError(
                f"cannot recompute authoritative attribution for row {row_id!r}"
            ) from exc
        arm_a_only.update((row_id, owner_index) for owner_index in left_matched - right_matched)
        arm_b_only.update((row_id, owner_index) for owner_index in right_matched - left_matched)
    return arm_a_only, arm_b_only


def _validate_reference_ledger(
    ledger: Mapping[str, Any],
    arm_a: ArtifactRows,
    arm_b: ArtifactRows,
    *,
    expected_total_refs: int | None,
) -> tuple[tuple[OwnerReference, ...], float]:
    policy = _mapping(ledger.get("policy"), "policy")
    threshold_value = policy.get("match_iou_threshold")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise ReviewPacketContractError("policy.match_iou_threshold must be numeric")
    threshold = float(threshold_value)
    if not math.isclose(threshold, 0.50, rel_tol=0.0, abs_tol=1e-12):
        raise ReviewPacketContractError(
            f"Phase Zero review requires match_iou_threshold=0.50, got {threshold}"
        )
    if policy.get("delta_convention") != "arm_b_minus_arm_a":
        raise ReviewPacketContractError("comparison delta convention is not arm_b_minus_arm_a")
    if policy.get("unmatched_predictions_are_not_hallucinations") is not True:
        raise ReviewPacketContractError(
            "comparison policy must preserve unmatched predictions as unresolved"
        )

    geometry = _mapping(ledger.get("common_owner_geometry"), "common_owner_geometry")
    arm_a_refs = _parse_declared_refs(
        geometry,
        field="arm_a_only_owner_refs",
        count_field="arm_a_only_owner_count",
        ledger_side="arm_a_only",
    )
    arm_b_refs = _parse_declared_refs(
        geometry,
        field="arm_b_only_owner_refs",
        count_field="arm_b_only_owner_count",
        ledger_side="arm_b_only",
    )
    declared_a = {ref.key for ref in arm_a_refs}
    declared_b = {ref.key for ref in arm_b_refs}
    if declared_a & declared_b:
        raise ReviewPacketContractError("owner reference appears on both comparison sides")

    recomputed_a, recomputed_b = _recompute_reference_sets(
        arm_a, arm_b, threshold=threshold
    )
    if declared_a != recomputed_a:
        missing = sorted(recomputed_a - declared_a)[:5]
        extra = sorted(declared_a - recomputed_a)[:5]
        raise ReviewPacketContractError(
            f"arm_a-only reference set disagrees with recomputation: missing={missing} extra={extra}"
        )
    if declared_b != recomputed_b:
        missing = sorted(recomputed_b - declared_b)[:5]
        extra = sorted(declared_b - recomputed_b)[:5]
        raise ReviewPacketContractError(
            f"arm_b-only reference set disagrees with recomputation: missing={missing} extra={extra}"
        )

    refs = arm_a_refs + arm_b_refs
    if expected_total_refs is not None and len(refs) != expected_total_refs:
        raise ReviewPacketContractError(
            f"Phase Zero ledger contains {len(refs)} changed-owner references, "
            f"expected {expected_total_refs}"
        )
    for ref in refs:
        if ref.row_id not in arm_a.rows:
            raise ReviewPacketContractError(f"owner reference row is absent: {ref.row_id!r}")
        gt = _list(arm_a.rows[ref.row_id].get("gt"), f"row {ref.row_id!r}.gt")
        if ref.owner_index >= len(gt):
            raise ReviewPacketContractError(
                f"owner reference {ref.key!r} is outside the GT index domain"
            )
    return refs, threshold


def _raw_span(event: Mapping[str, Any], label: str) -> str:
    value = event.get("raw_span_text", event.get("raw_text"))
    return _nonempty_text(value, f"{label}.raw_span_text")


def _validate_prediction_provenance(
    row: Mapping[str, Any], *, row_id: str
) -> tuple[Mapping[str, Any], ...]:
    predictions = _list(row.get("pred"), f"row {row_id!r}.pred")
    dropped = _list(row.get("dropped_predictions", []), f"row {row_id!r}.dropped_predictions")
    parsed_predictions, invalid_count = _pred_objects(dict(row))
    if invalid_count or len(parsed_predictions) != len(predictions):
        raise ReviewPacketContractError(
            f"row {row_id!r} accepted prediction list cannot be mapped exactly to comparator indices"
        )
    declared_valid = _nonnegative_int(
        row.get("valid_prediction_count"), f"row {row_id!r}.valid_prediction_count"
    )
    declared_dropped = _nonnegative_int(
        row.get("dropped_prediction_count"), f"row {row_id!r}.dropped_prediction_count"
    )
    if declared_valid != len(predictions) or declared_dropped != len(dropped):
        raise ReviewPacketContractError(
            f"row {row_id!r} prediction counts do not match accepted/dropped payloads"
        )

    events: list[dict[str, Any]] = []
    for prediction_index, value in enumerate(predictions):
        event = dict(_mapping(value, f"row {row_id!r}.pred[{prediction_index}]"))
        event["_accepted_prediction_index"] = prediction_index
        event["_event_kind"] = "accepted_prediction"
        events.append(event)
    for dropped_index, value in enumerate(dropped):
        event = dict(_mapping(value, f"row {row_id!r}.dropped_predictions[{dropped_index}]"))
        event["_accepted_prediction_index"] = None
        event["_event_kind"] = "dropped_prediction_unresolved"
        events.append(event)

    by_order: dict[int, dict[str, Any]] = {}
    for event in events:
        order = _nonnegative_int(
            event.get("generated_order"), f"row {row_id!r}.generated_order"
        )
        if order in by_order:
            raise ReviewPacketContractError(
                f"row {row_id!r} has duplicate raw generated_order={order}"
            )
        expected_span_id = f"{row_id}:span-{order}"
        if event.get("object_span_id") != expected_span_id:
            raise ReviewPacketContractError(
                f"row {row_id!r} prediction-index provenance mismatch at raw order {order}"
            )
        raw_span = _raw_span(event, f"row {row_id!r}.event[{order}]")
        if event.get("raw_span_sha256") != _sha256_text(raw_span):
            raise ReviewPacketContractError(
                f"row {row_id!r} raw span digest mismatch at generated_order={order}"
            )
        char_start = _nonnegative_int(
            event.get("char_start"), f"row {row_id!r}.event[{order}].char_start"
        )
        char_end = _nonnegative_int(
            event.get("char_end"), f"row {row_id!r}.event[{order}].char_end"
        )
        if char_end <= char_start or char_end - char_start != len(raw_span):
            raise ReviewPacketContractError(
                f"row {row_id!r} raw span offsets mismatch at generated_order={order}"
            )
        event["_raw_span"] = raw_span
        by_order[order] = event

    ordered_events = [by_order[index] for index in range(len(events)) if index in by_order]
    if len(ordered_events) != len(events):
        raise ReviewPacketContractError(
            f"row {row_id!r} raw generated-order domain is not contiguous"
        )
    previous_end = 0
    for event in ordered_events:
        if event["char_start"] != previous_end:
            raise ReviewPacketContractError(
                f"row {row_id!r} raw output spans are not contiguous"
            )
        previous_end = int(event["char_end"])
    raw_decode_text = _nonempty_text(
        row.get("raw_decode_text"), f"row {row_id!r}.raw_decode_text"
    )
    reconstructed = "".join(str(event["_raw_span"]) for event in ordered_events)
    if raw_decode_text != reconstructed + TERMINAL_TEXT:
        raise ReviewPacketContractError(
            f"row {row_id!r} raw output cannot be reconstructed from prediction provenance"
        )
    return tuple(ordered_events)


def _validate_image_and_gt(row: Mapping[str, Any], *, row_id: str) -> dict[str, Any]:
    width, height = _dimensions(row, row_id)
    image_path = _image_path(row, row_id)
    try:
        with Image.open(image_path) as image:
            observed_size = image.size
    except OSError as exc:
        raise ReviewPacketContractError(f"cannot decode image for row {row_id!r}") from exc
    if observed_size != (width, height):
        raise ReviewPacketContractError(
            f"image provenance mismatch for row {row_id!r}: "
            f"artifact={(width, height)} file={observed_size}"
        )
    expected_stem = row_id.rsplit("_", 1)[-1]
    if image_path.stem != expected_stem:
        raise ReviewPacketContractError(
            f"image filename does not preserve row identity for row {row_id!r}"
        )
    objects = _list(row.get("gt"), f"row {row_id!r}.gt")
    object_ids: set[str] = set()
    for index, raw in enumerate(objects):
        obj = _mapping(raw, f"row {row_id!r}.gt[{index}]")
        object_id = _nonempty_text(obj.get("object_id"), f"row {row_id!r}.gt[{index}].object_id")
        if object_id in object_ids:
            raise ReviewPacketContractError(
                f"row {row_id!r} has duplicate GT physical-owner id {object_id!r}"
            )
        object_ids.add(object_id)
    try:
        _gt_objects(dict(row), row_id=row_id)
    except ValueError as exc:
        raise ReviewPacketContractError(f"row {row_id!r} GT geometry is invalid") from exc
    return {
        "path": image_path,
        "sha256": sha256_file(image_path),
        "width": width,
        "height": height,
    }


def _matches_for_row(
    row: Mapping[str, Any], *, row_id: str, threshold: float
) -> tuple[
    list[tuple[str, tuple[float, float, float, float]]],
    list[tuple[str, tuple[float, float, float, float]]],
    list[tuple[int, int, float]],
]:
    gt = _gt_objects(dict(row), row_id=row_id)
    pred, invalid = _pred_objects(dict(row))
    if invalid:
        raise ReviewPacketContractError(
            f"row {row_id!r} has invalid accepted predictions in matcher input"
        )
    return gt, pred, _global_matches(gt, pred, threshold)


def _visual_row(
    row: Mapping[str, Any], *, row_id: str, image_path: Path
) -> VisualRow:
    width, height = _dimensions(row, row_id)
    gt_values = _list(row.get("gt"), f"row {row_id!r}.gt")
    pred_values = _list(row.get("pred"), f"row {row_id!r}.pred")
    gt_parsed = _gt_objects(dict(row), row_id=row_id)
    pred_parsed, invalid = _pred_objects(dict(row))
    if invalid or len(pred_parsed) != len(pred_values):
        raise ReviewPacketContractError(
            f"row {row_id!r} cannot be represented by the shared visualization types"
        )
    gt = tuple(
        VisualObject(
            index=index,
            description=str(_mapping(raw, f"gt[{index}]").get("description", "")),
            normalized_description=gt_parsed[index][0],
            bbox_pixel_xyxy=gt_parsed[index][1],
            source_bbox=tuple(_bbox(raw)),
            source_coord_space="norm1000",
        )
        for index, raw in enumerate(gt_values)
    )
    pred = tuple(
        VisualObject(
            index=index,
            description=str(_mapping(raw, f"pred[{index}]").get("description", "")),
            normalized_description=pred_parsed[index][0],
            bbox_pixel_xyxy=pred_parsed[index][1],
            source_bbox=tuple(_bbox(raw)),
            source_coord_space="pixel",
            source_coord_bins=(
                tuple(raw["coord_bins"])
                if isinstance(raw, Mapping)
                and isinstance(raw.get("coord_bins"), (list, tuple))
                and len(raw["coord_bins"]) == 4
                else None
            ),
        )
        for index, raw in enumerate(pred_values)
    )
    return VisualRow(
        row_id=row_id,
        row_index=int(row["row_index"]),
        image_path=image_path,
        source_image_path=image_path.name,
        image_width=width,
        image_height=height,
        gt=gt,
        pred=pred,
    )


def _match_result(
    *, gt_count: int, pred_count: int, matches: Sequence[tuple[int, int, float]]
) -> MatchResult:
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_pred = {pred_index for _, pred_index, _ in matches}
    return MatchResult(
        matches=tuple(
            MatchPair(pred_index=pred_index, gt_index=gt_index, iou=float(overlap))
            for gt_index, pred_index, overlap in matches
        ),
        missing_gt_indices=tuple(index for index in range(gt_count) if index not in matched_gt),
        fp_pred_indices=tuple(index for index in range(pred_count) if index not in matched_pred),
        duplicate_candidates=(),
    )


def _intersects(
    box: tuple[float, float, float, float],
    crop_box: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = box
    left, top, right, bottom = crop_box
    return min(x2, right) > max(x1, left) and min(y2, bottom) > max(y1, top)


def _crop_visual_row(
    row: VisualRow,
    match: MatchResult,
    *,
    image_path: Path,
    crop_box: tuple[int, int, int, int],
) -> tuple[VisualRow, MatchResult]:
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    selected_gt = [obj for obj in row.gt if _intersects(obj.bbox_pixel_xyxy, crop_box)]
    selected_pred = [obj for obj in row.pred if _intersects(obj.bbox_pixel_xyxy, crop_box)]
    gt_index_map = {obj.index: index for index, obj in enumerate(selected_gt)}
    pred_index_map = {obj.index: index for index, obj in enumerate(selected_pred)}

    def shifted(obj: VisualObject, *, index: int, label: str) -> VisualObject:
        x1, y1, x2, y2 = obj.bbox_pixel_xyxy
        return VisualObject(
            index=index,
            description=f"{obj.description} [{label}{obj.index}]",
            normalized_description=obj.normalized_description,
            bbox_pixel_xyxy=(
                max(0.0, x1 - left),
                max(0.0, y1 - top),
                min(float(crop_width), x2 - left),
                min(float(crop_height), y2 - top),
            ),
            source_bbox=obj.source_bbox,
            source_coord_space=obj.source_coord_space,
            source_coord_bins=obj.source_coord_bins,
        )

    cropped_matches = tuple(
        MatchPair(
            gt_index=gt_index_map[pair.gt_index],
            pred_index=pred_index_map[pair.pred_index],
            iou=pair.iou,
        )
        for pair in match.matches
        if pair.gt_index in gt_index_map and pair.pred_index in pred_index_map
    )
    matched_gt = {pair.gt_index for pair in cropped_matches}
    matched_pred = {pair.pred_index for pair in cropped_matches}
    cropped_row = VisualRow(
        row_id=row.row_id,
        row_index=row.row_index,
        image_path=image_path,
        source_image_path=image_path.name,
        image_width=crop_width,
        image_height=crop_height,
        gt=tuple(
            shifted(obj, index=index, label="ref G")
            for index, obj in enumerate(selected_gt)
        ),
        pred=tuple(
            shifted(obj, index=index, label="pred P")
            for index, obj in enumerate(selected_pred)
        ),
    )
    cropped_match = MatchResult(
        matches=cropped_matches,
        missing_gt_indices=tuple(
            index for index in range(len(selected_gt)) if index not in matched_gt
        ),
        fp_pred_indices=tuple(
            index for index in range(len(selected_pred)) if index not in matched_pred
        ),
        duplicate_candidates=(),
    )
    return cropped_row, cropped_match


def _crop_box(
    *,
    target_box: tuple[float, float, float, float],
    target_prediction_boxes: Iterable[tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
    halo: int,
) -> tuple[int, int, int, int]:
    boxes = [target_box, *target_prediction_boxes]
    left = max(0, math.floor(min(box[0] for box in boxes) - halo))
    top = max(0, math.floor(min(box[1] for box in boxes) - halo))
    right = min(image_width, math.ceil(max(box[2] for box in boxes) + halo))
    bottom = min(image_height, math.ceil(max(box[3] for box in boxes) + halo))
    if right <= left or bottom <= top:
        raise ReviewPacketContractError("target crop is empty")
    return left, top, right, bottom


def _attribution_payload(
    *,
    row: Mapping[str, Any],
    row_id: str,
    gt: Sequence[tuple[str, tuple[float, float, float, float]]],
    pred: Sequence[tuple[str, tuple[float, float, float, float]]],
    matches: Sequence[tuple[int, int, float]],
    events: Sequence[Mapping[str, Any]],
    owner_index: int,
) -> dict[str, Any]:
    match_by_pred = {
        pred_index: (gt_index, float(overlap))
        for gt_index, pred_index, overlap in matches
    }
    target_matches = [
        (pred_index, float(overlap))
        for gt_index, pred_index, overlap in matches
        if gt_index == owner_index
    ]
    accepted_values = _list(row.get("pred"), f"row {row_id!r}.pred")
    predictions: list[dict[str, Any]] = []
    for prediction_index, raw in enumerate(accepted_values):
        value = _mapping(raw, f"row {row_id!r}.pred[{prediction_index}]")
        matched = match_by_pred.get(prediction_index)
        official_attribution: dict[str, Any]
        if matched is None:
            official_attribution = {"status": "unmatched_unresolved"}
        else:
            official_attribution = {
                "status": "matched",
                "reference_index": matched[0],
                "intersection_over_union": matched[1],
            }
        predictions.append(
            {
                "prediction_index": prediction_index,
                "raw_output_order": int(value["generated_order"]),
                "category": str(value.get("description", "")),
                "normalized_category": pred[prediction_index][0],
                "bbox_pixel_xyxy": list(pred[prediction_index][1]),
                "bbox_norm1000": list(value.get("coord_bins", [])),
                "raw_span_text": _raw_span(value, f"pred[{prediction_index}]"),
                "official_attribution": official_attribution,
            }
        )

    output_events: list[dict[str, Any]] = []
    for event in events:
        accepted_index = event.get("_accepted_prediction_index")
        item: dict[str, Any] = {
            "raw_output_order": int(event["generated_order"]),
            "status": str(event["_event_kind"]),
            "raw_span_text": str(event["_raw_span"]),
        }
        if accepted_index is not None:
            item["prediction_index"] = int(accepted_index)
        else:
            item["drop_reason"] = str(event.get("reason", "unresolved"))
        output_events.append(item)

    matched_pred_indices = {pred_index for _, pred_index, _ in matches}
    matched_gt_indices = {gt_index for gt_index, _, _ in matches}
    target_payload: dict[str, Any]
    if target_matches:
        if len(target_matches) != 1:
            raise ReviewPacketContractError(
                f"row {row_id!r} target owner has non-unique global attribution"
            )
        target_payload = {
            "status": "matched",
            "prediction_index": target_matches[0][0],
            "intersection_over_union": target_matches[0][1],
        }
    else:
        target_payload = {"status": "unmatched_unresolved"}
    return {
        "prediction_index_domain": "accepted_prediction_list_index",
        "raw_output_order_domain": "generated_order_in_raw_decode_text",
        "predictions": predictions,
        "raw_output_events": output_events,
        "raw_output_text": str(row["raw_decode_text"]),
        "official_attribution": {
            "method": "cardinality_first_maximum_total_intersection_over_union",
            "threshold": 0.50,
            "matches": [
                {
                    "reference_index": gt_index,
                    "prediction_index": pred_index,
                    "intersection_over_union": float(overlap),
                }
                for gt_index, pred_index, overlap in matches
            ],
            "missing_reference_indices": [
                index for index in range(len(gt)) if index not in matched_gt_indices
            ],
            "unmatched_prediction_indices": [
                index for index in range(len(pred)) if index not in matched_pred_indices
            ],
            "unmatched_prediction_semantics": (
                "unresolved pending human review; not an automatic hallucination label"
            ),
        },
        "target_reference_attribution": target_payload,
    }


def _reference_payload(
    row: Mapping[str, Any],
    *,
    row_id: str,
    gt: Sequence[tuple[str, tuple[float, float, float, float]]],
) -> list[dict[str, Any]]:
    raw_gt = _list(row.get("gt"), f"row {row_id!r}.gt")
    return [
        {
            "reference_index": index,
            "physical_owner_id": str(_mapping(raw, f"gt[{index}]")["object_id"]),
            "category": str(_mapping(raw, f"gt[{index}]").get("description", "")),
            "normalized_category": gt[index][0],
            "bbox_norm1000": list(_bbox(raw)),
            "bbox_pixel_xyxy": list(gt[index][1]),
        }
        for index, raw in enumerate(raw_gt)
    ]


def _select_references(
    refs: Sequence[OwnerReference],
    requested: Sequence[tuple[str, int]] | None,
) -> tuple[OwnerReference, ...]:
    by_key = {ref.key: ref for ref in refs}
    if requested is None:
        return tuple(refs)
    if not requested:
        raise ReviewPacketContractError("case reference selection is empty")
    if len(set(requested)) != len(requested):
        raise ReviewPacketContractError("case reference selection contains duplicates")
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ReviewPacketContractError(
            f"requested case references are absent from the recomputed ledger: {missing}"
        )
    return tuple(by_key[key] for key in requested)


def _blind_order(
    refs: Sequence[OwnerReference], *, ledger_sha256: str
) -> list[tuple[str, OwnerReference, bool]]:
    decorated: list[tuple[str, OwnerReference, bool]] = []
    for ref in refs:
        digest = hashlib.sha256(
            f"{ledger_sha256}|{ref.row_id}|{ref.owner_index}".encode("utf-8")
        ).hexdigest()
        arm_a_is_view_a = int(digest[-2:], 16) % 2 == 0
        decorated.append((digest, ref, arm_a_is_view_a))
    decorated.sort(key=lambda item: item[0])
    return [
        (f"case_{index:04d}", ref, arm_a_is_view_a)
        for index, (_, ref, arm_a_is_view_a) in enumerate(decorated, start=1)
    ]


def _assert_reviewer_blind(reviewer_dir: Path) -> None:
    for path in sorted(reviewer_dir.rglob("*")):
        if path.suffix.lower() not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        match = _REVIEWER_LEAK_PATTERN.search(text)
        if match is not None:
            raise ReviewPacketContractError(
                f"reviewer-visible checkpoint or direction leak in {path.name}: {match.group(0)!r}"
            )


def build_packet(
    comparison_ledger_path: Path,
    output_dir: Path,
    *,
    case_refs: Sequence[tuple[str, int]] | None = None,
    crop_halo_px: int = DEFAULT_CROP_HALO_PX,
    expected_total_refs: int | None = None,
) -> dict[str, Any]:
    """Validate inputs and materialize a deterministic, immutable packet."""

    if isinstance(crop_halo_px, bool) or not isinstance(crop_halo_px, int) or crop_halo_px < 0:
        raise ReviewPacketContractError("crop_halo_px must be a non-negative integer")
    comparison_ledger_path = comparison_ledger_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ReviewPacketContractError(
            f"output directory already exists; packet outputs are immutable: {output_dir}"
        )
    ledger_sha256 = sha256_file(comparison_ledger_path)
    ledger = _read_json(comparison_ledger_path, "comparison ledger")
    inputs = _mapping(ledger.get("inputs"), "inputs")
    arm_a = _load_artifact(inputs.get("arm_a"), "inputs.arm_a")
    arm_b = _load_artifact(inputs.get("arm_b"), "inputs.arm_b")
    _validate_arm_pair(arm_a, arm_b)
    refs, threshold = _validate_reference_ledger(
        ledger,
        arm_a,
        arm_b,
        expected_total_refs=expected_total_refs,
    )
    selected_refs = _select_references(refs, case_refs)

    # Validate every selected row completely before writing any output.
    selected_row_ids = sorted({ref.row_id for ref in selected_refs})
    image_provenance: dict[str, dict[str, Any]] = {}
    event_provenance: dict[tuple[str, str], tuple[Mapping[str, Any], ...]] = {}
    for row_id in selected_row_ids:
        left = arm_a.rows[row_id]
        right = arm_b.rows[row_id]
        image_provenance[row_id] = _validate_image_and_gt(left, row_id=row_id)
        _validate_image_and_gt(right, row_id=row_id)
        event_provenance[("arm_a", row_id)] = _validate_prediction_provenance(
            left, row_id=row_id
        )
        event_provenance[("arm_b", row_id)] = _validate_prediction_provenance(
            right, row_id=row_id
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    reviewer_dir = output_dir / "reviewer"
    private_dir = output_dir / "private"
    reviewer_cases_dir = reviewer_dir / "cases"
    reviewer_images_dir = reviewer_dir / "images"
    reviewer_cases_dir.mkdir(parents=True)
    reviewer_images_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    private_ledger_path = private_dir / "original_comparison_ledger.json"
    private_ledger_path.write_bytes(comparison_ledger_path.read_bytes())

    reviewer_manifest_cases: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    unblinding_cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="heldout-owner-change-review-") as temp_value:
        temp_dir = Path(temp_value)
        for case_id, ref, arm_a_is_view_a in _blind_order(
            selected_refs, ledger_sha256=ledger_sha256
        ):
            left_row_raw = arm_a.rows[ref.row_id]
            right_row_raw = arm_b.rows[ref.row_id]
            image_info = image_provenance[ref.row_id]
            image_path = Path(image_info["path"])
            left_gt, left_pred, left_matches = _matches_for_row(
                left_row_raw,
                row_id=ref.row_id,
                threshold=threshold,
            )
            right_gt, right_pred, right_matches = _matches_for_row(
                right_row_raw,
                row_id=ref.row_id,
                threshold=threshold,
            )
            if left_gt != right_gt:
                raise ReviewPacketContractError(
                    f"normalized GT mismatch for row {ref.row_id!r}"
                )

            left_visual = _visual_row(
                left_row_raw, row_id=ref.row_id, image_path=image_path
            )
            right_visual = _visual_row(
                right_row_raw, row_id=ref.row_id, image_path=image_path
            )
            left_match_result = _match_result(
                gt_count=len(left_gt), pred_count=len(left_pred), matches=left_matches
            )
            right_match_result = _match_result(
                gt_count=len(right_gt), pred_count=len(right_pred), matches=right_matches
            )
            if arm_a_is_view_a:
                view_a_row, view_b_row = left_visual, right_visual
                view_a_match, view_b_match = left_match_result, right_match_result
                view_a_raw, view_b_raw = left_row_raw, right_row_raw
                view_a_pred, view_b_pred = left_pred, right_pred
                view_a_matches, view_b_matches = left_matches, right_matches
                view_a_events = event_provenance[("arm_a", ref.row_id)]
                view_b_events = event_provenance[("arm_b", ref.row_id)]
                view_a_arm, view_b_arm = "arm_a", "arm_b"
            else:
                view_a_row, view_b_row = right_visual, left_visual
                view_a_match, view_b_match = right_match_result, left_match_result
                view_a_raw, view_b_raw = right_row_raw, left_row_raw
                view_a_pred, view_b_pred = right_pred, left_pred
                view_a_matches, view_b_matches = right_matches, left_matches
                view_a_events = event_provenance[("arm_b", ref.row_id)]
                view_b_events = event_provenance[("arm_a", ref.row_id)]
                view_a_arm, view_b_arm = "arm_b", "arm_a"

            target_box = left_gt[ref.owner_index][1]
            target_prediction_boxes = [
                left_pred[pred_index][1]
                for gt_index, pred_index, _ in left_matches
                if gt_index == ref.owner_index
            ] + [
                right_pred[pred_index][1]
                for gt_index, pred_index, _ in right_matches
                if gt_index == ref.owner_index
            ]
            crop_box = _crop_box(
                target_box=target_box,
                target_prediction_boxes=target_prediction_boxes,
                image_width=int(image_info["width"]),
                image_height=int(image_info["height"]),
                halo=crop_halo_px,
            )
            crop_input_path = temp_dir / f"{case_id}.crop.png"
            with Image.open(image_path) as original:
                original.convert("RGB").crop(crop_box).save(crop_input_path, format="PNG")

            full_path = reviewer_images_dir / f"{case_id}.full.png"
            crop_path = reviewer_images_dir / f"{case_id}.crop.png"
            target_category = left_gt[ref.owner_index][0]
            title = f"OWNER REVIEW | {case_id} | target={target_category}"
            render_comparison_png(
                row=view_a_row,
                right_row=view_b_row,
                left_match=view_a_match,
                right_match=view_b_match,
                output_path=full_path,
                title=title,
                left_label="View A",
                right_label="View B",
            )
            cropped_view_a, cropped_match_a = _crop_visual_row(
                view_a_row,
                view_a_match,
                image_path=crop_input_path,
                crop_box=crop_box,
            )
            cropped_view_b, cropped_match_b = _crop_visual_row(
                view_b_row,
                view_b_match,
                image_path=crop_input_path,
                crop_box=crop_box,
            )
            render_comparison_png(
                row=cropped_view_a,
                right_row=cropped_view_b,
                left_match=cropped_match_a,
                right_match=cropped_match_b,
                output_path=crop_path,
                title=title + " | enlarged crop",
                left_label="View A",
                right_label="View B",
            )

            view_a_payload = _attribution_payload(
                row=view_a_raw,
                row_id=ref.row_id,
                gt=left_gt,
                pred=view_a_pred,
                matches=view_a_matches,
                events=view_a_events,
                owner_index=ref.owner_index,
            )
            view_b_payload = _attribution_payload(
                row=view_b_raw,
                row_id=ref.row_id,
                gt=left_gt,
                pred=view_b_pred,
                matches=view_b_matches,
                events=view_b_events,
                owner_index=ref.owner_index,
            )
            case_path = reviewer_cases_dir / f"{case_id}.json"
            case_payload = {
                "schema_version": SCHEMA_VERSION + ".blind_case",
                "case_id": case_id,
                "blinded": True,
                "image": {
                    "sha256": image_info["sha256"],
                    "width": image_info["width"],
                    "height": image_info["height"],
                },
                "reference_objects": _reference_payload(
                    left_row_raw, row_id=ref.row_id, gt=left_gt
                ),
                "target_reference": {
                    "category": target_category,
                    "bbox_norm1000": list(
                        _bbox(_list(left_row_raw.get("gt"), "gt")[ref.owner_index])
                    ),
                    "bbox_pixel_xyxy": list(target_box),
                },
                "views": {"view_a": view_a_payload, "view_b": view_b_payload},
                "artifacts": {
                    "full_image": f"images/{full_path.name}",
                    "full_image_sha256": sha256_file(full_path),
                    "enlarged_crop": f"images/{crop_path.name}",
                    "enlarged_crop_sha256": sha256_file(crop_path),
                    "crop_pixel_xyxy": list(crop_box),
                },
            }
            _write_json(case_path, case_payload)
            reviewer_manifest_cases.append(
                {
                    "case_id": case_id,
                    "case_file": f"cases/{case_path.name}",
                    "case_file_sha256": sha256_file(case_path),
                    "full_image": f"images/{full_path.name}",
                    "full_image_sha256": sha256_file(full_path),
                    "enlarged_crop": f"images/{crop_path.name}",
                    "enlarged_crop_sha256": sha256_file(crop_path),
                }
            )
            dispositions.append(
                {
                    "case_id": case_id,
                    "entity_category": "pending",
                    "geometry": "pending",
                    "notes": None,
                }
            )

            by_arm = {
                "arm_a": {
                    "matches": left_matches,
                    "target": _attribution_payload(
                        row=left_row_raw,
                        row_id=ref.row_id,
                        gt=left_gt,
                        pred=left_pred,
                        matches=left_matches,
                        events=event_provenance[("arm_a", ref.row_id)],
                        owner_index=ref.owner_index,
                    )["target_reference_attribution"],
                },
                "arm_b": {
                    "matches": right_matches,
                    "target": _attribution_payload(
                        row=right_row_raw,
                        row_id=ref.row_id,
                        gt=right_gt,
                        pred=right_pred,
                        matches=right_matches,
                        events=event_provenance[("arm_b", ref.row_id)],
                        owner_index=ref.owner_index,
                    )["target_reference_attribution"],
                },
            }
            unblinding_cases.append(
                {
                    "case_id": case_id,
                    "row_id": ref.row_id,
                    "owner_index": ref.owner_index,
                    "ledger_reference_side": ref.ledger_side,
                    "physical_owner_change_direction": (
                        "loss" if ref.ledger_side == "arm_a_only" else "gain"
                    ),
                    "views": {"view_a": view_a_arm, "view_b": view_b_arm},
                    "authoritative_attribution": by_arm,
                }
            )

    disposition_path = reviewer_dir / "dispositions.jsonl"
    disposition_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for item in dispositions
        ),
        encoding="utf-8",
    )
    instructions_path = reviewer_dir / "README.md"
    instructions_path.write_text(
        "# Held-out physical-owner change review\n\n"
        "Inspect each full view and enlarged crop without inferring checkpoint identity. "
        "View A and View B are deterministically shuffled per case.\n\n"
        "Record one entity/category disposition and one geometry disposition per case. "
        "Officially unmatched objects remain unresolved until human review; the automated "
        "match does not establish hallucination.\n\n"
        "Enlarged-crop labels use compact panel-local indices. Bracketed `ref G...` and "
        "`pred P...` labels preserve the corresponding full-view indices.\n\n"
        "Use `dispositions.jsonl` as the fill-in template. Do not change `case_id`.\n",
        encoding="utf-8",
    )
    reviewer_manifest = {
        "schema_version": SCHEMA_VERSION + ".blind_manifest",
        "blinded": True,
        "case_count": len(reviewer_manifest_cases),
        "hidden_fields": ["checkpoint_identity", "physical_owner_change_direction"],
        "official_matching": {
            "method": "cardinality_first_maximum_total_intersection_over_union",
            "threshold": threshold,
            "unmatched_prediction_semantics": "unresolved_pending_human_review",
        },
        "review_schema": {
            "entity_category": {
                "required": True,
                "allowed_values": list(ENTITY_CATEGORY_OPTIONS),
            },
            "geometry": {
                "required": True,
                "allowed_values": list(GEOMETRY_OPTIONS),
            },
            "notes": {"required": False, "type": "string_or_null"},
        },
        "instructions": instructions_path.name,
        "instructions_sha256": sha256_file(instructions_path),
        "disposition_template": disposition_path.name,
        "disposition_template_sha256": sha256_file(disposition_path),
        "cases": reviewer_manifest_cases,
    }
    reviewer_manifest_path = reviewer_dir / "manifest.json"
    _write_json(reviewer_manifest_path, reviewer_manifest)
    _assert_reviewer_blind(reviewer_dir)

    unblinding = {
        "schema_version": SCHEMA_VERSION + ".private_unblinding",
        "comparison_ledger": {
            "path": str(comparison_ledger_path),
            "sha256": ledger_sha256,
            "private_copy": private_ledger_path.name,
            "private_copy_sha256": sha256_file(private_ledger_path),
        },
        "checkpoint_roles": {
            "arm_a": "Source",
            "arm_b": "transition step 36",
        },
        "inputs": {
            "arm_a": {"path": str(arm_a.path), "sha256": arm_a.sha256},
            "arm_b": {"path": str(arm_b.path), "sha256": arm_b.sha256},
        },
        "attribution": {
            "implementation": (
                "scripts.research.compare_clean_rollout_owner_coverage._global_matches"
            ),
            "threshold": threshold,
            "prediction_index_domain": "accepted_prediction_list_index",
            "raw_output_order_domain": "generated_order_in_raw_decode_text",
        },
        "reviewer_manifest": {
            "path": "../reviewer/manifest.json",
            "sha256": sha256_file(reviewer_manifest_path),
        },
        "cases": unblinding_cases,
    }
    unblinding_path = private_dir / "unblinding.json"
    _write_json(unblinding_path, unblinding)
    receipt = {
        "schema_version": SCHEMA_VERSION + ".receipt",
        "output_dir": str(output_dir),
        "selected_case_count": len(selected_refs),
        "ledger_reference_count": len(refs),
        "reviewer_manifest": str(reviewer_manifest_path),
        "reviewer_manifest_sha256": sha256_file(reviewer_manifest_path),
        "private_unblinding": str(unblinding_path),
        "private_unblinding_sha256": sha256_file(unblinding_path),
    }
    _write_json(private_dir / "build_receipt.json", receipt)
    return receipt


def parse_case_ref(value: str) -> tuple[str, int]:
    row_id, separator, owner_text = value.rpartition(":")
    if not separator or not row_id or not owner_text:
        raise argparse.ArgumentTypeError("case reference must be ROW_ID:OWNER_INDEX")
    try:
        owner_index = int(owner_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("owner index must be an integer") from exc
    if owner_index < 0:
        raise argparse.ArgumentTypeError("owner index must be non-negative")
    return row_id, owner_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case-ref",
        action="append",
        type=parse_case_ref,
        default=None,
        metavar="ROW_ID:OWNER_INDEX",
        help="Build only selected recomputed references; repeat for a bounded smoke.",
    )
    parser.add_argument(
        "--crop-halo-px", type=int, default=DEFAULT_CROP_HALO_PX
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_packet(
            args.comparison_ledger,
            args.output_dir,
            case_refs=args.case_ref,
            crop_halo_px=args.crop_halo_px,
            expected_total_refs=EXPECTED_PHASE_ZERO_REFERENCE_COUNT,
        )
    except (ReviewPacketContractError, OSError) as exc:
        raise SystemExit(f"held-out owner-change review packet failed: {exc}") from exc
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
