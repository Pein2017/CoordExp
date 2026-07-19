#!/usr/bin/env python3
"""Build deterministic same-coverage prefix permutation cases.

This is an experiment-local case builder for the existing
``same_covered_set_prefix_order.case.v2`` runner.  It uses only already saved
rollout artifacts: no model, tokenizer, active training data, or current
ground-truth authority is loaded.

For each requested image, predictions from a current geometry-sorted rollout
and a secondary rollout are matched to their artifact-local ground-truth rows.
The common physical rows are ordered by the current sorted rollout, then
several permutations of the same selected prefix are emitted.  The final two
rows are kept byte-identical across the two-row comparisons so the intervention
is about earlier prefix order, not the immediately preceding rows.  The case
document retains the complete artifact-local ground-truth entity ledger even
though each arm uses only a selected common prefix.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


OBJECT_REF_START = "<|object_ref_start|>"
OBJECT_REF_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"

CASE_SCHEMA_VERSION = "same_covered_set_prefix_order.case.v2"
DEFAULT_PREFIX_DEPTHS = (6, 10)
DEFAULT_SHARED_SUFFIX_LENGTH = 2
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SHUFFLE_SEED = 17
DEFAULT_ROLLOUT_HORIZON_ROWS = 1


def permutation_inversion_count(permutation: Sequence[str], canonical: Sequence[str]) -> int:
    """Return Kendall tau inversion count relative to ``canonical`` order.

    The count is the number of unordered entity pairs whose relative order is
    reversed by ``permutation``.  It is deterministic and does not require a
    score for how plausible either prefix is.
    """

    canonical_values = [str(value) for value in canonical]
    permutation_values = [str(value) for value in permutation]
    if len(canonical_values) != len(permutation_values) or set(canonical_values) != set(permutation_values):
        raise ValueError("permutation and canonical order must contain the same unique entities")
    positions = {entity_id: index for index, entity_id in enumerate(canonical_values)}
    canonical_positions = [positions[entity_id] for entity_id in permutation_values]
    return sum(
        1
        for left_index in range(len(canonical_positions))
        for right_index in range(left_index + 1, len(canonical_positions))
        if canonical_positions[left_index] > canonical_positions[right_index]
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _as_box(value: Any, *, label: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{label} must contain four numeric coordinates")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain four numeric coordinates") from exc
    if any(not math.isfinite(item) for item in box):
        raise ValueError(f"{label} contains a non-finite coordinate")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} must have positive area")
    return box


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    left_box = _as_box(left, label="left box")
    right_box = _as_box(right, label="right box")
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_area = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _normalise_description(value: Any) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("description must be non-empty")
    return value


def _image_keys(row: Mapping[str, Any]) -> set[str]:
    """Return conservative selectors for one artifact row."""

    values: set[str] = set()
    for field in ("image_id", "example_id", "row_id"):
        if row.get(field) is not None:
            value = str(row[field])
            values.add(value)
            values.add(value.lstrip("0") or "0")
    path_value = row.get("image_path")
    if path_value is None:
        path_value = row.get("image")
        if isinstance(path_value, Mapping):
            path_value = path_value.get("path", path_value.get("image_path"))
    if path_value:
        stem = Path(str(path_value)).stem
        values.add(stem)
        values.add(stem.lstrip("0") or "0")
    for value in tuple(values):
        match = re.search(r"(\d+)$", value)
        if match:
            digits = match.group(1)
            values.add(digits)
            values.add(digits.lstrip("0") or "0")
    return values


def _object_description(value: Mapping[str, Any], *, label: str) -> str:
    """Read canonical artifact ``desc`` with the newer ``description`` alias."""

    raw = value.get("desc", value.get("description", ""))
    return _normalise_description(raw)


def _object_box(value: Mapping[str, Any], *, label: str) -> tuple[float, float, float, float]:
    """Read canonical artifact ``points`` with the newer ``bbox`` aliases."""

    raw = value.get("points", value.get("bbox", value.get("bbox_xyxy")))
    return _as_box(raw, label=label)


def _artifact_dimensions(row: Mapping[str, Any]) -> tuple[int, int]:
    width = row.get("width", row.get("image_width"))
    height = row.get("height", row.get("image_height"))
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("artifact width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("artifact height must be a positive integer")
    return int(width), int(height)


def load_artifact_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows without imposing a current-data or GT contract."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"artifact {path} is empty")
    return rows


def select_artifact_row(rows: Sequence[Mapping[str, Any]], image_id: str, *, label: str) -> dict[str, Any]:
    requested = str(image_id)
    matches = [dict(row) for row in rows if requested in _image_keys(row)]
    if not matches:
        raise ValueError(f"{label} has no row for image_id {image_id!r}")
    if len(matches) != 1:
        raise ValueError(f"{label} has {len(matches)} rows for image_id {image_id!r}; expected one")
    return matches[0]


def _match_predictions_to_gt(
    row: Mapping[str, Any],
    *,
    iou_threshold: float,
) -> dict[int, dict[str, Any]]:
    """Return one-to-one GT matches using deterministic global greedy pairs."""

    gt_values = row.get("gt")
    predictions = row.get("pred")
    if not isinstance(gt_values, list) or not isinstance(predictions, list):
        raise ValueError("artifact row must contain list-valued gt and pred arrays")
    candidate_pairs: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            continue
        try:
            prediction_description = _object_description(prediction, label=f"pred[{prediction_index}].description")
            prediction_box = _object_box(prediction, label=f"pred[{prediction_index}].bbox")
        except ValueError:
            continue
        for gt_index, gt in enumerate(gt_values):
            if not isinstance(gt, Mapping):
                raise ValueError(f"gt[{gt_index}] must be a JSON object")
            if prediction_description != _object_description(gt, label=f"gt[{gt_index}].description"):
                continue
            score = intersection_over_union(prediction_box, _object_box(gt, label=f"gt[{gt_index}].bbox"))
            if score >= float(iou_threshold):
                candidate_pairs.append((-score, prediction_index, gt_index))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    matches: dict[int, dict[str, Any]] = {}
    for negative_iou, prediction_index, gt_index in candidate_pairs:
        if prediction_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(prediction_index)
        used_gt.add(gt_index)
        matches[gt_index] = {
            "prediction_index": prediction_index,
            "intersection_over_union": -negative_iou,
        }
    return matches


def _pixel_to_coordinate_bin(value: float, extent: int, *, label: str) -> int:
    if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
        raise ValueError(f"{label} image extent must be a positive integer")
    result = int(round(float(value) * 1000.0 / float(extent)))
    if not 0 <= result <= 999:
        raise ValueError(f"{label} maps to coordinate bin {result}, outside [0,999]")
    return result


def render_row_text(description: str, pixel_box: Sequence[float], *, image_width: int, image_height: int) -> tuple[str, list[int]]:
    box = _as_box(pixel_box, label="GT bbox")
    bins = [
        _pixel_to_coordinate_bin(box[0], image_width, label="x1"),
        _pixel_to_coordinate_bin(box[1], image_height, label="y1"),
        _pixel_to_coordinate_bin(box[2], image_width, label="x2"),
        _pixel_to_coordinate_bin(box[3], image_height, label="y2"),
    ]
    if bins[2] <= bins[0] or bins[3] <= bins[1]:
        raise ValueError(f"GT bbox {list(pixel_box)} collapses after coordinate normalization")
    row_text = (
        f"{OBJECT_REF_START}{_normalise_description(description)}{OBJECT_REF_END}"
        f"{BOX_START}{''.join(f'<|coord_{value}|>' for value in bins)}{BOX_END}"
    )
    return row_text, bins


def _entity_rows(sorted_row: Mapping[str, Any], *, gt_indices: Sequence[int]) -> dict[str, dict[str, Any]]:
    gt_values = sorted_row.get("gt")
    width, height = _artifact_dimensions(sorted_row)
    if not isinstance(gt_values, list):
        raise ValueError("sorted artifact gt must be a list")
    entities: dict[str, dict[str, Any]] = {}
    for gt_index in gt_indices:
        if not 0 <= int(gt_index) < len(gt_values):
            raise ValueError(f"GT index {gt_index} is unavailable")
        gt = gt_values[int(gt_index)]
        if not isinstance(gt, Mapping):
            raise ValueError(f"gt[{gt_index}] must be a JSON object")
        description = _object_description(gt, label=f"gt[{gt_index}].description")
        pixel_box = _object_box(gt, label=f"gt[{gt_index}].bbox")
        row_text, bins = render_row_text(description, pixel_box, image_width=width, image_height=height)
        entity_id = f"gt_{int(gt_index):04d}"
        entities[entity_id] = {
            "entity_id": entity_id,
            "description": description,
            "bbox_pixel_xyxy": [float(value) for value in pixel_box],
            "bbox_norm1000": bins,
            "row_text": row_text,
            "source_gt_index": int(gt_index),
            "source_object_id": gt.get("object_id"),
            "row_text_source": "artifact_local_gt_pixel_box_normalized_once",
        }
    return entities


def _permutation_arms(
    selected_ids: list[str],
    random_relative_order: Sequence[str],
    *,
    shuffle_seed: int,
    image_id: str,
    depth: int,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    suffix = selected_ids[-DEFAULT_SHARED_SUFFIX_LENGTH:]
    earlier = selected_ids[:-DEFAULT_SHARED_SUFFIX_LENGTH]
    if len(set(suffix)) != DEFAULT_SHARED_SUFFIX_LENGTH:
        raise ValueError("selected suffix must contain two distinct entities")
    random_order = [entity_id for entity_id in random_relative_order if entity_id in set(earlier)]
    if len(random_order) != len(earlier):
        raise ValueError("secondary rollout does not contain all selected earlier entities")
    if len(earlier) < 2:
        raise ValueError("selected prefix must have at least two rows before the fixed two-row suffix")
    fixed_final_one = list(reversed(selected_ids[:-1])) + [selected_ids[-1]]
    adjacent_swap_before_final_two = (
        list(earlier[:-2])
        + [earlier[-1], earlier[-2]]
        + list(suffix)
    )
    seeded = list(earlier)
    # Keep the requested seed deterministic, while avoiding the accidental
    # duplicate arm that a single pseudo-random draw can produce.  The retry
    # remains a seeded shuffle; a genuinely impossible permutation set still
    # fails below rather than silently dropping an arm.
    for attempt in range(100):
        seeded = list(earlier)
        random.Random(f"{shuffle_seed}:{image_id}:{depth}:seeded:{attempt}").shuffle(seeded)
        if seeded + list(suffix) not in (
            list(selected_ids),
            random_order + list(suffix),
            list(reversed(earlier)) + list(suffix),
            fixed_final_one,
            adjacent_swap_before_final_two,
        ):
            break
    candidate_arms = {
        "current_sorted_rollout_order": list(selected_ids),
        "historical_random_relative_order_with_fixed_final_two": random_order + list(suffix),
        "reverse_earlier_rows_with_fixed_final_two": list(reversed(earlier)) + list(suffix),
        "reverse_earlier_rows_with_fixed_final_one": fixed_final_one,
        "adjacent_swap_before_fixed_final_two": adjacent_swap_before_final_two,
        "seeded_shuffle_earlier_rows_with_fixed_final_two": seeded + list(suffix),
    }
    arms: dict[str, list[str]] = {}
    omitted_comparisons: list[dict[str, str]] = []
    for name, values in candidate_arms.items():
        if set(values) != set(selected_ids) or values[-2:] != suffix:
            if name != "reverse_earlier_rows_with_fixed_final_one":
                raise ValueError(f"arm {name} violates same coverage or shared suffix")
        if name == "reverse_earlier_rows_with_fixed_final_one" and values[-1:] != selected_ids[-1:]:
            raise ValueError(f"arm {name} does not preserve the fixed final row")
        duplicate_of = next((existing_name for existing_name, existing_values in arms.items() if existing_values == values), None)
        if duplicate_of is not None:
            omitted_comparisons.append({
                "arm_name": name,
                "duplicate_of": duplicate_of,
                "reason": "identical_row_order",
            })
            continue
        arms[name] = values
    return arms, omitted_comparisons


def build_case_for_rows(
    sorted_row: Mapping[str, Any],
    random_row: Mapping[str, Any],
    *,
    image_id: str,
    prefix_depths: Sequence[int] = DEFAULT_PREFIX_DEPTHS,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    rollout_horizon_rows: int = DEFAULT_ROLLOUT_HORIZON_ROWS,
    sorted_source_path: str = "<in-memory-sorted-artifact>",
    random_source_path: str = "<in-memory-secondary-artifact>",
    sorted_source_digest: str | None = None,
    random_source_digest: str | None = None,
) -> dict[str, Any]:
    """Build one v2 case from two artifact-local rows."""

    if sorted_row.get("gt") != random_row.get("gt"):
        raise ValueError("sorted and secondary artifact-local GT arrays must be exactly equal")
    sorted_width, sorted_height = _artifact_dimensions(sorted_row)
    secondary_width, secondary_height = _artifact_dimensions(random_row)
    if (sorted_width, sorted_height) != (secondary_width, secondary_height):
        raise ValueError("sorted and secondary artifact dimensions must be identical")
    if isinstance(rollout_horizon_rows, bool) or not 1 <= int(rollout_horizon_rows) <= 4:
        raise ValueError("rollout_horizon_rows must be in [1,4]")
    depths = [int(value) for value in prefix_depths]
    if not depths or len(set(depths)) != len(depths) or any(value < 3 for value in depths):
        raise ValueError("prefix_depths must be unique integers at least 3")

    sorted_matches = _match_predictions_to_gt(sorted_row, iou_threshold=iou_threshold)
    random_matches = _match_predictions_to_gt(random_row, iou_threshold=iou_threshold)
    sorted_order = [gt_index for gt_index, _ in sorted(sorted_matches.items(), key=lambda item: item[1]["prediction_index"])]
    random_order = [gt_index for gt_index, _ in sorted(random_matches.items(), key=lambda item: item[1]["prediction_index"])]
    common_indices = [gt_index for gt_index in sorted_order if gt_index in random_matches]
    if not common_indices:
        raise ValueError(f"no common matched GT entities for image {image_id}")
    random_entity_order = [f"gt_{gt_index:04d}" for gt_index in random_order]
    gt_values = sorted_row.get("gt")
    if not isinstance(gt_values, list):
        raise ValueError("sorted artifact gt must be a list")
    # Keep the complete artifact-local ledger available for matching rollout
    # predictions.  Arms still reference only the selected common prefix; the
    # full ledger must not be mistaken for the covered set.
    entities = _entity_rows(sorted_row, gt_indices=range(len(gt_values)))
    all_entity_ids = list(entities)
    cases: list[dict[str, Any]] = []
    for depth in depths:
        if depth > len(common_indices):
            raise ValueError(f"prefix depth {depth} unavailable; only {len(common_indices)} common matched entities")
        selected_ids = [f"gt_{gt_index:04d}" for gt_index in common_indices[:depth]]
        if len(selected_ids) < DEFAULT_SHARED_SUFFIX_LENGTH:
            raise ValueError("selected entity set must contain the two-row suffix")
        arms, omitted_comparisons = _permutation_arms(
            selected_ids,
            random_entity_order,
            shuffle_seed=shuffle_seed,
            image_id=str(image_id),
            depth=depth,
        )
        comparison_names = [name for name in arms if name != "current_sorted_rollout_order"]
        arm_records = {
            name: {
                "entity_ids": values,
                "permutation_inversion_count_relative_to_canonical_order": permutation_inversion_count(
                    values, selected_ids
                ),
            }
            for name, values in arms.items()
        }
        comparisons = [
            {
                "comparison_id": f"current_sorted_rollout_order_vs_{name}",
                "arm_names": ["current_sorted_rollout_order", name],
                "require_same_covered_set": True,
                "shared_suffix_length": 1 if name == "reverse_earlier_rows_with_fixed_final_one" else DEFAULT_SHARED_SUFFIX_LENGTH,
                "rollout_horizon_rows": int(rollout_horizon_rows),
                "comparison_meaning": (
                    "Compare earlier prefix order while keeping the same physical rows and final one row."
                    if name == "reverse_earlier_rows_with_fixed_final_one"
                    else "Compare earlier prefix order while keeping the same physical rows and final two rows."
                ),
            }
            for name in comparison_names
        ]
        cases.append({
            "case_id": f"common_objects_depth_{depth}",
            "selected_common_entity_ids": selected_ids,
            "selected_common_gt_indices": [int(entity_id[3:]) for entity_id in selected_ids],
            "entity_ledger_scope": "full_artifact_local_gt",
            "entity_ledger_entity_count": len(all_entity_ids),
            "arms": arm_records,
            "comparisons": comparisons,
            "omitted_comparisons": omitted_comparisons,
            "depth_meaning": "Number of common physical entities included in the covered prefix.",
        })

    payload = {
        "schema_version": CASE_SCHEMA_VERSION,
        "image_id": str(image_id),
        "entities": list(entities.values()),
        "cases": cases,
        "metadata": {
            "experiment_name": "Common-object prefix permutation with a fixed final two-row suffix",
            "experiment_meaning": "Test whether earlier prefix order changes the next-row rollout when the same physical objects have already been covered.",
            "cohort_status": "artifact-local provisional cohort; selected from the intersection of two rollout matches; not current relabeled ground truth",
            "current_sorted_rollout_meaning": "Current geometry-sorted pure cross-entropy rollout used as the physical-order reference.",
            "secondary_rollout_meaning": "Historical or secondary rollout used only to provide an independent relative order for common matched entities.",
            "matching_rule": "Same description and intersection over union at or above threshold, globally greedily assigned by descending intersection over union, then prediction index, then ground-truth index.",
            "same_coverage_rule": "Every compared arm contains the same selected physical entities and the same byte-identical final one- or two-row suffix declared by its comparison.",
            "entity_ledger_scope": "full_artifact_local_gt",
            "entity_ledger_meaning": "The entities list is the complete artifact-local COCO-80 ground-truth set; each arm entity_ids list is only the selected common covered prefix.",
            "full_artifact_local_gt_entity_count": len(all_entity_ids),
            "selected_prefix_entity_counts_by_depth": {
                f"common_objects_depth_{depth}": depth for depth in depths
            },
            "selected_prefix_is_a_subset_of_full_entity_ledger": True,
            "prefix_likelihood_scoring": {
                "status": "unavailable_in_this_pilot",
                "meaning": "No prefix likelihood or plausibility score was computed; permutations are evaluated only by exact entity order and declared comparisons.",
            },
            "permutation_distance_metric": {
                "metric_full_name": "Kendall tau inversion count relative to canonical order",
                "operational_meaning": "For each arm, count unordered pairs of selected entities whose relative order is reversed compared with current_sorted_rollout_order; zero means identical order.",
                "per_arm_field": "permutation_inversion_count_relative_to_canonical_order",
            },
            "omitted_comparison_rule": {
                "reason_value": "identical_row_order",
                "meaning": "If a predefined permutation is byte-identical in entity order to an earlier arm for one image and depth, omit only that redundant comparison and record its duplicate arm.",
            },
            "prefix_depths": depths,
            "intersection_over_union_threshold": float(iou_threshold),
            "shuffle_seed": int(shuffle_seed),
            "rollout_horizon_rows": int(rollout_horizon_rows),
            "shared_suffix_length": DEFAULT_SHARED_SUFFIX_LENGTH,
            "source_artifacts": {
                "current_sorted_rollout_jsonl": str(sorted_source_path),
                "secondary_rollout_jsonl": str(random_source_path),
                "current_sorted_rollout_file_sha256": sorted_source_digest,
                "secondary_rollout_file_sha256": random_source_digest,
                "digest_meaning": "SHA-256 cryptographic digest of the complete source JSONL file when a filesystem path was supplied.",
            },
            "source_artifact_row_sha256": {
                "current_sorted_rollout_row": sha256_json(dict(sorted_row)),
                "secondary_rollout_row": sha256_json(dict(random_row)),
                "digest_meaning": "SHA-256 cryptographic digest of the exact selected artifact row.",
            },
            "source_gt_arrays_are_exactly_equal": True,
            "common_matched_gt_indices": [int(value) for value in common_indices],
            "current_sorted_matched_gt_order": [int(value) for value in sorted_order],
            "secondary_matched_gt_order": [int(value) for value in random_order],
            "image_width": sorted_width,
            "image_height": sorted_height,
        },
    }
    return payload


def _safe_filename(image_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id)).strip(".")
    if not value:
        raise ValueError("image_id cannot become an empty filename")
    return value + ".json"


def build_cases(
    sorted_path: Path,
    secondary_path: Path,
    *,
    image_ids: Sequence[str],
    output_directory: Path,
    prefix_depths: Sequence[int] = DEFAULT_PREFIX_DEPTHS,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    rollout_horizon_rows: int = DEFAULT_ROLLOUT_HORIZON_ROWS,
) -> list[Path]:
    sorted_rows = load_artifact_rows(sorted_path)
    secondary_rows = load_artifact_rows(secondary_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for image_id in image_ids:
        current = select_artifact_row(sorted_rows, image_id, label="current sorted artifact")
        secondary = select_artifact_row(secondary_rows, image_id, label="secondary artifact")
        payload = build_case_for_rows(
            current,
            secondary,
            image_id=str(image_id),
            prefix_depths=prefix_depths,
            iou_threshold=iou_threshold,
            shuffle_seed=shuffle_seed,
            rollout_horizon_rows=rollout_horizon_rows,
            sorted_source_path=str(sorted_path.resolve()),
            random_source_path=str(secondary_path.resolve()),
            sorted_source_digest=sha256_file(sorted_path),
            random_source_digest=sha256_file(secondary_path),
        )
        output_path = output_directory / _safe_filename(str(image_id))
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(output_path)
    return written


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not values:
        raise ValueError("value must contain at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-sorted-rollout", type=Path, required=True)
    parser.add_argument("--secondary-rollout", type=Path, required=True)
    parser.add_argument("--image-ids", required=True, help="Comma-separated image identifiers")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prefix-depths", default=",".join(map(str, DEFAULT_PREFIX_DEPTHS)))
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--rollout-horizon-rows", type=int, default=DEFAULT_ROLLOUT_HORIZON_ROWS)
    args = parser.parse_args()
    image_ids = tuple(piece.strip() for piece in args.image_ids.split(",") if piece.strip())
    if not image_ids:
        raise SystemExit("--image-ids must contain at least one identifier")
    written = build_cases(
        args.current_sorted_rollout,
        args.secondary_rollout,
        image_ids=image_ids,
        output_directory=args.output_directory,
        prefix_depths=_parse_int_list(args.prefix_depths),
        iou_threshold=args.iou_threshold,
        shuffle_seed=args.shuffle_seed,
        rollout_horizon_rows=args.rollout_horizon_rows,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
