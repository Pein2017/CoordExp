"""Deterministically reduce saved Lane A production cells; never runs a model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from probes.training_set_completion import readback_selectors


EOS_TOKEN_ID = 151645
IOU_THRESHOLD = 0.5
SCHEMA = "address_readout_pilot.production_reduction.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    data = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"reduction output collision: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _condition_name(condition: Any) -> str:
    if isinstance(condition, str):
        return condition
    if isinstance(condition, Mapping):
        for key in ("name", "condition", "id"):
            if condition.get(key) is not None:
                return str(condition[key])
    raise ValueError(f"condition lacks a stable name: {condition!r}")


def _case_key(case: Mapping[str, Any]) -> str:
    value = case.get("row_id", case.get("case_id", case.get("image_id")))
    if value is None:
        raise ValueError("case lacks row_id/case_id/image_id")
    return str(value)


def _image_id(case: Mapping[str, Any]) -> int:
    if type(case.get("image_id")) is int:
        return int(case["image_id"])
    record = case.get("input_record")
    if isinstance(record, Mapping) and type(record.get("image_id")) is int:
        return int(record["image_id"])
    raise ValueError(f"case lacks integer image_id: {_case_key(case)}")


def _cohort(case: Mapping[str, Any]) -> str:
    return str(case.get("cohort", case.get("split", "unknown")))


def _coord(value: Any) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, str) and value.startswith("<|coord_") and value.endswith("|>"):
        result = int(value[8:-2])
    else:
        raise ValueError(f"invalid coordinate token: {value!r}")
    if not 0 <= result <= 999:
        raise ValueError(f"coordinate outside 0..999: {result}")
    return result


def _references(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    record = case.get("input_record", case)
    objects = record.get("objects", []) if isinstance(record, Mapping) else []
    references = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or obj.get("coco_ann_id") is None:
            continue
        bins = obj.get("bbox_2d")
        if not isinstance(bins, list) or len(bins) != 4:
            continue
        try:
            coord_bins = [_coord(value) for value in bins]
        except ValueError:
            continue
        if coord_bins[0] >= coord_bins[2] or coord_bins[1] >= coord_bins[3]:
            continue
        references.append({
            "owner_id": str(obj["coco_ann_id"]),
            "reference_coord_bins_1000": coord_bins,
            "description": obj.get("desc"),
            "reference_index": index,
        })
    return references


def _predictions(parsed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_predictions = parsed.get("predictions", parsed.get("pred", []))
    dropped = parsed.get("dropped_predictions", [])
    if not isinstance(raw_predictions, list) or not isinstance(dropped, list):
        raise ValueError("parsed predictions and dropped_predictions must be lists")
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(raw_predictions):
        if not isinstance(item, Mapping):
            continue
        bins = item.get("coord_bins", item.get("coord_bins_1000"))
        prediction = {
            "prediction_id": str(item.get("prediction_id", f"p{item.get('generated_order', index)}")),
            "generated_order": int(item.get("generated_order", index)),
            "description": item.get("description"),
            "coord_bins_1000": list(bins) if isinstance(bins, list) else None,
            "raw": dict(item),
        }
        valid.append(prediction)
    return valid, [dict(item) for item in dropped if isinstance(item, Mapping)]


_COMPLETE_CELL_STATUSES = frozenset({"complete", "candidate_complete", "success", "ok"})


def _cell_is_complete(cell: Mapping[str, Any]) -> bool:
    status = cell.get("status")
    # Qualification fixtures predate the explicit status field; a payload with
    # its complete schema remains admissible. Once status is present, fail
    # closed on HOLD/interrupted/failed cells.
    return status is None or str(status) in _COMPLETE_CELL_STATUSES


def _hold_cell(cell: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    token_ids = cell.get("token_ids", [])
    return {
        "row_id": str(cell["row_id"]),
        "condition": str(cell["condition"]),
        "cell_status": "HOLD",
        "reason": reason,
        "source_status": cell.get("status"),
        "partial_generation": cell.get("partial_generation"),
        "token_count": len(token_ids) if isinstance(token_ids, list) else 0,
        "stop_reason": cell.get("stop_reason", "unknown"),
    }


def _geometry_valid(prediction: Mapping[str, Any]) -> bool:
    bins = prediction.get("coord_bins_1000")
    return (
        isinstance(bins, list)
        and len(bins) == 4
        and all(type(value) is int and 0 <= value <= 999 for value in bins)
        and bins[0] < bins[2]
        and bins[1] < bins[3]
    )


def _dropped_geometry_count(dropped: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        "geometry" in str(item.get("reason", "")).lower()
        or "bbox" in str(item.get("reason", "")).lower()
        for item in dropped
    )


def _owner_candidates(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        if not _geometry_valid(prediction):
            continue
        candidates = []
        for reference in references:
            overlap = readback_selectors.iou_xyxy(
                prediction["coord_bins_1000"], reference["reference_coord_bins_1000"]
            )
            if overlap >= IOU_THRESHOLD:
                candidates.append({"owner_id": str(reference["owner_id"]), "iou": overlap})
        candidates.sort(key=lambda item: (-float(item["iou"]), str(item["owner_id"])))
        result[str(prediction["prediction_id"])] = candidates
    return result


def _native_cell(cell: Mapping[str, Any], case: Mapping[str, Any], *, cap: int) -> dict[str, Any]:
    parsed = cell.get("parsed")
    if not isinstance(parsed, Mapping):
        raise ValueError(f"native cell parsed payload missing: {cell.get('row_id')} {cell.get('condition')}")
    valid, dropped = _predictions(parsed)
    geometry_invalid = _dropped_geometry_count(dropped)
    geometry_valid_predictions = []
    invalid_geometry_predictions = 0
    for prediction in valid:
        if _geometry_valid(prediction):
            geometry_valid_predictions.append(prediction)
        else:
            invalid_geometry_predictions += 1
    references = _references(case)
    candidates = _owner_candidates(references, geometry_valid_predictions)
    proxy_assignments = []
    by_owner: dict[str, list[int]] = {}
    for prediction in sorted(geometry_valid_predictions, key=lambda item: int(item["generated_order"])):
        options = candidates.get(str(prediction["prediction_id"]), [])
        owner = options[0] if options else None
        if owner is not None:
            proxy_assignments.append({
                "prediction_id": prediction["prediction_id"],
                "generated_order": prediction["generated_order"],
                "owner_id": owner["owner_id"],
                "iou": owner["iou"],
                "description": prediction.get("description"),
                "basis": "annotation_iou_0_5_proxy",
            })
            by_owner.setdefault(owner["owner_id"], []).append(int(prediction["generated_order"]))
    one_to_one = readback_selectors.one_to_one_matches(references, geometry_valid_predictions, IOU_THRESHOLD)
    covered = sorted({str(item["reference_owner_id"]) for item in one_to_one})
    proxy_owner_ids = sorted(by_owner)
    repeats = sum(max(0, len(orders) - 1) for orders in by_owner.values())
    burst_max = 0
    contiguous_run_max = 0
    current_owner: str | None = None
    current_length = 0
    previous_order: int | None = None
    for assignment in proxy_assignments:
        owner = str(assignment["owner_id"])
        order = int(assignment["generated_order"])
        if owner == current_owner and previous_order is not None and order == previous_order + 1:
            current_length += 1
        else:
            current_owner, current_length = owner, 1
        contiguous_run_max = max(contiguous_run_max, current_length)
        burst_max = max(burst_max, max(0, current_length - 1))
        previous_order = order
    token_ids = cell.get("token_ids", [])
    if not isinstance(token_ids, list):
        token_ids = []
    stop_reason = str(cell.get("stop_reason", "unknown"))
    termination = readback_selectors.termination_metrics(len(token_ids), token_ids, stop_reason, cap=cap, eos_token_id=EOS_TOKEN_ID)
    one_to_one_prediction_ids = {str(item["prediction_id"]) for item in one_to_one}
    annotation_unmatched = [str(item["prediction_id"]) for item in geometry_valid_predictions if str(item["prediction_id"]) not in one_to_one_prediction_ids]
    unknown_proxy = [str(item["prediction_id"]) for item in geometry_valid_predictions if str(item["prediction_id"]) not in {str(item["prediction_id"]) for item in proxy_assignments}]
    return {
        "row_id": str(cell["row_id"]),
        "image_id": _image_id(case),
        "condition": str(cell["condition"]),
        "cohort": _cohort(case),
        "stratum": case.get("stratum"),
        "cell_status": "complete",
        "token_count": len(token_ids),
        "cap": cap,
        "stop_reason": stop_reason,
        "termination": termination,
        "prediction_counts": {
            "parsed_valid": len(valid),
            "geometry_valid": len(geometry_valid_predictions),
            "parser_dropped": len(dropped),
            "geometry_invalid": geometry_invalid + invalid_geometry_predictions,
            "annotation_unmatched_one_to_one": len(annotation_unmatched),
            "owner_proxy_unknown": len(unknown_proxy),
        },
        "owner_proxy": {
            "references": len(references),
            "covered_owner_ids": covered,
            "proxy_owner_ids": proxy_owner_ids,
            "assignments": proxy_assignments,
            "same_owner_revisits": repeats,
            "same_owner_revisit_burst_max": burst_max,
            "same_owner_contiguous_run_max": contiguous_run_max,
            "annotation_unmatched_prediction_ids": annotation_unmatched,
            "unknown_prediction_ids": unknown_proxy,
            "basis": "class-agnostic IoU>=0.5 annotation proxy; not physical adjudication",
        },
        "parse": {"dropped_predictions": dropped},
    }


def _load_cell_files(cells_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    cells: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(cells_root.rglob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or "row_id" not in value or "condition" not in value:
            continue
        if "parsed" not in value and "teacher_coordinate_ce_sum" not in value and str(value.get("status")) not in {"HOLD", "partial", "failed", "technical_invalid"}:
            continue
        cells.append((path, dict(value)))
    return cells


def _expected_cases(manifest: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    cases = manifest.get(key, [])
    if not isinstance(cases, list):
        raise ValueError(f"manifest {key} must be a list")
    return [dict(case) for case in cases]


def _conditions(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get("conditions", [])
    if not isinstance(values, list) or not values:
        raise ValueError("manifest conditions must be a nonempty list")
    result = [_condition_name(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError("manifest conditions must be unique")
    return result


def _native_reduction(manifest: Mapping[str, Any], cells: Sequence[tuple[Path, dict[str, Any]]], *, cap: int) -> dict[str, Any]:
    cases = _expected_cases(manifest, "native_cases")
    conditions = _conditions(manifest)
    by_case = {_case_key(case): case for case in cases}
    expected = {(_case_key(case), condition) for case in cases for condition in conditions}
    found: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    duplicates = []
    unknown_cells = []
    for path, cell in cells:
        key = (str(cell["row_id"]), str(cell["condition"]))
        if key not in expected:
            unknown_cells.append(str(path))
            continue
        if key in found:
            duplicates.append({"key": list(key), "paths": [str(found[key][0]), str(path)]})
            continue
        found[key] = (path, cell)
    missing = [{"row_id": row_id, "condition": condition, "reason": "no_saved_cell"} for row_id, condition in sorted(expected) if (row_id, condition) not in found]
    held = [{"row_id": row_id, "condition": condition, "reason": "saved_cell_not_complete", "status": found[(row_id, condition)][1].get("status")}
            for row_id, condition in sorted(expected)
            if (row_id, condition) in found and not _cell_is_complete(found[(row_id, condition)][1])]
    per_image = []
    for case in cases:
        row_id = _case_key(case)
        row_cells = {}
        for condition in conditions:
            item = found.get((row_id, condition))
            row_cells[condition] = (
                _native_cell(item[1], case, cap=cap)
                if item and _cell_is_complete(item[1])
                else _hold_cell(item[1], reason="saved_cell_not_complete")
                if item
                else {"row_id": row_id, "condition": condition, "cell_status": "HOLD", "reason": "missing_saved_cell"}
            )
        complete = all(item.get("cell_status") == "complete" for item in row_cells.values())
        baseline = conditions[0]
        baseline_owners = set(row_cells[baseline].get("owner_proxy", {}).get("covered_owner_ids", [])) if complete else set()
        comparisons = {}
        for condition, item in row_cells.items():
            owners = set(item.get("owner_proxy", {}).get("covered_owner_ids", [])) if complete else set()
            comparisons[condition] = {"gained": sorted(owners - baseline_owners), "lost": sorted(baseline_owners - owners), "baseline_condition": baseline, "comparison_status": "complete_block" if complete else "HOLD_partial_block"}
        per_image.append({"row_id": row_id, "image_id": _image_id(case), "cohort": _cohort(case), "stratum": case.get("stratum"), "block_status": "complete_5_condition_block" if complete else "partial_block_HOLD", "conditions": row_cells, "owner_proxy_comparison": comparisons})
    complete_blocks = [row for row in per_image if row["block_status"] == "complete_5_condition_block"]
    def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        aggregate: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            condition_rows = [row["conditions"][condition] for row in rows]
            aggregate[condition] = {
                "complete_cells": len(condition_rows),
                "distinct_owner_proxy_count": sum(len(row["owner_proxy"]["covered_owner_ids"]) for row in condition_rows),
                "same_owner_revisits": sum(row["owner_proxy"]["same_owner_revisits"] for row in condition_rows),
                "same_owner_revisit_burst_max": max((row["owner_proxy"]["same_owner_revisit_burst_max"] for row in condition_rows), default=0),
                "same_owner_contiguous_run_max": max((row["owner_proxy"]["same_owner_contiguous_run_max"] for row in condition_rows), default=0),
                "geometry_invalid": sum(row["prediction_counts"]["geometry_invalid"] for row in condition_rows),
                "owner_proxy_unknown": sum(row["prediction_counts"]["owner_proxy_unknown"] for row in condition_rows),
                "natural_eos": sum(bool(row["termination"]["natural_eos"]) for row in condition_rows),
                "cap_debt": sum(int(row["termination"]["cap_debt"]) for row in condition_rows),
            }
        return aggregate

    # The pooled value is retained as a convenience view. Cohort views are the
    # decision-bearing denominators for development recurrence versus fresh
    # natural evaluation.
    aggregate = aggregate_rows(complete_blocks)
    by_cohort = {}
    for cohort in ("development_recurrence", "fresh_natural_evaluation"):
        by_cohort[cohort] = aggregate_rows([row for row in complete_blocks if row["cohort"] == cohort])
    comparisons = {
        condition: {
            "gained_count": sum(len(row["owner_proxy_comparison"][condition]["gained"]) for row in complete_blocks),
            "lost_count": sum(len(row["owner_proxy_comparison"][condition]["lost"]) for row in complete_blocks),
            "per_image": [
                {"row_id": row["row_id"], **row["owner_proxy_comparison"][condition]}
                for row in complete_blocks
            ],
        }
        for condition in conditions
    }
    return {
        "planned": {"cases": len(cases), "conditions": len(conditions), "cells": len(expected)},
        "observed": {"saved_cells": len(found), "missing_cells": len(missing), "held_cells": len(held), "complete_cells": sum(row["block_status"] == "complete_5_condition_block" for row in per_image) * len(conditions), "complete_blocks": len(complete_blocks), "partial_blocks": len(per_image) - len(complete_blocks)},
        "missing_cells": missing,
        "held_cells": held,
        "duplicate_cells": duplicates,
        "unknown_cell_files": unknown_cells,
        "per_image": per_image,
        "aggregate_complete_blocks": aggregate,
        "aggregate_complete_blocks_by_cohort": by_cohort,
        "aggregate_complete_blocks_scope": "pooled convenience view; use cohort aggregates for decision denominators",
        "owner_proxy_comparisons": comparisons,
        "basis": "one-to-one annotation IoU>=0.5 owner proxies; unmatched predictions remain UNKNOWN; exact-coordinate overlap is not a same-owner judgment",
    }


def _reference_bins(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [_coord(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if result[0] < result[2] and result[1] < result[3] else None


def _unique_description_reference(case: Mapping[str, Any], description: Any) -> list[int] | None:
    if not isinstance(description, str) or not description:
        return None
    record = case.get("input_record", case)
    objects = record.get("objects", []) if isinstance(record, Mapping) else []
    matches = [obj for obj in objects if isinstance(obj, Mapping) and obj.get("desc") == description]
    if len(matches) != 1:
        return None
    return _reference_bins(matches[0].get("bbox_2d"))


def _free_reference(manifest: Mapping[str, Any], case: Mapping[str, Any]) -> list[int] | None:
    keys = (
        "free_reference_coord_bins_1000",
        "referent_coord_bins_1000",
        "diagnostic_reference_coord_bins_1000",
        "reference_coord_bins_1000",
        "target_coord_bins_1000",
    )
    for key in keys:
        value = _reference_bins(case.get(key))
        if value is not None:
            return value
    for key in ("calibration_referent", "diagnostic_reference", "referent"):
        nested = case.get(key)
        if isinstance(nested, Mapping):
            for child_key in keys:
                value = _reference_bins(nested.get(child_key))
                if value is not None:
                    return value
    referents = manifest.get("calibration_referents", manifest.get("diagnostic_cases", []))
    if isinstance(referents, Mapping):
        row_id = _case_key(case)
        value = referents.get(row_id)
        if isinstance(value, str):
            # Manifest-v4 deliberately binds only the unique description;
            # recover its frozen coordinates from this case's source object.
            return _unique_description_reference(case, value)
        if isinstance(value, Mapping):
            description = value.get("description", value.get("referent_description"))
            if description is not None:
                return _unique_description_reference(case, description)
            for key in keys:
                resolved = _reference_bins(value.get(key))
                if resolved is not None:
                    return resolved
        return None
    if isinstance(referents, list):
        row_id = _case_key(case)
        for item in referents:
            if isinstance(item, Mapping) and str(item.get("row_id", item.get("case_id", ""))) == row_id:
                description = item.get("description", item.get("referent_description"))
                if description is not None:
                    return _unique_description_reference(case, description)
                for key in keys:
                    value = _reference_bins(item.get(key))
                    if value is not None:
                        return value
    return None


def _free_diagnostic(free: Any, reference: list[int] | None, *, default_cap: int) -> dict[str, Any]:
    parsed = free.get("parsed", {}) if isinstance(free, Mapping) else {}
    valid, dropped = _predictions(parsed if isinstance(parsed, Mapping) else {})
    first = next((item for item in sorted(valid, key=lambda item: int(item["generated_order"])) if _geometry_valid(item)), None)
    token_ids = free.get("token_ids", []) if isinstance(free, Mapping) else []
    token_count = len(token_ids) if isinstance(token_ids, list) else 0
    cap = int(free.get("cap", default_cap)) if isinstance(free, Mapping) else default_cap
    stop_reason = str(free.get("stop_reason", "unknown")) if isinstance(free, Mapping) else "unknown"
    common = {"reference_coord_bins_1000": reference, "parser_dropped": len(dropped), "token_count": token_count, "cap": cap, "stop_reason": stop_reason}
    if first is None:
        return {"status": "missing_first_box", **common, "first_box": None, "absolute_error": None, "mean_absolute_error_norm1000_on_scored_case": None}
    if reference is None:
        return {"status": "reference_unavailable_HOLD", **{**common, "reference_coord_bins_1000": None}, "first_box": list(first["coord_bins_1000"]), "absolute_error": None, "mean_absolute_error_norm1000_on_scored_case": None}
    errors = [abs(int(left) - int(right)) for left, right in zip(first["coord_bins_1000"], reference, strict=True)]
    return {"status": "first_box_scored", **common, "first_box": list(first["coord_bins_1000"]), "absolute_error": errors, "mean_absolute_error_norm1000_on_scored_case": sum(errors) / 4000.0}


def _calibration_reduction(manifest: Mapping[str, Any], cells: Sequence[tuple[Path, dict[str, Any]]], *, cap: int) -> dict[str, Any]:
    cases = _expected_cases(manifest, "calibration_cases")
    conditions = _conditions(manifest)
    expected = {(_case_key(case), condition) for case in cases for condition in conditions}
    found: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    duplicates = []
    for path, cell in cells:
        if "teacher_coordinate_ce_sum" not in cell:
            continue
        key = (str(cell["row_id"]), str(cell["condition"]))
        if key not in expected:
            continue
        if key in found:
            duplicates.append({"key": list(key), "paths": [str(found[key][0]), str(path)]})
        else:
            found[key] = (path, cell)
    missing = [{"row_id": row_id, "condition": condition, "reason": "no_saved_cell"} for row_id, condition in sorted(expected) if (row_id, condition) not in found]
    held = [{"row_id": row_id, "condition": condition, "reason": "saved_cell_not_complete", "status": found[(row_id, condition)][1].get("status")}
            for row_id, condition in sorted(expected)
            if (row_id, condition) in found and not _cell_is_complete(found[(row_id, condition)][1])]
    rows = []
    for case in cases:
        row_id = _case_key(case)
        conditions_out = {}
        for condition in conditions:
            item = found.get((row_id, condition))
            if not item:
                conditions_out[condition] = {"row_id": row_id, "condition": condition, "cell_status": "HOLD", "reason": "missing_saved_cell"}
                continue
            cell = item[1]
            if not _cell_is_complete(cell):
                conditions_out[condition] = _hold_cell(cell, reason="saved_cell_not_complete")
                continue
            free = cell.get("free", {})
            parsed = free.get("parsed", {}) if isinstance(free, Mapping) else {}
            valid, dropped = _predictions(parsed if isinstance(parsed, Mapping) else {})
            token_count = int(cell.get("coordinate_tokens", 0))
            conditions_out[condition] = {
                "row_id": row_id,
                "condition": condition,
                "cell_status": "complete",
                "teacher_coordinate_ce_sum": float(cell.get("teacher_coordinate_ce_sum", 0.0)),
                "coordinate_tokens": token_count,
                "teacher_coordinate_absolute_error_sum": float(cell.get("coordinate_absolute_error_sum", 0.0)),
                "teacher_coordinate_ce_per_token": (float(cell.get("teacher_coordinate_ce_sum", 0.0)) / token_count if token_count else None),
                "teacher_coordinate_absolute_error_per_token": (float(cell.get("coordinate_absolute_error_sum", 0.0)) / token_count if token_count else None),
                "free": {"parsed_valid": len(valid), "parser_dropped": len(dropped), "token_ids": list(free.get("token_ids", [])) if isinstance(free, Mapping) and isinstance(free.get("token_ids", []), list) else [], "stop_reason": free.get("stop_reason", "unknown") if isinstance(free, Mapping) else "unknown", "cap": int(free.get("cap", cap)) if isinstance(free, Mapping) else cap, "diagnostic": _free_diagnostic(free, _free_reference(manifest, case), default_cap=cap)},
            }
        rows.append({"row_id": row_id, "image_id": _image_id(case), "cohort": _cohort(case), "stratum": case.get("stratum"), "block_status": "complete_5_condition_block" if all(item.get("cell_status") == "complete" for item in conditions_out.values()) else "partial_block_HOLD", "conditions": conditions_out})
    complete = [row for row in rows if row["block_status"] == "complete_5_condition_block"]
    aggregate = {}
    for condition in conditions:
        cells_complete = [row["conditions"][condition] for row in complete]
        ce_tokens = sum(item["coordinate_tokens"] for item in cells_complete)
        diagnostics = [item["free"]["diagnostic"] for item in cells_complete]
        scored = [item for item in diagnostics if item["status"] == "first_box_scored"]
        aggregate[condition] = {"complete_cells": len(cells_complete), "coordinate_tokens": ce_tokens, "coordinate_ce_sum": sum(item["teacher_coordinate_ce_sum"] for item in cells_complete), "coordinate_absolute_error_sum": sum(item["teacher_coordinate_absolute_error_sum"] for item in cells_complete), "coordinate_ce_per_token": (sum(item["teacher_coordinate_ce_sum"] for item in cells_complete) / ce_tokens if ce_tokens else None), "coordinate_absolute_error_per_token": (sum(item["teacher_coordinate_absolute_error_sum"] for item in cells_complete) / ce_tokens if ce_tokens else None), "free_diagnostic_case_count": len(cells_complete), "free_diagnostic_case_denominator": len(cells_complete), "free_first_box_scored": len(scored), "free_first_box_missing": sum(item["status"] == "missing_first_box" for item in diagnostics), "free_reference_unavailable": sum(item["status"] == "reference_unavailable_HOLD" for item in diagnostics), "free_absolute_error_sum": sum(sum(item["absolute_error"]) for item in scored), "free_error_denominator_scored_cases": len(scored), "free_mean_absolute_error_norm1000_on_scored_cases": (sum(sum(item["absolute_error"]) for item in scored) / (4000.0 * len(scored)) if scored else None)}
    return {"planned": {"cases": len(cases), "conditions": len(conditions), "cells": len(expected)}, "observed": {"saved_cells": len(found), "missing_cells": len(missing), "held_cells": len(held), "complete_cells": len(complete) * len(conditions), "complete_blocks": len(complete), "partial_blocks": len(rows) - len(complete)}, "missing_cells": missing, "held_cells": held, "duplicate_cells": duplicates, "per_image": rows, "aggregate_complete_blocks": aggregate, "cap": cap, "free_diagnostic_denominator": "all complete saved calibration cells; missing first boxes remain failures in the denominator"}


def reduce_production(*, manifest_path: Path, cells_root: Path, output_path: Path, cap: int = 3084) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("production manifest must be a JSON object")
    cells = _load_cell_files(cells_root)
    native = _native_reduction(manifest, cells, cap=cap)
    calibration = _calibration_reduction(manifest, cells, cap=8)
    result = {
        "schema": SCHEMA,
        "status": (
            "technical_invalid_duplicate_cells"
            if native["duplicate_cells"] or calibration["duplicate_cells"]
            else "candidate_reduction_with_HOLD_cells"
            if native["observed"]["missing_cells"] or native["observed"].get("held_cells") or calibration["observed"]["missing_cells"] or calibration["observed"].get("held_cells")
            else "candidate_complete_denominator"
        ),
        "manifest": _file_binding(manifest_path),
        "cells_root": {"path": str(cells_root.resolve()), "kind": "directory"},
        "native": native,
        "calibration": calibration,
        "source_semantics": {"owner_credit": "annotation coco_ann_id IoU>=0.5 proxy only unless an explicit physical review field is present", "unmatched": "UNKNOWN", "same_owner_recurrence": "repeated annotation owner proxy across generated order; exact-coordinate repetition alone is not used", "missing_cells": "HOLD and excluded from complete-block aggregates"},
        "producer": _file_binding(Path(__file__)),
    }
    _publish(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=3084)
    args = parser.parse_args()
    result = reduce_production(manifest_path=args.manifest, cells_root=args.cells_root, output_path=args.output, cap=args.cap)
    print(json.dumps({"status": result["status"], "native": result["native"]["observed"], "calibration": result["calibration"]["observed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
