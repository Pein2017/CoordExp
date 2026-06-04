from __future__ import annotations

from typing import Any, Mapping

from .controls import headline_eligibility_status


def build_summary(
    *,
    case_index_total: int,
    case_index_rows: int,
    planned: int,
    attempted: int,
    valid: int,
    assigned: int,
    control_status_by_type: Mapping[str, str],
    breakdowns: Mapping[str, Any] | None = None,
    checkpoint_id: str | None = None,
    checkpoint_path: str | None = None,
    validation_status: str = "ok",
) -> dict[str, Any]:
    valid_rate = valid / planned if planned else 0.0
    return {
        "validation_status": validation_status,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "headline_eligibility_status": headline_eligibility_status(
            control_status_by_type,
            valid_rate=valid_rate,
        ),
        "denominator_counts": {
            "indexed_cases": case_index_total,
            "indexed_gt_rows": case_index_rows,
            "sampled_gpu_cases": planned,
            "attempted_gpu_cases": attempted,
            "valid_gpu_cases": valid,
            "taxonomy_cases": assigned,
        },
        "row_counts": {
            "case_index_total_cases": case_index_total,
            "case_index_total_rows": case_index_rows,
            "gpu_probe_planned_cases": planned,
            "gpu_probe_attempted_cases": attempted,
            "gpu_probe_valid_cases": valid,
            "taxonomy_assigned_cases": assigned,
        },
        "control_status_by_type": dict(control_status_by_type),
        "row_count_breakdowns": dict(breakdowns or {}),
    }


def build_probe_breakdowns(
    x1_rows: list[Mapping[str, Any]],
    taxonomy_rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    axes = {
        "split": "split",
        "desc_text": "desc_text_canonical",
        "same_desc_count_bucket": "same_desc_gt_count_annotated",
        "pool_role": "pool_role",
        "fn_rescue_overlay_membership": "fn_rescue_overlay_membership",
        "prefix_condition": "prefix_condition",
    }
    breakdowns: dict[str, dict[str, dict[str, int]]] = {axis: {} for axis in axes}
    breakdowns["primary_bucket"] = {}
    for row in x1_rows:
        valid = row.get("probe_status") == "ok"
        for axis, field in axes.items():
            value = _axis_value(row, field)
            _increment_counts(breakdowns[axis], value, attempted=1, valid=1 if valid else 0)
    for row in taxonomy_rows:
        for axis, field in axes.items():
            value = _axis_value(row, field)
            _increment_counts(breakdowns[axis], value, taxonomy=1)
        _increment_taxonomy_counts(breakdowns["primary_bucket"], str(row.get("primary_bucket") or "unknown"))
    total_taxonomy = len(taxonomy_rows)
    for counts in breakdowns["primary_bucket"].values():
        counts["taxonomy_denominator"] = total_taxonomy
    return breakdowns


def _increment_counts(
    table: dict[str, dict[str, int]],
    key: str,
    *,
    attempted: int = 0,
    valid: int = 0,
    taxonomy: int = 0,
) -> None:
    counts = table.setdefault(key, {"attempted": 0, "valid": 0, "taxonomy": 0})
    counts["attempted"] += attempted
    counts["valid"] += valid
    counts["taxonomy"] += taxonomy


def _increment_taxonomy_counts(table: dict[str, dict[str, int]], key: str) -> None:
    counts = table.setdefault(key, {"taxonomy": 0})
    counts["taxonomy"] += 1


def _axis_value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if field == "same_desc_gt_count_annotated":
        return _same_desc_bucket(int(value or 0))
    if field == "fn_rescue_overlay_membership":
        return "true" if bool(value) else "false"
    return str(value or "unknown")


def _same_desc_bucket(count: int) -> str:
    if count <= 1:
        return "same_desc_1"
    if count == 2:
        return "same_desc_2"
    if count == 3:
        return "same_desc_3"
    if count <= 5:
        return "same_desc_4_5"
    return "same_desc_6_plus"
