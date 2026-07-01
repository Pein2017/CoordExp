from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence


PREFIX_MODE_POLICY_ID = "a3_3_semantic_prefix_modes_v1"
SUPPORTED_PREFIX_MODES = (
    "empty",
    "same_desc_good_prefix",
    "same_desc_bad_prefix",
    "different_desc_good_prefix",
    "rollout_good_prefix",
    "rollout_bad_prefix",
)

_MODE_ALIASES = {
    "minimal_or_empty_prefix": "empty",
    "canonical_sorted_gt_prefix_before_target": "same_desc_good_prefix",
    "clean_non_target_same_desc_prefix": "same_desc_good_prefix",
    "duplicate_same_desc_prefix": "same_desc_bad_prefix",
    "wrong_instance_same_desc_prefix": "same_desc_bad_prefix",
    "rollout_native_prefix_with_quality_label": "rollout_good_prefix",
}
_WS_RE = re.compile(r"\s+")


def canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def render_compact_prefix_rows(
    objects: Sequence[Mapping[str, Any]],
    *,
    row_separator: str,
) -> str:
    rendered = [_render_one_compact_row(obj) for obj in objects]
    separator = _separator(row_separator)
    return separator.join(rendered)


def render_forced_state_prompt(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    desc: str,
    forced_x1: int,
    forced_state: str,
    row_separator: str,
) -> str:
    if forced_state not in {"pre_x1", "post_x1"}:
        raise ValueError(f"unsupported forced_state for prefix materialization: {forced_state}")
    forced = _render_forced_desc_row(desc=desc, forced_x1=forced_x1 if forced_state == "post_x1" else None)
    prefix = render_compact_prefix_rows(prefix_objects, row_separator=row_separator)
    if not prefix:
        return forced
    return prefix + _separator(row_separator) + forced


def build_prefix_mode_rows(
    case: Mapping[str, Any],
    *,
    modes: Sequence[str],
    rollout_prefix_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requested_mode in modes:
        mode = _canonical_mode(requested_mode)
        if mode in {"rollout_good_prefix", "rollout_bad_prefix"}:
            rollout_row = _rollout_row_for_case(
                case,
                mode=mode,
                rollout_prefix_rows=rollout_prefix_rows,
            )
            if rollout_row is not None:
                rows.append(rollout_row)
            continue
        row = _synthetic_prefix_row(case, mode=mode, requested_mode=requested_mode)
        if row is not None:
            rows.append(row)
    return rows


def materialize_prefix_modes(
    cases: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
    rollout_prefix_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    rollout_prefix_missing_policy: str = "skip_with_manifest",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = [str(mode) for mode in modes]
    skipped_reasons: dict[str, str] = {}
    rollout_missing = rollout_prefix_rows is None and any(
        _canonical_mode(mode) in {"rollout_good_prefix", "rollout_bad_prefix"}
        for mode in requested
    )
    if rollout_missing:
        rollout_modes = [
            mode
            for mode in requested
            if _canonical_mode(mode) in {"rollout_good_prefix", "rollout_bad_prefix"}
        ]
        if rollout_prefix_missing_policy == "fail":
            raise ValueError(
                "rollout prefix modes requested but rollout_prefix_rows is missing"
            )
        if rollout_prefix_missing_policy not in {
            "skip",
            "skip_with_manifest",
            "mark",
            "mark_unmaterialized",
        }:
            raise ValueError(
                f"unsupported rollout_prefix_missing_policy: {rollout_prefix_missing_policy}"
            )
        for mode in rollout_modes:
            skipped_reasons[mode] = "rollout_prefix_source_missing"

    materializable_modes = [
        mode
        for mode in requested
        if mode not in skipped_reasons
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(
            build_prefix_mode_rows(
                case,
                modes=materializable_modes,
                rollout_prefix_rows=rollout_prefix_rows,
            )
        )

    materialized = _ordered_unique(row["prefix_mode"] for row in rows)
    summary = {
        "prefix_mode_policy_id": PREFIX_MODE_POLICY_ID,
        "prefix_modes_requested": requested,
        "prefix_modes_materialized": materialized,
        "prefix_modes_skipped": sorted(skipped_reasons),
        "rollout_prefix_missing_policy": rollout_prefix_missing_policy,
        "skipped_mode_reasons": skipped_reasons,
        "materialized_row_count": len(rows),
    }
    return rows, summary


def _synthetic_prefix_row(
    case: Mapping[str, Any],
    *,
    mode: str,
    requested_mode: str,
) -> dict[str, Any] | None:
    target_gt_idx = int(case["target_gt_idx"])
    objects_by_idx = _objects_by_idx(case)
    competitor_indices = [
        int(idx)
        for idx in case.get("competitor_gt_indices", ())
        if int(idx) in objects_by_idx and int(idx) != target_gt_idx
    ]
    desc = str(case["desc"])

    if mode == "empty":
        prefix_objects: list[dict[str, Any]] = []
        quality = "good"
        bad_bucket = None
        duplicate_source_gt_idx = None
        wrong_instance_replacement_gt_idx = None
        source_kind = "none"
    elif mode == "same_desc_good_prefix":
        prefix_objects = [_prefix_object(objects_by_idx[idx]) for idx in competitor_indices]
        quality = "good"
        bad_bucket = None
        duplicate_source_gt_idx = None
        wrong_instance_replacement_gt_idx = None
        source_kind = "gt_synthetic"
    elif mode == "same_desc_bad_prefix":
        if not competitor_indices:
            return None
        source_idx = competitor_indices[0]
        prefix_objects = [
            _prefix_object(objects_by_idx[source_idx]),
            _prefix_object(objects_by_idx[source_idx]),
        ]
        quality = "bad"
        bad_bucket = "duplicate_prefix"
        duplicate_source_gt_idx = source_idx
        wrong_instance_replacement_gt_idx = None
        source_kind = "gt_synthetic"
    elif mode == "different_desc_good_prefix":
        prefix_objects = [
            _prefix_object(obj)
            for idx, obj in sorted(objects_by_idx.items())
            if idx != target_gt_idx and canonical_desc(str(obj.get("desc") or "")) != canonical_desc(desc)
        ]
        quality = "good"
        bad_bucket = None
        duplicate_source_gt_idx = None
        wrong_instance_replacement_gt_idx = None
        source_kind = "gt_synthetic"
    else:
        raise ValueError(f"unsupported prefix mode: {mode}")

    target_leak = target_gt_idx in {int(obj["gt_idx"]) for obj in prefix_objects}
    if target_leak:
        quality = "target_leak"
        bad_bucket = "target_leak_prefix"
    return _base_prefix_row(
        case=case,
        mode=mode,
        requested_mode=requested_mode,
        prefix_source_kind=source_kind,
        prefix_source_artifact=None,
        prefix_source_line_id=None,
        prefix_quality=quality,
        prefix_objects=prefix_objects,
        target_leak=target_leak,
        duplicate_source_gt_idx=duplicate_source_gt_idx,
        wrong_instance_replacement_gt_idx=wrong_instance_replacement_gt_idx,
        match_iou_policy="gt_exact",
        denominator_eligible=not target_leak,
        bad_prefix_bucket=bad_bucket,
    )


def _rollout_row_for_case(
    case: Mapping[str, Any],
    *,
    mode: str,
    rollout_prefix_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    if rollout_prefix_rows is None:
        return None
    rollout = _lookup_rollout(case, rollout_prefix_rows)
    if rollout is None:
        return None
    prefix_objects = [_prefix_object(obj) for obj in rollout.get("prefix_objects", ())]
    quality = "good" if mode == "rollout_good_prefix" else "bad"
    bad_bucket = None if quality == "good" else str(rollout.get("bad_prefix_bucket") or "rollout_unmatched_prefix")
    target_gt_idx = int(case["target_gt_idx"])
    target_leak = target_gt_idx in {int(obj["gt_idx"]) for obj in prefix_objects if "gt_idx" in obj}
    return _base_prefix_row(
        case=case,
        mode=mode,
        requested_mode=mode,
        prefix_source_kind="rollout_native",
        prefix_source_artifact=rollout.get("prefix_source_artifact"),
        prefix_source_line_id=rollout.get("prefix_source_line_id"),
        prefix_quality=quality,
        prefix_objects=prefix_objects,
        target_leak=target_leak,
        duplicate_source_gt_idx=rollout.get("duplicate_source_gt_idx"),
        wrong_instance_replacement_gt_idx=rollout.get("wrong_instance_replacement_gt_idx"),
        match_iou_policy=str(rollout.get("match_iou_policy") or "rollout_iou50"),
        denominator_eligible=not target_leak and quality == "good",
        bad_prefix_bucket=bad_bucket,
    )


def _base_prefix_row(
    *,
    case: Mapping[str, Any],
    mode: str,
    requested_mode: str,
    prefix_source_kind: str,
    prefix_source_artifact: str | None,
    prefix_source_line_id: int | None,
    prefix_quality: str,
    prefix_objects: Sequence[Mapping[str, Any]],
    target_leak: bool,
    duplicate_source_gt_idx: int | None,
    wrong_instance_replacement_gt_idx: int | None,
    match_iou_policy: str,
    denominator_eligible: bool,
    bad_prefix_bucket: str | None,
) -> dict[str, Any]:
    objects = [dict(obj) for obj in prefix_objects]
    prefix_gt_indices = [
        int(obj["gt_idx"]) for obj in objects if obj.get("gt_idx") is not None
    ]
    row = {
        "case_id": case["case_id"],
        "prefix_mode": mode,
        "requested_prefix_mode": requested_mode,
        "prefix_mode_policy_id": PREFIX_MODE_POLICY_ID,
        "prefix_row_id": _prefix_row_id(str(case["case_id"]), mode, prefix_gt_indices),
        "desc": case.get("desc"),
        "target_gt_idx": int(case["target_gt_idx"]),
        "prefix_source_kind": prefix_source_kind,
        "prefix_source_artifact": prefix_source_artifact,
        "prefix_source_line_id": prefix_source_line_id,
        "prefix_quality": prefix_quality,
        "prefix_gt_indices": prefix_gt_indices,
        "prefix_objects": objects,
        "target_leak": bool(target_leak),
        "duplicate_source_gt_idx": duplicate_source_gt_idx,
        "wrong_instance_replacement_gt_idx": wrong_instance_replacement_gt_idx,
        "match_iou_policy": match_iou_policy,
        "denominator_eligible": bool(denominator_eligible),
        "bad_prefix_bucket": bad_prefix_bucket,
    }
    return row


def _render_one_compact_row(obj: Mapping[str, Any]) -> str:
    desc = canonical_desc(str(obj.get("desc") or obj.get("desc_text") or ""))
    if not desc:
        raise ValueError("prefix object is missing desc")
    bbox = _bbox(obj)
    tokens = "".join(_coord_token(value) for value in bbox)
    return f"<|object_ref_start|>{desc}<|box_start|>{tokens}"


def _render_forced_desc_row(*, desc: str, forced_x1: int | None) -> str:
    row = f"<|object_ref_start|>{canonical_desc(desc)}<|box_start|>"
    if forced_x1 is not None:
        row += _coord_token(forced_x1)
    return row


def _bbox(obj: Mapping[str, Any]) -> list[int]:
    bbox = obj.get("bbox_coord_token_xyxy", obj.get("bbox_xyxy"))
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("prefix object must carry bbox_coord_token_xyxy or bbox_xyxy")
    values = [int(value) for value in bbox]
    for value in values:
        if value < 0 or value > 999:
            raise ValueError(f"coordinate token value out of range: {value}")
    return values


def _coord_token(value: int) -> str:
    value = int(value)
    if value < 0 or value > 999:
        raise ValueError(f"coordinate token value out of range: {value}")
    return f"<|coord_{value}|>"


def _separator(row_separator: str) -> str:
    if row_separator == "none":
        return ""
    if row_separator == "newline":
        return "\n"
    raise ValueError(f"unsupported row_separator: {row_separator}")


def _canonical_mode(mode: str) -> str:
    canonical = _MODE_ALIASES.get(str(mode), str(mode))
    if canonical not in SUPPORTED_PREFIX_MODES:
        raise ValueError(f"unsupported prefix mode: {mode}")
    return canonical


def _objects_by_idx(case: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    objects = case.get("objects") or []
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("case must carry semantic objects for prefix materialization")
    result: dict[int, Mapping[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping) or "gt_idx" not in obj:
            continue
        result[int(obj["gt_idx"])] = obj
    return result


def _prefix_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    bbox = _bbox(obj)
    return {
        "gt_idx": int(obj["gt_idx"]) if obj.get("gt_idx") is not None else None,
        "desc": canonical_desc(str(obj.get("desc") or obj.get("desc_text") or "")),
        "bbox_coord_token_xyxy": bbox,
        "bbox_xyxy": bbox,
        "bbox_surface": obj.get("bbox_surface", "semantic_coord_token_xyxy"),
        "bbox_source_field": obj.get("bbox_source_field"),
    }


def _lookup_rollout(
    case: Mapping[str, Any],
    rollout_prefix_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    case_id = str(case["case_id"])
    if isinstance(rollout_prefix_rows, Mapping):
        row = rollout_prefix_rows.get(case_id)
        return row if isinstance(row, Mapping) else None
    for row in rollout_prefix_rows:
        if isinstance(row, Mapping) and str(row.get("case_id")) == case_id:
            return row
    return None


def _prefix_row_id(case_id: str, mode: str, gt_indices: Sequence[int]) -> str:
    payload = f"{case_id}|{mode}|{','.join(str(idx) for idx in gt_indices)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "PREFIX_MODE_POLICY_ID",
    "SUPPORTED_PREFIX_MODES",
    "build_prefix_mode_rows",
    "canonical_desc",
    "materialize_prefix_modes",
    "render_compact_prefix_rows",
    "render_forced_state_prompt",
]
