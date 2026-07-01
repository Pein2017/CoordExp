from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


DEFAULT_HINT_POLICY_ID = "desc_x1_r95_ladder_v1"
DEFAULT_BROAD_X1_RADIUS = 24
LOW_MARGIN_THRESHOLD = 0.05
HINT_LEVELS = ("none", "desc", "desc_x1", "desc_x1_y1")
SUPPORTED_SLOTS = ("x1", "y1", "x2", "y2")
PREFIX_CONDITIONS = (
    "empty_prefix",
    "rollout_prefix",
    "teacher_sorted_prefix",
    "teacher_oracle_remaining_prefix",
    "same_desc_removed_prefix",
    "same_desc_shuffled_prefix",
)
ANCHOR_POLICIES = {
    "empty_prefix": "empty_assistant_prefix_v1",
    "rollout_prefix": "native_rollout_emitted_prefix_v1",
    "teacher_sorted_prefix": "teacher_sorted_emitted_prefix_v1",
    "teacher_oracle_remaining_prefix": "teacher_oracle_remaining_prefix_v1",
    "same_desc_removed_prefix": "same_desc_removed_prefix_v1",
    "same_desc_shuffled_prefix": "same_desc_shuffled_prefix_v1",
}
BUCKET_AXES = (
    "not_rescued_under_valid_desc_x1_controls",
    "coord_binding_failure",
    "desc_selection_failure",
    "residual_accounting_failure",
    "prefix_suppression_flip",
    "low_margin_ambiguous",
    "probe_invalid_or_unscored",
)
PRIMARY_BUCKET_PRIORITY = (
    "probe_invalid_or_unscored",
    "prefix_suppression_flip",
    "residual_accounting_failure",
    "coord_binding_failure",
    "desc_selection_failure",
    "low_margin_ambiguous",
    "not_rescued_under_valid_desc_x1_controls",
)


def r95(axis_len: int) -> int:
    """Return the strict focused-Gaussian R95 radius for one coordinate axis."""

    axis_len = int(axis_len)
    if axis_len < 0:
        raise ValueError("axis_len must be non-negative")
    return math.floor(min(8, 0.04 * axis_len))


def build_fn_candidate_score_rows(
    candidate_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize externally supplied candidate scores into joinable rows.

    This is intentionally a pure row builder. Scores are supplied by a caller or
    test fixture; no model, tokenizer, image, or GPU module is imported here.
    """

    rows: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str]] = set()
    for index, score_row in enumerate(candidate_scores):
        probe_id = _required_str(score_row, "probe_id")
        score = _finite_float(score_row.get("score"), f"candidate_scores[{index}].score")
        candidate_id = str(
            score_row.get("candidate_id")
            or f"{probe_id}:candidate:{index}"
        )
        candidate_key = (probe_id, candidate_id)
        if candidate_key in seen_candidate_keys:
            raise ValueError(
                f"duplicate candidate_id for probe_id {probe_id}: {candidate_id}"
            )
        seen_candidate_keys.add(candidate_key)
        row = {
            "candidate_score_id": f"{probe_id}:score:{candidate_id}",
            "probe_id": probe_id,
            "fn_case_id": _required_str(score_row, "fn_case_id"),
            "candidate_id": candidate_id,
            "candidate_gt_idx": _optional_int(score_row.get("candidate_gt_idx")),
            "desc": str(score_row.get("desc", "")),
            "role": str(score_row.get("role", "candidate")),
            "score": score,
            "candidate_rank": None,
        }
        rows.append(_json_safe(row, "candidate_score_row"))

    ranks_by_key: dict[tuple[str, str], int] = {}
    for probe_id in sorted({str(row["probe_id"]) for row in rows}):
        probe_rows = [row for row in rows if row["probe_id"] == probe_id]
        ranked = sorted(
            probe_rows,
            key=lambda row: (
                -float(row["score"]),
                str(row["candidate_id"]),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            ranks_by_key[(probe_id, str(row["candidate_id"]))] = rank
    for row in rows:
        row["candidate_rank"] = ranks_by_key[
            (str(row["probe_id"]), str(row["candidate_id"]))
        ]
    return rows


def build_fn_slot_evidence_rows(
    slot_evidence: Sequence[Mapping[str, Any]],
    *,
    broad_x1_radius: int = DEFAULT_BROAD_X1_RADIUS,
) -> list[dict[str, Any]]:
    """Normalize externally supplied slot peaks into joinable evidence rows."""

    rows: list[dict[str, Any]] = []
    broad_x1_radius = int(broad_x1_radius)
    if broad_x1_radius < 0:
        raise ValueError("broad_x1_radius must be non-negative")
    for index, evidence in enumerate(slot_evidence):
        probe_id = _required_str(evidence, "probe_id")
        slot = str(evidence.get("slot", ""))
        if not slot:
            raise ValueError(f"slot_evidence[{index}].slot is required")
        slot_supported = slot in SUPPORTED_SLOTS
        axis_len = int(evidence.get("axis_len", 0))
        gt_value = int(evidence.get("gt_value", 0))
        peak_value = int(evidence.get("peak_value", 0))
        strict_radius = r95(axis_len)
        delta = abs(peak_value - gt_value)
        strict_hit = delta <= strict_radius
        broad_near = delta <= broad_x1_radius
        gt_idx = _optional_int(evidence.get("gt_idx"))
        hinted_control = bool(evidence.get("hinted_control", False))
        row: dict[str, Any] = {
            "slot_evidence_id": str(
                evidence.get("slot_evidence_id")
                or f"{probe_id}:slot:{slot}:{gt_idx}:{index}"
            ),
            "probe_id": probe_id,
            "fn_case_id": _required_str(evidence, "fn_case_id"),
            "slot": slot,
            "slot_supported": slot_supported,
            "axis_len": axis_len,
            "gt_idx": gt_idx,
            "gt_value": gt_value,
            "peak_value": peak_value,
            "peak_delta": delta,
            "score": (
                None
                if evidence.get("score") is None
                else _finite_float(evidence.get("score"), f"slot_evidence[{index}].score")
            ),
            "evidence_source": str(
                evidence.get(
                    "evidence_source",
                    "hint_control" if hinted_control else "model_decode",
                )
            ),
            "hinted_control": hinted_control,
            "model_predicted": bool(evidence.get("model_predicted", not hinted_control)),
            "strict_r95_radius": strict_radius,
            "strict_r95_hit": strict_hit,
            "broad_x1_radius": broad_x1_radius,
            "broad_near_24": broad_near,
        }
        row[f"{slot}_strict_r95_hit"] = strict_hit
        row[f"{slot}_broad_near_24"] = broad_near
        rows.append(_json_safe(row, "slot_evidence_row"))
    return rows


def build_fn_probe_rows(
    fn_cases: Sequence[Mapping[str, Any]],
    *,
    probe_specs: Sequence[Mapping[str, Any]],
    candidate_score_rows: Sequence[Mapping[str, Any]] = (),
    slot_evidence_rows: Sequence[Mapping[str, Any]] = (),
    hint_policy_id: str = DEFAULT_HINT_POLICY_ID,
    low_margin_threshold: float = LOW_MARGIN_THRESHOLD,
) -> list[dict[str, Any]]:
    """Build per-probe rows from CPU/mocked evidence supplied by the caller."""

    cases_by_id = {_required_str(case, "fn_case_id"): case for case in fn_cases}
    candidate_rows_by_probe = _rows_by_probe(candidate_score_rows)
    slot_rows_by_probe = _rows_by_probe(slot_evidence_rows)

    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(probe_specs):
        probe_id = _required_str(spec, "probe_id")
        fn_case_id = _required_str(spec, "fn_case_id")
        case = cases_by_id.get(fn_case_id)
        if case is None:
            raise ValueError(f"probe_specs[{spec_index}] references unknown fn_case_id: {fn_case_id}")

        hint_level = str(spec.get("hint_level", "none"))
        if hint_level not in HINT_LEVELS:
            raise ValueError(f"unsupported hint_level: {hint_level}")
        prefix_condition = str(spec.get("prefix_condition", "empty_prefix"))
        if prefix_condition not in PREFIX_CONDITIONS:
            raise ValueError(f"unsupported prefix_condition: {prefix_condition}")

        fn_gt_idx = int(case["fn_gt_idx"])
        fn_desc = str(case.get("fn_desc", ""))
        fn_bbox = _bbox_list(case.get("fn_bbox", []), "fn_bbox")
        emitted_gt_indices = _int_list(case.get("emitted_gt_indices", ()))
        residual_gt_indices = _int_list(case.get("residual_gt_indices", (fn_gt_idx,)))
        prefix_objects = _prefix_objects(spec.get("prefix_objects", ()))
        anchor_policy = str(spec.get("anchor_policy") or ANCHOR_POLICIES[prefix_condition])
        valid_parse = bool(spec.get("valid_parse", True))
        hint_text, hint_coords = _hint_text_and_coords(
            hint_level,
            fn_desc=fn_desc,
            fn_bbox=fn_bbox,
        )
        candidates = candidate_rows_by_probe.get(probe_id, [])
        slots = slot_rows_by_probe.get(probe_id, [])
        evidence = _summarize_probe_evidence(
            hint_level=hint_level,
            fn_desc=fn_desc,
            residual_gt_indices=residual_gt_indices,
            emitted_gt_indices=emitted_gt_indices,
            candidate_rows=candidates,
            slot_rows=slots,
            low_margin_threshold=low_margin_threshold,
        )
        axes = _base_bucket_axes(
            hint_level=hint_level,
            valid_parse=valid_parse,
            has_scores=bool(candidates),
            evidence=evidence,
        )
        row = {
            "probe_id": probe_id,
            "fn_case_id": fn_case_id,
            "checkpoint_role": case.get("checkpoint_role"),
            "split": case.get("split"),
            "image_id": case.get("image_id"),
            "width": _optional_int(case.get("width")),
            "height": _optional_int(case.get("height")),
            "fn_gt_idx": fn_gt_idx,
            "fn_desc": fn_desc,
            "fn_bbox": fn_bbox,
            "hint_level": hint_level,
            "hint_policy_id": str(hint_policy_id),
            "hint_text_sha256": _sha256_text(hint_text),
            "hint_coords": hint_coords,
            "valid_parse": valid_parse,
            "generated_continuation_sha256": _sha256_text(
                str(spec.get("generated_continuation", ""))
            ),
            "prefix_condition": prefix_condition,
            "anchor_policy": anchor_policy,
            "prefix_gt_indices": [
                int(obj["gt_idx"]) for obj in prefix_objects if obj["gt_idx"] is not None
            ],
            "prefix_pred_indices": [
                int(obj["pred_idx"]) for obj in prefix_objects if obj["pred_idx"] is not None
            ],
            "prefix_len": len(prefix_objects),
            "contains_future_gt_after_fn": any(
                obj["gt_idx"] is not None and int(obj["gt_idx"]) > fn_gt_idx
                for obj in prefix_objects
            ),
            "rendered_assistant_prefix_sha256": _sha256_json(
                {
                    "prefix_condition": prefix_condition,
                    "anchor_policy": anchor_policy,
                    "prefix_objects": prefix_objects,
                }
            ),
            "prefix_objects": prefix_objects,
            "emitted_gt_indices": emitted_gt_indices,
            "residual_gt_indices": residual_gt_indices,
            **evidence,
            **axes,
            "primary_bucket": None,
            "bucket_trace": [],
        }
        rows.append(_json_safe(row, "fn_probe_row"))

    _apply_prefix_suppression_flips(rows)
    for row in rows:
        row["primary_bucket"] = _primary_bucket(row)
        row["bucket_trace"] = [axis for axis in BUCKET_AXES if bool(row.get(axis))]
    return rows


def _summarize_probe_evidence(
    *,
    hint_level: str,
    fn_desc: str,
    residual_gt_indices: Sequence[int],
    emitted_gt_indices: Sequence[int],
    candidate_rows: Sequence[Mapping[str, Any]],
    slot_rows: Sequence[Mapping[str, Any]],
    low_margin_threshold: float,
) -> dict[str, Any]:
    ranked_candidates = sorted(
        candidate_rows,
        key=lambda row: (
            int(row.get("candidate_rank") or 999_999),
            -float(row.get("score", 0.0)),
            str(row.get("candidate_id", "")),
        ),
    )
    top_candidate = ranked_candidates[0] if ranked_candidates else None
    runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    top_desc = None if top_candidate is None else str(top_candidate.get("desc", ""))
    top_score = None if top_candidate is None else float(top_candidate.get("score", 0.0))
    margin = None
    if top_candidate is not None and runner_up is not None:
        margin = float(top_candidate.get("score", 0.0)) - float(
            runner_up.get("score", 0.0)
        )

    residual_set = {int(value) for value in residual_gt_indices}
    emitted_set = {int(value) for value in emitted_gt_indices}
    unsupported_slot_names = sorted(
        {
            str(row.get("slot"))
            for row in slot_rows
            if not bool(row.get("slot_supported", row.get("slot") in SUPPORTED_SLOTS))
        }
    )
    residual_x1_rows = [
        row
        for row in slot_rows
        if row.get("slot") == "x1"
        and bool(row.get("slot_supported", True))
        and _optional_int(row.get("gt_idx")) in residual_set
    ]
    emitted_x1_rows = [
        row
        for row in slot_rows
        if row.get("slot") == "x1"
        and bool(row.get("slot_supported", True))
        and _optional_int(row.get("gt_idx")) in emitted_set
    ]
    valid_residual_x1_control = bool(residual_x1_rows)
    residual_x1_strict_hit = any(bool(row.get("strict_r95_hit")) for row in residual_x1_rows)
    residual_x1_model_strict_hit = any(
        bool(row.get("strict_r95_hit")) and bool(row.get("model_predicted", True))
        for row in residual_x1_rows
    )
    residual_x1_hint_control_hit = any(
        bool(row.get("strict_r95_hit")) and bool(row.get("hinted_control"))
        for row in residual_x1_rows
    )
    emitted_same_desc_x1_strict_hit = any(
        bool(row.get("strict_r95_hit")) for row in emitted_x1_rows
    )
    residual_x1_broad_near_24 = any(bool(row.get("broad_near_24")) for row in residual_x1_rows)
    required_generated_slots = _required_generated_slots_for_hint(hint_level)
    required_slot_hits = {
        slot: any(
            str(row.get("slot")) == slot
            and _optional_int(row.get("gt_idx")) in residual_set
            and bool(row.get("slot_supported", True))
            and bool(row.get("model_predicted", True))
            and bool(row.get("strict_r95_hit"))
            for row in slot_rows
        )
        for slot in required_generated_slots
    }
    required_slot_count = len(required_generated_slots)
    required_slot_strict_hit_count = sum(
        1 for hit in required_slot_hits.values() if hit
    )
    all_required_slots_strict_hit = (
        required_slot_count > 0
        and required_slot_strict_hit_count == required_slot_count
    )
    desc_target_favored = top_desc == fn_desc if top_desc is not None else False
    low_margin_ambiguous = margin is not None and margin <= float(low_margin_threshold)
    residual_geometry_success = desc_target_favored and all_required_slots_strict_hit
    return {
        "top_candidate_desc": top_desc,
        "top_candidate_score": top_score,
        "candidate_margin": margin,
        "desc_target_favored": desc_target_favored,
        "valid_residual_x1_control": valid_residual_x1_control,
        "unsupported_slot_names": unsupported_slot_names,
        "residual_x1_strict_hit": residual_x1_strict_hit,
        "residual_x1_model_strict_hit": residual_x1_model_strict_hit,
        "residual_x1_hint_control_hit": residual_x1_hint_control_hit,
        "emitted_same_desc_x1_strict_hit": emitted_same_desc_x1_strict_hit,
        "x1_strict_r95_hit": residual_x1_strict_hit,
        "x1_broad_near_24": residual_x1_broad_near_24,
        "required_generated_slots": list(required_generated_slots),
        "required_slot_strict_hits": required_slot_hits,
        "required_slot_count": required_slot_count,
        "required_slot_strict_hit_count": required_slot_strict_hit_count,
        "all_required_slots_strict_hit": all_required_slots_strict_hit,
        "residual_geometry_success": residual_geometry_success,
        "residual_accounting_success": residual_geometry_success,
        "low_margin_ambiguous": low_margin_ambiguous,
    }


def _base_bucket_axes(
    *,
    hint_level: str,
    valid_parse: bool,
    has_scores: bool,
    evidence: Mapping[str, Any],
) -> dict[str, bool]:
    desc_target_favored = bool(evidence.get("desc_target_favored"))
    residual_geometry_success = bool(evidence.get("residual_geometry_success"))
    emitted_hit = bool(evidence.get("emitted_same_desc_x1_strict_hit"))
    valid_residual_x1_control = bool(evidence.get("valid_residual_x1_control"))
    requires_residual_x1_control = hint_level in {"desc_x1", "desc_x1_y1"}
    has_unsupported_slots = bool(evidence.get("unsupported_slot_names"))
    missing_required_x1_control = (
        requires_residual_x1_control and not valid_residual_x1_control
    )
    invalid_or_unscored = (
        not valid_parse
        or (not has_scores and not valid_residual_x1_control)
        or missing_required_x1_control
        or has_unsupported_slots
    )
    desc_selection_failure = (
        valid_parse
        and not invalid_or_unscored
        and has_scores
        and evidence.get("top_candidate_desc") is not None
        and not desc_target_favored
    )
    coord_binding_failure = (
        valid_parse
        and not invalid_or_unscored
        and valid_residual_x1_control
        and desc_target_favored
        and not residual_geometry_success
    )
    residual_accounting_failure = (
        valid_parse
        and desc_target_favored
        and emitted_hit
        and not residual_geometry_success
    )
    not_rescued = (
        valid_parse
        and not invalid_or_unscored
        and requires_residual_x1_control
        and valid_residual_x1_control
        and desc_target_favored
        and not residual_geometry_success
    )
    return {
        "not_rescued_under_valid_desc_x1_controls": not_rescued,
        "coord_binding_failure": coord_binding_failure,
        "desc_selection_failure": desc_selection_failure,
        "residual_accounting_failure": residual_accounting_failure,
        "prefix_suppression_flip": False,
        "low_margin_ambiguous": bool(evidence.get("low_margin_ambiguous")),
        "probe_invalid_or_unscored": invalid_or_unscored,
    }


def _apply_prefix_suppression_flips(rows: list[dict[str, Any]]) -> None:
    teacher_hit_by_key: dict[tuple[str, str], bool] = {}
    for row in rows:
        if row["prefix_condition"] != "teacher_sorted_prefix":
            continue
        if not _valid_prefix_flip_control(row):
            continue
        key = (str(row["fn_case_id"]), str(row["hint_level"]))
        teacher_hit_by_key[key] = teacher_hit_by_key.get(key, False) or bool(
            row.get("residual_geometry_success")
        )

    for row in rows:
        if row["prefix_condition"] != "rollout_prefix":
            continue
        if not _valid_prefix_flip_control(row):
            continue
        key = (str(row["fn_case_id"]), str(row["hint_level"]))
        if teacher_hit_by_key.get(key, False) and not bool(row.get("residual_geometry_success")):
            row["prefix_suppression_flip"] = True


def _valid_prefix_flip_control(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("valid_parse"))
        and not bool(row.get("probe_invalid_or_unscored"))
        and bool(row.get("valid_residual_x1_control"))
    )


def _required_generated_slots_for_hint(hint_level: str) -> tuple[str, ...]:
    if hint_level in {"none", "desc"}:
        return ("x1", "y1", "x2", "y2")
    if hint_level == "desc_x1":
        return ("y1", "x2", "y2")
    if hint_level == "desc_x1_y1":
        return ("x2", "y2")
    raise ValueError(f"unsupported hint_level: {hint_level}")


def _primary_bucket(row: Mapping[str, Any]) -> str:
    for axis in PRIMARY_BUCKET_PRIORITY:
        if bool(row.get(axis)):
            return axis
    if bool(row.get("residual_accounting_success")):
        return "rescued_residual_instance"
    return "unclassified_valid_probe"


def _hint_text_and_coords(
    hint_level: str,
    *,
    fn_desc: str,
    fn_bbox: Sequence[int],
) -> tuple[str, dict[str, int]]:
    if hint_level == "none":
        return "", {}
    if hint_level == "desc":
        return f"desc={fn_desc}", {}
    if hint_level == "desc_x1":
        coords = {"x1": int(fn_bbox[0])}
        return f"desc={fn_desc}|x1={coords['x1']}", coords
    if hint_level == "desc_x1_y1":
        coords = {"x1": int(fn_bbox[0]), "y1": int(fn_bbox[1])}
        return f"desc={fn_desc}|x1={coords['x1']}|y1={coords['y1']}", coords
    raise ValueError(f"unsupported hint_level: {hint_level}")


def _prefix_objects(prefix_objects: Any) -> list[dict[str, Any]]:
    if prefix_objects is None:
        return []
    rows: list[dict[str, Any]] = []
    for index, obj in enumerate(prefix_objects):
        if not isinstance(obj, Mapping):
            raise ValueError(f"prefix_objects[{index}] must be a mapping")
        rows.append(
            _json_safe(
                {
                    "source": str(obj.get("source", "")),
                    "gt_idx": _optional_int(obj.get("gt_idx")),
                    "pred_idx": _optional_int(obj.get("pred_idx")),
                    "desc": str(obj.get("desc", "")),
                    "bbox": _bbox_list(obj.get("bbox", []), f"prefix_objects[{index}].bbox"),
                    "order_idx": int(obj.get("order_idx", index)),
                },
                "prefix_object",
            )
        )
    return rows


def _rows_by_probe(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        probe_id = row.get("probe_id")
        if probe_id is None:
            raise ValueError(f"row {index} is missing probe_id")
        grouped.setdefault(str(probe_id), []).append(row)
    return grouped


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{key} is required")
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _bbox_list(value: Any, path: str) -> list[int]:
    items = list(value)
    if len(items) != 4:
        raise ValueError(f"{path} must contain four coordinates")
    return [int(item) for item in items]


def _finite_float(value: Any, path: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_text(payload)


def _json_safe(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str | int):
        return value
    return str(value)


__all__ = [
    "ANCHOR_POLICIES",
    "BUCKET_AXES",
    "DEFAULT_BROAD_X1_RADIUS",
    "DEFAULT_HINT_POLICY_ID",
    "HINT_LEVELS",
    "LOW_MARGIN_THRESHOLD",
    "PREFIX_CONDITIONS",
    "SUPPORTED_SLOTS",
    "build_fn_candidate_score_rows",
    "build_fn_probe_rows",
    "build_fn_slot_evidence_rows",
    "r95",
]
