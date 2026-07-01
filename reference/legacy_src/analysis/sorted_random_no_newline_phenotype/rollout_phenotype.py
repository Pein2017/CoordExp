from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DECODE_POLICY = "free_text_unconstrained_greedy_temp0"
CONSTRAINT_POLICY = "none"
SCHEMA_VERSION = "a3.2.rollout_phenotype.v1"
MATCH_IOU_THRESHOLD = 0.5
DUPLICATE_IOU_THRESHOLD = 0.95

_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")
_WS_RE = re.compile(r"\s+")


def normalize_rollout_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one A3.2 native rollout artifact row."""

    if not isinstance(row, Mapping):
        raise TypeError("rollout row must be a mapping")
    if _is_standard_eval_artifact_row(row):
        row = _standard_eval_artifact_to_normalized_rollout_row(row)

    decode_policy = _required_str(row, "decode_policy")
    if decode_policy != DECODE_POLICY:
        raise ValueError(f"decode_policy must be {DECODE_POLICY}")

    constraint_policy = _required_str(row, "constraint_policy")
    if constraint_policy != CONSTRAINT_POLICY:
        raise ValueError("constraint_policy must be none")

    template_contract = _required_mapping(row, "template_contract")
    row_separator = str(template_contract.get("row_separator") or "")
    if row_separator != "none":
        raise ValueError("template_contract.row_separator must be none")

    pred_rows_ordered = _required_list(row, "pred_rows_ordered")
    invalid_pred_rows = _required_list(row, "invalid_pred_rows")
    raw_output_text_sha256 = _raw_output_sha256(row)

    normalized = {
        "checkpoint_role": _required_str(row, "checkpoint_role"),
        "image_id": _required_str(row, "image_id"),
        "source_line_idx": _required_int(row, "source_line_idx"),
        "raw_output_text_sha256": raw_output_text_sha256,
        "pred_rows_ordered": list(pred_rows_ordered),
        "invalid_pred_rows": list(invalid_pred_rows),
        "rollout_stop_reason": _required_str(row, "rollout_stop_reason"),
        "decode_policy": decode_policy,
        "constraint_policy": constraint_policy,
        "native_prompt_ordering": _required_str(row, "native_prompt_ordering"),
        "template_contract": {"row_separator": "none"},
        "parse_failure": row.get("parse_failure"),
        "parse_error": row.get("parse_error") or row.get("parse_error_code"),
    }
    for key in ("runtime_kind", "checkpoint_fingerprint", "gpu_id"):
        value = row.get(key)
        if value is not None and str(value):
            normalized[key] = str(value)
    inline_gt = row.get("inline_gt")
    if isinstance(inline_gt, list):
        normalized["inline_gt"] = list(inline_gt)
    return normalized


def compute_rollout_phenotype(
    gt_rows: Any,
    rollout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-rollout phenotype rows from normalized A3.2 rollout records."""

    gt_by_image = _build_gt_by_image(gt_rows)
    phenotype_rows: list[dict[str, Any]] = []
    for raw_rollout_row in rollout_rows:
        rollout_row = normalize_rollout_row(raw_rollout_row)
        image_gt = gt_by_image.get(str(rollout_row["image_id"]))
        if image_gt is None:
            image_gt = _inline_gt_record_from_rollout_row(rollout_row)
        phenotype_rows.append(_compute_one_row(image_gt, rollout_row))
    return phenotype_rows


def materialize_rollout_phenotype(
    artifact_root: str | Path,
    gt_rows: Any,
    rollout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Persist Task-4 rollout phenotype rows and summary under ``rollout/``."""

    rollout_dir = Path(artifact_root) / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rows_path = rollout_dir / "rollout_phenotype_rows.jsonl"
    summary_path = rollout_dir / "rollout_summary.json"

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    summary = _summarize_phenotype_rows(rows)

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_strict_json_dumps(row))
            handle.write("\n")
    summary_path.write_text(
        _strict_json_dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"rows_path": str(rows_path), "summary_path": str(summary_path)}


def summarize_rollout_phenotype(
    gt_rows: Any,
    rollout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a JSON-safe A3.2 rollout phenotype summary."""

    rows = compute_rollout_phenotype(gt_rows, rollout_rows)
    return _summarize_phenotype_rows(rows)


def _summarize_phenotype_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicted_row_count_distribution = Counter(
        str(row["pred_row_count"]) for row in rows
    )

    gt_count = sum(int(row["gt_count"]) for row in rows)
    matched_gt_count = sum(int(row["matched_gt_count"]) for row in rows)
    gt_desc_group_count = sum(int(row["gt_desc_group_count"]) for row in rows)
    class_found_desc_count = sum(int(row["class_found_desc_count"]) for row in rows)
    class_found_but_instance_missed_desc_count = sum(
        int(row["class_found_but_instance_missed_desc_count"]) for row in rows
    )
    same_desc_fn_count = sum(int(row["same_desc_fn_count"]) for row in rows)
    valid_pred_row_count = sum(int(row["valid_pred_row_count"]) for row in rows)
    duplicate_side_label_count = sum(
        int(row["duplicate_side_label_count"]) for row in rows
    )
    gt_bearing_row_count = sum(1 for row in rows if int(row["gt_count"]) > 0)
    eos_after_partial_count = sum(
        1 for row in rows if bool(row["eos_after_partial_coverage"])
    )
    order_evaluable_count = sum(
        1 for row in rows if bool(row["sorted_gt_order_evaluable"])
    )
    order_agree_count = sum(
        1
        for row in rows
        if bool(row["sorted_gt_order_evaluable"])
        and bool(row["sorted_gt_order_agreement"])
    )
    invalid_pred_row_count = sum(int(row["invalid_pred_row_count"]) for row in rows)
    malformed_pred_row_count = sum(
        int(row["malformed_pred_row_count"]) for row in rows
    )
    degenerate_box_count = sum(int(row["degenerate_box_count"]) for row in rows)
    parse_failure_count = sum(int(row["parse_failure_count"]) for row in rows)
    early_eos_count = sum(1 for row in rows if bool(row["early_eos"]))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
        "row_count": len(rows),
        "metrics": {
            "class_any_recall": _safe_div(
                class_found_desc_count,
                gt_desc_group_count,
            ),
            "instance_recall": _safe_div(matched_gt_count, gt_count),
            "class_found_but_instance_missed_rate": _safe_div(
                class_found_but_instance_missed_desc_count,
                class_found_desc_count,
            ),
            "same_desc_fn_rate": _safe_div(same_desc_fn_count, gt_count),
            "same_desc_duplication_rate": _safe_div(
                duplicate_side_label_count,
                valid_pred_row_count,
            ),
            "predicted_row_count_distribution": dict(
                sorted(predicted_row_count_distribution.items())
            ),
            "eos_after_partial_coverage_rate": _safe_div(
                eos_after_partial_count,
                gt_bearing_row_count,
            ),
            "sorted_gt_order_agreement": _safe_div(
                order_agree_count,
                order_evaluable_count,
            ),
            "parse_invalid_count": invalid_pred_row_count,
            "parse_failure_count": parse_failure_count,
            "malformed_pred_row_count": malformed_pred_row_count,
            "degenerate_box_count": degenerate_box_count,
            "duplicate_side_label_count": duplicate_side_label_count,
            "early_eos_count": early_eos_count,
        },
    }
    _assert_json_finite(summary)
    return summary


def _is_standard_eval_artifact_row(row: Mapping[str, Any]) -> bool:
    required = {
        "image",
        "width",
        "height",
        "mode",
        "gt",
        "pred",
        "coord_mode",
        "raw_output_json",
        "raw_special_tokens",
        "raw_ends_with_im_end",
        "errors",
        "error_entries",
    }
    return required.issubset(row.keys())


def _standard_eval_artifact_to_normalized_rollout_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = row.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("standard artifact metadata must be a mapping when provided")

    return {
        "checkpoint_role": _policy_value(row, metadata, "checkpoint_role"),
        "image_id": _standard_artifact_image_id(row),
        "source_line_idx": _standard_artifact_source_line_idx(row, metadata),
        "raw_output_text_sha256": _standard_artifact_raw_output_sha256(row),
        "pred_rows_ordered": _standard_artifact_pred_rows(row),
        "invalid_pred_rows": _standard_artifact_invalid_rows(row),
        "rollout_stop_reason": _standard_artifact_stop_reason(row),
        "decode_policy": _policy_value(row, metadata, "decode_policy"),
        "constraint_policy": _policy_value(row, metadata, "constraint_policy"),
        "native_prompt_ordering": _policy_value(
            row,
            metadata,
            "native_prompt_ordering",
        ),
        "template_contract": _standard_artifact_template_contract(row, metadata),
        "inline_gt": _standard_artifact_inline_gt(row),
        "parse_failure": bool(row.get("errors")),
        "parse_error": _first_error_code(row),
        **_standard_artifact_runtime_metadata(row, metadata),
    }


def _standard_artifact_runtime_metadata(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for key in ("runtime_kind", "checkpoint_fingerprint", "gpu_id"):
        value = row.get(key, metadata.get(key))
        if value is not None and str(value):
            output[key] = str(value)
    return output


def _policy_value(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = row.get(key, metadata.get(key, default))
    if value is None or value == "":
        raise ValueError(f"{key} is required for standard artifact adaptation")
    return str(value)


def _standard_artifact_image_id(row: Mapping[str, Any]) -> str:
    value = row.get("image_id")
    if value is None or value == "":
        value = row.get("image")
    if value is None or value == "":
        raise ValueError("standard artifact image_id or image is required")
    return str(value)


def _standard_artifact_source_line_idx(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    sentinel = object()
    value = row.get("source_line_idx", metadata.get("source_line_idx", sentinel))
    if value is sentinel or value is None or value == "":
        raise ValueError("source_line_idx is required for standard artifact adaptation")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_line_idx must be an int") from exc


def _standard_artifact_raw_output_sha256(row: Mapping[str, Any]) -> str:
    existing = row.get("raw_output_text_sha256")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    raw_text = row.get("raw_output_text")
    if isinstance(raw_text, str):
        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_payload = {
        "raw_output_json": row.get("raw_output_json"),
        "raw_special_tokens": row.get("raw_special_tokens"),
        "raw_ends_with_im_end": row.get("raw_ends_with_im_end"),
    }
    encoded = json.dumps(
        raw_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _standard_artifact_pred_rows(row: Mapping[str, Any]) -> list[Any]:
    pred = row.get("pred")
    if isinstance(pred, list):
        return list(pred)
    raw_output_json = row.get("raw_output_json")
    if isinstance(raw_output_json, Mapping) and isinstance(
        raw_output_json.get("objects"),
        list,
    ):
        return list(raw_output_json["objects"])
    raise ValueError("standard artifact pred must be a list")


def _standard_artifact_invalid_rows(row: Mapping[str, Any]) -> list[Any]:
    invalid_rows: list[Any] = []
    error_entries = row.get("error_entries")
    if isinstance(error_entries, list):
        invalid_rows.extend(error_entries)
    errors = row.get("errors")
    if isinstance(errors, list):
        seen_codes = {
            str(entry.get("code"))
            for entry in invalid_rows
            if isinstance(entry, Mapping) and entry.get("code") is not None
        }
        for error in errors:
            code = str(error)
            if code not in seen_codes:
                invalid_rows.append({"code": code})
    return invalid_rows


def _standard_artifact_stop_reason(row: Mapping[str, Any]) -> str:
    explicit = row.get("rollout_stop_reason")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if bool(row.get("raw_ends_with_im_end")):
        return "eos"
    raw_special_tokens = row.get("raw_special_tokens")
    if isinstance(raw_special_tokens, list) and any(
        str(token) == "<|im_end|>" for token in raw_special_tokens
    ):
        return "eos"
    return "max_new_tokens"


def _standard_artifact_template_contract(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = row.get("template_contract", metadata.get("template_contract"))
    if raw is None:
        raise ValueError(
            "template_contract is required for standard artifact adaptation"
        )
    if not isinstance(raw, Mapping):
        raise ValueError("template_contract must be a mapping")
    return raw


def _standard_artifact_inline_gt(row: Mapping[str, Any]) -> list[Any]:
    gt = row.get("gt")
    if not isinstance(gt, list):
        raise ValueError("standard artifact gt must be a list")
    return list(gt)


def _first_error_code(row: Mapping[str, Any]) -> str | None:
    error_entries = row.get("error_entries")
    if isinstance(error_entries, list):
        for entry in error_entries:
            if isinstance(entry, Mapping) and entry.get("code") is not None:
                return str(entry["code"])
    errors = row.get("errors")
    if isinstance(errors, list) and errors:
        return str(errors[0])
    return None


def _inline_gt_record_from_rollout_row(
    rollout_row: Mapping[str, Any],
) -> dict[str, Any]:
    inline_gt = rollout_row.get("inline_gt")
    if not isinstance(inline_gt, list):
        return _empty_gt_record()
    gt_objects = _normalize_gt_objects(inline_gt)
    canonical_sorted_gt_indices = [
        int(obj["gt_idx"]) for obj in _canonical_sorted_gt_objects(gt_objects)
    ]
    return {
        "objects": gt_objects,
        "canonical_sorted_gt_indices": canonical_sorted_gt_indices,
    }


def _compute_one_row(
    image_gt: Mapping[str, Any],
    rollout_row: Mapping[str, Any],
) -> dict[str, Any]:
    gt_objects = list(image_gt["objects"])
    canonical_sorted_gt_indices = list(image_gt["canonical_sorted_gt_indices"])
    pred_stats = _normalize_prediction_rows(rollout_row["pred_rows_ordered"])
    valid_preds = pred_stats["valid_preds"]
    matches = _same_desc_greedy_matches(gt_objects, valid_preds)

    matched_gt_positions = {int(match["gt_pos"]) for match in matches}
    matched_gt_indices = {
        int(gt_objects[gt_pos]["gt_idx"]) for gt_pos in matched_gt_positions
    }
    matched_descs = {
        str(gt_objects[int(match["gt_pos"])]["desc"]) for match in matches
    }
    pred_descs = {str(pred["desc"]) for pred in valid_preds}
    desc_to_gt_positions: dict[str, list[int]] = defaultdict(list)
    for gt_pos, gt in enumerate(gt_objects):
        desc_to_gt_positions[str(gt["desc"])].append(gt_pos)

    class_found_but_instance_missed_descs = [
        desc
        for desc, gt_positions in sorted(desc_to_gt_positions.items())
        if desc in matched_descs and any(pos not in matched_gt_positions for pos in gt_positions)
    ]
    same_desc_fn_gt_indices = [
        int(gt["gt_idx"])
        for gt_pos, gt in enumerate(gt_objects)
        if gt_pos not in matched_gt_positions and str(gt["desc"]) in pred_descs
    ]
    duplicate_side_labels = _same_desc_duplicate_side_labels(valid_preds)
    matched_gt_indices_in_pred_order = [
        int(gt_objects[int(match["gt_pos"])]["gt_idx"])
        for match in sorted(matches, key=lambda item: int(item["pred_idx"]))
    ]
    sorted_gt_order_evaluable = len(matched_gt_indices_in_pred_order) >= 2
    sorted_gt_order_agreement = _is_subsequence_in_canonical_order(
        matched_gt_indices_in_pred_order,
        canonical_sorted_gt_indices,
    )
    early_eos = _is_early_eos(str(rollout_row["rollout_stop_reason"]))
    eos_after_partial_coverage = (
        early_eos
        and len(gt_objects) > 0
        and 0 < len(matched_gt_positions) < len(gt_objects)
    )
    parse_failure_count = 1 if rollout_row.get("parse_failure") or rollout_row.get("parse_error") else 0

    row = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_role": rollout_row["checkpoint_role"],
        "image_id": rollout_row["image_id"],
        "source_line_idx": rollout_row["source_line_idx"],
        "raw_output_text_sha256": rollout_row["raw_output_text_sha256"],
        "source_runtime_kind": rollout_row.get("runtime_kind"),
        "source_checkpoint_fingerprint": rollout_row.get("checkpoint_fingerprint"),
        "source_gpu_id": rollout_row.get("gpu_id"),
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
        "native_prompt_ordering": rollout_row["native_prompt_ordering"],
        "template_contract": {"row_separator": "none"},
        "rollout_stop_reason": rollout_row["rollout_stop_reason"],
        "early_eos": early_eos,
        "gt_count": len(gt_objects),
        "gt_desc_group_count": len(desc_to_gt_positions),
        "pred_row_count": len(rollout_row["pred_rows_ordered"]),
        "valid_pred_row_count": len(valid_preds),
        "invalid_pred_row_count": len(rollout_row["invalid_pred_rows"]),
        "malformed_pred_row_count": pred_stats["malformed_pred_row_count"],
        "degenerate_box_count": pred_stats["degenerate_box_count"],
        "parse_failure_count": parse_failure_count,
        "matched_gt_count": len(matched_gt_positions),
        "matched_pred_count": len({int(match["pred_idx"]) for match in matches}),
        "class_found_desc_count": len(matched_descs),
        "class_found_but_instance_missed_descs": (
            class_found_but_instance_missed_descs
        ),
        "class_found_but_instance_missed_desc_count": len(
            class_found_but_instance_missed_descs
        ),
        "same_desc_fn_gt_indices": same_desc_fn_gt_indices,
        "same_desc_fn_count": len(same_desc_fn_gt_indices),
        "duplicate_side_labels": duplicate_side_labels,
        "duplicate_side_label_count": len(duplicate_side_labels),
        "eos_after_partial_coverage": eos_after_partial_coverage,
        "canonical_sorted_gt_indices": canonical_sorted_gt_indices,
        "matched_gt_indices": sorted(matched_gt_indices),
        "matched_gt_indices_in_pred_order": matched_gt_indices_in_pred_order,
        "sorted_gt_order_evaluable": sorted_gt_order_evaluable,
        "sorted_gt_order_agreement": sorted_gt_order_agreement,
        "matches": matches,
    }
    _assert_json_finite(row)
    return row


def _build_gt_by_image(gt_rows: Any) -> dict[str, dict[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(gt_rows, Mapping):
        if "image_id" in gt_rows and "objects" in gt_rows:
            records = [gt_rows]
        else:
            records = [
                {"image_id": image_id, "objects": objects}
                for image_id, objects in gt_rows.items()
            ]
    elif isinstance(gt_rows, Sequence) and not isinstance(gt_rows, (str, bytes)):
        records = list(gt_rows)
    else:
        raise TypeError("gt_rows must be a mapping or sequence of mappings")

    by_image: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each GT row must be a mapping")
        image_id = _required_str(record, "image_id")
        raw_objects = record.get("objects") or record.get("gt_objects") or []
        if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
            raise ValueError("GT objects must be a sequence")
        gt_objects = _normalize_gt_objects(raw_objects)
        explicit_canonical = record.get("canonical_sorted_gt_indices")
        if explicit_canonical is None:
            canonical_sorted_gt_indices = [
                int(obj["gt_idx"]) for obj in _canonical_sorted_gt_objects(gt_objects)
            ]
        else:
            canonical_sorted_gt_indices = [int(value) for value in explicit_canonical]
        by_image[image_id] = {
            "objects": gt_objects,
            "canonical_sorted_gt_indices": canonical_sorted_gt_indices,
        }
    return by_image


def _normalize_gt_objects(raw_objects: Sequence[Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for fallback_idx, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            continue
        desc = _row_desc(raw)
        if not desc:
            continue
        try:
            bbox = _row_bbox(raw)
        except ValueError:
            continue
        if _is_degenerate_bbox(bbox):
            continue
        gt_idx = int(raw.get("gt_idx", raw.get("gt_index", fallback_idx)))
        objects.append(
            {
                "gt_idx": gt_idx,
                "desc": _canonical_desc(desc),
                "bbox_xyxy": list(bbox),
            }
        )
    return objects


def _normalize_prediction_rows(raw_preds: Sequence[Any]) -> dict[str, Any]:
    valid_preds: list[dict[str, Any]] = []
    malformed_pred_row_count = 0
    degenerate_box_count = 0
    for pred_idx, raw in enumerate(raw_preds):
        if not isinstance(raw, Mapping):
            malformed_pred_row_count += 1
            continue
        desc = _row_desc(raw)
        if not desc:
            malformed_pred_row_count += 1
            continue
        try:
            bbox = _row_bbox(raw)
        except ValueError:
            malformed_pred_row_count += 1
            continue
        if _is_degenerate_bbox(bbox):
            degenerate_box_count += 1
            continue
        valid_preds.append(
            {
                "pred_idx": pred_idx,
                "desc": _canonical_desc(desc),
                "bbox_xyxy": list(bbox),
            }
        )
    return {
        "valid_preds": valid_preds,
        "malformed_pred_row_count": malformed_pred_row_count,
        "degenerate_box_count": degenerate_box_count,
    }


def _same_desc_greedy_matches(
    gt_objects: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for pred in preds:
        pred_idx = int(pred["pred_idx"])
        for gt_pos, gt in enumerate(gt_objects):
            if str(pred["desc"]) != str(gt["desc"]):
                continue
            iou = bbox_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if iou >= MATCH_IOU_THRESHOLD:
                candidates.append((iou, pred_idx, gt_pos))

    used_preds: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, pred_idx, gt_pos in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if pred_idx in used_preds or gt_pos in used_gt:
            continue
        used_preds.add(pred_idx)
        used_gt.add(gt_pos)
        matches.append(
            {
                "pred_idx": pred_idx,
                "gt_idx": int(gt_objects[gt_pos]["gt_idx"]),
                "gt_pos": gt_pos,
                "desc": str(gt_objects[gt_pos]["desc"]),
                "iou": _json_float(iou),
            }
        )
    return sorted(matches, key=lambda item: int(item["pred_idx"]))


def _same_desc_duplicate_side_labels(
    preds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    duplicate_pred_indices: set[int] = set()
    for left_pos, left in enumerate(preds):
        for right in preds[left_pos + 1 :]:
            if str(left["desc"]) != str(right["desc"]):
                continue
            iou = bbox_iou(left["bbox_xyxy"], right["bbox_xyxy"])
            if iou <= DUPLICATE_IOU_THRESHOLD:
                continue
            pred_idx = int(right["pred_idx"])
            if pred_idx in duplicate_pred_indices:
                continue
            duplicate_pred_indices.add(pred_idx)
            labels.append(
                {
                    "label": "same_desc_duplicate",
                    "desc": str(right["desc"]),
                    "pred_idx": pred_idx,
                    "duplicate_of_pred_idx": int(left["pred_idx"]),
                    "iou": _json_float(iou),
                }
            )
    return labels


def bbox_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_box = _coerce_bbox_xyxy(left)
    right_box = _coerce_bbox_xyxy(right)
    left_area = _bbox_area(left_box)
    right_area = _bbox_area(right_box)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    ix1 = max(left_box[0], right_box[0])
    iy1 = max(left_box[1], right_box[1])
    ix2 = min(left_box[2], right_box[2])
    iy2 = min(left_box[3], right_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left_area + right_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _row_desc(row: Mapping[str, Any]) -> str:
    desc = (
        row.get("desc")
        or row.get("description")
        or row.get("desc_text")
        or row.get("desc_text_canonical")
        or row.get("category_name")
    )
    return _canonical_desc(str(desc)) if desc is not None else ""


def _row_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    for key in ("bbox_xyxy", "bbox_2d", "bbox"):
        if key in row:
            return _coerce_bbox_xyxy(row[key])
    raise ValueError("row is missing bbox")


def _coerce_bbox_xyxy(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("bbox must be a four-element sequence")
    if len(value) != 4:
        raise ValueError("bbox must contain four values")
    return tuple(_coerce_coord(coord) for coord in value)  # type: ignore[return-value]


def _coerce_coord(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bbox coordinates must not be bools")
    if isinstance(value, (int, float)):
        coord = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        match = _COORD_TOKEN_RE.match(stripped)
        if match:
            coord = float(match.group(1))
        else:
            try:
                coord = float(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid coordinate value: {value!r}") from exc
    else:
        raise ValueError(f"invalid coordinate value: {value!r}")
    if not math.isfinite(coord):
        raise ValueError(f"coordinate must be finite: {value!r}")
    return coord


def _is_degenerate_bbox(bbox: Sequence[Any]) -> bool:
    box = _coerce_bbox_xyxy(bbox)
    return _bbox_area(box) <= 0.0


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0,
        float(bbox[3]) - float(bbox[1]),
    )


def _canonical_sorted_gt_objects(
    gt_objects: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        gt_objects,
        key=lambda obj: (
            float(obj["bbox_xyxy"][1]),
            float(obj["bbox_xyxy"][0]),
            int(obj["gt_idx"]),
        ),
    )


def _is_subsequence_in_canonical_order(
    matched_gt_indices_in_pred_order: Sequence[int],
    canonical_sorted_gt_indices: Sequence[int],
) -> bool:
    canonical_rank = {
        int(gt_idx): rank for rank, gt_idx in enumerate(canonical_sorted_gt_indices)
    }
    ranks: list[int] = []
    for gt_idx in matched_gt_indices_in_pred_order:
        if int(gt_idx) not in canonical_rank:
            return False
        ranks.append(canonical_rank[int(gt_idx)])
    return ranks == sorted(ranks)


def _is_early_eos(stop_reason: str) -> bool:
    normalized = stop_reason.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "eos",
        "early_eos",
        "stop_token",
        "im_end",
        "end_of_sequence",
    }


def _raw_output_sha256(row: Mapping[str, Any]) -> str:
    existing = row.get("raw_output_text_sha256")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    raw_text = row.get("raw_output_text")
    if isinstance(raw_text, str):
        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raise ValueError("raw_output_text_sha256 is required")


def _empty_gt_record() -> dict[str, Any]:
    return {"objects": [], "canonical_sorted_gt_indices": []}


def _canonical_desc(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def _required_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_list(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required")
    return str(value)


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an int") from exc


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return _json_float(float(numerator) / float(denominator))


def _json_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _assert_json_finite(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in rollout phenotype output")
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_json_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_json_finite(child)


def _strict_json_dumps(value: Any, *, indent: int | None = None) -> str:
    _assert_json_finite(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=None if indent is not None else (",", ":"),
        indent=indent,
    )


__all__ = [
    "CONSTRAINT_POLICY",
    "DECODE_POLICY",
    "DUPLICATE_IOU_THRESHOLD",
    "MATCH_IOU_THRESHOLD",
    "SCHEMA_VERSION",
    "bbox_iou",
    "compute_rollout_phenotype",
    "materialize_rollout_phenotype",
    "normalize_rollout_row",
    "summarize_rollout_phenotype",
]
