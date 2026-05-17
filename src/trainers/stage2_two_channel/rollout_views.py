from dataclasses import replace
from typing import Any, Dict, List, Mapping, Tuple

from src.common.duplicate_control import duplicate_control_object_from_bbox
from src.training.stage2.rollout_codec import (
    CompactFullRolloutCodec,
    Stage2RolloutTemplatePolicy,
)

from ..rollout_matching.contracts import GTObject


def _response_text_from_rollout(
    *,
    tokenizer: Any,
    response_token_ids: List[int],
    response_text: str,
) -> str:
    text = str(response_text or "")
    if text:
        return text

    return str(
        tokenizer.decode(
            [int(t) for t in response_token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _view_from_bbox_objects(
    *,
    parsed_objects: List[GTObject],
    source_label: str,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    duplicate_diagnostics_fn: Any,
) -> tuple[List[Any], Mapping[str, Any]]:
    duplicate_control_objects_raw = [
        duplicate_control_object_from_bbox(
            index=int(index),
            desc=str(obj.desc),
            bbox_norm1000=obj.points_norm1000,
            source=str(source_label),
        )
        for index, obj in enumerate(parsed_objects)
    ]
    duplicate_metrics = duplicate_diagnostics_fn(
        parsed_objects,
        duplicate_iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
    )
    return duplicate_control_objects_raw, duplicate_metrics


def _build_compact_full_rollout_view(
    *,
    tokenizer: Any,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    max_new_tokens: int,
    rollout_result: Tuple[List[int], str, str, List[int]],
    source_label: str,
    duplicate_diagnostics_fn: Any,
    rollout_template_policy: Stage2RolloutTemplatePolicy,
) -> Dict[str, Any]:
    resp_ids, resp_text, rollout_decode_mode, prompt_ids = rollout_result
    resp_ids_local = [int(t) for t in resp_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    truncated_by_budget = False
    if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
        try:
            eos = int(eos_id) if eos_id is not None else -1
        except (TypeError, ValueError):
            eos = -1
        if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
            resp_ids_local.append(int(eos))
            truncated_by_budget = True

    response_text = _response_text_from_rollout(
        tokenizer=tokenizer,
        response_token_ids=resp_ids_local,
        response_text=resp_text,
    )
    codec = CompactFullRolloutCodec(rollout_template_policy)
    parse = codec.parse(response_text)
    parse = replace(
        parse,
        response_token_ids=tuple(int(t) for t in resp_ids_local),
        truncated=bool(parse.truncated or truncated_by_budget),
    )

    drop_reasons: Dict[str, int] = {}
    drop_bbox_invalid = 0
    parsed_bbox_objects_raw: List[GTObject] = []
    for pobj in list(parse.valid_objects):
        if pobj.geom_type != "bbox_2d" or pobj.bbox_norm1000 is None:
            drop_bbox_invalid += 1
            continue
        try:
            x1, y1, x2, y2 = [int(x) for x in pobj.bbox_norm1000]
        except (TypeError, ValueError):
            drop_bbox_invalid += 1
            continue
        if x2 <= x1 or y2 <= y1:
            drop_bbox_invalid += 1
            continue

        parsed_bbox_objects_raw.append(
            GTObject(
                index=int(pobj.index),
                geom_type="bbox_2d",
                points_norm1000=[x1, y1, x2, y2],
                desc=str(pobj.desc),
            )
        )

    if drop_bbox_invalid:
        drop_reasons["bbox_invalid"] = int(drop_bbox_invalid)

    parse_empty_after_geometry_filter = bool(
        parse.valid_objects and not parsed_bbox_objects_raw
    )
    if parse_empty_after_geometry_filter:
        parse = replace(
            parse,
            empty_valid_object_set=True,
            fallback_reason="empty_valid_object_set",
            dropped_invalid=int(parse.dropped_invalid) + int(drop_bbox_invalid),
            dropped_invalid_by_reason={
                **dict(parse.dropped_invalid_by_reason),
                "bbox_invalid": int(drop_bbox_invalid),
            },
        )

    duplicate_control_objects_raw, duplicate_metrics = _view_from_bbox_objects(
        parsed_objects=parsed_bbox_objects_raw,
        source_label=source_label,
        duplicate_iou_threshold=duplicate_iou_threshold,
        center_radius_scale=center_radius_scale,
        duplicate_diagnostics_fn=duplicate_diagnostics_fn,
    )
    fallback_applies = bool(parse.invalid_rollout or parse.empty_valid_object_set)

    return {
        "prompt_ids": [int(t) for t in prompt_ids],
        "decode_mode": str(rollout_decode_mode),
        "pred_objects": int(len(parse.valid_objects)),
        "parse_truncated": int(1 if bool(parse.truncated) else 0),
        "gen_new_tokens": int(len(parse.response_token_ids)),
        "parse": parse,
        "invalid_rollout": int(1 if bool(parse.invalid_rollout) else 0),
        "empty_valid_object_set": int(
            1 if bool(parse.empty_valid_object_set) else 0
        ),
        "fallback_reason": parse.fallback_reason,
        "compact_fallback_applies": int(1 if fallback_applies else 0),
        "drop_reasons": drop_reasons,
        "drop_poly": int(0),
        "drop_unknown": int(0),
        "drop_bbox_invalid": int(drop_bbox_invalid),
        "parsed_bbox_objects_raw": parsed_bbox_objects_raw,
        "duplicate_control_objects_raw": duplicate_control_objects_raw,
        "n_valid_pred": int(len(parsed_bbox_objects_raw)),
        "n_drop_invalid": int(drop_bbox_invalid),
        "duplicate_metrics": duplicate_metrics,
        "rollout_template_family": "compact_full",
        "rollout_parser_id": str(rollout_template_policy.parser_id),
        "rollout_append_policy_id": str(rollout_template_policy.append_policy_id),
        "rollout_counts_as_valid_rollout": int(0 if fallback_applies else 1),
        "rollout_fallback_loss_weight": (
            float(rollout_template_policy.fallback_loss_weight)
            if fallback_applies
            else 0.0
        ),
    }


def build_channel_b_rollout_view(
    *,
    tokenizer: Any,
    object_field_order: str,
    coord_id_to_bin: Mapping[int, int],
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    max_new_tokens: int,
    rollout_result: Tuple[List[int], str, str, List[int]],
    source_label: str,
    parse_rollout_for_matching_fn: Any,
    points_from_coord_tokens_fn: Any,
    duplicate_diagnostics_fn: Any,
    rollout_template_policy: Stage2RolloutTemplatePolicy | None = None,
) -> Dict[str, Any]:
    if (
        rollout_template_policy is not None
        and rollout_template_policy.template_family == "compact_full"
    ):
        return _build_compact_full_rollout_view(
            tokenizer=tokenizer,
            duplicate_iou_threshold=duplicate_iou_threshold,
            center_radius_scale=center_radius_scale,
            max_new_tokens=max_new_tokens,
            rollout_result=rollout_result,
            source_label=source_label,
            duplicate_diagnostics_fn=duplicate_diagnostics_fn,
            rollout_template_policy=rollout_template_policy,
        )

    resp_ids, _resp_text, rollout_decode_mode, prompt_ids = rollout_result
    resp_ids_local = [int(t) for t in resp_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if int(max_new_tokens) > 0 and int(len(resp_ids_local)) >= int(max_new_tokens):
        try:
            eos = int(eos_id) if eos_id is not None else -1
        except (TypeError, ValueError):
            eos = -1
        if eos >= 0 and (not resp_ids_local or int(resp_ids_local[-1]) != eos):
            resp_ids_local.append(int(eos))

    parse = parse_rollout_for_matching_fn(
        tokenizer=tokenizer,
        response_token_ids=resp_ids_local,
        object_field_order=object_field_order,
    )
    invalid_rollout = int(1 if bool(getattr(parse, "invalid_rollout", False)) else 0)

    drop_reasons: Dict[str, int] = {}
    raw = getattr(parse, "dropped_invalid_by_reason", None)
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            try:
                drop_reasons[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

    drop_poly = 0
    drop_unknown = 0
    drop_bbox_invalid = 0
    parsed_bbox_objects_raw: List[GTObject] = []
    for pobj in list(parse.valid_objects):
        if pobj.geom_type != "bbox_2d":
            if pobj.geom_type == "poly":
                drop_poly += 1
            else:
                drop_unknown += 1
            continue

        pts = points_from_coord_tokens_fn(
            response_token_ids=parse.response_token_ids,
            coord_token_indices=pobj.coord_token_indices,
            coord_id_to_bin=coord_id_to_bin,
        )
        if pts is None or len(pts) != 4:
            drop_bbox_invalid += 1
            continue
        try:
            x1, y1, x2, y2 = [int(x) for x in pts]
        except (TypeError, ValueError):
            drop_bbox_invalid += 1
            continue
        # Keep the rollout view contract aligned with duplicate-control:
        # zero-width / zero-height boxes are invalid and should be dropped.
        if x2 <= x1 or y2 <= y1:
            drop_bbox_invalid += 1
            continue

        parsed_bbox_objects_raw.append(
            GTObject(
                index=int(pobj.index),
                geom_type="bbox_2d",
                points_norm1000=[x1, y1, x2, y2],
                desc=str(pobj.desc),
            )
        )

    if drop_poly:
        drop_reasons["poly_unsupported"] = int(
            drop_reasons.get("poly_unsupported", 0)
        ) + int(drop_poly)
    if drop_unknown:
        drop_reasons["unknown_geom"] = int(drop_reasons.get("unknown_geom", 0)) + int(
            drop_unknown
        )
    if drop_bbox_invalid:
        drop_reasons["bbox_invalid"] = int(drop_reasons.get("bbox_invalid", 0)) + int(
            drop_bbox_invalid
        )

    n_valid_pred = int(len(parsed_bbox_objects_raw))
    n_drop_invalid = (
        int(getattr(parse, "dropped_invalid", 0) or 0)
        + int(getattr(parse, "dropped_ambiguous", 0) or 0)
        + int(drop_poly)
        + int(drop_unknown)
        + int(drop_bbox_invalid)
    )

    duplicate_control_objects_raw, duplicate_metrics = _view_from_bbox_objects(
        parsed_objects=parsed_bbox_objects_raw,
        source_label=source_label,
        duplicate_iou_threshold=duplicate_iou_threshold,
        center_radius_scale=center_radius_scale,
        duplicate_diagnostics_fn=duplicate_diagnostics_fn,
    )

    return {
        "prompt_ids": [int(t) for t in prompt_ids],
        "decode_mode": str(rollout_decode_mode),
        "pred_objects": int(len(parse.valid_objects)),
        "parse_truncated": int(1 if bool(getattr(parse, "truncated", False)) else 0),
        "gen_new_tokens": int(len(parse.response_token_ids)),
        "parse": parse,
        "invalid_rollout": int(invalid_rollout),
        "empty_valid_object_set": int(0),
        "fallback_reason": None,
        "compact_fallback_applies": int(0),
        "drop_reasons": drop_reasons,
        "drop_poly": int(drop_poly),
        "drop_unknown": int(drop_unknown),
        "drop_bbox_invalid": int(drop_bbox_invalid),
        "parsed_bbox_objects_raw": parsed_bbox_objects_raw,
        "duplicate_control_objects_raw": duplicate_control_objects_raw,
        "n_valid_pred": int(n_valid_pred),
        "n_drop_invalid": int(n_drop_invalid),
        "duplicate_metrics": duplicate_metrics,
        "rollout_template_family": "coordjson",
        "rollout_parser_id": (
            str(rollout_template_policy.parser_id)
            if rollout_template_policy is not None
            else "coordjson_legacy"
        ),
        "rollout_append_policy_id": (
            str(rollout_template_policy.append_policy_id)
            if rollout_template_policy is not None
            else "coordjson_append"
        ),
        "rollout_counts_as_valid_rollout": int(0 if invalid_rollout else 1),
        "rollout_fallback_loss_weight": float(0.0),
    }


__all__ = ["build_channel_b_rollout_view"]
