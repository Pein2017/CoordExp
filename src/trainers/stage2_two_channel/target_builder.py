from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.common.semantic_desc import normalize_desc
from src.common.object_field_order import build_object_payload
from src.utils.assistant_json import dumps_coordjson

from ..rollout_matching.contracts import GTObject
from ..rollout_matching.matching import associate_one_to_one_max_iou
from ..rollout_matching.parsing import decode_pieces, find_desc_value_char_spans
from .types import Stage2DeadAnchorSuppressionTarget


@dataclass(frozen=True)
class _ValueSpanObject:
    value_span: Tuple[int, int]


@dataclass(frozen=True)
class _CanonicalPrefixData:
    prefix_text: str
    prefix_token_ids: List[int]
    boundary_prefix_texts: List[str]
    object_value_spans: List[Tuple[int, int]]


@dataclass(frozen=True)
class _ChannelBTriageResult:
    association_pairs: List[Tuple[int, int]]
    anchor_gt_backed_indices: List[int]
    shielded_anchor_indices: List[int]
    dead_anchor_indices: List[int]
    dead_explorer_indices: List[int]
    recovered_gt_indices: List[int]
    kept_anchor_objects: List[GTObject]
    kept_anchor_new_index_by_old: Dict[int, int]
    dead_anchor_bursts_by_boundary: Dict[int, List[GTObject]]


def _bbox_iou_norm1000_xyxy(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
        bx1, by1, bx2, by2 = [int(v) for v in box_b]
    except (TypeError, ValueError):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = float(area_a + area_b - inter)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _compute_duplicate_diagnostics(
    parsed_bbox_objects_raw: Sequence[GTObject],
) -> Dict[str, float]:
    norm_descs = [normalize_desc(obj.desc) for obj in parsed_bbox_objects_raw]
    max_desc_count = 0
    if norm_descs:
        counts = Counter(norm_descs)
        max_desc_count = int(max(counts.values()))

    saturated = 0
    for obj in parsed_bbox_objects_raw:
        if any(int(v) in {0, 999} for v in obj.points_norm1000):
            saturated += 1

    near_same_desc = 0
    near_any_desc = 0
    for i, obj_i in enumerate(parsed_bbox_objects_raw):
        for j in range(int(i + 1), int(len(parsed_bbox_objects_raw))):
            obj_j = parsed_bbox_objects_raw[j]
            iou = _bbox_iou_norm1000_xyxy(obj_i.points_norm1000, obj_j.points_norm1000)
            if iou < 0.90:
                continue
            near_any_desc += 1
            if norm_descs[i] == norm_descs[j]:
                near_same_desc += 1

    n_raw = int(len(parsed_bbox_objects_raw))
    saturation_rate = (float(saturated) / float(n_raw)) if n_raw > 0 else 0.0
    return {
        "dup/max_desc_count": float(max_desc_count),
        "dup/saturation_rate": float(saturation_rate),
        "dup/near_iou90_pairs_same_desc_count": float(near_same_desc),
        "dup/near_iou90_pairs_any_desc_count": float(near_any_desc),
    }


def _sequential_dedup_bbox_objects(
    *,
    parsed_bbox_objects_raw: Sequence[GTObject],
    duplicate_iou_threshold: float,
) -> Tuple[List[GTObject], Dict[int, List[GTObject]]]:
    accepted_objects_clean: List[GTObject] = []
    accepted_norm_descs: List[str] = []
    duplicate_bursts_by_boundary: Dict[int, List[GTObject]] = {}

    for obj in parsed_bbox_objects_raw:
        boundary = int(len(accepted_objects_clean))
        obj_norm_desc = normalize_desc(obj.desc)
        is_duplicate = False

        for accepted, accepted_norm_desc in zip(
            accepted_objects_clean,
            accepted_norm_descs,
        ):
            if obj_norm_desc != accepted_norm_desc:
                continue
            if (
                _bbox_iou_norm1000_xyxy(
                    obj.points_norm1000,
                    accepted.points_norm1000,
                )
                < float(duplicate_iou_threshold)
            ):
                continue
            duplicate_bursts_by_boundary.setdefault(boundary, []).append(obj)
            is_duplicate = True
            break

        if is_duplicate:
            continue

        accepted_objects_clean.append(obj)
        accepted_norm_descs.append(obj_norm_desc)

    return accepted_objects_clean, duplicate_bursts_by_boundary


def _build_channel_b_triage(
    *,
    accepted_objects_clean: Sequence[GTObject],
    explorer_accepted_objects_clean: Sequence[GTObject],
    anchor_match_by_pred: Mapping[int, int],
    explorer_match_by_pred: Mapping[int, int],
    unlabeled_consistent_iou_threshold: float,
) -> _ChannelBTriageResult:
    association_pairs = [
        (int(anchor_i), int(explorer_i))
        for anchor_i, explorer_i in associate_one_to_one_max_iou(
            anchors=accepted_objects_clean,
            explorers=explorer_accepted_objects_clean,
            min_iou=float(unlabeled_consistent_iou_threshold),
        )
    ]
    anchor_to_explorer = {
        int(anchor_i): int(explorer_i)
        for anchor_i, explorer_i in association_pairs
    }

    anchor_gt_backed_indices = sorted(int(pred_i) for pred_i in anchor_match_by_pred.keys())
    shielded_anchor_indices: set[int] = set()
    dead_anchor_indices: set[int] = set()
    for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
        if int(anchor_i) in anchor_match_by_pred:
            continue
        explorer_i = anchor_to_explorer.get(int(anchor_i))
        if explorer_i is not None and explorer_i not in explorer_match_by_pred:
            conflicts_gt_backed = any(
                _bbox_iou_norm1000_xyxy(
                    anchor_obj.points_norm1000,
                    accepted_objects_clean[int(gt_anchor_i)].points_norm1000,
                )
                >= float(unlabeled_consistent_iou_threshold)
                for gt_anchor_i in anchor_gt_backed_indices
            )
            if not conflicts_gt_backed:
                shielded_anchor_indices.add(int(anchor_i))
                continue
        dead_anchor_indices.add(int(anchor_i))

    anchor_gt_indices = set(int(gt_i) for gt_i in anchor_match_by_pred.values())
    explorer_gt_indices = set(int(gt_i) for gt_i in explorer_match_by_pred.values())
    recovered_gt_indices = sorted(
        int(gt_i) for gt_i in (explorer_gt_indices - anchor_gt_indices)
    )
    dead_explorer_indices = sorted(
        int(explorer_i)
        for explorer_i in range(len(explorer_accepted_objects_clean))
        if int(explorer_i) not in explorer_match_by_pred
    )

    kept_anchor_objects: List[GTObject] = []
    kept_anchor_new_index_by_old: Dict[int, int] = {}
    dead_anchor_bursts_by_boundary: Dict[int, List[GTObject]] = {}
    kept_anchor_count = 0
    for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
        if int(anchor_i) in dead_anchor_indices:
            dead_anchor_bursts_by_boundary.setdefault(int(kept_anchor_count), []).append(
                anchor_obj
            )
            continue
        kept_anchor_new_index_by_old[int(anchor_i)] = int(len(kept_anchor_objects))
        kept_anchor_objects.append(anchor_obj)
        kept_anchor_count += 1

    return _ChannelBTriageResult(
        association_pairs=association_pairs,
        anchor_gt_backed_indices=[int(idx) for idx in anchor_gt_backed_indices],
        shielded_anchor_indices=[int(idx) for idx in sorted(shielded_anchor_indices)],
        dead_anchor_indices=[int(idx) for idx in sorted(dead_anchor_indices)],
        dead_explorer_indices=[int(idx) for idx in dead_explorer_indices],
        recovered_gt_indices=[int(idx) for idx in recovered_gt_indices],
        kept_anchor_objects=kept_anchor_objects,
        kept_anchor_new_index_by_old=kept_anchor_new_index_by_old,
        dead_anchor_bursts_by_boundary=dead_anchor_bursts_by_boundary,
    )


def _serialize_gt_object_entry(
    *,
    obj: GTObject,
    object_field_order: str,
) -> str:
    if str(obj.geom_type) != "bbox_2d":
        raise ValueError(
            f"Channel-B clean-prefix v1 only supports bbox_2d objects; got {obj.geom_type!r}"
        )
    if len(obj.points_norm1000) != 4:
        raise ValueError(
            "Channel-B clean-prefix v1 requires bbox_2d objects with four coord bins"
        )

    payload = build_object_payload(
        desc=str(obj.desc),
        geometry_key="bbox_2d",
        geometry_value=[f"<|coord_{int(v)}|>" for v in obj.points_norm1000],
        object_field_order=object_field_order,
    )
    return dumps_coordjson(payload)


def _build_canonical_prefix_text_data(
    *,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    empty_container = dumps_coordjson({"objects": []})
    if not str(empty_container).endswith("]}"):
        raise ValueError(
            "unexpected canonical CoordJSON container rendering for empty objects list"
        )

    prefix_text = str(empty_container[:-2])
    boundary_prefix_texts: List[str] = [str(prefix_text)]
    object_value_spans: List[Tuple[int, int]] = []

    for obj in objects:
        entry_text = _serialize_gt_object_entry(
            obj=obj,
            object_field_order=object_field_order,
        )
        if not prefix_text.endswith("["):
            prefix_text = prefix_text + ", "
        start = int(len(prefix_text))
        prefix_text = prefix_text + str(entry_text)
        object_value_spans.append((start, int(len(prefix_text))))
        boundary_prefix_texts.append(str(prefix_text))

    return prefix_text, boundary_prefix_texts, object_value_spans


def _build_canonical_prefix_data(
    *,
    tokenizer: Any,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> _CanonicalPrefixData:
    prefix_text, boundary_prefix_texts, object_value_spans = (
        _build_canonical_prefix_text_data(
            objects=objects,
            object_field_order=object_field_order,
        )
    )
    prefix_token_ids = [
        int(t) for t in tokenizer.encode(prefix_text, add_special_tokens=False)
    ]
    return _CanonicalPrefixData(
        prefix_text=str(prefix_text),
        prefix_token_ids=prefix_token_ids,
        boundary_prefix_texts=[str(t) for t in boundary_prefix_texts],
        object_value_spans=[tuple(span) for span in object_value_spans],
    )


def _build_canonical_closed_container_text(
    *,
    objects: Sequence[GTObject],
    object_field_order: str,
) -> str:
    prefix_text, _boundary_prefix_texts, _value_spans = _build_canonical_prefix_text_data(
        objects=objects,
        object_field_order=object_field_order,
    )
    return str(prefix_text) + "]}"


def _token_piece_char_spans(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    pieces = decode_pieces(tokenizer, token_ids)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = int(cursor)
        cursor += int(len(piece))
        spans.append((start, int(cursor)))
    return spans


def _first_safe_token_index_from_char_cut(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    cut_char_pos: int,
) -> int:
    if cut_char_pos <= 0 or not token_ids:
        return 0

    # Keep any token that starts before the clean boundary in the prefix/context.
    # This avoids retokenizing token-internal char cuts into synthetic positions that
    # do not exist in the actual teacher-forced target tokenization.
    for idx, (start, _end) in enumerate(
        _token_piece_char_spans(tokenizer=tokenizer, token_ids=token_ids)
    ):
        if int(start) >= int(cut_char_pos):
            return int(idx)
    return int(len(token_ids))


def _build_dead_anchor_suppression_targets(
    *,
    tokenizer: Any,
    y_train_ids: Sequence[int],
    clean_target_text: str,
    accepted_objects_clean: Sequence[GTObject],
    fn_objects: Sequence[GTObject],
    duplicate_bursts_by_boundary: Mapping[int, Sequence[GTObject]],
    boundary_prefix_texts: Sequence[str],
    object_field_order: str,
) -> Tuple[List[Stage2DeadAnchorSuppressionTarget], int, int]:
    targets_by_boundary_token: dict[
        tuple[int, int], Stage2DeadAnchorSuppressionTarget
    ] = {}
    skipped_no_divergence = 0

    y_train_ids_list = [int(t) for t in y_train_ids]
    clean_target_text_s = str(clean_target_text)

    for boundary, duplicates in sorted(duplicate_bursts_by_boundary.items()):
        boundary_i = int(boundary)
        if boundary_i < 0 or boundary_i >= len(boundary_prefix_texts):
            raise ValueError(
                f"duplicate burst boundary is outside clean-prefix range: {boundary_i}"
            )

        boundary_prefix_text = str(boundary_prefix_texts[boundary_i])
        if not clean_target_text_s.startswith(boundary_prefix_text):
            raise ValueError(
                "clean teacher-forced target does not share the declared boundary prefix"
            )

        boundary_char_pos = int(len(boundary_prefix_text))
        clean_boundary_token_idx = _first_safe_token_index_from_char_cut(
            tokenizer=tokenizer,
            token_ids=y_train_ids_list,
            cut_char_pos=boundary_char_pos,
        )

        for dup in duplicates:
            duplicate_target_text = _build_canonical_closed_container_text(
                objects=(
                    list(accepted_objects_clean[:boundary_i])
                    + [dup]
                    + list(accepted_objects_clean[boundary_i:])
                    + list(fn_objects)
                ),
                object_field_order=object_field_order,
            )
            if not duplicate_target_text.startswith(boundary_prefix_text):
                raise ValueError(
                    "duplicate continuation does not preserve the declared clean boundary prefix"
                )

            duplicate_target_ids = [
                int(t)
                for t in tokenizer.encode(
                    duplicate_target_text,
                    add_special_tokens=False,
                )
            ]
            duplicate_boundary_token_idx = _first_safe_token_index_from_char_cut(
                tokenizer=tokenizer,
                token_ids=duplicate_target_ids,
                cut_char_pos=boundary_char_pos,
            )

            clean_pos = int(clean_boundary_token_idx)
            duplicate_pos = int(duplicate_boundary_token_idx)
            while (
                clean_pos < len(y_train_ids_list)
                and duplicate_pos < len(duplicate_target_ids)
                and y_train_ids_list[clean_pos] == duplicate_target_ids[duplicate_pos]
            ):
                clean_pos += 1
                duplicate_pos += 1

            if clean_pos >= len(y_train_ids_list) or duplicate_pos >= len(
                duplicate_target_ids
            ):
                skipped_no_divergence += 1
                continue

            rel_pos = int(clean_pos)
            bad_token_id = int(duplicate_target_ids[duplicate_pos])
            candidate: Stage2DeadAnchorSuppressionTarget = {
                "boundary": int(boundary_i),
                "rel_pos": int(rel_pos),
                "token_id": int(bad_token_id),
            }
            key = (int(boundary_i), int(bad_token_id))
            existing = targets_by_boundary_token.get(key)
            if existing is None or int(candidate["rel_pos"]) < int(existing["rel_pos"]):
                targets_by_boundary_token[key] = candidate

    targets = sorted(
        targets_by_boundary_token.values(),
        key=lambda item: (
            int(item["boundary"]),
            int(item["rel_pos"]),
            int(item["token_id"]),
        ),
    )
    dead_anchor_suppression_boundary_count = len(
        {int(item["boundary"]) for item in targets}
    )
    return targets, int(dead_anchor_suppression_boundary_count), int(
        skipped_no_divergence
    )


def _desc_tail_positions_and_weights(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    object_weights: Sequence[float],
) -> Tuple[List[int], List[float]]:
    ids = [int(t) for t in token_ids]
    if not ids:
        return [], []

    pieces = decode_pieces(tokenizer, ids)
    text = "".join(pieces)
    desc_spans = find_desc_value_char_spans(text)
    if not desc_spans:
        return [], []
    if len(desc_spans) != len(object_weights):
        raise ValueError(
            "Channel-B FN desc spans do not align with fn_object_weights: "
            f"spans={len(desc_spans)} weights={len(object_weights)}"
        )

    token_spans = _token_piece_char_spans(tokenizer=tokenizer, token_ids=ids)
    positions: List[int] = []
    weights: List[float] = []
    for (start_char, end_char), weight in zip(desc_spans, object_weights):
        for token_i, (token_start, token_end) in enumerate(token_spans):
            if int(token_start) < int(end_char) and int(token_end) > int(start_char):
                positions.append(int(token_i))
                weights.append(float(weight))

    return positions, weights


__all__ = [
    "_ValueSpanObject",
    "_CanonicalPrefixData",
    "_ChannelBTriageResult",
    "_build_channel_b_triage",
    "_bbox_iou_norm1000_xyxy",
    "_compute_duplicate_diagnostics",
    "_build_canonical_prefix_data",
    "_build_dead_anchor_suppression_targets",
    "_desc_tail_positions_and_weights",
    "_sequential_dedup_bbox_objects",
]
