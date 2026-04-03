from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.common.duplicate_control import (
    DuplicateControlDecision,
    DuplicateControlResult,
    compute_duplicate_metrics,
    duplicate_control_object_from_bbox,
    apply_duplicate_policy,
    validate_duplicate_control_config,
)
from src.common.semantic_desc import normalize_desc
from src.common.object_field_order import build_object_payload
from src.utils.assistant_json import dumps_coordjson

from ..rollout_matching.contracts import GTObject, MatchResult
from ..rollout_matching.matching import associate_one_to_one_max_iou
from ..rollout_matching.parsing import decode_pieces, find_desc_value_char_spans
from .types import Stage2ChannelBMeta, Stage2DuplicateBurstUnlikelihoodTarget


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
class _ChannelBDuplicateControlResult:
    kept_anchor_objects: List[GTObject]
    suppressed_duplicate_objects_by_boundary: Dict[int, List[GTObject]]
    decisions: List[DuplicateControlDecision]
    support_counts: List[int]
    support_rates: List[float]
    survivor_anchor_indices: List[int]
    exempt_anchor_indices: List[int]
    suppressed_anchor_indices: List[int]
    counter_metrics: Dict[str, float]


@dataclass(frozen=True)
class _ChannelBTriageResult:
    association_pairs_by_view: List[List[Tuple[int, int]]]
    anchor_gt_backed_indices: List[int]
    anchor_support_counts: List[int]
    anchor_support_rates: List[float]
    shielded_anchor_indices: List[int]
    pseudo_positive_candidate_indices: List[int]
    pseudo_positive_anchor_indices: List[int]
    pseudo_positive_cluster_demoted_indices: List[int]
    dead_anchor_indices: List[int]
    lvis_verified_positive_dead_anchor_indices: List[int]
    lvis_verified_negative_dead_anchor_indices: List[int]
    lvis_not_exhaustive_anchor_indices: List[int]
    lvis_unevaluable_anchor_indices: List[int]
    dead_explorer_indices_by_view: List[List[int]]
    recovered_gt_indices: List[int]
    recovered_gt_support_counts: List[int]
    recovered_gt_support_rates: List[float]
    valid_explorer_count: int
    kept_anchor_objects: List[GTObject]
    kept_anchor_new_index_by_old: Dict[int, int]
    suppressed_duplicate_objects_by_boundary: Dict[int, List[GTObject]]


@dataclass(frozen=True)
class _ChannelBSupervisionTargets:
    clean_prefix: _CanonicalPrefixData
    prefix_len_raw_local: int
    prefix_bbox_groups: List[Dict[str, Any]]
    fn_bbox_groups: List[Dict[str, Any]]
    prefix_pos: List[int]
    prefix_bins: List[int]
    prefix_struct_pos: List[int]
    matched_gt_indices: List[int]
    fn_gt_indices_final: List[int]
    fn_objs: List[GTObject]
    fn_object_weights: List[float]
    fn_count_for_meta: int
    append_text: str
    append_ids: List[int]
    tail_desc_pos: List[int]
    tail_desc_weights: List[float]
    y_train_ids: List[int]
    clean_target_text: str
    duplicate_burst_unlikelihood_targets: List[Stage2DuplicateBurstUnlikelihoodTarget]
    duplicate_burst_unlikelihood_boundary_count: int
    duplicate_burst_unlikelihood_skipped_no_divergence: int


def _shift_bbox_groups_with_weights(
    *,
    groups: Sequence[Mapping[str, Any]],
    delta_prompt: int,
    lower: int,
    upper: int,
    encoded_len: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, Mapping):
            continue
        pos = g.get("pos")
        gb = g.get("gt_bins")
        weight_raw = g.get("weight", 1.0)
        if not isinstance(pos, Sequence) or not isinstance(gb, Sequence):
            continue
        if len(pos) != 4 or len(gb) != 4:
            continue
        try:
            pos_i = [int(p) + int(delta_prompt) for p in pos]
            gb_i = [int(x) for x in gb]
            weight_i = float(weight_raw)
        except (TypeError, ValueError):
            continue
        if any(p < int(lower) or p >= int(upper) for p in pos_i):
            raise ValueError(
                "stage2-ab bbox group pos escaped expected span after prompt shift "
                f"(possible truncation/misalignment). pos={pos_i} span=[{int(lower)},{int(upper)}) "
                f"delta_prompt={int(delta_prompt)}"
            )
        if any(p >= int(encoded_len) for p in pos_i):
            raise ValueError(
                "stage2-ab bbox group pos exceeds encoded_len after prompt shift "
                f"(possible truncation/misalignment). pos={pos_i} encoded_len={int(encoded_len)}"
            )
        shifted: Dict[str, Any] = {
            "pos": pos_i,
            "gt_bins": gb_i,
            "weight": float(weight_i),
        }
        prev_bins = g.get("adjacent_prev_gt_bins")
        if isinstance(prev_bins, Sequence) and len(prev_bins) == 4:
            try:
                shifted["adjacent_prev_gt_bins"] = [int(v) for v in prev_bins]
            except (TypeError, ValueError):
                pass
        same_desc = g.get("adjacent_same_desc_with_prev", None)
        if same_desc is not None:
            shifted["adjacent_same_desc_with_prev"] = bool(same_desc)
        out.append(shifted)
    return out


def _adjacent_repulsion_meta_for_objects(
    objects: Sequence[GTObject],
) -> Tuple[List[Optional[List[int]]], List[bool]]:
    prev_bins_by_index: List[Optional[List[int]]] = []
    same_desc_prev_by_index: List[bool] = []
    norm_descs = [normalize_desc(str(obj.desc)) for obj in objects]

    for obj_idx, obj in enumerate(objects):
        if int(obj_idx) <= 0:
            prev_bins_by_index.append(None)
            same_desc_prev_by_index.append(False)
            continue
        prev_obj = objects[int(obj_idx) - 1]
        prev_bins_by_index.append([int(v) for v in prev_obj.points_norm1000])
        same_desc_prev_by_index.append(
            bool(norm_descs[int(obj_idx)] == norm_descs[int(obj_idx) - 1])
        )

    return prev_bins_by_index, same_desc_prev_by_index


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
    *,
    duplicate_iou_threshold: float = 0.90,
    center_radius_scale: float = 0.80,
) -> Dict[str, float]:
    config = validate_duplicate_control_config(
        iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
    )
    duplicate_objects = [
        duplicate_control_object_from_bbox(
            index=int(index),
            desc=str(obj.desc),
            bbox_norm1000=obj.points_norm1000,
            source="anchor",
        )
        for index, obj in enumerate(parsed_bbox_objects_raw)
    ]
    metrics = compute_duplicate_metrics(duplicate_objects, config=config)
    return metrics


def _apply_channel_b_duplicate_control(
    *,
    anchor_objects_raw: Sequence[GTObject],
    explorer_objects_raw_by_view: Sequence[Sequence[GTObject]],
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    unlabeled_consistent_iou_threshold: float,
) -> _ChannelBDuplicateControlResult:
    config = validate_duplicate_control_config(
        iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
    )
    anchor_duplicate_objects = [
        duplicate_control_object_from_bbox(
            index=int(index),
            desc=str(obj.desc),
            bbox_norm1000=obj.points_norm1000,
            source="anchor",
        )
        for index, obj in enumerate(anchor_objects_raw)
    ]
    explorer_duplicate_objects_by_view = [
        [
            duplicate_control_object_from_bbox(
                index=int(index),
                desc=str(obj.desc),
                bbox_norm1000=obj.points_norm1000,
                source="explorer",
            )
            for index, obj in enumerate(explorer_objects_raw)
        ]
        for explorer_objects_raw in explorer_objects_raw_by_view
    ]
    result: DuplicateControlResult = apply_duplicate_policy(
        anchor_objects=anchor_duplicate_objects,
        explorer_objects_by_view=explorer_duplicate_objects_by_view,
        config=config,
        support_iou_threshold=float(unlabeled_consistent_iou_threshold),
    )

    kept_anchor_objects = [
        anchor_objects_raw[int(index)] for index in result.kept_indices
    ]
    kept_indices_sorted = [int(index) for index in result.kept_indices]
    kept_index_set = set(kept_indices_sorted)
    suppressed_duplicate_objects_by_boundary: Dict[int, List[GTObject]] = {}
    for suppressed_index in sorted(int(index) for index in result.suppressed_indices):
        boundary = int(
            sum(
                1
                for kept_index in kept_indices_sorted
                if int(kept_index) < int(suppressed_index)
            )
        )
        if int(suppressed_index) in kept_index_set:
            continue
        suppressed_duplicate_objects_by_boundary.setdefault(boundary, []).append(
            anchor_objects_raw[int(suppressed_index)]
        )

    return _ChannelBDuplicateControlResult(
        kept_anchor_objects=list(kept_anchor_objects),
        suppressed_duplicate_objects_by_boundary=suppressed_duplicate_objects_by_boundary,
        decisions=list(result.decisions),
        support_counts=[int(value) for value in result.support_counts],
        support_rates=[float(value) for value in result.support_rates],
        survivor_anchor_indices=[int(index) for index in result.survivor_indices],
        exempt_anchor_indices=[int(index) for index in result.exempt_indices],
        suppressed_anchor_indices=[int(index) for index in result.suppressed_indices],
        counter_metrics=dict(result.counter_metrics),
    )


def _build_channel_b_triage(
    *,
    accepted_objects_clean: Sequence[GTObject],
    suppressed_duplicate_objects_by_boundary: Optional[
        Mapping[int, Sequence[GTObject]]
    ] = None,
    explorer_objects_raw_by_view: Optional[Sequence[Sequence[GTObject]]] = None,
    anchor_match_by_pred: Mapping[int, int],
    explorer_match_by_pred_by_view: Sequence[Mapping[int, int]],
    unlabeled_consistent_iou_threshold: float,
    duplicate_iou_threshold: float,
    pseudo_positive_enabled: bool,
    anchor_policy_statuses: Sequence[Optional[str]] = (),
) -> _ChannelBTriageResult:
    if suppressed_duplicate_objects_by_boundary is None:
        suppressed_duplicate_objects_by_boundary = {}
    if explorer_objects_raw_by_view is None:
        explorer_objects_raw_by_view = []
    anchor_gt_backed_indices = sorted(int(pred_i) for pred_i in anchor_match_by_pred.keys())
    valid_explorer_count = int(len(explorer_objects_raw_by_view))
    anchor_support_counts = [0 for _ in range(len(accepted_objects_clean))]
    association_pairs_by_view: List[List[Tuple[int, int]]] = []
    dead_explorer_indices_by_view: List[List[int]] = []

    def _anchor_policy_status(anchor_i: int) -> Optional[str]:
        if int(anchor_i) < 0 or int(anchor_i) >= len(anchor_policy_statuses):
            return None
        raw_value = anchor_policy_statuses[int(anchor_i)]
        if raw_value is None:
            return None
        value = str(raw_value).strip().lower()
        return value or None

    def _policy_forced_dead(anchor_i: int) -> bool:
        return _anchor_policy_status(int(anchor_i)) in {
            "verified_positive",
            "verified_negative",
        }

    def _conflicts_gt_backed(anchor_obj: GTObject) -> bool:
        return any(
            _bbox_iou_norm1000_xyxy(
                anchor_obj.points_norm1000,
                accepted_objects_clean[int(gt_anchor_i)].points_norm1000,
            )
            >= float(unlabeled_consistent_iou_threshold)
            for gt_anchor_i in anchor_gt_backed_indices
        )

    for explorer_objects_raw, explorer_match_by_pred in zip(
        explorer_objects_raw_by_view,
        explorer_match_by_pred_by_view,
    ):
        association_pairs = [
            (int(anchor_i), int(explorer_i))
            for anchor_i, explorer_i in associate_one_to_one_max_iou(
                anchors=accepted_objects_clean,
                explorers=explorer_objects_raw,
                min_iou=float(unlabeled_consistent_iou_threshold),
            )
        ]
        association_pairs_by_view.append(list(association_pairs))
        anchor_to_explorer = {
            int(anchor_i): int(explorer_i) for anchor_i, explorer_i in association_pairs
        }
        dead_explorer_indices_by_view.append(
            [
                int(explorer_i)
                for explorer_i in range(len(explorer_objects_raw))
                if int(explorer_i) not in explorer_match_by_pred
            ]
        )
        for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
            if int(anchor_i) in anchor_match_by_pred:
                continue
            if _policy_forced_dead(int(anchor_i)):
                continue
            explorer_i = anchor_to_explorer.get(int(anchor_i))
            if explorer_i is None:
                continue
            if int(explorer_i) in explorer_match_by_pred:
                continue
            if _conflicts_gt_backed(anchor_obj):
                continue
            anchor_support_counts[int(anchor_i)] += 1

    anchor_support_rates = [
        (
            float(int(support_count)) / float(valid_explorer_count)
            if valid_explorer_count > 0
            else 0.0
        )
        for support_count in anchor_support_counts
    ]

    shielded_anchor_indices: set[int] = set()
    pseudo_positive_candidate_indices: List[int] = []
    dead_anchor_indices: set[int] = set()
    lvis_verified_positive_dead_anchor_indices: set[int] = set()
    lvis_verified_negative_dead_anchor_indices: set[int] = set()
    lvis_not_exhaustive_anchor_indices: set[int] = set()
    lvis_unevaluable_anchor_indices: set[int] = set()
    for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
        if int(anchor_i) in anchor_match_by_pred:
            continue
        anchor_status = _anchor_policy_status(int(anchor_i))
        if anchor_status == "verified_positive":
            dead_anchor_indices.add(int(anchor_i))
            lvis_verified_positive_dead_anchor_indices.add(int(anchor_i))
            continue
        if anchor_status == "verified_negative":
            dead_anchor_indices.add(int(anchor_i))
            lvis_verified_negative_dead_anchor_indices.add(int(anchor_i))
            continue
        if anchor_status == "not_exhaustive":
            lvis_not_exhaustive_anchor_indices.add(int(anchor_i))
        elif anchor_status is None:
            lvis_unevaluable_anchor_indices.add(int(anchor_i))
        support_count = int(anchor_support_counts[int(anchor_i)])
        support_rate = float(anchor_support_rates[int(anchor_i)])
        if support_count <= 0:
            dead_anchor_indices.add(int(anchor_i))
            continue
        shielded_anchor_indices.add(int(anchor_i))
        if (
            pseudo_positive_enabled
            and support_count >= 2
            and support_rate >= (2.0 / 3.0)
        ):
            pseudo_positive_candidate_indices.append(int(anchor_i))
        continue
    dead_anchor_indices = {
        int(anchor_i)
        for anchor_i in dead_anchor_indices
        if int(anchor_i) not in shielded_anchor_indices
    }

    pseudo_positive_anchor_indices: List[int] = []
    pseudo_positive_cluster_demoted_indices: List[int] = []
    if pseudo_positive_candidate_indices:
        candidate_set = set(int(idx) for idx in pseudo_positive_candidate_indices)
        adjacency: Dict[int, set[int]] = {
            int(idx): set() for idx in pseudo_positive_candidate_indices
        }
        candidate_list = [int(idx) for idx in pseudo_positive_candidate_indices]
        for left_pos, left_idx in enumerate(candidate_list):
            for right_idx in candidate_list[left_pos + 1 :]:
                if (
                    _bbox_iou_norm1000_xyxy(
                        accepted_objects_clean[int(left_idx)].points_norm1000,
                        accepted_objects_clean[int(right_idx)].points_norm1000,
                    )
                    >= float(duplicate_iou_threshold)
                ):
                    adjacency[int(left_idx)].add(int(right_idx))
                    adjacency[int(right_idx)].add(int(left_idx))

        visited: set[int] = set()
        for candidate_idx in candidate_list:
            if int(candidate_idx) in visited:
                continue
            stack = [int(candidate_idx)]
            component: List[int] = []
            while stack:
                current = int(stack.pop())
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in sorted(adjacency.get(current, set())):
                    if int(neighbor) not in visited:
                        stack.append(int(neighbor))
            winner = min(
                component,
                key=lambda idx: (-float(anchor_support_rates[int(idx)]), int(idx)),
            )
            pseudo_positive_anchor_indices.append(int(winner))
            for idx in component:
                if int(idx) == int(winner):
                    continue
                shielded_anchor_indices.add(int(idx))
                pseudo_positive_cluster_demoted_indices.append(int(idx))
            candidate_set.difference_update(component)
        pseudo_positive_anchor_indices = sorted(
            int(idx) for idx in pseudo_positive_anchor_indices
        )
        pseudo_positive_cluster_demoted_indices = sorted(
            int(idx) for idx in pseudo_positive_cluster_demoted_indices
        )
    shielded_anchor_indices = {
        int(idx)
        for idx in shielded_anchor_indices
        if int(idx) not in set(pseudo_positive_anchor_indices)
    }

    anchor_gt_indices = set(int(gt_i) for gt_i in anchor_match_by_pred.values())
    explorer_gt_support_counts: Dict[int, int] = {}
    for explorer_match_by_pred in explorer_match_by_pred_by_view:
        explorer_gt_indices = set(int(gt_i) for gt_i in explorer_match_by_pred.values())
        for gt_i in explorer_gt_indices:
            if int(gt_i) in anchor_gt_indices:
                continue
            explorer_gt_support_counts[int(gt_i)] = int(
                explorer_gt_support_counts.get(int(gt_i), 0)
            ) + 1
    recovered_gt_indices = sorted(int(gt_i) for gt_i in explorer_gt_support_counts.keys())
    recovered_gt_support_counts = [
        int(explorer_gt_support_counts[int(gt_i)]) for gt_i in recovered_gt_indices
    ]
    recovered_gt_support_rates = [
        (
            float(int(support_count)) / float(valid_explorer_count)
            if valid_explorer_count > 0
            else 0.0
        )
        for support_count in recovered_gt_support_counts
    ]

    kept_anchor_objects: List[GTObject] = []
    kept_anchor_new_index_by_old: Dict[int, int] = {}
    kept_anchor_count = 0
    for anchor_i, anchor_obj in enumerate(accepted_objects_clean):
        if int(anchor_i) in dead_anchor_indices:
            continue
        kept_anchor_new_index_by_old[int(anchor_i)] = int(len(kept_anchor_objects))
        kept_anchor_objects.append(anchor_obj)
        kept_anchor_count += 1

    # Duplicate-control suppression is discovered on the pre-triage accepted-anchor
    # surface, but duplicate UL is realized against the post-triage clean prefix.
    # Remap each pre-triage suppression boundary onto the number of kept anchors
    # that survive before it so duplicate UL still activates when earlier anchors
    # or the whole duplicate-cluster survivor die during triage.
    kept_prefix_count_by_old_boundary: List[int] = [0]
    kept_prefix_count = 0
    for anchor_i in range(len(accepted_objects_clean)):
        if int(anchor_i) not in dead_anchor_indices:
            kept_prefix_count += 1
        kept_prefix_count_by_old_boundary.append(int(kept_prefix_count))

    remapped_suppressed_duplicate_objects_by_boundary: Dict[int, List[GTObject]] = {}
    for boundary, duplicates in suppressed_duplicate_objects_by_boundary.items():
        boundary_i = int(boundary)
        if boundary_i < 0 or boundary_i > len(accepted_objects_clean):
            raise ValueError(
                "pre-triage duplicate suppression boundary escaped accepted-anchor span: "
                f"boundary={boundary_i} accepted={len(accepted_objects_clean)}"
            )
        kept_boundary = int(kept_prefix_count_by_old_boundary[boundary_i])
        remapped_suppressed_duplicate_objects_by_boundary.setdefault(
            kept_boundary, []
        ).extend(
            [obj for obj in duplicates]
        )

    return _ChannelBTriageResult(
        association_pairs_by_view=association_pairs_by_view,
        anchor_gt_backed_indices=[int(idx) for idx in anchor_gt_backed_indices],
        anchor_support_counts=[int(v) for v in anchor_support_counts],
        anchor_support_rates=[float(v) for v in anchor_support_rates],
        shielded_anchor_indices=[int(idx) for idx in sorted(shielded_anchor_indices)],
        pseudo_positive_candidate_indices=[
            int(idx) for idx in sorted(pseudo_positive_candidate_indices)
        ],
        pseudo_positive_anchor_indices=[
            int(idx) for idx in pseudo_positive_anchor_indices
        ],
        pseudo_positive_cluster_demoted_indices=[
            int(idx) for idx in pseudo_positive_cluster_demoted_indices
        ],
        dead_anchor_indices=[int(idx) for idx in sorted(dead_anchor_indices)],
        lvis_verified_positive_dead_anchor_indices=[
            int(idx) for idx in sorted(lvis_verified_positive_dead_anchor_indices)
        ],
        lvis_verified_negative_dead_anchor_indices=[
            int(idx) for idx in sorted(lvis_verified_negative_dead_anchor_indices)
        ],
        lvis_not_exhaustive_anchor_indices=[
            int(idx) for idx in sorted(lvis_not_exhaustive_anchor_indices)
        ],
        lvis_unevaluable_anchor_indices=[
            int(idx) for idx in sorted(lvis_unevaluable_anchor_indices)
        ],
        dead_explorer_indices_by_view=[
            [int(idx) for idx in dead_explorer_indices]
            for dead_explorer_indices in dead_explorer_indices_by_view
        ],
        recovered_gt_indices=[int(idx) for idx in recovered_gt_indices],
        recovered_gt_support_counts=[
            int(v) for v in recovered_gt_support_counts
        ],
        recovered_gt_support_rates=[float(v) for v in recovered_gt_support_rates],
        valid_explorer_count=int(valid_explorer_count),
        kept_anchor_objects=kept_anchor_objects,
        kept_anchor_new_index_by_old=kept_anchor_new_index_by_old,
        suppressed_duplicate_objects_by_boundary=(
            remapped_suppressed_duplicate_objects_by_boundary
        ),
    )


def _build_channel_b_supervision_targets(
    *,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    coord_id_set: set[int],
    gts: Sequence[GTObject],
    match: MatchResult,
    triage: _ChannelBTriageResult,
    recovered_ground_truth_weight_multiplier: float,
    pseudo_positive_enabled: bool,
    pseudo_positive_coord_weight: float,
    object_field_order: str,
    bbox_groups_from_token_ids_fn: Any,
    matched_prefix_structure_positions_fn: Any,
    serialize_append_fragment_fn: Any,
) -> _ChannelBSupervisionTargets:
    prefix_bbox_groups: List[Dict[str, Any]] = []
    fn_bbox_groups: List[Dict[str, Any]] = []
    prefix_pos: List[int] = []
    prefix_bins: List[int] = []
    matched_gt_for_supervision: set[int] = set()
    partial_pseudo_anchor_indices = [
        int(idx)
        for idx in triage.shielded_anchor_indices
        if int(idx) not in set(triage.pseudo_positive_cluster_demoted_indices)
    ]

    clean_prefix = _build_canonical_prefix_data(
        tokenizer=tokenizer,
        objects=triage.kept_anchor_objects,
        object_field_order=object_field_order,
    )
    prefix_len_raw_local = int(len(clean_prefix.prefix_token_ids))
    prefix_adjacent_prev_bins_by_index, prefix_adjacent_same_desc_by_index = (
        _adjacent_repulsion_meta_for_objects(list(triage.kept_anchor_objects))
    )

    prefix_coord_positions_all = [
        int(i)
        for i, tok_id in enumerate(clean_prefix.prefix_token_ids)
        if int(tok_id) in coord_id_set
    ]
    expected_prefix_coord_slots = int(len(triage.kept_anchor_objects) * 4)
    if len(prefix_coord_positions_all) != expected_prefix_coord_slots:
        raise ValueError(
            "clean-prefix canonical serialization produced an unexpected number "
            "of coord tokens for accepted Channel-B bbox objects: "
            f"got={len(prefix_coord_positions_all)} expected={expected_prefix_coord_slots}"
        )

    matched_clean_indices: List[int] = []
    for pred_i, gt_i in sorted(match.matched_pairs, key=lambda item: int(item[0])):
        if pred_i < 0 or pred_i >= len(triage.kept_anchor_objects) + len(triage.dead_anchor_indices):
            continue
        if gt_i < 0 or gt_i >= len(gts):
            continue
        if int(pred_i) not in triage.kept_anchor_new_index_by_old:
            continue

        matched_gt_for_supervision.add(int(gt_i))
        kept_pred_i = int(triage.kept_anchor_new_index_by_old[int(pred_i)])
        matched_clean_indices.append(int(kept_pred_i))

        coord_group = prefix_coord_positions_all[
            int(kept_pred_i) * 4 : int(kept_pred_i + 1) * 4
        ]
        if len(coord_group) != 4:
            raise ValueError(
                "clean-prefix Channel-B expected exactly four coord slots per bbox object"
            )

        gt_bins = list(gts[gt_i].points_norm1000)
        prefix_bbox_groups.append(
            {
                "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                "gt_bins": gt_bins,
                **(
                    {
                        "adjacent_prev_gt_bins": list(
                            prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] or []
                        ),
                        "adjacent_same_desc_with_prev": bool(
                            prefix_adjacent_same_desc_by_index[int(kept_pred_i)]
                        ),
                    }
                    if prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] is not None
                    else {}
                ),
            }
        )
        for local_idx, tbin in zip(coord_group, gt_bins):
            prefix_pos.append(int(local_idx))
            prefix_bins.append(int(tbin))

    for pred_i in triage.pseudo_positive_anchor_indices:
        if int(pred_i) not in triage.kept_anchor_new_index_by_old:
            continue
        kept_pred_i = int(triage.kept_anchor_new_index_by_old[int(pred_i)])
        coord_group = prefix_coord_positions_all[
            int(kept_pred_i) * 4 : int(kept_pred_i + 1) * 4
        ]
        if len(coord_group) != 4:
            raise ValueError(
                "clean-prefix Channel-B expected exactly four coord slots per bbox object"
            )
        gt_bins = list(triage.kept_anchor_objects[int(kept_pred_i)].points_norm1000)
        prefix_bbox_groups.append(
            {
                "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                "gt_bins": gt_bins,
                "weight": float(pseudo_positive_coord_weight),
                **(
                    {
                        "adjacent_prev_gt_bins": list(
                            prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] or []
                        ),
                        "adjacent_same_desc_with_prev": bool(
                            prefix_adjacent_same_desc_by_index[int(kept_pred_i)]
                        ),
                    }
                    if prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] is not None
                    else {}
                ),
            }
        )
        for local_idx, tbin in zip(coord_group, gt_bins):
            prefix_pos.append(int(local_idx))
            prefix_bins.append(int(tbin))

    if pseudo_positive_enabled:
        for pred_i in partial_pseudo_anchor_indices:
            if int(pred_i) not in triage.kept_anchor_new_index_by_old:
                continue
            kept_pred_i = int(triage.kept_anchor_new_index_by_old[int(pred_i)])
            coord_group = prefix_coord_positions_all[
                int(kept_pred_i) * 4 : int(kept_pred_i + 1) * 4
            ]
            if len(coord_group) != 4:
                raise ValueError(
                    "clean-prefix Channel-B expected exactly four coord slots per bbox object"
                )
            partial_weight = float(pseudo_positive_coord_weight) * float(
                triage.anchor_support_rates[int(pred_i)]
            )
            if partial_weight <= 0.0:
                continue
            gt_bins = list(triage.kept_anchor_objects[int(kept_pred_i)].points_norm1000)
            prefix_bbox_groups.append(
                {
                    "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                    "gt_bins": gt_bins,
                    "weight": float(partial_weight),
                    **(
                        {
                            "adjacent_prev_gt_bins": list(
                                prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] or []
                            ),
                            "adjacent_same_desc_with_prev": bool(
                                prefix_adjacent_same_desc_by_index[int(kept_pred_i)]
                            ),
                        }
                        if prefix_adjacent_prev_bins_by_index[int(kept_pred_i)] is not None
                        else {}
                    ),
                }
            )
            for local_idx, tbin in zip(coord_group, gt_bins):
                prefix_pos.append(int(local_idx))
                prefix_bins.append(int(tbin))

    matched_prefix_objects = [
        _ValueSpanObject(value_span=clean_prefix.object_value_spans[int(i)])
        for i in matched_clean_indices
        if 0 <= int(i) < len(clean_prefix.object_value_spans)
    ]
    prefix_struct_pos = matched_prefix_structure_positions_fn(
        tokenizer=tokenizer,
        prefix_token_ids=clean_prefix.prefix_token_ids,
        prefix_text=clean_prefix.prefix_text,
        matched_pred_objects=matched_prefix_objects,
    )

    fn_gt_indices_final = [
        i for i in range(len(gts)) if i not in matched_gt_for_supervision
    ]
    fn_objs = [gts[i] for i in fn_gt_indices_final]
    fn_object_weights = [
        float(recovered_ground_truth_weight_multiplier)
        if int(gt_i) in triage.recovered_gt_indices
        else 1.0
        for gt_i in fn_gt_indices_final
    ]
    fn_count_for_meta = int(len(fn_objs))
    full_adjacent_prev_bins_by_index, full_adjacent_same_desc_by_index = (
        _adjacent_repulsion_meta_for_objects(
            list(triage.kept_anchor_objects) + list(fn_objs)
        )
    )

    append_text = serialize_append_fragment_fn(
        fn_objects=fn_objs,
        prefix_text=clean_prefix.prefix_text,
        object_field_order=object_field_order,
    )
    append_ids = [
        int(t) for t in tokenizer.encode(append_text, add_special_tokens=False)
    ]

    tail_desc_pos, tail_desc_weights = _desc_tail_positions_and_weights(
        tokenizer=tokenizer,
        token_ids=append_ids,
        object_weights=fn_object_weights,
    )

    y_train_ids = list(clean_prefix.prefix_token_ids) + list(append_ids)
    clean_target_text = str(clean_prefix.prefix_text) + str(append_text)
    (
        duplicate_burst_unlikelihood_targets,
        duplicate_burst_unlikelihood_boundary_count,
        duplicate_burst_unlikelihood_skipped_no_divergence,
    ) = _build_duplicate_burst_unlikelihood_targets(
        tokenizer=tokenizer,
        y_train_ids=y_train_ids,
        clean_target_text=clean_target_text,
        accepted_objects_clean=triage.kept_anchor_objects,
        fn_objects=fn_objs,
        suppressed_duplicate_objects_by_boundary=(
            triage.suppressed_duplicate_objects_by_boundary
        ),
        boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
        object_field_order=object_field_order,
    )

    rel_groups = bbox_groups_from_token_ids_fn(
        token_ids=append_ids, coord_id_set=coord_id_set, gt_objs=fn_objs
    )
    prefix_object_count = int(len(triage.kept_anchor_objects))
    for fn_idx, (obj, rel_pos, obj_weight) in enumerate(
        zip(fn_objs, rel_groups, fn_object_weights)
    ):
        full_idx = int(prefix_object_count + fn_idx)
        fn_bbox_groups.append(
            {
                "pos": [
                    int(len(prompt_ids) + int(prefix_len_raw_local) + int(p))
                    for p in rel_pos
                ],
                "gt_bins": list(obj.points_norm1000),
                "weight": float(obj_weight),
                **(
                    {
                        "adjacent_prev_gt_bins": list(
                            full_adjacent_prev_bins_by_index[int(full_idx)] or []
                        ),
                        "adjacent_same_desc_with_prev": bool(
                            full_adjacent_same_desc_by_index[int(full_idx)]
                        ),
                    }
                    if full_adjacent_prev_bins_by_index[int(full_idx)] is not None
                    else {}
                ),
            }
        )

    return _ChannelBSupervisionTargets(
        clean_prefix=clean_prefix,
        prefix_len_raw_local=prefix_len_raw_local,
        prefix_bbox_groups=prefix_bbox_groups,
        fn_bbox_groups=fn_bbox_groups,
        prefix_pos=prefix_pos,
        prefix_bins=prefix_bins,
        prefix_struct_pos=[int(p) for p in prefix_struct_pos],
        matched_gt_indices=sorted(int(idx) for idx in matched_gt_for_supervision),
        fn_gt_indices_final=[int(idx) for idx in fn_gt_indices_final],
        fn_objs=fn_objs,
        fn_object_weights=[float(w) for w in fn_object_weights],
        fn_count_for_meta=int(fn_count_for_meta),
        append_text=str(append_text),
        append_ids=[int(t) for t in append_ids],
        tail_desc_pos=[int(p) for p in tail_desc_pos],
        tail_desc_weights=[float(w) for w in tail_desc_weights],
        y_train_ids=[int(t) for t in y_train_ids],
        clean_target_text=str(clean_target_text),
        duplicate_burst_unlikelihood_targets=list(
            duplicate_burst_unlikelihood_targets
        ),
        duplicate_burst_unlikelihood_boundary_count=int(
            duplicate_burst_unlikelihood_boundary_count
        ),
        duplicate_burst_unlikelihood_skipped_no_divergence=int(
            duplicate_burst_unlikelihood_skipped_no_divergence
        ),
    )


def _build_channel_b_meta_entry(
    *,
    tokenizer: Any,
    enc_ids_list: Sequence[int],
    prompt_len: int,
    prompt_ids: Sequence[int],
    train_len_eff: int,
    prefix_len_eff: int,
    encoded_len: int,
    parse: Any,
    invalid_rollout: int,
    seed_base: int,
    decode_mode: str,
    n_drop_invalid: int,
    valid_pred_objects: int,
    matched_for_supervision_count: int,
    match: MatchResult,
    gt_objects_count: int,
    fn_count_for_meta: int,
    prefix_pos: Sequence[int],
    prefix_bins: Sequence[int],
    prefix_struct_pos: Sequence[int],
    prefix_bbox_groups: Sequence[Mapping[str, Any]],
    fn_bbox_groups: Sequence[Mapping[str, Any]],
    tail_desc_pos: Sequence[int],
    tail_desc_weights: Sequence[float],
    fn_object_weights: Sequence[float],
    anchor_decode_mode: str,
    explorer_decode_mode: str,
    valid_explorer_count: int,
    duplicate_clusters_total: int,
    duplicate_clusters_exempt: int,
    duplicate_clusters_suppressed: int,
    duplicate_objects_suppressed: int,
    duplicate_survivor_anchor_indices: Sequence[int],
    duplicate_exempt_anchor_indices: Sequence[int],
    duplicate_suppressed_anchor_indices: Sequence[int],
    anchor_gt_backed_indices: Sequence[int],
    anchor_support_counts: Sequence[int],
    anchor_support_rates: Sequence[float],
    shielded_anchor_indices: Sequence[int],
    dead_anchor_indices: Sequence[int],
    lvis_verified_positive_dead_anchor_indices: Sequence[int],
    lvis_verified_negative_dead_anchor_indices: Sequence[int],
    lvis_not_exhaustive_anchor_indices: Sequence[int],
    lvis_unevaluable_anchor_indices: Sequence[int],
    pseudo_positive_anchor_indices: Sequence[int],
    dead_explorer_indices_by_view: Sequence[Sequence[int]],
    recovered_gt_indices: Sequence[int],
    recovered_gt_support_counts: Sequence[int],
    recovered_gt_support_rates: Sequence[float],
    duplicate_burst_unlikelihood_targets: Sequence[Stage2DuplicateBurstUnlikelihoodTarget],
    duplicate_burst_unlikelihood_boundary_count: int,
    duplicate_burst_unlikelihood_skipped_no_divergence: int,
    stage2_tail_closure_positions_fn: Any,
    stage2_semantic_stop_branch_metadata_fn: Any,
) -> Tuple[Stage2ChannelBMeta, int]:
    prompt_ids_local = [int(x) for x in enc_ids_list[: int(prompt_len)]]
    delta_prompt = int(prompt_len) - int(len(prompt_ids))

    bbox_groups_prefix = _shift_bbox_groups_with_weights(
        groups=prefix_bbox_groups,
        delta_prompt=int(delta_prompt),
        lower=int(prompt_len),
        upper=int(prompt_len + prefix_len_eff),
        encoded_len=int(encoded_len),
    )
    bbox_groups_fn = _shift_bbox_groups_with_weights(
        groups=fn_bbox_groups,
        delta_prompt=int(delta_prompt),
        lower=int(prompt_len + prefix_len_eff),
        upper=int(prompt_len + train_len_eff),
        encoded_len=int(encoded_len),
    )

    tail_desc_pos_eff: List[int] = []
    tail_desc_weights_eff: List[float] = []
    tail_cap = max(0, int(train_len_eff) - int(prefix_len_eff))
    tail_ignore_pos_eff: List[int] = []
    assistant_span_ids = list(enc_ids_list[int(prompt_len) : int(prompt_len) + int(train_len_eff)])
    semantic_stop_meta: Dict[str, Any] | None = None
    closure_supervision_drop_count = 0
    try:
        tail_closure_pos_eff = stage2_tail_closure_positions_fn(
            tokenizer=tokenizer,
            assistant_span_ids=assistant_span_ids,
            prefix_len=int(prefix_len_eff),
        )
    except ValueError:
        closure_supervision_drop_count = 1
        tail_closure_pos_eff = []
    try:
        semantic_stop_meta = stage2_semantic_stop_branch_metadata_fn(
            tokenizer=tokenizer,
            assistant_span_ids=assistant_span_ids,
            prefix_len=int(prefix_len_eff),
        )
    except ValueError:
        semantic_stop_meta = None

    for rel, weight in zip(tail_desc_pos, tail_desc_weights):
        try:
            rel_i = int(rel)
            weight_f = float(weight)
        except (TypeError, ValueError):
            continue
        if 0 <= rel_i < tail_cap:
            tail_desc_pos_eff.append(rel_i)
            tail_desc_weights_eff.append(weight_f)

    meta_entry: Stage2ChannelBMeta = {
        "stage2_channel": "B",
        "stage2_invalid_rollout": int(invalid_rollout),
        "rollout_seed_base": int(seed_base),
        "prompt_len": int(prompt_len),
        "prompt_ids": prompt_ids_local,
        "rollout_len": int(len(parse.response_token_ids)),
        "prefix_len": int(prefix_len_eff),
        "train_len": int(train_len_eff),
        "encoded_len": int(encoded_len),
        "decode_mode": str(decode_mode),
        "parse_dropped_invalid": int(parse.dropped_invalid),
        "parse_dropped_ambiguous": int(parse.dropped_ambiguous),
        "parse_truncated": bool(parse.truncated),
        "drop_invalid_total": int(n_drop_invalid),
        "valid_pred_objects": int(valid_pred_objects),
        "matched_for_supervision": int(matched_for_supervision_count),
        "matched_maskiou_sum": float(match.matched_maskiou_sum),
        "matched_maskiou_count": int(match.matched_maskiou_count),
        "gt_objects": int(gt_objects_count),
        "fn_count": int(fn_count_for_meta),
        "gating_rejections": int(match.gating_rejections),
        "excluded_from_supervision": int(0),
        "prefix_coord_pos": [int(p) for p in prefix_pos],
        "prefix_coord_target_bins": [int(b) for b in prefix_bins],
        "prefix_struct_pos": [int(p) for p in prefix_struct_pos],
        "tail_closure_pos": [int(p) for p in tail_closure_pos_eff],
        "tail_ignore_pos": tail_ignore_pos_eff,
        "tail_desc_pos": [int(p) for p in tail_desc_pos_eff],
        "tail_desc_weights": [float(w) for w in tail_desc_weights_eff],
        "stop_rel_pos": (
            int(semantic_stop_meta["stop_rel_pos"])
            if isinstance(semantic_stop_meta, Mapping)
            else None
        ),
        "stop_token_id": (
            int(semantic_stop_meta["stop_token_id"])
            if isinstance(semantic_stop_meta, Mapping)
            else None
        ),
        "continue_token_id": (
            int(semantic_stop_meta["continue_token_id"])
            if isinstance(semantic_stop_meta, Mapping)
            else None
        ),
        "fn_object_weights": [float(w) for w in fn_object_weights],
        "bbox_groups_prefix": bbox_groups_prefix,
        "bbox_groups_fn": bbox_groups_fn,
        "anchor_decode_mode": str(anchor_decode_mode),
        "explorer_decode_mode": str(explorer_decode_mode),
        "valid_explorer_count": int(valid_explorer_count),
        "duplicate_clusters_total": int(duplicate_clusters_total),
        "duplicate_clusters_exempt": int(duplicate_clusters_exempt),
        "duplicate_clusters_suppressed": int(duplicate_clusters_suppressed),
        "duplicate_objects_suppressed": int(duplicate_objects_suppressed),
        "duplicate_survivor_anchor_indices": [
            int(idx) for idx in duplicate_survivor_anchor_indices
        ],
        "duplicate_exempt_anchor_indices": [
            int(idx) for idx in duplicate_exempt_anchor_indices
        ],
        "duplicate_suppressed_anchor_indices": [
            int(idx) for idx in duplicate_suppressed_anchor_indices
        ],
        "anchor_gt_backed_indices": [int(idx) for idx in anchor_gt_backed_indices],
        "anchor_support_counts": [int(v) for v in anchor_support_counts],
        "anchor_support_rates": [float(v) for v in anchor_support_rates],
        "shielded_anchor_indices": [int(idx) for idx in shielded_anchor_indices],
        "dead_anchor_indices": [int(idx) for idx in dead_anchor_indices],
        "lvis_verified_positive_dead_anchor_indices": [
            int(idx) for idx in lvis_verified_positive_dead_anchor_indices
        ],
        "lvis_verified_negative_dead_anchor_indices": [
            int(idx) for idx in lvis_verified_negative_dead_anchor_indices
        ],
        "lvis_not_exhaustive_anchor_indices": [
            int(idx) for idx in lvis_not_exhaustive_anchor_indices
        ],
        "lvis_unevaluable_anchor_indices": [
            int(idx) for idx in lvis_unevaluable_anchor_indices
        ],
        "pseudo_positive_anchor_indices": [
            int(idx) for idx in pseudo_positive_anchor_indices
        ],
        "dead_explorer_indices_by_view": [
            [int(idx) for idx in dead_explorer_indices]
            for dead_explorer_indices in dead_explorer_indices_by_view
        ],
        "recovered_gt_indices": [int(idx) for idx in recovered_gt_indices],
        "recovered_gt_support_counts": [
            int(v) for v in recovered_gt_support_counts
        ],
        "recovered_gt_support_rates": [
            float(v) for v in recovered_gt_support_rates
        ],
        "duplicate_burst_unlikelihood_targets": [
            {
                "boundary": int(item["boundary"]),
                "rel_pos": int(item["rel_pos"]),
                "token_id": int(item["token_id"]),
            }
            for item in duplicate_burst_unlikelihood_targets
        ],
        "duplicate_burst_unlikelihood_boundary_count": int(
            duplicate_burst_unlikelihood_boundary_count
        ),
        "duplicate_burst_unlikelihood_skipped_no_divergence": int(
            duplicate_burst_unlikelihood_skipped_no_divergence
        ),
    }
    return meta_entry, int(closure_supervision_drop_count)


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


def _build_duplicate_burst_unlikelihood_targets(
    *,
    tokenizer: Any,
    y_train_ids: Sequence[int],
    clean_target_text: str,
    accepted_objects_clean: Sequence[GTObject],
    fn_objects: Sequence[GTObject],
    suppressed_duplicate_objects_by_boundary: Optional[
        Mapping[int, Sequence[GTObject]]
    ] = None,
    boundary_prefix_texts: Sequence[str],
    object_field_order: str,
) -> Tuple[List[Stage2DuplicateBurstUnlikelihoodTarget], int, int]:
    if suppressed_duplicate_objects_by_boundary is None:
        suppressed_duplicate_objects_by_boundary = {}
    targets_by_boundary_token: dict[
        tuple[int, int], Stage2DuplicateBurstUnlikelihoodTarget
    ] = {}
    skipped_no_divergence = 0

    y_train_ids_list = [int(t) for t in y_train_ids]
    clean_target_text_s = str(clean_target_text)

    for boundary, duplicates in sorted(
        suppressed_duplicate_objects_by_boundary.items()
    ):
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
            candidate: Stage2DuplicateBurstUnlikelihoodTarget = {
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
    duplicate_burst_unlikelihood_boundary_count = len(
        {int(item["boundary"]) for item in targets}
    )
    return targets, int(duplicate_burst_unlikelihood_boundary_count), int(
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
    "_ChannelBSupervisionTargets",
    "_build_channel_b_triage",
    "_build_channel_b_supervision_targets",
    "_build_channel_b_meta_entry",
    "_bbox_iou_norm1000_xyxy",
    "_apply_channel_b_duplicate_control",
    "_compute_duplicate_diagnostics",
    "_build_canonical_prefix_data",
    "_build_duplicate_burst_unlikelihood_targets",
    "_desc_tail_positions_and_weights",
]
