from dataclasses import dataclass, replace
import math
import random
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from src.common.duplicate_control import (
    DuplicateControlDecision,
    DuplicateControlResult,
    compute_duplicate_metrics,
    duplicate_control_object_from_bbox,
    apply_duplicate_policy,
    validate_duplicate_control_config,
)
from src.common.detection_sequence import (
    BOX_START_TOKEN,
    COMPACT_FULL_FORMAT,
    OBJECT_REF_START_TOKEN,
    render_compact_detection_sequence,
)
from src.common.detection_compact_rows import COMPACT_DESC_FORBIDDEN_SUBSTRINGS
from src.common.object_field_order import build_object_payload
from src.common.semantic_desc import normalize_desc
from src.training.stage2.rollout_codec import (
    FALLBACK_GT_FN_APPEND_ONLY,
    Stage2RolloutTemplatePolicy,
)
from src.training.teacher_forcing.constants import (
    MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
    TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
)
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.span_adapters.residual_boundary import (
    ResidualBoundaryAdapter,
    ResidualBoundaryObjectSpan,
    ResidualBoundarySlice,
)
from src.utils.assistant_json import dumps_coordjson

from ..rollout_matching.contracts import GTObject, MatchResult
from ..rollout_matching.matching import associate_one_to_one_greedy_iou
from ..rollout_matching.parsing import decode_pieces, find_desc_value_char_spans
from .types import Stage2ChannelBMeta, Stage2DuplicateControlDivergenceDiagnostic
from .trie_supervision import (
    Stage2TrieCandidate,
    Stage2TrieObjectSpan,
    Stage2TrieTokenTarget,
    build_fp_object_span,
    compile_stage2_trie_targets_for_rollout_group,
    stage2_trie_span_score_record_to_json,
)
from .residual_set import (
    CorrectionAtomDraft,
    CorrectionEvent,
    ObservedResidualRow,
    ResidualObject,
    ResidualRowScanResult,
    ResidualState,
    ValidAction,
    enumerate_valid_actions,
    scan_dirty_prefix_rows,
    transition_state,
)
from .projections import (
    DetectionAssignment,
    DuplicateFilteredRolloutPrediction,
    RolloutPrediction,
    assign_detection_scene_rollout_prediction,
    detection_scene_gt_objects,
    filter_rollout_prediction_duplicates,
)
from .rollout_views import CompactFullObjectTokenSpan, extract_compact_full_object_token_spans

_RESIDUAL_SET_OBJECTIVE_NAME = "residual_set_correction"
_RESIDUAL_STATE_TRIE_OBJECTIVE_NAMES = frozenset(
    {_RESIDUAL_SET_OBJECTIVE_NAME, "stage2_trie_ce"}
)
_DEFAULT_RESIDUAL_SET_ROLLIN_POLICY = "random_valid_branch"
_DEFAULT_RESIDUAL_SET_BASE_SEED = 17
_COORD_ROLE_BY_SLOT = ("x1", "y1", "x2", "y2")
_RESIDUAL_SET_OPTION_DEFAULTS: Dict[str, Any] = {
    "expected_num_rollouts": 4,
    "base_seed": _DEFAULT_RESIDUAL_SET_BASE_SEED,
    "lambda_ul_promoted": 0.5,
    "commit_iou_threshold": 0.75,
    "duplicate_burst_iou_threshold": 0.95,
    "duplicate_burst_prefix_rollback": False,
    "ul_cluster_iou_threshold": 0.9,
    "ul_gray_iou_low": 0.30,
    "min_ul_valid_rollouts": 4,
    "ul_consensus_ratio": 1.0,
}
_COMPACT_COORD_TOKEN_RE = re.compile(r"<\|coord_(0|[1-9]\d{0,2})\|>")


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
    prefix_desc_pos: List[int]
    prefix_desc_weights: List[float]
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
    duplicate_control_first_divergence_diagnostics: List[Stage2DuplicateControlDivergenceDiagnostic]
    duplicate_control_first_divergence_boundary_count: int
    duplicate_control_first_divergence_skipped_no_divergence: int
    rollout_template_family: str
    rollout_parser_id: str
    rollout_append_policy_id: str
    rollout_context: str
    rollout_fallback_reason: str | None
    rollout_fallback_loss_weight: float
    rollout_counts_as_valid_rollout: bool
    stage2_trie_object_spans: List[Stage2TrieObjectSpan]
    stage2_trie_weak_fp_span_level_fallback: bool


@dataclass(frozen=True)
class RolloutCorrectionTargetContextInput:
    """Parsed-rollout and policy facts needed before target realization."""

    sample_id: str
    gt_objects: Sequence[GTObject]
    accepted_objects_clean: Sequence[GTObject]
    suppressed_duplicate_objects_by_boundary: Mapping[int, Sequence[GTObject]]
    explorer_objects_raw_by_view: Sequence[Sequence[GTObject]]
    anchor_match_by_pred: Mapping[int, int]
    explorer_match_by_pred_by_view: Sequence[Mapping[int, int]]
    anchor_policy_statuses: Sequence[Optional[str]]
    unlabeled_consistent_iou_threshold: float
    duplicate_iou_threshold: float
    pseudo_positive_enabled: bool
    expected_peer_count: int
    detection_scene: Any | None = None
    rollout_prediction: RolloutPrediction | None = None
    assignment: DetectionAssignment | None = None
    duplicate_filter: DuplicateFilteredRolloutPrediction | None = None


@dataclass(frozen=True)
class RolloutCorrectionTargetContext:
    """Target context isolated from rollout/DDP/trainer lifecycle."""

    sample_id: str
    triage: _ChannelBTriageResult
    metrics: Dict[str, float]
    detection_scene: Any | None = None
    rollout_prediction: RolloutPrediction | None = None
    assignment: DetectionAssignment | None = None
    duplicate_filter: DuplicateFilteredRolloutPrediction | None = None
    supervision_view_authority: str = "DetectionSupervisionView"


@dataclass(frozen=True)
class _ResidualSetUniverseObject:
    object_id: str
    source: str
    source_index: int
    gt_object: GTObject
    residual_object: ResidualObject


@dataclass(frozen=True)
class _ResidualSetCorrectionBuildResult:
    y_train_ids: List[int]
    clean_target_text: str
    prefix_len_raw_local: int
    events: List[CorrectionEvent]
    event_summaries: List[Dict[str, Any]]
    metrics: Dict[str, float]


def _compact_coord_tokens(values: Sequence[object]) -> List[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("compact-full rollout-correction bbox requires four coord bins")
    if len(values) != 4:
        raise ValueError("compact-full rollout-correction bbox requires four coord bins")

    coords: List[str] = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError("compact-full rollout-correction bbox bins must not be bools")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("compact-full rollout-correction bbox bins must be integers") from exc
        if value < 0 or value > 999:
            raise ValueError("compact-full rollout-correction bbox bins must be in [0, 999]")
        coords.append(f"<|coord_{int(value)}|>")
    return coords


def _compact_payload_from_gt_object(obj: GTObject) -> Dict[str, object]:
    if str(obj.geom_type) != "bbox_2d":
        raise ValueError(
            f"compact-full rollout-correction only supports bbox_2d objects; got {obj.geom_type!r}"
        )
    return {
        "desc": str(obj.desc),
        "bbox_2d": _compact_coord_tokens(obj.points_norm1000),
    }


def _render_compact_objects(objects: Sequence[GTObject]) -> str:
    return render_compact_detection_sequence(
        {"objects": [_compact_payload_from_gt_object(obj) for obj in objects]},
        detection_sequence_format=COMPACT_FULL_FORMAT,
    )


def _compact_object_and_desc_spans(
    text: str,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    spans: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    cursor = 0
    for row in str(text).split("\n"):
        row_start = int(cursor)
        row_end = int(row_start + len(row))
        cursor = int(row_end + 1)
        if not row:
            continue

        box_pos = row.find(BOX_START_TOKEN)
        if not row.startswith(OBJECT_REF_START_TOKEN) or box_pos < len(
            OBJECT_REF_START_TOKEN
        ):
            raise ValueError("compact-full rollout-correction target row lost grammar markers")

        desc_start = int(row_start + len(OBJECT_REF_START_TOKEN))
        desc_end = int(row_start + box_pos)
        spans.append(((row_start, row_end), (desc_start, desc_end)))
    return spans


def _build_compact_prefix_text_data(
    *, objects: Sequence[GTObject]
) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    prefix_text = ""
    boundary_prefix_texts: List[str] = [""]
    object_value_spans: List[Tuple[int, int]] = []

    for obj in objects:
        row_text = _render_compact_objects([obj])
        if prefix_text:
            prefix_text = prefix_text + "\n"
        start = int(len(prefix_text))
        prefix_text = prefix_text + str(row_text)
        object_value_spans.append((start, int(len(prefix_text))))
        boundary_prefix_texts.append(str(prefix_text))

    return prefix_text, boundary_prefix_texts, object_value_spans


def _build_compact_prefix_data(
    *,
    tokenizer: Any,
    objects: Sequence[GTObject],
) -> _CanonicalPrefixData:
    prefix_text, boundary_prefix_texts, object_value_spans = (
        _build_compact_prefix_text_data(objects=objects)
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


def _token_indices_overlapping_char_span(
    *,
    token_spans: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    out: List[int] = []
    for idx, (start, end) in enumerate(token_spans):
        if int(end) <= int(char_start) or int(start) >= int(char_end):
            continue
        out.append(int(idx))
    return out


def _prefix_object_token_span(
    *,
    token_spans: Sequence[Tuple[int, int]],
    object_value_spans: Sequence[Tuple[int, int]],
    object_index: int,
) -> Tuple[int, int] | None:
    if int(object_index) < 0 or int(object_index) >= len(object_value_spans):
        return None

    char_start, char_end = object_value_spans[int(object_index)]
    token_indices = _token_indices_overlapping_char_span(
        token_spans=token_spans,
        char_start=int(char_start),
        char_end=int(char_end),
    )
    if not token_indices:
        return None

    return int(min(token_indices)), int(max(token_indices) + 1)


def _compact_object_marker_token_span(
    *,
    token_spans: Sequence[Tuple[int, int]],
    prefix_text: str,
    object_index: int,
) -> Tuple[int, int] | None:
    object_spans = _compact_object_and_desc_spans(prefix_text)
    if int(object_index) < 0 or int(object_index) >= len(object_spans):
        return None

    object_span, desc_span = object_spans[int(object_index)]
    marker_start = int(object_span[0])
    marker_end = int(desc_span[0])
    token_indices = [
        int(idx)
        for idx, (start, end) in enumerate(token_spans)
        if int(start) >= marker_start and int(end) <= marker_end
    ]
    if not token_indices:
        return None

    return int(min(token_indices)), int(max(token_indices) + 1)


def _build_stage2_trie_fp_object_spans(
    *,
    tokenizer: Any,
    clean_prefix: _CanonicalPrefixData,
    compact_full: bool,
    final_prefix_anchor_indices: Mapping[int, int],
    matched_kept_anchor_indices: Sequence[int],
    original_anchor_index_by_kept: Mapping[int, int],
    anchor_support_counts: Sequence[int],
    fp_policy_mode: str,
    fp_policy_weak_positive_weight: float,
    fp_policy_require_explorer_support: bool,
    fp_policy_min_support_count: int,
    fp_policy_require_token_score: bool,
) -> Tuple[List[Stage2TrieObjectSpan], bool]:
    if fp_policy_mode not in {"zero_loss_context", "weak_positive_context"}:
        raise ValueError(
            "Unknown Stage-2 false-positive policy mode "
            f"{fp_policy_mode!r}; expected zero_loss_context or weak_positive_context"
        )
    if (
        str(fp_policy_mode) == "weak_positive_context"
        and bool(fp_policy_require_token_score)
    ):
        raise ValueError(
            "stage2_rollout_correction.correction.fp_policy.require_token_score=True is reserved "
            "for token-score-gated weak-positive false positives, but that "
            "gating is not implemented in v0"
        )

    token_spans = _token_piece_char_spans(
        tokenizer=tokenizer,
        token_ids=clean_prefix.prefix_token_ids,
    )
    matched_kept = {int(idx) for idx in matched_kept_anchor_indices}

    object_spans: List[Stage2TrieObjectSpan] = []
    for final_object_index, kept_anchor_index in sorted(
        final_prefix_anchor_indices.items(),
        key=lambda item: int(item[0]),
    ):
        kept_idx = int(kept_anchor_index)
        if kept_idx in matched_kept:
            continue

        token_span = _prefix_object_token_span(
            token_spans=token_spans,
            object_value_spans=clean_prefix.object_value_spans,
            object_index=int(final_object_index),
        )
        if token_span is None:
            continue

        original_idx = int(original_anchor_index_by_kept.get(kept_idx, kept_idx))
        support_count = (
            int(anchor_support_counts[original_idx])
            if 0 <= original_idx < len(anchor_support_counts)
            else 0
        )
        support_satisfied = (
            (not bool(fp_policy_require_explorer_support))
            or support_count >= int(fp_policy_min_support_count)
        )
        weak_positive_satisfied = (
            str(fp_policy_mode) == "weak_positive_context"
            and support_satisfied
        )
        span_policy_mode = (
            "weak_positive_context"
            if weak_positive_satisfied
            else "zero_loss_context"
        )

        if span_policy_mode != "weak_positive_context" or not bool(compact_full):
            if span_policy_mode == "weak_positive_context":
                raise ValueError(
                    "stage2_rollout_correction.correction.fp_policy.mode=weak_positive_context "
                    "is compact_full-only in v0; non-compact FP marker "
                    "supervision is not implemented"
                )
            object_spans.append(
                build_fp_object_span(
                    token_start=int(token_span[0]),
                    token_end=int(token_span[1]),
                    policy_mode=span_policy_mode,
                    support_count=int(support_count),
                    weak_positive_weight=float(fp_policy_weak_positive_weight),
                    min_support_count=int(fp_policy_min_support_count),
                    require_explorer_support=bool(fp_policy_require_explorer_support),
                )
            )
            continue

        object_spans.append(
            build_fp_object_span(
                token_start=int(token_span[0]),
                token_end=int(token_span[1]),
                policy_mode="zero_loss_context",
                support_count=int(support_count),
                weak_positive_weight=float(fp_policy_weak_positive_weight),
                min_support_count=int(fp_policy_min_support_count),
                require_explorer_support=bool(fp_policy_require_explorer_support),
            )
        )
        weak_token_span = _compact_object_marker_token_span(
            token_spans=token_spans,
            prefix_text=clean_prefix.prefix_text,
            object_index=int(final_object_index),
        )
        if weak_token_span is None:
            raise ValueError(
                "stage2_rollout_correction.correction.fp_policy.mode=weak_positive_context "
                "could not isolate a compact_full object marker token span; "
                "refusing to weak-supervise FP description or coordinate tokens"
            )

        object_spans.append(
            build_fp_object_span(
                token_start=int(weak_token_span[0]),
                token_end=int(weak_token_span[1]),
                policy_mode="weak_positive_context",
                support_count=int(support_count),
                weak_positive_weight=float(fp_policy_weak_positive_weight),
                min_support_count=int(fp_policy_min_support_count),
                require_explorer_support=bool(fp_policy_require_explorer_support),
            )
        )

    return object_spans, any(
        span.role == "weak_positive_fp" for span in object_spans
    )


def _compact_prefix_structure_positions(
    *,
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    prefix_text: str,
    matched_object_indices: Sequence[int],
) -> List[int]:
    if not prefix_token_ids or not matched_object_indices:
        return []

    token_spans = _token_piece_char_spans(
        tokenizer=tokenizer,
        token_ids=prefix_token_ids,
    )
    object_spans = _compact_object_and_desc_spans(prefix_text)

    supervised: set[int] = set()
    for raw_idx in matched_object_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(object_spans):
            continue
        object_span, desc_span = object_spans[idx]
        entry_tokens = _token_indices_overlapping_char_span(
            token_spans=token_spans,
            char_start=int(object_span[0]),
            char_end=int(object_span[1]),
        )
        desc_tokens = set(
            _token_indices_overlapping_char_span(
                token_spans=token_spans,
                char_start=int(desc_span[0]),
                char_end=int(desc_span[1]),
            )
        )
        supervised.update(int(token) for token in entry_tokens if token not in desc_tokens)

    return sorted(int(p) for p in supervised)


def _compact_desc_tail_positions_and_weights(
    *,
    tokenizer: Any,
    token_ids: Sequence[int],
    tail_text: str,
    object_weights: Sequence[float],
) -> Tuple[List[int], List[float]]:
    if not token_ids or not tail_text:
        return [], []

    token_spans = _token_piece_char_spans(tokenizer=tokenizer, token_ids=token_ids)
    object_spans = _compact_object_and_desc_spans(tail_text)

    positions: List[int] = []
    weights: List[float] = []
    for obj_idx, (_object_span, desc_span) in enumerate(object_spans):
        weight = (
            float(object_weights[int(obj_idx)])
            if int(obj_idx) < len(object_weights)
            else 1.0
        )
        desc_tokens = _token_indices_overlapping_char_span(
            token_spans=token_spans,
            char_start=int(desc_span[0]),
            char_end=int(desc_span[1]),
        )
        positions.extend(int(pos) for pos in desc_tokens)
        weights.extend(float(weight) for _ in desc_tokens)

    return positions, weights


def _compact_desc_prefix_positions_and_weights(
    *,
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    prefix_text: str,
    object_weights_by_index: Mapping[int, float],
) -> Tuple[List[int], List[float]]:
    if not prefix_token_ids or not prefix_text or not object_weights_by_index:
        return [], []

    token_spans = _token_piece_char_spans(
        tokenizer=tokenizer,
        token_ids=prefix_token_ids,
    )
    object_spans = _compact_object_and_desc_spans(prefix_text)

    positions: List[int] = []
    weights: List[float] = []
    for obj_idx, (_object_span, desc_span) in enumerate(object_spans):
        if int(obj_idx) not in object_weights_by_index:
            continue
        weight = float(object_weights_by_index[int(obj_idx)])
        desc_tokens = _token_indices_overlapping_char_span(
            token_spans=token_spans,
            char_start=int(desc_span[0]),
            char_end=int(desc_span[1]),
        )
        positions.extend(int(pos) for pos in desc_tokens)
        weights.extend(float(weight) for _ in desc_tokens)

    return positions, weights


def _normalize_rollout_correction_insertion_order(insertion_order: str) -> str:
    value = str(insertion_order or "tail_append").strip().lower()
    if value not in {"tail_append", "sorted", "fn_slot_shuffle"}:
        raise ValueError(
            "stage2_rollout_correction.correction.insertion_order must be one of "
            "{'tail_append', 'sorted', 'fn_slot_shuffle'}"
        )
    return value


def _gt_object_topleft_anchor(obj: GTObject) -> Tuple[int, int]:
    points = [int(v) for v in obj.points_norm1000]
    if str(obj.geom_type) == "bbox_2d" and len(points) >= 2:
        return (int(points[1]), int(points[0]))

    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return (10**9, 10**9)
    return (int(min(ys)), int(min(xs)))


def _fn_slot_shuffle_entries(
    *,
    anchor_entries: Sequence[Dict[str, Any]],
    fn_entries: Sequence[Dict[str, Any]],
    shuffle_seed: int | None,
) -> List[Dict[str, Any]]:
    """Deterministically inject FN entries into accepted-anchor slots.

    The accepted rollout objects keep their relative order. False-negative
    objects are shuffled, then inserted into random slots so the target can
    test order robustness without making run replay nondeterministic.
    """

    rng = random.Random(0 if shuffle_seed is None else int(shuffle_seed))

    arranged_entries = [dict(entry) for entry in anchor_entries]
    shuffled_fn_entries = [dict(entry) for entry in fn_entries]
    rng.shuffle(shuffled_fn_entries)

    for fn_entry in shuffled_fn_entries:
        insert_at = rng.randint(0, len(arranged_entries))
        arranged_entries.insert(int(insert_at), dict(fn_entry))

    return arranged_entries


def _sorted_duplicate_bursts_by_boundary(
    *,
    sorted_objects: Sequence[GTObject],
    suppressed_duplicate_objects_by_boundary: Mapping[int, Sequence[GTObject]],
) -> Dict[int, List[GTObject]]:
    remapped: Dict[int, List[GTObject]] = {}
    sorted_base = list(sorted_objects)
    for duplicates in suppressed_duplicate_objects_by_boundary.values():
        for dup in duplicates:
            annotated = [(obj, False) for obj in sorted_base] + [(dup, True)]
            sorted_annotated = sorted(
                annotated,
                key=lambda item: _gt_object_topleft_anchor(item[0]),
            )
            boundary = next(
                (
                    int(idx)
                    for idx, (_obj, is_dup) in enumerate(sorted_annotated)
                    if bool(is_dup)
                ),
                None,
            )
            if boundary is None:
                raise ValueError(
                    "failed to place duplicate burst object into sorted rollout-correction sequence"
                )
            remapped.setdefault(int(boundary), []).append(dup)
    return remapped


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
        out.append(shifted)
    return out


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


def _apply_rollout_correction_duplicate_control(
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


def _sequential_dedup_bbox_objects(
    *,
    parsed_bbox_objects_raw: Sequence[GTObject],
    duplicate_iou_threshold: float,
) -> Tuple[List[GTObject], Dict[int, List[GTObject]]]:
    result = _apply_rollout_correction_duplicate_control(
        anchor_objects_raw=parsed_bbox_objects_raw,
        explorer_objects_raw_by_view=[],
        duplicate_iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=0.0,
        unlabeled_consistent_iou_threshold=0.0,
    )
    return (
        list(result.kept_anchor_objects),
        dict(result.suppressed_duplicate_objects_by_boundary),
    )


def _build_rollout_correction_triage(
    *,
    accepted_objects_clean: Sequence[GTObject],
    suppressed_duplicate_objects_by_boundary: Optional[
        Mapping[int, Sequence[GTObject]]
    ] = None,
    explorer_objects_raw_by_view: Optional[Sequence[Sequence[GTObject]]] = None,
    duplicate_bursts_by_boundary: Optional[Mapping[int, Sequence[GTObject]]] = None,
    explorer_accepted_objects_clean_by_view: Optional[
        Sequence[Sequence[GTObject]]
    ] = None,
    anchor_match_by_pred: Mapping[int, int],
    explorer_match_by_pred_by_view: Sequence[Mapping[int, int]],
    unlabeled_consistent_iou_threshold: float,
    duplicate_iou_threshold: float,
    pseudo_positive_enabled: bool,
    anchor_policy_statuses: Sequence[Optional[str]] = (),
    expected_peer_count: Optional[int] = None,
) -> _ChannelBTriageResult:
    if duplicate_bursts_by_boundary is not None:
        suppressed_duplicate_objects_by_boundary = duplicate_bursts_by_boundary
    if explorer_accepted_objects_clean_by_view is not None:
        explorer_objects_raw_by_view = explorer_accepted_objects_clean_by_view
    if suppressed_duplicate_objects_by_boundary is None:
        suppressed_duplicate_objects_by_boundary = {}
    if explorer_objects_raw_by_view is None:
        explorer_objects_raw_by_view = []
    anchor_gt_backed_indices = sorted(int(pred_i) for pred_i in anchor_match_by_pred.keys())
    valid_explorer_count = int(len(explorer_objects_raw_by_view))
    expected_peer_count_resolved = (
        int(valid_explorer_count)
        if expected_peer_count is None
        else max(0, int(expected_peer_count))
    )
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

    def _normalized_desc(obj: GTObject) -> str:
        return str(getattr(obj, "desc", "") or "").strip()

    def _same_desc(anchor_obj: GTObject, explorer_obj: GTObject) -> bool:
        return _normalized_desc(anchor_obj) == _normalized_desc(explorer_obj)

    for explorer_objects_raw, explorer_match_by_pred in zip(
        explorer_objects_raw_by_view,
        explorer_match_by_pred_by_view,
    ):
        association_pairs = [
            (int(anchor_i), int(explorer_i))
            for anchor_i, explorer_i in associate_one_to_one_greedy_iou(
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
            if not _same_desc(anchor_obj, explorer_objects_raw[int(explorer_i)]):
                continue
            anchor_support_counts[int(anchor_i)] += 1

    anchor_support_rates = [
        (
            float(int(support_count)) / float(expected_peer_count_resolved)
            if expected_peer_count_resolved > 0
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
            and (int(expected_peer_count_resolved) + 1) >= 4
            and support_count == int(expected_peer_count_resolved)
            and support_rate == 1.0
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
            float(int(support_count)) / float(expected_peer_count_resolved)
            if expected_peer_count_resolved > 0
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
    # surface, but diagnostic first-divergence metadata is recorded against the
    # post-triage clean prefix. Remap each pre-triage suppression boundary onto
    # the number of kept anchors that survive before it so diagnostics remain
    # aligned when earlier anchors or the whole duplicate-cluster survivor die
    # during triage.
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


def construct_rollout_correction_target_context(
    request: RolloutCorrectionTargetContextInput,
) -> RolloutCorrectionTargetContext:
    """Build target context from parsed rollout/GT facts and correction policy.

    This boundary intentionally receives already-parsed rollout objects and
    matching maps. Rollout backend calls, DDP/packing coordination, model
    forward/loss execution, and metric projection stay with the trainer.
    """

    triage = _build_rollout_correction_triage(
        accepted_objects_clean=request.accepted_objects_clean,
        suppressed_duplicate_objects_by_boundary=(
            request.suppressed_duplicate_objects_by_boundary
        ),
        explorer_objects_raw_by_view=request.explorer_objects_raw_by_view,
        anchor_match_by_pred=request.anchor_match_by_pred,
        explorer_match_by_pred_by_view=request.explorer_match_by_pred_by_view,
        anchor_policy_statuses=request.anchor_policy_statuses,
        unlabeled_consistent_iou_threshold=float(
            request.unlabeled_consistent_iou_threshold
        ),
        duplicate_iou_threshold=float(request.duplicate_iou_threshold),
        pseudo_positive_enabled=bool(request.pseudo_positive_enabled),
        expected_peer_count=int(request.expected_peer_count),
    )
    metrics = {
        "gt_objects": float(len(request.gt_objects)),
        "accepted_objects": float(len(request.accepted_objects_clean)),
        "anchor_gt_backed": float(len(triage.anchor_gt_backed_indices)),
        "dead_anchor": float(len(triage.dead_anchor_indices)),
        "recovered_gt": float(len(triage.recovered_gt_indices)),
        "pseudo_positive_candidate": float(
            len(triage.pseudo_positive_candidate_indices)
        ),
        "pseudo_positive_selected": float(len(triage.pseudo_positive_anchor_indices)),
        "valid_explorer_count": float(triage.valid_explorer_count),
    }
    if request.rollout_prediction is not None:
        metrics["rollout_prediction_valid_objects"] = float(
            len(request.rollout_prediction.valid_objects)
        )
        metrics["rollout_prediction_metric_bearing"] = float(
            1.0 if request.rollout_prediction.metric_bearing else 0.0
        )
    if request.assignment is not None:
        metrics["detection_assignment_matched_pairs"] = float(
            len(request.assignment.matched_pairs)
        )
    return RolloutCorrectionTargetContext(
        sample_id=str(request.sample_id),
        triage=triage,
        metrics=metrics,
        detection_scene=request.detection_scene,
        rollout_prediction=request.rollout_prediction,
        assignment=request.assignment,
        duplicate_filter=request.duplicate_filter,
    )


def construct_detection_scene_rollout_correction_target_context(
    *,
    scene: Any,
    rollout_prediction: RolloutPrediction,
    explorer_predictions: Sequence[RolloutPrediction] = (),
    unlabeled_consistent_iou_threshold: float,
    duplicate_iou_threshold: float,
    center_radius_scale: float,
    pseudo_positive_enabled: bool,
    expected_peer_count: int,
    assignment_iou_threshold: float = 0.5,
    anchor_policy_statuses: Sequence[Optional[str]] = (),
) -> RolloutCorrectionTargetContext:
    """Build Stage-2 target context from DetectionScene plus RolloutPrediction."""

    duplicate_filter = filter_rollout_prediction_duplicates(
        prediction=rollout_prediction,
        explorer_predictions=explorer_predictions,
        duplicate_iou_threshold=float(duplicate_iou_threshold),
        center_radius_scale=float(center_radius_scale),
        unlabeled_consistent_iou_threshold=float(unlabeled_consistent_iou_threshold),
    )
    assignment = assign_detection_scene_rollout_prediction(
        scene=scene,
        prediction=duplicate_filter.prediction,
        min_iou=float(assignment_iou_threshold),
    )
    explorer_match_by_pred_by_view: List[Dict[int, int]] = []
    for explorer_prediction in explorer_predictions:
        explorer_assignment = assign_detection_scene_rollout_prediction(
            scene=scene,
            prediction=explorer_prediction,
            min_iou=float(assignment_iou_threshold),
        )
        explorer_match_by_pred_by_view.append(explorer_assignment.anchor_match_by_pred)

    return construct_rollout_correction_target_context(
        RolloutCorrectionTargetContextInput(
            sample_id=str(getattr(scene, "image_id", "")),
            gt_objects=detection_scene_gt_objects(scene),
            accepted_objects_clean=duplicate_filter.prediction.valid_objects,
            suppressed_duplicate_objects_by_boundary=(
                duplicate_filter.suppressed_duplicate_objects_by_boundary
            ),
            explorer_objects_raw_by_view=[
                tuple(explorer_prediction.valid_objects)
                for explorer_prediction in explorer_predictions
            ],
            anchor_match_by_pred=assignment.anchor_match_by_pred,
            explorer_match_by_pred_by_view=explorer_match_by_pred_by_view,
            anchor_policy_statuses=anchor_policy_statuses,
            unlabeled_consistent_iou_threshold=float(
                unlabeled_consistent_iou_threshold
            ),
            duplicate_iou_threshold=float(duplicate_iou_threshold),
            pseudo_positive_enabled=bool(pseudo_positive_enabled),
            expected_peer_count=int(expected_peer_count),
            detection_scene=scene,
            rollout_prediction=rollout_prediction,
            assignment=assignment,
            duplicate_filter=duplicate_filter,
        )
    )


def _build_rollout_correction_supervision_targets(
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
    duplicate_iou_threshold: float | None = None,
    insertion_order: str = "tail_append",
    fp_policy_mode: str = "zero_loss_context",
    fp_policy_weak_positive_weight: float = 0.05,
    fp_policy_require_explorer_support: bool = True,
    fp_policy_min_support_count: int = 1,
    fp_policy_require_token_score: bool = False,
    object_field_order: str,
    bbox_groups_from_token_ids_fn: Any,
    matched_prefix_structure_positions_fn: Any,
    serialize_append_fragment_fn: Any,
    rollout_template_policy: Stage2RolloutTemplatePolicy | None = None,
    parse: Any | None = None,
    shuffle_seed: int | None = None,
) -> _ChannelBSupervisionTargets:
    insertion_order_resolved = _normalize_rollout_correction_insertion_order(insertion_order)
    rollout_template_family = (
        str(rollout_template_policy.template_family)
        if rollout_template_policy is not None
        else "coordjson"
    )
    rollout_parser_id = (
        str(rollout_template_policy.parser_id)
        if rollout_template_policy is not None
        else "coordjson_legacy"
    )
    rollout_append_policy_id = (
        str(rollout_template_policy.append_policy_id)
        if rollout_template_policy is not None
        else "coordjson_legacy_fn_append"
    )
    is_compact_full = rollout_template_family == "compact_full"
    compact_fallback_applies = bool(
        is_compact_full
        and parse is not None
        and (
            bool(getattr(parse, "invalid_rollout", False))
            or bool(getattr(parse, "empty_valid_object_set", False))
        )
    )
    rollout_fallback_loss_weight = (
        float(rollout_template_policy.fallback_loss_weight)
        if compact_fallback_applies and rollout_template_policy is not None
        else 0.0
    )
    rollout_fallback_reason = (
        str(getattr(parse, "fallback_reason", "") or "")
        if compact_fallback_applies
        else None
    )
    rollout_context = (
        FALLBACK_GT_FN_APPEND_ONLY
        if compact_fallback_applies
        else (
            "rollout_valid_with_fn_append"
            if is_compact_full
            else "legacy_coordjson_append"
        )
    )
    rollout_counts_as_valid_rollout = not bool(
        getattr(parse, "invalid_rollout", False)
        or getattr(parse, "empty_valid_object_set", False)
    )

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

    clean_prefix = (
        _build_compact_prefix_data(
            tokenizer=tokenizer,
            objects=triage.kept_anchor_objects,
        )
        if is_compact_full
        else _build_canonical_prefix_data(
            tokenizer=tokenizer,
            objects=triage.kept_anchor_objects,
            object_field_order=object_field_order,
        )
    )
    prefix_len_raw_local = int(len(clean_prefix.prefix_token_ids))

    prefix_coord_positions_all = [
        int(i)
        for i, tok_id in enumerate(clean_prefix.prefix_token_ids)
        if int(tok_id) in coord_id_set
    ]
    original_anchor_index_by_kept: Dict[int, int] = {
        int(new_idx): int(old_idx)
        for old_idx, new_idx in triage.kept_anchor_new_index_by_old.items()
    }
    final_prefix_anchor_indices: Dict[int, int] = {
        int(kept_idx): int(kept_idx)
        for kept_idx in range(len(triage.kept_anchor_objects))
    }
    expected_prefix_coord_slots = int(len(triage.kept_anchor_objects) * 4)
    if len(prefix_coord_positions_all) != expected_prefix_coord_slots:
        raise ValueError(
            "clean-prefix canonical serialization produced an unexpected number "
            "of coord tokens for accepted rollout-correction bbox objects: "
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
                "clean-prefix rollout-correction expected exactly four coord slots per bbox object"
            )

        gt_bins = list(gts[gt_i].points_norm1000)
        prefix_bbox_groups.append(
            {
                "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                "gt_bins": gt_bins,
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
                "clean-prefix rollout-correction expected exactly four coord slots per bbox object"
            )
        gt_bins = list(triage.kept_anchor_objects[int(kept_pred_i)].points_norm1000)
        prefix_bbox_groups.append(
            {
                "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                "gt_bins": gt_bins,
                "weight": float(pseudo_positive_coord_weight),
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
                    "clean-prefix rollout-correction expected exactly four coord slots per bbox object"
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
                }
            )
            for local_idx, tbin in zip(coord_group, gt_bins):
                prefix_pos.append(int(local_idx))
                prefix_bins.append(int(tbin))

    if is_compact_full:
        prefix_struct_pos = _compact_prefix_structure_positions(
            tokenizer=tokenizer,
            prefix_token_ids=clean_prefix.prefix_token_ids,
            prefix_text=clean_prefix.prefix_text,
            matched_object_indices=matched_clean_indices,
        )
    else:
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
    if compact_fallback_applies:
        fn_object_weights = [
            float(weight) * float(rollout_fallback_loss_weight)
            for weight in fn_object_weights
        ]
    fn_count_for_meta = int(len(fn_objs))
    prefix_desc_pos: List[int] = []
    prefix_desc_weights: List[float] = []
    if insertion_order_resolved in {"sorted", "fn_slot_shuffle"}:
        prefix_bbox_groups = []
        fn_bbox_groups = []
        prefix_pos = []
        prefix_bins = []
        anchor_entries: List[Dict[str, Any]] = []
        for kept_idx, obj in enumerate(triage.kept_anchor_objects):
            anchor_entries.append(
                {
                    "kind": "anchor",
                    "index": int(kept_idx),
                    "obj": obj,
                }
            )
        fn_entries: List[Dict[str, Any]] = []
        for fn_idx, (obj, obj_weight) in enumerate(zip(fn_objs, fn_object_weights)):
            fn_entries.append(
                {
                    "kind": "fn",
                    "index": int(fn_idx),
                    "obj": obj,
                    "weight": float(obj_weight),
                }
            )
        if insertion_order_resolved == "sorted":
            sorted_entries = sorted(
                list(anchor_entries) + list(fn_entries),
                key=lambda entry: _gt_object_topleft_anchor(entry["obj"]),
            )
        else:
            sorted_entries = _fn_slot_shuffle_entries(
                anchor_entries=anchor_entries,
                fn_entries=fn_entries,
                shuffle_seed=shuffle_seed,
            )
        sorted_objects = [entry["obj"] for entry in sorted_entries]
        clean_prefix = (
            _build_compact_prefix_data(
                tokenizer=tokenizer,
                objects=sorted_objects,
            )
            if is_compact_full
            else _build_canonical_prefix_data(
                tokenizer=tokenizer,
                objects=sorted_objects,
                object_field_order=object_field_order,
            )
        )
        prefix_len_raw_local = int(len(clean_prefix.prefix_token_ids))
        prefix_coord_positions_all = [
            int(i)
            for i, tok_id in enumerate(clean_prefix.prefix_token_ids)
            if int(tok_id) in coord_id_set
        ]
        expected_prefix_coord_slots = int(len(sorted_objects) * 4)
        if len(prefix_coord_positions_all) != expected_prefix_coord_slots:
            raise ValueError(
                "sorted rollout-correction canonical serialization produced an unexpected number "
                "of coord tokens: "
                f"got={len(prefix_coord_positions_all)} expected={expected_prefix_coord_slots}"
            )
        kept_idx_to_sorted_idx: Dict[int, int] = {}
        fn_idx_to_sorted_idx: Dict[int, int] = {}
        for sorted_idx, entry in enumerate(sorted_entries):
            kind = str(entry["kind"])
            if kind == "anchor":
                kept_idx_to_sorted_idx[int(entry["index"])] = int(sorted_idx)
                continue
            if kind == "fn":
                fn_idx_to_sorted_idx[int(entry["index"])] = int(sorted_idx)
        final_prefix_anchor_indices = {
            int(sorted_idx): int(kept_idx)
            for kept_idx, sorted_idx in kept_idx_to_sorted_idx.items()
        }

        remapped_matched_sorted_indices: List[int] = []
        for pred_i, gt_i in sorted(match.matched_pairs, key=lambda item: int(item[0])):
            if pred_i < 0 or pred_i >= len(triage.kept_anchor_objects) + len(triage.dead_anchor_indices):
                continue
            if gt_i < 0 or gt_i >= len(gts):
                continue
            if int(pred_i) not in triage.kept_anchor_new_index_by_old:
                continue

            matched_gt_for_supervision.add(int(gt_i))
            kept_pred_i = int(triage.kept_anchor_new_index_by_old[int(pred_i)])
            sorted_idx = kept_idx_to_sorted_idx.get(int(kept_pred_i))
            if sorted_idx is None:
                continue
            remapped_matched_sorted_indices.append(int(sorted_idx))
            coord_group = prefix_coord_positions_all[
                int(sorted_idx) * 4 : int(sorted_idx + 1) * 4
            ]
            if len(coord_group) != 4:
                raise ValueError(
                    "sorted rollout-correction expected exactly four coord slots per bbox object"
                )
            gt_bins = list(gts[gt_i].points_norm1000)
            prefix_bbox_groups.append(
                {
                    "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                    "gt_bins": gt_bins,
                }
            )
            for local_idx, tbin in zip(coord_group, gt_bins):
                prefix_pos.append(int(local_idx))
                prefix_bins.append(int(tbin))

        for pred_i in triage.pseudo_positive_anchor_indices:
            if int(pred_i) not in triage.kept_anchor_new_index_by_old:
                continue
            kept_pred_i = int(triage.kept_anchor_new_index_by_old[int(pred_i)])
            sorted_idx = kept_idx_to_sorted_idx.get(int(kept_pred_i))
            if sorted_idx is None:
                continue
            coord_group = prefix_coord_positions_all[
                int(sorted_idx) * 4 : int(sorted_idx + 1) * 4
            ]
            if len(coord_group) != 4:
                raise ValueError(
                    "sorted rollout-correction expected exactly four coord slots per bbox object"
                )
            gt_bins = list(triage.kept_anchor_objects[int(kept_pred_i)].points_norm1000)
            prefix_bbox_groups.append(
                {
                    "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                    "gt_bins": gt_bins,
                    "weight": float(pseudo_positive_coord_weight),
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
                sorted_idx = kept_idx_to_sorted_idx.get(int(kept_pred_i))
                if sorted_idx is None:
                    continue
                coord_group = prefix_coord_positions_all[
                    int(sorted_idx) * 4 : int(sorted_idx + 1) * 4
                ]
                if len(coord_group) != 4:
                    raise ValueError(
                        "sorted rollout-correction expected exactly four coord slots per bbox object"
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
                    },
                )
                for local_idx, tbin in zip(coord_group, gt_bins):
                    prefix_pos.append(int(local_idx))
                    prefix_bins.append(int(tbin))

        for fn_idx, (obj, obj_weight) in enumerate(zip(fn_objs, fn_object_weights)):
            sorted_idx = fn_idx_to_sorted_idx.get(int(fn_idx))
            if sorted_idx is None:
                continue
            coord_group = prefix_coord_positions_all[
                int(sorted_idx) * 4 : int(sorted_idx + 1) * 4
            ]
            if len(coord_group) != 4:
                raise ValueError(
                    "sorted rollout-correction expected exactly four coord slots per bbox object"
                )
            prefix_bbox_groups.append(
                {
                    "pos": [int(len(prompt_ids) + int(p)) for p in coord_group],
                    "gt_bins": list(obj.points_norm1000),
                    "weight": float(obj_weight),
                }
            )

        if is_compact_full:
            prefix_struct_pos = _compact_prefix_structure_positions(
                tokenizer=tokenizer,
                prefix_token_ids=clean_prefix.prefix_token_ids,
                prefix_text=clean_prefix.prefix_text,
                matched_object_indices=remapped_matched_sorted_indices,
            )
            prefix_desc_weight_by_sorted_idx = {
                int(sorted_idx): float(fn_object_weights[int(fn_idx)])
                for fn_idx, sorted_idx in fn_idx_to_sorted_idx.items()
                if 0 <= int(fn_idx) < len(fn_object_weights)
            }
            prefix_desc_pos, prefix_desc_weights = (
                _compact_desc_prefix_positions_and_weights(
                    tokenizer=tokenizer,
                    prefix_token_ids=clean_prefix.prefix_token_ids,
                    prefix_text=clean_prefix.prefix_text,
                    object_weights_by_index=prefix_desc_weight_by_sorted_idx,
                )
            )
            append_text = ""
            append_ids = []
            tail_desc_pos = []
            tail_desc_weights = []
            y_train_ids = list(clean_prefix.prefix_token_ids)
            clean_target_text = str(clean_prefix.prefix_text)
            duplicate_control_first_divergence_diagnostics = []
            duplicate_control_first_divergence_boundary_count = 0
            duplicate_control_first_divergence_skipped_no_divergence = 0
        else:
            matched_prefix_objects = [
                _ValueSpanObject(value_span=clean_prefix.object_value_spans[int(i)])
                for i in remapped_matched_sorted_indices
                if 0 <= int(i) < len(clean_prefix.object_value_spans)
            ]
            prefix_struct_pos = matched_prefix_structure_positions_fn(
                tokenizer=tokenizer,
                prefix_token_ids=clean_prefix.prefix_token_ids,
                prefix_text=clean_prefix.prefix_text,
                matched_pred_objects=matched_prefix_objects,
            )
            append_text = "]}"
            append_ids = [
                int(t) for t in tokenizer.encode(append_text, add_special_tokens=False)
            ]
            tail_desc_pos = []
            tail_desc_weights = []
            y_train_ids = list(clean_prefix.prefix_token_ids) + list(append_ids)
            clean_target_text = str(clean_prefix.prefix_text) + str(append_text)
            sorted_duplicate_bursts = _sorted_duplicate_bursts_by_boundary(
                sorted_objects=sorted_objects,
                suppressed_duplicate_objects_by_boundary=(
                    triage.suppressed_duplicate_objects_by_boundary
                ),
            )
            (
                duplicate_control_first_divergence_diagnostics,
                duplicate_control_first_divergence_boundary_count,
                duplicate_control_first_divergence_skipped_no_divergence,
            ) = _build_duplicate_control_divergence_diagnostics(
                tokenizer=tokenizer,
                y_train_ids=y_train_ids,
                clean_target_text=clean_target_text,
                accepted_objects_clean=sorted_objects,
                fn_objects=[],
                suppressed_duplicate_objects_by_boundary=sorted_duplicate_bursts,
                boundary_prefix_texts=clean_prefix.boundary_prefix_texts,
                object_field_order=object_field_order,
            )
    else:
        if is_compact_full:
            append_body = _render_compact_objects(fn_objs) if fn_objs else ""
            append_text = (
                ("\n" + append_body)
                if clean_prefix.prefix_text and append_body
                else append_body
            )
        else:
            append_text = serialize_append_fragment_fn(
                fn_objects=fn_objs,
                prefix_text=clean_prefix.prefix_text,
                object_field_order=object_field_order,
            )
        append_ids = [
            int(t) for t in tokenizer.encode(append_text, add_special_tokens=False)
        ]

        if is_compact_full:
            tail_desc_pos, tail_desc_weights = _compact_desc_tail_positions_and_weights(
                tokenizer=tokenizer,
                token_ids=append_ids,
                tail_text=str(append_text),
                object_weights=fn_object_weights,
            )
        else:
            tail_desc_pos, tail_desc_weights = _desc_tail_positions_and_weights(
                tokenizer=tokenizer,
                token_ids=append_ids,
                object_weights=fn_object_weights,
            )

        y_train_ids = list(clean_prefix.prefix_token_ids) + list(append_ids)
        clean_target_text = str(clean_prefix.prefix_text) + str(append_text)
        if is_compact_full:
            duplicate_control_first_divergence_diagnostics = []
            duplicate_control_first_divergence_boundary_count = 0
            duplicate_control_first_divergence_skipped_no_divergence = 0
        else:
            (
                duplicate_control_first_divergence_diagnostics,
                duplicate_control_first_divergence_boundary_count,
                duplicate_control_first_divergence_skipped_no_divergence,
            ) = _build_duplicate_control_divergence_diagnostics(
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
        for fn_idx, (obj, rel_pos, obj_weight) in enumerate(
            zip(fn_objs, rel_groups, fn_object_weights)
        ):
            fn_bbox_groups.append(
                {
                    "pos": [
                        int(len(prompt_ids) + int(prefix_len_raw_local) + int(p))
                        for p in rel_pos
                    ],
                    "gt_bins": list(obj.points_norm1000),
                    "weight": float(obj_weight),
                }
            )

    stage2_trie_object_spans, stage2_trie_weak_fp_span_level_fallback = (
        _build_stage2_trie_fp_object_spans(
            tokenizer=tokenizer,
            clean_prefix=clean_prefix,
            compact_full=bool(is_compact_full),
            final_prefix_anchor_indices=final_prefix_anchor_indices,
            matched_kept_anchor_indices=matched_clean_indices,
            original_anchor_index_by_kept=original_anchor_index_by_kept,
            anchor_support_counts=triage.anchor_support_counts,
            fp_policy_mode=str(fp_policy_mode),
            fp_policy_weak_positive_weight=float(fp_policy_weak_positive_weight),
            fp_policy_require_explorer_support=bool(
                fp_policy_require_explorer_support
            ),
            fp_policy_min_support_count=int(fp_policy_min_support_count),
            fp_policy_require_token_score=bool(fp_policy_require_token_score),
        )
    )

    return _ChannelBSupervisionTargets(
        clean_prefix=clean_prefix,
        prefix_len_raw_local=prefix_len_raw_local,
        prefix_bbox_groups=prefix_bbox_groups,
        fn_bbox_groups=fn_bbox_groups,
        prefix_pos=prefix_pos,
        prefix_bins=prefix_bins,
        prefix_struct_pos=[int(p) for p in prefix_struct_pos],
        prefix_desc_pos=[int(p) for p in prefix_desc_pos],
        prefix_desc_weights=[float(w) for w in prefix_desc_weights],
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
        duplicate_control_first_divergence_diagnostics=list(
            duplicate_control_first_divergence_diagnostics
        ),
        duplicate_control_first_divergence_boundary_count=int(
            duplicate_control_first_divergence_boundary_count
        ),
        duplicate_control_first_divergence_skipped_no_divergence=int(
            duplicate_control_first_divergence_skipped_no_divergence
        ),
        rollout_template_family=str(rollout_template_family),
        rollout_parser_id=str(rollout_parser_id),
        rollout_append_policy_id=str(rollout_append_policy_id),
        rollout_context=str(rollout_context),
        rollout_fallback_reason=rollout_fallback_reason,
        rollout_fallback_loss_weight=float(rollout_fallback_loss_weight),
        rollout_counts_as_valid_rollout=bool(rollout_counts_as_valid_rollout),
        stage2_trie_object_spans=list(stage2_trie_object_spans),
        stage2_trie_weak_fp_span_level_fallback=bool(
            stage2_trie_weak_fp_span_level_fallback
        ),
    )


def _build_residual_set_correction_events(
    *,
    tokenizer: Any,
    response_token_ids: Sequence[int],
    parsed_bbox_objects_raw: Sequence[GTObject],
    compact_full_object_spans: Sequence[CompactFullObjectTokenSpan],
    gts: Sequence[GTObject],
    accepted_objects_clean: Sequence[GTObject],
    match: MatchResult,
    ul_promoted_objects: Sequence[Mapping[str, Any]] = (),
    assignment_iou_threshold: float,
    commit_iou_threshold: float | None = None,
    duplicate_burst_iou_threshold: float | None = None,
    sample_id: str,
    rollout_index: int,
    lambda_ul_promoted: float,
    rollout_id: str | None = None,
    duplicate_burst_prefix_rollback: bool = False,
) -> _ResidualSetCorrectionBuildResult:
    """Build sampled-path residual correction events for compact-full rollout-correction."""

    object_start_token_id = _single_token_id_for_text(
        tokenizer, OBJECT_REF_START_TOKEN, label="object_start"
    )
    box_start_token_id = _single_token_id_for_text(
        tokenizer, BOX_START_TOKEN, label="box_start"
    )
    stop_token_id = _stop_token_id(tokenizer)

    universe = _build_residual_universe_objects(
        tokenizer=tokenizer,
        gts=gts,
        ul_promoted_objects=ul_promoted_objects,
        lambda_ul_promoted=float(lambda_ul_promoted),
    )
    universe_by_id = {item.object_id: item for item in universe}
    metrics: Dict[str, float] = {
        "residual_object_count": float(len(universe)),
        "ul_promoted_object_count": float(
            sum(1 for item in universe if item.source == "ul")
        ),
        "event_count": 0.0,
        "atom_count": 0.0,
        "no_event_exact_path": 0.0,
        "dropped_stop_at_empty_prefix": 0.0,
    }
    del parsed_bbox_objects_raw, compact_full_object_spans, accepted_objects_clean, match
    raw_ids = [int(token_id) for token_id in response_token_ids]
    commit_threshold = (
        float(assignment_iou_threshold)
        if commit_iou_threshold is None
        else float(commit_iou_threshold)
    )
    duplicate_threshold = (
        0.95
        if duplicate_burst_iou_threshold is None
        else float(duplicate_burst_iou_threshold)
    )
    metrics["scanner_commit_iou_threshold"] = float(commit_threshold)
    metrics["scanner_duplicate_burst_iou_threshold"] = float(duplicate_threshold)

    initial_state = ResidualState(
        objects=tuple(item.residual_object for item in universe),
        remaining_object_ids=frozenset(str(item.object_id) for item in universe),
        active_candidate_ids=frozenset(str(item.object_id) for item in universe),
        object_start_token_id=int(object_start_token_id),
        box_start_token_id=int(box_start_token_id),
        stop_token_id=int(stop_token_id),
    )
    observed_rows = _observed_residual_rows_from_compact_response(
        tokenizer=tokenizer,
        response_token_ids=raw_ids,
    )
    scan = scan_dirty_prefix_rows(
        initial_state,
        observed_rows,
        commit_iou_threshold=float(commit_threshold),
        duplicate_burst_iou_threshold=float(duplicate_threshold),
        sample_id=str(sample_id),
        rollback_duplicate_burst=bool(duplicate_burst_prefix_rollback),
    )
    _add_residual_scan_metrics(metrics, scan)

    if scan.dropped_sample:
        metrics["no_event_dropped_dirty_prefix"] = 1.0
        retained_end = max(0, int(scan.retained_prefix_end or 0))
        y_train_ids = raw_ids[:retained_end]
        clean_target_text = _decode_token_ids(tokenizer, y_train_ids)
        return _residual_build_result(
            y_train_ids=y_train_ids,
            clean_target_text=clean_target_text,
            events=[],
            event_summaries=[],
            metrics=metrics,
        )

    remaining_ids = [
        str(item.object_id)
        for item in universe
        if str(item.object_id) in scan.final_state.remaining_object_ids
    ]
    retained_prefix_end = _residual_retained_prefix_end(scan, observed_rows)
    (
        y_train_ids,
        clean_target_text,
        boundary_slice,
        prefix_object_count,
        target_position_offset,
    ) = (
        _render_residual_scanned_target(
            tokenizer=tokenizer,
            raw_ids=raw_ids,
            rows=observed_rows,
            retained_prefix_end=int(retained_prefix_end),
            remaining_ids=remaining_ids,
            universe_by_id=universe_by_id,
        )
    )

    events: List[CorrectionEvent] = list(scan.events)
    if remaining_ids:
        if boundary_slice is None:
            raise ValueError("residual boundary adapter did not produce a target slice")
        events = _remap_spatial_wrong_desc_events_to_clean_target(
            events,
            remaining_ids=remaining_ids,
            boundary_slice=boundary_slice,
            prefix_object_count=int(prefix_object_count),
            target_position_offset=int(target_position_offset),
        )
        target_span = boundary_slice.object_spans[int(prefix_object_count)]
        current_target = universe_by_id[remaining_ids[0]]
        event = _build_residual_event_from_specs(
            universe=universe,
            current_target=current_target,
            remaining_ids=remaining_ids,
            draft_specs=_residual_continuation_draft_specs_from_span(
                target_span=target_span,
                target_position_offset=int(target_position_offset),
            ),
            correction_kind=_residual_continuation_correction_kind(scan),
            sample_id=sample_id,
            rollout_index=int(rollout_index),
            rollout_id=rollout_id,
            observed_token_ids=y_train_ids,
            observed_object_start=0,
            object_start_token_id=int(object_start_token_id),
            box_start_token_id=int(box_start_token_id),
            stop_token_id=int(stop_token_id),
        )
        events.append(event)
    elif not events:
        metrics["no_event_exact_path"] = 1.0

    return _residual_build_result(
        y_train_ids=y_train_ids,
        clean_target_text=clean_target_text,
        events=events,
        event_summaries=[_residual_event_summary(event) for event in events],
        metrics=metrics,
    )


def _observed_residual_rows_from_compact_response(
    *,
    tokenizer: Any,
    response_token_ids: Sequence[int],
) -> Tuple[ObservedResidualRow, ...]:
    token_ids = [int(token_id) for token_id in response_token_ids]
    if not token_ids:
        return ()

    pieces = decode_pieces(tokenizer, token_ids)
    token_spans: List[Tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = int(cursor)
        cursor += int(len(piece))
        token_spans.append((start, int(cursor)))
    text = "".join(str(piece) for piece in pieces)

    terminal_positions = [
        pos
        for marker in ("<|im_end|>", "<|endoftext|>")
        if (pos := text.find(marker)) >= 0
    ]
    parse_end = min(terminal_positions) if terminal_positions else len(text)
    parse_text = text[:parse_end]
    if not parse_text.strip("\r\n"):
        return ()

    object_starts: List[int] = []
    search_start = 0
    while True:
        found = parse_text.find(OBJECT_REF_START_TOKEN, search_start)
        if found < 0:
            break
        object_starts.append(int(found))
        search_start = int(found + len(OBJECT_REF_START_TOKEN))

    if not object_starts or parse_text[: object_starts[0]].strip("\r\n"):
        return (
            ObservedResidualRow(
                desc=None,
                bbox_norm1000=None,
                object_start=0,
                object_end=_token_index_at_or_after_char(token_spans, parse_end),
                reliable_span=False,
                reliable_resync=False,
                malformed=True,
            ),
        )

    rows: List[ObservedResidualRow] = []
    for row_index, row_start in enumerate(object_starts):
        next_row_start = (
            int(object_starts[int(row_index) + 1])
            if int(row_index) + 1 < len(object_starts)
            else int(parse_end)
        )
        row_end = int(next_row_start)
        while row_end > int(row_start) and parse_text[int(row_end) - 1] in "\r\n":
            row_end -= 1
        row = parse_text[int(row_start) : int(row_end)]
        object_start_token = _single_token_index_for_char_span(
            token_spans=token_spans,
            char_start=int(row_start),
            char_end=int(row_start) + len(OBJECT_REF_START_TOKEN),
            default=_token_index_at_or_after_char(token_spans, int(row_start)),
        )
        next_object_start_token = _token_index_at_or_after_char(
            token_spans,
            int(next_row_start),
        )

        box_rel = row.rfind(BOX_START_TOKEN)
        coord_tail_start = int(box_rel) + len(BOX_START_TOKEN)
        coord_matches = (
            list(_COMPACT_COORD_TOKEN_RE.finditer(row, coord_tail_start))
            if box_rel >= len(OBJECT_REF_START_TOKEN)
            else []
        )
        coord_tail_complete = (
            len(coord_matches) == 4
            and coord_matches[0].start() == coord_tail_start
            and all(
                left.end() == right.start()
                for left, right in zip(coord_matches, coord_matches[1:])
            )
            and coord_matches[-1].end() == len(row)
        )
        if coord_tail_complete:
            desc_start_char = int(row_start) + len(OBJECT_REF_START_TOKEN)
            desc_end_char = int(row_start) + int(box_rel)
            desc_token_positions = _token_indices_overlapping_char_span(
                token_spans=token_spans,
                char_start=desc_start_char,
                char_end=desc_end_char,
            )
            coord_positions = [
                _single_token_index_for_char_span(
                    token_spans=token_spans,
                    char_start=int(row_start) + int(match.start()),
                    char_end=int(row_start) + int(match.end()),
                    default=_token_index_at_or_after_char(
                        token_spans,
                        int(row_start) + int(match.start()),
                    ),
                )
                for match in coord_matches
            ]
            rows.append(
                ObservedResidualRow(
                    desc=row[len(OBJECT_REF_START_TOKEN) : int(box_rel)],
                    bbox_norm1000=tuple(int(match.group(1)) for match in coord_matches),  # type: ignore[arg-type]
                    object_start=int(object_start_token),
                    object_end=int(coord_positions[-1]) + 1,
                    reliable_span=True,
                    reliable_resync=True,
                    desc_token_ids=tuple(
                        int(token_ids[int(pos)]) for pos in desc_token_positions
                    ),
                    desc_token_positions=tuple(int(pos) for pos in desc_token_positions),
                    metadata={"row_index": int(row_index)},
                )
            )
            continue

        incomplete = int(row_index) + 1 >= len(object_starts)
        rows.append(
            ObservedResidualRow(
                desc=None,
                bbox_norm1000=None,
                object_start=int(object_start_token),
                object_end=None if incomplete else int(next_object_start_token),
                reliable_span=False,
                reliable_resync=not incomplete,
                malformed=not incomplete,
                incomplete=bool(incomplete),
                metadata={"row_index": int(row_index)},
            )
        )

    return tuple(rows)


def _token_index_at_or_after_char(
    token_spans: Sequence[Tuple[int, int]],
    char_pos: int,
) -> int:
    for index, (start, _end) in enumerate(token_spans):
        if int(start) >= int(char_pos):
            return int(index)
    return int(len(token_spans))


def _single_token_index_for_char_span(
    *,
    token_spans: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
    default: int,
) -> int:
    positions = _token_indices_overlapping_char_span(
        token_spans=token_spans,
        char_start=int(char_start),
        char_end=int(char_end),
    )
    if len(positions) == 1:
        return int(positions[0])
    return int(default)


def _add_residual_scan_metrics(
    metrics: MutableMapping[str, float],
    scan: ResidualRowScanResult,
) -> None:
    metrics["scanner_row_count"] = float(len(scan.row_decisions))
    metrics["scanner_event_count"] = float(len(scan.events))
    metrics["scanner_dirty_context_span_count"] = float(len(scan.dirty_context_spans))
    metrics["scanner_masked_label_span_count"] = float(len(scan.masked_label_spans))
    metrics["scanner_type_loss_mask_span_count"] = float(len(scan.type_loss_mask_spans))
    metrics["scanner_dropped_sample"] = float(1.0 if scan.dropped_sample else 0.0)
    metrics["scanner_final_remaining_object_count"] = float(
        len(scan.final_state.remaining_object_ids)
    )
    if scan.retained_prefix_end is not None:
        metrics["scanner_retained_prefix_end"] = float(int(scan.retained_prefix_end))
    for decision in scan.row_decisions:
        metrics[f"scanner_row_decision/{decision.kind}"] = (
            float(metrics.get(f"scanner_row_decision/{decision.kind}", 0.0)) + 1.0
        )
        if (
            decision.kind == "duplicate_burst"
            and bool(decision.metadata.get("prefix_rollback", False))
        ):
            metrics["scanner_duplicate_prefix_rollback"] = (
                float(metrics.get("scanner_duplicate_prefix_rollback", 0.0)) + 1.0
            )
    for reason in scan.no_atom_reasons:
        key = f"scanner_no_atom_reason/{reason}"
        metrics[key] = float(metrics.get(key, 0.0)) + 1.0


def _residual_retained_prefix_end(
    scan: ResidualRowScanResult,
    rows: Sequence[ObservedResidualRow],
) -> int:
    if scan.retained_prefix_end is not None:
        return max(0, int(scan.retained_prefix_end))
    end = 0
    for row in rows[: len(scan.row_decisions)]:
        if row.incomplete:
            break
        if row.object_end is not None:
            end = max(int(end), int(row.object_end))
    return int(end)


def _render_residual_scanned_target(
    *,
    tokenizer: Any,
    raw_ids: Sequence[int],
    rows: Sequence[ObservedResidualRow],
    retained_prefix_end: int,
    remaining_ids: Sequence[str],
    universe_by_id: Mapping[str, _ResidualSetUniverseObject],
) -> Tuple[List[int], str, ResidualBoundarySlice | None, int, int]:
    prefix_objects = _observed_prefix_objects(
        rows=rows,
        retained_prefix_end=int(retained_prefix_end),
    )
    remaining_objects = _residual_target_objects(
        target_ids=remaining_ids,
        universe_by_id=universe_by_id,
    )
    can_render_prefix = _can_render_observed_prefix(
        rows=rows,
        retained_prefix_end=int(retained_prefix_end),
        prefix_objects=prefix_objects,
    )
    if remaining_objects:
        if not can_render_prefix:
            clean_tail_text, tail_ids, boundary_slice = _render_residual_target_slice(
                tokenizer=tokenizer,
                target_objects=remaining_objects,
                boundary="object",
                object_index=0,
            )
            prefix_ids = [int(token_id) for token_id in raw_ids[: int(retained_prefix_end)]]
            y_train_ids = prefix_ids + list(tail_ids)
            clean_target_text = _decode_token_ids(tokenizer, y_train_ids)
            del clean_tail_text
            return (
                y_train_ids,
                clean_target_text,
                boundary_slice,
                0,
                int(retained_prefix_end),
            )
        target_objects = list(prefix_objects) + list(remaining_objects)
        clean_target_text, y_train_ids, boundary_slice = _render_residual_target_slice(
            tokenizer=tokenizer,
            target_objects=target_objects,
            boundary="object",
            object_index=int(len(prefix_objects)),
        )
        return y_train_ids, clean_target_text, boundary_slice, int(len(prefix_objects)), 0
    if prefix_objects and can_render_prefix:
        clean_target_text, y_train_ids, _boundary_slice = _render_residual_target_slice(
            tokenizer=tokenizer,
            target_objects=prefix_objects,
            boundary="assistant_start",
        )
        return y_train_ids, clean_target_text, None, int(len(prefix_objects)), 0
    y_train_ids = [int(token_id) for token_id in raw_ids[: int(retained_prefix_end)]]
    return y_train_ids, _decode_token_ids(tokenizer, y_train_ids), None, 0, 0


def _observed_prefix_objects(
    *,
    rows: Sequence[ObservedResidualRow],
    retained_prefix_end: int,
) -> List[GTObject]:
    objects: List[GTObject] = []
    for row_index, row in enumerate(rows):
        if row.object_end is None or int(row.object_end) > int(retained_prefix_end):
            continue
        if row.desc is None or row.bbox_norm1000 is None:
            continue
        if not _is_renderable_compact_desc(str(row.desc)):
            continue
        objects.append(
            GTObject(
                index=int(row_index),
                geom_type="bbox_2d",
                points_norm1000=[int(value) for value in row.bbox_norm1000],
                desc=str(row.desc),
            )
        )
    return objects


def _is_renderable_compact_desc(desc: str) -> bool:
    text = str(desc)
    if not text.strip():
        return False
    return not any(
        str(forbidden) in text for forbidden in COMPACT_DESC_FORBIDDEN_SUBSTRINGS
    )


def _can_render_observed_prefix(
    *,
    rows: Sequence[ObservedResidualRow],
    retained_prefix_end: int,
    prefix_objects: Sequence[GTObject],
) -> bool:
    complete_rows = [
        row
        for row in rows
        if row.object_end is not None
        and int(row.object_end) <= int(retained_prefix_end)
        and row.desc is not None
        and row.bbox_norm1000 is not None
    ]
    if len(complete_rows) != len(prefix_objects):
        return False
    for row in complete_rows:
        if row.desc is None or not _is_renderable_compact_desc(str(row.desc)):
            return False
        if row.bbox_norm1000 is None:
            return False
        x1, y1, x2, y2 = [int(value) for value in row.bbox_norm1000]
        if x2 <= x1 or y2 <= y1:
            return False
    return True


def _residual_continuation_correction_kind(scan: ResidualRowScanResult) -> str:
    if not scan.row_decisions:
        return "premature_stop"
    last_kind = str(scan.row_decisions[-1].kind)
    if last_kind in {"invalid_geometry", "malformed_span", "trailing_incomplete"}:
        return last_kind
    if last_kind == "duplicate_burst":
        return "repeated_object_boundary"
    if last_kind == "unmatched_dirty_context":
        return "fp_boundary"
    if last_kind == "spatial_wrong_desc_conflict":
        return "spatial_wrong_desc_conflict"
    return "premature_stop"


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _build_residual_universe_objects(
    *,
    tokenizer: Any,
    gts: Sequence[GTObject],
    ul_promoted_objects: Sequence[Mapping[str, Any]],
    lambda_ul_promoted: float,
) -> List[_ResidualSetUniverseObject]:
    universe: List[_ResidualSetUniverseObject] = []
    for gt_index, obj in enumerate(gts):
        universe.append(
            _residual_universe_object_from_gt(
                tokenizer=tokenizer,
                object_id=f"gt:{int(gt_index)}",
                source="labeled",
                source_index=int(gt_index),
                obj=obj,
                loss_weight=1.0,
                support_provenance=("labeled",),
            )
        )
    for ul_index, item in enumerate(ul_promoted_objects):
        obj = item.get("object") if isinstance(item, Mapping) else None
        if not isinstance(obj, GTObject):
            continue
        if not _is_renderable_compact_desc(str(obj.desc)):
            continue
        try:
            loss_weight = float(item.get("loss_weight", lambda_ul_promoted))
        except (TypeError, ValueError):
            loss_weight = float(lambda_ul_promoted)
        support_provenance_raw = item.get("support_provenance", ("ul",))
        if isinstance(support_provenance_raw, str):
            support_provenance = (support_provenance_raw,)
        elif isinstance(support_provenance_raw, Sequence):
            support_provenance = tuple(str(value) for value in support_provenance_raw)
        else:
            support_provenance = ("ul",)
        universe.append(
            _residual_universe_object_from_gt(
                tokenizer=tokenizer,
                object_id=f"ul:{int(ul_index)}",
                source="ul",
                source_index=int(ul_index),
                obj=obj,
                loss_weight=max(0.0, float(loss_weight)),
                support_provenance=support_provenance,
            )
        )
    return universe


def _residual_continuation_draft_specs_from_span(
    *,
    target_span: ResidualBoundaryObjectSpan,
    target_position_offset: int,
) -> List[Tuple[str, int, int]]:
    """Return full selected-object continuation specs from compact target spans."""

    offset = int(target_position_offset)
    specs: List[Tuple[str, int, int]] = [
        ("object_start", offset + int(target_span.object_start), 0)
    ]
    desc_start = int(target_span.desc_start)
    desc_end = int(target_span.desc_end)
    for desc_prefix_len, target_position in enumerate(range(desc_start, desc_end)):
        specs.append(("desc", offset + int(target_position), int(desc_prefix_len)))
    specs.append(("box_start", offset + int(target_span.box_start), 0))
    for coord_role, target_position in zip(
        _COORD_ROLE_BY_SLOT,
        target_span.coord_positions,
    ):
        specs.append((str(coord_role), offset + int(target_position), 0))
    return specs


def _residual_universe_object_from_gt(
    *,
    tokenizer: Any,
    object_id: str,
    source: str,
    source_index: int,
    obj: GTObject,
    loss_weight: float,
    support_provenance: Sequence[str],
) -> _ResidualSetUniverseObject:
    desc_token_ids = _compact_context_desc_token_ids(tokenizer=tokenizer, obj=obj)
    if not desc_token_ids:
        raise ValueError("residual_set_correction requires nonempty object descriptions")
    coord_token_ids = {
        role: _single_token_id_for_text(
            tokenizer,
            f"<|coord_{int(value)}|>",
            label=f"coord_{role}",
        )
        for role, value in zip(_COORD_ROLE_BY_SLOT, obj.points_norm1000)
    }
    residual_object = ResidualObject(
        object_id=str(object_id),
        desc_token_ids=desc_token_ids,
        coord_token_ids=coord_token_ids,
        desc_token_texts=tuple(decode_pieces(tokenizer, list(desc_token_ids))),
        loss_weight=float(loss_weight),
        metadata={
            "source": str(source),
            "source_index": int(source_index),
            "support_provenance": tuple(str(item) for item in support_provenance),
            "bbox_norm1000": tuple(int(value) for value in obj.points_norm1000),
            "desc_norm": normalize_desc(str(obj.desc)),
        },
    )
    gt_object = GTObject(
        index=int(source_index),
        geom_type="bbox_2d",
        points_norm1000=[int(v) for v in obj.points_norm1000],
        desc=str(obj.desc),
    )
    return _ResidualSetUniverseObject(
        object_id=str(object_id),
        source=str(source),
        source_index=int(source_index),
        gt_object=gt_object,
        residual_object=residual_object,
    )


def _compact_context_desc_token_ids(*, tokenizer: Any, obj: GTObject) -> Tuple[int, ...]:
    row_object = GTObject(
        index=0,
        geom_type="bbox_2d",
        points_norm1000=[int(v) for v in obj.points_norm1000],
        desc=str(obj.desc),
    )
    rendered, row_token_ids, boundary_slice = _render_residual_target_slice(
        tokenizer=tokenizer,
        target_objects=[row_object],
        boundary="assistant_start",
    )
    del rendered
    span = boundary_slice.object_spans[0]
    return tuple(
        int(token_id)
        for token_id in row_token_ids[int(span.desc_start) : int(span.desc_end)]
    )


def _build_residual_event_from_specs(
    *,
    universe: Sequence[_ResidualSetUniverseObject],
    current_target: _ResidualSetUniverseObject,
    remaining_ids: Sequence[str],
    draft_specs: Sequence[Tuple[str, int, int]],
    correction_kind: str,
    sample_id: str,
    rollout_index: int,
    observed_token_ids: Sequence[int],
    observed_object_start: int,
    object_start_token_id: int,
    box_start_token_id: int,
    stop_token_id: int,
    rollout_id: str | None = None,
) -> CorrectionEvent:
    base_state = ResidualState(
        objects=tuple(item.residual_object for item in universe),
        remaining_object_ids=frozenset(str(item_id) for item_id in remaining_ids),
        active_candidate_ids=frozenset(str(item_id) for item_id in remaining_ids),
        object_start_token_id=int(object_start_token_id),
        box_start_token_id=int(box_start_token_id),
        stop_token_id=int(stop_token_id),
    )
    drafts: List[CorrectionAtomDraft] = []
    for slot, target_position, desc_prefix_len in draft_specs:
        state = _advance_residual_state_to_slot(
            base_state,
            current_target=current_target.residual_object,
            slot=str(slot),
            desc_prefix_len=int(desc_prefix_len),
        )
        actions = enumerate_valid_actions(state, slot=str(slot))
        selected_token_id = _selected_token_for_slot(
            current_target.residual_object,
            str(slot),
            desc_prefix_len=int(desc_prefix_len),
            object_start_token_id=int(object_start_token_id),
            box_start_token_id=int(box_start_token_id),
        )
        actions = _with_selected_path_metadata(
            actions=actions,
            selected_token_id=int(selected_token_id),
            selected_object=current_target.residual_object,
        )
        selected = _selected_action_for_token(actions, int(selected_token_id))
        observed_rel = int(target_position) - int(observed_object_start)
        observed_token_id = (
            int(observed_token_ids[int(observed_rel)])
            if 0 <= observed_rel < len(observed_token_ids)
            else None
        )
        drafts.append(
            CorrectionAtomDraft(
                correction_kind=correction_kind,  # type: ignore[arg-type]
                target_position=int(target_position),
                logit_position=int(target_position) - 1,
                valid_actions=actions,
                selected_action=selected,
                metadata={
                    "observed_token_id": observed_token_id,
                    "rollout_index": int(rollout_index),
                    **(
                        {"rollout_id": str(rollout_id)}
                        if rollout_id is not None
                        else {}
                    ),
                    "anchor_position": int(observed_object_start),
                    "slot": str(slot),
                    "selected_object_id": current_target.object_id,
                    "selected_object_source": current_target.source,
                    "selected_object_source_index": int(current_target.source_index),
                },
            )
        )
    return CorrectionEvent(
        correction_kind=correction_kind,  # type: ignore[arg-type]
        sample_id=str(sample_id),
        atom_drafts=tuple(drafts),
        metadata={
            "rollout_index": int(rollout_index),
            **({"rollout_id": str(rollout_id)} if rollout_id is not None else {}),
            "anchor_position": int(observed_object_start),
            "correction_builder": "stage2_residual_events_v1",
            "full_residual_events": True,
        },
    )


def _advance_residual_state_to_slot(
    state: ResidualState,
    *,
    current_target: ResidualObject,
    slot: str,
    desc_prefix_len: int,
) -> ResidualState:
    if slot == "object_start":
        return state
    object_start_action = _selected_action_for_token(
        enumerate_valid_actions(state, slot="object_start"),
        int(state.object_start_token_id or -1),
    )
    state = transition_state(state, object_start_action)
    desc_prefix_len = (
        int(desc_prefix_len)
        if slot == "desc"
        else len(current_target.desc_token_ids)
    )
    for token_id in current_target.desc_token_ids[:desc_prefix_len]:
        action = _selected_action_for_token(
            enumerate_valid_actions(state, slot="desc"),
            int(token_id),
        )
        state = transition_state(state, action)
    if slot == "desc":
        return state
    if slot == "box_start":
        return state
    box_action = _selected_action_for_token(
        enumerate_valid_actions(state, slot="box_start"),
        int(state.box_start_token_id or -1),
    )
    state = transition_state(state, box_action)
    for coord_role in _COORD_ROLE_BY_SLOT:
        if slot == coord_role:
            return state
        action = _selected_action_for_token(
            enumerate_valid_actions(state, slot=coord_role),
            int(current_target.coord_token_ids[coord_role]),
        )
        state = transition_state(state, action)
    raise ValueError(f"unsupported residual-set event slot: {slot!r}")


def _selected_token_for_slot(
    obj: ResidualObject,
    slot: str,
    *,
    desc_prefix_len: int,
    object_start_token_id: int,
    box_start_token_id: int,
) -> int:
    if slot == "object_start":
        return int(object_start_token_id)
    if slot == "box_start":
        return int(box_start_token_id)
    if slot == "desc":
        cursor = int(desc_prefix_len)
        if cursor < 0 or cursor >= len(obj.desc_token_ids):
            raise ValueError("residual-set desc draft prefix is out of range")
        return int(obj.desc_token_ids[cursor])
    if slot in _COORD_ROLE_BY_SLOT:
        return int(obj.coord_token_ids[slot])
    raise ValueError(f"unsupported residual-set event slot: {slot!r}")


def _with_selected_path_metadata(
    *,
    actions: Sequence[ValidAction],
    selected_token_id: int,
    selected_object: ResidualObject,
) -> Tuple[ValidAction, ...]:
    selected_support = selected_object.metadata.get("support_provenance", ("labeled",))
    if isinstance(selected_support, str):
        selected_support_tuple = (selected_support,)
    else:
        selected_support_tuple = tuple(str(item) for item in selected_support)
    out: List[ValidAction] = []
    for action in actions:
        if int(action.token_id) != int(selected_token_id):
            out.append(action)
            continue
        metadata = dict(action.metadata)
        metadata["loss_weight"] = float(selected_object.loss_weight)
        metadata["support_provenance"] = selected_support_tuple
        out.append(
            replace(
                action,
                selected_object_id=(
                    selected_object.object_id
                    if selected_object.object_id in action.candidate_ids_after
                    else action.selected_object_id
                ),
                metadata=metadata,
            )
        )
    return tuple(out)


def _selected_action_for_token(
    actions: Sequence[ValidAction],
    token_id: int,
) -> ValidAction:
    for action in actions:
        if int(action.token_id) == int(token_id):
            return action
    raise ValueError("residual_set_correction selected target token is not valid")


def _residual_target_objects(
    *,
    target_ids: Sequence[str],
    universe_by_id: Mapping[str, _ResidualSetUniverseObject],
) -> List[GTObject]:
    return [
        GTObject(
            index=int(row_index),
            geom_type="bbox_2d",
            points_norm1000=[
                int(value)
                for value in universe_by_id[str(object_id)].gt_object.points_norm1000
            ],
            desc=str(universe_by_id[str(object_id)].gt_object.desc),
        )
        for row_index, object_id in enumerate(target_ids)
    ]


def _render_residual_target_slice(
    *,
    tokenizer: Any,
    target_objects: Sequence[GTObject],
    boundary: str,
    object_index: int | None = None,
) -> Tuple[str, List[int], ResidualBoundarySlice]:
    if not target_objects:
        raise ValueError("residual boundary adapter requires at least one target object")
    adapter = ResidualBoundaryAdapter(tokenizer=tokenizer)
    rendered = adapter.render_objects(target_objects)
    boundary_slice = adapter.slice_from_boundary(
        rendered,
        boundary=boundary,
        object_index=object_index,
    )
    token_ids = [
        int(token_id)
        for token_id in (
            boundary_slice.retained_prefix_input_ids
            + boundary_slice.suffix_input_ids
        )
    ]
    return str(rendered.text), token_ids, boundary_slice


def _render_residual_target_ids(
    *,
    tokenizer: Any,
    target_ids: Sequence[str],
    universe_by_id: Mapping[str, _ResidualSetUniverseObject],
) -> Tuple[str, List[int]]:
    objects = _residual_target_objects(
        target_ids=target_ids,
        universe_by_id=universe_by_id,
    )
    if not objects:
        return "", []
    text, token_ids, _boundary_slice = _render_residual_target_slice(
        tokenizer=tokenizer,
        target_objects=objects,
        boundary="assistant_start",
    )
    return str(text), token_ids


def _residual_build_result(
    *,
    y_train_ids: Sequence[int],
    clean_target_text: str,
    events: Sequence[CorrectionEvent],
    event_summaries: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, float],
) -> _ResidualSetCorrectionBuildResult:
    metric_out = dict(metrics)
    metric_out["event_count"] = float(len(events))
    metric_out["atom_count"] = float(
        sum(len(event.atom_drafts) for event in events)
    )
    return _ResidualSetCorrectionBuildResult(
        y_train_ids=[int(token_id) for token_id in y_train_ids],
        clean_target_text=str(clean_target_text),
        prefix_len_raw_local=int(len(y_train_ids)),
        events=list(events),
        event_summaries=[dict(item) for item in event_summaries],
        metrics=metric_out,
    )


def _residual_event_summary(event: CorrectionEvent) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "correction_kind": str(event.correction_kind),
        "sample_id": str(event.sample_id),
        "atom_count": int(len(event.atom_drafts)),
        "target_positions": [
            int(draft.target_position) for draft in event.atom_drafts
        ],
        "slots": [
            str(draft.metadata.get("slot", "")) for draft in event.atom_drafts
        ],
        "support_provenance": sorted(
            {
                str(item)
                for draft in event.atom_drafts
                for action in draft.valid_actions
                for item in _metadata_sequence(
                    action.metadata.get("support_provenance", ("labeled",))
                )
            }
        ),
    }
    rollout_id = event.metadata.get("rollout_id")
    if rollout_id is None:
        for draft in event.atom_drafts:
            if draft.metadata.get("rollout_id") is not None:
                rollout_id = draft.metadata["rollout_id"]
                break
    if rollout_id is not None:
        summary["rollout_id"] = str(rollout_id)
    return summary


def _shift_residual_correction_events(
    events: Sequence[CorrectionEvent],
    *,
    position_offset: int,
) -> List[CorrectionEvent]:
    offset = int(position_offset)
    if offset == 0:
        return list(events)
    shifted_events: List[CorrectionEvent] = []
    for event in events:
        shifted_drafts: List[CorrectionAtomDraft] = []
        for draft in event.atom_drafts:
            metadata = dict(draft.metadata)
            if "anchor_position" in metadata:
                try:
                    metadata["anchor_position"] = int(metadata["anchor_position"]) + offset
                except (TypeError, ValueError):
                    pass
            shifted_drafts.append(
                CorrectionAtomDraft(
                    correction_kind=draft.correction_kind,
                    target_position=int(draft.target_position) + offset,
                    logit_position=int(draft.logit_position) + offset,
                    valid_actions=draft.valid_actions,
                    selected_action=draft.selected_action,
                    metadata=metadata,
                )
            )
        event_metadata = dict(event.metadata)
        if "anchor_position" in event_metadata:
            try:
                event_metadata["anchor_position"] = (
                    int(event_metadata["anchor_position"]) + offset
                )
            except (TypeError, ValueError):
                pass
        event_metadata["position_offset"] = offset
        shifted_events.append(
            CorrectionEvent(
                correction_kind=event.correction_kind,
                sample_id=event.sample_id,
                atom_drafts=tuple(shifted_drafts),
                metadata=event_metadata,
            )
        )
    return shifted_events


def _remap_spatial_wrong_desc_events_to_clean_target(
    events: Sequence[CorrectionEvent],
    *,
    remaining_ids: Sequence[str],
    boundary_slice: ResidualBoundarySlice,
    prefix_object_count: int,
    target_position_offset: int,
) -> List[CorrectionEvent]:
    remapped_events: List[CorrectionEvent] = []
    remaining_index_by_id = {
        str(object_id): int(index) for index, object_id in enumerate(remaining_ids)
    }
    for event in events:
        if event.correction_kind != "spatial_wrong_desc_conflict":
            remapped_events.append(event)
            continue
        object_id = event.metadata.get("conflicting_object_id")
        if object_id is None or str(object_id) not in remaining_index_by_id:
            remapped_events.append(event)
            continue
        span_index = int(prefix_object_count) + remaining_index_by_id[str(object_id)]
        if span_index < 0 or span_index >= len(boundary_slice.object_spans):
            remapped_events.append(event)
            continue
        target_span = boundary_slice.object_spans[span_index]

        shifted_drafts: List[CorrectionAtomDraft] = []
        for draft in event.atom_drafts:
            if draft.correction_kind != "spatial_wrong_desc_conflict":
                shifted_drafts.append(draft)
                continue
            metadata = dict(draft.metadata)
            try:
                divergence = int(metadata["desc_divergence"])
            except (KeyError, TypeError, ValueError):
                shifted_drafts.append(draft)
                continue
            target_position = int(target_position_offset) + int(target_span.desc_start) + divergence
            if target_position < int(target_position_offset) + int(target_span.desc_start):
                shifted_drafts.append(draft)
                continue
            if target_position >= int(target_position_offset) + int(target_span.desc_end):
                shifted_drafts.append(draft)
                continue
            metadata["raw_target_position"] = int(draft.target_position)
            shifted_drafts.append(
                CorrectionAtomDraft(
                    correction_kind=draft.correction_kind,
                    target_position=int(target_position),
                    logit_position=int(target_position) - 1,
                    valid_actions=draft.valid_actions,
                    selected_action=draft.selected_action,
                    metadata=metadata,
                )
            )
        remapped_events.append(
            CorrectionEvent(
                correction_kind=event.correction_kind,
                sample_id=event.sample_id,
                atom_drafts=tuple(shifted_drafts),
                metadata=event.metadata,
            )
        )
    return remapped_events


def _metadata_sequence(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _single_token_id_for_text(tokenizer: Any, token_text: str, *, label: str) -> int:
    token_ids = [int(token_id) for token_id in tokenizer.encode(str(token_text), add_special_tokens=False)]
    if len(token_ids) == 1:
        return int(token_ids[0])
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = int(convert(str(token_text)))
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is None or int(token_id) != int(unk_id):
            return int(token_id)
    raise ValueError(f"residual_set_correction {label} must resolve to one token")


def _stop_token_id(tokenizer: Any) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return _single_token_id_for_text(tokenizer, "<|im_end|>", label="stop")


def _objective_spec_get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _objective_spec_enabled_for_rollout_correction(spec: Any) -> bool:
    return bool(_objective_spec_get(spec, "enabled", True))


def _residual_set_option_positive_int(
    config: Mapping[str, Any],
    *,
    key: str,
) -> int:
    value = config.get(key, _RESIDUAL_SET_OPTION_DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
        raise ValueError(f"residual_set_correction.config.{key} must be a positive integer")
    return int(value)


def _residual_set_option_nonnegative_float(
    config: Mapping[str, Any],
    *,
    key: str,
) -> float:
    value = config.get(key, _RESIDUAL_SET_OPTION_DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"residual_set_correction.config.{key} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"residual_set_correction.config.{key} must be finite")
    if result < 0.0:
        raise ValueError(f"residual_set_correction.config.{key} must be nonnegative")
    return result


def _residual_set_option_threshold(
    config: Mapping[str, Any],
    *,
    key: str,
) -> float:
    value = config.get(key, _RESIDUAL_SET_OPTION_DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"residual_set_correction.config.{key} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"residual_set_correction.config.{key} must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"residual_set_correction.config.{key} must be in [0, 1]")
    return result


def _residual_set_option_bool(
    config: Mapping[str, Any],
    *,
    key: str,
) -> bool:
    value = config.get(key, _RESIDUAL_SET_OPTION_DEFAULTS[key])
    if not isinstance(value, bool):
        raise TypeError(f"residual_set_correction.config.{key} must be a boolean")
    return bool(value)


def _rollout_correction_residual_set_options(
    objective_specs: Sequence[Any] | None,
) -> Dict[str, Any] | None:
    for spec in objective_specs or ():
        spec_name = str(_objective_spec_get(spec, "name", "") or "")
        if spec_name not in _RESIDUAL_STATE_TRIE_OBJECTIVE_NAMES:
            continue
        if not _objective_spec_enabled_for_rollout_correction(spec):
            continue
        config_raw = _objective_spec_get(spec, "config", {})
        if not isinstance(config_raw, Mapping):
            raise TypeError("Stage-2 residual trie config must be a mapping")
        config = dict(config_raw)
        base_seed = config.get("base_seed", _RESIDUAL_SET_OPTION_DEFAULTS["base_seed"])
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValueError("Stage-2 residual trie config base_seed must be an integer")
        expected_num_rollouts = _residual_set_option_positive_int(
            config,
            key="expected_num_rollouts",
        )
        lambda_ul_promoted = _residual_set_option_nonnegative_float(
            config,
            key="lambda_ul_promoted",
        )
        min_ul_valid_rollouts = _residual_set_option_positive_int(
            config,
            key="min_ul_valid_rollouts",
        )
        ul_consensus_ratio = _residual_set_option_threshold(
            config,
            key="ul_consensus_ratio",
        )
        if float(ul_consensus_ratio) != 1.0:
            raise ValueError(
                "Stage-2 residual trie config ul_consensus_ratio must be 1.0 "
                "because mine_ul_consensus currently supports only consensus_ratio == 1.0"
            )
        ul_cluster_iou_threshold = _residual_set_option_threshold(
            config,
            key="ul_cluster_iou_threshold",
        )
        ul_gray_iou_low = _residual_set_option_threshold(
            config,
            key="ul_gray_iou_low",
        )
        if float(ul_gray_iou_low) > float(ul_cluster_iou_threshold):
            raise ValueError(
                "Stage-2 residual trie config ul_gray_iou_low must be <= "
                "ul_cluster_iou_threshold"
            )
        return {
            "strict_builder_invariants": bool(
                config.get("strict_builder_invariants", False)
            ),
            "expected_num_rollouts": int(expected_num_rollouts),
            "base_seed": int(base_seed),
            "lambda_ul_promoted": float(lambda_ul_promoted),
            "commit_iou_threshold": _residual_set_option_threshold(
                config,
                key="commit_iou_threshold",
            ),
            "duplicate_burst_iou_threshold": _residual_set_option_threshold(
                config,
                key="duplicate_burst_iou_threshold",
            ),
            "duplicate_burst_prefix_rollback": _residual_set_option_bool(
                config,
                key="duplicate_burst_prefix_rollback",
            ),
            "ul_cluster_iou_threshold": float(ul_cluster_iou_threshold),
            "ul_gray_iou_low": float(ul_gray_iou_low),
            "min_ul_valid_rollouts": int(min_ul_valid_rollouts),
            "ul_consensus_ratio": float(ul_consensus_ratio),
        }
    return None


def _rollout_correction_residual_set_enabled(
    objective_specs: Sequence[Any] | None,
) -> bool:
    return _rollout_correction_residual_set_options(objective_specs) is not None


def _residual_set_singleton_atom(
    *,
    enc_ids_list: Sequence[int],
    target_position: int,
    token_role: TokenRole,
    loss_weight: float,
    coord_role: str | None,
    provenance: Mapping[str, Any],
) -> SupervisionAtom | None:
    target_position_i = int(target_position)
    if target_position_i <= 0 or target_position_i >= len(enc_ids_list):
        return None
    selected_token_id = int(enc_ids_list[target_position_i])
    return SupervisionAtom(
        batch_index=0,
        logit_position=int(target_position_i - 1),
        target_position=target_position_i,
        allowed_token_roles=frozenset({token_role}),
        selected_token_role=token_role,
        valid_token_ids=frozenset({selected_token_id}),
        selected_token_id=selected_token_id,
        latent_valid_token_ids=frozenset({selected_token_id}),
        coverage_target_weights=None,
        loss_tags=frozenset({"stage2", "rollout_correction", "residual_set"}),
        loss_weight=max(0.0, float(loss_weight)),
        coord_role=coord_role,
        provenance=dict(provenance),
    )


def _build_residual_set_target_ir_from_meta_positions(
    *,
    enc_ids_list: Sequence[int],
    prompt_len: int,
    prefix_len: int,
    train_len: int,
    encoded_len: int,
    bbox_groups_prefix: Sequence[Mapping[str, Any]],
    bbox_groups_fn: Sequence[Mapping[str, Any]],
    prefix_desc_pos: Sequence[int],
    prefix_desc_weights: Sequence[float],
    tail_desc_pos: Sequence[int],
    tail_desc_weights: Sequence[float],
    rollin_policy: str,
    base_seed: int,
) -> TeacherForcingTargetIR:
    atoms: List[SupervisionAtom] = []
    seen_positions: set[int] = set()
    lower = int(prompt_len)
    upper = min(
        int(encoded_len),
        int(len(enc_ids_list)),
        int(prompt_len) + int(train_len),
    )

    def _append_atom(
        *,
        target_position: int,
        token_role: TokenRole,
        loss_weight: float,
        coord_role: str | None,
        provenance: Mapping[str, Any],
    ) -> None:
        target_position_i = int(target_position)
        if target_position_i in seen_positions:
            return
        if target_position_i < lower or target_position_i >= upper:
            return
        atom = _residual_set_singleton_atom(
            enc_ids_list=enc_ids_list,
            target_position=target_position_i,
            token_role=token_role,
            loss_weight=float(loss_weight),
            coord_role=coord_role,
            provenance=provenance,
        )
        if atom is None:
            return
        atoms.append(atom)
        seen_positions.add(target_position_i)

    for group_kind, groups in (
        ("prefix", bbox_groups_prefix),
        ("fn", bbox_groups_fn),
    ):
        for group_index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                continue
            positions = group.get("pos", ())
            gt_bins = group.get("gt_bins", ())
            if not isinstance(positions, Sequence) or isinstance(
                positions, (str, bytes)
            ):
                continue
            if len(positions) != 4:
                continue
            loss_weight = float(group.get("weight", 1.0))
            for slot_index, position in enumerate(positions):
                _append_atom(
                    target_position=int(position),
                    token_role=TokenRole.COORD,
                    loss_weight=loss_weight,
                    coord_role=_COORD_ROLE_BY_SLOT[int(slot_index)],
                    provenance={
                        "objective": _RESIDUAL_SET_OBJECTIVE_NAME,
                        "correction_kind": "selected_path_singleton",
                        "source": f"bbox_group_{group_kind}",
                        "source_position_kind": f"bbox_group_{group_kind}",
                        "bbox_group_index": int(group_index),
                        "coord_slot": _COORD_ROLE_BY_SLOT[int(slot_index)],
                        "gt_bin": (
                            int(gt_bins[int(slot_index)])
                            if isinstance(gt_bins, Sequence)
                            and not isinstance(gt_bins, (str, bytes))
                            and len(gt_bins) > int(slot_index)
                            else None
                        ),
                    },
                )

    for source, rel_positions, weights, base in (
        (
            "prefix_desc",
            prefix_desc_pos,
            prefix_desc_weights,
            int(prompt_len),
        ),
        (
            "tail_desc",
            tail_desc_pos,
            tail_desc_weights,
            int(prompt_len) + int(prefix_len),
        ),
    ):
        for desc_index, rel_position in enumerate(rel_positions):
            try:
                weight = float(weights[int(desc_index)])
            except (IndexError, TypeError, ValueError):
                weight = 1.0
            _append_atom(
                target_position=int(base) + int(rel_position),
                token_role=TokenRole.TEXT,
                loss_weight=weight,
                coord_role=None,
                provenance={
                    "objective": _RESIDUAL_SET_OBJECTIVE_NAME,
                    "correction_kind": "selected_path_singleton",
                    "source": source,
                    "source_position_kind": source,
                    "desc_index": int(desc_index),
                },
            )

    return TeacherForcingTargetIR(
        schema_version=TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION,
        atoms=tuple(atoms),
        metadata={
            "stage": "stage2",
            "stage2_surface": "rollout_correction",
            "objective": _RESIDUAL_SET_OBJECTIVE_NAME,
            "position_space": "segment_local",
            "correction_builder": "stage2_meta_position_singleton_v0",
            "correction_semantics": "selected_token_singleton_not_full_residual",
            "full_residual_events": False,
            "marginal_scope": MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN,
            "rollin_policy": str(rollin_policy),
            "base_seed": int(base_seed),
            "prompt_len": int(prompt_len),
            "prefix_len": int(prefix_len),
            "train_len": int(train_len),
            "encoded_len": int(encoded_len),
        },
    )


def _build_rollout_correction_meta_entry(
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
    prefix_desc_pos: Sequence[int],
    prefix_desc_weights: Sequence[float],
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
    duplicate_control_first_divergence_diagnostics: Sequence[Stage2DuplicateControlDivergenceDiagnostic],
    duplicate_control_first_divergence_boundary_count: int,
    duplicate_control_first_divergence_skipped_no_divergence: int,
    assignment_strategy: str,
    assignment_iou_threshold: float,
    rollout_template_family: str,
    rollout_parser_id: str,
    rollout_append_policy_id: str,
    rollout_context: str,
    rollout_fallback_reason: str | None,
    rollout_fallback_loss_weight: float,
    rollout_counts_as_valid_rollout: bool,
    y_train_ids: Sequence[int],
    sample_id: str,
    rollout_index: int,
    stage2_trie_candidates: Sequence[Stage2TrieCandidate] | None = None,
    stage2_trie_object_spans: Sequence[Stage2TrieObjectSpan],
    stage2_trie_weak_fp_span_level_fallback: bool,
    residual_set_selected: bool = False,
    residual_set_target_ir: TeacherForcingTargetIR | None = None,
    residual_set_event_summaries: Sequence[Mapping[str, Any]] = (),
    residual_set_metrics: Mapping[str, float] | None = None,
    residual_set_rollin_policy: str = _DEFAULT_RESIDUAL_SET_ROLLIN_POLICY,
    residual_set_base_seed: int = _DEFAULT_RESIDUAL_SET_BASE_SEED,
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
    prefix_desc_pos_eff: List[int] = []
    prefix_desc_weights_eff: List[float] = []
    prefix_cap = max(0, int(prefix_len_eff))
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

    for rel, weight in zip(prefix_desc_pos, prefix_desc_weights):
        try:
            rel_i = int(rel)
            weight_f = float(weight)
        except (TypeError, ValueError):
            continue
        if 0 <= rel_i < prefix_cap:
            prefix_desc_pos_eff.append(rel_i)
            prefix_desc_weights_eff.append(weight_f)

    meta_entry: Stage2ChannelBMeta = {
        "stage2_surface": "rollout_correction",
        "stage2_invalid_rollout": int(invalid_rollout),
        "rollout_seed_base": int(seed_base),
        "rollout_template_family": str(rollout_template_family),
        "rollout_parser_id": str(rollout_parser_id),
        "rollout_append_policy_id": str(rollout_append_policy_id),
        "rollout_context": str(rollout_context),
        "rollout_fallback_reason": rollout_fallback_reason,
        "rollout_fallback_loss_weight": float(rollout_fallback_loss_weight),
        "rollout_counts_as_valid_rollout": bool(rollout_counts_as_valid_rollout),
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
        "prefix_desc_pos": [int(p) for p in prefix_desc_pos_eff],
        "prefix_desc_weights": [float(w) for w in prefix_desc_weights_eff],
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
            and semantic_stop_meta.get("continue_token_id") is not None
            else None
        ),
        "fn_object_weights": [float(w) for w in fn_object_weights],
        "bbox_groups_prefix": bbox_groups_prefix,
        "bbox_groups_fn": bbox_groups_fn,
        "anchor_decode_mode": str(anchor_decode_mode),
        "explorer_decode_mode": str(explorer_decode_mode),
        "valid_explorer_count": int(valid_explorer_count),
        "current_decode_mode": str(anchor_decode_mode),
        "peer_reference_decode_mode": str(explorer_decode_mode),
        "valid_peer_count": int(valid_explorer_count),
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
        "duplicate_survivor_current_indices": [
            int(idx) for idx in duplicate_survivor_anchor_indices
        ],
        "duplicate_exempt_current_indices": [
            int(idx) for idx in duplicate_exempt_anchor_indices
        ],
        "duplicate_suppressed_current_indices": [
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
        "current_gt_backed_indices": [int(idx) for idx in anchor_gt_backed_indices],
        "current_support_counts": [int(v) for v in anchor_support_counts],
        "current_support_rates": [float(v) for v in anchor_support_rates],
        "shield_only_current_indices": [int(idx) for idx in shielded_anchor_indices],
        "dead_current_indices": [int(idx) for idx in dead_anchor_indices],
        "lvis_verified_positive_dead_current_indices": [
            int(idx) for idx in lvis_verified_positive_dead_anchor_indices
        ],
        "lvis_verified_negative_dead_current_indices": [
            int(idx) for idx in lvis_verified_negative_dead_anchor_indices
        ],
        "lvis_not_exhaustive_current_indices": [
            int(idx) for idx in lvis_not_exhaustive_anchor_indices
        ],
        "lvis_unevaluable_current_indices": [
            int(idx) for idx in lvis_unevaluable_anchor_indices
        ],
        "pseudo_positive_current_indices": [
            int(idx) for idx in pseudo_positive_anchor_indices
        ],
        "dead_peer_indices_by_view": [
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
        "duplicate_control_first_divergence_diagnostics": [
            {
                "boundary": int(item["boundary"]),
                "clean_rel_pos": int(item["clean_rel_pos"]),
                "duplicate_token_id": int(item["duplicate_token_id"]),
            }
            for item in duplicate_control_first_divergence_diagnostics
        ],
        "duplicate_control_first_divergence_boundary_count": int(
            duplicate_control_first_divergence_boundary_count
        ),
        "duplicate_control_first_divergence_skipped_no_divergence": int(
            duplicate_control_first_divergence_skipped_no_divergence
        ),
        "assignment_strategy": str(assignment_strategy),
        "assignment_iou_threshold": float(assignment_iou_threshold),
    }
    if bool(stage2_trie_weak_fp_span_level_fallback):
        meta_entry["stage2_trie_weak_fp_span_level_fallback"] = True
    if bool(residual_set_selected):
        if residual_set_target_ir is None:
            raise ValueError(
                "residual_set_correction production path requires a live "
                "CorrectionEvent-derived target IR; refusing singleton fallback"
            )
        meta_entry["residual_set_target_ir"] = residual_set_target_ir
        meta_entry["residual_set_event_summaries"] = [
            dict(item) for item in residual_set_event_summaries
        ]
        meta_entry["residual_set_rollin_policy"] = str(residual_set_rollin_policy)
        meta_entry["residual_set_base_seed"] = int(residual_set_base_seed)
        metrics = dict(residual_set_metrics or {})
        metrics.setdefault("atom_count", float(len(residual_set_target_ir.atoms)))
        meta_entry["residual_set_metrics"] = metrics
    return meta_entry, int(closure_supervision_drop_count)


def _attach_stage2_trie_sidecar_to_meta(
    *,
    meta_entry: MutableMapping[str, Any],
    y_train_ids: Sequence[int],
    assistant_span_ids: Sequence[int] | None = None,
    tokenizer: Any | None = None,
    prompt_len: int,
    sample_id: str,
    rollout_index: int,
    stage2_trie_candidates: Sequence[Stage2TrieCandidate] | None = None,
    stage2_trie_object_spans: Sequence[Stage2TrieObjectSpan] = (),
    stage2_trie_weak_fp_span_level_fallback: bool = False,
) -> None:
    label_position_start = int(prompt_len)
    if stage2_trie_candidates is None:
        if label_position_start <= 0:
            return
        candidates = _build_implicit_stage2_trie_candidates(
            meta_entry=meta_entry,
            y_train_ids=y_train_ids,
            sample_id=sample_id,
            rollout_index=rollout_index,
            stage2_trie_object_spans=stage2_trie_object_spans,
        )
        if not candidates:
            return
    else:
        candidates = list(stage2_trie_candidates)
        explicit_sample_ids = {str(candidate.sample_id) for candidate in candidates}
        if explicit_sample_ids and explicit_sample_ids != {str(sample_id)}:
            expected = ", ".join(sorted(explicit_sample_ids))
            raise ValueError(
                "Stage-2 trie explicit candidate group sample_id must match "
                f"metadata sample_id={sample_id!r}; got {expected}"
            )

    semantic_role_by_position = _build_stage2_trie_semantic_role_map(
        meta_entry=meta_entry,
        y_train_ids=y_train_ids,
        prompt_len=prompt_len,
        tokenizer=tokenizer,
    )
    extra_token_targets = _build_stage2_trie_extra_terminal_targets(
        assistant_span_ids=assistant_span_ids,
        y_train_ids=y_train_ids,
        prompt_len=prompt_len,
        candidates=candidates,
    )

    targets = compile_stage2_trie_targets_for_rollout_group(
        candidates,
        label_position_start=label_position_start,
        semantic_role_by_position=semantic_role_by_position,
        extra_token_targets=extra_token_targets,
    )

    meta_entry["stage2_trie_targets"] = targets
    meta_entry["stage2_trie_span_scores"] = [
        stage2_trie_span_score_record_to_json(record)
        for record in targets.span_score_records
    ]
    summary: dict[str, Any] = {
        "sample_id": str(candidates[0].sample_id),
        "candidate_count": int(targets.summary.candidate_count),
        "fallback_candidate_count": int(targets.summary.fallback_candidate_count),
        "fallback_loss_weight_sum": float(targets.summary.fallback_loss_weight_sum),
        "weak_positive_fp_count": int(targets.summary.weak_positive_fp_count),
        "label_position_start": label_position_start,
        "target_positions": int(targets.summary.target_positions),
        "branch_points": int(targets.summary.branch_points),
        "max_branching_factor": int(targets.summary.max_branching_factor),
        "rollout_indices": [
            int(candidate.rollout_index) for candidate in candidates
        ],
    }
    if len(candidates) == 1:
        candidate = candidates[0]
        summary.update(
            {
                "rollout_index": int(candidate.rollout_index),
                "source": str(candidate.source),
                "token_count": len(candidate.token_ids),
                "loss_weight": float(candidate.loss_weight),
            }
        )
    meta_entry["stage2_trie_candidate_summary"] = summary
    if bool(stage2_trie_weak_fp_span_level_fallback):
        meta_entry["stage2_trie_weak_fp_span_level_fallback"] = True


def _build_stage2_trie_semantic_role_map(
    *,
    meta_entry: Mapping[str, Any],
    y_train_ids: Sequence[int],
    prompt_len: int,
    tokenizer: Any | None,
) -> dict[int, str]:
    """Build segment-local token-role annotations for Stage-2 trie CE."""

    role_by_position: dict[int, str] = {}
    prompt_len_i = int(prompt_len)
    prefix_len = int(meta_entry.get("prefix_len", 0) or 0)
    y_train_len = int(len(y_train_ids))

    if str(meta_entry.get("rollout_template_family", "")) == "compact_full":
        for rel in _compact_desc_positions_from_token_ids(
            tokenizer=tokenizer,
            token_ids=y_train_ids,
        ):
            _set_stage2_trie_role(
                role_by_position,
                prompt_len_i + int(rel),
                "desc",
            )

    for rel in meta_entry.get("prefix_desc_pos") or []:
        _set_stage2_trie_role(
            role_by_position,
            prompt_len_i + int(rel),
            "desc",
        )

    for rel in meta_entry.get("tail_desc_pos") or []:
        _set_stage2_trie_role(
            role_by_position,
            prompt_len_i + prefix_len + int(rel),
            "desc",
        )

    for rel in meta_entry.get("prefix_coord_pos") or []:
        _set_stage2_trie_role(
            role_by_position,
            prompt_len_i + int(rel),
            "coord",
        )

    for group in list(meta_entry.get("bbox_groups_prefix") or []) + list(
        meta_entry.get("bbox_groups_fn") or []
    ):
        if not isinstance(group, Mapping):
            continue
        for pos in group.get("pos") or []:
            _set_stage2_trie_role(role_by_position, int(pos), "coord")

    # ...mark remaining compact row tokens as structure after desc/coord roles land.
    for local_index in range(y_train_len):
        role_by_position.setdefault(prompt_len_i + int(local_index), "struct")

    for rel in meta_entry.get("tail_closure_pos") or []:
        _set_stage2_trie_role(
            role_by_position,
            prompt_len_i + prefix_len + int(rel),
            "eos",
        )

    return role_by_position


def _compact_desc_positions_from_token_ids(
    *,
    tokenizer: Any | None,
    token_ids: Sequence[int],
) -> list[int]:
    """Locate compact-full description token positions from encoded rows."""

    if tokenizer is None or not token_ids:
        return []

    token_ids_list = [int(token_id) for token_id in token_ids]
    token_spans = _token_piece_char_spans(tokenizer=tokenizer, token_ids=token_ids_list)
    text = tokenizer.decode(
        token_ids_list,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    positions: set[int] = set()
    for _object_span, desc_span in _compact_object_and_desc_spans(str(text)):
        positions.update(
            _token_indices_overlapping_char_span(
                token_spans=token_spans,
                char_start=int(desc_span[0]),
                char_end=int(desc_span[1]),
            )
        )

    return sorted(int(pos) for pos in positions)


def _build_stage2_trie_extra_terminal_targets(
    *,
    assistant_span_ids: Sequence[int] | None,
    y_train_ids: Sequence[int],
    prompt_len: int,
    candidates: Sequence[Stage2TrieCandidate],
) -> tuple[Stage2TrieTokenTarget, ...]:
    """Build explicit terminal-token supervision after repaired row content."""

    if assistant_span_ids is None:
        return ()

    assistant_ids = [int(token_id) for token_id in assistant_span_ids]
    y_train_len = int(len(y_train_ids))
    if y_train_len >= len(assistant_ids):
        return ()

    terminal_token_id = int(assistant_ids[y_train_len])
    terminal_weight = _stage2_trie_terminal_loss_weight(candidates)
    if terminal_weight <= 0.0:
        return ()

    return (
        Stage2TrieTokenTarget(
            position=int(prompt_len) + y_train_len,
            positive_token_ids=(terminal_token_id,),
            source_weights=(float(terminal_weight),),
            semantic_role="eos",
        ),
    )


def _stage2_trie_terminal_loss_weight(
    candidates: Sequence[Stage2TrieCandidate],
) -> float:
    """Resolve terminal supervision weight without letting fallback dominate."""

    if any(candidate.source == "valid_rollout" for candidate in candidates):
        return 1.0

    fallback_weights = [
        float(candidate.loss_weight)
        for candidate in candidates
        if candidate.source == "fallback_gt_fn_append_only"
    ]
    if fallback_weights:
        return max(fallback_weights)

    return 0.0


def _set_stage2_trie_role(
    role_by_position: MutableMapping[int, str],
    position: int,
    role: str,
) -> None:
    """Set one role using the same precedence as trie target merging."""

    precedence = {"text": 0, "desc": 1, "struct": 2, "coord": 3, "eos": 4}
    current = str(role_by_position.get(int(position), "text"))
    if precedence[str(role)] >= precedence.get(current, 0):
        role_by_position[int(position)] = str(role)


def _build_implicit_stage2_trie_candidates(
    *,
    meta_entry: Mapping[str, Any],
    y_train_ids: Sequence[int],
    sample_id: str,
    rollout_index: int,
    stage2_trie_object_spans: Sequence[Stage2TrieObjectSpan] = (),
) -> list[Stage2TrieCandidate]:
    y_train_ids_list = [int(token_id) for token_id in y_train_ids]
    if not y_train_ids_list:
        return []

    source = (
        "fallback_gt_fn_append_only"
        if str(meta_entry.get("rollout_context", "")) == FALLBACK_GT_FN_APPEND_ONLY
        else "valid_rollout"
    )
    loss_weight = (
        float(meta_entry.get("rollout_fallback_loss_weight", 1.0))
        if source == "fallback_gt_fn_append_only"
        else 1.0
    )
    return [
        Stage2TrieCandidate(
            sample_id=str(sample_id),
            rollout_index=int(rollout_index),
            source=source,
            token_ids=y_train_ids_list,
            loss_weight=float(loss_weight),
            object_spans=list(stage2_trie_object_spans),
        )
    ]


def _serialize_gt_object_entry(
    *,
    obj: GTObject,
    object_field_order: str,
) -> str:
    if str(obj.geom_type) != "bbox_2d":
        raise ValueError(
            f"rollout-correction clean-prefix v1 only supports bbox_2d objects; got {obj.geom_type!r}"
        )
    if len(obj.points_norm1000) != 4:
        raise ValueError(
            "rollout-correction clean-prefix v1 requires bbox_2d objects with four coord bins"
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


def _build_duplicate_control_divergence_diagnostics(
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
) -> Tuple[List[Stage2DuplicateControlDivergenceDiagnostic], int, int]:
    if suppressed_duplicate_objects_by_boundary is None:
        suppressed_duplicate_objects_by_boundary = {}
    diagnostics_by_boundary_token: dict[
        tuple[int, int], Stage2DuplicateControlDivergenceDiagnostic
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
            duplicate_token_id = int(duplicate_target_ids[duplicate_pos])
            candidate: Stage2DuplicateControlDivergenceDiagnostic = {
                "boundary": int(boundary_i),
                "clean_rel_pos": int(rel_pos),
                "duplicate_token_id": int(duplicate_token_id),
            }
            key = (int(boundary_i), int(duplicate_token_id))
            existing = diagnostics_by_boundary_token.get(key)
            if existing is None or int(candidate["clean_rel_pos"]) < int(
                existing["clean_rel_pos"]
            ):
                diagnostics_by_boundary_token[key] = candidate

    diagnostics = sorted(
        diagnostics_by_boundary_token.values(),
        key=lambda item: (
            int(item["boundary"]),
            int(item["clean_rel_pos"]),
            int(item["duplicate_token_id"]),
        ),
    )
    duplicate_control_first_divergence_boundary_count = len(
        {int(item["boundary"]) for item in diagnostics}
    )
    return (
        diagnostics,
        int(duplicate_control_first_divergence_boundary_count),
        int(skipped_no_divergence),
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
            "rollout-correction FN desc spans do not align with fn_object_weights: "
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
    "RolloutCorrectionTargetContext",
    "RolloutCorrectionTargetContextInput",
    "construct_detection_scene_rollout_correction_target_context",
    "construct_rollout_correction_target_context",
    "_ValueSpanObject",
    "_CanonicalPrefixData",
    "_ChannelBTriageResult",
    "_ChannelBSupervisionTargets",
    "_build_rollout_correction_triage",
    "_build_rollout_correction_supervision_targets",
    "_build_residual_set_correction_events",
    "_shift_residual_correction_events",
    "_build_rollout_correction_meta_entry",
    "_rollout_correction_residual_set_enabled",
    "_rollout_correction_residual_set_options",
    "_bbox_iou_norm1000_xyxy",
    "_apply_rollout_correction_duplicate_control",
    "_compute_duplicate_diagnostics",
    "_build_canonical_prefix_data",
    "_build_duplicate_control_divergence_diagnostics",
    "_desc_tail_positions_and_weights",
]
