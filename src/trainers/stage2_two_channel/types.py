from typing import TYPE_CHECKING, Any, Dict, List, Literal, NotRequired, TypeAlias, TypedDict

if TYPE_CHECKING:
    from .trie_supervision import Stage2TrieTargets


class Stage2BBoxGroup(TypedDict):
    pos: List[int]
    gt_bins: List[int]
    weight: NotRequired[float]


class Stage2DuplicateControlDivergenceDiagnostic(TypedDict):
    boundary: int
    clean_rel_pos: int
    duplicate_token_id: int


class Stage2RolloutMetaBase(TypedDict):
    stage2_channel: Literal["A", "B"]
    prompt_len: int
    prompt_ids: List[int]
    rollout_len: int
    prefix_len: int
    train_len: int
    encoded_len: int
    decode_mode: str
    parse_dropped_invalid: int
    parse_dropped_ambiguous: int
    parse_truncated: bool
    valid_pred_objects: int
    matched_for_supervision: int
    matched_maskiou_sum: float
    matched_maskiou_count: int
    gt_objects: int
    fn_count: int
    gating_rejections: int
    excluded_from_supervision: int
    prefix_coord_pos: List[int]
    prefix_coord_target_bins: List[int]
    tail_closure_pos: List[int]
    tail_ignore_pos: List[int]
    tail_desc_pos: List[int]
    stop_rel_pos: int | None
    stop_token_id: int | None
    continue_token_id: int | None
    bbox_groups_prefix: List[Stage2BBoxGroup]
    bbox_groups_fn: List[Stage2BBoxGroup]


class Stage2ChannelAMeta(Stage2RolloutMetaBase):
    stage2_channel: Literal["A"]


class Stage2ChannelBMeta(Stage2RolloutMetaBase):
    stage2_channel: Literal["B"]
    stage2_invalid_rollout: int
    rollout_seed_base: int
    rollout_template_family: str
    rollout_parser_id: str
    rollout_append_policy_id: str
    rollout_context: str
    rollout_fallback_reason: str | None
    rollout_fallback_loss_weight: float
    rollout_counts_as_valid_rollout: bool
    drop_invalid_total: int
    valid_explorer_count: int
    prefix_struct_pos: List[int]
    prefix_desc_pos: NotRequired[List[int]]
    prefix_desc_weights: NotRequired[List[float]]
    tail_desc_weights: List[float]
    fn_object_weights: List[float]
    anchor_decode_mode: str
    explorer_decode_mode: str
    duplicate_clusters_total: int
    duplicate_clusters_exempt: int
    duplicate_clusters_suppressed: int
    duplicate_objects_suppressed: int
    duplicate_survivor_anchor_indices: List[int]
    duplicate_exempt_anchor_indices: List[int]
    duplicate_suppressed_anchor_indices: List[int]
    anchor_gt_backed_indices: List[int]
    anchor_support_counts: List[int]
    anchor_support_rates: List[float]
    shielded_anchor_indices: List[int]
    dead_anchor_indices: List[int]
    lvis_verified_positive_dead_anchor_indices: List[int]
    lvis_verified_negative_dead_anchor_indices: List[int]
    lvis_not_exhaustive_anchor_indices: List[int]
    lvis_unevaluable_anchor_indices: List[int]
    pseudo_positive_anchor_indices: List[int]
    dead_explorer_indices_by_view: List[List[int]]
    recovered_gt_indices: List[int]
    recovered_gt_support_counts: List[int]
    recovered_gt_support_rates: List[float]
    duplicate_control_first_divergence_diagnostics: List[Stage2DuplicateControlDivergenceDiagnostic]
    duplicate_control_first_divergence_boundary_count: int
    duplicate_control_first_divergence_skipped_no_divergence: int
    assignment_strategy: str
    assignment_iou_threshold: float
    stage2_trie_targets: NotRequired["Stage2TrieTargets"]
    stage2_trie_span_scores: NotRequired[List[Dict[str, Any]]]
    stage2_trie_candidate_summary: NotRequired[Dict[str, Any]]
    stage2_trie_skip_loss: NotRequired[bool]
    stage2_trie_weak_fp_span_level_fallback: NotRequired[bool]


Stage2RolloutMeta: TypeAlias = Stage2ChannelAMeta | Stage2ChannelBMeta
Stage2EncodedSample: TypeAlias = Dict[str, Any]
Stage2PreparedSegment: TypeAlias = tuple[Stage2EncodedSample, Stage2RolloutMeta, int]
Stage2BatchMetrics: TypeAlias = Dict[str, float]


__all__ = [
    "Stage2BBoxGroup",
    "Stage2DuplicateControlDivergenceDiagnostic",
    "Stage2RolloutMetaBase",
    "Stage2ChannelAMeta",
    "Stage2ChannelBMeta",
    "Stage2RolloutMeta",
    "Stage2EncodedSample",
    "Stage2PreparedSegment",
    "Stage2BatchMetrics",
]
