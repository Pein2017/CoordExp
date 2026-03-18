from typing import Any, Dict, List, Literal, NotRequired, TypeAlias, TypedDict


class Stage2BBoxGroup(TypedDict):
    pos: List[int]
    gt_bins: List[int]
    weight: NotRequired[float]


class Stage2DeadAnchorSuppressionTarget(TypedDict):
    boundary: int
    rel_pos: int
    token_id: int


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
    drop_invalid_total: int
    prefix_struct_pos: List[int]
    tail_desc_weights: List[float]
    fn_object_weights: List[float]
    anchor_decode_mode: str
    explorer_decode_mode: str
    anchor_gt_backed_indices: List[int]
    shielded_anchor_indices: List[int]
    dead_anchor_indices: List[int]
    dead_explorer_indices: List[int]
    recovered_gt_indices: List[int]
    dead_anchor_suppression_targets: List[Stage2DeadAnchorSuppressionTarget]
    dead_anchor_suppression_boundary_count: int
    dead_anchor_suppression_skipped_no_divergence: int


Stage2RolloutMeta: TypeAlias = Stage2ChannelAMeta | Stage2ChannelBMeta
Stage2EncodedSample: TypeAlias = Dict[str, Any]
Stage2PreparedSegment: TypeAlias = tuple[Stage2EncodedSample, Stage2RolloutMeta, int]
Stage2BatchMetrics: TypeAlias = Dict[str, float]


__all__ = [
    "Stage2BBoxGroup",
    "Stage2DeadAnchorSuppressionTarget",
    "Stage2RolloutMetaBase",
    "Stage2ChannelAMeta",
    "Stage2ChannelBMeta",
    "Stage2RolloutMeta",
    "Stage2EncodedSample",
    "Stage2PreparedSegment",
    "Stage2BatchMetrics",
]
