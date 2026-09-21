"""Loss context and protected V1 token-wise losses."""

from src.losses.base_ce import BaseTokenCE
from src.losses.context import LossContext
from src.losses.coord_gaussian_rps import (
    CoordGaussianRPSLoss,
    gaussian_soft_targets_from_r95,
    ranked_probability_score,
)
from src.losses.normalizers import (
    PlannedStepLossSlice,
    SegmentBalancedDenominator,
    SegmentBalancedLossResult,
    build_segment_balanced_denominator,
    reduce_segment_balanced_planned_step,
    segment_balanced_contribution,
)
from src.losses.runner import LossBundle, LossRunner, LossTermResult
from src.losses.rollout_calibration import (
    CandidatePath,
    CandidatePathScore,
    CoordinateBoundaryPreferenceResult,
    EntityTransitionPreferenceResult,
    EventGateResult,
    FieldBalancedDuplicateRejectionResult,
    GateSiteIdentity,
    LossFiniteDiagnostics,
    OwnerConditionedCandidateLossResult,
    OwnerScore,
    PositivePathImitationResult,
    RolloutGateSite,
    RolloutSiteTokenTypeGateResult,
    first_wrong_coordinate_preference,
    field_balanced_duplicate_rejection_loss,
    grouped_entity_transition_preference,
    owner_conditioned_candidate_loss,
    positive_path_imitation,
    positive_path_imitation_loss,
    rollout_site_token_type_gate,
)
from src.losses.token_type_gate import TokenTypeGateLoss
from src.losses.token_scores import aligned_token_logprobs
from src.losses.vocab import (
    KNOWN_CONTROL_TOKENS,
    V1_TOKEN_TYPES,
    TokenType,
    TokenVocabularyGroups,
    build_token_vocabulary_groups,
)

__all__ = [
    "BaseTokenCE",
    "CandidatePath",
    "CandidatePathScore",
    "CoordGaussianRPSLoss",
    "CoordinateBoundaryPreferenceResult",
    "EntityTransitionPreferenceResult",
    "EventGateResult",
    "FieldBalancedDuplicateRejectionResult",
    "GateSiteIdentity",
    "KNOWN_CONTROL_TOKENS",
    "LossContext",
    "LossBundle",
    "LossRunner",
    "LossTermResult",
    "LossFiniteDiagnostics",
    "OwnerConditionedCandidateLossResult",
    "OwnerScore",
    "PositivePathImitationResult",
    "PlannedStepLossSlice",
    "SegmentBalancedDenominator",
    "SegmentBalancedLossResult",
    "RolloutGateSite",
    "RolloutSiteTokenTypeGateResult",
    "TokenType",
    "TokenTypeGateLoss",
    "TokenVocabularyGroups",
    "V1_TOKEN_TYPES",
    "build_segment_balanced_denominator",
    "build_token_vocabulary_groups",
    "first_wrong_coordinate_preference",
    "field_balanced_duplicate_rejection_loss",
    "gaussian_soft_targets_from_r95",
    "grouped_entity_transition_preference",
    "owner_conditioned_candidate_loss",
    "positive_path_imitation",
    "positive_path_imitation_loss",
    "ranked_probability_score",
    "reduce_segment_balanced_planned_step",
    "rollout_site_token_type_gate",
    "segment_balanced_contribution",
    "aligned_token_logprobs",
]
