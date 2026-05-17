"""Stage-1 compact-full span adapter seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.training.encoding.view import EncodedDetectionView
from src.training.span_adapters.compact_projector import (
    CompactFullSpanProjector,
    CompactSpanProjection,
    require_projected_role,
)
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.context import SemanticMetadata
from src.training.supervision.distributions import (
    CoordinateSoftLossMode,
    CoordinateSoftTargetFamily,
    CoordinateSoftTokenDistribution,
    CoordinateSoftTokenWeight,
    MultiPositiveWeightInput,
    MultiPositiveTokenDistribution,
)
from src.training.supervision.spans import SupervisionSpan


@dataclass(frozen=True, slots=True)
class CompactTrieTargetSpec:
    """Stage-1 multi-positive trie target at one compact label position.

    :param label_position: Target-token label position to supervise.
    :param token_ids: Positive token ids accepted at the target position.
    :param token_weights: Optional positive multiplicity weights for token ids.
    """

    label_position: int
    token_ids: Sequence[int]
    token_weights: MultiPositiveWeightInput = None

    def __post_init__(self) -> None:
        """Validate the target position and normalize token ids."""

        # validate target-token position identity.
        object.__setattr__(
            self,
            "label_position",
            _validate_label_position(self.label_position),
        )

        # reuse the distribution contract for token-id normalization.
        distribution = MultiPositiveTokenDistribution(
            self.token_ids,
            token_weights=self.token_weights,
        )
        object.__setattr__(self, "token_ids", distribution.token_ids)
        object.__setattr__(self, "token_weights", distribution.token_weights)


@dataclass(frozen=True, slots=True)
class CompactCoordinateTokenWeightSpec:
    """Weighted coordinate-token candidate for a soft coordinate target.

    :param token_id: Coordinate token id.
    :param weight: Non-negative finite target weight.
    """

    token_id: int
    weight: float

    def to_distribution_weight(self) -> CoordinateSoftTokenWeight:
        """Return the supervision distribution weight for this spec."""

        return CoordinateSoftTokenWeight(token_id=self.token_id, weight=self.weight)


@dataclass(frozen=True, slots=True)
class CompactCoordinateSoftTargetSpec:
    """Stage-1 soft coordinate target at one compact coordinate position.

    :param label_position: Target-token coordinate label position to supervise.
    :param token_weights: Soft coordinate target weights.
    :param target_family: Semantic target recipe identifier.
    :param loss_mode: Semantic coordinate soft-loss mode.
    """

    label_position: int
    token_weights: Sequence[CompactCoordinateTokenWeightSpec]
    target_family: CoordinateSoftTargetFamily = "iou_gibbs_v0"
    loss_mode: CoordinateSoftLossMode = "full_vocab_ce"

    def __post_init__(self) -> None:
        """Validate the target position and normalize token weights."""

        # validate target-token position identity.
        object.__setattr__(
            self,
            "label_position",
            _validate_label_position(self.label_position),
        )

        # preserve typed coordinate weight specs before distribution construction.
        if isinstance(self.token_weights, (str, bytes, Mapping)) or not isinstance(
            self.token_weights,
            Sequence,
        ):
            raise TypeError("coordinate target weights must be a sequence")
        token_weights = tuple(self.token_weights)
        for token_weight in token_weights:
            if type(token_weight) is not CompactCoordinateTokenWeightSpec:
                raise TypeError(
                    "coordinate target weights must be CompactCoordinateTokenWeightSpec"
                )
        CoordinateSoftTokenDistribution(
            tuple(weight.to_distribution_weight() for weight in token_weights),
            target_family=self.target_family,
            loss_mode=self.loss_mode,
        )
        object.__setattr__(self, "token_weights", token_weights)


class Stage1CompactSpanAdapter:
    """Build semantic Stage-1 spans from compact-full target specs."""

    def __init__(self, projector: CompactFullSpanProjector | None = None) -> None:
        """Initialize the adapter.

        :param projector: Optional compact-full projector override.
        """

        self._projector = projector or CompactFullSpanProjector()

    def build_batch(
        self,
        *,
        sample_id: str,
        projection: CompactSpanProjection | EncodedDetectionView,
        trie_targets: Sequence[CompactTrieTargetSpec] = (),
        coordinate_targets: Sequence[CompactCoordinateSoftTargetSpec] = (),
        context_id: str | None = None,
        provenance: str | None = None,
        batch_id: str | None = None,
        metadata: SemanticMetadata | None = None,
    ) -> SupervisionBatch:
        """Build a semantic supervision batch for Stage-1 compact targets."""

        # resolve the compact projection boundary.
        compact_projection = self._coerce_projection(projection)

        # convert typed specs to supervision spans without logit shifting.
        spans: list[SupervisionSpan] = []
        for target in trie_targets:
            self._validate_trie_target(target, compact_projection)
            spans.append(
                SupervisionSpan(
                    sample_id=sample_id,
                    role=require_projected_role(
                        compact_projection,
                        target.label_position,
                    ),
                    label_positions=(target.label_position,),
                    distribution=MultiPositiveTokenDistribution(
                        target.token_ids,
                        token_weights=target.token_weights,
                    ),
                    context_id=context_id,
                    start=target.label_position,
                    end=target.label_position + 1,
                    provenance=provenance,
                )
            )

        for target in coordinate_targets:
            self._validate_coordinate_target(target, compact_projection)
            spans.append(
                SupervisionSpan(
                    sample_id=sample_id,
                    role=require_projected_role(
                        compact_projection,
                        target.label_position,
                    ),
                    label_positions=(target.label_position,),
                    distribution=CoordinateSoftTokenDistribution(
                        tuple(
                            weight.to_distribution_weight()
                            for weight in target.token_weights
                        ),
                        target_family=target.target_family,
                        loss_mode=target.loss_mode,
                    ),
                    context_id=context_id,
                    start=target.label_position,
                    end=target.label_position + 1,
                    provenance=provenance,
                )
            )

        return SupervisionBatch(
            spans=tuple(spans),
            batch_id=batch_id,
            metadata=metadata or {},
        )

    def _coerce_projection(
        self,
        projection: CompactSpanProjection | EncodedDetectionView,
    ) -> CompactSpanProjection:
        """Return a compact projection from an encoded view or projection."""

        # accept either boundary object to keep the seam convenient for callers.
        if type(projection) is CompactSpanProjection:
            return projection
        if type(projection) is EncodedDetectionView:
            return self._projector.project(projection)

        raise TypeError(
            "projection must be CompactSpanProjection or EncodedDetectionView"
        )

    def _validate_trie_target(
        self,
        target: CompactTrieTargetSpec,
        projection: CompactSpanProjection,
    ) -> None:
        """Validate a trie target against the compact projection."""

        # require explicit supervised target-token positions only.
        if type(target) is not CompactTrieTargetSpec:
            raise TypeError("trie targets must be CompactTrieTargetSpec")
        if not projection.contains_label_position(target.label_position):
            raise ValueError("trie target label position is not supervised")

        require_projected_role(projection, target.label_position)

    def _validate_coordinate_target(
        self,
        target: CompactCoordinateSoftTargetSpec,
        projection: CompactSpanProjection,
    ) -> None:
        """Validate a coordinate target against the compact projection."""

        # require coordinate targets to land on projected coordinate slots.
        if type(target) is not CompactCoordinateSoftTargetSpec:
            raise TypeError(
                "coordinate targets must be CompactCoordinateSoftTargetSpec"
            )
        if not projection.contains_label_position(target.label_position):
            raise ValueError("coordinate target label position is not supervised")
        if not projection.is_coordinate_position(target.label_position):
            raise ValueError("coordinate target label position is not coordinate")


def _validate_label_position(label_position: object) -> int:
    """Return a validated non-negative target-token label position."""

    if type(label_position) is not int:
        raise TypeError("label position must be an integer")
    if label_position < 0:
        raise ValueError("label position must be non-negative")

    return label_position
