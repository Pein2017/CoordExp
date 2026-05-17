"""Stage-2 compact-full span adapter seam."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.training.encoding.view import EncodedDetectionView
from src.training.span_adapters.compact_projector import (
    CompactFullSpanProjector,
    CompactSpanProjection,
    require_projected_role,
)
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.context import (
    SemanticMetadata,
    validate_optional_semantic_string,
    validate_required_semantic_string,
)
from src.training.supervision.distributions import (
    MultiPositiveTokenDistribution,
    MultiPositiveWeightInput,
)
from src.training.supervision.spans import SupervisionSpan


Stage2CompactChannel = Literal["channel_a", "channel_b"]


@dataclass(frozen=True, slots=True)
class Stage2SpanProvenance:
    """Scalar provenance carried by a Stage-2 compact supervision span.

    :param source_role: Semantic source role, such as matched GT or FN append.
    :param assignment_id: Optional assignment provenance identifier.
    :param duplicate_filter_id: Optional duplicate-filter provenance identifier.
    :param false_negative: Whether the span came from a false-negative path.
    :param source_object_instance_id: Optional source object identifier.
    :param state_weight: Optional objective state denominator weight.
    :param loss_weight: Optional objective numerator multiplier.
    :param support_weight: Optional trie support-term multiplier.
    :param balance_weight: Optional trie balance-term multiplier.
    """

    source_role: str
    assignment_id: str | None = None
    duplicate_filter_id: str | None = None
    false_negative: bool = False
    source_object_instance_id: str | None = None
    state_weight: float | None = None
    loss_weight: float | None = None
    support_weight: float | None = None
    balance_weight: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar semantic provenance fields."""

        # validate semantic strings and bool flags before metadata conversion.
        object.__setattr__(
            self,
            "source_role",
            validate_required_semantic_string(
                self.source_role,
                field_name="source_role",
            ),
        )
        object.__setattr__(
            self,
            "assignment_id",
            validate_optional_semantic_string(
                self.assignment_id,
                field_name="assignment_id",
            ),
        )
        object.__setattr__(
            self,
            "duplicate_filter_id",
            validate_optional_semantic_string(
                self.duplicate_filter_id,
                field_name="duplicate_filter_id",
            ),
        )
        if type(self.false_negative) is not bool:
            raise TypeError("false_negative must be a bool")
        object.__setattr__(
            self,
            "source_object_instance_id",
            validate_optional_semantic_string(
                self.source_object_instance_id,
                field_name="source_object_instance_id",
            ),
        )
        for field_name in (
            "state_weight",
            "loss_weight",
            "support_weight",
            "balance_weight",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_optional_objective_weight(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

    def to_metadata(self, *, channel: Stage2CompactChannel) -> SemanticMetadata:
        """Return scalar metadata for a supervision span."""

        metadata = {
            "channel": channel,
            "source_role": self.source_role,
            "assignment_id": self.assignment_id,
            "duplicate_filter_id": self.duplicate_filter_id,
            "false_negative": self.false_negative,
            "source_object_instance_id": self.source_object_instance_id,
        }
        for field_name in (
            "state_weight",
            "loss_weight",
            "support_weight",
            "balance_weight",
        ):
            value = getattr(self, field_name)
            if value is not None:
                metadata[field_name] = value

        return metadata


@dataclass(frozen=True, slots=True)
class Stage2CompactTokenTargetSpec:
    """Stage-2 multi-positive target at one compact label position.

    :param label_position: Target-token label position to supervise.
    :param token_ids: Positive token ids accepted at the target position.
    :param token_weights: Optional positive multiplicity weights for token ids.
    :param channel: Stage-2 channel ownership for this target.
    :param provenance: Scalar Stage-2 provenance metadata.
    :param span_provenance: Optional semantic source label for the span.
    """

    label_position: int
    token_ids: Sequence[int]
    channel: Stage2CompactChannel
    provenance: Stage2SpanProvenance
    span_provenance: str | None = None
    token_weights: MultiPositiveWeightInput = None

    def __post_init__(self) -> None:
        """Validate Stage-2 target spec fields."""

        # validate target-token identity and channel ownership.
        if type(self.label_position) is not int:
            raise TypeError("label position must be an integer")
        if self.label_position < 0:
            raise ValueError("label position must be non-negative")
        if self.channel not in ("channel_a", "channel_b"):
            raise ValueError("Stage-2 compact channel must be channel_a or channel_b")
        if type(self.provenance) is not Stage2SpanProvenance:
            raise TypeError("provenance must be Stage2SpanProvenance")

        # reuse the distribution contract for token-id normalization.
        distribution = MultiPositiveTokenDistribution(
            self.token_ids,
            token_weights=self.token_weights,
        )
        object.__setattr__(self, "token_ids", distribution.token_ids)
        object.__setattr__(self, "token_weights", distribution.token_weights)
        object.__setattr__(
            self,
            "span_provenance",
            validate_optional_semantic_string(
                self.span_provenance,
                field_name="span_provenance",
            ),
        )


class Stage2CompactSpanAdapter:
    """Build semantic Stage-2 spans from compact-full target specs."""

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
        targets: Sequence[Stage2CompactTokenTargetSpec],
        context_id: str | None = None,
        batch_id: str | None = None,
        metadata: SemanticMetadata | None = None,
    ) -> SupervisionBatch:
        """Build a semantic supervision batch for Stage-2 compact targets."""

        # resolve the compact projection boundary.
        compact_projection = self._coerce_projection(projection)

        # convert channel target specs to supervision spans without logit shifting.
        spans: list[SupervisionSpan] = []
        for target in targets:
            self._validate_target(target, compact_projection)
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
                    provenance=target.span_provenance,
                    metadata=target.provenance.to_metadata(channel=target.channel),
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

    def _validate_target(
        self,
        target: Stage2CompactTokenTargetSpec,
        projection: CompactSpanProjection,
    ) -> None:
        """Validate a Stage-2 target against the compact projection."""

        # require explicit supervised target-token positions only.
        if type(target) is not Stage2CompactTokenTargetSpec:
            raise TypeError("targets must be Stage2CompactTokenTargetSpec")
        if not projection.contains_label_position(target.label_position):
            raise ValueError("Stage-2 target label position is not supervised")

        require_projected_role(projection, target.label_position)


def _validate_optional_objective_weight(
    value: object,
    *,
    field_name: str,
) -> float | None:
    """Return an optional finite non-negative objective metadata weight."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be >= 0")

    return parsed
