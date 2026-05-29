from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from src.training.supervision.context import (
    SemanticMetadata,
    freeze_semantic_metadata,
    validate_optional_semantic_string,
    validate_required_semantic_string,
)
from src.training.supervision.distributions import (
    TargetDistribution,
    TargetDistributionRegistry,
)


SupervisionSpanRole: TypeAlias = Literal[
    "schema",
    "free_text",
    "coordinate",
    "object_boundary",
]

VALID_SUPERVISION_SPAN_ROLES: frozenset[str] = frozenset(
    (
        "schema",
        "free_text",
        "coordinate",
        "object_boundary",
    )
)


def validate_supervision_span_role(role: object) -> SupervisionSpanRole:
    """Return a validated supervision span role.

    :param role: Candidate span role.
    :returns: Supported supervision span role.
    :raises TypeError: If the role is not a plain string.
    :raises ValueError: If the role is unsupported.
    """

    if type(role) is not str:
        raise TypeError("supervision span role must be a semantic string")
    if role not in VALID_SUPERVISION_SPAN_ROLES:
        raise ValueError(f"unsupported supervision span role: {role!r}")

    return cast(SupervisionSpanRole, role)


def _normalize_label_positions(label_positions: object) -> tuple[int, ...]:
    """Return validated target-token label positions.

    These positions name target tokens in the encoded label sequence. They are
    not shifted logit rows; that mapping belongs to a future bridge layer.

    :param label_positions: Candidate target-token label positions.
    :returns: Immutable tuple of non-negative integer label positions.
    :raises TypeError: If positions are not integer sequence entries.
    :raises ValueError: If the sequence is empty or contains a negative value.
    """

    if isinstance(label_positions, (str, bytes, Mapping)) or not isinstance(
        label_positions,
        Sequence,
    ):
        raise TypeError("label positions must be a sequence of non-negative integers")

    normalized: list[int] = []
    for label_position in label_positions:
        if type(label_position) is not int:
            raise TypeError(
                "label positions must be a sequence of non-negative integers"
            )
        if label_position < 0:
            raise ValueError("label positions must be non-negative")
        normalized.append(label_position)

    if len(normalized) == 0:
        raise ValueError("span must contain at least one label position")

    return tuple(normalized)


def _normalize_optional_position(value: object | None, *, field_name: str) -> int | None:
    """Return a validated optional non-negative position.

    :param value: Candidate optional position.
    :param field_name: Field name for error reporting.
    :returns: Non-negative integer position or ``None``.
    :raises TypeError: If the value is not ``None`` or an integer.
    :raises ValueError: If the value is negative.
    """

    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a non-negative integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")

    return value


@dataclass(frozen=True, slots=True)
class SupervisionSpan:
    """Semantic span carrying target-token label positions.

    ``label_positions`` are target-token positions in the label sequence, not
    shifted logit rows. The future ``PredictionCoordinateMapper`` owns any
    label-position to logit-position mapping.

    :param sample_id: Stable per-example sample identifier.
    :param role: Semantic role for this span.
    :param label_positions: Non-empty target-token label positions.
    :param distribution: Registered semantic target distribution.
    :param context_id: Optional semantic context identifier.
    :param start: Optional inclusive semantic span boundary.
    :param end: Optional exclusive semantic span boundary.
    :param provenance: Optional semantic source label.
    :param metadata: Optional scalar semantic metadata.
    """

    sample_id: str
    role: SupervisionSpanRole
    label_positions: Sequence[int]
    distribution: TargetDistribution
    context_id: str | None = None
    start: int | None = None
    end: int | None = None
    provenance: str | None = None
    metadata: SemanticMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate local semantic span structure."""

        if type(self.distribution) not in (
            TargetDistributionRegistry.registered_distribution_classes()
        ):
            raise TypeError(
                "supervision span distribution must be a registered TargetDistribution"
            )

        start = _normalize_optional_position(self.start, field_name="start")
        end = _normalize_optional_position(self.end, field_name="end")
        if (start is None) != (end is None):
            raise ValueError("span start and end must both be present or both absent")
        if start is not None and end is not None and end <= start:
            raise ValueError("span end must be greater than start")

        object.__setattr__(
            self,
            "sample_id",
            validate_required_semantic_string(self.sample_id, field_name="sample_id"),
        )
        object.__setattr__(
            self,
            "context_id",
            validate_optional_semantic_string(
                self.context_id,
                field_name="context_id",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            validate_optional_semantic_string(
                self.provenance,
                field_name="provenance",
            ),
        )
        object.__setattr__(self, "role", validate_supervision_span_role(self.role))
        object.__setattr__(
            self,
            "label_positions",
            _normalize_label_positions(self.label_positions),
        )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "metadata", freeze_semantic_metadata(self.metadata))
