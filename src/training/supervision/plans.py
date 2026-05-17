from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from src.training.supervision.context import (
    SemanticMetadata,
    SupervisionChannel,
    SupervisionStage,
    freeze_semantic_metadata,
    validate_optional_semantic_string,
    validate_required_semantic_string,
    validate_stage_channel_pair,
    validate_supervision_channel,
    validate_supervision_stage,
)

SemanticBBoxInput: TypeAlias = Sequence[int | float] | None
SemanticBBox: TypeAlias = tuple[float, float, float, float]


def _normalize_bbox(
    bbox: SemanticBBoxInput,
) -> SemanticBBox | None:
    """Return a validated immutable bounding box.

    :param bbox: Optional candidate bounding box.
    :returns: Four-value floating point bounding box or ``None``.
    :raises TypeError: If the bounding box is not a sequence of numbers.
    :raises ValueError: If the bounding box does not contain four values.
    """

    if bbox is None:
        return None

    if isinstance(bbox, (str, bytes, Mapping)):
        raise TypeError("semantic bbox must be a four-value numeric sequence")

    if not isinstance(bbox, Sequence):
        raise TypeError("semantic bbox must be a four-value numeric sequence")

    if len(bbox) != 4:
        raise ValueError("semantic bbox must contain exactly four values")

    values: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("semantic bbox values must be finite numeric scalars")

        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("semantic bbox values must be finite")
        values.append(coordinate)

    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True, slots=True)
class SupervisionObject:
    """Semantic object entry owned by a per-example supervision plan.

    :param object_id: Stable object identity within the planned example.
    :param description: Semantic text description before any rendering.
    :param bbox: Optional finite numeric bounding box. Values are stored as an
        immutable four-float tuple; coordinate orientation is validated by later
        geometry-specific layers.
    :param provenance: Optional semantic source label for this object.
    :param metadata: Optional scalar semantic metadata.
    """

    object_id: str
    description: str
    bbox: SemanticBBoxInput = None
    provenance: str | None = None
    metadata: SemanticMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize geometry and freeze scalar metadata."""

        object.__setattr__(
            self,
            "object_id",
            validate_required_semantic_string(self.object_id, field_name="object_id"),
        )
        object.__setattr__(
            self,
            "description",
            validate_required_semantic_string(
                self.description,
                field_name="description",
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
        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(self, "metadata", freeze_semantic_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SupervisionPlan:
    """Semantic per-example supervision plan for Stage-1 and Stage-2.

    :param sample_id: Stable per-example sample identifier.
    :param stage: Training stage that owns this plan.
    :param template_id: Template family selected before rendering.
    :param objects: Semantic object entries for this example and channel.
    :param channel: Supervision channel ownership for this per-example plan.
    :param provenance: Optional semantic source label for this plan.
    :param context_id: Optional identifier linking to a ``SupervisionContext``.
    :param metadata: Optional scalar semantic metadata.
    """

    sample_id: str
    stage: SupervisionStage
    template_id: str
    objects: tuple[SupervisionObject, ...] = field(default_factory=tuple)
    channel: SupervisionChannel = "primary"
    provenance: str | None = None
    context_id: str | None = None
    metadata: SemanticMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate semantic ownership and freeze scalar metadata."""

        objects = tuple(self.objects)
        for supervision_object in objects:
            if type(supervision_object) is not SupervisionObject:
                raise TypeError(
                    "supervision plan objects must be SupervisionObject instances"
                )

        object.__setattr__(
            self,
            "sample_id",
            validate_required_semantic_string(self.sample_id, field_name="sample_id"),
        )
        object.__setattr__(
            self,
            "template_id",
            validate_required_semantic_string(self.template_id, field_name="template_id"),
        )
        object.__setattr__(
            self,
            "provenance",
            validate_optional_semantic_string(
                self.provenance,
                field_name="provenance",
            ),
        )
        object.__setattr__(
            self,
            "context_id",
            validate_optional_semantic_string(
                self.context_id,
                field_name="context_id",
            ),
        )
        stage = validate_supervision_stage(self.stage)
        channel = validate_supervision_channel(self.channel)
        validate_stage_channel_pair(stage=stage, channel=channel)

        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "metadata", freeze_semantic_metadata(self.metadata))
