from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias, TypeVar, cast


ObjectBBox: TypeAlias = tuple[float, float, float, float]
SemanticScalar: TypeAlias = str | int | float | bool | None


def _normalize_bbox(bbox: Sequence[int | float]) -> ObjectBBox:
    """Return a finite non-degenerate ``xyxy`` bounding box.

    :param bbox: Candidate four-value bounding box.
    :returns: Normalized four-float bounding box.
    :raises TypeError: If the box is not a numeric sequence.
    :raises ValueError: If the box is not finite or non-degenerate.
    """

    if isinstance(bbox, (str, bytes, Mapping)):
        raise TypeError("bbox must be a four-value numeric sequence")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")

    values: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("bbox values must be finite numeric scalars")
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise ValueError("bbox values must be finite")
        values.append(coordinate)

    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox must be non-degenerate xyxy coordinates")

    return (values[0], values[1], values[2], values[3])


def _freeze_metadata(
    metadata: Mapping[str, SemanticScalar] | None,
) -> Mapping[str, SemanticScalar]:
    """Return immutable scalar planning metadata."""

    if metadata is None:
        return cast(Mapping[str, SemanticScalar], MappingProxyType({}))

    frozen: dict[str, SemanticScalar] = {}
    for key, value in metadata.items():
        if type(key) is not str:
            raise TypeError("metadata keys must be strings")
        if (
            value is not None
            and type(value) is not str
            and type(value) is not int
            and type(value) is not float
            and type(value) is not bool
        ):
            raise TypeError("metadata values must be scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("metadata float values must be finite")
        frozen[key] = value

    return cast(Mapping[str, SemanticScalar], MappingProxyType(frozen))


class SupportsPlanningObject(Protocol):
    """Protocol for objects that can participate in ordering strategies."""

    object_id: str
    bbox: ObjectBBox


PlanningObjectT = TypeVar("PlanningObjectT", bound=SupportsPlanningObject)


@dataclass(frozen=True, slots=True)
class PlanningObject:
    """Small semantic object used by ordering-only tests and adapters.

    :param object_id: Stable object identity.
    :param bbox: Finite non-degenerate ``xyxy`` bounding box.
    :param metadata: Optional scalar provenance metadata.
    """

    object_id: str
    bbox: Sequence[int | float]
    metadata: Mapping[str, SemanticScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stable identity, geometry, and metadata."""

        if type(self.object_id) is not str:
            raise TypeError("object_id must be a string")
        if self.object_id == "":
            raise ValueError("object_id must not be empty")

        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ObjectOrderingStrategy(ABC):
    """Strategy for ordering accepted and false-negative planning objects."""

    strategy_id: str

    @abstractmethod
    def order(
        self,
        *,
        accepted_objects: Sequence[PlanningObjectT],
        false_negative_objects: Sequence[PlanningObjectT],
    ) -> tuple[PlanningObjectT, ...]:
        """Return ordered objects for a per-example supervision plan."""


class LegacyTailAppendOrdering(ObjectOrderingStrategy):
    """Ordering that preserves accepted rollout order before FN append order."""

    strategy_id = "legacy_tail_append"

    def order(
        self,
        *,
        accepted_objects: Sequence[PlanningObjectT],
        false_negative_objects: Sequence[PlanningObjectT],
    ) -> tuple[PlanningObjectT, ...]:
        """Return accepted objects followed by false-negative objects."""

        return tuple(accepted_objects) + tuple(false_negative_objects)


class TopLeftSpatialOrdering(ObjectOrderingStrategy):
    """Ordering that sorts all objects by top-left spatial position."""

    strategy_id = "top_left_spatial"

    def order(
        self,
        *,
        accepted_objects: Sequence[PlanningObjectT],
        false_negative_objects: Sequence[PlanningObjectT],
    ) -> tuple[PlanningObjectT, ...]:
        """Return all objects sorted by ``(min_y, min_x)`` with stable ties."""

        combined = tuple(accepted_objects) + tuple(false_negative_objects)

        return tuple(
            item
            for _, item in sorted(
                enumerate(combined),
                key=lambda indexed: (
                    float(indexed[1].bbox[1]),
                    float(indexed[1].bbox[0]),
                    int(indexed[0]),
                    str(indexed[1].object_id),
                ),
            )
        )


def resolve_object_ordering_strategy(mode: str) -> ObjectOrderingStrategy:
    """Return the ordering strategy for a config-compatible mode name.

    :param mode: Ordering mode name.
    :returns: Concrete object ordering strategy.
    :raises ValueError: If the mode is unsupported.
    """

    if mode in {"tail_append", "tail_append_legacy", "legacy_tail_append"}:
        return LegacyTailAppendOrdering()
    if mode in {"sorted", "top_left", "top_left_spatial"}:
        return TopLeftSpatialOrdering()

    raise ValueError(f"unsupported object ordering: {mode!r}")


__all__ = [
    "LegacyTailAppendOrdering",
    "ObjectBBox",
    "ObjectOrderingStrategy",
    "PlanningObject",
    "TopLeftSpatialOrdering",
    "resolve_object_ordering_strategy",
]
