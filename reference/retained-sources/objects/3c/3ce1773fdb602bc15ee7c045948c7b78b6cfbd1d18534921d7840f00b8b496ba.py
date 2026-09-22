"""Coordinate-token target metadata shared by template, packing, and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.errors import CoordExpError


class CoordinateTargetContractError(CoordExpError):
    default_code = "coordinate_target.contract"


@dataclass(frozen=True)
class CoordinateLossTarget:
    bbox: tuple[int, int, int, int]
    slot_index: int

    def __post_init__(self) -> None:
        bbox = tuple(int(value) for value in self.bbox)
        if len(bbox) != 4:
            raise CoordinateTargetContractError(
                "coordinate loss target bbox must contain four bins",
                code="coordinate_target.bbox_shape",
                context={"bbox": list(bbox)},
            )
        if any(value < 0 or value > 999 for value in bbox):
            raise CoordinateTargetContractError(
                "coordinate loss target bbox bins must stay within [0, 999]",
                code="coordinate_target.bbox_range",
                context={"bbox": list(bbox)},
            )
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise CoordinateTargetContractError(
                "coordinate loss target bbox must be non-degenerate x1,y1,x2,y2",
                code="coordinate_target.bbox_order",
                context={"bbox": list(bbox)},
            )
        slot = int(self.slot_index)
        if slot < 0 or slot > 3:
            raise CoordinateTargetContractError(
                "coordinate loss target slot_index must be in [0, 3]",
                code="coordinate_target.slot_index",
                context={"slot_index": slot},
            )
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "slot_index", slot)

    @property
    def axis_length(self) -> int:
        x1, y1, x2, y2 = self.bbox
        if self.slot_index in (0, 2):
            return x2 - x1
        return y2 - y1

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "slot_index": self.slot_index,
        }


def coordinate_target_to_artifact(
    target: CoordinateLossTarget | None,
) -> dict[str, Any] | None:
    if target is None:
        return None
    return target.to_artifact_dict()


__all__ = [
    "CoordinateLossTarget",
    "CoordinateTargetContractError",
    "coordinate_target_to_artifact",
]
