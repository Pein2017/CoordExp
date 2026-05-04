"""Template-independent semantic IR for normalized detection samples."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Literal, Mapping

from src.detection.data import NormalizedDetectionSample, ObjectOrderingPlan

_COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,3})\|>")
_SLOT_NAMES = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class DetectionCoordinateSlot:
    object_instance_id: str
    object_index: int
    slot_name: Literal["x1", "y1", "x2", "y2"]
    norm1000_value: int | None
    coord_token: str
    coord_token_id: int | None
    render_text: str


@dataclass(frozen=True)
class DetectionGeometry:
    slots: tuple[DetectionCoordinateSlot, ...]
    bbox_format: str


@dataclass(frozen=True)
class DetectionObjectEntry:
    object_instance_id: str
    object_index: int
    source_object_index: int
    desc: str
    geometry: DetectionGeometry
    field_order_policy: str
    category_id: int
    category_name: str
    coco_ann_id: int


@dataclass(frozen=True)
class DetectionDocument:
    objects: tuple[DetectionObjectEntry, ...]
    object_ordering: ObjectOrderingPlan
    coordinate_surface: str
    bbox_format: str
    metadata: Mapping[str, object]
    images: tuple[str, ...]
    width: int
    height: int
    image_id: int
    file_name: str
    realized_source_object_indices: tuple[int, ...]

    @classmethod
    def from_normalized_sample(
        cls,
        sample: NormalizedDetectionSample,
        *,
        coordinate_surface: str = "coord_token",
        bbox_format: str = "xyxy",
    ) -> "DetectionDocument":
        """Adapt a normalized detection sample without rendering or tokenizing it."""

        realized_source_object_indices = _realized_source_object_indices(sample)
        return cls(
            objects=tuple(
                _adapt_object(
                    obj,
                    coordinate_surface=coordinate_surface,
                    bbox_format=bbox_format,
                )
                for obj in sample.objects
            ),
            object_ordering=sample.object_ordering,
            coordinate_surface=coordinate_surface,
            bbox_format=bbox_format,
            metadata=MappingProxyType(
                {
                    "source": sample.metadata.source,
                    "split": sample.metadata.split,
                }
            ),
            images=sample.images,
            width=sample.width,
            height=sample.height,
            image_id=sample.image_id,
            file_name=sample.file_name,
            realized_source_object_indices=realized_source_object_indices,
        )


def detection_document_from_normalized_sample(
    sample: NormalizedDetectionSample,
    *,
    coordinate_surface: str = "coord_token",
    bbox_format: str = "xyxy",
) -> DetectionDocument:
    """Adapt a normalized detection sample into semantic IR."""

    return DetectionDocument.from_normalized_sample(
        sample,
        coordinate_surface=coordinate_surface,
        bbox_format=bbox_format,
    )


def _adapt_object(
    obj,
    *,
    coordinate_surface: str,
    bbox_format: str,
) -> DetectionObjectEntry:
    object_index = obj.normalized_object_index
    return DetectionObjectEntry(
        object_instance_id=obj.object_instance_id,
        object_index=object_index,
        source_object_index=obj.source_object_index,
        desc=obj.desc,
        geometry=DetectionGeometry(
            slots=_coordinate_slots(
                object_instance_id=obj.object_instance_id,
                object_index=object_index,
                tokens=obj.bbox_2d.tokens,
            ),
            bbox_format=bbox_format,
        ),
        field_order_policy="desc_then_bbox_2d",
        category_id=obj.category_id,
        category_name=obj.category_name,
        coco_ann_id=obj.coco_ann_id,
    )


def _coordinate_slots(
    *,
    object_instance_id: str,
    object_index: int,
    tokens: tuple[str, str, str, str],
) -> tuple[DetectionCoordinateSlot, ...]:
    return tuple(
        DetectionCoordinateSlot(
            object_instance_id=object_instance_id,
            object_index=object_index,
            slot_name=slot_name,
            norm1000_value=_parse_norm1000_value(coord_token),
            coord_token=coord_token,
            coord_token_id=None,
            render_text=coord_token,
        )
        for slot_name, coord_token in zip(_SLOT_NAMES, tokens, strict=True)
    )


def _parse_norm1000_value(coord_token: str) -> int | None:
    match = _COORD_TOKEN_RE.fullmatch(coord_token)
    if match is None:
        return None
    value = int(match.group(1))
    if value > 999:
        return None
    return value


def _realized_source_object_indices(
    sample: NormalizedDetectionSample,
) -> tuple[int, ...]:
    if sample.realized_source_object_indices:
        return sample.realized_source_object_indices
    return tuple(obj.source_object_index for obj in sample.objects)


__all__ = [
    "DetectionCoordinateSlot",
    "DetectionDocument",
    "DetectionGeometry",
    "DetectionObjectEntry",
    "detection_document_from_normalized_sample",
]
