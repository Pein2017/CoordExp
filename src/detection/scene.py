"""Semantic DetectionScene layer for detection examples.

The scene layer is the in-memory authority for one detection image/example.  Raw
JSONL rows remain intake/storage, while render/token/eval/trainer-specific views
stay owned by their projection layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import random
from typing import Any, Literal, Mapping, Sequence

from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
    RawDetectionObject,
    RawDetectionRow,
)

CoordinateFrame = Literal["image"]
CoordinateSpace = Literal["norm1000"]
BBoxChart = Literal["xyxy"]
DetectionGeometryKind = Literal["bbox_2d", "poly"]


@dataclass(frozen=True)
class DetectionGeometry:
    """Explicit geometry value for a detection object.

    Bbox geometry carries its coordinate frame, coordinate space, and bbox chart
    explicitly.  The chart is never inferred from coordinate position.
    """

    kind: DetectionGeometryKind
    coordinate_frame: CoordinateFrame
    coordinate_space: CoordinateSpace
    bbox_chart: BBoxChart | None = None
    bbox_2d: CoordinateTokenBox | None = None
    polygon: tuple[tuple[int, int], ...] | None = None

    @classmethod
    def from_bbox_2d(
        cls,
        bbox_2d: CoordinateTokenBox,
        *,
        coordinate_frame: CoordinateFrame,
        coordinate_space: CoordinateSpace,
        bbox_chart: BBoxChart,
    ) -> "DetectionGeometry":
        return cls(
            kind="bbox_2d",
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=bbox_chart,
            bbox_2d=bbox_2d,
            polygon=None,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: Sequence[Sequence[int]],
        *,
        coordinate_frame: CoordinateFrame,
        coordinate_space: CoordinateSpace,
    ) -> "DetectionGeometry":
        vertices = tuple(_parse_polygon_vertex(vertex) for vertex in polygon)
        if len(vertices) < 3:
            raise ValueError("poly geometry requires at least three vertices")
        return cls(
            kind="poly",
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=None,
            bbox_2d=None,
            polygon=vertices,
        )

    def __post_init__(self) -> None:
        if self.kind == "bbox_2d":
            if self.bbox_2d is None:
                raise ValueError("bbox_2d geometry requires bbox_2d coordinates")
            if self.bbox_chart is None:
                raise ValueError("bbox_2d geometry requires explicit bbox_chart")
            if self.polygon is not None:
                raise ValueError("bbox_2d geometry must not carry polygon vertices")
            return
        if self.kind == "poly":
            if self.polygon is None:
                raise ValueError("poly geometry requires polygon vertices")
            if self.bbox_2d is not None or self.bbox_chart is not None:
                raise ValueError("poly geometry must not carry bbox_2d or bbox_chart")
            return
        raise ValueError(f"unsupported detection geometry kind: {self.kind!r}")

    def require_bbox_2d(self) -> CoordinateTokenBox:
        if self.kind != "bbox_2d" or self.bbox_2d is None:
            raise ValueError("detection geometry is not bbox_2d")
        return self.bbox_2d


@dataclass(frozen=True)
class DetectionObject:
    """One annotated object in a DetectionScene."""

    scene_object_index: int
    source_object_index: int
    object_instance_id: str
    label: str
    desc: str
    geometry: DetectionGeometry
    category_id: int
    category_name: str
    coco_ann_id: int
    object_id: str | None = None
    source_role: str | None = None
    relation_snapshot: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DetectionScene:
    """Canonical in-memory semantics for exactly one detection image/example."""

    image_id: int
    image_reference: str
    source_image_reference: str
    file_name: str
    width: int
    height: int
    coordinate_frame: CoordinateFrame
    coordinate_space: CoordinateSpace
    bbox_chart: BBoxChart
    objects: tuple[DetectionObject, ...]
    object_ordering: ObjectOrderingPlan
    metadata: DetectionMetadata

    @property
    def realized_source_object_indices(self) -> tuple[int, ...]:
        return self.object_ordering.realized_source_object_indices

    @property
    def images(self) -> tuple[str, ...]:
        return (self.image_reference,)


def detection_scene_from_raw_row(
    raw: RawDetectionRow,
    *,
    object_ordering: ObjectOrderingPlan,
    image_reference: str,
    coordinate_frame: CoordinateFrame = "image",
    coordinate_space: CoordinateSpace = "norm1000",
    bbox_chart: BBoxChart = "xyxy",
) -> DetectionScene:
    """Project a raw intake row into the semantic scene layer."""

    source_image_reference = _single_image_reference(raw.images)
    resolved_image_reference = _require_resolved_image_reference(image_reference)
    source_indices = _realize_object_order(raw, object_ordering=object_ordering)
    objects = tuple(
        _scene_object_from_raw(
            raw,
            raw.objects[source_index],
            scene_object_index=scene_object_index,
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=bbox_chart,
        )
        for scene_object_index, source_index in enumerate(source_indices)
    )
    return DetectionScene(
        image_id=raw.image_id,
        image_reference=resolved_image_reference,
        source_image_reference=source_image_reference,
        file_name=raw.file_name,
        width=raw.width,
        height=raw.height,
        coordinate_frame=coordinate_frame,
        coordinate_space=coordinate_space,
        bbox_chart=bbox_chart,
        objects=objects,
        object_ordering=object_ordering.with_realized(source_indices),
        metadata=raw.metadata,
    )


def normalized_detection_sample_from_raw_row_bridge(
    raw: RawDetectionRow,
    *,
    object_ordering: ObjectOrderingPlan,
) -> NormalizedDetectionSample:
    """Build the temporary normalized bridge without creating a canonical scene.

    ``NormalizedDetectionSample`` still carries source-relative image strings for
    migration-era render/token paths.  Keep that behavior explicit here so
    canonical ``DetectionScene`` construction can require a resolved image
    reference.
    """

    source_indices = _realize_object_order(raw, object_ordering=object_ordering)
    normalized_objects = tuple(
        _normalized_object_from_raw(
            raw,
            raw.objects[source_index],
            normalized_object_index=normalized_index,
        )
        for normalized_index, source_index in enumerate(source_indices)
    )
    return NormalizedDetectionSample(
        images=raw.images,
        objects=normalized_objects,
        width=raw.width,
        height=raw.height,
        image_id=raw.image_id,
        file_name=raw.file_name,
        metadata=raw.metadata,
        object_ordering=object_ordering.with_realized(source_indices),
    )


def detection_scene_from_normalized_sample_bridge(
    sample: NormalizedDetectionSample,
    *,
    image_reference: str,
    coordinate_frame: CoordinateFrame = "image",
    coordinate_space: CoordinateSpace = "norm1000",
    bbox_chart: BBoxChart = "xyxy",
) -> DetectionScene:
    """Adapt the temporary normalized bridge into a DetectionScene.

    This is a migration bridge for retained projection paths.  Public canonical
    construction should prefer raw-intake loading with a resolved image
    reference.
    """

    source_image_reference = _single_image_reference(sample.images)
    resolved_image_reference = _require_resolved_image_reference(image_reference)
    realized = _validate_normalized_sample_ordering(sample)
    objects = tuple(
        _scene_object_from_normalized(
            obj,
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=bbox_chart,
        )
        for obj in sample.objects
    )
    return DetectionScene(
        image_id=sample.image_id,
        image_reference=resolved_image_reference,
        source_image_reference=source_image_reference,
        file_name=sample.file_name,
        width=sample.width,
        height=sample.height,
        coordinate_frame=coordinate_frame,
        coordinate_space=coordinate_space,
        bbox_chart=bbox_chart,
        objects=objects,
        object_ordering=sample.object_ordering.with_realized(realized),
        metadata=sample.metadata,
    )


def normalized_detection_sample_from_scene(
    scene: DetectionScene,
) -> NormalizedDetectionSample:
    """Project scene semantics into the temporary normalized bridge."""

    objects = tuple(
        NormalizedDetectionObject(
            normalized_object_index=obj.scene_object_index,
            source_object_index=obj.source_object_index,
            object_instance_id=obj.object_instance_id,
            desc=obj.desc,
            bbox_2d=obj.geometry.require_bbox_2d(),
            category_id=obj.category_id,
            category_name=obj.category_name,
            coco_ann_id=obj.coco_ann_id,
            object_id=obj.object_id,
            source_role=obj.source_role,
            relation_snapshot=obj.relation_snapshot,
        )
        for obj in scene.objects
    )
    return NormalizedDetectionSample(
        images=(scene.source_image_reference,),
        objects=objects,
        width=scene.width,
        height=scene.height,
        image_id=scene.image_id,
        file_name=scene.file_name,
        metadata=scene.metadata,
        object_ordering=scene.object_ordering,
    )


def _scene_object_from_raw(
    raw: RawDetectionRow,
    obj: RawDetectionObject,
    *,
    scene_object_index: int,
    coordinate_frame: CoordinateFrame,
    coordinate_space: CoordinateSpace,
    bbox_chart: BBoxChart,
) -> DetectionObject:
    relation_snapshot = _relation_snapshot_for_object(raw.metadata, obj.object_id)
    return DetectionObject(
        scene_object_index=scene_object_index,
        source_object_index=obj.source_object_index,
        object_instance_id=_stable_object_instance_id(raw, obj),
        label=obj.category_name,
        desc=obj.desc,
        geometry=DetectionGeometry.from_bbox_2d(
            obj.bbox_2d,
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=bbox_chart,
        ),
        category_id=obj.category_id,
        category_name=obj.category_name,
        coco_ann_id=obj.coco_ann_id,
        object_id=obj.object_id,
        source_role=_source_role_from_snapshot(relation_snapshot),
        relation_snapshot=relation_snapshot,
    )


def _normalized_object_from_raw(
    raw: RawDetectionRow,
    obj: RawDetectionObject,
    *,
    normalized_object_index: int,
) -> NormalizedDetectionObject:
    relation_snapshot = _relation_snapshot_for_object(raw.metadata, obj.object_id)
    return NormalizedDetectionObject(
        normalized_object_index=normalized_object_index,
        source_object_index=obj.source_object_index,
        object_instance_id=_stable_object_instance_id(raw, obj),
        desc=obj.desc,
        bbox_2d=obj.bbox_2d,
        category_id=obj.category_id,
        category_name=obj.category_name,
        coco_ann_id=obj.coco_ann_id,
        object_id=obj.object_id,
        source_role=_source_role_from_snapshot(relation_snapshot),
        relation_snapshot=relation_snapshot,
    )


def _scene_object_from_normalized(
    obj: NormalizedDetectionObject,
    *,
    coordinate_frame: CoordinateFrame,
    coordinate_space: CoordinateSpace,
    bbox_chart: BBoxChart,
) -> DetectionObject:
    return DetectionObject(
        scene_object_index=obj.normalized_object_index,
        source_object_index=obj.source_object_index,
        object_instance_id=obj.object_instance_id,
        label=obj.category_name,
        desc=obj.desc,
        geometry=DetectionGeometry.from_bbox_2d(
            obj.bbox_2d,
            coordinate_frame=coordinate_frame,
            coordinate_space=coordinate_space,
            bbox_chart=bbox_chart,
        ),
        category_id=obj.category_id,
        category_name=obj.category_name,
        coco_ann_id=obj.coco_ann_id,
        object_id=obj.object_id,
        source_role=obj.source_role,
        relation_snapshot=obj.relation_snapshot,
    )


def _single_image_reference(images: Sequence[str]) -> str:
    if len(images) != 1:
        raise ValueError(
            "DetectionScene requires exactly one image reference; "
            f"got {len(images)}"
        )
    image_reference = str(images[0])
    if not image_reference:
        raise ValueError("DetectionScene image reference must be non-empty")
    return image_reference


def _require_resolved_image_reference(image_reference: str) -> str:
    if not isinstance(image_reference, str) or not image_reference:
        raise ValueError(
            "DetectionScene requires a resolved image_reference supplied by the "
            "caller"
        )
    if not Path(image_reference).expanduser().is_absolute():
        raise ValueError(
            "DetectionScene image_reference must be an absolute local path; "
            f"got {image_reference!r}"
        )
    return image_reference


def _validate_normalized_sample_ordering(
    sample: NormalizedDetectionSample,
) -> tuple[int, ...]:
    realized = sample.realized_source_object_indices
    if not realized:
        realized = tuple(obj.source_object_index for obj in sample.objects)
    if len(realized) != len(sample.objects):
        raise ValueError(
            "realized_source_object_indices length must match normalized objects; "
            f"got {len(realized)} for {len(sample.objects)} objects"
        )
    object_source_indices = tuple(obj.source_object_index for obj in sample.objects)
    if realized != object_source_indices:
        raise ValueError(
            "realized_source_object_indices must agree with normalized object "
            f"source indices; got realized={realized}, objects={object_source_indices}"
        )
    for expected_index, obj in enumerate(sample.objects):
        if int(obj.normalized_object_index) != expected_index:
            raise ValueError(
                "normalized_object_index must match object tuple position; "
                f"got object at position {expected_index} with "
                f"normalized_object_index={obj.normalized_object_index}"
            )
    return realized


def _realize_object_order(
    raw: RawDetectionRow, *, object_ordering: ObjectOrderingPlan
) -> tuple[int, ...]:
    indices = list(range(len(raw.objects)))
    if object_ordering.strategy == "sorted":
        geometry_keys = tuple(
            (
                obj.bbox_2d.values[1],
                obj.bbox_2d.values[0],
                obj.source_object_index,
            )
            for obj in raw.objects
        )
        if geometry_keys != tuple(sorted(geometry_keys)):
            raise ValueError(
                "sorted object_ordering requires source objects to be sorted by "
                "(y1, x1, source_object_index)"
            )
        return tuple(indices)
    if object_ordering.strategy == "random_permutation":
        if object_ordering.seed is None:
            raise ValueError("random_permutation object ordering requires seed")
        rng = random.Random(object_ordering.seed)
        rng.shuffle(indices)
        return tuple(indices)
    raise ValueError(
        "object_ordering.strategy must be one of {'sorted', 'random_permutation'}; "
        f"got {object_ordering.strategy!r}"
    )


def _stable_object_instance_id(raw: RawDetectionRow, obj: RawDetectionObject) -> str:
    return (
        f"{raw.metadata.source}:{raw.metadata.split}:"
        f"image_id={raw.image_id}:source_object_index={obj.source_object_index}:"
        f"coco_ann_id={obj.coco_ann_id}"
    )


def _relation_snapshot_for_object(
    metadata: DetectionMetadata, object_id: str | None
) -> Mapping[str, Any] | None:
    if object_id is None or metadata.supervision is None:
        return None
    object_supervision = metadata.supervision.get("object_supervision")
    if not isinstance(object_supervision, Mapping):
        return None
    snapshot = object_supervision.get(object_id)
    if not isinstance(snapshot, Mapping):
        return None
    return MappingProxyType(dict(snapshot))


def _source_role_from_snapshot(snapshot: Mapping[str, Any] | None) -> str | None:
    if snapshot is None or "source_role" not in snapshot:
        return None
    source_role = snapshot["source_role"]
    if source_role is None:
        return None
    if not isinstance(source_role, str):
        raise ValueError("metadata supervision source_role must be a string when present")
    return source_role


def _parse_polygon_vertex(vertex: Sequence[int]) -> tuple[int, int]:
    if len(vertex) != 2:
        raise ValueError("poly geometry vertices must contain exactly two coordinates")
    x, y = vertex
    if not isinstance(x, int) or isinstance(x, bool):
        raise ValueError("poly geometry x coordinates must be integers")
    if not isinstance(y, int) or isinstance(y, bool):
        raise ValueError("poly geometry y coordinates must be integers")
    if x < 0 or x > 999 or y < 0 or y > 999:
        raise ValueError("poly geometry coordinates must be in the 0..999 range")
    return (x, y)


__all__ = [
    "BBoxChart",
    "CoordinateFrame",
    "CoordinateSpace",
    "DetectionGeometry",
    "DetectionGeometryKind",
    "DetectionObject",
    "DetectionScene",
    "detection_scene_from_normalized_sample_bridge",
    "detection_scene_from_raw_row",
    "normalized_detection_sample_from_scene",
    "normalized_detection_sample_from_raw_row_bridge",
]
