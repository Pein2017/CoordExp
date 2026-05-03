"""Strict typed contract for raw and normalized detection rows."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Literal, Mapping, Sequence

_RAW_TOP_LEVEL_KEYS = frozenset(
    {"file_name", "height", "image_id", "images", "metadata", "objects", "width"}
)
_RAW_OBJECT_KEYS = frozenset(
    {"bbox_2d", "category_id", "category_name", "coco_ann_id", "desc"}
)
_RAW_METADATA_KEYS = frozenset({"source", "split"})
_COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,3})\|>")

ObjectOrderingStrategy = Literal["sorted", "random_permutation"]


@dataclass(frozen=True)
class CoordinateTokenBox:
    x1: str
    y1: str
    x2: str
    y2: str

    @property
    def tokens(self) -> tuple[str, str, str, str]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class DetectionMetadata:
    source: str
    split: str


@dataclass(frozen=True)
class RawDetectionObject:
    source_object_index: int
    desc: str
    bbox_2d: CoordinateTokenBox
    category_id: int
    category_name: str
    coco_ann_id: int


@dataclass(frozen=True)
class RawDetectionRow:
    images: tuple[str, ...]
    objects: tuple[RawDetectionObject, ...]
    width: int
    height: int
    image_id: int
    file_name: str
    metadata: DetectionMetadata


@dataclass(frozen=True)
class NormalizedDetectionObject:
    normalized_object_index: int
    source_object_index: int
    object_instance_id: str
    desc: str
    bbox_2d: CoordinateTokenBox
    category_id: int
    category_name: str
    coco_ann_id: int


@dataclass(frozen=True)
class ObjectOrderingPlan:
    strategy: ObjectOrderingStrategy
    seed: int | None
    seed_source: str
    realized_source_object_indices: tuple[int, ...] = ()

    @classmethod
    def sorted(cls, *, seed_source: str = "source_order") -> "ObjectOrderingPlan":
        return cls(
            strategy="sorted",
            seed=None,
            seed_source=seed_source,
            realized_source_object_indices=(),
        )

    @classmethod
    def random_permutation(
        cls, *, seed: int | None, seed_source: str
    ) -> "ObjectOrderingPlan":
        if seed is None:
            raise ValueError("random_permutation object ordering requires seed")
        return cls(
            strategy="random_permutation",
            seed=int(seed),
            seed_source=seed_source,
            realized_source_object_indices=(),
        )

    def with_realized(
        self, realized_source_object_indices: Sequence[int]
    ) -> "ObjectOrderingPlan":
        return ObjectOrderingPlan(
            strategy=self.strategy,
            seed=self.seed,
            seed_source=self.seed_source,
            realized_source_object_indices=tuple(realized_source_object_indices),
        )


@dataclass(frozen=True)
class NormalizedDetectionSample:
    images: tuple[str, ...]
    objects: tuple[NormalizedDetectionObject, ...]
    width: int
    height: int
    image_id: int
    file_name: str
    metadata: DetectionMetadata
    object_ordering: ObjectOrderingPlan

    @property
    def realized_source_object_indices(self) -> tuple[int, ...]:
        return self.object_ordering.realized_source_object_indices


def parse_raw_detection_row(row: Mapping[str, Any]) -> RawDetectionRow:
    """Parse a source COCO coord-token JSONL row into frozen typed containers."""

    _validate_key_set(row, expected=_RAW_TOP_LEVEL_KEYS, label="top-level")
    metadata_raw = _require_mapping(row["metadata"], path="metadata")
    _validate_key_set(metadata_raw, expected=_RAW_METADATA_KEYS, label="metadata")

    objects_raw = _require_sequence(row["objects"], path="objects")
    if not objects_raw:
        raise ValueError("objects must contain at least one object")
    images_raw = _require_sequence(row["images"], path="images")
    if not images_raw:
        raise ValueError("images must contain at least one image")

    objects = tuple(
        _parse_raw_object(obj, index=index) for index, obj in enumerate(objects_raw)
    )
    return RawDetectionRow(
        images=tuple(
            _require_str(item, path=f"images[{idx}]")
            for idx, item in enumerate(images_raw)
        ),
        objects=objects,
        width=_require_int(row["width"], path="width"),
        height=_require_int(row["height"], path="height"),
        image_id=_require_int(row["image_id"], path="image_id"),
        file_name=_require_str(row["file_name"], path="file_name"),
        metadata=DetectionMetadata(
            source=_require_str(metadata_raw["source"], path="metadata.source"),
            split=_require_str(metadata_raw["split"], path="metadata.split"),
        ),
    )


def normalize_detection_row(
    raw: RawDetectionRow, *, object_ordering: ObjectOrderingPlan
) -> NormalizedDetectionSample:
    """Normalize raw detection objects while preserving source provenance."""

    source_indices = _realize_object_order(raw, object_ordering=object_ordering)
    normalized_objects: list[NormalizedDetectionObject] = []
    for normalized_index, source_index in enumerate(source_indices):
        source = raw.objects[source_index]
        normalized_objects.append(
            NormalizedDetectionObject(
                normalized_object_index=normalized_index,
                source_object_index=source.source_object_index,
                object_instance_id=_stable_object_instance_id(raw, source),
                desc=source.desc,
                bbox_2d=source.bbox_2d,
                category_id=source.category_id,
                category_name=source.category_name,
                coco_ann_id=source.coco_ann_id,
            )
        )

    return NormalizedDetectionSample(
        images=raw.images,
        objects=tuple(normalized_objects),
        width=raw.width,
        height=raw.height,
        image_id=raw.image_id,
        file_name=raw.file_name,
        metadata=raw.metadata,
        object_ordering=object_ordering.with_realized(source_indices),
    )


def _parse_raw_object(obj: Any, *, index: int) -> RawDetectionObject:
    path = f"objects[{index}]"
    obj_raw = _require_mapping(obj, path=path)
    _validate_key_set(obj_raw, expected=_RAW_OBJECT_KEYS, label=path, key_name="object")
    return RawDetectionObject(
        source_object_index=index,
        desc=_require_str(obj_raw["desc"], path=f"{path}.desc"),
        bbox_2d=_parse_coordinate_token_box(obj_raw["bbox_2d"], path=f"{path}.bbox_2d"),
        category_id=_require_int(obj_raw["category_id"], path=f"{path}.category_id"),
        category_name=_require_str(obj_raw["category_name"], path=f"{path}.category_name"),
        coco_ann_id=_require_int(obj_raw["coco_ann_id"], path=f"{path}.coco_ann_id"),
    )


def _parse_coordinate_token_box(value: Any, *, path: str) -> CoordinateTokenBox:
    values = _require_sequence(value, path=path)
    if len(values) != 4:
        raise ValueError(f"{path} must contain exactly four coordinate-token strings")
    tokens = tuple(
        _require_coordinate_token(token, path=f"{path}[{idx}]")
        for idx, token in enumerate(values)
    )
    return CoordinateTokenBox(*tokens)


def _require_coordinate_token(value: Any, *, path: str) -> str:
    token = _require_str(value, path=path)
    match = _COORD_TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"{path} must be a compact-v1 coordinate token string")
    coord_value = int(match.group(1))
    if coord_value > 999:
        raise ValueError(f"{path} coordinate token must be in the 0..999 range")
    return token


def _realize_object_order(
    raw: RawDetectionRow, *, object_ordering: ObjectOrderingPlan
) -> tuple[int, ...]:
    indices = list(range(len(raw.objects)))
    if object_ordering.strategy == "sorted":
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


def _validate_key_set(
    mapping: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
    key_name: str = "top-level",
) -> None:
    keys = set(mapping.keys())
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        raise ValueError(f"{label} missing {key_name} keys: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} unsupported {key_name} keys: {', '.join(extra)}")


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return value


def _require_str(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if not value:
        raise ValueError(f"{path} must be non-empty")
    return value


def _require_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


__all__ = [
    "CoordinateTokenBox",
    "DetectionMetadata",
    "NormalizedDetectionObject",
    "NormalizedDetectionSample",
    "ObjectOrderingPlan",
    "ObjectOrderingStrategy",
    "RawDetectionObject",
    "RawDetectionRow",
    "normalize_detection_row",
    "parse_raw_detection_row",
]
