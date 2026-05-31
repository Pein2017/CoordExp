"""Strict typed contract for raw and normalized detection rows."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Literal, Mapping, Sequence

_RAW_TOP_LEVEL_KEYS = frozenset(
    {"file_name", "height", "image_id", "images", "metadata", "objects", "width"}
)
_RAW_REQUIRED_OBJECT_KEYS = frozenset(
    {"bbox_2d", "category_id", "category_name", "coco_ann_id", "desc"}
)
_RAW_OPTIONAL_OBJECT_KEYS = frozenset({"object_id"})
_RAW_METADATA_KEYS = frozenset({"source", "split"})
_RAW_OPTIONAL_METADATA_KEYS = frozenset({"supervision"})
_COORD_TOKEN_RE = re.compile(r"<\|coord_(\d{1,3})\|>")

ObjectOrderingStrategy = Literal["sorted", "random_permutation"]


@dataclass(frozen=True)
class CoordinateTokenBox:
    x1: str | int
    y1: str | int
    x2: str | int
    y2: str | int

    @property
    def tokens(self) -> tuple[str, str, str, str]:
        return tuple(
            _coordinate_component_token(component)
            for component in (self.x1, self.y1, self.x2, self.y2)
        )

    @property
    def values(self) -> tuple[int, int, int, int]:
        return tuple(
            _coordinate_component_value(component)
            for component in (self.x1, self.y1, self.x2, self.y2)
        )


@dataclass(frozen=True)
class DetectionMetadata:
    source: str
    split: str
    supervision: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RawDetectionObject:
    source_object_index: int
    desc: str
    bbox_2d: CoordinateTokenBox
    category_id: int
    category_name: str
    coco_ann_id: int
    object_id: str | None = None


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
    object_id: str | None = None
    source_role: str | None = None
    relation_snapshot: Mapping[str, Any] | None = None


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
    """Parse legacy coord-token or canonical norm1000 rows into typed containers."""

    _validate_key_set(row, expected=_RAW_TOP_LEVEL_KEYS, label="top-level")
    metadata_raw = _require_mapping(row["metadata"], path="metadata")
    _validate_key_set(
        metadata_raw,
        expected=_RAW_METADATA_KEYS,
        optional=_RAW_OPTIONAL_METADATA_KEYS,
        label="metadata",
    )
    supervision = _parse_optional_supervision(metadata_raw)

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
            supervision=supervision,
        ),
    )


def normalize_detection_row(
    raw: RawDetectionRow, *, object_ordering: ObjectOrderingPlan
) -> NormalizedDetectionSample:
    """Normalize raw detection objects while preserving source provenance."""

    from src.detection.scene import normalized_detection_sample_from_raw_row_bridge

    return normalized_detection_sample_from_raw_row_bridge(
        raw,
        object_ordering=object_ordering,
    )


def _parse_raw_object(obj: Any, *, index: int) -> RawDetectionObject:
    path = f"objects[{index}]"
    obj_raw = _require_mapping(obj, path=path)
    _validate_key_set(
        obj_raw,
        expected=_RAW_REQUIRED_OBJECT_KEYS,
        optional=_RAW_OPTIONAL_OBJECT_KEYS,
        label=path,
        key_name="object",
    )
    bbox_2d = _parse_coordinate_box(obj_raw["bbox_2d"], path=f"{path}.bbox_2d")
    object_id = (
        _require_str(obj_raw["object_id"], path=f"{path}.object_id")
        if "object_id" in obj_raw
        else None
    )
    if _is_norm1000_integer_box(bbox_2d) and object_id is None:
        raise ValueError(f"{path}.object_id is required for norm1000 integer bbox_2d rows")

    return RawDetectionObject(
        source_object_index=index,
        desc=_require_str(obj_raw["desc"], path=f"{path}.desc"),
        bbox_2d=bbox_2d,
        category_id=_require_int(obj_raw["category_id"], path=f"{path}.category_id"),
        category_name=_require_str(obj_raw["category_name"], path=f"{path}.category_name"),
        coco_ann_id=_require_int(obj_raw["coco_ann_id"], path=f"{path}.coco_ann_id"),
        object_id=object_id,
    )


def _parse_optional_supervision(
    metadata_raw: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if "supervision" not in metadata_raw:
        return None

    supervision_raw = _require_mapping(
        metadata_raw["supervision"],
        path="metadata.supervision",
    )
    if "object_supervision" in supervision_raw:
        object_supervision = _require_mapping(
            supervision_raw["object_supervision"],
            path="metadata.supervision.object_supervision",
        )
        for object_id, snapshot in object_supervision.items():
            if not isinstance(object_id, str) or not object_id:
                raise ValueError(
                    "metadata.supervision.object_supervision keys must be non-empty strings"
                )
            _require_mapping(
                snapshot,
                path=f"metadata.supervision.object_supervision[{object_id!r}]",
            )
    if "support_objects" in supervision_raw:
        _require_sequence(
            supervision_raw["support_objects"],
            path="metadata.supervision.support_objects",
        )

    return _copy_json_safe_metadata(supervision_raw, path="metadata.supervision")


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
    return _copy_json_safe_metadata(
        snapshot,
        path=f"metadata.supervision.object_supervision[{object_id!r}]",
    )


def _source_role_from_snapshot(snapshot: Mapping[str, Any] | None) -> str | None:
    if snapshot is None or "source_role" not in snapshot:
        return None
    source_role = snapshot["source_role"]
    if source_role is None:
        return None
    if not isinstance(source_role, str):
        raise ValueError("metadata supervision source_role must be a string when present")
    return source_role


def _copy_json_safe_metadata(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            copied[key] = _copy_json_safe_metadata(item, path=f"{path}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _copy_json_safe_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{path} must contain JSON-safe metadata values")


def _parse_coordinate_box(value: Any, *, path: str) -> CoordinateTokenBox:
    values = _require_sequence(value, path=path)
    if len(values) != 4:
        raise ValueError(
            f"{path} must contain exactly four coordinate-token strings "
            "or norm1000 integers"
        )
    if _all_norm1000_integer_components(values):
        coords = tuple(
            _require_norm1000_integer_coord(coord, path=f"{path}[{idx}]")
            for idx, coord in enumerate(values)
        )
        _validate_non_inverted_xyxy(coords, path=path, format_label="norm1000 integer")
        return CoordinateTokenBox(*coords)
    if all(isinstance(component, str) for component in values):
        tokens = tuple(
            _require_coordinate_token(token, path=f"{path}[{idx}]")
            for idx, token in enumerate(values)
        )
        coords = tuple(_coordinate_token_value(token) for token in tokens)
        _validate_non_inverted_xyxy(
            coords, path=path, format_label="coordinate-token"
        )
        return CoordinateTokenBox(*tokens)
    raise ValueError(
        f"{path} must contain either four norm1000 integers or four coordinate-token strings"
    )


def _require_coordinate_token(value: Any, *, path: str) -> str:
    token = _require_str(value, path=path)
    match = _COORD_TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"{path} must be a compact-v1 coordinate token string")
    coord_text = match.group(1)
    coord_value = int(coord_text)
    if str(coord_value) != coord_text:
        raise ValueError(f"{path} coordinate token must use canonical integer text")
    if coord_value > 999:
        raise ValueError(f"{path} coordinate token must be in the 0..999 range")
    return token


def _coordinate_token_value(token: str) -> int:
    match = _COORD_TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"expected coordinate token, got {token!r}")
    return int(match.group(1))


def _coordinate_component_token(value: str | int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"coordinate value must be a string token or integer, got {value!r}")
    if value < 0 or value > 999:
        raise ValueError(f"coordinate integer must be in the 0..999 range, got {value}")
    return f"<|coord_{value}|>"


def _coordinate_component_value(value: str | int) -> int:
    if isinstance(value, str):
        return _coordinate_token_value(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"coordinate value must be a string token or integer, got {value!r}")
    return value


def _all_norm1000_integer_components(values: Sequence[Any]) -> bool:
    return all(isinstance(value, int) and not isinstance(value, bool) for value in values)


def _require_norm1000_integer_coord(value: Any, *, path: str) -> int:
    coord = _require_int(value, path=path)
    if coord < 0 or coord > 999:
        raise ValueError(f"{path} norm1000 coordinate must be in the 0..999 range")
    return coord


def _validate_non_inverted_xyxy(
    coords: tuple[int, int, int, int], *, path: str, format_label: str
) -> None:
    x1, y1, x2, y2 = coords
    if x1 > x2 or y1 > y2:
        raise ValueError(
            f"{path} must be a non-inverted xyxy {format_label} box; "
            f"got x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )


def _is_norm1000_integer_box(box: CoordinateTokenBox) -> bool:
    return all(
        isinstance(component, int) and not isinstance(component, bool)
        for component in (box.x1, box.y1, box.x2, box.y2)
    )


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


def _validate_key_set(
    mapping: Mapping[str, Any],
    *,
    expected: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
    key_name: str = "top-level",
) -> None:
    keys = set(mapping.keys())
    missing = sorted(expected - keys)
    extra = sorted(keys - expected - optional)
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
