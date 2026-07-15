"""Frozen working-data, identity, ordering, and runtime-layout models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from src.common.errors import DataContractError
from src.data.examples import JsonFrozen, freeze_json, thaw_json
from src.data.geometry import validate_bbox_bins
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY


Split = Literal["train", "val"]
RUNTIME_RELATIVE_ROOT = Path(
    "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
)
SHARED_IMAGE_RELATIVE_ROOT = Path("public_data/coco/rescale_32_1024_bbox/images")
SELECTED_SOURCE_RELATIVE_ROOT = Path(
    "public_data/coco/rescale_32_1024_bbox_len12000"
)
WORKING_ROW_FIELDS = frozenset(
    {"file_name", "height", "image_id", "images", "metadata", "objects", "width"}
)
WORKING_OBJECT_REQUIRED_FIELDS = frozenset(
    {"bbox_2d", "desc", "category_id", "category_name", "coco_ann_id"}
)
WORKING_OBJECT_FIELDS = WORKING_OBJECT_REQUIRED_FIELDS | {"metadata"}


@dataclass(frozen=True, order=True)
class TaskIdentity:
    split: Split
    image_id: int

    def __post_init__(self) -> None:
        _validate_split(self.split)
        _integer(self.image_id, field="image_id", positive=True)


@dataclass(frozen=True)
class ObjectIdentity:
    """Hidden stable region identity plus deterministic tie-order provenance."""

    region_key: str
    coco_ann_id: int | None
    prior_rank: int | None = None
    creation_ordinal: int | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.region_key, field="region_key")
        if self.coco_ann_id is not None:
            _integer(self.coco_ann_id, field="coco_ann_id", nonzero=True)
        if self.prior_rank is not None:
            _integer(self.prior_rank, field="prior_rank", minimum=0)
        if self.creation_ordinal is not None:
            _integer(self.creation_ordinal, field="creation_ordinal", minimum=0)
        if self.prior_rank is None and self.creation_ordinal is None:
            raise DataContractError(
                "object identity requires prior rank or creation ordinal",
                code="label_studio.object_order_identity",
                context={"region_key": self.region_key},
            )


@dataclass(frozen=True)
class WorkingObject:
    bbox_2d: tuple[int, int, int, int]
    desc: str
    category_id: int
    category_name: str
    coco_ann_id: int
    metadata: Mapping[str, JsonFrozen] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bbox_2d",
            validate_bbox_bins(self.bbox_2d, field="bbox_2d"),
        )
        category = COCO80_REGISTRY.validate(self.category_name, self.category_id)
        if self.desc != category.name:
            raise DataContractError(
                "desc and category_name must equal the canonical COCO-80 name",
                code="label_studio.object_desc",
                context={"desc": self.desc, "category_name": self.category_name},
            )
        _integer(self.coco_ann_id, field="coco_ann_id", nonzero=True)
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise DataContractError(
                    "object metadata must be a JSON object",
                    code="label_studio.object_metadata",
                    context={"value_type": type(self.metadata).__name__},
                )
            object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @classmethod
    def from_mapping(cls, value: Any, *, field: str = "object") -> "WorkingObject":
        mapping = _mapping(value, field=field)
        fields = frozenset(str(key) for key in mapping)
        _exact_fields(
            fields,
            required=WORKING_OBJECT_REQUIRED_FIELDS,
            allowed=WORKING_OBJECT_FIELDS,
            field=field,
        )
        return cls(
            bbox_2d=mapping["bbox_2d"],
            desc=mapping["desc"],
            category_id=mapping["category_id"],
            category_name=mapping["category_name"],
            coco_ann_id=mapping["coco_ann_id"],
            metadata=mapping.get("metadata"),
        )

    def to_json_dict(self, *, coord_tokens: bool = False) -> dict[str, Any]:
        bbox: list[int | str]
        if coord_tokens:
            bbox = [f"<|coord_{value}|>" for value in self.bbox_2d]
        else:
            bbox = list(self.bbox_2d)
        payload: dict[str, Any] = {
            "bbox_2d": bbox,
            "desc": self.desc,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "coco_ann_id": self.coco_ann_id,
        }
        if self.metadata is not None:
            payload["metadata"] = thaw_json(self.metadata)
        return payload


@dataclass(frozen=True)
class OrderedWorkingObject:
    identity: ObjectIdentity
    object: WorkingObject


def stable_top_left_order(
    values: Sequence[OrderedWorkingObject],
) -> tuple[OrderedWorkingObject, ...]:
    """Apply identity order first, then stable ``(y1, x1)`` geometric order."""

    seeded = sorted(values, key=lambda value: _identity_order_key(value.identity))
    return tuple(sorted(seeded, key=lambda value: (value.object.bbox_2d[1], value.object.bbox_2d[0])))


@dataclass(frozen=True)
class WorkingRow:
    file_name: str
    height: int
    image_id: int
    images: tuple[str, ...]
    metadata: Mapping[str, JsonFrozen]
    objects: tuple[WorkingObject, ...]
    width: int

    def __post_init__(self) -> None:
        file_name = _relative_path_string(self.file_name, field="file_name")
        object.__setattr__(self, "file_name", file_name)
        _integer(self.image_id, field="image_id", positive=True)
        _integer(self.width, field="width", positive=True)
        _integer(self.height, field="height", positive=True)
        images = tuple(self.images)
        if len(images) != 1:
            raise DataContractError(
                "working row must contain exactly one image locator",
                code="label_studio.image_count",
                context={"count": len(images)},
            )
        image = _relative_path_string(images[0], field="images[0]")
        if not image.endswith(file_name):
            raise DataContractError(
                "file_name must match the working image locator suffix",
                code="label_studio.image_locator_mismatch",
                context={"file_name": file_name, "image": image},
            )
        object.__setattr__(self, "images", (image,))
        if not isinstance(self.metadata, Mapping):
            raise DataContractError(
                "working row metadata must be a JSON object",
                code="label_studio.row_metadata",
                context={"value_type": type(self.metadata).__name__},
            )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        objects = tuple(self.objects)
        if not objects:
            raise DataContractError(
                "working row objects must not be empty",
                code="label_studio.objects_empty",
            )
        if any(not isinstance(item, WorkingObject) for item in objects):
            raise DataContractError(
                "working row objects must be WorkingObject values",
                code="label_studio.object_type",
            )
        ids = [item.coco_ann_id for item in objects]
        if len(set(ids)) != len(ids):
            raise DataContractError(
                "coco_ann_id values must be unique within a working row",
                code="label_studio.object_id_duplicate",
                context={"image_id": self.image_id},
            )
        object.__setattr__(self, "objects", objects)

    @classmethod
    def from_mapping(cls, value: Any, *, field: str = "row") -> "WorkingRow":
        mapping = _mapping(value, field=field)
        _exact_fields(
            frozenset(str(key) for key in mapping),
            required=WORKING_ROW_FIELDS,
            allowed=WORKING_ROW_FIELDS,
            field=field,
        )
        raw_objects = mapping["objects"]
        if not isinstance(raw_objects, list):
            raise DataContractError(
                "working row objects must be a list",
                code="label_studio.objects_shape",
                context={"field": f"{field}.objects"},
            )
        raw_images = mapping["images"]
        if not isinstance(raw_images, list):
            raise DataContractError(
                "working row images must be a list",
                code="label_studio.images_shape",
                context={"field": f"{field}.images"},
            )
        return cls(
            file_name=mapping["file_name"],
            height=mapping["height"],
            image_id=mapping["image_id"],
            images=tuple(raw_images),
            metadata=mapping["metadata"],
            objects=tuple(
                WorkingObject.from_mapping(item, field=f"{field}.objects[{index}]")
                for index, item in enumerate(raw_objects)
            ),
            width=mapping["width"],
        )

    def to_json_dict(self, *, coord_tokens: bool = False) -> dict[str, Any]:
        return {
            "images": list(self.images),
            "objects": [item.to_json_dict(coord_tokens=coord_tokens) for item in self.objects],
            "width": self.width,
            "height": self.height,
            "image_id": self.image_id,
            "file_name": self.file_name,
            "metadata": thaw_json(self.metadata),
        }

    def validate_for_split(self, split: Split) -> "WorkingRow":
        """Validate the exact rebased locator and source split for one project."""

        split = _validate_split(split)
        expected_locator = f"images/{split}2017/{self.image_id:012d}.jpg"
        if self.images[0] != expected_locator:
            raise DataContractError(
                "working image locator must use the canonical split and image_id filename",
                code="label_studio.split_image_locator",
                context={
                    "split": split,
                    "image_id": self.image_id,
                    "expected": expected_locator,
                    "actual": self.images[0],
                },
            )
        if self.file_name != expected_locator:
            raise DataContractError(
                "immutable file_name must equal the canonical split and image_id locator",
                code="label_studio.working_file_name",
                context={
                    "split": split,
                    "image_id": self.image_id,
                    "expected": expected_locator,
                    "actual": self.file_name,
                },
            )
        if self.metadata.get("split") != split:
            raise DataContractError(
                "working row source task identity metadata split does not match the project",
                code="label_studio.row_split",
                context={"expected": split, "actual": self.metadata.get("split")},
            )
        return self


@dataclass(frozen=True)
class RefinementRuntimeLayout:
    """Exact repository-bound one-instance/two-split mutable runtime paths."""

    repository_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_root",
            Path(self.repository_root).expanduser().resolve(),
        )

    @classmethod
    def under_repository(cls, repository_root: Path) -> "RefinementRuntimeLayout":
        return cls(Path(repository_root))

    @property
    def root(self) -> Path:
        return self.repository_root / RUNTIME_RELATIVE_ROOT

    @property
    def image_root(self) -> Path:
        return self.repository_root / SHARED_IMAGE_RELATIVE_ROOT

    @property
    def label_studio_state(self) -> Path:
        return self.root / "label-studio" / "state"

    def split_root(self, split: Split) -> Path:
        return self.root / _validate_split(split)

    def project_manifest(self, split: Split) -> Path:
        return self.split_root(split) / "project.json"

    def working_norm(self, split: Split) -> Path:
        return self.split_root(split) / "working.norm.jsonl"

    def working_coord(self, split: Split) -> Path:
        return self.split_root(split) / "working.coord.jsonl"

    def journal(self, split: Split) -> Path:
        return self.split_root(split) / "journal.jsonl"

    def images_link(self, split: Split) -> Path:
        return self.split_root(split) / "images"

    def commit_lock(self, split: Split) -> Path:
        return self.split_root(split) / ".commit.lock"

    def selected_source(self, split: Split) -> Path:
        return self.repository_root / SELECTED_SOURCE_RELATIVE_ROOT / f"{_validate_split(split)}.norm.jsonl"


def _identity_order_key(identity: ObjectIdentity) -> tuple[int, int, str]:
    if identity.prior_rank is not None:
        return 0, identity.prior_rank, identity.region_key
    assert identity.creation_ordinal is not None
    return 1, identity.creation_ordinal, identity.region_key


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataContractError(
            "value must be a JSON object",
            code="label_studio.mapping_shape",
            context={"field": field, "value_type": type(value).__name__},
        )
    return value


def _exact_fields(
    fields: frozenset[str],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    field: str,
) -> None:
    missing = sorted(required - fields)
    unknown = sorted(fields - allowed)
    if missing or unknown:
        raise DataContractError(
            "working JSONL fields do not match the accepted schema",
            code="label_studio.fields",
            context={"field": field, "missing": missing, "unknown": unknown},
        )


def _integer(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonzero: bool = False,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(
            "field must be an integer",
            code="label_studio.integer_type",
            context={"field": field, "value": value},
        )
    if positive and value <= 0 or nonzero and value == 0 or minimum is not None and value < minimum:
        raise DataContractError(
            "integer field is outside the accepted range",
            code="label_studio.integer_range",
            context={"field": field, "value": value},
        )
    return value


def _non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DataContractError(
            "field must be a non-empty canonical string",
            code="label_studio.string",
            context={"field": field, "value": value},
        )
    return value


def _relative_path_string(value: Any, *, field: str) -> str:
    path_text = _non_empty_string(value, field=field)
    path = PurePosixPath(path_text)
    if "\\" in path_text or path.is_absolute() or ".." in path.parts or path_text == ".":
        raise DataContractError(
            "working image locator must be relative and non-escaping",
            code="label_studio.relative_path",
            context={"field": field, "value": path_text},
        )
    return path_text


def _validate_split(split: Any) -> Split:
    if split not in ("train", "val"):
        raise DataContractError(
            "split must be train or val",
            code="label_studio.split",
            context={"split": split},
        )
    return split


__all__ = [
    "ObjectIdentity",
    "OrderedWorkingObject",
    "RefinementRuntimeLayout",
    "RUNTIME_RELATIVE_ROOT",
    "SELECTED_SOURCE_RELATIVE_ROOT",
    "SHARED_IMAGE_RELATIVE_ROOT",
    "Split",
    "TaskIdentity",
    "WORKING_OBJECT_FIELDS",
    "WORKING_ROW_FIELDS",
    "WorkingObject",
    "WorkingRow",
    "stable_top_left_order",
]
