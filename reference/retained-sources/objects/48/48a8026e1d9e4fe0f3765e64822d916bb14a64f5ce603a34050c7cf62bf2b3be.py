"""Immutable V1 raw-example records and row factories."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.common.errors import DataContractError
from src.data.geometry import parse_source_bbox_tokens, validate_bbox_bins
from src.data.images import (
    image_stat_fingerprint,
    resolve_image_path,
    validate_declared_dimension,
)


JsonFrozen = str | int | float | bool | None | tuple["JsonFrozen", ...] | Mapping[str, "JsonFrozen"]

CANONICAL_TOP_LEVEL_FIELDS = frozenset({"example_id", "image", "objects", "metadata"})
CURRENT_COORD_JSONL_FIELDS = frozenset(
    {"file_name", "height", "image_id", "images", "metadata", "objects", "width"}
)


@dataclass(frozen=True)
class SourceProvenance:
    source_path: Path
    row_number: int
    row_sha256: str
    source_format: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if isinstance(self.row_number, bool) or not isinstance(self.row_number, int):
            raise DataContractError(
                "source row number must be an integer",
                code="data.source_row_number_type",
                context={"value": self.row_number, "value_type": type(self.row_number).__name__},
            )
        if self.row_number <= 0:
            raise DataContractError(
                "source row number must be positive",
                code="data.source_row_number_range",
                context={"value": self.row_number},
            )
        object.__setattr__(
            self,
            "row_sha256",
            _required_non_empty_str(self.row_sha256, field="source.row_sha256"),
        )
        object.__setattr__(
            self,
            "source_format",
            _required_non_empty_str(self.source_format, field="source.source_format"),
        )

    def to_artifact_dict(self) -> dict[str, str | int]:
        return {
            "source_path": str(self.source_path),
            "row_number": self.row_number,
            "row_sha256": self.row_sha256,
            "source_format": self.source_format,
        }


@dataclass(frozen=True)
class ImageRef:
    declared_path: str
    path: Path
    width: int
    height: int
    stat: Mapping[str, JsonFrozen]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "declared_path",
            _required_non_empty_str(self.declared_path, field="image.declared_path"),
        )
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(
            self,
            "width",
            validate_declared_dimension(self.width, field="image.width"),
        )
        object.__setattr__(
            self,
            "height",
            validate_declared_dimension(self.height, field="image.height"),
        )
        object.__setattr__(self, "stat", freeze_json(self.stat))

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "declared_path": self.declared_path,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "stat": thaw_json(self.stat),
        }


@dataclass(frozen=True)
class RawObject:
    object_id: str
    description: str
    bbox: tuple[int, int, int, int]
    metadata: Mapping[str, JsonFrozen]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_id",
            _normalize_identifier(self.object_id, field="object.object_id"),
        )
        object.__setattr__(
            self,
            "description",
            _description(self.description, field="object.description"),
        )
        object.__setattr__(
            self,
            "bbox",
            validate_bbox_bins(self.bbox, field="object.bbox"),
        )
        object.__setattr__(
            self,
            "metadata",
            _optional_metadata(self.metadata, field="object.metadata"),
        )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "description": self.description,
            "bbox": list(self.bbox),
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class RawExample:
    example_id: str
    image: ImageRef
    objects: tuple[RawObject, ...]
    metadata: Mapping[str, JsonFrozen]
    source: SourceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "example_id",
            _required_non_empty_str(self.example_id, field="example_id"),
        )
        if not isinstance(self.image, ImageRef):
            raise DataContractError(
                "RawExample.image must be an ImageRef",
                code="data.image_ref_type",
                context={"value_type": type(self.image).__name__},
            )
        if not isinstance(self.objects, Sequence) or isinstance(self.objects, (str, bytes)):
            raise DataContractError(
                "RawExample.objects must be a sequence of RawObject values",
                code="data.objects_shape",
                context={"value_type": type(self.objects).__name__},
            )
        objects = tuple(self.objects)
        if not objects:
            raise DataContractError("objects must not be empty", code="data.objects_empty")
        for index, obj in enumerate(objects):
            if not isinstance(obj, RawObject):
                raise DataContractError(
                    "RawExample.objects must contain RawObject values",
                    code="data.object_type",
                    context={"index": index, "value_type": type(obj).__name__},
                )
        seen_object_ids: set[str] = set()
        for obj in objects:
            if obj.object_id in seen_object_ids:
                raise DataContractError(
                    "object ids must be unique within one raw example",
                    code="data.duplicate_object_id",
                    context={"example_id": self.example_id, "object_id": obj.object_id},
                )
            seen_object_ids.add(obj.object_id)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "metadata", _optional_metadata(self.metadata, field="metadata"))
        if not isinstance(self.source, SourceProvenance):
            raise DataContractError(
                "RawExample.source must be SourceProvenance",
                code="data.source_type",
                context={"value_type": type(self.source).__name__},
            )

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "image": self.image.to_artifact_dict(),
            "objects": [obj.to_artifact_dict() for obj in self.objects],
            "metadata": thaw_json(self.metadata),
            "source": self.source.to_artifact_dict(),
        }


def raw_example_from_jsonl_row(
    row: Mapping[str, Any],
    *,
    jsonl_path: Path,
    row_number: int,
    raw_line: str,
) -> RawExample:
    if not isinstance(row, Mapping):
        raise DataContractError(
            "JSONL row must be an object",
            code="data.row_mapping",
            context={"path": str(jsonl_path), "row_number": row_number},
        )
    if "video" in row or "videos" in row:
        raise DataContractError(
            "video payloads are not supported in V1 raw examples",
            code="data.video_unsupported",
            context={"path": str(jsonl_path), "row_number": row_number},
        )

    row_sha256 = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
    source = SourceProvenance(
        source_path=jsonl_path,
        row_number=row_number,
        row_sha256=row_sha256,
        source_format=_detect_source_format(row),
    )
    if source.source_format == "canonical_raw_example":
        return _canonical_raw_example(row, jsonl_path=jsonl_path, source=source)
    if source.source_format == "coord_jsonl_len12000":
        return _current_coord_jsonl_example(row, jsonl_path=jsonl_path, source=source)
    raise DataContractError(
        "unsupported raw-example row shape",
        code="data.unsupported_row_shape",
        context={
            "path": str(jsonl_path),
            "row_number": row_number,
            "fields": sorted(str(key) for key in row.keys()),
        },
    )


def _detect_source_format(row: Mapping[str, Any]) -> str:
    fields = frozenset(str(key) for key in row.keys())
    if "image" in row or "example_id" in row:
        _reject_unknown_fields(fields, CANONICAL_TOP_LEVEL_FIELDS, source_format="canonical_raw_example")
        return "canonical_raw_example"
    if fields & CURRENT_COORD_JSONL_FIELDS:
        _reject_unknown_fields(fields, CURRENT_COORD_JSONL_FIELDS, source_format="coord_jsonl_len12000")
        return "coord_jsonl_len12000"
    return "unknown"


def _canonical_raw_example(
    row: Mapping[str, Any],
    *,
    jsonl_path: Path,
    source: SourceProvenance,
) -> RawExample:
    example_id = _required_non_empty_str(row.get("example_id"), field="example_id")
    image = _canonical_image_ref(row.get("image"), jsonl_path=jsonl_path)
    objects = _canonical_objects(row.get("objects"))
    metadata = _optional_metadata(row.get("metadata"), field="metadata")
    return RawExample(
        example_id=example_id,
        image=image,
        objects=objects,
        metadata=metadata,
        source=source,
    )


def _current_coord_jsonl_example(
    row: Mapping[str, Any],
    *,
    jsonl_path: Path,
    source: SourceProvenance,
) -> RawExample:
    _require_fields(
        row,
        CURRENT_COORD_JSONL_FIELDS,
        source_format="coord_jsonl_len12000",
        row_number=source.row_number,
    )
    metadata = _current_source_metadata(row.get("metadata"))
    file_name = _source_file_name(row.get("file_name"))
    images = row.get("images")
    if not isinstance(images, list):
        raise DataContractError(
            "current coord JSONL row must contain images as a list",
            code="data.images_shape",
            context={"row_number": source.row_number, "value_type": type(images).__name__},
        )
    if len(images) != 1:
        raise DataContractError(
            "V1 raw examples must have exactly one image",
            code="data.image_count",
            context={"row_number": source.row_number, "image_count": len(images)},
        )
    image_ref = _source_image_ref(images[0], file_name=file_name)

    width = validate_declared_dimension(row.get("width"), field="width")
    height = validate_declared_dimension(row.get("height"), field="height")
    declared_path, resolved_path = resolve_image_path(
        image_ref,
        root=jsonl_path.parent,
        field="images[0]",
    )
    image = ImageRef(
        declared_path=declared_path,
        path=resolved_path,
        width=width,
        height=height,
        stat=freeze_json(image_stat_fingerprint(resolved_path)),
    )

    source_metadata = _source_row_metadata(
        row,
        metadata=metadata,
        file_name=file_name,
        image_ref=image_ref,
    )
    source_id = _source_example_id(row, metadata=metadata)
    return RawExample(
        example_id=source_id,
        image=image,
        objects=_current_source_objects(row.get("objects")),
        metadata=freeze_json({"source": source_metadata}),
        source=source,
    )


def _canonical_image_ref(value: Any, *, jsonl_path: Path) -> ImageRef:
    if not isinstance(value, Mapping):
        raise DataContractError(
            "image must be an object",
            code="data.image_shape",
            context={"field": "image", "value_type": type(value).__name__},
        )
    fields = frozenset(str(key) for key in value.keys())
    _reject_unknown_fields(fields, frozenset({"path", "width", "height"}), source_format="image")
    declared_path, resolved_path = resolve_image_path(
        value.get("path"),
        root=jsonl_path.parent,
        field="image.path",
        require_under_root=True,
    )
    return ImageRef(
        declared_path=declared_path,
        path=resolved_path,
        width=validate_declared_dimension(value.get("width"), field="image.width"),
        height=validate_declared_dimension(value.get("height"), field="image.height"),
        stat=freeze_json(image_stat_fingerprint(resolved_path)),
    )


def _canonical_objects(value: Any) -> tuple[RawObject, ...]:
    if not isinstance(value, list):
        raise DataContractError(
            "objects must be a list",
            code="data.objects_shape",
            context={"value_type": type(value).__name__},
        )
    if not value:
        raise DataContractError(
            "objects must not be empty",
            code="data.objects_empty",
        )
    objects: list[RawObject] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DataContractError(
                "object entry must be an object",
                code="data.object_shape",
                context={"field": f"objects[{index}]", "value_type": type(item).__name__},
            )
        fields = frozenset(str(key) for key in item.keys())
        _reject_unknown_fields(
            fields,
            frozenset({"object_id", "description", "bbox", "metadata"}),
            source_format=f"objects[{index}]",
        )
        obj = RawObject(
            object_id=_required_unique_id(
                item.get("object_id"),
                seen=seen_ids,
                field=f"objects[{index}].object_id",
            ),
            description=_description(item.get("description"), field=f"objects[{index}].description"),
            bbox=validate_bbox_bins(item.get("bbox"), field=f"objects[{index}].bbox"),
            metadata=_optional_metadata(item.get("metadata"), field=f"objects[{index}].metadata"),
        )
        objects.append(obj)
    return tuple(objects)


def _current_source_objects(value: Any) -> tuple[RawObject, ...]:
    if not isinstance(value, list):
        raise DataContractError(
            "current coord JSONL objects must be a list",
            code="data.objects_shape",
            context={"value_type": type(value).__name__},
        )
    if not value:
        raise DataContractError("objects must not be empty", code="data.objects_empty")
    objects: list[RawObject] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DataContractError(
                "object entry must be an object",
                code="data.object_shape",
                context={"field": f"objects[{index}]", "value_type": type(item).__name__},
            )
        allowed = frozenset(
            {"bbox_2d", "desc", "category_id", "category_name", "coco_ann_id", "metadata"}
        )
        _reject_unknown_fields(
            frozenset(str(key) for key in item.keys()),
            allowed,
            source_format=f"objects[{index}]",
        )
        object_id = item.get("coco_ann_id")
        if object_id is None:
            raise DataContractError(
                "current coord JSONL object must include coco_ann_id as object id",
                code="data.object_id_missing",
                context={"field": f"objects[{index}].coco_ann_id"},
            )
        bbox_tokens = _source_bbox_token_list(
            item.get("bbox_2d"),
            field=f"objects[{index}].bbox_2d",
        )
        bbox = parse_source_bbox_tokens(
            bbox_tokens,
            field=f"objects[{index}].bbox_2d",
        )
        metadata = {
            "source": {
                "bbox_2d": list(bbox_tokens),
                "category_id": item.get("category_id"),
                "category_name": item.get("category_name"),
                "coco_ann_id": item.get("coco_ann_id"),
            }
        }
        if item.get("metadata") is not None:
            metadata["source"]["metadata"] = item["metadata"]
        objects.append(
            RawObject(
                object_id=_required_unique_id(
                    object_id,
                    seen=seen_ids,
                    field=f"objects[{index}].coco_ann_id",
                ),
                description=_description(item.get("desc"), field=f"objects[{index}].desc"),
                bbox=bbox,
                metadata=freeze_json(metadata),
            )
        )
    return tuple(objects)


def _source_example_id(row: Mapping[str, Any], *, metadata: Mapping[str, str]) -> str:
    dataset = metadata["source"]
    split = metadata["split"]
    image_id = row.get("image_id")
    if isinstance(image_id, bool) or image_id is None or not isinstance(image_id, int):
        raise DataContractError(
            "current coord JSONL row must include numeric image_id",
            code="data.example_id_type",
            context={"image_id": image_id, "value_type": type(image_id).__name__},
        )
    return f"{dataset}_{split}_{image_id:012d}"


def _source_row_metadata(
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    file_name: str,
    image_ref: str,
) -> dict[str, Any]:
    return {
        "dataset": metadata["source"],
        "split": metadata["split"],
        "image_id": row.get("image_id"),
        "file_name": file_name,
        "source_format": "coord_jsonl_len12000",
        "original_images": [image_ref],
    }


def _required_non_empty_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise DataContractError(
            "field must be a string",
            code="data.string_type",
            context={"field": field, "value_type": type(value).__name__},
        )
    stripped = value.strip()
    if not stripped:
        raise DataContractError(
            "field must not be empty",
            code="data.string_empty",
            context={"field": field},
        )
    return stripped


def _description(value: Any, *, field: str) -> str:
    description = _required_non_empty_str(value, field=field)
    if any(char in description for char in "\n\r\t"):
        raise DataContractError(
            "description contains unsupported control whitespace",
            code="data.description_control_whitespace",
            context={"field": field, "description": description},
        )
    return description


def _required_unique_id(value: Any, *, seen: set[str], field: str) -> str:
    identifier = _normalize_identifier(value, field=field)
    if identifier in seen:
        raise DataContractError(
            "identifier must be unique within the example",
            code="data.duplicate_object_id",
            context={"field": field, "object_id": identifier},
        )
    seen.add(identifier)
    return identifier


def _normalize_identifier(value: Any, *, field: str) -> str:
    if value is None:
        raise DataContractError(
            "object id is required",
            code="data.object_id_missing",
            context={"field": field},
        )
    if isinstance(value, bool):
        raise DataContractError(
            "object id must not be boolean",
            code="data.object_id_type",
            context={"field": field, "value": value},
        )
    return _required_non_empty_str(str(value) if not isinstance(value, str) else value, field=field)


def _optional_metadata(value: Any, *, field: str) -> Mapping[str, JsonFrozen]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise DataContractError(
            "metadata must be an object",
            code="data.metadata_shape",
            context={"field": field, "value_type": type(value).__name__},
        )
    return freeze_json(value)


def _reject_unknown_fields(fields: frozenset[str], allowed: frozenset[str], *, source_format: str) -> None:
    unknown = sorted(fields - allowed)
    if unknown:
        raise DataContractError(
            "raw example contains unsupported fields",
            code="data.unknown_fields",
            context={"source_format": source_format, "unknown_fields": unknown},
        )


def _require_fields(
    row: Mapping[str, Any],
    required: frozenset[str],
    *,
    source_format: str,
    row_number: int,
) -> None:
    missing = sorted(required - frozenset(str(key) for key in row.keys()))
    if missing:
        raise DataContractError(
            "raw example is missing required fields",
            code="data.missing_fields",
            context={
                "source_format": source_format,
                "row_number": row_number,
                "missing_fields": missing,
            },
        )


def _current_source_metadata(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DataContractError(
            "current coord JSONL metadata must be an object",
            code="data.source_metadata_shape",
            context={"value_type": type(value).__name__},
        )
    return {
        "source": _required_non_empty_str(value.get("source"), field="metadata.source"),
        "split": _required_non_empty_str(value.get("split"), field="metadata.split"),
    }


def _source_file_name(value: Any) -> str:
    file_name = _required_non_empty_str(value, field="file_name")
    path = Path(file_name)
    if path.is_absolute() or ".." in path.parts:
        raise DataContractError(
            "current coord JSONL file_name must be a fixture-relative provenance path",
            code="data.source_file_name",
            context={"file_name": file_name},
        )
    return file_name


def _source_image_ref(value: Any, *, file_name: str) -> str:
    image_ref = _required_non_empty_str(value, field="images[0]")
    if Path(image_ref).is_absolute():
        raise DataContractError(
            "current coord JSONL image reference must be relative to the JSONL source",
            code="data.source_image_path",
            context={"image_ref": image_ref},
        )
    if not image_ref.endswith(file_name):
        raise DataContractError(
            "current coord JSONL file_name must match images[0] suffix",
            code="data.source_image_mismatch",
            context={"file_name": file_name, "image_ref": image_ref},
        )
    return image_ref


def _source_bbox_token_list(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataContractError(
            "source bbox_2d must be a four-token sequence",
            code="data.bbox_shape",
            context={"field": field, "value_type": type(value).__name__},
        )
    return value


def freeze_json(value: Any) -> JsonFrozen:
    if isinstance(value, Mapping):
        frozen: dict[str, JsonFrozen] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise DataContractError(
                    "metadata/provenance keys must be strings",
                    code="data.json_key_type",
                    context={"key": key, "key_type": type(key).__name__},
                )
            frozen[key] = freeze_json(child)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataContractError(
                "metadata/provenance floats must be finite JSON numbers",
                code="data.json_nonfinite_number",
                context={"value": repr(value)},
            )
        return value
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise DataContractError(
        "metadata/provenance must use JSON primitive values",
        code="data.json_primitive",
        context={"value_type": type(value).__name__},
    )


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = [
    "ImageRef",
    "RawExample",
    "RawObject",
    "SourceProvenance",
    "freeze_json",
    "raw_example_from_jsonl_row",
    "thaw_json",
]
