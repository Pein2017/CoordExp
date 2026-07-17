"""Canonicalize native browser objects without vendor Draft/result shapes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from src.common.errors import DataContractError
from src.coco_refinement.models import CanonicalDraft, NativeObject, Split


_OBJECT_FIELDS = frozenset(
    {
        "region_key",
        "bbox_2d",
        "category_name",
        "category_id",
        "coco_ann_id",
        "metadata",
        "presentation",
    }
)
_OBJECT_REQUIRED_FIELDS = frozenset(
    {"region_key", "bbox_2d", "category_name", "category_id"}
)
_INFERENCE_METADATA_FIELDS = frozenset(
    {
        "inference_origin",
        "receipt_id",
        "request_id",
        "result_id",
        "draft_revision",
    }
)
_SOURCE_KEY = re.compile(r"^(train|val):coco:([1-9][0-9]*)$")
_ROI_KEY = re.compile(
    r"^roi:([A-Za-z0-9][A-Za-z0-9_.-]*):([A-Za-z0-9][A-Za-z0-9_.:-]*)$"
)


def canonicalize_objects(
    values: Sequence[Mapping[str, Any]], *, split: Split
) -> CanonicalDraft:
    """Validate one full native object list and derive stable/exact hashes."""

    if split not in ("train", "val"):
        raise DataContractError(
            "split must be train or val",
            code="coco_refinement.split",
            context={"split": split},
        )
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DataContractError(
            "Draft objects must be a JSON array",
            code="coco_refinement.objects_shape",
        )

    objects: list[NativeObject] = []
    region_keys: set[str] = set()
    coco_ann_ids: set[int] = set()
    receipts: set[str] = set()
    for index, value in enumerate(values):
        obj, receipt_id = _canonical_object(value, split=split, index=index)
        if obj.region_key in region_keys:
            raise DataContractError(
                "region_key values must be unique within a Draft",
                code="coco_refinement.region_key_duplicate",
                context={"region_key": obj.region_key},
            )
        region_keys.add(obj.region_key)
        if obj.coco_ann_id is not None:
            if obj.coco_ann_id in coco_ann_ids:
                raise DataContractError(
                    "coco_ann_id values must be unique within a Draft",
                    code="coco_refinement.coco_ann_id_duplicate",
                    context={"coco_ann_id": obj.coco_ann_id},
                )
            coco_ann_ids.add(obj.coco_ann_id)
        if receipt_id is not None:
            receipts.add(receipt_id)
        objects.append(obj)

    semantic_projection = sorted(
        (item.semantic_dict() for item in objects),
        key=lambda item: item["region_key"],
    )
    exact_payload = [item.to_json_dict() for item in objects]
    return CanonicalDraft(
        split=split,
        objects=tuple(objects),
        semantic_hash=_sha256_json(semantic_projection),
        result_hash=_sha256_json(exact_payload),
        inference_receipts=tuple(sorted(receipts)),
    )


def _canonical_object(
    value: Mapping[str, Any], *, split: Split, index: int
) -> tuple[NativeObject, str | None]:
    if not isinstance(value, Mapping):
        raise DataContractError(
            "Draft object must be a JSON object",
            code="coco_refinement.object_shape",
            context={"index": index},
        )
    fields = frozenset(str(key) for key in value)
    missing = _OBJECT_REQUIRED_FIELDS - fields
    extra = fields - _OBJECT_FIELDS
    if missing or extra:
        raise DataContractError(
            "Draft object fields do not match the native contract",
            code="coco_refinement.object_fields",
            context={"index": index, "missing": sorted(missing), "extra": sorted(extra)},
        )
    presentation = value.get("presentation")
    if presentation is not None and not isinstance(presentation, Mapping):
        raise DataContractError(
            "presentation must be an object when supplied",
            code="coco_refinement.presentation_shape",
            context={"index": index},
        )

    raw_metadata = value.get("metadata")
    metadata, receipt_id = _canonical_metadata(raw_metadata, index=index)
    coco_ann_id = value.get("coco_ann_id")
    if coco_ann_id is not None and (
        isinstance(coco_ann_id, bool) or not isinstance(coco_ann_id, int) or coco_ann_id == 0
    ):
        raise DataContractError(
            "coco_ann_id must be a nonzero integer when supplied",
            code="coco_refinement.coco_ann_id",
            context={"index": index, "coco_ann_id": coco_ann_id},
        )

    region_key = value["region_key"]
    if not isinstance(region_key, str):
        raise DataContractError(
            "region_key must be a string",
            code="coco_refinement.region_key",
            context={"index": index},
        )
    source_match = _SOURCE_KEY.fullmatch(region_key)
    roi_match = _ROI_KEY.fullmatch(region_key)
    if source_match is not None:
        key_split, raw_id = source_match.groups()
        source_id = int(raw_id)
        if key_split != split:
            raise DataContractError(
                "source region key belongs to another split",
                code="coco_refinement.region_split",
                context={"index": index, "region_key": region_key, "split": split},
            )
        if coco_ann_id != source_id:
            raise DataContractError(
                "source region key and positive coco_ann_id must match",
                code="coco_refinement.source_identity",
                context={"index": index, "region_key": region_key, "coco_ann_id": coco_ann_id},
            )
        if metadata:
            raise DataContractError(
                "source regions cannot carry inference provenance",
                code="coco_refinement.metadata_key",
                context={"index": index},
            )
    elif region_key.startswith("local:"):
        raw_uuid = region_key.removeprefix("local:")
        try:
            parsed_uuid = UUID(raw_uuid)
        except (ValueError, AttributeError) as exc:
            raise DataContractError(
                "local region keys require a canonical UUID",
                code="coco_refinement.local_key",
                context={"index": index, "region_key": region_key},
                cause=exc,
            ) from exc
        if str(parsed_uuid) != raw_uuid:
            raise DataContractError(
                "local region keys require a canonical lowercase UUID",
                code="coco_refinement.local_key",
                context={"index": index, "region_key": region_key},
            )
        _validate_new_identity(coco_ann_id, index=index)
        if metadata:
            raise DataContractError(
                "local human regions cannot carry inference provenance",
                code="coco_refinement.metadata_key",
                context={"index": index},
            )
    elif roi_match is not None:
        _validate_new_identity(coco_ann_id, index=index)
        key_receipt, key_result = roi_match.groups()
        if not metadata or receipt_id is None:
            raise DataContractError(
                "ROI regions require complete inference provenance",
                code="coco_refinement.roi_metadata",
                context={"index": index},
            )
        if receipt_id != key_receipt or metadata["result_id"] != key_result:
            raise DataContractError(
                "ROI region key must match receipt_id and result_id",
                code="coco_refinement.roi_identity",
                context={"index": index, "region_key": region_key},
            )
    else:
        raise DataContractError(
            "region_key must be source, local UUID, or ROI identity",
            code="coco_refinement.region_key",
            context={"index": index, "region_key": region_key},
        )

    return (
        NativeObject(
            region_key=region_key,
            bbox_2d=value["bbox_2d"],
            category_name=value["category_name"],
            category_id=value["category_id"],
            coco_ann_id=coco_ann_id,
            metadata=metadata or None,
        ),
        receipt_id,
    )


def _canonical_metadata(
    value: object, *, index: int
) -> tuple[dict[str, Any], str | None]:
    if value is None:
        return {}, None
    if not isinstance(value, Mapping):
        raise DataContractError(
            "metadata must be a JSON object",
            code="coco_refinement.metadata_shape",
            context={"index": index},
        )
    fields = frozenset(str(key) for key in value)
    if not fields:
        return {}, None
    if fields != _INFERENCE_METADATA_FIELDS:
        raise DataContractError(
            "metadata fields are not in the inference provenance allowlist",
            code="coco_refinement.metadata_fields",
            context={"index": index, "fields": sorted(fields)},
        )
    if value.get("inference_origin") is not True:
        raise DataContractError(
            "inference_origin must be true",
            code="coco_refinement.metadata_origin",
            context={"index": index},
        )
    normalized: dict[str, Any] = {"inference_origin": True}
    for field in ("receipt_id", "request_id", "result_id", "draft_revision"):
        field_value = value.get(field)
        if not isinstance(field_value, str) or not field_value:
            raise DataContractError(
                "inference metadata identifiers must be non-empty strings",
                code="coco_refinement.metadata_value",
                context={"index": index, "field": field},
            )
        normalized[field] = field_value
    return normalized, normalized["receipt_id"]


def _validate_new_identity(value: object, *, index: int) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value >= 0
    ):
        raise DataContractError(
            "local and ROI coco_ann_id values must be allocated negative IDs",
            code="coco_refinement.new_identity",
            context={"index": index, "coco_ann_id": value},
        )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["canonicalize_objects"]
