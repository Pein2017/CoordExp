"""Canonical Label Studio Draft conversion for the refinement runtime.

The vendor database owns durable Draft bytes.  This module validates one saved
Label Studio ``result`` array, converts rectangle percentages back to the
approved norm1000 lattice, and retains the complete result/meta payload inside
the immutable queue member.  It does not query Label Studio or mutate dataset
state.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.common.errors import DataContractError
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY, Coco80Registry
from src.label_studio_coco_refinement.geometry import label_studio_xywh_to_norm1000
from src.label_studio_coco_refinement.store import semantic_hash, sha256_json


class DraftContractError(ValueError):
    """A saved vendor Draft cannot enter the canonical batch boundary."""


@dataclass(frozen=True)
class CanonicalDraftPayload:
    """Canonical semantics plus full retained vendor payload attestation."""

    regions: tuple[Mapping[str, Any], ...]
    semantic_hash: str
    result_hash: str
    inference_receipts: tuple[str, ...]

    def to_json_regions(self) -> list[dict[str, Any]]:
        """Return a detached ordinary-JSON copy for a queue/catalog boundary."""

        return [_thaw_json(region) for region in self.regions]


def merge_stable_identity_metadata(
    results: Sequence[Mapping[str, Any]],
    *,
    region_id_mapping: Mapping[str, int],
    committed_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Merge terminal identity baselines into a possibly newer Draft.

    Stable region keys are the only join key.  Matching live results receive
    only the hidden committed ID and bbox baseline; geometry, class, result
    membership, inference/presentation metadata, ordering, and all other live
    payload bytes survive unchanged.  Committed regions absent from the live
    Draft are not recreated.
    """

    if not isinstance(region_id_mapping, Mapping):
        raise DraftContractError("region ID mapping must be a JSON object")
    if not isinstance(committed_row, Mapping):
        raise DraftContractError("committed row must be a JSON object")
    committed_objects = committed_row.get("objects")
    if not isinstance(committed_objects, list):
        raise DraftContractError("committed row objects must be a JSON array")

    normalized_mapping: dict[str, int] = {}
    mapped_ids: set[int] = set()
    for raw_key, raw_id in region_id_mapping.items():
        key = _normalized_text(raw_key, field="stable region key")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id == 0:
            raise DraftContractError("mapped coco_ann_id must be a nonzero integer")
        if raw_id in mapped_ids:
            raise DraftContractError("region ID mapping must contain unique IDs")
        normalized_mapping[key] = raw_id
        mapped_ids.add(raw_id)

    committed_bbox_by_id: dict[int, list[int]] = {}
    for ordinal, raw_object in enumerate(committed_objects):
        if not isinstance(raw_object, Mapping):
            raise DraftContractError(
                f"committed row object[{ordinal}] must be a JSON object"
            )
        object_id = raw_object.get("coco_ann_id")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id == 0
        ):
            raise DraftContractError(
                "committed object coco_ann_id must be a nonzero integer"
            )
        bbox = raw_object.get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > 999
                for value in bbox
            )
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            raise DraftContractError(
                "committed object bbox_2d must be strict norm1000 integers"
            )
        if object_id in committed_bbox_by_id:
            raise DraftContractError("committed row contains duplicate coco_ann_id")
        committed_bbox_by_id[object_id] = list(bbox)
    missing_ids = mapped_ids - set(committed_bbox_by_id)
    if missing_ids:
        raise DraftContractError(
            "region ID mapping references an object outside the committed row"
        )

    merged = _json_copy(list(results), field="Draft result")
    seen_keys: set[str] = set()
    for ordinal, result in enumerate(merged):
        if not isinstance(result, dict):
            raise DraftContractError(f"Draft result[{ordinal}] must be a JSON object")
        key = _normalized_text(result.get("id"), field="region id")
        if key in seen_keys:
            raise DraftContractError(f"duplicate region key: {key}")
        seen_keys.add(key)
        object_id = normalized_mapping.get(key)
        if object_id is None:
            continue
        meta = result.get("meta")
        if not isinstance(meta, dict):
            raise DraftContractError("rectangle meta must be a JSON object")
        prior_key = meta.get("coordexp_region_key")
        if prior_key not in (None, key):
            raise DraftContractError(
                "hidden region key does not match the Label Studio result id"
            )
        prior_id = meta.get("coco_ann_id")
        if prior_id not in (None, object_id):
            raise DraftContractError(
                "stable region key already carries a different coco_ann_id"
            )
        meta["coordexp_region_key"] = key
        meta["coco_ann_id"] = object_id
        meta["last_committed_bbox"] = list(committed_bbox_by_id[object_id])
    return merged


def canonicalize_label_studio_draft(
    results: Sequence[Mapping[str, Any]],
    *,
    split: str,
    image_id: int,
    image_width: int,
    image_height: int,
    registry: Coco80Registry = COCO80_REGISTRY,
) -> CanonicalDraftPayload:
    """Validate and freeze one authoritative Label Studio Draft result list.

    ``results`` may be empty so Label Studio can preserve a reviewed-empty
    Draft.  The working store independently rejects that payload at the V1
    Commit gate.
    """

    if split not in {"train", "val"}:
        raise DraftContractError("split must be train or val")
    _positive_integer(image_id, field="image_id")
    _positive_integer(image_width, field="image_width")
    _positive_integer(image_height, field="image_height")
    if not isinstance(results, Sequence) or isinstance(
        results, (str, bytes, bytearray)
    ):
        raise DraftContractError("Draft result must be a JSON array")

    raw_results = _json_copy(list(results), field="Draft result")
    canonical: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for ordinal, result in enumerate(raw_results):
        if not isinstance(result, Mapping):
            raise DraftContractError(f"Draft result[{ordinal}] must be a JSON object")
        canonical_region = _canonicalize_region(
            result,
            split=split,
            image_width=image_width,
            image_height=image_height,
            ordinal=ordinal,
            registry=registry,
        )
        key = canonical_region["region_key"]
        if key in seen_keys:
            raise DraftContractError(f"duplicate region key: {key}")
        seen_keys.add(key)
        canonical.append(canonical_region)

    canonical.sort(key=lambda region: region["region_key"])
    receipt_ids = tuple(
        sorted(
            {
                str(region["metadata"]["receipt_id"])
                for region in canonical
                if region.get("metadata", {}).get("inference_origin") is True
            }
        )
    )
    frozen_regions = tuple(_freeze_json(region) for region in canonical)

    return CanonicalDraftPayload(
        regions=frozen_regions,
        semantic_hash=semantic_hash(canonical),
        result_hash=sha256_json(raw_results),
        inference_receipts=receipt_ids,
    )


def _canonicalize_region(
    result: Mapping[str, Any],
    *,
    split: str,
    image_width: int,
    image_height: int,
    ordinal: int,
    registry: Coco80Registry,
) -> dict[str, Any]:
    if result.get("type") != "rectanglelabels":
        raise DraftContractError("only rectanglelabels Draft results are supported")
    if result.get("from_name") != "bbox" or result.get("to_name") != "image":
        raise DraftContractError(
            "rectangle result must use the fixed bbox/image binding"
        )
    if (
        result.get("original_width") != image_width
        or result.get("original_height") != image_height
    ):
        raise DraftContractError(
            "rectangle original dimensions do not match the task image"
        )
    if not _is_zero(result.get("image_rotation")):
        raise DraftContractError("image rotation must be zero")

    result_id = _normalized_text(result.get("id"), field="region id")
    meta = result.get("meta", {})
    if not isinstance(meta, Mapping):
        raise DraftContractError("rectangle meta must be a JSON object")
    meta_copy = _json_copy(dict(meta), field="rectangle meta")
    meta_key = meta_copy.get("coordexp_region_key")
    if meta_key is not None and meta_key != result_id:
        raise DraftContractError(
            "hidden region key does not match the Label Studio result id"
        )
    region_key = result_id

    value = result.get("value")
    if not isinstance(value, Mapping):
        raise DraftContractError("rectangle value must be a JSON object")
    if not _is_zero(value.get("rotation")):
        raise DraftContractError("bbox rotation must be zero")
    labels = value.get("rectanglelabels")
    if not isinstance(labels, list) or len(labels) != 1:
        raise DraftContractError("bbox must contain exactly one canonical class")
    try:
        category = registry.by_name(labels[0])
    except DataContractError as exc:
        raise DraftContractError(
            "bbox class must be an exact canonical COCO-80 name"
        ) from exc

    previous_bbox = meta_copy.get("last_committed_bbox")
    try:
        bbox = label_studio_xywh_to_norm1000(
            value.get("x"),
            value.get("y"),
            value.get("width"),
            value.get("height"),
            previous_bbox=previous_bbox,
        )
    except DataContractError as exc:
        raise DraftContractError(f"invalid rectangle geometry: {exc}") from exc

    coco_ann_id = meta_copy.get("coco_ann_id")
    if coco_ann_id is not None:
        if (
            isinstance(coco_ann_id, bool)
            or not isinstance(coco_ann_id, int)
            or coco_ann_id == 0
        ):
            raise DraftContractError(
                "coco_ann_id must be a nonzero integer when present"
            )
    source_prefix = f"{split}:coco:"
    if region_key.startswith(source_prefix):
        suffix = region_key.removeprefix(source_prefix)
        try:
            source_id = int(suffix)
        except ValueError as exc:
            raise DraftContractError(
                "source region key has an invalid coco_ann_id"
            ) from exc
        if source_id <= 0 or coco_ann_id not in (None, source_id):
            raise DraftContractError("source region key and coco_ann_id disagree")
        coco_ann_id = source_id
    elif coco_ann_id is not None and coco_ann_id > 0:
        raise DraftContractError(
            "positive coco_ann_id requires the canonical split:coco:<id> region key"
        )

    creation_ordinal = meta_copy.get("coordexp_creation_ordinal", ordinal)
    if (
        isinstance(creation_ordinal, bool)
        or not isinstance(creation_ordinal, int)
        or creation_ordinal < 0
    ):
        raise DraftContractError("creation ordinal must be a non-negative integer")

    canonical: dict[str, Any] = {
        "region_key": region_key,
        "bbox_2d": list(bbox),
        "category_name": category.name,
        "category_id": category.id,
        "creation_ordinal": creation_ordinal,
        "draft_meta": meta_copy,
        "label_studio_result": copy.deepcopy(result),
    }
    if coco_ann_id is not None:
        canonical["coco_ann_id"] = coco_ann_id
    training_metadata = meta_copy.get("coordexp_training_metadata")
    metadata: dict[str, Any] = {}
    if training_metadata is not None:
        if not isinstance(training_metadata, Mapping):
            raise DraftContractError("coordexp_training_metadata must be a JSON object")
        metadata = copy.deepcopy(dict(training_metadata))
        reserved = {
            "inference_origin",
            "receipt_id",
            "request_id",
            "result_id",
            "draft_revision",
        }
        if reserved & set(metadata):
            raise DraftContractError(
                "coordexp_training_metadata cannot override inference linkage"
            )

    inference_values = {
        "receipt_id": meta_copy.get("coordexp_inference_receipt_id"),
        "request_id": meta_copy.get("coordexp_inference_request_id"),
        "result_id": meta_copy.get("coordexp_inference_result_id"),
        "draft_revision": meta_copy.get("coordexp_inference_source_draft_revision"),
    }
    if any(value is not None for value in inference_values.values()):
        if any(value is None for value in inference_values.values()):
            raise DraftContractError(
                "inference-origin region requires receipt, request, result, and "
                "source Draft revision IDs"
            )
        for label, value in inference_values.items():
            _normalized_text(value, field=f"inference {label}")
        metadata.update({"inference_origin": True, **inference_values})
    if metadata:
        canonical["metadata"] = metadata
    return canonical


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DraftContractError(f"{field} must be finite ordinary JSON") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DraftContractError(f"{field} must be a positive integer")
    return value


def _normalized_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DraftContractError(f"{field} must be non-empty normalized text")
    return value


def _is_zero(value: Any) -> bool:
    return (
        not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0
    )


__all__ = [
    "CanonicalDraftPayload",
    "DraftContractError",
    "canonicalize_label_studio_draft",
    "merge_stable_identity_metadata",
]
