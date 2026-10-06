"""Canonical JSON, request codecs and pure row/identity validation.

These functions do not publish or recover working generations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .records import (
    BatchMember,
    BatchRequest,
    CategoryRegistry,
    CommitRequest,
    DraftSaveReceipt,
    RecoveryError,
    StoreError,
    ValidationError,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_nonfinite_json_constant)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_semantic_projection(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project Draft semantics while excluding UI order and view metadata."""

    projected: list[dict[str, Any]] = []
    for region in regions:
        key = _region_key(region)
        projected.append(
            {
                "region_key": key,
                "bbox_2d": list(region.get("bbox_2d", ())),
                "category_name": region.get("category_name", region.get("desc")),
                "category_id": region.get("category_id"),
                "coco_ann_id": region.get("coco_ann_id"),
            }
        )
    projected.sort(key=lambda item: item["region_key"])
    return projected


def semantic_hash(regions: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(canonical_semantic_projection(regions))


_BATCH_PAYLOAD_FIELDS = frozenset(
    {"batch_id", "split", "current_user_id", "base_generation", "members"}
)

_BATCH_MEMBER_PAYLOAD_FIELDS = frozenset({"source_row_index", "request"})

_COMMIT_REQUEST_PAYLOAD_FIELDS = frozenset(
    {
        "commit_id",
        "split",
        "image_id",
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "base_row_hash",
        "observed_generation",
        "regions",
        "draft_save",
        "inference_receipts",
    }
)

_DRAFT_SAVE_PAYLOAD_FIELDS = frozenset(
    {
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "durable",
    }
)

_QUEUE_ENQUEUE_RECORD_FIELDS = frozenset(
    {
        "kind",
        "batch_id",
        "payload_hash",
        "split",
        "base_generation",
        "member_count",
        "payload",
        "timestamp",
        "prev_record_hash",
        "record_hash",
    }
)

_QUEUE_CLAIM_RECORD_FIELDS = frozenset(
    {
        "kind",
        "batch_id",
        "payload_hash",
        "timestamp",
        "prev_record_hash",
        "record_hash",
    }
)

_QUEUE_TERMINAL_REQUIRED_RECORD_FIELDS = frozenset(
    {
        "kind",
        "batch_id",
        "payload_hash",
        "status",
        "generation",
        "working_sha256",
        "timestamp",
        "prev_record_hash",
        "record_hash",
    }
)

_QUEUE_TERMINAL_OPTIONAL_RECORD_FIELDS = frozenset({"error"})


def _payload_error(error_type: type[StoreError], path: str, message: str) -> None:
    raise error_type(f"{path}: {message}")


def _validate_queue_record_envelope(record: dict[str, Any]) -> None:
    """Validate the exact durable queue envelope before interpreting a record."""

    kind = record.get("kind")
    if kind == "enqueue":
        required = allowed = _QUEUE_ENQUEUE_RECORD_FIELDS
    elif kind == "claim":
        required = allowed = _QUEUE_CLAIM_RECORD_FIELDS
    elif kind == "queue_terminal":
        required = _QUEUE_TERMINAL_REQUIRED_RECORD_FIELDS
        allowed = required | _QUEUE_TERMINAL_OPTIONAL_RECORD_FIELDS
    else:
        raise RecoveryError(
            f"queue record kind is not supported by the canonical schema: {kind!r}"
        )

    actual = frozenset(record)
    missing = sorted(required - actual)
    unknown = sorted(actual - allowed, key=repr)
    if missing or unknown:
        raise RecoveryError(
            f"queue {kind} record fields do not match the canonical schema "
            f"(missing={missing}, unknown={unknown})"
        )
    if type(record["batch_id"]) is not str or not record["batch_id"].strip():
        raise RecoveryError("queue record has invalid batch identity")
    if not _is_sha256(record["payload_hash"]):
        raise RecoveryError("queue record has invalid payload hash")
    if type(record["timestamp"]) is not str or not record["timestamp"].strip():
        raise RecoveryError("queue record has invalid timestamp")
    previous_hash = record["prev_record_hash"]
    if previous_hash is not None and not _is_sha256(previous_hash):
        raise RecoveryError("queue record has invalid previous hash")
    if not _is_sha256(record["record_hash"]):
        raise RecoveryError("queue record has invalid record hash")


def _validate_ordinary_json(
    value: Any,
    *,
    path: str,
    error_type: type[StoreError],
) -> None:
    """Require values whose type and value survive a JSON round trip exactly."""

    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _payload_error(error_type, path, "number must be finite ordinary JSON")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_ordinary_json(
                item,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _payload_error(error_type, path, "JSON object keys must be strings")
            _validate_ordinary_json(
                item,
                path=f"{path}.{key}",
                error_type=error_type,
            )
        return
    _payload_error(
        error_type,
        path,
        f"value is not canonical ordinary JSON: {type(value).__name__}",
    )


def _require_payload_object(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
    error_type: type[StoreError],
) -> dict[str, Any]:
    if type(value) is not dict:
        _payload_error(error_type, path, "must be a JSON object")
    actual = frozenset(value)
    if actual != fields:
        _payload_error(
            error_type,
            path,
            "fields do not match the canonical schema "
            f"(missing={sorted(fields - actual)}, unknown={sorted(actual - fields)})",
        )
    return value


def _require_payload_string(
    value: Any,
    *,
    path: str,
    error_type: type[StoreError],
    sha256: bool = False,
    trimmed: bool = False,
) -> str:
    if type(value) is not str or not value.strip():
        _payload_error(error_type, path, "must be non-empty text")
    if trimmed and value != value.strip():
        _payload_error(error_type, path, "must be trimmed text")
    if sha256 and not _is_sha256(value):
        _payload_error(error_type, path, "must be a lowercase SHA-256 digest")
    return value


def _require_payload_integer(
    value: Any,
    *,
    path: str,
    error_type: type[StoreError],
) -> int:
    if type(value) is not int or value < 0:
        _payload_error(error_type, path, "must be a non-negative integer")
    return value


def _validate_commit_request_payload(
    payload: Any,
    *,
    path: str,
    error_type: type[StoreError],
) -> dict[str, Any]:
    request = _require_payload_object(
        payload,
        fields=_COMMIT_REQUEST_PAYLOAD_FIELDS,
        path=path,
        error_type=error_type,
    )
    for field_name in (
        "commit_id",
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
    ):
        _require_payload_string(
            request[field_name],
            path=f"{path}.{field_name}",
            error_type=error_type,
        )
    split = _require_payload_string(
        request["split"], path=f"{path}.split", error_type=error_type
    )
    if split not in {"train", "val"}:
        _payload_error(error_type, f"{path}.split", "must be 'train' or 'val'")
    _require_payload_integer(
        request["image_id"], path=f"{path}.image_id", error_type=error_type
    )
    _require_payload_integer(
        request["observed_generation"],
        path=f"{path}.observed_generation",
        error_type=error_type,
    )
    for field_name in ("annotation_revision", "draft_updated_at"):
        _require_payload_string(
            request[field_name],
            path=f"{path}.{field_name}",
            error_type=error_type,
        )
    for field_name in ("semantic_hash", "result_hash", "base_row_hash"):
        _require_payload_string(
            request[field_name],
            path=f"{path}.{field_name}",
            error_type=error_type,
            sha256=True,
        )
    regions = request["regions"]
    if type(regions) is not list:
        _payload_error(error_type, f"{path}.regions", "must be a JSON array")
    for index, region in enumerate(regions):
        if type(region) is not dict:
            _payload_error(
                error_type,
                f"{path}.regions[{index}]",
                "must be a JSON object",
            )

    receipt_path = f"{path}.draft_save"
    receipt = _require_payload_object(
        request["draft_save"],
        fields=_DRAFT_SAVE_PAYLOAD_FIELDS,
        path=receipt_path,
        error_type=error_type,
    )
    for field_name in (
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
    ):
        _require_payload_string(
            receipt[field_name],
            path=f"{receipt_path}.{field_name}",
            error_type=error_type,
        )
    for field_name in ("semantic_hash", "result_hash"):
        _require_payload_string(
            receipt[field_name],
            path=f"{receipt_path}.{field_name}",
            error_type=error_type,
            sha256=True,
        )
    if type(receipt["durable"]) is not bool:
        _payload_error(
            error_type,
            f"{receipt_path}.durable",
            "must be a boolean",
        )

    inference_receipts = request["inference_receipts"]
    if type(inference_receipts) is not list:
        _payload_error(
            error_type,
            f"{path}.inference_receipts",
            "must be a JSON array",
        )
    for index, receipt_id in enumerate(inference_receipts):
        _require_payload_string(
            receipt_id,
            path=f"{path}.inference_receipts[{index}]",
            error_type=error_type,
        )
    return request


def _validate_batch_payload(
    payload: Any,
    *,
    error_type: type[StoreError],
) -> dict[str, Any]:
    _validate_ordinary_json(payload, path="batch payload", error_type=error_type)
    batch = _require_payload_object(
        payload,
        fields=_BATCH_PAYLOAD_FIELDS,
        path="batch payload",
        error_type=error_type,
    )
    _require_payload_string(
        batch["batch_id"], path="batch payload.batch_id", error_type=error_type
    )
    split = _require_payload_string(
        batch["split"], path="batch payload.split", error_type=error_type
    )
    if split not in {"train", "val"}:
        _payload_error(error_type, "batch payload.split", "must be 'train' or 'val'")
    _require_payload_string(
        batch["current_user_id"],
        path="batch payload.current_user_id",
        error_type=error_type,
        trimmed=True,
    )
    _require_payload_integer(
        batch["base_generation"],
        path="batch payload.base_generation",
        error_type=error_type,
    )
    members = batch["members"]
    if type(members) is not list or not members:
        _payload_error(
            error_type, "batch payload.members", "must be a non-empty JSON array"
        )
    source_row_indices: list[int] = []
    for index, value in enumerate(members):
        member_path = f"batch payload.members[{index}]"
        member = _require_payload_object(
            value,
            fields=_BATCH_MEMBER_PAYLOAD_FIELDS,
            path=member_path,
            error_type=error_type,
        )
        source_row_indices.append(
            _require_payload_integer(
                member["source_row_index"],
                path=f"{member_path}.source_row_index",
                error_type=error_type,
            )
        )
        _validate_commit_request_payload(
            member["request"],
            path=f"{member_path}.request",
            error_type=error_type,
        )
    if len(set(source_row_indices)) != len(source_row_indices):
        _payload_error(
            error_type,
            "batch payload.members",
            "duplicate source row index",
        )
    if source_row_indices != sorted(source_row_indices):
        _payload_error(
            error_type,
            "batch payload.members",
            "source row indices must be ascending",
        )
    return batch


def _commit_request_payload(request: CommitRequest) -> dict[str, Any]:
    if not isinstance(request, CommitRequest):
        raise ValidationError("batch payload member request has an invalid type")
    receipt = request.draft_save
    if not isinstance(receipt, DraftSaveReceipt):
        raise ValidationError("batch payload Draft-save receipt has an invalid type")
    if isinstance(request.regions, (str, bytes, bytearray)) or not isinstance(
        request.regions, Sequence
    ):
        raise ValidationError("batch payload regions must be a sequence of objects")
    if any(type(region) is not dict for region in request.regions):
        raise ValidationError("batch payload regions must contain JSON objects")
    if isinstance(
        request.inference_receipts, (str, bytes, bytearray)
    ) or not isinstance(request.inference_receipts, Sequence):
        raise ValidationError(
            "batch payload inference receipts must be a sequence of strings"
        )
    return {
        "commit_id": request.commit_id,
        "split": request.split,
        "image_id": request.image_id,
        "project_id": request.project_id,
        "task_id": request.task_id,
        "annotation_id": request.annotation_id,
        "draft_id": request.draft_id,
        "annotation_revision": request.annotation_revision,
        "draft_updated_at": request.draft_updated_at,
        "semantic_hash": request.semantic_hash,
        "result_hash": request.result_hash,
        "base_row_hash": request.base_row_hash,
        "observed_generation": request.observed_generation,
        "regions": copy.deepcopy(list(request.regions)),
        "draft_save": {
            "project_id": receipt.project_id,
            "task_id": receipt.task_id,
            "annotation_id": receipt.annotation_id,
            "draft_id": receipt.draft_id,
            "annotation_revision": receipt.annotation_revision,
            "draft_updated_at": receipt.draft_updated_at,
            "semantic_hash": receipt.semantic_hash,
            "result_hash": receipt.result_hash,
            "durable": receipt.durable,
        },
        "inference_receipts": list(request.inference_receipts),
    }


def _commit_request_from_payload(
    payload: Any,
    *,
    error_type: type[StoreError] = ValidationError,
) -> CommitRequest:
    request = _validate_commit_request_payload(
        payload,
        path="commit request payload",
        error_type=error_type,
    )
    receipt = request["draft_save"]
    return CommitRequest(
        commit_id=request["commit_id"],
        split=request["split"],
        image_id=request["image_id"],
        project_id=request["project_id"],
        task_id=request["task_id"],
        annotation_id=request["annotation_id"],
        draft_id=request["draft_id"],
        annotation_revision=request["annotation_revision"],
        draft_updated_at=request["draft_updated_at"],
        semantic_hash=request["semantic_hash"],
        result_hash=request["result_hash"],
        base_row_hash=request["base_row_hash"],
        observed_generation=request["observed_generation"],
        regions=copy.deepcopy(request["regions"]),
        draft_save=DraftSaveReceipt(
            project_id=receipt["project_id"],
            task_id=receipt["task_id"],
            annotation_id=receipt["annotation_id"],
            draft_id=receipt["draft_id"],
            annotation_revision=receipt["annotation_revision"],
            draft_updated_at=receipt["draft_updated_at"],
            semantic_hash=receipt["semantic_hash"],
            result_hash=receipt["result_hash"],
            durable=receipt["durable"],
        ),
        inference_receipts=tuple(request["inference_receipts"]),
    )


def _batch_request_from_payload(
    payload: Any,
    *,
    error_type: type[StoreError] = ValidationError,
) -> BatchRequest:
    batch = _validate_batch_payload(payload, error_type=error_type)
    return BatchRequest(
        batch_id=batch["batch_id"],
        split=batch["split"],
        current_user_id=batch["current_user_id"],
        base_generation=batch["base_generation"],
        members=tuple(
            BatchMember(
                source_row_index=member["source_row_index"],
                request=_commit_request_from_payload(
                    member["request"], error_type=error_type
                ),
            )
            for member in batch["members"]
        ),
    )


def _request_identity_payload(
    request: CommitRequest,
    materialized_region_projection: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = request.draft_save
    return {
        "split": request.split,
        "image_id": request.image_id,
        "project_id": request.project_id,
        "task_id": request.task_id,
        "annotation_id": request.annotation_id,
        "draft_id": request.draft_id,
        "annotation_revision": request.annotation_revision,
        "draft_updated_at": request.draft_updated_at,
        "semantic_hash": request.semantic_hash,
        "result_hash": request.result_hash,
        "base_row_hash": request.base_row_hash,
        "observed_generation": request.observed_generation,
        "draft_save": {
            "project_id": receipt.project_id,
            "task_id": receipt.task_id,
            "annotation_id": receipt.annotation_id,
            "draft_id": receipt.draft_id,
            "annotation_revision": receipt.annotation_revision,
            "draft_updated_at": receipt.draft_updated_at,
            "semantic_hash": receipt.semantic_hash,
            "result_hash": receipt.result_hash,
            "durable": receipt.durable,
        },
        "inference_receipts": list(request.inference_receipts),
        "materialized_region_projection": copy.deepcopy(materialized_region_projection),
        "materialized_region_projection_hash": sha256_json(
            materialized_region_projection
        ),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _region_key(region: Mapping[str, Any]) -> str:
    value = region.get("region_key", region.get("stable_region_key"))
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("every Draft region requires a hidden stable region key")
    return value


def _source_region_key(split: str, object_id: int) -> str:
    return f"{split}:coco:{object_id}"


def _source_id_from_region_key(region_key: str, split: str) -> int | None:
    prefix = f"{split}:coco:"
    if not region_key.startswith(prefix):
        if ":coco:" in region_key or region_key.startswith("coco:"):
            raise ValidationError(
                "source region keys must use the current split:coco:<id> namespace"
            )
        return None
    try:
        object_id = int(region_key[len(prefix) :])
    except ValueError as exc:
        raise ValidationError(
            "coco region keys must end in a positive integer"
        ) from exc
    if object_id <= 0:
        raise ValidationError("coco region keys must end in a positive integer")
    return object_id


def _validate_bbox(value: Any) -> list[int]:
    from src.data.geometry import validate_bbox_bins

    try:
        return list(validate_bbox_bins(value, field="bbox_2d"))
    except Exception as exc:
        raise ValidationError(str(exc)) from exc


def _stable_order(
    values: Sequence[tuple[dict[str, Any], str, int | None, int | None]],
) -> list[dict[str, Any]]:
    from .models import (
        ObjectIdentity,
        OrderedWorkingObject,
        WorkingObject,
        stable_top_left_order,
    )

    ordered = stable_top_left_order(
        tuple(
            OrderedWorkingObject(
                identity=ObjectIdentity(
                    region_key=key,
                    coco_ann_id=int(obj["coco_ann_id"]),
                    prior_rank=prior_rank,
                    creation_ordinal=creation_ordinal,
                ),
                object=WorkingObject.from_mapping(obj),
            )
            for obj, key, prior_rank, creation_ordinal in values
        )
    )
    return [item.object.to_json_dict() for item in ordered]


def _training_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("metadata must be a mapping")
    _validate_ordinary_json(
        value,
        path="metadata",
        error_type=ValidationError,
    )
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in _PRESENTATION_METADATA_FIELDS
    }


def _validate_split(split: str) -> str:
    if split not in {"train", "val"}:
        raise ValidationError("split must be 'train' or 'val'")
    return split


def _task_id(split: str, image_id: int) -> str:
    return f"{split}:{image_id}"


def _parse_working_jsonl_row(raw: bytes, index: int) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise ValidationError("working JSONL must end every row with a newline")
    try:
        row = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"invalid working JSONL row {index + 1}") from exc
    if not isinstance(row, dict):
        raise ValidationError(f"working JSONL row {index + 1} is not an object")
    return row


def _parse_jsonl_line(line: str, source: Path, line_no: int) -> dict[str, Any]:
    if not line.endswith("\n"):
        raise ValidationError(
            f"source row {line_no} in {source} has no trailing newline"
        )
    try:
        row = _strict_json_loads(line)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"invalid source JSONL row {line_no}") from exc
    if not isinstance(row, dict):
        raise ValidationError(f"source row {line_no} is not an object")
    return row


def _validate_source_row(
    row: Mapping[str, Any], split: str, registry: CategoryRegistry
) -> None:
    required = {"images", "objects", "width", "height", "image_id", "file_name"}
    if not required.issubset(row):
        raise ValidationError(
            f"source row missing fields: {sorted(required - set(row))}"
        )
    if not isinstance(row["images"], list) or len(row["images"]) != 1:
        raise ValidationError("source row must contain exactly one image locator")
    if not isinstance(row["objects"], list) or not row["objects"]:
        raise ValidationError("source row objects must be non-empty")
    ids: set[int] = set()
    for obj in row["objects"]:
        _validate_bbox(obj.get("bbox_2d"))
        name = obj.get("category_name")
        category_id = obj.get("category_id")
        if (
            obj.get("desc") != name
            or not isinstance(name, str)
            or not isinstance(category_id, int)
        ):
            raise ValidationError("source class fields are inconsistent")
        registry.validate(name, category_id)
        object_id = obj.get("coco_ann_id")
        if not isinstance(object_id, int) or object_id <= 0 or object_id in ids:
            raise ValidationError("source coco_ann_id must be unique and positive")
        ids.add(object_id)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("split") not in (None, split):
        raise ValidationError("source metadata split mismatch")


def _register_split_wide_ids(
    row: Mapping[str, Any],
    object_owners: dict[int, int],
    image_ids: set[int],
    *,
    error_type: type[StoreError],
) -> None:
    image_id = int(row["image_id"])
    if image_id in image_ids:
        raise error_type(f"split-wide duplicate image_id: {image_id}")
    image_ids.add(image_id)
    for obj in row["objects"]:
        object_id = int(obj["coco_ann_id"])
        prior_image = object_owners.setdefault(object_id, image_id)
        if prior_image != image_id:
            raise error_type(f"split-wide duplicate coco_ann_id: {object_id}")


_PRESENTATION_METADATA_FIELDS = {
    "color",
    "visual_color",
    "visual_group",
    "visual_badge",
    "visual_policy",
}
