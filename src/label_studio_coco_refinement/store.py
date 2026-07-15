"""Durable batch-first mutable JSONL store for the COCO refinement workflow.

The source JSONL is never a transaction target.  A store owns one derived
split directory, freezes immutable batches in an append-only queue, and
serializes whole-batch publication through a non-blocking advisory lock.  The
append-only journal is the recovery authority; the manifest is published only
after the complete working JSONL is durable.  Single-sample Commit remains a
compatibility path over the same publication authority.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


SCHEMA_VERSION = 2
_ACCEPTED_OBJECT_FIELDS = {
    "bbox_2d",
    "desc",
    "category_id",
    "category_name",
    "coco_ann_id",
    "metadata",
}
_PRESENTATION_METADATA_FIELDS = {
    "color",
    "visual_color",
    "visual_group",
    "visual_badge",
    "visual_policy",
}


class StoreError(RuntimeError):
    """Base error for a working dataset store."""


class StoreBusyError(StoreError):
    """The split is already owned by another commit/recovery operation."""


class StaleCommitError(StoreError):
    """The request does not describe the currently authoritative base."""


class CommitConflictError(StoreError):
    """A commit id was reused for a different semantic snapshot."""


class CommitRolledBackError(StoreError):
    """The idempotent commit id has a definite rolled-back outcome."""


class CommitOutcomeUnknown(StoreError):
    """The working file may have been replaced; status lookup is required."""


class RecoveryError(StoreError):
    """Journal, working file, and manifest cannot be reconciled safely."""


class ManifestDriftError(StoreError):
    """An existing project does not match its immutable bootstrap contract."""


class ValidationError(StoreError):
    """A Draft or working row violates the store boundary."""


class InjectedCrash(BaseException):
    """Test-only process-stop analogue raised by :class:`FaultInjector`."""


class CategoryRegistry(Protocol):
    """Narrow sibling-registry boundary; the store owns no COCO ID table."""

    def validate(self, category_name: str, category_id: int) -> Any: ...


class AuthoritativeAnnotationVerifier(Protocol):
    """Legacy per-Draft lookup; adapters may additionally expose verify_batch."""

    def verify(self, identity: "AuthoritativeDraftIdentity") -> bool: ...


class InferenceReceiptResolver(Protocol):
    """Resolve one durable ROI receipt into its immutable target/result links."""

    def resolve(self, receipt_id: str) -> "InferenceReceiptLink | None": ...


FaultInjector = Callable[[str], None]


class CommitStatus(str, Enum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    OUTCOME_UNKNOWN = "outcome_unknown"
    NOT_FOUND = "not_found"


class BatchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RECONCILING = "reconciling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class DraftSaveReceipt:
    """Proof returned only after Label Studio durably saves the frozen Draft."""

    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: str
    draft_updated_at: str
    semantic_hash: str
    result_hash: str
    durable: bool = True


@dataclass(frozen=True)
class CommitRequest:
    commit_id: str
    split: str
    image_id: int
    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: str
    draft_updated_at: str
    semantic_hash: str
    result_hash: str
    base_row_hash: str
    observed_generation: int
    regions: Sequence[Mapping[str, Any]]
    draft_save: DraftSaveReceipt
    inference_receipts: Sequence[str] = ()


@dataclass(frozen=True)
class BatchMember:
    """One frozen task snapshot at its immutable zero-based source row."""

    source_row_index: int
    request: CommitRequest


@dataclass(frozen=True)
class BatchRequest:
    batch_id: str
    split: str
    current_user_id: str
    base_generation: int
    members: Sequence[BatchMember]


@dataclass(frozen=True)
class BatchEnqueueReceipt:
    batch_id: str
    payload_hash: str
    status: BatchStatus
    split: str
    member_count: int
    base_generation: int


@dataclass(frozen=True)
class BatchStatusView:
    batch_id: str
    payload_hash: str | None
    status: BatchStatus
    split: str | None
    member_count: int
    base_generation: int | None
    generation: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    payload_hash: str
    status: BatchStatus
    split: str
    generation: int
    working_sha256: str
    members: Sequence[CommitResult]
    error: str | None = None


@dataclass(frozen=True)
class AuthoritativeDraftIdentity:
    split: str
    image_id: int
    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: str
    draft_updated_at: str
    semantic_hash: str
    result_hash: str

    @classmethod
    def from_request(cls, request: CommitRequest) -> "AuthoritativeDraftIdentity":
        return cls(
            split=request.split,
            image_id=request.image_id,
            project_id=request.project_id,
            task_id=request.task_id,
            annotation_id=request.annotation_id,
            draft_id=request.draft_id,
            annotation_revision=request.annotation_revision,
            draft_updated_at=request.draft_updated_at,
            semantic_hash=request.semantic_hash,
            result_hash=request.result_hash,
        )


@dataclass(frozen=True)
class InferenceReceiptLink:
    receipt_id: str
    request_id: str
    project_id: str
    task_id: str
    image_id: int
    annotation_id: str
    current_user_id: str
    draft_id: str
    draft_revision: str
    terminal_status: str
    result_region_keys: Mapping[str, str]


@dataclass(frozen=True)
class CommitResult:
    commit_id: str
    status: CommitStatus
    split: str
    image_id: int
    generation: int
    row_hash: str
    semantic_hash: str
    region_id_mapping: Mapping[str, int]
    committed_row: Mapping[str, Any]


@dataclass(frozen=True)
class DraftRestore:
    split: str
    image_id: int
    generation: int
    row_hash: str
    row: Mapping[str, Any]
    region_id_mapping: Mapping[str, int]


@dataclass(frozen=True)
class BootstrapSpec:
    split: str
    source_path: Path
    runtime_root: Path
    image_root: Path
    expected_source_sha256: str
    project_id: str
    storage_id: str
    adapter_version: str
    vendor_revision: str
    registry_fingerprint: str
    label_config_fingerprint: str
    instance_id: str = "label-studio-coco-refinement"
    storage_subdir: str | None = None
    extra_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSeed:
    """Parent-adapter seed: exactly one editable annotation, no prediction."""

    task_id: str
    split: str
    image_id: int
    source_row_index: int
    image_locator: str
    authoritative_annotation_id: str
    annotations: Sequence[Mapping[str, Any]]
    predictions: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class BootstrapResult:
    store: "WorkingDatasetStore"
    created: bool
    task_count: int
    task_manifest_hash: str


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


class WorkingDatasetStore:
    """One split's canonical mutable ``working.norm.jsonl`` authority."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        annotation_verifier: AuthoritativeAnnotationVerifier,
        inference_receipt_resolver: InferenceReceiptResolver,
        registry: CategoryRegistry | None = None,
        fault_injector: FaultInjector | None = None,
        recover: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir).resolve()
        self.working_path = self.split_dir / "working.norm.jsonl"
        self.manifest_path = self.split_dir / "project.json"
        self.journal_path = self.split_dir / "journal.jsonl"
        self.queue_path = self.split_dir / "queue.jsonl"
        self.task_index_path = self.split_dir / "task_index.json"
        self.lock_path = self.split_dir / ".commit.lock"
        self.queue_lock_path = self.split_dir / ".queue.lock"
        self.registry = registry if registry is not None else _default_registry()
        self.annotation_verifier = annotation_verifier
        self.inference_receipt_resolver = inference_receipt_resolver
        self._fault_injector = fault_injector
        self._records: list[dict[str, Any]] = []
        self._region_to_id: dict[tuple[int, str], int] = {}
        self._id_to_region: dict[int, tuple[int, str]] = {}
        self._tombstones: set[int] = set()
        self._reserved_negative_ids: set[int] = set()
        self._batch_reservations: dict[str, tuple[int, int]] = {}
        self._source_row_by_image: dict[int, int] = {}
        self._row_image_cache: dict[int, int] = {}
        self._row_object_ids_cache: dict[int, set[int]] = {}
        self._row_hash_cache: dict[int, str] = {}
        self._cache_generation = -1
        self._cache_working_sha256 = ""
        self._cache_line_count = 0
        self._recovery_required = False
        if recover:
            self.recover()
        else:
            self._reload_journal_index()
        self._load_task_index()
        self._load_row_cache()

    @classmethod
    def bootstrap(
        cls,
        spec: BootstrapSpec,
        *,
        annotation_verifier: AuthoritativeAnnotationVerifier,
        inference_receipt_resolver: InferenceReceiptResolver,
        registry: CategoryRegistry | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> BootstrapResult:
        """Idempotently create one derived split without mutating the source.

        Label Studio API publication remains an adapter concern.  The returned
        :meth:`iter_task_seeds` surface provides stable ``(split, image_id)``
        task identities and exactly one authoritative editable annotation.
        """

        split = _validate_split(spec.split)
        registry = registry if registry is not None else _default_registry()
        source = spec.source_path.resolve(strict=True)
        _validate_selected_source(source, split)
        image_root = spec.image_root.resolve(strict=True)
        if not source.is_file() or not image_root.is_dir():
            raise ValidationError(
                "source_path must be a file and image_root a directory"
            )
        actual_source_hash = sha256_file(source)
        if actual_source_hash != spec.expected_source_sha256:
            raise ManifestDriftError("source_sha256")

        split_dir = (spec.runtime_root / split).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)
        images_link = split_dir / "images"
        static_contract = {
            "schema_version": SCHEMA_VERSION,
            "instance_id": spec.instance_id,
            "split": split,
            "source_path": str(source),
            "source_sha256": actual_source_hash,
            "image_root": str(image_root),
            "document_root": str(image_root),
            "managed_image_link": str(images_link),
            "project_id": spec.project_id,
            "storage": {
                "storage_id": spec.storage_id,
                "subdir": spec.storage_subdir or f"{split}2017",
                "project_bound": True,
            },
            "adapter_version": spec.adapter_version,
            "vendor_revision": spec.vendor_revision,
            "registry_fingerprint": spec.registry_fingerprint,
            "label_config_fingerprint": spec.label_config_fingerprint,
            "extra_fingerprints": dict(spec.extra_fingerprints),
            "task_policy": {
                "identity": "(split,image_id)",
                "authoritative_annotations": 1,
                "predictions": 0,
                "alternate_annotations_enabled": False,
                "native_submit_enabled": False,
                "native_skip_enabled": False,
            },
        }

        if (split_dir / "project.json").exists():
            store = cls(
                split_dir,
                annotation_verifier=annotation_verifier,
                inference_receipt_resolver=inference_receipt_resolver,
                registry=registry,
                fault_injector=fault_injector,
            )
            manifest = store._read_manifest()
            for key, expected in static_contract.items():
                if manifest.get(key) != expected:
                    raise ManifestDriftError(key)
            _validate_managed_link(images_link, image_root)
            if sha256_file(store.working_path) != manifest["working_sha256"]:
                raise ManifestDriftError("working_sha256")
            return BootstrapResult(
                store=store,
                created=False,
                task_count=int(manifest["task_count"]),
                task_manifest_hash=str(manifest["task_manifest_hash"]),
            )

        unexpected = [
            path.name
            for path in split_dir.iterdir()
            if path.name not in {".commit.lock", ".queue.lock"}
        ]
        if unexpected:
            raise ManifestDriftError(f"partial bootstrap state: {sorted(unexpected)}")
        os.symlink(image_root, images_link, target_is_directory=True)
        _validate_managed_link(images_link, image_root)

        task_digest = hashlib.sha256()
        task_count = 0
        split_object_ids: dict[int, int] = {}
        split_image_ids: set[int] = set()
        task_index_entries: list[dict[str, Any]] = []
        working_tmp: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".working.norm.jsonl.", dir=split_dir
            )
            working_tmp = Path(tmp_name)
            with (
                os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output,
                source.open("r", encoding="utf-8") as input_handle,
            ):
                for line_no, line in enumerate(input_handle, start=1):
                    row = _parse_jsonl_line(line, source, line_no)
                    _validate_source_row(row, split, registry)
                    image_id = int(row["image_id"])
                    if image_id in split_image_ids:
                        raise ValidationError(
                            f"split-wide duplicate image_id: {image_id}"
                        )
                    split_image_ids.add(image_id)
                    for obj in row["objects"]:
                        object_id = int(obj["coco_ann_id"])
                        prior_image = split_object_ids.setdefault(object_id, image_id)
                        if prior_image != image_id:
                            raise ValidationError(
                                f"split-wide duplicate coco_ann_id: {object_id}"
                            )
                    row = copy.deepcopy(row)
                    row["images"] = [
                        _working_image_locator(row, split, source, image_root)
                    ]
                    encoded = canonical_json(row) + "\n"
                    output.write(encoded)
                    seed_identity = {
                        "task_id": _task_id(split, image_id),
                        "image_id": image_id,
                        "row_hash": sha256_json(row),
                        "source_line": line_no,
                    }
                    task_digest.update(
                        (canonical_json(seed_identity) + "\n").encode("utf-8")
                    )
                    task_index_entries.append(
                        {
                            "source_row_index": task_count,
                            "source_line": line_no,
                            "image_id": image_id,
                            "task_id": _task_id(split, image_id),
                        }
                    )
                    task_count += 1
                output.flush()
                os.fsync(output.fileno())
            if task_count == 0:
                raise ValidationError("source JSONL must contain at least one row")
            os.replace(working_tmp, split_dir / "working.norm.jsonl")
            _fsync_directory(split_dir)
            working_tmp = None

            task_index = {
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "entries": task_index_entries,
            }
            _atomic_replace_json(split_dir / "task_index.json", task_index)
            (split_dir / "journal.jsonl").touch(exist_ok=False)
            with (split_dir / "journal.jsonl").open("ab") as journal:
                journal.flush()
                os.fsync(journal.fileno())
            (split_dir / "queue.jsonl").touch(exist_ok=False)
            with (split_dir / "queue.jsonl").open("ab") as queue:
                queue.flush()
                os.fsync(queue.fileno())
            manifest = {
                **static_contract,
                "task_count": task_count,
                "task_manifest_hash": task_digest.hexdigest(),
                "task_index_sha256": sha256_file(split_dir / "task_index.json"),
                "generation": 0,
                "working_sha256": sha256_file(split_dir / "working.norm.jsonl"),
                "working_line_count": task_count,
                "last_commit_id": None,
                "last_batch_id": None,
            }
            _atomic_replace_json(split_dir / "project.json", manifest)
        except BaseException:
            if working_tmp is not None:
                working_tmp.unlink(missing_ok=True)
            raise

        store = cls(
            split_dir,
            annotation_verifier=annotation_verifier,
            inference_receipt_resolver=inference_receipt_resolver,
            registry=registry,
            fault_injector=fault_injector,
        )
        return BootstrapResult(
            store=store,
            created=True,
            task_count=task_count,
            task_manifest_hash=task_digest.hexdigest(),
        )

    def iter_task_seeds(self) -> Iterator[TaskSeed]:
        with self._supported_reader_lock():
            manifest = self._read_manifest()
            split = str(manifest["split"])
            for source_row_index, row, _ in self._iter_rows():
                image_id = int(row["image_id"])
                task_id = _task_id(split, image_id)
                regions = []
                for obj in row["objects"]:
                    obj_copy = copy.deepcopy(obj)
                    obj_copy["region_key"] = self._key_for_object(
                        obj_copy, image_id, split
                    )
                    regions.append(obj_copy)
                yield TaskSeed(
                    task_id=task_id,
                    split=split,
                    image_id=image_id,
                    source_row_index=source_row_index,
                    image_locator=str(row["images"][0]),
                    authoritative_annotation_id=f"{task_id}:annotation",
                    annotations=({"id": f"{task_id}:annotation", "regions": regions},),
                )

    def resolve_source_row_index(
        self,
        *,
        split: str,
        project_id: str,
        task_id: str,
        image_id: int,
    ) -> int:
        """Resolve one server-owned task identity to its immutable row index.

        Batch adapters use this instead of accepting a browser-provided row
        position.  The supported-reader barrier also makes manifest/task-index
        reconciliation part of the lookup authority.
        """

        with self._supported_reader_lock():
            manifest = self._read_manifest()
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id < 0
            ):
                raise StaleCommitError("invalid task image identity")
            if split != manifest.get("split"):
                raise StaleCommitError("split mismatch")
            if project_id != manifest.get("project_id"):
                raise StaleCommitError("project mismatch")
            expected_task_id = _task_id(split, image_id)
            if task_id != expected_task_id:
                raise StaleCommitError("task identity mismatch")
            source_row_index = self._source_row_by_image.get(image_id)
            if source_row_index is None:
                raise StaleCommitError("unknown task image identity")
            return source_row_index

    @contextmanager
    def committed_generation_guard(self) -> Iterator[None]:
        """Hold one fully reconciled generation for a complete-output consumer."""

        self._assert_serving_ready()
        with self._exclusive_lock():
            with self._shared_queue_lock():
                queue_records = self._read_queue_records()
                reason = self._batch_reconciliation_reason(queue_records)
                if reason is not None:
                    raise RecoveryError(f"store requires recovery: {reason}")
                yield

    def enqueue_batch(self, request: BatchRequest) -> BatchEnqueueReceipt:
        """Durably freeze a bounded same-split batch without publishing JSONL."""

        self._assert_serving_ready()
        if not isinstance(request.batch_id, str) or not request.batch_id.strip():
            raise ValidationError("batch id must be non-empty text")
        split = _validate_split(request.split)
        payload = self._canonical_batch_payload(request)
        payload_hash = sha256_json(payload)

        fast_enqueue: Mapping[str, Any] | None = None
        fast_records: list[dict[str, Any]] | None = None
        with self._serialized_queue_admission_lock():
            queue_records = self._read_queue_records()
            existing = self._queue_enqueue_for_batch(queue_records, request.batch_id)
            if existing is not None:
                # Retry identity is derived without consulting mutable live
                # Draft state; later edits cannot invalidate an exact receipt.
                if existing["payload_hash"] != payload_hash:
                    raise CommitConflictError(
                        "batch id reused with a different immutable payload"
                    )
                fast_enqueue = existing
                fast_records = queue_records
            else:
                active = self._active_queue_enqueue(queue_records)
                if active is not None:
                    fast_enqueue = active
                    fast_records = queue_records
        if fast_enqueue is not None and fast_records is not None:
            return self._enqueue_receipt_under_barrier(fast_enqueue, fast_records)

        # Freeze the exact payload before any external authority lookup.  The
        # verifier may safely call supported store readers because neither an
        # exclusive queue lock nor the transaction lock is held here.
        attested_request = _batch_request_from_payload(payload)
        with self._batch_admission_reader_lock():
            manifest = self._read_manifest()
            self._validate_batch_store_contract(attested_request, manifest)
        self._attest_batch_authority(attested_request)

        final_enqueue: Mapping[str, Any] | None = None
        final_records: list[dict[str, Any]] | None = None
        with self._serialized_queue_admission_lock():
            queue_records = self._read_queue_records()
            existing = self._queue_enqueue_for_batch(queue_records, request.batch_id)
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    raise CommitConflictError(
                        "batch id reused with a different immutable payload"
                    )
                final_enqueue = existing
                final_records = queue_records
            else:
                active = self._active_queue_enqueue(queue_records)
                if active is not None:
                    final_enqueue = active
                    final_records = queue_records
            if final_enqueue is None:
                # Admission observes a reconciled transaction projection but
                # never waits behind the long-running worker exclusive lock.
                with self._shared_lock():
                    final_request = _batch_request_from_payload(payload)
                    self._validate_batch_store_contract(
                        final_request, self._read_manifest()
                    )
                    reconciliation_reason = self._batch_reconciliation_reason(
                        queue_records
                    )
                    if reconciliation_reason is not None:
                        raise RecoveryError(
                            f"store requires recovery: {reconciliation_reason}"
                        )
                    record = {
                        "kind": "enqueue",
                        "batch_id": final_request.batch_id,
                        "payload_hash": payload_hash,
                        "split": split,
                        "base_generation": final_request.base_generation,
                        "member_count": len(payload["members"]),
                        "payload": payload,
                    }
                    queued = self._append_queue_record(record, "enqueue")
                    final_enqueue = queued
                    final_records = queue_records + [queued]
        if final_enqueue is None or final_records is None:  # pragma: no cover
            raise StoreError("batch admission produced no durable identity")
        return self._enqueue_receipt_under_barrier(final_enqueue, final_records)

    def get_batch_status(self, batch_id: str) -> BatchStatusView:
        with self._shared_lock():
            with self._shared_queue_lock():
                queue_records = self._read_queue_records()
            reconciliation_reason = self._batch_reconciliation_reason(queue_records)
            enqueue = self._queue_enqueue_for_batch(queue_records, batch_id)
            if enqueue is None:
                if reconciliation_reason is not None:
                    raise RecoveryError(
                        f"store requires recovery: {reconciliation_reason}"
                    )
                return BatchStatusView(
                    batch_id=batch_id,
                    payload_hash=None,
                    status=BatchStatus.NOT_FOUND,
                    split=None,
                    member_count=0,
                    base_generation=None,
                )
            journal_records = list(self._records)
            terminals = [
                record
                for record in journal_records
                if record.get("kind") == "batch_terminal"
                and record.get("batch_id") == batch_id
            ]
            prepared = any(
                record.get("kind") == "batch_prepared"
                and record.get("batch_id") == batch_id
                for record in journal_records
            )
            queue_terminals = [
                record
                for record in queue_records
                if record.get("kind") == "queue_terminal"
                and record.get("batch_id") == batch_id
            ]
            if len(terminals) > 1:
                raise RecoveryError(f"batch {batch_id} has multiple terminal outcomes")
            if terminals:
                terminal = terminals[0]
                projection_matches = any(
                    record.get("payload_hash") == terminal.get("payload_hash")
                    and record.get("status") == terminal.get("status")
                    and record.get("generation") == terminal.get("generation")
                    and record.get("working_sha256") == terminal.get("working_sha256")
                    and record.get("error") == terminal.get("error")
                    for record in queue_terminals
                )
                if projection_matches:
                    status = BatchStatus(terminal["status"])
                    generation = terminal["generation"]
                    error = terminal.get("error")
                else:
                    status = BatchStatus.RECONCILING
                    generation = int(terminal["generation"])
                    error = "queue terminal projection requires repair"
            elif prepared or queue_terminals:
                status = BatchStatus.RECONCILING
                generation = None
                error = "batch transaction requires reconciliation"
            else:
                status, generation, error = self._queue_batch_state(
                    enqueue, queue_records
                )
            if reconciliation_reason is not None:
                status = BatchStatus.RECONCILING
                error = reconciliation_reason
            return BatchStatusView(
                batch_id=batch_id,
                payload_hash=enqueue["payload_hash"],
                status=status,
                split=enqueue["split"],
                member_count=enqueue["member_count"],
                base_generation=enqueue["base_generation"],
                generation=generation,
                error=error,
            )

    def _validate_batch_store_contract(
        self,
        request: BatchRequest,
        manifest: Mapping[str, Any],
    ) -> None:
        members = sorted(request.members, key=lambda member: member.source_row_index)
        seen_indices: set[int] = set()
        seen_images: set[int] = set()
        seen_tasks: set[str] = set()
        seen_commits: set[str] = set()
        if request.split != manifest.get("split"):
            raise StaleCommitError("split mismatch")
        if request.base_generation > int(manifest["generation"]):
            raise StaleCommitError("batch base generation is from the future")
        for member in members:
            index = member.source_row_index
            if index >= int(manifest["task_count"]):
                raise ValidationError("source row index out of range")
            if index in seen_indices:
                raise ValidationError("duplicate source row index")
            seen_indices.add(index)
            request_member = member.request
            expected_index = self._source_row_by_image.get(request_member.image_id)
            if expected_index != index:
                raise ValidationError("source row index mismatch")
            if request_member.image_id in seen_images:
                raise ValidationError("duplicate batch image identity")
            if request_member.task_id in seen_tasks:
                raise ValidationError("duplicate batch task identity")
            if request_member.commit_id in seen_commits:
                raise ValidationError("duplicate batch member commit id")
            seen_images.add(request_member.image_id)
            seen_tasks.add(request_member.task_id)
            seen_commits.add(request_member.commit_id)
            if request_member.split != request.split:
                raise ValidationError("batch member split mismatch")
            if request_member.project_id != manifest.get("project_id"):
                raise StaleCommitError("project mismatch")
            if request_member.task_id != _task_id(
                request.split, request_member.image_id
            ):
                raise StaleCommitError("task identity mismatch")
            self._validate_batch_frozen_request(
                request_member,
                current_user_id=request.current_user_id,
            )

    def _attest_batch_authority(self, request: BatchRequest) -> None:
        verify_batch = getattr(self.annotation_verifier, "verify_batch", None)
        if verify_batch is not None:
            if not callable(verify_batch) or not verify_batch(request):
                raise StaleCommitError("authoritative batch snapshot was not attested")
            return
        for member in request.members:
            if not self.annotation_verifier.verify(
                AuthoritativeDraftIdentity.from_request(member.request)
            ):
                raise StaleCommitError(
                    "authoritative annotation snapshot was not attested"
                )

    def _canonical_batch_payload(self, request: BatchRequest) -> dict[str, Any]:
        if request.split not in {"train", "val"}:
            raise ValidationError("batch payload split must be 'train' or 'val'")
        if (
            isinstance(request.base_generation, bool)
            or not isinstance(request.base_generation, int)
            or request.base_generation < 0
        ):
            raise ValidationError(
                "batch payload base generation must be a non-negative integer"
            )
        if (
            not isinstance(request.current_user_id, str)
            or not request.current_user_id
            or request.current_user_id != request.current_user_id.strip()
        ):
            raise ValidationError(
                "batch payload current_user_id must be non-empty trimmed text"
            )
        if isinstance(request.members, (str, bytes, bytearray)) or not isinstance(
            request.members, Sequence
        ):
            raise ValidationError("batch payload members must be a sequence")
        members = list(request.members)
        if not members:
            raise ValidationError("batch payload must contain at least one member")
        if any(not isinstance(member, BatchMember) for member in members):
            raise ValidationError("batch payload members have an invalid type")
        if any(
            isinstance(member.source_row_index, bool)
            or not isinstance(member.source_row_index, int)
            or member.source_row_index < 0
            for member in members
        ):
            raise ValidationError("batch payload source row index out of range")
        members.sort(key=lambda member: member.source_row_index)
        payload = {
            "batch_id": request.batch_id,
            "split": request.split,
            "current_user_id": request.current_user_id,
            "base_generation": request.base_generation,
            "members": [
                {
                    "source_row_index": member.source_row_index,
                    "request": _commit_request_payload(member.request),
                }
                for member in members
            ],
        }
        _validate_batch_payload(payload, error_type=ValidationError)
        try:
            canonical_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValidationError("batch payload must be finite ordinary JSON") from exc
        return payload

    def _enqueue_receipt(
        self,
        enqueue: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        *,
        status_override: BatchStatus | None = None,
    ) -> BatchEnqueueReceipt:
        status, _, _ = self._queue_batch_state(enqueue, records)
        return BatchEnqueueReceipt(
            batch_id=enqueue["batch_id"],
            payload_hash=enqueue["payload_hash"],
            status=status if status_override is None else status_override,
            split=enqueue["split"],
            member_count=enqueue["member_count"],
            base_generation=enqueue["base_generation"],
        )

    def _enqueue_receipt_under_barrier(
        self,
        enqueue: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> BatchEnqueueReceipt:
        status, _, _ = self._queue_batch_state(enqueue, records)
        try:
            with self._shared_lock():
                reconciliation_reason = self._batch_reconciliation_reason(records)
        except StoreBusyError:
            reconciliation_reason = (
                None
                if status in {BatchStatus.QUEUED, BatchStatus.RUNNING}
                else "transaction barrier is temporarily unavailable"
            )
        return self._enqueue_receipt(
            enqueue,
            records,
            status_override=(
                BatchStatus.RECONCILING if reconciliation_reason is not None else None
            ),
        )

    def _queue_batch_state(
        self,
        enqueue: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[BatchStatus, int | None, str | None]:
        matching = [
            record
            for record in records
            if record.get("batch_id") == enqueue.get("batch_id")
        ]
        terminal = next(
            (
                record
                for record in reversed(matching)
                if record.get("kind") == "queue_terminal"
            ),
            None,
        )
        if terminal is not None:
            return (
                BatchStatus(terminal["status"]),
                terminal["generation"],
                terminal.get("error"),
            )
        if any(record.get("kind") == "claim" for record in matching):
            return BatchStatus.RUNNING, None, None
        return BatchStatus.QUEUED, None, None

    @staticmethod
    def _queue_enqueue_for_batch(
        records: Sequence[Mapping[str, Any]], batch_id: str
    ) -> dict[str, Any] | None:
        matches = [
            record
            for record in records
            if record.get("kind") == "enqueue" and record.get("batch_id") == batch_id
        ]
        if len(matches) > 1:
            raise RecoveryError(f"duplicate enqueue record for batch {batch_id}")
        return dict(matches[0]) if matches else None

    def _active_queue_enqueue(
        self, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        active: list[dict[str, Any]] = []
        for record in records:
            if record.get("kind") != "enqueue":
                continue
            status, _, _ = self._queue_batch_state(record, records)
            if status in {
                BatchStatus.QUEUED,
                BatchStatus.RUNNING,
                BatchStatus.RECONCILING,
            }:
                active.append(dict(record))
        if len(active) > 1:
            raise RecoveryError("split queue contains multiple active batches")
        return active[0] if active else None

    def _batch_reconciliation_reason(
        self, queue_records: Sequence[Mapping[str, Any]]
    ) -> str | None:
        """Return durable cross-process projection disagreement, if any.

        The caller holds the transaction lock. Journal replay refreshes the
        process-local identity index, while queue/manifest checks prevent a
        pre-opened peer from serving across a crashed publication boundary.
        """

        self._reload_journal_index()
        legacy_prepared_by_hash = {
            str(record["record_hash"]): record
            for record in self._records
            if record.get("kind") == "prepared"
        }
        legacy_terminals = [
            record for record in self._records if record.get("kind") == "terminal"
        ]
        legacy_terminal_by_prepared = {
            str(record["prepared_record_hash"]): record for record in legacy_terminals
        }
        for prepared_hash, prepared in legacy_prepared_by_hash.items():
            if prepared_hash not in legacy_terminal_by_prepared:
                return (
                    f"commit {prepared['commit_id']} has an unresolved prepared "
                    "transaction"
                )

        prepared_by_hash = {
            str(record["record_hash"]): record
            for record in self._records
            if record.get("kind") == "batch_prepared"
        }
        terminals = [
            record for record in self._records if record.get("kind") == "batch_terminal"
        ]
        terminal_by_prepared = {
            str(record["prepared_record_hash"]): record
            for record in terminals
            if record.get("prepared_record_hash") is not None
        }
        for prepared_hash, prepared in prepared_by_hash.items():
            if prepared_hash not in terminal_by_prepared:
                return f"batch {prepared['batch_id']} has an unresolved prepared transaction"

        queue_enqueues = {
            str(record["batch_id"]): record
            for record in queue_records
            if record.get("kind") == "enqueue"
        }
        queue_terminals = [
            record for record in queue_records if record.get("kind") == "queue_terminal"
        ]
        journal_terminal_by_batch = {
            str(record["batch_id"]): record for record in terminals
        }
        for projection in queue_terminals:
            terminal = journal_terminal_by_batch.get(str(projection["batch_id"]))
            if terminal is None:
                return "queue terminal projection has no journal authority"
            if not self._queue_terminal_matches_journal(projection, terminal):
                if not any(
                    self._queue_terminal_matches_journal(candidate, terminal)
                    for candidate in queue_terminals
                    if candidate.get("batch_id") == terminal.get("batch_id")
                ):
                    return (
                        f"batch {terminal['batch_id']} queue terminal requires repair"
                    )

        latest_success: tuple[int, str, Mapping[str, Any]] | None = None
        for terminal in legacy_terminals:
            if terminal.get("status") != CommitStatus.COMMITTED.value:
                continue
            prepared = legacy_prepared_by_hash[str(terminal["prepared_record_hash"])]
            generation = int(terminal["generation"])
            label = f"commit {prepared['commit_id']}"
            if latest_success is None or generation > latest_success[0]:
                latest_success = (generation, label, prepared)
            elif generation == latest_success[0] and label != latest_success[1]:
                return f"generation {generation} has multiple publication authorities"

        for terminal in terminals:
            batch_id = str(terminal["batch_id"])
            enqueue = queue_enqueues.get(batch_id)
            if enqueue is None:
                return f"batch {batch_id} has no durable enqueue identity"
            matching_projection = any(
                projection.get("batch_id") == batch_id
                and self._queue_terminal_matches_journal(projection, terminal)
                for projection in queue_terminals
            )
            if not matching_projection:
                return f"batch {batch_id} queue terminal requires repair"
            if terminal.get("status") != BatchStatus.SUCCEEDED.value:
                continue
            prepared_hash = str(terminal.get("prepared_record_hash"))
            prepared = prepared_by_hash.get(prepared_hash)
            if prepared is None:
                return f"batch {batch_id} terminal has no prepared authority"
            generation = int(terminal["generation"])
            label = f"batch {batch_id}"
            if latest_success is None or generation > latest_success[0]:
                latest_success = (generation, label, prepared)
            elif generation == latest_success[0] and label != latest_success[1]:
                return f"generation {generation} has multiple publication authorities"

        if latest_success is not None:
            generation, label, prepared = latest_success
            manifest = self._read_manifest()
            manifest_generation = int(manifest["generation"])
            if manifest_generation < generation:
                return f"{label} manifest projection is behind"
            if manifest_generation == generation and (
                sha256_json(manifest) != prepared.get("candidate_manifest_hash")
                or manifest.get("working_sha256")
                != prepared.get("candidate_working_sha256")
                or int(manifest.get("working_line_count", -1))
                != int(prepared["candidate_manifest"]["working_line_count"])
            ):
                return f"{label} manifest projection requires repair"
        return None

    @staticmethod
    def _queue_terminal_matches_journal(
        projection: Mapping[str, Any], terminal: Mapping[str, Any]
    ) -> bool:
        return all(
            projection.get(key) == terminal.get(key)
            for key in (
                "batch_id",
                "payload_hash",
                "status",
                "generation",
                "working_sha256",
                "error",
            )
        )

    def process_next_batch(self) -> BatchResult | None:
        """Claim and publish one queued batch for this split."""

        self._assert_serving_ready()
        with self._exclusive_queue_lock():
            queue_records = self._read_queue_records()
            enqueue = self._active_queue_enqueue(queue_records)
            if enqueue is None:
                return None
            if not any(
                record.get("kind") == "claim"
                and record.get("batch_id") == enqueue["batch_id"]
                for record in queue_records
            ):
                self._append_queue_record(
                    {
                        "kind": "claim",
                        "batch_id": enqueue["batch_id"],
                        "payload_hash": enqueue["payload_hash"],
                    },
                    "claim",
                )

        with self._exclusive_lock():
            self._reload_journal_index()
            self._refresh_row_cache_from_journal(self._read_manifest())
            unfinished = [
                record
                for record in self._records
                if record.get("kind") == "batch_prepared"
                and record.get("batch_id") == enqueue["batch_id"]
                and self._batch_terminal_record(str(enqueue["batch_id"])) is None
            ]
            if len(unfinished) > 1:
                raise RecoveryError(
                    f"batch {enqueue['batch_id']} has multiple prepared transactions"
                )
            if unfinished:
                self._recover_batch_prepared(unfinished[0])
                self._reload_journal_index()
            prior_terminal = self._batch_terminal_record(str(enqueue["batch_id"]))
            if prior_terminal is not None:
                result = self._batch_result_from_terminal(prior_terminal)
            else:
                try:
                    result = self._execute_batch(enqueue)
                except InjectedCrash:
                    raise
                except (StaleCommitError, ValidationError) as exc:
                    # A failure may be injected immediately after a durable
                    # reservation append, before the in-memory chain advances.
                    self._reload_journal_index()
                    terminal = self._append_batch_failure(enqueue, str(exc))
                    result = self._batch_result_from_terminal(terminal)

        self._repair_one_queue_terminal(result)
        return result

    def get_batch_result(self, batch_id: str) -> BatchResult:
        with self._supported_reader_lock():
            records = list(self._records)
            terminals = [
                record
                for record in records
                if record.get("kind") == "batch_terminal"
                and record.get("batch_id") == batch_id
            ]
            if not terminals:
                with self._shared_queue_lock():
                    queue_records = self._read_queue_records()
                if self._queue_enqueue_for_batch(queue_records, batch_id) is None:
                    raise StoreError(f"unknown batch id: {batch_id}")
                raise CommitOutcomeUnknown(batch_id)
            if len(terminals) != 1:
                raise RecoveryError(f"batch {batch_id} has multiple terminal outcomes")
            return self._batch_result_from_terminal(terminals[0], records=records)

    def _execute_batch(self, enqueue: Mapping[str, Any]) -> BatchResult:
        payload = copy.deepcopy(enqueue["payload"])
        if sha256_json(payload) != enqueue.get("payload_hash"):
            raise RecoveryError("queued batch payload hash disagreement")
        frozen_request = _batch_request_from_payload(payload, error_type=RecoveryError)
        members = [
            (member.source_row_index, member.request)
            for member in frozen_request.members
        ]
        if [index for index, _ in members] != sorted(index for index, _ in members):
            raise RecoveryError("queued batch members are not source-row ordered")
        manifest = self._read_manifest()
        if (
            self._cache_generation != int(manifest["generation"])
            or self._cache_working_sha256 != manifest["working_sha256"]
            or self._cache_line_count != int(manifest["working_line_count"])
        ):
            raise RecoveryError("row freshness cache is not publication-attested")
        for source_row_index, request in members:
            self._validate_batch_frozen_request(
                request,
                current_user_id=frozen_request.current_user_id,
            )
            if self._row_image_cache.get(source_row_index) != request.image_id:
                raise StaleCommitError("source row image identity changed")
            if self._row_hash_cache.get(source_row_index) != request.base_row_hash:
                raise StaleCommitError("base row hash changed")
            self._validate_request_against_cached_identity(
                request,
                self._row_object_ids_cache[source_row_index],
            )
        self._reserve_batch_ids(enqueue, members)

        if payload.get("split") != manifest.get("split"):
            raise RecoveryError("queued batch split disagrees with the manifest")
        member_by_index = {index: request for index, request in members}
        fd, temp_name = tempfile.mkstemp(prefix=".working.batch.", dir=self.split_dir)
        temp_path = Path(temp_name)
        replaced = False
        prepared: dict[str, Any] | None = None
        try:
            input_digest = hashlib.sha256()
            candidate_digest = hashlib.sha256()
            input_line_count = 0
            seen_member_indices: set[int] = set()
            object_owners: dict[int, int] = {}
            image_ids: set[int] = set()
            prepared_members: list[dict[str, Any]] = []
            candidate_row_images: dict[int, int] = {}
            candidate_row_object_ids: dict[int, set[int]] = {}
            candidate_row_hashes: dict[int, str] = {}
            with (
                os.fdopen(fd, "wb") as output,
                self.working_path.open("rb") as input_handle,
            ):
                for source_row_index, raw in enumerate(input_handle):
                    if not raw.endswith(b"\n"):
                        raise ValidationError(
                            "working JSONL must end every row with a newline"
                        )
                    input_digest.update(raw)
                    input_line_count += 1
                    try:
                        before_row = _strict_json_loads(raw)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        ValueError,
                    ) as exc:
                        raise ValidationError(
                            f"invalid working JSONL row {source_row_index + 1}"
                        ) from exc
                    image_id = int(before_row.get("image_id", -1))
                    if self._source_row_by_image.get(image_id) != source_row_index:
                        raise RecoveryError(
                            "working source row index attestation failed"
                        )
                    request = member_by_index.get(source_row_index)
                    if request is None:
                        after_row = before_row
                        encoded = raw
                    else:
                        if request.image_id != image_id:
                            raise StaleCommitError("source row image identity changed")
                        before_hash = sha256_json(before_row)
                        if before_hash != request.base_row_hash:
                            raise StaleCommitError("base row hash changed")
                        after_objects, mapping, _, projection = (
                            self._materialize_objects(
                                before_row,
                                request.regions,
                                split=request.split,
                                image_id=request.image_id,
                            )
                        )
                        after_row = copy.deepcopy(before_row)
                        after_row["objects"] = after_objects
                        self._validate_row(after_row)
                        after_hash = sha256_json(after_row)
                        tombstones = sorted(
                            {int(obj["coco_ann_id"]) for obj in before_row["objects"]}
                            - {int(obj["coco_ann_id"]) for obj in after_objects}
                        )
                        prepared_members.append(
                            {
                                "source_row_index": source_row_index,
                                "commit_id": request.commit_id,
                                "image_id": image_id,
                                "semantic_hash": request.semantic_hash,
                                "request_identity": _request_identity_payload(
                                    request, projection
                                ),
                                "before_row": before_row,
                                "after_row": after_row,
                                "before_row_hash": before_hash,
                                "after_row_hash": after_hash,
                                "region_id_mapping": mapping,
                                "tombstones": tombstones,
                                "inference_receipts": list(request.inference_receipts),
                            }
                        )
                        seen_member_indices.add(source_row_index)
                        encoded = (canonical_json(after_row) + "\n").encode("utf-8")
                    self._validate_row(after_row)
                    _register_split_wide_ids(
                        after_row,
                        object_owners,
                        image_ids,
                        error_type=ValidationError,
                    )
                    candidate_row_images[source_row_index] = image_id
                    candidate_row_object_ids[source_row_index] = {
                        int(obj["coco_ann_id"]) for obj in after_row["objects"]
                    }
                    candidate_row_hashes[source_row_index] = sha256_json(after_row)
                    output.write(encoded)
                    candidate_digest.update(encoded)

                if seen_member_indices != set(member_by_index):
                    raise StaleCommitError("one or more batch source rows are missing")
                input_hash = input_digest.hexdigest()
                if input_line_count != int(manifest["working_line_count"]):
                    raise RecoveryError("working input line-count attestation failed")
                if input_hash != manifest["working_sha256"]:
                    raise RecoveryError("working input hash attestation failed")
                output.flush()
                self._fault("batch_working_temp_flushed")
                os.fsync(output.fileno())
                self._fault("batch_working_temp_fsynced")

            candidate_generation = int(manifest["generation"]) + 1
            candidate_working_hash = candidate_digest.hexdigest()
            candidate_manifest = copy.deepcopy(manifest)
            candidate_manifest.update(
                {
                    "generation": candidate_generation,
                    "working_sha256": candidate_working_hash,
                    "working_line_count": input_line_count,
                    "last_batch_id": enqueue["batch_id"],
                }
            )
            prepared = self._append_record(
                {
                    "kind": "batch_prepared",
                    "batch_id": enqueue["batch_id"],
                    "payload_hash": enqueue["payload_hash"],
                    "split": enqueue["split"],
                    "request_base_generation": enqueue["base_generation"],
                    "base_generation": int(manifest["generation"]),
                    "candidate_generation": candidate_generation,
                    "before_working_sha256": input_hash,
                    "before_working_line_count": input_line_count,
                    "candidate_working_sha256": candidate_working_hash,
                    "candidate_working_line_count": input_line_count,
                    "before_manifest_hash": sha256_json(manifest),
                    "candidate_manifest": candidate_manifest,
                    "candidate_manifest_hash": sha256_json(candidate_manifest),
                    "members": prepared_members,
                },
                "batch_prepared",
            )
            os.replace(temp_path, self.working_path)
            replaced = True
            self._fault("batch_working_replaced")
            _fsync_directory(self.split_dir)
            self._fault("batch_working_directory_fsynced")
            self._publish_manifest(
                candidate_manifest, str(prepared["candidate_manifest_hash"])
            )
            terminal = self._append_batch_terminal(
                prepared,
                BatchStatus.SUCCEEDED,
                generation=candidate_generation,
                working_sha256=candidate_working_hash,
            )
            self._row_image_cache = candidate_row_images
            self._row_object_ids_cache = candidate_row_object_ids
            self._row_hash_cache = candidate_row_hashes
            self._cache_generation = candidate_generation
            self._cache_working_sha256 = candidate_working_hash
            self._cache_line_count = input_line_count
            return self._batch_result_from_terminal(terminal)
        except InjectedCrash:
            raise
        except Exception:
            if replaced:
                raise CommitOutcomeUnknown(
                    f"batch {enqueue['batch_id']} outcome requires reconciliation"
                )
            raise
        finally:
            temp_path.unlink(missing_ok=True)

    def _validate_batch_frozen_request(
        self,
        request: CommitRequest,
        *,
        current_user_id: str,
    ) -> None:
        self._validate_draft_handshake(request)
        if not request.regions:
            raise ValidationError(
                "empty Draft is preserved, but V1 Commit requires an object"
            )
        self._validate_inference_linkage(
            request,
            current_user_id=current_user_id,
        )
        keys: set[str] = set()
        for region in request.regions:
            key = _region_key(region)
            if key in keys:
                raise ValidationError(f"duplicate region key: {key}")
            keys.add(key)
            _source_id_from_region_key(key, request.split)
            supplied_id = region.get("coco_ann_id")
            if supplied_id is not None and (
                type(supplied_id) is not int or supplied_id == 0
            ):
                raise ValidationError("coco_ann_id must be null or a nonzero integer")
            name = region.get("category_name", region.get("desc"))
            category_id = region.get("category_id")
            if not isinstance(name, str) or type(category_id) is not int:
                raise ValidationError(
                    "category_name and integer category_id are required"
                )
            self.registry.validate(name, category_id)
            _validate_bbox(region.get("bbox_2d"))
            if "creation_ordinal" in region:
                creation_ordinal = region["creation_ordinal"]
                if type(creation_ordinal) is not int or creation_ordinal < 0:
                    raise ValidationError(
                        "creation_ordinal must be a non-negative integer"
                    )
            _training_metadata(region.get("metadata"))

    def _validate_request_against_cached_identity(
        self, request: CommitRequest, before_ids: set[int]
    ) -> None:
        used_ids: set[int] = set()
        for region in request.regions:
            key = _region_key(region)
            supplied_id = region.get("coco_ann_id")
            known_id = self._region_to_id.get((request.image_id, key))
            implicit_source_id = _source_id_from_region_key(key, request.split)
            if implicit_source_id is not None:
                if implicit_source_id not in before_ids:
                    if implicit_source_id in self._tombstones:
                        raise ValidationError(
                            f"tombstoned source identity cannot be restored: {key!r}"
                        )
                    raise ValidationError(
                        f"source identity {key!r} is not present in the current before row"
                    )
                known_id = implicit_source_id
            elif supplied_id is not None and (
                isinstance(supplied_id, bool)
                or not isinstance(supplied_id, int)
                or supplied_id > 0
            ):
                raise ValidationError(
                    "positive source identity requires its canonical split:coco:<id> region key"
                )
            if known_id is not None and supplied_id not in (None, known_id):
                raise ValidationError(f"region {key!r} changed its hidden identity")
            if known_id is None:
                if supplied_id is not None:
                    raise ValidationError(
                        "unmapped coco_ann_id values are not accepted from the Draft"
                    )
                continue
            if known_id in used_ids:
                raise ValidationError(f"duplicate coco_ann_id: {known_id}")
            if known_id in self._tombstones and known_id not in before_ids:
                raise ValidationError(
                    f"tombstoned coco_ann_id cannot be restored: {known_id}"
                )
            used_ids.add(known_id)

    def _reserve_batch_ids(
        self,
        enqueue: Mapping[str, Any],
        members: Sequence[tuple[int, CommitRequest]],
    ) -> None:
        pending: list[tuple[int, str]] = []
        for _, request in members:
            new_keys = sorted(
                {
                    _region_key(region)
                    for region in request.regions
                    if _source_id_from_region_key(_region_key(region), request.split)
                    is None
                }
            )
            for key in new_keys:
                reservation = self._batch_reservations.get(key)
                if reservation is not None:
                    owner_image, _ = reservation
                    if owner_image != request.image_id:
                        raise ValidationError(
                            f"stable region key is already bound to another task: {key!r}"
                        )
                    continue
                prior_owners = {
                    (mapped_image_id, mapped_id)
                    for (
                        mapped_image_id,
                        mapped_key,
                    ), mapped_id in self._region_to_id.items()
                    if mapped_key == key
                }
                if any(
                    owner_image != request.image_id for owner_image, _ in prior_owners
                ):
                    raise ValidationError(
                        f"stable region key is already bound to another task: {key!r}"
                    )
                known = self._region_to_id.get((request.image_id, key))
                if known is not None:
                    continue
                pending.append((request.image_id, key))

        next_negative = min(self._reserved_negative_ids | {0}) - 1
        for image_id, key in pending:
            while next_negative in self._reserved_negative_ids:
                next_negative -= 1
            object_id = next_negative
            next_negative -= 1
            record = self._append_record(
                {
                    "kind": "reservation",
                    "batch_id": enqueue["batch_id"],
                    "payload_hash": enqueue["payload_hash"],
                    "split": enqueue["split"],
                    "image_id": image_id,
                    "stable_region_key": key,
                    "coco_ann_id": object_id,
                },
                "reservation",
            )
            self._register_reservation_record(record)

    def _append_batch_failure(
        self, enqueue: Mapping[str, Any], error: str
    ) -> dict[str, Any]:
        manifest = self._read_manifest()
        return self._append_batch_terminal(
            None,
            BatchStatus.FAILED,
            batch_id=enqueue["batch_id"],
            payload_hash=enqueue["payload_hash"],
            generation=int(manifest["generation"]),
            working_sha256=str(manifest["working_sha256"]),
            error=error,
        )

    def _append_batch_terminal(
        self,
        prepared: Mapping[str, Any] | None,
        status: BatchStatus,
        *,
        batch_id: str | None = None,
        payload_hash: str | None = None,
        generation: int,
        working_sha256: str,
        error: str | None = None,
        recovery: bool = False,
    ) -> dict[str, Any]:
        resolved_batch_id = str(
            prepared["batch_id"] if prepared is not None else batch_id
        )
        prior = self._batch_terminal_record(resolved_batch_id)
        if prior is not None:
            return prior
        return self._append_record(
            {
                "kind": "batch_terminal",
                "batch_id": resolved_batch_id,
                "payload_hash": (
                    prepared["payload_hash"] if prepared is not None else payload_hash
                ),
                "prepared_record_hash": (
                    prepared["record_hash"] if prepared is not None else None
                ),
                "status": status.value,
                "generation": generation,
                "working_sha256": working_sha256,
                "error": error,
                "recovery": recovery,
            },
            "batch_terminal",
        )

    def _batch_terminal_record(self, batch_id: str) -> dict[str, Any] | None:
        matches = [
            record
            for record in self._records
            if record.get("kind") == "batch_terminal"
            and record.get("batch_id") == batch_id
        ]
        if len(matches) > 1:
            raise RecoveryError(f"batch {batch_id} has multiple terminal outcomes")
        return matches[0] if matches else None

    def _batch_result_from_terminal(
        self,
        terminal: Mapping[str, Any],
        *,
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> BatchResult:
        records = self._records if records is None else records
        prepared = None
        prepared_hash = terminal.get("prepared_record_hash")
        if prepared_hash is not None:
            prepared = next(
                (
                    record
                    for record in records
                    if record.get("kind") == "batch_prepared"
                    and record.get("record_hash") == prepared_hash
                ),
                None,
            )
            if prepared is None:
                raise RecoveryError("batch terminal has no matching prepared record")
        member_results: list[CommitResult] = []
        if prepared is not None and terminal["status"] == BatchStatus.SUCCEEDED.value:
            for member in prepared["members"]:
                member_results.append(
                    CommitResult(
                        commit_id=str(member["commit_id"]),
                        status=CommitStatus.COMMITTED,
                        split=str(prepared["split"]),
                        image_id=int(member["image_id"]),
                        generation=int(prepared["candidate_generation"]),
                        row_hash=str(member["after_row_hash"]),
                        semantic_hash=str(member["semantic_hash"]),
                        region_id_mapping={
                            str(key): int(value)
                            for key, value in member["region_id_mapping"].items()
                        },
                        committed_row=copy.deepcopy(member["after_row"]),
                    )
                )
        return BatchResult(
            batch_id=str(terminal["batch_id"]),
            payload_hash=str(terminal["payload_hash"]),
            status=BatchStatus(str(terminal["status"])),
            split=str(
                prepared["split"]
                if prepared is not None
                else self._read_manifest()["split"]
            ),
            generation=int(terminal["generation"]),
            working_sha256=str(terminal["working_sha256"]),
            members=tuple(member_results),
            error=terminal.get("error"),
        )

    def _repair_one_queue_terminal(self, result: BatchResult) -> None:
        with self._exclusive_queue_lock():
            records = self._read_queue_records()
            existing = [
                record
                for record in records
                if record.get("kind") == "queue_terminal"
                and record.get("batch_id") == result.batch_id
            ]
            expected = {
                "payload_hash": result.payload_hash,
                "status": result.status.value,
                "generation": result.generation,
                "working_sha256": result.working_sha256,
                "error": result.error,
            }
            if existing:
                if any(
                    all(record.get(key) == value for key, value in expected.items())
                    for record in existing
                ):
                    return
            self._append_queue_record(
                {
                    "kind": "queue_terminal",
                    "batch_id": result.batch_id,
                    **expected,
                },
                "queue_terminal",
            )

    def commit(
        self,
        request: CommitRequest,
        *,
        current_user_id: str | None = None,
    ) -> CommitResult:
        """Commit the exact durably-saved Draft or return its prior outcome."""

        self._validate_draft_handshake(request)
        with self._legacy_commit_lock():
            self._reload_journal_index()
            prior = self._records_for_commit(request.commit_id)
            if prior:
                result = self._resolve_idempotent_request(request, prior)
                self._validate_inference_linkage(
                    request,
                    current_user_id=current_user_id,
                )
                return result
            manifest = self._read_manifest()
            if request.split != manifest["split"]:
                raise StaleCommitError("split mismatch")
            if request.project_id != manifest["project_id"]:
                raise StaleCommitError("project mismatch")
            if request.task_id != _task_id(request.split, request.image_id):
                raise StaleCommitError("task identity mismatch")
            if not self.annotation_verifier.verify(
                AuthoritativeDraftIdentity.from_request(request)
            ):
                raise StaleCommitError(
                    "authoritative annotation snapshot was not attested"
                )
            if request.observed_generation != int(manifest["generation"]):
                raise StaleCommitError("project generation changed")
            if sha256_file(self.working_path) != manifest["working_sha256"]:
                raise RecoveryError("working file does not match the manifest")

            row_index, before_row, _ = self._find_row(request.image_id)
            before_hash = sha256_json(before_row)
            if before_hash != request.base_row_hash:
                raise StaleCommitError("base row hash changed")
            if not request.regions:
                raise ValidationError(
                    "empty Draft is preserved, but V1 Commit requires an object"
                )

            self._validate_inference_linkage(
                request,
                current_user_id=current_user_id,
            )
            after_objects, mapping, allocations, materialized_projection = (
                self._materialize_objects(
                    before_row,
                    request.regions,
                    split=request.split,
                    image_id=request.image_id,
                )
            )
            after_row = copy.deepcopy(before_row)
            after_row["objects"] = after_objects
            self._validate_row(after_row)
            after_hash = sha256_json(after_row)
            tombstones = sorted(
                {int(obj["coco_ann_id"]) for obj in before_row["objects"]}
                - {int(obj["coco_ann_id"]) for obj in after_objects}
            )

            candidate_generation = int(manifest["generation"]) + 1
            candidate_working_hash = self._candidate_working_hash(row_index, after_row)
            candidate_manifest = copy.deepcopy(manifest)
            candidate_manifest.update(
                {
                    "generation": candidate_generation,
                    "working_sha256": candidate_working_hash,
                    "last_commit_id": request.commit_id,
                }
            )
            candidate_manifest_hash = sha256_json(candidate_manifest)
            request_identity = _request_identity_payload(
                request, materialized_projection
            )
            prepared = {
                "kind": "prepared",
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
                "request_identity": request_identity,
                "request_identity_hash": sha256_json(request_identity),
                "base_generation": int(manifest["generation"]),
                "candidate_generation": candidate_generation,
                "before_working_sha256": manifest["working_sha256"],
                "candidate_working_sha256": candidate_working_hash,
                "before_manifest_hash": sha256_json(manifest),
                "candidate_manifest": candidate_manifest,
                "candidate_manifest_hash": candidate_manifest_hash,
                "before_row": before_row,
                "after_row": after_row,
                "before_row_hash": before_hash,
                "after_row_hash": after_hash,
                "region_id_mapping": mapping,
                "allocated_mappings": allocations,
                "tombstones": tombstones,
                "inference_receipts": list(request.inference_receipts),
            }
            prepared = self._append_record(prepared, "prepared")

            working_replaced = False
            try:
                self._rewrite_working(row_index, after_row, candidate_working_hash)
                working_replaced = True
                self._publish_manifest(candidate_manifest, candidate_manifest_hash)
                self._append_terminal(prepared, CommitStatus.COMMITTED)
                self._fault("before_response")
            except InjectedCrash:
                raise
            except CommitOutcomeUnknown:
                raise
            except Exception as exc:
                if working_replaced:
                    raise CommitOutcomeUnknown(
                        f"commit {request.commit_id} outcome requires reconciliation"
                    ) from exc
                self._append_terminal(
                    prepared, CommitStatus.ROLLED_BACK, error=str(exc)
                )
                raise

            self._reload_journal_index()
            self._load_row_cache()
            return self._result_from_prepared(prepared, CommitStatus.COMMITTED)

    def status(self, commit_id: str) -> CommitStatus:
        with self._shared_lock():
            self._reload_journal_index()
            records = self._records_for_commit(commit_id)
            if not records:
                return CommitStatus.NOT_FOUND
            terminal = next(
                (r for r in reversed(records) if r["kind"] == "terminal"), None
            )
            if terminal is None:
                return CommitStatus.OUTCOME_UNKNOWN
            return CommitStatus(terminal["status"])

    def result(self, commit_id: str) -> CommitResult:
        with self._shared_lock():
            self._reload_journal_index()
            records = self._records_for_commit(commit_id)
            if not records:
                raise StoreError(f"unknown commit id: {commit_id}")
            prepared = next(
                record for record in records if record["kind"] == "prepared"
            )
            terminal = next(
                (
                    record
                    for record in reversed(records)
                    if record["kind"] == "terminal"
                ),
                None,
            )
            if terminal is None:
                raise CommitOutcomeUnknown(commit_id)
            status = CommitStatus(terminal["status"])
            if status is CommitStatus.ROLLED_BACK:
                raise CommitRolledBackError(commit_id)
            return self._result_from_prepared(prepared, status)

    def restore_draft(self, image_id: int) -> DraftRestore:
        """Return the committed row used by persisted-Draft reset/reload hooks."""

        return self.restore_drafts((image_id,))[0]

    def restore_drafts(self, image_ids: Sequence[int]) -> tuple[DraftRestore, ...]:
        """Atomically return committed baselines in the exact requested order.

        Duplicate identities are rejected rather than collapsed.  Every row
        and the one manifest generation are read while holding a single
        supported-reader barrier, so callers cannot assemble a cross-generation
        batch from repeated single-row reads.
        """

        if isinstance(image_ids, (str, bytes, bytearray)) or not isinstance(
            image_ids, Sequence
        ):
            raise ValidationError("image_ids must be a sequence of integers")
        requested = tuple(image_ids)
        seen: set[int] = set()
        for image_id in requested:
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id < 0
            ):
                raise ValidationError("image_id must be a non-negative integer")
            if image_id in seen:
                raise ValidationError(f"duplicate image_id: {image_id}")
            seen.add(image_id)

        with self._supported_reader_lock():
            manifest = self._read_manifest()
            split = str(manifest["split"])
            generation = int(manifest["generation"])
            rows = self._scan_attested_restore_rows(seen, manifest)

            restores: list[DraftRestore] = []
            for image_id in requested:
                row = rows.get(image_id)
                if row is None:
                    raise StaleCommitError(f"unknown image_id: {image_id}")
                mapping = {
                    self._key_for_object(obj, image_id, split): int(obj["coco_ann_id"])
                    for obj in row["objects"]
                }
                restores.append(
                    DraftRestore(
                        split=split,
                        image_id=image_id,
                        generation=generation,
                        row_hash=sha256_json(row),
                        row=copy.deepcopy(row),
                        region_id_mapping=copy.deepcopy(mapping),
                    )
                )
            return tuple(restores)

    def recover(self) -> None:
        """Reconcile an interrupted transaction exactly once before serving."""

        if not self.split_dir.exists():
            raise RecoveryError(f"missing split directory: {self.split_dir}")
        terminal_results: list[BatchResult] = []
        with self._exclusive_lock():
            self._reload_journal_index(repair_torn_tail=True)
            for prepared in [r for r in self._records if r["kind"] == "prepared"]:
                if any(
                    r["kind"] == "terminal"
                    and r.get("prepared_record_hash") == prepared["record_hash"]
                    for r in self._records
                ):
                    continue
                self._recover_prepared(prepared)
                self._reload_journal_index()
            for prepared in [r for r in self._records if r["kind"] == "batch_prepared"]:
                if self._batch_terminal_record(str(prepared["batch_id"])) is not None:
                    continue
                self._recover_batch_prepared(prepared)
                self._reload_journal_index()
            self._cleanup_orphan_batch_candidates()
            self._validate_authority()
            terminal_results = [
                self._batch_result_from_terminal(record)
                for record in self._records
                if record["kind"] == "batch_terminal"
            ]
        with self._exclusive_queue_lock():
            self._read_queue_records(repair_torn_tail=True)
        for result in terminal_results:
            self._repair_one_queue_terminal(result)
        self._recovery_required = False

    def _cleanup_orphan_batch_candidates(self) -> None:
        removed = False
        for candidate in self.split_dir.glob(".working.batch.*"):
            if candidate.is_file():
                candidate.unlink()
                removed = True
        if removed:
            _fsync_directory(self.split_dir)

    def _recover_prepared(self, prepared: Mapping[str, Any]) -> None:
        manifest = self._read_manifest()
        working_hash = sha256_file(self.working_path)
        manifest_hash = sha256_json(manifest)
        before_working = prepared["before_working_sha256"]
        candidate_working = prepared["candidate_working_sha256"]
        before_manifest = prepared["before_manifest_hash"]
        candidate_manifest = prepared["candidate_manifest_hash"]

        if working_hash == before_working and manifest_hash == before_manifest:
            self._append_terminal(prepared, CommitStatus.ROLLED_BACK, recovery=True)
            return
        if working_hash == candidate_working and manifest_hash == before_manifest:
            self._publish_manifest(
                prepared["candidate_manifest"], str(candidate_manifest), recovery=True
            )
            self._append_terminal(prepared, CommitStatus.COMMITTED, recovery=True)
            return
        if working_hash == candidate_working and manifest_hash == candidate_manifest:
            self._append_terminal(prepared, CommitStatus.COMMITTED, recovery=True)
            return
        raise RecoveryError(
            "journal/hash disagreement for commit "
            f"{prepared['commit_id']}: working={working_hash}, manifest={manifest_hash}"
        )

    def _recover_batch_prepared(self, prepared: Mapping[str, Any]) -> None:
        manifest = self._read_manifest()
        working_hash = sha256_file(self.working_path)
        manifest_hash = sha256_json(manifest)
        before_working = prepared["before_working_sha256"]
        candidate_working = prepared["candidate_working_sha256"]
        before_manifest = prepared["before_manifest_hash"]
        candidate_manifest_hash = prepared["candidate_manifest_hash"]

        if working_hash == before_working and manifest_hash == before_manifest:
            self._append_batch_terminal(
                prepared,
                BatchStatus.FAILED,
                generation=int(prepared["base_generation"]),
                working_sha256=str(before_working),
                error="recovered before batch publication",
                recovery=True,
            )
            return
        if working_hash == candidate_working and manifest_hash == before_manifest:
            _fsync_directory(self.split_dir)
            self._publish_manifest(
                prepared["candidate_manifest"],
                str(candidate_manifest_hash),
                recovery=True,
            )
            self._append_batch_terminal(
                prepared,
                BatchStatus.SUCCEEDED,
                generation=int(prepared["candidate_generation"]),
                working_sha256=str(candidate_working),
                recovery=True,
            )
            return
        if (
            working_hash == candidate_working
            and manifest_hash == candidate_manifest_hash
        ):
            self._append_batch_terminal(
                prepared,
                BatchStatus.SUCCEEDED,
                generation=int(prepared["candidate_generation"]),
                working_sha256=str(candidate_working),
                recovery=True,
            )
            return
        raise RecoveryError(
            "journal/hash disagreement for batch "
            f"{prepared['batch_id']}: working={working_hash}, manifest={manifest_hash}"
        )

    def _validate_draft_handshake(self, request: CommitRequest) -> None:
        receipt = request.draft_save
        if not receipt.durable:
            raise ValidationError("Commit requires a durable Draft-save receipt")
        for value, label in (
            (request.annotation_revision, "annotation revision"),
            (request.draft_updated_at, "Draft updated_at"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{label} must be an opaque non-empty string")
        for value, label in (
            (request.semantic_hash, "semantic hash"),
            (request.result_hash, "result hash"),
        ):
            if not _is_sha256(value):
                raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
        matched = (
            receipt.project_id == request.project_id
            and receipt.task_id == request.task_id
            and receipt.annotation_id == request.annotation_id
            and receipt.draft_id == request.draft_id
            and receipt.annotation_revision == request.annotation_revision
            and receipt.draft_updated_at == request.draft_updated_at
            and receipt.semantic_hash == request.semantic_hash
            and receipt.result_hash == request.result_hash
        )
        if not matched:
            raise StaleCommitError(
                "Draft-save receipt does not match the Commit snapshot"
            )
        actual_semantic_hash = semantic_hash(request.regions)
        if actual_semantic_hash != request.semantic_hash:
            raise StaleCommitError(
                "submitted regions do not match the saved Draft hash"
            )

    def _validate_inference_linkage(
        self,
        request: CommitRequest,
        *,
        current_user_id: str | None,
    ) -> None:
        declared = tuple(request.inference_receipts)
        if any(not isinstance(value, str) or not value for value in declared):
            raise ValidationError(
                "declared inference receipt IDs must be non-empty text"
            )
        if len(set(declared)) != len(declared):
            raise ValidationError("declared inference receipt IDs must be unique")

        used: set[str] = set()
        resolved: dict[str, InferenceReceiptLink] = {}
        provenance_fields = {
            "receipt_id",
            "request_id",
            "result_id",
            "draft_revision",
        }
        for region in request.regions:
            metadata = region.get("metadata")
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise ValidationError("metadata must be a mapping")
            has_provenance = bool(provenance_fields & set(metadata))
            inference_origin = metadata.get("inference_origin")
            if not has_provenance and inference_origin is not True:
                continue
            if inference_origin is not True or not provenance_fields.issubset(metadata):
                raise ValidationError(
                    "inference-origin objects require complete inference linkage metadata"
                )
            receipt_id = metadata["receipt_id"]
            request_id = metadata["request_id"]
            result_id = metadata["result_id"]
            draft_revision = metadata["draft_revision"]
            if any(
                not isinstance(value, str) or not value
                for value in (
                    receipt_id,
                    request_id,
                    result_id,
                    draft_revision,
                )
            ):
                raise ValidationError(
                    "inference-origin objects require complete inference linkage metadata"
                )
            if receipt_id not in declared:
                raise ValidationError(
                    f"object receipt {receipt_id!r} is not a declared inference receipt"
                )
            link = resolved.get(receipt_id)
            if link is None:
                if (
                    not isinstance(current_user_id, str)
                    or not current_user_id
                    or current_user_id != current_user_id.strip()
                ):
                    raise ValidationError(
                        "inference-origin objects require an authenticated current_user_id"
                    )
                link = self.inference_receipt_resolver.resolve(receipt_id)
                if link is None:
                    raise ValidationError(f"unknown inference receipt: {receipt_id}")
                resolved[receipt_id] = link
            if (
                link.receipt_id != receipt_id
                or link.request_id != request_id
                or link.project_id != request.project_id
                or link.task_id != request.task_id
                or link.image_id != request.image_id
                or link.annotation_id != request.annotation_id
                or link.current_user_id != current_user_id
                or link.draft_id != request.draft_id
                or link.terminal_status not in {"accepted", "accepted_with_drops"}
            ):
                raise ValidationError(
                    f"inference receipt target mismatch: {receipt_id}"
                )
            if link.draft_revision != draft_revision:
                raise ValidationError(
                    f"inference receipt source Draft revision mismatch: {receipt_id}"
                )
            key = _region_key(region)
            if link.result_region_keys.get(result_id) != key:
                raise ValidationError(
                    f"inference receipt result linkage mismatch: {receipt_id}/{result_id}"
                )
            used.add(receipt_id)

        if used != set(declared):
            unlinked = sorted(set(declared) - used)
            raise ValidationError(
                f"declared inference receipt has no committed object linkage: {unlinked}"
            )

    def _resolve_idempotent_request(
        self, request: CommitRequest, records: Sequence[Mapping[str, Any]]
    ) -> CommitResult:
        prepared = next(record for record in records if record["kind"] == "prepared")
        try:
            _, _, _, materialized_projection = self._materialize_objects(
                prepared["before_row"],
                request.regions,
                split=request.split,
                image_id=request.image_id,
                identity_region_id_mapping=prepared["region_id_mapping"],
            )
            identity = _request_identity_payload(request, materialized_projection)
        except Exception as exc:
            raise CommitConflictError(
                "commit id reused with a different immutable request identity"
            ) from exc
        if prepared.get("request_identity") != identity or prepared.get(
            "request_identity_hash"
        ) != sha256_json(identity):
            raise CommitConflictError(
                "commit id reused with a different immutable request identity"
            )
        terminal = next((r for r in reversed(records) if r["kind"] == "terminal"), None)
        if terminal is None:
            raise CommitOutcomeUnknown(request.commit_id)
        status = CommitStatus(terminal["status"])
        if status is CommitStatus.ROLLED_BACK:
            raise CommitRolledBackError(request.commit_id)
        return self._result_from_prepared(prepared, status)

    def _result_from_prepared(
        self, prepared: Mapping[str, Any], status: CommitStatus
    ) -> CommitResult:
        return CommitResult(
            commit_id=str(prepared["commit_id"]),
            status=status,
            split=str(prepared["split"]),
            image_id=int(prepared["image_id"]),
            generation=int(prepared["candidate_generation"]),
            row_hash=str(prepared["after_row_hash"]),
            semantic_hash=str(prepared["semantic_hash"]),
            region_id_mapping={
                str(key): int(value)
                for key, value in prepared["region_id_mapping"].items()
            },
            committed_row=copy.deepcopy(prepared["after_row"]),
        )

    def _materialize_objects(
        self,
        before_row: Mapping[str, Any],
        regions: Sequence[Mapping[str, Any]],
        *,
        split: str,
        image_id: int,
        identity_region_id_mapping: Mapping[str, Any] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
        dict[str, int],
        dict[str, Any],
    ]:
        before_ids = {int(obj["coco_ann_id"]): obj for obj in before_row["objects"]}
        before_rank = {
            int(obj["coco_ann_id"]): rank
            for rank, obj in enumerate(before_row["objects"])
        }
        used_ids: set[int] = set()
        mapping: dict[str, int] = {}
        allocations: dict[str, int] = {}
        materialized: list[tuple[dict[str, Any], str, int | None, int | None]] = []

        if identity_region_id_mapping is None:
            region_to_id = self._region_to_id
            reserved_negative_ids = self._reserved_negative_ids
            tombstones = self._tombstones
        else:
            # Retry identity is reconstructed from the original prepared record,
            # never from mappings or tombstones introduced by later commits.
            region_to_id = {
                (image_id, str(key)): int(value)
                for key, value in identity_region_id_mapping.items()
            }
            reserved_negative_ids = {
                object_id for object_id in region_to_id.values() if object_id < 0
            }
            tombstones = set()

        next_negative = min(reserved_negative_ids | {0}) - 1
        for ordinal, region in enumerate(regions):
            key = _region_key(region)
            if key in mapping:
                raise ValidationError(f"duplicate region key: {key}")
            supplied_id = region.get("coco_ann_id")
            namespace = (image_id, key)
            known_id = region_to_id.get(namespace)
            implicit_source_id = _source_id_from_region_key(key, split)
            if implicit_source_id is not None:
                if implicit_source_id not in before_ids:
                    if implicit_source_id in tombstones:
                        raise ValidationError(
                            f"tombstoned source identity cannot be restored: {key!r}"
                        )
                    raise ValidationError(
                        f"source identity {key!r} is not present in the current before row"
                    )
                known_id = implicit_source_id
            elif supplied_id is not None and (
                isinstance(supplied_id, bool)
                or not isinstance(supplied_id, int)
                or supplied_id > 0
            ):
                raise ValidationError(
                    "positive source identity requires its canonical split:coco:<id> region key"
                )
            if known_id is not None and supplied_id not in (None, known_id):
                raise ValidationError(f"region {key!r} changed its hidden identity")
            if known_id is not None:
                object_id = known_id
            elif supplied_id is not None:
                raise ValidationError(
                    "unmapped coco_ann_id values are not accepted from the Draft"
                )
            else:
                while (
                    next_negative in reserved_negative_ids or next_negative in used_ids
                ):
                    next_negative -= 1
                object_id = next_negative
                next_negative -= 1
                allocations[key] = object_id
            if object_id in used_ids:
                raise ValidationError(f"duplicate coco_ann_id: {object_id}")
            if object_id in tombstones and before_ids.get(object_id) is None:
                raise ValidationError(
                    f"tombstoned coco_ann_id cannot be restored: {object_id}"
                )
            used_ids.add(object_id)
            mapping[key] = object_id

            name = region.get("category_name", region.get("desc"))
            category_id = region.get("category_id")
            if not isinstance(name, str) or type(category_id) is not int:
                raise ValidationError(
                    "category_name and integer category_id are required"
                )
            self.registry.validate(name, category_id)
            bbox = _validate_bbox(region.get("bbox_2d"))
            obj: dict[str, Any] = {
                "bbox_2d": bbox,
                "desc": name,
                "category_id": category_id,
                "category_name": name,
                "coco_ann_id": object_id,
            }
            metadata = _training_metadata(region.get("metadata"))
            if metadata:
                obj["metadata"] = metadata
            prior_rank = before_rank.get(object_id)
            creation_ordinal = None
            if prior_rank is None:
                creation_ordinal = region.get("creation_ordinal", ordinal)
                if type(creation_ordinal) is not int or creation_ordinal < 0:
                    raise ValidationError(
                        "creation_ordinal must be a non-negative integer"
                    )
            materialized.append((obj, key, prior_rank, creation_ordinal))

        objects = _stable_order(materialized)
        ordering_seeds = [
            {
                "region_key": key,
                "coco_ann_id": int(obj["coco_ann_id"]),
                "prior_rank": prior_rank,
                "creation_ordinal": creation_ordinal,
            }
            for obj, key, prior_rank, creation_ordinal in materialized
        ]
        ordering_seeds.sort(key=lambda value: value["region_key"])
        projection = {
            "objects": copy.deepcopy(objects),
            "ordering_seeds": ordering_seeds,
        }
        return objects, mapping, allocations, projection

    def _validate_row(self, row: Mapping[str, Any]) -> None:
        try:
            from .models import WorkingRow
        except ImportError:  # pragma: no cover - partial sibling landing fallback
            WorkingRow = None  # type: ignore[assignment,misc]
        if WorkingRow is not None:
            try:
                WorkingRow.from_mapping(row)
            except Exception as exc:
                raise ValidationError(str(exc)) from exc
            return
        required = {"images", "objects", "width", "height", "image_id", "file_name"}
        if not required.issubset(row):
            raise ValidationError(f"row missing fields: {sorted(required - set(row))}")
        if not isinstance(row["objects"], list) or not row["objects"]:
            raise ValidationError("working row objects must be non-empty")
        ids: set[int] = set()
        for obj in row["objects"]:
            if set(obj) - _ACCEPTED_OBJECT_FIELDS:
                raise ValidationError("working object contains unsupported fields")
            _validate_bbox(obj.get("bbox_2d"))
            name = obj.get("category_name")
            if obj.get("desc") != name or not isinstance(name, str):
                raise ValidationError("desc must equal canonical category_name")
            category_id = obj.get("category_id")
            if not isinstance(category_id, int):
                raise ValidationError("category_id must be an integer")
            self.registry.validate(name, category_id)
            object_id = obj.get("coco_ann_id")
            if not isinstance(object_id, int) or object_id == 0 or object_id in ids:
                raise ValidationError("coco_ann_id must be a unique nonzero integer")
            ids.add(object_id)

    def _candidate_working_hash(
        self, row_index: int, after_row: Mapping[str, Any]
    ) -> str:
        digest = hashlib.sha256()
        object_owners: dict[int, int] = {}
        image_ids: set[int] = set()
        for index, current_row, raw in self._iter_rows():
            row = after_row if index == row_index else current_row
            self._validate_row(row)
            _register_split_wide_ids(
                row, object_owners, image_ids, error_type=ValidationError
            )
            encoded = (
                (canonical_json(after_row) + "\n").encode("utf-8")
                if index == row_index
                else raw
            )
            digest.update(encoded)
        return digest.hexdigest()

    def _rewrite_working(
        self, row_index: int, after_row: Mapping[str, Any], expected_hash: str
    ) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=".working.norm.jsonl.", dir=self.split_dir
        )
        temp_path = Path(temp_name)
        replaced = False
        try:
            with os.fdopen(fd, "wb") as output:
                for index, _, raw in self._iter_rows():
                    output.write(
                        (canonical_json(after_row) + "\n").encode("utf-8")
                        if index == row_index
                        else raw
                    )
                output.flush()
                self._fault("working_temp_flushed")
                os.fsync(output.fileno())
                self._fault("working_temp_fsynced")
            if sha256_file(temp_path) != expected_hash:
                raise RecoveryError(
                    "candidate working hash disagrees with prepared record"
                )
            os.replace(temp_path, self.working_path)
            replaced = True
            self._fault("working_replaced")
            _fsync_directory(self.split_dir)
            self._fault("working_directory_fsynced")
        except InjectedCrash:
            raise
        except Exception as exc:
            if replaced:
                raise CommitOutcomeUnknown(
                    "working JSONL was replaced before a durability step failed"
                ) from exc
            raise
        finally:
            temp_path.unlink(missing_ok=True)

    def _publish_manifest(
        self,
        manifest: Mapping[str, Any],
        expected_hash: str,
        *,
        recovery: bool = False,
    ) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=".project.json.", dir=self.split_dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                output.write(canonical_json(manifest) + "\n")
                output.flush()
                self._fault("manifest_temp_flushed", recovery=recovery)
                os.fsync(output.fileno())
                self._fault("manifest_temp_fsynced", recovery=recovery)
            if sha256_json(manifest) != expected_hash:
                raise RecoveryError(
                    "candidate manifest hash disagrees with prepared record"
                )
            os.replace(temp_path, self.manifest_path)
            self._fault("manifest_replaced", recovery=recovery)
            _fsync_directory(self.split_dir)
            self._fault("manifest_directory_fsynced", recovery=recovery)
        finally:
            temp_path.unlink(missing_ok=True)

    def _append_record(self, record: Mapping[str, Any], stage: str) -> dict[str, Any]:
        payload = copy.deepcopy(dict(record))
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["prev_record_hash"] = (
            self._records[-1]["record_hash"] if self._records else None
        )
        payload["record_hash"] = sha256_json(payload)
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        with self.journal_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            self._fault(f"{stage}_journal_flushed")
            os.fsync(handle.fileno())
            self._fault(f"{stage}_journal_fsynced")
        self._records.append(payload)
        return payload

    def _append_terminal(
        self,
        prepared: Mapping[str, Any],
        status: CommitStatus,
        *,
        recovery: bool = False,
        error: str | None = None,
    ) -> None:
        if any(
            record["kind"] == "terminal"
            and record.get("prepared_record_hash") == prepared["record_hash"]
            for record in self._records
        ):
            return
        terminal = {
            "kind": "terminal",
            "commit_id": prepared["commit_id"],
            "prepared_record_hash": prepared["record_hash"],
            "status": status.value,
            "generation": (
                prepared["candidate_generation"]
                if status is CommitStatus.COMMITTED
                else prepared["base_generation"]
            ),
            "recovery": recovery,
            "error": error,
        }
        self._append_record(terminal, "terminal")

    def _reload_journal_index(self, *, repair_torn_tail: bool = False) -> None:
        self._records = []
        self._region_to_id = {}
        self._id_to_region = {}
        self._tombstones = set()
        self._reserved_negative_ids = set()
        self._batch_reservations = {}
        self._records = self._read_journal_records(repair_torn_tail=repair_torn_tail)

        legacy_prepared: dict[str, dict[str, Any]] = {}
        legacy_commit_ids: set[str] = set()
        legacy_terminal_hashes: set[str] = set()
        batch_prepared: dict[str, dict[str, Any]] = {}
        batch_prepared_ids: set[str] = set()
        batch_terminals: dict[str, dict[str, Any]] = {}
        for record in self._records:
            kind = record.get("kind")
            if kind == "reservation":
                self._register_reservation_record(record)
                continue
            if kind == "prepared":
                commit_id = str(record.get("commit_id"))
                if commit_id in legacy_commit_ids:
                    raise RecoveryError(f"duplicate prepared commit id: {commit_id}")
                legacy_commit_ids.add(commit_id)
                legacy_prepared[str(record["record_hash"])] = record
                continue
            if kind == "terminal":
                prepared_hash = str(record.get("prepared_record_hash"))
                prepared = legacy_prepared.get(prepared_hash)
                if prepared is None:
                    raise RecoveryError(
                        "terminal journal record has no preceding prepared record"
                    )
                if prepared_hash in legacy_terminal_hashes:
                    raise RecoveryError(
                        "prepared journal record has multiple terminal outcomes"
                    )
                legacy_terminal_hashes.add(prepared_hash)
                if record.get("commit_id") != prepared.get("commit_id"):
                    raise RecoveryError(
                        "terminal commit identity disagrees with prepared record"
                    )
                if record.get("status") not in {
                    CommitStatus.COMMITTED.value,
                    CommitStatus.ROLLED_BACK.value,
                }:
                    raise RecoveryError("terminal journal status is invalid")
                continue
            if kind == "batch_prepared":
                batch_id = str(record.get("batch_id"))
                if batch_id in batch_prepared_ids:
                    raise RecoveryError(f"duplicate prepared batch id: {batch_id}")
                batch_prepared_ids.add(batch_id)
                batch_prepared[str(record["record_hash"])] = record
                continue
            if kind == "batch_terminal":
                batch_id = str(record.get("batch_id"))
                if batch_id in batch_terminals:
                    raise RecoveryError(
                        f"batch {batch_id} has multiple terminal outcomes"
                    )
                status = record.get("status")
                if status not in {
                    BatchStatus.SUCCEEDED.value,
                    BatchStatus.FAILED.value,
                }:
                    raise RecoveryError("batch terminal journal status is invalid")
                prepared_hash = record.get("prepared_record_hash")
                if prepared_hash is not None:
                    prepared = batch_prepared.get(str(prepared_hash))
                    if prepared is None:
                        raise RecoveryError(
                            "batch terminal has no preceding prepared record"
                        )
                    if record.get("batch_id") != prepared.get("batch_id") or record.get(
                        "payload_hash"
                    ) != prepared.get("payload_hash"):
                        raise RecoveryError(
                            "batch terminal identity disagrees with prepared record"
                        )
                elif status == BatchStatus.SUCCEEDED.value:
                    raise RecoveryError(
                        "successful batch terminal requires a prepared record"
                    )
                batch_terminals[batch_id] = record
                continue
            raise RecoveryError(f"unsupported journal record kind: {kind!r}")

        committed_legacy = {
            prepared_hash
            for prepared_hash in legacy_terminal_hashes
            if next(
                record
                for record in self._records
                if record.get("prepared_record_hash") == prepared_hash
                and record.get("kind") == "terminal"
            )["status"]
            == CommitStatus.COMMITTED.value
        }
        for record in self._records:
            if record["kind"] != "prepared":
                continue
            image_id = int(record["image_id"])
            for key, value in record.get("allocated_mappings", {}).items():
                self._register_region_mapping(image_id, str(key), int(value))
                self._reserved_negative_ids.add(int(value))
            if record["record_hash"] in committed_legacy:
                for key, value in record.get("region_id_mapping", {}).items():
                    self._register_region_mapping(image_id, str(key), int(value))
                self._tombstones.update(
                    int(value) for value in record.get("tombstones", ())
                )

        for terminal in batch_terminals.values():
            if terminal["status"] != BatchStatus.SUCCEEDED.value:
                continue
            prepared = batch_prepared[str(terminal["prepared_record_hash"])]
            for member in prepared.get("members", ()):
                image_id = int(member["image_id"])
                for key, value in member.get("region_id_mapping", {}).items():
                    self._register_region_mapping(image_id, str(key), int(value))
                    if int(value) < 0:
                        self._reserved_negative_ids.add(int(value))
                self._tombstones.update(
                    int(value) for value in member.get("tombstones", ())
                )

    def _read_journal_records(self, *, repair_torn_tail: bool) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        data = self.journal_path.read_bytes()
        complete = data
        tail = b""
        if data and not data.endswith(b"\n"):
            boundary = data.rfind(b"\n") + 1
            complete = data[:boundary]
            tail = data[boundary:]

        previous_hash: str | None = None
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(complete.splitlines(keepends=True), start=1):
            try:
                line = raw.decode("utf-8")
                record = _strict_json_loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RecoveryError(
                    f"invalid journal record at line {line_no}"
                ) from exc
            if record.get("prev_record_hash") != previous_hash:
                raise RecoveryError(f"broken journal chain at line {line_no}")
            recorded_hash = record.get("record_hash")
            check = dict(record)
            check.pop("record_hash", None)
            if recorded_hash != sha256_json(check):
                raise RecoveryError(f"journal hash disagreement at line {line_no}")
            previous_hash = recorded_hash
            records.append(record)

        if tail:
            if not repair_torn_tail:
                raise RecoveryError(
                    f"incomplete journal record at line {len(records) + 1}"
                )
            if not tail.startswith(b"{") or b"\n" in tail:
                raise RecoveryError(
                    "final journal bytes are not an unambiguous torn frame"
                )
            with self.journal_path.open("r+b") as handle:
                handle.truncate(len(complete))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.split_dir)
        return records

    def _validate_queue_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        enqueues: dict[str, Mapping[str, Any]] = {}
        claims: set[str] = set()
        for record in records:
            kind = record.get("kind")
            batch_id = record.get("batch_id")
            if type(batch_id) is not str or not batch_id.strip():
                raise RecoveryError("queue record has invalid batch identity")
            if kind == "enqueue":
                if batch_id in enqueues:
                    raise RecoveryError(
                        f"duplicate enqueue record for batch {batch_id}"
                    )
                payload = record.get("payload")
                payload_hash = record.get("payload_hash")
                split = record.get("split")
                base_generation = record.get("base_generation")
                member_count = record.get("member_count")
                if (
                    not _is_sha256(payload_hash)
                    or type(split) is not str
                    or type(base_generation) is not int
                    or base_generation < 0
                    or type(member_count) is not int
                    or member_count < 1
                ):
                    raise RecoveryError("queue enqueue payload attestation failed")
                batch = _validate_batch_payload(payload, error_type=RecoveryError)
                if (
                    sha256_json(batch) != payload_hash
                    or batch["batch_id"] != batch_id
                    or batch["split"] != split
                    or batch["base_generation"] != base_generation
                    or len(batch["members"]) != member_count
                ):
                    raise RecoveryError("queue enqueue payload attestation failed")
                enqueues[batch_id] = record
                continue
            enqueue = enqueues.get(batch_id)
            if enqueue is None:
                raise RecoveryError("queue projection has no preceding enqueue record")
            if record.get("payload_hash") != enqueue.get("payload_hash"):
                raise RecoveryError("queue projection payload identity disagrees")
            if kind == "claim":
                if batch_id in claims:
                    raise RecoveryError(f"batch {batch_id} has multiple queue claims")
                claims.add(batch_id)
                continue
            if kind == "queue_terminal":
                if record.get("status") not in {
                    BatchStatus.SUCCEEDED.value,
                    BatchStatus.FAILED.value,
                }:
                    raise RecoveryError("queue terminal status is invalid")
                if (
                    type(record.get("generation")) is not int
                    or record["generation"] < 0
                    or not _is_sha256(record.get("working_sha256"))
                    or (
                        record.get("error") is not None
                        and type(record.get("error")) is not str
                    )
                ):
                    raise RecoveryError("queue terminal attestation is invalid")
                continue
            raise RecoveryError(f"unsupported queue record kind: {kind!r}")

    def _register_region_mapping(self, image_id: int, key: str, object_id: int) -> None:
        namespace = (image_id, key)
        prior = self._region_to_id.setdefault(namespace, object_id)
        if prior != object_id:
            raise RecoveryError(
                f"region identity disagreement for image {image_id}, key {key!r}"
            )
        prior_owner = self._id_to_region.setdefault(object_id, namespace)
        if prior_owner != namespace:
            raise RecoveryError(
                f"split-wide coco_ann_id {object_id} is mapped to multiple regions"
            )

    def _register_reservation_record(self, record: Mapping[str, Any]) -> None:
        key = record.get("stable_region_key")
        image_id = record.get("image_id")
        object_id = record.get("coco_ann_id")
        if (
            not isinstance(key, str)
            or not key
            or isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id >= 0
        ):
            raise RecoveryError("invalid batch allocation reservation")
        prior = self._batch_reservations.setdefault(key, (image_id, object_id))
        if prior != (image_id, object_id):
            raise RecoveryError(f"stable region key reservation disagreement: {key!r}")
        self._register_region_mapping(image_id, key, object_id)
        self._reserved_negative_ids.add(object_id)

    def _validate_authority(self) -> None:
        manifest = self._read_manifest()
        actual = sha256_file(self.working_path)
        if actual != manifest.get("working_sha256"):
            raise RecoveryError("working JSONL does not match published manifest")
        object_owners: dict[int, int] = {}
        image_ids: set[int] = set()
        line_count = 0
        for _, row, _ in self._iter_rows():
            line_count += 1
            self._validate_row(row)
            _register_split_wide_ids(
                row, object_owners, image_ids, error_type=RecoveryError
            )
            image_id = int(row["image_id"])
            for obj in row["objects"]:
                object_id = int(obj["coco_ann_id"])
                if object_id < 0:
                    owner = self._id_to_region.get(object_id)
                    if owner is None or owner[0] != image_id:
                        raise RecoveryError(
                            f"negative coco_ann_id {object_id} has no authoritative journal mapping"
                        )
        if line_count != int(manifest.get("working_line_count", -1)):
            raise RecoveryError(
                "working JSONL line count does not match published manifest"
            )

    def _records_for_commit(self, commit_id: str) -> list[dict[str, Any]]:
        return [
            record for record in self._records if record.get("commit_id") == commit_id
        ]

    def _read_manifest(self) -> dict[str, Any]:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = _strict_json_loads(handle.read())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RecoveryError(f"cannot read manifest: {self.manifest_path}") from exc
        return manifest

    def _load_task_index(self) -> None:
        manifest = self._read_manifest()
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ManifestDriftError("schema_version")
        expected_hash = manifest.get("task_index_sha256")
        if not isinstance(expected_hash, str) or not self.task_index_path.is_file():
            raise ManifestDriftError("task_index_sha256")
        if sha256_file(self.task_index_path) != expected_hash:
            raise ManifestDriftError("task_index_sha256")
        try:
            payload = _strict_json_loads(
                self.task_index_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ManifestDriftError("task_index") from exc
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("split") != manifest.get("split")
            or not isinstance(payload.get("entries"), list)
        ):
            raise ManifestDriftError("task_index")
        entries = payload["entries"]
        if len(entries) != int(manifest.get("task_count", -1)):
            raise ManifestDriftError("task_index")
        image_to_index: dict[int, int] = {}
        for expected_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ManifestDriftError("task_index")
            image_id = entry.get("image_id")
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or entry.get("source_row_index") != expected_index
                or entry.get("source_line") != expected_index + 1
                or entry.get("task_id") != _task_id(str(manifest["split"]), image_id)
                or image_id in image_to_index
            ):
                raise ManifestDriftError("task_index")
            image_to_index[image_id] = expected_index
        self._source_row_by_image = image_to_index

    def _load_row_cache(self) -> None:
        images: dict[int, int] = {}
        object_ids: dict[int, set[int]] = {}
        hashes: dict[int, str] = {}
        for source_row_index, row, _ in self._iter_rows():
            image_id = int(row.get("image_id", -1))
            if self._source_row_by_image.get(image_id) != source_row_index:
                raise RecoveryError("working source row index attestation failed")
            images[source_row_index] = image_id
            object_ids[source_row_index] = {
                int(obj["coco_ann_id"]) for obj in row["objects"]
            }
            hashes[source_row_index] = sha256_json(row)
        manifest = self._read_manifest()
        if len(images) != int(manifest.get("working_line_count", -1)):
            raise RecoveryError("working input line-count attestation failed")
        self._row_image_cache = images
        self._row_object_ids_cache = object_ids
        self._row_hash_cache = hashes
        self._cache_generation = int(manifest["generation"])
        self._cache_working_sha256 = str(manifest["working_sha256"])
        self._cache_line_count = len(images)

    def _refresh_row_cache_from_journal(self, manifest: Mapping[str, Any]) -> None:
        target_generation = int(manifest["generation"])
        if target_generation < self._cache_generation:
            raise RecoveryError("published generation moved behind the row cache")
        if target_generation == self._cache_generation:
            if (
                manifest.get("working_sha256") != self._cache_working_sha256
                or int(manifest.get("working_line_count", -1)) != self._cache_line_count
            ):
                raise RecoveryError(
                    "row freshness cache publication attestation drifted"
                )
            return

        updates: dict[int, tuple[str, int, Sequence[Mapping[str, Any]]]] = {}
        for prepared in self._records:
            kind = prepared.get("kind")
            if kind == "prepared":
                terminal = next(
                    (
                        record
                        for record in self._records
                        if record.get("kind") == "terminal"
                        and record.get("prepared_record_hash")
                        == prepared.get("record_hash")
                        and record.get("status") == CommitStatus.COMMITTED.value
                    ),
                    None,
                )
                if terminal is None:
                    continue
                generation = int(prepared["candidate_generation"])
                rows: Sequence[Mapping[str, Any]] = (
                    {
                        "image_id": prepared["image_id"],
                        "after_row": prepared["after_row"],
                        "after_row_hash": prepared["after_row_hash"],
                    },
                )
                candidate_hash = str(prepared["candidate_working_sha256"])
                line_count = self._cache_line_count
            elif kind == "batch_prepared":
                terminal = next(
                    (
                        record
                        for record in self._records
                        if record.get("kind") == "batch_terminal"
                        and record.get("prepared_record_hash")
                        == prepared.get("record_hash")
                        and record.get("status") == BatchStatus.SUCCEEDED.value
                    ),
                    None,
                )
                if terminal is None:
                    continue
                generation = int(prepared["candidate_generation"])
                rows = prepared["members"]
                candidate_hash = str(prepared["candidate_working_sha256"])
                line_count = int(prepared["candidate_working_line_count"])
            else:
                continue
            if generation in updates:
                raise RecoveryError(
                    f"multiple committed transactions at generation {generation}"
                )
            updates[generation] = (candidate_hash, line_count, rows)

        current_generation = self._cache_generation
        for generation in sorted(
            value for value in updates if value > current_generation
        ):
            if generation != current_generation + 1:
                raise RecoveryError(
                    "journal cannot advance the row freshness cache contiguously"
                )
            candidate_hash, line_count, rows = updates[generation]
            for row_update in rows:
                image_id = int(row_update["image_id"])
                source_row_index = self._source_row_by_image.get(image_id)
                if source_row_index is None:
                    raise RecoveryError(
                        "journal row is absent from the immutable task index"
                    )
                after_row = row_update["after_row"]
                if sha256_json(after_row) != row_update["after_row_hash"]:
                    raise RecoveryError("journal row hash attestation failed")
                self._row_image_cache[source_row_index] = image_id
                self._row_object_ids_cache[source_row_index] = {
                    int(obj["coco_ann_id"]) for obj in after_row["objects"]
                }
                self._row_hash_cache[source_row_index] = str(
                    row_update["after_row_hash"]
                )
            self._cache_working_sha256 = candidate_hash
            self._cache_line_count = line_count
            current_generation = generation
        self._cache_generation = current_generation
        if (
            self._cache_generation != target_generation
            or self._cache_working_sha256 != manifest.get("working_sha256")
            or self._cache_line_count != int(manifest.get("working_line_count", -1))
        ):
            raise RecoveryError("journal cannot attest the current row freshness cache")

    def _append_queue_record(
        self, record: Mapping[str, Any], stage: str
    ) -> dict[str, Any]:
        records = self._read_queue_records()
        payload = copy.deepcopy(dict(record))
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["prev_record_hash"] = records[-1]["record_hash"] if records else None
        payload["record_hash"] = sha256_json(payload)
        _validate_queue_record_envelope(payload)
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        with self.queue_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            self._fault(f"{stage}_queue_flushed")
            os.fsync(handle.fileno())
            self._fault(f"{stage}_queue_fsynced")
        return payload

    def _read_queue_records(
        self, *, repair_torn_tail: bool = False
    ) -> list[dict[str, Any]]:
        if not self.queue_path.exists():
            raise RecoveryError(f"missing queue: {self.queue_path}")
        data = self.queue_path.read_bytes()
        complete = data
        tail = b""
        if data and not data.endswith(b"\n"):
            boundary = data.rfind(b"\n") + 1
            complete = data[:boundary]
            tail = data[boundary:]
        previous_hash: str | None = None
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(complete.splitlines(), start=1):
            try:
                record = _strict_json_loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RecoveryError(f"invalid queue record at line {line_no}") from exc
            if type(record) is not dict:
                raise RecoveryError(f"invalid queue record at line {line_no}")
            _validate_queue_record_envelope(record)
            if record.get("prev_record_hash") != previous_hash:
                raise RecoveryError(f"broken queue chain at line {line_no}")
            recorded_hash = record.get("record_hash")
            check = dict(record)
            check.pop("record_hash", None)
            if recorded_hash != sha256_json(check):
                raise RecoveryError(f"queue hash disagreement at line {line_no}")
            previous_hash = recorded_hash
            records.append(record)
        if tail:
            if not repair_torn_tail:
                raise RecoveryError("incomplete queue record")
            if not tail.startswith(b"{") or b"\n" in tail:
                raise RecoveryError(
                    "final queue bytes are not an unambiguous torn frame"
                )
            with self.queue_path.open("r+b") as handle:
                handle.truncate(len(complete))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.split_dir)
        self._validate_queue_records(records)
        return records

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any], bytes]]:
        with self.working_path.open("rb") as handle:
            for index, raw in enumerate(handle):
                row = _parse_working_jsonl_row(raw, index)
                yield index, row, raw

    def _scan_attested_restore_rows(
        self,
        requested_image_ids: set[int],
        manifest: Mapping[str, Any],
    ) -> dict[int, dict[str, Any]]:
        """Scan once, attest the complete file, then expose requested rows."""

        digest = hashlib.sha256()
        line_count = 0
        rows: dict[int, dict[str, Any]] = {}
        validation_error: ValidationError | None = None
        with self.working_path.open("rb") as handle:
            for index, raw in enumerate(handle):
                digest.update(raw)
                line_count += 1
                try:
                    row = _parse_working_jsonl_row(raw, index)
                except ValidationError as exc:
                    if validation_error is None:
                        validation_error = exc
                    continue
                row_image_id = row.get("image_id")
                if row_image_id not in requested_image_ids:
                    continue
                if row_image_id in rows:
                    if validation_error is None:
                        validation_error = ValidationError(
                            f"duplicate image_id: {row_image_id}"
                        )
                    continue
                rows[row_image_id] = row

        if digest.hexdigest() != manifest.get("working_sha256") or line_count != int(
            manifest.get("working_line_count", -1)
        ):
            raise RecoveryError(
                "working file does not match published hash/line-count attestation"
            )
        if validation_error is not None:
            raise validation_error
        return rows

    def _find_row(self, image_id: int) -> tuple[int, dict[str, Any], bytes]:
        found: tuple[int, dict[str, Any], bytes] | None = None
        for item in self._iter_rows():
            if int(item[1].get("image_id", -1)) == image_id:
                if found is not None:
                    raise ValidationError(f"duplicate image_id: {image_id}")
                found = item
        if found is None:
            raise StaleCommitError(f"unknown image_id: {image_id}")
        return found

    def _key_for_object(self, obj: Mapping[str, Any], image_id: int, split: str) -> str:
        object_id = int(obj["coco_ann_id"])
        if object_id > 0:
            return _source_region_key(split, object_id)
        keys = [
            key
            for (mapped_image_id, key), value in self._region_to_id.items()
            if mapped_image_id == image_id and value == object_id
        ]
        if len(keys) != 1:
            raise RecoveryError(f"cannot rehydrate region key for {object_id}")
        return keys[0]

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(
                    f"split is already locked: {self.split_dir}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _shared_lock(self) -> Iterator[None]:
        self._assert_serving_ready()
        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(f"split is reconciling: {self.split_dir}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _supported_reader_lock(self) -> Iterator[None]:
        with self._shared_lock():
            with self._shared_queue_lock():
                queue_records = self._read_queue_records()
            reason = self._batch_reconciliation_reason(queue_records)
            if reason is not None:
                raise RecoveryError(f"store requires recovery: {reason}")
            yield

    @contextmanager
    def _exclusive_queue_lock(self) -> Iterator[None]:
        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.queue_lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(
                    f"split queue is already locked: {self.split_dir}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _serialized_queue_admission_lock(self) -> Iterator[None]:
        """Serialize only the short queue admission/recheck critical sections."""

        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.queue_lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _batch_admission_reader_lock(self) -> Iterator[None]:
        """Hold reconciled shared queue/file authority without exclusive locks."""

        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.queue_lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                with self._shared_lock():
                    queue_records = self._read_queue_records()
                    reason = self._batch_reconciliation_reason(queue_records)
                    if reason is not None:
                        raise RecoveryError(f"store requires recovery: {reason}")
                    yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _legacy_commit_lock(self) -> Iterator[None]:
        self._assert_serving_ready()
        with self._exclusive_queue_lock():
            records = self._read_queue_records()
            active = self._active_queue_enqueue(records)
            if active is not None:
                raise StoreBusyError(
                    f"split has active batch {active['batch_id']}: {self.split_dir}"
                )
            with self._exclusive_lock():
                reason = self._batch_reconciliation_reason(records)
                if reason is not None:
                    raise RecoveryError(f"store requires recovery: {reason}")
                yield

    @contextmanager
    def _shared_queue_lock(self) -> Iterator[None]:
        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.queue_lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(
                    f"split queue is being updated: {self.split_dir}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _fault(self, boundary: str, *, recovery: bool = False) -> None:
        if self._fault_injector is not None and not recovery:
            try:
                self._fault_injector(boundary)
            except InjectedCrash:
                self._recovery_required = True
                raise

    def _assert_serving_ready(self) -> None:
        if self._recovery_required:
            raise RecoveryError(
                "store requires recovery after an interrupted transaction"
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


def _validate_bbox(value: Any) -> list[int]:
    try:
        from src.data.geometry import validate_bbox_bins
    except ImportError:  # pragma: no cover - partial sibling landing fallback
        validate_bbox_bins = None  # type: ignore[assignment]
    if validate_bbox_bins is not None:
        try:
            return list(validate_bbox_bins(value, field="bbox_2d"))
        except Exception as exc:
            raise ValidationError(str(exc)) from exc
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValidationError("bbox_2d must contain four integer norm1000 edges")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        raise ValidationError("bbox_2d must be strict and inside 0..999")
    return [x1, y1, x2, y2]


def _stable_order(
    values: Sequence[tuple[dict[str, Any], str, int | None, int | None]],
) -> list[dict[str, Any]]:
    try:
        from .models import (
            ObjectIdentity,
            OrderedWorkingObject,
            WorkingObject,
            stable_top_left_order,
        )
    except ImportError:  # pragma: no cover - partial sibling landing fallback
        seeded = sorted(
            values,
            key=lambda item: (
                0 if item[2] is not None else 1,
                item[2] if item[2] is not None else item[3],
                item[1],
            ),
        )
        seeded.sort(key=lambda item: (item[0]["bbox_2d"][1], item[0]["bbox_2d"][0]))
        return [item[0] for item in seeded]

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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_selected_source(source: Path, split: str) -> None:
    expected_suffix = (
        Path("public_data/coco/rescale_32_1024_bbox_len12000") / f"{split}.norm.jsonl"
    )
    if source.parts[-len(expected_suffix.parts) :] != expected_suffix.parts:
        raise ManifestDriftError("source_path")


def _default_registry() -> CategoryRegistry:
    try:
        from .categories import COCO80_REGISTRY
    except ImportError as exc:  # pragma: no cover - only during partial sibling landing
        raise StoreError(
            "Coco80Registry is not available; pass registry explicitly"
        ) from exc
    return COCO80_REGISTRY


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


def _working_image_locator(
    row: Mapping[str, Any], split: str, source: Path, image_root: Path
) -> str:
    locator = row["images"][0]
    if not isinstance(locator, str):
        raise ValidationError("image locator must be a string")
    resolved = (source.parent / locator).resolve(strict=True)
    try:
        relative = resolved.relative_to(image_root)
    except ValueError as exc:
        raise ValidationError(
            "source image escapes the allowlisted image root"
        ) from exc
    expected_dir = f"{split}2017"
    if not relative.parts or relative.parts[0] != expected_dir:
        raise ValidationError("source image does not belong to its split storage")
    return (Path("images") / relative).as_posix()


def _validate_managed_link(link: Path, expected_target: Path) -> None:
    if not link.is_symlink() or link.resolve(strict=True) != expected_target:
        raise ManifestDriftError("managed_image_link")


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
