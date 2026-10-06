"""Working-dataset requests, receipts, errors and durable schema identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from .store import WorkingDatasetStore


SCHEMA_VERSION = 2
TASK_INDEX_IDENTITY_SCHEMA_VERSION = 1

BOOTSTRAP_TASK_INDEX_DIRECTORY_NAME = "bootstrap-task-index"

BOOTSTRAP_TASK_INDEX_SCHEMA_KEY = "bootstrap_task_index_schema_version"

BOOTSTRAP_TASK_INDEX_BUILD_KEY = "bootstrap_task_index_build_key"

BOOTSTRAP_TASK_INDEX_SHA256_KEY = "bootstrap_task_index_sha256"

BOOTSTRAP_TASK_MANIFEST_KEY = "bootstrap_task_manifest_fingerprint"

_BOOTSTRAP_TASK_INDEX_BINDING_KEYS = frozenset(
    {
        BOOTSTRAP_TASK_INDEX_SCHEMA_KEY,
        BOOTSTRAP_TASK_INDEX_BUILD_KEY,
        BOOTSTRAP_TASK_INDEX_SHA256_KEY,
        BOOTSTRAP_TASK_MANIFEST_KEY,
    }
)


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
    """Proof returned only after an adapter durably saves the frozen Draft."""

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
class StoreStateSnapshot:
    """Small public store authority for status APIs; never scans working rows."""

    split: str
    project_id: str
    generation: int
    task_count: int


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
