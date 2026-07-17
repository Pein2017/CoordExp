"""Thin asynchronous Commit and project-state service over the proven core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any, Protocol

from src.common.errors import RuntimeContractError
from src.coco_refinement.models import Split
from src.coco_refinement.repository import SqliteDraftRepository
from src.label_studio_coco_refinement.runtime import (
    AuthenticatedPrincipal,
    BatchStatusReceipt,
    DraftCatalogError,
    NoEligibleDraftsError,
)
from src.label_studio_coco_refinement.store import (
    BatchStatus,
    CommitConflictError,
    RecoveryError,
    StoreBusyError,
    StoreError,
    StoreStateSnapshot,
    ValidationError,
    WorkingDatasetStore,
)


LOCAL_OPERATOR = "local-operator"
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_WORKER_STATES = {
    "stopped",
    "recovering",
    "busy",
    "idle",
    "running",
    "stopping",
    "failed",
}


class CommitServiceError(RuntimeContractError):
    """One Commit/status request cannot be served from public authority."""


class _RuntimeCore(Protocol):
    def capture_and_enqueue(
        self, *, split: str, batch_id: str, principal: AuthenticatedPrincipal
    ) -> BatchStatusReceipt: ...

    def batch_status(self, *, split: str, batch_id: str) -> BatchStatusReceipt: ...

    def worker_health(self, split: str | None = None) -> object: ...


@dataclass(frozen=True)
class SafeCommitStatus:
    batch_id: str
    payload_hash: str | None
    status: str
    split: str
    member_count: int
    base_generation: int | None
    generation: int | None

    @classmethod
    def from_receipt(
        cls,
        receipt: object,
        *,
        expected_split: Split,
        expected_batch_id: str | None = None,
    ) -> SafeCommitStatus:
        if not isinstance(receipt, BatchStatusReceipt):
            _invalid_commit_status()
        assert isinstance(receipt, BatchStatusReceipt)
        allowed_statuses = {
            BatchStatus.QUEUED.value,
            BatchStatus.RUNNING.value,
            BatchStatus.RECONCILING.value,
            BatchStatus.SUCCEEDED.value,
            BatchStatus.FAILED.value,
        }
        if (
            not _is_batch_id(receipt.batch_id)
            or (
                expected_batch_id is not None
                and receipt.batch_id != expected_batch_id
            )
            or receipt.split != expected_split
            or not isinstance(receipt.status, str)
            or receipt.status not in allowed_statuses
            or not _is_sha256(receipt.payload_hash)
            or not _is_nonnegative_int(receipt.member_count)
            or receipt.member_count == 0
            or not _is_nonnegative_int(receipt.base_generation)
            or not (receipt.error is None or isinstance(receipt.error, str))
        ):
            _invalid_commit_status()
        base_generation = receipt.base_generation
        generation = receipt.generation
        if receipt.status in {BatchStatus.QUEUED.value, BatchStatus.RUNNING.value}:
            if generation is not None:
                _invalid_commit_status()
        elif receipt.status == BatchStatus.RECONCILING.value:
            if generation is not None and (
                not _is_nonnegative_int(generation) or generation < base_generation
            ):
                _invalid_commit_status()
        elif receipt.status == BatchStatus.SUCCEEDED.value:
            if not _is_nonnegative_int(generation) or generation <= base_generation:
                _invalid_commit_status()
        elif not _is_nonnegative_int(generation) or generation < base_generation:
            _invalid_commit_status()
        return cls(
            batch_id=receipt.batch_id,
            payload_hash=receipt.payload_hash,
            status=receipt.status,
            split=expected_split,
            member_count=receipt.member_count,
            base_generation=receipt.base_generation,
            generation=receipt.generation,
        )

    def to_dict(self) -> dict[str, Any]:
        detail: dict[str, str] | None = None
        if self.status == BatchStatus.RECONCILING.value:
            detail = {
                "code": "coco_refinement.commit_reconciling",
                "message": "Commit authority is reconciling; retry status shortly.",
            }
        elif self.status == BatchStatus.FAILED.value:
            detail = {
                "code": "coco_refinement.commit_failed",
                "message": "Commit failed; inspect the local durable receipt before retrying.",
            }
        return {
            "base_generation": self.base_generation,
            "batch_id": self.batch_id,
            "detail": detail,
            "generation": self.generation,
            "member_count": self.member_count,
            "payload_hash": self.payload_hash,
            "split": self.split,
            "status": self.status,
        }


@dataclass(frozen=True)
class ProjectState:
    split: Split
    project_id: str
    task_count: int
    generation: int
    pending_draft_count: int
    accepting_writes: bool
    worker: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepting_writes": self.accepting_writes,
            "generation": self.generation,
            "pending_draft_count": self.pending_draft_count,
            "project_id": self.project_id,
            "split": self.split,
            "task_count": self.task_count,
            "worker": dict(self.worker),
        }


class CommitService:
    """Expose durable batch enqueue/status without owning worker lifecycle."""

    def __init__(
        self,
        *,
        repository: SqliteDraftRepository,
        runtime: _RuntimeCore,
        stores: Mapping[str, WorkingDatasetStore],
        project_ids: Mapping[str, str],
        accepting_writes: Callable[[], bool],
        state_attempts: int = 3,
    ) -> None:
        if set(stores) != {"train", "val"} or set(project_ids) != {"train", "val"}:
            raise CommitServiceError(
                "Commit service requires exact train and val bindings",
                code="coco_refinement.commit_service_splits",
            )
        if isinstance(state_attempts, bool) or not 1 <= state_attempts <= 5:
            raise CommitServiceError(
                "state_attempts must be from 1 through 5",
                code="coco_refinement.commit_state_attempts",
            )
        self.repository = repository
        self.runtime = runtime
        self.stores = {split: stores[split] for split in ("train", "val")}
        self.project_ids = {
            split: str(project_ids[split]) for split in ("train", "val")
        }
        self.accepting_writes = accepting_writes
        self.state_attempts = state_attempts

    @classmethod
    def from_runtime(cls, runtime: object) -> CommitService:
        workspace = getattr(runtime, "workspace", None)
        stores = getattr(runtime, "stores", None)
        project_ids = getattr(runtime, "project_ids", None)
        core = getattr(runtime, "runtime", None)
        if (
            workspace is None
            or not isinstance(stores, Mapping)
            or not isinstance(project_ids, Mapping)
            or core is None
        ):
            raise CommitServiceError(
                "standalone runtime lacks Commit service bindings",
                code="coco_refinement.commit_service_runtime",
            )
        return cls(
            repository=workspace.repository,
            runtime=core,
            stores=stores,
            project_ids=project_ids,
            accepting_writes=lambda: bool(getattr(runtime, "accepting_writes", False)),
        )

    def enqueue(self, *, split: str, batch_id: str) -> SafeCommitStatus:
        selected = _split(split)
        if not self.accepting_writes():
            try:
                return self.status(split=selected, batch_id=batch_id)
            except CommitServiceError as exc:
                if exc.code != "coco_refinement.commit_not_found":
                    raise
                raise CommitServiceError(
                    "runtime is not accepting Commit writes",
                    code="coco_refinement.runtime_not_ready",
                    cause=exc,
                ) from exc
        try:
            receipt = self.runtime.capture_and_enqueue(
                split=selected,
                batch_id=batch_id,
                principal=AuthenticatedPrincipal(
                    user_id=LOCAL_OPERATOR,
                    authenticated=True,
                ),
            )
        except NoEligibleDraftsError as exc:
            raise CommitServiceError(
                "no eligible durable Drafts are available",
                code="coco_refinement.no_pending_drafts",
                cause=exc,
            ) from exc
        except DraftCatalogError as exc:
            raise CommitServiceError(
                "durable Draft capture is unavailable",
                code="coco_refinement.commit_unavailable",
                cause=exc,
            ) from exc
        except CommitConflictError as exc:
            raise CommitServiceError(
                "batch identity conflicts with a durable batch",
                code="coco_refinement.commit_conflict",
                cause=exc,
            ) from exc
        except StoreBusyError as exc:
            raise CommitServiceError(
                "split Commit authority is busy",
                code="coco_refinement.commit_busy",
                cause=exc,
            ) from exc
        except RecoveryError as exc:
            raise CommitServiceError(
                "split Commit authority requires recovery",
                code="coco_refinement.commit_recovery",
                cause=exc,
            ) from exc
        except ValidationError as exc:
            raise CommitServiceError(
                "one or more durable Drafts are invalid",
                code="coco_refinement.commit_invalid_draft",
                cause=exc,
            ) from exc
        except StoreError as exc:
            raise CommitServiceError(
                "Commit enqueue authority is unavailable",
                code="coco_refinement.commit_unavailable",
                cause=exc,
            ) from exc
        safe = SafeCommitStatus.from_receipt(receipt, expected_split=selected)
        if safe.batch_id != batch_id:
            raise CommitServiceError(
                "another batch already owns the split queue",
                code="coco_refinement.commit_busy",
            )
        return safe

    def status(self, *, split: str, batch_id: str) -> SafeCommitStatus:
        selected = _split(split)
        try:
            receipt = self.runtime.batch_status(split=selected, batch_id=batch_id)
        except RecoveryError as exc:
            raise CommitServiceError(
                "split Commit authority requires recovery",
                code="coco_refinement.commit_recovery",
                cause=exc,
            ) from exc
        except StoreError as exc:
            raise CommitServiceError(
                "Commit status authority is unavailable",
                code="coco_refinement.commit_unavailable",
                cause=exc,
            ) from exc
        if not isinstance(receipt, BatchStatusReceipt):
            _invalid_commit_status()
        if receipt.status == BatchStatus.NOT_FOUND.value:
            if (
                not _is_batch_id(receipt.batch_id)
                or receipt.batch_id != batch_id
                or receipt.payload_hash is not None
                or receipt.split is not None
                or type(receipt.member_count) is not int
                or receipt.member_count != 0
                or receipt.base_generation is not None
                or receipt.generation is not None
                or receipt.error is not None
            ):
                _invalid_commit_status()
            raise CommitServiceError(
                "batch was not found",
                code="coco_refinement.commit_not_found",
            )
        return SafeCommitStatus.from_receipt(
            receipt,
            expected_split=selected,
            expected_batch_id=batch_id,
        )

    def project_state(self, *, split: str) -> ProjectState:
        selected = _split(split)
        store = self.stores[selected]
        project_id = self.project_ids[selected]
        for _attempt in range(self.state_attempts):
            try:
                first = store.state_snapshot()
                first_pending = self.repository.capture_pending_draft_states(
                    project_id=project_id, split=selected
                )
                task_count = self.repository.count_tasks(project_id=project_id)
                second_pending = self.repository.capture_pending_draft_states(
                    project_id=project_id, split=selected
                )
                second = store.state_snapshot()
            except StoreError as exc:
                raise CommitServiceError(
                    "project store state is unavailable",
                    code="coco_refinement.commit_recovery",
                    cause=exc,
                ) from exc
            if (
                first != second
                or first_pending != second_pending
                or not _state_matches(
                    first,
                    split=selected,
                    project_id=project_id,
                    generation=second_pending.current_generation,
                    task_count=task_count,
                )
            ):
                continue
            return ProjectState(
                split=selected,
                project_id=project_id,
                task_count=task_count,
                generation=first.generation,
                pending_draft_count=len(second_pending.states),
                accepting_writes=bool(self.accepting_writes()),
                worker=_safe_worker_health(
                    self.runtime.worker_health(selected), expected_split=selected
                ),
            )
        raise CommitServiceError(
            "SQLite and store project state did not converge",
            code="coco_refinement.commit_state_reconciling",
        )


def _state_matches(
    snapshot: StoreStateSnapshot,
    *,
    split: Split,
    project_id: str,
    generation: int,
    task_count: int,
) -> bool:
    return (
        snapshot.split == split
        and snapshot.project_id == project_id
        and snapshot.generation == generation
        and snapshot.task_count == task_count
    )


def _safe_worker_health(value: object, *, expected_split: Split) -> dict[str, Any]:
    try:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
    except Exception as exc:
        raise CommitServiceError(
            "worker health cannot be materialized",
            code="coco_refinement.worker_health_invalid",
            cause=exc,
        ) from exc
    if not isinstance(value, Mapping):
        raise CommitServiceError(
            "worker health has an invalid shape",
            code="coco_refinement.worker_health_invalid",
        )
    split = value.get("split")
    state = value.get("state")
    thread_alive = value.get("thread_alive")
    healthy = value.get("healthy")
    processed_batches = value.get("processed_batches")
    last_batch_id = value.get("last_batch_id")
    error = value.get("error")
    expected_healthy = thread_alive is True and state in {"idle", "running"}
    if (
        split != expected_split
        or not isinstance(state, str)
        or state not in _WORKER_STATES
        or type(thread_alive) is not bool
        or type(healthy) is not bool
        or healthy is not expected_healthy
        or not _is_nonnegative_int(processed_batches)
        or not (last_batch_id is None or _is_batch_id(last_batch_id))
        or not (error is None or isinstance(error, str))
    ):
        raise CommitServiceError(
            "worker health has invalid fields",
            code="coco_refinement.worker_health_invalid",
        )
    return {
        "error": (
            None if error is None else "worker failure; inspect the local runtime receipt"
        ),
        "healthy": healthy,
        "last_batch_id": last_batch_id,
        "processed_batches": processed_batches,
        "split": split,
        "state": state,
        "thread_alive": thread_alive,
    }


def _invalid_commit_status() -> None:
    raise CommitServiceError(
        "runtime returned an invalid batch status receipt",
        code="coco_refinement.commit_status_invalid",
    )


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_batch_id(value: object) -> bool:
    return isinstance(value, str) and _BATCH_ID_RE.fullmatch(value) is not None


def _split(value: str) -> Split:
    if value not in ("train", "val"):
        raise CommitServiceError(
            "split must be train or val",
            code="coco_refinement.split",
        )
    return value  # type: ignore[return-value]


__all__ = [
    "CommitService",
    "CommitServiceError",
    "ProjectState",
    "SafeCommitStatus",
]
