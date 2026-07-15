"""Vendor-neutral Draft capture and asynchronous split batch coordination.

The browser never constructs a :class:`BatchRequest`.  An authenticated server
adapter captures the current user's complete canonical Draft snapshots, while
this module resolves immutable task positions from the exact working store and
signals a split-local worker only after durable enqueue has returned.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from src.label_studio_coco_refinement.store import (
    BatchEnqueueReceipt,
    BatchMember,
    BatchRequest,
    BatchStatus,
    BatchStatusView,
    CommitRequest,
    DraftSaveReceipt,
    StoreBusyError,
    StoreError,
    ValidationError,
    WorkingDatasetStore,
    semantic_hash,
)


_JsonScalar = str | int | float | bool | None
_FrozenJson = _JsonScalar | tuple["_FrozenJson", ...] | Mapping[str, "_FrozenJson"]


class RuntimeError(StoreError):
    """Base error for the parent-owned refinement runtime."""


class AuthenticationError(RuntimeError):
    """Draft capture was requested without a server-authenticated principal."""


class DraftCatalogError(RuntimeError):
    """The authoritative Draft catalog returned an invalid capture."""


class WorkerStopError(RuntimeError):
    """A split worker did not reach a clean stopped state."""


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Server-derived user identity; browser claims are not authority."""

    user_id: str
    authenticated: bool

    def __post_init__(self) -> None:
        if (
            self.authenticated is not True
            or not isinstance(self.user_id, str)
            or not self.user_id.strip()
        ):
            raise AuthenticationError(
                "an authenticated current-user identity is required"
            )


@dataclass(frozen=True)
class DraftCatalogRequest:
    split: str
    project_id: str
    principal: AuthenticatedPrincipal


@dataclass(frozen=True)
class AuthoritativeDraftSnapshot:
    """One deeply immutable, server-captured canonical Draft payload."""

    split: str
    project_id: str
    image_id: int
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
    inference_receipts: Sequence[str] = ()

    def __post_init__(self) -> None:
        frozen_regions = _freeze_json(list(self.regions))
        if not isinstance(frozen_regions, tuple) or not all(
            isinstance(region, Mapping) for region in frozen_regions
        ):
            raise DraftCatalogError("Draft regions must be a JSON array of objects")
        object.__setattr__(self, "regions", frozen_regions)
        object.__setattr__(self, "inference_receipts", tuple(self.inference_receipts))


@dataclass(frozen=True)
class DraftCatalogCapture:
    """A bounded current-user capture made in one authoritative server call."""

    split: str
    project_id: str
    current_user_id: str
    base_generation: int
    snapshots: Sequence[AuthoritativeDraftSnapshot]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))


class DraftCatalog(Protocol):
    """Authenticated vendor bridge for authoritative current-user Drafts.

    Implementations must derive ``principal`` from the authenticated server
    request and copy complete canonical result/meta content before returning.
    Mutable Draft IDs or browser task enumerations are not valid captures.
    """

    def capture_current_user_drafts(
        self, request: DraftCatalogRequest
    ) -> DraftCatalogCapture: ...


@dataclass(frozen=True)
class BatchStatusReceipt:
    """Deterministic ordinary-JSON receipt for enqueue and status responses."""

    batch_id: str
    payload_hash: str | None
    status: str
    split: str | None
    member_count: int
    base_generation: int | None
    generation: int | None
    error: str | None

    @classmethod
    def from_status_view(cls, view: BatchStatusView) -> "BatchStatusReceipt":
        return cls(
            batch_id=view.batch_id,
            payload_hash=view.payload_hash,
            status=view.status.value,
            split=view.split,
            member_count=view.member_count,
            base_generation=view.base_generation,
            generation=view.generation,
            error=view.error,
        )

    @classmethod
    def from_enqueue(cls, receipt: BatchEnqueueReceipt) -> "BatchStatusReceipt":
        return cls(
            batch_id=receipt.batch_id,
            payload_hash=receipt.payload_hash,
            status=receipt.status.value,
            split=receipt.split,
            member_count=receipt.member_count,
            base_generation=receipt.base_generation,
            generation=None,
            error=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_generation": self.base_generation,
            "batch_id": self.batch_id,
            "error": self.error,
            "generation": self.generation,
            "member_count": self.member_count,
            "payload_hash": self.payload_hash,
            "split": self.split,
            "status": self.status,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


class WorkerState(str, Enum):
    STOPPED = "stopped"
    RECOVERING = "recovering"
    BUSY = "busy"
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerHealth:
    split: str
    state: WorkerState
    thread_alive: bool
    processed_batches: int
    last_batch_id: str | None
    error: str | None

    @property
    def healthy(self) -> bool:
        return self.thread_alive and self.state in {
            WorkerState.IDLE,
            WorkerState.RUNNING,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "healthy": self.healthy,
            "last_batch_id": self.last_batch_id,
            "processed_batches": self.processed_batches,
            "split": self.split,
            "state": self.state.value,
            "thread_alive": self.thread_alive,
        }


@dataclass
class _SplitWorker:
    split: str
    store: WorkingDatasetStore
    poll_interval: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    wake: threading.Event = field(default_factory=threading.Event)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    state: WorkerState = WorkerState.STOPPED
    processed_batches: int = 0
    last_batch_id: str | None = None
    error: str | None = None

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_requested = threading.Event()
            self.wake = threading.Event()
            self.state = WorkerState.RECOVERING
            self.error = None
            self.thread = threading.Thread(
                target=self._run,
                name=f"coco-refinement-{self.split}-worker",
                daemon=True,
            )
            self.thread.start()
            self.wake.set()

    def signal(self) -> None:
        self.wake.set()

    def stop(self, timeout: float) -> None:
        with self.lock:
            thread = self.thread
            if thread is None or not thread.is_alive():
                if self.state is not WorkerState.FAILED:
                    self.state = WorkerState.STOPPED
                return
            self.state = WorkerState.STOPPING
            self.stop_requested.set()
            self.wake.set()
        thread.join(timeout)
        if thread.is_alive():
            with self.lock:
                self.state = WorkerState.FAILED
                self.error = "worker did not stop before timeout"
            raise WorkerStopError(f"{self.split} worker did not stop before timeout")

    def health(self) -> WorkerHealth:
        with self.lock:
            thread = self.thread
            return WorkerHealth(
                split=self.split,
                state=self.state,
                thread_alive=thread is not None and thread.is_alive(),
                processed_batches=self.processed_batches,
                last_batch_id=self.last_batch_id,
                error=self.error,
            )

    def _set_state(self, state: WorkerState) -> None:
        with self.lock:
            self.state = state
            if state in {WorkerState.IDLE, WorkerState.RUNNING, WorkerState.STOPPED}:
                self.error = None

    def _set_busy(self, exc: StoreBusyError) -> None:
        with self.lock:
            self.state = WorkerState.BUSY
            self.error = f"{type(exc).__name__}: {exc}"

    def _fail(self, exc: BaseException) -> None:
        with self.lock:
            self.state = WorkerState.FAILED
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        try:
            while not self.stop_requested.is_set():
                try:
                    self.store.recover()
                    break
                except StoreBusyError as exc:
                    self._set_busy(exc)
                    self.stop_requested.wait(self.poll_interval)
            if self.stop_requested.is_set():
                self._set_state(WorkerState.STOPPED)
                return
            self._set_state(WorkerState.IDLE)
            while not self.stop_requested.is_set():
                self.wake.wait(self.poll_interval)
                self.wake.clear()
                if self.stop_requested.is_set():
                    break
                while not self.stop_requested.is_set():
                    self._set_state(WorkerState.RUNNING)
                    try:
                        result = self.store.process_next_batch()
                    except StoreBusyError as exc:
                        # Shared status/read barriers are intentionally
                        # non-blocking.  Observe Busy and retry later without
                        # weakening the store's publication authority.
                        self._set_busy(exc)
                        break
                    if result is None:
                        self._set_state(WorkerState.IDLE)
                        break
                    with self.lock:
                        self.processed_batches += 1
                        self.last_batch_id = result.batch_id
            self._set_state(WorkerState.STOPPED)
        except BaseException as exc:  # fail closed at the thread boundary
            self._fail(exc)


class BatchCoordinator:
    """Own one independently startable background worker per working split."""

    def __init__(
        self,
        stores: Mapping[str, WorkingDatasetStore],
        *,
        poll_interval: float = 0.25,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not stores:
            raise ValueError("at least one split store is required")
        self._stores = dict(stores)
        if any(split not in {"train", "val"} for split in self._stores):
            raise ValueError("split stores must be named 'train' or 'val'")
        self._workers = {
            split: _SplitWorker(split, store, poll_interval)
            for split, store in self._stores.items()
        }

    def start(self, split: str | None = None) -> None:
        for worker in self._selected_workers(split):
            worker.start()

    def stop(self, split: str | None = None, *, timeout: float = 5.0) -> None:
        errors: list[WorkerStopError] = []
        for worker in self._selected_workers(split):
            try:
                worker.stop(timeout)
            except WorkerStopError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def enqueue_batch(self, request: BatchRequest) -> BatchEnqueueReceipt:
        worker = self._worker(request.split)
        receipt = worker.store.enqueue_batch(request)
        # The store has returned only after its queue append/fsync completed.
        worker.signal()
        return receipt

    def signal(self, split: str) -> None:
        self._worker(split).signal()

    def health(
        self, split: str | None = None
    ) -> WorkerHealth | dict[str, WorkerHealth]:
        if split is not None:
            return self._worker(split).health()
        return {name: self._workers[name].health() for name in sorted(self._workers)}

    def _selected_workers(self, split: str | None) -> list[_SplitWorker]:
        if split is not None:
            return [self._worker(split)]
        return [self._workers[name] for name in sorted(self._workers)]

    def _worker(self, split: str) -> _SplitWorker:
        try:
            return self._workers[split]
        except KeyError as exc:
            raise ValueError(f"unknown split: {split}") from exc


class RefinementRuntime:
    """Status-first bridge from authoritative Drafts to durable split batches."""

    def __init__(
        self,
        *,
        catalog: DraftCatalog,
        stores: Mapping[str, WorkingDatasetStore],
        project_ids: Mapping[str, str],
        coordinator: BatchCoordinator | None = None,
    ) -> None:
        self.catalog = catalog
        self.stores = dict(stores)
        self.project_ids = dict(project_ids)
        if set(self.stores) != set(self.project_ids):
            raise ValueError("stores and project_ids must cover the same splits")
        self.coordinator = coordinator or BatchCoordinator(self.stores)
        self._capture_locks = {split: threading.Lock() for split in self.stores}

    def start_workers(self, split: str | None = None) -> None:
        self.coordinator.start(split)

    def stop_workers(self, split: str | None = None, *, timeout: float = 5.0) -> None:
        self.coordinator.stop(split, timeout=timeout)

    def worker_health(
        self, split: str | None = None
    ) -> WorkerHealth | dict[str, WorkerHealth]:
        return self.coordinator.health(split)

    def batch_status(self, *, split: str, batch_id: str) -> BatchStatusReceipt:
        store = self._store(split)
        return BatchStatusReceipt.from_status_view(store.get_batch_status(batch_id))

    def capture_and_enqueue(
        self,
        *,
        split: str,
        batch_id: str,
        principal: AuthenticatedPrincipal,
    ) -> BatchStatusReceipt:
        """Capture and durably enqueue without waiting for background work.

        An existing batch ID is resolved before the catalog is consulted.  This
        is the lost-response retry path and prevents recapture from a newer live
        Draft revision.
        """

        if not isinstance(principal, AuthenticatedPrincipal):
            raise AuthenticationError("server-authenticated principal is required")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValidationError("batch id must be non-empty text")
        store = self._store(split)
        with self._capture_locks[split]:
            existing = store.get_batch_status(batch_id)
            if existing.status is not BatchStatus.NOT_FOUND:
                return BatchStatusReceipt.from_status_view(existing)

            project_id = self.project_ids[split]
            capture = self.catalog.capture_current_user_drafts(
                DraftCatalogRequest(
                    split=split,
                    project_id=project_id,
                    principal=principal,
                )
            )
            request = self._build_batch_request(
                store=store,
                batch_id=batch_id,
                split=split,
                project_id=project_id,
                principal=principal,
                capture=capture,
            )
            receipt = self.coordinator.enqueue_batch(request)
            return BatchStatusReceipt.from_enqueue(receipt)

    def _build_batch_request(
        self,
        *,
        store: WorkingDatasetStore,
        batch_id: str,
        split: str,
        project_id: str,
        principal: AuthenticatedPrincipal,
        capture: DraftCatalogCapture,
    ) -> BatchRequest:
        if not isinstance(capture, DraftCatalogCapture):
            raise DraftCatalogError("catalog returned an invalid capture type")
        if (
            capture.split != split
            or capture.project_id != project_id
            or capture.current_user_id != principal.user_id
        ):
            raise DraftCatalogError("catalog capture identity mismatch")
        if (
            isinstance(capture.base_generation, bool)
            or not isinstance(capture.base_generation, int)
            or capture.base_generation < 0
        ):
            raise DraftCatalogError("catalog base generation is invalid")
        if not capture.snapshots:
            raise DraftCatalogError("catalog capture contains no eligible Drafts")

        members: list[BatchMember] = []
        seen_tasks: set[str] = set()
        for snapshot in capture.snapshots:
            self._validate_snapshot(snapshot, split=split, project_id=project_id)
            if snapshot.task_id in seen_tasks:
                raise DraftCatalogError("catalog capture contains duplicate tasks")
            seen_tasks.add(snapshot.task_id)
            regions = _thaw_json(snapshot.regions)
            if not isinstance(regions, list):
                raise DraftCatalogError("Draft regions must thaw to a JSON array")
            if semantic_hash(regions) != snapshot.semantic_hash:
                raise DraftCatalogError(
                    "Draft semantic hash does not match its payload"
                )
            source_row_index = store.resolve_source_row_index(
                split=split,
                project_id=project_id,
                task_id=snapshot.task_id,
                image_id=snapshot.image_id,
            )
            draft_save = DraftSaveReceipt(
                project_id=project_id,
                task_id=snapshot.task_id,
                annotation_id=snapshot.annotation_id,
                draft_id=snapshot.draft_id,
                annotation_revision=snapshot.annotation_revision,
                draft_updated_at=snapshot.draft_updated_at,
                semantic_hash=snapshot.semantic_hash,
                result_hash=snapshot.result_hash,
                durable=True,
            )
            request = CommitRequest(
                commit_id=f"{batch_id}:member:{snapshot.task_id}",
                split=split,
                image_id=snapshot.image_id,
                project_id=project_id,
                task_id=snapshot.task_id,
                annotation_id=snapshot.annotation_id,
                draft_id=snapshot.draft_id,
                annotation_revision=snapshot.annotation_revision,
                draft_updated_at=snapshot.draft_updated_at,
                semantic_hash=snapshot.semantic_hash,
                result_hash=snapshot.result_hash,
                base_row_hash=snapshot.base_row_hash,
                observed_generation=snapshot.observed_generation,
                regions=regions,
                draft_save=draft_save,
                inference_receipts=tuple(snapshot.inference_receipts),
            )
            members.append(BatchMember(source_row_index, request))
        members.sort(key=lambda member: member.source_row_index)
        return BatchRequest(
            batch_id=batch_id,
            split=split,
            base_generation=capture.base_generation,
            members=tuple(members),
        )

    def _store(self, split: str) -> WorkingDatasetStore:
        try:
            return self.stores[split]
        except KeyError as exc:
            raise ValueError(f"unknown split: {split}") from exc

    @staticmethod
    def _validate_snapshot(
        snapshot: AuthoritativeDraftSnapshot, *, split: str, project_id: str
    ) -> None:
        if not isinstance(snapshot, AuthoritativeDraftSnapshot):
            raise DraftCatalogError("catalog snapshot has an invalid type")
        if snapshot.split != split or snapshot.project_id != project_id:
            raise DraftCatalogError("Draft snapshot split/project mismatch")
        if (
            isinstance(snapshot.image_id, bool)
            or not isinstance(snapshot.image_id, int)
            or snapshot.image_id < 0
        ):
            raise DraftCatalogError("Draft image identity is invalid")
        if (
            not isinstance(snapshot.annotation_revision, str)
            or not snapshot.annotation_revision.strip()
        ):
            raise DraftCatalogError("Draft annotation revision is invalid")
        if (
            isinstance(snapshot.observed_generation, bool)
            or not isinstance(snapshot.observed_generation, int)
            or snapshot.observed_generation < 0
        ):
            raise DraftCatalogError("Draft observed generation is invalid")
        for value, label in (
            (snapshot.task_id, "task id"),
            (snapshot.annotation_id, "annotation id"),
            (snapshot.draft_id, "draft id"),
            (snapshot.draft_updated_at, "draft updated_at"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise DraftCatalogError(f"Draft {label} is invalid")
        for value, label in (
            (snapshot.semantic_hash, "semantic hash"),
            (snapshot.result_hash, "result hash"),
            (snapshot.base_row_hash, "base row hash"),
        ):
            if not _is_sha256(value):
                raise DraftCatalogError(f"Draft {label} is invalid")


def _freeze_json(value: Any) -> _FrozenJson:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DraftCatalogError("Draft payload must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, _FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DraftCatalogError("Draft JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise DraftCatalogError(f"Draft payload is not JSON-safe: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "AuthenticatedPrincipal",
    "AuthoritativeDraftSnapshot",
    "BatchCoordinator",
    "BatchStatusReceipt",
    "DraftCatalog",
    "DraftCatalogCapture",
    "DraftCatalogError",
    "DraftCatalogRequest",
    "RefinementRuntime",
    "WorkerHealth",
    "WorkerState",
]
