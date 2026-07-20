"""Persistent small-image Focus Queue orchestration for the local editor."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from src.common.errors import RuntimeContractError
from src.coco_refinement.commit_service import CommitService, CommitServiceError
from src.coco_refinement.dataset_publisher import (
    DEFAULT_TRAINING_CONFIG,
    CommittedGenerationPublisher,
    CoordExpSwiftTokenBudgetValidator,
    DatasetPublicationReceipt,
)
from src.coco_refinement.repository import (
    FocusQueueRecord,
    RepositoryError,
    SqliteDraftRepository,
)


_ACTIVE_COMMIT = {"waiting", "queued", "running", "reconciling"}
_ACTIVE_PUBLICATION = {"waiting", "running"}


class FocusQueueServiceError(RuntimeContractError):
    """A Focus Queue command cannot be completed from durable authority."""


class _Publisher(Protocol):
    def publish(self) -> DatasetPublicationReceipt: ...


PublisherFactory = Callable[[str], _Publisher]


class FocusQueueService:
    """Own path mapping, scoped Commit, and background training publication."""

    def __init__(
        self,
        *,
        repository: SqliteDraftRepository,
        commit_service: CommitService,
        project_ids: Mapping[str, str],
        image_roots: Mapping[str, str | Path],
        publisher_factory: PublisherFactory,
        poll_interval: float = 0.25,
    ) -> None:
        if set(project_ids) != {"train", "val"} or set(image_roots) != {
            "train",
            "val",
        }:
            raise FocusQueueServiceError(
                "Focus Queue requires exact train and val bindings",
                code="coco_refinement.focus_bindings",
            )
        if not isinstance(repository, SqliteDraftRepository):
            raise FocusQueueServiceError(
                "Focus Queue repository has an invalid type",
                code="coco_refinement.focus_repository",
            )
        if not isinstance(commit_service, CommitService):
            raise FocusQueueServiceError(
                "Focus Queue Commit service has an invalid type",
                code="coco_refinement.focus_commit_service",
            )
        if not callable(publisher_factory):
            raise FocusQueueServiceError(
                "Focus Queue publisher factory is unavailable",
                code="coco_refinement.focus_publisher",
            )
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise FocusQueueServiceError(
                "Focus Queue poll interval must be positive",
                code="coco_refinement.focus_poll_interval",
            )
        self.repository = repository
        self.commit_service = commit_service
        self.project_ids = {
            split: str(project_ids[split]) for split in ("train", "val")
        }
        self.image_roots = {
            split: Path(image_roots[split]).resolve(strict=True)
            for split in ("train", "val")
        }
        self.publisher_factory = publisher_factory
        self.poll_interval = float(poll_interval)
        self._monitor_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._enqueue_handoff: tuple[str, str] | None = None
        self._ensure_monitor()

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        commit_service: CommitService,
        poll_interval: float = 0.25,
    ) -> "FocusQueueService":
        workspace = getattr(runtime, "workspace", None)
        inspections = getattr(runtime, "inspections", ())
        project_ids = getattr(runtime, "project_ids", None)
        runtime_root = getattr(workspace, "runtime_root", None)
        repository_root = getattr(runtime, "repository_root", None)
        if (
            workspace is None
            or runtime_root is None
            or repository_root is None
            or not isinstance(project_ids, Mapping)
        ):
            raise FocusQueueServiceError(
                "standalone runtime lacks Focus Queue bindings",
                code="coco_refinement.focus_runtime",
            )
        image_roots = {
            str(receipt.split): receipt.image_root for receipt in inspections
        }
        root = Path(repository_root).resolve(strict=True)
        mutable_root = Path(runtime_root).resolve(strict=True)

        def publisher_factory(split: str) -> _Publisher:
            return CommittedGenerationPublisher(
                repository_root=root,
                runtime_root=mutable_root,
                split=split,  # type: ignore[arg-type]
                token_budget_validator=CoordExpSwiftTokenBudgetValidator(
                    root / DEFAULT_TRAINING_CONFIG,
                    max_total_tokens=12000,
                ),
            )

        return cls(
            repository=workspace.repository,
            commit_service=commit_service,
            project_ids=project_ids,
            image_roots=image_roots,
            publisher_factory=publisher_factory,
            poll_interval=poll_interval,
        )

    def create(self, image_paths: Sequence[str]) -> dict[str, Any]:
        values = tuple(image_paths)
        if not values:
            raise FocusQueueServiceError(
                "at least one image path is required",
                code="coco_refinement.focus_paths",
            )
        locators: list[str] = []
        selected_split: str | None = None
        seen_paths: set[Path] = set()
        errors: list[dict[str, Any]] = []
        for position, raw in enumerate(values):
            try:
                path, split, locator = self._resolve_image_path(raw)
                if path in seen_paths:
                    raise ValueError("duplicate image path")
                if selected_split is not None and split != selected_split:
                    raise ValueError("all Focus Queue images must belong to one split")
                selected_split = split if selected_split is None else selected_split
                seen_paths.add(path)
                locators.append(locator)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(
                    {"position": position, "image_path": str(raw), "reason": str(exc)}
                )
        if errors or selected_split is None:
            raise FocusQueueServiceError(
                "Focus Queue image validation failed",
                code="coco_refinement.focus_paths",
                context={"errors": errors},
            )
        try:
            tasks = self.repository.resolve_tasks_by_locators(locators)
            if any(
                task.project_id != self.project_ids[selected_split]
                or task.identity.split != selected_split
                for task in tasks
            ):
                raise FocusQueueServiceError(
                    "Focus Queue image index differs from the requested split",
                    code="coco_refinement.focus_tasks",
                )
            record = self.repository.create_focus_queue(
                queue_id=f"focus-{uuid.uuid4()}",
                project_id=self.project_ids[selected_split],
                split=selected_split,
                task_ids=tuple(task.task_id for task in tasks),
            )
        except RepositoryError as exc:
            context = getattr(exc, "context", None)
            if exc.code == "coco_refinement.focus_locator_resolution" and isinstance(
                context, Mapping
            ):
                missing = set(context.get("missing", ()))
                ambiguous = set(context.get("ambiguous", ()))
                context = {
                    "errors": [
                        {
                            "position": position,
                            "image_path": str(values[position]),
                            "reason": (
                                "image is not present in the max_len12000 task index"
                                if locator in missing
                                else "image locator is ambiguous in the task index"
                            ),
                        }
                        for position, locator in enumerate(locators)
                        if locator in missing or locator in ambiguous
                    ]
                }
            raise FocusQueueServiceError(
                str(exc),
                code=exc.code,
                context=context,
                cause=exc,
            ) from exc
        return self._status_dict(record)

    def status(self) -> dict[str, Any]:
        record = self.repository.get_focus_queue()
        if record is None:
            return {"active": False, "queue": None}
        self._ensure_monitor()
        return self._status_dict(record)

    def release(self) -> dict[str, Any]:
        record = self._require_queue()
        try:
            self.repository.release_focus_queue(queue_id=record.queue_id)
        except RepositoryError as exc:
            raise FocusQueueServiceError(
                str(exc),
                code=exc.code,
                context=getattr(exc, "context", None),
                cause=exc,
            ) from exc
        return {"active": False, "released_queue_id": record.queue_id, "queue": None}

    def enqueue(self, *, batch_id: str) -> dict[str, Any]:
        record = self._require_queue()
        if record.batch_id == batch_id:
            self._ensure_monitor()
            return self._status_dict(record)
        if (
            record.commit_status == "succeeded"
            and record.publication_status == "failed"
        ):
            raise FocusQueueServiceError(
                "retry or release the failed Focus publication before another Commit",
                code="coco_refinement.focus_publication_retry_required",
            )
        pending = self.repository.count_pending_drafts(
            project_id=record.project_id, task_ids=record.task_ids
        )
        if pending == 0:
            return self._empty_status(record)
        self._begin_enqueue_handoff(record.queue_id, batch_id)
        try:
            try:
                record = self.repository.bind_focus_batch(
                    queue_id=record.queue_id,
                    expected_updated_at=record.updated_at,
                    batch_id=batch_id,
                )
            except RepositoryError as exc:
                raise FocusQueueServiceError(
                    str(exc),
                    code=exc.code,
                    context=getattr(exc, "context", None),
                    cause=exc,
                ) from exc
            try:
                status = self.commit_service.enqueue(
                    split=record.split,
                    batch_id=batch_id,
                    task_ids=record.task_ids,
                    focus_queue_id=record.queue_id,
                )
            except CommitServiceError as exc:
                if exc.code == "coco_refinement.no_pending_drafts":
                    try:
                        record = self.repository.clear_focus_batch_intent(
                            queue_id=record.queue_id,
                            expected_updated_at=record.updated_at,
                            batch_id=batch_id,
                        )
                    except RepositoryError as cleanup_exc:
                        self._record_enqueue_failure(record, cleanup_exc)
                        raise FocusQueueServiceError(
                            str(cleanup_exc),
                            code=cleanup_exc.code,
                            context=getattr(cleanup_exc, "context", None),
                            cause=cleanup_exc,
                        ) from cleanup_exc
                    return self._empty_status(record)
                self._record_enqueue_failure(record, exc)
                raise FocusQueueServiceError(
                    str(exc),
                    code=exc.code,
                    context=getattr(exc, "context", None),
                    cause=exc,
                ) from exc
            try:
                record = self.repository.update_focus_queue(
                    queue_id=record.queue_id,
                    expected_updated_at=record.updated_at,
                    commit_status=status.status,
                    publication_status="waiting",
                    error=None,
                )
            except RepositoryError as exc:
                self._record_enqueue_failure(record, exc)
                raise FocusQueueServiceError(
                    str(exc),
                    code=exc.code,
                    context=getattr(exc, "context", None),
                    cause=exc,
                ) from exc
            return self._status_dict(record)
        finally:
            self._end_enqueue_handoff(record.queue_id, batch_id)

    def _empty_status(self, record: FocusQueueRecord) -> dict[str, Any]:
        value = self._status_dict(record)
        value.update({"empty": True, "status": "empty"})
        return value

    def retry_publication(self) -> dict[str, Any]:
        record = self._require_queue()
        if (
            record.commit_status != "succeeded"
            or record.committed_generation is None
            or record.publication_status != "failed"
        ):
            raise FocusQueueServiceError(
                "Focus publication is not retryable",
                code="coco_refinement.focus_publication_not_retryable",
            )
        record = self.repository.update_focus_queue(
            queue_id=record.queue_id,
            expected_updated_at=record.updated_at,
            publication_status="waiting",
            publication_receipt=None,
            error=None,
        )
        self._ensure_monitor()
        return self._status_dict(record)

    def _resolve_image_path(self, raw: str) -> tuple[Path, str, str]:
        if not isinstance(raw, str) or not raw:
            raise ValueError("image path must be non-empty text")
        supplied = Path(raw).expanduser()
        if not supplied.is_absolute():
            raise ValueError("image path must be absolute")
        resolved = supplied.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("image path is not a file")
        matches: list[tuple[str, str]] = []
        for split, root in self.image_roots.items():
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            locator = PurePosixPath(*relative.parts).as_posix()
            if relative.parts and relative.parts[0] == f"{split}2017":
                matches.append((split, locator))
        if len(matches) != 1:
            raise ValueError("image path is outside its approved split image root")
        return resolved, matches[0][0], matches[0][1]

    def _require_queue(self) -> FocusQueueRecord:
        record = self.repository.get_focus_queue()
        if record is None:
            raise FocusQueueServiceError(
                "no Focus Queue is active", code="coco_refinement.focus_not_found"
            )
        return record

    def _status_dict(self, record: FocusQueueRecord) -> dict[str, Any]:
        pending = self.repository.count_pending_drafts(
            project_id=record.project_id,
            task_ids=tuple(member.task.task_id for member in record.members),
        )
        return {
            "active": True,
            "queue": {
                "queue_id": record.queue_id,
                "project_id": record.project_id,
                "split": record.split,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "pending_draft_count": pending,
                "members": [
                    {
                        "ordinal": member.position + 1,
                        "task_id": member.task.task_id,
                        "image_id": member.task.identity.image_id,
                        "source_row_index": member.task.identity.source_row_index,
                        "image_width": member.task.image_width,
                        "image_height": member.task.image_height,
                    }
                    for member in record.members
                ],
                "batch": (
                    None
                    if record.batch_id is None
                    else {
                        "batch_id": record.batch_id,
                        "status": record.commit_status,
                        "generation": record.committed_generation,
                    }
                ),
                "publication": {
                    "status": record.publication_status,
                    "generation": record.committed_generation,
                    "receipt": record.publication_receipt,
                    "error": record.error,
                },
            },
        }

    def _ensure_monitor(self) -> None:
        record = self.repository.get_focus_queue()
        if record is None or record.batch_id is None:
            return
        if (
            record.commit_status not in _ACTIVE_COMMIT
            and record.publication_status not in _ACTIVE_PUBLICATION
        ):
            return
        with self._monitor_lock:
            if self._enqueue_handoff == (record.queue_id, record.batch_id):
                return
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._monitor_thread = threading.Thread(
                target=self._monitor_entry,
                name="coco-refinement-focus-publisher",
                daemon=True,
            )
            self._monitor_thread.start()

    def _monitor_entry(self) -> None:
        """Close the handoff race between a terminal monitor and new work."""

        try:
            self._monitor()
        finally:
            current = threading.current_thread()
            with self._monitor_lock:
                if self._monitor_thread is current:
                    self._monitor_thread = None
            self._ensure_monitor()

    def _monitor(self) -> None:
        while True:
            record = self.repository.get_focus_queue()
            if record is None or record.batch_id is None:
                return
            with self._monitor_lock:
                local_handoff = self._enqueue_handoff == (
                    record.queue_id,
                    record.batch_id,
                )
            if local_handoff:
                time.sleep(self.poll_interval)
                continue
            if record.commit_status in _ACTIVE_COMMIT:
                try:
                    status = self.commit_service.status(
                        split=record.split, batch_id=record.batch_id
                    )
                    if status.status != record.commit_status or (
                        status.generation is not None
                        and status.generation != record.committed_generation
                    ):
                        terminal_failure = status.status == "failed"
                        record = self.repository.update_focus_queue(
                            queue_id=record.queue_id,
                            expected_updated_at=record.updated_at,
                            commit_status=status.status,
                            publication_status=(
                                "idle"
                                if terminal_failure
                                else record.publication_status
                            ),
                            committed_generation=status.generation,
                            error=(
                                "Focus Commit failed; retry the queue Commit or release it."
                                if terminal_failure
                                else None
                            ),
                        )
                except CommitServiceError as exc:
                    if exc.code == "coco_refinement.commit_not_found":
                        self._fail_missing_commit(record)
                        return
                    self._record_monitor_observation_error(record, exc)
                    time.sleep(self.poll_interval)
                    continue
                if record.commit_status in _ACTIVE_COMMIT:
                    time.sleep(self.poll_interval)
                    continue
            if record.commit_status == "failed":
                return
            if (
                record.commit_status != "succeeded"
                or record.committed_generation is None
            ):
                self._fail_monitor(
                    record, RuntimeError("Focus Commit status is invalid")
                )
                return
            if record.publication_status == "succeeded":
                return
            if record.publication_status == "failed":
                return
            try:
                if record.publication_status != "running":
                    record = self.repository.update_focus_queue(
                        queue_id=record.queue_id,
                        expected_updated_at=record.updated_at,
                        publication_status="running",
                        error=None,
                    )
                receipt = self.publisher_factory(record.split).publish()
                if receipt.generation != record.committed_generation:
                    raise FocusQueueServiceError(
                        "published generation differs from Focus Commit",
                        code="coco_refinement.focus_publication_generation",
                    )
                self.repository.update_focus_queue(
                    queue_id=record.queue_id,
                    expected_updated_at=record.updated_at,
                    publication_status="succeeded",
                    publication_receipt=receipt.to_artifact_dict(),
                    error=None,
                )
            except BaseException as exc:
                self._fail_monitor(record, exc)
            return

    def _begin_enqueue_handoff(self, queue_id: str, batch_id: str) -> None:
        with self._monitor_lock:
            if self._enqueue_handoff is not None:
                raise FocusQueueServiceError(
                    "another Focus Commit is entering the durable queue",
                    code="coco_refinement.focus_busy",
                )
            self._enqueue_handoff = (queue_id, batch_id)

    def _end_enqueue_handoff(self, queue_id: str, batch_id: str) -> None:
        with self._monitor_lock:
            if self._enqueue_handoff == (queue_id, batch_id):
                self._enqueue_handoff = None
        self._ensure_monitor()

    def _record_enqueue_failure(
        self, record: FocusQueueRecord, exc: BaseException
    ) -> None:
        try:
            latest = self.repository.get_focus_queue()
            if latest is None or latest.queue_id != record.queue_id:
                return
            self.repository.update_focus_queue(
                queue_id=latest.queue_id,
                expected_updated_at=latest.updated_at,
                commit_status="failed",
                publication_status="idle",
                error=f"{type(exc).__name__}: {exc}",
            )
        except BaseException:
            return

    def _fail_monitor(self, record: FocusQueueRecord, exc: BaseException) -> None:
        try:
            latest = self.repository.get_focus_queue()
            if latest is None or latest.queue_id != record.queue_id:
                return
            self.repository.update_focus_queue(
                queue_id=latest.queue_id,
                expected_updated_at=latest.updated_at,
                publication_status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except BaseException:
            return

    def _fail_missing_commit(self, record: FocusQueueRecord) -> None:
        """Close a crash window where intent persisted before durable enqueue."""

        try:
            latest = self.repository.get_focus_queue()
            if latest is None or latest.queue_id != record.queue_id:
                return
            self.repository.update_focus_queue(
                queue_id=latest.queue_id,
                expected_updated_at=latest.updated_at,
                commit_status="failed",
                publication_status="idle",
                error="Focus Commit was not durably enqueued; retry with a new batch ID.",
            )
        except BaseException:
            return

    def _record_monitor_observation_error(
        self, record: FocusQueueRecord, exc: BaseException
    ) -> None:
        """Keep unknown Commit ownership reserved while status is retried."""

        try:
            latest = self.repository.get_focus_queue()
            if latest is None or latest.queue_id != record.queue_id:
                return
            self.repository.update_focus_queue(
                queue_id=latest.queue_id,
                expected_updated_at=latest.updated_at,
                error=f"{type(exc).__name__}: {exc}",
            )
        except BaseException:
            return


__all__ = ["FocusQueueService", "FocusQueueServiceError", "PublisherFactory"]
