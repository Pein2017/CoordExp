"""Authoritative task, image, and full-Draft application service."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from src.common.errors import RuntimeContractError
from src.coco_refinement.bootstrap import resolve_indexed_image
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import CanonicalDraft, Split
from src.coco_refinement.repository import (
    CompactTaskRecord,
    DraftSaveApplied,
    DraftSaveConflict,
    DraftSaveResponse,
    SaveDraftRequest,
    SqliteDraftRepository,
    TaskDraftState,
)
from src.label_studio_coco_refinement.store import (
    DraftRestore,
    StoreError,
    WorkingDatasetStore,
)


MAX_TASK_PAGE = 200


class TaskServiceError(RuntimeContractError):
    """A safe task projection could not be produced."""


class CrossAuthorityConflict(TaskServiceError):
    """SQLite and the working store did not converge within the bounded read."""


@dataclass(frozen=True)
class TaskPage:
    split: Split
    cursor: int
    limit: int
    total: int
    next_cursor: int | None
    tasks: tuple[CompactTaskRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "cursor": self.cursor,
            "limit": self.limit,
            "total": self.total,
            "next_cursor": self.next_cursor,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "image_id": task.identity.image_id,
                    "source_row_index": task.identity.source_row_index,
                    "image_width": task.image_width,
                    "image_height": task.image_height,
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True)
class AuthoritativeTask:
    task: CompactTaskRecord
    state: TaskDraftState
    committed: CanonicalDraft
    objects: CanonicalDraft
    source: Literal["committed", "draft"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.task.identity.split,
            "task_id": self.task.task_id,
            "image_id": self.task.identity.image_id,
            "source_row_index": self.task.identity.source_row_index,
            "image_width": self.task.image_width,
            "image_height": self.task.image_height,
            "image_url": (
                f"/api/splits/{self.task.identity.split}/tasks/"
                f"{self.task.task_id}/image"
            ),
            "authority": self.source,
            "revision": self.state.revision,
            "epoch": self.state.epoch,
            "generation": self.state.current_generation,
            "base_row_hash": self.state.base_row_hash,
            "committed_result_hash": self.state.committed_result_hash,
            "semantic_hash": self.objects.semantic_hash,
            "result_hash": self.objects.result_hash,
            "objects": self.objects.to_json_regions(),
        }


@dataclass(frozen=True)
class DraftMutationOutcome:
    status: Literal["applied", "conflict"]
    mutation_id: str
    authoritative: AuthoritativeTask
    retired: bool | None = None
    conflict_reason: str | None = None
    expected: int | str | None = None
    actual: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = self.authoritative.to_dict()
        value.update(
            {
                "status": self.status,
                "mutation_id": self.mutation_id,
            }
        )
        if self.status == "applied":
            value["retired"] = self.retired
        else:
            value["conflict"] = {
                "reason": self.conflict_reason,
                "expected": self.expected,
                "actual": self.actual,
            }
        return value


@dataclass(frozen=True)
class ImagePayload:
    body: bytes
    media_type: str
    etag: str


class TaskService:
    """One vendor-free boundary over SQLite task state and working JSONL."""

    def __init__(
        self,
        *,
        repository: SqliteDraftRepository,
        stores: Mapping[str, WorkingDatasetStore],
        project_ids: Mapping[str, str],
        image_roots: Mapping[str, str | Path],
        write_ready: Callable[[], bool] = lambda: True,
        authority_attempts: int = 3,
    ) -> None:
        if set(stores) != {"train", "val"} or set(project_ids) != {
            "train",
            "val",
        } or set(image_roots) != {"train", "val"}:
            raise TaskServiceError(
                "task service requires exact train and val bindings",
                code="coco_refinement.task_service_splits",
            )
        if (
            isinstance(authority_attempts, bool)
            or not isinstance(authority_attempts, int)
            or not 1 <= authority_attempts <= 5
        ):
            raise TaskServiceError(
                "authority_attempts must be from 1 through 5",
                code="coco_refinement.authority_attempts",
            )
        self.repository = repository
        self.stores = {split: stores[split] for split in ("train", "val")}
        self.project_ids = {split: str(project_ids[split]) for split in ("train", "val")}
        self.image_roots = {
            split: Path(image_roots[split]).resolve(strict=True)
            for split in ("train", "val")
        }
        self.write_ready = write_ready
        self.authority_attempts = authority_attempts

    @classmethod
    def from_runtime(cls, runtime: object) -> TaskService:
        workspace = getattr(runtime, "workspace", None)
        inspections = getattr(runtime, "inspections", ())
        stores = getattr(runtime, "stores", None)
        project_ids = getattr(runtime, "project_ids", None)
        if workspace is None or not isinstance(stores, Mapping) or not isinstance(
            project_ids, Mapping
        ):
            raise TaskServiceError(
                "standalone runtime lacks task service bindings",
                code="coco_refinement.task_service_runtime",
            )
        image_roots = {
            str(receipt.split): receipt.image_root for receipt in inspections
        }
        return cls(
            repository=workspace.repository,
            stores=stores,
            project_ids=project_ids,
            image_roots=image_roots,
            write_ready=lambda: bool(getattr(runtime, "accepting_writes", False)),
        )

    def list_tasks(self, *, split: str, cursor: int, limit: int) -> TaskPage:
        selected = _split(split)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise TaskServiceError(
                "cursor must be a non-negative integer",
                code="coco_refinement.task_cursor",
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_TASK_PAGE
        ):
            raise TaskServiceError(
                f"limit must be from 1 through {MAX_TASK_PAGE}",
                code="coco_refinement.task_limit",
            )
        project, tasks = self.repository.list_compact_tasks(
            project_id=self.project_ids[selected], cursor=cursor, limit=limit
        )
        if project.split != selected:
            raise CrossAuthorityConflict(
                "project split differs from the server split binding",
                code="coco_refinement.project_split",
            )
        end = cursor + len(tasks)
        return TaskPage(
            split=selected,
            cursor=cursor,
            limit=limit,
            total=project.task_count,
            next_cursor=end if end < project.task_count else None,
            tasks=tasks,
        )

    def read_task(self, *, split: str, task_id: str) -> AuthoritativeTask:
        selected = _split(split)
        project_id = self.project_ids[selected]
        store = self.stores[selected]
        for _attempt in range(self.authority_attempts):
            first = self.repository.get_task_authority(project_id, task_id)
            task, state = first
            try:
                self._validate_task_binding(selected, task, state, store)
                restore = store.restore_draft(task.identity.image_id)
            except StoreError as exc:
                raise CrossAuthorityConflict(
                    "working task authority is unavailable",
                    code="coco_refinement.working_authority",
                    cause=exc,
                ) from exc
            second = self.repository.get_task_authority(project_id, task_id)
            if first != second:
                continue
            committed = _canonical_committed_restore(task, restore)
            if (
                restore.generation != state.current_generation
                or restore.row_hash != state.base_row_hash
                or committed.result_hash != state.committed_result_hash
            ):
                continue
            selected_objects = state.draft if state.draft is not None else committed
            return AuthoritativeTask(
                task=task,
                state=state,
                committed=committed,
                objects=selected_objects,
                source="draft" if state.draft is not None else "committed",
            )
        raise CrossAuthorityConflict(
            "SQLite task state and working generation did not converge",
            code="coco_refinement.cross_authority",
            context={"split": selected, "task_id": task_id},
        )

    def save_draft(
        self,
        *,
        split: str,
        task_id: str,
        mutation_id: str,
        expected_revision: int,
        expected_generation: int,
        expected_base_row_hash: str,
        objects: Sequence[Mapping[str, Any]],
    ) -> DraftMutationOutcome:
        selected = _split(split)
        if not self.write_ready():
            raise TaskServiceError(
                "runtime is not accepting Draft writes",
                code="coco_refinement.runtime_not_ready",
            )
        draft = canonicalize_objects(objects, split=selected)
        before = self.read_task(split=selected, task_id=task_id)
        request = SaveDraftRequest(
            project_id=self.project_ids[selected],
            task_id=task_id,
            mutation_id=mutation_id,
            expected_revision=expected_revision,
            expected_generation=expected_generation,
            expected_base_row_hash=expected_base_row_hash,
            committed=before.committed,
            draft=draft,
        )
        response = self.repository.save_draft(request)
        current = self.read_task(split=selected, task_id=task_id)
        _validate_save_transition(before, response, current)
        if isinstance(response, DraftSaveApplied):
            return DraftMutationOutcome(
                status="applied",
                mutation_id=response.mutation_id,
                authoritative=current,
                retired=response.retired,
            )
        return DraftMutationOutcome(
            status="conflict",
            mutation_id=response.mutation_id,
            authoritative=current,
            conflict_reason=response.reason,
            expected=response.expected,
            actual=_current_conflict_actual(response.reason, current.state),
        )

    def read_image(self, *, split: str, task_id: str) -> ImagePayload:
        selected = _split(split)
        project_id = self.project_ids[selected]
        first = self.repository.get_task_authority(project_id, task_id)
        task, state = first
        try:
            self._validate_task_binding(selected, task, state, self.stores[selected])
        except StoreError as exc:
            raise CrossAuthorityConflict(
                "working task authority is unavailable",
                code="coco_refinement.working_authority",
                cause=exc,
            ) from exc
        resolved = resolve_indexed_image(task, self.image_roots[selected])
        body = _read_image_without_symlinks(
            root=self.image_roots[selected],
            locator=task.image_locator,
            expected_path=resolved,
        )
        digest = hashlib.sha256(body).hexdigest()
        if digest != task.image_fingerprint:
            raise CrossAuthorityConflict(
                "indexed image bytes changed during the request",
                code="coco_refinement.image_hash",
            )
        try:
            with Image.open(io.BytesIO(body)) as image:
                size = image.size
                image_format = image.format
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise TaskServiceError(
                "indexed image could not be decoded",
                code="coco_refinement.image_decode",
                cause=exc,
            ) from exc
        if size != (task.image_width, task.image_height):
            raise CrossAuthorityConflict(
                "indexed image dimensions changed during the request",
                code="coco_refinement.image_dimensions",
            )
        if self.repository.get_task_authority(project_id, task_id)[0] != task:
            raise CrossAuthorityConflict(
                "task image binding changed during the request",
                code="coco_refinement.image_binding",
            )
        return ImagePayload(
            body=body,
            media_type=_image_media_type(image_format),
            etag=f'"sha256:{digest}"',
        )

    def _validate_task_binding(
        self,
        split: Split,
        task: CompactTaskRecord,
        state: TaskDraftState,
        store: WorkingDatasetStore,
    ) -> None:
        if (
            task.project_id != self.project_ids[split]
            or task.identity.split != split
            or state.identity != task.identity
            or state.project_id != task.project_id
            or task.current_generation != state.current_generation
            or task.base_row_hash != state.base_row_hash
            or task.committed_result_hash != state.committed_result_hash
        ):
            raise CrossAuthorityConflict(
                "task bindings disagree within SQLite",
                code="coco_refinement.task_binding",
            )
        row_index = store.resolve_source_row_index(
            split=split,
            project_id=task.project_id,
            task_id=task.task_id,
            image_id=task.identity.image_id,
        )
        if row_index != task.identity.source_row_index:
            raise CrossAuthorityConflict(
                "working task index differs from SQLite",
                code="coco_refinement.task_row",
            )


def _canonical_committed_restore(
    task: CompactTaskRecord, restore: DraftRestore
) -> CanonicalDraft:
    if (
        restore.split != task.identity.split
        or restore.image_id != task.identity.image_id
        or restore.row.get("image_id") != task.identity.image_id
    ):
        raise CrossAuthorityConflict(
            "working restore differs from the task identity",
            code="coco_refinement.restore_identity",
        )
    reverse_mapping = {object_id: key for key, object_id in restore.region_id_mapping.items()}
    if len(reverse_mapping) != len(restore.region_id_mapping):
        raise CrossAuthorityConflict(
            "working restore contains duplicate object identity mappings",
            code="coco_refinement.restore_mapping",
        )
    raw_objects = restore.row.get("objects")
    if not isinstance(raw_objects, list):
        raise CrossAuthorityConflict(
            "working restore objects are not an array",
            code="coco_refinement.restore_objects",
        )
    values: list[dict[str, Any]] = []
    for obj in raw_objects:
        if not isinstance(obj, Mapping):
            raise CrossAuthorityConflict(
                "working restore object is not an object",
                code="coco_refinement.restore_objects",
            )
        object_id = obj.get("coco_ann_id")
        region_key = reverse_mapping.get(object_id)
        if region_key is None:
            raise CrossAuthorityConflict(
                "working restore object lacks one stable region mapping",
                code="coco_refinement.restore_mapping",
            )
        value = {
            "region_key": region_key,
            "bbox_2d": obj.get("bbox_2d"),
            "category_name": obj.get("category_name", obj.get("desc")),
            "category_id": obj.get("category_id"),
            "coco_ann_id": object_id,
        }
        if obj.get("metadata") is not None:
            value["metadata"] = obj["metadata"]
        values.append(value)
    return canonicalize_objects(values, split=task.identity.split)


def _validate_save_transition(
    before: AuthoritativeTask,
    response: DraftSaveResponse,
    current: AuthoritativeTask,
) -> None:
    if current.task.task_id != before.task.task_id:
        raise CrossAuthorityConflict(
            "Draft response changed task identity",
            code="coco_refinement.save_identity",
        )
    if isinstance(response, DraftSaveApplied):
        if current.state.revision < response.state.revision:
            raise CrossAuthorityConflict(
                "applied Draft revision is absent from current authority",
                code="coco_refinement.save_projection",
            )
        return
    if not isinstance(response, DraftSaveConflict):
        raise TaskServiceError(
            "repository returned an unsupported Draft response",
            code="coco_refinement.save_response",
        )
    if current.state.revision < response.state.revision:
        raise CrossAuthorityConflict(
            "conflict authority is newer than the current projection",
            code="coco_refinement.save_projection",
        )


def _current_conflict_actual(reason: str, state: TaskDraftState) -> int | str:
    if reason == "revision":
        return state.revision
    if reason == "generation":
        return state.current_generation
    if reason == "base_row_hash":
        return state.base_row_hash
    raise TaskServiceError(
        "repository returned an unsupported conflict reason",
        code="coco_refinement.save_response",
    )


def _read_image_without_symlinks(
    *, root: Path, locator: str, expected_path: Path
) -> bytes:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TaskServiceError(
            "indexed image locator is invalid",
            code="coco_refinement.image_locator",
        )
    if expected_path != root.joinpath(*relative.parts).resolve(strict=True):
        raise CrossAuthorityConflict(
            "indexed image path differs from the manifest binding",
            code="coco_refinement.image_binding",
        )
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    current_fd = directory_fd
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            if current_fd != directory_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=current_fd,
        )
        try:
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise TaskServiceError(
                    "indexed image is not a regular file",
                    code="coco_refinement.image_type",
                )
            with os.fdopen(file_fd, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise TaskServiceError(
            "indexed image could not be opened safely",
            code="coco_refinement.image_open",
            cause=exc,
        ) from exc
    finally:
        if current_fd != directory_fd:
            os.close(current_fd)
        os.close(directory_fd)


def _image_media_type(image_format: str | None) -> str:
    mapping = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    media_type = mapping.get(image_format or "")
    if media_type is None:
        raise TaskServiceError(
            "indexed image format is unsupported",
            code="coco_refinement.image_format",
        )
    return media_type


def _split(value: str) -> Split:
    if value not in ("train", "val"):
        raise TaskServiceError(
            "split must be train or val",
            code="coco_refinement.split",
        )
    return value  # type: ignore[return-value]


__all__ = [
    "AuthoritativeTask",
    "CrossAuthorityConflict",
    "DraftMutationOutcome",
    "ImagePayload",
    "MAX_TASK_PAGE",
    "TaskPage",
    "TaskService",
    "TaskServiceError",
]
