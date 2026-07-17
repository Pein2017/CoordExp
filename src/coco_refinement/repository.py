"""Durable SQLite authority for compact tasks and sparse native Drafts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, TypeAlias

from src.common.errors import RuntimeContractError
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import CanonicalDraft, NativeTaskIdentity, Split


_SCHEMA_VERSION = 1
_DEFAULT_BUSY_TIMEOUT_MS = 5_000


class RepositoryError(RuntimeContractError):
    """Base class for typed standalone-editor persistence failures."""


class RepositoryInvariantError(RepositoryError):
    """Raised when durable state disagrees with a supplied immutable contract."""


class RepositoryCorruptionError(RepositoryError):
    """Raised when stored SQLite payloads cannot reproduce their recorded hashes."""


class ProjectNotFoundError(RepositoryError):
    """Raised when a project identity is absent."""


class TaskNotFoundError(RepositoryError):
    """Raised when a task identity is absent from its complete project index."""


class MutationCollisionError(RepositoryError):
    """Raised when a mutation ID is reused for a different request."""


@dataclass(frozen=True)
class ProjectRecord:
    """Immutable identity of one exact-source split workspace."""

    project_id: str
    split: Split
    source_fingerprint: str
    task_count: int

    def __post_init__(self) -> None:
        _nonempty(self.project_id, field="project_id")
        _split(self.split)
        _digest(self.source_fingerprint, field="source_fingerprint")
        _integer(self.task_count, field="task_count", minimum=0)


@dataclass(frozen=True)
class CompactTaskRecord:
    """Compact task row; committed object arrays deliberately live elsewhere."""

    project_id: str
    identity: NativeTaskIdentity
    image_locator: str
    image_width: int
    image_height: int
    image_fingerprint: str
    current_generation: int
    base_row_hash: str
    committed_result_hash: str

    def __post_init__(self) -> None:
        _nonempty(self.project_id, field="project_id")
        if not isinstance(self.identity, NativeTaskIdentity):
            raise RepositoryInvariantError(
                "identity must be a NativeTaskIdentity",
                code="coco_refinement.task_identity",
            )
        _relative_locator(self.image_locator)
        _integer(self.image_width, field="image_width", minimum=1)
        _integer(self.image_height, field="image_height", minimum=1)
        _digest(self.image_fingerprint, field="image_fingerprint")
        _integer(self.current_generation, field="current_generation", minimum=0)
        _digest(self.base_row_hash, field="base_row_hash")
        _digest(self.committed_result_hash, field="committed_result_hash")

    @property
    def task_id(self) -> str:
        return self.identity.task_key


@dataclass(frozen=True)
class TaskDraftState:
    """Current task authority plus an optional sparse native Draft."""

    project_id: str
    identity: NativeTaskIdentity
    revision: int
    epoch: int
    current_generation: int
    base_row_hash: str
    committed_result_hash: str
    updated_at: str
    draft: CanonicalDraft | None

    def __post_init__(self) -> None:
        _nonempty(self.project_id, field="project_id")
        if not isinstance(self.identity, NativeTaskIdentity):
            raise RepositoryInvariantError(
                "identity must be a NativeTaskIdentity",
                code="coco_refinement.task_identity",
            )
        _integer(self.revision, field="revision", minimum=0)
        _integer(self.epoch, field="epoch", minimum=0)
        _integer(self.current_generation, field="current_generation", minimum=0)
        _digest(self.base_row_hash, field="base_row_hash")
        _digest(self.committed_result_hash, field="committed_result_hash")
        _timestamp(self.updated_at, field="updated_at")
        if self.draft is None:
            return
        if not isinstance(self.draft, CanonicalDraft):
            raise RepositoryInvariantError(
                "draft must be a CanonicalDraft when present",
                code="coco_refinement.draft_type",
            )
        if self.draft.split != self.identity.split:
            raise RepositoryInvariantError(
                "Draft split differs from task identity",
                code="coco_refinement.draft_split",
            )
        _digest(self.draft.semantic_hash, field="semantic_hash")
        _digest(self.draft.result_hash, field="result_hash")
        reproduced = canonicalize_objects(
            self.draft.to_json_regions(), split=self.draft.split
        )
        if (
            reproduced.semantic_hash != self.draft.semantic_hash
            or reproduced.result_hash != self.draft.result_hash
            or reproduced.inference_receipts != self.draft.inference_receipts
        ):
            raise RepositoryInvariantError(
                "Draft hashes or receipts do not reproduce from canonical objects",
                code="coco_refinement.draft_integrity",
            )

    @property
    def task_id(self) -> str:
        return self.identity.task_key

    @property
    def has_draft(self) -> bool:
        return self.draft is not None


ConflictReason: TypeAlias = Literal[
    "revision", "generation", "base_row_hash"
]


@dataclass(frozen=True)
class SaveDraftRequest:
    """One idempotent full-Draft CAS request."""

    project_id: str
    task_id: str
    mutation_id: str
    expected_revision: int
    expected_generation: int
    expected_base_row_hash: str
    draft: CanonicalDraft

    def __post_init__(self) -> None:
        _nonempty(self.project_id, field="project_id")
        _nonempty(self.task_id, field="task_id")
        _nonempty(self.mutation_id, field="mutation_id")
        _integer(self.expected_revision, field="expected_revision", minimum=0)
        _integer(self.expected_generation, field="expected_generation", minimum=0)
        _digest(self.expected_base_row_hash, field="expected_base_row_hash")
        if not isinstance(self.draft, CanonicalDraft):
            raise RepositoryInvariantError(
                "draft must already be a canonical native Draft",
                code="coco_refinement.draft_type",
            )


@dataclass(frozen=True)
class DraftSaveApplied:
    """Durable successful write, including sparse-baseline retirement."""

    mutation_id: str
    state: TaskDraftState
    retired: bool

    def __post_init__(self) -> None:
        _nonempty(self.mutation_id, field="mutation_id")
        if not isinstance(self.state, TaskDraftState):
            raise RepositoryInvariantError(
                "state must be a TaskDraftState",
                code="coco_refinement.response_state",
            )
        if not isinstance(self.retired, bool):
            raise RepositoryInvariantError(
                "retired must be boolean",
                code="coco_refinement.response_retired",
            )
        if self.retired != (self.state.draft is None):
            raise RepositoryInvariantError(
                "retirement response disagrees with sparse Draft state",
                code="coco_refinement.response_retired",
            )


@dataclass(frozen=True)
class DraftSaveConflict:
    """Durable authoritative conflict response with no semantic mutation."""

    mutation_id: str
    reason: ConflictReason
    expected: int | str
    actual: int | str
    state: TaskDraftState

    def __post_init__(self) -> None:
        _nonempty(self.mutation_id, field="mutation_id")
        if not isinstance(self.state, TaskDraftState):
            raise RepositoryInvariantError(
                "state must be a TaskDraftState",
                code="coco_refinement.response_state",
            )
        if self.reason == "revision":
            _integer(self.expected, field="expected_revision", minimum=0)
            _integer(self.actual, field="actual_revision", minimum=0)
            authoritative: int | str = self.state.revision
        elif self.reason == "generation":
            _integer(self.expected, field="expected_generation", minimum=0)
            _integer(self.actual, field="actual_generation", minimum=0)
            authoritative = self.state.current_generation
        elif self.reason == "base_row_hash":
            _digest(self.expected, field="expected_base_row_hash")
            _digest(self.actual, field="actual_base_row_hash")
            authoritative = self.state.base_row_hash
        else:
            raise RepositoryInvariantError(
                "unknown Draft conflict reason",
                code="coco_refinement.conflict_reason",
                context={"reason": self.reason},
            )
        if self.actual != authoritative:
            raise RepositoryInvariantError(
                "conflict actual value differs from authoritative state",
                code="coco_refinement.conflict_state",
                context={"reason": self.reason},
            )


DraftSaveResponse: TypeAlias = DraftSaveApplied | DraftSaveConflict


@dataclass(frozen=True)
class SqliteConnectionSettings:
    journal_mode: str
    foreign_keys: bool
    busy_timeout_ms: int


class SqliteDraftRepository:
    """Single-file SQLite repository with per-save immediate transactions."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.path = Path(path)
        _integer(busy_timeout_ms, field="busy_timeout_ms", minimum=1)
        self.busy_timeout_ms = busy_timeout_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def bootstrap_project(
        self,
        project: ProjectRecord,
        tasks: Sequence[CompactTaskRecord],
    ) -> None:
        """Create or exactly verify one complete compact project index."""

        if not isinstance(project, ProjectRecord):
            raise RepositoryInvariantError(
                "project must be a ProjectRecord",
                code="coco_refinement.project_type",
            )
        task_values = tuple(tasks)
        if len(task_values) != project.task_count:
            raise RepositoryInvariantError(
                "project task_count does not equal supplied compact rows",
                code="coco_refinement.task_count",
                context={"expected": project.task_count, "actual": len(task_values)},
            )
        task_ids: set[str] = set()
        row_indexes: set[int] = set()
        image_ids: set[int] = set()
        for task in task_values:
            if not isinstance(task, CompactTaskRecord):
                raise RepositoryInvariantError(
                    "tasks must be CompactTaskRecord values",
                    code="coco_refinement.task_type",
                )
            if task.project_id != project.project_id or task.identity.split != project.split:
                raise RepositoryInvariantError(
                    "compact task belongs to another project or split",
                    code="coco_refinement.task_project",
                    context={"task_id": task.task_id},
                )
            if (
                task.task_id in task_ids
                or task.identity.source_row_index in row_indexes
                or task.identity.image_id in image_ids
            ):
                raise RepositoryInvariantError(
                    "compact task identities must be unique",
                    code="coco_refinement.task_duplicate",
                    context={"task_id": task.task_id},
                )
            task_ids.add(task.task_id)
            row_indexes.add(task.identity.source_row_index)
            image_ids.add(task.identity.image_id)
        if task_values and (
            min(row_indexes) != 0 or max(row_indexes) != project.task_count - 1
        ):
            raise RepositoryInvariantError(
                "source_row_index values must be exactly 0 through task_count - 1",
                code="coco_refinement.source_row_sequence",
                context={"task_count": project.task_count},
            )

        with self._transaction(immediate=True) as connection:
            existing_project = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project.project_id,),
            ).fetchone()
            if existing_project is None:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO projects(
                        project_id, split, source_fingerprint, task_count, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project.project_id,
                        project.split,
                        project.source_fingerprint,
                        project.task_count,
                        now,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO tasks(
                        task_id, project_id, split, image_id, source_row_index,
                        image_locator, image_width, image_height, image_fingerprint,
                        revision, epoch, current_generation, base_row_hash,
                        committed_result_hash, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
                    """,
                    (
                        (
                            task.task_id,
                            task.project_id,
                            task.identity.split,
                            task.identity.image_id,
                            task.identity.source_row_index,
                            task.image_locator,
                            task.image_width,
                            task.image_height,
                            task.image_fingerprint,
                            task.current_generation,
                            task.base_row_hash,
                            task.committed_result_hash,
                            now,
                        )
                        for task in task_values
                    ),
                )
                return

            actual_project = _project_from_row(existing_project)
            if actual_project != project:
                raise RepositoryInvariantError(
                    "existing project identity differs from bootstrap contract",
                    code="coco_refinement.project_drift",
                    context={"project_id": project.project_id},
                )
            rows = connection.execute(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY source_row_index",
                (project.project_id,),
            ).fetchall()
            if len(rows) != project.task_count:
                raise RepositoryInvariantError(
                    "existing compact task count differs from project contract",
                    code="coco_refinement.task_index_drift",
                    context={"expected": project.task_count, "actual": len(rows)},
                )
            expected_by_id = {task.task_id: task for task in task_values}
            if set(expected_by_id) != {str(row["task_id"]) for row in rows}:
                raise RepositoryInvariantError(
                    "existing compact task identities differ from bootstrap contract",
                    code="coco_refinement.task_index_drift",
                    context={"project_id": project.project_id},
                )
            for row in rows:
                expected = expected_by_id[str(row["task_id"])]
                if _compact_task_from_row(row) != expected:
                    raise RepositoryInvariantError(
                        "existing compact task row differs from bootstrap contract",
                        code="coco_refinement.task_index_drift",
                        context={"task_id": expected.task_id},
                    )

    def save_draft(self, request: SaveDraftRequest) -> DraftSaveResponse:
        """Apply or replay one full-Draft save under monotonic task CAS."""

        if not isinstance(request, SaveDraftRequest):
            raise RepositoryInvariantError(
                "request must be a SaveDraftRequest",
                code="coco_refinement.save_request_type",
            )
        request_fingerprint = _request_fingerprint(request)
        with self._transaction(immediate=True) as connection:
            mutation = connection.execute(
                "SELECT request_fingerprint, response_json FROM mutations WHERE mutation_id = ?",
                (request.mutation_id,),
            ).fetchone()
            if mutation is not None:
                if str(mutation["request_fingerprint"]) != request_fingerprint:
                    raise MutationCollisionError(
                        "mutation_id was already bound to a different request",
                        code="coco_refinement.mutation_collision",
                        context={"mutation_id": request.mutation_id},
                    )
                return _decode_response(str(mutation["response_json"]))

            task_row = connection.execute(
                "SELECT * FROM tasks WHERE project_id = ? AND task_id = ?",
                (request.project_id, request.task_id),
            ).fetchone()
            if task_row is None:
                project = connection.execute(
                    "SELECT 1 FROM projects WHERE project_id = ?",
                    (request.project_id,),
                ).fetchone()
                if project is None:
                    raise ProjectNotFoundError(
                        "project does not exist",
                        code="coco_refinement.project_not_found",
                        context={"project_id": request.project_id},
                    )
                raise TaskNotFoundError(
                    "task does not exist in the complete project index",
                    code="coco_refinement.task_not_found",
                    context={
                        "project_id": request.project_id,
                        "task_id": request.task_id,
                    },
                )
            if str(task_row["split"]) != request.draft.split:
                raise RepositoryInvariantError(
                    "Draft split differs from task authority",
                    code="coco_refinement.draft_split",
                    context={"task_id": request.task_id},
                )

            # CAS is intentionally checked before generation/base bindings.
            conflict: tuple[ConflictReason, int | str, int | str] | None = None
            if request.expected_revision != int(task_row["revision"]):
                conflict = (
                    "revision",
                    request.expected_revision,
                    int(task_row["revision"]),
                )
            elif request.expected_generation != int(task_row["current_generation"]):
                conflict = (
                    "generation",
                    request.expected_generation,
                    int(task_row["current_generation"]),
                )
            elif request.expected_base_row_hash != str(task_row["base_row_hash"]):
                conflict = (
                    "base_row_hash",
                    request.expected_base_row_hash,
                    str(task_row["base_row_hash"]),
                )
            if conflict is not None:
                state = self._state_from_rows(
                    task_row,
                    connection.execute(
                        "SELECT * FROM drafts WHERE task_id = ?",
                        (request.task_id,),
                    ).fetchone(),
                )
                response = DraftSaveConflict(
                    mutation_id=request.mutation_id,
                    reason=conflict[0],
                    expected=conflict[1],
                    actual=conflict[2],
                    state=state,
                )
                self._record_mutation(
                    connection,
                    request=request,
                    request_fingerprint=request_fingerprint,
                    response=response,
                )
                return response

            next_revision = int(task_row["revision"]) + 1
            now = _now()
            retired = request.draft.result_hash == str(
                task_row["committed_result_hash"]
            )
            if retired:
                connection.execute(
                    "DELETE FROM drafts WHERE task_id = ?",
                    (request.task_id,),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO drafts(task_id, objects_json, semantic_hash, result_hash)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        objects_json = excluded.objects_json,
                        semantic_hash = excluded.semantic_hash,
                        result_hash = excluded.result_hash
                    """,
                    (
                        request.task_id,
                        _json(request.draft.to_json_regions()),
                        request.draft.semantic_hash,
                        request.draft.result_hash,
                    ),
                )
            connection.execute(
                "UPDATE tasks SET revision = ?, updated_at = ? WHERE task_id = ?",
                (next_revision, now, request.task_id),
            )
            updated_task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (request.task_id,),
            ).fetchone()
            updated_draft = connection.execute(
                "SELECT * FROM drafts WHERE task_id = ?",
                (request.task_id,),
            ).fetchone()
            response = DraftSaveApplied(
                mutation_id=request.mutation_id,
                state=self._state_from_rows(updated_task, updated_draft),
                retired=retired,
            )
            self._record_mutation(
                connection,
                request=request,
                request_fingerprint=request_fingerprint,
                response=response,
            )
            return response

    def get_task_state(self, project_id: str, task_id: str) -> TaskDraftState:
        _nonempty(project_id, field="project_id")
        _nonempty(task_id, field="task_id")
        with closing(self._connect()) as connection:
            task_row = connection.execute(
                "SELECT * FROM tasks WHERE project_id = ? AND task_id = ?",
                (project_id, task_id),
            ).fetchone()
            if task_row is None:
                raise TaskNotFoundError(
                    "task does not exist in the complete project index",
                    code="coco_refinement.task_not_found",
                    context={"project_id": project_id, "task_id": task_id},
                )
            draft_row = connection.execute(
                "SELECT * FROM drafts WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return self._state_from_rows(task_row, draft_row)

    def count_tasks(self, *, project_id: str) -> int:
        with closing(self._connect()) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE project_id = ?", (project_id,)
                ).fetchone()[0]
            )

    def count_drafts(self, *, project_id: str | None = None) -> int:
        with closing(self._connect()) as connection:
            if project_id is None:
                row = connection.execute("SELECT COUNT(*) FROM drafts").fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM drafts AS d
                    JOIN tasks AS t ON t.task_id = d.task_id
                    WHERE t.project_id = ?
                    """,
                    (project_id,),
                ).fetchone()
            return int(row[0])

    def count_mutations(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM mutations").fetchone()[0])

    def list_task_identities(
        self, *, project_id: str
    ) -> tuple[NativeTaskIdentity, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT split, image_id, source_row_index
                FROM tasks WHERE project_id = ? ORDER BY source_row_index
                """,
                (project_id,),
            ).fetchall()
        return tuple(
            NativeTaskIdentity(
                split=str(row["split"]),  # type: ignore[arg-type]
                image_id=int(row["image_id"]),
                source_row_index=int(row["source_row_index"]),
            )
            for row in rows
        )

    def connection_settings(self) -> SqliteConnectionSettings:
        with closing(self._connect()) as connection:
            return SqliteConnectionSettings(
                journal_mode=str(connection.execute("PRAGMA journal_mode").fetchone()[0]),
                foreign_keys=bool(
                    connection.execute("PRAGMA foreign_keys").fetchone()[0]
                ),
                busy_timeout_ms=int(
                    connection.execute("PRAGMA busy_timeout").fetchone()[0]
                ),
            )

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, _SCHEMA_VERSION):
                raise RepositoryInvariantError(
                    "SQLite schema version is unsupported",
                    code="coco_refinement.schema_version",
                    context={"expected": _SCHEMA_VERSION, "actual": version},
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    split TEXT NOT NULL CHECK(split IN ('train', 'val')),
                    source_fingerprint TEXT NOT NULL,
                    task_count INTEGER NOT NULL CHECK(task_count >= 0),
                    created_at TEXT NOT NULL,
                    UNIQUE(split)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id)
                        ON UPDATE RESTRICT ON DELETE RESTRICT,
                    split TEXT NOT NULL CHECK(split IN ('train', 'val')),
                    image_id INTEGER NOT NULL CHECK(image_id > 0),
                    source_row_index INTEGER NOT NULL CHECK(source_row_index >= 0),
                    image_locator TEXT NOT NULL,
                    image_width INTEGER NOT NULL CHECK(image_width > 0),
                    image_height INTEGER NOT NULL CHECK(image_height > 0),
                    image_fingerprint TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    epoch INTEGER NOT NULL CHECK(epoch >= 0),
                    current_generation INTEGER NOT NULL CHECK(current_generation >= 0),
                    base_row_hash TEXT NOT NULL,
                    committed_result_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, source_row_index),
                    UNIQUE(project_id, image_id)
                );

                CREATE INDEX IF NOT EXISTS tasks_project_order
                    ON tasks(project_id, source_row_index);

                CREATE TABLE IF NOT EXISTS drafts (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id)
                        ON UPDATE RESTRICT ON DELETE RESTRICT,
                    objects_json TEXT NOT NULL,
                    semantic_hash TEXT NOT NULL,
                    result_hash TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mutations (
                    mutation_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id)
                        ON UPDATE RESTRICT ON DELETE RESTRICT,
                    request_fingerprint TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS mutations_task
                    ON mutations(task_id, created_at);
                """
            )
            if version == 0:
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _transaction(self, *, immediate: bool) -> _Transaction:
        return _Transaction(self._connect(), immediate=immediate)

    def _state_from_rows(
        self,
        task_row: sqlite3.Row,
        draft_row: sqlite3.Row | None,
    ) -> TaskDraftState:
        split = str(task_row["split"])
        draft: CanonicalDraft | None = None
        if draft_row is not None:
            try:
                values = json.loads(str(draft_row["objects_json"]))
                draft = canonicalize_objects(values, split=split)  # type: ignore[arg-type]
            except Exception as exc:
                if isinstance(exc, RepositoryCorruptionError):
                    raise
                raise RepositoryCorruptionError(
                    "stored Draft payload is not canonical",
                    code="coco_refinement.draft_corrupt",
                    context={"task_id": str(task_row["task_id"])},
                    cause=exc,
                ) from exc
            if (
                draft.semantic_hash != str(draft_row["semantic_hash"])
                or draft.result_hash != str(draft_row["result_hash"])
            ):
                raise RepositoryCorruptionError(
                    "stored Draft hashes do not reproduce from objects",
                    code="coco_refinement.draft_hash_corrupt",
                    context={"task_id": str(task_row["task_id"])},
                )
        try:
            return TaskDraftState(
                project_id=str(task_row["project_id"]),
                identity=NativeTaskIdentity(
                    split=split,  # type: ignore[arg-type]
                    image_id=int(task_row["image_id"]),
                    source_row_index=int(task_row["source_row_index"]),
                ),
                revision=int(task_row["revision"]),
                epoch=int(task_row["epoch"]),
                current_generation=int(task_row["current_generation"]),
                base_row_hash=str(task_row["base_row_hash"]),
                committed_result_hash=str(task_row["committed_result_hash"]),
                updated_at=str(task_row["updated_at"]),
                draft=draft,
            )
        except Exception as exc:
            if isinstance(exc, RepositoryCorruptionError):
                raise
            raise RepositoryCorruptionError(
                "stored task authority is invalid",
                code="coco_refinement.task_state_corrupt",
                context={"task_id": str(task_row["task_id"])},
                cause=exc,
            ) from exc

    def _record_mutation(
        self,
        connection: sqlite3.Connection,
        *,
        request: SaveDraftRequest,
        request_fingerprint: str,
        response: DraftSaveResponse,
    ) -> None:
        connection.execute(
            """
            INSERT INTO mutations(
                mutation_id, task_id, request_fingerprint, response_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                request.mutation_id,
                request.task_id,
                request_fingerprint,
                _json(_response_to_json(response)),
                _now(),
            ),
        )


class _Transaction:
    def __init__(self, connection: sqlite3.Connection, *, immediate: bool) -> None:
        self.connection = connection
        self.immediate = immediate

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE" if self.immediate else "BEGIN")
        return self.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()


def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
    return ProjectRecord(
        project_id=str(row["project_id"]),
        split=str(row["split"]),  # type: ignore[arg-type]
        source_fingerprint=str(row["source_fingerprint"]),
        task_count=int(row["task_count"]),
    )


def _compact_task_from_row(row: sqlite3.Row) -> CompactTaskRecord:
    return CompactTaskRecord(
        project_id=str(row["project_id"]),
        identity=NativeTaskIdentity(
            split=str(row["split"]),  # type: ignore[arg-type]
            image_id=int(row["image_id"]),
            source_row_index=int(row["source_row_index"]),
        ),
        image_locator=str(row["image_locator"]),
        image_width=int(row["image_width"]),
        image_height=int(row["image_height"]),
        image_fingerprint=str(row["image_fingerprint"]),
        current_generation=int(row["current_generation"]),
        base_row_hash=str(row["base_row_hash"]),
        committed_result_hash=str(row["committed_result_hash"]),
    )


def _request_fingerprint(request: SaveDraftRequest) -> str:
    return _sha256_json(
        {
            "project_id": request.project_id,
            "task_id": request.task_id,
            "expected_revision": request.expected_revision,
            "expected_generation": request.expected_generation,
            "expected_base_row_hash": request.expected_base_row_hash,
            "split": request.draft.split,
            "objects": request.draft.to_json_regions(),
            "semantic_hash": request.draft.semantic_hash,
            "result_hash": request.draft.result_hash,
        }
    )


def _response_to_json(response: DraftSaveResponse) -> dict[str, Any]:
    value: dict[str, Any] = {
        "mutation_id": response.mutation_id,
        "state": _state_to_json(response.state),
    }
    if isinstance(response, DraftSaveApplied):
        value.update({"kind": "applied", "retired": response.retired})
    else:
        value.update(
            {
                "kind": "conflict",
                "reason": response.reason,
                "expected": response.expected,
                "actual": response.actual,
            }
        )
    return value


def _state_to_json(state: TaskDraftState) -> dict[str, Any]:
    draft: dict[str, Any] | None = None
    if state.draft is not None:
        draft = {
            "split": state.draft.split,
            "objects": state.draft.to_json_regions(),
            "semantic_hash": state.draft.semantic_hash,
            "result_hash": state.draft.result_hash,
        }
    return {
        "project_id": state.project_id,
        "split": state.identity.split,
        "image_id": state.identity.image_id,
        "source_row_index": state.identity.source_row_index,
        "revision": state.revision,
        "epoch": state.epoch,
        "current_generation": state.current_generation,
        "base_row_hash": state.base_row_hash,
        "committed_result_hash": state.committed_result_hash,
        "updated_at": state.updated_at,
        "draft": draft,
    }


def _decode_response(encoded: str) -> DraftSaveResponse:
    try:
        value = json.loads(encoded)
        state_value = value["state"]
        identity = NativeTaskIdentity(
            split=state_value["split"],
            image_id=state_value["image_id"],
            source_row_index=state_value["source_row_index"],
        )
        draft_value = state_value["draft"]
        draft = None
        if draft_value is not None:
            draft = canonicalize_objects(
                draft_value["objects"], split=draft_value["split"]
            )
            if (
                draft.semantic_hash != draft_value["semantic_hash"]
                or draft.result_hash != draft_value["result_hash"]
            ):
                raise ValueError("response Draft hashes do not reproduce")
        state = TaskDraftState(
            project_id=state_value["project_id"],
            identity=identity,
            revision=state_value["revision"],
            epoch=state_value["epoch"],
            current_generation=state_value["current_generation"],
            base_row_hash=state_value["base_row_hash"],
            committed_result_hash=state_value["committed_result_hash"],
            updated_at=state_value["updated_at"],
            draft=draft,
        )
        if value["kind"] == "applied":
            return DraftSaveApplied(
                mutation_id=value["mutation_id"],
                state=state,
                retired=value["retired"],
            )
        if value["kind"] == "conflict" and value["reason"] in (
            "revision",
            "generation",
            "base_row_hash",
        ):
            return DraftSaveConflict(
                mutation_id=value["mutation_id"],
                reason=value["reason"],
                expected=value["expected"],
                actual=value["actual"],
                state=state,
            )
        raise ValueError("unknown response kind")
    except Exception as exc:
        raise RepositoryCorruptionError(
            "stored mutation response is invalid",
            code="coco_refinement.mutation_response_corrupt",
            cause=exc,
        ) from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepositoryInvariantError(
            f"{field} must be a non-empty string",
            code="coco_refinement.repository_field",
            context={"field": field},
        )
    return value


def _relative_locator(value: object) -> str:
    locator = value if isinstance(value, str) else ""
    parts = locator.split("/")
    if (
        not locator
        or "\\" in locator
        or PurePosixPath(locator).is_absolute()
        or PureWindowsPath(locator).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise RepositoryInvariantError(
            "image_locator must be a normalized relative POSIX path",
            code="coco_refinement.image_locator",
            context={"image_locator": locator},
        )
    return locator


def _timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RepositoryInvariantError(
            f"{field} must be a UTC ISO-8601 timestamp",
            code="coco_refinement.repository_timestamp",
            context={"field": field},
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RepositoryInvariantError(
            f"{field} must be a UTC ISO-8601 timestamp",
            code="coco_refinement.repository_timestamp",
            context={"field": field},
            cause=exc,
        ) from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise RepositoryInvariantError(
            f"{field} must be a UTC ISO-8601 timestamp",
            code="coco_refinement.repository_timestamp",
            context={"field": field},
        )
    return value


def _split(value: object) -> Split:
    if value not in ("train", "val"):
        raise RepositoryInvariantError(
            "split must be train or val",
            code="coco_refinement.repository_split",
            context={"split": value},
        )
    return value  # type: ignore[return-value]


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RepositoryInvariantError(
            f"{field} must be a lowercase SHA-256 digest",
            code="coco_refinement.repository_digest",
            context={"field": field},
        )
    return value


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RepositoryInvariantError(
            f"{field} must be an integer >= {minimum}",
            code="coco_refinement.repository_integer",
            context={"field": field, "value": value},
        )
    return value


__all__ = [
    "CompactTaskRecord",
    "DraftSaveApplied",
    "DraftSaveConflict",
    "DraftSaveResponse",
    "MutationCollisionError",
    "ProjectNotFoundError",
    "ProjectRecord",
    "RepositoryCorruptionError",
    "RepositoryError",
    "RepositoryInvariantError",
    "SaveDraftRequest",
    "SqliteConnectionSettings",
    "SqliteDraftRepository",
    "TaskDraftState",
    "TaskNotFoundError",
]
