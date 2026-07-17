from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import NativeTaskIdentity
from src.coco_refinement.repository import (
    CompactTaskRecord,
    DraftSaveApplied,
    DraftSaveConflict,
    MutationCollisionError,
    ProjectRecord,
    RepositoryCorruptionError,
    RepositoryInvariantError,
    SaveDraftRequest,
    SqliteDraftRepository,
)


LOCAL_KEY = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"


def _object(*, x1: int = 10) -> dict[str, object]:
    return {
        "region_key": LOCAL_KEY,
        "bbox_2d": [x1, 20, 300, 400],
        "category_name": "person",
        "category_id": 1,
    }


def _digest(character: str) -> str:
    return character * 64


def _task(
    image_id: int = 7,
    *,
    row: int = 0,
    generation: int = 3,
    committed_result_hash: str | None = None,
) -> CompactTaskRecord:
    baseline = canonicalize_objects(
        [
            {
                "region_key": f"train:coco:{image_id}",
                "bbox_2d": [1, 2, 3, 4],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": image_id,
            }
        ],
        split="train",
    )
    return CompactTaskRecord(
        project_id="coco:train",
        identity=NativeTaskIdentity(
            split="train", image_id=image_id, source_row_index=row
        ),
        image_locator=f"train2017/{image_id:012d}.jpg",
        image_width=640,
        image_height=480,
        image_fingerprint=_digest("c"),
        current_generation=generation,
        base_row_hash=_digest("b"),
        committed_result_hash=committed_result_hash or baseline.result_hash,
    )


def _repository(
    path: Path,
    *,
    tasks: tuple[CompactTaskRecord, ...] | None = None,
) -> SqliteDraftRepository:
    selected = tasks or (_task(),)
    repository = SqliteDraftRepository(path)
    repository.bootstrap_project(
        ProjectRecord(
            project_id="coco:train",
            split="train",
            source_fingerprint=_digest("a"),
            task_count=len(selected),
        ),
        selected,
    )
    return repository


def _request(
    draft: object,
    *,
    mutation_id: str,
    committed: object | None = None,
    expected_revision: int = 0,
    expected_generation: int = 3,
    expected_base_row_hash: str | None = None,
) -> SaveDraftRequest:
    return SaveDraftRequest(
        project_id="coco:train",
        task_id="train:7",
        mutation_id=mutation_id,
        expected_revision=expected_revision,
        expected_generation=expected_generation,
        expected_base_row_hash=expected_base_row_hash or _digest("b"),
        committed=(
            canonicalize_objects(
                [
                    {
                        "region_key": "train:coco:7",
                        "bbox_2d": [1, 2, 3, 4],
                        "category_name": "person",
                        "category_id": 1,
                        "coco_ann_id": 7,
                    }
                ],
                split="train",
            )
            if committed is None
            else committed
        ),  # type: ignore[arg-type]
        draft=draft,  # type: ignore[arg-type]
    )


def test_absent_draft_becomes_revision_one_and_uses_sqlite_safety_pragmas(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    draft = canonicalize_objects([_object()], split="train")

    response = repository.save_draft(
        _request(draft, mutation_id="save-first")
    )

    assert isinstance(response, DraftSaveApplied)
    assert response.state.revision == 1
    assert response.state.epoch == 0
    assert response.state.current_generation == 3
    assert response.state.base_row_hash == _digest("b")
    assert response.state.draft == draft
    assert response.retired is False
    assert repository.count_drafts(project_id="coco:train") == 1

    with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    assert repository.connection_settings().foreign_keys is True
    assert repository.connection_settings().busy_timeout_ms == 5_000


def test_stale_cas_returns_current_authoritative_state_without_merging(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    first = canonicalize_objects([_object()], split="train")
    changed = canonicalize_objects([_object(x1=11)], split="train")
    repository.save_draft(_request(first, mutation_id="save-1"))

    response = repository.save_draft(
        _request(changed, mutation_id="stale", expected_revision=0)
    )

    assert isinstance(response, DraftSaveConflict)
    assert response.reason == "revision"
    assert response.expected == 0
    assert response.actual == 1
    assert response.state.revision == 1
    assert response.state.draft == first
    assert repository.get_task_state("coco:train", "train:7").draft == first


def test_same_mutation_replays_authoritative_response_across_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    repository = _repository(path)
    request = _request(
        canonicalize_objects([_object()], split="train"),
        mutation_id="response-lost",
    )
    applied = repository.save_draft(request)

    restarted = SqliteDraftRepository(path)
    replayed = restarted.save_draft(request)

    assert replayed == applied
    assert restarted.get_task_state("coco:train", "train:7").revision == 1
    assert restarted.count_mutations() == 1


def test_replay_uses_caller_fingerprint_after_committed_generation_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    repository = _repository(path)
    request = _request(
        canonicalize_objects([_object()], split="train"),
        mutation_id="replay-after-generation",
    )
    applied = repository.save_draft(request)
    newer_committed = canonicalize_objects([], split="train")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE tasks
            SET current_generation = 4, base_row_hash = ?, committed_result_hash = ?
            WHERE task_id = 'train:7'
            """,
            (_digest("e"), newer_committed.result_hash),
        )

    replayed = repository.save_draft(
        replace(request, committed=newer_committed)
    )

    assert replayed == applied
    assert repository.count_mutations() == 1


def test_replay_rejects_response_with_wrong_mutation_identity(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    repository = _repository(path)
    request = _request(
        canonicalize_objects([_object()], split="train"),
        mutation_id="ledger-key",
    )
    repository.save_draft(request)
    with sqlite3.connect(path) as connection:
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM mutations WHERE mutation_id = ?",
                (request.mutation_id,),
            ).fetchone()[0]
        )
        response["mutation_id"] = "different-response-id"
        connection.execute(
            "UPDATE mutations SET response_json = ? WHERE mutation_id = ?",
            (json.dumps(response), request.mutation_id),
        )

    with pytest.raises(RepositoryCorruptionError) as exc_info:
        repository.save_draft(request)
    assert exc_info.value.code == "coco_refinement.mutation_response_identity"


def test_reusing_mutation_id_for_different_request_fails_closed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    first = _request(
        canonicalize_objects([_object()], split="train"), mutation_id="same-id"
    )
    repository.save_draft(first)
    collision = replace(
        first,
        draft=canonicalize_objects([_object(x1=12)], split="train"),
    )

    with pytest.raises(MutationCollisionError) as exc_info:
        repository.save_draft(collision)

    assert exc_info.value.code == "coco_refinement.mutation_collision"
    assert repository.get_task_state("coco:train", "train:7").revision == 1
    assert repository.count_mutations() == 1


def test_baseline_retirement_is_replayable_and_next_write_is_n_plus_one(
    tmp_path: Path,
) -> None:
    baseline = canonicalize_objects([], split="train")
    path = tmp_path / "state.sqlite3"
    repository = _repository(
        path,
        tasks=(_task(committed_result_hash=baseline.result_hash),),
    )
    changed = canonicalize_objects([_object()], split="train")
    first = repository.save_draft(
        _request(changed, mutation_id="draft", committed=baseline)
    )
    assert isinstance(first, DraftSaveApplied)
    assert first.state.revision == 1

    retire_request = _request(
        baseline,
        mutation_id="retire-response-lost",
        expected_revision=1,
        committed=baseline,
    )
    retired = repository.save_draft(retire_request)
    assert isinstance(retired, DraftSaveApplied)
    assert retired.retired is True
    assert retired.state.revision == 2
    assert retired.state.draft is None
    assert repository.count_drafts(project_id="coco:train") == 0

    restarted = SqliteDraftRepository(path)
    assert restarted.save_draft(retire_request) == retired
    next_response = restarted.save_draft(
        _request(
            changed,
            mutation_id="after-retire",
            expected_revision=2,
            committed=baseline,
        )
    )
    assert isinstance(next_response, DraftSaveApplied)
    assert next_response.state.revision == 3
    assert restarted.count_mutations() == 3


def test_semantically_equal_reordered_baseline_retires_sparse_draft(
    tmp_path: Path,
) -> None:
    values = [
        {
            "region_key": f"train:coco:{object_id}",
            "bbox_2d": [object_id, 20, 300, 400],
            "category_name": "person",
            "category_id": 1,
            "coco_ann_id": object_id,
        }
        for object_id in (7, 8)
    ]
    baseline = canonicalize_objects(values, split="train")
    reordered = canonicalize_objects(list(reversed(values)), split="train")
    assert reordered.semantic_hash == baseline.semantic_hash
    assert reordered.result_hash != baseline.result_hash
    repository = _repository(
        tmp_path / "state.sqlite3",
        tasks=(_task(committed_result_hash=baseline.result_hash),),
    )

    response = repository.save_draft(
        _request(
            reordered,
            mutation_id="semantic-retirement",
            committed=baseline,
        )
    )

    assert isinstance(response, DraftSaveApplied)
    assert response.retired is True
    assert response.state.draft is None
    assert repository.count_drafts(project_id="coco:train") == 0


def test_untrusted_committed_baseline_and_preallocated_new_ids_fail_before_ledger(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    wrong_committed = canonicalize_objects([], split="train")
    with pytest.raises(RepositoryInvariantError) as baseline_error:
        repository.save_draft(
            _request(
                canonicalize_objects([_object()], split="train"),
                mutation_id="wrong-baseline",
                committed=wrong_committed,
            )
        )
    assert baseline_error.value.code == "coco_refinement.committed_baseline"

    invented = canonicalize_objects(
        [{**_object(), "coco_ann_id": -99}], split="train"
    )
    with pytest.raises(RepositoryInvariantError) as identity_error:
        repository.save_draft(
            _request(invented, mutation_id="invented-negative")
        )
    assert identity_error.value.code == "coco_refinement.task_bound_identity"
    assert repository.get_task_state("coco:train", "train:7").revision == 0
    assert repository.count_mutations() == 0


def test_stale_save_returns_cas_conflict_before_baseline_and_identity_validation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    stale_foreign_object = canonicalize_objects(
        [
            {
                "region_key": "train:coco:999",
                "bbox_2d": [1, 2, 3, 4],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": 999,
            }
        ],
        split="train",
    )

    response = repository.save_draft(
        _request(
            stale_foreign_object,
            mutation_id="stale-foreign-identity",
            committed=canonicalize_objects([], split="train"),
            expected_revision=1,
        )
    )

    assert isinstance(response, DraftSaveConflict)
    assert response.reason == "revision"
    assert response.expected == 1
    assert response.actual == 0
    assert response.state.draft is None
    assert repository.count_mutations() == 1


@pytest.mark.parametrize(
    ("updates", "reason", "expected", "actual"),
    [
        ({"expected_generation": 2}, "generation", 2, 3),
        (
            {"expected_base_row_hash": _digest("d")},
            "base_row_hash",
            _digest("d"),
            _digest("b"),
        ),
    ],
)
def test_generation_or_base_hash_mismatch_fails_before_write(
    tmp_path: Path,
    updates: dict[str, object],
    reason: str,
    expected: object,
    actual: object,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    request = _request(
        canonicalize_objects([_object()], split="train"),
        mutation_id=f"mismatch-{reason}",
    )

    response = repository.save_draft(replace(request, **updates))

    assert isinstance(response, DraftSaveConflict)
    assert response.reason == reason
    assert response.expected == expected
    assert response.actual == actual
    assert response.state.revision == 0
    assert response.state.draft is None
    assert repository.count_drafts(project_id="coco:train") == 0


def test_complete_compact_task_index_has_no_baseline_object_copies(
    tmp_path: Path,
) -> None:
    tasks = tuple(_task(image_id=7 + index, row=index) for index in range(3))
    path = tmp_path / "state.sqlite3"
    repository = _repository(path, tasks=tasks)

    assert repository.count_tasks(project_id="coco:train") == 3
    assert repository.count_drafts(project_id="coco:train") == 0
    assert repository.list_task_identities(project_id="coco:train") == tuple(
        task.identity for task in tasks
    )

    with sqlite3.connect(path) as connection:
        task_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tasks)")
        }
        assert "objects_json" not in task_columns
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0

    restarted = SqliteDraftRepository(path)
    restarted.bootstrap_project(
        ProjectRecord(
            project_id="coco:train",
            split="train",
            source_fingerprint=_digest("a"),
            task_count=3,
        ),
        tasks,
    )
    assert restarted.count_tasks(project_id="coco:train") == 3


def test_bootstrap_rejects_source_row_index_gap(tmp_path: Path) -> None:
    repository = SqliteDraftRepository(tmp_path / "state.sqlite3")
    tasks = (_task(image_id=7, row=0), _task(image_id=8, row=2))

    with pytest.raises(RepositoryInvariantError) as exc_info:
        repository.bootstrap_project(
            ProjectRecord(
                project_id="coco:train",
                split="train",
                source_fingerprint=_digest("a"),
                task_count=2,
            ),
            tasks,
        )

    assert exc_info.value.code == "coco_refinement.source_row_sequence"
    assert repository.count_tasks(project_id="coco:train") == 0


@pytest.mark.parametrize(
    "locator",
    [
        "",
        ".",
        "/data/images/0001.jpg",
        "../images/0001.jpg",
        "train2017/../0001.jpg",
        "train2017/./0001.jpg",
        "train2017//0001.jpg",
        "train2017/",
        "train2017\\0001.jpg",
        "C:/images/0001.jpg",
    ],
)
def test_compact_task_rejects_unsafe_image_locator(locator: str) -> None:
    with pytest.raises(RepositoryInvariantError) as exc_info:
        replace(_task(), image_locator=locator)
    assert exc_info.value.code == "coco_refinement.image_locator"


def test_mutation_replay_rejects_corrupt_authoritative_response(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    repository = _repository(path)
    request = _request(
        canonicalize_objects([_object()], split="train"),
        mutation_id="corrupt-response",
    )
    repository.save_draft(request)
    with sqlite3.connect(path) as connection:
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM mutations WHERE mutation_id = ?",
                (request.mutation_id,),
            ).fetchone()[0]
        )
        response["state"]["revision"] = "1"
        connection.execute(
            "UPDATE mutations SET response_json = ? WHERE mutation_id = ?",
            (json.dumps(response), request.mutation_id),
        )

    with pytest.raises(RepositoryCorruptionError) as exc_info:
        SqliteDraftRepository(path).save_draft(request)
    assert exc_info.value.code == "coco_refinement.mutation_response_corrupt"


def test_unsupported_schema_version_is_rejected_before_creating_tables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE future_owner(value TEXT)")
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(RepositoryInvariantError) as exc_info:
        SqliteDraftRepository(path)
    assert exc_info.value.code == "coco_refinement.schema_version"

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {"future_owner"}


def test_unversioned_nonempty_database_is_rejected_without_schema_blessing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign-v0.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE projects(foo TEXT)")
        before_schema = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        before_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert before_version == 0

    with pytest.raises(RepositoryInvariantError) as exc_info:
        SqliteDraftRepository(path)
    assert exc_info.value.code == "coco_refinement.unversioned_database"

    with sqlite3.connect(path) as connection:
        after_schema = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        after_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert after_schema == before_schema
    assert after_version == before_version


def test_empty_unversioned_database_is_initialized_atomically(tmp_path: Path) -> None:
    path = tmp_path / "empty-v0.sqlite3"
    path.touch()

    SqliteDraftRepository(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {"projects", "tasks", "drafts", "mutations"}


def test_version_one_missing_tables_is_rejected_without_self_healing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete-v1.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE projects(foo TEXT)")
        connection.execute("PRAGMA user_version = 1")
        before_schema = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

    with pytest.raises(RepositoryInvariantError) as exc_info:
        SqliteDraftRepository(path)
    assert exc_info.value.code == "coco_refinement.schema_missing"

    with sqlite3.connect(path) as connection:
        after_schema = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert after_schema == before_schema


def test_version_one_missing_table_constraints_is_rejected_without_self_healing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unconstrained-v1.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE projects (
                project_id TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                task_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(project_id)
                    ON UPDATE RESTRICT ON DELETE RESTRICT,
                split TEXT NOT NULL,
                image_id INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                image_locator TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                image_fingerprint TEXT NOT NULL,
                revision INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                current_generation INTEGER NOT NULL,
                base_row_hash TEXT NOT NULL,
                committed_result_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX tasks_project_order
                ON tasks(project_id, source_row_index);
            CREATE TABLE drafts (
                task_id TEXT PRIMARY KEY REFERENCES tasks(task_id)
                    ON UPDATE RESTRICT ON DELETE RESTRICT,
                objects_json TEXT NOT NULL,
                semantic_hash TEXT NOT NULL,
                result_hash TEXT NOT NULL
            );
            CREATE TABLE mutations (
                mutation_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id)
                    ON UPDATE RESTRICT ON DELETE RESTRICT,
                request_fingerprint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX mutations_task ON mutations(task_id, created_at);
            PRAGMA user_version = 1;
            """
        )
        connection.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            (
                ("project-1", "invalid", "fingerprint-1", -1, "created-1"),
                ("project-2", "invalid", "fingerprint-2", -2, "created-2"),
            ),
        )
        task_values = (
            "invalid",
            0,
            -1,
            "image.jpg",
            0,
            -1,
            "image-fingerprint",
            -1,
            -1,
            -1,
            "base-row-hash",
            "committed-result-hash",
            "updated-at",
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ("task-1", "project-1", *task_values),
                ("task-2", "project-1", *task_values),
            ),
        )
        before_schema = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "ORDER BY type, name"
        ).fetchall()
        before_projects = connection.execute(
            "SELECT * FROM projects ORDER BY project_id"
        ).fetchall()
        before_tasks = connection.execute(
            "SELECT * FROM tasks ORDER BY task_id"
        ).fetchall()

    with pytest.raises(RepositoryInvariantError) as exc_info:
        SqliteDraftRepository(path)
    assert exc_info.value.code == "coco_refinement.schema_constraints"

    with sqlite3.connect(path) as connection:
        after_schema = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "ORDER BY type, name"
        ).fetchall()
        after_projects = connection.execute(
            "SELECT * FROM projects ORDER BY project_id"
        ).fetchall()
        after_tasks = connection.execute(
            "SELECT * FROM tasks ORDER BY task_id"
        ).fetchall()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert after_schema == before_schema
    assert after_projects == before_projects
    assert after_tasks == before_tasks


def test_empty_object_list_is_repository_data_not_a_commit_policy(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "state.sqlite3")
    empty = canonicalize_objects([], split="train")

    response = repository.save_draft(_request(empty, mutation_id="empty"))

    assert isinstance(response, DraftSaveApplied)
    assert response.state.revision == 1
    assert response.state.draft == empty
    assert response.state.draft.to_json_regions() == []
