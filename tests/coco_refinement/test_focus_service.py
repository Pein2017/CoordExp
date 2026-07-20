from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.commit_service import (
    CommitService,
    CommitServiceError,
    SafeCommitStatus,
)
from src.coco_refinement.focus_service import FocusQueueService, FocusQueueServiceError
from src.coco_refinement.models import NativeTaskIdentity
from src.coco_refinement.repository import (
    CompactTaskRecord,
    ProjectRecord,
    SaveDraftRequest,
    SqliteDraftRepository,
)
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService


ORIGIN = "http://127.0.0.1:9141"


def _digest(character: str) -> str:
    return character * 64


def _committed(image_id: int):
    return canonicalize_objects(
        [
            {
                "region_key": f"val:coco:{image_id}",
                "bbox_2d": [1, 2, 30, 40],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": image_id,
            }
        ],
        split="val",
    )


def _workspace(tmp_path: Path):
    image_root = tmp_path / "images"
    for split in ("train", "val"):
        (image_root / f"{split}2017").mkdir(parents=True, exist_ok=True)
    tasks = []
    for row, image_id in enumerate((16228, 14038)):
        locator = f"val2017/{image_id:012d}.jpg"
        path = image_root / locator
        path.write_bytes(f"image-{image_id}".encode())
        committed = _committed(image_id)
        tasks.append(
            CompactTaskRecord(
                project_id="coco:val",
                identity=NativeTaskIdentity(
                    split="val", image_id=image_id, source_row_index=row
                ),
                image_locator=locator,
                image_width=640,
                image_height=480,
                image_fingerprint=_digest(chr(ord("a") + row)),
                current_generation=0,
                base_row_hash=_digest(chr(ord("c") + row)),
                committed_result_hash=committed.result_hash,
            )
        )
    repository = SqliteDraftRepository(tmp_path / "state.sqlite3")
    repository.bootstrap_project(
        ProjectRecord(
            project_id="coco:val",
            split="val",
            source_fingerprint=_digest("f"),
            task_count=2,
        ),
        tasks,
    )
    first = tasks[0]
    committed = _committed(first.identity.image_id)
    draft = canonicalize_objects(
        [
            *committed.to_json_regions(),
            {
                "region_key": "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938",
                "bbox_2d": [50, 60, 100, 120],
                "category_name": "tie",
                "category_id": 32,
            },
        ],
        split="val",
    )
    repository.save_draft(
        SaveDraftRequest(
            project_id="coco:val",
            task_id=first.task_id,
            mutation_id="focus-draft",
            expected_revision=0,
            expected_generation=0,
            expected_base_row_hash=first.base_row_hash,
            committed=committed,
            draft=draft,
        )
    )
    return repository, image_root, tasks


class _CommitStub(CommitService):
    def __init__(self) -> None:
        self.enqueue_calls: list[dict[str, object]] = []
        self.status_value = "succeeded"
        self.generation = 1

    def enqueue(self, **kwargs):
        self.enqueue_calls.append(dict(kwargs))
        return SafeCommitStatus(
            batch_id=str(kwargs["batch_id"]),
            payload_hash="a" * 64,
            status="queued",
            split=str(kwargs["split"]),
            member_count=1,
            base_generation=0,
            generation=None,
        )

    def status(self, *, split: str, batch_id: str):
        return SafeCommitStatus(
            batch_id=batch_id,
            payload_hash="a" * 64,
            status=self.status_value,
            split=split,
            member_count=1,
            base_generation=0,
            generation=self.generation if self.status_value == "succeeded" else None,
        )


class _MissingCommitStub(_CommitStub):
    def status(self, *, split: str, batch_id: str):
        raise CommitServiceError("missing", code="coco_refinement.commit_not_found")


class _EmptyCaptureCommitStub(_CommitStub):
    def enqueue(self, **kwargs):
        self.enqueue_calls.append(dict(kwargs))
        raise CommitServiceError(
            "no pending Drafts", code="coco_refinement.no_pending_drafts"
        )


class _BlockingEnqueueCommitStub(_CommitStub):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def enqueue(self, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=2)
        return super().enqueue(**kwargs)


@dataclass(frozen=True)
class _Receipt:
    generation: int

    def to_artifact_dict(self):
        return {"generation": self.generation, "status": "published"}


class _Publisher:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def publish(self):
        self.calls.append("publish")
        if self.fail:
            raise RuntimeError("synthetic publish failure")
        return _Receipt(generation=1)


def _service(
    repository: SqliteDraftRepository,
    image_root: Path,
    commit: _CommitStub,
    publisher_factory,
) -> FocusQueueService:
    return FocusQueueService(
        repository=repository,
        commit_service=commit,
        project_ids={"train": "coco:train", "val": "coco:val"},
        image_roots={"train": image_root, "val": image_root},
        publisher_factory=publisher_factory,
        poll_interval=0.005,
    )


def _wait_for(service: FocusQueueService, status: str) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = service.status()
        if value["queue"]["publication"]["status"] == status:
            return value
        time.sleep(0.01)
    raise AssertionError(f"publication did not reach {status}: {service.status()}")


def test_exact_order_persists_and_focus_commit_publishes_then_releases(
    tmp_path: Path,
) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    commit = _CommitStub()
    publish_calls: list[str] = []
    service = _service(
        repository,
        image_root,
        commit,
        lambda _split: _Publisher(publish_calls),
    )

    created = service.create(
        [
            str(image_root / tasks[1].image_locator),
            str(image_root / tasks[0].image_locator),
        ]
    )

    assert [member["image_id"] for member in created["queue"]["members"]] == [
        14038,
        16228,
    ]
    assert created["queue"]["pending_draft_count"] == 1
    restarted = _service(
        SqliteDraftRepository(repository.path),
        image_root,
        commit,
        lambda _split: _Publisher(publish_calls),
    )
    assert [member["task_id"] for member in restarted.status()["queue"]["members"]] == [
        tasks[1].task_id,
        tasks[0].task_id,
    ]

    queued = restarted.enqueue(batch_id="focus-batch-1")
    assert queued["queue"]["batch"]["status"] == "queued"
    assert commit.enqueue_calls == [
        {
            "split": "val",
            "batch_id": "focus-batch-1",
            "task_ids": (tasks[1].task_id, tasks[0].task_id),
            "focus_queue_id": queued["queue"]["queue_id"],
        }
    ]
    published = _wait_for(restarted, "succeeded")
    assert published["queue"]["batch"] == {
        "batch_id": "focus-batch-1",
        "status": "succeeded",
        "generation": 1,
    }
    assert publish_calls == ["publish"]
    assert restarted.release()["active"] is False
    assert repository.count_pending_drafts(project_id="coco:val") == 1


def test_create_validation_is_atomic_for_duplicates_mixed_split_and_escape(
    tmp_path: Path,
) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    service = _service(
        repository, image_root, _CommitStub(), lambda _split: _Publisher([])
    )
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    first = str(image_root / tasks[0].image_locator)
    train = image_root / "train2017/000000000001.jpg"
    train.write_bytes(b"train")

    for paths in ([first, first], [first, str(train)], [first, str(outside)]):
        with pytest.raises(FocusQueueServiceError, match="validation failed"):
            service.create(paths)
        assert repository.get_focus_queue() is None


def test_failed_publication_retains_queue_and_retries_without_recapture(
    tmp_path: Path,
) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    commit = _CommitStub()
    publish_calls: list[str] = []
    attempts = {"value": 0}

    def factory(_split: str):
        attempts["value"] += 1
        return _Publisher(publish_calls, fail=attempts["value"] == 1)

    service = _service(repository, image_root, commit, factory)
    service.create([str(image_root / tasks[0].image_locator)])
    service.enqueue(batch_id="focus-fail-1")
    failed = _wait_for(service, "failed")
    assert failed["active"] is True
    assert len(commit.enqueue_calls) == 1

    service.retry_publication()
    succeeded = _wait_for(service, "succeeded")
    assert succeeded["queue"]["publication"]["receipt"]["generation"] == 1
    assert len(commit.enqueue_calls) == 1
    assert publish_calls == ["publish", "publish"]


def test_restart_closes_intent_before_enqueue_crash_window(tmp_path: Path) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    queue = repository.create_focus_queue(
        queue_id="focus-pre-enqueue-crash",
        project_id="coco:val",
        split="val",
        task_ids=(tasks[0].task_id,),
    )
    repository.bind_focus_batch(
        queue_id=queue.queue_id,
        expected_updated_at=queue.updated_at,
        batch_id="missing-after-restart",
    )

    service = _service(
        repository,
        image_root,
        _MissingCommitStub(),
        lambda _split: _Publisher([]),
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        queue = repository.get_focus_queue()
        if queue is not None and queue.commit_status == "failed":
            break
        time.sleep(0.01)
    assert queue is not None
    assert queue.commit_status == "failed"
    assert queue.publication_status == "idle"
    assert service.release()["active"] is False


def test_terminal_commit_failure_never_publishes_and_allows_retry(
    tmp_path: Path,
) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    commit = _CommitStub()
    commit.status_value = "failed"
    publish_calls: list[str] = []
    service = _service(
        repository,
        image_root,
        commit,
        lambda _split: _Publisher(publish_calls),
    )
    service.create([str(image_root / tasks[0].image_locator)])
    service.enqueue(batch_id="focus-commit-fails")

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        queue = repository.get_focus_queue()
        if queue is not None and queue.commit_status == "failed":
            break
        time.sleep(0.01)
    assert queue is not None
    assert queue.commit_status == "failed"
    assert queue.publication_status == "idle"
    assert publish_calls == []

    commit.status_value = "succeeded"
    service.enqueue(batch_id="focus-commit-retry")
    succeeded = _wait_for(service, "succeeded")
    assert succeeded["queue"]["batch"]["batch_id"] == "focus-commit-retry"
    assert publish_calls == ["publish"]


def test_capture_that_becomes_empty_clears_batch_identity(tmp_path: Path) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    commit = _EmptyCaptureCommitStub()
    service = _service(repository, image_root, commit, lambda _split: _Publisher([]))
    service.create([str(image_root / tasks[0].image_locator)])

    result = service.enqueue(batch_id="focus-becomes-empty")

    queue = repository.get_focus_queue()
    assert result["empty"] is True
    assert result["status"] == "empty"
    assert queue is not None
    assert queue.batch_id is None
    assert queue.commit_status == "idle"
    assert queue.publication_status == "idle"
    assert len(commit.enqueue_calls) == 1


def test_concurrent_status_does_not_misclassify_live_enqueue_handoff(
    tmp_path: Path,
) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    commit = _BlockingEnqueueCommitStub(entered, release)
    service = _service(repository, image_root, commit, lambda _split: _Publisher([]))
    service.create([str(image_root / tasks[0].image_locator)])
    outcome: list[object] = []

    worker = threading.Thread(
        target=lambda: outcome.append(service.enqueue(batch_id="focus-live-handoff"))
    )
    worker.start()
    assert entered.wait(timeout=2)

    observed = service.status()
    assert observed["queue"]["batch"]["status"] == "waiting"
    assert repository.get_focus_queue().commit_status == "waiting"

    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(outcome) == 1
    succeeded = _wait_for(service, "succeeded")
    assert succeeded["queue"]["batch"]["status"] == "succeeded"


def test_focus_http_create_status_and_release_use_csrf(tmp_path: Path) -> None:
    repository, image_root, tasks = _workspace(tmp_path)
    commit = _CommitStub()
    focus = _service(repository, image_root, commit, lambda _split: _Publisher([]))
    task_service = TaskService(
        repository=repository,
        stores={"train": object(), "val": object()},  # unused by Focus routes
        project_ids={"train": "coco:train", "val": "coco:val"},
        image_roots={"train": image_root, "val": image_root},
    )
    app = create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9141,
        commit_service=commit,
        focus_service=focus,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        headers = {"Origin": ORIGIN, "x-csrf-token": csrf}
        invalid = client.post(
            "/api/focus",
            json={"image_paths": [str(tmp_path / "missing.jpg")]},
            headers=headers,
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["details"]["errors"][0]["position"] == 0
        unindexed_path = image_root / "val2017/000000099999.jpg"
        unindexed_path.write_bytes(b"not-indexed")
        unindexed = client.post(
            "/api/focus",
            json={"image_paths": [str(unindexed_path)]},
            headers=headers,
        )
        assert unindexed.status_code == 422
        assert unindexed.json()["error"]["details"]["errors"] == [
            {
                "position": 0,
                "image_path": str(unindexed_path),
                "reason": "image is not present in the max_len12000 task index",
            }
        ]
        created = client.post(
            "/api/focus",
            json={"image_paths": [str(image_root / tasks[0].image_locator)]},
            headers=headers,
        )
        assert created.status_code == 201
        duplicate = client.post(
            "/api/focus",
            json={"image_paths": [str(image_root / tasks[0].image_locator)]},
            headers=headers,
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "coco_refinement.focus_active"
        assert (
            client.get("/api/focus").json()["queue"]["queue_id"]
            == created.json()["queue"]["queue_id"]
        )
        assert client.get("/api/focus").json()["queue"]["split"] == "val"
        released = client.delete("/api/focus", headers=headers)
        assert released.status_code == 200
        assert released.json()["released_queue_id"].startswith("focus-")
