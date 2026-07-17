from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.coco_refinement.adapters import (
    SqliteDraftCatalog,
    SqliteDraftVerifier,
    SqliteTerminalReconciler,
)
from src.coco_refinement.bootstrap import BootstrapSourceContract, bootstrap_workspace
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.commit_service import CommitService
from src.coco_refinement.http_security import CSRF_HEADER
from src.coco_refinement.repository import SaveDraftRequest, SqliteDraftRepository
from src.coco_refinement.service import create_service_app
from src.coco_refinement.task_service import TaskService
from src.label_studio_coco_refinement.runtime import (
    BatchStatusReceipt,
    DraftCatalogError,
    NoEligibleDraftsError,
    RefinementRuntime,
    WorkerHealth,
    WorkerState,
)
from src.label_studio_coco_refinement.store import (
    BatchRequest,
    BatchStatus,
    CommitConflictError,
    RecoveryError,
    StoreBusyError,
    ValidationError,
    canonical_json,
    sha256_file,
)


HOST = "127.0.0.1:9141"
ORIGIN = f"http://{HOST}"


class _ReceiptResolver:
    def resolve(self, _receipt_id: str) -> None:
        return None


class _PauseAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, boundary: str) -> None:
        if boundary == self.boundary:
            self.entered.set()
            assert self.release.wait(3.0)


class _PausedCoordinator:
    def __init__(self, stores: dict[str, Any]) -> None:
        self.stores = stores
        self.requests: list[BatchRequest] = []

    def enqueue_batch(self, request: BatchRequest):
        receipt = self.stores[request.split].enqueue_batch(request)
        self.requests.append(request)
        return receipt

    def health(self, split: str | None = None):
        def one(name: str) -> WorkerHealth:
            return WorkerHealth(
                split=name,
                state=WorkerState.IDLE,
                thread_alive=True,
                processed_batches=0,
                last_batch_id=None,
                error=None,
            )

        if split is not None:
            return one(split)
        return {name: one(name) for name in sorted(self.stores)}


class _StatusRuntime:
    def __init__(
        self,
        receipt: BatchStatusReceipt | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.receipt = receipt
        self.error = error

    def capture_and_enqueue(self, **_kwargs: object) -> BatchStatusReceipt:
        if self.error is not None:
            raise self.error
        assert self.receipt is not None
        return self.receipt

    def batch_status(self, **_kwargs: object) -> BatchStatusReceipt:
        if self.error is not None:
            raise self.error
        assert self.receipt is not None
        return self.receipt

    def worker_health(self, split: str | None = None) -> WorkerHealth:
        return WorkerHealth(
            split=split or "train",
            state=WorkerState.IDLE,
            thread_alive=True,
            processed_batches=0,
            last_batch_id=None,
            error=None,
        )


def _row(split: str, image_id: int) -> dict[str, Any]:
    object_id = image_id * 100
    return {
        "images": [f"../rescale_32_1024_bbox/images/{split}2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 300, 400],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": object_id,
            }
        ],
        "width": 32,
        "height": 24,
        "image_id": image_id,
        "file_name": f"images/{split}2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": split},
    }


def _committed(split: str, image_id: int):
    object_id = image_id * 100
    return canonicalize_objects(
        [
            {
                "region_key": f"{split}:coco:{object_id}",
                "bbox_2d": [10, 20, 300, 400],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": object_id,
            }
        ],
        split=split,  # type: ignore[arg-type]
    )


def _local(key: str, *, x1: int) -> dict[str, Any]:
    return {
        "region_key": key,
        "bbox_2d": [x1, 40, 500, 600],
        "category_name": "bicycle",
        "category_id": 2,
    }


@pytest.fixture
def commit_api(tmp_path: Path):
    source_root = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data/coco/rescale_32_1024_bbox/images"
    source_root.mkdir(parents=True)
    contracts: list[BootstrapSourceContract] = []
    source_before: dict[str, bytes] = {}
    source_paths: dict[str, Path] = {}
    image_before: dict[Path, bytes] = {}
    for split, image_ids in (("train", (1, 3)), ("val", (2,))):
        split_images = image_root / f"{split}2017"
        split_images.mkdir(parents=True)
        for image_id in image_ids:
            path = split_images / f"{image_id:012d}.jpg"
            Image.new("RGB", (32, 24), color="red").save(path, format="JPEG")
            image_before[path] = path.read_bytes()
        source = source_root / f"{split}.norm.jsonl"
        source.write_text(
            "".join(canonical_json(_row(split, value)) + "\n" for value in image_ids),
            encoding="utf-8",
        )
        source_before[split] = source.read_bytes()
        source_paths[split] = source
        contracts.append(
            BootstrapSourceContract(
                split=split,  # type: ignore[arg-type]
                source_path=source,
                image_root=image_root,
                expected_source_sha256=sha256_file(source),
                expected_row_count=len(image_ids),
            )
        )

    runtime_root = tmp_path / "runtime"
    repository = SqliteDraftRepository(runtime_root / "state.sqlite3")
    verifier = SqliteDraftVerifier(repository, current_user_id="local-operator")
    workspace = bootstrap_workspace(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        repository=repository,
        annotation_verifier=verifier,
        inference_receipt_resolver=_ReceiptResolver(),
    )
    stores = {split: workspace.splits[split].store for split in ("train", "val")}
    project_ids = {
        split: workspace.splits[split].project.project_id for split in ("train", "val")
    }
    for index, task in enumerate(workspace.splits["train"].tasks):
        committed = _committed("train", task.identity.image_id)
        draft = canonicalize_objects(
            [
                *committed.to_json_regions(),
                _local(
                    f"local:3f5dd17d-46ee-43dd-9fc0-51a5fd60393{8 + index}",
                    x1=30 + index,
                ),
            ],
            split="train",
        )
        repository.save_draft(
            SaveDraftRequest(
                project_id=task.project_id,
                task_id=task.task_id,
                mutation_id=f"seed-{task.task_id}",
                expected_revision=0,
                expected_generation=task.current_generation,
                expected_base_row_hash=task.base_row_hash,
                committed=committed,
                draft=draft,
            )
        )

    coordinator = _PausedCoordinator(stores)
    core = RefinementRuntime(
        catalog=SqliteDraftCatalog(repository, current_user_id="local-operator"),
        stores=stores,
        project_ids=project_ids,
        coordinator=coordinator,  # type: ignore[arg-type]
    )
    ready = {"value": True}
    commit_service = CommitService(
        repository=repository,
        runtime=core,
        stores=stores,
        project_ids=project_ids,
        accepting_writes=lambda: ready["value"],
    )
    task_service = TaskService(
        repository=repository,
        stores=stores,
        project_ids=project_ids,
        image_roots={"train": image_root, "val": image_root},
    )
    app = create_service_app(
        task_service,
        bind_host="127.0.0.1",
        port=9141,
        commit_service=commit_service,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        yield {
            "client": client,
            "csrf": csrf,
            "repository": repository,
            "stores": stores,
            "project_ids": project_ids,
            "coordinator": coordinator,
            "commit_service": commit_service,
            "task_service": task_service,
            "ready": ready,
            "source_before": source_before,
            "source_paths": source_paths,
            "image_before": image_before,
        }


def _post(client: TestClient, csrf: str, batch_id: str):
    return client.post(
        "/api/splits/train/commits",
        json={"batch_id": batch_id},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )


def test_paused_worker_captures_all_drafts_and_lost_retry_never_recaptures(
    commit_api,
) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    repository = commit_api["repository"]
    coordinator = commit_api["coordinator"]
    source_before = commit_api["source_before"]
    source_paths = commit_api["source_paths"]
    image_before = commit_api["image_before"]

    queued = _post(client, csrf, "batch-paused")
    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert queued.json()["member_count"] == 2
    request = coordinator.requests[0]
    assert request.current_user_id == "local-operator"
    assert [member.source_row_index for member in request.members] == [0, 1]
    frozen_hashes = [member.request.result_hash for member in request.members]

    commit_api["ready"]["value"] = False
    lost_response_retry = _post(client, csrf, "batch-paused")
    assert lost_response_retry.status_code == 202
    assert lost_response_retry.json()["status"] == "queued"
    assert len(coordinator.requests) == 1
    assert _post(client, csrf, "new-while-not-ready").status_code == 503
    commit_api["ready"]["value"] = True

    task = repository.get_task_state("coco-refinement:train", "train:1")
    assert task.draft is not None
    changed_values = task.draft.to_json_regions()
    changed_values[-1]["bbox_2d"] = [90, 100, 700, 800]
    repository.save_draft(
        SaveDraftRequest(
            project_id=task.project_id,
            task_id=task.task_id,
            mutation_id="later-while-queued",
            expected_revision=task.revision,
            expected_generation=task.current_generation,
            expected_base_row_hash=task.base_row_hash,
            committed=_committed("train", task.identity.image_id),
            draft=canonicalize_objects(changed_values, split="train"),
        )
    )

    replayed = _post(client, csrf, "batch-paused")
    assert replayed.status_code == 202
    assert len(coordinator.requests) == 1
    assert [member.request.result_hash for member in request.members] == frozen_hashes
    assert client.get("/api/splits/train/tasks/train:1/draft").status_code == 200
    assert (
        client.get("/api/splits/train/commits/batch-paused").json()["status"]
        == "queued"
    )
    assert _post(client, csrf, "second-while-active").status_code == 409
    for split, before in source_before.items():
        assert source_paths[split].read_bytes() == before
    assert all(path.read_bytes() == before for path, before in image_before.items())


def test_project_state_and_reads_remain_available_when_commit_writes_are_disabled(
    commit_api,
) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    commit_api["ready"]["value"] = False

    rejected = _post(client, csrf, "not-ready")
    state = client.get("/api/splits/train/state")
    missing = client.get("/api/splits/train/commits/not-ready")
    task = client.get("/api/splits/train/tasks/train:1/draft")

    assert rejected.status_code == 503
    assert state.status_code == 200
    assert state.json()["accepting_writes"] is False
    assert state.json()["task_count"] == 2
    assert state.json()["generation"] == 0
    assert state.json()["pending_draft_count"] == 2
    assert state.json()["worker"]["state"] == "idle"
    assert missing.status_code == 404
    assert task.status_code == 200


def test_real_paused_publication_keeps_http_editable_and_reconciles_newer_draft(
    commit_api,
) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    store = commit_api["stores"]["train"]
    repository = commit_api["repository"]
    queued = _post(client, csrf, "batch-live-pause")
    assert queued.status_code == 202
    pause = _PauseAt("batch_working_temp_fsynced")
    store._fault_injector = pause
    observer_entered = threading.Event()
    observer_release = threading.Event()
    reconciler = SqliteTerminalReconciler(repository, current_user_id="local-operator")
    outcomes: list[Any] = []
    errors: list[BaseException] = []

    def observe(request, result) -> None:
        observer_entered.set()
        assert observer_release.wait(3.0)
        reconciler.reconcile_batch(request, result)

    def process() -> None:
        try:
            outcomes.append(store.process_next_batch(terminal_observer=observe))
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert pause.entered.wait(1.0)
    initial = client.get("/api/splits/train/tasks/train:1/draft")
    running = client.get("/api/splits/train/commits/batch-live-pause")
    state = client.get("/api/splits/train/state")
    assert initial.status_code == 200
    assert running.status_code == 200
    assert running.json()["status"] == "running"
    assert state.status_code == 200
    changed = initial.json()["objects"]
    changed[-1]["bbox_2d"] = [101, 102, 701, 802]
    later = client.put(
        "/api/splits/train/tasks/train:1/draft",
        json={
            "mutation_id": "later-during-candidate-build",
            "expected_revision": initial.json()["revision"],
            "expected_generation": initial.json()["generation"],
            "expected_base_row_hash": initial.json()["base_row_hash"],
            "objects": changed,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert later.status_code == 200

    pause.release.set()
    assert observer_entered.wait(1.0)
    reconciling = client.get("/api/splits/train/commits/batch-live-pause")
    replay = _post(client, csrf, "batch-live-pause")
    rejected_new = _post(client, csrf, "batch-before-reconcile")
    task_page = client.get("/api/splits/train/tasks?cursor=0&limit=2")
    projected = client.get("/api/splits/train/tasks/train:1/draft")
    image = client.get("/api/splits/train/tasks/train:1/image")
    assert reconciling.status_code == 200
    assert reconciling.json()["status"] == "reconciling"
    with pytest.raises(RecoveryError, match="store requires recovery"):
        store.restore_draft(1)
    assert replay.status_code == 200
    assert replay.json()["status"] == "reconciling"
    assert rejected_new.status_code == 503
    assert task_page.status_code == 200
    assert [task["task_id"] for task in task_page.json()["tasks"]] == [
        "train:1",
        "train:3",
    ]
    assert projected.status_code == 200
    assert image.status_code == 200
    projected_objects = projected.json()["objects"]
    projected_objects[-1]["bbox_2d"] = [111, 112, 711, 812]
    saved_while_reconciling = client.put(
        "/api/splits/train/tasks/train:1/draft",
        json={
            "mutation_id": "later-during-terminal-observer",
            "expected_revision": projected.json()["revision"],
            "expected_generation": projected.json()["generation"],
            "expected_base_row_hash": projected.json()["base_row_hash"],
            "objects": projected_objects,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert saved_while_reconciling.status_code == 200
    assert saved_while_reconciling.json()["generation"] == 0
    assert (
        saved_while_reconciling.json()["base_row_hash"]
        == projected.json()["base_row_hash"]
    )
    assert "coco_ann_id" not in saved_while_reconciling.json()["objects"][-1]
    assert len(commit_api["coordinator"].requests) == 1

    observer_release.set()
    worker.join(3.0)
    assert not worker.is_alive()
    assert errors == []
    assert len(outcomes) == 1
    assert (
        client.get("/api/splits/train/commits/batch-live-pause").json()["status"]
        == "succeeded"
    )
    preserved = client.get("/api/splits/train/tasks/train:1/draft")
    assert preserved.status_code == 200
    assert preserved.json()["authority"] == "draft"
    assert preserved.json()["generation"] == 1
    assert preserved.json()["objects"][-1]["coco_ann_id"] < 0
    assert preserved.json()["objects"][-1]["bbox_2d"] == [111, 112, 711, 812]


def test_failed_terminal_observer_keeps_http_editable_until_restart_repair(
    commit_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    store = commit_api["stores"]["train"]
    repository = commit_api["repository"]
    queued = _post(client, csrf, "batch-failed-observer")
    assert queued.status_code == 202

    def reject_frozen_request(*_args: object, **_kwargs: object) -> None:
        raise ValidationError("deliberate terminal failure")

    monkeypatch.setattr(store, "_validate_batch_frozen_request", reject_frozen_request)
    observer_entered = threading.Event()
    observer_release = threading.Event()
    errors: list[BaseException] = []

    def failed_observer(_request: object, _result: object) -> None:
        observer_entered.set()
        assert observer_release.wait(3.0)
        raise RuntimeError("deliberate SQLite observer failure")

    def process() -> None:
        try:
            store.process_next_batch(terminal_observer=failed_observer)
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert observer_entered.wait(1.0)
    status = client.get("/api/splits/train/commits/batch-failed-observer")
    task_page = client.get("/api/splits/train/tasks?cursor=0&limit=2")
    task = client.get("/api/splits/train/tasks/train:1/draft")
    assert status.status_code == 200
    assert status.json()["status"] == "reconciling"
    assert task_page.status_code == 200
    assert task.status_code == 200

    changed = task.json()["objects"]
    changed[-1]["bbox_2d"] = [121, 122, 721, 822]
    saved = client.put(
        "/api/splits/train/tasks/train:1/draft",
        json={
            "mutation_id": "later-during-failed-observer",
            "expected_revision": task.json()["revision"],
            "expected_generation": task.json()["generation"],
            "expected_base_row_hash": task.json()["base_row_hash"],
            "objects": changed,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert saved.status_code == 200
    replay = _post(client, csrf, "batch-failed-observer")
    assert replay.status_code == 200
    assert replay.json()["status"] == "reconciling"
    assert _post(client, csrf, "new-after-failed-observer").status_code == 503

    observer_release.set()
    worker.join(3.0)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert (
        client.get("/api/splits/train/commits/batch-failed-observer").json()["status"]
        == "reconciling"
    )
    assert client.get("/api/splits/train/tasks/train:1/draft").status_code == 200

    reconciler = SqliteTerminalReconciler(repository, current_user_id="local-operator")
    store.recover(
        terminal_observer=lambda request, result: reconciler.reconcile_batch(
            request, result
        )
    )
    repaired = client.get("/api/splits/train/commits/batch-failed-observer")
    assert repaired.status_code == 200
    assert repaired.json()["status"] == "failed"
    preserved = client.get("/api/splits/train/tasks/train:1/draft")
    assert preserved.status_code == 200
    assert preserved.json()["objects"][-1]["bbox_2d"] == [121, 122, 721, 822]


def test_post_sqlite_pre_queue_terminal_keeps_http_editable_and_identity_stable(
    commit_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = commit_api["client"]
    csrf = commit_api["csrf"]
    store = commit_api["stores"]["train"]
    repository = commit_api["repository"]
    queued = _post(client, csrf, "batch-post-sqlite-pause")
    assert queued.status_code == 202

    reconciler = SqliteTerminalReconciler(
        repository, current_user_id="local-operator"
    )
    sqlite_projected = threading.Event()
    observer_release = threading.Event()
    outcomes: list[Any] = []
    errors: list[BaseException] = []

    def observe(request, result) -> None:
        reconciler.reconcile_batch(request, result)
        sqlite_projected.set()
        assert observer_release.wait(3.0)

    def process() -> None:
        try:
            outcomes.append(store.process_next_batch(terminal_observer=observe))
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert sqlite_projected.wait(1.0)

    status = client.get(
        "/api/splits/train/commits/batch-post-sqlite-pause"
    )
    task_page = client.get("/api/splits/train/tasks?cursor=0&limit=2")
    image = client.get("/api/splits/train/tasks/train:1/image")
    projected = client.get("/api/splits/train/tasks/train:1/draft")
    assert status.status_code == 200
    assert status.json()["status"] == "reconciling"
    assert task_page.status_code == 200
    assert image.status_code == 200
    assert projected.status_code == 200
    assert projected.json()["generation"] == 1
    assert projected.json()["authority"] == "committed"
    allocated_id = projected.json()["objects"][-1]["coco_ann_id"]
    assert allocated_id < 0

    journal_read_entered = threading.Event()
    journal_read_release = threading.Event()
    original_journal_read = store._read_journal_records

    def paused_journal_read(*, repair_torn_tail: bool):
        journal_read_entered.set()
        assert journal_read_release.wait(3.0)
        return original_journal_read(repair_torn_tail=repair_torn_tail)

    monkeypatch.setattr(store, "_read_journal_records", paused_journal_read)
    reload_errors: list[BaseException] = []
    concurrent_reads: list[Any] = []
    read_finished = threading.Event()

    def reload_index() -> None:
        try:
            store._reload_journal_index()
        except BaseException as exc:  # pragma: no cover - assertion receipt
            reload_errors.append(exc)

    def read_during_reload() -> None:
        try:
            concurrent_reads.append(
                client.get("/api/splits/train/tasks/train:1/draft")
            )
        finally:
            read_finished.set()

    reload_thread = threading.Thread(target=reload_index)
    reload_thread.start()
    assert journal_read_entered.wait(1.0)
    read_thread = threading.Thread(target=read_during_reload)
    read_thread.start()
    assert not read_finished.wait(0.05)
    journal_read_release.set()
    reload_thread.join(3.0)
    read_thread.join(3.0)
    assert not reload_thread.is_alive()
    assert not read_thread.is_alive()
    assert reload_errors == []
    assert len(concurrent_reads) == 1
    assert concurrent_reads[0].status_code == 200
    assert concurrent_reads[0].json()["objects"][-1]["coco_ann_id"] == allocated_id

    changed = projected.json()["objects"]
    changed[-1]["bbox_2d"] = [131, 132, 731, 832]
    saved = client.put(
        "/api/splits/train/tasks/train:1/draft",
        json={
            "mutation_id": "later-after-sqlite-projection",
            "expected_revision": projected.json()["revision"],
            "expected_generation": projected.json()["generation"],
            "expected_base_row_hash": projected.json()["base_row_hash"],
            "objects": changed,
        },
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert saved.status_code == 200
    assert saved.json()["generation"] == 1
    assert saved.json()["authority"] == "draft"
    assert saved.json()["objects"][-1]["coco_ann_id"] == allocated_id
    replay = _post(client, csrf, "batch-post-sqlite-pause")
    assert replay.status_code == 200
    assert replay.json()["status"] == "reconciling"
    assert _post(client, csrf, "new-post-sqlite-pause").status_code == 503

    observer_release.set()
    worker.join(3.0)
    assert not worker.is_alive()
    assert errors == []
    assert len(outcomes) == 1
    assert client.get(
        "/api/splits/train/commits/batch-post-sqlite-pause"
    ).json()["status"] == "succeeded"
    preserved = client.get("/api/splits/train/tasks/train:1/draft")
    assert preserved.status_code == 200
    assert preserved.json()["authority"] == "draft"
    assert preserved.json()["objects"][-1]["coco_ann_id"] == allocated_id
    assert preserved.json()["objects"][-1]["bbox_2d"] == [131, 132, 731, 832]


def test_project_state_retries_when_pending_draft_capture_changes(
    commit_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = commit_api["repository"]
    original = repository.capture_pending_draft_states
    first = original(project_id="coco-refinement:train", split="train")
    calls = 0

    def capture(*, project_id: str, split: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        if calls == 2:
            return type(first)(
                project_id=first.project_id,
                split=first.split,
                current_generation=first.current_generation,
                states=first.states[:1],
            )
        return original(project_id=project_id, split=split)

    monkeypatch.setattr(repository, "capture_pending_draft_states", capture)

    state = commit_api["commit_service"].project_state(split="train")

    assert calls == 4
    assert state.pending_draft_count == 2


def test_empty_durable_draft_is_rejected_without_enqueuing(commit_api) -> None:
    repository = commit_api["repository"]
    task = repository.get_task_state("coco-refinement:train", "train:1")
    repository.save_draft(
        SaveDraftRequest(
            project_id=task.project_id,
            task_id=task.task_id,
            mutation_id="empty-draft",
            expected_revision=task.revision,
            expected_generation=task.current_generation,
            expected_base_row_hash=task.base_row_hash,
            committed=_committed("train", task.identity.image_id),
            draft=canonicalize_objects([], split="train"),
        )
    )

    response = _post(commit_api["client"], commit_api["csrf"], "batch-empty")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "coco_refinement.commit_invalid_draft"
    assert repository.count_drafts(project_id="coco-refinement:train") == 2
    assert (
        commit_api["stores"]["train"].get_batch_status("batch-empty").status
        is BatchStatus.NOT_FOUND
    )
    assert commit_api["coordinator"].requests == []


@pytest.mark.parametrize("status", ["running", "reconciling", "succeeded", "failed"])
def test_status_exposes_exact_state_without_leaking_durable_error(
    commit_api, status: str
) -> None:
    raw_error = "failed opening /tmp/secret/model.json" if status == "failed" else None
    receipt = BatchStatusReceipt(
        batch_id=f"batch-{status}",
        payload_hash=hashlib.sha256(status.encode()).hexdigest(),
        status=status,
        split="train",
        member_count=2,
        base_generation=0,
        generation=1 if status in {"succeeded", "failed"} else None,
        error=raw_error,
    )
    service = CommitService(
        repository=commit_api["repository"],
        runtime=_StatusRuntime(receipt=receipt),
        stores=commit_api["stores"],
        project_ids=commit_api["project_ids"],
        accepting_writes=lambda: True,
    )
    app = create_service_app(
        commit_api["task_service"],
        bind_host="127.0.0.1",
        port=9141,
        commit_service=service,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        client.get("/api/session")
        response = client.get(f"/api/splits/train/commits/batch-{status}")

    assert response.status_code == 200
    assert response.json()["status"] == status
    assert "/tmp/secret" not in response.text
    if status == "failed":
        assert response.json()["detail"]["code"] == "coco_refinement.commit_failed"


def test_status_rejects_receipt_for_another_batch(commit_api) -> None:
    receipt = BatchStatusReceipt(
        batch_id="other-batch",
        payload_hash=hashlib.sha256(b"other").hexdigest(),
        status="running",
        split="train",
        member_count=1,
        base_generation=0,
        generation=None,
        error=None,
    )
    service = CommitService(
        repository=commit_api["repository"],
        runtime=_StatusRuntime(receipt=receipt),
        stores=commit_api["stores"],
        project_ids=commit_api["project_ids"],
        accepting_writes=lambda: True,
    )
    app = create_service_app(
        commit_api["task_service"],
        bind_host="127.0.0.1",
        port=9141,
        commit_service=service,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        client.get("/api/session")
        response = client.get("/api/splits/train/commits/requested-batch")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "coco_refinement.commit_status_invalid"


@pytest.mark.parametrize(
    "receipt",
    [
        object(),
        BatchStatusReceipt(
            batch_id="batch-malformed",
            payload_hash="/tmp/private/hash",
            status="running",
            split="train",
            member_count=1,
            base_generation=0,
            generation=None,
            error=None,
        ),
        BatchStatusReceipt(
            batch_id="batch-malformed",
            payload_hash=hashlib.sha256(b"malformed").hexdigest(),
            status="succeeded",
            split="train",
            member_count=-7,
            base_generation=False,
            generation=False,
            error="/tmp/private/error",
        ),
    ],
)
def test_status_rejects_malformed_runtime_receipt_safely(
    commit_api, receipt: object
) -> None:
    service = CommitService(
        repository=commit_api["repository"],
        runtime=_StatusRuntime(receipt=receipt),  # type: ignore[arg-type]
        stores=commit_api["stores"],
        project_ids=commit_api["project_ids"],
        accepting_writes=lambda: True,
    )
    app = create_service_app(
        commit_api["task_service"],
        bind_host="127.0.0.1",
        port=9141,
        commit_service=service,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        client.get("/api/session")
        response = client.get("/api/splits/train/commits/batch-malformed")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "coco_refinement.commit_status_invalid"
    assert "/tmp/private" not in response.text


def test_status_rejects_batch_id_that_enqueue_cannot_create(commit_api) -> None:
    response = commit_api["client"].get("/api/splits/train/commits/invalid%20batch")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "coco_refinement.request_invalid"


def test_project_state_rejects_malformed_worker_health_safely(
    commit_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        commit_api["commit_service"].runtime,
        "worker_health",
        lambda _split: {
            "error": "/tmp/private/worker.log",
            "healthy": "yes",
            "last_batch_id": "/tmp/private/batch",
            "processed_batches": -1,
            "split": "val",
            "state": "nonsense",
            "thread_alive": "true",
        },
    )

    response = commit_api["client"].get("/api/splits/train/state")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "coco_refinement.worker_health_invalid"
    assert "/tmp/private" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            NoEligibleDraftsError("empty"),
            409,
            "coco_refinement.no_pending_drafts",
        ),
        (
            DraftCatalogError("corrupt /tmp/secret"),
            503,
            "coco_refinement.commit_unavailable",
        ),
        (StoreBusyError("busy /tmp/secret"), 409, "coco_refinement.commit_busy"),
        (
            CommitConflictError("conflict /tmp/secret"),
            409,
            "coco_refinement.commit_conflict",
        ),
        (RecoveryError("repair /tmp/secret"), 503, "coco_refinement.commit_recovery"),
    ],
)
def test_enqueue_error_mapping_is_stable_and_safe(
    commit_api, error: BaseException, expected_status: int, expected_code: str
) -> None:
    service = CommitService(
        repository=commit_api["repository"],
        runtime=_StatusRuntime(error=error),
        stores=commit_api["stores"],
        project_ids=commit_api["project_ids"],
        accepting_writes=lambda: True,
    )
    app = create_service_app(
        commit_api["task_service"],
        bind_host="127.0.0.1",
        port=9141,
        commit_service=service,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        csrf = client.get("/api/session").json()["csrf_token"]
        response = _post(client, csrf, "batch-error")

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.headers["cache-control"] == "no-store"
    assert "/tmp/secret" not in response.text
