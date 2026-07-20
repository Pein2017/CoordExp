from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.runtime import (
    AuthenticationError,
    AuthenticatedPrincipal,
    AuthoritativeDraftSnapshot,
    BatchCoordinator,
    DraftCatalogCapture,
    DraftCatalogError,
    DraftCatalogRequest,
    RefinementRuntime,
    WorkerState,
)
from src.label_studio_coco_refinement.store import (
    AuthoritativeDraftIdentity,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BootstrapSpec,
    InferenceReceiptLink,
    RecoveryError,
    StoreBusyError,
    WorkingDatasetStore,
    canonical_json,
    semantic_hash,
    sha256_file,
    sha256_json,
)


class AcceptingAnnotationVerifier:
    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        return True


class EmptyInferenceResolver:
    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        return None


class MutableCatalog:
    def __init__(self) -> None:
        self.captures: dict[str, DraftCatalogCapture] = {}
        self.calls: list[DraftCatalogRequest] = []

    def capture_current_user_drafts(
        self, request: DraftCatalogRequest
    ) -> DraftCatalogCapture:
        self.calls.append(request)
        return self.captures[request.split]


def _row(split: str, image_id: int) -> dict[str, Any]:
    return {
        "images": [f"../rescale_32_1024_bbox/images/{split}2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 30, 40],
                "desc": "cat",
                "category_id": 17,
                "category_name": "cat",
                "coco_ann_id": image_id * 100 + 1,
            }
        ],
        "width": 640,
        "height": 480,
        "image_id": image_id,
        "file_name": f"images/{split}2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": split},
    }


@pytest.fixture
def stores(
    tmp_path: Path,
) -> tuple[
    dict[str, WorkingDatasetStore], AcceptingAnnotationVerifier, EmptyInferenceResolver
]:
    source_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox" / "images"
    source_root.mkdir(parents=True)
    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceResolver()
    result: dict[str, WorkingDatasetStore] = {}
    for split in ("train", "val"):
        image_split = image_root / f"{split}2017"
        image_split.mkdir(parents=True)
        for image_id in (1, 2):
            (image_split / f"{image_id:012d}.jpg").write_bytes(
                f"{split}-image-{image_id}".encode()
            )
        source = source_root / f"{split}.norm.jsonl"
        source.write_text(
            "".join(
                canonical_json(_row(split, image_id)) + "\n" for image_id in (1, 2)
            ),
            encoding="utf-8",
        )
        bootstrapped = WorkingDatasetStore.bootstrap(
            BootstrapSpec(
                split=split,
                source_path=source,
                runtime_root=tmp_path / "runtime",
                image_root=image_root,
                expected_source_sha256=sha256_file(source),
                project_id=f"project-{split}",
                storage_id=f"storage-{split}",
                adapter_version="adapter-v1",
                vendor_revision="label-studio-rev",
                registry_fingerprint=COCO80_REGISTRY.fingerprint,
                label_config_fingerprint="label-config-v1",
            ),
            annotation_verifier=verifier,
            inference_receipt_resolver=resolver,
        )
        result[split] = bootstrapped.store
    return result, verifier, resolver


def _snapshot(
    store: WorkingDatasetStore,
    split: str,
    image_id: int,
    *,
    revision: str = "annotation-v1",
    draft_updated_at: str = "2026-07-15T00:00:01Z",
    x1: int = 11,
) -> AuthoritativeDraftSnapshot:
    restored = store.restore_draft(image_id)
    regions: list[dict[str, Any]] = []
    for obj in restored.row["objects"]:
        copied = dict(obj)
        copied["region_key"] = next(
            key
            for key, object_id in restored.region_id_mapping.items()
            if object_id == obj["coco_ann_id"]
        )
        regions.append(copied)
    regions[0]["bbox_2d"] = [x1, 20, 30, 40]
    return AuthoritativeDraftSnapshot(
        split=split,
        project_id=f"project-{split}",
        image_id=image_id,
        task_id=f"{split}:{image_id}",
        annotation_id=f"annotation-{split}-{image_id}",
        draft_id=f"draft-{split}-{image_id}",
        annotation_revision=revision,
        draft_updated_at=draft_updated_at,
        semantic_hash=semantic_hash(regions),
        result_hash=sha256_json(regions),
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
    )


def _set_capture(
    catalog: MutableCatalog,
    store: WorkingDatasetStore,
    split: str,
    image_ids: tuple[int, ...],
    *,
    revision: str = "annotation-v1",
    draft_updated_at: str = "2026-07-15T00:00:01Z",
    x1: int = 11,
) -> None:
    snapshots = tuple(
        _snapshot(
            store,
            split,
            image_id,
            revision=revision,
            draft_updated_at=draft_updated_at,
            x1=x1,
        )
        for image_id in image_ids
    )
    catalog.captures[split] = DraftCatalogCapture(
        split=split,
        project_id=f"project-{split}",
        current_user_id="reviewer",
        base_generation=max(snapshot.observed_generation for snapshot in snapshots),
        snapshots=snapshots,
    )


def _runtime(
    stores: dict[str, WorkingDatasetStore], catalog: MutableCatalog
) -> RefinementRuntime:
    coordinator = BatchCoordinator(stores, poll_interval=0.01)
    return RefinementRuntime(
        catalog=catalog,
        stores=stores,
        project_ids={split: f"project-{split}" for split in stores},
        coordinator=coordinator,
    )


def _wait_until(predicate: Any, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def _batch_status_is(
    store: WorkingDatasetStore, batch_id: str, expected: BatchStatus
) -> bool:
    try:
        return store.get_batch_status(batch_id).status is expected
    except StoreBusyError:
        # Status intentionally fails closed while the worker owns a projection
        # boundary; polling retries instead of inventing an intermediate state.
        return False


def test_enqueue_precedes_blocked_worker_and_queued_payload_is_immutable(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]
    catalog = MutableCatalog()
    _set_capture(catalog, store, "train", (1,))
    captured = catalog.captures["train"].snapshots[0]
    original_process = store.process_next_batch
    worker_entered = threading.Event()
    release_worker = threading.Event()

    def blocked_process() -> Any:
        worker_entered.set()
        assert release_worker.wait(3.0)
        return original_process()

    monkeypatch.setattr(store, "process_next_batch", blocked_process)
    runtime = _runtime({"train": store}, catalog)
    runtime.start_workers()
    try:
        assert worker_entered.wait(1.0)
        receipt = runtime.capture_and_enqueue(
            split="train",
            batch_id="batch-blocked",
            principal=AuthenticatedPrincipal("reviewer", authenticated=True),
        )
        assert receipt.status == "queued"
        assert store.get_batch_status("batch-blocked").status is BatchStatus.QUEUED

        # Deep immutability protects both the capture and durable queue payload.
        with pytest.raises(TypeError):
            captured.regions[0]["bbox_2d"] = [999, 20, 30, 40]
        queued = [
            json.loads(line) for line in store.queue_path.read_text().splitlines()
        ]
        queued_request = queued[0]["payload"]["members"][0]["request"]
        bbox = queued_request["regions"][0]["bbox_2d"]
        assert bbox == [11, 20, 30, 40]
        assert queued[0]["payload"]["current_user_id"] == "reviewer"
        assert queued_request["annotation_revision"] == "annotation-v1"
        assert queued_request["draft_updated_at"] == "2026-07-15T00:00:01Z"
        assert queued_request["result_hash"] == captured.result_hash
        assert runtime.worker_health("train").state is WorkerState.RUNNING

        release_worker.set()
        _wait_until(
            lambda: _batch_status_is(store, "batch-blocked", BatchStatus.SUCCEEDED)
        )
        assert store.restore_draft(1).row["objects"][0]["bbox_2d"] == [11, 20, 30, 40]
    finally:
        release_worker.set()
        runtime.stop_workers()


def test_lost_response_retry_queries_status_without_recapturing(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, _, _ = stores
    catalog = MutableCatalog()
    _set_capture(catalog, split_stores["train"], "train", (1,))
    runtime = _runtime({"train": split_stores["train"]}, catalog)
    principal = AuthenticatedPrincipal("reviewer", authenticated=True)

    first = runtime.capture_and_enqueue(
        split="train", batch_id="lost-response", principal=principal
    )
    _set_capture(
        catalog,
        split_stores["train"],
        "train",
        (1,),
        revision="annotation-v2",
        draft_updated_at="2026-07-15T00:00:02Z",
        x1=12,
    )
    retry = runtime.capture_and_enqueue(
        split="train", batch_id="lost-response", principal=principal
    )

    assert retry == first
    assert retry.to_json() == first.to_json()
    assert len(catalog.calls) == 1


def test_task_scope_reaches_catalog_and_lost_retry_still_uses_status(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, _, _ = stores
    catalog = MutableCatalog()
    _set_capture(catalog, split_stores["train"], "train", (2,))
    runtime = _runtime({"train": split_stores["train"]}, catalog)
    principal = AuthenticatedPrincipal("reviewer", authenticated=True)

    first = runtime.capture_and_enqueue(
        split="train",
        batch_id="scoped-lost-response",
        principal=principal,
        task_ids=("train:2",),
    )
    retry = runtime.capture_and_enqueue(
        split="train",
        batch_id="scoped-lost-response",
        principal=principal,
        task_ids=("invalid", "invalid"),
    )

    assert retry == first
    assert len(catalog.calls) == 1
    assert catalog.calls[0].task_ids == ("train:2",)


def test_same_split_has_one_active_batch_and_server_resolves_member_order(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]
    catalog = MutableCatalog()
    _set_capture(catalog, store, "train", (2, 1))
    runtime = _runtime({"train": store}, catalog)
    principal = AuthenticatedPrincipal("reviewer", authenticated=True)

    first = runtime.capture_and_enqueue(
        split="train", batch_id="first", principal=principal
    )
    _set_capture(
        catalog,
        store,
        "train",
        (2,),
        revision="annotation-v2",
        draft_updated_at="2026-07-15T00:00:02Z",
        x1=12,
    )
    second = runtime.capture_and_enqueue(
        split="train", batch_id="second", principal=principal
    )

    assert second.batch_id == first.batch_id == "first"
    assert store.get_batch_status("second").status is BatchStatus.NOT_FOUND
    queued = json.loads(store.queue_path.read_text().splitlines()[0])
    assert [member["source_row_index"] for member in queued["payload"]["members"]] == [
        0,
        1,
    ]
    assert [
        member["request"]["task_id"] for member in queued["payload"]["members"]
    ] == ["train:1", "train:2"]


def test_train_and_val_workers_progress_independently(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_stores, _, _ = stores
    catalog = MutableCatalog()
    for split in ("train", "val"):
        _set_capture(catalog, split_stores[split], split, (1,))
    train_original = split_stores["train"].process_next_batch
    train_entered = threading.Event()
    release_train = threading.Event()

    def blocked_train() -> Any:
        train_entered.set()
        assert release_train.wait(3.0)
        return train_original()

    monkeypatch.setattr(split_stores["train"], "process_next_batch", blocked_train)
    runtime = _runtime(split_stores, catalog)
    runtime.start_workers()
    try:
        assert train_entered.wait(1.0)
        principal = AuthenticatedPrincipal("reviewer", authenticated=True)
        runtime.capture_and_enqueue(
            split="train", batch_id="train-batch", principal=principal
        )
        runtime.capture_and_enqueue(
            split="val", batch_id="val-batch", principal=principal
        )
        _wait_until(
            lambda: _batch_status_is(
                split_stores["val"], "val-batch", BatchStatus.SUCCEEDED
            )
        )
        assert (
            split_stores["train"].get_batch_status("train-batch").status
            is BatchStatus.QUEUED
        )
        release_train.set()
        _wait_until(
            lambda: _batch_status_is(
                split_stores["train"], "train-batch", BatchStatus.SUCCEEDED
            )
        )
    finally:
        release_train.set()
        runtime.stop_workers()


def test_clean_stop_then_restart_recovers_queued_batch(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, verifier, resolver = stores
    store = split_stores["train"]
    catalog = MutableCatalog()
    _set_capture(catalog, store, "train", (1,))
    runtime = _runtime({"train": store}, catalog)
    runtime.start_workers()
    _wait_until(lambda: runtime.worker_health("train").state is WorkerState.IDLE)
    runtime.stop_workers()
    stopped = runtime.worker_health("train")
    assert stopped.state is WorkerState.STOPPED
    assert stopped.healthy is False

    runtime.capture_and_enqueue(
        split="train",
        batch_id="restart-batch",
        principal=AuthenticatedPrincipal("reviewer", authenticated=True),
    )
    assert store.get_batch_status("restart-batch").status is BatchStatus.QUEUED

    reopened = WorkingDatasetStore(
        store.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    restarted = BatchCoordinator({"train": reopened}, poll_interval=0.01)
    restarted.start()
    try:
        _wait_until(
            lambda: _batch_status_is(reopened, "restart-batch", BatchStatus.SUCCEEDED)
            or restarted.health("train").state is WorkerState.FAILED
        )
        health = restarted.health("train")
        assert _batch_status_is(reopened, "restart-batch", BatchStatus.SUCCEEDED), (
            health
        )
        assert health.last_batch_id == "restart-batch"
        assert health.processed_batches == 1
    finally:
        restarted.stop()


def test_terminal_observer_runs_after_publication_before_processed_count(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]
    catalog = MutableCatalog()
    _set_capture(catalog, store, "train", (1,))
    observed: list[tuple[str, WorkingDatasetStore, BatchResult, int]] = []
    coordinator: BatchCoordinator

    def observe(
        *,
        split: str,
        store: WorkingDatasetStore,
        request: BatchRequest,
        result: BatchResult,
    ) -> None:
        assert request.batch_id == result.batch_id
        observed.append(
            (split, store, result, coordinator.health("train").processed_batches)
        )

    coordinator = BatchCoordinator(
        {"train": store}, poll_interval=0.01, on_batch_result=observe
    )
    runtime = RefinementRuntime(
        catalog=catalog,
        stores={"train": store},
        project_ids={"train": "project-train"},
        coordinator=coordinator,
    )
    runtime.start_workers()
    try:
        _wait_until(lambda: runtime.worker_health("train").state is WorkerState.IDLE)
        runtime.capture_and_enqueue(
            split="train",
            batch_id="observed-terminal",
            principal=AuthenticatedPrincipal("reviewer", authenticated=True),
        )
        _wait_until(lambda: len(observed) == 1)
        split, observed_store, result, count_during_callback = observed[0]
        assert split == "train"
        assert observed_store is store
        assert result.batch_id == "observed-terminal"
        assert result.status is BatchStatus.SUCCEEDED
        assert count_during_callback == 0
        _wait_until(
            lambda: runtime.worker_health("train").processed_batches == 1
        )
    finally:
        runtime.stop_workers()


def test_terminal_observer_failure_stops_worker_after_durable_terminal(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]
    catalog = MutableCatalog()
    _set_capture(catalog, store, "train", (1,))
    fail_projection = {"value": True}
    reconciled: list[str] = []

    def reject_projection(
        *,
        split: str,
        store: WorkingDatasetStore,
        request: BatchRequest,
        result: BatchResult,
    ) -> None:
        del split, store, request
        if fail_projection["value"]:
            raise ValueError("terminal projection failed")
        reconciled.append(result.batch_id)

    coordinator = BatchCoordinator(
        {"train": store},
        poll_interval=0.01,
        on_batch_result=reject_projection,
    )
    runtime = RefinementRuntime(
        catalog=catalog,
        stores={"train": store},
        project_ids={"train": "project-train"},
        coordinator=coordinator,
    )
    runtime.start_workers()
    _wait_until(lambda: runtime.worker_health("train").state is WorkerState.IDLE)
    runtime.capture_and_enqueue(
        split="train",
        batch_id="projection-failure",
        principal=AuthenticatedPrincipal("reviewer", authenticated=True),
    )

    _wait_until(lambda: runtime.worker_health("train").state is WorkerState.FAILED)
    health = runtime.worker_health("train")
    assert (
        store.get_batch_status("projection-failure").status
        is BatchStatus.RECONCILING
    )
    assert health.thread_alive is False
    assert health.last_batch_id == "projection-failure"
    assert health.processed_batches == 0
    assert health.error == "ValueError: terminal projection failed"

    fail_projection["value"] = False
    runtime.start_workers()
    _wait_until(lambda: runtime.worker_health("train").state is WorkerState.IDLE)
    assert reconciled == ["projection-failure"]
    assert store.get_batch_status("projection-failure").status is BatchStatus.SUCCEEDED
    runtime.stop_workers()


def test_worker_busy_is_queryable_and_retries_fail_closed(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]
    original_recover = store.recover
    release = threading.Event()

    def busy_recovery() -> None:
        if not release.is_set():
            raise StoreBusyError("recovery lock busy")
        original_recover()

    monkeypatch.setattr(store, "recover", busy_recovery)
    coordinator = BatchCoordinator({"train": store}, poll_interval=0.01)
    coordinator.start()
    _wait_until(lambda: coordinator.health("train").state is WorkerState.BUSY)
    health = coordinator.health("train")
    assert health.thread_alive is True
    assert health.healthy is False
    assert health.error == "StoreBusyError: recovery lock busy"
    release.set()
    _wait_until(lambda: coordinator.health("train").state is WorkerState.IDLE)
    coordinator.stop()


def test_worker_recovery_error_is_queryable_and_fail_closed(
    stores: tuple[
        dict[str, WorkingDatasetStore],
        AcceptingAnnotationVerifier,
        EmptyInferenceResolver,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_stores, _, _ = stores
    store = split_stores["train"]

    def failed_recovery() -> None:
        raise RecoveryError("projection disagreement")

    monkeypatch.setattr(store, "recover", failed_recovery)
    coordinator = BatchCoordinator({"train": store}, poll_interval=0.01)
    coordinator.start()
    _wait_until(lambda: coordinator.health("train").state is WorkerState.FAILED)
    health = coordinator.health("train")
    assert health.thread_alive is False
    assert health.healthy is False
    assert health.error == "RecoveryError: projection disagreement"


def test_principal_requires_explicit_server_authentication() -> None:
    with pytest.raises(TypeError):
        AuthenticatedPrincipal("reviewer")  # type: ignore[call-arg]
    with pytest.raises(AuthenticationError, match="authenticated current-user"):
        AuthenticatedPrincipal("reviewer", authenticated=False)
    with pytest.raises(AuthenticationError, match="authenticated current-user"):
        AuthenticatedPrincipal("   ", authenticated=True)


def test_snapshot_rejects_nonfinite_nested_json_before_catalog_capture() -> None:
    with pytest.raises(DraftCatalogError, match="finite numbers"):
        AuthoritativeDraftSnapshot(
            split="train",
            project_id="project-train",
            image_id=1,
            task_id="train:1",
            annotation_id="annotation-train-1",
            draft_id="draft-train-1",
            annotation_revision="annotation-v1",
            draft_updated_at="2026-07-15T00:00:01Z",
            semantic_hash="0" * 64,
            result_hash="1" * 64,
            base_row_hash="2" * 64,
            observed_generation=0,
            regions=(
                {
                    "region_key": "drawn:1",
                    "bbox_2d": [1, 2, 3, 4],
                    "category_name": "cat",
                    "category_id": 17,
                    "label_studio_result": {"score": float("nan")},
                },
            ),
        )


@pytest.mark.parametrize("inference_receipts", [(123,), "receipt-1", None])
def test_snapshot_rejects_non_string_or_non_sequence_inference_receipts(
    inference_receipts: Any,
) -> None:
    with pytest.raises(DraftCatalogError, match="inference receipt"):
        AuthoritativeDraftSnapshot(
            split="train",
            project_id="project-train",
            image_id=1,
            task_id="train:1",
            annotation_id="annotation-train-1",
            draft_id="draft-train-1",
            annotation_revision="annotation-v1",
            draft_updated_at="2026-07-15T00:00:01Z",
            semantic_hash="0" * 64,
            result_hash="1" * 64,
            base_row_hash="2" * 64,
            observed_generation=0,
            regions=(
                {
                    "region_key": "drawn:1",
                    "bbox_2d": [1, 2, 3, 4],
                    "category_name": "cat",
                    "category_id": 17,
                },
            ),
            inference_receipts=inference_receipts,
        )
