from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from src.coco_refinement.bootstrap import BootstrapSourceContract
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.preflight import (
    LaunchPreflight,
    RuntimeRootBusyError,
    SUPPORTED_DEPENDENCIES,
    RuntimeRootLock,
    run_launch_preflight,
)
from src.coco_refinement.repository import SaveDraftRequest
from src.coco_refinement.runtime import (
    AdapterFactories,
    RuntimeAssemblyError,
    RuntimeShutdownError,
    RuntimeStartupError,
    RuntimeState,
    SourceInspectionError,
    StoreTerminalPairProvider,
    create_standalone_runtime,
    inspect_source_contracts_read_only,
    production_adapter_factories,
)
from src.label_studio_coco_refinement.store import (
    BatchRequest,
    BatchResult,
    BatchStatus,
    WorkingDatasetStore,
    canonical_json,
    sha256_file,
)
from src.label_studio_coco_refinement.runtime import AuthenticatedPrincipal


def _supported_version(distribution: str) -> str:
    return SUPPORTED_DEPENDENCIES[distribution]


def _row(split: str, image_id: int) -> dict[str, object]:
    return {
        "images": [f"../rescale_32_1024_bbox/images/{split}2017/{image_id:012d}.jpg"],
        "objects": [
            {
                "bbox_2d": [10, 20, 300, 400],
                "desc": "person",
                "category_id": 1,
                "category_name": "person",
                "coco_ann_id": image_id * 100,
            }
        ],
        "width": 32,
        "height": 24,
        "image_id": image_id,
        "file_name": f"images/{split}2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": split},
    }


def _dual_contracts(tmp_path: Path) -> tuple[BootstrapSourceContract, ...]:
    source_root = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data/coco/rescale_32_1024_bbox/images"
    source_root.mkdir(parents=True, exist_ok=True)
    contracts: list[BootstrapSourceContract] = []
    for split, image_id, color in (("train", 1, "red"), ("val", 2, "blue")):
        split_images = image_root / f"{split}2017"
        split_images.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color=color).save(
            split_images / f"{image_id:012d}.jpg", format="JPEG"
        )
        source = source_root / f"{split}.norm.jsonl"
        source.write_text(
            canonical_json(_row(split, image_id)) + "\n", encoding="utf-8"
        )
        contracts.append(
            BootstrapSourceContract(
                split=split,  # type: ignore[arg-type]
                source_path=source,
                image_root=image_root,
                expected_source_sha256=sha256_file(source),
                expected_row_count=1,
            )
        )
    return tuple(contracts)


def _dual_contracts_with_two_train_tasks(
    tmp_path: Path,
) -> tuple[BootstrapSourceContract, ...]:
    contracts = list(_dual_contracts(tmp_path))
    train = contracts[0]
    second_image_id = 3
    Image.new("RGB", (32, 24), color="green").save(
        train.image_root / f"train2017/{second_image_id:012d}.jpg",
        format="JPEG",
    )
    train.source_path.write_text(
        "".join(
            canonical_json(_row("train", image_id)) + "\n"
            for image_id in (1, second_image_id)
        ),
        encoding="utf-8",
    )
    contracts[0] = replace(
        train,
        expected_source_sha256=sha256_file(train.source_path),
        expected_row_count=2,
    )
    return tuple(contracts)


def _local_region(
    region_key: str,
    *,
    bbox: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "region_key": region_key,
        "bbox_2d": list(bbox),
        "category_name": "person",
        "category_id": 1,
    }


def _committed_region(split: str, image_id: int) -> dict[str, object]:
    object_id = image_id * 100
    return {
        "region_key": f"{split}:coco:{object_id}",
        "bbox_2d": [10, 20, 300, 400],
        "category_name": "person",
        "category_id": 1,
        "coco_ann_id": object_id,
    }


class _Verifier:
    def verify(self, _identity: object) -> bool:
        return False


class _ReceiptResolver:
    def resolve(self, _receipt_id: str) -> None:
        return None


class _Catalog:
    pass


class _TerminalReconciler:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reconcile_existing_terminals(self, pairs: object) -> None:
        del pairs
        self.events.append("reconcile:existing")

    def reconcile_batch(self, request: object, result: object) -> None:
        del result
        self.events.append(f"terminal:{request.split}")


class _TerminalPairProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def existing_pairs(self, *, split: str, store: object) -> tuple[()]:
        del store
        self.events.append(f"pairs:{split}")
        return ()

    def request_for_result(
        self, *, split: str, store: object, result: object
    ) -> object:
        del store, result
        self.events.append(f"request:{split}")
        return SimpleNamespace(split=split)


class _Store:
    def __init__(self, split: str, events: list[str]) -> None:
        self.split = split
        self.events = events

    def recover(self) -> None:
        self.events.append(f"recover:{self.split}")


class _Repository:
    def __init__(self, path: Path, events: list[str]) -> None:
        self.path = path
        self.events = events

    def bootstrap_project(self, project: object, tasks: object) -> None:
        del tasks
        self.events.append(f"attest:{project.split}")


class _FakeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.started = False
        self.stop_fails = False
        self.stop_leaves_alive = False
        self.start_fails = False
        self.health_fails = False
        self.val_ready_after = 0
        self.started_health_calls = 0

    def start_workers(self, split: str | None = None) -> None:
        del split
        self.events.append("workers:start")
        self.started = True

    def stop_workers(self, split: str | None = None, *, timeout: float = 5.0) -> None:
        del split, timeout
        self.events.append("workers:stop")
        if self.stop_fails:
            raise RuntimeError("injected stop failure")
        if self.stop_leaves_alive:
            return
        self.started = False

    def worker_health(self, split: str | None = None) -> object:
        del split
        if self.health_fails:
            raise RuntimeError("live health unavailable")
        if self.started:
            self.started_health_calls += 1
        return {
            name: {
                "split": name,
                "state": (
                    "failed"
                    if self.started and self.start_fails and name == "val"
                    else "recovering"
                    if self.started
                    and name == "val"
                    and self.started_health_calls <= self.val_ready_after
                    else "idle"
                    if self.started
                    else "stopped"
                ),
                "thread_alive": self.started,
                "healthy": self.started
                and not (self.start_fails and name == "val")
                and not (
                    name == "val" and self.started_health_calls <= self.val_ready_after
                ),
                "processed_batches": 0,
                "last_batch_id": None,
                "error": None,
            }
            for name in ("train", "val")
        }


class _RecordingPreflight:
    def __init__(self, inner: LaunchPreflight, events: list[str]) -> None:
        self.inner = inner
        self.events = events

    @property
    def writer_lock(self):  # type: ignore[no-untyped-def]
        return self.inner.writer_lock

    def release(self) -> None:
        self.events.append("lock:release")
        self.inner.release()


def _fake_workspace(
    runtime_root: Path,
    stores: Mapping[str, _Store],
    repository: _Repository,
) -> Any:
    return SimpleNamespace(
        runtime_root=runtime_root,
        repository=repository,
        splits={
            split: SimpleNamespace(
                store=store,
                project=SimpleNamespace(
                    project_id=f"coco-refinement:{split}", split=split
                ),
                tasks=(),
            )
            for split, store in stores.items()
        },
    )


def _assemble_fake(
    tmp_path: Path,
    *,
    events: list[str] | None = None,
    source_inspector: Callable[..., Any] = inspect_source_contracts_read_only,
) -> tuple[Any, _FakeRuntime, list[str]]:
    observed = [] if events is None else events
    contracts = _dual_contracts(tmp_path)
    runtime_root = tmp_path / "runtime"
    stores = {split: _Store(split, observed) for split in ("train", "val")}
    fake_runtime = _FakeRuntime(observed)
    terminal = _TerminalReconciler(observed)
    pair_provider = _TerminalPairProvider(observed)

    def preflight_runner(root: Path, **kwargs: object) -> _RecordingPreflight:
        observed.append("preflight")
        return _RecordingPreflight(run_launch_preflight(root, **kwargs), observed)  # type: ignore[arg-type]

    def inspect_both(values: Any) -> Any:
        observed.extend(["inspect:train", "inspect:val"])
        return source_inspector(values)

    def repository_factory(path: Path) -> object:
        assert observed[:3] == ["preflight", "inspect:train", "inspect:val"]
        observed.append("repository")
        return _Repository(path, observed)

    adapters = AdapterFactories(
        verifier=lambda _repo: observed.append("adapter:verifier") or _Verifier(),
        inference_receipt_resolver=lambda _repo: observed.append("adapter:resolver")
        or _ReceiptResolver(),
        catalog=lambda _repo, _stores, _projects: observed.append("adapter:catalog")
        or _Catalog(),
        terminal_reconciler=lambda _repo, _stores, _projects: observed.append(
            "adapter:terminal"
        )
        or terminal,
    )

    def bootstrapper(*_args: object, **kwargs: object) -> Any:
        observed.append("bootstrap")
        assert kwargs["attest_repository"] is False
        repository = kwargs["repository"]
        assert isinstance(repository, _Repository)
        return _fake_workspace(runtime_root, stores, repository)

    def coordinator_factory(
        _stores: object, *, on_batch_result: Callable[..., None]
    ) -> _FakeRuntime:
        observed.append("coordinator")
        on_batch_result(
            split="train",
            store=stores["train"],
            request=SimpleNamespace(split="train"),
            result=object(),
        )
        return fake_runtime

    def runtime_factory(**kwargs: object) -> _FakeRuntime:
        observed.append("runtime")
        return kwargs["coordinator"]  # type: ignore[return-value]

    runtime = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapters,
        source_inspector=inspect_both,
        repository_factory=repository_factory,  # type: ignore[arg-type]
        workspace_bootstrapper=bootstrapper,
        coordinator_factory=coordinator_factory,  # type: ignore[arg-type]
        refinement_runtime_factory=runtime_factory,  # type: ignore[arg-type]
        terminal_pair_provider=pair_provider,  # type: ignore[arg-type]
        preflight_runner=preflight_runner,  # type: ignore[arg-type]
        environment={},
        version_resolver=_supported_version,
        startup_timeout=0.2,
        poll_interval=0.001,
    )
    return runtime, fake_runtime, observed


def test_factory_orders_dual_inspection_recovery_reconciliation_and_callback(
    tmp_path: Path,
) -> None:
    runtime, _, events = _assemble_fake(tmp_path)

    assert events[:20] == [
        "preflight",
        "inspect:train",
        "inspect:val",
        "repository",
        "adapter:verifier",
        "adapter:resolver",
        "bootstrap",
        "adapter:catalog",
        "adapter:terminal",
        "recover:train",
        "pairs:train",
        "reconcile:existing",
        "recover:val",
        "pairs:val",
        "reconcile:existing",
        "attest:train",
        "attest:val",
        "coordinator",
        "terminal:train",
        "runtime",
    ]
    assert runtime.lock_held is True
    runtime.shutdown()


def test_start_is_idempotent_and_waits_for_both_splits_before_accepting(
    tmp_path: Path,
) -> None:
    runtime, _, events = _assemble_fake(tmp_path)
    runtime.runtime.val_ready_after = 1

    first = runtime.start()
    second = runtime.start()

    assert events.count("workers:start") == 1
    assert runtime.runtime.started_health_calls >= 2
    assert runtime.state is RuntimeState.READY
    assert runtime.accepting_writes is True
    assert first["workers"]["train"]["state"] == "idle"
    assert second["workers"]["val"]["state"] == "idle"
    assert (
        json.loads(runtime.health_receipt_path.read_text())["accepting_writes"] is True
    )
    runtime.shutdown()


def test_shutdown_stops_before_unlock_and_is_idempotent(tmp_path: Path) -> None:
    runtime, _, events = _assemble_fake(tmp_path)
    runtime.start()

    stopped = runtime.shutdown()
    repeated = runtime.shutdown()

    assert events.index("workers:stop") < events.index("lock:release")
    assert events.count("workers:stop") == 1
    assert events.count("lock:release") == 1
    assert stopped["lock_held"] is False
    assert repeated["state"] == "stopped"
    assert runtime.lock_held is False


def test_stop_failure_retains_lock_until_a_later_clean_stop(tmp_path: Path) -> None:
    runtime, fake, events = _assemble_fake(tmp_path)
    runtime.start()
    fake.stop_fails = True

    with pytest.raises(RuntimeShutdownError, match="lock retained"):
        runtime.shutdown(timeout=0.01)

    assert runtime.state is RuntimeState.STOP_FAILED
    assert runtime.lock_held is True
    assert "lock:release" not in events
    with pytest.raises(RuntimeRootBusyError):
        RuntimeRootLock(runtime.workspace.runtime_root).acquire()

    fake.stop_fails = False
    runtime.shutdown()
    assert runtime.lock_held is False


def test_stop_returning_with_a_live_worker_retains_lock(tmp_path: Path) -> None:
    runtime, fake, _ = _assemble_fake(tmp_path)
    runtime.start()
    fake.stop_leaves_alive = True

    with pytest.raises(RuntimeShutdownError, match="lock retained"):
        runtime.shutdown(timeout=0.01)

    assert runtime.lock_held is True
    fake.stop_leaves_alive = False
    runtime.shutdown()


def test_worker_start_failure_stops_before_releasing_root_lock(tmp_path: Path) -> None:
    runtime, fake, events = _assemble_fake(tmp_path)
    fake.start_fails = True

    with pytest.raises(RuntimeStartupError, match="failed before becoming ready"):
        runtime.start()

    assert events.index("workers:stop") < events.index("lock:release")
    assert runtime.lock_held is False
    assert runtime.accepting_writes is False


def test_duplicate_runtime_root_is_rejected_before_second_source_inspection(
    tmp_path: Path,
) -> None:
    first, _, _ = _assemble_fake(tmp_path)
    second_events: list[str] = []

    with pytest.raises(RuntimeRootBusyError):
        _assemble_fake(tmp_path, events=second_events)

    assert second_events == ["preflight"]
    first.shutdown()


def test_invalid_terminal_pair_provider_fails_before_preflight(tmp_path: Path) -> None:
    contracts = _dual_contracts(tmp_path)
    runtime_root = tmp_path / "runtime-no-provider"
    terminal = _TerminalReconciler([])
    adapters = AdapterFactories(
        verifier=lambda _repository: _Verifier(),
        inference_receipt_resolver=lambda _repository: _ReceiptResolver(),
        catalog=lambda _repository, _stores, _projects: _Catalog(),
        terminal_reconciler=lambda _repository, _stores, _projects: terminal,
    )

    with pytest.raises(RuntimeAssemblyError, match="lacks durable pair"):
        create_standalone_runtime(
            tmp_path,
            runtime_root=runtime_root,
            source_contracts=contracts,
            adapter_factories=adapters,
            terminal_pair_provider=object(),  # type: ignore[arg-type]
            environment={},
            version_resolver=_supported_version,
        )

    assert not runtime_root.exists()


def test_store_terminal_pair_provider_uses_durable_public_store_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(WorkingDatasetStore)
    store.split_dir = tmp_path / "train"
    request = BatchRequest(
        batch_id="batch-1",
        split="train",
        current_user_id="local-operator",
        base_generation=0,
        members=(),
    )
    result = BatchResult(
        batch_id="batch-1",
        payload_hash="payload",
        status=BatchStatus.FAILED,
        split="train",
        generation=0,
        working_sha256="working",
        members=(),
        error="failed",
    )
    monkeypatch.setattr(store, "terminal_batch_pairs", lambda: ((request, result),))
    monkeypatch.setattr(store, "get_batch_request", lambda _batch_id: request)
    monkeypatch.setattr(store, "get_batch_result", lambda _batch_id: result)
    provider = StoreTerminalPairProvider()

    assert provider.existing_pairs(split="train", store=store) == ((request, result),)
    assert (
        provider.request_for_result(split="train", store=store, result=result)
        is request
    )
    with pytest.raises(RuntimeAssemblyError, match="store differs"):
        provider.existing_pairs(split="val", store=store)


def test_store_terminal_pair_provider_rejects_callback_result_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(WorkingDatasetStore)
    store.split_dir = tmp_path / "train"
    request = BatchRequest("batch-1", "train", "local-operator", 0, ())
    callback_result = BatchResult(
        "batch-1", "payload", BatchStatus.FAILED, "train", 0, "working", (), "failed"
    )
    durable_result = replace(callback_result, working_sha256="other")
    monkeypatch.setattr(store, "get_batch_request", lambda _batch_id: request)
    monkeypatch.setattr(store, "get_batch_result", lambda _batch_id: durable_result)

    with pytest.raises(RuntimeAssemblyError, match="differs from the durable"):
        StoreTerminalPairProvider().request_for_result(
            split="train", store=store, result=callback_result
        )


def test_source_inspection_failure_releases_lock_before_sqlite_creation(
    tmp_path: Path,
) -> None:
    def fail(_contracts: object) -> object:
        raise SourceInspectionError(
            "injected source failure", code="coco_refinement.test_source_failure"
        )

    with pytest.raises(SourceInspectionError, match="injected source failure"):
        _assemble_fake(tmp_path, source_inspector=fail)

    with RuntimeRootLock(tmp_path / "runtime"):
        pass


@pytest.mark.parametrize("failure", ["bbox", "missing", "symlink", "dimensions"])
def test_val_semantic_or_image_drift_creates_no_sqlite_or_split_state(
    tmp_path: Path, failure: str
) -> None:
    contracts = list(_dual_contracts(tmp_path))
    val = contracts[1]
    row = json.loads(val.source_path.read_text(encoding="utf-8"))
    image_path = val.image_root / "val2017/000000000002.jpg"
    if failure == "bbox":
        row["objects"][0]["bbox_2d"] = [10, 20, 10, 40]
        val.source_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
        contracts[1] = replace(val, expected_source_sha256=sha256_file(val.source_path))
    elif failure == "missing":
        image_path.unlink()
    elif failure == "symlink":
        outside = tmp_path / "outside.jpg"
        image_path.replace(outside)
        image_path.symlink_to(outside)
    else:
        row["width"] = 31
        val.source_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
        contracts[1] = replace(val, expected_source_sha256=sha256_file(val.source_path))

    runtime_root = tmp_path / "runtime-invalid"
    events: list[str] = []
    terminal = _TerminalReconciler(events)
    adapters = AdapterFactories(
        verifier=lambda _repository: _Verifier(),
        inference_receipt_resolver=lambda _repository: _ReceiptResolver(),
        catalog=lambda _repository, _stores, _projects: _Catalog(),
        terminal_reconciler=lambda _repository, _stores, _projects: terminal,
    )
    with pytest.raises(SourceInspectionError, match="semantic inspection"):
        create_standalone_runtime(
            tmp_path,
            runtime_root=runtime_root,
            source_contracts=tuple(contracts),
            adapter_factories=adapters,
            terminal_pair_provider=_TerminalPairProvider(events),  # type: ignore[arg-type]
            environment={},
            version_resolver=_supported_version,
        )

    assert not (runtime_root / "state.sqlite3").exists()
    assert not (runtime_root / "train").exists()
    assert not (runtime_root / "val").exists()
    with RuntimeRootLock(runtime_root):
        pass


def test_health_snapshot_never_uses_stale_receipt_when_live_health_fails(
    tmp_path: Path,
) -> None:
    runtime, fake, _ = _assemble_fake(tmp_path)
    runtime.start()
    runtime.health_receipt_path.write_text(
        '{"state":"ready","workers":{"stale":true}}\n', encoding="utf-8"
    )
    fake.health_fails = True

    current = runtime.health_snapshot()

    assert current["workers"] is None
    assert "live health unavailable" in current["health_error"]
    fake.health_fails = False
    runtime.shutdown()


def test_bounded_real_bootstrap_worker_lifecycle_and_restart_reconciliation(
    tmp_path: Path,
) -> None:
    contracts = _dual_contracts(tmp_path)
    source_bytes = tuple(contract.source_path.read_bytes() for contract in contracts)
    runtime_root = tmp_path / "runtime-real"

    def adapter_factories() -> AdapterFactories:
        return production_adapter_factories(
            inference_receipt_store=_ReceiptResolver(),
            current_user_id="local-operator",
        )

    first = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapter_factories(),
        environment={},
        version_resolver=_supported_version,
        startup_timeout=2.0,
        poll_interval=0.005,
    )
    assert all(
        type(store.annotation_verifier).__name__ == "SqliteDraftVerifier"
        for store in first.stores.values()
    )
    assert all(
        type(store.inference_receipt_resolver).__name__
        == "SqliteInferenceReceiptResolver"
        for store in first.stores.values()
    )
    with first:
        assert first.accepting_writes is True
    assert first.lock_held is False

    restarted = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapter_factories(),
        environment={},
        version_resolver=_supported_version,
        startup_timeout=2.0,
        poll_interval=0.005,
    )
    with restarted:
        assert restarted.accepting_writes is True
    assert (
        tuple(contract.source_path.read_bytes() for contract in contracts)
        == source_bytes
    )


def test_startup_replays_published_terminal_before_strict_sqlite_attestation(
    tmp_path: Path,
) -> None:
    contracts = _dual_contracts_with_two_train_tasks(tmp_path)
    runtime_root = tmp_path / "runtime-terminal-replay"

    def adapter_factories() -> AdapterFactories:
        return production_adapter_factories(
            inference_receipt_store=_ReceiptResolver(),
            current_user_id="local-operator",
        )

    first = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapter_factories(),
        environment={},
        version_resolver=_supported_version,
    )
    repository = first.workspace.repository
    exact_task, newer_task = first.workspace.splits["train"].tasks
    exact_key = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"
    newer_key = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603939"

    exact_save = repository.save_draft(
        SaveDraftRequest(
            project_id=exact_task.project_id,
            task_id=exact_task.task_id,
            mutation_id="startup-replay-exact",
            expected_revision=0,
            expected_generation=exact_task.current_generation,
            expected_base_row_hash=exact_task.base_row_hash,
            committed=canonicalize_objects(
                [_committed_region("train", exact_task.identity.image_id)],
                split="train",
            ),
            draft=canonicalize_objects(
                [_local_region(exact_key, bbox=(20, 30, 320, 430))],
                split="train",
            ),
        )
    )
    newer_save = repository.save_draft(
        SaveDraftRequest(
            project_id=newer_task.project_id,
            task_id=newer_task.task_id,
            mutation_id="startup-replay-captured",
            expected_revision=0,
            expected_generation=newer_task.current_generation,
            expected_base_row_hash=newer_task.base_row_hash,
            committed=canonicalize_objects(
                [_committed_region("train", newer_task.identity.image_id)],
                split="train",
            ),
            draft=canonicalize_objects(
                [_local_region(newer_key, bbox=(40, 50, 340, 450))],
                split="train",
            ),
        )
    )
    first.runtime.capture_and_enqueue(  # type: ignore[attr-defined]
        split="train",
        batch_id="startup-replay-batch",
        principal=AuthenticatedPrincipal(
            user_id="local-operator", authenticated=True
        ),
    )
    later_bbox = (60, 70, 360, 470)
    repository.save_draft(
        SaveDraftRequest(
            project_id=newer_task.project_id,
            task_id=newer_task.task_id,
            mutation_id="startup-replay-newer",
            expected_revision=newer_save.state.revision,
            expected_generation=newer_task.current_generation,
            expected_base_row_hash=newer_task.base_row_hash,
            committed=canonicalize_objects(
                [_committed_region("train", newer_task.identity.image_id)],
                split="train",
            ),
            draft=canonicalize_objects(
                [_local_region(newer_key, bbox=later_bbox)], split="train"
            ),
        )
    )

    # Process directly through the store to model publication followed by a
    # process crash before BatchCoordinator can invoke its SQLite callback.
    terminal = first.stores["train"].process_next_batch()
    assert terminal is not None
    assert terminal.status is BatchStatus.SUCCEEDED
    assert terminal.generation == 1
    assert repository.get_task_state(
        exact_task.project_id, exact_task.task_id
    ).current_generation == 0
    assert repository.get_task_state(
        newer_task.project_id, newer_task.task_id
    ).current_generation == 0
    assert exact_save.state.draft is not None
    first.shutdown()

    restarted = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapter_factories(),
        environment={},
        version_resolver=_supported_version,
    )
    exact = restarted.workspace.repository.get_task_state(
        exact_task.project_id, exact_task.task_id
    )
    newer = restarted.workspace.repository.get_task_state(
        newer_task.project_id, newer_task.task_id
    )
    assert exact.current_generation == 1
    assert exact.draft is None
    assert newer.current_generation == 1
    assert newer.draft is not None
    assert len(newer.draft.objects) == 1
    assert newer.draft.objects[0].region_key == newer_key
    assert newer.draft.objects[0].bbox_2d == later_bbox
    assert newer.draft.objects[0].coco_ann_id is not None
    assert newer.draft.objects[0].coco_ann_id < 0
    mutation_count = restarted.workspace.repository.count_mutations()
    restarted.shutdown()

    repeated = create_standalone_runtime(
        tmp_path,
        runtime_root=runtime_root,
        source_contracts=contracts,
        adapter_factories=adapter_factories(),
        environment={},
        version_resolver=_supported_version,
    )
    assert repeated.workspace.repository.get_task_state(
        exact_task.project_id, exact_task.task_id
    ) == exact
    assert repeated.workspace.repository.get_task_state(
        newer_task.project_id, newer_task.task_id
    ) == newer
    assert repeated.workspace.repository.count_mutations() == mutation_count
    assert repeated.stores["train"].get_batch_result(
        "startup-replay-batch"
    ).generation == 1
    repeated.shutdown()
