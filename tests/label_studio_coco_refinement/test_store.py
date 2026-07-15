from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
import src.label_studio_coco_refinement.store as store_module
from src.label_studio_coco_refinement.store import (
    AuthoritativeDraftIdentity,
    BootstrapSpec,
    CommitConflictError,
    CommitOutcomeUnknown,
    CommitRequest,
    CommitRolledBackError,
    CommitStatus,
    DraftSaveReceipt,
    InjectedCrash,
    InferenceReceiptLink,
    ManifestDriftError,
    RecoveryError,
    StaleCommitError,
    StoreBusyError,
    ValidationError,
    WorkingDatasetStore,
    canonical_json,
    semantic_hash,
    sha256_file,
    sha256_json,
)


class AcceptingAnnotationVerifier:
    def __init__(self) -> None:
        self.calls: list[AuthoritativeDraftIdentity] = []
        self.accept = True

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        self.calls.append(identity)
        return self.accept


class DictInferenceReceiptResolver:
    def __init__(self) -> None:
        self.links: dict[str, InferenceReceiptLink] = {}
        self.calls: list[str] = []

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        self.calls.append(receipt_id)
        return self.links.get(receipt_id)


def _row(split: str, image_id: int, *, second_object: bool = False) -> dict[str, Any]:
    objects = [
        {
            "bbox_2d": [10, 20, 30, 40],
            "desc": "cat",
            "category_id": 17,
            "category_name": "cat",
            "coco_ann_id": image_id * 100 + 1,
        }
    ]
    if second_object:
        objects.append(
            {
                "bbox_2d": [50, 60, 70, 80],
                "desc": "dog",
                "category_id": 18,
                "category_name": "dog",
                "coco_ann_id": image_id * 100 + 2,
            }
        )
    return {
        "images": [f"../rescale_32_1024_bbox/images/{split}2017/{image_id:012d}.jpg"],
        "objects": objects,
        "width": 640,
        "height": 480,
        "image_id": image_id,
        "file_name": f"images/{split}2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": split},
    }


@pytest.fixture
def project(tmp_path: Path) -> tuple[WorkingDatasetStore, BootstrapSpec, Path]:
    source_dir = (
        tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_len12000"
    )
    image_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox" / "images"
    image_split = image_root / "train2017"
    source_dir.mkdir(parents=True)
    image_split.mkdir(parents=True)
    for image_id in (1, 2, 3):
        (image_split / f"{image_id:012d}.jpg").write_bytes(f"image-{image_id}".encode())
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        "".join(
            canonical_json(_row("train", image_id, second_object=image_id == 1)) + "\n"
            for image_id in (1, 2, 3)
        ),
        encoding="utf-8",
    )
    spec = BootstrapSpec(
        split="train",
        source_path=source,
        runtime_root=tmp_path / "runtime",
        image_root=image_root,
        expected_source_sha256=sha256_file(source),
        project_id="project-train",
        storage_id="storage-train",
        adapter_version="adapter-v1",
        vendor_revision="label-studio-rev",
        registry_fingerprint=COCO80_REGISTRY.fingerprint,
        label_config_fingerprint="label-config-v1",
    )
    annotation_verifier = AcceptingAnnotationVerifier()
    inference_resolver = DictInferenceReceiptResolver()
    result = WorkingDatasetStore.bootstrap(
        spec,
        annotation_verifier=annotation_verifier,
        inference_receipt_resolver=inference_resolver,
    )
    return result.store, spec, source


def _regions(
    store: WorkingDatasetStore,
    image_id: int = 1,
    *,
    include_new: bool = True,
) -> list[dict[str, Any]]:
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
    regions[0]["bbox_2d"] = [11, 21, 31, 41]
    if include_new:
        regions.append(
            {
                "region_key": "drawn:new-1",
                "bbox_2d": [100, 110, 120, 130],
                "desc": "dog",
                "category_name": "dog",
                "category_id": 18,
                "creation_ordinal": 4,
                "metadata": {"visual_color": "red", "human_note": "new box"},
            }
        )
    return regions


def _request(
    store: WorkingDatasetStore,
    *,
    commit_id: str = "commit-1",
    regions: list[dict[str, Any]] | None = None,
    image_id: int = 1,
) -> CommitRequest:
    restored = store.restore_draft(image_id)
    regions = _regions(store, image_id, include_new=True) if regions is None else regions
    projection_hash = semantic_hash(regions)
    receipt = DraftSaveReceipt(
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision=7,
        semantic_hash=projection_hash,
    )
    return CommitRequest(
        commit_id=commit_id,
        split="train",
        image_id=image_id,
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision=7,
        semantic_hash=projection_hash,
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
        draft_save=receipt,
        inference_receipts=(),
    )


class CrashAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.seen: list[str] = []

    def __call__(self, boundary: str) -> None:
        self.seen.append(boundary)
        if boundary == self.boundary:
            raise InjectedCrash(boundary)


def _reopen(store: WorkingDatasetStore) -> WorkingDatasetStore:
    return WorkingDatasetStore(
        store.split_dir,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
    )


def _batch_request(
    store: WorkingDatasetStore,
    *,
    batch_id: str = "batch-1",
    image_ids: tuple[int, ...] = (1, 2),
    reverse_members: bool = False,
) -> Any:
    members = []
    for image_id in image_ids:
        regions = _regions(store, image_id)
        regions[-1]["region_key"] = f"drawn:batch-{image_id}"
        request = _request(
            store,
            commit_id=f"{batch_id}:member:{image_id}",
            image_id=image_id,
            regions=regions,
        )
        members.append(
            store_module.BatchMember(source_row_index=image_id - 1, request=request)
        )
    if reverse_members:
        members.reverse()
    return store_module.BatchRequest(
        batch_id=batch_id,
        split="train",
        base_generation=store.restore_draft(1).generation,
        members=tuple(members),
    )


def test_batch_enqueue_is_durable_immutable_and_does_not_publish(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store, reverse_members=True)
    working_before = store.working_path.read_bytes()
    manifest_before = store.manifest_path.read_bytes()

    receipt = store.enqueue_batch(request)

    assert receipt.batch_id == "batch-1"
    assert receipt.status is store_module.BatchStatus.QUEUED
    assert receipt.member_count == 2
    assert len(receipt.payload_hash) == 64
    assert store.working_path.read_bytes() == working_before
    assert store.manifest_path.read_bytes() == manifest_before
    queue_before_mutation = store.queue_path.read_bytes()
    assert queue_before_mutation.endswith(b"\n")
    queued = json.loads(queue_before_mutation)
    assert [member["source_row_index"] for member in queued["payload"]["members"]] == [0, 1]

    request.members[0].request.regions[0]["bbox_2d"][0] = 777
    assert store.queue_path.read_bytes() == queue_before_mutation


def test_batch_enqueue_exact_retry_is_byte_stable_and_payload_drift_conflicts(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store)
    first = store.enqueue_batch(request)
    queue_bytes = store.queue_path.read_bytes()

    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)
    verifier.accept = False
    assert store.enqueue_batch(copy.deepcopy(request)) == first
    assert store.queue_path.read_bytes() == queue_bytes

    changed = _batch_request(store, batch_id="batch-1", image_ids=(1, 3))
    with pytest.raises(CommitConflictError, match="batch id"):
        store.enqueue_batch(changed)
    assert store.queue_path.read_bytes() == queue_bytes


def test_batch_enqueue_returns_existing_active_batch_identity(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    active = store.enqueue_batch(_batch_request(store, batch_id="batch-active"))

    returned = store.enqueue_batch(
        _batch_request(store, batch_id="batch-other", image_ids=(3,))
    )

    assert returned == active
    assert store.get_batch_status("batch-other").status is store_module.BatchStatus.NOT_FOUND
    assert store.queue_path.read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize(
    "claimed,expected_status",
    [
        (False, store_module.BatchStatus.QUEUED),
        (True, store_module.BatchStatus.RUNNING),
    ],
)
def test_batch_enqueue_retries_return_queue_identity_while_worker_locked(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    claimed: bool,
    expected_status: Any,
) -> None:
    store, _, _ = project
    peer = _reopen(store)
    request = _batch_request(store, batch_id="batch-active", image_ids=(1,))
    other = _batch_request(peer, batch_id="batch-other", image_ids=(2,))
    accepted = store.enqueue_batch(request)
    if claimed:
        with store._exclusive_queue_lock():
            store._append_queue_record(
                {
                    "kind": "claim",
                    "batch_id": accepted.batch_id,
                    "payload_hash": accepted.payload_hash,
                },
                "claim",
            )
    queue_bytes = store.queue_path.read_bytes()

    with store._exclusive_lock():
        exact = peer.enqueue_batch(copy.deepcopy(request))
        active = peer.enqueue_batch(other)

    assert exact.batch_id == accepted.batch_id
    assert exact.payload_hash == accepted.payload_hash
    assert exact.status is expected_status
    assert active == exact
    assert store.queue_path.read_bytes() == queue_bytes


def test_batch_enqueue_terminal_retry_is_reconciling_while_worker_locked(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store, batch_id="batch-finished", image_ids=(1,))
    peer = _reopen(store)
    accepted = store.enqueue_batch(request)
    result = store.process_next_batch()
    assert result is not None
    assert result.status is store_module.BatchStatus.SUCCEEDED
    queue_bytes = store.queue_path.read_bytes()

    with store._exclusive_lock():
        retry = peer.enqueue_batch(copy.deepcopy(request))

    assert retry.batch_id == accepted.batch_id
    assert retry.payload_hash == accepted.payload_hash
    assert retry.status is store_module.BatchStatus.RECONCILING
    assert store.queue_path.read_bytes() == queue_bytes


def test_batch_enqueue_retries_reconcile_unlocked_journal_terminal_gap(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    worker, _, _ = project
    peer = _reopen(worker)
    request = _batch_request(worker, batch_id="batch-interrupted", image_ids=(1,))
    other = _batch_request(peer, batch_id="batch-other", image_ids=(2,))
    accepted = worker.enqueue_batch(request)
    worker._fault_injector = CrashAt("batch_terminal_journal_fsynced")

    with pytest.raises(InjectedCrash):
        worker.process_next_batch()
    queue_bytes = worker.queue_path.read_bytes()

    exact = peer.enqueue_batch(copy.deepcopy(request))
    active = peer.enqueue_batch(other)

    assert exact.batch_id == accepted.batch_id
    assert exact.payload_hash == accepted.payload_hash
    assert exact.status is store_module.BatchStatus.RECONCILING
    assert active == exact
    assert worker.queue_path.read_bytes() == queue_bytes


def test_legacy_commit_cannot_bypass_an_active_batch(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    working_before = store.working_path.read_bytes()

    with pytest.raises(StoreBusyError, match="active batch"):
        store.commit(_request(store, commit_id="legacy-during-batch", image_id=2))

    assert store.working_path.read_bytes() == working_before


@pytest.mark.parametrize(
    "members,error",
    [
        (((0, 2),), "source row index mismatch"),
        (((3, 3),), "source row index out of range"),
        (((0, 1), (0, 1)), "duplicate source row index"),
    ],
)
def test_batch_enqueue_verifies_immutable_source_row_index(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    members: tuple[tuple[int, int], ...],
    error: str,
) -> None:
    store, _, _ = project
    batch_members = []
    for ordinal, (source_row_index, image_id) in enumerate(members):
        batch_members.append(
            store_module.BatchMember(
                source_row_index=source_row_index,
                request=_request(
                    store,
                    commit_id=f"invalid-index:{ordinal}",
                    image_id=image_id,
                ),
            )
        )
    request = store_module.BatchRequest(
        batch_id="invalid-index-batch",
        split="train",
        base_generation=0,
        members=tuple(batch_members),
    )

    with pytest.raises(ValidationError, match=error):
        store.enqueue_batch(request)
    assert store.queue_path.read_bytes() == b""


def test_process_batch_publishes_every_member_once_in_source_order(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    receipt = store.enqueue_batch(_batch_request(store, reverse_members=True))

    result = store.process_next_batch()

    assert result is not None
    assert result.batch_id == receipt.batch_id
    assert result.status is store_module.BatchStatus.SUCCEEDED
    assert result.generation == 1
    assert [member.image_id for member in result.members] == [1, 2]
    assert [
        member.region_id_mapping[f"drawn:batch-{member.image_id}"]
        for member in result.members
    ] == [-1, -2]
    rows = [json.loads(line) for line in store.working_path.read_text().splitlines()]
    assert [row["image_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["objects"][0]["bbox_2d"] == [11, 21, 31, 41]
    assert rows[1]["objects"][0]["bbox_2d"] == [11, 21, 31, 41]
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["generation"] == 1
    assert manifest["working_line_count"] == 3
    assert manifest["working_sha256"] == result.working_sha256
    assert store.get_batch_status(receipt.batch_id).status is store_module.BatchStatus.SUCCEEDED
    assert store.get_batch_result(receipt.batch_id) == result
    records = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    assert [record["kind"] for record in records].count("batch_prepared") == 1
    assert [record["kind"] for record in records].count("batch_terminal") == 1
    assert [record["kind"] for record in records].count("reservation") == 2


def test_batch_freshness_is_per_row_not_global_generation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    worker_store = _reopen(store)
    old_request = _request(
        worker_store, commit_id="old-generation-member", image_id=1
    )
    store.commit(
        _request(
            store,
            commit_id="unrelated-legacy",
            image_id=3,
            regions=_regions(store, 3, include_new=False),
        )
    )
    batch = store_module.BatchRequest(
        batch_id="old-generation-batch",
        split="train",
        base_generation=old_request.observed_generation,
        members=(store_module.BatchMember(source_row_index=0, request=old_request),),
    )

    worker_store.enqueue_batch(batch)
    result = worker_store.process_next_batch()

    assert result is not None
    assert result.status is store_module.BatchStatus.SUCCEEDED
    assert result.generation == 2


def test_stale_batch_member_fails_all_members_without_replacement(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    old_one = _request(store, commit_id="stale-one", image_id=1)
    old_two = _request(store, commit_id="stale-two", image_id=2)
    store.commit(_request(store, commit_id="advance-row-one", image_id=1))
    batch = store_module.BatchRequest(
        batch_id="stale-batch",
        split="train",
        base_generation=0,
        members=(
            store_module.BatchMember(source_row_index=0, request=old_one),
            store_module.BatchMember(source_row_index=1, request=old_two),
        ),
    )
    store.enqueue_batch(batch)
    working_before = store.working_path.read_bytes()
    manifest_before = store.manifest_path.read_bytes()

    result = store.process_next_batch()

    assert result is not None
    assert result.status is store_module.BatchStatus.FAILED
    assert "base row hash changed" in (result.error or "")
    assert store.working_path.read_bytes() == working_before
    assert store.manifest_path.read_bytes() == manifest_before
    assert store.get_batch_status("stale-batch").status is store_module.BatchStatus.FAILED


def test_sequential_batches_preserve_prior_rows_and_use_frozen_snapshot(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first = _batch_request(store, batch_id="first", image_ids=(1,))
    store.enqueue_batch(first)
    first.members[0].request.regions[0]["bbox_2d"] = [200, 210, 220, 230]
    first_result = store.process_next_batch()
    assert first_result is not None and first_result.status is store_module.BatchStatus.SUCCEEDED
    row_one_after_first = store.working_path.read_bytes().splitlines(keepends=True)[0]
    assert json.loads(row_one_after_first)["objects"][0]["bbox_2d"] == [11, 21, 31, 41]

    store.enqueue_batch(_batch_request(store, batch_id="second", image_ids=(2,)))
    second_result = store.process_next_batch()

    assert second_result is not None and second_result.status is store_module.BatchStatus.SUCCEEDED
    assert store.working_path.read_bytes().splitlines(keepends=True)[0] == row_one_after_first


def test_batch_worker_does_not_use_legacy_prescan_or_temp_rehash(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy whole-file helper was used")

    monkeypatch.setattr(store, "_find_row", forbidden)
    monkeypatch.setattr(store, "_candidate_working_hash", forbidden)
    monkeypatch.setattr(store, "_rewrite_working", forbidden)
    monkeypatch.setattr(store, "_iter_rows", forbidden)
    monkeypatch.setattr(store_module, "sha256_file", forbidden)

    result = store.process_next_batch()

    assert result is not None
    assert result.status is store_module.BatchStatus.SUCCEEDED


def test_failed_batch_reservation_is_reused_but_never_given_to_another_key(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, batch_id="will-fail", image_ids=(1,)))

    def fail_after_reservation(boundary: str) -> None:
        if boundary == "reservation_journal_fsynced":
            raise ValidationError("injected failure after durable reservation")

    store._fault_injector = fail_after_reservation
    failed = store.process_next_batch()
    assert failed is not None and failed.status is store_module.BatchStatus.FAILED
    store._fault_injector = None
    reservations = [
        json.loads(line)
        for line in store.journal_path.read_text().splitlines()
        if json.loads(line)["kind"] == "reservation"
    ]
    assert [(record["stable_region_key"], record["coco_ann_id"]) for record in reservations] == [
        ("drawn:batch-1", -1)
    ]

    corrected = _batch_request(store, batch_id="corrected", image_ids=(1,))
    store.enqueue_batch(corrected)
    corrected_result = store.process_next_batch()
    assert corrected_result is not None
    assert corrected_result.members[0].region_id_mapping["drawn:batch-1"] == -1
    assert sum(
        json.loads(line)["kind"] == "reservation"
        for line in store.journal_path.read_text().splitlines()
    ) == 1

    other = _batch_request(store, batch_id="other-key", image_ids=(2,))
    other.members[0].request.regions[-1]["region_key"] = "drawn:other-key"
    regions = other.members[0].request.regions
    other_member = _request(
        store,
        commit_id="other-key:member:2",
        image_id=2,
        regions=regions,
    )
    other = replace(
        other,
        members=(store_module.BatchMember(source_row_index=1, request=other_member),),
        base_generation=store.restore_draft(2).generation,
    )
    store.enqueue_batch(other)
    other_result = store.process_next_batch()
    assert other_result is not None
    assert other_result.members[0].region_id_mapping["drawn:other-key"] == -2


@pytest.mark.parametrize(
    "boundary,expected_status,expected_generation",
    [
        ("batch_prepared_journal_fsynced", store_module.BatchStatus.FAILED, 0),
        ("batch_working_replaced", store_module.BatchStatus.SUCCEEDED, 1),
        ("batch_terminal_journal_fsynced", store_module.BatchStatus.SUCCEEDED, 1),
    ],
)
def test_batch_reopen_reconciles_transaction_and_queue_projection(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    boundary: str,
    expected_status: Any,
    expected_generation: int,
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    crash = CrashAt(boundary)
    store._fault_injector = crash

    with pytest.raises(InjectedCrash):
        store.process_next_batch()
    with pytest.raises(RecoveryError, match="requires recovery"):
        store.restore_draft(1)

    reopened = _reopen(store)
    status = reopened.get_batch_status("batch-1")
    assert status.status is expected_status
    assert status.generation == expected_generation
    assert json.loads(reopened.manifest_path.read_text())["generation"] == expected_generation
    assert any(
        json.loads(line).get("kind") == "queue_terminal"
        for line in reopened.queue_path.read_text().splitlines()
    )


@pytest.mark.parametrize(
    "boundary,terminal_projection_complete",
    [
        ("batch_prepared_journal_fsynced", False),
        ("batch_working_replaced", False),
        ("batch_working_directory_fsynced", False),
        ("manifest_temp_fsynced", False),
        ("manifest_replaced", False),
        ("batch_terminal_journal_fsynced", False),
        ("queue_terminal_queue_fsynced", True),
    ],
)
def test_preopened_peer_never_crosses_batch_publication_authorities(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    boundary: str,
    terminal_projection_complete: bool,
) -> None:
    worker, _, _ = project
    peer = _reopen(worker)
    next_request = _batch_request(peer, batch_id="batch-next", image_ids=(2,))
    worker.enqueue_batch(_batch_request(worker, image_ids=(1,)))
    worker._fault_injector = CrashAt(boundary)

    with pytest.raises(InjectedCrash):
        worker.process_next_batch()

    if terminal_projection_complete:
        restored = peer.restore_draft(1)
        assert restored.generation == 1
        assert restored.row["objects"][0]["bbox_2d"] == [11, 21, 31, 41]
        assert list(peer.iter_task_seeds())[0].annotations[0]["regions"][0][
            "bbox_2d"
        ] == [11, 21, 31, 41]
        assert peer.get_batch_status("batch-1").status is store_module.BatchStatus.SUCCEEDED
        accepted = peer.enqueue_batch(next_request)
        assert accepted.batch_id == "batch-next"
        assert accepted.status is store_module.BatchStatus.QUEUED
    else:
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.restore_draft(1)
        with pytest.raises(RecoveryError, match="requires recovery"):
            list(peer.iter_task_seeds())
        assert peer.get_batch_status("batch-1").status is store_module.BatchStatus.RECONCILING
        receipt = peer.enqueue_batch(next_request)
        assert receipt.batch_id == "batch-1"
        assert receipt.status is store_module.BatchStatus.RECONCILING
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.get_batch_status("batch-next")
        assert not any(
            json.loads(line).get("batch_id") == "batch-next"
            for line in peer.queue_path.read_text().splitlines()
        )


@pytest.mark.parametrize(
    "boundary,terminal_complete",
    [
        ("working_replaced", False),
        ("manifest_replaced", False),
        ("terminal_journal_fsynced", True),
    ],
)
def test_preopened_peer_never_crosses_legacy_publication_authorities(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    boundary: str,
    terminal_complete: bool,
) -> None:
    worker, _, _ = project
    peer = _reopen(worker)
    interrupted = _request(worker, commit_id="legacy-interrupted", image_id=1)
    later_legacy = _request(peer, commit_id="legacy-later", image_id=2)
    later_batch = _batch_request(peer, batch_id="batch-after-legacy", image_ids=(2,))
    worker._fault_injector = CrashAt(boundary)

    with pytest.raises(InjectedCrash):
        worker.commit(interrupted)

    if terminal_complete:
        restored = peer.restore_draft(1)
        assert restored.generation == 1
        assert restored.row["objects"][0]["bbox_2d"] == [11, 21, 31, 41]
        assert list(peer.iter_task_seeds())[0].annotations[0]["regions"][0][
            "bbox_2d"
        ] == [11, 21, 31, 41]
        assert peer.status("legacy-interrupted") is CommitStatus.COMMITTED
        assert peer.result("legacy-interrupted").generation == 1
        fresh_later_legacy = _request(
            peer, commit_id="legacy-later-fresh", image_id=2
        )
        assert peer.commit(fresh_later_legacy).generation == 2
    else:
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.restore_draft(1)
        with pytest.raises(RecoveryError, match="requires recovery"):
            list(peer.iter_task_seeds())
        assert peer.status("legacy-interrupted") is CommitStatus.OUTCOME_UNKNOWN
        with pytest.raises(CommitOutcomeUnknown):
            peer.result("legacy-interrupted")
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.commit(later_legacy)
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.enqueue_batch(later_batch)
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.get_batch_status("batch-after-legacy")
        assert peer.queue_path.read_bytes() == b""

        peer.recover()
        assert peer.restore_draft(1).generation == 1
        assert peer.status("legacy-interrupted") is CommitStatus.COMMITTED


@pytest.mark.parametrize(
    "transaction,boundary",
    [
        ("legacy", "working_replaced"),
        ("legacy", "manifest_replaced"),
        ("batch", "batch_working_replaced"),
        ("batch", "batch_terminal_journal_fsynced"),
    ],
)
def test_committed_generation_guard_blocks_unreconciled_publication(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    transaction: str,
    boundary: str,
) -> None:
    worker, _, _ = project
    peer = _reopen(worker)
    worker._fault_injector = CrashAt(boundary)

    with pytest.raises(InjectedCrash):
        if transaction == "legacy":
            worker.commit(
                _request(worker, commit_id="guard-interrupted", image_id=1)
            )
        else:
            worker.enqueue_batch(
                _batch_request(worker, batch_id="guard-interrupted", image_ids=(1,))
            )
            worker.process_next_batch()

    with pytest.raises(RecoveryError, match="requires recovery"):
        with peer.committed_generation_guard():
            pytest.fail("guard yielded an unreconciled generation")

    peer.recover()
    with peer.committed_generation_guard():
        assert json.loads(peer.manifest_path.read_text())["generation"] == 1


def test_committed_generation_guard_is_exclusive_without_hashing_working(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = project
    peer = _reopen(store)

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("committed generation guard hashed the working file")

    monkeypatch.setattr(store_module, "sha256_file", forbidden)
    with store.committed_generation_guard():
        assert json.loads(store.manifest_path.read_text())["generation"] == 0
        with pytest.raises(StoreBusyError, match="reconciling"):
            peer.restore_draft(1)


def test_lost_enqueue_response_reopens_as_same_queued_batch(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    crash = CrashAt("enqueue_queue_fsynced")
    store._fault_injector = crash

    with pytest.raises(InjectedCrash):
        store.enqueue_batch(_batch_request(store, image_ids=(1,)))

    reopened = _reopen(store)
    assert reopened.get_batch_status("batch-1").status is store_module.BatchStatus.QUEUED
    result = reopened.process_next_batch()
    assert result is not None and result.status is store_module.BatchStatus.SUCCEEDED


def test_bootstrap_persists_verified_zero_based_task_index(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    assert [seed.source_row_index for seed in store.iter_task_seeds()] == [0, 1, 2]
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["working_line_count"] == 3
    assert manifest["task_index_sha256"] == sha256_file(store.task_index_path)

    task_index = json.loads(store.task_index_path.read_text())
    task_index["entries"][1]["source_row_index"] = 0
    store.task_index_path.write_text(canonical_json(task_index) + "\n", encoding="utf-8")
    manifest["task_index_sha256"] = sha256_file(store.task_index_path)
    store.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ManifestDriftError, match="task_index"):
        _reopen(store)


def test_reopen_truncates_only_an_unambiguous_torn_queue_tail(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    durable = store.queue_path.read_bytes()
    with store.queue_path.open("ab") as handle:
        handle.write(b'{"kind":"claim"')
        handle.flush()

    reopened = _reopen(store)

    assert reopened.queue_path.read_bytes() == durable
    assert reopened.get_batch_status("batch-1").status is store_module.BatchStatus.QUEUED


def test_reopen_removes_orphan_batch_candidate_and_keeps_queue_dispatchable(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    orphan = store.split_dir / ".working.batch.orphan"
    orphan.write_bytes(b"partial candidate")

    reopened = _reopen(store)

    assert not orphan.exists()
    result = reopened.process_next_batch()
    assert result is not None and result.status is store_module.BatchStatus.SUCCEEDED


def test_batch_candidate_rejects_untouched_input_drift_before_replacement(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    lines = store.working_path.read_bytes().splitlines(keepends=True)
    lines[2] = lines[2].rstrip(b"\n") + b" \n"
    drifted = b"".join(lines)
    store.working_path.write_bytes(drifted)

    with pytest.raises(RecoveryError, match="input hash attestation"):
        store.process_next_batch()

    assert store.working_path.read_bytes() == drifted
    assert not any(
        json.loads(line)["kind"] in {"batch_prepared", "batch_terminal"}
        for line in store.journal_path.read_text().splitlines()
    )
    with pytest.raises(RecoveryError, match="published manifest"):
        _reopen(store)


def test_supported_reads_and_new_admission_fail_closed_behind_transaction_barrier(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store, image_ids=(1,))
    with store.lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(StoreBusyError, match="reconciling"):
            store.restore_draft(1)
        with pytest.raises(StoreBusyError, match="reconciling"):
            store.get_batch_status("missing")
        with pytest.raises(StoreBusyError, match="reconciling"):
            store.enqueue_batch(request)
    assert store.queue_path.read_bytes() == b""


def test_bootstrap_is_idempotent_source_safe_and_yields_stable_single_annotations(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, spec, source = project
    source_before = source.read_bytes()
    repeated = WorkingDatasetStore.bootstrap(
        spec,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
    )

    assert repeated.created is False
    assert repeated.task_count == 3
    assert source.read_bytes() == source_before
    assert store.working_path.resolve() != source.resolve()
    assert (store.split_dir / "images").is_symlink()
    assert (store.split_dir / "images").resolve() == spec.image_root.resolve()
    seeds = list(repeated.store.iter_task_seeds())
    assert [seed.task_id for seed in seeds] == ["train:1", "train:2", "train:3"]
    assert all(len(seed.annotations) == 1 and seed.predictions == () for seed in seeds)
    assert all(seed.image_locator.startswith("images/train2017/") for seed in seeds)
    assert all(
        region["region_key"].startswith("train:coco:")
        for seed in seeds
        for region in seed.annotations[0]["regions"]
    )
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["document_root"] == str(spec.image_root.resolve())
    assert manifest["storage"] == {
        "storage_id": "storage-train",
        "subdir": "train2017",
        "project_bound": True,
    }
    assert manifest["task_policy"]["authoritative_annotations"] == 1
    assert manifest["task_policy"]["alternate_annotations_enabled"] is False


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("vendor_revision", "other"),
        ("registry_fingerprint", "contiguous-evaluator-map"),
        ("label_config_fingerprint", "drifted"),
        ("project_id", "different-project"),
    ],
)
def test_bootstrap_fails_closed_on_manifest_drift(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], field: str, replacement: str
) -> None:
    store, spec, _ = project
    manifest = json.loads(store.manifest_path.read_text())
    manifest[field] = replacement
    store.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ManifestDriftError, match=field):
        WorkingDatasetStore.bootstrap(
            spec,
            annotation_verifier=store.annotation_verifier,
            inference_receipt_resolver=store.inference_receipt_resolver,
        )


def test_bootstrap_rejects_non_selected_path_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "train.norm.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    spec = BootstrapSpec(
        split="train",
        source_path=source,
        runtime_root=tmp_path / "runtime",
        image_root=tmp_path,
        expected_source_sha256=sha256_file(source),
        project_id="p",
        storage_id="s",
        adapter_version="a",
        vendor_revision="v",
        registry_fingerprint="r",
        label_config_fingerprint="l",
    )
    with pytest.raises(ManifestDriftError, match="source_path"):
        WorkingDatasetStore.bootstrap(
            spec,
            annotation_verifier=AcceptingAnnotationVerifier(),
            inference_receipt_resolver=DictInferenceReceiptResolver(),
        )


def test_commit_preserves_positive_ids_allocates_negative_once_and_rehydrates(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _request(store)
    result = store.commit(request)

    assert result.status is CommitStatus.COMMITTED
    assert result.generation == 1
    assert result.region_id_mapping["train:coco:101"] == 101
    assert result.region_id_mapping["train:coco:102"] == 102
    assert result.region_id_mapping["drawn:new-1"] == -1
    assert [obj["coco_ann_id"] for obj in result.committed_row["objects"]] == [101, 102, -1]
    assert result.committed_row["objects"][-1]["metadata"] == {"human_note": "new box"}
    restored = _reopen(store).restore_draft(1)
    assert restored.region_id_mapping["drawn:new-1"] == -1
    assert restored.row_hash == result.row_hash


def test_retry_is_idempotent_and_conflicting_hash_is_rejected(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _request(store)
    first = store.commit(request)
    journal_before = store.journal_path.read_bytes()
    second = store.commit(request)
    assert second == first
    assert store.journal_path.read_bytes() == journal_before

    changed = list(request.regions)
    changed[0] = {**changed[0], "bbox_2d": [12, 22, 32, 42]}
    conflict = _request(store, commit_id=request.commit_id, regions=changed)
    with pytest.raises(CommitConflictError):
        store.commit(conflict)


def test_committed_retry_survives_later_deletion_of_its_new_object_byte_exact(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first_request = _request(store, commit_id="first")
    first_result = store.commit(first_request)
    assert first_result.region_id_mapping["drawn:new-1"] == -1

    retained = [
        region
        for region in _regions(store, include_new=False)
        if region["region_key"] != "drawn:new-1"
    ]
    store.commit(_request(store, commit_id="delete-new", regions=retained))
    assert store.status(first_request.commit_id) is CommitStatus.COMMITTED
    before_retry = {
        path: path.read_bytes()
        for path in (store.journal_path, store.working_path, store.manifest_path)
    }

    retried = store.commit(first_request)

    assert retried == first_result
    assert {
        path: path.read_bytes()
        for path in (store.journal_path, store.working_path, store.manifest_path)
    } == before_retry

    changed_regions = copy.deepcopy(first_request.regions)
    changed_regions[-1]["metadata"]["human_note"] = "changed after deletion"
    changed_retry = replace(first_request, regions=changed_regions)
    with pytest.raises(CommitConflictError, match="immutable request identity"):
        store.commit(changed_retry)
    assert {
        path: path.read_bytes()
        for path in (store.journal_path, store.working_path, store.manifest_path)
    } == before_retry


def test_retry_rejects_changed_retained_metadata_with_same_semantic_hash(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _request(store)
    store.commit(request)
    changed_regions = copy.deepcopy(request.regions)
    changed_regions[-1]["metadata"]["human_note"] = "changed after lost response"
    assert semantic_hash(changed_regions) == request.semantic_hash

    changed = replace(request, regions=changed_regions)
    with pytest.raises(CommitConflictError, match="immutable request identity"):
        store.commit(changed)


def test_retry_rejects_changed_order_seed_with_same_semantic_hash(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    regions = _regions(store, include_new=False)
    regions.extend(
        [
            {
                "region_key": "drawn:tie-a",
                "bbox_2d": [200, 200, 240, 240],
                "category_name": "cat",
                "category_id": 17,
                "creation_ordinal": 1,
            },
            {
                "region_key": "drawn:tie-b",
                "bbox_2d": [200, 200, 250, 250],
                "category_name": "dog",
                "category_id": 18,
                "creation_ordinal": 2,
            },
        ]
    )
    request = _request(store, regions=regions)
    first = store.commit(request)
    assert [obj["coco_ann_id"] for obj in first.committed_row["objects"]][-2:] == [-1, -2]

    reordered = copy.deepcopy(regions)
    reordered[-2]["creation_ordinal"] = 2
    reordered[-1]["creation_ordinal"] = 1
    assert semantic_hash(reordered) == request.semantic_hash
    changed = replace(request, regions=reordered)
    with pytest.raises(CommitConflictError, match="immutable request identity"):
        store.commit(changed)


def test_draft_save_is_not_commit_and_empty_commit_is_rejected_without_mutation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    before = store.working_path.read_bytes()
    empty_hash = semantic_hash([])
    receipt = DraftSaveReceipt(
        project_id="project-train",
        task_id="train:1",
        annotation_id="annotation-1",
        draft_id="draft-1",
        annotation_revision=8,
        semantic_hash=empty_hash,
    )
    # Holding a durable empty Draft receipt has no working-data side effect.
    assert store.working_path.read_bytes() == before
    restored = store.restore_draft(1)
    request = CommitRequest(
        commit_id="empty",
        split="train",
        image_id=1,
        project_id="project-train",
        task_id="train:1",
        annotation_id="annotation-1",
        draft_id="draft-1",
        annotation_revision=8,
        semantic_hash=empty_hash,
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=[],
        draft_save=receipt,
    )
    with pytest.raises(ValidationError, match="empty Draft"):
        store.commit(request)
    assert store.working_path.read_bytes() == before
    assert json.loads(store.manifest_path.read_text())["generation"] == 0


def test_commit_requires_exact_durable_draft_save_receipt(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _request(store)
    mismatched = CommitRequest(
        **{
            **request.__dict__,
            "draft_save": DraftSaveReceipt(
                **{**request.draft_save.__dict__, "annotation_revision": 6}
            ),
        }
    )
    with pytest.raises(StaleCommitError, match="Draft-save receipt"):
        store.commit(mismatched)

    not_durable = CommitRequest(
        **{
            **request.__dict__,
            "draft_save": DraftSaveReceipt(
                **{**request.draft_save.__dict__, "durable": False}
            ),
        }
    )
    with pytest.raises(ValidationError, match="durable Draft-save"):
        store.commit(not_durable)


def test_commit_rewrites_once_preserves_row_order_and_untouched_bytes(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    before_lines = store.working_path.read_bytes().splitlines(keepends=True)
    before_inode = store.working_path.stat().st_ino
    store.commit(_request(store))
    after_lines = store.working_path.read_bytes().splitlines(keepends=True)

    assert store.working_path.stat().st_ino != before_inode
    assert [json.loads(line)["image_id"] for line in after_lines] == [1, 2, 3]
    assert after_lines[1:] == before_lines[1:]
    assert json.loads(store.manifest_path.read_text())["working_sha256"] == sha256_file(
        store.working_path
    )


def test_manifest_is_published_after_working_directory_durability(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    events: list[str] = []
    store._fault_injector = events.append
    store.commit(_request(store))
    assert events.index("working_replaced") < events.index("working_directory_fsynced")
    assert events.index("working_directory_fsynced") < events.index("manifest_temp_flushed")
    assert events.index("manifest_directory_fsynced") < events.index(
        "terminal_journal_flushed"
    )
    assert events[-1] == "before_response"


PRE_REPLACEMENT_CUTS = [
    "prepared_journal_flushed",
    "prepared_journal_fsynced",
    "working_temp_flushed",
    "working_temp_fsynced",
]
POST_REPLACEMENT_CUTS = [
    "working_replaced",
    "working_directory_fsynced",
    "manifest_temp_flushed",
    "manifest_temp_fsynced",
    "manifest_replaced",
    "manifest_directory_fsynced",
    "terminal_journal_flushed",
    "terminal_journal_fsynced",
    "before_response",
]


@pytest.mark.parametrize("boundary", PRE_REPLACEMENT_CUTS)
def test_startup_recovery_rolls_back_every_pre_replacement_cut(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], boundary: str
) -> None:
    store, _, _ = project
    request = _request(store)
    store._fault_injector = CrashAt(boundary)
    with pytest.raises(InjectedCrash):
        store.commit(request)

    recovered = _reopen(store)
    assert recovered.status(request.commit_id) is CommitStatus.ROLLED_BACK
    assert recovered.restore_draft(1).generation == 0
    with pytest.raises(CommitRolledBackError):
        recovered.result(request.commit_id)
    terminal_count = sum(
        json.loads(line)["kind"] == "terminal"
        for line in recovered.journal_path.read_text().splitlines()
    )
    recovered.recover()
    assert sum(
        json.loads(line)["kind"] == "terminal"
        for line in recovered.journal_path.read_text().splitlines()
    ) == terminal_count


@pytest.mark.parametrize("boundary", POST_REPLACEMENT_CUTS)
def test_startup_recovery_commits_every_post_replacement_cut_exactly_once(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], boundary: str
) -> None:
    store, _, _ = project
    request = _request(store)
    store._fault_injector = CrashAt(boundary)
    with pytest.raises(InjectedCrash):
        store.commit(request)

    unresolved = WorkingDatasetStore(
        store.split_dir,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
        recover=False,
    )
    if boundary not in {"terminal_journal_flushed", "terminal_journal_fsynced", "before_response"}:
        assert unresolved.status(request.commit_id) is CommitStatus.OUTCOME_UNKNOWN
        with pytest.raises(CommitOutcomeUnknown):
            unresolved.result(request.commit_id)

    recovered = _reopen(store)
    result = recovered.result(request.commit_id)
    assert result.status is CommitStatus.COMMITTED
    assert result.generation == 1
    assert recovered.restore_draft(1).region_id_mapping["drawn:new-1"] == -1
    terminal_count = sum(
        json.loads(line)["kind"] == "terminal"
        for line in recovered.journal_path.read_text().splitlines()
    )
    recovered.recover()
    assert sum(
        json.loads(line)["kind"] == "terminal"
        for line in recovered.journal_path.read_text().splitlines()
    ) == terminal_count == 1


def test_prepared_allocation_is_never_reused_after_rollback(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first = _request(store, commit_id="rolled-back")
    store._fault_injector = CrashAt("prepared_journal_fsynced")
    with pytest.raises(InjectedCrash):
        store.commit(first)
    recovered = _reopen(store)
    assert recovered.status(first.commit_id) is CommitStatus.ROLLED_BACK

    second = _request(recovered, commit_id="retry-new-id")
    result = recovered.commit(second)
    assert result.region_id_mapping["drawn:new-1"] == -1
    third_regions = _regions(recovered, include_new=False)
    third_regions.append(
        {
            "region_key": "drawn:new-2",
            "bbox_2d": [200, 210, 220, 230],
            "category_name": "cat",
            "category_id": 17,
        }
    )
    third = _request(recovered, commit_id="next-allocation", regions=third_regions)
    assert recovered.commit(third).region_id_mapping["drawn:new-2"] == -2


def test_committed_deletion_tombstones_identity(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    original = _regions(store, include_new=False)
    deleted_key = original[1]["region_key"]
    first = _request(store, commit_id="delete", regions=[original[0]])
    store.commit(first)

    redraw = _regions(store, include_new=False)
    redraw.append(
        {
            "region_key": deleted_key,
            "bbox_2d": [300, 310, 320, 330],
            "category_name": "dog",
            "category_id": 18,
        }
    )
    with pytest.raises(ValidationError, match="tombstoned"):
        store.commit(_request(store, commit_id="restore-tombstone", regions=redraw))


def test_equal_top_left_ties_preserve_prior_rank_then_creation_ordinal(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    existing = _regions(store, include_new=False)
    for region in existing:
        region["bbox_2d"] = [100, 100, 200, 200]
    regions = [
        {
            "region_key": "new-later",
            "bbox_2d": [100, 100, 150, 150],
            "category_name": "cat",
            "category_id": 17,
            "creation_ordinal": 20,
        },
        existing[1],
        {
            "region_key": "new-earlier",
            "bbox_2d": [100, 100, 160, 160],
            "category_name": "dog",
            "category_id": 18,
            "creation_ordinal": 10,
        },
        existing[0],
    ]
    result = store.commit(_request(store, regions=regions))
    ids = [obj["coco_ann_id"] for obj in result.committed_row["objects"]]
    assert ids[:2] == [101, 102]
    assert [result.region_id_mapping[key] for key in ("new-earlier", "new-later")] == [-2, -1]
    assert ids[2:] == [-2, -1]


def test_stale_generation_row_and_category_are_rejected_atomically(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _request(store)
    stale_generation = CommitRequest(**{**request.__dict__, "observed_generation": 99})
    with pytest.raises(StaleCommitError, match="generation"):
        store.commit(stale_generation)
    stale_row = CommitRequest(**{**request.__dict__, "base_row_hash": "0" * 64})
    with pytest.raises(StaleCommitError, match="base row"):
        store.commit(stale_row)

    invalid_regions = list(request.regions)
    invalid_regions[-1] = {**invalid_regions[-1], "category_id": 2}
    invalid = _request(store, regions=invalid_regions)
    before = store.working_path.read_bytes()
    with pytest.raises(Exception):
        store.commit(invalid)
    assert store.working_path.read_bytes() == before


def test_fail_fast_advisory_lock_rejects_overlapping_owner(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    with store.lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(StoreBusyError):
                store.commit(_request(store))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_recovery_fails_closed_on_journal_hash_disagreement(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.commit(_request(store))
    records = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    records[0]["after_row_hash"] = "0" * 64
    store.journal_path.write_text(
        "".join(canonical_json(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(RecoveryError, match="journal hash disagreement"):
        _reopen(store)


def test_recovery_fails_closed_on_unknown_working_hash(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store._fault_injector = CrashAt("prepared_journal_fsynced")
    with pytest.raises(InjectedCrash):
        store.commit(_request(store))
    with store.working_path.open("ab") as handle:
        handle.write(b"{}\n")
        handle.flush()
    with pytest.raises(RecoveryError, match="journal/hash disagreement"):
        _reopen(store)


def test_post_replacement_runtime_error_requires_outcome_reconciliation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _, _ = project
    request = _request(store)

    def fail_manifest(*args: Any, **kwargs: Any) -> None:
        raise OSError("manifest device error")

    monkeypatch.setattr(store, "_publish_manifest", fail_manifest)
    with pytest.raises(CommitOutcomeUnknown):
        store.commit(request)
    assert store.status(request.commit_id) is CommitStatus.OUTCOME_UNKNOWN
    recovered = _reopen(store)
    assert recovered.status(request.commit_id) is CommitStatus.COMMITTED


def test_manifest_and_working_hashes_are_stable_canonical_receipts(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    result = store.commit(_request(store))
    manifest = json.loads(store.manifest_path.read_text())
    prepared = json.loads(store.journal_path.read_text().splitlines()[0])
    assert manifest["working_sha256"] == hashlib.sha256(store.working_path.read_bytes()).hexdigest()
    assert prepared["candidate_manifest_hash"] == sha256_json(manifest)
    assert prepared["after_row_hash"] == result.row_hash


def _open_copied_store(path: Path, store: WorkingDatasetStore) -> WorkingDatasetStore:
    return WorkingDatasetStore(
        path,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
    )


def test_recovery_truncates_only_a_torn_final_prepared_frame_and_reopens_twice(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], tmp_path: Path
) -> None:
    store, _, _ = project
    store._fault_injector = CrashAt("prepared_journal_fsynced")
    with pytest.raises(InjectedCrash):
        store.commit(_request(store))
    frame = store.journal_path.read_bytes()
    assert frame.endswith(b"\n")

    for index, cut in enumerate((0, 1, len(frame) // 2, len(frame) - 1)):
        case = tmp_path / f"prepared-tail-{index}"
        shutil.copytree(store.split_dir, case, symlinks=True)
        (case / "journal.jsonl").write_bytes(frame[:cut])
        first = _open_copied_store(case, store)
        after_first = first.journal_path.read_bytes()
        assert after_first == b""
        assert first.restore_draft(1).generation == 0
        second = _open_copied_store(case, store)
        assert second.journal_path.read_bytes() == after_first
        assert second.restore_draft(1).generation == 0


def test_recovery_truncates_torn_terminal_then_finalizes_once_and_reopens_twice(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], tmp_path: Path
) -> None:
    store, _, _ = project
    request = _request(store)
    store.commit(request)
    prepared, terminal = store.journal_path.read_bytes().splitlines(keepends=True)
    assert prepared.endswith(b"\n") and terminal.endswith(b"\n")

    for index, cut in enumerate((0, 1, len(terminal) // 2, len(terminal) - 1)):
        case = tmp_path / f"terminal-tail-{index}"
        shutil.copytree(store.split_dir, case, symlinks=True)
        (case / "journal.jsonl").write_bytes(prepared + terminal[:cut])
        first = _open_copied_store(case, store)
        after_first = first.journal_path.read_bytes()
        assert first.status(request.commit_id) is CommitStatus.COMMITTED
        assert len(after_first.splitlines()) == 2
        second = _open_copied_store(case, store)
        assert second.journal_path.read_bytes() == after_first
        assert second.status(request.commit_id) is CommitStatus.COMMITTED


def test_recovery_fails_closed_on_corruption_before_a_torn_tail(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], tmp_path: Path
) -> None:
    store, _, _ = project
    store.commit(_request(store))
    prepared_raw, terminal_raw = store.journal_path.read_bytes().splitlines(keepends=True)
    prepared = json.loads(prepared_raw)
    prepared["semantic_hash"] = "0" * 64
    corrupted = (canonical_json(prepared) + "\n").encode() + terminal_raw[:9]
    case = tmp_path / "corrupt-before-tail"
    shutil.copytree(store.split_dir, case, symlinks=True)
    (case / "journal.jsonl").write_bytes(corrupted)

    with pytest.raises(RecoveryError, match="journal hash disagreement"):
        _open_copied_store(case, store)
    assert (case / "journal.jsonl").read_bytes() == corrupted


def test_same_region_key_on_different_tasks_has_independent_stable_mapping(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first = store.commit(_request(store, commit_id="task-1", image_id=1))
    second = store.commit(_request(store, commit_id="task-2", image_id=2))
    assert first.region_id_mapping["drawn:new-1"] == -1
    assert second.region_id_mapping["drawn:new-1"] == -2
    reopened = _reopen(store)
    assert reopened.restore_draft(1).region_id_mapping["drawn:new-1"] == -1
    assert reopened.restore_draft(2).region_id_mapping["drawn:new-1"] == -2


@pytest.mark.parametrize(
    "image_id,region_key,supplied_id",
    [
        (2, "train:coco:101", 101),
        (1, "train:coco:201", 201),
        (1, "forged-positive", 101),
        (1, "train:coco:999", None),
    ],
)
def test_positive_source_identity_cannot_be_forged_or_hijacked(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    image_id: int,
    region_key: str,
    supplied_id: int | None,
) -> None:
    store, _, _ = project
    regions = _regions(store, image_id, include_new=False)
    forged = {
        "region_key": region_key,
        "bbox_2d": [200, 210, 220, 230],
        "category_name": "cat",
        "category_id": 17,
    }
    if supplied_id is not None:
        forged["coco_ann_id"] = supplied_id
    regions.append(forged)
    with pytest.raises(ValidationError, match="source identity|positive source"):
        store.commit(_request(store, commit_id="forged", image_id=image_id, regions=regions))


def test_bootstrap_and_recovery_reject_split_wide_duplicate_coco_ann_id(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], tmp_path: Path
) -> None:
    store, spec, source = project
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[1]["objects"][0]["coco_ann_id"] = rows[0]["objects"][0]["coco_ann_id"]
    source.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    duplicate_spec = replace(
        spec,
        runtime_root=tmp_path / "duplicate-bootstrap",
        expected_source_sha256=sha256_file(source),
    )
    with pytest.raises(ValidationError, match="split-wide duplicate coco_ann_id"):
        WorkingDatasetStore.bootstrap(
            duplicate_spec,
            annotation_verifier=store.annotation_verifier,
            inference_receipt_resolver=store.inference_receipt_resolver,
        )

    working_rows = [json.loads(line) for line in store.working_path.read_text().splitlines()]
    working_rows[1]["objects"][0]["coco_ann_id"] = working_rows[0]["objects"][0][
        "coco_ann_id"
    ]
    store.working_path.write_text(
        "".join(canonical_json(row) + "\n" for row in working_rows), encoding="utf-8"
    )
    manifest = json.loads(store.manifest_path.read_text())
    manifest["working_sha256"] = sha256_file(store.working_path)
    store.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(RecoveryError, match="split-wide duplicate coco_ann_id"):
        _reopen(store)


def _matching_draft_receipt(request: CommitRequest) -> DraftSaveReceipt:
    return DraftSaveReceipt(
        project_id=request.project_id,
        task_id=request.task_id,
        annotation_id=request.annotation_id,
        draft_id=request.draft_id,
        annotation_revision=request.annotation_revision,
        semantic_hash=request.semantic_hash,
    )


def test_commit_id_retry_binds_every_immutable_request_identity_field(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    original = _request(store)
    store.commit(original)
    changed_regions = [dict(region) for region in original.regions]
    changed_regions[0]["bbox_2d"] = [12, 22, 32, 42]
    changed_hash = semantic_hash(changed_regions)
    candidates = [
        replace(original, split="val"),
        replace(original, image_id=2),
        replace(original, project_id="project-other"),
        replace(original, task_id="train:2"),
        replace(original, annotation_id="annotation-other"),
        replace(original, draft_id="draft-other"),
        replace(original, annotation_revision=8),
        replace(original, base_row_hash="0" * 64),
        replace(original, observed_generation=99),
        replace(original, inference_receipts=("receipt-forged",)),
        replace(original, regions=changed_regions, semantic_hash=changed_hash),
    ]
    for candidate in candidates:
        candidate = replace(candidate, draft_save=_matching_draft_receipt(candidate))
        with pytest.raises(CommitConflictError, match="immutable request identity"):
            store.commit(candidate)


def test_commit_requires_external_authoritative_annotation_attestation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)
    verifier.accept = False
    request = _request(store)
    with pytest.raises(StaleCommitError, match="authoritative annotation"):
        store.commit(request)
    assert verifier.calls[-1] == AuthoritativeDraftIdentity.from_request(request)
    assert store.journal_path.read_bytes() == b""


def _inference_regions(store: WorkingDatasetStore) -> list[dict[str, Any]]:
    regions = _regions(store, include_new=False)
    regions.append(
        {
            "region_key": "inferred:region-1",
            "bbox_2d": [100, 110, 120, 130],
            "category_name": "dog",
            "category_id": 18,
            "metadata": {
                "inference_origin": True,
                "receipt_id": "receipt-1",
                "request_id": "request-1",
                "result_id": "result-1",
            },
        }
    )
    return regions


def _valid_inference_link(**changes: Any) -> InferenceReceiptLink:
    values: dict[str, Any] = {
        "receipt_id": "receipt-1",
        "request_id": "request-1",
        "project_id": "project-train",
        "task_id": "train:1",
        "image_id": 1,
        "annotation_id": "annotation-1",
        "terminal_status": "accepted",
        "result_region_keys": {"result-1": "inferred:region-1"},
    }
    values.update(changes)
    return InferenceReceiptLink(**values)


def test_inference_origin_requires_authoritative_exact_receipt_linkage(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    resolver = store.inference_receipt_resolver
    assert isinstance(resolver, DictInferenceReceiptResolver)
    regions = _inference_regions(store)

    with pytest.raises(ValidationError, match="declared inference receipt"):
        store.commit(_request(store, commit_id="omitted", regions=regions))

    with pytest.raises(ValidationError, match="unknown inference receipt"):
        store.commit(
            replace(
                _request(store, commit_id="unknown", regions=regions),
                inference_receipts=("receipt-1",),
            )
        )

    for commit_id, link, error in (
        ("cross-task", _valid_inference_link(task_id="train:2"), "target mismatch"),
        ("cross-image", _valid_inference_link(image_id=2), "target mismatch"),
        (
            "cross-annotation",
            _valid_inference_link(annotation_id="annotation-2"),
            "target mismatch",
        ),
        (
            "wrong-region",
            _valid_inference_link(result_region_keys={"result-1": "other-region"}),
            "result linkage mismatch",
        ),
    ):
        resolver.links["receipt-1"] = link
        with pytest.raises(ValidationError, match=error):
            store.commit(
                replace(
                    _request(store, commit_id=commit_id, regions=regions),
                    inference_receipts=("receipt-1",),
                )
            )

    incomplete = [dict(region) for region in regions]
    incomplete[-1] = {
        **incomplete[-1],
        "metadata": {"inference_origin": True, "receipt_id": "receipt-1"},
    }
    with pytest.raises(ValidationError, match="complete inference linkage"):
        store.commit(_request(store, commit_id="incomplete", regions=incomplete))

    resolver.links["receipt-1"] = _valid_inference_link()
    request = replace(
        _request(store, commit_id="valid-inference", regions=regions),
        inference_receipts=("receipt-1",),
    )
    result = store.commit(request)
    assert result.region_id_mapping["inferred:region-1"] == -1
    prepared = json.loads(store.journal_path.read_text().splitlines()[0])
    assert prepared["inference_receipts"] == ["receipt-1"]


def test_human_only_commit_has_no_inference_receipt_or_resolver_call(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    resolver = store.inference_receipt_resolver
    assert isinstance(resolver, DictInferenceReceiptResolver)
    store.commit(_request(store))
    assert resolver.calls == []
    prepared = json.loads(store.journal_path.read_text().splitlines()[0])
    assert prepared["inference_receipts"] == []
