from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import shutil
import threading
from contextlib import contextmanager
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


class BatchReadingAnnotationVerifier:
    def __init__(self, *, barrier: threading.Barrier | None = None) -> None:
        self.store: WorkingDatasetStore | None = None
        self.barrier = barrier
        self.batch_calls: list[store_module.BatchRequest] = []
        self.fallback_calls: list[AuthoritativeDraftIdentity] = []

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        self.fallback_calls.append(identity)
        return True

    def verify_batch(self, request: store_module.BatchRequest) -> bool:
        self.batch_calls.append(request)
        if self.store is not None:
            image_ids = tuple(member.request.image_id for member in request.members)
            restored = self.store.restore_drafts(image_ids)
            assert [item.image_id for item in restored] == list(image_ids)
            for member in request.members:
                assert (
                    self.store.resolve_source_row_index(
                        split=request.split,
                        project_id=member.request.project_id,
                        task_id=member.request.task_id,
                        image_id=member.request.image_id,
                    )
                    == member.source_row_index
                )
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        return True


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
    source_dir = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_len12000"
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


def _bootstrap_store_with_images(
    tmp_path: Path,
    image_ids: tuple[int, ...],
    annotation_verifier: Any,
) -> WorkingDatasetStore:
    source_dir = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data" / "coco" / "rescale_32_1024_bbox" / "images"
    image_split = image_root / "train2017"
    source_dir.mkdir(parents=True)
    image_split.mkdir(parents=True)
    for image_id in image_ids:
        (image_split / f"{image_id:012d}.jpg").write_bytes(f"image-{image_id}".encode())
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        "".join(
            canonical_json(_row("train", image_id)) + "\n" for image_id in image_ids
        ),
        encoding="utf-8",
    )
    result = WorkingDatasetStore.bootstrap(
        BootstrapSpec(
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
        ),
        annotation_verifier=annotation_verifier,
        inference_receipt_resolver=DictInferenceReceiptResolver(),
    )
    return result.store


def test_public_state_snapshot_uses_manifest_authority_without_row_payloads(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, spec, _source = project

    snapshot = store.state_snapshot()

    assert snapshot.split == "train"
    assert snapshot.project_id == spec.project_id
    assert snapshot.generation == 0
    assert snapshot.task_count == 3
    assert not hasattr(snapshot, "working_path")


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
    regions = (
        _regions(store, image_id, include_new=True) if regions is None else regions
    )
    projection_hash = semantic_hash(regions)
    receipt = DraftSaveReceipt(
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision="annotation-v7",
        draft_updated_at="2026-07-15T00:00:07Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
    )
    return CommitRequest(
        commit_id=commit_id,
        split="train",
        image_id=image_id,
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision="annotation-v7",
        draft_updated_at="2026-07-15T00:00:07Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
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


class PauseAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, boundary: str) -> None:
        if boundary == self.boundary:
            self.entered.set()
            assert self.release.wait(3.0)


def _reopen(store: WorkingDatasetStore) -> WorkingDatasetStore:
    return WorkingDatasetStore(
        store.split_dir,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
    )


def _rewrite_queue_chain_after_envelope_tamper(
    store: WorkingDatasetStore,
    *,
    kind: str,
    operation: str,
) -> None:
    records = [json.loads(line) for line in store.queue_path.read_text().splitlines()]
    target = next(record for record in records if record["kind"] == kind)
    if operation == "add":
        target["unknown_envelope_field"] = "must-not-be-ignored"
    elif operation == "remove":
        target.pop("timestamp")
    else:  # pragma: no cover - test helper contract
        raise AssertionError(operation)
    previous_hash: str | None = None
    for record in records:
        record.pop("record_hash")
        record["prev_record_hash"] = previous_hash
        record["record_hash"] = sha256_json(record)
        previous_hash = record["record_hash"]
    store.queue_path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _seed_queue_record_kind(store: WorkingDatasetStore, kind: str) -> str:
    batch_id = f"exact-envelope-{kind}"
    receipt = store.enqueue_batch(
        _batch_request(store, batch_id=batch_id, image_ids=(1,))
    )
    if kind == "enqueue":
        return batch_id
    if kind == "claim":
        with store._exclusive_queue_lock():
            store._append_queue_record(
                {
                    "kind": "claim",
                    "batch_id": receipt.batch_id,
                    "payload_hash": receipt.payload_hash,
                },
                "claim",
            )
        return batch_id
    if kind == "queue_terminal":
        result = store.process_next_batch()
        assert (
            result is not None and result.status is store_module.BatchStatus.SUCCEEDED
        )
        return batch_id
    raise AssertionError(kind)  # pragma: no cover - test helper contract


def _batch_request(
    store: WorkingDatasetStore,
    *,
    batch_id: str = "batch-1",
    image_ids: tuple[int, ...] = (1, 2),
    reverse_members: bool = False,
    current_user_id: str = "reviewer",
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
        current_user_id=current_user_id,
        base_generation=store.restore_draft(1).generation,
        members=tuple(members),
    )


def _replace_only_batch_member_request(
    batch: store_module.BatchRequest,
    request: CommitRequest,
    *,
    source_row_index: Any | None = None,
) -> store_module.BatchRequest:
    member = batch.members[0]
    return replace(
        batch,
        members=(
            replace(
                member,
                source_row_index=(
                    member.source_row_index
                    if source_row_index is None
                    else source_row_index
                ),
                request=request,
            ),
        ),
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
    assert [member["source_row_index"] for member in queued["payload"]["members"]] == [
        0,
        1,
    ]

    request.members[0].request.regions[0]["bbox_2d"][0] = 777
    assert store.queue_path.read_bytes() == queue_before_mutation


def test_hashed_batch_payload_rehydrates_without_type_or_value_changes(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(
        _batch_request(
            store,
            batch_id="identity-preserving-rehydration",
            image_ids=(1, 2),
            reverse_members=True,
        )
    )
    payload = json.loads(store.queue_path.read_text(encoding="utf-8"))["payload"]

    rehydrated = store_module._batch_request_from_payload(payload)

    assert store._canonical_batch_payload(rehydrated) == payload


def test_batch_enqueue_exact_retry_is_byte_stable_and_payload_drift_conflicts(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store)
    first = store.enqueue_batch(request)
    queue_bytes = store.queue_path.read_bytes()

    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)
    calls_after_first = len(verifier.calls)
    verifier.accept = False
    assert store.enqueue_batch(copy.deepcopy(request)) == first
    assert len(verifier.calls) == calls_after_first
    assert store.queue_path.read_bytes() == queue_bytes

    changed = _batch_request(store, batch_id="batch-1", image_ids=(1, 3))
    with pytest.raises(CommitConflictError, match="batch id"):
        store.enqueue_batch(changed)
    assert store.queue_path.read_bytes() == queue_bytes


def test_batch_verifier_runs_once_outside_locks_and_can_read_store(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    original, _, _ = project
    verifier = BatchReadingAnnotationVerifier()
    store = WorkingDatasetStore(
        original.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=original.inference_receipt_resolver,
    )
    verifier.store = store
    request = _batch_request(store, batch_id="batch-verified", image_ids=(1, 2, 3))

    receipt = store.enqueue_batch(request)

    assert receipt.status is store_module.BatchStatus.QUEUED
    assert len(verifier.batch_calls) == 1
    assert verifier.batch_calls[0].current_user_id == "reviewer"
    assert verifier.fallback_calls == []


def test_batch_verifier_fallback_calls_each_identity_once(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)

    store.enqueue_batch(_batch_request(store, batch_id="fallback", image_ids=(1, 2)))

    assert [identity.image_id for identity in verifier.calls] == [1, 2]


def test_ten_member_batch_uses_one_batch_verifier_call(tmp_path: Path) -> None:
    verifier = BatchReadingAnnotationVerifier()
    store = _bootstrap_store_with_images(
        tmp_path,
        tuple(range(1, 11)),
        verifier,
    )
    verifier.store = store

    store.enqueue_batch(
        _batch_request(
            store,
            batch_id="ten-member",
            image_ids=tuple(range(1, 11)),
        )
    )

    assert len(verifier.batch_calls) == 1
    assert len(verifier.batch_calls[0].members) == 10
    assert verifier.fallback_calls == []


def test_current_user_is_payload_identity_and_must_be_trimmed(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    original = _batch_request(
        store,
        batch_id="current-user",
        image_ids=(1,),
        current_user_id="reviewer",
    )
    store.enqueue_batch(original)
    queue_before = store.queue_path.read_bytes()
    queued = json.loads(queue_before)
    assert queued["payload"]["current_user_id"] == "reviewer"

    changed = replace(original, current_user_id="another-reviewer")
    with pytest.raises(CommitConflictError, match="batch id"):
        store.enqueue_batch(changed)
    assert store.queue_path.read_bytes() == queue_before

    for invalid in ("", " reviewer "):
        with pytest.raises(ValidationError, match="current_user_id"):
            store.enqueue_batch(
                _batch_request(
                    store,
                    batch_id=f"invalid-user-{invalid!r}",
                    image_ids=(2,),
                    current_user_id=invalid,
                )
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annotation_revision", "annotation-v8"),
        ("draft_updated_at", "2026-07-15T00:00:08Z"),
        ("result_hash", "f" * 64),
    ],
)
def test_batch_retry_identity_binds_exact_draft_tokens_and_result_hash(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    field: str,
    value: str,
) -> None:
    store, _, _ = project
    original = _batch_request(store, batch_id="draft-token-batch", image_ids=(1,))
    store.enqueue_batch(original)
    queue_bytes = store.queue_path.read_bytes()
    member = original.members[0]
    changed_request = replace(member.request, **{field: value})
    changed_request = replace(
        changed_request,
        draft_save=_matching_draft_receipt(changed_request),
    )
    changed = replace(
        original,
        members=(replace(member, request=changed_request),),
    )

    with pytest.raises(CommitConflictError, match="batch id"):
        store.enqueue_batch(changed)
    assert store.queue_path.read_bytes() == queue_bytes


def test_nonfinite_batch_payload_is_rejected_before_queue_growth(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    original = _batch_request(store, batch_id="nonfinite", image_ids=(1,))
    member = original.members[0]
    regions = copy.deepcopy(list(member.request.regions))
    regions[0]["label_studio_result"] = {"score": float("nan")}
    changed_request = replace(member.request, regions=regions, result_hash="f" * 64)
    changed_request = replace(
        changed_request,
        draft_save=_matching_draft_receipt(changed_request),
    )
    changed = replace(original, members=(replace(member, request=changed_request),))
    queue_before = store.queue_path.read_bytes()

    with pytest.raises(ValidationError, match="finite ordinary JSON"):
        store.enqueue_batch(changed)
    assert store.queue_path.read_bytes() == queue_before
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"value": float("inf")})


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("image_id", True),
        ("image_id", 1.0),
        ("image_id", "1"),
        ("source_row_index", True),
        ("source_row_index", "0"),
        ("source_row_index", 0.0),
        ("base_generation", True),
        ("base_generation", 0.0),
        ("observed_generation", False),
        ("observed_generation", 0.0),
        ("durable", "false"),
        ("durable", 1),
        ("commit_id", 7),
        ("annotation_id", 7),
        ("inference_receipts", (123,)),
        ("inference_receipts", "receipt-1"),
    ],
)
def test_batch_payload_rejects_scalar_type_confusion_before_verifier_or_queue(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    target: str,
    value: Any,
) -> None:
    store, _, _ = project
    batch = _batch_request(store, batch_id=f"invalid-scalar-{target}", image_ids=(1,))
    request = batch.members[0].request
    source_row_index: Any | None = None
    if target == "source_row_index":
        source_row_index = value
    elif target == "base_generation":
        batch = replace(batch, base_generation=value)
    elif target == "durable":
        request = replace(
            request,
            draft_save=replace(request.draft_save, durable=value),
        )
    elif target == "inference_receipts":
        request = replace(request, inference_receipts=value)
    else:
        request = replace(request, **{target: value})
        if target == "annotation_id":
            request = replace(
                request,
                draft_save=replace(request.draft_save, annotation_id=value),
            )
    batch = _replace_only_batch_member_request(
        batch,
        request,
        source_row_index=source_row_index,
    )
    queue_before = store.queue_path.read_bytes()
    working_before = store.working_path.read_bytes()
    journal_before = store.journal_path.read_bytes()
    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)

    with pytest.raises(ValidationError, match="payload"):
        store.enqueue_batch(batch)

    assert verifier.calls == []
    assert store.queue_path.read_bytes() == queue_before
    assert store.working_path.read_bytes() == working_before
    assert store.journal_path.read_bytes() == journal_before


def test_batch_payload_rejects_non_string_json_object_key_before_verifier(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    batch = _batch_request(store, batch_id="invalid-metadata-key", image_ids=(1,))
    request = batch.members[0].request
    regions = copy.deepcopy(list(request.regions))
    regions[-1]["metadata"] = {1: "not-an-ordinary-json-key"}
    changed = replace(
        request,
        regions=regions,
        result_hash=sha256_json(regions),
    )
    changed = replace(changed, draft_save=_matching_draft_receipt(changed))
    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)

    with pytest.raises(ValidationError, match="JSON object keys must be strings"):
        store.enqueue_batch(_replace_only_batch_member_request(batch, changed))

    assert verifier.calls == []
    assert store.queue_path.read_bytes() == b""
    assert store.journal_path.read_bytes() == b""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category_id", True),
        ("creation_ordinal", "4"),
        ("coco_ann_id", True),
    ],
)
def test_batch_payload_rejects_region_scalar_type_confusion_before_verifier(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    field: str,
    value: Any,
) -> None:
    store, _, _ = project
    batch = _batch_request(store, batch_id=f"invalid-region-{field}", image_ids=(1,))
    request = batch.members[0].request
    regions = copy.deepcopy(list(request.regions))
    regions[-1][field] = value
    changed = replace(
        request,
        regions=regions,
        semantic_hash=semantic_hash(regions),
        result_hash=sha256_json(regions),
    )
    changed = replace(changed, draft_save=_matching_draft_receipt(changed))
    batch = _replace_only_batch_member_request(batch, changed)
    verifier = store.annotation_verifier
    assert isinstance(verifier, AcceptingAnnotationVerifier)

    with pytest.raises(ValidationError):
        store.enqueue_batch(batch)

    assert verifier.calls == []
    assert store.queue_path.read_bytes() == b""
    assert store.journal_path.read_bytes() == b""


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("image_id", True),
        ("source_row_index", "0"),
        ("durable", "false"),
        ("annotation_id", 7),
    ],
)
def test_tampered_hashed_queue_payload_fails_recovery_without_file_mutation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    target: str,
    value: Any,
) -> None:
    store, _, _ = project
    store.enqueue_batch(
        _batch_request(store, batch_id=f"tampered-{target}", image_ids=(1,))
    )
    record = json.loads(store.queue_path.read_text(encoding="utf-8"))
    member = record["payload"]["members"][0]
    request = member["request"]
    if target == "source_row_index":
        member["source_row_index"] = value
    elif target == "durable":
        request["draft_save"]["durable"] = value
    else:
        request[target] = value
        if target == "annotation_id":
            request["draft_save"][target] = value
    record["payload_hash"] = sha256_json(record["payload"])
    record.pop("record_hash")
    record["record_hash"] = sha256_json(record)
    store.queue_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    queue_before = store.queue_path.read_bytes()
    working_before = store.working_path.read_bytes()
    journal_before = store.journal_path.read_bytes()

    with pytest.raises(RecoveryError, match="payload"):
        store.process_next_batch()
    with pytest.raises(RecoveryError, match="payload"):
        _reopen(store)

    assert store.queue_path.read_bytes() == queue_before
    assert store.working_path.read_bytes() == working_before
    assert store.journal_path.read_bytes() == journal_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("member_count", True),
        ("base_generation", 0.0),
    ],
)
def test_tampered_queue_envelope_rejects_equal_but_different_scalar_type(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    field: str,
    value: Any,
) -> None:
    store, _, _ = project
    store.enqueue_batch(
        _batch_request(store, batch_id=f"tampered-envelope-{field}", image_ids=(1,))
    )
    record = json.loads(store.queue_path.read_text(encoding="utf-8"))
    record[field] = value
    record.pop("record_hash")
    record["record_hash"] = sha256_json(record)
    store.queue_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    queue_before = store.queue_path.read_bytes()
    working_before = store.working_path.read_bytes()

    with pytest.raises(RecoveryError, match="payload attestation"):
        _reopen(store)

    assert store.queue_path.read_bytes() == queue_before
    assert store.working_path.read_bytes() == working_before


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("enqueue", "add"),
        ("enqueue", "remove"),
        ("claim", "add"),
        ("claim", "remove"),
        ("queue_terminal", "add"),
        ("queue_terminal", "remove"),
    ],
)
def test_queue_record_exact_schema_rejects_one_field_add_or_remove(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    kind: str,
    operation: str,
) -> None:
    store, _, _ = project
    _seed_queue_record_kind(store, kind)
    _rewrite_queue_chain_after_envelope_tamper(
        store,
        kind=kind,
        operation=operation,
    )
    queue_before = store.queue_path.read_bytes()
    journal_before = store.journal_path.read_bytes()
    working_before = store.working_path.read_bytes()
    manifest_before = store.manifest_path.read_bytes()

    with pytest.raises(RecoveryError, match="canonical schema"):
        store.process_next_batch()
    with pytest.raises(RecoveryError, match="canonical schema"):
        _reopen(store)

    assert store.queue_path.read_bytes() == queue_before
    assert store.journal_path.read_bytes() == journal_before
    assert store.working_path.read_bytes() == working_before
    assert store.manifest_path.read_bytes() == manifest_before


def test_success_queue_terminal_allows_legacy_omitted_error_field(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    batch_id = _seed_queue_record_kind(store, "queue_terminal")
    records = [json.loads(line) for line in store.queue_path.read_text().splitlines()]
    terminal = next(record for record in records if record["kind"] == "queue_terminal")
    assert terminal.pop("error") is None
    previous_hash: str | None = None
    for record in records:
        record.pop("record_hash")
        record["prev_record_hash"] = previous_hash
        record["record_hash"] = sha256_json(record)
        previous_hash = record["record_hash"]
    store.queue_path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    queue_before = store.queue_path.read_bytes()

    reopened = _reopen(store)

    assert (
        reopened.get_batch_status(batch_id).status is store_module.BatchStatus.SUCCEEDED
    )
    assert store.queue_path.read_bytes() == queue_before


def test_queue_parser_rejects_nonstandard_json_constants(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.queue_path.write_text('{"kind":"enqueue","value":NaN}\n', encoding="utf-8")

    with pytest.raises(RecoveryError, match="invalid queue record"):
        store.get_batch_status("missing")


def test_batch_enqueue_returns_existing_active_batch_identity(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    active = store.enqueue_batch(_batch_request(store, batch_id="batch-active"))

    returned = store.enqueue_batch(
        _batch_request(store, batch_id="batch-other", image_ids=(3,))
    )

    assert returned == active
    assert (
        store.get_batch_status("batch-other").status
        is store_module.BatchStatus.NOT_FOUND
    )
    assert store.queue_path.read_text(encoding="utf-8").count("\n") == 1


def test_concurrent_post_attestation_admission_enqueues_only_one_batch(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    original, _, _ = project
    barrier = threading.Barrier(2)
    verifier = BatchReadingAnnotationVerifier(barrier=barrier)
    store = WorkingDatasetStore(
        original.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=original.inference_receipt_resolver,
    )
    request_a = _batch_request(store, batch_id="racing-a", image_ids=(1,))
    request_b = _batch_request(store, batch_id="racing-b", image_ids=(2,))
    receipts: list[store_module.BatchEnqueueReceipt] = []
    errors: list[BaseException] = []

    def enqueue(request: store_module.BatchRequest) -> None:
        try:
            receipts.append(store.enqueue_batch(request))
        except BaseException as exc:  # capture thread failures for the assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=enqueue, args=(request_a,)),
        threading.Thread(target=enqueue, args=(request_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(receipts) == 2
    assert len({receipt.batch_id for receipt in receipts}) == 1
    queue_records = [
        json.loads(line) for line in store.queue_path.read_text().splitlines()
    ]
    assert [record["kind"] for record in queue_records] == ["enqueue"]
    assert queue_records[0]["batch_id"] in {"racing-a", "racing-b"}
    assert len(verifier.batch_calls) == 2


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


def test_batch_candidate_build_keeps_old_generation_readable(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store, batch_id="batch-paused-build", image_ids=(1,))
    store.enqueue_batch(request)
    pause = PauseAt("batch_working_temp_fsynced")
    store._fault_injector = pause
    outcome: list[store_module.BatchResult] = []
    errors: list[BaseException] = []

    def process() -> None:
        try:
            result = store.process_next_batch()
            assert result is not None
            outcome.append(result)
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert pause.entered.wait(1.0)
    try:
        status = store.get_batch_status("batch-paused-build")
        restored = store.restore_draft(1)
        state = store.state_snapshot()
        assert status.status is store_module.BatchStatus.RUNNING
        assert restored.generation == 0
        assert restored.row_hash == request.members[0].request.base_row_hash
        assert state.generation == 0
        with pytest.raises(StoreBusyError, match="processor is already active"):
            store.process_next_batch()
    finally:
        pause.release.set()
        worker.join(3.0)

    assert not worker.is_alive()
    assert errors == []
    assert len(outcome) == 1
    assert outcome[0].status is store_module.BatchStatus.SUCCEEDED
    assert (
        store.get_batch_status("batch-paused-build").status
        is store_module.BatchStatus.SUCCEEDED
    )
    assert store.restore_draft(1).generation == 1


def test_batch_materialization_uses_frozen_identity_while_readers_reload(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = project
    peer = _reopen(store)
    request = _batch_request(store, batch_id="batch-frozen-maps", image_ids=(1,))
    store.enqueue_batch(request)
    entered = threading.Event()
    release = threading.Event()
    original = store._materialize_objects
    first_call = True

    def blocked_materialize(*args: Any, **kwargs: Any):
        nonlocal first_call
        if first_call:
            first_call = False
            entered.set()
            assert release.wait(3.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_materialize_objects", blocked_materialize)
    outcome: list[store_module.BatchResult] = []
    errors: list[BaseException] = []

    def process() -> None:
        try:
            result = store.process_next_batch()
            assert result is not None
            outcome.append(result)
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert entered.wait(1.0)
    candidates = tuple(store.split_dir.glob(".working.batch.*"))
    try:
        assert store.restore_draft(1).generation == 0
        assert (
            store.get_batch_status("batch-frozen-maps").status
            is store_module.BatchStatus.RUNNING
        )
        with pytest.raises(StoreBusyError, match="processor is already active"):
            peer.recover()
        assert tuple(store.split_dir.glob(".working.batch.*")) == candidates
    finally:
        release.set()
        worker.join(3.0)

    assert not worker.is_alive()
    assert errors == []
    assert len(outcome) == 1
    assert outcome[0].members[0].region_id_mapping["drawn:batch-1"] == -1
    records = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    assert all(
        record.get("prev_record_hash")
        == (records[index - 1]["record_hash"] if index else None)
        for index, record in enumerate(records)
    )


def test_batch_phase_c_rejects_changed_cache_before_publication(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(
        _batch_request(store, batch_id="batch-cache-reattest", image_ids=(1,))
    )
    working_before = store.working_path.read_bytes()
    pause = PauseAt("batch_working_temp_fsynced")
    store._fault_injector = pause
    errors: list[BaseException] = []

    def process() -> None:
        try:
            store.process_next_batch()
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    worker = threading.Thread(target=process)
    worker.start()
    assert pause.entered.wait(1.0)
    store._row_hash_cache[0] = "0" * 64
    pause.release.set()
    worker.join(3.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RecoveryError)
    assert "authority changed before publication" in str(errors[0])
    assert store.working_path.read_bytes() == working_before
    assert not any(
        json.loads(line).get("kind") == "batch_prepared"
        for line in store.journal_path.read_text().splitlines()
    )

    with store._exclusive_lock():
        store._load_row_cache()
    store._fault_injector = None
    retried = store.process_next_batch()
    assert retried is not None
    assert retried.status is store_module.BatchStatus.SUCCEEDED


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
        current_user_id="reviewer",
        base_generation=0,
        members=tuple(batch_members),
    )

    with pytest.raises(ValidationError, match=error):
        store.enqueue_batch(request)
    assert store.queue_path.read_bytes() == b""


@pytest.mark.parametrize(
    "identity,error",
    [
        ({"split": "val"}, "split mismatch"),
        ({"project_id": "project-other"}, "project mismatch"),
        ({"task_id": "train:2"}, "task identity mismatch"),
        ({"image_id": 999, "task_id": "train:999"}, "unknown task image identity"),
        ({"image_id": True}, "invalid task image identity"),
    ],
)
def test_resolve_source_row_index_rejects_every_identity_mismatch(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    identity: dict[str, Any],
    error: str,
) -> None:
    store, _, _ = project
    values: dict[str, Any] = {
        "split": "train",
        "project_id": "project-train",
        "task_id": "train:1",
        "image_id": 1,
    }
    values.update(identity)

    with pytest.raises(StaleCommitError, match=error):
        store.resolve_source_row_index(**values)


def test_restore_drafts_uses_one_barrier_and_preserves_requested_order(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = project
    store.commit(_request(store, commit_id="baseline-generation", image_id=2))
    original_lock = store._supported_reader_lock
    original_manifest = store._read_manifest
    original_scan = store._scan_attested_restore_rows
    state = {"active": False, "entries": 0, "scans": 0}

    @contextmanager
    def tracked_reader_lock():
        assert state["active"] is False
        state["active"] = True
        state["entries"] += 1
        try:
            with original_lock():
                yield
        finally:
            state["active"] = False

    def checked_manifest() -> dict[str, Any]:
        assert state["active"] is True
        return original_manifest()

    def checked_scan(
        requested_image_ids: set[int], manifest: dict[str, Any]
    ) -> dict[int, dict[str, Any]]:
        assert state["active"] is True
        state["scans"] += 1
        return original_scan(requested_image_ids, manifest)

    monkeypatch.setattr(store, "_supported_reader_lock", tracked_reader_lock)
    monkeypatch.setattr(store, "_read_manifest", checked_manifest)
    monkeypatch.setattr(store, "_scan_attested_restore_rows", checked_scan)

    restored = store.restore_drafts((3, 1, 2))

    assert state == {"active": False, "entries": 1, "scans": 1}
    assert [item.image_id for item in restored] == [3, 1, 2]
    assert {item.generation for item in restored} == {1}
    assert store.restore_draft(2).image_id == 2
    assert state == {"active": False, "entries": 2, "scans": 2}


def test_task_navigation_random_reads_only_the_requested_row(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = project
    baseline = store.restore_draft(2)
    original_parse = store_module._parse_working_jsonl_row
    parsed_indexes: list[int] = []

    def tracked_parse(raw: bytes, source_row_index: int) -> dict[str, Any]:
        parsed_indexes.append(source_row_index)
        return original_parse(raw, source_row_index)

    def forbidden_scan(*_args: Any, **_kwargs: Any) -> dict[int, dict[str, Any]]:
        raise AssertionError("task navigation must not scan the complete working file")

    monkeypatch.setattr(store_module, "_parse_working_jsonl_row", tracked_parse)
    monkeypatch.setattr(store, "_scan_attested_restore_rows", forbidden_scan)

    restored = store.restore_task_navigation(
        2,
        projected_generation=baseline.generation,
        projected_row_hash=baseline.row_hash,
    )

    assert restored == baseline
    assert parsed_indexes == [1]


def test_task_navigation_rejects_same_generation_file_drift(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    baseline = store.restore_draft(1)
    store.working_path.write_bytes(store.working_path.read_bytes() + b"drift")

    with pytest.raises(RecoveryError, match="navigation index attestation"):
        store.restore_task_navigation(
            1,
            projected_generation=baseline.generation,
            projected_row_hash=baseline.row_hash,
        )


def test_preopened_navigation_index_rebuilds_after_generation_advance(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    worker, _, _ = project
    peer = _reopen(worker)
    prior = peer.restore_draft(1)
    peer.restore_task_navigation(
        1,
        projected_generation=prior.generation,
        projected_row_hash=prior.row_hash,
    )

    worker.commit(_request(worker, commit_id="navigation-generation", image_id=1))
    later = worker.restore_draft(3)
    restored_later = peer.restore_task_navigation(
        3,
        projected_generation=later.generation,
        projected_row_hash=later.row_hash,
    )
    current = worker.restore_draft(1)
    restored = peer.restore_task_navigation(
        1,
        projected_generation=current.generation,
        projected_row_hash=current.row_hash,
    )

    assert restored_later == later
    assert restored == current
    assert restored.generation == 1


@pytest.mark.parametrize(
    ("image_ids", "error_type", "message"),
    [
        ((True,), ValidationError, "non-negative integer"),
        ((1.0,), ValidationError, "non-negative integer"),
        (("1",), ValidationError, "non-negative integer"),
        ((-1,), ValidationError, "non-negative integer"),
        ((1, 1), ValidationError, "duplicate image_id"),
        ((999,), StaleCommitError, "unknown image_id"),
    ],
)
def test_restore_drafts_rejects_invalid_duplicate_and_unknown_image_ids(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    image_ids: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    store, _, _ = project

    with pytest.raises(error_type, match=message):
        store.restore_drafts(image_ids)


def test_restore_drafts_returns_mutation_isolated_ordinary_json(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first, second = store.restore_drafts((1, 2))
    second_before = copy.deepcopy(second.row)
    first.row["objects"][0]["bbox_2d"][0] = 999
    first.row["metadata"]["caller_mutation"] = True
    first.region_id_mapping["caller:key"] = -999

    assert second.row == second_before
    assert "caller:key" not in second.region_id_mapping
    json.dumps(first.row, allow_nan=False)
    fresh_first, fresh_second = store.restore_drafts((1, 2))
    assert fresh_first.row["objects"][0]["bbox_2d"] == [10, 20, 30, 40]
    assert "caller_mutation" not in fresh_first.row["metadata"]
    assert "caller:key" not in fresh_first.region_id_mapping
    assert fresh_second.row == second_before


def test_restore_drafts_rejects_drift_in_an_unrequested_row(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    lines = store.working_path.read_bytes().splitlines(keepends=True)
    unrequested = json.loads(lines[1])
    unrequested["objects"][0]["bbox_2d"] = [11, 20, 30, 40]
    lines[1] = (canonical_json(unrequested) + "\n").encode("utf-8")
    store.working_path.write_bytes(b"".join(lines))

    with pytest.raises(RecoveryError, match="hash/line-count attestation"):
        store.restore_drafts((1,))


def test_restore_drafts_rejects_truncated_tail_after_requested_row(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    working = store.working_path.read_bytes()
    store.working_path.write_bytes(working[:-7])

    with pytest.raises(RecoveryError, match="hash/line-count attestation"):
        store.restore_drafts((1,))


def test_empty_restore_drafts_still_attests_the_complete_working_file(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    assert store.restore_drafts(()) == ()
    store.working_path.write_bytes(store.working_path.read_bytes() + b"drift")

    with pytest.raises(RecoveryError, match="hash/line-count attestation"):
        store.restore_drafts(())


def test_restore_drafts_fails_closed_during_batch_reconciliation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    store._fault_injector = CrashAt("batch_working_replaced")
    with pytest.raises(InjectedCrash):
        store.process_next_batch()

    with pytest.raises(RecoveryError, match="requires recovery"):
        store.restore_drafts((1, 2))


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
    assert (
        store.get_batch_status(receipt.batch_id).status
        is store_module.BatchStatus.SUCCEEDED
    )
    assert store.get_batch_result(receipt.batch_id) == result
    records = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    assert [record["kind"] for record in records].count("batch_prepared") == 1
    assert [record["kind"] for record in records].count("batch_terminal") == 1
    assert [record["kind"] for record in records].count("reservation") == 2


def test_durable_batch_request_and_terminal_pairs_reopen_exactly(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    request = _batch_request(store, image_ids=(1, 2))

    store.enqueue_batch(request)
    assert store.get_batch_request(request.batch_id) == request
    assert store.terminal_batch_pairs() == ()

    result = store.process_next_batch()
    assert result is not None
    assert store.get_batch_request(request.batch_id) == request
    assert store.terminal_batch_pairs() == ((request, result),)

    reopened = _reopen(store)
    assert reopened.get_batch_request(request.batch_id) == request
    assert reopened.terminal_batch_pairs() == ((request, result),)


def test_batch_freshness_is_per_row_not_global_generation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    worker_store = _reopen(store)
    old_request = _request(worker_store, commit_id="old-generation-member", image_id=1)
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
        current_user_id="reviewer",
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
        current_user_id="reviewer",
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
    assert (
        store.get_batch_status("stale-batch").status is store_module.BatchStatus.FAILED
    )


def test_sequential_batches_preserve_prior_rows_and_use_frozen_snapshot(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    first = _batch_request(store, batch_id="first", image_ids=(1,))
    store.enqueue_batch(first)
    first.members[0].request.regions[0]["bbox_2d"] = [200, 210, 220, 230]
    first_result = store.process_next_batch()
    assert (
        first_result is not None
        and first_result.status is store_module.BatchStatus.SUCCEEDED
    )
    row_one_after_first = store.working_path.read_bytes().splitlines(keepends=True)[0]
    assert json.loads(row_one_after_first)["objects"][0]["bbox_2d"] == [11, 21, 31, 41]

    store.enqueue_batch(_batch_request(store, batch_id="second", image_ids=(2,)))
    second_result = store.process_next_batch()

    assert (
        second_result is not None
        and second_result.status is store_module.BatchStatus.SUCCEEDED
    )
    assert (
        store.working_path.read_bytes().splitlines(keepends=True)[0]
        == row_one_after_first
    )


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
    assert [
        (record["stable_region_key"], record["coco_ann_id"]) for record in reservations
    ] == [("drawn:batch-1", -1)]

    corrected = _batch_request(store, batch_id="corrected", image_ids=(1,))
    store.enqueue_batch(corrected)
    corrected_result = store.process_next_batch()
    assert corrected_result is not None
    assert corrected_result.members[0].region_id_mapping["drawn:batch-1"] == -1
    assert (
        sum(
            json.loads(line)["kind"] == "reservation"
            for line in store.journal_path.read_text().splitlines()
        )
        == 1
    )

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


def test_batch_rejects_cross_task_stable_key_collision_before_reservation(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    batch = _batch_request(store, batch_id="colliding-keys", image_ids=(1, 2))
    first, second = batch.members
    second_regions = copy.deepcopy(list(second.request.regions))
    second_regions[-1]["region_key"] = "drawn:batch-1"
    second_request = _request(
        store,
        commit_id=second.request.commit_id,
        image_id=second.request.image_id,
        regions=second_regions,
    )
    batch = replace(
        batch,
        members=(first, replace(second, request=second_request)),
    )
    working_before = store.working_path.read_bytes()
    manifest_before = store.manifest_path.read_bytes()
    store.enqueue_batch(batch)

    result = store.process_next_batch()

    assert result is not None and result.status is store_module.BatchStatus.FAILED
    assert "stable region key is already bound to another task" in (result.error or "")
    assert store.working_path.read_bytes() == working_before
    assert store.manifest_path.read_bytes() == manifest_before
    records = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    assert [record["kind"] for record in records] == ["batch_terminal"]
    reopened = _reopen(store)
    assert reopened.get_batch_result(batch.batch_id).status is store_module.BatchStatus.FAILED


def test_terminal_batch_retry_does_not_republish_or_allocate_again(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    batch = _batch_request(store, batch_id="terminal-retry", image_ids=(1,))
    first_receipt = store.enqueue_batch(batch)
    first_result = store.process_next_batch()
    assert first_result is not None
    artifacts_before = {
        path: path.read_bytes()
        for path in (
            store.working_path,
            store.manifest_path,
            store.journal_path,
            store.queue_path,
        )
    }

    retry_receipt = store.enqueue_batch(copy.deepcopy(batch))

    assert retry_receipt.batch_id == first_receipt.batch_id
    assert retry_receipt.payload_hash == first_receipt.payload_hash
    assert retry_receipt.status is store_module.BatchStatus.SUCCEEDED
    assert store.get_batch_result(batch.batch_id) == first_result
    assert store.process_next_batch() is None
    assert {
        path: path.read_bytes()
        for path in artifacts_before
    } == artifacts_before


def test_batch_deletion_tombstones_identity_against_later_redraw(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    store.enqueue_batch(_batch_request(store, batch_id="add", image_ids=(1,)))
    added = store.process_next_batch()
    assert added is not None and added.status is store_module.BatchStatus.SUCCEEDED
    assert added.members[0].region_id_mapping["drawn:batch-1"] == -1

    delete_regions = [
        region
        for region in _regions(store, image_id=1, include_new=False)
        if region["region_key"] != "drawn:batch-1"
    ]
    delete_request = _request(
        store,
        commit_id="delete:member:1",
        image_id=1,
        regions=delete_regions,
    )
    delete_batch = store_module.BatchRequest(
        batch_id="delete",
        split="train",
        current_user_id="reviewer",
        base_generation=delete_request.observed_generation,
        members=(store_module.BatchMember(0, delete_request),),
    )
    store.enqueue_batch(delete_batch)
    deleted = store.process_next_batch()
    assert deleted is not None and deleted.status is store_module.BatchStatus.SUCCEEDED

    redraw_regions = _regions(store, image_id=1, include_new=False)
    redraw_regions.append(
        {
            "region_key": "drawn:batch-1",
            "bbox_2d": [300, 310, 320, 330],
            "category_name": "dog",
            "category_id": 18,
        }
    )
    redraw_request = _request(
        store,
        commit_id="redraw:member:1",
        image_id=1,
        regions=redraw_regions,
    )
    redraw_batch = store_module.BatchRequest(
        batch_id="redraw",
        split="train",
        current_user_id="reviewer",
        base_generation=redraw_request.observed_generation,
        members=(store_module.BatchMember(0, redraw_request),),
    )
    store.enqueue_batch(redraw_batch)

    redrawn = store.process_next_batch()

    assert redrawn is not None and redrawn.status is store_module.BatchStatus.FAILED
    assert "tombstoned" in (redrawn.error or "")


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
    assert (
        json.loads(reopened.manifest_path.read_text())["generation"]
        == expected_generation
    )
    assert any(
        json.loads(line).get("kind") == "queue_terminal"
        for line in reopened.queue_path.read_text().splitlines()
    )


def test_prepublication_recovery_observer_exposes_exact_prior_navigation_row(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
) -> None:
    store, _, _ = project
    prior = store.restore_draft(1)
    store.enqueue_batch(_batch_request(store, image_ids=(1,)))
    store._fault_injector = CrashAt("batch_prepared_journal_fsynced")
    with pytest.raises(InjectedCrash):
        store.process_next_batch()
    store._fault_injector = None

    observer_entered = threading.Event()
    observer_release = threading.Event()
    errors: list[BaseException] = []

    def observe(_request: Any, result: Any) -> None:
        assert result.status is store_module.BatchStatus.FAILED
        assert result.error == "recovered before batch publication"
        observer_entered.set()
        assert observer_release.wait(3.0)

    def recover() -> None:
        try:
            store.recover(terminal_observer=observe)
        except BaseException as exc:  # pragma: no cover - assertion receipt
            errors.append(exc)

    recovery = threading.Thread(target=recover)
    recovery.start()
    assert observer_entered.wait(1.0)
    with pytest.raises(RecoveryError, match="requires recovery"):
        store.restore_draft(1)
    projected = store.restore_task_navigation(
        1,
        projected_generation=prior.generation,
        projected_row_hash=prior.row_hash,
    )
    assert projected == prior

    observer_release.set()
    recovery.join(3.0)
    assert not recovery.is_alive()
    assert errors == []
    assert store.get_batch_status("batch-1").status is store_module.BatchStatus.FAILED


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
        assert (
            peer.get_batch_status("batch-1").status
            is store_module.BatchStatus.SUCCEEDED
        )
        accepted = peer.enqueue_batch(next_request)
        assert accepted.batch_id == "batch-next"
        assert accepted.status is store_module.BatchStatus.QUEUED
        assert peer.state_snapshot().generation == 1
    else:
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.restore_draft(1)
        with pytest.raises(RecoveryError, match="requires recovery"):
            list(peer.iter_task_seeds())
        with pytest.raises(RecoveryError, match="requires recovery"):
            peer.state_snapshot()
        assert (
            peer.get_batch_status("batch-1").status
            is store_module.BatchStatus.RECONCILING
        )
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
        fresh_later_legacy = _request(peer, commit_id="legacy-later-fresh", image_id=2)
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
            worker.commit(_request(worker, commit_id="guard-interrupted", image_id=1))
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
    assert (
        reopened.get_batch_status("batch-1").status is store_module.BatchStatus.QUEUED
    )
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
    store.task_index_path.write_text(
        canonical_json(task_index) + "\n", encoding="utf-8"
    )
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
    assert (
        reopened.get_batch_status("batch-1").status is store_module.BatchStatus.QUEUED
    )


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


def test_bootstrap_reuses_exact_preexisting_managed_image_link(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    tmp_path: Path,
) -> None:
    store, spec, _ = project
    adapter_spec = replace(spec, runtime_root=tmp_path / "adapter-runtime")
    split_dir = adapter_spec.runtime_root / adapter_spec.split
    split_dir.mkdir(parents=True)
    images_link = split_dir / "images"
    images_link.symlink_to(adapter_spec.image_root.resolve(), target_is_directory=True)
    link_inode = images_link.lstat().st_ino

    result = WorkingDatasetStore.bootstrap(
        adapter_spec,
        annotation_verifier=store.annotation_verifier,
        inference_receipt_resolver=store.inference_receipt_resolver,
    )

    assert result.created is True
    assert images_link.is_symlink()
    assert images_link.resolve(strict=True) == adapter_spec.image_root.resolve()
    assert images_link.lstat().st_ino == link_inode


@pytest.mark.parametrize(
    "preexisting_kind",
    ["wrong_symlink", "broken_symlink", "file", "directory", "other_residue"],
)
def test_bootstrap_rejects_and_preserves_non_exact_preexisting_state(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    tmp_path: Path,
    preexisting_kind: str,
) -> None:
    store, spec, _ = project
    adapter_spec = replace(
        spec,
        runtime_root=tmp_path / f"adapter-runtime-{preexisting_kind}",
    )
    split_dir = adapter_spec.runtime_root / adapter_spec.split
    split_dir.mkdir(parents=True)
    images_link = split_dir / "images"
    residue = images_link
    if preexisting_kind == "wrong_symlink":
        wrong_target = tmp_path / "wrong-images"
        wrong_target.mkdir()
        images_link.symlink_to(wrong_target, target_is_directory=True)
    elif preexisting_kind == "broken_symlink":
        images_link.symlink_to(tmp_path / "missing-images", target_is_directory=True)
    elif preexisting_kind == "file":
        images_link.write_text("not a managed link", encoding="utf-8")
    elif preexisting_kind == "directory":
        images_link.mkdir()
    else:
        residue = split_dir / "unexpected"
        residue.write_text("partial bootstrap state", encoding="utf-8")
    residue_inode = residue.lstat().st_ino

    with pytest.raises(ManifestDriftError):
        WorkingDatasetStore.bootstrap(
            adapter_spec,
            annotation_verifier=store.annotation_verifier,
            inference_receipt_resolver=store.inference_receipt_resolver,
        )

    assert residue.lstat().st_ino == residue_inode
    assert not (split_dir / "project.json").exists()


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
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    field: str,
    replacement: str,
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
    assert [obj["coco_ann_id"] for obj in result.committed_row["objects"]] == [
        101,
        102,
        -1,
    ]
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
    assert [obj["coco_ann_id"] for obj in first.committed_row["objects"]][-2:] == [
        -1,
        -2,
    ]

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
        annotation_revision="annotation-v8",
        draft_updated_at="2026-07-15T00:00:08Z",
        semantic_hash=empty_hash,
        result_hash=sha256_json([]),
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
        annotation_revision="annotation-v8",
        draft_updated_at="2026-07-15T00:00:08Z",
        semantic_hash=empty_hash,
        result_hash=sha256_json([]),
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
                **{
                    **request.draft_save.__dict__,
                    "annotation_revision": "annotation-v6",
                }
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

    integer_revision = replace(
        request,
        annotation_revision=7,  # type: ignore[arg-type]
        draft_save=replace(
            request.draft_save,
            annotation_revision=7,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(ValidationError, match="opaque non-empty string"):
        store.commit(integer_revision)


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
    assert events.index("working_directory_fsynced") < events.index(
        "manifest_temp_flushed"
    )
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
    assert (
        sum(
            json.loads(line)["kind"] == "terminal"
            for line in recovered.journal_path.read_text().splitlines()
        )
        == terminal_count
    )


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
    if boundary not in {
        "terminal_journal_flushed",
        "terminal_journal_fsynced",
        "before_response",
    }:
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
    assert (
        sum(
            json.loads(line)["kind"] == "terminal"
            for line in recovered.journal_path.read_text().splitlines()
        )
        == terminal_count
        == 1
    )


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
    assert [result.region_id_mapping[key] for key in ("new-earlier", "new-later")] == [
        -2,
        -1,
    ]
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
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path],
    monkeypatch: pytest.MonkeyPatch,
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
    assert (
        manifest["working_sha256"]
        == hashlib.sha256(store.working_path.read_bytes()).hexdigest()
    )
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
    prepared_raw, terminal_raw = store.journal_path.read_bytes().splitlines(
        keepends=True
    )
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
        store.commit(
            _request(store, commit_id="forged", image_id=image_id, regions=regions)
        )


def test_bootstrap_and_recovery_reject_split_wide_duplicate_coco_ann_id(
    project: tuple[WorkingDatasetStore, BootstrapSpec, Path], tmp_path: Path
) -> None:
    store, spec, source = project
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[1]["objects"][0]["coco_ann_id"] = rows[0]["objects"][0]["coco_ann_id"]
    source.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
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

    working_rows = [
        json.loads(line) for line in store.working_path.read_text().splitlines()
    ]
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
        draft_updated_at=request.draft_updated_at,
        semantic_hash=request.semantic_hash,
        result_hash=request.result_hash,
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
        replace(original, annotation_revision="annotation-v8"),
        replace(original, draft_updated_at="2026-07-15T00:00:08Z"),
        replace(original, result_hash="f" * 64),
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
                "draft_revision": "2026-07-15T00:00:06Z",
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
        "current_user_id": "reviewer",
        "draft_id": "draft-1",
        "draft_revision": "2026-07-15T00:00:06Z",
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
            ),
            current_user_id="reviewer",
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
            "cross-user",
            _valid_inference_link(current_user_id="other-reviewer"),
            "target mismatch",
        ),
        (
            "replaced-draft",
            _valid_inference_link(draft_id="draft-replacement"),
            "target mismatch",
        ),
        (
            "draft-revision-drift",
            _valid_inference_link(draft_revision="2026-07-15T00:00:08Z"),
            "source Draft revision mismatch",
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
                ),
                current_user_id="reviewer",
            )

    incomplete = [dict(region) for region in regions]
    incomplete[-1] = {
        **incomplete[-1],
        "metadata": {"inference_origin": True, "receipt_id": "receipt-1"},
    }
    with pytest.raises(ValidationError, match="complete inference linkage"):
        store.commit(_request(store, commit_id="incomplete", regions=incomplete))

    resolver.links["receipt-1"] = _valid_inference_link()
    with pytest.raises(ValidationError, match="authenticated current_user_id"):
        store.commit(
            replace(
                _request(store, commit_id="missing-user", regions=regions),
                inference_receipts=("receipt-1",),
            )
        )

    request = replace(
        _request(store, commit_id="valid-inference", regions=regions),
        inference_receipts=("receipt-1",),
    )
    assert request.draft_updated_at != resolver.links["receipt-1"].draft_revision
    result = store.commit(request, current_user_id="reviewer")
    assert result.region_id_mapping["inferred:region-1"] == -1
    with pytest.raises(ValidationError, match="target mismatch"):
        store.commit(request, current_user_id="other-reviewer")
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
