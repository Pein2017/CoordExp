from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest
from PIL import Image

from src.coco_refinement.adapters import (
    AdapterContractError,
    SqliteDraftCatalog,
    SqliteDraftVerifier,
    SqliteInferenceReceiptResolver,
    SqliteTerminalReconciler,
)
from src.coco_refinement.bootstrap import BootstrapSourceContract, bootstrap_workspace
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import NativeTaskIdentity
from src.coco_refinement.repository import (
    CompactTaskRecord,
    ProjectRecord,
    SaveDraftRequest,
    SqliteDraftRepository,
)
from src.label_studio_coco_refinement.runtime import (
    AuthenticatedPrincipal,
    DraftCatalogRequest,
    RefinementRuntime,
)
from src.label_studio_coco_refinement.store import (
    BatchMember,
    BatchRequest,
    BatchResult,
    BatchStatus,
    CommitRequest,
    CommitResult,
    CommitStatus,
    DraftSaveReceipt,
    InferenceReceiptLink,
    canonical_json,
    sha256_file,
    sha256_json,
)


PROJECT_ID = "coco-refinement:train"
PRINCIPAL = "local-operator"
LOCAL_A = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"
LOCAL_B = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603939"
LOCAL_C = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603940"
ROI_KEY = "roi:receipt-1:result-1"


def _digest(character: str) -> str:
    return character * 64


def _source_object(image_id: int) -> dict[str, object]:
    return {
        "region_key": f"train:coco:{image_id}",
        "bbox_2d": [1, 2, 300, 400],
        "category_name": "person",
        "category_id": 1,
        "coco_ann_id": image_id,
    }


def _local_object(
    key: str = LOCAL_A,
    *,
    bbox: tuple[int, int, int, int] = (10, 20, 300, 400),
    category_name: str = "person",
    category_id: int = 1,
    coco_ann_id: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "region_key": key,
        "bbox_2d": list(bbox),
        "category_name": category_name,
        "category_id": category_id,
    }
    if coco_ann_id is not None:
        value["coco_ann_id"] = coco_ann_id
    return value


def _roi_object(
    *,
    bbox: tuple[int, int, int, int] = (100, 120, 350, 460),
    category_name: str = "person",
    category_id: int = 1,
    coco_ann_id: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "region_key": ROI_KEY,
        "bbox_2d": list(bbox),
        "category_name": category_name,
        "category_id": category_id,
        "metadata": {
            "inference_origin": True,
            "receipt_id": "receipt-1",
            "request_id": "request-1",
            "result_id": "result-1",
            "draft_revision": "source-revision-1",
        },
    }
    if coco_ann_id is not None:
        value["coco_ann_id"] = coco_ann_id
    return value


def _task(image_id: int, row: int, *, generation: int = 3) -> CompactTaskRecord:
    baseline = canonicalize_objects([_source_object(image_id)], split="train")
    return CompactTaskRecord(
        project_id=PROJECT_ID,
        identity=NativeTaskIdentity(
            split="train", image_id=image_id, source_row_index=row
        ),
        image_locator=f"train2017/{image_id:012d}.jpg",
        image_width=640,
        image_height=480,
        image_fingerprint=_digest("c"),
        current_generation=generation,
        base_row_hash=_digest(chr(ord("a") + row)),
        committed_result_hash=baseline.result_hash,
    )


def _repository(
    path: Path,
    *,
    tasks: tuple[CompactTaskRecord, ...] | None = None,
) -> SqliteDraftRepository:
    selected = tasks or (_task(7, 0),)
    repository = SqliteDraftRepository(path)
    repository.bootstrap_project(
        ProjectRecord(
            project_id=PROJECT_ID,
            split="train",
            source_fingerprint=_digest("f"),
            task_count=len(selected),
        ),
        selected,
    )
    return repository


def _save(
    repository: SqliteDraftRepository,
    *,
    task: CompactTaskRecord,
    objects: list[dict[str, object]],
    revision: int,
    mutation_id: str,
    committed_objects: list[dict[str, object]] | None = None,
) -> None:
    repository.save_draft(
        SaveDraftRequest(
            project_id=PROJECT_ID,
            task_id=task.task_id,
            mutation_id=mutation_id,
            expected_revision=revision,
            expected_generation=task.current_generation,
            expected_base_row_hash=task.base_row_hash,
            committed=canonicalize_objects(
                (
                    [_source_object(task.identity.image_id)]
                    if committed_objects is None
                    else committed_objects
                ),
                split="train",
            ),
            draft=canonicalize_objects(objects, split="train"),
        )
    )


def _capture(
    repository: SqliteDraftRepository,
):
    return SqliteDraftCatalog(
        repository, current_user_id=PRINCIPAL
    ).capture_current_user_drafts(
        DraftCatalogRequest(
            split="train",
            project_id=PROJECT_ID,
            principal=AuthenticatedPrincipal(
                user_id=PRINCIPAL,
                authenticated=True,
            ),
        )
    )


def _batch_from_capture(
    capture,
    *,
    batch_id: str = "batch-1",
    source_rows: tuple[int, ...] | None = None,
) -> BatchRequest:
    members: list[BatchMember] = []
    selected_rows = source_rows or tuple(range(len(capture.snapshots)))
    for source_row_index, snapshot in zip(
        selected_rows, capture.snapshots, strict=True
    ):
        receipt = DraftSaveReceipt(
            project_id=snapshot.project_id,
            task_id=snapshot.task_id,
            annotation_id=snapshot.annotation_id,
            draft_id=snapshot.draft_id,
            annotation_revision=snapshot.annotation_revision,
            draft_updated_at=snapshot.draft_updated_at,
            semantic_hash=snapshot.semantic_hash,
            result_hash=snapshot.result_hash,
            durable=True,
        )
        request = CommitRequest(
            commit_id=f"{batch_id}:member:{snapshot.task_id}",
            split=snapshot.split,
            image_id=snapshot.image_id,
            project_id=snapshot.project_id,
            task_id=snapshot.task_id,
            annotation_id=snapshot.annotation_id,
            draft_id=snapshot.draft_id,
            annotation_revision=snapshot.annotation_revision,
            draft_updated_at=snapshot.draft_updated_at,
            semantic_hash=snapshot.semantic_hash,
            result_hash=snapshot.result_hash,
            base_row_hash=snapshot.base_row_hash,
            observed_generation=snapshot.observed_generation,
            regions=[_thaw(region) for region in snapshot.regions],
            draft_save=receipt,
            inference_receipts=tuple(snapshot.inference_receipts),
        )
        members.append(BatchMember(source_row_index, request))
    return BatchRequest(
        batch_id=batch_id,
        split="train",
        current_user_id=PRINCIPAL,
        base_generation=capture.base_generation,
        members=tuple(members),
    )


def _thaw(value):
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _batch_payload_hash(batch: BatchRequest) -> str:
    members = []
    for member in sorted(batch.members, key=lambda value: value.source_row_index):
        request = member.request
        receipt = request.draft_save
        members.append(
            {
                "source_row_index": member.source_row_index,
                "request": {
                    "commit_id": request.commit_id,
                    "split": request.split,
                    "image_id": request.image_id,
                    "project_id": request.project_id,
                    "task_id": request.task_id,
                    "annotation_id": request.annotation_id,
                    "draft_id": request.draft_id,
                    "annotation_revision": request.annotation_revision,
                    "draft_updated_at": request.draft_updated_at,
                    "semantic_hash": request.semantic_hash,
                    "result_hash": request.result_hash,
                    "base_row_hash": request.base_row_hash,
                    "observed_generation": request.observed_generation,
                    "regions": list(request.regions),
                    "draft_save": {
                        "project_id": receipt.project_id,
                        "task_id": receipt.task_id,
                        "annotation_id": receipt.annotation_id,
                        "draft_id": receipt.draft_id,
                        "annotation_revision": receipt.annotation_revision,
                        "draft_updated_at": receipt.draft_updated_at,
                        "semantic_hash": receipt.semantic_hash,
                        "result_hash": receipt.result_hash,
                        "durable": receipt.durable,
                    },
                    "inference_receipts": list(request.inference_receipts),
                },
            }
        )
    return sha256_json(
        {
            "batch_id": batch.batch_id,
            "split": batch.split,
            "current_user_id": batch.current_user_id,
            "base_generation": batch.base_generation,
            "members": members,
        }
    )


def _committed_object(region: dict[str, object], object_id: int) -> dict[str, object]:
    value: dict[str, object] = {
        "bbox_2d": list(region["bbox_2d"]),
        "desc": region["category_name"],
        "category_name": region["category_name"],
        "category_id": region["category_id"],
        "coco_ann_id": object_id,
    }
    if "metadata" in region:
        value["metadata"] = region["metadata"]
    return value


def _success_result(
    batch: BatchRequest,
    *,
    mappings: tuple[dict[str, int], ...] | None = None,
) -> BatchResult:
    chosen = mappings or tuple(
        {
            str(region["region_key"]): -(ordinal + 1)
            for ordinal, region in enumerate(member.request.regions)
        }
        for member in batch.members
    )
    results: list[CommitResult] = []
    for member, mapping in zip(batch.members, chosen, strict=True):
        by_key = {
            str(region["region_key"]): dict(region)
            for region in member.request.regions
        }
        committed_objects = [
            _committed_object(by_key[key], object_id)
            for key, object_id in mapping.items()
        ]
        committed_row = {
            "image_id": member.request.image_id,
            "objects": committed_objects,
        }
        results.append(
            CommitResult(
                commit_id=member.request.commit_id,
                status=CommitStatus.COMMITTED,
                split="train",
                image_id=member.request.image_id,
                generation=4,
                row_hash=sha256_json(committed_row),
                semantic_hash=member.request.semantic_hash,
                region_id_mapping=mapping,
                committed_row=committed_row,
            )
        )
    return BatchResult(
        batch_id=batch.batch_id,
        payload_hash=_batch_payload_hash(batch),
        status=BatchStatus.SUCCEEDED,
        split="train",
        generation=4,
        working_sha256=_digest("9"),
        members=tuple(results),
        error=None,
    )


def test_catalog_captures_only_pending_drafts_in_source_order_and_one_generation(
    tmp_path: Path,
) -> None:
    tasks = (_task(7, 0), _task(8, 1), _task(9, 2))
    repository = _repository(tmp_path / "state.sqlite3", tasks=tasks)
    _save(
        repository,
        task=tasks[2],
        objects=[_local_object(LOCAL_C)],
        revision=0,
        mutation_id="save-row-2",
    )
    _save(
        repository,
        task=tasks[0],
        objects=[_local_object(LOCAL_A)],
        revision=0,
        mutation_id="save-row-0",
    )

    capture = _capture(repository)

    assert capture.current_user_id == PRINCIPAL
    assert capture.base_generation == 3
    assert [snapshot.task_id for snapshot in capture.snapshots] == [
        "train:7",
        "train:9",
    ]
    assert [snapshot.image_id for snapshot in capture.snapshots] == [7, 9]
    assert all(snapshot.observed_generation == 3 for snapshot in capture.snapshots)
    assert all(snapshot.annotation_revision == "1" for snapshot in capture.snapshots)


def test_catalog_rejects_principal_or_cross_generation_authority(
    tmp_path: Path,
) -> None:
    tasks = (_task(7, 0), _task(8, 1))
    repository = _repository(tmp_path / "state.sqlite3", tasks=tasks)
    for index, task in enumerate(tasks):
        _save(
            repository,
            task=task,
            objects=[_local_object(LOCAL_A if index == 0 else LOCAL_B)],
            revision=0,
            mutation_id=f"save-{index}",
        )
    with repository._transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE tasks SET current_generation = 4 WHERE task_id = 'train:8'"
        )

    catalog = SqliteDraftCatalog(repository, current_user_id=PRINCIPAL)
    with pytest.raises(AdapterContractError, match="generation"):
        _capture(repository)
    with pytest.raises(AdapterContractError, match="principal"):
        catalog.capture_current_user_drafts(
            DraftCatalogRequest(
                split="train",
                project_id=PROJECT_ID,
                principal=AuthenticatedPrincipal(
                    user_id="other-local-user", authenticated=True
                ),
            )
        )


def test_exact_verifier_uses_historical_authority_after_later_draft_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    task = _task(7, 0)
    repository = _repository(tmp_path / "state.sqlite3", tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured",
    )
    batch = _batch_from_capture(_capture(repository))
    _save(
        repository,
        task=task,
        objects=[_local_object(bbox=(50, 60, 350, 460))],
        revision=1,
        mutation_id="later",
    )
    verifier = SqliteDraftVerifier(repository, current_user_id=PRINCIPAL)

    assert verifier.verify_batch(batch) is True

    original = batch.members[0]
    forged_regions = [dict(region) for region in original.request.regions]
    forged_regions[0]["bbox_2d"] = [100, 100, 500, 500]
    forged_payload = replace(
        batch,
        members=(
            replace(
                original,
                request=replace(original.request, regions=forged_regions),
            ),
        ),
    )
    forged_token = replace(
        batch,
        members=(
            replace(
                original,
                request=replace(original.request, annotation_revision="999"),
            ),
        ),
    )
    assert verifier.verify_batch(forged_payload) is False
    assert verifier.verify_batch(forged_token) is False
    malformed_regions = [dict(region) for region in original.request.regions]
    malformed_regions[0]["bbox_2d"] = [1, 2, 3]
    assert verifier.verify_batch(
        replace(
            batch,
            members=(
                replace(
                    original,
                    request=replace(original.request, regions=malformed_regions),
                ),
            ),
        )
    ) is False
    assert not verifier.verify_batch(replace(batch, current_user_id="other"))


class _ReceiptDelegate:
    def __init__(self, value: object) -> None:
        self.value = value

    def resolve(self, receipt_id: str) -> object:
        return self.value


def _receipt_link(*, receipt_id: str = "receipt-1") -> InferenceReceiptLink:
    return InferenceReceiptLink(
        receipt_id=receipt_id,
        request_id="request-1",
        project_id=PROJECT_ID,
        task_id="train:7",
        image_id=7,
        annotation_id="train:7:annotation",
        current_user_id=PRINCIPAL,
        draft_id="train:7:draft",
        draft_revision="source-revision-1",
        terminal_status="accepted",
        result_region_keys={"result-1": ROI_KEY},
    )


def test_receipt_resolver_delegates_and_fails_closed_on_malformed_links() -> None:
    link = _receipt_link()
    assert SqliteInferenceReceiptResolver(_ReceiptDelegate(link)).resolve(
        "receipt-1"
    ) == link
    assert SqliteInferenceReceiptResolver(_ReceiptDelegate(None)).resolve(
        "receipt-1"
    ) is None
    with pytest.raises(AdapterContractError, match="receipt"):
        SqliteInferenceReceiptResolver(
            _ReceiptDelegate(_receipt_link(receipt_id="other"))
        ).resolve("receipt-1")
    with pytest.raises(AdapterContractError, match="receipt"):
        SqliteInferenceReceiptResolver(_ReceiptDelegate({"receipt_id": "receipt-1"})).resolve(
            "receipt-1"
        )


def test_terminal_success_retires_exact_captured_draft_idempotently_across_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    task = _task(7, 0)
    repository = _repository(path, tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured",
    )
    batch = _batch_from_capture(_capture(repository))
    result = _success_result(batch)
    reconciler = SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    )

    first = reconciler.reconcile_batch(batch, result)
    state = repository.get_task_state(PROJECT_ID, task.task_id)

    assert first.status is BatchStatus.SUCCEEDED
    assert first.tasks[0].retired is True
    assert first.tasks[0].newer_draft_preserved is False
    assert state.revision == 2
    assert state.epoch == 1
    assert state.current_generation == 4
    assert state.base_row_hash == result.members[0].row_hash
    assert state.draft is None

    restarted = SqliteDraftRepository(path)
    replayed = SqliteTerminalReconciler(
        restarted, current_user_id=PRINCIPAL
    ).reconcile_batch(batch, result)
    assert replayed == first
    assert restarted.get_task_state(PROJECT_ID, task.task_id) == state


def test_terminal_reconciliation_rejects_forged_payload_hash_without_mutation(
    tmp_path: Path,
) -> None:
    task = _task(7, 0)
    repository = _repository(tmp_path / "state.sqlite3", tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured",
    )
    batch = _batch_from_capture(_capture(repository))
    before = repository.get_task_state(PROJECT_ID, task.task_id)
    forged = replace(_success_result(batch), payload_hash=_digest("e"))

    with pytest.raises(AdapterContractError, match="terminal result"):
        SqliteTerminalReconciler(
            repository, current_user_id=PRINCIPAL
        ).reconcile_batch(batch, forged)

    assert repository.get_task_state(PROJECT_ID, task.task_id) == before


def test_terminal_success_merges_only_identity_into_newer_draft_without_resurrection(
    tmp_path: Path,
) -> None:
    task = _task(7, 0)
    repository = _repository(tmp_path / "state.sqlite3", tasks=(task,))
    captured_objects = [_local_object(LOCAL_A), _roi_object()]
    _save(
        repository,
        task=task,
        objects=captured_objects,
        revision=0,
        mutation_id="captured",
    )
    batch = _batch_from_capture(_capture(repository))
    newer_objects = [
        _local_object(LOCAL_C, bbox=(5, 6, 100, 120)),
        _roi_object(
            bbox=(300, 310, 700, 710),
            category_name="car",
            category_id=3,
        ),
    ]
    _save(
        repository,
        task=task,
        objects=newer_objects,
        revision=1,
        mutation_id="later",
    )
    before = repository.get_task_state(PROJECT_ID, task.task_id)
    result = _success_result(
        batch,
        mappings=({LOCAL_A: -1, ROI_KEY: -2},),
    )

    receipt = SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    ).reconcile_batch(batch, result)
    after = repository.get_task_state(PROJECT_ID, task.task_id)

    assert receipt.tasks[0].retired is False
    assert receipt.tasks[0].newer_draft_preserved is True
    assert after.revision == before.revision + 1
    assert after.epoch == before.epoch + 1
    assert after.current_generation == 4
    assert after.draft is not None
    objects = after.draft.to_json_regions()
    assert [obj["region_key"] for obj in objects] == [LOCAL_C, ROI_KEY]
    assert LOCAL_A not in {obj["region_key"] for obj in objects}
    assert objects[0] == newer_objects[0]
    assert objects[1] == {**newer_objects[1], "coco_ann_id": -2}


def test_terminal_success_restores_newer_old_baseline_intent_from_mutation_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    task = _task(7, 0)
    repository = _repository(path, tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured-replacement",
    )
    batch = _batch_from_capture(_capture(repository))

    # While the frozen replacement is publishing, the operator explicitly
    # returns to the old committed source object.  It is sparse-retired against
    # generation 3, but that exact requested payload remains permanent history.
    _save(
        repository,
        task=task,
        objects=[_source_object(7)],
        revision=1,
        mutation_id="return-to-old-baseline",
    )
    retired_before_terminal = repository.get_task_state(PROJECT_ID, task.task_id)
    assert retired_before_terminal.revision == 2
    assert retired_before_terminal.draft is None

    result = _success_result(batch)
    first = SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    ).reconcile_batch(batch, result)
    restored = repository.get_task_state(PROJECT_ID, task.task_id)

    assert first.tasks[0].newer_draft_preserved is True
    assert first.tasks[0].retired is False
    assert restored.revision == 3
    assert restored.current_generation == 4
    assert restored.draft == canonicalize_objects([_source_object(7)], split="train")

    restarted = SqliteDraftRepository(path)
    replayed = SqliteTerminalReconciler(
        restarted, current_user_id=PRINCIPAL
    ).reconcile_batch(batch, result)
    assert replayed == first
    assert restarted.get_task_state(PROJECT_ID, task.task_id) == restored


def test_terminal_reconciliation_fails_closed_when_retired_revision_lacks_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    task = _task(7, 0)
    repository = _repository(path, tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured-replacement",
    )
    batch = _batch_from_capture(_capture(repository))
    _save(
        repository,
        task=task,
        objects=[_source_object(7)],
        revision=1,
        mutation_id="legacy-retired-without-request",
    )
    with sqlite3.connect(path) as connection:
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM mutations WHERE mutation_id = ?",
                ("legacy-retired-without-request",),
            ).fetchone()[0]
        )
        response.pop("requested_draft")
        connection.execute(
            "UPDATE mutations SET response_json = ? WHERE mutation_id = ?",
            (json.dumps(response), "legacy-retired-without-request"),
        )
    before = repository.get_task_state(PROJECT_ID, task.task_id)

    with pytest.raises(AdapterContractError, match="requested Draft"):
        SqliteTerminalReconciler(
            repository, current_user_id=PRINCIPAL
        ).reconcile_batch(batch, _success_result(batch))

    assert repository.get_task_state(PROJECT_ID, task.task_id) == before


def test_later_unrelated_draft_rebinds_to_terminal_project_generation_and_verifies(
    tmp_path: Path,
) -> None:
    tasks = (_task(7, 0), _task(8, 1))
    repository = _repository(tmp_path / "state.sqlite3", tasks=tasks)
    _save(
        repository,
        task=tasks[0],
        objects=[_local_object(LOCAL_A)],
        revision=0,
        mutation_id="captured-first-task",
    )
    first_batch = _batch_from_capture(_capture(repository))

    # This task was not in the immutable capture, but is edited while the
    # worker publishes it.  Its row hash is unchanged by that publication.
    _save(
        repository,
        task=tasks[1],
        objects=[_local_object(LOCAL_B)],
        revision=0,
        mutation_id="later-unrelated-task",
    )
    SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    ).reconcile_batch(first_batch, _success_result(first_batch))

    second_capture = _capture(repository)
    assert second_capture.base_generation == 4
    assert [snapshot.task_id for snapshot in second_capture.snapshots] == ["train:8"]
    second_batch = _batch_from_capture(
        second_capture, batch_id="batch-2", source_rows=(1,)
    )
    # The exact requested Draft came from generation 3; compact task history
    # plus the unchanged base-row binding proves its generation-4 rebinding.
    assert SqliteDraftVerifier(
        repository, current_user_id=PRINCIPAL
    ).verify_batch(second_batch)


def test_terminal_failure_preserves_draft_and_is_duplicate_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    task = _task(7, 0)
    repository = _repository(path, tasks=(task,))
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="captured",
    )
    batch = _batch_from_capture(_capture(repository))
    failed = BatchResult(
        batch_id=batch.batch_id,
        payload_hash=_batch_payload_hash(batch),
        status=BatchStatus.FAILED,
        split="train",
        generation=3,
        working_sha256=_digest("9"),
        members=(),
        error="validation failed",
    )
    before = repository.get_task_state(PROJECT_ID, task.task_id)

    first = SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    ).reconcile_batch(batch, failed)
    second = SqliteTerminalReconciler(
        SqliteDraftRepository(path), current_user_id=PRINCIPAL
    ).reconcile_batch(batch, failed)

    assert first == second
    assert first.status is BatchStatus.FAILED
    assert first.tasks == ()
    assert SqliteDraftRepository(path).get_task_state(PROJECT_ID, task.task_id) == before


def test_runtime_store_terminal_round_trip_uses_exact_payload_and_reconciles(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data/coco/rescale_32_1024_bbox/images"
    image_dir = image_root / "train2017"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_path = image_dir / "000000000007.jpg"
    Image.new("RGB", (32, 24), color="red").save(image_path, format="JPEG")
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        canonical_json(
            {
                "images": [
                    "../rescale_32_1024_bbox/images/train2017/000000000007.jpg"
                ],
                "objects": [
                    {
                        "bbox_2d": [10, 20, 300, 400],
                        "desc": "person",
                        "category_id": 1,
                        "category_name": "person",
                        "coco_ann_id": 701,
                    }
                ],
                "width": 32,
                "height": 24,
                "image_id": 7,
                "file_name": "images/train2017/000000000007.jpg",
                "metadata": {"source": "coco2017", "split": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workspace = bootstrap_workspace(
        tmp_path,
        runtime_root=tmp_path / "runtime",
        source_contracts=(
            BootstrapSourceContract(
                split="train",
                source_path=source,
                image_root=image_root,
                expected_source_sha256=sha256_file(source),
                expected_row_count=1,
            ),
        ),
    )
    repository = workspace.repository
    split = workspace.splits["train"]
    task = split.tasks[0]
    _save(
        repository,
        task=task,
        objects=[_local_object()],
        revision=0,
        mutation_id="real-store-draft",
        committed_objects=[
            {
                "region_key": "train:coco:701",
                "bbox_2d": [10, 20, 300, 400],
                "category_name": "person",
                "category_id": 1,
                "coco_ann_id": 701,
            }
        ],
    )
    catalog = SqliteDraftCatalog(repository, current_user_id=PRINCIPAL)
    verifier = SqliteDraftVerifier(repository, current_user_id=PRINCIPAL)
    split.store.annotation_verifier = verifier
    split.store.inference_receipt_resolver = SqliteInferenceReceiptResolver(
        _ReceiptDelegate(None)
    )
    capture = _capture(repository)
    expected_request = _batch_from_capture(capture, batch_id="real-store-batch")
    runtime = RefinementRuntime(
        catalog=catalog,
        stores={"train": split.store},
        project_ids={"train": PROJECT_ID},
    )

    enqueue = runtime.capture_and_enqueue(
        split="train",
        batch_id="real-store-batch",
        principal=AuthenticatedPrincipal(user_id=PRINCIPAL, authenticated=True),
    )
    assert enqueue.payload_hash == _batch_payload_hash(expected_request)
    terminal = split.store.process_next_batch()
    assert terminal is not None
    assert terminal.status is BatchStatus.SUCCEEDED

    receipt = SqliteTerminalReconciler(
        repository, current_user_id=PRINCIPAL
    ).reconcile_batch(expected_request, terminal)
    assert receipt.status is BatchStatus.SUCCEEDED
    assert receipt.tasks[0].retired is True
    state = repository.get_task_state(PROJECT_ID, task.task_id)
    assert state.current_generation == 1
    assert state.draft is None
