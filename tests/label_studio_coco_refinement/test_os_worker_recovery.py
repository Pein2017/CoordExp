from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.store import (
    AuthoritativeDraftIdentity,
    BatchMember,
    BatchRequest,
    BatchStatus,
    BootstrapSpec,
    CommitRequest,
    DraftSaveReceipt,
    InferenceReceiptLink,
    StoreBusyError,
    WorkingDatasetStore,
    canonical_json,
    semantic_hash,
    sha256_file,
    sha256_json,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/status").exists(),
    reason="the regression requires POSIX signals and Linux process-state receipts",
)

_WAIT_SECONDS = 5.0
_OBSERVATION_SECONDS = 1.0


class _AcceptingAnnotationVerifier:
    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        return True


class _EmptyInferenceReceiptResolver:
    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        return None


@dataclass(frozen=True)
class _RecoveryCut:
    name: str
    boundary: str
    expected_status: BatchStatus
    expected_generation: int
    queue_lock_held: bool = False


_RECOVERY_CUTS = (
    _RecoveryCut(
        "before-working-rename",
        "batch_prepared_journal_fsynced",
        BatchStatus.FAILED,
        0,
    ),
    _RecoveryCut(
        "after-working-rename",
        "batch_working_replaced",
        BatchStatus.SUCCEEDED,
        1,
    ),
    _RecoveryCut(
        "after-working-directory-fsync",
        "batch_working_directory_fsynced",
        BatchStatus.SUCCEEDED,
        1,
    ),
    _RecoveryCut(
        "after-manifest-replacement",
        "manifest_replaced",
        BatchStatus.SUCCEEDED,
        1,
    ),
    _RecoveryCut(
        "after-manifest-directory-fsync",
        "manifest_directory_fsynced",
        BatchStatus.SUCCEEDED,
        1,
    ),
    _RecoveryCut(
        "after-authoritative-journal-terminal",
        "batch_terminal_journal_fsynced",
        BatchStatus.SUCCEEDED,
        1,
    ),
    _RecoveryCut(
        "after-queue-terminal-projection",
        "queue_terminal_queue_fsynced",
        BatchStatus.SUCCEEDED,
        1,
        queue_lock_held=True,
    ),
)


def _row(image_id: int) -> dict[str, Any]:
    return {
        "images": [f"../rescale_32_1024_bbox/images/train2017/{image_id:012d}.jpg"],
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
        "file_name": f"images/train2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": "train"},
    }


def _open_store(split_dir: Path) -> WorkingDatasetStore:
    return WorkingDatasetStore(
        split_dir,
        annotation_verifier=_AcceptingAnnotationVerifier(),
        inference_receipt_resolver=_EmptyInferenceReceiptResolver(),
    )


def _bootstrap_tiny_store(tmp_path: Path) -> WorkingDatasetStore:
    source_dir = tmp_path / "public_data/coco/rescale_32_1024_bbox_len12000"
    image_root = tmp_path / "public_data/coco/rescale_32_1024_bbox/images"
    image_split = image_root / "train2017"
    source_dir.mkdir(parents=True)
    image_split.mkdir(parents=True)
    for image_id in (1, 2, 3):
        (image_split / f"{image_id:012d}.jpg").write_bytes(f"image-{image_id}".encode())
    source = source_dir / "train.norm.jsonl"
    source.write_text(
        "".join(canonical_json(_row(image_id)) + "\n" for image_id in (1, 2, 3)),
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
        annotation_verifier=_AcceptingAnnotationVerifier(),
        inference_receipt_resolver=_EmptyInferenceReceiptResolver(),
    )
    return result.store


def _regions(store: WorkingDatasetStore, image_id: int) -> list[dict[str, Any]]:
    restored = store.restore_draft(image_id)
    source = restored.row["objects"][0]
    source_id = int(source["coco_ann_id"])
    region_key = next(
        key
        for key, coco_ann_id in restored.region_id_mapping.items()
        if coco_ann_id == source_id
    )
    edited = dict(source)
    edited["region_key"] = region_key
    edited["bbox_2d"] = [11, 21, 31, 41]
    return [
        edited,
        {
            "region_key": f"drawn:os-kill-{image_id}",
            "bbox_2d": [100, 110, 120, 130],
            "desc": "dog",
            "category_name": "dog",
            "category_id": 18,
            "creation_ordinal": image_id,
        },
    ]


def _commit_request(
    store: WorkingDatasetStore,
    *,
    batch_id: str,
    image_id: int,
) -> CommitRequest:
    restored = store.restore_draft(image_id)
    regions = _regions(store, image_id)
    projection_hash = semantic_hash(regions)
    receipt = DraftSaveReceipt(
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision="annotation-v1",
        draft_updated_at="2026-07-16T00:00:00Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
    )
    return CommitRequest(
        commit_id=f"{batch_id}:member:{image_id}",
        split="train",
        image_id=image_id,
        project_id="project-train",
        task_id=f"train:{image_id}",
        annotation_id=f"annotation-{image_id}",
        draft_id=f"draft-{image_id}",
        annotation_revision="annotation-v1",
        draft_updated_at="2026-07-16T00:00:00Z",
        semantic_hash=projection_hash,
        result_hash=sha256_json(regions),
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
        draft_save=receipt,
        inference_receipts=(),
    )


def _batch_request(
    store: WorkingDatasetStore,
    *,
    batch_id: str,
    image_ids: tuple[int, ...],
) -> BatchRequest:
    return BatchRequest(
        batch_id=batch_id,
        split="train",
        current_user_id="reviewer",
        base_generation=store.restore_draft(image_ids[0]).generation,
        members=tuple(
            BatchMember(
                source_row_index=image_id - 1,
                request=_commit_request(
                    store,
                    batch_id=batch_id,
                    image_id=image_id,
                ),
            )
            for image_id in image_ids
        ),
    )


def _write_fsynced_marker(marker_path: Path, boundary: str) -> None:
    descriptor = os.open(
        marker_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, (boundary + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stopping_worker(split_dir: Path, boundary: str, marker_path: Path) -> None:
    store = _open_store(split_dir)

    def stop_at_boundary(observed: str) -> None:
        if observed != boundary:
            return
        _write_fsynced_marker(marker_path, observed)
        os.kill(os.getpid(), signal.SIGSTOP)
        raise AssertionError("the stopped crash worker unexpectedly resumed")

    store._fault_injector = stop_at_boundary
    store.process_next_batch()


def _process_state(process_id: int) -> str | None:
    status_path = Path(f"/proc/{process_id}/status")
    try:
        status = status_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    state_line = next(
        (line for line in status.splitlines() if line.startswith("State:")),
        "",
    )
    return state_line.split()[1] if state_line else None


def _wait_for_stopped_worker(
    process: multiprocessing.Process,
    marker_path: Path,
    boundary: str,
) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        if marker_path.exists():
            assert marker_path.read_text(encoding="utf-8") == boundary + "\n"
            if process.pid is not None and _process_state(process.pid) == "T":
                return
        if not process.is_alive():
            raise AssertionError(
                f"crash worker exited before stopping at {boundary}: "
                f"exitcode={process.exitcode}"
            )
        time.sleep(0.01)
    raise AssertionError(f"crash worker did not stop at {boundary} within timeout")


def _observation_entry(
    sender: Any,
    observation: Callable[[], dict[str, Any]],
) -> None:
    try:
        payload = ("ok", observation())
    except Exception as exc:  # noqa: BLE001 - the exception type is the receipt
        payload = (
            "error",
            {"type": type(exc).__name__, "message": str(exc)},
        )
    try:
        sender.send(payload)
    finally:
        sender.close()


def _observe_with_timeout(
    context: multiprocessing.context.BaseContext,
    observation: Callable[[], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_observation_entry, args=(sender, observation))
    process.start()
    sender.close()
    try:
        if not receiver.poll(_OBSERVATION_SECONDS):
            process.kill()
            process.join(_WAIT_SECONDS)
            assert process.exitcode == -signal.SIGKILL
            return "timeout", {"seconds": _OBSERVATION_SECONDS}
        result = receiver.recv()
        process.join(_WAIT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(_WAIT_SECONDS)
            raise AssertionError("observation process did not exit after sending")
        assert process.exitcode == 0
        return result
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
            process.join(_WAIT_SECONDS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _observe_admission(
    peer: WorkingDatasetStore,
    request: BatchRequest,
) -> dict[str, Any]:
    receipt = peer.enqueue_batch(request)
    return {"batch_id": receipt.batch_id, "status": receipt.status.value}


@pytest.mark.parametrize("cut", _RECOVERY_CUTS, ids=lambda cut: cut.name)
def test_os_killed_batch_worker_recovers_exactly_once(
    tmp_path: Path,
    cut: _RecoveryCut,
) -> None:
    started = time.monotonic()
    store = _bootstrap_tiny_store(tmp_path)
    batch_id = "os-kill-batch"
    before_rows = _read_jsonl(store.working_path)
    enqueue = store.enqueue_batch(
        _batch_request(store, batch_id=batch_id, image_ids=(1, 2))
    )
    peer = _open_store(store.split_dir)
    second_request = _batch_request(
        peer,
        batch_id="must-not-be-admitted",
        image_ids=(3,),
    )
    marker_path = tmp_path / f"{cut.name}.marker"
    context = multiprocessing.get_context("fork")
    worker = context.Process(
        target=_stopping_worker,
        args=(store.split_dir, cut.boundary, marker_path),
    )
    worker.start()

    try:
        _wait_for_stopped_worker(worker, marker_path, cut.boundary)

        reader_observation = _observe_with_timeout(
            context,
            lambda: {
                "generations": [
                    draft.generation for draft in peer.restore_drafts((1, 2, 3))
                ]
            },
        )
        status_observation = _observe_with_timeout(
            context,
            lambda: {
                "status": peer.get_batch_status(batch_id).status.value,
            },
        )
        admission_observation = _observe_with_timeout(
            context,
            lambda: _observe_admission(peer, second_request),
        )

        for observation in (reader_observation, status_observation):
            assert observation[0] == "error"
            assert observation[1]["type"] == StoreBusyError.__name__
        if cut.queue_lock_held:
            assert admission_observation[0] == "timeout"
        else:
            assert admission_observation == (
                "ok",
                {"batch_id": batch_id, "status": BatchStatus.RUNNING.value},
            )
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(_WAIT_SECONDS)
        if worker.is_alive():
            worker.kill()
            worker.join(_WAIT_SECONDS)

    assert worker.exitcode == -signal.SIGKILL

    recovered = _open_store(store.split_dir)
    # The pre-rename cut strands the durable candidate; recovery must remove
    # that residue, and no later cut may leave another candidate behind.
    assert not list(recovered.split_dir.glob(".working.batch.*"))
    status = recovered.get_batch_status(batch_id)
    result = recovered.get_batch_result(batch_id)
    manifest = json.loads(recovered.manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(recovered.working_path)
    journal = _read_jsonl(recovered.journal_path)
    queue = _read_jsonl(recovered.queue_path)

    prepared = [record for record in journal if record["kind"] == "batch_prepared"]
    terminals = [record for record in journal if record["kind"] == "batch_terminal"]
    queue_terminals = [record for record in queue if record["kind"] == "queue_terminal"]
    assert len(prepared) == len(terminals) == len(queue_terminals) == 1
    prepared_record = prepared[0]
    terminal = terminals[0]
    queue_terminal = queue_terminals[0]

    expected_rows = list(before_rows)
    if cut.expected_status is BatchStatus.SUCCEEDED:
        for member in prepared_record["members"]:
            expected_rows[int(member["source_row_index"])] = member["after_row"]
    assert rows == expected_rows
    assert rows[2] == before_rows[2]
    assert all(
        (rows[index] == before_rows[index])
        == (cut.expected_status is BatchStatus.FAILED)
        for index in (0, 1)
    )

    negative_ids = [
        int(obj["coco_ann_id"])
        for row in rows
        for obj in row["objects"]
        if int(obj["coco_ann_id"]) < 0
    ]
    reservation_ids = [
        int(record["coco_ann_id"])
        for record in journal
        if record["kind"] == "reservation"
    ]
    assert len(negative_ids) == len(set(negative_ids))
    assert len(reservation_ids) == len(set(reservation_ids)) == 2
    if cut.expected_status is BatchStatus.SUCCEEDED:
        assert set(negative_ids) == set(reservation_ids)
    else:
        assert negative_ids == []

    assert status.status is result.status is cut.expected_status
    assert status.generation == result.generation == cut.expected_generation
    assert manifest["generation"] == cut.expected_generation
    assert manifest["working_sha256"] == result.working_sha256
    assert sha256_file(recovered.working_path) == result.working_sha256
    assert terminal["prepared_record_hash"] == prepared_record["record_hash"]
    for key in (
        "batch_id",
        "payload_hash",
        "status",
        "generation",
        "working_sha256",
        "error",
    ):
        assert queue_terminal.get(key) == terminal.get(key)
    assert terminal["batch_id"] == enqueue.batch_id == batch_id
    assert terminal["status"] == cut.expected_status.value
    assert len(result.members) == (
        2 if cut.expected_status is BatchStatus.SUCCEEDED else 0
    )
    assert [record["kind"] for record in queue].count("enqueue") == 1
    assert [record["kind"] for record in queue].count("claim") == 1
    assert not any(
        record.get("batch_id") == second_request.batch_id for record in queue
    )

    recovered_bytes = {
        path.name: path.read_bytes()
        for path in (
            recovered.working_path,
            recovered.manifest_path,
            recovered.journal_path,
            recovered.queue_path,
        )
    }
    reopened = _open_store(store.split_dir)
    assert reopened.get_batch_status(batch_id) == status
    assert reopened.get_batch_result(batch_id) == result
    assert recovered_bytes == {
        path.name: path.read_bytes()
        for path in (
            reopened.working_path,
            reopened.manifest_path,
            reopened.journal_path,
            reopened.queue_path,
        )
    }

    elapsed = time.monotonic() - started
    print(
        f"cut={cut.name} status={result.status.value} "
        f"generation={result.generation} sha256={result.working_sha256} "
        f"runtime_s={elapsed:.3f}"
    )
