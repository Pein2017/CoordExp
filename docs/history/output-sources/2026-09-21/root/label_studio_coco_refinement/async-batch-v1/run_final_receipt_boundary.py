from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

from src.label_studio_coco_refinement.store import (
    BatchMember,
    BatchRequest,
    BatchStatus,
    InjectedCrash,
    WorkingDatasetStore,
)


ROOT = Path("/data/CoordExp/outputs/label_studio_coco_refinement/async-batch-v1")
SUPPORT_PATH = ROOT / "run_async_batch_probe.py"
FINAL_STORE_HASH = "77ac5fa04dba619b10f783506d8e598e1ff62160932a98fa021b3f0179569d39"
FINAL_TEST_HASH = "a6841ea3d69183de66925ab84a0dbaec6097710a5a28cddbec1a64410c69efdf"

SPEC = importlib.util.spec_from_file_location("receipt_boundary_support", SUPPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


class CrashAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.seen: list[str] = []

    def __call__(self, boundary: str) -> None:
        self.seen.append(boundary)
        if boundary == self.boundary:
            raise InjectedCrash(boundary)


def queue_receipt(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    return {
        "bytes": len(raw),
        "sha256": support.sha256_bytes(raw),
        "kinds": [record["kind"] for record in records],
        "records": len(records),
    }


def identity(receipt: object) -> dict[str, object]:
    return support.asdict(receipt)


def main() -> None:
    code_before = {
        "store_py_sha256": support.sha256_file(support.STORE_PY),
        "test_store_py_sha256": support.sha256_file(support.TEST_STORE_PY),
    }
    if code_before != {
        "store_py_sha256": FINAL_STORE_HASH,
        "test_store_py_sha256": FINAL_TEST_HASH,
    }:
        raise AssertionError(f"unexpected code state: {code_before}")
    canonical_before = support.source_receipt()

    root = ROOT / "final-receipt-boundary"
    source = root / "source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    support.write_slice(source, support.read_source_prefix(3))
    verifier = support.AcceptingAnnotationVerifier()
    resolver = support.EmptyInferenceResolver()
    boot = WorkingDatasetStore.bootstrap(
        support.make_spec(source, root / "runtime", "final-receipt-boundary"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    worker = boot.store
    peer = WorkingDatasetStore(
        worker.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    seeds = list(itertools.islice(worker.iter_task_seeds(), 2))
    active_request = support.request_for_seed(
        worker,
        seeds[0],
        batch_id="receipt-active",
        ordinal=0,
    )
    active_batch = BatchRequest(
        batch_id="receipt-active",
        split="train",
        base_generation=0,
        members=(BatchMember(seeds[0].source_row_index, active_request),),
    )
    different_request = support.request_for_seed(
        peer,
        seeds[1],
        batch_id="receipt-different",
        ordinal=1,
    )
    different_batch = BatchRequest(
        batch_id="receipt-different",
        split="train",
        base_generation=0,
        members=(BatchMember(seeds[1].source_row_index, different_request),),
    )
    accepted = worker.enqueue_batch(active_batch)

    queue_queued_before = queue_receipt(worker.queue_path)
    queued_started = time.perf_counter()
    with worker._exclusive_lock():
        exact_queued = peer.enqueue_batch(copy.deepcopy(active_batch))
        different_queued = peer.enqueue_batch(copy.deepcopy(different_batch))
    queued_seconds = time.perf_counter() - queued_started
    queue_queued_after = queue_receipt(worker.queue_path)

    with worker._exclusive_queue_lock():
        worker._append_queue_record(
            {
                "kind": "claim",
                "batch_id": accepted.batch_id,
                "payload_hash": accepted.payload_hash,
            },
            "claim",
        )
    queue_running_before = queue_receipt(worker.queue_path)
    running_started = time.perf_counter()
    with worker._exclusive_lock():
        exact_running = peer.enqueue_batch(copy.deepcopy(active_batch))
        different_running = peer.enqueue_batch(copy.deepcopy(different_batch))
    running_seconds = time.perf_counter() - running_started
    queue_running_after = queue_receipt(worker.queue_path)

    crash = CrashAt("batch_terminal_journal_fsynced")
    worker._fault_injector = crash
    crash_started = time.perf_counter()
    try:
        worker.process_next_batch()
        crash_outcome = "returned"
    except BaseException as error:
        crash_outcome = f"{type(error).__name__}: {error}"
    crash_seconds = time.perf_counter() - crash_started
    queue_gap_before = queue_receipt(worker.queue_path)
    journal_kinds = [
        json.loads(line)["kind"] for line in worker.journal_path.read_text().splitlines()
    ]
    manifest_gap = json.loads(worker.manifest_path.read_text())

    gap_started = time.perf_counter()
    exact_gap = peer.enqueue_batch(copy.deepcopy(active_batch))
    different_gap = peer.enqueue_batch(copy.deepcopy(different_batch))
    gap_seconds = time.perf_counter() - gap_started
    queue_gap_after = queue_receipt(worker.queue_path)

    accepted_identity = identity(accepted)
    assertions = {
        "a_exact_queued": exact_queued.status is BatchStatus.QUEUED,
        "a_different_queued_original_identity": identity(different_queued)
        == identity(exact_queued)
        == accepted_identity,
        "a_queued_queue_bytes_unchanged": queue_queued_before == queue_queued_after,
        "a_exact_running": exact_running.status is BatchStatus.RUNNING,
        "a_different_running_original_identity": identity(different_running)
        == identity(exact_running),
        "a_running_queue_bytes_unchanged": queue_running_before == queue_running_after,
        "b_crashed_at_terminal_fsync": crash_outcome
        == "InjectedCrash: batch_terminal_journal_fsynced",
        "b_journal_terminal_durable": "batch_terminal" in journal_kinds,
        "b_queue_terminal_missing": "queue_terminal" not in queue_gap_before["kinds"],
        "b_lock_released_exact_reconciling": exact_gap.status is BatchStatus.RECONCILING,
        "b_lock_released_different_reconciling": different_gap.status
        is BatchStatus.RECONCILING,
        "b_original_identity_preserved": identity(exact_gap) == identity(different_gap)
        and exact_gap.batch_id == accepted.batch_id
        and exact_gap.payload_hash == accepted.payload_hash,
        "c_gap_queue_bytes_unchanged": queue_gap_before == queue_gap_after,
        "candidate_generation_published_once": manifest_gap["generation"] == 1,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    code_after = {
        "store_py_sha256": support.sha256_file(support.STORE_PY),
        "test_store_py_sha256": support.sha256_file(support.TEST_STORE_PY),
    }
    canonical_after = support.source_receipt()
    if code_after != code_before:
        raise AssertionError("code state drifted during receipt boundary probe")
    if canonical_after != canonical_before:
        raise AssertionError("canonical source changed during receipt boundary probe")

    receipt = {
        "command": "PYTHONPATH=. /root/miniconda3/envs/ms/bin/python outputs/label_studio_coco_refinement/async-batch-v1/run_final_receipt_boundary.py",
        "exit": 0,
        "code_before": code_before,
        "code_after": code_after,
        "canonical_source_before": canonical_before,
        "canonical_source_after": canonical_after,
        "bootstrap": {
            "rows": boot.task_count,
            "runtime": str(root / "runtime"),
            "managed_images": support.managed_link_receipt(worker),
        },
        "accepted_identity": accepted_identity,
        "a_txn_ex_locked": {
            "queued": {
                "exact": identity(exact_queued),
                "different": identity(different_queued),
                "pair_seconds": queued_seconds,
                "queue_before": queue_queued_before,
                "queue_after": queue_queued_after,
            },
            "running": {
                "exact": identity(exact_running),
                "different": identity(different_running),
                "pair_seconds": running_seconds,
                "queue_before": queue_running_before,
                "queue_after": queue_running_after,
            },
        },
        "b_terminal_projection_gap_lock_released": {
            "fault_boundary": crash.boundary,
            "faults_seen": crash.seen,
            "process_seconds_until_crash": crash_seconds,
            "crash_outcome": crash_outcome,
            "journal_kinds": journal_kinds,
            "manifest_generation": manifest_gap["generation"],
            "manifest_working_sha256": manifest_gap["working_sha256"],
            "exact": identity(exact_gap),
            "different": identity(different_gap),
            "pair_seconds": gap_seconds,
            "queue_before": queue_gap_before,
            "queue_after": queue_gap_after,
        },
        "assertions": assertions,
        "historical_receipts": {
            "pre_final_full": {
                "path": str(ROOT / "receipt.json"),
                "store_hash": "5cc77532703a580b3fe59de30b5113e8115b6b42201d00c1529f5686252f25d1",
                "classification": "pre-final background-path evidence only",
            },
            "intermediate_bounded": {
                "path": str(ROOT / "final_bounded_receipt.json"),
                "store_hash": "578309999d73e30d62cb56229d94437a1d2ef6ec546817572ac8cfe696e8e2b1",
                "classification": "intermediate receipt-boundary hash; not final",
            },
        },
    }
    destination = ROOT / "final_receipt_boundary.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(destination),
                "code_state": code_after,
                "assertions": assertions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
