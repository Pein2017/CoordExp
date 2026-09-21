from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

from src.label_studio_coco_refinement.store import BatchMember, BatchRequest, BatchStatus, WorkingDatasetStore


ROOT = Path("/data/CoordExp/outputs/label_studio_coco_refinement/async-batch-v1")
PROBE_PATH = ROOT / "run_async_batch_probe.py"
SPEC = importlib.util.spec_from_file_location("async_batch_probe_support", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def run_concurrent_retry(source: Path) -> dict:
    runtime = ROOT / "final-concurrent-retry/runtime"
    verifier = support.AcceptingAnnotationVerifier()
    resolver = support.EmptyInferenceResolver()
    boot = WorkingDatasetStore.bootstrap(
        support.make_spec(source, runtime, "final-concurrent-retry"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    store = boot.store
    peer = WorkingDatasetStore(
        store.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    seeds = list(itertools.islice(store.iter_task_seeds(), 2))
    request = support.request_for_seed(
        store,
        seeds[0],
        batch_id="final-active",
        ordinal=0,
    )
    batch = BatchRequest(
        batch_id="final-active",
        split="train",
        base_generation=0,
        members=(BatchMember(seeds[0].source_row_index, request),),
    )
    other_request = support.request_for_seed(
        peer,
        seeds[1],
        batch_id="final-other",
        ordinal=1,
    )
    other = BatchRequest(
        batch_id="final-other",
        split="train",
        base_generation=0,
        members=(BatchMember(seeds[1].source_row_index, other_request),),
    )
    accepted = store.enqueue_batch(batch)
    queue_after_enqueue = store.queue_path.read_bytes()
    verifier_count_after_enqueue = len(verifier.calls)

    queued_started = time.perf_counter()
    with store._exclusive_lock():
        exact_queued = peer.enqueue_batch(copy.deepcopy(batch))
        active_queued = peer.enqueue_batch(copy.deepcopy(other))
    queued_seconds = time.perf_counter() - queued_started
    queue_after_queued_retries = store.queue_path.read_bytes()

    with store._exclusive_queue_lock():
        store._append_queue_record(
            {
                "kind": "claim",
                "batch_id": accepted.batch_id,
                "payload_hash": accepted.payload_hash,
            },
            "claim",
        )
    queue_after_claim = store.queue_path.read_bytes()
    running_started = time.perf_counter()
    with store._exclusive_lock():
        exact_running = peer.enqueue_batch(copy.deepcopy(batch))
        active_running = peer.enqueue_batch(copy.deepcopy(other))
    running_seconds = time.perf_counter() - running_started
    queue_after_running_retries = store.queue_path.read_bytes()
    assertions = {
        "initial_queued": accepted.status is BatchStatus.QUEUED,
        "exact_queued": exact_queued.status is BatchStatus.QUEUED,
        "active_queued_returns_same": active_queued == exact_queued == accepted,
        "queued_retries_do_not_append": queue_after_queued_retries == queue_after_enqueue,
        "exact_running": exact_running.status is BatchStatus.RUNNING,
        "active_running_returns_same": active_running == exact_running,
        "running_retries_do_not_append": queue_after_running_retries == queue_after_claim,
        "retry_fast_path_skips_live_verifier": len(verifier.calls)
        == verifier_count_after_enqueue,
        "working_generation_still_zero": json.loads(store.manifest_path.read_text())[
            "generation"
        ]
        == 0,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return {
        "runtime": str(runtime),
        "accepted": support.asdict(accepted),
        "exact_queued": support.asdict(exact_queued),
        "active_queued": support.asdict(active_queued),
        "exact_running": support.asdict(exact_running),
        "active_running": support.asdict(active_running),
        "queued_retry_pair_seconds": queued_seconds,
        "running_retry_pair_seconds": running_seconds,
        "queue_after_enqueue": {
            "bytes": len(queue_after_enqueue),
            "sha256": support.sha256_bytes(queue_after_enqueue),
        },
        "queue_after_claim": {
            "bytes": len(queue_after_claim),
            "sha256": support.sha256_bytes(queue_after_claim),
        },
        "verifier_call_count": len(verifier.calls),
        "assertions": assertions,
        "worker_not_called": True,
    }


def main() -> None:
    store_hash = support.sha256_file(support.STORE_PY)
    test_hash = support.sha256_file(support.TEST_STORE_PY)
    prefix = support.read_source_prefix(10_000)
    cases = {}
    for count in (1_000, 10_000):
        case_root = ROOT / f"final-case-{count}"
        source = case_root / "source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
        support.write_slice(source, prefix[:count])
        cases[str(count)] = support.run_case(
            name=f"final-{count}",
            source=source,
            runtime=case_root / "runtime",
            expected_rows=count,
        )
    retry_source = ROOT / "final-concurrent-retry/source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    support.write_slice(retry_source, prefix[:1_000])
    receipt = {
        "command": "PYTHONPATH=. /root/miniconda3/envs/ms/bin/python outputs/label_studio_coco_refinement/async-batch-v1/run_final_bounded_refresh.py",
        "code_state": {
            "store_py_sha256": store_hash,
            "test_store_py_sha256": test_hash,
        },
        "cases": cases,
        "concurrent_retry": run_concurrent_retry(retry_source),
        "canonical_source_after": support.source_receipt(),
        "pre_final_full_evidence": {
            "receipt": str(ROOT / "receipt.json"),
            "store_py_sha256": "5cc77532703a580b3fe59de30b5113e8115b6b42201d00c1529f5686252f25d1",
            "test_store_py_sha256": "f309bc56ac9ae9f9728053f4aca80e2d7d532a3e45dd546d2f56baa7da029cb4",
            "not_final_code_execution": True,
            "full_worker_seconds": 12.935362521559,
            "full_amortized_seconds_per_member": 1.2935362521559,
        },
    }
    if support.sha256_file(support.STORE_PY) != store_hash or support.sha256_file(
        support.TEST_STORE_PY
    ) != test_hash:
        raise AssertionError("code state drifted during final bounded refresh")
    destination = ROOT / "final_bounded_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(destination),
                "code_state": receipt["code_state"],
                "cases": {
                    name: {
                        "enqueue_seconds": case["batch"]["enqueue"]["seconds"],
                        "worker_seconds": case["batch"]["worker"]["seconds"],
                    }
                    for name, case in cases.items()
                },
                "concurrent_retry": receipt["concurrent_retry"]["assertions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
