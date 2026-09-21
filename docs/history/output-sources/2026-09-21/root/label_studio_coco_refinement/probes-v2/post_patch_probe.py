from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from src.label_studio_coco_refinement.store import CommitConflictError, CommitRequest, WorkingDatasetStore


ROOT = Path("/data/CoordExp/outputs/label_studio_coco_refinement/probes-v2")
MODULE_PATH = ROOT / "run_attestation.py"
SPEC = importlib.util.spec_from_file_location("run_attestation_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ROOT / "rows-1000/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    runtime = ROOT / "post-patch-1000/runtime"
    verifier = probe.AcceptingAnnotationVerifier()
    resolver = probe.EmptyInferenceReceiptResolver()
    source_hashes_before = probe.hashes(probe.SELECTED_FAMILY)
    started = time.perf_counter()
    boot = WorkingDatasetStore.bootstrap(
        probe.make_spec(source, runtime, "post-patch-1000"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    bootstrap_seconds = time.perf_counter() - started
    request, identity = probe.make_request(
        boot.store,
        commit_id="post-patch-retry-identity",
        revision=1,
        add_human_region=True,
        toggle=1,
    )
    started = time.perf_counter()
    first = boot.store.commit(request)
    first_seconds = time.perf_counter() - started
    journal_after_first = boot.store.journal_path.read_bytes()
    started = time.perf_counter()
    identical = boot.store.commit(request)
    identical_retry_seconds = time.perf_counter() - started
    journal_after_identical = boot.store.journal_path.read_bytes()

    changed_regions = [dict(region) for region in request.regions]
    changed_regions[0]["metadata"] = {"human_note": "retry-metadata-changed"}
    changed_request = CommitRequest(**{**request.__dict__, "regions": changed_regions})
    started = time.perf_counter()
    try:
        boot.store.commit(changed_request)
        conflict = {"raised": False}
    except Exception as error:
        conflict = {
            "raised": True,
            "type": type(error).__name__,
            "message": str(error),
            "is_commit_conflict": isinstance(error, CommitConflictError),
        }
    conflict_seconds = time.perf_counter() - started
    journal_after_conflict = boot.store.journal_path.read_bytes()
    reopened1 = WorkingDatasetStore(
        boot.store.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    recovery1 = {
        "status": reopened1.status(request.commit_id).value,
        "generation": reopened1.restore_draft(request.image_id).generation,
        "working_sha256": file_hash(reopened1.working_path),
        "journal_sha256": file_hash(reopened1.journal_path),
    }
    reopened2 = WorkingDatasetStore(
        boot.store.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    recovery2 = {
        "status": reopened2.status(request.commit_id).value,
        "generation": reopened2.restore_draft(request.image_id).generation,
        "working_sha256": file_hash(reopened2.working_path),
        "journal_sha256": file_hash(reopened2.journal_path),
    }

    receipt = {
        "command": "PYTHONPATH=. /root/miniconda3/envs/ms/bin/python outputs/label_studio_coco_refinement/probes-v2/post_patch_probe.py",
        "code_state": {
            "git_head": "c06c188349cece5d922358baaa6ae507adce3203",
            "store_sha256": file_hash(Path("/data/CoordExp/src/label_studio_coco_refinement/store.py")),
            "test_store_sha256": file_hash(Path("/data/CoordExp/tests/label_studio_coco_refinement/test_store.py")),
        },
        "source_path": str(source),
        "source_sha256": file_hash(source),
        "bootstrap_seconds": bootstrap_seconds,
        "task_count": boot.task_count,
        "identity": identity,
        "first_commit": {
            "seconds": first_seconds,
            "status": first.status.value,
            "generation": first.generation,
            "row_hash": first.row_hash,
        },
        "identical_retry": {
            "seconds": identical_retry_seconds,
            "same_result": identical == first,
            "journal_unchanged": journal_after_identical == journal_after_first,
        },
        "changed_metadata_retry": {
            **conflict,
            "seconds": conflict_seconds,
            "journal_unchanged": journal_after_conflict == journal_after_first,
        },
        "recovery_reopen_1": recovery1,
        "recovery_reopen_2": recovery2,
        "recovery_stable": recovery1 == recovery2,
        "verifier_calls": [dataclasses.asdict(call) for call in verifier.calls],
        "resolver_calls": resolver.calls,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": probe.hashes(probe.SELECTED_FAMILY),
    }
    receipt["source_hashes_unchanged"] = (
        receipt["source_hashes_before"] == receipt["source_hashes_after"]
    )
    destination = ROOT / "post_patch_receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
