from __future__ import annotations

import hashlib
import itertools
import json
import os
import resource
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

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
    WorkingDatasetStore,
    canonical_json,
    semantic_hash,
    sha256_file,
)


REPO = Path("/data/CoordExp")
ROOT = REPO / "outputs/label_studio_coco_refinement/async-batch-v1"
SOURCE = REPO / "public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
IMAGE_ROOT = REPO / "public_data/coco/rescale_32_1024_bbox/images"
RECEIPT = ROOT / "receipt.json"
STORE_PY = REPO / "src/label_studio_coco_refinement/store.py"
TEST_STORE_PY = REPO / "tests/label_studio_coco_refinement/test_store.py"


class AcceptingAnnotationVerifier:
    def __init__(self) -> None:
        self.calls: list[AuthoritativeDraftIdentity] = []

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        self.calls.append(identity)
        return True


class EmptyInferenceResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        self.calls.append(receipt_id)
        return None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stat_receipt(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "resolved": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": oct(stat.st_mode),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def source_receipt() -> dict[str, Any]:
    return {**stat_receipt(SOURCE), "sha256": sha256_file(SOURCE)}


def memory_receipt() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _ = line.split()
            values[key.rstrip(":")] = int(value)
    values["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def file_receipt(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def manifest_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    return {
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "generation": payload["generation"],
        "working_sha256": payload["working_sha256"],
        "working_line_count": payload["working_line_count"],
        "raw_hex_prefix": raw[:32].hex(),
    }


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def save(result: Mapping[str, Any]) -> None:
    temporary = RECEIPT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, RECEIPT)


def selected_raw_lines(path: Path, indices: set[int]) -> dict[int, bytes]:
    found: dict[int, bytes] = {}
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                found[index] = line
                if len(found) == len(indices):
                    break
    if set(found) != indices:
        raise AssertionError(f"missing selected rows: {sorted(indices - set(found))}")
    return found


def read_source_prefix(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in itertools.islice(handle, count):
            rows.append(json.loads(line))
    if len(rows) != count:
        raise AssertionError(f"expected {count} source rows, got {len(rows)}")
    return rows


def rebase_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    image_name = Path(copied["images"][0]).name
    copied["images"] = [str(IMAGE_ROOT / "train2017" / image_name)]
    return copied


def write_slice(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(rebase_source_row(row)) + "\n")


def make_spec(source: Path, runtime: Path, project_id: str) -> BootstrapSpec:
    return BootstrapSpec(
        split="train",
        source_path=source,
        runtime_root=runtime,
        image_root=IMAGE_ROOT,
        expected_source_sha256=sha256_file(source),
        project_id=project_id,
        storage_id=f"storage-{project_id}",
        adapter_version="label-studio-coco-refinement-v1",
        vendor_revision="async-batch-v1-probe",
        registry_fingerprint=COCO80_REGISTRY.fingerprint,
        label_config_fingerprint="human-only-async-batch-v1-probe",
    )


def regions_for_seed(
    store: WorkingDatasetStore,
    seed: Any,
    *,
    add_region: bool,
    ordinal: int,
) -> tuple[list[dict[str, Any]], Any]:
    restored = store.restore_draft(seed.image_id)
    regions: list[dict[str, Any]] = []
    for obj in restored.row["objects"]:
        copied = dict(obj)
        copied["region_key"] = next(
            key
            for key, object_id in restored.region_id_mapping.items()
            if object_id == obj["coco_ann_id"]
        )
        regions.append(copied)
    if add_region:
        regions.append(
            {
                "region_key": f"drawn:async-batch:{seed.image_id}",
                "bbox_2d": [100 + ordinal, 110 + ordinal, 140 + ordinal, 160 + ordinal],
                "category_name": "person",
                "category_id": 1,
                "creation_ordinal": 10_000 + ordinal,
                "metadata": {"human_note": "async batch execution probe"},
            }
        )
    return regions, restored


def request_for_seed(
    store: WorkingDatasetStore,
    seed: Any,
    *,
    batch_id: str,
    ordinal: int,
    add_region: bool = True,
) -> CommitRequest:
    regions, restored = regions_for_seed(
        store,
        seed,
        add_region=add_region,
        ordinal=ordinal,
    )
    semantic = semantic_hash(regions)
    project_id = json.loads(store.manifest_path.read_text())["project_id"]
    commit_id = f"{batch_id}:member:{seed.image_id}"
    draft_id = f"draft:{commit_id}"
    draft_save = DraftSaveReceipt(
        project_id=project_id,
        task_id=seed.task_id,
        annotation_id=seed.authoritative_annotation_id,
        draft_id=draft_id,
        annotation_revision=ordinal + 1,
        semantic_hash=semantic,
    )
    return CommitRequest(
        commit_id=commit_id,
        split=seed.split,
        image_id=seed.image_id,
        project_id=project_id,
        task_id=seed.task_id,
        annotation_id=seed.authoritative_annotation_id,
        draft_id=draft_id,
        annotation_revision=ordinal + 1,
        semantic_hash=semantic,
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
        draft_save=draft_save,
        inference_receipts=(),
    )


def freeze_batch(
    store: WorkingDatasetStore,
    *,
    batch_id: str,
    member_count: int,
) -> tuple[BatchRequest, list[Any], float]:
    started = time.perf_counter()
    seeds = list(itertools.islice(store.iter_task_seeds(), member_count + 1))
    if len(seeds) != member_count + 1:
        raise AssertionError("not enough store rows for members and sentinel")
    members = tuple(
        BatchMember(
            source_row_index=seed.source_row_index,
            request=request_for_seed(
                store,
                seed,
                batch_id=batch_id,
                ordinal=ordinal,
            ),
        )
        for ordinal, seed in enumerate(seeds[:member_count])
    )
    generation = json.loads(store.manifest_path.read_text())["generation"]
    request = BatchRequest(
        batch_id=batch_id,
        split="train",
        base_generation=generation,
        members=members,
    )
    return request, seeds, time.perf_counter() - started


def managed_link_receipt(store: WorkingDatasetStore) -> dict[str, Any]:
    link = store.split_dir / "images"
    return {
        "path": str(link),
        "is_symlink": link.is_symlink(),
        "readlink": os.readlink(link),
        "resolved": str(link.resolve(strict=True)),
        "shared_root": str(IMAGE_ROOT.resolve(strict=True)),
        "samefile": os.path.samefile(link, IMAGE_ROOT),
    }


def run_case(
    *,
    name: str,
    source: Path,
    runtime: Path,
    expected_rows: int,
) -> dict[str, Any]:
    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceResolver()
    memory_before = memory_receipt()
    bootstrap_started = time.perf_counter()
    boot = WorkingDatasetStore.bootstrap(
        make_spec(source, runtime, f"async-batch-{name}"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    store = boot.store
    if boot.task_count != expected_rows:
        raise AssertionError(f"bootstrap rows {boot.task_count} != {expected_rows}")
    link_receipt = managed_link_receipt(store)
    if not (link_receipt["is_symlink"] and link_receipt["samefile"]):
        raise AssertionError("managed image link is not the exact shared root")

    batch_id = f"batch-{name}-10"
    request, seeds, freeze_seconds = freeze_batch(
        store,
        batch_id=batch_id,
        member_count=10,
    )
    member_indices = {member.source_row_index for member in request.members}
    sentinel_index = seeds[10].source_row_index
    before_rows = selected_raw_lines(store.working_path, member_indices | {sentinel_index})
    working_before = file_receipt(store.working_path)
    manifest_before = manifest_receipt(store.manifest_path)
    manifest_raw_before = store.manifest_path.read_bytes()
    memory_before_enqueue = memory_receipt()
    enqueue_started = time.perf_counter()
    enqueue = store.enqueue_batch(request)
    enqueue_seconds = time.perf_counter() - enqueue_started
    memory_after_enqueue = memory_receipt()
    working_after_enqueue = file_receipt(store.working_path)
    manifest_after_enqueue = manifest_receipt(store.manifest_path)
    manifest_raw_after_enqueue = store.manifest_path.read_bytes()
    status_after_enqueue = store.get_batch_status(batch_id)
    enqueue_immutable = (
        working_after_enqueue == working_before
        and manifest_raw_after_enqueue == manifest_raw_before
        and manifest_after_enqueue == manifest_before
    )
    if enqueue.status is not BatchStatus.QUEUED or not enqueue_immutable:
        raise AssertionError("enqueue published working or manifest state")

    queue_after_enqueue = file_receipt(store.queue_path)
    memory_before_worker = memory_receipt()
    worker_started = time.perf_counter()
    worker_result = store.process_next_batch()
    worker_seconds = time.perf_counter() - worker_started
    memory_after_worker = memory_receipt()
    if worker_result is None or worker_result.status is not BatchStatus.SUCCEEDED:
        raise AssertionError(f"batch worker failed: {worker_result}")
    manifest_after_worker = manifest_receipt(store.manifest_path)
    if manifest_after_worker["generation"] != manifest_before["generation"] + 1:
        raise AssertionError("batch publication did not increment generation exactly once")
    if len(worker_result.members) != 10:
        raise AssertionError("batch result did not publish all ten members")
    after_rows = selected_raw_lines(store.working_path, member_indices | {sentinel_index})
    changed_indices = sorted(index for index in member_indices if after_rows[index] != before_rows[index])
    sentinel_unchanged = after_rows[sentinel_index] == before_rows[sentinel_index]
    if changed_indices != sorted(member_indices) or not sentinel_unchanged:
        raise AssertionError("member or untouched-sentinel publication mismatch")
    expected_keys = {
        member.request.image_id: f"drawn:async-batch:{member.request.image_id}"
        for member in request.members
    }
    member_assertions = []
    for result_member in worker_result.members:
        key = expected_keys[result_member.image_id]
        assigned = result_member.region_id_mapping.get(key)
        has_object = any(
            obj["coco_ann_id"] == assigned
            and obj.get("metadata") == {"human_note": "async batch execution probe"}
            for obj in result_member.committed_row["objects"]
        )
        member_assertions.append(
            {
                "image_id": result_member.image_id,
                "source_row_index": next(
                    member.source_row_index
                    for member in request.members
                    if member.request.image_id == result_member.image_id
                ),
                "region_key": key,
                "coco_ann_id": assigned,
                "row_hash": result_member.row_hash,
                "written": assigned is not None and assigned < 0 and has_object,
            }
        )
    if not all(item["written"] for item in member_assertions):
        raise AssertionError("one or more batch members were not written exactly")
    status_after_worker = store.get_batch_status(batch_id)
    if status_after_worker.status is not BatchStatus.SUCCEEDED:
        raise AssertionError("batch status projection is not succeeded")

    return {
        "name": name,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "expected_rows": expected_rows,
        "bootstrap": {
            "seconds": bootstrap_seconds,
            "created": boot.created,
            "task_count": boot.task_count,
            "task_manifest_hash": boot.task_manifest_hash,
            "memory_before": memory_before,
            "memory_after": memory_receipt(),
        },
        "managed_images_link": link_receipt,
        "draft_freeze_seconds": freeze_seconds,
        "batch": {
            "batch_id": batch_id,
            "member_count": 10,
            "base_generation": request.base_generation,
            "member_source_row_indices": sorted(member_indices),
            "sentinel_source_row_index": sentinel_index,
            "sentinel_image_id": seeds[10].image_id,
            "enqueue": {
                "seconds": enqueue_seconds,
                "receipt": asdict(enqueue),
                "status_view": asdict(status_after_enqueue),
                "working_manifest_unchanged": enqueue_immutable,
                "working_before": working_before,
                "working_after": working_after_enqueue,
                "manifest_before": manifest_before,
                "manifest_after": manifest_after_enqueue,
                "queue_after": queue_after_enqueue,
                "memory_before": memory_before_enqueue,
                "memory_after": memory_after_enqueue,
            },
            "worker": {
                "seconds": worker_seconds,
                "amortized_seconds_per_member": worker_seconds / 10,
                "status": worker_result.status.value,
                "generation": worker_result.generation,
                "working_sha256": worker_result.working_sha256,
                "generation_increment": worker_result.generation - request.base_generation,
                "member_assertions": member_assertions,
                "all_ten_rows_changed": len(changed_indices) == 10,
                "changed_source_row_indices": changed_indices,
                "sentinel_unchanged": sentinel_unchanged,
                "manifest_after": manifest_after_worker,
                "working_after": file_receipt(store.working_path),
                "queue_after": file_receipt(store.queue_path),
                "journal_after": file_receipt(store.journal_path),
                "memory_before": memory_before_worker,
                "memory_after": memory_after_worker,
            },
        },
        "storage": {
            "split_root": str(store.split_dir),
            "runtime_root": str(runtime),
            "tree_bytes": tree_bytes(runtime),
            "working_bytes": store.working_path.stat().st_size,
            "queue_bytes": store.queue_path.stat().st_size,
            "journal_bytes": store.journal_path.stat().st_size,
            "manifest_bytes": store.manifest_path.stat().st_size,
            "task_index_bytes": store.task_index_path.stat().st_size,
        },
        "verifier_call_count": len(verifier.calls),
        "verifier_identities": [asdict(call) for call in verifier.calls],
        "resolver_calls": resolver.calls,
    }


def run_blocked_worker(source: Path) -> dict[str, Any]:
    runtime = ROOT / "blocked-worker/runtime"
    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceResolver()
    boot = WorkingDatasetStore.bootstrap(
        make_spec(source, runtime, "blocked-worker"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    worker = boot.store
    peer = WorkingDatasetStore(
        worker.split_dir,
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    batch, seeds, _ = freeze_batch(worker, batch_id="blocked-batch", member_count=2)
    snapshot_seed = list(itertools.islice(peer.iter_task_seeds(), 21))[20]
    before_restore = peer.restore_draft(snapshot_seed.image_id)
    before_working = file_receipt(worker.working_path)
    before_manifest = manifest_receipt(worker.manifest_path)
    enqueue = worker.enqueue_batch(batch)
    status_worker = worker.get_batch_status(batch.batch_id)
    after_restore = peer.restore_draft(snapshot_seed.image_id)
    peer_seed = next(
        seed
        for seed in itertools.islice(peer.iter_task_seeds(), 21)
        if seed.image_id == snapshot_seed.image_id
    )
    draft_snapshot = request_for_seed(
        peer,
        peer_seed,
        batch_id="independent-draft-snapshot",
        ordinal=99,
        add_region=False,
    )
    peer._validate_draft_handshake(draft_snapshot)
    authoritative_snapshot_valid = verifier.verify(
        AuthoritativeDraftIdentity.from_request(draft_snapshot)
    )
    status_peer = peer.get_batch_status(batch.batch_id)
    after_working = file_receipt(worker.working_path)
    after_manifest = manifest_receipt(worker.manifest_path)
    assertions = {
        "enqueue_queued": enqueue.status is BatchStatus.QUEUED,
        "worker_status_queued": status_worker.status is BatchStatus.QUEUED,
        "peer_status_queued": status_peer.status is BatchStatus.QUEUED,
        "old_generation_readable": after_restore.generation == before_restore.generation == 0,
        "old_row_hash_stable": after_restore.row_hash == before_restore.row_hash,
        "draft_snapshot_handshake_valid": authoritative_snapshot_valid,
        "working_unchanged": after_working == before_working,
        "manifest_unchanged": after_manifest == before_manifest,
        "not_processed": not any(
            json.loads(line).get("kind") == "claim"
            for line in worker.queue_path.read_text().splitlines()
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"blocked worker assertions failed: {assertions}")
    return {
        "runtime": str(runtime),
        "batch_id": batch.batch_id,
        "enqueue_receipt": asdict(enqueue),
        "worker_status": asdict(status_worker),
        "peer_status": asdict(status_peer),
        "peer_snapshot": {
            "image_id": snapshot_seed.image_id,
            "task_id": snapshot_seed.task_id,
            "annotation_id": snapshot_seed.authoritative_annotation_id,
            "generation_before": before_restore.generation,
            "generation_after_enqueue": after_restore.generation,
            "row_hash": after_restore.row_hash,
            "draft_semantic_hash": draft_snapshot.semantic_hash,
        },
        "assertions": assertions,
        "queue": file_receipt(worker.queue_path),
        "working": after_working,
        "manifest": after_manifest,
        "core_attested": [
            "queued publication leaves working and manifest unchanged",
            "pre-opened peer reads the old committed generation",
            "an independent Draft snapshot passes the core immutable handshake",
        ],
        "ui_unattested": [
            "Label Studio browser Draft-save HTTP/UI completion while worker is blocked",
            "navigation modal behavior",
        ],
    }


def image_samples() -> dict[str, Any]:
    indices = {0, 58_633, 117_265}
    rows: dict[int, dict[str, Any]] = {}
    with SOURCE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                rows[index] = json.loads(line)
    samples = []
    for index in sorted(indices):
        locator = rows[index]["images"][0]
        image = (SOURCE.parent / locator).resolve(strict=True)
        samples.append(
            {
                "source_row_index": index,
                "image_id": rows[index]["image_id"],
                "declared_locator": locator,
                **stat_receipt(image),
                "sha256": sha256_file(image),
            }
        )
    return {
        "image_root": stat_receipt(IMAGE_ROOT),
        "sample_count": len(samples),
        "samples": samples,
    }


def main() -> None:
    result: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "PYTHONPATH=. /root/miniconda3/envs/ms/bin/python outputs/label_studio_coco_refinement/async-batch-v1/run_async_batch_probe.py",
        "code_state": {
            "store_py": {"path": str(STORE_PY), "sha256": sha256_file(STORE_PY)},
            "test_store_py": {
                "path": str(TEST_STORE_PY),
                "sha256": sha256_file(TEST_STORE_PY),
            },
        },
        "canonical_before": {
            "source": source_receipt(),
            "images": image_samples(),
        },
        "cases": {},
    }
    save(result)
    prefix_10k = read_source_prefix(10_000)
    for count in (1_000, 10_000):
        case_root = ROOT / f"case-{count}"
        source = case_root / "source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
        write_slice(source, prefix_10k[:count])
        result["cases"][str(count)] = run_case(
            name=str(count),
            source=source,
            runtime=case_root / "runtime",
            expected_rows=count,
        )
        save(result)
    result["cases"]["117266"] = run_case(
        name="117266",
        source=SOURCE,
        runtime=ROOT / "case-117266/runtime",
        expected_rows=117_266,
    )
    save(result)
    blocked_source = ROOT / "blocked-worker/source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    write_slice(blocked_source, prefix_10k[:1_000])
    result["blocked_worker"] = run_blocked_worker(blocked_source)
    result["canonical_after"] = {
        "source": source_receipt(),
        "images": image_samples(),
    }
    result["canonical_unchanged"] = result["canonical_before"] == result["canonical_after"]
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(result)
    print(
        json.dumps(
            {
                "receipt": str(RECEIPT),
                "cases": {
                    name: {
                        "bootstrap_seconds": case["bootstrap"]["seconds"],
                        "enqueue_seconds": case["batch"]["enqueue"]["seconds"],
                        "worker_seconds": case["batch"]["worker"]["seconds"],
                        "amortized_seconds_per_member": case["batch"]["worker"][
                            "amortized_seconds_per_member"
                        ],
                    }
                    for name, case in result["cases"].items()
                },
                "blocked_worker_queued": result["blocked_worker"]["assertions"][
                    "enqueue_queued"
                ],
                "canonical_unchanged": result["canonical_unchanged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "store_sha256": sha256_file(STORE_PY),
            "test_store_sha256": sha256_file(TEST_STORE_PY),
        }
        (ROOT / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
