from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from src.data import load_raw_examples
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.materialize import (
    CommittedGenerationReceipt,
    WorkingCoordMaterializer,
)
from src.label_studio_coco_refinement.models import RefinementRuntimeLayout
from src.label_studio_coco_refinement.store import (
    AuthoritativeDraftIdentity,
    BootstrapSpec,
    CommitRequest,
    DraftSaveReceipt,
    InjectedCrash,
    InferenceReceiptLink,
    WorkingDatasetStore,
    canonical_json,
    semantic_hash,
    sha256_file,
)


REPO = Path("/data/CoordExp")
ROOT = REPO / "outputs/label_studio_coco_refinement/probes-v2"
SELECTED = REPO / "public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
SELECTED_FAMILY = [
    REPO / "public_data/coco/rescale_32_1024_bbox_len12000/train.jsonl",
    SELECTED,
    REPO / "public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl",
]
IMAGE_ROOT = REPO / "public_data/coco/rescale_32_1024_bbox/images"
RECEIPT = ROOT / "receipt.json"


class AcceptingAnnotationVerifier:
    def __init__(self) -> None:
        self.calls: list[AuthoritativeDraftIdentity] = []

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        self.calls.append(identity)
        return True


class EmptyInferenceReceiptResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        self.calls.append(receipt_id)
        return None


class CrashAt:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.seen: list[str] = []

    def __call__(self, boundary: str) -> None:
        self.seen.append(boundary)
        if boundary == self.boundary:
            raise InjectedCrash(boundary)


def hashes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def memory_receipt() -> dict[str, int]:
    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            status[key.rstrip(":")] = int(value)
    status["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return status


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def save(result: dict[str, Any]) -> None:
    temp = RECEIPT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, RECEIPT)


def rebase_row(row: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(row))
    relative = Path(row["images"][0])
    image_name = relative.name
    row["images"] = [str(IMAGE_ROOT / "train2017" / image_name)]
    return row


def write_slice(destination: Path, rows: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(rebase_row(row)) + "\n")


def make_spec(source: Path, runtime_root: Path, project_id: str) -> BootstrapSpec:
    return BootstrapSpec(
        split="train",
        source_path=source,
        runtime_root=runtime_root,
        image_root=IMAGE_ROOT,
        expected_source_sha256=sha256_file(source),
        project_id=project_id,
        storage_id=f"storage-{project_id}",
        adapter_version="label-studio-coco-refinement-v1",
        vendor_revision="probe-current-tree",
        registry_fingerprint=COCO80_REGISTRY.fingerprint,
        label_config_fingerprint="probe-human-only-v2",
    )


def make_request(
    store: WorkingDatasetStore,
    *,
    commit_id: str,
    revision: int,
    add_human_region: bool,
    toggle: int,
) -> tuple[CommitRequest, dict[str, Any]]:
    seed = next(store.iter_task_seeds())
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
    bbox = list(regions[0]["bbox_2d"])
    bbox[0] += 1 if toggle % 2 else -1
    regions[0]["bbox_2d"] = bbox
    if add_human_region:
        regions.append(
            {
                "region_key": f"drawn:human-probe-{revision}",
                "bbox_2d": [100, 110, 130, 150],
                "category_name": "person",
                "category_id": 1,
                "creation_ordinal": 1000 + revision,
                "metadata": {"human_note": "execution attestation"},
            }
        )
    projection_hash = semantic_hash(regions)
    draft_id = f"draft-{commit_id}"
    receipt = DraftSaveReceipt(
        project_id=json.loads(store.manifest_path.read_text())["project_id"],
        task_id=seed.task_id,
        annotation_id=seed.authoritative_annotation_id,
        draft_id=draft_id,
        annotation_revision=revision,
        semantic_hash=projection_hash,
    )
    request = CommitRequest(
        commit_id=commit_id,
        split=seed.split,
        image_id=seed.image_id,
        project_id=receipt.project_id,
        task_id=seed.task_id,
        annotation_id=seed.authoritative_annotation_id,
        draft_id=draft_id,
        annotation_revision=revision,
        semantic_hash=projection_hash,
        base_row_hash=restored.row_hash,
        observed_generation=restored.generation,
        regions=regions,
        draft_save=receipt,
        inference_receipts=(),
    )
    return request, {
        "project_id": request.project_id,
        "task_id": request.task_id,
        "annotation_id": request.annotation_id,
        "image_id": request.image_id,
        "inference_receipts": list(request.inference_receipts),
    }


def bootstrap_and_commit(
    source: Path,
    runtime_root: Path,
    project_id: str,
    row_count: int,
    *,
    full: bool,
) -> dict[str, Any]:
    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceReceiptResolver()
    memory_before = memory_receipt()
    started = time.perf_counter()
    boot = WorkingDatasetStore.bootstrap(
        make_spec(source, runtime_root, project_id),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    bootstrap_s = time.perf_counter() - started
    store = boot.store
    commit_receipts: list[dict[str, Any]] = []
    target = 1
    for index in range(8):
        if index >= target:
            break
        request, identity = make_request(
            store,
            commit_id=f"{project_id}-commit-{index + 1}",
            revision=index + 1,
            add_human_region=index == 0,
            toggle=index + 1,
        )
        before = memory_receipt()
        commit_started = time.perf_counter()
        committed = store.commit(request)
        elapsed = time.perf_counter() - commit_started
        after = memory_receipt()
        commit_receipts.append(
            {
                "index": index + 1,
                "seconds": elapsed,
                "status": committed.status.value,
                "generation": committed.generation,
                "row_hash": committed.row_hash,
                "working_sha256": sha256_file(store.working_path),
                "identity": identity,
                "negative_ids": [
                    obj["coco_ann_id"]
                    for obj in committed.committed_row["objects"]
                    if obj["coco_ann_id"] < 0
                ],
                "memory_before": before,
                "memory_after": after,
            }
        )
        if full and index == 0 and elapsed <= 2.0:
            target = 8
        if full and elapsed > 5.0:
            break
    latencies = [entry["seconds"] for entry in commit_receipts]
    manifest = json.loads(store.manifest_path.read_text())
    return {
        "requested_rows": row_count,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "bootstrap_created": boot.created,
        "bootstrap_seconds": bootstrap_s,
        "bootstrap_task_count": boot.task_count,
        "task_manifest_hash": boot.task_manifest_hash,
        "commit_count": len(commit_receipts),
        "commit_seconds": latencies,
        "commit_p95_seconds_linear": percentile(latencies, 0.95),
        "commit_max_seconds": max(latencies),
        "commit_mean_seconds": statistics.mean(latencies),
        "commit_hard_stop_exceeded": any(value > 5.0 for value in latencies),
        "commit_target_p95_le_2s": percentile(latencies, 0.95) <= 2.0,
        "commits": commit_receipts,
        "working_path": str(store.working_path),
        "working_bytes": store.working_path.stat().st_size,
        "journal_bytes": store.journal_path.stat().st_size,
        "manifest_bytes": store.manifest_path.stat().st_size,
        "runtime_tree_bytes": tree_bytes(runtime_root),
        "storage_root": str(runtime_root),
        "manifest_generation": manifest["generation"],
        "manifest_working_sha256": manifest["working_sha256"],
        "verifier_calls": [call.__dict__ for call in verifier.calls],
        "resolver_calls": resolver.calls,
        "memory_before": memory_before,
        "memory_after": memory_receipt(),
    }


def run_faults(source: Path) -> list[dict[str, Any]]:
    pre = [
        "prepared_journal_flushed",
        "prepared_journal_fsynced",
        "working_temp_flushed",
        "working_temp_fsynced",
    ]
    post = [
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
    results: list[dict[str, Any]] = []
    for boundary in pre + post:
        verifier = AcceptingAnnotationVerifier()
        resolver = EmptyInferenceReceiptResolver()
        runtime = ROOT / "faults" / boundary / "runtime"
        crash = CrashAt(boundary)
        boot = WorkingDatasetStore.bootstrap(
            make_spec(source, runtime, f"fault-{boundary}"),
            annotation_verifier=verifier,
            inference_receipt_resolver=resolver,
        )
        store = boot.store
        store._fault_injector = crash
        request, identity = make_request(
            store,
            commit_id=f"commit-{boundary}",
            revision=1,
            add_human_region=True,
            toggle=1,
        )
        try:
            store.commit(request)
            commit_outcome = "returned"
        except BaseException as error:
            commit_outcome = f"{type(error).__name__}: {error}"
        recovered1 = WorkingDatasetStore(
            store.split_dir,
            annotation_verifier=verifier,
            inference_receipt_resolver=resolver,
        )
        status1 = recovered1.status(request.commit_id).value
        generation1 = recovered1.restore_draft(request.image_id).generation
        terminal1 = sum(
            json.loads(line)["kind"] == "terminal"
            for line in recovered1.journal_path.read_text().splitlines()
        )
        hash1 = sha256_file(recovered1.working_path)
        recovered2 = WorkingDatasetStore(
            store.split_dir,
            annotation_verifier=verifier,
            inference_receipt_resolver=resolver,
        )
        status2 = recovered2.status(request.commit_id).value
        generation2 = recovered2.restore_draft(request.image_id).generation
        terminal2 = sum(
            json.loads(line)["kind"] == "terminal"
            for line in recovered2.journal_path.read_text().splitlines()
        )
        hash2 = sha256_file(recovered2.working_path)
        expected = "rolled_back" if boundary in pre else "committed"
        results.append(
            {
                "boundary": boundary,
                "expected_status": expected,
                "commit_outcome": commit_outcome,
                "seen_boundaries": crash.seen,
                "identity": identity,
                "first_reopen": {
                    "status": status1,
                    "generation": generation1,
                    "terminal_records": terminal1,
                    "working_sha256": hash1,
                },
                "second_reopen": {
                    "status": status2,
                    "generation": generation2,
                    "terminal_records": terminal2,
                    "working_sha256": hash2,
                },
                "attested": status1 == expected
                and status2 == expected
                and generation1 == generation2
                and terminal1 == terminal2
                and hash1 == hash2,
            }
        )
    return results


def run_materializer(one_row: dict[str, Any]) -> dict[str, Any]:
    repo = ROOT / "materializer-repository"
    layout = RefinementRuntimeLayout.under_repository(repo)
    source = layout.selected_source("train")
    source.parent.mkdir(parents=True, exist_ok=True)
    image_parent = layout.image_root.parent
    image_parent.mkdir(parents=True, exist_ok=True)
    layout.image_root.symlink_to(IMAGE_ROOT, target_is_directory=True)
    write_slice(source, [one_row])
    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceReceiptResolver()
    boot = WorkingDatasetStore.bootstrap(
        make_spec(source, layout.root, "materializer-project"),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    store = boot.store
    request, identity = make_request(
        store,
        commit_id="materializer-negative-id",
        revision=1,
        add_human_region=True,
        toggle=1,
    )
    committed = store.commit(request)
    manifest = json.loads(store.manifest_path.read_text())
    generation = CommittedGenerationReceipt(
        split="train",
        generation=manifest["generation"],
        working_sha256=manifest["working_sha256"],
    )
    source_before = store.working_path.read_bytes()
    materialized = WorkingCoordMaterializer(
        layout,
        "train",
        exclusive_lock=store._exclusive_lock,
    ).materialize(generation)
    loaded = load_raw_examples(layout.working_coord("train"))
    return {
        "identity": identity,
        "committed_generation": generation.__dict__,
        "commit_negative_ids": [
            obj["coco_ann_id"]
            for obj in committed.committed_row["objects"]
            if obj["coco_ann_id"] < 0
        ],
        "materialization_receipt": materialized.to_artifact_dict(),
        "working_norm_unchanged_by_materializer": source_before
        == store.working_path.read_bytes(),
        "layout": {
            "repository_root": str(layout.repository_root),
            "root": str(layout.root),
            "image_root_declared": str(layout.image_root),
            "image_root_resolved": str(layout.image_root.resolve()),
            "images_link": str(layout.images_link("train")),
            "images_link_resolved": str(layout.images_link("train").resolve()),
        },
        "loader_api": "src.data.load_raw_examples(layout.working_coord('train'))",
        "loader_row_count": len(loaded),
        "loader_example_id": loaded[0].example_id,
        "loader_object_ids": [obj.object_id for obj in loaded[0].objects],
        "loader_bboxes": [list(obj.bbox) for obj in loaded[0].objects],
        "loader_image": str(loaded[0].image.path),
        "verifier_calls": [call.__dict__ for call in verifier.calls],
        "resolver_calls": resolver.calls,
    }


def main() -> None:
    result: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "PYTHONPATH=. conda run -n ms python outputs/label_studio_coco_refinement/probes-v2/run_attestation.py",
        "source_hashes_before": hashes(SELECTED_FAMILY),
        "source_rows": sum(1 for _ in SELECTED.open("rb")),
        "runs": {},
    }
    save(result)
    source_rows: list[dict[str, Any]] = []
    with SELECTED.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 10_000:
                break
            source_rows.append(json.loads(line))
    for count in (1_000, 10_000):
        base = ROOT / f"rows-{count}"
        source = base / "public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
        write_slice(source, source_rows[:count])
        result["runs"][str(count)] = bootstrap_and_commit(
            source,
            base / "runtime",
            f"probe-{count}",
            count,
            full=False,
        )
        save(result)
    result["runs"]["117266"] = bootstrap_and_commit(
        SELECTED,
        ROOT / "rows-117266/runtime",
        "probe-117266",
        117_266,
        full=True,
    )
    save(result)
    fault_source = ROOT / "faults/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
    write_slice(fault_source, source_rows[:100])
    result["faults"] = run_faults(fault_source)
    save(result)
    result["materializer"] = run_materializer(source_rows[0])
    result["source_hashes_after"] = hashes(SELECTED_FAMILY)
    result["source_hashes_unchanged"] = (
        result["source_hashes_before"] == result["source_hashes_after"]
    )
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(result)
    print(
        json.dumps(
            {
                "receipt": str(RECEIPT),
                "run_commit_seconds": {
                    key: value["commit_seconds"] for key, value in result["runs"].items()
                },
                "faults_attested": sum(item["attested"] for item in result["faults"]),
                "source_hashes_unchanged": result["source_hashes_unchanged"],
                "materializer_loader_object_ids": result["materializer"]["loader_object_ids"],
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
        }
        (ROOT / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
