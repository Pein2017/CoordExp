#!/usr/bin/env python3
"""Disposable current-contract probe for asynchronous batch publication."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label_studio_coco_refinement.categories import COCO80_REGISTRY  # noqa: E402
from src.label_studio_coco_refinement.store import (  # noqa: E402
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
    sha256_json,
)
import src.label_studio_coco_refinement.store as store_module  # noqa: E402


CANONICAL_SOURCE = (
    REPO_ROOT / "public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
)
CANONICAL_IMAGE_ROOT = REPO_ROOT / "public_data/coco/rescale_32_1024_bbox/images"
CURRENT_USER_ID = "async-batch-probe"
PINNED_CONTRACT_FIELDS = {
    "DraftSaveReceipt": (
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "durable",
    ),
    "CommitRequest": (
        "commit_id",
        "split",
        "image_id",
        "project_id",
        "task_id",
        "annotation_id",
        "draft_id",
        "annotation_revision",
        "draft_updated_at",
        "semantic_hash",
        "result_hash",
        "base_row_hash",
        "observed_generation",
        "regions",
        "draft_save",
        "inference_receipts",
    ),
    "BatchMember": ("source_row_index", "request"),
    "BatchRequest": (
        "batch_id",
        "split",
        "current_user_id",
        "base_generation",
        "members",
    ),
}


class AcceptingAnnotationVerifier:
    """Probe-only authority that records the exact frozen batch it attests."""

    def __init__(self) -> None:
        self.batch_calls: list[BatchRequest] = []
        self.fallback_calls: list[AuthoritativeDraftIdentity] = []

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        self.fallback_calls.append(identity)
        return True

    def verify_batch(self, request: BatchRequest) -> bool:
        self.batch_calls.append(request)
        return True


class EmptyInferenceResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        self.calls.append(receipt_id)
        return None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _pinned_contract_fields() -> dict[str, list[str]]:
    observed = {
        "DraftSaveReceipt": tuple(field.name for field in fields(DraftSaveReceipt)),
        "CommitRequest": tuple(field.name for field in fields(CommitRequest)),
        "BatchMember": tuple(field.name for field in fields(BatchMember)),
        "BatchRequest": tuple(field.name for field in fields(BatchRequest)),
    }
    if observed != PINNED_CONTRACT_FIELDS:
        raise AssertionError(
            f"async batch dataclass contract drift: expected "
            f"{PINNED_CONTRACT_FIELDS}, observed {observed}"
        )
    return {name: list(field_names) for name, field_names in observed.items()}


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def _memory_identity() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            values[f"{key.rstrip(':')}_kb"] = int(value)
    values["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _code_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    store = Path(store_module.__file__).resolve()
    return {
        "git_head": _git_output("rev-parse", "HEAD"),
        "store": {
            "path": str(store),
            "sha256": sha256_file(store),
            "git_blob_sha1": _git_output("hash-object", str(store)),
        },
        "script": {
            "path": str(script),
            "sha256": sha256_file(script),
            "git_blob_sha1": _git_output("hash-object", str(script)),
        },
    }


def _is_git_ignored(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "--", str(relative)],
        check=False,
    )
    return result.returncode == 0


def _validate_output_root(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    candidate = candidate.resolve(strict=False)
    if os.path.lexists(candidate):
        raise ValueError(f"output root already exists: {candidate}")
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"output root must stay inside {REPO_ROOT}") from exc
    if not _is_git_ignored(candidate):
        raise ValueError(f"output root is not ignored by git: {candidate}")
    return candidate


def _source_row_count(source: Path, *, stop_after: int | None = None) -> int:
    count = 0
    with source.open("rb") as handle:
        for count, _line in enumerate(handle, start=1):
            if stop_after is not None and count >= stop_after:
                break
    return count


def _selected_rows(source: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                selected[index] = json.loads(line)
                if len(selected) == len(indices):
                    break
    if set(selected) != indices:
        raise ValueError(
            f"source is missing sampled rows: {sorted(indices - set(selected))}"
        )
    return selected


def _resolve_source_image(
    source: Path, image_root: Path, row: Mapping[str, Any]
) -> Path:
    locator = Path(row["images"][0])
    image = locator if locator.is_absolute() else source.parent / locator
    image = image.resolve(strict=True)
    try:
        image.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"sampled image escapes image root: {image}") from exc
    return image


def _sample_image_identities(
    source: Path,
    image_root: Path,
    *,
    selected_row_count: int,
) -> list[dict[str, Any]]:
    indices = {0, selected_row_count // 2, selected_row_count - 1}
    rows = _selected_rows(source, indices)
    samples = []
    for index in sorted(indices):
        row = rows[index]
        image = _resolve_source_image(source, image_root, row)
        samples.append(
            {
                "source_row_index": index,
                "image_id": row["image_id"],
                "declared_locator": row["images"][0],
                **_file_identity(image),
            }
        )
    return samples


def _write_source_slice(
    source: Path,
    destination: Path,
    image_root: Path,
    *,
    rows: int,
) -> None:
    destination.parent.mkdir(parents=True)
    written = 0
    with (
        source.open(encoding="utf-8") as input_handle,
        destination.open("w", encoding="utf-8", newline="\n") as output_handle,
    ):
        for line in itertools.islice(input_handle, rows):
            row = json.loads(line)
            row = copy.deepcopy(row)
            row["images"] = [str(_resolve_source_image(source, image_root, row))]
            output_handle.write(canonical_json(row) + "\n")
            written += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if written != rows:
        raise ValueError(f"requested {rows} rows but source has only {written}")


def _bootstrap_spec(
    source: Path, runtime_root: Path, image_root: Path
) -> BootstrapSpec:
    return BootstrapSpec(
        split="train",
        source_path=source,
        runtime_root=runtime_root,
        image_root=image_root,
        expected_source_sha256=sha256_file(source),
        project_id="async-batch-probe",
        storage_id="async-batch-probe-storage",
        adapter_version="label-studio-coco-refinement-current",
        vendor_revision="async-batch-current-api-probe",
        registry_fingerprint=COCO80_REGISTRY.fingerprint,
        label_config_fingerprint="async-batch-current-api-probe",
    )


def _regions_with_new_box(
    store: WorkingDatasetStore,
    image_id: int,
    *,
    ordinal: int,
) -> tuple[list[dict[str, Any]], Any, str]:
    restored = store.restore_draft(image_id)
    object_to_key = {
        object_id: region_key
        for region_key, object_id in restored.region_id_mapping.items()
    }
    regions: list[dict[str, Any]] = []
    for obj in restored.row["objects"]:
        copied = copy.deepcopy(obj)
        copied["region_key"] = object_to_key[obj["coco_ann_id"]]
        regions.append(copied)
    region_key = f"drawn:async-batch-probe:{image_id}"
    offset = ordinal % 20
    regions.append(
        {
            "region_key": region_key,
            "bbox_2d": [1 + offset, 2 + offset, 21 + offset, 22 + offset],
            "desc": "person",
            "category_name": "person",
            "category_id": 1,
            "creation_ordinal": 10_000 + ordinal,
            "metadata": {"human_note": "current async batch probe"},
        }
    )
    return regions, restored, region_key


def _freeze_batch(
    store: WorkingDatasetStore,
    *,
    member_count: int,
    batch_id: str,
) -> tuple[BatchRequest, list[Any], dict[int, str]]:
    seeds = list(itertools.islice(store.iter_task_seeds(), member_count + 1))
    if len(seeds) != member_count + 1:
        raise ValueError("rows must exceed members so one untouched sentinel exists")
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    members: list[BatchMember] = []
    region_keys: dict[int, str] = {}
    frozen_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for ordinal, seed in enumerate(seeds[:member_count]):
        regions, restored, region_key = _regions_with_new_box(
            store, seed.image_id, ordinal=ordinal
        )
        projection_hash = semantic_hash(regions)
        result_hash = sha256_json(regions)
        revision = f"probe-revision-{ordinal + 1}"
        draft_id = f"draft:{batch_id}:{seed.image_id}"
        draft_save = DraftSaveReceipt(
            project_id=manifest["project_id"],
            task_id=seed.task_id,
            annotation_id=seed.authoritative_annotation_id,
            draft_id=draft_id,
            annotation_revision=revision,
            draft_updated_at=frozen_at,
            semantic_hash=projection_hash,
            result_hash=result_hash,
        )
        request = CommitRequest(
            commit_id=f"{batch_id}:member:{seed.image_id}",
            split=seed.split,
            image_id=seed.image_id,
            project_id=manifest["project_id"],
            task_id=seed.task_id,
            annotation_id=seed.authoritative_annotation_id,
            draft_id=draft_id,
            annotation_revision=revision,
            draft_updated_at=frozen_at,
            semantic_hash=projection_hash,
            result_hash=result_hash,
            base_row_hash=restored.row_hash,
            observed_generation=restored.generation,
            regions=regions,
            draft_save=draft_save,
        )
        members.append(
            BatchMember(source_row_index=seed.source_row_index, request=request)
        )
        region_keys[seed.image_id] = region_key
    return (
        BatchRequest(
            batch_id=batch_id,
            split="train",
            current_user_id=CURRENT_USER_ID,
            base_generation=int(manifest["generation"]),
            members=tuple(members),
        ),
        seeds,
        region_keys,
    )


def _raw_rows(path: Path, indices: set[int]) -> dict[int, bytes]:
    rows: dict[int, bytes] = {}
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            if index in indices:
                rows[index] = raw
                if len(rows) == len(indices):
                    break
    if set(rows) != indices:
        raise AssertionError(
            f"working JSONL is missing rows: {sorted(indices - set(rows))}"
        )
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_receipt_atomic(output_root: Path, receipt: Mapping[str, Any]) -> Path:
    receipt_path = output_root / "receipt.json"
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".receipt.json.", dir=output_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt_path)
        directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt_path


def run_probe(
    *,
    output_root: Path,
    members: int,
    rows: int | None,
    source: Path = CANONICAL_SOURCE,
    image_root: Path = CANONICAL_IMAGE_ROOT,
) -> dict[str, Any]:
    """Run one disposable probe; ``rows=None`` selects the full source."""

    contracts = _pinned_contract_fields()
    output_root = _validate_output_root(output_root)
    source = source.resolve(strict=True)
    image_root = image_root.resolve(strict=True)
    if members <= 0:
        raise ValueError("members must be positive")
    if rows is not None and rows <= 0:
        raise ValueError("rows must be positive")
    selected_row_count = (
        _source_row_count(source)
        if rows is None
        else _source_row_count(source, stop_after=rows)
    )
    if rows is not None and selected_row_count != rows:
        raise ValueError(f"requested {rows} rows but source has {selected_row_count}")
    if selected_row_count <= members:
        raise ValueError("rows must exceed members so one untouched sentinel exists")

    source_before = _file_identity(source)
    images_before = _sample_image_identities(
        source, image_root, selected_row_count=selected_row_count
    )
    code_before = _code_identity()
    output_root.mkdir(parents=True, exist_ok=False)
    selected_source = source
    if rows is not None:
        selected_source = (
            output_root
            / "source/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl"
        )
        _write_source_slice(
            source, selected_source, image_root, rows=selected_row_count
        )

    verifier = AcceptingAnnotationVerifier()
    resolver = EmptyInferenceResolver()
    memory: dict[str, dict[str, int]] = {"before_bootstrap": _memory_identity()}
    timings: dict[str, float] = {}
    bootstrap_started = time.perf_counter()
    bootstrap = WorkingDatasetStore.bootstrap(
        _bootstrap_spec(selected_source, output_root / "runtime", image_root),
        annotation_verifier=verifier,
        inference_receipt_resolver=resolver,
    )
    timings["bootstrap_seconds"] = time.perf_counter() - bootstrap_started
    store = bootstrap.store
    memory["after_bootstrap"] = _memory_identity()
    if bootstrap.task_count != selected_row_count:
        raise AssertionError("bootstrap task count disagrees with the selected rows")
    images_link = store.split_dir / "images"
    if not images_link.is_symlink() or not os.path.samefile(images_link, image_root):
        raise AssertionError("bootstrap did not retain the shared image symlink")

    batch_id = f"async-batch-{selected_row_count}-{members}"
    freeze_started = time.perf_counter()
    batch, seeds, region_keys = _freeze_batch(
        store, member_count=members, batch_id=batch_id
    )
    timings["freeze_seconds"] = time.perf_counter() - freeze_started
    memory["after_freeze"] = _memory_identity()

    member_indices = {member.source_row_index for member in batch.members}
    sentinel_index = seeds[members].source_row_index
    observed_indices = member_indices | {sentinel_index}
    rows_before = _raw_rows(store.working_path, observed_indices)
    working_before = _artifact_identity(store.working_path)
    manifest_before = _artifact_identity(store.manifest_path)
    manifest_payload_before = json.loads(
        store.manifest_path.read_text(encoding="utf-8")
    )

    enqueue_started = time.perf_counter()
    enqueue = store.enqueue_batch(batch)
    timings["enqueue_seconds"] = time.perf_counter() - enqueue_started
    memory["after_enqueue"] = _memory_identity()
    working_after_enqueue = _artifact_identity(store.working_path)
    manifest_after_enqueue = _artifact_identity(store.manifest_path)
    enqueue_unchanged = (
        working_after_enqueue == working_before
        and manifest_after_enqueue == manifest_before
    )
    if enqueue.status is not BatchStatus.QUEUED or not enqueue_unchanged:
        raise AssertionError("enqueue changed published working or manifest state")
    status_after_enqueue = store.get_batch_status(batch_id)

    background_started = time.perf_counter()
    result = store.process_next_batch()
    timings["background_seconds"] = time.perf_counter() - background_started
    memory["after_background"] = _memory_identity()
    if result is None or result.status is not BatchStatus.SUCCEEDED:
        raise AssertionError(f"background worker did not succeed: {result}")
    if len(result.members) != members:
        raise AssertionError("background worker returned an incomplete batch")

    final_manifest_payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    rows_after = _raw_rows(store.working_path, observed_indices)
    changed_indices = sorted(
        index for index in member_indices if rows_after[index] != rows_before[index]
    )
    sentinel_unchanged = rows_after[sentinel_index] == rows_before[sentinel_index]
    generation_once = (
        result.generation == batch.base_generation + 1
        and int(final_manifest_payload["generation"]) == batch.base_generation + 1
        and int(manifest_payload_before["generation"]) == batch.base_generation
    )

    result_by_image = {member.image_id: member for member in result.members}
    member_receipts = []
    negative_ids: list[int] = []
    for member in batch.members:
        commit_result = result_by_image[member.request.image_id]
        assigned = commit_result.region_id_mapping[region_keys[member.request.image_id]]
        negative_ids.append(assigned)
        member_receipts.append(
            {
                "image_id": member.request.image_id,
                "source_row_index": member.source_row_index,
                "draft_updated_at": member.request.draft_updated_at,
                "result_hash": member.request.result_hash,
                "generation": commit_result.generation,
                "negative_coco_ann_id": assigned,
                "row_hash": commit_result.row_hash,
            }
        )
    negative_ids_valid = (
        len(negative_ids) == members
        and len(set(negative_ids)) == members
        and all(type(value) is int and value < 0 for value in negative_ids)
    )
    all_members_changed = changed_indices == sorted(member_indices)
    all_member_generations_match = all(
        item["generation"] == result.generation for item in member_receipts
    )
    status_final = store.get_batch_status(batch_id)
    atomic_publication = all(
        (
            enqueue_unchanged,
            all_members_changed,
            sentinel_unchanged,
            generation_once,
            all_member_generations_match,
            status_final.status is BatchStatus.SUCCEEDED,
        )
    )
    if not atomic_publication:
        raise AssertionError("batch members were not published as one generation")
    if not negative_ids_valid:
        raise AssertionError("batch did not allocate distinct negative object IDs")

    source_after = _file_identity(source)
    images_after = _sample_image_identities(
        source, image_root, selected_row_count=selected_row_count
    )
    source_unchanged = source_after == source_before
    images_unchanged = images_after == images_before
    if not source_unchanged or not images_unchanged:
        raise AssertionError("probe mutated its source JSONL or sampled images")
    code_after = _code_identity()
    code_identity_unchanged = code_after == code_before
    if not code_identity_unchanged:
        raise AssertionError(
            "store, probe script, or git HEAD changed during the probe"
        )
    memory["final"] = _memory_identity()
    final_artifacts = {
        "queue": _artifact_identity(store.queue_path),
        "journal": _artifact_identity(store.journal_path),
        "manifest": _artifact_identity(store.manifest_path),
        "working": _artifact_identity(store.working_path),
    }
    if final_artifacts["working"]["sha256"] != final_manifest_payload["working_sha256"]:
        raise AssertionError("final working hash disagrees with the manifest")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration": {
            "source": str(source),
            "selected_source": str(selected_source),
            "full_source": rows is None,
            "rows": selected_row_count,
            "members": members,
            "output_root": str(output_root),
        },
        "contracts": contracts,
        "code_identity_before": code_before,
        "code_identity_after": code_after,
        "code_identity_unchanged": code_identity_unchanged,
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "source_unchanged": source_unchanged,
        "sampled_images_before": images_before,
        "sampled_images_after": images_after,
        "sampled_images_unchanged": images_unchanged,
        "shared_image_symlink": {
            "path": str(images_link),
            "target": os.readlink(images_link),
            "samefile": os.path.samefile(images_link, image_root),
        },
        "timings": timings,
        "memory_kb": memory,
        "bootstrap": {
            "created": bootstrap.created,
            "task_count": bootstrap.task_count,
            "task_manifest_hash": bootstrap.task_manifest_hash,
        },
        "batch": {
            "batch_id": batch.batch_id,
            "current_user_id": batch.current_user_id,
            "base_generation": batch.base_generation,
            "member_count": len(batch.members),
            "member_source_row_indices": sorted(member_indices),
            "sentinel_source_row_index": sentinel_index,
            "enqueue_receipt": asdict(enqueue),
            "status_after_enqueue": asdict(status_after_enqueue),
            "enqueue_working_manifest_unchanged": enqueue_unchanged,
            "changed_source_row_indices": changed_indices,
            "all_members_changed": all_members_changed,
            "sentinel_unchanged": sentinel_unchanged,
            "generation_increment": result.generation - batch.base_generation,
            "generation_exactly_once": generation_once,
            "atomic_publication": atomic_publication,
            "negative_ids_valid": negative_ids_valid,
            "members": member_receipts,
            "final_status": asdict(status_final),
        },
        "authority": {
            "batch_verify_calls": len(verifier.batch_calls),
            "fallback_verify_calls": len(verifier.fallback_calls),
            "inference_resolver_calls": resolver.calls,
        },
        "final_artifacts": final_artifacts,
        "final_manifest": final_manifest_payload,
        "receipt_path": str(output_root / "receipt.json"),
    }
    receipt_path = _write_receipt_atomic(output_root, receipt)
    if receipt_path != Path(receipt["receipt_path"]):  # pragma: no cover
        raise AssertionError("atomic receipt writer returned the wrong path")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--rows",
        dest="rows",
        type=_positive_int,
        default=1000,
        help="copy this many source rows into the disposable root (default: 1000)",
    )
    selection.add_argument(
        "--full-source",
        dest="rows",
        action="store_const",
        const=None,
        help="bootstrap directly from the canonical full source",
    )
    parser.add_argument("--members", type=_positive_int, default=10)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_probe(
        output_root=args.output_root,
        members=args.members,
        rows=args.rows,
    )
    print(
        json.dumps(
            {
                "receipt": receipt["receipt_path"],
                "timings": receipt["timings"],
                "atomic_publication": receipt["batch"]["atomic_publication"],
                "source_unchanged": receipt["source_unchanged"],
                "sampled_images_unchanged": receipt["sampled_images_unchanged"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
