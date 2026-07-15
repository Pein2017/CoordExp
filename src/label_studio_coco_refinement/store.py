"""Durable mutable JSONL store for the COCO refinement workflow.

The source JSONL is never a transaction target.  A store owns one derived
split directory and serializes every sample commit through a non-blocking
advisory lock.  The append-only journal is the recovery authority; the
manifest is published only after the complete working JSONL is durable.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
_ACCEPTED_OBJECT_FIELDS = {
    "bbox_2d",
    "desc",
    "category_id",
    "category_name",
    "coco_ann_id",
    "metadata",
}
_PRESENTATION_METADATA_FIELDS = {
    "color",
    "visual_color",
    "visual_group",
    "visual_badge",
    "visual_policy",
}


class StoreError(RuntimeError):
    """Base error for a working dataset store."""


class StoreBusyError(StoreError):
    """The split is already owned by another commit/recovery operation."""


class StaleCommitError(StoreError):
    """The request does not describe the currently authoritative base."""


class CommitConflictError(StoreError):
    """A commit id was reused for a different semantic snapshot."""


class CommitRolledBackError(StoreError):
    """The idempotent commit id has a definite rolled-back outcome."""


class CommitOutcomeUnknown(StoreError):
    """The working file may have been replaced; status lookup is required."""


class RecoveryError(StoreError):
    """Journal, working file, and manifest cannot be reconciled safely."""


class ManifestDriftError(StoreError):
    """An existing project does not match its immutable bootstrap contract."""


class ValidationError(StoreError):
    """A Draft or working row violates the store boundary."""


class InjectedCrash(BaseException):
    """Test-only process-stop analogue raised by :class:`FaultInjector`."""


class CategoryRegistry(Protocol):
    """Narrow sibling-registry boundary; the store owns no COCO ID table."""

    def validate(self, category_name: str, category_id: int) -> Any: ...


class AuthoritativeAnnotationVerifier(Protocol):
    """Durable Label Studio lookup required before a Commit can begin."""

    def verify(self, identity: "AuthoritativeDraftIdentity") -> bool: ...


class InferenceReceiptResolver(Protocol):
    """Resolve one durable ROI receipt into its immutable target/result links."""

    def resolve(self, receipt_id: str) -> "InferenceReceiptLink | None": ...


FaultInjector = Callable[[str], None]


class CommitStatus(str, Enum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    OUTCOME_UNKNOWN = "outcome_unknown"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class DraftSaveReceipt:
    """Proof returned only after Label Studio durably saves the frozen Draft."""

    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: int
    semantic_hash: str
    durable: bool = True


@dataclass(frozen=True)
class CommitRequest:
    commit_id: str
    split: str
    image_id: int
    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: int
    semantic_hash: str
    base_row_hash: str
    observed_generation: int
    regions: Sequence[Mapping[str, Any]]
    draft_save: DraftSaveReceipt
    inference_receipts: Sequence[str] = ()


@dataclass(frozen=True)
class AuthoritativeDraftIdentity:
    split: str
    image_id: int
    project_id: str
    task_id: str
    annotation_id: str
    draft_id: str
    annotation_revision: int
    semantic_hash: str

    @classmethod
    def from_request(cls, request: CommitRequest) -> "AuthoritativeDraftIdentity":
        return cls(
            split=request.split,
            image_id=request.image_id,
            project_id=request.project_id,
            task_id=request.task_id,
            annotation_id=request.annotation_id,
            draft_id=request.draft_id,
            annotation_revision=request.annotation_revision,
            semantic_hash=request.semantic_hash,
        )


@dataclass(frozen=True)
class InferenceReceiptLink:
    receipt_id: str
    request_id: str
    project_id: str
    task_id: str
    image_id: int
    annotation_id: str
    terminal_status: str
    result_region_keys: Mapping[str, str]


@dataclass(frozen=True)
class CommitResult:
    commit_id: str
    status: CommitStatus
    split: str
    image_id: int
    generation: int
    row_hash: str
    semantic_hash: str
    region_id_mapping: Mapping[str, int]
    committed_row: Mapping[str, Any]


@dataclass(frozen=True)
class DraftRestore:
    split: str
    image_id: int
    generation: int
    row_hash: str
    row: Mapping[str, Any]
    region_id_mapping: Mapping[str, int]


@dataclass(frozen=True)
class BootstrapSpec:
    split: str
    source_path: Path
    runtime_root: Path
    image_root: Path
    expected_source_sha256: str
    project_id: str
    storage_id: str
    adapter_version: str
    vendor_revision: str
    registry_fingerprint: str
    label_config_fingerprint: str
    instance_id: str = "label-studio-coco-refinement"
    storage_subdir: str | None = None
    extra_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSeed:
    """Parent-adapter seed: exactly one editable annotation, no prediction."""

    task_id: str
    split: str
    image_id: int
    image_locator: str
    authoritative_annotation_id: str
    annotations: Sequence[Mapping[str, Any]]
    predictions: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class BootstrapResult:
    store: "WorkingDatasetStore"
    created: bool
    task_count: int
    task_manifest_hash: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_semantic_projection(regions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project Draft semantics while excluding UI order and view metadata."""

    projected: list[dict[str, Any]] = []
    for region in regions:
        key = _region_key(region)
        projected.append(
            {
                "region_key": key,
                "bbox_2d": list(region.get("bbox_2d", ())),
                "category_name": region.get("category_name", region.get("desc")),
                "category_id": region.get("category_id"),
                "coco_ann_id": region.get("coco_ann_id"),
            }
        )
    projected.sort(key=lambda item: item["region_key"])
    return projected


def semantic_hash(regions: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(canonical_semantic_projection(regions))


def _request_identity_payload(
    request: CommitRequest,
    materialized_region_projection: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = request.draft_save
    return {
        "split": request.split,
        "image_id": request.image_id,
        "project_id": request.project_id,
        "task_id": request.task_id,
        "annotation_id": request.annotation_id,
        "draft_id": request.draft_id,
        "annotation_revision": request.annotation_revision,
        "semantic_hash": request.semantic_hash,
        "base_row_hash": request.base_row_hash,
        "observed_generation": request.observed_generation,
        "draft_save": {
            "project_id": receipt.project_id,
            "task_id": receipt.task_id,
            "annotation_id": receipt.annotation_id,
            "draft_id": receipt.draft_id,
            "annotation_revision": receipt.annotation_revision,
            "semantic_hash": receipt.semantic_hash,
            "durable": receipt.durable,
        },
        "inference_receipts": list(request.inference_receipts),
        "materialized_region_projection": copy.deepcopy(materialized_region_projection),
        "materialized_region_projection_hash": sha256_json(materialized_region_projection),
    }


class WorkingDatasetStore:
    """One split's canonical mutable ``working.norm.jsonl`` authority."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        annotation_verifier: AuthoritativeAnnotationVerifier,
        inference_receipt_resolver: InferenceReceiptResolver,
        registry: CategoryRegistry | None = None,
        fault_injector: FaultInjector | None = None,
        recover: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir).resolve()
        self.working_path = self.split_dir / "working.norm.jsonl"
        self.manifest_path = self.split_dir / "project.json"
        self.journal_path = self.split_dir / "journal.jsonl"
        self.lock_path = self.split_dir / ".commit.lock"
        self.registry = registry if registry is not None else _default_registry()
        self.annotation_verifier = annotation_verifier
        self.inference_receipt_resolver = inference_receipt_resolver
        self._fault_injector = fault_injector
        self._records: list[dict[str, Any]] = []
        self._region_to_id: dict[tuple[int, str], int] = {}
        self._id_to_region: dict[int, tuple[int, str]] = {}
        self._tombstones: set[int] = set()
        self._reserved_negative_ids: set[int] = set()
        if recover:
            self.recover()
        else:
            self._reload_journal_index()

    @classmethod
    def bootstrap(
        cls,
        spec: BootstrapSpec,
        *,
        annotation_verifier: AuthoritativeAnnotationVerifier,
        inference_receipt_resolver: InferenceReceiptResolver,
        registry: CategoryRegistry | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> BootstrapResult:
        """Idempotently create one derived split without mutating the source.

        Label Studio API publication remains an adapter concern.  The returned
        :meth:`iter_task_seeds` surface provides stable ``(split, image_id)``
        task identities and exactly one authoritative editable annotation.
        """

        split = _validate_split(spec.split)
        registry = registry if registry is not None else _default_registry()
        source = spec.source_path.resolve(strict=True)
        _validate_selected_source(source, split)
        image_root = spec.image_root.resolve(strict=True)
        if not source.is_file() or not image_root.is_dir():
            raise ValidationError("source_path must be a file and image_root a directory")
        actual_source_hash = sha256_file(source)
        if actual_source_hash != spec.expected_source_sha256:
            raise ManifestDriftError("source_sha256")

        split_dir = (spec.runtime_root / split).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)
        images_link = split_dir / "images"
        static_contract = {
            "schema_version": SCHEMA_VERSION,
            "instance_id": spec.instance_id,
            "split": split,
            "source_path": str(source),
            "source_sha256": actual_source_hash,
            "image_root": str(image_root),
            "document_root": str(image_root),
            "managed_image_link": str(images_link),
            "project_id": spec.project_id,
            "storage": {
                "storage_id": spec.storage_id,
                "subdir": spec.storage_subdir or f"{split}2017",
                "project_bound": True,
            },
            "adapter_version": spec.adapter_version,
            "vendor_revision": spec.vendor_revision,
            "registry_fingerprint": spec.registry_fingerprint,
            "label_config_fingerprint": spec.label_config_fingerprint,
            "extra_fingerprints": dict(spec.extra_fingerprints),
            "task_policy": {
                "identity": "(split,image_id)",
                "authoritative_annotations": 1,
                "predictions": 0,
                "alternate_annotations_enabled": False,
                "native_submit_enabled": False,
                "native_skip_enabled": False,
            },
        }

        if (split_dir / "project.json").exists():
            store = cls(
                split_dir,
                annotation_verifier=annotation_verifier,
                inference_receipt_resolver=inference_receipt_resolver,
                registry=registry,
                fault_injector=fault_injector,
            )
            manifest = store._read_manifest()
            for key, expected in static_contract.items():
                if manifest.get(key) != expected:
                    raise ManifestDriftError(key)
            _validate_managed_link(images_link, image_root)
            if sha256_file(store.working_path) != manifest["working_sha256"]:
                raise ManifestDriftError("working_sha256")
            return BootstrapResult(
                store=store,
                created=False,
                task_count=int(manifest["task_count"]),
                task_manifest_hash=str(manifest["task_manifest_hash"]),
            )

        unexpected = [
            path.name
            for path in split_dir.iterdir()
            if path.name not in {".commit.lock"}
        ]
        if unexpected:
            raise ManifestDriftError(f"partial bootstrap state: {sorted(unexpected)}")
        os.symlink(image_root, images_link, target_is_directory=True)
        _validate_managed_link(images_link, image_root)

        task_digest = hashlib.sha256()
        task_count = 0
        split_object_ids: dict[int, int] = {}
        split_image_ids: set[int] = set()
        working_tmp: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".working.norm.jsonl.", dir=split_dir)
            working_tmp = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output, source.open(
                "r", encoding="utf-8"
            ) as input_handle:
                for line_no, line in enumerate(input_handle, start=1):
                    row = _parse_jsonl_line(line, source, line_no)
                    _validate_source_row(row, split, registry)
                    image_id = int(row["image_id"])
                    if image_id in split_image_ids:
                        raise ValidationError(f"split-wide duplicate image_id: {image_id}")
                    split_image_ids.add(image_id)
                    for obj in row["objects"]:
                        object_id = int(obj["coco_ann_id"])
                        prior_image = split_object_ids.setdefault(object_id, image_id)
                        if prior_image != image_id:
                            raise ValidationError(
                                f"split-wide duplicate coco_ann_id: {object_id}"
                            )
                    row = copy.deepcopy(row)
                    row["images"] = [_working_image_locator(row, split, source, image_root)]
                    encoded = canonical_json(row) + "\n"
                    output.write(encoded)
                    seed_identity = {
                        "task_id": _task_id(split, image_id),
                        "image_id": image_id,
                        "row_hash": sha256_json(row),
                        "source_line": line_no,
                    }
                    task_digest.update((canonical_json(seed_identity) + "\n").encode("utf-8"))
                    task_count += 1
                output.flush()
                os.fsync(output.fileno())
            if task_count == 0:
                raise ValidationError("source JSONL must contain at least one row")
            os.replace(working_tmp, split_dir / "working.norm.jsonl")
            _fsync_directory(split_dir)
            working_tmp = None

            (split_dir / "journal.jsonl").touch(exist_ok=False)
            with (split_dir / "journal.jsonl").open("ab") as journal:
                journal.flush()
                os.fsync(journal.fileno())
            manifest = {
                **static_contract,
                "task_count": task_count,
                "task_manifest_hash": task_digest.hexdigest(),
                "generation": 0,
                "working_sha256": sha256_file(split_dir / "working.norm.jsonl"),
                "last_commit_id": None,
            }
            _atomic_replace_json(split_dir / "project.json", manifest)
        except BaseException:
            if working_tmp is not None:
                working_tmp.unlink(missing_ok=True)
            raise

        store = cls(
            split_dir,
            annotation_verifier=annotation_verifier,
            inference_receipt_resolver=inference_receipt_resolver,
            registry=registry,
            fault_injector=fault_injector,
        )
        return BootstrapResult(
            store=store,
            created=True,
            task_count=task_count,
            task_manifest_hash=task_digest.hexdigest(),
        )

    def iter_task_seeds(self) -> Iterator[TaskSeed]:
        manifest = self._read_manifest()
        split = str(manifest["split"])
        for _, row, _ in self._iter_rows():
            image_id = int(row["image_id"])
            task_id = _task_id(split, image_id)
            regions = []
            for obj in row["objects"]:
                obj_copy = copy.deepcopy(obj)
                obj_copy["region_key"] = self._key_for_object(obj_copy, image_id, split)
                regions.append(obj_copy)
            yield TaskSeed(
                task_id=task_id,
                split=split,
                image_id=image_id,
                image_locator=str(row["images"][0]),
                authoritative_annotation_id=f"{task_id}:annotation",
                annotations=({"id": f"{task_id}:annotation", "regions": regions},),
            )

    def commit(self, request: CommitRequest) -> CommitResult:
        """Commit the exact durably-saved Draft or return its prior outcome."""

        self._validate_draft_handshake(request)
        with self._exclusive_lock():
            self._reload_journal_index()
            prior = self._records_for_commit(request.commit_id)
            if prior:
                return self._resolve_idempotent_request(request, prior)
            manifest = self._read_manifest()
            if request.split != manifest["split"]:
                raise StaleCommitError("split mismatch")
            if request.project_id != manifest["project_id"]:
                raise StaleCommitError("project mismatch")
            if request.task_id != _task_id(request.split, request.image_id):
                raise StaleCommitError("task identity mismatch")
            if not self.annotation_verifier.verify(AuthoritativeDraftIdentity.from_request(request)):
                raise StaleCommitError("authoritative annotation snapshot was not attested")
            if request.observed_generation != int(manifest["generation"]):
                raise StaleCommitError("project generation changed")
            if sha256_file(self.working_path) != manifest["working_sha256"]:
                raise RecoveryError("working file does not match the manifest")

            row_index, before_row, _ = self._find_row(request.image_id)
            before_hash = sha256_json(before_row)
            if before_hash != request.base_row_hash:
                raise StaleCommitError("base row hash changed")
            if not request.regions:
                raise ValidationError("empty Draft is preserved, but V1 Commit requires an object")

            self._validate_inference_linkage(request)
            after_objects, mapping, allocations, materialized_projection = self._materialize_objects(
                before_row,
                request.regions,
                split=request.split,
                image_id=request.image_id,
            )
            after_row = copy.deepcopy(before_row)
            after_row["objects"] = after_objects
            self._validate_row(after_row)
            after_hash = sha256_json(after_row)
            tombstones = sorted(
                {int(obj["coco_ann_id"]) for obj in before_row["objects"]}
                - {int(obj["coco_ann_id"]) for obj in after_objects}
            )

            candidate_generation = int(manifest["generation"]) + 1
            candidate_working_hash = self._candidate_working_hash(row_index, after_row)
            candidate_manifest = copy.deepcopy(manifest)
            candidate_manifest.update(
                {
                    "generation": candidate_generation,
                    "working_sha256": candidate_working_hash,
                    "last_commit_id": request.commit_id,
                }
            )
            candidate_manifest_hash = sha256_json(candidate_manifest)
            request_identity = _request_identity_payload(request, materialized_projection)
            prepared = {
                "kind": "prepared",
                "commit_id": request.commit_id,
                "split": request.split,
                "image_id": request.image_id,
                "project_id": request.project_id,
                "task_id": request.task_id,
                "annotation_id": request.annotation_id,
                "draft_id": request.draft_id,
                "annotation_revision": request.annotation_revision,
                "semantic_hash": request.semantic_hash,
                "request_identity": request_identity,
                "request_identity_hash": sha256_json(request_identity),
                "base_generation": int(manifest["generation"]),
                "candidate_generation": candidate_generation,
                "before_working_sha256": manifest["working_sha256"],
                "candidate_working_sha256": candidate_working_hash,
                "before_manifest_hash": sha256_json(manifest),
                "candidate_manifest": candidate_manifest,
                "candidate_manifest_hash": candidate_manifest_hash,
                "before_row": before_row,
                "after_row": after_row,
                "before_row_hash": before_hash,
                "after_row_hash": after_hash,
                "region_id_mapping": mapping,
                "allocated_mappings": allocations,
                "tombstones": tombstones,
                "inference_receipts": list(request.inference_receipts),
            }
            prepared = self._append_record(prepared, "prepared")

            working_replaced = False
            try:
                self._rewrite_working(row_index, after_row, candidate_working_hash)
                working_replaced = True
                self._publish_manifest(candidate_manifest, candidate_manifest_hash)
                self._append_terminal(prepared, CommitStatus.COMMITTED)
                self._fault("before_response")
            except InjectedCrash:
                raise
            except CommitOutcomeUnknown:
                raise
            except Exception as exc:
                if working_replaced:
                    raise CommitOutcomeUnknown(
                        f"commit {request.commit_id} outcome requires reconciliation"
                    ) from exc
                self._append_terminal(prepared, CommitStatus.ROLLED_BACK, error=str(exc))
                raise

            self._reload_journal_index()
            return self._result_from_prepared(prepared, CommitStatus.COMMITTED)

    def status(self, commit_id: str) -> CommitStatus:
        self._reload_journal_index()
        records = self._records_for_commit(commit_id)
        if not records:
            return CommitStatus.NOT_FOUND
        terminal = next((r for r in reversed(records) if r["kind"] == "terminal"), None)
        if terminal is None:
            return CommitStatus.OUTCOME_UNKNOWN
        return CommitStatus(terminal["status"])

    def result(self, commit_id: str) -> CommitResult:
        self._reload_journal_index()
        records = self._records_for_commit(commit_id)
        if not records:
            raise StoreError(f"unknown commit id: {commit_id}")
        prepared = next(record for record in records if record["kind"] == "prepared")
        status = self.status(commit_id)
        if status is CommitStatus.OUTCOME_UNKNOWN:
            raise CommitOutcomeUnknown(commit_id)
        if status is CommitStatus.ROLLED_BACK:
            raise CommitRolledBackError(commit_id)
        return self._result_from_prepared(prepared, status)

    def restore_draft(self, image_id: int) -> DraftRestore:
        """Return the committed row used by persisted-Draft reset/reload hooks."""

        manifest = self._read_manifest()
        _, row, _ = self._find_row(image_id)
        split = str(manifest["split"])
        mapping = {
            self._key_for_object(obj, image_id, split): int(obj["coco_ann_id"])
            for obj in row["objects"]
        }
        return DraftRestore(
            split=split,
            image_id=image_id,
            generation=int(manifest["generation"]),
            row_hash=sha256_json(row),
            row=row,
            region_id_mapping=mapping,
        )

    def recover(self) -> None:
        """Reconcile an interrupted transaction exactly once before serving."""

        if not self.split_dir.exists():
            raise RecoveryError(f"missing split directory: {self.split_dir}")
        with self._exclusive_lock():
            self._reload_journal_index(repair_torn_tail=True)
            for prepared in [r for r in self._records if r["kind"] == "prepared"]:
                if any(
                    r["kind"] == "terminal" and r.get("prepared_record_hash") == prepared["record_hash"]
                    for r in self._records
                ):
                    continue
                self._recover_prepared(prepared)
                self._reload_journal_index()
            self._validate_authority()

    def _recover_prepared(self, prepared: Mapping[str, Any]) -> None:
        manifest = self._read_manifest()
        working_hash = sha256_file(self.working_path)
        manifest_hash = sha256_json(manifest)
        before_working = prepared["before_working_sha256"]
        candidate_working = prepared["candidate_working_sha256"]
        before_manifest = prepared["before_manifest_hash"]
        candidate_manifest = prepared["candidate_manifest_hash"]

        if working_hash == before_working and manifest_hash == before_manifest:
            self._append_terminal(prepared, CommitStatus.ROLLED_BACK, recovery=True)
            return
        if working_hash == candidate_working and manifest_hash == before_manifest:
            self._publish_manifest(
                prepared["candidate_manifest"], str(candidate_manifest), recovery=True
            )
            self._append_terminal(prepared, CommitStatus.COMMITTED, recovery=True)
            return
        if working_hash == candidate_working and manifest_hash == candidate_manifest:
            self._append_terminal(prepared, CommitStatus.COMMITTED, recovery=True)
            return
        raise RecoveryError(
            "journal/hash disagreement for commit "
            f"{prepared['commit_id']}: working={working_hash}, manifest={manifest_hash}"
        )

    def _validate_draft_handshake(self, request: CommitRequest) -> None:
        receipt = request.draft_save
        if not receipt.durable:
            raise ValidationError("Commit requires a durable Draft-save receipt")
        matched = (
            receipt.project_id == request.project_id
            and receipt.task_id == request.task_id
            and receipt.annotation_id == request.annotation_id
            and receipt.draft_id == request.draft_id
            and receipt.annotation_revision == request.annotation_revision
            and receipt.semantic_hash == request.semantic_hash
        )
        if not matched:
            raise StaleCommitError("Draft-save receipt does not match the Commit snapshot")
        actual_semantic_hash = semantic_hash(request.regions)
        if actual_semantic_hash != request.semantic_hash:
            raise StaleCommitError("submitted regions do not match the saved Draft hash")

    def _validate_inference_linkage(self, request: CommitRequest) -> None:
        declared = tuple(request.inference_receipts)
        if any(not isinstance(value, str) or not value for value in declared):
            raise ValidationError("declared inference receipt IDs must be non-empty text")
        if len(set(declared)) != len(declared):
            raise ValidationError("declared inference receipt IDs must be unique")

        used: set[str] = set()
        resolved: dict[str, InferenceReceiptLink] = {}
        provenance_fields = {"receipt_id", "request_id", "result_id"}
        for region in request.regions:
            metadata = region.get("metadata")
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise ValidationError("metadata must be a mapping")
            has_provenance = bool(provenance_fields & set(metadata))
            inference_origin = metadata.get("inference_origin")
            if not has_provenance and inference_origin is not True:
                continue
            if inference_origin is not True or not provenance_fields.issubset(metadata):
                raise ValidationError(
                    "inference-origin objects require complete inference linkage metadata"
                )
            receipt_id = metadata["receipt_id"]
            request_id = metadata["request_id"]
            result_id = metadata["result_id"]
            if any(
                not isinstance(value, str) or not value
                for value in (receipt_id, request_id, result_id)
            ):
                raise ValidationError(
                    "inference-origin objects require complete inference linkage metadata"
                )
            if receipt_id not in declared:
                raise ValidationError(
                    f"object receipt {receipt_id!r} is not a declared inference receipt"
                )
            link = resolved.get(receipt_id)
            if link is None:
                link = self.inference_receipt_resolver.resolve(receipt_id)
                if link is None:
                    raise ValidationError(f"unknown inference receipt: {receipt_id}")
                resolved[receipt_id] = link
            if (
                link.receipt_id != receipt_id
                or link.request_id != request_id
                or link.project_id != request.project_id
                or link.task_id != request.task_id
                or link.image_id != request.image_id
                or link.annotation_id != request.annotation_id
                or link.terminal_status not in {"accepted", "accepted_with_drops"}
            ):
                raise ValidationError(f"inference receipt target mismatch: {receipt_id}")
            key = _region_key(region)
            if link.result_region_keys.get(result_id) != key:
                raise ValidationError(
                    f"inference receipt result linkage mismatch: {receipt_id}/{result_id}"
                )
            used.add(receipt_id)

        if used != set(declared):
            unlinked = sorted(set(declared) - used)
            raise ValidationError(
                f"declared inference receipt has no committed object linkage: {unlinked}"
            )

    def _resolve_idempotent_request(
        self, request: CommitRequest, records: Sequence[Mapping[str, Any]]
    ) -> CommitResult:
        prepared = next(record for record in records if record["kind"] == "prepared")
        try:
            _, _, _, materialized_projection = self._materialize_objects(
                prepared["before_row"],
                request.regions,
                split=request.split,
                image_id=request.image_id,
                identity_region_id_mapping=prepared["region_id_mapping"],
            )
            identity = _request_identity_payload(request, materialized_projection)
        except Exception as exc:
            raise CommitConflictError(
                "commit id reused with a different immutable request identity"
            ) from exc
        if (
            prepared.get("request_identity") != identity
            or prepared.get("request_identity_hash") != sha256_json(identity)
        ):
            raise CommitConflictError(
                "commit id reused with a different immutable request identity"
            )
        terminal = next((r for r in reversed(records) if r["kind"] == "terminal"), None)
        if terminal is None:
            raise CommitOutcomeUnknown(request.commit_id)
        status = CommitStatus(terminal["status"])
        if status is CommitStatus.ROLLED_BACK:
            raise CommitRolledBackError(request.commit_id)
        return self._result_from_prepared(prepared, status)

    def _result_from_prepared(
        self, prepared: Mapping[str, Any], status: CommitStatus
    ) -> CommitResult:
        return CommitResult(
            commit_id=str(prepared["commit_id"]),
            status=status,
            split=str(prepared["split"]),
            image_id=int(prepared["image_id"]),
            generation=int(prepared["candidate_generation"]),
            row_hash=str(prepared["after_row_hash"]),
            semantic_hash=str(prepared["semantic_hash"]),
            region_id_mapping={
                str(key): int(value) for key, value in prepared["region_id_mapping"].items()
            },
            committed_row=copy.deepcopy(prepared["after_row"]),
        )

    def _materialize_objects(
        self,
        before_row: Mapping[str, Any],
        regions: Sequence[Mapping[str, Any]],
        *,
        split: str,
        image_id: int,
        identity_region_id_mapping: Mapping[str, Any] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
        dict[str, int],
        dict[str, Any],
    ]:
        before_ids = {int(obj["coco_ann_id"]): obj for obj in before_row["objects"]}
        before_rank = {
            int(obj["coco_ann_id"]): rank for rank, obj in enumerate(before_row["objects"])
        }
        used_ids: set[int] = set()
        mapping: dict[str, int] = {}
        allocations: dict[str, int] = {}
        materialized: list[tuple[dict[str, Any], str, int | None, int | None]] = []

        if identity_region_id_mapping is None:
            region_to_id = self._region_to_id
            reserved_negative_ids = self._reserved_negative_ids
            tombstones = self._tombstones
        else:
            # Retry identity is reconstructed from the original prepared record,
            # never from mappings or tombstones introduced by later commits.
            region_to_id = {
                (image_id, str(key)): int(value)
                for key, value in identity_region_id_mapping.items()
            }
            reserved_negative_ids = {
                object_id for object_id in region_to_id.values() if object_id < 0
            }
            tombstones = set()

        next_negative = min(reserved_negative_ids | {0}) - 1
        for ordinal, region in enumerate(regions):
            key = _region_key(region)
            if key in mapping:
                raise ValidationError(f"duplicate region key: {key}")
            supplied_id = region.get("coco_ann_id")
            namespace = (image_id, key)
            known_id = region_to_id.get(namespace)
            implicit_source_id = _source_id_from_region_key(key, split)
            if implicit_source_id is not None:
                if implicit_source_id not in before_ids:
                    if implicit_source_id in tombstones:
                        raise ValidationError(
                            f"tombstoned source identity cannot be restored: {key!r}"
                        )
                    raise ValidationError(
                        f"source identity {key!r} is not present in the current before row"
                    )
                known_id = implicit_source_id
            elif supplied_id is not None and (
                isinstance(supplied_id, bool)
                or not isinstance(supplied_id, int)
                or supplied_id > 0
            ):
                raise ValidationError(
                    "positive source identity requires its canonical split:coco:<id> region key"
                )
            if known_id is not None and supplied_id not in (None, known_id):
                raise ValidationError(f"region {key!r} changed its hidden identity")
            if known_id is not None:
                object_id = known_id
            elif supplied_id is not None:
                raise ValidationError("unmapped coco_ann_id values are not accepted from the Draft")
            else:
                while next_negative in reserved_negative_ids or next_negative in used_ids:
                    next_negative -= 1
                object_id = next_negative
                next_negative -= 1
                allocations[key] = object_id
            if object_id in used_ids:
                raise ValidationError(f"duplicate coco_ann_id: {object_id}")
            if object_id in tombstones and before_ids.get(object_id) is None:
                raise ValidationError(f"tombstoned coco_ann_id cannot be restored: {object_id}")
            used_ids.add(object_id)
            mapping[key] = object_id

            name = region.get("category_name", region.get("desc"))
            category_id = region.get("category_id")
            if not isinstance(name, str) or not isinstance(category_id, int):
                raise ValidationError("category_name and integer category_id are required")
            self.registry.validate(name, category_id)
            bbox = _validate_bbox(region.get("bbox_2d"))
            obj: dict[str, Any] = {
                "bbox_2d": bbox,
                "desc": name,
                "category_id": category_id,
                "category_name": name,
                "coco_ann_id": object_id,
            }
            metadata = _training_metadata(region.get("metadata"))
            if metadata:
                obj["metadata"] = metadata
            prior_rank = before_rank.get(object_id)
            creation_ordinal = (
                None if prior_rank is not None else int(region.get("creation_ordinal", ordinal))
            )
            materialized.append((obj, key, prior_rank, creation_ordinal))

        objects = _stable_order(materialized)
        ordering_seeds = [
            {
                "region_key": key,
                "coco_ann_id": int(obj["coco_ann_id"]),
                "prior_rank": prior_rank,
                "creation_ordinal": creation_ordinal,
            }
            for obj, key, prior_rank, creation_ordinal in materialized
        ]
        ordering_seeds.sort(key=lambda value: value["region_key"])
        projection = {
            "objects": copy.deepcopy(objects),
            "ordering_seeds": ordering_seeds,
        }
        return objects, mapping, allocations, projection

    def _validate_row(self, row: Mapping[str, Any]) -> None:
        try:
            from .models import WorkingRow
        except ImportError:  # pragma: no cover - partial sibling landing fallback
            WorkingRow = None  # type: ignore[assignment,misc]
        if WorkingRow is not None:
            try:
                WorkingRow.from_mapping(row)
            except Exception as exc:
                raise ValidationError(str(exc)) from exc
            return
        required = {"images", "objects", "width", "height", "image_id", "file_name"}
        if not required.issubset(row):
            raise ValidationError(f"row missing fields: {sorted(required - set(row))}")
        if not isinstance(row["objects"], list) or not row["objects"]:
            raise ValidationError("working row objects must be non-empty")
        ids: set[int] = set()
        for obj in row["objects"]:
            if set(obj) - _ACCEPTED_OBJECT_FIELDS:
                raise ValidationError("working object contains unsupported fields")
            _validate_bbox(obj.get("bbox_2d"))
            name = obj.get("category_name")
            if obj.get("desc") != name or not isinstance(name, str):
                raise ValidationError("desc must equal canonical category_name")
            category_id = obj.get("category_id")
            if not isinstance(category_id, int):
                raise ValidationError("category_id must be an integer")
            self.registry.validate(name, category_id)
            object_id = obj.get("coco_ann_id")
            if not isinstance(object_id, int) or object_id == 0 or object_id in ids:
                raise ValidationError("coco_ann_id must be a unique nonzero integer")
            ids.add(object_id)

    def _candidate_working_hash(self, row_index: int, after_row: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        object_owners: dict[int, int] = {}
        image_ids: set[int] = set()
        for index, current_row, raw in self._iter_rows():
            row = after_row if index == row_index else current_row
            self._validate_row(row)
            _register_split_wide_ids(row, object_owners, image_ids, error_type=ValidationError)
            encoded = (
                (canonical_json(after_row) + "\n").encode("utf-8")
                if index == row_index
                else raw
            )
            digest.update(encoded)
        return digest.hexdigest()

    def _rewrite_working(
        self, row_index: int, after_row: Mapping[str, Any], expected_hash: str
    ) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=".working.norm.jsonl.", dir=self.split_dir)
        temp_path = Path(temp_name)
        replaced = False
        try:
            with os.fdopen(fd, "wb") as output:
                for index, _, raw in self._iter_rows():
                    output.write(
                        (canonical_json(after_row) + "\n").encode("utf-8")
                        if index == row_index
                        else raw
                    )
                output.flush()
                self._fault("working_temp_flushed")
                os.fsync(output.fileno())
                self._fault("working_temp_fsynced")
            if sha256_file(temp_path) != expected_hash:
                raise RecoveryError("candidate working hash disagrees with prepared record")
            os.replace(temp_path, self.working_path)
            replaced = True
            self._fault("working_replaced")
            _fsync_directory(self.split_dir)
            self._fault("working_directory_fsynced")
        except InjectedCrash:
            raise
        except Exception as exc:
            if replaced:
                raise CommitOutcomeUnknown(
                    "working JSONL was replaced before a durability step failed"
                ) from exc
            raise
        finally:
            temp_path.unlink(missing_ok=True)

    def _publish_manifest(
        self,
        manifest: Mapping[str, Any],
        expected_hash: str,
        *,
        recovery: bool = False,
    ) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=".project.json.", dir=self.split_dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                output.write(canonical_json(manifest) + "\n")
                output.flush()
                self._fault("manifest_temp_flushed", recovery=recovery)
                os.fsync(output.fileno())
                self._fault("manifest_temp_fsynced", recovery=recovery)
            if sha256_json(manifest) != expected_hash:
                raise RecoveryError("candidate manifest hash disagrees with prepared record")
            os.replace(temp_path, self.manifest_path)
            self._fault("manifest_replaced", recovery=recovery)
            _fsync_directory(self.split_dir)
            self._fault("manifest_directory_fsynced", recovery=recovery)
        finally:
            temp_path.unlink(missing_ok=True)

    def _append_record(self, record: Mapping[str, Any], stage: str) -> dict[str, Any]:
        payload = copy.deepcopy(dict(record))
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["prev_record_hash"] = self._records[-1]["record_hash"] if self._records else None
        payload["record_hash"] = sha256_json(payload)
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        with self.journal_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            self._fault(f"{stage}_journal_flushed")
            os.fsync(handle.fileno())
            self._fault(f"{stage}_journal_fsynced")
        self._records.append(payload)
        return payload

    def _append_terminal(
        self,
        prepared: Mapping[str, Any],
        status: CommitStatus,
        *,
        recovery: bool = False,
        error: str | None = None,
    ) -> None:
        if any(
            record["kind"] == "terminal"
            and record.get("prepared_record_hash") == prepared["record_hash"]
            for record in self._records
        ):
            return
        terminal = {
            "kind": "terminal",
            "commit_id": prepared["commit_id"],
            "prepared_record_hash": prepared["record_hash"],
            "status": status.value,
            "generation": (
                prepared["candidate_generation"]
                if status is CommitStatus.COMMITTED
                else prepared["base_generation"]
            ),
            "recovery": recovery,
            "error": error,
        }
        self._append_record(terminal, "terminal")

    def _reload_journal_index(self, *, repair_torn_tail: bool = False) -> None:
        self._records = []
        self._region_to_id = {}
        self._id_to_region = {}
        self._tombstones = set()
        self._reserved_negative_ids = set()
        self._records = self._read_journal_records(repair_torn_tail=repair_torn_tail)

        prepared_by_hash: dict[str, dict[str, Any]] = {}
        prepared_commit_ids: set[str] = set()
        terminal_prepared_hashes: set[str] = set()
        for record in self._records:
            kind = record.get("kind")
            if kind == "prepared":
                commit_id = str(record.get("commit_id"))
                if commit_id in prepared_commit_ids:
                    raise RecoveryError(f"duplicate prepared commit id: {commit_id}")
                prepared_commit_ids.add(commit_id)
                prepared_by_hash[str(record["record_hash"])] = record
                continue
            if kind != "terminal":
                raise RecoveryError(f"unsupported journal record kind: {kind!r}")
            prepared_hash = str(record.get("prepared_record_hash"))
            prepared = prepared_by_hash.get(prepared_hash)
            if prepared is None:
                raise RecoveryError("terminal journal record has no preceding prepared record")
            if prepared_hash in terminal_prepared_hashes:
                raise RecoveryError("prepared journal record has multiple terminal outcomes")
            terminal_prepared_hashes.add(prepared_hash)
            if record.get("commit_id") != prepared.get("commit_id"):
                raise RecoveryError("terminal commit identity disagrees with prepared record")
            if record.get("status") not in {
                CommitStatus.COMMITTED.value,
                CommitStatus.ROLLED_BACK.value,
            }:
                raise RecoveryError("terminal journal status is invalid")

        committed_prepared = {
            prepared_hash
            for prepared_hash in terminal_prepared_hashes
            if next(
                record
                for record in self._records
                if record.get("prepared_record_hash") == prepared_hash
            )["status"]
            == CommitStatus.COMMITTED.value
        }
        for record in self._records:
            if record["kind"] != "prepared":
                continue
            image_id = int(record["image_id"])
            for key, value in record.get("allocated_mappings", {}).items():
                self._register_region_mapping(image_id, str(key), int(value))
                self._reserved_negative_ids.add(int(value))
            if record["record_hash"] in committed_prepared:
                for key, value in record.get("region_id_mapping", {}).items():
                    self._register_region_mapping(image_id, str(key), int(value))
                self._tombstones.update(int(value) for value in record.get("tombstones", ()))

    def _read_journal_records(self, *, repair_torn_tail: bool) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        data = self.journal_path.read_bytes()
        complete = data
        tail = b""
        if data and not data.endswith(b"\n"):
            boundary = data.rfind(b"\n") + 1
            complete = data[:boundary]
            tail = data[boundary:]

        previous_hash: str | None = None
        records: list[dict[str, Any]] = []
        for line_no, raw in enumerate(complete.splitlines(keepends=True), start=1):
            try:
                line = raw.decode("utf-8")
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RecoveryError(f"invalid journal record at line {line_no}") from exc
            if record.get("prev_record_hash") != previous_hash:
                raise RecoveryError(f"broken journal chain at line {line_no}")
            recorded_hash = record.get("record_hash")
            check = dict(record)
            check.pop("record_hash", None)
            if recorded_hash != sha256_json(check):
                raise RecoveryError(f"journal hash disagreement at line {line_no}")
            previous_hash = recorded_hash
            records.append(record)

        if tail:
            if not repair_torn_tail:
                raise RecoveryError(f"incomplete journal record at line {len(records) + 1}")
            if not tail.startswith(b"{") or b"\n" in tail:
                raise RecoveryError("final journal bytes are not an unambiguous torn frame")
            with self.journal_path.open("r+b") as handle:
                handle.truncate(len(complete))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.split_dir)
        return records

    def _register_region_mapping(self, image_id: int, key: str, object_id: int) -> None:
        namespace = (image_id, key)
        prior = self._region_to_id.setdefault(namespace, object_id)
        if prior != object_id:
            raise RecoveryError(
                f"region identity disagreement for image {image_id}, key {key!r}"
            )
        prior_owner = self._id_to_region.setdefault(object_id, namespace)
        if prior_owner != namespace:
            raise RecoveryError(
                f"split-wide coco_ann_id {object_id} is mapped to multiple regions"
            )

    def _validate_authority(self) -> None:
        manifest = self._read_manifest()
        actual = sha256_file(self.working_path)
        if actual != manifest.get("working_sha256"):
            raise RecoveryError("working JSONL does not match published manifest")
        object_owners: dict[int, int] = {}
        image_ids: set[int] = set()
        for _, row, _ in self._iter_rows():
            self._validate_row(row)
            _register_split_wide_ids(row, object_owners, image_ids, error_type=RecoveryError)
            image_id = int(row["image_id"])
            for obj in row["objects"]:
                object_id = int(obj["coco_ann_id"])
                if object_id < 0:
                    owner = self._id_to_region.get(object_id)
                    if owner is None or owner[0] != image_id:
                        raise RecoveryError(
                            f"negative coco_ann_id {object_id} has no authoritative journal mapping"
                        )

    def _records_for_commit(self, commit_id: str) -> list[dict[str, Any]]:
        return [record for record in self._records if record.get("commit_id") == commit_id]

    def _read_manifest(self) -> dict[str, Any]:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryError(f"cannot read manifest: {self.manifest_path}") from exc
        return manifest

    def _iter_rows(self) -> Iterator[tuple[int, dict[str, Any], bytes]]:
        with self.working_path.open("rb") as handle:
            for index, raw in enumerate(handle):
                if not raw.endswith(b"\n"):
                    raise ValidationError("working JSONL must end every row with a newline")
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValidationError(f"invalid working JSONL row {index + 1}") from exc
                yield index, row, raw

    def _find_row(self, image_id: int) -> tuple[int, dict[str, Any], bytes]:
        found: tuple[int, dict[str, Any], bytes] | None = None
        for item in self._iter_rows():
            if int(item[1].get("image_id", -1)) == image_id:
                if found is not None:
                    raise ValidationError(f"duplicate image_id: {image_id}")
                found = item
        if found is None:
            raise StaleCommitError(f"unknown image_id: {image_id}")
        return found

    def _key_for_object(self, obj: Mapping[str, Any], image_id: int, split: str) -> str:
        object_id = int(obj["coco_ann_id"])
        if object_id > 0:
            return _source_region_key(split, object_id)
        keys = [
            key
            for (mapped_image_id, key), value in self._region_to_id.items()
            if mapped_image_id == image_id and value == object_id
        ]
        if len(keys) != 1:
            raise RecoveryError(f"cannot rehydrate region key for {object_id}")
        return keys[0]

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.split_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusyError(f"split is already locked: {self.split_dir}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _fault(self, boundary: str, *, recovery: bool = False) -> None:
        if self._fault_injector is not None and not recovery:
            self._fault_injector(boundary)


def _region_key(region: Mapping[str, Any]) -> str:
    value = region.get("region_key", region.get("stable_region_key"))
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("every Draft region requires a hidden stable region key")
    return value


def _source_region_key(split: str, object_id: int) -> str:
    return f"{split}:coco:{object_id}"


def _source_id_from_region_key(region_key: str, split: str) -> int | None:
    prefix = f"{split}:coco:"
    if not region_key.startswith(prefix):
        if ":coco:" in region_key or region_key.startswith("coco:"):
            raise ValidationError(
                "source region keys must use the current split:coco:<id> namespace"
            )
        return None
    try:
        object_id = int(region_key[len(prefix) :])
    except ValueError as exc:
        raise ValidationError("coco region keys must end in a positive integer") from exc
    if object_id <= 0:
        raise ValidationError("coco region keys must end in a positive integer")
    return object_id


def _register_split_wide_ids(
    row: Mapping[str, Any],
    object_owners: dict[int, int],
    image_ids: set[int],
    *,
    error_type: type[StoreError],
) -> None:
    image_id = int(row["image_id"])
    if image_id in image_ids:
        raise error_type(f"split-wide duplicate image_id: {image_id}")
    image_ids.add(image_id)
    for obj in row["objects"]:
        object_id = int(obj["coco_ann_id"])
        prior_image = object_owners.setdefault(object_id, image_id)
        if prior_image != image_id:
            raise error_type(f"split-wide duplicate coco_ann_id: {object_id}")


def _validate_bbox(value: Any) -> list[int]:
    try:
        from src.data.geometry import validate_bbox_bins
    except ImportError:  # pragma: no cover - partial sibling landing fallback
        validate_bbox_bins = None  # type: ignore[assignment]
    if validate_bbox_bins is not None:
        try:
            return list(validate_bbox_bins(value, field="bbox_2d"))
        except Exception as exc:
            raise ValidationError(str(exc)) from exc
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValidationError("bbox_2d must contain four integer norm1000 edges")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        raise ValidationError("bbox_2d must be strict and inside 0..999")
    return [x1, y1, x2, y2]


def _stable_order(
    values: Sequence[tuple[dict[str, Any], str, int | None, int | None]],
) -> list[dict[str, Any]]:
    try:
        from .models import ObjectIdentity, OrderedWorkingObject, WorkingObject, stable_top_left_order
    except ImportError:  # pragma: no cover - partial sibling landing fallback
        seeded = sorted(
            values,
            key=lambda item: (
                0 if item[2] is not None else 1,
                item[2] if item[2] is not None else item[3],
                item[1],
            ),
        )
        seeded.sort(key=lambda item: (item[0]["bbox_2d"][1], item[0]["bbox_2d"][0]))
        return [item[0] for item in seeded]

    ordered = stable_top_left_order(
        tuple(
            OrderedWorkingObject(
                identity=ObjectIdentity(
                    region_key=key,
                    coco_ann_id=int(obj["coco_ann_id"]),
                    prior_rank=prior_rank,
                    creation_ordinal=creation_ordinal,
                ),
                object=WorkingObject.from_mapping(obj),
            )
            for obj, key, prior_rank, creation_ordinal in values
        )
    )
    return [item.object.to_json_dict() for item in ordered]


def _training_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("metadata must be a mapping")
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in _PRESENTATION_METADATA_FIELDS
    }


def _validate_split(split: str) -> str:
    if split not in {"train", "val"}:
        raise ValidationError("split must be 'train' or 'val'")
    return split


def _validate_selected_source(source: Path, split: str) -> None:
    expected_suffix = Path(
        "public_data/coco/rescale_32_1024_bbox_len12000"
    ) / f"{split}.norm.jsonl"
    if source.parts[-len(expected_suffix.parts) :] != expected_suffix.parts:
        raise ManifestDriftError("source_path")


def _default_registry() -> CategoryRegistry:
    try:
        from .categories import COCO80_REGISTRY
    except ImportError as exc:  # pragma: no cover - only during partial sibling landing
        raise StoreError("Coco80Registry is not available; pass registry explicitly") from exc
    return COCO80_REGISTRY


def _task_id(split: str, image_id: int) -> str:
    return f"{split}:{image_id}"


def _parse_jsonl_line(line: str, source: Path, line_no: int) -> dict[str, Any]:
    if not line.endswith("\n"):
        raise ValidationError(f"source row {line_no} in {source} has no trailing newline")
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid source JSONL row {line_no}") from exc
    if not isinstance(row, dict):
        raise ValidationError(f"source row {line_no} is not an object")
    return row


def _validate_source_row(
    row: Mapping[str, Any], split: str, registry: CategoryRegistry
) -> None:
    required = {"images", "objects", "width", "height", "image_id", "file_name"}
    if not required.issubset(row):
        raise ValidationError(f"source row missing fields: {sorted(required - set(row))}")
    if not isinstance(row["images"], list) or len(row["images"]) != 1:
        raise ValidationError("source row must contain exactly one image locator")
    if not isinstance(row["objects"], list) or not row["objects"]:
        raise ValidationError("source row objects must be non-empty")
    ids: set[int] = set()
    for obj in row["objects"]:
        _validate_bbox(obj.get("bbox_2d"))
        name = obj.get("category_name")
        category_id = obj.get("category_id")
        if obj.get("desc") != name or not isinstance(name, str) or not isinstance(category_id, int):
            raise ValidationError("source class fields are inconsistent")
        registry.validate(name, category_id)
        object_id = obj.get("coco_ann_id")
        if not isinstance(object_id, int) or object_id <= 0 or object_id in ids:
            raise ValidationError("source coco_ann_id must be unique and positive")
        ids.add(object_id)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("split") not in (None, split):
        raise ValidationError("source metadata split mismatch")


def _working_image_locator(
    row: Mapping[str, Any], split: str, source: Path, image_root: Path
) -> str:
    locator = row["images"][0]
    if not isinstance(locator, str):
        raise ValidationError("image locator must be a string")
    resolved = (source.parent / locator).resolve(strict=True)
    try:
        relative = resolved.relative_to(image_root)
    except ValueError as exc:
        raise ValidationError("source image escapes the allowlisted image root") from exc
    expected_dir = f"{split}2017"
    if not relative.parts or relative.parts[0] != expected_dir:
        raise ValidationError("source image does not belong to its split storage")
    return (Path("images") / relative).as_posix()


def _validate_managed_link(link: Path, expected_target: Path) -> None:
    if not link.is_symlink() or link.resolve(strict=True) != expected_target:
        raise ManifestDriftError("managed_image_link")


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
