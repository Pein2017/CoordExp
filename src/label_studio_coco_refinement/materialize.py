"""Locked atomic materialization of one committed working split."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol

from src.common.errors import DataContractError
from src.data import iter_raw_examples
from src.data.examples import JsonFrozen, freeze_json, thaw_json
from src.label_studio_coco_refinement.models import (
    RefinementRuntimeLayout,
    Split,
    WorkingRow,
)
from src.label_studio_coco_refinement.store import (
    BOOTSTRAP_TASK_INDEX_BUILD_KEY,
    BOOTSTRAP_TASK_INDEX_SCHEMA_KEY,
    BOOTSTRAP_TASK_INDEX_SHA256_KEY,
    BOOTSTRAP_TASK_MANIFEST_KEY,
    TASK_INDEX_IDENTITY_SCHEMA_VERSION,
    WorkingDatasetStore,
    sha256_json,
)


MATERIALIZER_VERSION = "label-studio-working-coord-v4"
OPERATOR_RECEIPT_SCHEMA_VERSION = 1
OPERATOR_RECEIPT_CODE = "label_studio.working_coord_materialized"
OPERATOR_RECEIPT_NAME = "working.coord.receipt.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_GenerationGuard = Callable[[], AbstractContextManager[None]]
_EXACT_COMMITTED_GENERATION_GUARD = WorkingDatasetStore.committed_generation_guard
_OPERATOR_RECEIPT_TEST_HOOK: Callable[[str], None] | None = None


@dataclass(frozen=True)
class CommittedGenerationReceipt:
    """Caller-held proof of the exact committed split generation to export."""

    split: Split
    generation: int
    working_sha256: str
    task_count: int
    task_manifest_hash: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "val"):
            raise DataContractError(
                "committed generation split must be train or val",
                code="label_studio.materialize_split",
                context={"split": self.split},
            )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise DataContractError(
                "committed generation must be a non-negative integer",
                code="label_studio.materialize_generation",
                context={"generation": self.generation},
            )
        if (
            not isinstance(self.working_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.working_sha256) is None
        ):
            raise DataContractError(
                "committed working hash must be a lowercase SHA-256 digest",
                code="label_studio.materialize_working_hash",
                context={"working_sha256": self.working_sha256},
            )
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count <= 0
        ):
            raise DataContractError(
                "committed task count must be a positive integer",
                code="label_studio.materialize_task_count",
                context={"task_count": self.task_count},
            )
        _require_sha256_digest(
            self.task_manifest_hash,
            field="task_manifest_hash",
            code="label_studio.materialize_task_manifest_hash",
        )


@dataclass(frozen=True)
class SourceTaskIdentityReceipt:
    """Immutable source-row identity attested by the project/task authority."""

    split: Split
    image_id: int
    file_name: str
    width: int
    height: int
    metadata: Mapping[str, JsonFrozen]
    source_line: int
    task_row_fingerprint: str
    task_manifest_hash: str
    image_sha256: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "val"):
            raise DataContractError(
                "source task identity split must be train or val",
                code="label_studio.source_task_split",
                context={"split": self.split},
            )
        for field, value in (
            ("image_id", self.image_id),
            ("width", self.width),
            ("height", self.height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DataContractError(
                    "source task identity integer must be positive",
                    code="label_studio.source_task_integer",
                    context={"field": field, "value": value},
                )
        if (
            isinstance(self.source_line, bool)
            or not isinstance(self.source_line, int)
            or self.source_line <= 0
        ):
            raise DataContractError(
                "source task identity source_line must be a positive integer",
                code="label_studio.source_task_line",
                context={"source_line": self.source_line},
            )
        expected_locator = _canonical_image_locator(self.split, self.image_id)
        if self.file_name != expected_locator:
            raise DataContractError(
                "source task identity file_name is not the canonical image_id locator",
                code="label_studio.source_task_file_name",
                context={"expected": expected_locator, "actual": self.file_name},
            )
        if not isinstance(self.metadata, Mapping):
            raise DataContractError(
                "source task identity metadata must be a JSON object",
                code="label_studio.source_task_metadata",
                context={"value_type": type(self.metadata).__name__},
            )
        frozen_metadata = freeze_json(self.metadata)
        object.__setattr__(self, "metadata", frozen_metadata)
        expected_metadata = {"source": "coco2017", "split": self.split}
        if thaw_json(frozen_metadata) != expected_metadata:
            raise DataContractError(
                "source task identity metadata must exactly name the selected COCO split",
                code="label_studio.source_task_metadata",
                context={
                    "expected": expected_metadata,
                    "actual": thaw_json(frozen_metadata),
                },
            )
        for field, value in (
            ("task_row_fingerprint", self.task_row_fingerprint),
            ("task_manifest_hash", self.task_manifest_hash),
            ("image_sha256", self.image_sha256),
            ("source_fingerprint", self.source_fingerprint),
        ):
            _require_sha256_digest(
                value,
                field=field,
                code="label_studio.source_task_fingerprint",
            )
        expected_fingerprint = _source_task_fingerprint(
            split=self.split,
            image_id=self.image_id,
            file_name=self.file_name,
            width=self.width,
            height=self.height,
            metadata=frozen_metadata,
            source_line=self.source_line,
            task_row_fingerprint=self.task_row_fingerprint,
            task_manifest_hash=self.task_manifest_hash,
            image_sha256=self.image_sha256,
        )
        if self.source_fingerprint != expected_fingerprint:
            raise DataContractError(
                "source task identity fingerprint does not match its immutable fields",
                code="label_studio.source_task_fingerprint",
                context={
                    "expected": expected_fingerprint,
                    "actual": self.source_fingerprint,
                },
            )

    @classmethod
    def capture(
        cls,
        *,
        split: Split,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        metadata: Mapping[str, Any],
        source_line: int,
        task_row_fingerprint: str,
        task_manifest_hash: str,
        image_sha256: str,
    ) -> "SourceTaskIdentityReceipt":
        """Build a deterministic receipt for fakes or an attested adapter."""

        frozen_metadata = freeze_json(metadata)
        return cls(
            split=split,
            image_id=image_id,
            file_name=file_name,
            width=width,
            height=height,
            metadata=frozen_metadata,
            source_line=source_line,
            task_row_fingerprint=task_row_fingerprint,
            task_manifest_hash=task_manifest_hash,
            image_sha256=image_sha256,
            source_fingerprint=_source_task_fingerprint(
                split=split,
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                metadata=frozen_metadata,
                source_line=source_line,
                task_row_fingerprint=task_row_fingerprint,
                task_manifest_hash=task_manifest_hash,
                image_sha256=image_sha256,
            ),
        )

    def validate_row(self, row: WorkingRow) -> None:
        """Reject any working-row drift from this source/task receipt."""

        mismatches: list[str] = []
        if row.image_id != self.image_id:
            mismatches.append("image_id")
        if row.images != (self.file_name,):
            mismatches.append("images[0]")
        if row.file_name != self.file_name:
            mismatches.append("file_name")
        if row.width != self.width:
            mismatches.append("width")
        if row.height != self.height:
            mismatches.append("height")
        if row.metadata != self.metadata:
            mismatches.append("metadata")
        observed_fingerprint = _source_task_fingerprint(
            split=self.split,
            image_id=row.image_id,
            file_name=row.file_name,
            width=row.width,
            height=row.height,
            metadata=row.metadata,
            source_line=self.source_line,
            task_row_fingerprint=self.task_row_fingerprint,
            task_manifest_hash=self.task_manifest_hash,
            image_sha256=self.image_sha256,
        )
        if observed_fingerprint != self.source_fingerprint:
            mismatches.append("source_fingerprint")
        if mismatches:
            raise DataContractError(
                "working row does not match its source task identity receipt",
                code="label_studio.materialize_source_task_identity",
                context={
                    "split": self.split,
                    "image_id": row.image_id,
                    "mismatches": mismatches,
                    "expected_source_fingerprint": self.source_fingerprint,
                    "actual_source_fingerprint": observed_fingerprint,
                },
            )


class SourceTaskIdentityResolver(Protocol):
    """Narrow authority boundary for one immutable source/task identity."""

    def resolve(
        self,
        *,
        split: Split,
        image_id: int,
    ) -> SourceTaskIdentityReceipt: ...


class ProjectTaskIndexSourceIdentityResolver:
    """Production resolver over the frozen schema-v3 project task index."""

    def __init__(
        self,
        layout: RefinementRuntimeLayout,
        split: Split,
        *,
        store: WorkingDatasetStore,
    ) -> None:
        self._initialize(layout, split, store, allow_store_fixture=False)

    @classmethod
    def _for_bounded_probe(
        cls,
        layout: RefinementRuntimeLayout,
        split: Split,
        *,
        store: WorkingDatasetStore,
    ) -> "ProjectTaskIndexSourceIdentityResolver":
        """Use the store's frozen bootstrap identity only for a bounded probe."""

        resolver = cls.__new__(cls)
        resolver._initialize(layout, split, store, allow_store_fixture=True)
        return resolver

    def _initialize(
        self,
        layout: RefinementRuntimeLayout,
        split: Split,
        store: WorkingDatasetStore,
        *,
        allow_store_fixture: bool,
    ) -> None:
        if not isinstance(layout, RefinementRuntimeLayout):
            raise DataContractError(
                "source resolver requires RefinementRuntimeLayout",
                code="label_studio.source_task_layout",
            )
        layout.split_root(split)
        if type(store) is not WorkingDatasetStore:
            raise DataContractError(
                "source resolver requires the exact WorkingDatasetStore",
                code="label_studio.source_task_store",
            )
        if store.split_dir != layout.split_root(split):
            raise DataContractError(
                "source resolver store belongs to another split root",
                code="label_studio.source_task_store_binding",
            )
        self.layout = layout
        self.split = split
        self.store = store
        self._allow_store_fixture = allow_store_fixture
        self._active_session: (
            _CanonicalIdentitySession | _StoreIdentitySession | None
        ) = None

    @contextmanager
    def committed_generation_session(
        self,
        committed_generation: CommittedGenerationReceipt,
    ) -> Iterator["ProjectTaskIndexSourceIdentityResolver"]:
        """Bind one resolver cursor to the caller-held committed generation."""

        if self._active_session is not None:
            raise DataContractError(
                "source resolver generation session is already active",
                code="label_studio.source_task_session",
            )
        manifest = _read_strict_mapping(
            self.store.manifest_path,
            code="label_studio.source_task_manifest_decode",
        )
        observed_generation = _read_committed_generation(
            self.layout,
            self.split,
            self.store.manifest_path,
        )
        _require_generation(
            observed_generation,
            committed_generation,
            stage="source_resolver",
        )
        extras = manifest.get("extra_fingerprints")
        if not isinstance(extras, Mapping):
            raise DataContractError(
                "project manifest has no bootstrap fingerprint mapping",
                code="label_studio.source_task_bootstrap_binding",
            )
        binding_fields = {
            BOOTSTRAP_TASK_INDEX_SCHEMA_KEY,
            BOOTSTRAP_TASK_INDEX_BUILD_KEY,
            BOOTSTRAP_TASK_INDEX_SHA256_KEY,
            BOOTSTRAP_TASK_MANIFEST_KEY,
        }
        if binding_fields.issubset(extras):
            session: _CanonicalIdentitySession | _StoreIdentitySession = (
                _CanonicalIdentitySession(
                    layout=self.layout,
                    split=self.split,
                    committed_generation=committed_generation,
                    manifest=manifest,
                )
            )
        elif self._allow_store_fixture:
            session = _StoreIdentitySession(
                layout=self.layout,
                split=self.split,
                committed_generation=committed_generation,
                manifest=manifest,
                task_index_path=self.store.task_index_path,
            )
        else:
            raise DataContractError(
                "project manifest is not bound to the canonical bootstrap task index",
                code="label_studio.source_task_bootstrap_binding",
                context={"missing": sorted(binding_fields - set(extras))},
            )
        self._active_session = session
        try:
            yield self
            session.complete()
        finally:
            session.close()
            self._active_session = None

    def resolve(
        self,
        *,
        split: Split,
        image_id: int,
    ) -> SourceTaskIdentityReceipt:
        session = self._active_session
        if session is None:
            raise DataContractError(
                "source task resolution requires the committed-generation session",
                code="label_studio.source_task_session",
            )
        return session.resolve(split=split, image_id=image_id)


class _CanonicalIdentitySession:
    def __init__(
        self,
        *,
        layout: RefinementRuntimeLayout,
        split: Split,
        committed_generation: CommittedGenerationReceipt,
        manifest: Mapping[str, Any],
    ) -> None:
        from src.label_studio_coco_refinement.project import (
            SOURCE_CONTRACTS,
            TASK_INDEX_SCHEMA_VERSION,
            ProjectContractError,
            Split as ProjectSplit,
            load_canonical_task_index_receipt,
        )

        self.layout = layout
        self.split = split
        self.committed_generation = committed_generation
        self._count = 0
        self._task_manifest_digest = hashlib.sha256()
        extras = manifest["extra_fingerprints"]
        expected_schema = str(TASK_INDEX_SCHEMA_VERSION)
        if extras.get(BOOTSTRAP_TASK_INDEX_SCHEMA_KEY) != expected_schema:
            raise DataContractError(
                "bootstrap task-index schema binding is stale",
                code="label_studio.source_task_bootstrap_schema",
            )
        try:
            receipt = load_canonical_task_index_receipt(
                layout.repository_root,
                ProjectSplit(split),
                expected_task_manifest_fingerprint=extras.get(
                    BOOTSTRAP_TASK_MANIFEST_KEY
                ),
            )
        except ProjectContractError as exc:
            raise DataContractError(
                "canonical bootstrap task index cannot be attested",
                code="label_studio.source_task_bootstrap_receipt",
                cause=exc,
            ) from exc
        if receipt.build_key != extras.get(
            BOOTSTRAP_TASK_INDEX_BUILD_KEY
        ) or receipt.sha256 != extras.get(BOOTSTRAP_TASK_INDEX_SHA256_KEY):
            raise DataContractError(
                "canonical bootstrap task-index binding drift",
                code="label_studio.source_task_bootstrap_binding",
            )
        contract = SOURCE_CONTRACTS[ProjectSplit(split)]
        if (
            manifest.get("source_path") != str(layout.selected_source(split).resolve())
            or manifest.get("source_sha256") != contract.sha256
        ):
            raise DataContractError(
                "mutable store is not bound to the canonical selected source",
                code="label_studio.source_task_selected_source",
            )
        self._receipt = receipt
        self._records = iter(receipt.iter_records())
        self._source_path = layout.selected_source(split)
        self._source_handle, self._source_before = _open_stable_binary_source(
            self._source_path
        )
        self._source_digest = hashlib.sha256()

    def resolve(self, *, split: Split, image_id: int) -> SourceTaskIdentityReceipt:
        if split != self.split:
            raise LookupError(f"source task split drift: {split}")
        raw_line = self._source_handle.readline()
        if not raw_line:
            raise LookupError(f"source task is absent: {split}:{image_id}")
        self._source_digest.update(raw_line)
        source_line = self._count + 1
        source_row = _parse_jsonl_row(self._source_path, source_line, raw_line)
        if not isinstance(source_row, Mapping):
            raise DataContractError(
                "canonical source row is not an object",
                code="label_studio.source_task_source_row",
                context={"source_line": source_line},
            )
        from src.label_studio_coco_refinement.project import (
            ProjectContractError,
            validate_source_row,
        )

        try:
            identity = validate_source_row(source_row, split=split)
            entry, _payload = next(self._records)
        except (ProjectContractError, StopIteration) as exc:
            raise DataContractError(
                "canonical source and bootstrap task index are not aligned",
                code="label_studio.source_task_inventory",
                context={"source_line": source_line},
                cause=exc,
            ) from exc
        if (
            identity.image_id != image_id
            or entry.identity.image_id != image_id
            or entry.source_line != source_line
            or entry.working_image_locator != source_row.get("file_name")
            or entry.source_image_locator != source_row.get("images", [None])[0]
        ):
            raise DataContractError(
                "canonical source row does not match its bootstrap task-index entry",
                code="label_studio.source_task_inventory",
                context={"source_line": source_line, "image_id": image_id},
            )
        working_row = dict(source_row)
        working_row["images"] = [source_row["file_name"]]
        receipt = SourceTaskIdentityReceipt.capture(
            split=split,
            image_id=image_id,
            file_name=str(source_row["file_name"]),
            width=int(source_row["width"]),
            height=int(source_row["height"]),
            metadata=source_row["metadata"],
            source_line=source_line,
            task_row_fingerprint=sha256_json(working_row),
            task_manifest_hash=self.committed_generation.task_manifest_hash,
            image_sha256=entry.image_sha256,
        )
        self._task_manifest_digest.update(_task_manifest_frame(receipt))
        self._count += 1
        return receipt

    def complete(self) -> None:
        if self._source_handle.read(1) != b"":
            raise DataContractError(
                "canonical source has rows missing from the working inventory",
                code="label_studio.source_task_inventory",
            )
        try:
            next(self._records)
        except StopIteration:
            pass
        else:
            raise DataContractError(
                "bootstrap task index has rows missing from the working inventory",
                code="label_studio.source_task_inventory",
            )
        source_after = os.fstat(self._source_handle.fileno())
        try:
            path_after = self._source_path.lstat()
        except OSError as exc:
            raise DataContractError(
                "canonical source changed during identity resolution",
                code="label_studio.source_task_source_changed",
                cause=exc,
            ) from exc

        def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            signature(self._source_before) != signature(source_after)
            or signature(self._source_before) != signature(path_after)
            or not stat.S_ISREG(path_after.st_mode)
            or self._source_digest.hexdigest() != self._receipt.source_sha256
            or self._count != self.committed_generation.task_count
            or self._task_manifest_digest.hexdigest()
            != self.committed_generation.task_manifest_hash
        ):
            raise DataContractError(
                "canonical source identity inventory changed or is incomplete",
                code="label_studio.source_task_inventory",
            )

    def close(self) -> None:
        self._source_handle.close()


class _StoreIdentitySession:
    """Bounded-probe resolver over the store's immutable bootstrap task index."""

    def __init__(
        self,
        *,
        layout: RefinementRuntimeLayout,
        split: Split,
        committed_generation: CommittedGenerationReceipt,
        manifest: Mapping[str, Any],
        task_index_path: Path,
    ) -> None:
        self.layout = layout
        self.split = split
        self.committed_generation = committed_generation
        if _sha256_file(task_index_path) != manifest.get("task_index_sha256"):
            raise DataContractError(
                "bounded task-index bytes do not match the store manifest",
                code="label_studio.source_task_fixture_index",
            )
        payload = _read_strict_mapping(
            task_index_path,
            code="label_studio.source_task_fixture_index",
        )
        if (
            payload.get("identity_schema_version") != TASK_INDEX_IDENTITY_SCHEMA_VERSION
            or payload.get("split") != split
            or not isinstance(payload.get("entries"), list)
        ):
            raise DataContractError(
                "bounded task index lacks the frozen identity schema",
                code="label_studio.source_task_fixture_index",
            )
        self._entries = iter(payload["entries"])
        self._count = 0
        self._task_manifest_digest = hashlib.sha256()

    def resolve(self, *, split: Split, image_id: int) -> SourceTaskIdentityReceipt:
        try:
            entry = next(self._entries)
        except StopIteration as exc:
            raise LookupError(f"source task is absent: {split}:{image_id}") from exc
        source_line = self._count + 1
        if (
            split != self.split
            or not isinstance(entry, Mapping)
            or entry.get("source_line") != source_line
            or entry.get("source_row_index") != self._count
            or entry.get("image_id") != image_id
        ):
            raise DataContractError(
                "bounded task-index inventory is out of order",
                code="label_studio.source_task_fixture_index",
            )
        receipt = SourceTaskIdentityReceipt.capture(
            split=split,
            image_id=image_id,
            file_name=entry.get("file_name"),
            width=entry.get("width"),
            height=entry.get("height"),
            metadata=entry.get("metadata"),
            source_line=source_line,
            task_row_fingerprint=entry.get("task_row_fingerprint"),
            task_manifest_hash=self.committed_generation.task_manifest_hash,
            image_sha256=entry.get("image_sha256"),
        )
        self._task_manifest_digest.update(_task_manifest_frame(receipt))
        self._count += 1
        return receipt

    def complete(self) -> None:
        try:
            next(self._entries)
        except StopIteration:
            pass
        else:
            raise DataContractError(
                "bounded task index has rows missing from the working inventory",
                code="label_studio.source_task_fixture_index",
            )
        if (
            self._count != self.committed_generation.task_count
            or self._task_manifest_digest.hexdigest()
            != self.committed_generation.task_manifest_hash
        ):
            raise DataContractError(
                "bounded task-index manifest inventory drift",
                code="label_studio.source_task_fixture_index",
            )

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class MaterializationReceipt:
    split: Split
    generation: int
    source_path: Path
    destination_path: Path
    source_sha256: str
    destination_sha256: str
    row_count: int
    object_count: int
    loader_row_count: int
    summary_fingerprints: Mapping[str, str]
    materializer_version: str = MATERIALIZER_VERSION

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "generation": self.generation,
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "source_sha256": self.source_sha256,
            "destination_sha256": self.destination_sha256,
            "row_count": self.row_count,
            "object_count": self.object_count,
            "loader_row_count": self.loader_row_count,
            "summary_fingerprints": dict(self.summary_fingerprints),
            "materializer_version": self.materializer_version,
        }


@dataclass(frozen=True)
class OperatorMaterializationReceipt:
    """Durable operator-facing proof for one fixed split output."""

    materialization: MaterializationReceipt
    store_manifest_sha256: str
    task_index_sha256: str
    bootstrap_task_index_binding: Mapping[str, str]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPERATOR_RECEIPT_SCHEMA_VERSION,
            "code": OPERATOR_RECEIPT_CODE,
            "materialization": self.materialization.to_artifact_dict(),
            "store": {
                "manifest_sha256": self.store_manifest_sha256,
                "task_index_sha256": self.task_index_sha256,
                "generation": self.materialization.generation,
            },
            "bootstrap_task_index_binding": dict(self.bootstrap_task_index_binding),
            "loader_attestation": {
                "seam": "src.data.iter_raw_examples",
                "row_count": self.materialization.loader_row_count,
                "status": "passed",
            },
            "contract_summary": {
                "task_identity": "source-order (split,image_id) inventory",
                "category": "canonical COCO-80 id/name pairs",
                "object_order": "working object order preserved",
                "geometry": "positive-area norm1000 boxes emitted as coord tokens",
                "image_content": "bootstrap-frozen per-image SHA-256",
                "metadata": "row and object metadata preserved",
            },
        }


class WorkingCoordMaterializer:
    """Export exactly one locked committed split to its fixed coord sibling."""

    def __init__(
        self,
        layout: RefinementRuntimeLayout,
        split: Split,
        *,
        store: WorkingDatasetStore,
        source_identity_resolver: SourceTaskIdentityResolver,
    ) -> None:
        self._initialize_common(layout, split, source_identity_resolver)
        if type(store) is not WorkingDatasetStore:
            raise DataContractError(
                "materializer public construction requires WorkingDatasetStore",
                code="label_studio.materialize_store",
                context={"value_type": type(store).__name__},
            )
        expected_paths = {
            "split_dir": self.layout.split_root(self.split),
            "working_path": self.layout.working_norm(self.split),
            "manifest_path": self.layout.project_manifest(self.split),
            "lock_path": self.layout.commit_lock(self.split),
        }
        actual_paths = {
            "split_dir": store.split_dir,
            "working_path": store.working_path,
            "manifest_path": store.manifest_path,
            "lock_path": store.lock_path,
        }
        mismatches = [
            field
            for field, expected in expected_paths.items()
            if actual_paths[field] != expected
        ]
        if mismatches:
            raise DataContractError(
                "WorkingDatasetStore is not bound to the exact materializer split",
                code="label_studio.materialize_store_binding",
                context={
                    "split": self.split,
                    "mismatches": mismatches,
                    "expected": {
                        key: str(value) for key, value in expected_paths.items()
                    },
                    "actual": {key: str(value) for key, value in actual_paths.items()},
                },
            )
        generation_guard = getattr(store, "committed_generation_guard", None)
        if (
            getattr(generation_guard, "__self__", None) is not store
            or getattr(generation_guard, "__func__", None)
            is not _EXACT_COMMITTED_GENERATION_GUARD
        ):
            raise DataContractError(
                "materializer requires the exact bound committed generation guard",
                code="label_studio.materialize_generation_guard",
                context={"store_type": type(store).__name__},
            )
        self._committed_generation_guard = generation_guard

    @classmethod
    def _for_test(
        cls,
        layout: RefinementRuntimeLayout,
        split: Split,
        *,
        exclusive_lock: _GenerationGuard,
        source_identity_resolver: SourceTaskIdentityResolver,
    ) -> "WorkingCoordMaterializer":
        """Private unit-test seam; production callers must bind a real store."""

        if not callable(exclusive_lock):
            raise DataContractError(
                "test materializer requires an injected generation guard",
                code="label_studio.materialize_test_guard",
            )
        materializer = cls.__new__(cls)
        materializer._initialize_common(layout, split, source_identity_resolver)
        materializer._committed_generation_guard = exclusive_lock
        return materializer

    def _initialize_common(
        self,
        layout: RefinementRuntimeLayout,
        split: Split,
        source_identity_resolver: SourceTaskIdentityResolver,
    ) -> None:
        if not isinstance(layout, RefinementRuntimeLayout):
            raise DataContractError(
                "materializer requires RefinementRuntimeLayout",
                code="label_studio.materialize_layout",
                context={"value_type": type(layout).__name__},
            )
        # Validate the split eagerly before binding any generation guard.
        layout.split_root(split)
        if not callable(getattr(source_identity_resolver, "resolve", None)):
            raise DataContractError(
                "materializer requires a source task identity resolver",
                code="label_studio.materialize_source_task_resolver",
                context={"value_type": type(source_identity_resolver).__name__},
            )
        self.layout = layout
        self.split = split
        self._source_identity_resolver = source_identity_resolver

    def materialize(
        self,
        committed_generation: CommittedGenerationReceipt,
    ) -> MaterializationReceipt:
        materialization, _operator_receipt = self._materialize_generation(
            committed_generation,
            publish_operator_receipt=False,
        )
        return materialization

    def _materialize_generation(
        self,
        committed_generation: CommittedGenerationReceipt,
        *,
        publish_operator_receipt: bool,
    ) -> tuple[MaterializationReceipt, OperatorMaterializationReceipt | None]:
        if not isinstance(committed_generation, CommittedGenerationReceipt):
            raise DataContractError(
                "materialization requires a committed generation receipt",
                code="label_studio.materialize_receipt_type",
                context={"value_type": type(committed_generation).__name__},
            )
        if committed_generation.split != self.split:
            raise DataContractError(
                "committed generation receipt belongs to another split",
                code="label_studio.materialize_receipt_split",
                context={
                    "materializer_split": self.split,
                    "receipt_split": committed_generation.split,
                },
            )

        source = self.layout.working_norm(self.split)
        output = self.layout.working_coord(self.split)
        manifest = self.layout.project_manifest(self.split)
        receipt_path = self.layout.split_root(self.split) / OPERATOR_RECEIPT_NAME
        materialization: MaterializationReceipt | None = None
        operator_receipt: OperatorMaterializationReceipt | None = None

        with self._committed_generation_guard():
            _validate_bound_paths(self.layout, self.split)
            before = _read_committed_generation(self.layout, self.split, manifest)
            _require_generation(before, committed_generation, stage="before_read")
            source_hash_before = _sha256_file(source)
            _require_working_hash(
                source_hash_before,
                committed_generation,
                stage="before_read",
            )

            session_factory = getattr(
                self._source_identity_resolver,
                "committed_generation_session",
                None,
            )
            resolver_context = (
                session_factory(committed_generation)
                if callable(session_factory)
                else nullcontext(self._source_identity_resolver)
            )
            candidate: _CandidateReceipt | None = None
            try:
                with resolver_context as active_resolver:
                    candidate = self._materialize_candidate(
                        source=source,
                        output=output,
                        committed_generation=committed_generation,
                        source_identity_resolver=active_resolver,
                    )
            except BaseException:
                if candidate is not None:
                    candidate._candidate_path.unlink(missing_ok=True)
                raise
            assert candidate is not None
            candidate_path: Path | None = candidate._candidate_path
            try:
                # Re-read both manifest authority and working bytes immediately before
                # replacement while the committed-generation guard remains held.
                after = _read_committed_generation(self.layout, self.split, manifest)
                _require_generation(after, committed_generation, stage="before_replace")
                source_hash_after = _sha256_file(source)
                _require_working_hash(
                    source_hash_after,
                    committed_generation,
                    stage="before_replace",
                )
                if source_hash_after != candidate.source_sha256:
                    raise DataContractError(
                        "working JSONL changed during materialization",
                        code="label_studio.materialize_source_changed",
                        context={
                            "expected": candidate.source_sha256,
                            "actual": source_hash_after,
                        },
                    )
                _validate_bound_paths(self.layout, self.split)
                _revalidate_image_contents(
                    self.layout,
                    self.split,
                    candidate.image_attestations,
                )
                _invalidate_operator_receipt(receipt_path)
                _run_operator_receipt_test_hook("after_receipt_invalidation")
                os.replace(candidate._candidate_path, output)
                candidate_path = None
                _fsync_directory(output.parent)
                _run_operator_receipt_test_hook("after_output_replace")
                materialization = MaterializationReceipt(
                    split=self.split,
                    generation=committed_generation.generation,
                    source_path=source,
                    destination_path=output,
                    source_sha256=candidate.source_sha256,
                    destination_sha256=candidate.destination_sha256,
                    row_count=candidate.row_count,
                    object_count=candidate.object_count,
                    loader_row_count=candidate.loader_row_count,
                    summary_fingerprints=candidate.summary_fingerprints,
                )
                if publish_operator_receipt:
                    manifest_payload = _read_strict_mapping(
                        manifest,
                        code="label_studio.materialize_manifest_decode",
                    )
                    task_index_sha256 = _require_sha256_digest(
                        manifest_payload.get("task_index_sha256"),
                        field="task_index_sha256",
                        code="label_studio.materialize_task_index_hash",
                    )
                    extras = manifest_payload.get("extra_fingerprints")
                    binding = {
                        key: str(extras[key])
                        for key in (
                            BOOTSTRAP_TASK_INDEX_SCHEMA_KEY,
                            BOOTSTRAP_TASK_INDEX_BUILD_KEY,
                            BOOTSTRAP_TASK_INDEX_SHA256_KEY,
                            BOOTSTRAP_TASK_MANIFEST_KEY,
                        )
                        if isinstance(extras, Mapping) and key in extras
                    }
                    operator_receipt = OperatorMaterializationReceipt(
                        materialization=materialization,
                        store_manifest_sha256=_sha256_file(manifest),
                        task_index_sha256=task_index_sha256,
                        bootstrap_task_index_binding=binding,
                    )
                    _run_operator_receipt_test_hook("before_receipt_publish")
                    _atomic_replace_json(
                        receipt_path,
                        operator_receipt.to_artifact_dict(),
                    )
            finally:
                if candidate_path is not None:
                    candidate_path.unlink(missing_ok=True)

        assert materialization is not None
        return materialization, operator_receipt

    def materialize_current_generation(self) -> OperatorMaterializationReceipt:
        """Materialize current authority and atomically publish its fixed receipt."""

        manifest_path = self.layout.project_manifest(self.split)
        generation = _read_committed_generation(
            self.layout,
            self.split,
            manifest_path,
        )
        _materialization, receipt = self._materialize_generation(
            generation,
            publish_operator_receipt=True,
        )
        assert receipt is not None
        return receipt

    def _materialize_candidate(
        self,
        *,
        source: Path,
        output: Path,
        committed_generation: CommittedGenerationReceipt,
        source_identity_resolver: SourceTaskIdentityResolver,
    ) -> "_CandidateReceipt":
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        row_count = 0
        object_count = 0
        seen_image_ids: set[int] = set()
        task_manifest_digest = hashlib.sha256()
        image_attestations: list[_ImageContentAttestation] = []
        summary_digests = {
            name: _CanonicalArrayDigest()
            for name in (
                "task_identity",
                "category",
                "object_order",
                "geometry",
                "image_content",
                "metadata",
            )
        }
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temp_path = Path(temp_name)
        keep_candidate = False
        try:
            with (
                source.open("rb") as source_handle,
                os.fdopen(
                    descriptor,
                    "wb",
                ) as output_handle,
            ):
                for row_number, raw_line in enumerate(source_handle, start=1):
                    source_digest.update(raw_line)
                    payload = _parse_jsonl_row(source, row_number, raw_line)
                    row = WorkingRow.from_mapping(
                        payload,
                        field=f"row[{row_number}]",
                    ).validate_for_split(self.split)
                    try:
                        source_identity = source_identity_resolver.resolve(
                            split=self.split,
                            image_id=row.image_id,
                        )
                    except Exception as exc:
                        raise DataContractError(
                            "source task identity resolver could not attest the working row",
                            code="label_studio.materialize_source_task_resolver",
                            context={
                                "split": self.split,
                                "image_id": row.image_id,
                                "row_number": row_number,
                            },
                            cause=exc,
                        ) from exc
                    if not isinstance(source_identity, SourceTaskIdentityReceipt):
                        raise DataContractError(
                            "source task identity resolver returned an invalid receipt",
                            code="label_studio.materialize_source_task_receipt",
                            context={
                                "split": self.split,
                                "image_id": row.image_id,
                                "value_type": type(source_identity).__name__,
                            },
                        )
                    if (
                        source_identity.task_manifest_hash
                        != committed_generation.task_manifest_hash
                    ):
                        raise DataContractError(
                            "source task receipt belongs to another committed task manifest",
                            code="label_studio.materialize_source_task_manifest",
                            context={
                                "image_id": row.image_id,
                                "expected": committed_generation.task_manifest_hash,
                                "actual": source_identity.task_manifest_hash,
                            },
                        )
                    if source_identity.source_line != row_number:
                        raise DataContractError(
                            "source task receipt source line does not match working row position",
                            code="label_studio.materialize_source_task_line",
                            context={
                                "image_id": row.image_id,
                                "expected": row_number,
                                "actual": source_identity.source_line,
                            },
                        )
                    source_identity.validate_row(row)
                    resolved_image = _validate_resolved_image(
                        self.layout, self.split, row
                    )
                    image_sha256 = _sha256_file(resolved_image)
                    if image_sha256 != source_identity.image_sha256:
                        raise DataContractError(
                            "working row image content does not match its source task receipt",
                            code="label_studio.materialize_image_content",
                            context={
                                "image_id": row.image_id,
                                "path": str(resolved_image),
                                "expected": source_identity.image_sha256,
                                "actual": image_sha256,
                            },
                        )
                    if row.image_id in seen_image_ids:
                        raise DataContractError(
                            "image_id must be unique within a working split",
                            code="label_studio.image_id_duplicate",
                            context={
                                "image_id": row.image_id,
                                "row_number": row_number,
                            },
                        )
                    seen_image_ids.add(row.image_id)
                    task_manifest_digest.update(_task_manifest_frame(source_identity))
                    summary_digests["task_identity"].add(
                        {
                            "image_id": row.image_id,
                            "source_fingerprint": source_identity.source_fingerprint,
                        }
                    )
                    summary_digests["category"].add(
                        [[item.category_id, item.category_name] for item in row.objects]
                    )
                    summary_digests["object_order"].add(
                        [item.coco_ann_id for item in row.objects]
                    )
                    summary_digests["geometry"].add(
                        [list(item.bbox_2d) for item in row.objects]
                    )
                    summary_digests["image_content"].add(
                        [row.image_id, source_identity.image_sha256]
                    )
                    summary_digests["metadata"].add(
                        {
                            "row": thaw_json(row.metadata),
                            "objects": [
                                None
                                if item.metadata is None
                                else thaw_json(item.metadata)
                                for item in row.objects
                            ],
                        }
                    )
                    image_attestations.append(
                        _ImageContentAttestation(
                            image_id=row.image_id,
                            expected_sha256=source_identity.image_sha256,
                        )
                    )
                    encoded = (
                        json.dumps(
                            row.to_json_dict(coord_tokens=True),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    output_handle.write(encoded)
                    output_digest.update(encoded)
                    row_count += 1
                    object_count += len(row.objects)
                if row_count == 0:
                    raise DataContractError(
                        "working JSONL must contain at least one row",
                        code="label_studio.rows_empty",
                        context={"path": str(source)},
                    )
                observed_task_manifest_hash = task_manifest_digest.hexdigest()
                if (
                    row_count != committed_generation.task_count
                    or observed_task_manifest_hash
                    != committed_generation.task_manifest_hash
                ):
                    raise DataContractError(
                        "working rows do not match the committed task manifest inventory",
                        code="label_studio.materialize_task_inventory",
                        context={
                            "expected_task_count": committed_generation.task_count,
                            "actual_task_count": row_count,
                            "expected_task_manifest_hash": (
                                committed_generation.task_manifest_hash
                            ),
                            "actual_task_manifest_hash": observed_task_manifest_hash,
                        },
                    )
                output_handle.flush()
                os.fsync(output_handle.fileno())

            source_sha256 = source_digest.hexdigest()
            _require_working_hash(
                source_sha256,
                committed_generation,
                stage="candidate_read",
            )
            loader_rows = sum(1 for _ in iter_raw_examples(temp_path))
            if loader_rows != row_count:
                raise DataContractError(
                    "current loader row count differs from materialized row count",
                    code="label_studio.loader_row_count",
                    context={"expected": row_count, "actual": loader_rows},
                )
            keep_candidate = True
            return _CandidateReceipt(
                _candidate_path=temp_path,
                source_sha256=source_sha256,
                destination_sha256=output_digest.hexdigest(),
                row_count=row_count,
                object_count=object_count,
                loader_row_count=loader_rows,
                summary_fingerprints={
                    name: digest.fingerprint for name, digest in summary_digests.items()
                },
                image_attestations=tuple(image_attestations),
            )
        finally:
            if not keep_candidate:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _CandidateReceipt:
    _candidate_path: Path
    source_sha256: str
    destination_sha256: str
    row_count: int
    object_count: int
    loader_row_count: int
    summary_fingerprints: Mapping[str, str]
    image_attestations: tuple["_ImageContentAttestation", ...]


class _CanonicalArrayDigest:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._count = 0

    def add(self, value: Any) -> None:
        if self._count:
            self._digest.update(b",")
        self._digest.update(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self._count += 1

    @property
    def fingerprint(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


@dataclass(frozen=True)
class _ImageContentAttestation:
    image_id: int
    expected_sha256: str


def _canonical_image_locator(split: Split, image_id: int) -> str:
    return f"images/{split}2017/{image_id:012d}.jpg"


def _source_task_fingerprint(
    *,
    split: Split,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    metadata: Mapping[str, JsonFrozen],
    source_line: int,
    task_row_fingerprint: str,
    task_manifest_hash: str,
    image_sha256: str,
) -> str:
    payload = {
        "split": split,
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "metadata": thaw_json(metadata),
        "source_line": source_line,
        "task_row_fingerprint": task_row_fingerprint,
        "task_manifest_hash": task_manifest_hash,
        "image_sha256": image_sha256,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_manifest_frame(receipt: SourceTaskIdentityReceipt) -> bytes:
    seed_identity = {
        "task_id": f"{receipt.split}:{receipt.image_id}",
        "image_id": receipt.image_id,
        "row_hash": receipt.task_row_fingerprint,
        "source_line": receipt.source_line,
    }
    return (
        json.dumps(
            seed_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256_digest(value: Any, *, field: str, code: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DataContractError(
            "identity field must be a lowercase SHA-256 digest",
            code=code,
            context={"field": field, "value": value},
        )
    return value


def _validate_bound_paths(layout: RefinementRuntimeLayout, split: Split) -> None:
    split_root = layout.split_root(split)
    source = layout.working_norm(split)
    output = layout.working_coord(split)
    manifest = layout.project_manifest(split)
    if not split_root.exists() or not split_root.is_dir() or split_root.is_symlink():
        raise DataContractError(
            "materializer split root must be an ordinary dedicated directory",
            code="label_studio.materialize_split_root",
            context={"path": str(split_root)},
        )
    if source.parent != split_root or output.parent != split_root:
        raise DataContractError(
            "materializer paths must be exact split-runtime children",
            code="label_studio.materialize_bound_path",
        )
    if not source.exists() or not source.is_file() or source.is_symlink():
        raise DataContractError(
            "exact split working.norm.jsonl must be an ordinary file, not a symlink",
            code="label_studio.materialize_source_file",
            context={"path": str(source)},
        )
    if output.exists() and (not output.is_file() or output.is_symlink()):
        raise DataContractError(
            "exact split working.coord.jsonl must be an ordinary file, not a symlink",
            code="label_studio.materialize_output_file",
            context={"path": str(output)},
        )
    if not manifest.exists() or not manifest.is_file() or manifest.is_symlink():
        raise DataContractError(
            "exact split project.json must be an ordinary manifest file",
            code="label_studio.materialize_manifest_file",
            context={"path": str(manifest)},
        )

    for canonical_split in ("train", "val"):
        canonical = layout.selected_source(canonical_split)
        for role, candidate in (("input", source), ("output", output)):
            if candidate == canonical or _same_existing_file(candidate, canonical):
                raise DataContractError(
                    "canonical selected source cannot be a materialization input or output target",
                    code="label_studio.materialize_canonical_source",
                    context={"role": role, "path": str(candidate)},
                )

    image_root = _resolved_directory(layout.image_root, field="image_root")
    images_link = layout.images_link(split)
    if not images_link.is_symlink():
        raise DataContractError(
            "working images path must be the validated managed symlink",
            code="label_studio.materialize_images_link",
            context={"path": str(images_link)},
        )
    try:
        resolved_link = images_link.resolve(strict=True)
    except OSError as exc:
        raise DataContractError(
            "managed images link cannot be resolved",
            code="label_studio.materialize_images_link",
            context={"path": str(images_link)},
            cause=exc,
        ) from exc
    if resolved_link != image_root:
        raise DataContractError(
            "managed images link does not resolve to the allowlisted shared image root",
            code="label_studio.materialize_image_root",
            context={"expected": str(image_root), "actual": str(resolved_link)},
        )


def _read_committed_generation(
    layout: RefinementRuntimeLayout,
    split: Split,
    manifest_path: Path,
) -> CommittedGenerationReceipt:
    try:
        payload: Any = json.loads(
            manifest_path.read_bytes(),
            parse_constant=_reject_json_constant,
        )
    except DataContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "project manifest is not readable strict JSON",
            code="label_studio.materialize_manifest_decode",
            context={"path": str(manifest_path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping):
        raise DataContractError(
            "project manifest must be a JSON object",
            code="label_studio.materialize_manifest_shape",
            context={"path": str(manifest_path)},
        )
    receipt = CommittedGenerationReceipt(
        split=payload.get("split"),
        generation=payload.get("generation"),
        working_sha256=payload.get("working_sha256"),
        task_count=payload.get("task_count"),
        task_manifest_hash=payload.get("task_manifest_hash"),
    )
    if receipt.split != split:
        raise DataContractError(
            "project manifest belongs to another split",
            code="label_studio.materialize_manifest_split",
            context={"expected": split, "actual": receipt.split},
        )
    expected_image_root = _resolved_directory(layout.image_root, field="image_root")
    for field in ("image_root", "document_root"):
        actual = _resolved_directory(payload.get(field), field=field)
        if actual != expected_image_root:
            raise DataContractError(
                "project manifest image root is not the exact allowlisted root",
                code="label_studio.materialize_manifest_image_root",
                context={
                    "field": field,
                    "expected": str(expected_image_root),
                    "actual": str(actual),
                },
            )
    managed_link = payload.get("managed_image_link")
    if not isinstance(managed_link, str) or Path(managed_link) != layout.images_link(
        split
    ):
        raise DataContractError(
            "project manifest managed image link does not match the split runtime",
            code="label_studio.materialize_manifest_images_link",
            context={
                "expected": str(layout.images_link(split)),
                "actual": managed_link,
            },
        )
    return receipt


def _require_generation(
    actual: CommittedGenerationReceipt,
    expected: CommittedGenerationReceipt,
    *,
    stage: str,
) -> None:
    if actual != expected:
        raise DataContractError(
            "project manifest does not match the committed generation receipt",
            code="label_studio.materialize_stale_generation",
            context={
                "stage": stage,
                "expected_generation": expected.generation,
                "actual_generation": actual.generation,
                "expected_working_sha256": expected.working_sha256,
                "actual_working_sha256": actual.working_sha256,
                "expected_task_count": expected.task_count,
                "actual_task_count": actual.task_count,
                "expected_task_manifest_hash": expected.task_manifest_hash,
                "actual_task_manifest_hash": actual.task_manifest_hash,
            },
        )


def _require_working_hash(
    actual: str,
    expected: CommittedGenerationReceipt,
    *,
    stage: str,
) -> None:
    if actual != expected.working_sha256:
        raise DataContractError(
            "working JSONL bytes do not match the committed generation receipt",
            code="label_studio.materialize_stale_working",
            context={
                "stage": stage,
                "expected": expected.working_sha256,
                "actual": actual,
            },
        )


def _validate_resolved_image(
    layout: RefinementRuntimeLayout,
    split: Split,
    row: WorkingRow,
) -> Path:
    return _resolve_image(
        layout,
        split,
        image_id=row.image_id,
        locator=row.images[0],
    )


def _resolve_image(
    layout: RefinementRuntimeLayout,
    split: Split,
    *,
    image_id: int,
    locator: str,
) -> Path:
    image_root = _resolved_directory(layout.image_root, field="image_root")
    declared = layout.split_root(split) / locator
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise DataContractError(
            "working row image does not resolve through the managed image link",
            code="label_studio.materialize_image_missing",
            context={"image_id": image_id, "image": locator},
            cause=exc,
        ) from exc
    try:
        relative = resolved.relative_to(image_root)
    except ValueError as exc:
        raise DataContractError(
            "working row image resolves outside the allowlisted shared image root",
            code="label_studio.materialize_image_escape",
            context={"image_id": image_id, "resolved": str(resolved)},
            cause=exc,
        ) from exc
    if (
        not resolved.is_file()
        or not relative.parts
        or relative.parts[0] != f"{split}2017"
    ):
        raise DataContractError(
            "working row image does not resolve inside its split-specific image directory",
            code="label_studio.materialize_image_split",
            context={"image_id": image_id, "resolved": str(resolved)},
        )
    return resolved


def _revalidate_image_contents(
    layout: RefinementRuntimeLayout,
    split: Split,
    attestations: tuple[_ImageContentAttestation, ...],
) -> None:
    for attestation in attestations:
        locator = _canonical_image_locator(split, attestation.image_id)
        resolved = _resolve_image(
            layout,
            split,
            image_id=attestation.image_id,
            locator=locator,
        )
        actual = _sha256_file(resolved)
        if actual != attestation.expected_sha256:
            raise DataContractError(
                "working row image content changed before coord replacement",
                code="label_studio.materialize_image_content_changed",
                context={
                    "image_id": attestation.image_id,
                    "path": str(resolved),
                    "expected": attestation.expected_sha256,
                    "actual": actual,
                },
            )


def _parse_jsonl_row(source: Path, row_number: int, raw_line: bytes) -> Any:
    raw_json = raw_line.rstrip(b"\r\n")
    if not raw_json.strip():
        raise DataContractError(
            "blank working JSONL rows are not allowed",
            code="label_studio.blank_row",
            context={"path": str(source), "row_number": row_number},
        )
    try:
        return json.loads(raw_json, parse_constant=_reject_json_constant)
    except DataContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "working JSONL row is not strict UTF-8 JSON",
            code="label_studio.json_decode",
            context={"path": str(source), "row_number": row_number},
            cause=exc,
        ) from exc


def _read_strict_mapping(path: Path, *, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_bytes(), parse_constant=_reject_json_constant)
    except DataContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "authority artifact is not readable strict JSON",
            code=code,
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, Mapping):
        raise DataContractError(
            "authority artifact must be a JSON object",
            code=code,
            context={"path": str(path)},
        )
    return payload


def _open_stable_binary_source(path: Path) -> tuple[BinaryIO, os.stat_result]:
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise DataContractError(
            "canonical source is unavailable",
            code="label_studio.source_task_source_changed",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise DataContractError(
            "canonical source must be a regular non-symlink file",
            code="label_studio.source_task_source_changed",
            context={"path": str(path)},
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DataContractError(
            "canonical source cannot be opened safely",
            code="label_studio.source_task_source_changed",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
        )

    if signature(path_before) != signature(opened) or not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise DataContractError(
            "canonical source changed while opening",
            code="label_studio.source_task_source_changed",
            context={"path": str(path)},
        )
    return os.fdopen(descriptor, "rb"), opened


def _resolved_directory(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise DataContractError(
            "manifest directory path must be a string",
            code="label_studio.materialize_manifest_path",
            context={"field": field, "value_type": type(value).__name__},
        )
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise DataContractError(
            "manifest directory path does not exist",
            code="label_studio.materialize_manifest_path",
            context={"field": field, "value": str(value)},
            cause=exc,
        ) from exc
    if not path.is_dir():
        raise DataContractError(
            "manifest path must identify a directory",
            code="label_studio.materialize_manifest_path",
            context={"field": field, "value": str(value)},
        )
    return path


def _same_existing_file(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise DataContractError(
        "working data and manifests must not contain NaN or Infinity",
        code="label_studio.json_constant",
        context={"constant": value},
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _invalidate_operator_receipt(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_file():
        raise DataContractError(
            "operator receipt path must be an ordinary fixed split child",
            code="label_studio.materialize_receipt_path",
            context={"path": str(path)},
        )
    path.unlink()
    _fsync_directory(path.parent)


def _run_operator_receipt_test_hook(stage: str) -> None:
    hook = _OPERATOR_RECEIPT_TEST_HOOK
    if hook is not None:
        hook(stage)


__all__ = [
    "CommittedGenerationReceipt",
    "MATERIALIZER_VERSION",
    "MaterializationReceipt",
    "OPERATOR_RECEIPT_CODE",
    "OPERATOR_RECEIPT_NAME",
    "OperatorMaterializationReceipt",
    "ProjectTaskIndexSourceIdentityResolver",
    "SourceTaskIdentityReceipt",
    "SourceTaskIdentityResolver",
    "WorkingCoordMaterializer",
]
