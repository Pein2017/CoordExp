"""Locked atomic materialization of one committed working split."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.common.errors import DataContractError
from src.data import iter_raw_examples
from src.data.examples import JsonFrozen, freeze_json, thaw_json
from src.label_studio_coco_refinement.models import (
    RefinementRuntimeLayout,
    Split,
    WorkingRow,
)


MATERIALIZER_VERSION = "label-studio-working-coord-v4"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ExclusiveLock = Callable[[], AbstractContextManager[None]]


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
        if not isinstance(self.working_sha256, str) or _SHA256_PATTERN.fullmatch(
            self.working_sha256
        ) is None:
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
    materializer_version: str = MATERIALIZER_VERSION

    def to_artifact_dict(self) -> dict[str, str | int]:
        return {
            "split": self.split,
            "generation": self.generation,
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "source_sha256": self.source_sha256,
            "destination_sha256": self.destination_sha256,
            "row_count": self.row_count,
            "object_count": self.object_count,
            "materializer_version": self.materializer_version,
        }


class WorkingCoordMaterializer:
    """Export exactly one locked committed split to its fixed coord sibling."""

    def __init__(
        self,
        layout: RefinementRuntimeLayout,
        split: Split,
        *,
        exclusive_lock: ExclusiveLock,
        source_identity_resolver: SourceTaskIdentityResolver,
    ) -> None:
        if not isinstance(layout, RefinementRuntimeLayout):
            raise DataContractError(
                "materializer requires RefinementRuntimeLayout",
                code="label_studio.materialize_layout",
                context={"value_type": type(layout).__name__},
            )
        # Validate the split eagerly before accepting the injected lock.
        layout.split_root(split)
        if not callable(exclusive_lock):
            raise DataContractError(
                "materializer requires an injected exclusive split lock",
                code="label_studio.materialize_lock",
            )
        if not callable(getattr(source_identity_resolver, "resolve", None)):
            raise DataContractError(
                "materializer requires a source task identity resolver",
                code="label_studio.materialize_source_task_resolver",
                context={"value_type": type(source_identity_resolver).__name__},
            )
        self.layout = layout
        self.split = split
        self._exclusive_lock = exclusive_lock
        self._source_identity_resolver = source_identity_resolver

    def materialize(
        self,
        committed_generation: CommittedGenerationReceipt,
    ) -> MaterializationReceipt:
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

        with self._exclusive_lock():
            _validate_bound_paths(self.layout, self.split)
            before = _read_committed_generation(self.layout, self.split, manifest)
            _require_generation(before, committed_generation, stage="before_read")
            source_hash_before = _sha256_file(source)
            _require_working_hash(
                source_hash_before,
                committed_generation,
                stage="before_read",
            )

            candidate = self._materialize_candidate(
                source=source,
                output=output,
                committed_generation=committed_generation,
            )
            candidate_path: Path | None = candidate._candidate_path
            try:
                # Re-read both manifest authority and working bytes immediately before
                # replacement while the injected store-equivalent lock remains held.
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
                os.replace(candidate._candidate_path, output)
                candidate_path = None
                _fsync_directory(output.parent)
            finally:
                if candidate_path is not None:
                    candidate_path.unlink(missing_ok=True)

        return MaterializationReceipt(
            split=self.split,
            generation=committed_generation.generation,
            source_path=source,
            destination_path=output,
            source_sha256=candidate.source_sha256,
            destination_sha256=candidate.destination_sha256,
            row_count=candidate.row_count,
            object_count=candidate.object_count,
        )

    def _materialize_candidate(
        self,
        *,
        source: Path,
        output: Path,
        committed_generation: CommittedGenerationReceipt,
    ) -> "_CandidateReceipt":
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        row_count = 0
        object_count = 0
        seen_image_ids: set[int] = set()
        task_manifest_digest = hashlib.sha256()
        image_attestations: list[_ImageContentAttestation] = []
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temp_path = Path(temp_name)
        keep_candidate = False
        try:
            with source.open("rb") as source_handle, os.fdopen(
                descriptor,
                "wb",
            ) as output_handle:
                for row_number, raw_line in enumerate(source_handle, start=1):
                    source_digest.update(raw_line)
                    payload = _parse_jsonl_row(source, row_number, raw_line)
                    row = WorkingRow.from_mapping(
                        payload,
                        field=f"row[{row_number}]",
                    ).validate_for_split(self.split)
                    try:
                        source_identity = self._source_identity_resolver.resolve(
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
                    if source_identity.task_manifest_hash != committed_generation.task_manifest_hash:
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
                    resolved_image = _validate_resolved_image(self.layout, self.split, row)
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
                            context={"image_id": row.image_id, "row_number": row_number},
                        )
                    seen_image_ids.add(row.image_id)
                    task_manifest_digest.update(_task_manifest_frame(source_identity))
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
    image_attestations: tuple["_ImageContentAttestation", ...]


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
                context={"field": field, "expected": str(expected_image_root), "actual": str(actual)},
            )
    managed_link = payload.get("managed_image_link")
    if not isinstance(managed_link, str) or Path(managed_link) != layout.images_link(split):
        raise DataContractError(
            "project manifest managed image link does not match the split runtime",
            code="label_studio.materialize_manifest_images_link",
            context={"expected": str(layout.images_link(split)), "actual": managed_link},
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
    if not resolved.is_file() or not relative.parts or relative.parts[0] != f"{split}2017":
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


__all__ = [
    "CommittedGenerationReceipt",
    "ExclusiveLock",
    "MATERIALIZER_VERSION",
    "MaterializationReceipt",
    "SourceTaskIdentityReceipt",
    "SourceTaskIdentityResolver",
    "WorkingCoordMaterializer",
]
