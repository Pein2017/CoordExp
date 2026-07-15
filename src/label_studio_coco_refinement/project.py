"""Pure contracts and bootstrap planning for COCO Label Studio refinement.

This module deliberately does not import Label Studio or perform project, file, or
symlink mutations.  It validates the two approved source splits and produces a
deterministic plan for a separate adapter to apply.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import quote, unquote


ADAPTER_VERSION = "label-studio-coco-refinement-v1"
DATASET_NAME = "rescale_32_1024_bbox_len12000"
SOURCE_DIRECTORY = PurePosixPath("public_data/coco/rescale_32_1024_bbox_len12000")
SHARED_IMAGE_ROOT = PurePosixPath("public_data/coco/rescale_32_1024_bbox/images")
RUNTIME_ROOT = PurePosixPath(
    "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
)
LOCAL_FILES_URL_PREFIX = "/data/local-files/?d="
SOURCE_ROW_FIELDS = frozenset(
    {"images", "objects", "width", "height", "image_id", "file_name", "metadata"}
)
SOURCE_OBJECT_FIELDS = frozenset(
    {"bbox_2d", "desc", "category_id", "category_name", "coco_ann_id"}
)
IMMUTABLE_ROW_FIELDS = ("file_name", "image_id", "width", "height", "metadata")


class ProjectContractError(ValueError):
    """An exact source, path, task, or project contract was violated."""


class ManifestDriftError(ProjectContractError):
    """An existing project differs from the deterministic desired manifest."""

    def __init__(self, mismatches: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__("project manifest drift: " + ", ".join(self.mismatches))


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"


def _split(value: Split | str) -> Split:
    try:
        return value if isinstance(value, Split) else Split(value)
    except ValueError as exc:
        raise ProjectContractError(
            f"unsupported split {value!r}; expected 'train' or 'val'"
        ) from exc


@dataclass(frozen=True)
class SourceContract:
    split: Split
    relative_path: PurePosixPath
    sha256: str
    row_count: int
    box_count: int
    image_subdirectory: str

    def path(self, repo_root: Path) -> Path:
        return _absolute(repo_root) / Path(self.relative_path)


SOURCE_CONTRACTS: Mapping[Split, SourceContract] = MappingProxyType(
    {
        Split.TRAIN: SourceContract(
            split=Split.TRAIN,
            relative_path=SOURCE_DIRECTORY / "train.norm.jsonl",
            sha256="d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a",
            row_count=117_266,
            box_count=849_947,
            image_subdirectory="train2017",
        ),
        Split.VAL: SourceContract(
            split=Split.VAL,
            relative_path=SOURCE_DIRECTORY / "val.norm.jsonl",
            sha256="a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795",
            row_count=4_952,
            box_count=36_335,
            image_subdirectory="val2017",
        ),
    }
)


class Coco80RegistryProtocol(Protocol):
    """Narrow category dependency supplied by ``categories.Coco80Registry``."""

    @property
    def names(self) -> Sequence[str]: ...

    @property
    def fingerprint(self) -> str: ...

    def validate(self, name: str, category_id: int) -> object: ...


BboxToLabelStudio = Callable[[Sequence[int]], tuple[float, float, float, float]]


class RefinementProjectAdapter(Protocol):
    """Read-only seam for a Label Studio-specific live-state attestor.

    Implementations must derive this receipt from current vendor/runtime state,
    not merely return the CoordExp-side saved project manifest.  Project
    planning remains pure and does not prescribe an HTTP framework.
    """

    def attest_project(self, split: Split) -> "LiveProjectAttestation | None": ...

    def attest_bootstrap_manifest(self) -> Mapping[str, Any] | None:
        """Return the parent-owned one-instance manifest, never vendor guesses."""

        ...


def default_registry() -> Coco80RegistryProtocol:
    """Load the sibling canonical registry lazily to keep this module decoupled."""

    from .categories import COCO80_REGISTRY

    return COCO80_REGISTRY


def default_bbox_converter() -> BboxToLabelStudio:
    """Load the sibling norm1000 conversion lazily, without duplicating it."""

    from .geometry import norm1000_bbox_to_label_studio_xywh

    return norm1000_bbox_to_label_studio_xywh


@dataclass(frozen=True, order=True)
class TaskIdentity:
    split: Split
    image_id: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.image_id, bool)
            or not isinstance(self.image_id, int)
            or self.image_id < 0
        ):
            raise ProjectContractError("image_id must be a non-negative integer")

    @property
    def key(self) -> str:
        return f"{self.split.value}:{self.image_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split.value, "image_id": self.image_id, "key": self.key}


@dataclass(frozen=True)
class SourceInspection:
    contract: SourceContract
    source_path: str
    sha256: str
    row_count: int
    box_count: int
    task_identity_fingerprint: str

    def body(self) -> dict[str, Any]:
        return {
            "split": self.contract.split.value,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "box_count": self.box_count,
            "task_identity_fingerprint": self.task_identity_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class SplitRuntimeLayout:
    root: Path
    project_manifest: Path
    working_norm_jsonl: Path
    task_index_json: Path
    queue_jsonl: Path
    journal_jsonl: Path
    lock_file: Path
    images_link: Path


@dataclass(frozen=True)
class RuntimeLayout:
    repo_root: Path
    root: Path
    label_studio_state: Path
    image_root: Path

    @classmethod
    def for_repo(cls, repo_root: Path) -> "RuntimeLayout":
        root = _absolute(repo_root)
        return cls(
            repo_root=root,
            root=root / Path(RUNTIME_ROOT),
            label_studio_state=root / Path(RUNTIME_ROOT) / "label-studio" / "state",
            image_root=root / Path(SHARED_IMAGE_ROOT),
        )

    def for_split(self, split: Split | str) -> SplitRuntimeLayout:
        split_value = _split(split).value
        root = self.root / split_value
        return SplitRuntimeLayout(
            root=root,
            project_manifest=root / "project.json",
            working_norm_jsonl=root / "working.norm.jsonl",
            task_index_json=root / "task_index.json",
            queue_jsonl=root / "queue.jsonl",
            journal_jsonl=root / "journal.jsonl",
            lock_file=root / "project.lock",
            images_link=root / "images",
        )


@dataclass(frozen=True)
class ManagedImageLinkPlan:
    split: Split
    link_path: Path
    target_path: Path

    def assert_existing_matches(self) -> None:
        """Fail closed unless an existing managed link resolves to the allowlist root."""

        if not self.link_path.is_symlink():
            raise ProjectContractError(
                f"managed image path is not a symlink: {self.link_path}"
            )
        try:
            observed_target = self.link_path.resolve(strict=True)
            expected_target = self.target_path.resolve(strict=True)
        except OSError as exc:
            raise ProjectContractError(
                f"managed image link cannot be resolved for {self.split.value}: {self.link_path}"
            ) from exc
        if observed_target != expected_target:
            raise ProjectContractError(
                f"managed image link target drift for {self.split.value}: {self.link_path}"
            )


def managed_image_link_plan(
    layout: RuntimeLayout, split: Split | str
) -> ManagedImageLinkPlan:
    split_value = _split(split)
    return ManagedImageLinkPlan(
        split=split_value,
        link_path=layout.for_split(split_value).images_link,
        target_path=layout.image_root,
    )


def resolve_working_image(
    locator: str,
    *,
    layout: RuntimeLayout,
    split: Split | str,
) -> Path:
    """Resolve ``images/<split2017>/...`` inside the exact shared image root."""

    contract = SOURCE_CONTRACTS[_split(split)]
    relative = _safe_relative(locator, field="working image locator")
    expected_prefix = PurePosixPath("images") / contract.image_subdirectory
    try:
        inside_root = relative.relative_to(expected_prefix)
    except ValueError as exc:
        raise ProjectContractError(
            f"working image locator must be under {expected_prefix.as_posix()}/"
        ) from exc
    if not inside_root.parts:
        raise ProjectContractError("working image locator must name a file")
    root = layout.image_root.resolve(strict=False)
    resolved = (root / Path(contract.image_subdirectory) / Path(inside_root)).resolve(
        strict=False
    )
    if not resolved.is_relative_to(root):
        raise ProjectContractError(
            "working image locator escapes the shared image root"
        )
    return resolved


def local_files_image_locator(file_name: str, *, split: Split | str) -> str:
    """Map a managed working locator to the exact Label Studio local-files URL."""

    split_value = _split(split)
    contract = SOURCE_CONTRACTS[split_value]
    relative = _safe_relative(file_name, field="file_name")
    expected_prefix = PurePosixPath("images") / contract.image_subdirectory
    try:
        inside_document_root = relative.relative_to(PurePosixPath("images"))
        inside_document_root.relative_to(contract.image_subdirectory)
    except ValueError as exc:
        raise ProjectContractError(
            f"file_name must be under {expected_prefix.as_posix()}/"
        ) from exc
    if len(inside_document_root.parts) != 2:
        raise ProjectContractError(
            "Label Studio image locator must name one split JPEG"
        )
    return LOCAL_FILES_URL_PREFIX + quote(inside_document_root.as_posix(), safe="/")


def local_files_image_locator_from_data(locator: Any, *, split: Split | str) -> str:
    """Canonicalize and validate an imported local-files task locator."""

    if not isinstance(locator, str) or not locator.startswith(LOCAL_FILES_URL_PREFIX):
        raise ProjectContractError(
            "task image must use the local-files locator contract"
        )
    encoded = locator[len(LOCAL_FILES_URL_PREFIX) :]
    if not encoded:
        raise ProjectContractError("task image locator must name a file")
    decoded = unquote(encoded)
    return local_files_image_locator(f"images/{decoded}", split=split)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_exact_source_path(
    path: Path,
    *,
    repo_root: Path,
    split: Split | str,
) -> SourceContract:
    contract, observed = _assert_exact_source_location(
        path,
        repo_root=repo_root,
        split=split,
    )
    observed_hash = sha256_file(observed)
    if observed_hash != contract.sha256:
        raise ProjectContractError(
            f"source hash drift for {contract.split.value}: expected {contract.sha256}, "
            f"got {observed_hash}"
        )
    return contract


def _assert_exact_source_location(
    path: Path,
    *,
    repo_root: Path,
    split: Split | str,
) -> tuple[SourceContract, Path]:
    contract = SOURCE_CONTRACTS[_split(split)]
    expected_path = contract.path(repo_root)
    expected_lexical = _lexical_absolute(expected_path)
    observed_lexical = _lexical_absolute(path)
    if observed_lexical != expected_lexical:
        raise ProjectContractError(
            f"source path drift for {contract.split.value}: expected {expected_lexical}, "
            f"got {observed_lexical}"
        )
    expected = expected_path.resolve(strict=True)
    observed = path.resolve(strict=True)
    if observed != expected:
        raise ProjectContractError(f"source target drift for {contract.split.value}")
    return contract, observed


def validate_source_row(
    row: Mapping[str, Any],
    *,
    split: Split | str,
    registry: Coco80RegistryProtocol | None = None,
) -> TaskIdentity:
    """Validate the exact selected-source schema and return stable row identity."""

    split_value = _split(split)
    contract = SOURCE_CONTRACTS[split_value]
    _require_exact_keys(row, SOURCE_ROW_FIELDS, context="source row")
    image_id = _require_int(row["image_id"], "image_id", minimum=0)
    _require_int(row["width"], "width", minimum=1)
    _require_int(row["height"], "height", minimum=1)
    file_name = _require_string(row["file_name"], "file_name")
    expected_prefix = PurePosixPath("images") / contract.image_subdirectory
    file_path = _safe_relative(file_name, field="file_name")
    try:
        image_leaf = file_path.relative_to(expected_prefix)
    except ValueError as exc:
        raise ProjectContractError(
            f"file_name must be under {expected_prefix.as_posix()}/"
        ) from exc
    if len(image_leaf.parts) != 1 or image_leaf.suffix.lower() != ".jpg":
        raise ProjectContractError(
            "file_name must identify one COCO JPEG below the split root"
        )
    if not image_leaf.stem.isdigit() or int(image_leaf.stem) != image_id:
        raise ProjectContractError("file_name stem must equal image_id")

    images = row["images"]
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        raise ProjectContractError("images must contain exactly one string locator")
    expected_source_locator = (
        PurePosixPath("..") / "rescale_32_1024_bbox" / file_path
    ).as_posix()
    if images[0] != expected_source_locator:
        raise ProjectContractError(
            f"source image locator drift: expected {expected_source_locator!r}, got {images[0]!r}"
        )

    metadata = row["metadata"]
    if not isinstance(metadata, Mapping) or dict(metadata) != {
        "source": "coco2017",
        "split": split_value.value,
    }:
        raise ProjectContractError(
            "metadata must be exactly the selected COCO source and split"
        )

    objects = row["objects"]
    if not isinstance(objects, list) or not objects:
        raise ProjectContractError("source objects must be a non-empty list")
    seen_annotation_ids: set[int] = set()
    for index, obj in enumerate(objects):
        _validate_source_object(obj, index=index, registry=registry)
        annotation_id = int(obj["coco_ann_id"])
        if annotation_id in seen_annotation_ids:
            raise ProjectContractError(
                f"duplicate coco_ann_id {annotation_id} in source row"
            )
        seen_annotation_ids.add(annotation_id)
    return TaskIdentity(split_value, image_id)


def validate_immutable_row_fields(
    source_row: Mapping[str, Any],
    working_row: Mapping[str, Any],
    *,
    split: Split | str,
) -> None:
    """Require immutable fields unchanged while enforcing the deliberate locator rebase."""

    split_value = _split(split)
    validate_source_row(source_row, split=split_value)
    _require_exact_keys(working_row, SOURCE_ROW_FIELDS, context="working row")
    mismatches = [
        field
        for field in IMMUTABLE_ROW_FIELDS
        if working_row.get(field) != source_row.get(field)
    ]
    if mismatches:
        raise ProjectContractError(
            "immutable row field drift: " + ", ".join(mismatches)
        )
    expected_locator = source_row["file_name"]
    if working_row.get("images") != [expected_locator]:
        raise ProjectContractError(
            f"working images must be the managed-link locator {[expected_locator]!r}"
        )


def inspect_source(
    path: Path,
    *,
    repo_root: Path,
    split: Split | str,
    registry: Coco80RegistryProtocol | None = None,
) -> SourceInspection:
    """Hash and exhaustively validate one approved JSONL source in one stream."""

    return _inspect_source_stream(
        path,
        repo_root=repo_root,
        split=split,
        registry=registry,
    )


SourceRowConsumer = Callable[[int, Mapping[str, Any], TaskIdentity], None]


def _inspect_source_stream(
    path: Path,
    *,
    repo_root: Path,
    split: Split | str,
    registry: Coco80RegistryProtocol | None,
    consume_row: SourceRowConsumer | None = None,
) -> SourceInspection:
    """Consume, hash, and validate the exact bytes used to derive a plan."""

    contract, observed_path = _assert_exact_source_location(
        path,
        repo_root=repo_root,
        split=split,
    )
    digest = hashlib.sha256()
    identities: list[str] = []
    box_count = 0
    seen: set[TaskIdentity] = set()
    with observed_path.open("rb") as handle:
        for line_number, encoded_line in enumerate(handle, start=1):
            digest.update(encoded_line)
            try:
                row = json.loads(encoded_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProjectContractError(
                    f"invalid JSON at source line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ProjectContractError(
                    f"source line {line_number} is not an object"
                )
            identity = validate_source_row(row, split=contract.split, registry=registry)
            if identity in seen:
                raise ProjectContractError(f"duplicate task identity {identity.key}")
            _assert_source_image_exists(
                row,
                repo_root=repo_root,
                split=contract.split,
                line_number=line_number,
            )
            seen.add(identity)
            identities.append(identity.key)
            box_count += len(row["objects"])
            if consume_row is not None:
                consume_row(line_number, row, identity)
    observed_hash = digest.hexdigest()
    if observed_hash != contract.sha256:
        raise ProjectContractError(
            f"source hash drift for {contract.split.value}: expected {contract.sha256}, "
            f"got {observed_hash}"
        )
    if len(identities) != contract.row_count:
        raise ProjectContractError(
            f"source row-count drift: expected {contract.row_count}, got {len(identities)}"
        )
    if box_count != contract.box_count:
        raise ProjectContractError(
            f"source box-count drift: expected {contract.box_count}, got {box_count}"
        )
    return SourceInspection(
        contract=contract,
        source_path=str(_lexical_absolute(path)),
        sha256=observed_hash,
        row_count=len(identities),
        box_count=box_count,
        task_identity_fingerprint=fingerprint_json(identities),
    )


@dataclass(frozen=True)
class FrozenTaskImport:
    """One recursively immutable canonical Label Studio task import."""

    canonical_json: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, bytes):
            raise ProjectContractError("frozen task import must use canonical bytes")
        payload = _decode_task_import(self.canonical_json)
        validate_authoritative_task_payload(payload)
        if _canonical_json_bytes(payload) != self.canonical_json:
            raise ProjectContractError(
                "frozen task import bytes are not canonical JSON"
            )

    @classmethod
    def freeze(cls, payload: Mapping[str, Any]) -> "FrozenTaskImport":
        """Validate and detach a mutable task payload into canonical bytes."""

        validate_authoritative_task_payload(payload)
        return cls(_canonical_json_bytes(payload))

    def _thaw(self) -> dict[str, Any]:
        """Internal validation copy; public mutable copies come from the action seam."""

        return _decode_task_import(self.canonical_json)


@dataclass(frozen=True)
class TaskManifestEntry:
    identity: TaskIdentity
    source_line: int
    source_image_locator: str
    working_image_locator: str
    label_studio_image_locator: str
    task_data_fingerprint: str
    authoritative_annotation_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "source_line": self.source_line,
            "source_image_locator": self.source_image_locator,
            "working_image_locator": self.working_image_locator,
            "label_studio_image_locator": self.label_studio_image_locator,
            "task_data_fingerprint": self.task_data_fingerprint,
            "authoritative_annotation_fingerprint": self.authoritative_annotation_fingerprint,
        }


@dataclass(frozen=True)
class TaskManifest:
    split: Split
    entries: tuple[TaskManifestEntry, ...]

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "task_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())

    @property
    def identity_fingerprint(self) -> str:
        return fingerprint_json(sorted(entry.identity.key for entry in self.entries))

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class StorageManifest:
    split: Split
    project_identity: str
    storage_identity: str
    document_root: str
    storage_subdirectory: str
    task_locator_prefix: str
    managed_link: str
    managed_link_target: str

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "project_identity": self.project_identity,
            "storage_identity": self.storage_identity,
            "document_root": self.document_root,
            "storage_subdirectory": self.storage_subdirectory,
            "task_locator_prefix": self.task_locator_prefix,
            "managed_link": self.managed_link,
            "managed_link_target": self.managed_link_target,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class ProjectControls:
    authoritative_annotations_per_task: int = 1
    allow_alternate_annotations: bool = False
    allow_annotation_deletion: bool = False
    show_native_submit: bool = False
    show_native_skip: bool = False
    allow_region_crud: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "authoritative_annotations_per_task": self.authoritative_annotations_per_task,
            "allow_alternate_annotations": self.allow_alternate_annotations,
            "allow_annotation_deletion": self.allow_annotation_deletion,
            "show_native_submit": self.show_native_submit,
            "show_native_skip": self.show_native_skip,
            "allow_region_crud": self.allow_region_crud,
        }


@dataclass(frozen=True)
class ProjectManifest:
    split: Split
    project_identity: str
    source_path: str
    source_sha256: str
    source_row_count: int
    source_box_count: int
    source_inspection_fingerprint: str
    adapter_version: str
    vendor_revision: str
    category_registry_fingerprint: str
    label_config_fingerprint: str
    authoritative_annotation_policy_fingerprint: str
    task_manifest_fingerprint: str
    storage_manifest_fingerprint: str
    image_root_identity: str

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "project_identity": self.project_identity,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_row_count": self.source_row_count,
            "source_box_count": self.source_box_count,
            "source_inspection_fingerprint": self.source_inspection_fingerprint,
            "adapter_version": self.adapter_version,
            "vendor_revision": self.vendor_revision,
            "category_registry_fingerprint": self.category_registry_fingerprint,
            "label_config_fingerprint": self.label_config_fingerprint,
            "authoritative_annotation_policy_fingerprint": (
                self.authoritative_annotation_policy_fingerprint
            ),
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "storage_manifest_fingerprint": self.storage_manifest_fingerprint,
            "image_root_identity": self.image_root_identity,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class SplitProjectPlan:
    split: Split
    source_inspection: SourceInspection
    manifest: ProjectManifest
    task_manifest: TaskManifest
    storage_manifest: StorageManifest
    controls: ProjectControls
    label_config: str
    task_imports: tuple[FrozenTaskImport, ...]


class BootstrapAction(str, Enum):
    CREATE = "create"
    RECONCILE = "reconcile"
    REUSE = "reuse"


@dataclass(frozen=True)
class PlannedProjectBootstrap:
    action: BootstrapAction
    project: SplitProjectPlan
    missing_task_imports: tuple[FrozenTaskImport, ...]
    reused_task_identities: tuple[TaskIdentity, ...]
    live_attestation_fingerprint: str | None = None

    def task_imports_for_adapter_send(self) -> tuple[dict[str, Any], ...]:
        """Return detached ordinary JSON only after revalidating the action payloads."""

        return _task_imports_for_adapter_send(self)


@dataclass(frozen=True)
class InstanceBootstrapPlan:
    runtime_layout: RuntimeLayout
    manifest: "InstanceBootstrapManifest"
    projects: tuple[PlannedProjectBootstrap, ...]


@dataclass(frozen=True)
class InstanceBootstrapManifest:
    """Deterministic parent-owned manifest for one instance and two projects."""

    adapter_version: str
    dataset_name: str
    runtime_root: str
    label_studio_state: str
    local_files_document_root: str
    project_manifest_fingerprints: Mapping[str, str]
    task_manifest_fingerprints: Mapping[str, str]
    storage_manifest_fingerprints: Mapping[str, str]

    def body(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "dataset_name": self.dataset_name,
            "runtime_root": self.runtime_root,
            "label_studio_state": self.label_studio_state,
            "local_files_document_root": self.local_files_document_root,
            "project_manifest_fingerprints": dict(self.project_manifest_fingerprints),
            "task_manifest_fingerprints": dict(self.task_manifest_fingerprints),
            "storage_manifest_fingerprints": dict(self.storage_manifest_fingerprints),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "fingerprint": self.fingerprint}


LiveIdentifier = int | str


@dataclass(frozen=True)
class LiveTaskAttestation:
    """Current Label Studio task/annotation counts and stable identities."""

    identity: TaskIdentity
    source_line: int
    image_locator: str
    task_data_fingerprint: str
    task_id: LiveIdentifier
    annotation_count: int
    authoritative_annotation_id: LiveIdentifier
    authoritative_annotation_revision: LiveIdentifier
    authoritative_annotation_fingerprint: str
    authoritative_annotation_ground_truth: bool
    alternate_annotation_count: int
    prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "source_line": self.source_line,
            "image_locator": self.image_locator,
            "task_data_fingerprint": self.task_data_fingerprint,
            "task_id": self.task_id,
            "annotation_count": self.annotation_count,
            "authoritative_annotation_id": self.authoritative_annotation_id,
            "authoritative_annotation_revision": self.authoritative_annotation_revision,
            "authoritative_annotation_fingerprint": (
                self.authoritative_annotation_fingerprint
            ),
            "authoritative_annotation_ground_truth": (
                self.authoritative_annotation_ground_truth
            ),
            "alternate_annotation_count": self.alternate_annotation_count,
            "prediction_count": self.prediction_count,
        }


@dataclass(frozen=True)
class LiveProjectAttestation:
    """Read-only receipt derived from current Label Studio and managed storage."""

    split: Split
    project_id: LiveIdentifier
    project_identity: str
    saved_manifest: Mapping[str, Any]
    vendor_revision: str
    label_config: str
    controls: ProjectControls
    storage_manifest: StorageManifest
    managed_link_is_symlink: bool
    managed_link_resolved_target: str
    tasks: tuple[LiveTaskAttestation, ...]

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "project_id": self.project_id,
            "project_identity": self.project_identity,
            "saved_manifest": dict(self.saved_manifest),
            "vendor_revision": self.vendor_revision,
            "label_config_fingerprint": fingerprint_json(
                {"label_config_xml": self.label_config}
            ),
            "controls": self.controls.to_dict(),
            "storage_manifest": self.storage_manifest.to_dict(),
            "managed_link_is_symlink": self.managed_link_is_symlink,
            "managed_link_resolved_target": self.managed_link_resolved_target,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.body())


def make_task_import_payload(
    row: Mapping[str, Any],
    *,
    split: Split | str,
    source_line: int,
    registry: Coco80RegistryProtocol | None = None,
    bbox_converter: BboxToLabelStudio | None = None,
) -> dict[str, Any]:
    """Build one task with exactly one editable annotation and no predictions."""

    split_value = _split(split)
    _require_int(source_line, "source_line", minimum=1)
    registry = registry or default_registry()
    bbox_converter = bbox_converter or default_bbox_converter()
    identity = validate_source_row(row, split=split_value, registry=registry)
    contract = SOURCE_CONTRACTS[split_value]
    file_name = str(row["file_name"])
    relative_to_document_root = PurePosixPath(file_name).relative_to("images")
    if relative_to_document_root.parts[0] != contract.image_subdirectory:
        raise ProjectContractError("task image does not belong to its project storage")
    results: list[dict[str, Any]] = []
    for obj in row["objects"]:
        x, y, width, height = bbox_converter(obj["bbox_2d"])
        region_key = f"{split_value.value}:coco:{obj['coco_ann_id']}"
        results.append(
            {
                "id": region_key,
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "original_width": row["width"],
                "original_height": row["height"],
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "rectanglelabels": [obj["category_name"]],
                },
                "meta": {
                    "coordexp_region_key": region_key,
                    "last_committed_bbox": list(obj["bbox_2d"]),
                    "coco_ann_id": obj["coco_ann_id"],
                },
            }
        )
    image_locator = local_files_image_locator(file_name, split=split_value)
    payload = {
        "data": {
            "image": image_locator,
            "coordexp_task_key": identity.key,
            "split": split_value.value,
            "image_id": identity.image_id,
            "source_line": source_line,
        },
        "annotations": [{"result": results, "ground_truth": False}],
    }
    validate_authoritative_task_payload(payload, registry=registry)
    return payload


def validate_authoritative_task_payload(
    payload: Mapping[str, Any],
    *,
    registry: Coco80RegistryProtocol | None = None,
) -> None:
    """Validate the imported one-annotation, bbox-only, zero-rotation boundary."""

    if "predictions" in payload:
        raise ProjectContractError("refinement tasks must never contain predictions")
    if set(payload) != {"data", "annotations"}:
        raise ProjectContractError("task import payload has an unsupported shape")
    data = payload.get("data")
    expected_data_fields = {
        "image",
        "coordexp_task_key",
        "split",
        "image_id",
        "source_line",
    }
    if not isinstance(data, Mapping) or set(data) != expected_data_fields:
        raise ProjectContractError("task data has an unsupported shape")
    split_value = _split(data["split"])
    identity = TaskIdentity(
        split_value, _require_int(data["image_id"], "image_id", minimum=0)
    )
    if data["coordexp_task_key"] != identity.key:
        raise ProjectContractError("task source identity does not match split/image_id")
    _require_int(data["source_line"], "source_line", minimum=1)
    canonical_locator = local_files_image_locator_from_data(
        data["image"], split=split_value
    )
    contract = SOURCE_CONTRACTS[split_value]
    identity_locator = local_files_image_locator(
        (f"images/{contract.image_subdirectory}/{identity.image_id:012d}.jpg"),
        split=split_value,
    )
    if data["image"] != canonical_locator or data["image"] != identity_locator:
        raise ProjectContractError("task image locator is not canonical")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ProjectContractError(
            "task must contain exactly one authoritative annotation"
        )
    annotation = annotations[0]
    if not isinstance(annotation, Mapping) or set(annotation) != {
        "result",
        "ground_truth",
    }:
        raise ProjectContractError("authoritative annotation has an unsupported shape")
    if annotation["ground_truth"] is not False:
        raise ProjectContractError(
            "authoritative annotation ground_truth must be false"
        )
    results = annotation["result"]
    if not isinstance(results, list) or not results:
        raise ProjectContractError(
            "authoritative annotation must contain source rectangles"
        )
    registry = registry or default_registry()
    seen_region_keys: set[str] = set()
    seen_annotation_ids: set[int] = set()
    result_fields = {
        "id",
        "type",
        "from_name",
        "to_name",
        "original_width",
        "original_height",
        "image_rotation",
        "value",
        "meta",
    }
    value_fields = {"x", "y", "width", "height", "rotation", "rectanglelabels"}
    meta_fields = {"coordexp_region_key", "last_committed_bbox", "coco_ann_id"}
    for result in results:
        if not isinstance(result, Mapping) or set(result) != result_fields:
            raise ProjectContractError("rectangle result has an unsupported shape")
        if result.get("type") != "rectanglelabels":
            raise ProjectContractError("only rectanglelabels results are supported")
        if result.get("from_name") != "bbox" or result.get("to_name") != "image":
            raise ProjectContractError("rectangle result is not bound to bbox/image")
        if result.get("image_rotation") != 0:
            raise ProjectContractError("image_rotation must be zero")
        value = result.get("value")
        if not isinstance(value, Mapping) or set(value) != value_fields:
            raise ProjectContractError("bbox value has an unsupported shape")
        if value.get("rotation") != 0:
            raise ProjectContractError("bbox rotation must be zero")
        original_width = _require_int(
            result["original_width"], "original_width", minimum=1
        )
        original_height = _require_int(
            result["original_height"], "original_height", minimum=1
        )
        if not original_width or not original_height:
            raise ProjectContractError("rectangle canvas must be positive")
        x = _require_finite_number(value["x"], "bbox x", minimum=0.0, maximum=100.0)
        y = _require_finite_number(value["y"], "bbox y", minimum=0.0, maximum=100.0)
        width = _require_finite_number(
            value["width"], "bbox width", minimum=0.0, maximum=100.0
        )
        height = _require_finite_number(
            value["height"], "bbox height", minimum=0.0, maximum=100.0
        )
        if (
            width <= 0
            or height <= 0
            or x + width > 100.0 + 1e-9
            or y + height > 100.0 + 1e-9
        ):
            raise ProjectContractError("bbox percentage geometry is invalid")
        labels = value.get("rectanglelabels")
        if (
            not isinstance(labels, list)
            or len(labels) != 1
            or labels[0] not in registry.names
        ):
            raise ProjectContractError(
                "bbox must have exactly one canonical COCO-80 label"
            )
        meta = result.get("meta")
        if not isinstance(meta, Mapping) or set(meta) != meta_fields:
            raise ProjectContractError(
                "bbox hidden identity metadata has an unsupported shape"
            )
        annotation_id = _require_int(meta["coco_ann_id"], "coco_ann_id", minimum=1)
        region_key = f"{split_value.value}:coco:{annotation_id}"
        if result["id"] != region_key or meta["coordexp_region_key"] != region_key:
            raise ProjectContractError("bbox hidden region identity is inconsistent")
        if region_key in seen_region_keys or annotation_id in seen_annotation_ids:
            raise ProjectContractError("duplicate bbox hidden region identity")
        seen_region_keys.add(region_key)
        seen_annotation_ids.add(annotation_id)
        bbox = meta["last_committed_bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ProjectContractError("last_committed_bbox must contain four integers")
        x1, y1, x2, y2 = (
            _require_int(item, "last_committed_bbox", minimum=0, maximum=999)
            for item in bbox
        )
        if x1 >= x2 or y1 >= y2:
            raise ProjectContractError("last_committed_bbox must have positive area")


def build_split_project_plan(
    source_path: Path,
    *,
    repo_root: Path,
    split: Split | str,
    vendor_revision: str,
    label_config: str,
    label_config_fingerprint: str,
    registry: Coco80RegistryProtocol | None = None,
    bbox_converter: BboxToLabelStudio | None = None,
) -> SplitProjectPlan:
    """Build a plan from the exact pinned source bytes consumed in this call."""

    split_value = _split(split)
    registry = registry or default_registry()
    bbox_converter = bbox_converter or default_bbox_converter()
    from .label_config import validate_label_config

    validate_label_config(label_config, expected_names=registry.names)
    if label_config_fingerprint != fingerprint_json({"label_config_xml": label_config}):
        raise ProjectContractError(
            "label_config_fingerprint does not match label_config"
        )
    layout = RuntimeLayout.for_repo(repo_root)
    project_identity = f"coco-refinement:{DATASET_NAME}:{split_value.value}"
    imports: list[FrozenTaskImport] = []
    entries: list[TaskManifestEntry] = []

    def consume_row(
        line_number: int,
        row: Mapping[str, Any],
        identity: TaskIdentity,
    ) -> None:
        task = make_task_import_payload(
            row,
            split=split_value,
            source_line=line_number,
            registry=registry,
            bbox_converter=bbox_converter,
        )
        imports.append(FrozenTaskImport.freeze(task))
        entries.append(
            TaskManifestEntry(
                identity=identity,
                source_line=line_number,
                source_image_locator=row["images"][0],
                working_image_locator=row["file_name"],
                label_studio_image_locator=task["data"]["image"],
                task_data_fingerprint=fingerprint_json(task["data"]),
                authoritative_annotation_fingerprint=fingerprint_json(
                    task["annotations"][0]
                ),
            )
        )

    source_inspection = _inspect_source_stream(
        source_path,
        repo_root=repo_root,
        split=split_value,
        registry=registry,
        consume_row=consume_row,
    )
    contract = source_inspection.contract
    task_manifest = TaskManifest(split_value, tuple(entries))
    split_layout = layout.for_split(split_value)
    storage_manifest = StorageManifest(
        split=split_value,
        project_identity=project_identity,
        storage_identity=f"local-files:{DATASET_NAME}:{split_value.value}",
        document_root=str(layout.image_root),
        storage_subdirectory=contract.image_subdirectory,
        task_locator_prefix=LOCAL_FILES_URL_PREFIX,
        managed_link=str(split_layout.images_link),
        managed_link_target=str(layout.image_root),
    )
    controls = ProjectControls()
    policy_fingerprint = fingerprint_json(controls.to_dict())
    manifest = ProjectManifest(
        split=split_value,
        project_identity=project_identity,
        source_path=source_inspection.source_path,
        source_sha256=source_inspection.sha256,
        source_row_count=source_inspection.row_count,
        source_box_count=source_inspection.box_count,
        source_inspection_fingerprint=source_inspection.fingerprint,
        adapter_version=ADAPTER_VERSION,
        vendor_revision=_require_string(vendor_revision, "vendor_revision"),
        category_registry_fingerprint=registry.fingerprint,
        label_config_fingerprint=label_config_fingerprint,
        authoritative_annotation_policy_fingerprint=policy_fingerprint,
        task_manifest_fingerprint=task_manifest.fingerprint,
        storage_manifest_fingerprint=storage_manifest.fingerprint,
        image_root_identity=str(layout.image_root),
    )
    return SplitProjectPlan(
        split=split_value,
        source_inspection=source_inspection,
        manifest=manifest,
        task_manifest=task_manifest,
        storage_manifest=storage_manifest,
        controls=controls,
        label_config=label_config,
        task_imports=tuple(imports),
    )


def compare_manifests(
    expected: ProjectManifest | InstanceBootstrapManifest | Mapping[str, Any],
    observed: ProjectManifest | InstanceBootstrapManifest | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return deterministic field paths that differ, including missing/extra fields."""

    expected_payload = _manifest_payload(expected)
    observed_payload = _manifest_payload(observed)
    differences: list[str] = []
    _collect_differences(
        expected_payload, observed_payload, path="", output=differences
    )
    return tuple(differences)


def assert_manifest_matches(
    expected: ProjectManifest | InstanceBootstrapManifest | Mapping[str, Any],
    observed: ProjectManifest | InstanceBootstrapManifest | Mapping[str, Any],
) -> None:
    differences = compare_manifests(expected, observed)
    if differences:
        raise ManifestDriftError(differences)


def _manifest_payload(
    value: ProjectManifest | InstanceBootstrapManifest | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return value.to_dict()


def plan_instance_bootstrap(
    desired: Mapping[Split | str, SplitProjectPlan],
    adapter: RefinementProjectAdapter,
) -> InstanceBootstrapPlan:
    """Plan create/reconcile/reuse from live source identities, failing closed on drift."""

    desired_normalized = {_split(key): value for key, value in desired.items()}
    required = {Split.TRAIN, Split.VAL}
    if set(desired_normalized) != required:
        raise ProjectContractError(
            "bootstrap requires exactly train and val desired projects"
        )
    for split_value in (Split.TRAIN, Split.VAL):
        _validate_split_project_plan(desired_normalized[split_value])
    actions: list[PlannedProjectBootstrap] = []
    repo_root = Path(desired_normalized[Split.TRAIN].manifest.source_path).parents[3]
    project_identities = {
        project.manifest.project_identity for project in desired_normalized.values()
    }
    storage_identities = {
        project.storage_manifest.storage_identity
        for project in desired_normalized.values()
    }
    if len(project_identities) != 2 or len(storage_identities) != 2:
        raise ProjectContractError(
            "train and val require distinct project and storage namespaces"
        )
    manifest = build_instance_bootstrap_manifest(desired_normalized)
    attest_manifest = getattr(adapter, "attest_bootstrap_manifest", None)
    if not callable(attest_manifest):
        raise ProjectContractError(
            "adapter must attest the parent-owned bootstrap manifest"
        )
    observed_manifest = attest_manifest()
    live_projects = {
        split_value: adapter.attest_project(split_value)
        for split_value in (Split.TRAIN, Split.VAL)
    }
    any_live_project = any(project is not None for project in live_projects.values())
    if observed_manifest is not None:
        assert_manifest_matches(manifest, observed_manifest)
    elif any_live_project:
        raise ManifestDriftError(("bootstrap_manifest (missing)",))

    live_project_ids: set[tuple[type[Any], Any]] = set()
    for split_value in (Split.TRAIN, Split.VAL):
        project = desired_normalized[split_value]
        if project.split is not split_value:
            raise ProjectContractError(
                "desired project is stored under the wrong split key"
            )
        expected_source_path = str(SOURCE_CONTRACTS[split_value].path(repo_root))
        if project.manifest.source_path != expected_source_path:
            raise ProjectContractError(
                f"desired {split_value.value} project is outside the shared bootstrap root"
            )
        live_attestation = live_projects[split_value]
        if live_attestation is None:
            action = BootstrapAction.CREATE
            attestation_fingerprint = None
            missing_task_imports = project.task_imports
            reused_task_identities: tuple[TaskIdentity, ...] = ()
        else:
            if not isinstance(live_attestation, LiveProjectAttestation):
                raise ProjectContractError(
                    "adapter must return a live project attestation, not a saved manifest"
                )
            project_id_key = (
                type(live_attestation.project_id),
                live_attestation.project_id,
            )
            if project_id_key in live_project_ids:
                raise ManifestDriftError(("live.project_id (cross-split duplicate)",))
            live_project_ids.add(project_id_key)
            reused_task_identities = _assert_live_project_matches(
                project, live_attestation
            )
            reused_set = set(reused_task_identities)
            missing_task_imports = tuple(
                task_import
                for entry, task_import in zip(
                    project.task_manifest.entries,
                    project.task_imports,
                    strict=True,
                )
                if entry.identity not in reused_set
            )
            action = (
                BootstrapAction.REUSE
                if not missing_task_imports
                else BootstrapAction.RECONCILE
            )
            attestation_fingerprint = live_attestation.fingerprint
        planned = PlannedProjectBootstrap(
            action=action,
            project=project,
            missing_task_imports=missing_task_imports,
            reused_task_identities=reused_task_identities,
            live_attestation_fingerprint=attestation_fingerprint,
        )
        _validate_planned_action_imports(planned)
        actions.append(planned)
    return InstanceBootstrapPlan(
        runtime_layout=RuntimeLayout.for_repo(repo_root),
        manifest=manifest,
        projects=tuple(actions),
    )


def _assert_live_project_matches(
    desired: SplitProjectPlan,
    observed: LiveProjectAttestation,
) -> tuple[TaskIdentity, ...]:
    """Validate all observed tasks and return reusable identities in source order."""

    if observed.split is not desired.split:
        raise ManifestDriftError(("live.split",))
    _require_live_identifier(observed.project_id, "live.project_id")
    if observed.project_identity != desired.manifest.project_identity:
        raise ManifestDriftError(("live.project_identity",))
    assert_manifest_matches(desired.manifest, observed.saved_manifest)

    mismatches: list[str] = []
    if observed.vendor_revision != desired.manifest.vendor_revision:
        mismatches.append("live.vendor_revision")
    observed_label_config_fingerprint = fingerprint_json(
        {"label_config_xml": observed.label_config}
    )
    if observed_label_config_fingerprint != desired.manifest.label_config_fingerprint:
        mismatches.append("live.label_config_fingerprint")
    observed_controls_fingerprint = fingerprint_json(observed.controls.to_dict())
    if (
        observed_controls_fingerprint
        != desired.manifest.authoritative_annotation_policy_fingerprint
    ):
        mismatches.append("live.project_capabilities")
    if observed.storage_manifest.fingerprint != desired.storage_manifest.fingerprint:
        mismatches.append("live.storage_manifest_fingerprint")
    if not observed.managed_link_is_symlink:
        mismatches.append("live.managed_link_is_symlink")
    expected_link_target = str(
        Path(desired.storage_manifest.managed_link_target).resolve(strict=False)
    )
    observed_link_target = str(
        Path(observed.managed_link_resolved_target).resolve(strict=False)
    )
    if observed_link_target != expected_link_target:
        mismatches.append("live.managed_link_resolved_target")

    expected_entries = {
        entry.identity: entry for entry in desired.task_manifest.entries
    }

    seen_identities: set[TaskIdentity] = set()
    seen_task_ids: set[tuple[type[Any], Any]] = set()
    seen_annotation_ids: set[tuple[type[Any], Any]] = set()
    for task in observed.tasks:
        prefix = f"live.tasks[{task.identity.key}]"
        if task.identity in seen_identities:
            mismatches.append(prefix + ".identity (duplicate)")
        seen_identities.add(task.identity)
        expected_entry = expected_entries.get(task.identity)
        if expected_entry is None:
            mismatches.append(prefix + ".identity (unexpected)")
        task_id = _live_identifier_key(task.task_id, prefix + ".task_id", mismatches)
        if task_id is not None:
            if task_id in seen_task_ids:
                mismatches.append(prefix + ".task_id (duplicate)")
            seen_task_ids.add(task_id)
        if task.annotation_count != 1:
            mismatches.append(prefix + ".annotation_count")
        if task.alternate_annotation_count != 0:
            mismatches.append(prefix + ".alternate_annotation_count")
        if task.prediction_count != 0:
            mismatches.append(prefix + ".prediction_count")
        if task.authoritative_annotation_ground_truth is not False:
            mismatches.append(prefix + ".authoritative_annotation_ground_truth")
        annotation_id = _live_identifier_key(
            task.authoritative_annotation_id,
            prefix + ".authoritative_annotation_id",
            mismatches,
        )
        if annotation_id is not None:
            if annotation_id in seen_annotation_ids:
                mismatches.append(prefix + ".authoritative_annotation_id (duplicate)")
            seen_annotation_ids.add(annotation_id)
        _live_identifier_key(
            task.authoritative_annotation_revision,
            prefix + ".authoritative_annotation_revision",
            mismatches,
        )
        if expected_entry is not None:
            if task.source_line != expected_entry.source_line:
                mismatches.append(prefix + ".source_line")
            if task.image_locator != expected_entry.label_studio_image_locator:
                mismatches.append(prefix + ".image_locator")
            if task.task_data_fingerprint != expected_entry.task_data_fingerprint:
                mismatches.append(prefix + ".task_data_fingerprint")
            if (
                task.authoritative_annotation_fingerprint
                != expected_entry.authoritative_annotation_fingerprint
            ):
                mismatches.append(prefix + ".authoritative_annotation_fingerprint")
    if mismatches:
        raise ManifestDriftError(tuple(dict.fromkeys(mismatches)))
    return tuple(
        entry.identity
        for entry in desired.task_manifest.entries
        if entry.identity in seen_identities
    )


def _build_instance_bootstrap_manifest(
    desired: Mapping[Split, SplitProjectPlan],
    *,
    layout: RuntimeLayout,
) -> InstanceBootstrapManifest:
    return InstanceBootstrapManifest(
        adapter_version=ADAPTER_VERSION,
        dataset_name=DATASET_NAME,
        runtime_root=str(layout.root),
        label_studio_state=str(layout.label_studio_state),
        local_files_document_root=str(layout.image_root),
        project_manifest_fingerprints=MappingProxyType(
            {
                split.value: desired[split].manifest.fingerprint
                for split in (Split.TRAIN, Split.VAL)
            }
        ),
        task_manifest_fingerprints=MappingProxyType(
            {
                split.value: desired[split].task_manifest.fingerprint
                for split in (Split.TRAIN, Split.VAL)
            }
        ),
        storage_manifest_fingerprints=MappingProxyType(
            {
                split.value: desired[split].storage_manifest.fingerprint
                for split in (Split.TRAIN, Split.VAL)
            }
        ),
    )


def build_instance_bootstrap_manifest(
    desired: Mapping[Split | str, SplitProjectPlan],
) -> InstanceBootstrapManifest:
    """Build the exact one-instance/two-split manifest without touching live state."""

    normalized = {_split(key): value for key, value in desired.items()}
    if set(normalized) != {Split.TRAIN, Split.VAL}:
        raise ProjectContractError(
            "bootstrap manifest requires exactly train and val projects"
        )
    repo_root = Path(normalized[Split.TRAIN].manifest.source_path).parents[3]
    for split_value in (Split.TRAIN, Split.VAL):
        project = normalized[split_value]
        _validate_split_project_plan(project)
        if project.split is not split_value:
            raise ProjectContractError(
                "bootstrap manifest project is stored under wrong split"
            )
        expected_source = str(SOURCE_CONTRACTS[split_value].path(repo_root))
        if project.manifest.source_path != expected_source:
            raise ProjectContractError(
                "bootstrap manifest projects do not share one repo root"
            )
    return _build_instance_bootstrap_manifest(
        normalized,
        layout=RuntimeLayout.for_repo(repo_root),
    )


def _validate_split_project_plan(project: SplitProjectPlan) -> None:
    """Fail before live adapter calls if a stored pure plan is internally inconsistent."""

    if not isinstance(project.task_imports, tuple):
        raise ProjectContractError("task imports must be an immutable tuple")
    if project.task_manifest.split is not project.split:
        raise ProjectContractError("task manifest split does not match project split")
    if project.storage_manifest.split is not project.split:
        raise ProjectContractError(
            "storage manifest split does not match project split"
        )
    if project.manifest.split is not project.split:
        raise ProjectContractError(
            "project manifest split does not match project split"
        )
    if project.source_inspection.contract.split is not project.split:
        raise ProjectContractError(
            "source inspection split does not match project split"
        )
    if len(project.task_imports) != len(project.task_manifest.entries):
        raise ProjectContractError("task import count does not match task manifest")
    if len(project.task_manifest.entries) != project.source_inspection.row_count:
        raise ProjectContractError(
            "task manifest count does not match source inspection"
        )
    if project.manifest.source_sha256 != project.source_inspection.sha256:
        raise ProjectContractError(
            "project source hash does not match source inspection"
        )
    if project.manifest.source_row_count != project.source_inspection.row_count:
        raise ProjectContractError(
            "project source row count does not match source inspection"
        )
    if project.manifest.source_box_count != project.source_inspection.box_count:
        raise ProjectContractError(
            "project source box count does not match source inspection"
        )
    if (
        project.manifest.source_inspection_fingerprint
        != project.source_inspection.fingerprint
    ):
        raise ProjectContractError("source inspection fingerprint drift")
    if project.manifest.task_manifest_fingerprint != project.task_manifest.fingerprint:
        raise ProjectContractError("task manifest fingerprint drift")
    if (
        project.manifest.storage_manifest_fingerprint
        != project.storage_manifest.fingerprint
    ):
        raise ProjectContractError("storage manifest fingerprint drift")
    if project.manifest.adapter_version != ADAPTER_VERSION:
        raise ProjectContractError("adapter version drift")
    if project.manifest.category_registry_fingerprint != default_registry().fingerprint:
        raise ProjectContractError("category registry fingerprint drift")
    if project.manifest.label_config_fingerprint != fingerprint_json(
        {"label_config_xml": project.label_config}
    ):
        raise ProjectContractError("label config fingerprint drift")
    if project.manifest.authoritative_annotation_policy_fingerprint != fingerprint_json(
        project.controls.to_dict()
    ):
        raise ProjectContractError("authoritative annotation policy fingerprint drift")

    seen_identities: set[TaskIdentity] = set()
    seen_lines: set[int] = set()
    for task_import, entry in zip(
        project.task_imports,
        project.task_manifest.entries,
        strict=True,
    ):
        if entry.identity in seen_identities:
            raise ProjectContractError(f"duplicate task identity {entry.identity.key}")
        if entry.source_line in seen_lines:
            raise ProjectContractError(f"duplicate source line {entry.source_line}")
        seen_identities.add(entry.identity)
        seen_lines.add(entry.source_line)
        _validated_frozen_task_import(
            task_import,
            entry=entry,
            split=project.split,
        )
    expected_lines = set(range(1, len(project.task_manifest.entries) + 1))
    if seen_lines != expected_lines:
        raise ProjectContractError(
            "task manifest source lines are not complete and ordered"
        )


def _validated_frozen_task_import(
    task_import: FrozenTaskImport,
    *,
    entry: TaskManifestEntry,
    split: Split,
) -> dict[str, Any]:
    if not isinstance(task_import, FrozenTaskImport):
        raise ProjectContractError("task import is not a frozen canonical payload")
    payload = task_import._thaw()
    validate_authoritative_task_payload(payload)
    data = payload["data"]
    identity = TaskIdentity(_split(data["split"]), data["image_id"])
    if identity != entry.identity or identity.split is not split:
        raise ProjectContractError("task import identity does not match task manifest")
    if data["source_line"] != entry.source_line:
        raise ProjectContractError(
            "task import source line does not match task manifest"
        )
    if data["image"] != entry.label_studio_image_locator:
        raise ProjectContractError(
            "task import image locator does not match task manifest"
        )
    if fingerprint_json(data) != entry.task_data_fingerprint:
        raise ProjectContractError("task data fingerprint does not match task manifest")
    if (
        fingerprint_json(payload["annotations"][0])
        != entry.authoritative_annotation_fingerprint
    ):
        raise ProjectContractError(
            "authoritative annotation fingerprint does not match task manifest"
        )
    return payload


def _validate_planned_action_imports(action: PlannedProjectBootstrap) -> None:
    if not isinstance(action.missing_task_imports, tuple):
        raise ProjectContractError("planned task imports must be an immutable tuple")
    if not isinstance(action.reused_task_identities, tuple):
        raise ProjectContractError("reused task identities must be an immutable tuple")
    entries = {entry.identity: entry for entry in action.project.task_manifest.entries}
    stored_imports = {
        entry.identity: task_import
        for entry, task_import in zip(
            action.project.task_manifest.entries,
            action.project.task_imports,
            strict=True,
        )
    }
    missing_identities: list[TaskIdentity] = []
    for task_import in action.missing_task_imports:
        if not isinstance(task_import, FrozenTaskImport):
            raise ProjectContractError("planned task import is not frozen")
        payload = task_import._thaw()
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ProjectContractError("planned task import data is invalid")
        identity = TaskIdentity(_split(data.get("split")), data.get("image_id"))
        entry = entries.get(identity)
        if entry is None:
            raise ProjectContractError(
                "planned task import is absent from task manifest"
            )
        _validated_frozen_task_import(
            task_import, entry=entry, split=action.project.split
        )
        if task_import != stored_imports[identity]:
            raise ProjectContractError(
                "planned task import differs from stored project plan"
            )
        missing_identities.append(identity)
    if len(set(missing_identities)) != len(missing_identities):
        raise ProjectContractError("planned task imports contain duplicate identities")

    reused = action.reused_task_identities
    if len(set(reused)) != len(reused) or any(
        identity not in entries for identity in reused
    ):
        raise ProjectContractError("reused task identities do not match task manifest")
    missing_set = set(missing_identities)
    reused_set = set(reused)
    if missing_set & reused_set or missing_set | reused_set != set(entries):
        raise ProjectContractError(
            "planned create/reuse identities do not partition tasks"
        )
    if action.action is BootstrapAction.CREATE and reused:
        raise ProjectContractError("create action cannot reuse existing tasks")
    if action.action is BootstrapAction.REUSE and action.missing_task_imports:
        raise ProjectContractError("reuse action cannot emit task imports")
    if (
        action.action is BootstrapAction.RECONCILE
        and action.live_attestation_fingerprint is None
    ):
        raise ProjectContractError("reconcile action requires an observed live project")


def _task_imports_for_adapter_send(
    action: PlannedProjectBootstrap,
) -> tuple[dict[str, Any], ...]:
    """Explicit mutable-JSON boundary for a future authenticated live adapter."""

    _validate_split_project_plan(action.project)
    _validate_planned_action_imports(action)
    entries = {entry.identity: entry for entry in action.project.task_manifest.entries}
    payloads: list[dict[str, Any]] = []
    for task_import in action.missing_task_imports:
        preview = task_import._thaw()
        data = preview["data"]
        identity = TaskIdentity(_split(data["split"]), data["image_id"])
        payloads.append(
            _validated_frozen_task_import(
                task_import,
                entry=entries[identity],
                split=action.project.split,
            )
        )
    return tuple(payloads)


def _require_live_identifier(value: LiveIdentifier, field: str) -> None:
    mismatches: list[str] = []
    _live_identifier_key(value, field, mismatches)
    if mismatches:
        raise ManifestDriftError(tuple(mismatches))


def _live_identifier_key(
    value: LiveIdentifier,
    field: str,
    mismatches: list[str],
) -> tuple[type[Any], Any] | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        mismatches.append(field)
        return None
    if isinstance(value, int) and value < 0:
        mismatches.append(field)
        return None
    if isinstance(value, str) and not value:
        mismatches.append(field)
        return None
    return type(value), value


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProjectContractError("fingerprint payload is not canonical JSON") from exc


def _decode_task_import(encoded: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectContractError("frozen task import is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProjectContractError("frozen task import must decode to an object")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def fingerprint_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_source_object(
    obj: Any,
    *,
    index: int,
    registry: Coco80RegistryProtocol | None,
) -> None:
    if not isinstance(obj, Mapping):
        raise ProjectContractError(f"source object {index} is not an object")
    _require_exact_keys(obj, SOURCE_OBJECT_FIELDS, context=f"source object {index}")
    bbox = obj["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProjectContractError(
            f"source object {index} bbox_2d must have four integers"
        )
    x1, y1, x2, y2 = (
        _require_int(value, f"source object {index} bbox_2d", minimum=0, maximum=999)
        for value in bbox
    )
    if x1 >= x2 or y1 >= y2:
        raise ProjectContractError(
            f"source object {index} bbox_2d must have positive area"
        )
    name = _require_string(obj["category_name"], f"source object {index} category_name")
    desc = _require_string(obj["desc"], f"source object {index} desc")
    if desc != name:
        raise ProjectContractError(f"source object {index} desc/category_name mismatch")
    category_id = _require_int(
        obj["category_id"], f"source object {index} category_id", minimum=1
    )
    _require_int(obj["coco_ann_id"], f"source object {index} coco_ann_id", minimum=1)
    if registry is not None:
        try:
            registry.validate(name, category_id)
        except Exception as exc:
            raise ProjectContractError(
                f"source object {index} is not a canonical COCO-80 category"
            ) from exc


def _assert_source_image_exists(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    split: Split,
    line_number: int,
) -> None:
    """Attest the shared source image without opening or copying its bytes."""

    layout = RuntimeLayout.for_repo(repo_root)
    image_path = resolve_working_image(
        str(row["file_name"]),
        layout=layout,
        split=split,
    )
    if not image_path.is_file():
        raise ProjectContractError(
            f"source image missing at line {line_number}: {row['file_name']}"
        )


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProjectContractError(
            f"{context} schema drift: missing={missing}, extra={extra}"
        )


def _require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectContractError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ProjectContractError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ProjectContractError(f"{field} must be at most {maximum}")
    return value


def _require_finite_number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectContractError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise ProjectContractError(f"{field} must be between {minimum} and {maximum}")
    return numeric


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectContractError(f"{field} must be a non-empty string")
    return value


def _safe_relative(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ProjectContractError(f"{field} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ProjectContractError(f"{field} must be a normalized relative path")
    return path


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _collect_differences(
    expected: Any,
    observed: Any,
    *,
    path: str,
    output: list[str],
) -> None:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                output.append(child + " (unexpected)")
            elif key not in observed:
                output.append(child + " (missing)")
            else:
                _collect_differences(
                    expected[key], observed[key], path=child, output=output
                )
        return
    if expected != observed:
        output.append(path or "<root>")
