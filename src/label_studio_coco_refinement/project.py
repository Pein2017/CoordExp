"""Pure contracts and bootstrap planning for COCO Label Studio refinement.

This module deliberately does not import Label Studio or perform project, file, or
symlink mutations.  It validates the two approved source splits and produces a
deterministic plan for a separate adapter to apply.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import quote


ADAPTER_VERSION = "label-studio-coco-refinement-v1"
DATASET_NAME = "rescale_32_1024_bbox_len12000"
SOURCE_DIRECTORY = PurePosixPath("public_data/coco/rescale_32_1024_bbox_len12000")
SHARED_IMAGE_ROOT = PurePosixPath("public_data/coco/rescale_32_1024_bbox/images")
RUNTIME_ROOT = PurePosixPath(
    "outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
)
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
        raise ProjectContractError(f"unsupported split {value!r}; expected 'train' or 'val'") from exc


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
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, int) or self.image_id < 0:
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
            raise ProjectContractError(f"managed image path is not a symlink: {self.link_path}")
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


def managed_image_link_plan(layout: RuntimeLayout, split: Split | str) -> ManagedImageLinkPlan:
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
    resolved = (root / Path(contract.image_subdirectory) / Path(inside_root)).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ProjectContractError("working image locator escapes the shared image root")
    return resolved


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
        raise ProjectContractError("file_name must identify one COCO JPEG below the split root")
    if not image_leaf.stem.isdigit() or int(image_leaf.stem) != image_id:
        raise ProjectContractError("file_name stem must equal image_id")

    images = row["images"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ProjectContractError("images must contain exactly one string locator")
    expected_source_locator = (PurePosixPath("..") / "rescale_32_1024_bbox" / file_path).as_posix()
    if images[0] != expected_source_locator:
        raise ProjectContractError(
            f"source image locator drift: expected {expected_source_locator!r}, got {images[0]!r}"
        )

    metadata = row["metadata"]
    if not isinstance(metadata, Mapping) or dict(metadata) != {
        "source": "coco2017",
        "split": split_value.value,
    }:
        raise ProjectContractError("metadata must be exactly the selected COCO source and split")

    objects = row["objects"]
    if not isinstance(objects, list) or not objects:
        raise ProjectContractError("source objects must be a non-empty list")
    seen_annotation_ids: set[int] = set()
    for index, obj in enumerate(objects):
        _validate_source_object(obj, index=index, registry=registry)
        annotation_id = int(obj["coco_ann_id"])
        if annotation_id in seen_annotation_ids:
            raise ProjectContractError(f"duplicate coco_ann_id {annotation_id} in source row")
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
        field for field in IMMUTABLE_ROW_FIELDS if working_row.get(field) != source_row.get(field)
    ]
    if mismatches:
        raise ProjectContractError("immutable row field drift: " + ", ".join(mismatches))
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
                raise ProjectContractError(f"invalid JSON at source line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ProjectContractError(f"source line {line_number} is not an object")
            identity = validate_source_row(row, split=contract.split, registry=registry)
            if identity in seen:
                raise ProjectContractError(f"duplicate task identity {identity.key}")
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
class TaskManifestEntry:
    identity: TaskIdentity
    source_line: int
    source_image_locator: str
    working_image_locator: str
    authoritative_annotation_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "source_line": self.source_line,
            "source_image_locator": self.source_image_locator,
            "working_image_locator": self.working_image_locator,
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
    managed_link: str
    managed_link_target: str

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "project_identity": self.project_identity,
            "storage_identity": self.storage_identity,
            "document_root": self.document_root,
            "storage_subdirectory": self.storage_subdirectory,
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
    task_imports: tuple[dict[str, Any], ...]


class BootstrapAction(str, Enum):
    CREATE = "create"
    REUSE = "reuse"


@dataclass(frozen=True)
class PlannedProjectBootstrap:
    action: BootstrapAction
    project: SplitProjectPlan
    live_attestation_fingerprint: str | None = None


@dataclass(frozen=True)
class InstanceBootstrapPlan:
    runtime_layout: RuntimeLayout
    projects: tuple[PlannedProjectBootstrap, ...]


LiveIdentifier = int | str


@dataclass(frozen=True)
class LiveTaskAttestation:
    """Current Label Studio task/annotation counts and stable identities."""

    identity: TaskIdentity
    task_id: LiveIdentifier
    annotation_count: int
    authoritative_annotation_id: LiveIdentifier
    authoritative_annotation_revision: LiveIdentifier
    alternate_annotation_count: int
    prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "task_id": self.task_id,
            "annotation_count": self.annotation_count,
            "authoritative_annotation_id": self.authoritative_annotation_id,
            "authoritative_annotation_revision": self.authoritative_annotation_revision,
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
    payload = {
        "data": {
            "image": "/data/local-files/?d="
            + quote(relative_to_document_root.as_posix(), safe="/"),
            "coordexp_task_key": identity.key,
            "split": split_value.value,
            "image_id": identity.image_id,
            "source_line": source_line,
        },
        "annotations": [{"result": results}],
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
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ProjectContractError("task must contain exactly one authoritative annotation")
    annotation = annotations[0]
    if not isinstance(annotation, Mapping) or set(annotation) != {"result"}:
        raise ProjectContractError("authoritative annotation has an unsupported shape")
    results = annotation["result"]
    if not isinstance(results, list) or not results:
        raise ProjectContractError("authoritative annotation must contain source rectangles")
    registry = registry or default_registry()
    for result in results:
        if not isinstance(result, Mapping) or result.get("type") != "rectanglelabels":
            raise ProjectContractError("only rectanglelabels results are supported")
        if result.get("from_name") != "bbox" or result.get("to_name") != "image":
            raise ProjectContractError("rectangle result is not bound to bbox/image")
        if result.get("image_rotation") != 0:
            raise ProjectContractError("image_rotation must be zero")
        value = result.get("value")
        if not isinstance(value, Mapping) or value.get("rotation") != 0:
            raise ProjectContractError("bbox rotation must be zero")
        labels = value.get("rectanglelabels")
        if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in registry.names:
            raise ProjectContractError("bbox must have exactly one canonical COCO-80 label")


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
        raise ProjectContractError("label_config_fingerprint does not match label_config")
    layout = RuntimeLayout.for_repo(repo_root)
    project_identity = f"coco-refinement:{DATASET_NAME}:{split_value.value}"
    imports: list[dict[str, Any]] = []
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
        imports.append(task)
        entries.append(
            TaskManifestEntry(
                identity=identity,
                source_line=line_number,
                source_image_locator=row["images"][0],
                working_image_locator=row["file_name"],
                authoritative_annotation_fingerprint=fingerprint_json(task["annotations"]),
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
    expected: ProjectManifest | Mapping[str, Any],
    observed: ProjectManifest | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return deterministic field paths that differ, including missing/extra fields."""

    expected_payload = expected.to_dict() if isinstance(expected, ProjectManifest) else dict(expected)
    observed_payload = observed.to_dict() if isinstance(observed, ProjectManifest) else dict(observed)
    differences: list[str] = []
    _collect_differences(expected_payload, observed_payload, path="", output=differences)
    return tuple(differences)


def assert_manifest_matches(
    expected: ProjectManifest | Mapping[str, Any],
    observed: ProjectManifest | Mapping[str, Any],
) -> None:
    differences = compare_manifests(expected, observed)
    if differences:
        raise ManifestDriftError(differences)


def plan_instance_bootstrap(
    desired: Mapping[Split | str, SplitProjectPlan],
    adapter: RefinementProjectAdapter,
) -> InstanceBootstrapPlan:
    """Plan create/reuse from current adapter attestations, failing closed on drift."""

    desired_normalized = {_split(key): value for key, value in desired.items()}
    required = {Split.TRAIN, Split.VAL}
    if set(desired_normalized) != required:
        raise ProjectContractError("bootstrap requires exactly train and val desired projects")
    actions: list[PlannedProjectBootstrap] = []
    repo_root = Path(desired_normalized[Split.TRAIN].manifest.source_path).parents[3]
    for split_value in (Split.TRAIN, Split.VAL):
        project = desired_normalized[split_value]
        if project.split is not split_value:
            raise ProjectContractError("desired project is stored under the wrong split key")
        expected_source_path = str(SOURCE_CONTRACTS[split_value].path(repo_root))
        if project.manifest.source_path != expected_source_path:
            raise ProjectContractError(
                f"desired {split_value.value} project is outside the shared bootstrap root"
            )
        live_attestation = adapter.attest_project(split_value)
        if live_attestation is None:
            action = BootstrapAction.CREATE
            attestation_fingerprint = None
        else:
            if not isinstance(live_attestation, LiveProjectAttestation):
                raise ProjectContractError(
                    "adapter must return a live project attestation, not a saved manifest"
                )
            _assert_live_project_matches(project, live_attestation)
            action = BootstrapAction.REUSE
            attestation_fingerprint = live_attestation.fingerprint
        actions.append(
            PlannedProjectBootstrap(
                action=action,
                project=project,
                live_attestation_fingerprint=attestation_fingerprint,
            )
        )
    return InstanceBootstrapPlan(
        runtime_layout=RuntimeLayout.for_repo(repo_root),
        projects=tuple(actions),
    )


def _assert_live_project_matches(
    desired: SplitProjectPlan,
    observed: LiveProjectAttestation,
) -> None:
    """Recompute every reuse-critical identity from the live-state receipt."""

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
    expected_link_target = str(Path(desired.storage_manifest.managed_link_target).resolve(strict=False))
    observed_link_target = str(Path(observed.managed_link_resolved_target).resolve(strict=False))
    if observed_link_target != expected_link_target:
        mismatches.append("live.managed_link_resolved_target")

    expected_identity_fingerprint = desired.task_manifest.identity_fingerprint
    observed_identity_keys = sorted(task.identity.key for task in observed.tasks)
    if fingerprint_json(observed_identity_keys) != expected_identity_fingerprint:
        mismatches.append("live.task_identity_fingerprint")
    if len(observed.tasks) != len(desired.task_manifest.entries):
        mismatches.append("live.task_count")

    seen_identities: set[TaskIdentity] = set()
    seen_task_ids: set[tuple[type[Any], Any]] = set()
    seen_annotation_ids: set[tuple[type[Any], Any]] = set()
    for task in observed.tasks:
        prefix = f"live.tasks[{task.identity.key}]"
        if task.identity in seen_identities:
            mismatches.append(prefix + ".identity (duplicate)")
        seen_identities.add(task.identity)
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
    if mismatches:
        raise ManifestDriftError(tuple(dict.fromkeys(mismatches)))


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


def fingerprint_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProjectContractError("fingerprint payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


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
        raise ProjectContractError(f"source object {index} bbox_2d must have four integers")
    x1, y1, x2, y2 = (
        _require_int(value, f"source object {index} bbox_2d", minimum=0, maximum=999)
        for value in bbox
    )
    if x1 >= x2 or y1 >= y2:
        raise ProjectContractError(f"source object {index} bbox_2d must have positive area")
    name = _require_string(obj["category_name"], f"source object {index} category_name")
    desc = _require_string(obj["desc"], f"source object {index} desc")
    if desc != name:
        raise ProjectContractError(f"source object {index} desc/category_name mismatch")
    category_id = _require_int(obj["category_id"], f"source object {index} category_id", minimum=1)
    _require_int(obj["coco_ann_id"], f"source object {index} coco_ann_id", minimum=1)
    if registry is not None:
        try:
            registry.validate(name, category_id)
        except Exception as exc:
            raise ProjectContractError(
                f"source object {index} is not a canonical COCO-80 category"
            ) from exc


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, context: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProjectContractError(f"{context} schema drift: missing={missing}, extra={extra}")


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
                _collect_differences(expected[key], observed[key], path=child, output=output)
        return
    if expected != observed:
        output.append(path or "<root>")
