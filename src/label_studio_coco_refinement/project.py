"""Pure contracts and bootstrap planning for COCO Label Studio refinement.

This module deliberately does not import Label Studio or mutate vendor state.  It
validates the two approved source splits, atomically publishes immutable task-index
sidecars below the ignored runtime root, and produces a deterministic plan for a
separate adapter to apply.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Callable, Iterator, Mapping, Protocol, Sequence
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
TASK_INDEX_SCHEMA_VERSION = 2
TASK_INDEX_DIRECTORY_NAME = "bootstrap-task-index"
TASK_INDEX_IDENTITY_SORT_CHUNK_SIZE = 2_048
TASK_INDEX_SORT_FAN_IN = 8
TASK_INDEX_MAX_SORT_LEVELS = 64
CANONICAL_BBOX_CONVERTER_SEMANTICS = (
    '{"contract":"coordexp.norm1000_xyxy_int_to_label_studio_percent_xywh",'
    '"version":1}'
)
# SHA-256 of the canonical JSON semantic contract.  This is intentionally a
# stable data-contract marker, never a Python function name, repr, or identity.
CANONICAL_BBOX_CONVERTER_FINGERPRINT = (
    "def0701da8ff3e6a22ef6dc230424b58ed35ed56c4a91f53529f4e9624dabef9"
)
UNFINGERPRINTED_CUSTOM_BBOX_CONVERTER_FINGERPRINT = (
    "0d9372be053631d6cb31438cfdcc5d1ad58b80b29d9c68a8c26cb5dba90e573d"
)


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


@dataclass(frozen=True)
class _BboxConverterContract:
    converter: BboxToLabelStudio
    fingerprint: str
    reuse_anchor: bool
    publish_anchor: bool


class RefinementProjectAdapter(Protocol):
    """Read-only seam for a Label Studio-specific live-state attestor.

    Implementations must derive this receipt from current vendor/runtime state,
    not merely return the CoordExp-side saved project manifest.  Project
    planning remains pure and does not prescribe an HTTP framework.
    """

    def attest_project(
        self, desired: "SplitProjectPlan"
    ) -> "LiveProjectAttestation | None": ...

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


def _resolve_bbox_converter_contract(
    bbox_converter: BboxToLabelStudio | None,
    converter_fingerprint: str | None,
) -> _BboxConverterContract:
    canonical = default_bbox_converter()
    converter = canonical if bbox_converter is None else bbox_converter
    if not callable(converter):
        raise ProjectContractError("bbox_converter must be callable")
    if converter is canonical:
        if converter_fingerprint is not None:
            observed = _require_sha256(
                converter_fingerprint,
                "converter_fingerprint",
            )
            if observed != CANONICAL_BBOX_CONVERTER_FINGERPRINT:
                raise ProjectContractError(
                    "default bbox converter fingerprint does not match its "
                    "canonical semantics"
                )
        return _BboxConverterContract(
            converter=converter,
            fingerprint=CANONICAL_BBOX_CONVERTER_FINGERPRINT,
            reuse_anchor=True,
            publish_anchor=True,
        )
    if converter_fingerprint is None:
        return _BboxConverterContract(
            converter=converter,
            fingerprint=UNFINGERPRINTED_CUSTOM_BBOX_CONVERTER_FINGERPRINT,
            reuse_anchor=False,
            publish_anchor=False,
        )
    observed = _require_sha256(converter_fingerprint, "converter_fingerprint")
    if observed in {
        CANONICAL_BBOX_CONVERTER_FINGERPRINT,
        UNFINGERPRINTED_CUSTOM_BBOX_CONVERTER_FINGERPRINT,
    }:
        raise ProjectContractError(
            "custom bbox converter fingerprint uses a reserved semantics marker"
        )
    return _BboxConverterContract(
        converter=converter,
        fingerprint=observed,
        reuse_anchor=False,
        publish_anchor=True,
    )


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
    ).inspection


SourceRowConsumer = Callable[[int, Mapping[str, Any], TaskIdentity], None]


@dataclass(frozen=True)
class _SourceStreamResult:
    inspection: SourceInspection
    sorted_identity_fingerprint: str


class _IdentityFingerprintBuilder:
    """Bounded-memory external sorter for the legacy sorted identity digest."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._chunk: list[str] = []
        self._levels: dict[int, list[Path]] = {}
        self._count = 0

    def add(self, key: str) -> None:
        self._chunk.append(key)
        self._count += 1
        if len(self._chunk) >= TASK_INDEX_IDENTITY_SORT_CHUNK_SIZE:
            self._flush_run()

    def finish(self) -> tuple[int, str]:
        self._flush_run()
        runs = [path for paths in self._levels.values() for path in paths]
        self._levels.clear()
        while len(runs) > TASK_INDEX_SORT_FAN_IN:
            compacted: list[Path] = []
            for offset in range(0, len(runs), TASK_INDEX_SORT_FAN_IN):
                group = runs[offset : offset + TASK_INDEX_SORT_FAN_IN]
                if len(group) == 1:
                    compacted.extend(group)
                    continue
                merged = self._merge_runs(group)
                self._unlink_runs(group)
                compacted.append(merged)
            runs = compacted
        digest = hashlib.sha256()
        digest.update(b"[")
        previous: str | None = None
        first = True
        handles: list[BinaryIO] = []
        try:
            handles = [path.open("rb") for path in runs]
            streams = (
                (encoded.rstrip(b"\n").decode("utf-8") for encoded in handle)
                for handle in handles
            )
            for key in heapq.merge(*streams):
                if key == previous:
                    raise ProjectContractError(f"duplicate task identity {key}")
                previous = key
                if not first:
                    digest.update(b",")
                digest.update(_canonical_json_bytes(key))
                first = False
        finally:
            for handle in handles:
                handle.close()
            self._unlink_runs(runs)
        digest.update(b"]")
        return self._count, digest.hexdigest()

    def _flush_run(self) -> None:
        if not self._chunk:
            return
        self._chunk.sort()
        fd, temporary = tempfile.mkstemp(
            dir=self._directory,
            prefix="identity-run-",
            suffix=".txt",
        )
        path = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as handle:
                for key in self._chunk:
                    handle.write(key.encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        self._chunk.clear()
        self._add_run(path, level=0)

    def _add_run(self, path: Path, *, level: int) -> None:
        if level >= TASK_INDEX_MAX_SORT_LEVELS:
            raise ProjectContractError("task identity external sort exceeds its bound")
        bucket = self._levels.setdefault(level, [])
        bucket.append(path)
        if len(bucket) < TASK_INDEX_SORT_FAN_IN:
            return
        inputs = tuple(bucket)
        del self._levels[level]
        merged = self._merge_runs(inputs)
        self._unlink_runs(inputs)
        self._add_run(merged, level=level + 1)

    def _merge_runs(self, paths: Sequence[Path]) -> Path:
        if not paths or len(paths) > TASK_INDEX_SORT_FAN_IN:
            raise ProjectContractError("invalid task identity merge fan-in")
        fd, temporary = tempfile.mkstemp(
            dir=self._directory,
            prefix="identity-merge-",
            suffix=".txt",
        )
        output = Path(temporary)
        handles: list[BinaryIO] = []
        try:
            handles = [path.open("rb") for path in paths]
            streams = (
                (encoded.rstrip(b"\n").decode("utf-8") for encoded in handle)
                for handle in handles
            )
            previous: str | None = None
            with os.fdopen(fd, "wb") as destination:
                for key in heapq.merge(*streams):
                    if key == previous:
                        raise ProjectContractError(f"duplicate task identity {key}")
                    previous = key
                    destination.write(key.encode("utf-8") + b"\n")
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                output.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            for handle in handles:
                handle.close()
        return output

    @staticmethod
    def _unlink_runs(paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _inspect_source_stream(
    path: Path,
    *,
    repo_root: Path,
    split: Split | str,
    registry: Coco80RegistryProtocol | None,
    consume_row: SourceRowConsumer | None = None,
) -> _SourceStreamResult:
    """Consume, hash, and validate the exact bytes used to derive a plan."""

    contract, observed_path = _assert_exact_source_location(
        path,
        repo_root=repo_root,
        split=split,
    )
    digest = hashlib.sha256()
    source_identity_digest = hashlib.sha256()
    source_identity_digest.update(b"[")
    box_count = 0
    row_count = 0
    sort_parent = (
        RuntimeLayout.for_repo(repo_root).for_split(contract.split).root
        / TASK_INDEX_DIRECTORY_NAME
    )
    sort_parent.mkdir(parents=True, exist_ok=True)
    temporary_sort_root = Path(
        tempfile.mkdtemp(dir=sort_parent, prefix=".identity-sort-")
    )
    identity_builder = _IdentityFingerprintBuilder(temporary_sort_root)
    try:
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
                identity = validate_source_row(
                    row, split=contract.split, registry=registry
                )
                _assert_source_image_exists(
                    row,
                    repo_root=repo_root,
                    split=contract.split,
                    line_number=line_number,
                )
                identity_builder.add(identity.key)
                if row_count:
                    source_identity_digest.update(b",")
                source_identity_digest.update(_canonical_json_bytes(identity.key))
                row_count += 1
                box_count += len(row["objects"])
                if consume_row is not None:
                    consume_row(line_number, row, identity)
        identity_count, identity_fingerprint = identity_builder.finish()
    finally:
        shutil.rmtree(temporary_sort_root, ignore_errors=True)
    observed_hash = digest.hexdigest()
    source_identity_digest.update(b"]")
    if observed_hash != contract.sha256:
        raise ProjectContractError(
            f"source hash drift for {contract.split.value}: expected {contract.sha256}, "
            f"got {observed_hash}"
        )
    if row_count != contract.row_count:
        raise ProjectContractError(
            f"source row-count drift: expected {contract.row_count}, got {row_count}"
        )
    if identity_count != row_count:
        raise ProjectContractError("source identity count drift")
    if box_count != contract.box_count:
        raise ProjectContractError(
            f"source box-count drift: expected {contract.box_count}, got {box_count}"
        )
    return _SourceStreamResult(
        inspection=SourceInspection(
            contract=contract,
            source_path=str(_lexical_absolute(path)),
            sha256=observed_hash,
            row_count=row_count,
            box_count=box_count,
            task_identity_fingerprint=source_identity_digest.hexdigest(),
        ),
        sorted_identity_fingerprint=identity_fingerprint,
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskManifestEntry":
        required = {
            "identity",
            "source_line",
            "source_image_locator",
            "working_image_locator",
            "label_studio_image_locator",
            "task_data_fingerprint",
            "authoritative_annotation_fingerprint",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ProjectContractError("task-index entry shape drift")
        identity_payload = payload["identity"]
        if not isinstance(identity_payload, Mapping) or set(identity_payload) != {
            "split",
            "image_id",
            "key",
        }:
            raise ProjectContractError("task-index identity shape drift")
        identity = TaskIdentity(
            _split(identity_payload["split"]),
            _require_int(identity_payload["image_id"], "image_id", minimum=0),
        )
        if identity_payload["key"] != identity.key:
            raise ProjectContractError("task-index identity key drift")
        return cls(
            identity=identity,
            source_line=_require_int(
                payload["source_line"], "source_line", minimum=1
            ),
            source_image_locator=_require_string(
                payload["source_image_locator"], "source_image_locator"
            ),
            working_image_locator=_require_string(
                payload["working_image_locator"], "working_image_locator"
            ),
            label_studio_image_locator=_require_string(
                payload["label_studio_image_locator"],
                "label_studio_image_locator",
            ),
            task_data_fingerprint=_require_sha256(
                payload["task_data_fingerprint"], "task_data_fingerprint"
            ),
            authoritative_annotation_fingerprint=_require_sha256(
                payload["authoritative_annotation_fingerprint"],
                "authoritative_annotation_fingerprint",
            ),
        )


@dataclass(frozen=True)
class TaskIndexReceipt:
    split: Split
    path: str
    sha256: str
    task_count: int
    source_sha256: str
    source_row_count: int
    source_box_count: int
    task_manifest_fingerprint: str
    identity_fingerprint: str
    converter_fingerprint: str
    build_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_INDEX_SCHEMA_VERSION,
            "split": self.split.value,
            "path": self.path,
            "sha256": self.sha256,
            "task_count": self.task_count,
            "source_sha256": self.source_sha256,
            "source_row_count": self.source_row_count,
            "source_box_count": self.source_box_count,
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "identity_fingerprint": self.identity_fingerprint,
            "converter_fingerprint": self.converter_fingerprint,
            "build_key": self.build_key,
        }

    def validate(self) -> None:
        _validate_task_index_receipt(self)

    def validate_for_repo(self, repo_root: Path) -> str:
        """Validate the receipt and all referenced images, returning source-order ID hash."""

        return _validate_task_index_receipt(self, repo_root=repo_root)

    def iter_records(self) -> Iterator[tuple[TaskManifestEntry, dict[str, Any]]]:
        """Validate the entire immutable index, then stream fresh decoded records."""

        self.validate()
        yield from _iter_task_index_records(self)


@dataclass(frozen=True)
class TaskManifest:
    split: Split
    task_count: int
    fingerprint: str
    identity_fingerprint: str
    task_index: TaskIndexReceipt

    def body(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "task_count": self.task_count,
            "fingerprint": self.fingerprint,
            "identity_fingerprint": self.identity_fingerprint,
            "task_index": self.task_index.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.body()

    def iter_entries(self) -> Iterator[TaskManifestEntry]:
        for entry, _payload in self.task_index.iter_records():
            yield entry


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

    def iter_task_import_chunks(
        self, chunk_size: int
    ) -> Iterator[tuple[dict[str, Any], ...]]:
        """Stream detached canonical payloads from the fully validated sidecar."""

        _require_chunk_size(chunk_size)
        chunk: list[dict[str, Any]] = []
        for _entry, payload in self.task_manifest.task_index.iter_records():
            chunk.append(payload)
            if len(chunk) == chunk_size:
                yield tuple(chunk)
                chunk = []
        if chunk:
            yield tuple(chunk)


class BootstrapAction(str, Enum):
    CREATE = "create"
    RECONCILE = "reconcile"
    REUSE = "reuse"


@dataclass(frozen=True)
class PlannedProjectBootstrap:
    action: BootstrapAction
    project: SplitProjectPlan
    observed_task_count: int
    missing_task_count: int
    live_attestation_fingerprint: str | None = None

    @property
    def task_import_count(self) -> int:
        if self.action is BootstrapAction.REUSE:
            return 0
        return self.project.task_manifest.task_count

    def iter_task_import_chunks(
        self, chunk_size: int
    ) -> Iterator[tuple[dict[str, Any], ...]]:
        """CREATE/RECONCILE stream all desired rows; chunk import is idempotent."""

        _validate_planned_action(self)
        if self.action is BootstrapAction.REUSE:
            return
        yield from self.project.iter_task_import_chunks(chunk_size)


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


class CanonicalJsonArrayFingerprint:
    """Incrementally hash a canonical JSON array without retaining its members."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, payload: Any) -> None:
        if self._count:
            self._digest.update(b",")
        self._digest.update(_canonical_json_bytes(payload))
        self._count += 1

    @property
    def fingerprint(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


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
class LiveTaskSetAttestation:
    """Bounded aggregate proving the live rows are an exact desired subset."""

    expected_task_manifest_fingerprint: str
    observed_task_count: int
    missing_task_count: int
    content_fingerprint: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.expected_task_manifest_fingerprint,
            "expected_task_manifest_fingerprint",
        )
        _require_sha256(self.content_fingerprint, "content_fingerprint")
        _require_int(self.observed_task_count, "observed_task_count", minimum=0)
        _require_int(self.missing_task_count, "missing_task_count", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_task_manifest_fingerprint": (
                self.expected_task_manifest_fingerprint
            ),
            "observed_task_count": self.observed_task_count,
            "missing_task_count": self.missing_task_count,
            "content_fingerprint": self.content_fingerprint,
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
    task_set: LiveTaskSetAttestation

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
            "task_set": self.task_set.to_dict(),
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


class _TaskIndexWriter:
    """Write one deterministic payload-bearing sidecar in bounded memory."""

    def __init__(
        self,
        path: Path,
        *,
        split: Split,
        contract: SourceContract,
        converter_fingerprint: str,
        build_key: str,
    ) -> None:
        self.path = path
        self.split = split
        self.contract = contract
        self.converter_fingerprint = converter_fingerprint
        self.build_key = build_key
        self._handle = path.open("xb")
        self._file_digest = hashlib.sha256()
        self._manifest_digest = hashlib.sha256()
        self._manifest_digest.update(b'{"entries":[')
        self._count = 0
        header_body = {
            "kind": "header",
            "schema_version": TASK_INDEX_SCHEMA_VERSION,
            "split": split.value,
            "source_sha256": contract.sha256,
            "source_row_count": contract.row_count,
            "source_box_count": contract.box_count,
            "converter_fingerprint": converter_fingerprint,
            "build_key": build_key,
        }
        self._previous_record_hash = fingerprint_json(header_body)
        self._write_line({**header_body, "record_hash": self._previous_record_hash})

    def add(self, entry: TaskManifestEntry, payload: Mapping[str, Any]) -> None:
        if entry.source_line != self._count + 1:
            raise ProjectContractError("task-index source sequence drift")
        detached = _decode_task_import(_canonical_json_bytes(payload))
        _validate_task_payload_against_entry(
            detached,
            entry=entry,
            split=self.split,
        )
        entry_payload = entry.to_dict()
        if self._count:
            self._manifest_digest.update(b",")
        self._manifest_digest.update(_canonical_json_bytes(entry_payload))
        body = {
            "kind": "task",
            "sequence": entry.source_line,
            "previous_record_hash": self._previous_record_hash,
            "entry": entry_payload,
            "payload": detached,
        }
        record_hash = fingerprint_json(body)
        self._write_line({**body, "record_hash": record_hash})
        self._previous_record_hash = record_hash
        self._count += 1

    def finish(
        self,
        source_inspection: SourceInspection,
        *,
        sorted_identity_fingerprint: str,
    ) -> tuple[str, str]:
        if source_inspection.row_count != self._count:
            raise ProjectContractError("task-index count does not match source inspection")
        self._manifest_digest.update(b'],"split":')
        self._manifest_digest.update(_canonical_json_bytes(self.split.value))
        self._manifest_digest.update(b',"task_count":')
        self._manifest_digest.update(str(self._count).encode("ascii"))
        self._manifest_digest.update(b"}")
        task_manifest_fingerprint = self._manifest_digest.hexdigest()
        trailer_body = {
            "kind": "trailer",
            "schema_version": TASK_INDEX_SCHEMA_VERSION,
            "split": self.split.value,
            "source_sha256": source_inspection.sha256,
            "source_row_count": source_inspection.row_count,
            "source_box_count": source_inspection.box_count,
            "task_count": self._count,
            "task_manifest_fingerprint": task_manifest_fingerprint,
            "identity_fingerprint": sorted_identity_fingerprint,
            "converter_fingerprint": self.converter_fingerprint,
            "build_key": self.build_key,
            "previous_record_hash": self._previous_record_hash,
        }
        trailer_hash = fingerprint_json(trailer_body)
        self._write_line({**trailer_body, "record_hash": trailer_hash})
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.chmod(self.path, 0o444)
        return task_manifest_fingerprint, self._file_digest.hexdigest()

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        encoded = _canonical_json_bytes(payload) + b"\n"
        self._handle.write(encoded)
        self._file_digest.update(encoded)


def _task_index_build_key(
    *,
    contract: SourceContract,
    registry_fingerprint: str,
    label_config_fingerprint: str,
    converter_fingerprint: str,
) -> str:
    return fingerprint_json(
        {
            "schema_version": TASK_INDEX_SCHEMA_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "split": contract.split.value,
            "source_relative_path": contract.relative_path.as_posix(),
            "source_sha256": contract.sha256,
            "source_row_count": contract.row_count,
            "source_box_count": contract.box_count,
            "registry_fingerprint": registry_fingerprint,
            "label_config_fingerprint": label_config_fingerprint,
            "converter_fingerprint": converter_fingerprint,
        }
    )


def _task_index_sidecar_name(
    *, converter_fingerprint: str, content_sha256: str
) -> str:
    return (
        f"task-index-v{TASK_INDEX_SCHEMA_VERSION}-{converter_fingerprint}-"
        f"{content_sha256}.jsonl"
    )


def _task_index_anchor_path(
    directory: Path, *, converter_fingerprint: str, build_key: str
) -> Path:
    return directory / (
        f"build-v{TASK_INDEX_SCHEMA_VERSION}-{converter_fingerprint}-{build_key}.json"
    )


def _publish_task_index(
    temporary_path: Path,
    *,
    split: Split,
    content_sha256: str,
    task_count: int,
    source_inspection: SourceInspection,
    task_manifest_fingerprint: str,
    identity_fingerprint: str,
    converter_fingerprint: str,
    build_key: str,
    publish_anchor: bool,
) -> TaskIndexReceipt:
    directory = temporary_path.parent.parent
    destination = directory / _task_index_sidecar_name(
        converter_fingerprint=converter_fingerprint,
        content_sha256=content_sha256,
    )
    try:
        os.link(temporary_path, destination)
        _fsync_directory(directory)
    except FileExistsError:
        pass
    receipt = TaskIndexReceipt(
        split=split,
        path=str(destination),
        sha256=content_sha256,
        task_count=task_count,
        source_sha256=source_inspection.sha256,
        source_row_count=source_inspection.row_count,
        source_box_count=source_inspection.box_count,
        task_manifest_fingerprint=task_manifest_fingerprint,
        identity_fingerprint=identity_fingerprint,
        converter_fingerprint=converter_fingerprint,
        build_key=build_key,
    )
    receipt.validate()
    if not publish_anchor:
        return receipt
    anchor_payload = {
        "schema_version": TASK_INDEX_SCHEMA_VERSION,
        "build_key": build_key,
        "sidecar_name": destination.name,
        "sidecar_sha256": content_sha256,
        "split": split.value,
        "task_count": task_count,
        "task_manifest_fingerprint": task_manifest_fingerprint,
        "identity_fingerprint": identity_fingerprint,
        "converter_fingerprint": converter_fingerprint,
    }
    _publish_small_immutable_file(
        _task_index_anchor_path(
            directory,
            converter_fingerprint=converter_fingerprint,
            build_key=build_key,
        ),
        _canonical_json_bytes(anchor_payload) + b"\n",
    )
    return receipt


def _publish_small_immutable_file(path: Path, content: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                or path.read_bytes() != content
            ):
                raise ProjectContractError(
                    f"task-index build receipt conflict for {path.name}"
                ) from None
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _task_manifest_from_sidecar(
    receipt: TaskIndexReceipt,
) -> TaskManifest:
    return TaskManifest(
        split=receipt.split,
        task_count=receipt.task_count,
        fingerprint=receipt.task_manifest_fingerprint,
        identity_fingerprint=receipt.identity_fingerprint,
        task_index=receipt,
    )


def _load_published_task_index(
    *,
    directory: Path,
    contract: SourceContract,
    converter_fingerprint: str,
    build_key: str,
) -> TaskIndexReceipt | None:
    anchor = _task_index_anchor_path(
        directory,
        converter_fingerprint=converter_fingerprint,
        build_key=build_key,
    )
    if not os.path.lexists(anchor):
        return None
    try:
        metadata = anchor.lstat()
    except OSError as exc:
        raise ProjectContractError("task-index build receipt is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or anchor.is_symlink():
        raise ProjectContractError("task-index build receipt must be a regular file")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ProjectContractError("task-index build receipt must be immutable/read-only")
    raw = anchor.read_bytes()
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ProjectContractError("task-index build receipt has trailing data")
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectContractError("task-index build receipt is invalid JSON") from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload) + b"\n":
        raise ProjectContractError("task-index build receipt is not canonical JSON")
    fields = {
        "schema_version",
        "build_key",
        "sidecar_name",
        "sidecar_sha256",
        "split",
        "task_count",
        "task_manifest_fingerprint",
        "identity_fingerprint",
        "converter_fingerprint",
    }
    if set(payload) != fields:
        raise ProjectContractError("task-index build receipt shape drift")
    _require_int(
        payload["schema_version"],
        "task_index.build_receipt.schema_version",
        minimum=TASK_INDEX_SCHEMA_VERSION,
        maximum=TASK_INDEX_SCHEMA_VERSION,
    )
    _require_int(
        payload["task_count"],
        "task_index.build_receipt.task_count",
        minimum=1,
    )
    for field in (
        "build_key",
        "sidecar_sha256",
        "task_manifest_fingerprint",
        "identity_fingerprint",
        "converter_fingerprint",
    ):
        _require_sha256(payload[field], f"task_index.build_receipt.{field}")
    sidecar_name = _require_string(
        payload["sidecar_name"], "task_index.build_receipt.sidecar_name"
    )
    expected_sidecar_name = _task_index_sidecar_name(
        converter_fingerprint=converter_fingerprint,
        content_sha256=payload["sidecar_sha256"],
    )
    if (
        payload["build_key"] != build_key
        or payload["converter_fingerprint"] != converter_fingerprint
        or payload["split"] != contract.split.value
        or payload["task_count"] != contract.row_count
        or sidecar_name != expected_sidecar_name
        or Path(sidecar_name).name != sidecar_name
    ):
        raise ProjectContractError("task-index build receipt contract drift")
    return TaskIndexReceipt(
        split=contract.split,
        path=str(directory / sidecar_name),
        sha256=payload["sidecar_sha256"],
        task_count=payload["task_count"],
        source_sha256=contract.sha256,
        source_row_count=contract.row_count,
        source_box_count=contract.box_count,
        task_manifest_fingerprint=payload["task_manifest_fingerprint"],
        identity_fingerprint=payload["identity_fingerprint"],
        converter_fingerprint=payload["converter_fingerprint"],
        build_key=build_key,
    )


def _reuse_published_task_index(
    *,
    source_path: Path,
    observed_source: Path,
    repo_root: Path,
    contract: SourceContract,
    directory: Path,
    converter_fingerprint: str,
    build_key: str,
) -> tuple[SourceInspection, TaskIndexReceipt] | None:
    receipt = _load_published_task_index(
        directory=directory,
        contract=contract,
        converter_fingerprint=converter_fingerprint,
        build_key=build_key,
    )
    if receipt is None:
        return None
    observed_hash = sha256_file(observed_source)
    if observed_hash != contract.sha256:
        raise ProjectContractError(
            f"source hash drift for {contract.split.value}: expected {contract.sha256}, "
            f"got {observed_hash}"
        )
    source_identity_fingerprint = receipt.validate_for_repo(repo_root)
    return (
        SourceInspection(
            contract=contract,
            source_path=str(_lexical_absolute(source_path)),
            sha256=observed_hash,
            row_count=contract.row_count,
            box_count=contract.box_count,
            task_identity_fingerprint=source_identity_fingerprint,
        ),
        receipt,
    )


def _build_task_index_from_source(
    source_path: Path,
    *,
    repo_root: Path,
    split: Split,
    contract: SourceContract,
    registry: Coco80RegistryProtocol,
    bbox_converter: BboxToLabelStudio,
    converter_fingerprint: str,
    build_key: str,
    index_directory: Path,
    publish_anchor: bool,
) -> tuple[SourceInspection, TaskIndexReceipt]:
    temporary_root = Path(
        tempfile.mkdtemp(dir=index_directory, prefix=".builder-")
    )
    temporary_index = temporary_root / "task-index.jsonl"
    writer = _TaskIndexWriter(
        temporary_index,
        split=split,
        contract=contract,
        converter_fingerprint=converter_fingerprint,
        build_key=build_key,
    )

    def consume_row(
        line_number: int,
        row: Mapping[str, Any],
        identity: TaskIdentity,
    ) -> None:
        task = make_task_import_payload(
            row,
            split=split,
            source_line=line_number,
            registry=registry,
            bbox_converter=bbox_converter,
        )
        writer.add(
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
            ),
            task,
        )

    try:
        source_result = _inspect_source_stream(
            source_path,
            repo_root=repo_root,
            split=split,
            registry=registry,
            consume_row=consume_row,
        )
        source_inspection = source_result.inspection
        task_manifest_fingerprint, content_sha256 = writer.finish(
            source_inspection,
            sorted_identity_fingerprint=source_result.sorted_identity_fingerprint,
        )
        task_index = _publish_task_index(
            temporary_index,
            split=split,
            content_sha256=content_sha256,
            task_count=source_inspection.row_count,
            source_inspection=source_inspection,
            task_manifest_fingerprint=task_manifest_fingerprint,
            identity_fingerprint=source_result.sorted_identity_fingerprint,
            converter_fingerprint=converter_fingerprint,
            build_key=build_key,
            publish_anchor=publish_anchor,
        )
        return source_inspection, task_index
    except BaseException:
        writer.abort()
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


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
    converter_fingerprint: str | None = None,
) -> SplitProjectPlan:
    """Build from exact pinned bytes or a fully re-attested immutable receipt."""

    split_value = _split(split)
    registry = registry or default_registry()
    converter_contract = _resolve_bbox_converter_contract(
        bbox_converter,
        converter_fingerprint,
    )
    from .label_config import validate_label_config

    validate_label_config(label_config, expected_names=registry.names)
    if label_config_fingerprint != fingerprint_json({"label_config_xml": label_config}):
        raise ProjectContractError(
            "label_config_fingerprint does not match label_config"
        )
    layout = RuntimeLayout.for_repo(repo_root)
    project_identity = f"coco-refinement:{DATASET_NAME}:{split_value.value}"
    contract, observed_source = _assert_exact_source_location(
        source_path,
        repo_root=repo_root,
        split=split_value,
    )
    index_directory = (
        layout.for_split(split_value).root / TASK_INDEX_DIRECTORY_NAME
    )
    index_directory.mkdir(parents=True, exist_ok=True)
    build_key = _task_index_build_key(
        contract=contract,
        registry_fingerprint=registry.fingerprint,
        label_config_fingerprint=label_config_fingerprint,
        converter_fingerprint=converter_contract.fingerprint,
    )
    existing = None
    if converter_contract.reuse_anchor:
        existing = _reuse_published_task_index(
            source_path=source_path,
            observed_source=observed_source,
            repo_root=repo_root,
            contract=contract,
            directory=index_directory,
            converter_fingerprint=converter_contract.fingerprint,
            build_key=build_key,
        )
    if existing is None:
        source_inspection, task_index = _build_task_index_from_source(
            source_path,
            repo_root=repo_root,
            split=split_value,
            contract=contract,
            registry=registry,
            bbox_converter=converter_contract.converter,
            converter_fingerprint=converter_contract.fingerprint,
            build_key=build_key,
            index_directory=index_directory,
            publish_anchor=converter_contract.publish_anchor,
        )
    else:
        source_inspection, task_index = existing
    task_manifest = _task_manifest_from_sidecar(task_index)
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
    )


def _validate_task_index_receipt(
    receipt: TaskIndexReceipt,
    *,
    repo_root: Path | None = None,
) -> str:
    if not isinstance(receipt, TaskIndexReceipt):
        raise ProjectContractError("task index receipt has the wrong type")
    _require_sha256(receipt.sha256, "task_index.sha256")
    _require_sha256(
        receipt.task_manifest_fingerprint,
        "task_index.task_manifest_fingerprint",
    )
    _require_sha256(
        receipt.identity_fingerprint,
        "task_index.identity_fingerprint",
    )
    _require_sha256(receipt.source_sha256, "task_index.source_sha256")
    _require_sha256(
        receipt.converter_fingerprint,
        "task_index.converter_fingerprint",
    )
    _require_sha256(receipt.build_key, "task_index.build_key")
    _require_int(receipt.task_count, "task_index.task_count", minimum=1)
    _require_int(
        receipt.source_row_count,
        "task_index.source_row_count",
        minimum=1,
    )
    _require_int(
        receipt.source_box_count,
        "task_index.source_box_count",
        minimum=1,
    )
    path = Path(receipt.path)
    expected_name = _task_index_sidecar_name(
        converter_fingerprint=receipt.converter_fingerprint,
        content_sha256=receipt.sha256,
    )
    if path.name != expected_name:
        raise ProjectContractError("task index path is not content addressed")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProjectContractError(f"task index is unavailable: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ProjectContractError("task index must be a regular non-symlink file")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ProjectContractError("task index must be immutable/read-only")
    identity_digest = hashlib.sha256()
    identity_digest.update(b"[")
    count = 0
    for entry, _payload in _iter_task_index_records(receipt):
        if count:
            identity_digest.update(b",")
        identity_digest.update(_canonical_json_bytes(entry.identity.key))
        count += 1
        if repo_root is not None:
            _assert_task_index_image_exists(entry, repo_root=repo_root)
    identity_digest.update(b"]")
    return identity_digest.hexdigest()


def _iter_task_index_records(
    receipt: TaskIndexReceipt,
) -> Iterator[tuple[TaskManifestEntry, dict[str, Any]]]:
    path = Path(receipt.path)
    file_digest = hashlib.sha256()
    manifest_digest = hashlib.sha256()
    manifest_digest.update(b'{"entries":[')
    task_count = 0
    with path.open("rb") as handle:
        encoded_header = handle.readline()
        header = _decode_task_index_line(encoded_header, file_digest=file_digest)
        header_fields = {
            "kind",
            "schema_version",
            "split",
            "source_sha256",
            "source_row_count",
            "source_box_count",
            "converter_fingerprint",
            "build_key",
            "record_hash",
        }
        if set(header) != header_fields or header.get("kind") != "header":
            raise ProjectContractError("task index header shape drift")
        _require_int(
            header["schema_version"],
            "task_index.header.schema_version",
            minimum=TASK_INDEX_SCHEMA_VERSION,
            maximum=TASK_INDEX_SCHEMA_VERSION,
        )
        _require_int(
            header["source_row_count"],
            "task_index.header.source_row_count",
            minimum=1,
        )
        _require_int(
            header["source_box_count"],
            "task_index.header.source_box_count",
            minimum=1,
        )
        _require_sha256(header["source_sha256"], "task_index.header.source_sha256")
        _require_sha256(
            header["converter_fingerprint"],
            "task_index.header.converter_fingerprint",
        )
        _require_sha256(header["build_key"], "task_index.header.build_key")
        _require_sha256(header["record_hash"], "task_index.header.record_hash")
        header_body = {key: header[key] for key in header_fields if key != "record_hash"}
        if header["record_hash"] != fingerprint_json(header_body):
            raise ProjectContractError("task index header hash drift")
        if (
            header["schema_version"] != TASK_INDEX_SCHEMA_VERSION
            or header["split"] != receipt.split.value
            or header["source_sha256"] != receipt.source_sha256
            or header["source_row_count"] != receipt.source_row_count
            or header["source_box_count"] != receipt.source_box_count
            or header["converter_fingerprint"] != receipt.converter_fingerprint
            or header["build_key"] != receipt.build_key
        ):
            raise ProjectContractError("task index header contract drift")
        previous_record_hash = header["record_hash"]

        while True:
            encoded = handle.readline()
            if encoded == b"":
                raise ProjectContractError("task index is missing its trailer")
            record = _decode_task_index_line(encoded, file_digest=file_digest)
            kind = record.get("kind")
            if kind == "task":
                fields = {
                    "kind",
                    "sequence",
                    "previous_record_hash",
                    "entry",
                    "payload",
                    "record_hash",
                }
                if set(record) != fields:
                    raise ProjectContractError("task index record shape drift")
                _require_int(
                    record["sequence"],
                    "task_index.task.sequence",
                    minimum=1,
                )
                _require_sha256(
                    record["previous_record_hash"],
                    "task_index.task.previous_record_hash",
                )
                _require_sha256(
                    record["record_hash"],
                    "task_index.task.record_hash",
                )
                body = {key: record[key] for key in fields if key != "record_hash"}
                if record["previous_record_hash"] != previous_record_hash:
                    raise ProjectContractError("task index hash-chain drift")
                if record["record_hash"] != fingerprint_json(body):
                    raise ProjectContractError("task index record hash drift")
                task_count += 1
                if record["sequence"] != task_count:
                    raise ProjectContractError("task index sequence drift")
                entry = TaskManifestEntry.from_dict(record["entry"])
                if entry.identity.split is not receipt.split:
                    raise ProjectContractError("task index entry split drift")
                if entry.source_line != task_count:
                    raise ProjectContractError("task index entry sequence drift")
                payload = record["payload"]
                if not isinstance(payload, dict):
                    raise ProjectContractError("task index payload is not an object")
                _validate_task_payload_against_entry(
                    payload,
                    entry=entry,
                    split=receipt.split,
                )
                if task_count > 1:
                    manifest_digest.update(b",")
                manifest_digest.update(_canonical_json_bytes(entry.to_dict()))
                previous_record_hash = record["record_hash"]
                yield entry, payload
                continue
            if kind != "trailer":
                raise ProjectContractError("task index record kind drift")
            fields = {
                "kind",
                "schema_version",
                "split",
                "source_sha256",
                "source_row_count",
                "source_box_count",
                "task_count",
                "task_manifest_fingerprint",
                "identity_fingerprint",
                "converter_fingerprint",
                "build_key",
                "previous_record_hash",
                "record_hash",
            }
            if set(record) != fields:
                raise ProjectContractError("task index trailer shape drift")
            _require_int(
                record["schema_version"],
                "task_index.trailer.schema_version",
                minimum=TASK_INDEX_SCHEMA_VERSION,
                maximum=TASK_INDEX_SCHEMA_VERSION,
            )
            for field in ("source_row_count", "source_box_count", "task_count"):
                _require_int(
                    record[field],
                    f"task_index.trailer.{field}",
                    minimum=1,
                )
            for field in (
                "source_sha256",
                "task_manifest_fingerprint",
                "identity_fingerprint",
                "converter_fingerprint",
                "build_key",
                "previous_record_hash",
                "record_hash",
            ):
                _require_sha256(record[field], f"task_index.trailer.{field}")
            trailer_body = {
                key: record[key] for key in fields if key != "record_hash"
            }
            if record["previous_record_hash"] != previous_record_hash:
                raise ProjectContractError("task index trailer hash-chain drift")
            if record["record_hash"] != fingerprint_json(trailer_body):
                raise ProjectContractError("task index trailer hash drift")
            if handle.read(1) != b"":
                raise ProjectContractError("task index has data after its trailer")
            break

    manifest_digest.update(b'],"split":')
    manifest_digest.update(_canonical_json_bytes(receipt.split.value))
    manifest_digest.update(b',"task_count":')
    manifest_digest.update(str(task_count).encode("ascii"))
    manifest_digest.update(b"}")
    manifest_fingerprint = manifest_digest.hexdigest()
    if file_digest.hexdigest() != receipt.sha256:
        raise ProjectContractError("task index content hash drift")
    if task_count != receipt.task_count or task_count != receipt.source_row_count:
        raise ProjectContractError("task index task-count drift")
    if (
        manifest_fingerprint != receipt.task_manifest_fingerprint
        or record["task_manifest_fingerprint"] != receipt.task_manifest_fingerprint
    ):
        raise ProjectContractError("task index manifest fingerprint drift")
    if (
        record["schema_version"] != TASK_INDEX_SCHEMA_VERSION
        or record["split"] != receipt.split.value
        or record["source_sha256"] != receipt.source_sha256
        or record["source_row_count"] != receipt.source_row_count
        or record["source_box_count"] != receipt.source_box_count
        or record["task_count"] != receipt.task_count
        or record["identity_fingerprint"] != receipt.identity_fingerprint
        or record["converter_fingerprint"] != receipt.converter_fingerprint
        or record["build_key"] != receipt.build_key
    ):
        raise ProjectContractError("task index trailer contract drift")


def _decode_task_index_line(
    encoded: bytes,
    *,
    file_digest: "hashlib._Hash",
) -> dict[str, Any]:
    if not encoded or not encoded.endswith(b"\n"):
        raise ProjectContractError("task index contains a torn JSONL record")
    file_digest.update(encoded)
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectContractError("task index contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProjectContractError("task index record is not an object")
    if encoded != _canonical_json_bytes(payload) + b"\n":
        raise ProjectContractError("task index record is not canonical JSON")
    return payload


def _validate_task_payload_against_entry(
    payload: Mapping[str, Any],
    *,
    entry: TaskManifestEntry,
    split: Split,
) -> None:
    validate_authoritative_task_payload(payload)
    expected_source_locator = (
        PurePosixPath("..")
        / "rescale_32_1024_bbox"
        / PurePosixPath(entry.working_image_locator)
    ).as_posix()
    if entry.source_image_locator != expected_source_locator:
        raise ProjectContractError("task manifest source image locator drift")
    if (
        local_files_image_locator(entry.working_image_locator, split=split)
        != entry.label_studio_image_locator
    ):
        raise ProjectContractError("task manifest working image locator drift")
    data = payload["data"]
    identity = TaskIdentity(_split(data["split"]), data["image_id"])
    if identity != entry.identity or identity.split is not split:
        raise ProjectContractError("task import identity does not match task manifest")
    if data["source_line"] != entry.source_line:
        raise ProjectContractError("task import source line does not match task manifest")
    if data["image"] != entry.label_studio_image_locator:
        raise ProjectContractError("task import image locator does not match task manifest")
    if fingerprint_json(data) != entry.task_data_fingerprint:
        raise ProjectContractError("task data fingerprint does not match task manifest")
    if (
        fingerprint_json(payload["annotations"][0])
        != entry.authoritative_annotation_fingerprint
    ):
        raise ProjectContractError(
            "authoritative annotation fingerprint does not match task manifest"
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
    manifest = _build_instance_bootstrap_manifest(
        desired_normalized,
        layout=RuntimeLayout.for_repo(repo_root),
    )
    attest_manifest = getattr(adapter, "attest_bootstrap_manifest", None)
    if not callable(attest_manifest):
        raise ProjectContractError(
            "adapter must attest the parent-owned bootstrap manifest"
        )
    observed_manifest = attest_manifest()
    live_projects = {
        split_value: adapter.attest_project(desired_normalized[split_value])
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
            observed_task_count = 0
            missing_task_count = project.task_manifest.task_count
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
            observed_task_count, missing_task_count = _assert_live_project_matches(
                project, live_attestation
            )
            action = (
                BootstrapAction.REUSE
                if missing_task_count == 0
                else BootstrapAction.RECONCILE
            )
            attestation_fingerprint = live_attestation.fingerprint
        planned = PlannedProjectBootstrap(
            action=action,
            project=project,
            observed_task_count=observed_task_count,
            missing_task_count=missing_task_count,
            live_attestation_fingerprint=attestation_fingerprint,
        )
        _validate_planned_action(planned)
        actions.append(planned)
    return InstanceBootstrapPlan(
        runtime_layout=RuntimeLayout.for_repo(repo_root),
        manifest=manifest,
        projects=tuple(actions),
    )


def _assert_live_project_matches(
    desired: SplitProjectPlan,
    observed: LiveProjectAttestation,
) -> tuple[int, int]:
    """Validate an adapter's bounded exact-subset receipt."""

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

    task_set = observed.task_set
    if not isinstance(task_set, LiveTaskSetAttestation):
        mismatches.append("live.task_set")
    else:
        if (
            task_set.expected_task_manifest_fingerprint
            != desired.task_manifest.fingerprint
        ):
            mismatches.append("live.task_set.expected_task_manifest_fingerprint")
        if (
            task_set.observed_task_count + task_set.missing_task_count
            != desired.task_manifest.task_count
        ):
            mismatches.append("live.task_set.task_count")
    if mismatches:
        raise ManifestDriftError(tuple(dict.fromkeys(mismatches)))
    return task_set.observed_task_count, task_set.missing_task_count


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
    """Fail before live adapter calls if a bounded plan is inconsistent."""

    if not isinstance(project, SplitProjectPlan):
        raise ProjectContractError("desired project has the wrong plan type")
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
    if project.task_manifest.task_count != project.source_inspection.row_count:
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
    task_index = project.task_manifest.task_index
    if task_index.split is not project.split:
        raise ProjectContractError("task index split does not match project split")
    if (
        task_index.task_count != project.task_manifest.task_count
        or task_index.task_manifest_fingerprint != project.task_manifest.fingerprint
        or task_index.identity_fingerprint
        != project.task_manifest.identity_fingerprint
        or task_index.source_sha256 != project.source_inspection.sha256
        or task_index.source_row_count != project.source_inspection.row_count
        or task_index.source_box_count != project.source_inspection.box_count
    ):
        raise ProjectContractError("task index receipt does not match project plan")
    expected_build_key = _task_index_build_key(
        contract=project.source_inspection.contract,
        registry_fingerprint=project.manifest.category_registry_fingerprint,
        label_config_fingerprint=project.manifest.label_config_fingerprint,
        converter_fingerprint=task_index.converter_fingerprint,
    )
    if task_index.build_key != expected_build_key:
        raise ProjectContractError("task index build key semantics drift")
    repo_root = Path(project.manifest.source_path).parents[3]
    expected_index_directory = (
        RuntimeLayout.for_repo(repo_root).for_split(project.split).root
        / TASK_INDEX_DIRECTORY_NAME
    ).resolve(strict=False)
    observed_index_path = Path(task_index.path).resolve(strict=False)
    if observed_index_path.parent != expected_index_directory:
        raise ProjectContractError("task index is outside the ignored split runtime")
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
    task_index.validate_for_repo(repo_root)


def _validate_planned_action(action: PlannedProjectBootstrap) -> None:
    if not isinstance(action, PlannedProjectBootstrap):
        raise ProjectContractError("planned action has the wrong type")
    total = action.project.task_manifest.task_count
    _require_int(action.observed_task_count, "observed_task_count", minimum=0)
    _require_int(action.missing_task_count, "missing_task_count", minimum=0)
    if action.observed_task_count + action.missing_task_count != total:
        raise ProjectContractError("planned task counts do not partition the source")
    if action.action is BootstrapAction.CREATE and (
        action.observed_task_count != 0 or action.missing_task_count != total
    ):
        raise ProjectContractError("create action task counts are inconsistent")
    if action.action is BootstrapAction.REUSE and action.missing_task_count != 0:
        raise ProjectContractError("reuse action cannot have missing tasks")
    if (
        action.action is BootstrapAction.RECONCILE
        and action.live_attestation_fingerprint is None
    ):
        raise ProjectContractError("reconcile action requires an observed live project")


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


def _assert_task_index_image_exists(
    entry: TaskManifestEntry,
    *,
    repo_root: Path,
) -> None:
    image_path = resolve_working_image(
        entry.working_image_locator,
        layout=RuntimeLayout.for_repo(repo_root),
        split=entry.identity.split,
    )
    if not image_path.is_file():
        raise ProjectContractError(
            "task-index image missing at source line "
            f"{entry.source_line}: {entry.working_image_locator}"
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


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProjectContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_chunk_size(value: Any) -> int:
    return _require_int(value, "chunk_size", minimum=1)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
