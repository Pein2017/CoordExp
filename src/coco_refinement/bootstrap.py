"""Exact-source bootstrap and indexed-image boundary for the standalone editor."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.common.errors import RuntimeContractError
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import NativeTaskIdentity, Split
from src.coco_refinement.repository import (
    CompactTaskRecord,
    ProjectRecord,
    SqliteDraftRepository,
)
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.project import (
    SHARED_IMAGE_ROOT,
    SOURCE_CONTRACTS,
    Split as ProjectSplit,
)
from src.label_studio_coco_refinement.store import (
    BootstrapSpec,
    WorkingDatasetStore,
    sha256_file,
    sha256_json,
)


DEFAULT_RUNTIME_RELATIVE = Path("outputs/coco_refinement/rescale_32_1024_bbox_len12000")
_ADAPTER_VERSION = "coco-refinement-native-v1"
_NO_VENDOR_REVISION = "standalone-no-vendor"
_NATIVE_LABEL_CONTRACT = "native-norm1000-coco80-v1"


class BootstrapContractError(RuntimeContractError):
    """Raised when the compact workspace or an indexed image drifts."""


@dataclass(frozen=True)
class BootstrapSourceContract:
    """Injectable exact-source contract used by production and bounded probes."""

    split: Split
    source_path: Path
    image_root: Path
    expected_source_sha256: str
    expected_row_count: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "val"):
            raise BootstrapContractError(
                "split must be train or val",
                code="coco_refinement.bootstrap_split",
                context={"split": self.split},
            )
        if not _is_digest(self.expected_source_sha256):
            raise BootstrapContractError(
                "expected_source_sha256 must be a lowercase SHA-256 digest",
                code="coco_refinement.bootstrap_source_hash",
            )
        if (
            isinstance(self.expected_row_count, bool)
            or not isinstance(self.expected_row_count, int)
            or self.expected_row_count <= 0
        ):
            raise BootstrapContractError(
                "expected_row_count must be a positive integer",
                code="coco_refinement.bootstrap_row_count",
            )


@dataclass(frozen=True)
class SplitBootstrapResult:
    """One split's store plus its complete compact SQLite projection."""

    project: ProjectRecord
    tasks: tuple[CompactTaskRecord, ...]
    store: WorkingDatasetStore
    store_created: bool


@dataclass(frozen=True)
class WorkspaceBootstrapResult:
    """The new runtime root, repository, and requested split projections."""

    runtime_root: Path
    repository: SqliteDraftRepository
    splits: Mapping[Split, SplitBootstrapResult]


class _BootstrapOnlyAnnotationVerifier:
    def verify(self, _identity: object) -> bool:
        return False


class _BootstrapOnlyReceiptResolver:
    def resolve(self, _receipt_id: str) -> None:
        return None


def production_source_contracts(
    repo_root: str | Path,
) -> tuple[BootstrapSourceContract, BootstrapSourceContract]:
    """Return the exact approved max_len12000 train/validation contracts."""

    root = Path(repo_root).resolve(strict=True)
    image_root = root / Path(SHARED_IMAGE_ROOT)
    values: list[BootstrapSourceContract] = []
    for split in (ProjectSplit.TRAIN, ProjectSplit.VAL):
        contract = SOURCE_CONTRACTS[split]
        values.append(
            BootstrapSourceContract(
                split=split.value,  # type: ignore[arg-type]
                source_path=contract.path(root),
                image_root=image_root,
                expected_source_sha256=contract.sha256,
                expected_row_count=contract.row_count,
            )
        )
    return values[0], values[1]


def bootstrap_workspace(
    repo_root: str | Path,
    *,
    runtime_root: str | Path | None = None,
    source_contracts: Sequence[BootstrapSourceContract] | None = None,
    repository: SqliteDraftRepository | None = None,
    annotation_verifier: object | None = None,
    inference_receipt_resolver: object | None = None,
) -> WorkspaceBootstrapResult:
    """Bootstrap stores and a complete compact index without baseline arrays.

    Empty ``extra_fingerprints`` are deliberate: the standalone workspace is
    derived directly from its source, working JSONL, and store-owned task index,
    never from a legacy Label Studio task-index sidecar.
    """

    root = Path(repo_root).resolve(strict=True)
    selected_runtime = (
        (root / DEFAULT_RUNTIME_RELATIVE)
        if runtime_root is None
        else Path(runtime_root)
    ).resolve()
    selected_runtime.mkdir(parents=True, exist_ok=True)
    selected_contracts = tuple(
        production_source_contracts(root)
        if source_contracts is None
        else source_contracts
    )
    if not selected_contracts:
        raise BootstrapContractError(
            "at least one source contract is required",
            code="coco_refinement.bootstrap_contracts",
        )
    if len({contract.split for contract in selected_contracts}) != len(
        selected_contracts
    ):
        raise BootstrapContractError(
            "source contracts contain duplicate splits",
            code="coco_refinement.bootstrap_contracts",
        )

    selected_repository = repository or SqliteDraftRepository(
        selected_runtime / "state.sqlite3"
    )
    verifier = annotation_verifier or _BootstrapOnlyAnnotationVerifier()
    resolver = inference_receipt_resolver or _BootstrapOnlyReceiptResolver()
    results: dict[Split, SplitBootstrapResult] = {}
    for contract in selected_contracts:
        result = _bootstrap_split(
            contract,
            runtime_root=selected_runtime,
            repository=selected_repository,
            annotation_verifier=verifier,
            inference_receipt_resolver=resolver,
        )
        results[contract.split] = result
    return WorkspaceBootstrapResult(
        runtime_root=selected_runtime,
        repository=selected_repository,
        splits=results,
    )


def _bootstrap_split(
    contract: BootstrapSourceContract,
    *,
    runtime_root: Path,
    repository: SqliteDraftRepository,
    annotation_verifier: object,
    inference_receipt_resolver: object,
) -> SplitBootstrapResult:
    project_id = f"coco-refinement:{contract.split}"
    bootstrap = WorkingDatasetStore.bootstrap(
        BootstrapSpec(
            split=contract.split,
            source_path=contract.source_path,
            runtime_root=runtime_root,
            image_root=contract.image_root,
            expected_source_sha256=contract.expected_source_sha256,
            project_id=project_id,
            storage_id=f"shared-coco-images-{contract.split}",
            adapter_version=_ADAPTER_VERSION,
            vendor_revision=_NO_VENDOR_REVISION,
            registry_fingerprint=COCO80_REGISTRY.fingerprint,
            label_config_fingerprint=_NATIVE_LABEL_CONTRACT,
            instance_id="coco-refinement",
            storage_subdir=f"{contract.split}2017",
            extra_fingerprints={},
        ),
        annotation_verifier=annotation_verifier,  # type: ignore[arg-type]
        inference_receipt_resolver=inference_receipt_resolver,  # type: ignore[arg-type]
    )
    if bootstrap.task_count != contract.expected_row_count:
        raise BootstrapContractError(
            "working-store task count differs from the exact source contract",
            code="coco_refinement.bootstrap_row_count",
            context={
                "split": contract.split,
                "expected": contract.expected_row_count,
                "actual": bootstrap.task_count,
            },
        )

    tasks = _project_compact_tasks(
        bootstrap.store,
        project_id=project_id,
        split=contract.split,
        expected_count=contract.expected_row_count,
    )
    project = ProjectRecord(
        project_id=project_id,
        split=contract.split,
        source_fingerprint=contract.expected_source_sha256,
        task_count=contract.expected_row_count,
    )
    repository.bootstrap_project(project, tasks)
    return SplitBootstrapResult(
        project=project,
        tasks=tasks,
        store=bootstrap.store,
        store_created=bootstrap.created,
    )


def _project_compact_tasks(
    store: WorkingDatasetStore,
    *,
    project_id: str,
    split: Split,
    expected_count: int,
) -> tuple[CompactTaskRecord, ...]:
    manifest = _load_json_object(store.manifest_path, field="project manifest")
    task_index = _load_json_object(store.task_index_path, field="task index")
    entries = task_index.get("entries")
    if (
        task_index.get("identity_schema_version") is None
        or not isinstance(entries, list)
        or len(entries) != expected_count
        or manifest.get("task_count") != expected_count
        or manifest.get("working_line_count") != expected_count
    ):
        raise BootstrapContractError(
            "store-owned compact identity index is absent or incomplete",
            code="coco_refinement.bootstrap_task_index",
            context={"split": split},
        )
    if manifest.get("extra_fingerprints") != {}:
        raise BootstrapContractError(
            "standalone bootstrap unexpectedly depends on external fingerprints",
            code="coco_refinement.bootstrap_legacy_binding",
            context={"split": split},
        )
    generation = manifest.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise BootstrapContractError(
            "working generation is invalid",
            code="coco_refinement.bootstrap_generation",
        )

    tasks: list[CompactTaskRecord] = []
    seeds = iter(store.iter_task_seeds())
    with store.working_path.open("rb") as rows:
        for source_row_index, (entry, raw_row) in enumerate(zip(entries, rows)):
            try:
                row = json.loads(raw_row)
                seed = next(seeds)
            except (json.JSONDecodeError, UnicodeDecodeError, StopIteration) as exc:
                raise BootstrapContractError(
                    "working row and store task seed projection disagree",
                    code="coco_refinement.bootstrap_projection",
                    context={"split": split, "source_row_index": source_row_index},
                    cause=exc,
                ) from exc
            if not isinstance(row, dict) or not isinstance(entry, dict):
                raise BootstrapContractError(
                    "working row or task-index entry is not an object",
                    code="coco_refinement.bootstrap_projection",
                )
            image_id = row.get("image_id")
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or image_id <= 0
                or entry.get("source_row_index") != source_row_index
                or entry.get("image_id") != image_id
                or seed.source_row_index != source_row_index
                or seed.image_id != image_id
            ):
                raise BootstrapContractError(
                    "working row, task index, and task seed identities disagree",
                    code="coco_refinement.bootstrap_projection",
                    context={"split": split, "source_row_index": source_row_index},
                )
            native_regions = [
                _native_region(value) for value in seed.annotations[0]["regions"]
            ]
            committed = canonicalize_objects(native_regions, split=split)
            tasks.append(
                CompactTaskRecord(
                    project_id=project_id,
                    identity=NativeTaskIdentity(
                        split=split,
                        image_id=image_id,
                        source_row_index=source_row_index,
                    ),
                    image_locator=_compact_image_locator(
                        seed.image_locator, split=split
                    ),
                    image_width=_positive_integer(entry.get("width"), field="width"),
                    image_height=_positive_integer(entry.get("height"), field="height"),
                    image_fingerprint=_digest(
                        entry.get("image_sha256"), field="image_sha256"
                    ),
                    current_generation=generation,
                    base_row_hash=sha256_json(row),
                    committed_result_hash=committed.result_hash,
                )
            )
        if rows.read(1):
            raise BootstrapContractError(
                "working JSONL contains rows beyond the task index",
                code="coco_refinement.bootstrap_projection",
            )
    try:
        next(seeds)
    except StopIteration:
        pass
    else:
        raise BootstrapContractError(
            "task seeds contain rows beyond the compact projection",
            code="coco_refinement.bootstrap_projection",
        )
    if len(tasks) != expected_count:
        raise BootstrapContractError(
            "working JSONL is shorter than the exact source contract",
            code="coco_refinement.bootstrap_projection",
            context={"expected": expected_count, "actual": len(tasks)},
        )
    return tuple(tasks)


def resolve_indexed_image(task: CompactTaskRecord, image_root: str | Path) -> Path:
    """Resolve only the image bound to ``task`` and re-attest its bytes/dimensions."""

    if not isinstance(task, CompactTaskRecord):
        raise BootstrapContractError(
            "task must be a CompactTaskRecord",
            code="coco_refinement.image_task",
        )
    root = Path(image_root).resolve(strict=True)
    if not root.is_dir():
        raise BootstrapContractError(
            "image_root must resolve to a directory",
            code="coco_refinement.image_root",
        )
    relative = _strict_image_relative(task.image_locator, split=task.identity.split)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise BootstrapContractError(
                "indexed image is unavailable",
                code="coco_refinement.image_missing",
                context={"task_id": task.task_id},
                cause=exc,
            ) from exc
        if stat.S_ISLNK(mode):
            raise BootstrapContractError(
                "indexed image path contains a descendant symlink",
                code="coco_refinement.image_symlink",
                context={"task_id": task.task_id},
            )
        final = index == len(relative.parts) - 1
        if (not final and not stat.S_ISDIR(mode)) or (final and not stat.S_ISREG(mode)):
            raise BootstrapContractError(
                "indexed image path has an unexpected filesystem type",
                code="coco_refinement.image_type",
                context={"task_id": task.task_id},
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise BootstrapContractError(
            "indexed image cannot be resolved",
            code="coco_refinement.image_missing",
            cause=exc,
        ) from exc
    if not resolved.is_relative_to(root):
        raise BootstrapContractError(
            "indexed image escapes the allowlisted root",
            code="coco_refinement.image_escape",
        )
    actual_hash = sha256_file(resolved)
    if actual_hash != task.image_fingerprint:
        raise BootstrapContractError(
            "indexed image bytes differ from the compact task binding",
            code="coco_refinement.image_hash",
            context={"task_id": task.task_id},
        )
    try:
        with Image.open(resolved) as image:
            actual_size = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise BootstrapContractError(
            "indexed image cannot be decoded",
            code="coco_refinement.image_decode",
            context={"task_id": task.task_id},
            cause=exc,
        ) from exc
    expected_size = (task.image_width, task.image_height)
    if actual_size != expected_size:
        raise BootstrapContractError(
            "indexed image dimensions differ from the compact task binding",
            code="coco_refinement.image_dimensions",
            context={
                "task_id": task.task_id,
                "expected": expected_size,
                "actual": actual_size,
            },
        )
    return resolved


def _native_region(value: Mapping[str, Any]) -> dict[str, Any]:
    region = {
        "region_key": value["region_key"],
        "bbox_2d": value["bbox_2d"],
        "category_name": value["category_name"],
        "category_id": value["category_id"],
        "coco_ann_id": value["coco_ann_id"],
    }
    if value.get("metadata") is not None:
        region["metadata"] = value["metadata"]
    return region


def _compact_image_locator(locator: str, *, split: Split) -> str:
    relative = PurePosixPath(locator)
    expected = ("images", f"{split}2017")
    if (
        relative.is_absolute()
        or len(relative.parts) != 3
        or relative.parts[:2] != expected
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise BootstrapContractError(
            "working image locator is outside its split allowlist",
            code="coco_refinement.image_locator",
            context={"split": split, "locator": locator},
        )
    return PurePosixPath(*relative.parts[1:]).as_posix()


def _strict_image_relative(locator: str, *, split: Split) -> PurePosixPath:
    if not isinstance(locator, str) or "\\" in locator:
        raise BootstrapContractError(
            "indexed image locator is not a canonical relative POSIX path",
            code="coco_refinement.image_locator",
        )
    relative = PurePosixPath(locator)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != f"{split}2017"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise BootstrapContractError(
            "indexed image locator is outside its split allowlist",
            code="coco_refinement.image_locator",
            context={"split": split, "locator": locator},
        )
    return relative


def _load_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BootstrapContractError(
            f"cannot read {field}",
            code="coco_refinement.bootstrap_json",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(value, dict):
        raise BootstrapContractError(
            f"{field} must be a JSON object",
            code="coco_refinement.bootstrap_json",
        )
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BootstrapContractError(
            f"{field} must be a positive integer",
            code="coco_refinement.bootstrap_task_index",
        )
    return value


def _digest(value: object, *, field: str) -> str:
    if not _is_digest(value):
        raise BootstrapContractError(
            f"{field} must be a lowercase SHA-256 digest",
            code="coco_refinement.bootstrap_task_index",
        )
    return value  # type: ignore[return-value]


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "BootstrapContractError",
    "BootstrapSourceContract",
    "DEFAULT_RUNTIME_RELATIVE",
    "SplitBootstrapResult",
    "WorkspaceBootstrapResult",
    "bootstrap_workspace",
    "production_source_contracts",
    "resolve_indexed_image",
]
