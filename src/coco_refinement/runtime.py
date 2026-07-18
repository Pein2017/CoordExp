"""Single-process lifecycle for the standalone COCO refinement workspace."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from src.common.errors import DataContractError, RuntimeContractError
from src.coco_refinement.bootstrap import (
    DEFAULT_RUNTIME_RELATIVE,
    BootstrapSourceContract,
    WorkspaceBootstrapResult,
    bootstrap_workspace,
    production_source_contracts,
    resume_workspace,
)
from src.coco_refinement.dataset_publisher import PUBLISHER_VERSION, RECEIPT_NAME
from src.coco_refinement.preflight import (
    LaunchPreflight,
    VersionResolver,
    run_launch_preflight,
)
from src.coco_refinement.repository import SqliteDraftRepository
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement.runtime import (
    BatchCoordinator,
    BatchResultObserver,
    RefinementRuntime,
)
from src.label_studio_coco_refinement.store import (
    BatchRequest,
    BatchResult,
    ValidationError,
    WorkingDatasetStore,
    _bootstrap_image_sha256,
    _parse_jsonl_line,
    _validate_selected_source,
    _validate_source_row,
    _working_image_locator,
    canonical_json,
    sha256_file,
)
from src.label_studio_coco_refinement.models import WorkingRow


HEALTH_RECEIPT_NAME = "runtime-health.json"
_SPLITS = ("train", "val")


class StandaloneRuntimeError(RuntimeContractError):
    """Base class for standalone runtime assembly and lifecycle failures."""


class SourceInspectionError(StandaloneRuntimeError):
    """One exact source failed read-only inspection before workspace creation."""


class RuntimeAssemblyError(StandaloneRuntimeError):
    """Required native adapters or lifecycle collaborators cannot be assembled."""


class RuntimeStartupError(StandaloneRuntimeError):
    """Both split workers did not become live and ready for writes."""


class RuntimeShutdownError(StandaloneRuntimeError):
    """Workers did not stop cleanly, so the root lock remains held."""


def _validate_positive_finite_timing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeAssemblyError(
            f"{name.replace('_', ' ')} must be a positive finite real number",
            code="coco_refinement.runtime_timing",
        )
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeAssemblyError(
            f"{name.replace('_', ' ')} must be a positive finite real number",
            code="coco_refinement.runtime_timing",
            cause=exc,
        ) from exc
    if normalized <= 0 or not math.isfinite(normalized):
        raise RuntimeAssemblyError(
            f"{name.replace('_', ' ')} must be a positive finite real number",
            code="coco_refinement.runtime_timing",
        )
    return normalized


class RuntimeState(str, Enum):
    INITIALIZED = "initialized"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOP_FAILED = "stop_failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class SourceInspectionReceipt:
    split: str
    source_path: Path
    source_sha256: str
    row_count: int
    image_root: Path
    authority: str = "bootstrap_source"
    bootstrap_source_sha256: str | None = None
    publication_receipt_path: Path | None = None
    publication_receipt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "authority": self.authority,
            "image_root": str(self.image_root),
            "row_count": self.row_count,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "split": self.split,
        }
        if self.bootstrap_source_sha256 is not None:
            payload["bootstrap_source_sha256"] = self.bootstrap_source_sha256
        if self.publication_receipt_path is not None:
            payload["publication_receipt_path"] = str(
                self.publication_receipt_path
            )
        if self.publication_receipt_sha256 is not None:
            payload["publication_receipt_sha256"] = (
                self.publication_receipt_sha256
            )
        return payload


@dataclass(frozen=True)
class AdapterFactories:
    """Two-phase native adapter construction around working-store bootstrap."""

    verifier: Callable[[SqliteDraftRepository], object]
    inference_receipt_resolver: Callable[[SqliteDraftRepository], object]
    catalog: Callable[
        [
            SqliteDraftRepository,
            Mapping[str, WorkingDatasetStore],
            Mapping[str, str],
        ],
        object,
    ]
    terminal_reconciler: Callable[
        [
            SqliteDraftRepository,
            Mapping[str, WorkingDatasetStore],
            Mapping[str, str],
        ],
        object,
    ]


@dataclass(frozen=True)
class NativeAdapterBindings:
    verifier: object
    inference_receipt_resolver: object
    catalog: object
    terminal_reconciler: object

    def identities(self) -> dict[str, str]:
        return {
            "catalog": _qualified_type(self.catalog),
            "inference_receipt_resolver": _qualified_type(
                self.inference_receipt_resolver
            ),
            "terminal_reconciler": _qualified_type(self.terminal_reconciler),
            "verifier": _qualified_type(self.verifier),
        }


class CoordinatorFactory(Protocol):
    def __call__(
        self,
        stores: Mapping[str, WorkingDatasetStore],
        *,
        on_batch_result: BatchResultObserver,
    ) -> object: ...


class RefinementRuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        catalog: object,
        stores: Mapping[str, WorkingDatasetStore],
        project_ids: Mapping[str, str],
        coordinator: object,
    ) -> object: ...


class TerminalPairProvider(Protocol):
    """Recover the immutable request paired with each durable terminal result."""

    def existing_pairs(
        self, *, split: str, store: WorkingDatasetStore
    ) -> Sequence[tuple[BatchRequest, BatchResult]]: ...

    def request_for_result(
        self,
        *,
        split: str,
        store: WorkingDatasetStore,
        result: BatchResult,
    ) -> BatchRequest: ...


@dataclass(frozen=True)
class StoreTerminalPairProvider:
    """Resolve terminal pairs only through the store's durable public authority."""

    def existing_pairs(
        self, *, split: str, store: WorkingDatasetStore
    ) -> tuple[tuple[BatchRequest, BatchResult], ...]:
        self._validate_store(split, store)
        pairs = store.terminal_batch_pairs()
        for request, result in pairs:
            self._validate_pair(split, request, result)
        return pairs

    def request_for_result(
        self,
        *,
        split: str,
        store: WorkingDatasetStore,
        result: BatchResult,
    ) -> BatchRequest:
        self._validate_store(split, store)
        if not isinstance(result, BatchResult) or result.split != split:
            raise RuntimeAssemblyError(
                "worker callback result differs from its split/store",
                code="coco_refinement.terminal_pair_result",
            )
        request = store.get_batch_request(result.batch_id)
        durable_result = store.get_batch_result(result.batch_id)
        if durable_result != result:
            raise RuntimeAssemblyError(
                "worker callback result differs from the durable terminal result",
                code="coco_refinement.terminal_pair_result",
                context={"batch_id": result.batch_id},
            )
        self._validate_pair(split, request, durable_result)
        return request

    @staticmethod
    def _validate_store(split: str, store: WorkingDatasetStore) -> None:
        if (
            split not in _SPLITS
            or not isinstance(store, WorkingDatasetStore)
            or store.split_dir.name != split
        ):
            raise RuntimeAssemblyError(
                "terminal pair store differs from its split",
                code="coco_refinement.terminal_pair_store",
                context={"split": split},
            )

    @staticmethod
    def _validate_pair(split: str, request: BatchRequest, result: BatchResult) -> None:
        if (
            not isinstance(request, BatchRequest)
            or not isinstance(result, BatchResult)
            or request.split != split
            or result.split != split
            or request.batch_id != result.batch_id
        ):
            raise RuntimeAssemblyError(
                "durable terminal request/result pair identity mismatch",
                code="coco_refinement.terminal_pair_identity",
            )


class _RuntimeFacade(Protocol):
    def start_workers(self, split: str | None = None) -> None: ...

    def stop_workers(
        self, split: str | None = None, *, timeout: float = 5.0
    ) -> None: ...

    def worker_health(self, split: str | None = None) -> object: ...


class StandaloneRefinementRuntime:
    """Own live workers and the sole runtime-root lock through clean shutdown."""

    def __init__(
        self,
        *,
        preflight: LaunchPreflight,
        workspace: WorkspaceBootstrapResult,
        inspections: Sequence[SourceInspectionReceipt],
        adapters: NativeAdapterBindings,
        runtime: _RuntimeFacade,
        startup_timeout: float,
        poll_interval: float,
        startup_cleanup_timeout: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        startup_timeout = _validate_positive_finite_timing(
            "startup_timeout", startup_timeout
        )
        startup_cleanup_timeout = _validate_positive_finite_timing(
            "startup_cleanup_timeout", startup_cleanup_timeout
        )
        poll_interval = _validate_positive_finite_timing(
            "poll_interval", poll_interval
        )
        if set(workspace.splits) != set(_SPLITS):
            raise RuntimeAssemblyError(
                "standalone runtime requires both train and val stores",
                code="coco_refinement.runtime_splits",
            )
        self.preflight = preflight
        self.workspace = workspace
        self.inspections = tuple(inspections)
        self.adapters = adapters
        self.runtime = runtime
        self.startup_timeout = startup_timeout
        self.startup_cleanup_timeout = startup_cleanup_timeout
        self.poll_interval = poll_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = RuntimeState.INITIALIZED
        self._accepting_writes = False
        self._lifecycle_lock = threading.RLock()
        self.health_receipt_path = workspace.runtime_root / HEALTH_RECEIPT_NAME
        self._write_health_snapshot()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def accepting_writes(self) -> bool:
        return self._accepting_writes

    @property
    def lock_held(self) -> bool:
        return self.preflight.writer_lock.acquired

    @property
    def stores(self) -> dict[str, WorkingDatasetStore]:
        return {split: self.workspace.splits[split].store for split in _SPLITS}

    @property
    def project_ids(self) -> dict[str, str]:
        return {
            split: self.workspace.splits[split].project.project_id for split in _SPLITS
        }

    def start(self) -> dict[str, Any]:
        """Start exactly two workers and admit writes only after both are ready."""

        with self._lifecycle_lock:
            if self._state is RuntimeState.READY:
                return self.health_snapshot()
            if self._state is RuntimeState.STOPPED or not self.lock_held:
                raise RuntimeStartupError(
                    "a released standalone runtime cannot be restarted",
                    code="coco_refinement.runtime_released",
                )
            if self._state in {RuntimeState.STARTING, RuntimeState.STOPPING}:
                raise RuntimeStartupError(
                    "runtime lifecycle transition is already active",
                    code="coco_refinement.runtime_transition",
                )
            self._state = RuntimeState.STARTING
            self._accepting_writes = False
            self._write_health_snapshot()
            try:
                self.runtime.start_workers()
                deadline = self._monotonic() + self.startup_timeout
                while True:
                    workers = self._live_worker_health()
                    if _workers_ready(workers):
                        self._state = RuntimeState.READY
                        self._accepting_writes = True
                        return self._write_health_snapshot(workers=workers)
                    if _workers_failed(workers):
                        raise RuntimeStartupError(
                            "a split worker failed before becoming ready",
                            code="coco_refinement.worker_startup",
                            context={"workers": workers},
                        )
                    if self._monotonic() >= deadline:
                        raise RuntimeStartupError(
                            "split workers did not become ready before timeout",
                            code="coco_refinement.worker_startup_timeout",
                            context={"workers": workers},
                        )
                    self._sleep(self.poll_interval)
            except BaseException as exc:
                self._accepting_writes = False
                cleanup_error = self._stop_workers_and_verify(
                    self.startup_cleanup_timeout
                )
                if cleanup_error is None:
                    self._state = RuntimeState.STOPPED
                    self._write_health_snapshot(lock_held=False)
                    self.preflight.release()
                else:
                    self._state = RuntimeState.STOP_FAILED
                    self._write_health_snapshot(health_error=str(cleanup_error))
                if isinstance(exc, RuntimeStartupError):
                    raise
                raise RuntimeStartupError(
                    "worker startup failed",
                    code="coco_refinement.worker_startup",
                    cause=exc,
                ) from exc

    def shutdown(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """Stop workers before releasing the root lock; safe to call repeatedly."""

        if timeout <= 0:
            raise RuntimeShutdownError(
                "shutdown timeout must be positive",
                code="coco_refinement.shutdown_timeout",
            )
        with self._lifecycle_lock:
            if not self.lock_held:
                self._state = RuntimeState.STOPPED
                self._accepting_writes = False
                return self.health_snapshot()
            self._accepting_writes = False
            self._state = RuntimeState.STOPPING
            self._write_health_snapshot()
            cleanup_error = self._stop_workers_and_verify(timeout)
            if cleanup_error is not None:
                self._state = RuntimeState.STOP_FAILED
                self._write_health_snapshot(health_error=str(cleanup_error))
                raise RuntimeShutdownError(
                    "workers remain live; runtime-root lock retained",
                    code="coco_refinement.worker_stop",
                    cause=cleanup_error,
                ) from cleanup_error
            self._state = RuntimeState.STOPPED
            final = self._write_health_snapshot(lock_held=False)
            self.preflight.release()
            return final

    def health_snapshot(self) -> dict[str, Any]:
        """Return current worker health; never substitute a prior disk receipt."""

        try:
            workers = self._live_worker_health()
        except BaseException as exc:
            return self._snapshot(
                workers=None, health_error=f"{type(exc).__name__}: {exc}"
            )
        return self._snapshot(workers=workers)

    def __enter__(self) -> StandaloneRefinementRuntime:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown()

    def _stop_workers_and_verify(self, timeout: float) -> BaseException | None:
        try:
            self.runtime.stop_workers(timeout=timeout)
            workers = self._live_worker_health()
            live = {
                split: health
                for split, health in workers.items()
                if health.get("thread_alive") is not False
            }
            if live:
                return RuntimeShutdownError(
                    "worker stop returned while a thread remained live",
                    code="coco_refinement.worker_still_live",
                    context={"workers": live},
                )
        except BaseException as exc:
            return exc
        return None

    def _live_worker_health(self) -> dict[str, dict[str, Any]]:
        raw = self.runtime.worker_health()
        if not isinstance(raw, Mapping) or set(raw) != set(_SPLITS):
            raise RuntimeAssemblyError(
                "worker health must cover exactly train and val",
                code="coco_refinement.worker_health_shape",
            )
        normalized: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            value = raw[split]
            if hasattr(value, "to_dict") and callable(value.to_dict):
                value = value.to_dict()
            if not isinstance(value, Mapping):
                raise RuntimeAssemblyError(
                    "worker health entry must be a mapping or expose to_dict",
                    code="coco_refinement.worker_health_shape",
                    context={"split": split},
                )
            entry = dict(value)
            if entry.get("split") != split:
                raise RuntimeAssemblyError(
                    "worker health split identity mismatch",
                    code="coco_refinement.worker_health_split",
                    context={"split": split},
                )
            _assert_json_safe(entry, field=f"worker health {split}")
            normalized[split] = entry
        return normalized

    def _snapshot(
        self,
        *,
        workers: Mapping[str, Mapping[str, Any]] | None,
        health_error: str | None = None,
        lock_held: bool | None = None,
    ) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "kind": "coco_refinement_runtime_health",
            "state": self._state.value,
            "accepting_writes": self._accepting_writes,
            "lock_held": self.lock_held if lock_held is None else lock_held,
            "runtime_root": str(self.workspace.runtime_root),
            "project_ids": self.project_ids,
            "adapters": self.adapters.identities(),
            "sources": [receipt.to_dict() for receipt in self.inspections],
            "workers": None if workers is None else dict(workers),
            "health_error": health_error,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _assert_json_safe(value, field="runtime health")
        return value

    def _write_health_snapshot(
        self,
        *,
        workers: Mapping[str, Mapping[str, Any]] | None = None,
        health_error: str | None = None,
        lock_held: bool | None = None,
    ) -> dict[str, Any]:
        if workers is None and health_error is None:
            try:
                workers = self._live_worker_health()
            except BaseException as exc:
                health_error = f"{type(exc).__name__}: {exc}"
        snapshot = self._snapshot(
            workers=workers,
            health_error=health_error,
            lock_held=lock_held,
        )
        _atomic_write_json(self.health_receipt_path, snapshot)
        return snapshot


def inspect_source_contracts_read_only(
    contracts: Sequence[BootstrapSourceContract],
) -> tuple[SourceInspectionReceipt, ...]:
    """Fully read both exact JSONL files without creating mutable workspace state."""

    selected = _validate_dual_contracts(contracts)
    receipts: list[SourceInspectionReceipt] = []
    for contract in selected:
        try:
            source = contract.source_path.resolve(strict=True)
            image_root = contract.image_root.resolve(strict=True)
        except OSError as exc:
            raise SourceInspectionError(
                "source or image root cannot be resolved",
                code="coco_refinement.source_inspection_path",
                context={"split": contract.split},
                cause=exc,
            ) from exc
        if not source.is_file() or not image_root.is_dir():
            raise SourceInspectionError(
                "source must be a file and image root must be a directory",
                code="coco_refinement.source_inspection_type",
                context={"split": contract.split},
            )
        try:
            _validate_selected_source(source, contract.split)
        except ValidationError as exc:
            raise SourceInspectionError(
                "source path differs from the approved max_len12000 contract",
                code="coco_refinement.source_inspection_path",
                context={"split": contract.split},
                cause=exc,
            ) from exc
        digest = hashlib.sha256()
        row_count = 0
        split_image_ids: set[int] = set()
        split_object_ids: dict[int, int] = {}
        try:
            with source.open("rb") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    digest.update(raw_line)
                    try:
                        row = _parse_jsonl_line(
                            raw_line.decode("utf-8"), source, line_no
                        )
                        _validate_source_row(row, contract.split, COCO80_REGISTRY)
                        image_id = row["image_id"]
                        if (
                            isinstance(image_id, bool)
                            or not isinstance(image_id, int)
                            or image_id <= 0
                            or image_id in split_image_ids
                        ):
                            raise ValidationError(
                                f"split-wide duplicate or invalid image_id: {image_id}"
                            )
                        split_image_ids.add(image_id)
                        for obj in row["objects"]:
                            object_id = int(obj["coco_ann_id"])
                            prior_image = split_object_ids.setdefault(
                                object_id, image_id
                            )
                            if prior_image != image_id:
                                raise ValidationError(
                                    f"split-wide duplicate coco_ann_id: {object_id}"
                                )
                        working_locator = _working_image_locator(
                            row, contract.split, source, image_root
                        )
                        if working_locator != row.get("file_name"):
                            raise ValidationError(
                                "source image locator and file_name disagree"
                            )
                        image_path = _strict_source_image_path(
                            row, split=contract.split, image_root=image_root
                        )
                        _bootstrap_image_sha256(
                            row, split=contract.split, image_root=image_root
                        )
                        width = row.get("width")
                        height = row.get("height")
                        if (
                            isinstance(width, bool)
                            or not isinstance(width, int)
                            or width <= 0
                            or isinstance(height, bool)
                            or not isinstance(height, int)
                            or height <= 0
                        ):
                            raise ValidationError(
                                "source image dimensions must be positive integers"
                            )
                        with Image.open(image_path) as image:
                            actual_dimensions = image.size
                        if actual_dimensions != (width, height):
                            raise ValidationError(
                                "source dimensions differ from the bound image"
                            )
                    except SourceInspectionError:
                        raise
                    except (
                        UnicodeDecodeError,
                        ValidationError,
                        RuntimeContractError,
                        OSError,
                        UnidentifiedImageError,
                    ) as exc:
                        raise SourceInspectionError(
                            "source row or image failed semantic inspection",
                            code="coco_refinement.source_inspection_semantics",
                            context={"split": contract.split, "line": line_no},
                            cause=exc,
                        ) from exc
                    row_count += 1
        except OSError as exc:
            raise SourceInspectionError(
                "source cannot be read",
                code="coco_refinement.source_inspection_read",
                context={"split": contract.split},
                cause=exc,
            ) from exc
        source_hash = digest.hexdigest()
        if source_hash != contract.expected_source_sha256:
            raise SourceInspectionError(
                "source fingerprint differs from the approved contract",
                code="coco_refinement.source_inspection_hash",
                context={"split": contract.split},
            )
        if row_count != contract.expected_row_count:
            raise SourceInspectionError(
                "source row count differs from the approved contract",
                code="coco_refinement.source_inspection_count",
                context={
                    "split": contract.split,
                    "expected": contract.expected_row_count,
                    "actual": row_count,
                },
            )
        receipts.append(
            SourceInspectionReceipt(
                split=contract.split,
                source_path=source,
                source_sha256=source_hash,
                row_count=row_count,
                image_root=image_root,
            )
        )
    return tuple(receipts)


def inspect_source_contracts_for_resume(
    contracts: Sequence[BootstrapSourceContract],
    runtime_root: str | Path,
) -> tuple[SourceInspectionReceipt, ...]:
    """Attest an existing runtime against bootstrap or published source bytes.

    Published training files are never used to rebuild the store.  Their only
    accepted drift is an exact terminal-generation publisher receipt whose
    working and journal authorities still match the existing runtime.
    """

    selected = _validate_dual_contracts(contracts)
    selected_runtime = Path(runtime_root).resolve(strict=True)
    actual_hashes = {
        contract.split: sha256_file(contract.source_path.resolve(strict=True))
        for contract in selected
    }
    if all(
        actual_hashes[contract.split] == contract.expected_source_sha256
        for contract in selected
    ):
        return inspect_source_contracts_read_only(selected)

    receipts: list[SourceInspectionReceipt] = []
    for contract in selected:
        source = contract.source_path.resolve(strict=True)
        image_root = contract.image_root.resolve(strict=True)
        actual_hash = actual_hashes[contract.split]
        if actual_hash == contract.expected_source_sha256:
            row_count = _validate_resume_rows(
                source,
                split=contract.split,
                image_root=image_root,
                allow_negative=False,
            )[0]
            if row_count != contract.expected_row_count:
                raise SourceInspectionError(
                    "bootstrap source row count differs during resume",
                    code="coco_refinement.resume_source_count",
                    context={"split": contract.split},
                )
            receipts.append(
                SourceInspectionReceipt(
                    split=contract.split,
                    source_path=source,
                    source_sha256=actual_hash,
                    row_count=row_count,
                    image_root=image_root,
                )
            )
            continue
        receipts.append(
            _inspect_published_iteration(
                contract,
                runtime_root=selected_runtime,
                source=source,
                source_sha256=actual_hash,
                image_root=image_root,
            )
        )
    return tuple(receipts)


def _inspect_published_iteration(
    contract: BootstrapSourceContract,
    *,
    runtime_root: Path,
    source: Path,
    source_sha256: str,
    image_root: Path,
) -> SourceInspectionReceipt:
    split_root = runtime_root / contract.split
    transaction_path = split_root / ".training.publish.transaction.json"
    if transaction_path.exists():
        raise SourceInspectionError(
            "training publication transaction requires recovery before resume",
            code="coco_refinement.resume_publication_transaction",
            context={"split": contract.split},
        )
    receipt_path = split_root / RECEIPT_NAME
    manifest_path = split_root / "project.json"
    working_path = split_root / "working.norm.jsonl"
    journal_path = split_root / "journal.jsonl"
    coord_path = source.with_name(f"{contract.split}.coord.jsonl")
    receipt = _read_json_artifact(receipt_path, field="training publication receipt")
    manifest = _read_json_artifact(manifest_path, field="runtime project manifest")
    outputs = receipt.get("outputs")
    norm_output = outputs.get("norm") if isinstance(outputs, Mapping) else None
    coord_output = outputs.get("coord") if isinstance(outputs, Mapping) else None
    working = receipt.get("working")
    identity = receipt.get("identity_authority")
    token_budget = receipt.get("token_budget")
    loader = receipt.get("loader_attestation")
    norm_schema = receipt.get("norm_schema_attestation")
    images = receipt.get("images")
    expected = (
        receipt.get("schema_version") == 2
        and receipt.get("code")
        == "coco_refinement.committed_generation_published"
        and receipt.get("publisher_version") == PUBLISHER_VERSION
        and receipt.get("split") == contract.split
        and receipt.get("runtime_root") == str(runtime_root)
        and receipt.get("generation") == manifest.get("generation")
        and receipt.get("row_count") == contract.expected_row_count
        and isinstance(norm_output, Mapping)
        and norm_output.get("path") == str(source)
        and norm_output.get("sha256") == source_sha256
        and isinstance(coord_output, Mapping)
        and coord_output.get("path") == str(coord_path)
        and coord_output.get("sha256") == sha256_file(coord_path)
        and isinstance(working, Mapping)
        and working.get("path") == str(working_path)
        and working.get("sha256") == manifest.get("working_sha256")
        and working.get("sha256") == sha256_file(working_path)
        and isinstance(identity, Mapping)
        and identity.get("journal_path") == str(journal_path)
        and identity.get("journal_sha256") == sha256_file(journal_path)
        and identity.get("status") == "passed"
        and isinstance(token_budget, Mapping)
        and token_budget.get("status") == "passed"
        and token_budget.get("max_total_tokens") == 12000
        and isinstance(loader, Mapping)
        and loader.get("status") == "passed"
        and isinstance(norm_schema, Mapping)
        and norm_schema.get("status") == "passed"
        and isinstance(images, Mapping)
        and images.get("copied") is False
    )
    if not expected:
        raise SourceInspectionError(
            "published iteration receipt does not match runtime and target authority",
            code="coco_refinement.resume_publication_receipt",
            context={"split": contract.split},
        )
    row_count, object_count, negative_count, normalized_sha256 = _validate_resume_rows(
        source,
        split=contract.split,
        image_root=image_root,
        allow_negative=True,
    )
    if (
        row_count != contract.expected_row_count
        or receipt.get("object_count") != object_count
        or identity.get("negative_object_count") != negative_count
        or normalized_sha256 != manifest.get("working_sha256")
    ):
        raise SourceInspectionError(
            "published iteration inventory differs from its receipt",
            code="coco_refinement.resume_publication_inventory",
            context={"split": contract.split},
        )
    return SourceInspectionReceipt(
        split=contract.split,
        source_path=source,
        source_sha256=source_sha256,
        row_count=row_count,
        image_root=image_root,
        authority="published_iteration",
        bootstrap_source_sha256=contract.expected_source_sha256,
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=sha256_file(receipt_path),
    )


def _validate_resume_rows(
    source: Path,
    *,
    split: str,
    image_root: Path,
    allow_negative: bool,
) -> tuple[int, int, int, str]:
    row_count = 0
    object_count = 0
    negative_count = 0
    seen_image_ids: set[int] = set()
    normalized_digest = hashlib.sha256()
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = _parse_jsonl_line(line, source, line_no)
            if allow_negative:
                try:
                    normalized_row = dict(row)
                    normalized_row["images"] = [
                        _working_image_locator(row, split, source, image_root)
                    ]
                    working = WorkingRow.from_mapping(
                        normalized_row, field=f"published[{line_no}]"
                    ).validate_for_split(split)  # type: ignore[arg-type]
                except (DataContractError, RuntimeContractError, ValidationError) as exc:
                    raise SourceInspectionError(
                        "published iteration row is not a valid working row",
                        code="coco_refinement.resume_publication_semantics",
                        context={"split": split, "line": line_no},
                        cause=exc,
                    ) from exc
                objects = working.objects
                image_id = working.image_id
                locator = str(row["images"][0])
                normalized_payload = working.to_json_dict(coord_tokens=False)
            else:
                try:
                    _validate_source_row(row, split, COCO80_REGISTRY)
                except ValidationError as exc:
                    raise SourceInspectionError(
                        "bootstrap source semantics drifted during resume",
                        code="coco_refinement.resume_source_semantics",
                        context={"split": split, "line": line_no},
                        cause=exc,
                    ) from exc
                objects = row["objects"]
                image_id = int(row["image_id"])
                locator = str(row["images"][0])
                normalized_payload = row
            if image_id in seen_image_ids:
                raise SourceInspectionError(
                    "resume source contains duplicate image identity",
                    code="coco_refinement.resume_source_identity",
                    context={"split": split, "image_id": image_id},
                )
            seen_image_ids.add(image_id)
            resolved_image = (source.parent / locator).resolve(strict=True)
            try:
                resolved_image.relative_to(image_root)
            except ValueError as exc:
                raise SourceInspectionError(
                    "resume source image locator escapes the shared root",
                    code="coco_refinement.resume_source_image",
                    context={"split": split, "line": line_no},
                    cause=exc,
                ) from exc
            object_count += len(objects)
            negative_count += sum(
                1
                for obj in objects
                if int(
                    obj.coco_ann_id if hasattr(obj, "coco_ann_id") else obj["coco_ann_id"]
                )
                < 0
            )
            normalized_digest.update(
                (canonical_json(normalized_payload) + "\n").encode("utf-8")
            )
            row_count += 1
    return row_count, object_count, negative_count, normalized_digest.hexdigest()


def _read_json_artifact(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceInspectionError(
            f"{field} is unavailable or invalid",
            code="coco_refinement.resume_publication_receipt",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(value, dict):
        raise SourceInspectionError(
            f"{field} must be a JSON object",
            code="coco_refinement.resume_publication_receipt",
            context={"path": str(path)},
        )
    return value


def _strict_source_image_path(
    row: Mapping[str, Any], *, split: str, image_root: Path
) -> Path:
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or "\\" in file_name:
        raise ValidationError("source file_name must be a relative POSIX path")
    relative = Path(file_name)
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("images", f"{split}2017")
        or len(relative.parts) != 3
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValidationError("source file_name is outside its split image root")
    current = image_root
    for index, part in enumerate(relative.parts[1:]):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as exc:
            raise ValidationError(f"source image is unavailable: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError("source image path contains a symlink")
        final = index == len(relative.parts[1:]) - 1
        if (not final and not stat.S_ISDIR(mode)) or (final and not stat.S_ISREG(mode)):
            raise ValidationError("source image path has an unexpected file type")
    return current


def production_adapter_factories(
    *,
    inference_receipt_store: object,
    current_user_id: str = "local-operator",
) -> AdapterFactories:
    """Load the native SQLite adapters lazily and fail closed if unavailable."""

    try:
        from src.coco_refinement.adapters import (
            SqliteDraftCatalog,
            SqliteDraftVerifier,
            SqliteInferenceReceiptResolver,
            SqliteTerminalReconciler,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeAssemblyError(
            "native SQLite adapters are unavailable",
            code="coco_refinement.adapters_unavailable",
            cause=exc,
        ) from exc

    return AdapterFactories(
        verifier=lambda repository: SqliteDraftVerifier(
            repository, current_user_id=current_user_id
        ),
        inference_receipt_resolver=lambda _repository: SqliteInferenceReceiptResolver(
            inference_receipt_store
        ),
        catalog=lambda repository, _stores, _project_ids: SqliteDraftCatalog(
            repository, current_user_id=current_user_id
        ),
        terminal_reconciler=lambda repository,
        _stores,
        _project_ids: SqliteTerminalReconciler(
            repository, current_user_id=current_user_id
        ),
    )


def create_standalone_runtime(
    repo_root: str | Path,
    *,
    runtime_root: str | Path | None = None,
    source_contracts: Sequence[BootstrapSourceContract] | None = None,
    adapter_factories: AdapterFactories | None = None,
    inference_receipt_store: object | None = None,
    inference_receipt_store_factory: Callable[[], object] | None = None,
    current_user_id: str = "local-operator",
    terminal_pair_provider: TerminalPairProvider | None = None,
    source_inspector: Callable[
        [Sequence[BootstrapSourceContract]], Sequence[SourceInspectionReceipt]
    ] = inspect_source_contracts_read_only,
    resume_source_inspector: Callable[
        [Sequence[BootstrapSourceContract], str | Path],
        Sequence[SourceInspectionReceipt],
    ] = inspect_source_contracts_for_resume,
    repository_factory: Callable[[Path], SqliteDraftRepository] = SqliteDraftRepository,
    workspace_bootstrapper: Callable[
        ..., WorkspaceBootstrapResult
    ] = bootstrap_workspace,
    workspace_resumer: Callable[..., WorkspaceBootstrapResult] = resume_workspace,
    coordinator_factory: CoordinatorFactory | None = None,
    refinement_runtime_factory: RefinementRuntimeFactory = RefinementRuntime,
    preflight_runner: Callable[..., LaunchPreflight] = run_launch_preflight,
    reload: bool = False,
    workers: int = 1,
    environment: Mapping[str, str] | None = None,
    version_resolver: VersionResolver | None = None,
    startup_timeout: float = 5.0,
    startup_cleanup_timeout: float = 5.0,
    poll_interval: float = 0.01,
) -> StandaloneRefinementRuntime:
    """Assemble a locked dual-split runtime without binding an HTTP port."""

    startup_timeout = _validate_positive_finite_timing(
        "startup_timeout", startup_timeout
    )
    startup_cleanup_timeout = _validate_positive_finite_timing(
        "startup_cleanup_timeout", startup_cleanup_timeout
    )
    poll_interval = _validate_positive_finite_timing("poll_interval", poll_interval)

    root = Path(repo_root).resolve(strict=True)
    selected_runtime = (
        root / DEFAULT_RUNTIME_RELATIVE if runtime_root is None else Path(runtime_root)
    ).resolve()
    selected_contracts = _validate_dual_contracts(
        production_source_contracts(root)
        if source_contracts is None
        else source_contracts
    )
    factories = adapter_factories
    if factories is None:
        if (inference_receipt_store is None) == (
            inference_receipt_store_factory is None
        ):
            raise RuntimeAssemblyError(
                "production adapters require exactly one inference receipt store source",
                code="coco_refinement.receipt_store",
            )
    elif inference_receipt_store is not None or inference_receipt_store_factory is not None:
        raise RuntimeAssemblyError(
            "explicit adapter factories cannot also receive an inference receipt store",
            code="coco_refinement.receipt_store",
        )
    selected_pair_provider = terminal_pair_provider or StoreTerminalPairProvider()
    if not callable(
        getattr(selected_pair_provider, "existing_pairs", None)
    ) or not callable(getattr(selected_pair_provider, "request_for_result", None)):
        raise RuntimeAssemblyError(
            "terminal_pair_provider lacks durable pair lookup methods",
            code="coco_refinement.terminal_pair_provider",
        )
    selected_coordinator_factory = coordinator_factory or _default_coordinator_factory

    preflight = preflight_runner(
        selected_runtime,
        reload=reload,
        workers=workers,
        environment=environment,
        version_resolver=version_resolver,
    )
    try:
        if factories is None:
            receipt_store = (
                inference_receipt_store
                if inference_receipt_store_factory is None
                else inference_receipt_store_factory()
            )
            factories = production_adapter_factories(
                inference_receipt_store=receipt_store,
                current_user_id=current_user_id,
            )
        split_manifests = tuple(
            selected_runtime / split / "project.json" for split in _SPLITS
        )
        existing_count = sum(path.is_file() for path in split_manifests)
        if existing_count not in (0, len(_SPLITS)):
            raise RuntimeAssemblyError(
                "runtime root contains only one split manifest",
                code="coco_refinement.resume_partial_runtime",
            )
        resuming = existing_count == len(_SPLITS)
        inspections = tuple(
            resume_source_inspector(selected_contracts, selected_runtime)
            if resuming
            else source_inspector(selected_contracts)
        )
        _validate_inspection_receipts(inspections, selected_contracts)

        repository = repository_factory(selected_runtime / "state.sqlite3")
        verifier = factories.verifier(repository)
        receipt_resolver = factories.inference_receipt_resolver(repository)
        workspace_factory = workspace_resumer if resuming else workspace_bootstrapper
        workspace = workspace_factory(
            root,
            runtime_root=selected_runtime,
            source_contracts=selected_contracts,
            repository=repository,
            annotation_verifier=verifier,
            inference_receipt_resolver=receipt_resolver,
            attest_repository=False,
        )
        if set(workspace.splits) != set(_SPLITS):
            raise RuntimeAssemblyError(
                "workspace bootstrap did not return both train and val",
                code="coco_refinement.runtime_splits",
            )
        if workspace.repository is not repository:
            raise RuntimeAssemblyError(
                "workspace bootstrap replaced the locked SQLite authority",
                code="coco_refinement.runtime_repository",
            )
        stores = {split: workspace.splits[split].store for split in _SPLITS}
        project_ids = {
            split: workspace.splits[split].project.project_id for split in _SPLITS
        }
        catalog = factories.catalog(repository, stores, project_ids)
        terminal_reconciler = factories.terminal_reconciler(
            repository, stores, project_ids
        )
        reconcile_existing = getattr(
            terminal_reconciler, "reconcile_existing_terminals", None
        )
        reconcile_batch = getattr(terminal_reconciler, "reconcile_batch", None)
        if not callable(reconcile_existing) or not callable(reconcile_batch):
            raise RuntimeAssemblyError(
                "terminal reconciler lacks its exact batch reconciliation contract",
                code="coco_refinement.terminal_reconcile",
            )
        for split in _SPLITS:
            stores[split].recover()
            pairs = selected_pair_provider.existing_pairs(
                split=split, store=stores[split]
            )
            reconcile_existing(pairs)
        # Store recovery is the generation authority.  A durable terminal may
        # have published generation N+1 immediately before the process died,
        # while SQLite still projects N.  Replay those exact terminal pairs
        # before comparing the recovered compact rows with SQLite; the strict
        # bootstrap attestation below then rejects every unexplained drift.
        for split in _SPLITS:
            projection = workspace.splits[split]
            repository.bootstrap_project(projection.project, projection.tasks)

        def on_batch_result(
            *,
            split: str,
            store: WorkingDatasetStore,
            request: BatchRequest,
            result: BatchResult,
        ) -> None:
            if request.split != split:
                raise RuntimeAssemblyError(
                    "terminal observer request split mismatch",
                    code="coco_refinement.terminal_reconcile",
                )
            reconcile_batch(request, result)

        coordinator = selected_coordinator_factory(
            stores, on_batch_result=on_batch_result
        )
        runtime = refinement_runtime_factory(
            catalog=catalog,
            stores=stores,
            project_ids=project_ids,
            coordinator=coordinator,
        )
        bindings = NativeAdapterBindings(
            verifier=verifier,
            inference_receipt_resolver=receipt_resolver,
            catalog=catalog,
            terminal_reconciler=terminal_reconciler,
        )
        return StandaloneRefinementRuntime(
            preflight=preflight,
            workspace=workspace,
            inspections=inspections,
            adapters=bindings,
            runtime=runtime,  # type: ignore[arg-type]
            startup_timeout=startup_timeout,
            startup_cleanup_timeout=startup_cleanup_timeout,
            poll_interval=poll_interval,
        )
    except BaseException:
        preflight.release()
        raise


def _default_coordinator_factory(
    stores: Mapping[str, WorkingDatasetStore],
    *,
    on_batch_result: BatchResultObserver,
) -> object:
    try:
        return BatchCoordinator(  # type: ignore[call-arg]
            stores,
            on_batch_result=on_batch_result,
        )
    except TypeError as exc:
        raise RuntimeAssemblyError(
            "BatchCoordinator lacks the required on_batch_result callback contract",
            code="coco_refinement.coordinator_callback",
            cause=exc,
        ) from exc


def _validate_dual_contracts(
    contracts: Sequence[BootstrapSourceContract],
) -> tuple[BootstrapSourceContract, BootstrapSourceContract]:
    selected = tuple(contracts)
    if len(selected) != 2 or {contract.split for contract in selected} != set(_SPLITS):
        raise SourceInspectionError(
            "standalone runtime requires exact train and val source contracts",
            code="coco_refinement.source_inspection_splits",
        )
    by_split = {contract.split: contract for contract in selected}
    return by_split["train"], by_split["val"]


def _validate_inspection_receipts(
    receipts: Sequence[SourceInspectionReceipt],
    contracts: Sequence[BootstrapSourceContract],
) -> None:
    if len(receipts) != 2 or {receipt.split for receipt in receipts} != set(_SPLITS):
        raise SourceInspectionError(
            "source inspector must attest both train and val",
            code="coco_refinement.source_inspection_receipts",
        )
    by_split = {receipt.split: receipt for receipt in receipts}
    for contract in contracts:
        receipt = by_split[contract.split]
        authority_matches = (
            receipt.authority == "bootstrap_source"
            and receipt.source_sha256 == contract.expected_source_sha256
        ) or (
            receipt.authority == "published_iteration"
            and receipt.bootstrap_source_sha256
            == contract.expected_source_sha256
            and receipt.publication_receipt_path is not None
            and receipt.publication_receipt_sha256 is not None
        )
        if (
            not authority_matches
            or receipt.row_count != contract.expected_row_count
            or receipt.source_path != contract.source_path.resolve(strict=True)
            or receipt.image_root != contract.image_root.resolve(strict=True)
        ):
            raise SourceInspectionError(
                "source inspection receipt does not match its contract",
                code="coco_refinement.source_inspection_receipts",
                context={"split": contract.split},
            )


def _workers_ready(workers: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        value.get("thread_alive") is True
        and value.get("healthy") is True
        and value.get("state") in {"idle", "running"}
        for value in workers.values()
    )


def _workers_failed(workers: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(value.get("state") == "failed" for value in workers.values())


def _qualified_type(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _assert_json_safe(value: object, *, field: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeAssemblyError(
            f"{field} is not JSON-safe",
            code="coco_refinement.runtime_json",
            cause=exc,
        ) from exc


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "AdapterFactories",
    "HEALTH_RECEIPT_NAME",
    "NativeAdapterBindings",
    "RuntimeAssemblyError",
    "RuntimeShutdownError",
    "RuntimeStartupError",
    "RuntimeState",
    "SourceInspectionError",
    "SourceInspectionReceipt",
    "StoreTerminalPairProvider",
    "StandaloneRefinementRuntime",
    "StandaloneRuntimeError",
    "TerminalPairProvider",
    "create_standalone_runtime",
    "inspect_source_contracts_read_only",
    "inspect_source_contracts_for_resume",
    "production_adapter_factories",
]
