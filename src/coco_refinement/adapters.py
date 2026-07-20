"""Exact native SQLite adapters for the reusable refinement runtime/store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.common.errors import CoordExpError, RuntimeContractError
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import CanonicalDraft, DraftSnapshotBinding
from src.coco_refinement.repository import (
    RepositoryError,
    SqliteDraftRepository,
    TerminalTaskCommit,
)
from src.label_studio_coco_refinement.runtime import (
    AuthoritativeDraftSnapshot,
    DraftCatalogCapture,
    DraftCatalogRequest,
)
from src.label_studio_coco_refinement.store import (
    AuthoritativeDraftIdentity,
    BatchRequest,
    BatchResult,
    BatchStatus,
    CommitRequest,
    CommitResult,
    CommitStatus,
    DraftSaveReceipt,
    InferenceReceiptLink,
    sha256_json,
)


class AdapterContractError(RuntimeContractError):
    """A native adapter boundary cannot prove the supplied authority."""


class _ReceiptDelegate(Protocol):
    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None: ...


@dataclass(frozen=True)
class TerminalTaskReceipt:
    task_id: str
    revision: int
    retired: bool
    newer_draft_preserved: bool


@dataclass(frozen=True)
class TerminalReconciliationReceipt:
    batch_id: str
    status: BatchStatus
    generation: int
    tasks: tuple[TerminalTaskReceipt, ...]
    error: str | None


class SqliteDraftCatalog:
    """Read-transaction capture of every eligible same-project native Draft."""

    def __init__(
        self,
        repository: SqliteDraftRepository,
        *,
        current_user_id: str,
    ) -> None:
        if not isinstance(repository, SqliteDraftRepository):
            raise AdapterContractError(
                "catalog repository must be SqliteDraftRepository",
                code="coco_refinement.adapter_repository",
            )
        self.repository = repository
        self.current_user_id = _nonempty(current_user_id, field="current_user_id")

    def capture_current_user_drafts(
        self, request: DraftCatalogRequest
    ) -> DraftCatalogCapture:
        if not isinstance(request, DraftCatalogRequest):
            raise AdapterContractError(
                "catalog request has an invalid type",
                code="coco_refinement.catalog_request",
            )
        if (
            request.principal.user_id != self.current_user_id
            or request.principal.authenticated is not True
        ):
            raise AdapterContractError(
                "catalog principal differs from the fixed local operator",
                code="coco_refinement.catalog_principal",
            )
        try:
            capture = self.repository.capture_pending_draft_states(
                project_id=request.project_id,
                split=request.split,  # type: ignore[arg-type]
                task_ids=request.task_ids,
            )
        except RepositoryError as exc:
            raise AdapterContractError(
                f"cannot capture authoritative Drafts: {exc}",
                code="coco_refinement.catalog_capture",
                cause=exc,
            ) from exc
        snapshots: list[AuthoritativeDraftSnapshot] = []
        for state in capture.states:
            if state.draft is None:
                raise AdapterContractError(
                    "captured sparse Draft is absent",
                    code="coco_refinement.catalog_capture",
                )
            snapshots.append(
                state.draft.to_authoritative_snapshot(
                    DraftSnapshotBinding(
                        split=state.identity.split,
                        project_id=state.project_id,
                        image_id=state.identity.image_id,
                        task_id=state.task_id,
                        annotation_id=_annotation_id(state.task_id),
                        draft_id=_draft_id(state.task_id),
                        revision=state.revision,
                        updated_at=state.updated_at,
                        base_row_hash=state.base_row_hash,
                        observed_generation=state.current_generation,
                    )
                )
            )
        return DraftCatalogCapture(
            split=capture.split,
            project_id=capture.project_id,
            current_user_id=self.current_user_id,
            base_generation=capture.current_generation,
            snapshots=tuple(snapshots),
        )


class SqliteDraftVerifier:
    """Exact frozen-batch verifier backed by permanent task mutation history."""

    def __init__(
        self,
        repository: SqliteDraftRepository,
        *,
        current_user_id: str,
    ) -> None:
        if not isinstance(repository, SqliteDraftRepository):
            raise AdapterContractError(
                "verifier repository must be SqliteDraftRepository",
                code="coco_refinement.adapter_repository",
            )
        self.repository = repository
        self.current_user_id = _nonempty(current_user_id, field="current_user_id")

    def verify(self, identity: AuthoritativeDraftIdentity) -> bool:
        """Fail closed: the legacy identity DTO omits frozen payload/base fields."""

        return False

    def verify_batch(self, request: BatchRequest) -> bool:
        try:
            if not isinstance(request, BatchRequest):
                return False
            if (
                request.current_user_id != self.current_user_id
                or request.split not in {"train", "val"}
                or isinstance(request.base_generation, bool)
                or not isinstance(request.base_generation, int)
                or request.base_generation < 0
                or not isinstance(request.members, Sequence)
                or not request.members
            ):
                return False
            source_rows: list[int] = []
            task_ids: set[str] = set()
            commit_ids: set[str] = set()
            for member in request.members:
                if (
                    isinstance(member.source_row_index, bool)
                    or not isinstance(member.source_row_index, int)
                    or member.source_row_index < 0
                ):
                    return False
                source_rows.append(member.source_row_index)
                frozen = member.request
                if not isinstance(frozen, CommitRequest):
                    return False
                if (
                    frozen.task_id in task_ids
                    or frozen.commit_id in commit_ids
                    or frozen.commit_id
                    != f"{request.batch_id}:member:{frozen.task_id}"
                    or frozen.split != request.split
                    or frozen.task_id != f"{request.split}:{frozen.image_id}"
                    or frozen.annotation_id != _annotation_id(frozen.task_id)
                    or frozen.draft_id != _draft_id(frozen.task_id)
                    or frozen.observed_generation != request.base_generation
                ):
                    return False
                task_ids.add(frozen.task_id)
                commit_ids.add(frozen.commit_id)
                try:
                    revision = int(frozen.annotation_revision)
                except (TypeError, ValueError):
                    return False
                if str(revision) != frozen.annotation_revision or revision < 1:
                    return False
                draft = canonicalize_objects(
                    list(frozen.regions), split=frozen.split  # type: ignore[arg-type]
                )
                if (
                    draft.semantic_hash != frozen.semantic_hash
                    or draft.result_hash != frozen.result_hash
                    or tuple(draft.inference_receipts)
                    != tuple(frozen.inference_receipts)
                    or not _receipt_matches_request(frozen.draft_save, frozen)
                ):
                    return False
                if not self.repository.attest_historical_draft(
                    project_id=frozen.project_id,
                    task_id=frozen.task_id,
                    split=frozen.split,  # type: ignore[arg-type]
                    image_id=frozen.image_id,
                    source_row_index=member.source_row_index,
                    revision=revision,
                    updated_at=frozen.draft_updated_at,
                    observed_generation=frozen.observed_generation,
                    base_row_hash=frozen.base_row_hash,
                    draft=draft,
                ):
                    return False
            return source_rows == sorted(source_rows) and len(source_rows) == len(
                set(source_rows)
            )
        except (CoordExpError, TypeError, ValueError):
            return False


class SqliteInferenceReceiptResolver:
    """Strict shape guard around the existing append-only receipt store."""

    def __init__(self, receipt_store: _ReceiptDelegate) -> None:
        resolve = getattr(receipt_store, "resolve", None)
        if not callable(resolve):
            raise AdapterContractError(
                "receipt store must expose resolve(receipt_id)",
                code="coco_refinement.receipt_delegate",
            )
        self.receipt_store = receipt_store

    def resolve(self, receipt_id: str) -> InferenceReceiptLink | None:
        expected = _nonempty(receipt_id, field="receipt_id")
        resolved = self.receipt_store.resolve(expected)
        if resolved is None:
            return None
        if not isinstance(resolved, InferenceReceiptLink):
            raise AdapterContractError(
                "receipt delegate returned an invalid receipt link",
                code="coco_refinement.receipt_link",
            )
        _validate_receipt_link(resolved, expected_receipt_id=expected)
        return resolved


class SqliteTerminalReconciler:
    """Project terminal store results into sparse SQLite Draft authority."""

    def __init__(
        self,
        repository: SqliteDraftRepository,
        *,
        current_user_id: str,
    ) -> None:
        if not isinstance(repository, SqliteDraftRepository):
            raise AdapterContractError(
                "reconciler repository must be SqliteDraftRepository",
                code="coco_refinement.adapter_repository",
            )
        self.repository = repository
        self.current_user_id = _nonempty(current_user_id, field="current_user_id")
        self.verifier = SqliteDraftVerifier(
            repository, current_user_id=current_user_id
        )

    def on_batch_result(
        self, *, request: BatchRequest, result: BatchResult
    ) -> TerminalReconciliationReceipt:
        return self.reconcile_batch(request, result)

    def reconcile_existing_terminals(
        self,
        terminals: Iterable[tuple[BatchRequest, BatchResult]],
    ) -> tuple[TerminalReconciliationReceipt, ...]:
        return tuple(
            self.reconcile_batch(request, result) for request, result in terminals
        )

    def reconcile_batch(
        self, request: BatchRequest, result: BatchResult
    ) -> TerminalReconciliationReceipt:
        if not self.verifier.verify_batch(request):
            raise AdapterContractError(
                "terminal BatchRequest is not exact historical Draft authority",
                code="coco_refinement.terminal_request",
            )
        _validate_terminal_envelope(request, result)
        if result.status is BatchStatus.FAILED:
            if result.members or not isinstance(result.error, str) or not result.error:
                raise AdapterContractError(
                    "failed terminal result has an invalid member/error shape",
                    code="coco_refinement.terminal_failure",
                )
            return TerminalReconciliationReceipt(
                batch_id=result.batch_id,
                status=result.status,
                generation=result.generation,
                tasks=(),
                error=result.error,
            )

        request_by_commit = {
            member.request.commit_id: member.request for member in request.members
        }
        result_by_commit: dict[str, CommitResult] = {}
        split_wide_object_ids: set[int] = set()
        for member in result.members:
            if not isinstance(member, CommitResult) or member.commit_id in result_by_commit:
                raise AdapterContractError(
                    "successful terminal result contains invalid members",
                    code="coco_refinement.terminal_members",
                )
            if not isinstance(member.region_id_mapping, Mapping) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value == 0
                for value in member.region_id_mapping.values()
            ):
                raise AdapterContractError(
                    "terminal member region mapping is invalid",
                    code="coco_refinement.terminal_members",
                )
            member_ids = set(member.region_id_mapping.values())
            if split_wide_object_ids & member_ids:
                raise AdapterContractError(
                    "terminal members reuse a split-wide object identity",
                    code="coco_refinement.terminal_members",
                )
            split_wide_object_ids.update(member_ids)
            result_by_commit[member.commit_id] = member
        if set(result_by_commit) != set(request_by_commit):
            raise AdapterContractError(
                "terminal members differ from the immutable request",
                code="coco_refinement.terminal_members",
            )
        commits: list[TerminalTaskCommit] = []
        for batch_member in request.members:
            frozen = batch_member.request
            terminal = result_by_commit[frozen.commit_id]
            if (
                terminal.status is not CommitStatus.COMMITTED
                or terminal.split != frozen.split
                or terminal.image_id != frozen.image_id
                or terminal.generation != result.generation
                or terminal.semantic_hash != frozen.semantic_hash
            ):
                raise AdapterContractError(
                    "terminal member identity differs from its frozen request",
                    code="coco_refinement.terminal_member",
                    context={"task_id": frozen.task_id},
                )
            committed = _committed_native_draft(frozen, terminal)
            commits.append(
                TerminalTaskCommit(
                    project_id=frozen.project_id,
                    task_id=frozen.task_id,
                    split=frozen.split,  # type: ignore[arg-type]
                    image_id=frozen.image_id,
                    captured_revision=int(frozen.annotation_revision),
                    captured_generation=frozen.observed_generation,
                    captured_base_row_hash=frozen.base_row_hash,
                    captured_result_hash=frozen.result_hash,
                    committed_generation=terminal.generation,
                    committed_base_row_hash=terminal.row_hash,
                    committed_draft=committed,
                    region_id_mapping=terminal.region_id_mapping,
                )
            )
        try:
            reconciled = self.repository.reconcile_terminal_success(
                batch_id=request.batch_id,
                current_user_id=self.current_user_id,
                commits=commits,
            )
        except RepositoryError as exc:
            raise AdapterContractError(
                f"terminal SQLite reconciliation failed: {exc}",
                code="coco_refinement.terminal_reconciliation",
                cause=exc,
            ) from exc
        return TerminalReconciliationReceipt(
            batch_id=result.batch_id,
            status=result.status,
            generation=result.generation,
            tasks=tuple(
                TerminalTaskReceipt(
                    task_id=value.task_id,
                    revision=value.state.revision,
                    retired=value.retired,
                    newer_draft_preserved=value.newer_draft_preserved,
                )
                for value in reconciled.tasks
            ),
            error=None,
        )


def _receipt_matches_request(receipt: object, request: CommitRequest) -> bool:
    return isinstance(receipt, DraftSaveReceipt) and receipt == DraftSaveReceipt(
        project_id=request.project_id,
        task_id=request.task_id,
        annotation_id=request.annotation_id,
        draft_id=request.draft_id,
        annotation_revision=request.annotation_revision,
        draft_updated_at=request.draft_updated_at,
        semantic_hash=request.semantic_hash,
        result_hash=request.result_hash,
        durable=True,
    )


def _committed_native_draft(
    request: CommitRequest, result: CommitResult
) -> CanonicalDraft:
    row = result.committed_row
    if (
        not isinstance(row, Mapping)
        or row.get("image_id") != request.image_id
        or not _is_digest(result.row_hash)
        or sha256_json(row) != result.row_hash
    ):
        raise AdapterContractError(
            "committed row image identity is invalid",
            code="coco_refinement.committed_row",
        )
    objects = row.get("objects")
    mapping = result.region_id_mapping
    if not isinstance(objects, list) or not isinstance(mapping, Mapping):
        raise AdapterContractError(
            "committed row or region mapping has an invalid shape",
            code="coco_refinement.committed_row",
        )
    frozen_keys = [str(value.get("region_key")) for value in request.regions]
    if len(set(frozen_keys)) != len(frozen_keys) or set(mapping) != set(frozen_keys):
        raise AdapterContractError(
            "committed region mapping differs from frozen membership",
            code="coco_refinement.committed_mapping",
        )
    id_to_key: dict[int, str] = {}
    for key, object_id in mapping.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id == 0
            or object_id in id_to_key
        ):
            raise AdapterContractError(
                "committed region mapping is not one-to-one",
                code="coco_refinement.committed_mapping",
            )
        id_to_key[object_id] = key
    native: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw in objects:
        if not isinstance(raw, Mapping):
            raise AdapterContractError(
                "committed object is not a mapping",
                code="coco_refinement.committed_row",
            )
        object_id = raw.get("coco_ann_id")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id not in id_to_key
            or object_id in seen_ids
            or raw.get("desc") != raw.get("category_name")
        ):
            raise AdapterContractError(
                "committed object identity or class projection is invalid",
                code="coco_refinement.committed_row",
            )
        seen_ids.add(object_id)
        value: dict[str, Any] = {
            "region_key": id_to_key[object_id],
            "bbox_2d": raw.get("bbox_2d"),
            "category_name": raw.get("category_name"),
            "category_id": raw.get("category_id"),
            "coco_ann_id": object_id,
        }
        if raw.get("metadata") is not None:
            value["metadata"] = raw["metadata"]
        native.append(value)
    if seen_ids != set(id_to_key):
        raise AdapterContractError(
            "committed row and region mapping contain different object IDs",
            code="coco_refinement.committed_mapping",
        )
    try:
        return canonicalize_objects(native, split=request.split)  # type: ignore[arg-type]
    except Exception as exc:
        raise AdapterContractError(
            f"committed row cannot rebuild the native baseline: {exc}",
            code="coco_refinement.committed_row",
            cause=exc,
        ) from exc


def _validate_terminal_envelope(request: BatchRequest, result: BatchResult) -> None:
    if not isinstance(result, BatchResult):
        raise AdapterContractError(
            "terminal result has an invalid type",
            code="coco_refinement.terminal_result",
        )
    if (
        result.batch_id != request.batch_id
        or result.split != request.split
        or result.payload_hash != _batch_payload_hash(request)
        or result.status not in {BatchStatus.SUCCEEDED, BatchStatus.FAILED}
        or isinstance(result.generation, bool)
        or not isinstance(result.generation, int)
        or result.generation < request.base_generation
        or not _is_digest(result.payload_hash)
        or not _is_digest(result.working_sha256)
    ):
        raise AdapterContractError(
            "terminal result envelope differs from the immutable batch",
            code="coco_refinement.terminal_result",
        )
    if result.status is BatchStatus.SUCCEEDED and (
        result.generation != request.base_generation + 1 or result.error is not None
    ):
        raise AdapterContractError(
            "successful result does not advance exactly one generation",
            code="coco_refinement.terminal_generation",
        )
    if result.status is BatchStatus.FAILED and result.generation != request.base_generation:
        raise AdapterContractError(
            "failed result must preserve the captured generation",
            code="coco_refinement.terminal_generation",
        )


def _batch_payload_hash(request: BatchRequest) -> str:
    members = sorted(request.members, key=lambda value: value.source_row_index)
    return sha256_json(
        {
            "batch_id": request.batch_id,
            "split": request.split,
            "current_user_id": request.current_user_id,
            "base_generation": request.base_generation,
            "members": [
                {
                    "source_row_index": member.source_row_index,
                    "request": _commit_request_payload(member.request),
                }
                for member in members
            ],
        }
    )


def _commit_request_payload(request: CommitRequest) -> dict[str, Any]:
    receipt = request.draft_save
    return {
        "commit_id": request.commit_id,
        "split": request.split,
        "image_id": request.image_id,
        "project_id": request.project_id,
        "task_id": request.task_id,
        "annotation_id": request.annotation_id,
        "draft_id": request.draft_id,
        "annotation_revision": request.annotation_revision,
        "draft_updated_at": request.draft_updated_at,
        "semantic_hash": request.semantic_hash,
        "result_hash": request.result_hash,
        "base_row_hash": request.base_row_hash,
        "observed_generation": request.observed_generation,
        "regions": list(request.regions),
        "draft_save": {
            "project_id": receipt.project_id,
            "task_id": receipt.task_id,
            "annotation_id": receipt.annotation_id,
            "draft_id": receipt.draft_id,
            "annotation_revision": receipt.annotation_revision,
            "draft_updated_at": receipt.draft_updated_at,
            "semantic_hash": receipt.semantic_hash,
            "result_hash": receipt.result_hash,
            "durable": receipt.durable,
        },
        "inference_receipts": list(request.inference_receipts),
    }
def _validate_receipt_link(
    link: InferenceReceiptLink, *, expected_receipt_id: str
) -> None:
    fields = (
        link.receipt_id,
        link.request_id,
        link.project_id,
        link.task_id,
        link.annotation_id,
        link.current_user_id,
        link.draft_id,
        link.draft_revision,
        link.terminal_status,
    )
    if (
        link.receipt_id != expected_receipt_id
        or any(not isinstance(value, str) or not value for value in fields)
        or isinstance(link.image_id, bool)
        or not isinstance(link.image_id, int)
        or link.image_id <= 0
        or link.terminal_status not in {"accepted", "accepted_with_drops"}
        or not isinstance(link.result_region_keys, Mapping)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in link.result_region_keys.items()
        )
    ):
        raise AdapterContractError(
            "receipt delegate returned a malformed or mismatched receipt link",
            code="coco_refinement.receipt_link",
        )


def _annotation_id(task_id: str) -> str:
    return f"{task_id}:annotation"


def _draft_id(task_id: str) -> str:
    return f"{task_id}:draft"


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AdapterContractError(
            f"{field} must be non-empty trimmed text",
            code="coco_refinement.adapter_field",
            context={"field": field},
        )
    return value


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "AdapterContractError",
    "SqliteDraftCatalog",
    "SqliteDraftVerifier",
    "SqliteInferenceReceiptResolver",
    "SqliteTerminalReconciler",
    "TerminalReconciliationReceipt",
    "TerminalTaskReceipt",
]
