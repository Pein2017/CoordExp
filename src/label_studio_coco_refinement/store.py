"""Application interface retained for the separately versioned Label Studio repo.

The deployed ``coordexp_refinement`` application imports these names and invokes
bootstrap without a task-index binder. New CoordExp callers use the defining
``working_dataset`` owners directly. This adapter supplies only Label Studio's
bootstrap policy; every storage, publication and recovery method is inherited
unchanged from the UI-neutral core. Retire it only with an explicit migration of
that external consumer, not as an unused internal compatibility alias.
"""

from __future__ import annotations

from src.working_dataset.codec import canonical_json, semantic_hash, sha256_json
from src.working_dataset.records import (
    AuthoritativeAnnotationVerifier,
    AuthoritativeDraftIdentity,
    BatchMember,
    BatchRequest,
    BatchStatus,
    BootstrapResult,
    BootstrapSpec,
    CategoryRegistry,
    CommitConflictError,
    CommitRequest,
    DraftRestore,
    DraftSaveReceipt,
    FaultInjector,
    InferenceReceiptResolver,
    ManifestDriftError,
    RecoveryError,
    StaleCommitError,
    StoreBusyError,
    StoreError,
    ValidationError,
)
from src.working_dataset.store import WorkingDatasetStore as CoreWorkingDatasetStore

from .project import bind_bootstrap_task_index_fingerprints


class WorkingDatasetStore(CoreWorkingDatasetStore):
    """Bind the Label Studio bootstrap policy, not a second dataset authority."""

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
        return super().bootstrap(
            spec,
            annotation_verifier=annotation_verifier,
            inference_receipt_resolver=inference_receipt_resolver,
            registry=registry,
            fault_injector=fault_injector,
            task_index_binder=bind_bootstrap_task_index_fingerprints,
        )


__all__ = [
    "AuthoritativeDraftIdentity",
    "BatchMember",
    "BatchRequest",
    "BatchStatus",
    "BootstrapSpec",
    "CommitConflictError",
    "CommitRequest",
    "DraftRestore",
    "DraftSaveReceipt",
    "ManifestDriftError",
    "RecoveryError",
    "StaleCommitError",
    "StoreBusyError",
    "StoreError",
    "ValidationError",
    "WorkingDatasetStore",
    "canonical_json",
    "semantic_hash",
    "sha256_json",
]
