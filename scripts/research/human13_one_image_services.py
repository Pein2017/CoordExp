"""Production-shaped Task-5 lifecycle owner for the one-image all-HF vertical.

The backend is injectable so prelaunch tests remain model/GPU/network inert,
but this owner is the concrete :class:`OneImageServices` implementation: it
owns append-only lost-reservation recovery, phase receipts, the split CUDA
proposal lifecycle, private-checkpoint containment, unconditional one-shot
rollback, Source reproduction, cleanup, and durable terminal publication.
Scientific objective math remains in the admitted Task-2/Task-3 owners and
``CudaHFVerticalAdapter``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
import errno
import json
import os
from pathlib import Path
import shutil
from typing import Any, Protocol, cast, runtime_checkable

from scripts.research.human13_cuda_cpu_adapter import (
    CudaHFVerticalAdapter,
)
from scripts.research.human13_hf_native_one_image_owner import (
    HFNativeAdmissionRequest,
    HFNativeOneImageAdmission,
    HFNativeOneImageOwner,
    HFNativeOneImageOwnerError,
    PreAcquisitionSourceOwners,
    SourceOwnerRequest,
)
from scripts.research.run_human13_all_hf_shared_surface_vertical import (
    DualGPUResourceReceipt,
    EntryConfig,
    OneImageTerminalReceipt,
    SourceAssemblyReceipt,
    acquired_h_owner_ids_from_trajectory,
)
from src.artifacts.json_values import json_sha256


RECOVERY_SCHEMA = "human13_one_image_reservation_recovery.v1"
RESERVATION_SCHEMA = "human13_one_image_run_reservation.v1"
PHASE_SCHEMA = "human13_one_image_phase.v1"
TERMINAL_ENVELOPE_SCHEMA = "human13_one_image_terminal_envelope.v1"
FIXED_RETRY_CEILING = 1
PhaseWriter = Callable[[Path, Mapping[str, Any]], None]


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(path) from error
        raise
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class ProductionAcquisition:
    """Admitted acquisition plus the exact Task-2/Task-3 CUDA proposal input."""

    parity_passed: bool
    trusted_h_owner_ids: tuple[str, ...]
    cuda_proposal_input: object
    task2_resource_sha256: str
    trajectory_ledger_sha256: str
    compiler_ledger_sha256: str
    sample_forward_count: int = 0
    replay_forward_count: int = 0

    def __post_init__(self) -> None:
        for field in (
            "task2_resource_sha256",
            "trajectory_ledger_sha256",
            "compiler_ledger_sha256",
        ):
            _digest(getattr(self, field), field)
        if any(
            not isinstance(owner, str) or not owner
            for owner in self.trusted_h_owner_ids
        ):
            raise ValueError("trusted H owner IDs must be nonempty strings")
        for field in ("sample_forward_count", "replay_forward_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")


Task5RuntimeFactory = Callable[..., ProductionAcquisition]


class Task5RuntimeEvidenceError(RuntimeError):
    """The real Task-2/Task-3 proposal evidence is absent or unadmitted."""


@dataclass(frozen=True)
class AdmittedTask5RuntimeEvidence:
    """Live graph-bearing owners required by the repository default factory."""

    trajectory_ledger: object
    compiler_ledger: object
    compiler_compact_logits: object | None
    witness_bank: object
    realized_margin_probe: Callable[[], Mapping[str, float]]

    def __post_init__(self) -> None:
        if not callable(self.realized_margin_probe):
            raise Task5RuntimeEvidenceError(
                "admitted Task2/Task3 runtime evidence lacks a margin probe"
            )


@dataclass(frozen=True)
class Task5ProductionContextRequest:
    """Exact live objects that need admitted Task-2/Task-3 owner evidence.

    This request deliberately keeps graph-bearing values as object references.
    The repository provider may only return evidence owned by these exact live
    objects; it must not reconstruct publications or objective math.
    """

    assembly: object
    session: object
    sampled_groups: tuple[object, ...]
    replay_groups: tuple[object, ...]
    replay_logprob_tensors: Mapping[str, object]
    manifest: object
    manifest_image: object
    config: EntryConfig

    def __post_init__(self) -> None:
        if len(self.sampled_groups) != 4 or len(self.replay_groups) != 4:
            raise Task5RuntimeEvidenceError(
                "production context requires four exact sampled/replayed groups"
            )
        if not self.replay_logprob_tensors:
            raise Task5RuntimeEvidenceError(
                "production context requires live replay-logprob tensors"
            )
        if getattr(self.assembly, "model", None) is None:
            raise Task5RuntimeEvidenceError("production context lacks live model")
        if getattr(self.assembly, "optimizer", None) is None:
            raise Task5RuntimeEvidenceError("production context lacks live optimizer")
        if self.config.image_id != 1584:
            raise Task5RuntimeEvidenceError("production context image differs from 1584")
        if getattr(self.manifest_image, "image_id", None) != 1584:
            raise Task5RuntimeEvidenceError("production manifest image differs from 1584")

    @property
    def model(self) -> object:
        return getattr(self.assembly, "model")

    @property
    def optimizer(self) -> object:
        return getattr(self.assembly, "optimizer")


@dataclass(frozen=True)
class Task5ProductionContextFailureReceipt:
    """Value receipt for the one legitimate unresolved live-owner seam."""

    reason_code: str
    config_sha256: str
    manifest_sha256: str
    source_identity_sha256: str
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    sampled_group_object_ids: tuple[int, ...]
    replay_group_object_ids: tuple[int, ...]
    replay_tensor_object_ids: tuple[tuple[str, int], ...]
    assembly_object_id: int
    session_object_id: int
    model_object_id: int
    optimizer_object_id: int
    manifest_object_id: int
    manifest_image_object_id: int
    process_id: int
    missing_owner_publications: tuple[str, ...] = (
        "hf_one_image_trajectory_credit_admission",
        "pre_acquisition_source_compiler_graph",
        "pre_acquisition_frozen_witness_and_realized_probe",
    )
    required_owner_phase_order: tuple[str, ...] = (
        "source_audits",
        "source_compiler_and_witness_freeze",
        "hf_sample_and_replay",
        "one_image_trajectory_credit_admission",
    )
    schema_version: str = "human13_task5_production_context_failure.v2"

    def __post_init__(self) -> None:
        if self.reason_code != "live_task2_owner_publications_unavailable":
            raise ValueError("production context failure reason differs")
        if self.missing_owner_publications != (
            "hf_one_image_trajectory_credit_admission",
            "pre_acquisition_source_compiler_graph",
            "pre_acquisition_frozen_witness_and_realized_probe",
        ):
            raise ValueError("production context missing-owner set differs")
        if self.required_owner_phase_order != (
            "source_audits",
            "source_compiler_and_witness_freeze",
            "hf_sample_and_replay",
            "one_image_trajectory_credit_admission",
        ):
            raise ValueError("production context owner phase order differs")
        for field in (
            "config_sha256",
            "manifest_sha256",
            "source_identity_sha256",
        ):
            _digest(getattr(self, field), field)
        if len(self.sampled_group_sha256s) != 4:
            raise ValueError("context failure must bind four sampled groups")
        if len(self.replay_group_sha256s) != 4:
            raise ValueError("context failure must bind four replay groups")
        if len(self.sampled_group_object_ids) != 4:
            raise ValueError("context failure must bind four sampled group objects")
        if len(self.replay_group_object_ids) != 4:
            raise ValueError("context failure must bind four replay group objects")
        for index, digest in enumerate(self.sampled_group_sha256s):
            _digest(digest, f"sampled_group_sha256s[{index}]")
        for index, digest in enumerate(self.replay_group_sha256s):
            _digest(digest, f"replay_group_sha256s[{index}]")
        if not self.replay_tensor_object_ids:
            raise ValueError("context failure must bind replay tensors")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id <= 0
            for name, object_id in self.replay_tensor_object_ids
        ):
            raise ValueError("replay tensor object identities are malformed")
        object_ids = (
            *self.sampled_group_object_ids,
            *self.replay_group_object_ids,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in object_ids
        ):
            raise ValueError("group object identities are malformed")
        for field in (
            "assembly_object_id",
            "session_object_id",
            "model_object_id",
            "optimizer_object_id",
            "manifest_object_id",
            "manifest_image_object_id",
            "process_id",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reason_code": self.reason_code,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "sampled_group_sha256s": list(self.sampled_group_sha256s),
            "replay_group_sha256s": list(self.replay_group_sha256s),
            "sampled_group_object_ids": list(self.sampled_group_object_ids),
            "replay_group_object_ids": list(self.replay_group_object_ids),
            "replay_tensor_object_ids": [
                [name, object_id]
                for name, object_id in self.replay_tensor_object_ids
            ],
            "assembly_object_id": self.assembly_object_id,
            "session_object_id": self.session_object_id,
            "model_object_id": self.model_object_id,
            "optimizer_object_id": self.optimizer_object_id,
            "manifest_object_id": self.manifest_object_id,
            "manifest_image_object_id": self.manifest_image_object_id,
            "process_id": self.process_id,
            "missing_owner_publications": list(self.missing_owner_publications),
            "required_owner_phase_order": list(self.required_owner_phase_order),
        }

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return self._payload() | {"content_sha256": self.content_sha256}


class Task5ProductionContextUnavailable(Task5RuntimeEvidenceError):
    """The repo lacks exact admitted live publications for this acquisition."""

    def __init__(self, receipt: Task5ProductionContextFailureReceipt) -> None:
        self.receipt = receipt
        super().__init__(
            f"{receipt.reason_code}: {receipt.content_sha256}"
        )


@dataclass(frozen=True)
class Task5ObservedAcquisitionFailureReceipt:
    """Closed shared-surface observation for a post-replay context failure."""

    context_failure_receipt_sha256: str
    shared_surface_resource_receipt_sha256: str
    config_sha256: str
    manifest_sha256: str
    source_identity_sha256: str
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    sample_forward_count: int
    replay_forward_count: int
    source_owner_forward_count: int
    total_forward_count: int
    no_cache_forward_count: int
    model_object_id: int
    cleanup_state: str
    cleanup_reason: str
    cleanup_failures: tuple[str, ...]
    cleanup_call_count: int
    schema_version: str = "human13_task5_observed_acquisition_failure.v2"

    def __post_init__(self) -> None:
        for field in (
            "context_failure_receipt_sha256",
            "shared_surface_resource_receipt_sha256",
            "config_sha256",
            "manifest_sha256",
            "source_identity_sha256",
        ):
            _digest(getattr(self, field), field)
        if len(self.sampled_group_sha256s) != 4:
            raise ValueError("observed failure must bind four sampled groups")
        if len(self.replay_group_sha256s) != 4:
            raise ValueError("observed failure must bind four replay groups")
        for index, digest in enumerate(self.sampled_group_sha256s):
            _digest(digest, f"sampled_group_sha256s[{index}]")
        for index, digest in enumerate(self.replay_group_sha256s):
            _digest(digest, f"replay_group_sha256s[{index}]")
        for field in (
            "sample_forward_count",
            "replay_forward_count",
            "source_owner_forward_count",
            "total_forward_count",
            "no_cache_forward_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.total_forward_count != (
            self.sample_forward_count
            + self.replay_forward_count
            + self.source_owner_forward_count
        ):
            raise ValueError("observed shared-surface forward counts differ")
        if self.no_cache_forward_count != self.total_forward_count:
            raise ValueError("every observed shared-surface forward must be no-cache")
        if (
            isinstance(self.model_object_id, bool)
            or not isinstance(self.model_object_id, int)
            or self.model_object_id <= 0
        ):
            raise ValueError("model_object_id must be a positive integer")
        if self.cleanup_state != "closed":
            raise ValueError("shared-surface cleanup must be terminally closed")
        if self.cleanup_reason not in ("completed", "failed"):
            raise ValueError("shared-surface cleanup reason differs")
        if any(not isinstance(item, str) or not item for item in self.cleanup_failures):
            raise ValueError("cleanup failures must be nonempty strings")
        if self.cleanup_reason == "completed" and self.cleanup_failures:
            raise ValueError("completed cleanup cannot contain failures")
        if self.cleanup_call_count != 1:
            raise ValueError("shared-surface cleanup must execute exactly once")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_failure_receipt_sha256": self.context_failure_receipt_sha256,
            "shared_surface_resource_receipt_sha256": (
                self.shared_surface_resource_receipt_sha256
            ),
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "sampled_group_sha256s": list(self.sampled_group_sha256s),
            "replay_group_sha256s": list(self.replay_group_sha256s),
            "sample_forward_count": self.sample_forward_count,
            "replay_forward_count": self.replay_forward_count,
            "source_owner_forward_count": self.source_owner_forward_count,
            "total_forward_count": self.total_forward_count,
            "no_cache_forward_count": self.no_cache_forward_count,
            "model_object_id": self.model_object_id,
            "cleanup_state": self.cleanup_state,
            "cleanup_reason": self.cleanup_reason,
            "cleanup_failures": list(self.cleanup_failures),
            "cleanup_call_count": self.cleanup_call_count,
        }

    @property
    def content_sha256(self) -> str:
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return self._payload() | {"content_sha256": self.content_sha256}


class _SharedSurfaceTrainingCloseError(RuntimeError):
    def __init__(self, resource_receipt: object, cause: BaseException) -> None:
        self.resource_receipt = resource_receipt
        super().__init__(f"{type(cause).__name__}: {cause}")


@runtime_checkable
class Task5ProductionContextProvider(Protocol):
    def provide(
        self, request: Task5ProductionContextRequest
    ) -> AdmittedTask5RuntimeEvidence: ...


class RepositoryTask5ProductionContextProvider:
    """Repository-owned fail-closed seam for not-yet-produced live owners.

    ``HFSharedSurfaceSession`` owns graph-bearing sampled/replayed tensors, but
    it does not own admitted trajectory-credit publications, the durable Source
    compiler panel, or the frozen witness/realized-probe surface.  Converting
    its groups into those owners here would change research meaning, so the
    public path emits an exact typed receipt until a real owner publishes them.
    """

    @staticmethod
    def _group_hashes(groups: Sequence[object], *, label: str) -> tuple[str, ...]:
        result = tuple(getattr(group, "content_sha256", None) for group in groups)
        if len(result) != 4 or any(not isinstance(value, str) for value in result):
            raise Task5RuntimeEvidenceError(
                f"{label} groups lack admitted content identities"
            )
        return tuple(_digest(value, f"{label}_group_sha256") for value in result)

    def provide(
        self, request: Task5ProductionContextRequest
    ) -> AdmittedTask5RuntimeEvidence:
        config = request.config
        manifest_sha256 = config.manifest_sha256
        if manifest_sha256 is None:
            raise Task5RuntimeEvidenceError(
                "production context lacks frozen manifest identity"
            )
        receipt = Task5ProductionContextFailureReceipt(
            reason_code="live_task2_owner_publications_unavailable",
            config_sha256=config.content_sha256,
            manifest_sha256=manifest_sha256,
            source_identity_sha256=json_sha256(
                {
                    "source_checkpoint_path": config.source_checkpoint_path,
                    "base_model_path": config.base_model_path,
                    "adapter_path": config.adapter_path,
                    "special_embedding_path": config.special_embedding_path,
                    "source_adapter_sha256": config.source_adapter_sha256,
                    "special_embedding_sha256": config.special_embedding_sha256,
                }
            ),
            sampled_group_sha256s=self._group_hashes(
                request.sampled_groups, label="sampled"
            ),
            replay_group_sha256s=self._group_hashes(
                request.replay_groups, label="replay"
            ),
            sampled_group_object_ids=tuple(
                id(group) for group in request.sampled_groups
            ),
            replay_group_object_ids=tuple(
                id(group) for group in request.replay_groups
            ),
            replay_tensor_object_ids=tuple(
                (name, id(tensor))
                for name, tensor in sorted(request.replay_logprob_tensors.items())
            ),
            assembly_object_id=id(request.assembly),
            session_object_id=id(request.session),
            model_object_id=id(request.model),
            optimizer_object_id=id(request.optimizer),
            manifest_object_id=id(request.manifest),
            manifest_image_object_id=id(request.manifest_image),
            process_id=os.getpid(),
        )
        raise Task5ProductionContextUnavailable(receipt)


def default_task5_runtime_factory(
    *,
    assembly: object,
    session: object,
    sampled_groups: Sequence[object],
    replay_groups: Sequence[object],
    replay_logprob_tensors: Mapping[str, object],
    manifest: object,
    manifest_image: object,
    config: object,
    runtime_evidence: AdmittedTask5RuntimeEvidence | None = None,
) -> ProductionAcquisition:
    """Compose existing admitted owners; never synthesize objective evidence."""

    del manifest, config, session
    evidence = runtime_evidence
    if type(evidence) is not AdmittedTask5RuntimeEvidence:
        raise Task5RuntimeEvidenceError(
            "explicit admitted Task2/Task3 runtime evidence is required"
        )
    from scripts.research import human13_greedy_compiler as compiler_owner
    from scripts.research.human13_adamw_proposal_preservation import (
        FrozenWitnessBank,
        ProposalBinding,
    )
    from scripts.research.human13_all_hf_vertical import ARM_ID, UNIT_ID
    from scripts.research.human13_cuda_cpu_adapter import (
        CudaProposalInput,
        compute_cuda_objective_binding_sha256,
    )
    from scripts.research.human13_hf_shared_surface import (
        GradientReplayGroup,
        SampledHFGroup,
    )
    from scripts.research.human13_training_transaction import (
        TrainingStateTransaction,
        UpdateCounter,
    )
    import torch
    from scripts.research.human13_trajectory_credit import (
        _require_scientific_ledger_admission,
    )

    sampled = tuple(sampled_groups)
    replayed = tuple(replay_groups)
    if (
        len(sampled) != 4
        or len(replayed) != 4
        or any(type(group) is not SampledHFGroup for group in sampled)
        or any(type(group) is not GradientReplayGroup for group in replayed)
    ):
        raise Task5RuntimeEvidenceError(
            "admitted Task2/Task3 runtime evidence requires four exact live groups"
        )
    sampled_live = cast(tuple[SampledHFGroup, ...], sampled)
    replayed_live = cast(tuple[GradientReplayGroup, ...], replayed)
    replay_tensors = cast(Mapping[str, torch.Tensor], replay_logprob_tensors)
    trajectory = _require_scientific_ledger_admission(evidence.trajectory_ledger)
    compiler = compiler_owner._require_compiler_admission(evidence.compiler_ledger)
    witness = evidence.witness_bank
    if type(witness) is not FrozenWitnessBank or not witness.constraints:
        raise Task5RuntimeEvidenceError(
            "admitted Task2/Task3 runtime evidence lacks preservation witnesses"
        )
    model = getattr(assembly, "model", None)
    optimizer = getattr(assembly, "optimizer", None)
    if model is None or optimizer is None:
        raise Task5RuntimeEvidenceError(
            "admitted Task2/Task3 runtime evidence lacks live model ownership"
        )
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise Task5RuntimeEvidenceError("live Task-5 trainable surface is empty")
    counter = UpdateCounter()
    transaction = TrainingStateTransaction(
        named,
        optimizer=optimizer,
        scheduler=getattr(assembly, "scheduler", None),
        update_counter=counter,
        runtime=getattr(assembly, "runtime", None),
        capture_cuda=True,
    )
    identity = sampled_live[0].identity
    placeholder = ProposalBinding(
        unit_id=UNIT_ID,
        arm_id=ARM_ID,
        training_rp="1.0",
        seed_group="35001..35016",
        source_checkpoint_sha256=trajectory.source_sha256,
        manifest_sha256=trajectory.manifest_sha256,
        objective_ledger_sha256="0" * 64,
    )
    surface = CudaProposalInput(
        model=model,
        named_trainable_parameters=named,
        optimizer=optimizer,
        transaction=transaction,
        update_counter=counter,
        objective=None,
        witness_bank=witness,
        proposal_binding=placeholder,
        realized_margin_probe=evidence.realized_margin_probe,
        surface_identity=identity,
        sampled_groups=sampled_live,
        replay_groups=replayed_live,
        replay_logprob_tensors=dict(replay_tensors),
        trajectory_ledger=trajectory,
        compiler_ledger=compiler,
        compiler_compact_logits=cast(Any, evidence.compiler_compact_logits),
    )
    surface = replace(
        surface,
        proposal_binding=replace(
            placeholder,
            objective_ledger_sha256=compute_cuda_objective_binding_sha256(surface),
        ),
    )
    sample_forwards = sum(len(group.active_batch_steps) for group in sampled_live)
    replay_forwards = sample_forwards
    return ProductionAcquisition(
        parity_passed=True,
        trusted_h_owner_ids=acquired_h_owner_ids_from_trajectory(
            trajectory,
            manifest_image,
        ),
        cuda_proposal_input=surface,
        task2_resource_sha256=json_sha256(
            {
                "sampled_group_sha256s": [
                    group.content_sha256 for group in sampled_live
                ],
                "replay_group_sha256s": [
                    group.content_sha256 for group in replayed_live
                ],
                "sample_forward_count": sample_forwards,
                "replay_forward_count": replay_forwards,
            }
        ),
        trajectory_ledger_sha256=trajectory.content_sha256,
        compiler_ledger_sha256=compiler.content_sha256,
        sample_forward_count=sample_forwards,
        replay_forward_count=replay_forwards,
    )


@runtime_checkable
class SplitCudaAdapter(Protocol):
    def apply_private_proposal(self) -> object: ...

    def rollback_private_proposal(self) -> object: ...


@runtime_checkable
class ProductionOneImageBackend(Protocol):
    """Real-owner boundary; tests may inject value/CPU implementations."""

    def preflight_source_assembly(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> SourceAssemblyReceipt: ...

    def open_training(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object: ...

    def open_audit(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object: ...

    def source_audit(
        self, session: object, /, repetition_penalty: float
    ) -> Mapping[str, Any]: ...

    def acquire_and_replay(
        self, session: object, /, config: EntryConfig
    ) -> ProductionAcquisition: ...

    def build_cuda_adapter(self, proposal_input: object) -> SplitCudaAdapter: ...

    def write_private_checkpoint(
        self, session: object, proposal: object, output_root: Path, /
    ) -> object: ...

    def proposal_audit(
        self, session: object, private: object, repetition_penalty: float, /
    ) -> Mapping[str, Any]: ...

    def reproduce_source(
        self, session: object, repetition_penalties: tuple[float, float], /
    ) -> Mapping[float, Mapping[str, Any]]: ...

    def cleanup_private_checkpoint(self, private: object, /) -> None: ...

    def close_training(self, session: object, /) -> object | None: ...

    def close_audit(self, session: object, /) -> None: ...


@dataclass(frozen=True)
class _LiveTrainingHandle:
    assembly: Any
    session: Any


@dataclass(frozen=True)
class _LiveAuditHandle:
    config: EntryConfig
    manifest: OneImageAuditManifest


@dataclass(frozen=True)
class OneImageAuditManifest:
    """One-image evaluator envelope retaining immutable parent binding."""

    binding: object
    images: tuple[object, ...]
    parent_manifest_sha256: str

    def __post_init__(self) -> None:
        _digest(self.parent_manifest_sha256, "parent_manifest_sha256")
        if len(self.images) != 1 or getattr(self.images[0], "image_id", None) != 1584:
            raise ValueError("audit manifest must contain only image 1584")


@dataclass(frozen=True)
class _PrivateCheckpoint:
    checkpoint_path: str
    checkpoint_payload_sha256: str


class ExistingOwnersProductionBackend:
    """Compose admitted live owners around an explicit Task-5 runtime factory.

    The runtime factory receives the exact sampled/replayed live objects and
    must return the existing Task-2/Task-3 ledgers, witness, and proposal input
    in a :class:`ProductionAcquisition`.  This lifecycle owner never rebuilds
    objective or projection math.
    """

    def __init__(
        self,
        *,
        manifest: object,
        manifest_path: str | Path,
        repo_root: str | Path,
        source_config_path: str | Path,
        runtime_factory: Task5RuntimeFactory,
        runtime_context_provider: Task5ProductionContextProvider | None = None,
        hf_native_owner: HFNativeOneImageOwner | None = None,
        assemble_model: Callable[..., object] | None = None,
        build_skeletons: Callable[..., Mapping[int, object]] | None = None,
        open_surface: Callable[..., object] | None = None,
        evaluate_checkpoint: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
        checkpoint_writer_factory: Callable[[str | Path], object] | None = None,
        checkpoint_write: Callable[[object, object], object] | None = None,
        checkpoint_readback: Callable[..., object] | None = None,
        checkpoint_hasher: Callable[[str | Path], str] | None = None,
        device_scope: Callable[[int], Any] | None = None,
    ) -> None:
        if not callable(runtime_factory):
            raise TypeError("Task-5 runtime factory must be callable")
        self._manifest = manifest
        self._manifest_path = Path(manifest_path).expanduser().resolve()
        self._repo_root = Path(repo_root).expanduser().resolve()
        self._source_config_path = Path(source_config_path).expanduser().resolve()
        self._runtime_factory = runtime_factory
        self._runtime_context_provider = (
            runtime_context_provider or RepositoryTask5ProductionContextProvider()
        )
        if not isinstance(
            self._runtime_context_provider, Task5ProductionContextProvider
        ):
            raise TypeError("Task-5 production context provider must be callable")
        if hf_native_owner is not None and not isinstance(
            hf_native_owner, HFNativeOneImageOwner
        ):
            raise TypeError("HF-native one-image owner must implement the public seam")
        self._hf_native_owner = hf_native_owner
        self._active_training_handle: _LiveTrainingHandle | None = None
        self._source_owner_audits: dict[float, Mapping[str, Any]] = {}
        self._pre_acquisition_source: PreAcquisitionSourceOwners | None = None
        live_training_boundary = assemble_model is None
        live_audit_boundary = evaluate_checkpoint is None
        if assemble_model is None or build_skeletons is None:
            from scripts.research.human13_live_model import (
                assemble_human13_live_model,
                build_human13_processor_skeletons,
            )

            assemble_model = assemble_model or assemble_human13_live_model
            build_skeletons = build_skeletons or build_human13_processor_skeletons
        if open_surface is None:
            from scripts.research.human13_hf_shared_surface_live import (
                open_hf_shared_surface,
            )

            open_surface = open_hf_shared_surface
        if evaluate_checkpoint is None or checkpoint_hasher is None:
            from scripts.research.human13_live_eval import (
                checkpoint_payload_sha256,
                evaluate_hf_checkpoint,
            )

            evaluate_checkpoint = evaluate_checkpoint or evaluate_hf_checkpoint
            checkpoint_hasher = checkpoint_hasher or checkpoint_payload_sha256
        if checkpoint_writer_factory is None or checkpoint_readback is None:
            from scripts.research.human13_live_model import (
                build_human13_checkpoint_writer,
                readback_human13_checkpoint,
            )

            checkpoint_writer_factory = (
                checkpoint_writer_factory or build_human13_checkpoint_writer
            )
            checkpoint_readback = checkpoint_readback or readback_human13_checkpoint
        self._assemble_model: Any = assemble_model
        self._build_skeletons: Any = build_skeletons
        self._open_surface: Any = open_surface
        self._evaluate_checkpoint: Any = evaluate_checkpoint
        self._checkpoint_writer_factory: Any = checkpoint_writer_factory
        self._checkpoint_write: Any = checkpoint_write or self._write_checkpoint
        self._checkpoint_readback: Any = checkpoint_readback
        self._checkpoint_hasher: Any = checkpoint_hasher
        production_scope = device_scope or self._cuda_device_scope
        self._training_device_scope = (
            production_scope if live_training_boundary else lambda _index: nullcontext()
        )
        self._audit_device_scope = (
            production_scope if live_audit_boundary else lambda _index: nullcontext()
        )

    @staticmethod
    def _cuda_device_scope(index: int) -> Any:
        import torch

        return torch.cuda.device(index)

    @staticmethod
    def _source_plan() -> Any:
        from scripts.research.human13_live_model import (
            build_human13_all_hf_vertical_source_plan,
        )

        return build_human13_all_hf_vertical_source_plan()

    @staticmethod
    def _write_checkpoint(writer: object, assembly: Any) -> object:
        from scripts.research.human13_live_model import build_human13_checkpoint_kwargs

        write = getattr(writer, "write_checkpoint", None)
        if not callable(write):
            raise TypeError("checkpoint writer lacks write_checkpoint")
        return write(
            step=1,
            model=getattr(assembly, "model"),
            **build_human13_checkpoint_kwargs(assembly),
        )

    def _image(self) -> object:
        selected = tuple(
            item
            for item in tuple(getattr(self._manifest, "images", ()))
            if getattr(item, "image_id", None) == 1584
        )
        if len(selected) != 1:
            raise ValueError("manifest must contain exactly one image-1584 record")
        return selected[0]

    def preflight_source_assembly(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> SourceAssemblyReceipt:
        plan = self._source_plan()
        source = getattr(plan, "source")
        binding = getattr(self._manifest, "binding", None)
        panel = getattr(binding, "panel", None)
        surface = getattr(binding, "surface", None)
        image = self._image()
        manifest_sha = hashlib_sha256(self._manifest_path.read_bytes())
        source_checkpoint = config.source_checkpoint_path
        manifest_config_sha = config.manifest_sha256
        if source_checkpoint is None or manifest_config_sha is None:
            raise ValueError(
                "entry config lacks Source checkpoint or manifest identity"
            )
        if manifest_sha != manifest_config_sha:
            raise ValueError("loaded manifest differs from the frozen config digest")
        expected = (
            (getattr(source, "checkpoint_path"), config.source_checkpoint_path),
            (getattr(source, "base_model_path"), config.base_model_path),
            (getattr(source, "adapter_path"), config.adapter_path),
            (getattr(source, "special_embedding_path"), config.special_embedding_path),
            (getattr(source, "adapter_sha256"), config.source_adapter_sha256),
            (
                getattr(source, "special_embedding_sha256"),
                config.special_embedding_sha256,
            ),
        )
        if any(
            left is None or right is None or left != right for left, right in expected
        ):
            raise ValueError("Source plan and entry config identity differ")
        base = SourceAssemblyReceipt(
            source_plan_sha256=config.content_sha256,
            training_gpu=resources.training_gpu,
            audit_gpu=resources.audit_gpu,
            training_surface="bf16/flash_attention_2",
            audit_surface="fp32/sdpa/batch1",
            checkpoint_path=config.source_checkpoint_path,
            base_model_path=config.base_model_path,
            adapter_path=config.adapter_path,
            special_embedding_path=config.special_embedding_path,
            adapter_sha256=config.source_adapter_sha256,
            special_embedding_sha256=config.special_embedding_sha256,
            checkpoint_sha256=self._checkpoint_hasher(source_checkpoint),
            tokenizer_sha256=getattr(surface, "tokenizer_sha256", None),
            prompt_policy_fingerprint=getattr(
                surface, "prompt_policy_fingerprint", None
            ),
            panel_sha256=getattr(panel, "panel_sha256", None),
            image_sha256=getattr(image, "image_sha256", None),
            manifest_sha256=manifest_sha,
            source_validation_sha256="0" * 64,
            assembly_receipt_sha256="0" * 64,
        )
        return replace(
            base,
            source_validation_sha256=base.source_validation_content_sha256,
            assembly_receipt_sha256=base.assembly_receipt_content_sha256,
        )

    def open_training(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> _LiveTrainingHandle:
        del config, resources
        plan = self._source_plan()
        with self._training_device_scope(0):
            assembly = self._assemble_model(
                plan, pack_count=1, repo_root=self._repo_root
            )
            skeletons = self._build_skeletons(
                self._manifest,
                getattr(assembly, "components"),
                repo_root=self._repo_root,
            )
            skeleton = skeletons.get(1584)
            if skeleton is None:
                raise ValueError("processor skeletons omit image 1584")
            from scripts.research.human13_hf_shared_surface import plan_image1584_k16

            live = self._open_surface(plan_image1584_k16(), assembly, skeleton)
        handle = _LiveTrainingHandle(assembly=assembly, session=live)
        self._active_training_handle = handle
        return handle

    def open_audit(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> _LiveAuditHandle:
        del resources
        if config.manifest_sha256 is None:
            raise ValueError("entry config lacks manifest identity")
        return _LiveAuditHandle(
            config=config,
            manifest=OneImageAuditManifest(
                binding=getattr(self._manifest, "binding"),
                images=(self._image(),),
                parent_manifest_sha256=config.manifest_sha256,
            ),
        )

    def _audit(
        self,
        session: _LiveAuditHandle,
        *,
        checkpoint_path: str | Path,
        arm_id: str,
        milestone: int,
        repetition_penalty: float,
    ) -> Mapping[str, Any]:
        config = session.config
        manifest_sha = config.manifest_sha256
        if manifest_sha is None:
            raise ValueError("entry config lacks manifest identity")
        with self._audit_device_scope(1):
            outputs = tuple(
                self._evaluate_checkpoint(
                    manifest=session.manifest,
                    manifest_sha256=manifest_sha,
                    checkpoint_path=checkpoint_path,
                    arm_id=arm_id,
                    milestone=milestone,
                    run_id=f"task5-{arm_id}",
                    run_root=str(Path(checkpoint_path).expanduser().resolve()),
                    resolved_arm_plan_sha256=config.content_sha256,
                    resolved_config_sha256=config.content_sha256,
                    source_config_path=self._source_config_path,
                    repetition_penalty=repetition_penalty,
                )
            )
        selected = tuple(row for row in outputs if row.get("image_id") == 1584)
        if len(selected) != 1:
            raise ValueError("HF audit must return exactly one image-1584 row")
        return selected[0]

    def source_audit(
        self, session: object, repetition_penalty: float
    ) -> Mapping[str, Any]:
        if not isinstance(session, _LiveAuditHandle):
            raise TypeError("Source audit requires the live audit handle")
        checkpoint = session.config.source_checkpoint_path
        if checkpoint is None:
            raise ValueError("Source checkpoint path is absent")
        result = self._audit(
            session,
            checkpoint_path=checkpoint,
            arm_id="frozen_source",
            milestone=0,
            repetition_penalty=repetition_penalty,
        )
        owner = self._hf_native_owner
        if owner is not None and self._pre_acquisition_source is None:
            if repetition_penalty in self._source_owner_audits:
                raise ValueError("Source owner audit RP was observed more than once")
            self._source_owner_audits[repetition_penalty] = result
            if tuple(self._source_owner_audits) == (1.0, 1.1):
                training = self._active_training_handle
                if training is None:
                    raise RuntimeError("Source owner lacks the active training session")
                try:
                    self._pre_acquisition_source = owner.prepare_source(
                        SourceOwnerRequest(
                            assembly=training.assembly,
                            session=training.session,
                            manifest=self._manifest,
                            manifest_image=self._image(),
                            config=session.config,
                            source_audits=dict(self._source_owner_audits),
                        )
                    )
                except BaseException as error:
                    # The RP=1.1 audit has already performed a real GPU1
                    # forward.  Preserve that observation on the primary
                    # owner error so the services/entry owners can receipt
                    # it even though source-owner preparation failed.
                    for name, value in (
                        ("_source_audit_forward_observed", True),
                        ("_source_audit_result", result),
                        ("_source_audit_repetition_penalty", repetition_penalty),
                    ):
                        try:
                            setattr(error, name, value)
                        except BaseException:
                            pass
                    raise
        return result

    def acquire_and_replay(
        self, session: object, config: EntryConfig
    ) -> ProductionAcquisition:
        if not isinstance(session, _LiveTrainingHandle):
            raise TypeError("acquisition requires the live training handle")
        sampled = tuple(
            session.session.sample_group(group) for group in config.seed_groups
        )
        replayed = tuple(session.session.replay_group(group) for group in sampled)
        tensors = getattr(session.session, "_live_replay_tensors", None)
        if not isinstance(tensors, Mapping):
            raise Task5RuntimeEvidenceError(
                "live shared-surface session lacks replay-logprob tensor ownership"
            )
        context_request = Task5ProductionContextRequest(
            assembly=session.assembly,
            session=session.session,
            sampled_groups=sampled,
            replay_groups=replayed,
            replay_logprob_tensors=tensors,
            manifest=self._manifest,
            manifest_image=self._image(),
            config=config,
        )
        owner = self._hf_native_owner
        if owner is None:
            runtime_evidence = self._runtime_context_provider.provide(context_request)
        else:
            source = self._pre_acquisition_source
            if source is None:
                raise Task5RuntimeEvidenceError(
                    "HF-native Source/compiler/witness owner was not frozen"
                )
            hf_admission = owner.admit_after_replay(
                HFNativeAdmissionRequest(
                    assembly=session.assembly,
                    session=session.session,
                    manifest=self._manifest,
                    manifest_image=self._image(),
                    config=config,
                    sampled_groups=sampled,
                    replay_groups=replayed,
                    replay_logprob_tensors=tensors,
                ),
                source,
            )
            if type(hf_admission) is not HFNativeOneImageAdmission:
                raise Task5RuntimeEvidenceError(
                    "HF-native owner returned an unadmitted one-image join"
                )
            runtime_evidence = hf_admission.runtime_evidence
        if type(runtime_evidence) is not AdmittedTask5RuntimeEvidence:
            raise Task5RuntimeEvidenceError(
                "production context provider returned unadmitted evidence"
            )
        acquisition = self._runtime_factory(
            assembly=session.assembly,
            session=session.session,
            sampled_groups=sampled,
            replay_groups=replayed,
            replay_logprob_tensors=tensors,
            manifest=self._manifest,
            manifest_image=self._image(),
            config=config,
            runtime_evidence=runtime_evidence,
        )
        if not isinstance(acquisition, ProductionAcquisition):
            raise TypeError("Task-5 runtime factory must return ProductionAcquisition")
        proposal = acquisition.cuda_proposal_input
        if (
            tuple(getattr(proposal, "sampled_groups", ())) != sampled
            or tuple(getattr(proposal, "replay_groups", ())) != replayed
            or dict(getattr(proposal, "replay_logprob_tensors", {})) != tensors
        ):
            raise HFNativeOneImageOwnerError(
                "Task-5 runtime factory changed live Task-2 group lineage",
                disposition="task2_task3_lineage_mismatch",
            )
        return acquisition

    def build_cuda_adapter(self, proposal_input: object) -> SplitCudaAdapter:
        return CudaHFVerticalAdapter(proposal_input)  # type: ignore[arg-type]

    def write_private_checkpoint(
        self, session: object, proposal: object, output_root: Path
    ) -> _PrivateCheckpoint:
        del proposal
        if not isinstance(session, _LiveTrainingHandle):
            raise TypeError("checkpoint write requires the live training handle")
        writer = self._checkpoint_writer_factory(output_root / "private")
        result = self._checkpoint_write(writer, session.assembly)
        checkpoint = Path(getattr(result, "checkpoint_dir")).expanduser().resolve()
        self._checkpoint_readback(
            checkpoint, expected_step=1, assembly=session.assembly
        )
        return _PrivateCheckpoint(
            checkpoint_path=str(checkpoint),
            checkpoint_payload_sha256=self._checkpoint_hasher(checkpoint),
        )

    def proposal_audit(
        self, session: object, private: object, repetition_penalty: float
    ) -> Mapping[str, Any]:
        if not isinstance(session, _LiveAuditHandle):
            raise TypeError("proposal audit requires the live audit handle")
        return self._audit(
            session,
            checkpoint_path=getattr(private, "checkpoint_path"),
            arm_id="private_proposal",
            milestone=1,
            repetition_penalty=repetition_penalty,
        )

    def reproduce_source(
        self, session: object, repetition_penalties: tuple[float, float]
    ) -> Mapping[float, Mapping[str, Any]]:
        return {rp: self.source_audit(session, rp) for rp in repetition_penalties}

    def cleanup_private_checkpoint(self, private: object) -> None:
        checkpoint = Path(getattr(private, "checkpoint_path")).expanduser().resolve()
        private_root = checkpoint.parents[1]
        if private_root.name != "private" or checkpoint.parent.name != "checkpoints":
            raise ValueError("private checkpoint containment differs")
        shutil.rmtree(private_root)
        _fsync_directory(private_root.parent)

    def close_training(self, session: object) -> object:
        return self._close_training(session, aborted=False)

    def close_training_failed(self, session: object) -> object:
        """Close a source-only abort without claiming completed K16."""

        return self._close_training(session, aborted=True)

    def _close_training(self, session: object, *, aborted: bool) -> object:
        if not isinstance(session, _LiveTrainingHandle):
            raise TypeError("close requires the live training handle")
        if self._active_training_handle is not session:
            raise ValueError("close training handle differs from the active owner")
        try:
            try:
                if aborted:
                    close_failed = getattr(session.session, "close_failed", None)
                    if not callable(close_failed):
                        raise TypeError(
                            "shared-surface session lacks the source-only failed close"
                        )
                    return close_failed()
                return session.session.close()
            except Exception as error:
                try:
                    receipt = session.session.resource_receipt
                except Exception:
                    raise error
                raise _SharedSurfaceTrainingCloseError(receipt, error) from error
        finally:
            self._active_training_handle = None
            self._pre_acquisition_source = None
            self._source_owner_audits.clear()

    def close_audit(self, session: object) -> None:
        if not isinstance(session, _LiveAuditHandle):
            raise TypeError("close requires the live audit handle")


class ProductionOneImageServices:
    """One concrete, append-only production lifecycle for ``run_one_image``."""

    retry_count = 0
    fallback_used = False

    def __init__(
        self,
        *,
        backend: ProductionOneImageBackend,
        stale_reservation_path: str | Path,
        successor_root: str | Path,
        attempt_id: str,
        recovery_authority: str = "explicit_task5_owner",
        stale_owner_pid: int | None = None,
        pid_is_alive: Callable[[int], bool] = _pid_is_alive,
        phase_writer: PhaseWriter | None = None,
    ) -> None:
        if not isinstance(backend, ProductionOneImageBackend):
            raise TypeError("production services require a complete backend")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id must be nonempty")
        if not isinstance(recovery_authority, str) or not recovery_authority:
            raise ValueError("recovery authority must be nonempty")
        if stale_owner_pid is not None and (
            isinstance(stale_owner_pid, bool) or stale_owner_pid <= 0
        ):
            raise ValueError("stale owner PID must be a positive integer")
        self._backend = backend
        self._stale = Path(stale_reservation_path).expanduser().resolve()
        self._root = Path(successor_root).expanduser().resolve()
        if (
            self._root == self._stale.parent
            or self._root.parent != self._stale.parent.parent
        ):
            raise ValueError("successor root must be a fresh sibling of the stale root")
        self._attempt_id = attempt_id
        self._recovery_authority = recovery_authority
        self._stale_owner_pid = stale_owner_pid
        self._pid_is_alive = pid_is_alive
        self._phase_writer = phase_writer or _write_exclusive_json
        self._recovery_sha256: str | None = None
        self._reserved = False
        self._phase_index = 0
        self._phase_hashes: list[str] = []
        self._adapter: SplitCudaAdapter | None = None
        self._proposal: object | None = None
        self._rollback_attempted = False
        self._source_reproduced = False
        self._close_called = False
        self._source_only_close_requested = False
        self._audit_session: object | None = None
        self._source_audits: dict[float, Mapping[str, Any]] = {}
        self._pending_context_failure: Task5ProductionContextFailureReceipt | None = (
            None
        )
        self._shared_surface_resource_receipt: object | None = None
        self._observed_acquisition_failure: (
            Task5ObservedAcquisitionFailureReceipt | None
        ) = None
        self._accounted_shared_surface_forwards = 0
        self._accounted_source_owner_forwards = 0
        self._actions = {
            "model_loads": 0,
            "forwards": 0,
            "backwards": 0,
            "optimizer_steps": 0,
            "gpu_allocations": 0,
            "network_actions": 0,
            "output_creations": 0,
        }

    @property
    def successor_root(self) -> Path:
        return self._root

    def action_counters(self) -> dict[str, int]:
        return dict(self._actions)

    @property
    def source_reproduced(self) -> bool:
        return self._source_reproduced

    @property
    def shared_surface_resource_receipt(self) -> object | None:
        return self._shared_surface_resource_receipt

    @property
    def observed_acquisition_failure_receipt(
        self,
    ) -> Task5ObservedAcquisitionFailureReceipt | None:
        return self._observed_acquisition_failure

    @property
    def phase_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(self._phase_hashes)

    @property
    def phase_ledger_sha256(self) -> str | None:
        if not self._phase_hashes:
            return None
        return json_sha256(
            {
                "schema_version": "human13_all_hf_phase_ledger.v1",
                "phase_receipt_sha256s": list(self._phase_hashes),
            }
        )

    def _record(
        self,
        phase: str,
        *,
        status: str = "completed",
        evidence: Mapping[str, Any] | None = None,
    ) -> str:
        self._phase_index += 1
        payload = {
            "schema_version": PHASE_SCHEMA,
            "attempt_id": self._attempt_id,
            "phase_index": self._phase_index,
            "phase": phase,
            "status": status,
            "recovery_successor_sha256": self._recovery_sha256,
            "evidence": {} if evidence is None else dict(evidence),
        }
        content = json_sha256(payload)
        self._phase_writer(
            self._root / "receipts" / f"{self._phase_index:03d}-{phase}.json",
            payload | {"content_sha256": content},
        )
        self._phase_hashes.append(content)
        return content

    def _reserve_successor(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> None:
        if self._reserved or self._root.exists():
            raise RuntimeError("successor attempt is already reserved")
        parent_bytes = self._stale.read_bytes()
        parent = json.loads(parent_bytes)
        if not isinstance(parent, Mapping):
            raise ValueError("stale reservation must be a JSON object")
        pid = parent.get("pid")
        parent_pid_source = "reservation"
        if pid is None and self._stale_owner_pid is not None:
            pid = self._stale_owner_pid
            parent_pid_source = "explicit_recovery_witness"
        run_id = parent.get("run_id")
        actions = parent.get("model_actions")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(actions, Mapping)
            or any(value != 0 for value in actions.values())
            or any(
                key in parent
                for key in ("terminal", "terminal_status", "terminal_sha256")
            )
        ):
            raise ValueError("parent reservation is not a zero-action lost-owner run")
        if self._pid_is_alive(pid):
            raise RuntimeError("parent reservation PID is still alive")
        recovery_payload = {
            "schema_version": RECOVERY_SCHEMA,
            "parent_reservation_path": str(self._stale),
            "parent_reservation_sha256": hashlib_sha256(parent_bytes),
            "parent_run_id": run_id,
            "parent_pid": pid,
            "parent_pid_source": parent_pid_source,
            "parent_owner_lost": True,
            "successor_run_id": self._attempt_id,
            "successor_root": str(self._root),
            "config_sha256": config.content_sha256,
            "manifest_sha256": config.manifest_sha256,
            "retry_ceiling": FIXED_RETRY_CEILING,
            "recovery_authority": self._recovery_authority,
        }
        recovery_sha = json_sha256(recovery_payload)
        claim_payload = {
            "schema_version": "human13_one_image_recovery_claim.v1",
            "parent_reservation_path": str(self._stale),
            "parent_reservation_sha256": recovery_payload["parent_reservation_sha256"],
            "parent_run_id": run_id,
            "parent_pid": pid,
            "parent_pid_source": parent_pid_source,
            "successor_run_id": self._attempt_id,
            "successor_root": str(self._root),
            "config_sha256": config.content_sha256,
            "manifest_sha256": config.manifest_sha256,
            "retry_ceiling": FIXED_RETRY_CEILING,
            "recovery_authority": self._recovery_authority,
        }
        claim_path = (
            self._stale.parent
            / ".reservation-recovery-claims"
            / f"{recovery_payload['parent_reservation_sha256']}.json"
        )
        try:
            _write_exclusive_json(
                claim_path,
                claim_payload | {"content_sha256": json_sha256(claim_payload)},
            )
        except FileExistsError as error:
            raise RuntimeError(
                "parent reservation recovery claim already consumed its retry ceiling"
            ) from error
        self._root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(self._root.parent)
        _write_exclusive_json(
            self._root / "reservation-recovery.v1.json",
            recovery_payload | {"content_sha256": recovery_sha},
        )
        reservation_payload = {
            "schema_version": RESERVATION_SCHEMA,
            "run_id": self._attempt_id,
            "pid": os.getpid(),
            "parent_pid_source": parent_pid_source,
            "output_root": str(self._root),
            "config_sha256": config.content_sha256,
            "manifest_sha256": config.manifest_sha256,
            "training_gpu": resources.training_gpu,
            "audit_gpu": resources.audit_gpu,
            "retry_ceiling": FIXED_RETRY_CEILING,
            "recovery_successor_sha256": recovery_sha,
            "model_actions": dict(self._actions),
        }
        _write_exclusive_json(
            self._root / "run-reservation.json",
            reservation_payload | {"content_sha256": json_sha256(reservation_payload)},
        )
        self._recovery_sha256 = recovery_sha
        self._reserved = True
        self._record("reservation_recovery")

    def preflight_source_assembly(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> SourceAssemblyReceipt:
        if resources.training_gpu != 0 or resources.audit_gpu != 1:
            raise ValueError("production lifecycle requires GPU 0 training/GPU 1 audit")
        self._reserve_successor(config, resources)
        try:
            receipt = self._backend.preflight_source_assembly(config, resources)
        except Exception as error:
            self._record(
                "source_assembly_preflight",
                status="failed",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        if not isinstance(receipt, SourceAssemblyReceipt):
            raise TypeError(
                "backend source preflight must return SourceAssemblyReceipt"
            )
        self._record(
            "source_assembly_preflight",
            evidence={"source_assembly_sha256": receipt.content_sha256},
        )
        return receipt

    def open_training(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object:
        session = self._backend.open_training(config, resources)
        self._actions["model_loads"] += 1
        self._actions["gpu_allocations"] += 1
        self._record("gpu0_training_open")
        return session

    def open_audit(
        self, config: EntryConfig, resources: DualGPUResourceReceipt
    ) -> object:
        session = self._backend.open_audit(config, resources)
        self._audit_session = session
        self._actions["model_loads"] += 1
        self._actions["gpu_allocations"] += 1
        self._record("gpu1_fp32_sdpa_audit_open")
        return session

    def source_audit(
        self, audit_session: object, repetition_penalty: float
    ) -> Mapping[str, Any]:
        try:
            result = self._backend.source_audit(audit_session, repetition_penalty)
        except BaseException as error:
            self._source_only_close_requested = True
            if bool(getattr(error, "_source_audit_forward_observed", False)):
                observed = getattr(error, "_source_audit_result", None)
                observed_rp = getattr(
                    error, "_source_audit_repetition_penalty", repetition_penalty
                )
                if isinstance(observed, Mapping) and isinstance(
                    observed_rp, (int, float)
                ) and not isinstance(observed_rp, bool):
                    observed_rp = float(observed_rp)
                    self._source_audits[observed_rp] = observed
                    self._actions["forwards"] += 1
                    phase = f"source_audit_rp_{observed_rp:g}"
                    evidence: dict[str, Any] = {
                        "source_audit_forward_observed": True,
                        "repetition_penalty": observed_rp,
                    }
                    try:
                        evidence["source_audit_sha256"] = json_sha256(observed)
                    except BaseException:
                        evidence["source_audit_sha256"] = None
                    reconciliation = getattr(
                        error, "reconciliation_receipt", None
                    )
                    if reconciliation is not None and callable(
                        getattr(reconciliation, "to_dict", None)
                    ):
                        evidence["surface_reconciliation_receipt"] = (
                            reconciliation.to_dict()
                        )
                    try:
                        self._record(phase, status="failed", evidence=evidence)
                    except BaseException as record_error:
                        try:
                            setattr(error, "_source_audit_phase_record_error", record_error)
                        except BaseException:
                            pass
                    else:
                        try:
                            setattr(error, "_source_audit_phase_recorded", True)
                            setattr(error, "_source_audit_phase", phase)
                        except BaseException:
                            pass
            raise
        self._source_audits[repetition_penalty] = result
        self._actions["forwards"] += 1
        self._record(f"source_audit_rp_{repetition_penalty:g}")
        return result

    def acquire_and_replay(
        self, training_session: object, config: EntryConfig
    ) -> ProductionAcquisition:
        try:
            acquisition = self._backend.acquire_and_replay(training_session, config)
        except Task5ProductionContextUnavailable as error:
            self._pending_context_failure = error.receipt
            self._record(
                "k16_acquisition_replay",
                status="runtime_context_failure",
                evidence={"context_failure_receipt": error.receipt.to_dict()},
            )
            raise
        except HFNativeOneImageOwnerError as error:
            owner_error: dict[str, object] = {
                "type": type(error).__name__,
                "reason": error.reason,
                "disposition": error.disposition,
            }
            if error.reconciliation_receipt is not None:
                owner_error["surface_reconciliation_receipt"] = (
                    error.reconciliation_receipt.to_dict()
                )
            self._record(
                "k16_acquisition_replay",
                status="hf_native_owner_failure",
                evidence={"owner_error": owner_error},
            )
            raise
        if not isinstance(acquisition, ProductionAcquisition):
            raise TypeError("backend acquisition must return ProductionAcquisition")
        self._account_shared_surface_forwards(
            acquisition.sample_forward_count,
            acquisition.replay_forward_count,
        )
        self._record(
            "k16_acquisition_replay",
            status="completed" if acquisition.parity_passed else "parity_failure",
            evidence={
                "task2_resource_sha256": acquisition.task2_resource_sha256,
                "trajectory_ledger_sha256": acquisition.trajectory_ledger_sha256,
                "compiler_ledger_sha256": acquisition.compiler_ledger_sha256,
                "sample_forward_count": acquisition.sample_forward_count,
                "replay_forward_count": acquisition.replay_forward_count,
            },
        )
        return acquisition

    def _account_shared_surface_forwards(
        self,
        sample_forward_count: int,
        replay_forward_count: int,
        source_owner_forward_count: int = 0,
    ) -> None:
        for field, value in (
            ("sample_forward_count", sample_forward_count),
            ("replay_forward_count", replay_forward_count),
            ("source_owner_forward_count", source_owner_forward_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        observed = sample_forward_count + replay_forward_count
        if self._accounted_shared_surface_forwards:
            if observed != self._accounted_shared_surface_forwards:
                raise ValueError("shared-surface forward observation changed")
            if source_owner_forward_count < self._accounted_source_owner_forwards:
                raise ValueError("Source-owner forward observation regressed")
            self._actions["forwards"] += (
                source_owner_forward_count - self._accounted_source_owner_forwards
            )
            self._accounted_source_owner_forwards = source_owner_forward_count
            return
        self._actions["forwards"] += observed + source_owner_forward_count
        self._accounted_shared_surface_forwards = observed
        self._accounted_source_owner_forwards = source_owner_forward_count

    @staticmethod
    def _admit_shared_surface_receipt(receipt: object) -> Mapping[str, Any]:
        to_dict = getattr(receipt, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("shared-surface close must return a typed resource receipt")
        value = to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("shared-surface resource receipt must serialize to an object")
        content_sha256 = value.get("content_sha256")
        _digest(content_sha256, "shared_surface_resource_receipt_sha256")
        payload = {key: item for key, item in value.items() if key != "content_sha256"}
        if json_sha256(payload) != content_sha256:
            raise ValueError("shared-surface resource receipt hash differs")
        required = {
            "sample_forward_count",
            "replay_forward_count",
            "source_owner_forward_count",
            "total_forward_count",
            "no_cache_forward_count",
            "sampled_group_sha256s",
            "replay_group_sha256s",
            "model_object_id",
            "retained_graph_count",
            "session_held_reference_count",
            "cleanup_state",
            "cleanup_reason",
            "cleanup_failures",
            "cleanup_call_count",
        }
        if not required <= set(value):
            raise ValueError("shared-surface resource receipt fields are incomplete")
        counts = tuple(
            value[field]
            for field in (
                "sample_forward_count",
                "replay_forward_count",
                "source_owner_forward_count",
                "total_forward_count",
                "no_cache_forward_count",
            )
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts
        ):
            raise ValueError("shared-surface forward counts are malformed")
        sample, replay, source_owner, total, no_cache = cast(
            tuple[int, int, int, int, int], counts
        )
        if total != sample + replay + source_owner or no_cache != total:
            raise ValueError("shared-surface no-cache forward counts differ")
        if value["retained_graph_count"] != 0:
            raise ValueError("shared-surface close retained graph ownership")
        if value["session_held_reference_count"] != 0:
            raise ValueError("shared-surface close retained group ownership")
        if value["cleanup_state"] != "closed" or value["cleanup_call_count"] != 1:
            raise ValueError("shared-surface cleanup is not exactly-once terminal")
        sampled_hashes = value["sampled_group_sha256s"]
        replay_hashes = value["replay_group_sha256s"]
        if not isinstance(sampled_hashes, (list, tuple)) or not isinstance(
            replay_hashes, (list, tuple)
        ):
            raise ValueError("shared-surface group hashes must be sequences")
        for label, hashes in (
            ("sampled", sampled_hashes),
            ("replay", replay_hashes),
        ):
            for index, digest in enumerate(hashes):
                _digest(digest, f"{label}_group_sha256s[{index}]")
        return value

    def _observe_shared_surface_close(self, receipt: object) -> dict[str, Any]:
        value = self._admit_shared_surface_receipt(receipt)
        if self._shared_surface_resource_receipt is not None:
            raise RuntimeError("shared-surface resource receipt was already observed")
        self._shared_surface_resource_receipt = receipt
        sample = cast(int, value["sample_forward_count"])
        replay = cast(int, value["replay_forward_count"])
        source_owner = cast(int, value["source_owner_forward_count"])
        self._account_shared_surface_forwards(sample, replay, source_owner)
        evidence: dict[str, Any] = {
            "shared_surface_resource_receipt_sha256": value["content_sha256"],
            "sample_forward_count": sample,
            "replay_forward_count": replay,
            "source_owner_forward_count": source_owner,
            "no_cache_forward_count": value["no_cache_forward_count"],
            "cleanup_state": value["cleanup_state"],
            "cleanup_reason": value["cleanup_reason"],
            "cleanup_call_count": value["cleanup_call_count"],
        }
        context = self._pending_context_failure
        if context is None:
            return evidence
        sampled_hashes = tuple(value["sampled_group_sha256s"])
        replay_hashes = tuple(value["replay_group_sha256s"])
        if sampled_hashes != context.sampled_group_sha256s:
            raise ValueError("failure sampled-group lineage differs at close")
        if replay_hashes != context.replay_group_sha256s:
            raise ValueError("failure replay-group lineage differs at close")
        if value["model_object_id"] != context.model_object_id:
            raise ValueError("failure model ownership differs at close")
        cleanup_failures = value["cleanup_failures"]
        if not isinstance(cleanup_failures, (list, tuple)):
            raise ValueError("cleanup failures must be a sequence")
        observed = Task5ObservedAcquisitionFailureReceipt(
            context_failure_receipt_sha256=context.content_sha256,
            shared_surface_resource_receipt_sha256=cast(
                str, value["content_sha256"]
            ),
            config_sha256=context.config_sha256,
            manifest_sha256=context.manifest_sha256,
            source_identity_sha256=context.source_identity_sha256,
            sampled_group_sha256s=sampled_hashes,
            replay_group_sha256s=replay_hashes,
            sample_forward_count=sample,
            replay_forward_count=replay,
            source_owner_forward_count=source_owner,
            total_forward_count=cast(int, value["total_forward_count"]),
            no_cache_forward_count=cast(int, value["no_cache_forward_count"]),
            model_object_id=cast(int, value["model_object_id"]),
            cleanup_state=cast(str, value["cleanup_state"]),
            cleanup_reason=cast(str, value["cleanup_reason"]),
            cleanup_failures=tuple(cast(Sequence[str], cleanup_failures)),
            cleanup_call_count=cast(int, value["cleanup_call_count"]),
        )
        self._observed_acquisition_failure = observed
        evidence["acquisition_failure_receipt"] = observed.to_dict()
        return evidence

    @staticmethod
    def _require_task2_task3_lineage(acquisition: ProductionAcquisition) -> None:
        surface = acquisition.cuda_proposal_input
        if (
            getattr(surface, "surface_identity", None) is None
            or len(tuple(getattr(surface, "sampled_groups", ()))) != 4
            or len(tuple(getattr(surface, "replay_groups", ()))) != 4
            or not getattr(surface, "replay_logprob_tensors", None)
            or getattr(surface, "trajectory_ledger", None) is None
            or getattr(surface, "compiler_ledger", None) is None
        ):
            raise HFNativeOneImageOwnerError(
                "complete current Task2/Task3 lineage is required",
                disposition="task2_task3_lineage_mismatch",
            )

    def apply_private_update(
        self,
        training_session: object,
        acquisition: object,
        config: EntryConfig,
    ) -> object:
        del config
        if not isinstance(acquisition, ProductionAcquisition):
            raise TypeError("private update requires ProductionAcquisition")
        if not acquisition.parity_passed:
            raise RuntimeError("shared-surface parity failure forbids backward")
        try:
            self._require_task2_task3_lineage(acquisition)
        except HFNativeOneImageOwnerError as error:
            self._record(
                "private_update_applied",
                status="hf_native_owner_failure",
                evidence={
                    "owner_error": {
                        "type": type(error).__name__,
                        "reason": error.reason,
                        "disposition": error.disposition,
                    }
                },
            )
            raise
        adapter = self._backend.build_cuda_adapter(acquisition.cuda_proposal_input)
        if not isinstance(adapter, (CudaHFVerticalAdapter, SplitCudaAdapter)):
            raise TypeError("backend must build the split CUDA adapter lifecycle")
        self._adapter = adapter
        try:
            proposal = adapter.apply_private_proposal()
        except Exception as error:
            self._record(
                "private_update_applied",
                status="failed",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        self._proposal = proposal
        self._actions["backwards"] += 1
        self._actions["optimizer_steps"] += 1
        try:
            self._record(
                "private_update_applied",
                evidence={
                    "proposal_receipt_sha256": getattr(proposal, "content_sha256")
                },
            )
        except Exception as journal_error:
            try:
                self.rollback_and_reproduce_source(training_session, proposal)
            except Exception as rollback_error:
                journal_error.add_note(
                    "post-apply rollback failure: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
            raise
        return proposal

    def write_private_proposal(
        self, training_session: object, proposal: object, output_root: Path
    ) -> object:
        if proposal is not self._proposal:
            raise ValueError("private checkpoint proposal identity differs")
        if Path(output_root).resolve() != self._root:
            raise ValueError("private checkpoint output root differs from successor")
        try:
            private = self._backend.write_private_checkpoint(
                training_session, proposal, self._root
            )
        except Exception as error:
            self._record(
                "private_checkpoint_written",
                status="failed",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        path = Path(getattr(private, "checkpoint_path", "")).expanduser().resolve()
        try:
            path.relative_to(self._root)
        except ValueError as error:
            raise ValueError("private checkpoint escaped the successor root") from error
        payload_sha = _digest(
            getattr(private, "checkpoint_payload_sha256", None),
            "private checkpoint payload",
        )
        self._actions["output_creations"] += 1
        self._record(
            "private_checkpoint_written",
            evidence={
                "checkpoint_path": str(path),
                "checkpoint_payload_sha256": payload_sha,
            },
        )
        return private

    def proposal_audit(
        self, audit_session: object, private: object, repetition_penalty: float
    ) -> Mapping[str, Any]:
        try:
            result = self._backend.proposal_audit(
                audit_session, private, repetition_penalty
            )
        except Exception as error:
            self._record(
                f"proposal_audit_rp_{repetition_penalty:g}",
                status="failed",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        self._actions["forwards"] += 1
        self._record(f"proposal_audit_rp_{repetition_penalty:g}")
        return result

    def rollback_and_reproduce_source(
        self, training_session: object, proposal: object
    ) -> bool:
        if proposal is not self._proposal or self._adapter is None:
            raise ValueError("rollback proposal identity differs")
        if self._rollback_attempted:
            raise RuntimeError("private proposal rollback was already attempted")
        self._rollback_attempted = True
        try:
            rollback = self._adapter.rollback_private_proposal()
        except Exception as error:
            self._record(
                "rollback_source_reproduction",
                status="rollback_failure",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        del training_session
        if self._audit_session is None or tuple(self._source_audits) != (1.0, 1.1):
            raise RuntimeError(
                "Source reproduction requires both ordered Source audits"
            )
        reproduced = self._backend.reproduce_source(self._audit_session, (1.0, 1.1))
        restored = (
            isinstance(reproduced, Mapping)
            and set(reproduced) == set(self._source_audits)
            and all(
                json_sha256(reproduced[rp]) == json_sha256(self._source_audits[rp])
                for rp in (1.0, 1.1)
            )
        )
        self._source_reproduced = restored
        self._record(
            "rollback_source_reproduction",
            status="completed" if restored else "failed",
            evidence={
                "rollback_receipt_sha256": getattr(rollback, "content_sha256"),
                "rollback_decision": getattr(rollback, "rollback_decision", None),
                "source_reproduced": restored,
            },
        )
        return restored

    def cleanup_private_proposal(self, private: object) -> None:
        try:
            self._backend.cleanup_private_checkpoint(private)
        except Exception as error:
            self._record(
                "private_checkpoint_cleanup",
                status="failed",
                evidence={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        self._record("private_checkpoint_cleanup")

    def close(
        self, training_session: object | None, audit_session: object | None
    ) -> None:
        if self._close_called:
            raise RuntimeError("one-image sessions were already closed")
        self._close_called = True
        try:
            failures: list[Exception] = []
            if audit_session is not None:
                try:
                    self._backend.close_audit(audit_session)
                    self._record("audit_session_closed")
                except Exception as error:
                    failures.append(error)
                    if self._reserved:
                        self._record(
                            "audit_session_closed",
                            status="failed",
                            evidence={"error": f"{type(error).__name__}: {error}"},
                        )
            if training_session is not None:
                try:
                    close_failed = getattr(self._backend, "close_training_failed", None)
                    if self._source_only_close_requested and callable(close_failed):
                        receipt = close_failed(training_session)
                    else:
                        receipt = self._backend.close_training(training_session)
                    if receipt is None:
                        if self._pending_context_failure is not None:
                            raise RuntimeError(
                                "runtime-context failure close omitted "
                                "shared-surface receipt"
                            )
                        evidence: Mapping[str, Any] = {}
                    else:
                        evidence = self._observe_shared_surface_close(receipt)
                    self._record("training_session_closed", evidence=evidence)
                except _SharedSurfaceTrainingCloseError as error:
                    try:
                        evidence = self._observe_shared_surface_close(
                            error.resource_receipt
                        )
                    except Exception as receipt_error:
                        error.add_note(
                            "shared-surface receipt observation failed: "
                            f"{type(receipt_error).__name__}: {receipt_error}"
                        )
                        evidence = {}
                    failures.append(error)
                    if self._reserved:
                        self._record(
                            "training_session_closed",
                            status="failed",
                            evidence=evidence
                            | {"error": f"{type(error).__name__}: {error}"},
                        )
                except Exception as error:
                    failures.append(error)
                    if self._reserved:
                        self._record(
                            "training_session_closed",
                            status="failed",
                            evidence={"error": f"{type(error).__name__}: {error}"},
                        )
            if failures:
                primary = failures[0]
                for extra in failures[1:]:
                    primary.add_note(
                        f"additional close failure: {type(extra).__name__}: {extra}"
                    )
                raise primary
        finally:
            self._adapter = None
            self._proposal = None
            self._audit_session = None
            self._source_audits.clear()

    def persist_terminal(self, terminal: OneImageTerminalReceipt) -> None:
        if not self._reserved or self._recovery_sha256 is None:
            raise RuntimeError("terminal persistence requires an admitted successor")
        if not isinstance(terminal, OneImageTerminalReceipt):
            raise TypeError("terminal persistence requires OneImageTerminalReceipt")
        if (
            terminal.phase_receipt_sha256s != self.phase_receipt_sha256s
            or terminal.phase_ledger_sha256 != self.phase_ledger_sha256
        ):
            raise ValueError(
                "terminal phase ledger differs from durable phase evidence"
            )
        context_failure = self._pending_context_failure
        observed_failure = self._observed_acquisition_failure
        if context_failure is not None and observed_failure is None:
            raise ValueError(
                "runtime-context terminal lacks closed shared-surface observation"
            )
        shared_resource_sha256 = None
        if self._shared_surface_resource_receipt is not None:
            shared_value = self._admit_shared_surface_receipt(
                self._shared_surface_resource_receipt
            )
            shared_resource_sha256 = shared_value["content_sha256"]
        payload = {
            "schema_version": TERMINAL_ENVELOPE_SCHEMA,
            "attempt_id": self._attempt_id,
            "recovery_successor_sha256": self._recovery_sha256,
            "terminal_status": terminal.terminal_status,
            "terminal_sha256": terminal.content_sha256,
            "phase_receipt_count": len(self._phase_hashes),
            "phase_receipt_sha256s": list(self._phase_hashes),
            "phase_ledger_sha256": self.phase_ledger_sha256,
            "context_failure_receipt_sha256": None
            if context_failure is None
            else context_failure.content_sha256,
            "shared_surface_resource_receipt_sha256": shared_resource_sha256,
            "observed_acquisition_failure_receipt": None
            if observed_failure is None
            else observed_failure.to_dict(),
            "terminal": terminal.to_dict(),
        }
        _write_exclusive_json(
            self._root / "terminal.json",
            payload | {"content_sha256": json_sha256(payload)},
        )
        _fsync_directory(self._root)


def hashlib_sha256(value: bytes) -> str:
    """Keep raw-file hashing explicit and independent of JSON canonicalization."""

    import hashlib

    return hashlib.sha256(value).hexdigest()


__all__ = [
    "AdmittedTask5RuntimeEvidence",
    "ExistingOwnersProductionBackend",
    "FIXED_RETRY_CEILING",
    "OneImageAuditManifest",
    "ProductionAcquisition",
    "ProductionOneImageBackend",
    "ProductionOneImageServices",
    "RepositoryTask5ProductionContextProvider",
    "Task5ProductionContextFailureReceipt",
    "Task5ProductionContextProvider",
    "Task5ProductionContextRequest",
    "Task5ProductionContextUnavailable",
    "Task5ObservedAcquisitionFailureReceipt",
    "Task5RuntimeFactory",
    "Task5RuntimeEvidenceError",
    "default_task5_runtime_factory",
]
