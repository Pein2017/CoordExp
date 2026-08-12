"""Production bridge from the training runtime to exact-state artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from src.artifacts.training_state import (
    REQUIRED_IDENTITY_KINDS,
    TRAINING_STATE_DIRECTORY,
    AdmittedTrainingState,
    PublishedTrainingState,
    RankTrainingStatePayload,
    RestoredRankTrainingState,
    TrainingStateContributionSession,
    TrainingStateExpectations,
    TrainingStatePublicationPlan,
    abort_training_state_contributions,
    admit_training_state,
    begin_training_state_contributions,
    build_resume_compatibility_projection,
    capture_runtime_state_expectations,
    commit_training_state_contributions,
    load_training_state_manifest,
    publish_rank_training_state_contribution,
    restore_decoded_rank_training_state,
    serialize_rank_training_state,
)
from src.common.errors import ArtifactContractError


@dataclass(frozen=True)
class RankCudaDeviceBinding:
    """Logical process device bound to its physical CUDA identity."""

    rank: int
    world_size: int
    logical_device: int
    physical_device: str


@dataclass(frozen=True)
class RankRngSnapshot:
    """One rank's process RNG state and canonical local CUDA topology."""

    rank: int
    world_size: int
    python_state: tuple[Any, ...]
    numpy_state: tuple[Any, ...]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]
    cuda_device_topology: tuple[str, ...]
    cuda_device_binding: RankCudaDeviceBinding


@dataclass(frozen=True)
class DistributedExactResumeStatus:
    """Bounded collective control record; it never carries runtime payload bytes."""

    phase: str
    rank: int
    world_size: int
    ok: bool
    error_code: str | None = None
    error_type: str | None = None
    session: TrainingStateContributionSession | None = None
    manifest_digest: str | None = None
    plan_digest: str | None = None


@dataclass(frozen=True)
class DistributedExactResumeRestoreStatus:
    """Bounded control record for one phase of a distributed restore."""

    phase: str
    rank: int
    world_size: int
    ok: bool
    error_code: str | None = None
    error_type: str | None = None
    applied: bool = False
    rolled_back: bool = False
    manifest_digest: str | None = None


@dataclass(frozen=True)
class DistributedExactResumeRestoreReceipt:
    """Converged receipt returned after every rank applies the same checkpoint."""

    rank: int
    world_size: int
    manifest_digest: str
    restored: RestoredRankTrainingState
    admission_statuses: tuple[DistributedExactResumeRestoreStatus, ...]
    apply_statuses: tuple[DistributedExactResumeRestoreStatus, ...]


@dataclass(frozen=True)
class _RuntimeRollbackSnapshot:
    trainable_model: Mapping[str, torch.Tensor]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any] | None
    scaler: Mapping[str, Any] | None
    python_rng: tuple[Any, ...]
    numpy_rng: tuple[Any, ...]
    torch_cpu_rng: torch.Tensor
    torch_cuda_rng: tuple[torch.Tensor, ...]


def build_exact_resume_identities(
    *,
    base_model: str,
    cache: str,
    dependencies: str,
    policy: str,
    resolved_config: Mapping[str, Any],
    topology: str,
    trainable_surface: str,
) -> Mapping[str, str]:
    """Build the complete strict identity map expected by the artifact core."""

    if not isinstance(resolved_config, Mapping):
        _fail(
            "resolved_config must be a string-keyed mapping",
            code="training_state.schema",
            context={"field": "resolved_config"},
        )
    config_bytes = _canonical_json_bytes(resolved_config, field="resolved_config")
    if not resolved_config:
        _fail(
            "resolved_config must be nonempty",
            code="training_state.incomplete",
            context={"field": "resolved_config"},
        )
    resume_compatibility = build_resume_compatibility_projection(resolved_config)
    identities = {
        "base_model": base_model,
        "cache": cache,
        "dependencies": dependencies,
        "policy": policy,
        "resolved_config": hashlib.sha256(config_bytes + b"\n").hexdigest(),
        "resume_compatibility": hashlib.sha256(
            _canonical_json_bytes(resume_compatibility, field="resume_compatibility")
            + b"\n"
        ).hexdigest(),
        "topology": topology,
        "trainable_surface": trainable_surface,
    }
    if tuple(identities) != REQUIRED_IDENTITY_KINDS:
        _fail(
            "exact-resume identity builder differs from the artifact core",
            code="training_state.schema",
            context={
                "expected": list(REQUIRED_IDENTITY_KINDS),
                "observed": list(identities),
            },
        )
    for name, value in identities.items():
        _require_sha256(value, field=f"identities.{name}")
    return MappingProxyType(identities)


def build_exact_resume_topology_identity(
    bindings: Sequence[RankCudaDeviceBinding],
) -> str:
    """Digest the complete rank-to-logical-to-physical CUDA assignment."""

    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        _fail(
            "CUDA bindings must be a sequence",
            code="training_state.schema",
            context={"field": "bindings"},
        )
    ordered = tuple(sorted(bindings, key=lambda item: getattr(item, "rank", -1)))
    if not ordered or any(
        not isinstance(item, RankCudaDeviceBinding) for item in ordered
    ):
        _fail(
            "CUDA bindings must contain RankCudaDeviceBinding values",
            code="training_state.schema",
            context={"field": "bindings"},
        )
    world_size = ordered[0].world_size
    if len(ordered) != world_size or tuple(item.rank for item in ordered) != tuple(
        range(world_size)
    ):
        _fail(
            "CUDA bindings must cover the complete ordered rank set",
            code="training_state.incomplete_rank_set",
            context={"world_size": world_size},
        )
    records: list[dict[str, Any]] = []
    for item in ordered:
        _validate_cuda_device_binding(item, rank=item.rank, world_size=world_size)
        records.append(
            {
                "logical_device": item.logical_device,
                "physical_device": item.physical_device,
                "rank": item.rank,
                "world_size": item.world_size,
            }
        )
    return hashlib.sha256(
        _canonical_json_bytes(records, field="cuda_device_bindings") + b"\n"
    ).hexdigest()


def capture_rank_rng_snapshot(
    *,
    rank: int,
    world_size: int,
    logical_cuda_device: int | None = None,
    physical_cuda_device: str | None = None,
) -> RankRngSnapshot:
    """Capture every repository-required RNG owner for the current rank."""

    _validate_rank(rank=rank, world_size=world_size)
    current_logical_device = torch.cuda.current_device()
    if (
        logical_cuda_device is not None
        and logical_cuda_device != current_logical_device
    ):
        _fail(
            "RNG capture must use the rank's current CUDA device",
            code="training_state.incompatible",
            context={
                "current_logical_device": current_logical_device,
                "requested_logical_device": logical_cuda_device,
            },
        )
    logical_device = current_logical_device
    if (
        isinstance(logical_device, bool)
        or not isinstance(logical_device, int)
        or logical_device < 0
    ):
        _fail(
            "logical CUDA device must be a nonnegative integer",
            code="training_state.runtime_state",
            context={"field": "logical_cuda_device"},
        )
    binding = RankCudaDeviceBinding(
        rank=rank,
        world_size=world_size,
        logical_device=logical_device,
        physical_device=(
            _resolve_physical_cuda_device(logical_device)
            if physical_cuda_device is None
            else physical_cuda_device
        ),
    )
    _validate_cuda_device_binding(binding, rank=rank, world_size=world_size)
    cuda_states = (torch.cuda.get_rng_state(logical_device).detach().cpu().clone(),)
    numpy_state = np.random.get_state()
    snapshot = RankRngSnapshot(
        rank=rank,
        world_size=world_size,
        python_state=random.getstate(),
        numpy_state=(
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch_cpu_state=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda_states=cuda_states,
        cuda_device_topology=("cuda:0",),
        cuda_device_binding=binding,
    )
    _validate_rng_snapshot(snapshot, rank=rank, world_size=world_size)
    return snapshot


def current_cuda_device_topology() -> tuple[str, ...]:
    """Return the one-device rank-local topology stored by the artifact core."""

    torch.cuda.current_device()
    return ("cuda:0",)


def serialize_current_rank_training_state(
    *,
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    cursor: Mapping[str, Any],
    next_rank_local_micro_step: int,
    accumulation_microstep: int,
    rng_snapshot: RankRngSnapshot | None = None,
) -> RankTrainingStatePayload:
    """Serialize current owners at the first contract's optimizer-step boundary."""

    _validate_rank(rank=rank, world_size=world_size)
    if (
        isinstance(accumulation_microstep, bool)
        or not isinstance(accumulation_microstep, int)
        or accumulation_microstep != 0
    ):
        _fail(
            "mid-accumulation exact training-state serialization is unsupported",
            code="training_state.unsupported_mid_accumulation",
            context={"accumulation_microstep": accumulation_microstep},
        )
    selected_rng = rng_snapshot or capture_rank_rng_snapshot(
        rank=rank, world_size=world_size
    )
    _validate_rng_snapshot(selected_rng, rank=rank, world_size=world_size)
    return serialize_rank_training_state(
        rank=rank,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        python_rng_state=selected_rng.python_state,
        numpy_rng_state=selected_rng.numpy_state,
        torch_cpu_rng_state=selected_rng.torch_cpu_state,
        torch_cuda_rng_states=selected_rng.torch_cuda_states,
        cursor=cursor,
        next_rank_local_micro_step=next_rank_local_micro_step,
    )


def admit_and_restore_current_rank(
    checkpoint_dir: Path | str,
    *,
    checkpoint_step: int,
    rank: int,
    world_size: int,
    identities: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    cuda_device_topology: Sequence[str] | None = None,
    get_torch_cuda_rng_states: Callable[[], Sequence[torch.Tensor]] | None = None,
    set_torch_cuda_rng_states: Callable[[Sequence[torch.Tensor]], None] | None = None,
) -> RestoredRankTrainingState:
    """Admit all files, then delegate atomic current-rank restore to the core."""

    _validate_rank(rank=rank, world_size=world_size)
    admitted = _admit_current_rank_for_restore(
        checkpoint_dir,
        checkpoint_step=checkpoint_step,
        rank=rank,
        world_size=world_size,
        identities=identities,
        resolved_config=resolved_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    topology = tuple(
        cuda_device_topology
        if cuda_device_topology is not None
        else current_cuda_device_topology()
    )

    get_cuda = get_torch_cuda_rng_states or _get_current_cuda_rng_states
    set_cuda = set_torch_cuda_rng_states or _set_current_cuda_rng_states
    restored = restore_decoded_rank_training_state(
        admitted.decoded_rank,
        current_rank=rank,
        current_world_size=world_size,
        current_cuda_device_topology=topology,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        get_torch_cuda_rng_states=get_cuda,
        set_torch_cuda_rng_states=set_cuda,
    )
    if not isinstance(restored, RestoredRankTrainingState):
        _fail(
            "exact-resume restore callback returned an unsupported result",
            code="training_state.restore_failed",
        )
    return restored


def restore_distributed_exact_resume(
    checkpoint_dir: Path | str,
    *,
    checkpoint_step: int,
    expected_manifest_digest: str,
    rank: int,
    world_size: int,
    identities: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    cuda_device_topology: Sequence[str] | None,
    gather_status: Callable[
        [DistributedExactResumeRestoreStatus],
        Sequence[DistributedExactResumeRestoreStatus],
    ],
    get_torch_cuda_rng_states: Callable[[], Sequence[torch.Tensor]] | None = None,
    set_torch_cuda_rng_states: Callable[[Sequence[torch.Tensor]], None] | None = None,
) -> DistributedExactResumeRestoreReceipt:
    """Admit, apply, and converge an all-rank restore transaction."""

    _validate_rank(rank=rank, world_size=world_size)
    admitted_manifest_digest = _require_sha256(
        expected_manifest_digest,
        field="expected_manifest_digest",
    )
    topology = tuple(
        cuda_device_topology
        if cuda_device_topology is not None
        else current_cuda_device_topology()
    )
    get_cuda = get_torch_cuda_rng_states or _get_current_cuda_rng_states
    set_cuda = set_torch_cuda_rng_states or _set_current_cuda_rng_states
    admitted: AdmittedTrainingState | None = None
    rollback_snapshot: _RuntimeRollbackSnapshot | None = None
    try:
        admitted = _admit_current_rank_for_restore(
            checkpoint_dir,
            checkpoint_step=checkpoint_step,
            rank=rank,
            world_size=world_size,
            identities=identities,
            resolved_config=resolved_config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        _validate_decoded_topology(
            admitted,
            rank=rank,
            world_size=world_size,
            cuda_device_topology=topology,
        )
        if admitted.manifest.aggregate_digest != admitted_manifest_digest:
            _fail(
                "training-state manifest changed after prior read-only admission",
                code="training_state.admitted_manifest_drift",
                context={
                    "admitted": admitted_manifest_digest,
                    "readmitted": admitted.manifest.aggregate_digest,
                },
            )
        rollback_snapshot = _capture_runtime_rollback_snapshot(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            get_torch_cuda_rng_states=get_cuda,
        )
        admission_local = DistributedExactResumeRestoreStatus(
            phase="restore_admit",
            rank=rank,
            world_size=world_size,
            ok=True,
            manifest_digest=admitted.manifest.aggregate_digest,
        )
    except BaseException as exc:
        admission_local = _restore_error_status(
            phase="restore_admit", rank=rank, world_size=world_size, exc=exc
        )
    admission_statuses = _gather_restore_statuses(
        gather_status,
        admission_local,
        phase="restore_admit",
        world_size=world_size,
    )
    admission_failures = _restore_failure_rows(admission_statuses)
    manifest_digests = {
        status.manifest_digest for status in admission_statuses if status.ok
    }
    if not admission_failures and len(manifest_digests) != 1:
        _fail(
            "distributed restore ranks admitted different manifests",
            code="training_state.distributed_control",
            context={"phase": "restore_admit"},
        )
    if admission_failures:
        raise ArtifactContractError(
            "distributed exact-state admission failed before mutation",
            code="training_state.distributed_restore_failed",
            context={"admission_failures": admission_failures},
        )
    assert admitted is not None
    assert rollback_snapshot is not None
    manifest_digest = next(iter(manifest_digests))

    restored: RestoredRankTrainingState | None = None
    try:
        restored = restore_decoded_rank_training_state(
            admitted.decoded_rank,
            current_rank=rank,
            current_world_size=world_size,
            current_cuda_device_topology=topology,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            get_torch_cuda_rng_states=get_cuda,
            set_torch_cuda_rng_states=set_cuda,
        )
        apply_local = DistributedExactResumeRestoreStatus(
            phase="restore_apply",
            rank=rank,
            world_size=world_size,
            ok=True,
            applied=True,
            manifest_digest=manifest_digest,
        )
    except BaseException as exc:
        apply_local = _restore_error_status(
            phase="restore_apply", rank=rank, world_size=world_size, exc=exc
        )
    apply_statuses = _gather_restore_statuses(
        gather_status,
        apply_local,
        phase="restore_apply",
        world_size=world_size,
    )
    apply_failures = _restore_failure_rows(apply_statuses)
    if not apply_failures:
        assert restored is not None
        return DistributedExactResumeRestoreReceipt(
            rank=rank,
            world_size=world_size,
            manifest_digest=manifest_digest,
            restored=restored,
            admission_statuses=admission_statuses,
            apply_statuses=apply_statuses,
        )

    try:
        _restore_runtime_rollback_snapshot(
            rollback_snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            set_torch_cuda_rng_states=set_cuda,
        )
        rollback_local = DistributedExactResumeRestoreStatus(
            phase="restore_rollback",
            rank=rank,
            world_size=world_size,
            ok=True,
            rolled_back=True,
            manifest_digest=manifest_digest,
        )
    except BaseException as exc:
        rollback_local = _restore_error_status(
            phase="restore_rollback", rank=rank, world_size=world_size, exc=exc
        )
    rollback_statuses = _gather_restore_statuses(
        gather_status,
        rollback_local,
        phase="restore_rollback",
        world_size=world_size,
    )
    rollback_failures = _restore_failure_rows(rollback_statuses)
    if rollback_failures:
        raise ArtifactContractError(
            "distributed exact-state rollback failed",
            code="training_state.distributed_restore_rollback_failed",
            context={
                "apply_failures": apply_failures,
                "rollback_failures": rollback_failures,
            },
        )
    raise ArtifactContractError(
        "distributed exact-state restore failed and was rolled back on every rank",
        code="training_state.distributed_restore_failed",
        context={
            "apply_failures": apply_failures,
            "rolled_back_ranks": [
                status.rank for status in rollback_statuses if status.rolled_back
            ],
        },
    )


def prepare_distributed_exact_resume_contribution(
    *,
    plan: TrainingStatePublicationPlan,
    rank: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    cursor: Mapping[str, Any],
    next_rank_local_micro_step: int,
    rng_snapshot: RankRngSnapshot | None,
    gather_status: Callable[
        [DistributedExactResumeStatus], Sequence[DistributedExactResumeStatus]
    ],
) -> RankTrainingStatePayload:
    """Serialize locally and converge control status before contribution I/O."""

    if not isinstance(plan, TrainingStatePublicationPlan):
        _fail(
            "publication plan has an unsupported type",
            code="training_state.schema",
        )
    _validate_rank(rank=rank, world_size=plan.world_size)
    payload: RankTrainingStatePayload | None = None
    plan_digest: str | None = None
    try:
        validated_plan = TrainingStatePublicationPlan.from_dict(plan.to_dict())
        plan_digest = validated_plan.digest
        if validated_plan.scheduler_applicable != (scheduler is not None):
            _fail(
                "scheduler applicability differs from the publication plan",
                code="training_state.incompatible",
                context={"field": "scheduler_applicable"},
            )
        if validated_plan.scaler_applicable != (scaler is not None):
            _fail(
                "scaler applicability differs from the publication plan",
                code="training_state.incompatible",
                context={"field": "scaler_applicable"},
            )
        payload = serialize_current_rank_training_state(
            rank=rank,
            world_size=validated_plan.world_size,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=cursor,
            next_rank_local_micro_step=next_rank_local_micro_step,
            accumulation_microstep=validated_plan.accumulation_microstep,
            rng_snapshot=rng_snapshot,
        )
        local_status = _ok_status(
            phase="serialize",
            rank=rank,
            world_size=plan.world_size,
            plan_digest=plan_digest,
        )
    except BaseException as exc:
        local_status = _error_status(
            phase="serialize",
            rank=rank,
            world_size=plan.world_size,
            exc=exc,
            plan_digest=plan_digest,
        )
    statuses = _gather_statuses(
        gather_status,
        local_status,
        phase="serialize",
        world_size=plan.world_size,
    )
    failures = _failure_rows(statuses)
    observed_plan_digests = {
        status.plan_digest for status in statuses if status.plan_digest is not None
    }
    if not failures and len(observed_plan_digests) != 1:
        _fail(
            "ranks prepared different publication plans",
            code="training_state.distributed_control",
            context={"phase": "serialize"},
        )
    if failures:
        raise ArtifactContractError(
            "distributed exact-state serialization failed before contribution",
            code="training_state.distributed_publication_failed",
            context={"failures": failures},
        )
    assert payload is not None
    return payload


def publish_distributed_exact_resume(
    checkpoint_dir: Path | str,
    *,
    plan: TrainingStatePublicationPlan,
    rank: int,
    local_payload: RankTrainingStatePayload,
    barrier: Callable[[], None],
    gather_status: Callable[
        [DistributedExactResumeStatus], Sequence[DistributedExactResumeStatus]
    ],
) -> PublishedTrainingState:
    """Publish rank-local bytes while collectives exchange control records only."""

    if not isinstance(plan, TrainingStatePublicationPlan):
        _fail(
            "publication plan has an unsupported type",
            code="training_state.schema",
        )
    _validate_rank(rank=rank, world_size=plan.world_size)
    session: TrainingStateContributionSession | None = None
    published: PublishedTrainingState | None = None

    if rank == 0:
        try:
            session = begin_training_state_contributions(checkpoint_dir, plan)
            begin_local = _ok_status(
                phase="begin",
                rank=rank,
                world_size=plan.world_size,
                session=session,
            )
        except BaseException as exc:
            begin_local = _error_status(
                phase="begin", rank=rank, world_size=plan.world_size, exc=exc
            )
    else:
        begin_local = _ok_status(phase="begin", rank=rank, world_size=plan.world_size)
    begin_statuses = _gather_statuses(
        gather_status,
        begin_local,
        phase="begin",
        world_size=plan.world_size,
    )
    begin_failures = _failure_rows(begin_statuses)
    if not begin_failures:
        session = begin_statuses[0].session
        if session is None:
            _fail(
                "rank zero did not publish the contribution session",
                code="training_state.distributed_control",
                context={"phase": "begin"},
            )

    if begin_failures:
        contribute_local = DistributedExactResumeStatus(
            phase="contribute",
            rank=rank,
            world_size=plan.world_size,
            ok=False,
            error_code="training_state.begin_failed",
            error_type="DistributedControl",
        )
    else:
        assert session is not None
        try:
            if local_payload.rank != rank:
                _fail(
                    "local payload rank differs from the current rank",
                    code="training_state.rank_mismatch",
                    context={
                        "current_rank": rank,
                        "payload_rank": local_payload.rank,
                    },
                )
            publish_rank_training_state_contribution(
                checkpoint_dir, session, local_payload
            )
            contribute_local = _ok_status(
                phase="contribute", rank=rank, world_size=plan.world_size
            )
        except BaseException as exc:
            contribute_local = _error_status(
                phase="contribute", rank=rank, world_size=plan.world_size, exc=exc
            )
    barrier()
    contribution_statuses = _gather_statuses(
        gather_status,
        contribute_local,
        phase="contribute",
        world_size=plan.world_size,
    )
    contribution_failures = _failure_rows(contribution_statuses)
    root_failures = begin_failures or contribution_failures

    if rank == 0:
        if root_failures:
            first = root_failures[0]
            commit_local = DistributedExactResumeStatus(
                phase="commit",
                rank=rank,
                world_size=plan.world_size,
                ok=False,
                error_code=first["error_code"],
                error_type=first["error_type"],
            )
        else:
            assert session is not None
            try:
                published = commit_training_state_contributions(checkpoint_dir, session)
                commit_local = _ok_status(
                    phase="commit",
                    rank=rank,
                    world_size=plan.world_size,
                    manifest_digest=published.manifest.aggregate_digest,
                )
            except BaseException as exc:
                commit_local = _error_status(
                    phase="commit", rank=rank, world_size=plan.world_size, exc=exc
                )
    else:
        commit_local = _ok_status(phase="commit", rank=rank, world_size=plan.world_size)
    barrier()
    commit_statuses = _gather_statuses(
        gather_status,
        commit_local,
        phase="commit",
        world_size=plan.world_size,
    )
    failures = root_failures or _failure_rows(commit_statuses)
    if failures:
        if rank == 0:
            try:
                if session is not None:
                    abort_training_state_contributions(checkpoint_dir, session)
                    if os.path.lexists(session.stage_path):
                        _fail(
                            "owned contribution stage remains after abort",
                            code="training_state.contribution_abort_failed",
                            context={"stage_path": str(session.stage_path)},
                        )
                abort_local = _ok_status(
                    phase="abort", rank=rank, world_size=plan.world_size
                )
            except BaseException as exc:
                abort_local = _error_status(
                    phase="abort", rank=rank, world_size=plan.world_size, exc=exc
                )
        else:
            abort_local = _ok_status(
                phase="abort", rank=rank, world_size=plan.world_size
            )
        abort_statuses = _gather_statuses(
            gather_status,
            abort_local,
            phase="abort",
            world_size=plan.world_size,
        )
        abort_failures = _failure_rows(abort_statuses)
        if abort_failures:
            raise ArtifactContractError(
                "distributed exact-state publication abort failed",
                code="training_state.distributed_publication_abort_failed",
                context={
                    "abort_failures": abort_failures,
                    "publication_failures": failures,
                },
            )
        raise ArtifactContractError(
            "distributed exact-state publication failed",
            code="training_state.distributed_publication_failed",
            context={"failures": failures},
        )
    expected_digest = commit_statuses[0].manifest_digest
    if expected_digest is None:
        _fail(
            "rank-zero commit status has no manifest digest",
            code="training_state.distributed_control",
            context={"phase": "commit"},
        )
    if published is None:
        manifest = load_training_state_manifest(checkpoint_dir)
        published = PublishedTrainingState(
            path=(
                Path(checkpoint_dir).expanduser().absolute() / TRAINING_STATE_DIRECTORY
            ),
            manifest=manifest,
        )
    if published.manifest.aggregate_digest != expected_digest:
        _fail(
            "rank-local committed manifest differs from rank-zero status",
            code="training_state.publication_reload_mismatch",
            context={
                "expected": expected_digest,
                "observed": published.manifest.aggregate_digest,
            },
        )
    return published


def _admit_current_rank_for_restore(
    checkpoint_dir: Path | str,
    *,
    checkpoint_step: int,
    rank: int,
    world_size: int,
    identities: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
) -> AdmittedTrainingState:
    current_config_digest = hashlib.sha256(
        _canonical_json_bytes(resolved_config, field="resolved_config") + b"\n"
    ).hexdigest()
    if identities.get("resolved_config") != current_config_digest:
        _fail(
            "current resolved configuration does not match its identity",
            code="training_state.incompatible",
            context={"field": "identities.resolved_config"},
        )
    current_resume_compatibility = build_resume_compatibility_projection(
        resolved_config
    )
    current_resume_digest = hashlib.sha256(
        _canonical_json_bytes(
            current_resume_compatibility, field="resume_compatibility"
        )
        + b"\n"
    ).hexdigest()
    if identities.get("resume_compatibility") != current_resume_digest:
        _fail(
            "current resume compatibility does not match its identity",
            code="training_state.incompatible",
            context={"field": "identities.resume_compatibility"},
        )
    expectations = TrainingStateExpectations(
        checkpoint_step=checkpoint_step,
        world_size=world_size,
        identities=identities,
        scheduler_applicable=scheduler is not None,
        scaler_applicable=scaler is not None,
        resolved_config=resolved_config,
        runtime_state=capture_runtime_state_expectations(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        ),
        resume_compatibility=current_resume_compatibility,
    )
    admitted = admit_training_state(
        checkpoint_dir,
        expectations,
        current_rank=rank,
    )
    assert isinstance(admitted, AdmittedTrainingState)
    return admitted


def _validate_decoded_topology(
    admitted: AdmittedTrainingState,
    *,
    rank: int,
    world_size: int,
    cuda_device_topology: Sequence[str],
) -> None:
    decoded = admitted.decoded_rank
    mismatches: list[dict[str, Any]] = []
    for field, checkpoint_value, current_value in (
        ("rank", decoded.rank, rank),
        ("world_size", decoded.world_size, world_size),
        (
            "cuda_device_topology",
            list(decoded.cuda_device_topology),
            list(cuda_device_topology),
        ),
    ):
        if checkpoint_value != current_value:
            mismatches.append(
                {
                    "checkpoint": checkpoint_value,
                    "current": current_value,
                    "field": field,
                }
            )
    if mismatches:
        _fail(
            "current rank topology cannot restore the admitted state",
            code="training_state.incompatible",
            context={"mismatches": mismatches},
        )


def _capture_runtime_rollback_snapshot(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    get_torch_cuda_rng_states: Callable[[], Sequence[torch.Tensor]],
) -> _RuntimeRollbackSnapshot:
    numpy_state = np.random.get_state()
    return _RuntimeRollbackSnapshot(
        trainable_model=MappingProxyType(
            {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        ),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=(
            copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
        ),
        scaler=copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        python_rng=random.getstate(),
        numpy_rng=(
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch_cpu_rng=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda_rng=tuple(
            state.detach().cpu().clone() for state in get_torch_cuda_rng_states()
        ),
    )


def _restore_runtime_rollback_snapshot(
    snapshot: _RuntimeRollbackSnapshot,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    set_torch_cuda_rng_states: Callable[[Sequence[torch.Tensor]], None],
) -> None:
    errors: list[str] = []

    def attempt(owner: str, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:
            errors.append(f"{owner}:{type(exc).__name__}")

    def restore_model() -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in snapshot.trainable_model.items():
                parameter = parameters[name]
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )

    attempt("model", restore_model)
    attempt("optimizer", lambda: optimizer.load_state_dict(snapshot.optimizer))
    if scheduler is not None:
        assert snapshot.scheduler is not None
        attempt("scheduler", lambda: scheduler.load_state_dict(snapshot.scheduler))
    if scaler is not None:
        assert snapshot.scaler is not None
        attempt("scaler", lambda: scaler.load_state_dict(snapshot.scaler))
    attempt("python_rng", lambda: random.setstate(snapshot.python_rng))
    attempt("numpy_rng", lambda: np.random.set_state(snapshot.numpy_rng))
    attempt("torch_cpu_rng", lambda: torch.set_rng_state(snapshot.torch_cpu_rng))
    attempt(
        "torch_cuda_rng",
        lambda: set_torch_cuda_rng_states(snapshot.torch_cuda_rng),
    )
    if errors:
        _fail(
            "runtime rollback did not restore every owner",
            code="training_state.restore_rollback_failed",
            context={"rollback_errors": errors},
        )


def _get_current_cuda_rng_states() -> tuple[torch.Tensor, ...]:
    device = torch.cuda.current_device()
    return (torch.cuda.get_rng_state(device).detach().cpu().clone(),)


def _set_current_cuda_rng_states(states: Sequence[torch.Tensor]) -> None:
    if len(states) != 1:
        _fail(
            "rank-owned CUDA RNG restore requires exactly one state",
            code="training_state.incompatible",
            context={"cuda_rng_state_count": len(states)},
        )
    torch.cuda.set_rng_state(states[0], device=torch.cuda.current_device())


def _gather_restore_statuses(
    gather_status: Callable[
        [DistributedExactResumeRestoreStatus],
        Sequence[DistributedExactResumeRestoreStatus],
    ],
    local_status: DistributedExactResumeRestoreStatus,
    *,
    phase: str,
    world_size: int,
) -> tuple[DistributedExactResumeRestoreStatus, ...]:
    observed = gather_status(local_status)
    if not isinstance(observed, Sequence) or isinstance(
        observed, (str, bytes, bytearray)
    ):
        _fail(
            "restore status gather must return a sequence",
            code="training_state.distributed_control",
            context={"phase": phase},
        )
    statuses = tuple(observed)
    if (
        len(statuses) != world_size
        or any(
            not isinstance(item, DistributedExactResumeRestoreStatus)
            for item in statuses
        )
        or tuple(item.rank for item in statuses) != tuple(range(world_size))
        or any(item.world_size != world_size for item in statuses)
        or any(item.phase != phase for item in statuses)
    ):
        _fail(
            "restore status gather does not contain the complete ordered rank set",
            code="training_state.distributed_control",
            context={
                "phase": phase,
                "world_size": world_size,
                "observed_world_sizes": [
                    getattr(item, "world_size", None) for item in statuses
                ],
            },
        )
    for item in statuses:
        if item.ok != (item.error_code is None and item.error_type is None):
            _fail(
                "restore status success and error fields disagree",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.applied and not (phase == "restore_apply" and item.ok):
            _fail(
                "applied flag appeared outside successful restore apply",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.rolled_back and not (phase == "restore_rollback" and item.ok):
            _fail(
                "rollback flag appeared outside successful restore rollback",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.ok:
            if item.manifest_digest is None:
                _fail(
                    "successful restore status has no manifest digest",
                    code="training_state.distributed_control",
                    context={"phase": phase, "rank": item.rank},
                )
            _require_sha256(item.manifest_digest, field="status.manifest_digest")
        elif item.manifest_digest is not None:
            _fail(
                "failed restore status unexpectedly has a manifest digest",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
    successful_digests = {
        item.manifest_digest for item in statuses if item.manifest_digest is not None
    }
    if len(successful_digests) > 1:
        _fail(
            "restore statuses disagree on the authenticated manifest",
            code="training_state.distributed_control",
            context={"phase": phase},
        )
    return statuses


def _restore_error_status(
    *, phase: str, rank: int, world_size: int, exc: BaseException
) -> DistributedExactResumeRestoreStatus:
    return DistributedExactResumeRestoreStatus(
        phase=phase,
        rank=rank,
        world_size=world_size,
        ok=False,
        error_code=getattr(exc, "code", "training_state.unexpected_failure"),
        error_type=type(exc).__name__,
    )


def _restore_failure_rows(
    statuses: Sequence[DistributedExactResumeRestoreStatus],
) -> list[dict[str, Any]]:
    return [
        {
            "error_code": status.error_code,
            "error_type": status.error_type,
            "phase": status.phase,
            "rank": status.rank,
        }
        for status in statuses
        if not status.ok
    ]


def _gather_statuses(
    gather_status: Callable[
        [DistributedExactResumeStatus], Sequence[DistributedExactResumeStatus]
    ],
    local_status: DistributedExactResumeStatus,
    *,
    phase: str,
    world_size: int,
) -> tuple[DistributedExactResumeStatus, ...]:
    observed = gather_status(local_status)
    if not isinstance(observed, Sequence) or isinstance(
        observed, (str, bytes, bytearray)
    ):
        _fail(
            "status gather must return a sequence",
            code="training_state.distributed_control",
            context={"phase": phase},
        )
    statuses = tuple(observed)
    if (
        len(statuses) != world_size
        or any(not isinstance(item, DistributedExactResumeStatus) for item in statuses)
        or tuple(item.rank for item in statuses) != tuple(range(world_size))
        or any(item.world_size != world_size for item in statuses)
        or any(item.phase != phase for item in statuses)
    ):
        _fail(
            "status gather does not contain the complete ordered rank set",
            code="training_state.distributed_control",
            context={
                "phase": phase,
                "world_size": world_size,
                "observed_ranks": [getattr(item, "rank", None) for item in statuses],
                "observed_world_sizes": [
                    getattr(item, "world_size", None) for item in statuses
                ],
            },
        )
    for item in statuses:
        if item.ok != (item.error_code is None and item.error_type is None):
            _fail(
                "distributed status success and error fields disagree",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.session is not None and not (
            phase == "begin" and item.rank == 0 and item.ok
        ):
            _fail(
                "contribution session appeared outside rank-zero begin status",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.manifest_digest is not None and not (
            phase == "commit" and item.rank == 0 and item.ok
        ):
            _fail(
                "manifest digest appeared outside rank-zero commit status",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
        if item.manifest_digest is not None:
            _require_sha256(item.manifest_digest, field="status.manifest_digest")
        if item.plan_digest is not None:
            if phase != "serialize":
                _fail(
                    "plan digest appeared outside serialization status",
                    code="training_state.distributed_control",
                    context={"phase": phase, "rank": item.rank},
                )
            _require_sha256(item.plan_digest, field="status.plan_digest")
        if phase == "serialize" and item.ok and item.plan_digest is None:
            _fail(
                "successful serialization status has no plan digest",
                code="training_state.distributed_control",
                context={"phase": phase, "rank": item.rank},
            )
    return statuses


def _ok_status(
    *,
    phase: str,
    rank: int,
    world_size: int,
    session: TrainingStateContributionSession | None = None,
    manifest_digest: str | None = None,
    plan_digest: str | None = None,
) -> DistributedExactResumeStatus:
    return DistributedExactResumeStatus(
        phase=phase,
        rank=rank,
        world_size=world_size,
        ok=True,
        session=session,
        manifest_digest=manifest_digest,
        plan_digest=plan_digest,
    )


def _error_status(
    *,
    phase: str,
    rank: int,
    world_size: int,
    exc: BaseException,
    plan_digest: str | None = None,
) -> DistributedExactResumeStatus:
    return DistributedExactResumeStatus(
        phase=phase,
        rank=rank,
        world_size=world_size,
        ok=False,
        error_code=getattr(exc, "code", "training_state.unexpected_failure"),
        error_type=type(exc).__name__,
        plan_digest=plan_digest,
    )


def _failure_rows(
    statuses: Sequence[DistributedExactResumeStatus],
) -> list[dict[str, Any]]:
    return [
        {
            "error_code": status.error_code,
            "error_type": status.error_type,
            "phase": status.phase,
            "rank": status.rank,
        }
        for status in statuses
        if not status.ok
    ]


def _resolve_physical_cuda_device(logical_device: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        tokens = tuple(token.strip() for token in visible_devices.split(","))
        if logical_device < len(tokens) and tokens[logical_device]:
            return tokens[logical_device]
    return f"cuda:{logical_device}"


def _validate_cuda_device_binding(
    binding: RankCudaDeviceBinding, *, rank: int, world_size: int
) -> None:
    if not isinstance(binding, RankCudaDeviceBinding):
        _fail(
            "CUDA device binding has an unsupported type",
            code="training_state.schema",
            context={"field": "cuda_device_binding"},
        )
    _validate_rank(rank=binding.rank, world_size=binding.world_size)
    if binding.rank != rank or binding.world_size != world_size:
        _fail(
            "CUDA device binding belongs to another rank or world",
            code="training_state.rank_mismatch",
            context={"rank": rank, "world_size": world_size},
        )
    if (
        isinstance(binding.logical_device, bool)
        or not isinstance(binding.logical_device, int)
        or binding.logical_device < 0
    ):
        _fail(
            "logical CUDA device must be a nonnegative integer",
            code="training_state.schema",
            context={"field": "cuda_device_binding.logical_device"},
        )
    if not isinstance(binding.physical_device, str) or not binding.physical_device:
        _fail(
            "physical CUDA device must be a nonempty string",
            code="training_state.schema",
            context={"field": "cuda_device_binding.physical_device"},
        )


def _validate_rng_snapshot(
    snapshot: RankRngSnapshot, *, rank: int, world_size: int
) -> None:
    if not isinstance(snapshot, RankRngSnapshot):
        _fail(
            "rng_snapshot has an unsupported type",
            code="training_state.schema",
            context={"field": "rng_snapshot"},
        )
    if snapshot.rank != rank or snapshot.world_size != world_size:
        _fail(
            "RNG snapshot is bound to another rank or world",
            code="training_state.rank_mismatch",
            context={
                "current_rank": rank,
                "current_world_size": world_size,
                "snapshot_rank": snapshot.rank,
                "snapshot_world_size": snapshot.world_size,
            },
        )
    if snapshot.cuda_device_binding is None:
        _fail(
            "RNG snapshot has no explicit CUDA device binding",
            code="training_state.incomplete",
            context={"field": "rng_snapshot.cuda_device_binding"},
        )
    _validate_cuda_device_binding(
        snapshot.cuda_device_binding, rank=rank, world_size=world_size
    )
    expected_topology = ("cuda:0",)
    if snapshot.cuda_device_topology != expected_topology:
        _fail(
            "RNG snapshot topology is not the canonical local CUDA inventory",
            code="training_state.incompatible",
            context={
                "expected": list(expected_topology),
                "observed": list(snapshot.cuda_device_topology),
            },
        )
    if len(snapshot.torch_cuda_states) != 1:
        _fail(
            "RNG snapshot must contain exactly the rank-owned CUDA state",
            code="training_state.incompatible",
            context={"cuda_rng_state_count": len(snapshot.torch_cuda_states)},
        )
    states = (snapshot.torch_cpu_state, *snapshot.torch_cuda_states)
    if any(
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
        or state.numel() == 0
        for state in states
    ):
        _fail(
            "Torch RNG states must be nonempty uint8 vectors",
            code="training_state.runtime_state",
            context={"field": "rng_snapshot"},
        )


def _validate_rank(*, rank: int, world_size: int) -> None:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        _fail(
            "world_size must be a positive integer",
            code="training_state.schema",
            context={"world_size": world_size},
        )
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= world_size
    ):
        _fail(
            "rank must belong to the declared world",
            code="training_state.schema",
            context={"rank": rank, "world_size": world_size},
        )


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            f"{field} must be a lowercase SHA-256 digest",
            code="training_state.schema",
            context={"field": field},
        )
    return value


def _canonical_json_bytes(value: Any, *, field: str) -> bytes:
    normalized = _normalize_json(value, field=field)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalize_json(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        _fail(
            f"{field} contains a non-finite float",
            code="training_state.schema",
            context={"field": field},
        )
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _normalize_json(item, field=f"{field}.{key}")
            for key, item in sorted(value.items())
        }
    _fail(
        f"{field} contains a value outside strict JSON",
        code="training_state.schema",
        context={"field": field, "value_type": type(value).__name__},
    )
    raise AssertionError("unreachable")


def _fail(
    message: str,
    *,
    code: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    raise ArtifactContractError(message, code=code, context=context)
