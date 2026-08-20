"""Strict exact-training-state publication and admission primitives."""

from __future__ import annotations

import ctypes
import copy
import errno
import hashlib
import io
import json
import math
import os
import random
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np
import torch

from src.common.errors import ArtifactContractError


TRAINING_STATE_DIRECTORY = "training_state"
TRAINING_STATE_MANIFEST = "manifest.json"
TRAINING_STATE_SCHEMA = "coordexp-swift-exact-training-state"
TRAINING_STATE_SCHEMA_VERSION = 2
TRAINING_STATE_ARTIFACT_TYPE = "exact_training_state"
TRAINING_STATE_COMMIT_STATUS = "committed"
TRAINING_STATE_SAVE_BOUNDARY = "optimizer_step"
TRAINING_STATE_CURSOR_SCHEMA = "coordexp-swift-exact-rank-cursor"
TRAINING_STATE_CURSOR_SCHEMA_VERSION = 1
TRAINING_STATE_DATA_CURSOR_SCHEMA = "coordexp-swift-data-cursor"
TRAINING_STATE_PACK_CURSOR_SCHEMA = "coordexp-swift-pack-cursor"
TRAINING_STATE_RESOLVED_CONFIG = "resolved_config.json"
TRAINING_STATE_RESUME_COMPATIBILITY = "resume_compatibility.json"
TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA = "coordexp-swift-exact-resume-compatibility"
TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA_VERSION = 1
TRAINING_STATE_CONTRIBUTION_PLAN = "contribution-plan.json"
TRAINING_STATE_RANK_CONTRIBUTION = "contribution.json"
TRAINING_STATE_TERMINAL_FORENSIC = "terminal-forensic.json"
TRAINING_STATE_CONTRIBUTION_SCHEMA = "coordexp-swift-training-state-contributions"
TRAINING_STATE_CONTRIBUTION_SCHEMA_VERSION = 1
TRAINING_STATE_TRAINABLE_MODEL_SCHEMA = "coordexp-swift-trainable-model-state"
TRAINING_STATE_OPTIMIZER_SCHEMA = "coordexp-swift-optimizer-state"
TRAINING_STATE_SCHEDULER_SCHEMA = "coordexp-swift-scheduler-state"
TRAINING_STATE_SCALER_SCHEMA = "coordexp-swift-scaler-state"
TRAINING_STATE_PYTHON_RNG_SCHEMA = "coordexp-swift-python-rng-state"
TRAINING_STATE_NUMPY_RNG_SCHEMA = "coordexp-swift-numpy-rng-state"
TRAINING_STATE_TORCH_CPU_RNG_SCHEMA = "coordexp-swift-torch-cpu-rng-state"
TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA = "coordexp-swift-torch-cuda-rng-state"
TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA_VERSION = 2
TRAINING_STATE_RUNTIME_COMPONENT_VERSION = 1
REQUIRED_IDENTITY_KINDS = (
    "base_model",
    "cache",
    "dependencies",
    "policy",
    "resolved_config",
    "resume_compatibility",
    "topology",
    "trainable_surface",
)
REQUIRED_RNG_KINDS = ("numpy", "python", "torch_cpu", "torch_cuda")

_SHA256_LENGTH = 64
_MANIFEST_FIELDS = frozenset(
    {
        "aggregate_digest",
        "applicability",
        "artifact_type",
        "checkpoint_step",
        "commit_status",
        "continuation_index",
        "identities",
        "parent_run_id",
        "parent_segment_id",
        "ranks",
        "resolved_config",
        "resume_compatibility",
        "save_boundary",
        "schema",
        "schema_version",
        "world_size",
    }
)
_RANK_FIELDS = frozenset(
    {
        "files",
        "next_rank_local_micro_step",
        "rank",
        "rng_kinds",
        "runtime_signature",
    }
)
_FILE_FIELDS = frozenset({"path", "role", "sha256", "size"})
_APPLICABILITY_FIELDS = frozenset({"optimizer", "scaler", "scheduler"})
_CURSOR_FIELDS = frozenset(
    {
        "data",
        "next_rank_local_micro_step",
        "pack",
        "schema",
        "schema_version",
    }
)
_CURSOR_OWNER_FIELDS = frozenset(
    {"next_rank_local_micro_step", "owner", "schema", "schema_version", "state"}
)
_PUBLICATION_PLAN_FIELDS = frozenset(
    {
        "accumulation_microstep",
        "checkpoint_step",
        "continuation_index",
        "identities",
        "parent_run_id",
        "parent_segment_id",
        "resolved_config",
        "resume_compatibility",
        "scaler_applicable",
        "scheduler_applicable",
        "world_size",
    }
)
_T = TypeVar("_T")


#: Top-level resolved-config blocks that carry NO training semantics and are
#: therefore projected out of exact-resume compatibility.
#:
#: * `run` and `resume` are continuation identity: a child necessarily differs.
#: * `observability` is rank-zero PRESENTATION only (console/TensorBoard
#:   cadence).  DECLARED FLIP (add-coordexp-swift-training-observability, task
#:   5.1): changing only `observability.steps` between an admitted parent and
#:   its continuation must not make otherwise identical training state
#:   incompatible.  This WIDENS admission, so no schema-version fence is
#:   added: an older checkpoint whose config predates the block projects
#:   byte-identically, and no previously admitted continuation is refused.
#:
#: Everything else -- forward, loss, optimizer, scheduler, data order, RNG,
#: cadence, and precision -- stays strict.
_RESUME_NON_SEMANTIC_CONFIG_BLOCKS = frozenset({"observability", "resume", "run"})


def build_resume_compatibility_projection(
    resolved_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project a full resolved-config artifact onto exact-resume semantics."""

    normalized = _require_mapping(
        _normalize_json_value(resolved_config, field="resolved_config"),
        field="resolved_config",
    )
    _require_exact_fields(
        normalized, frozenset({"config", "resolution"}), field="resolved_config"
    )
    config = _require_mapping(normalized["config"], field="resolved_config.config")
    _require_mapping(normalized["resolution"], field="resolved_config.resolution")
    if not config:
        _fail(
            "resolved training configuration must be nonempty",
            code="training_state.incomplete",
            context={"field": "resolved_config.config"},
        )
    missing_continuation_fields = {"run", "resume"} - set(config)
    if missing_continuation_fields:
        _fail(
            "resolved training configuration lacks continuation metadata",
            code="training_state.schema",
            context={
                "field": "resolved_config.config",
                "missing": sorted(missing_continuation_fields),
            },
        )
    semantic_config = {
        key: copy.deepcopy(value)
        for key, value in sorted(config.items())
        if key not in _RESUME_NON_SEMANTIC_CONFIG_BLOCKS
    }
    if not semantic_config:
        _fail(
            "resume compatibility projection has no training semantics",
            code="training_state.incomplete",
        )
    return MappingProxyType(
        {
            "schema": TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA,
            "schema_version": TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA_VERSION,
            "semantic_config": semantic_config,
        }
    )


@dataclass(frozen=True)
class RankTrainingStatePayload:
    """Opaque serialized state contributed by one rank."""

    rank: int
    trainable_model: bytes
    optimizer: bytes
    scheduler: bytes | None
    scaler: bytes | None
    rng: Mapping[str, bytes]
    cursor: Mapping[str, Any]
    next_rank_local_micro_step: int


@dataclass(frozen=True)
class TrainingStatePublication:
    """Complete caller-owned input needed to build one immutable publication."""

    parent_run_id: str
    parent_segment_id: str
    checkpoint_step: int
    continuation_index: int
    world_size: int
    identities: Mapping[str, str]
    scheduler_applicable: bool
    scaler_applicable: bool
    rank_payloads: Sequence[RankTrainingStatePayload]
    accumulation_microstep: int = 0
    resolved_config: Mapping[str, Any] | None = None
    resume_compatibility: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TrainingStatePublicationPlan:
    """Small rank-shared publication metadata with no runtime payload bytes."""

    parent_run_id: str
    parent_segment_id: str
    checkpoint_step: int
    continuation_index: int
    world_size: int
    identities: Mapping[str, str]
    scheduler_applicable: bool
    scaler_applicable: bool
    resolved_config: Mapping[str, Any]
    resume_compatibility: Mapping[str, Any]
    accumulation_microstep: int = 0

    @classmethod
    def from_publication(
        cls, publication: TrainingStatePublication
    ) -> TrainingStatePublicationPlan:
        if publication.resolved_config is None:
            _fail(
                "publication has no resolved configuration",
                code="training_state.incomplete",
            )
        if publication.resume_compatibility is None:
            _fail(
                "publication has no resume compatibility projection",
                code="training_state.incomplete",
            )
        return cls(
            parent_run_id=publication.parent_run_id,
            parent_segment_id=publication.parent_segment_id,
            checkpoint_step=publication.checkpoint_step,
            continuation_index=publication.continuation_index,
            world_size=publication.world_size,
            identities=publication.identities,
            scheduler_applicable=publication.scheduler_applicable,
            scaler_applicable=publication.scaler_applicable,
            resolved_config=publication.resolved_config,
            resume_compatibility=publication.resume_compatibility,
            accumulation_microstep=publication.accumulation_microstep,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accumulation_microstep": self.accumulation_microstep,
            "checkpoint_step": self.checkpoint_step,
            "continuation_index": self.continuation_index,
            "identities": dict(sorted(self.identities.items())),
            "parent_run_id": self.parent_run_id,
            "parent_segment_id": self.parent_segment_id,
            "resolved_config": _normalize_json_value(
                self.resolved_config, field="publication_plan.resolved_config"
            ),
            "resume_compatibility": _normalize_json_value(
                self.resume_compatibility,
                field="publication_plan.resume_compatibility",
            ),
            "scaler_applicable": self.scaler_applicable,
            "scheduler_applicable": self.scheduler_applicable,
            "world_size": self.world_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingStatePublicationPlan:
        _require_exact_fields(value, _PUBLICATION_PLAN_FIELDS, field="publication_plan")
        plan = cls(
            parent_run_id=value["parent_run_id"],
            parent_segment_id=value["parent_segment_id"],
            checkpoint_step=value["checkpoint_step"],
            continuation_index=value["continuation_index"],
            world_size=value["world_size"],
            identities=_require_mapping(
                value["identities"], field="publication_plan.identities"
            ),
            scheduler_applicable=value["scheduler_applicable"],
            scaler_applicable=value["scaler_applicable"],
            resolved_config=_require_mapping(
                value["resolved_config"], field="publication_plan.resolved_config"
            ),
            resume_compatibility=_require_mapping(
                value["resume_compatibility"],
                field="publication_plan.resume_compatibility",
            ),
            accumulation_microstep=value["accumulation_microstep"],
        )
        _validate_publication_plan(plan)
        return plan

    @property
    def digest(self) -> str:
        return _sha256(_canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class TrainingStateContributionSession:
    checkpoint_path: Path
    checkpoint_device: int
    checkpoint_inode: int
    stage_name: str
    plan_digest: str
    session_id: str
    world_size: int

    @property
    def stage_path(self) -> Path:
        return self.checkpoint_path / self.stage_name


@dataclass(frozen=True)
class PublishedRankContribution:
    path: Path
    rank: int
    runtime_signature: str


@dataclass(frozen=True)
class TrainingStateExpectations:
    """Current-run compatibility keys checked before any restore callback."""

    checkpoint_step: int
    world_size: int
    identities: Mapping[str, str]
    scheduler_applicable: bool
    scaler_applicable: bool
    rng_kinds: Sequence[str] = REQUIRED_RNG_KINDS
    resolved_config: Mapping[str, Any] | None = None
    runtime_state: RuntimeStateExpectations | None = None
    resume_compatibility: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeStateExpectations:
    """Structural runtime signature expected before mutable state restore."""

    structure: Mapping[str, Any]
    signature: str


@dataclass(frozen=True)
class DecodedRankTrainingState:
    """Repository-decoded state safe to hand to the restore callback."""

    rank: int
    world_size: int
    trainable_model: Mapping[str, torch.Tensor]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any] | None
    scaler: Mapping[str, Any] | None
    python_rng_state: tuple[Any, ...]
    numpy_rng_state: tuple[Any, ...]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]
    cuda_device_topology: tuple[str, ...]
    cuda_device_count: int
    cursor: Mapping[str, Any]
    structure: Mapping[str, Any]
    signature: str


@dataclass(frozen=True)
class RestoredRankTrainingState:
    cursor: Mapping[str, Any]
    next_rank_local_micro_step: int
    runtime_signature: str


@dataclass(frozen=True)
class TrainingStateFile:
    path: str
    role: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingStateFile":
        _require_exact_fields(value, _FILE_FIELDS, field="file")
        path = _require_safe_relative_path(value["path"], field="file.path")
        role = _require_nonempty_string(value["role"], field="file.role")
        size = _require_positive_int(value["size"], field="file.size")
        sha256 = _require_sha256(value["sha256"], field="file.sha256")
        return cls(path=path, role=role, size=size, sha256=sha256)


@dataclass(frozen=True)
class TrainingStateRank:
    rank: int
    rng_kinds: tuple[str, ...]
    next_rank_local_micro_step: int
    runtime_signature: str
    files: tuple[TrainingStateFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "next_rank_local_micro_step": self.next_rank_local_micro_step,
            "rank": self.rank,
            "rng_kinds": list(self.rng_kinds),
            "runtime_signature": self.runtime_signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingStateRank":
        _require_exact_fields(value, _RANK_FIELDS, field="rank")
        rank = _require_nonnegative_int(value["rank"], field="rank.rank")
        next_microstep = _require_nonnegative_int(
            value["next_rank_local_micro_step"],
            field="rank.next_rank_local_micro_step",
        )
        rng_kinds = _require_string_sequence(value["rng_kinds"], field="rank.rng_kinds")
        runtime_signature = _require_sha256(
            value["runtime_signature"], field="rank.runtime_signature"
        )
        raw_files = value["files"]
        if not isinstance(raw_files, list) or not raw_files:
            _fail("rank.files must be a nonempty list", code="training_state.schema")
        files = tuple(
            TrainingStateFile.from_dict(_require_mapping(item, field="rank.files[]"))
            for item in raw_files
        )
        if tuple(sorted(files, key=lambda item: item.path)) != files:
            _fail("rank files must be sorted by path", code="training_state.schema")
        paths = [item.path for item in files]
        roles = [item.role for item in files]
        if len(paths) != len(set(paths)) or len(roles) != len(set(roles)):
            _fail(
                "rank file paths and roles must be unique",
                code="training_state.schema",
                context={"rank": rank},
            )
        return cls(
            rank=rank,
            rng_kinds=rng_kinds,
            next_rank_local_micro_step=next_microstep,
            runtime_signature=runtime_signature,
            files=files,
        )


@dataclass(frozen=True)
class TrainingStateManifest:
    schema: str
    schema_version: int
    artifact_type: str
    commit_status: str
    parent_run_id: str
    parent_segment_id: str
    checkpoint_step: int
    continuation_index: int
    world_size: int
    save_boundary: str
    identities: Mapping[str, str]
    optimizer_applicable: bool
    scheduler_applicable: bool
    scaler_applicable: bool
    resolved_config: TrainingStateFile
    resume_compatibility: TrainingStateFile
    ranks: tuple[TrainingStateRank, ...]
    aggregate_digest: str

    def to_dict(self, *, include_aggregate_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "applicability": {
                "optimizer": self.optimizer_applicable,
                "scaler": self.scaler_applicable,
                "scheduler": self.scheduler_applicable,
            },
            "artifact_type": self.artifact_type,
            "checkpoint_step": self.checkpoint_step,
            "commit_status": self.commit_status,
            "continuation_index": self.continuation_index,
            "identities": dict(sorted(self.identities.items())),
            "parent_run_id": self.parent_run_id,
            "parent_segment_id": self.parent_segment_id,
            "ranks": [rank.to_dict() for rank in self.ranks],
            "resolved_config": self.resolved_config.to_dict(),
            "resume_compatibility": self.resume_compatibility.to_dict(),
            "save_boundary": self.save_boundary,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "world_size": self.world_size,
        }
        if include_aggregate_digest:
            value["aggregate_digest"] = self.aggregate_digest
        return value

    def computed_aggregate_digest(self) -> str:
        return _sha256(
            _canonical_json_bytes(self.to_dict(include_aggregate_digest=False))
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingStateManifest":
        if (
            value.get("schema") == TRAINING_STATE_SCHEMA
            and value.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION
        ):
            _fail(
                "training-state schema is unsupported",
                code="training_state.unsupported_schema",
                context={
                    "schema": value.get("schema"),
                    "schema_version": value.get("schema_version"),
                },
            )
        _require_exact_fields(value, _MANIFEST_FIELDS, field="manifest")
        schema = _require_nonempty_string(value["schema"], field="manifest.schema")
        schema_version = _require_positive_int(
            value["schema_version"], field="manifest.schema_version"
        )
        artifact_type = _require_nonempty_string(
            value["artifact_type"], field="manifest.artifact_type"
        )
        commit_status = _require_nonempty_string(
            value["commit_status"], field="manifest.commit_status"
        )
        if (
            schema != TRAINING_STATE_SCHEMA
            or schema_version != TRAINING_STATE_SCHEMA_VERSION
        ):
            _fail(
                "training-state schema is unsupported",
                code="training_state.unsupported_schema",
                context={"schema": schema, "schema_version": schema_version},
            )
        if artifact_type != TRAINING_STATE_ARTIFACT_TYPE:
            _fail(
                "artifact is not exact training state",
                code="training_state.model_only",
                context={"artifact_type": artifact_type},
            )
        if commit_status != TRAINING_STATE_COMMIT_STATUS:
            _fail(
                "training-state manifest is not committed",
                code="training_state.uncommitted",
                context={"commit_status": commit_status},
            )
        parent_run_id = _require_nonempty_string(
            value["parent_run_id"], field="manifest.parent_run_id"
        )
        parent_segment_id = _require_nonempty_string(
            value["parent_segment_id"], field="manifest.parent_segment_id"
        )
        checkpoint_step = _require_positive_int(
            value["checkpoint_step"], field="manifest.checkpoint_step"
        )
        continuation_index = _require_nonnegative_int(
            value["continuation_index"], field="manifest.continuation_index"
        )
        world_size = _require_positive_int(
            value["world_size"], field="manifest.world_size"
        )
        save_boundary = _require_nonempty_string(
            value["save_boundary"], field="manifest.save_boundary"
        )
        if save_boundary != TRAINING_STATE_SAVE_BOUNDARY:
            _fail(
                "mid-accumulation training-state checkpoints are unsupported",
                code="training_state.unsupported_mid_accumulation",
                context={"save_boundary": save_boundary},
            )
        identities = _validate_identities(value["identities"])
        applicability = _require_mapping(
            value["applicability"], field="manifest.applicability"
        )
        _require_exact_fields(
            applicability, _APPLICABILITY_FIELDS, field="manifest.applicability"
        )
        optimizer_applicable = _require_bool(
            applicability["optimizer"], field="manifest.applicability.optimizer"
        )
        scheduler_applicable = _require_bool(
            applicability["scheduler"], field="manifest.applicability.scheduler"
        )
        scaler_applicable = _require_bool(
            applicability["scaler"], field="manifest.applicability.scaler"
        )
        resolved_config = TrainingStateFile.from_dict(
            _require_mapping(value["resolved_config"], field="manifest.resolved_config")
        )
        resume_compatibility = TrainingStateFile.from_dict(
            _require_mapping(
                value["resume_compatibility"],
                field="manifest.resume_compatibility",
            )
        )
        if not optimizer_applicable:
            _fail(
                "exact training state requires optimizer state",
                code="training_state.incomplete",
            )
        raw_ranks = value["ranks"]
        if not isinstance(raw_ranks, list) or not raw_ranks:
            _fail(
                "manifest ranks must be a nonempty list",
                code="training_state.incomplete",
            )
        ranks = tuple(
            TrainingStateRank.from_dict(
                _require_mapping(item, field="manifest.ranks[]")
            )
            for item in raw_ranks
        )
        if tuple(rank.rank for rank in ranks) != tuple(range(world_size)):
            _fail(
                "manifest must contain the complete ordered rank set",
                code="training_state.incomplete_rank_set",
                context={
                    "expected_ranks": list(range(world_size)),
                    "observed_ranks": [rank.rank for rank in ranks],
                },
            )
        aggregate_digest = _require_sha256(
            value["aggregate_digest"], field="manifest.aggregate_digest"
        )
        manifest = cls(
            schema=schema,
            schema_version=schema_version,
            artifact_type=artifact_type,
            commit_status=commit_status,
            parent_run_id=parent_run_id,
            parent_segment_id=parent_segment_id,
            checkpoint_step=checkpoint_step,
            continuation_index=continuation_index,
            world_size=world_size,
            save_boundary=save_boundary,
            identities=MappingProxyType(identities),
            optimizer_applicable=optimizer_applicable,
            scheduler_applicable=scheduler_applicable,
            scaler_applicable=scaler_applicable,
            resolved_config=resolved_config,
            resume_compatibility=resume_compatibility,
            ranks=ranks,
            aggregate_digest=aggregate_digest,
        )
        _validate_manifest_roles(manifest)
        computed = manifest.computed_aggregate_digest()
        if computed != aggregate_digest:
            _fail(
                "training-state aggregate digest is invalid",
                code="training_state.corrupt_manifest",
                context={"expected": aggregate_digest, "observed": computed},
            )
        return manifest


@dataclass(frozen=True)
class PublishedTrainingState:
    path: Path
    manifest: TrainingStateManifest


@dataclass(frozen=True)
class AdmittedTrainingState:
    path: Path
    manifest: TrainingStateManifest
    files: Mapping[str, bytes]
    resolved_config: Mapping[str, Any]
    resume_compatibility: Mapping[str, Any]
    decoded_ranks: Mapping[int, DecodedRankTrainingState]

    @property
    def current_rank(self) -> int:
        return next(iter(self.decoded_ranks))

    @property
    def decoded_rank(self) -> DecodedRankTrainingState:
        return self.decoded_ranks[self.current_rank]

    def bytes_for(self, *, rank: int, role: str) -> bytes:
        if rank != self.current_rank:
            _fail(
                "admission retains bytes only for the authenticated current rank",
                code="training_state.missing_component",
                context={"current_rank": self.current_rank, "rank": rank, "role": role},
            )
        for rank_record in self.manifest.ranks:
            if rank_record.rank != rank:
                continue
            for file_record in rank_record.files:
                if file_record.role == role:
                    return self.files[file_record.path]
        _fail(
            "admitted training state has no file for the requested rank and role",
            code="training_state.missing_component",
            context={"rank": rank, "role": role},
        )
        raise AssertionError("unreachable")


class TrainingStatePublicationError(ArtifactContractError):
    """Publication failed with explicit ownership and reload evidence."""

    def __init__(
        self,
        message: str,
        *,
        installed_by_this_call: bool,
        reloaded_exact: bool,
        cause: BaseException,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.installed_by_this_call = installed_by_this_call
        self.reloaded_exact = reloaded_exact
        super().__init__(
            message,
            code="training_state.publication_failed",
            context={
                **dict(context or {}),
                "installed_by_this_call": installed_by_this_call,
                "reloaded_exact": reloaded_exact,
            },
            cause=cause,
        )


class TrainingStateContributionError(ArtifactContractError):
    """A distributed contribution stage failed and remains forensic evidence."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        stage_path: Path,
        session_id: str,
        owns_stage: bool,
        terminal_forensic_only: bool = False,
        cause: BaseException | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.stage_path = stage_path
        self.session_id = session_id
        self.owns_stage = owns_stage
        self.terminal_forensic_only = terminal_forensic_only
        self.retryable = not terminal_forensic_only
        super().__init__(
            message,
            code=code,
            context={
                **dict(context or {}),
                "owns_stage": owns_stage,
                "retryable": self.retryable,
                "session_id": session_id,
                "stage_path": str(stage_path),
                "terminal_forensic_only": terminal_forensic_only,
            },
            cause=cause,
        )


@dataclass
class _CheckpointAnchor:
    path: Path
    descriptor: int
    device: int
    inode: int

    @classmethod
    def open(cls, path: Path | str) -> "_CheckpointAnchor":
        selected = Path(path).expanduser().absolute()
        _assert_no_symlink_components(selected)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(selected, flags)
        except OSError as exc:
            _fail(
                "checkpoint_dir must be an existing real directory",
                code="training_state.invalid_checkpoint",
                context={"path": str(selected), "error_type": type(exc).__name__},
                cause=exc,
            )
        identity = os.fstat(descriptor)
        if not stat.S_ISDIR(identity.st_mode):
            os.close(descriptor)
            _fail(
                "checkpoint_dir must be a directory",
                code="training_state.invalid_checkpoint",
                context={"path": str(selected)},
            )
        anchor = cls(
            path=selected,
            descriptor=descriptor,
            device=identity.st_dev,
            inode=identity.st_ino,
        )
        try:
            anchor.assert_path_owner_unchanged()
        except BaseException:
            anchor.close()
            raise
        return anchor

    def assert_path_owner_unchanged(self) -> None:
        try:
            _assert_no_symlink_components(self.path)
            current = os.stat(self.path, follow_symlinks=False)
        except BaseException as exc:
            if isinstance(exc, ArtifactContractError):
                cause = exc
            else:
                cause = exc
            _fail(
                "selected checkpoint path owner changed after it was anchored",
                code="training_state.path_owner_drift",
                context={"path": str(self.path)},
                cause=cause,
            )
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != self.device
            or current.st_ino != self.inode
        ):
            _fail(
                "selected checkpoint path owner changed after it was anchored",
                code="training_state.path_owner_drift",
                context={
                    "anchored_device": self.device,
                    "anchored_inode": self.inode,
                    "current_device": current.st_dev,
                    "current_inode": current.st_ino,
                    "path": str(self.path),
                },
            )

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def capture_runtime_state_expectations(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
) -> RuntimeStateExpectations:
    """Capture the exact structural state expected from a current runtime."""

    trainable = _trainable_model_envelope(model)
    optimizer_envelope = _optimizer_envelope(model, optimizer)
    scheduler_envelope = (
        _stateful_envelope(TRAINING_STATE_SCHEDULER_SCHEMA, scheduler)
        if scheduler is not None
        else None
    )
    scaler_envelope = (
        _stateful_envelope(TRAINING_STATE_SCALER_SCHEMA, scaler)
        if scaler is not None
        else None
    )
    structure = _runtime_structure(
        trainable=trainable,
        optimizer=optimizer_envelope,
        scheduler=scheduler_envelope,
        scaler=scaler_envelope,
    )
    return RuntimeStateExpectations(
        structure=MappingProxyType(structure),
        signature=_sha256(_canonical_json_bytes(structure)),
    )


def serialize_rank_training_state(
    *,
    rank: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    python_rng_state: tuple[Any, ...],
    numpy_rng_state: tuple[Any, ...],
    torch_cpu_rng_state: torch.Tensor,
    torch_cuda_rng_states: Sequence[torch.Tensor],
    cursor: Mapping[str, Any],
    next_rank_local_micro_step: int,
) -> RankTrainingStatePayload:
    """Serialize actual runtime owners into repository-versioned strict envelopes."""

    trainable = _trainable_model_envelope(model)
    optimizer_envelope = _optimizer_envelope(model, optimizer)
    scheduler_bytes = (
        _torch_envelope_bytes(
            _stateful_envelope(TRAINING_STATE_SCHEDULER_SCHEMA, scheduler)
        )
        if scheduler is not None
        else None
    )
    scaler_bytes = (
        _torch_envelope_bytes(_stateful_envelope(TRAINING_STATE_SCALER_SCHEMA, scaler))
        if scaler is not None
        else None
    )
    payload = RankTrainingStatePayload(
        rank=rank,
        trainable_model=_torch_envelope_bytes(trainable),
        optimizer=_torch_envelope_bytes(optimizer_envelope),
        scheduler=scheduler_bytes,
        scaler=scaler_bytes,
        rng={
            "python": _python_rng_bytes(python_rng_state),
            "numpy": _numpy_rng_bytes(numpy_rng_state),
            "torch_cpu": _torch_rng_bytes(
                TRAINING_STATE_TORCH_CPU_RNG_SCHEMA, (torch_cpu_rng_state,)
            ),
            "torch_cuda": _torch_rng_bytes(
                TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA,
                torch_cuda_rng_states,
                cuda_device_topology=tuple(
                    f"cuda:{index}" for index in range(len(torch_cuda_rng_states))
                ),
            ),
        },
        cursor=cursor,
        next_rank_local_micro_step=next_rank_local_micro_step,
    )
    # Decoding here prevents publication of bytes the repository itself cannot admit.
    _decode_rank_payload(payload)
    return payload


def begin_training_state_contributions(
    checkpoint_dir: Path | str,
    plan: TrainingStatePublicationPlan,
) -> TrainingStateContributionSession:
    """Create one fresh shared stage containing no rank payload bytes."""

    _validate_publication_plan(plan)
    anchor = _CheckpointAnchor.open(checkpoint_dir)
    session_id = uuid.uuid4().hex
    stage_name = f".{TRAINING_STATE_DIRECTORY}.{session_id}.tmp"
    stage_created = False
    try:
        if _entry_exists_at(anchor.descriptor, TRAINING_STATE_DIRECTORY):
            _fail(
                "exact training-state target already exists",
                code="training_state.immutable_collision",
                context={"path": str(anchor.path / TRAINING_STATE_DIRECTORY)},
            )
        os.mkdir(stage_name, mode=0o700, dir_fd=anchor.descriptor)
        stage_created = True
        stage_fd = _open_directory_at(anchor.descriptor, stage_name)
        try:
            resolved_config_bytes = _canonical_json_bytes(plan.resolved_config) + b"\n"
            _write_new_file_durable_at(
                stage_fd, TRAINING_STATE_RESOLVED_CONFIG, resolved_config_bytes
            )
            resume_compatibility_bytes = (
                _canonical_json_bytes(plan.resume_compatibility) + b"\n"
            )
            _write_new_file_durable_at(
                stage_fd,
                TRAINING_STATE_RESUME_COMPATIBILITY,
                resume_compatibility_bytes,
            )
            record = _contribution_session_record(
                anchor=anchor,
                plan=plan,
                session_id=session_id,
                stage_name=stage_name,
            )
            _write_new_file_durable_at(
                stage_fd,
                TRAINING_STATE_CONTRIBUTION_PLAN,
                _canonical_json_bytes(record) + b"\n",
            )
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        os.fsync(anchor.descriptor)
        return TrainingStateContributionSession(
            checkpoint_path=anchor.path,
            checkpoint_device=anchor.device,
            checkpoint_inode=anchor.inode,
            stage_name=stage_name,
            plan_digest=plan.digest,
            session_id=session_id,
            world_size=plan.world_size,
        )
    except BaseException:
        if stage_created:
            _remove_tree_at(anchor.descriptor, stage_name)
        raise
    finally:
        anchor.close()


def abort_training_state_contributions(
    checkpoint_dir: Path | str,
    session: TrainingStateContributionSession,
) -> None:
    """Remove only an authenticated pre-manifest contribution stage."""

    anchor = _CheckpointAnchor.open(checkpoint_dir)
    stage_fd = -1
    try:
        _, stage_fd = _authenticate_contribution_session(anchor, session)
        stage_identity = os.fstat(stage_fd)
        if _entry_exists_at(stage_fd, TRAINING_STATE_TERMINAL_FORENSIC) or (
            _entry_exists_at(stage_fd, TRAINING_STATE_MANIFEST)
        ):
            _fail(
                "post-manifest contribution stage is terminal forensic evidence",
                code="training_state.terminal_forensic_only",
                context={"stage_path": str(session.stage_path)},
            )
        anchor.assert_path_owner_unchanged()
        current = os.stat(
            session.stage_name,
            dir_fd=anchor.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_dev != stage_identity.st_dev
            or current.st_ino != stage_identity.st_ino
        ):
            _fail(
                "contribution stage owner changed after authentication",
                code="training_state.contribution_session_mismatch",
                context={"stage_path": str(session.stage_path)},
            )
        os.close(stage_fd)
        stage_fd = -1
        _remove_tree_at(anchor.descriptor, session.stage_name)
        os.fsync(anchor.descriptor)
        anchor.assert_path_owner_unchanged()
        if _entry_exists_at(anchor.descriptor, session.stage_name):
            _fail(
                "authenticated contribution stage remains after abort",
                code="training_state.contribution_abort_failed",
                context={"stage_path": str(session.stage_path)},
            )
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        anchor.close()


def publish_rank_training_state_contribution(
    checkpoint_dir: Path | str,
    session: TrainingStateContributionSession,
    payload: RankTrainingStatePayload,
) -> PublishedRankContribution:
    """Durably install one strictly decoded rank payload into a shared stage."""

    anchor = _CheckpointAnchor.open(checkpoint_dir)
    stage_fd = -1
    temporary_name = f".rank-{payload.rank:05d}.{uuid.uuid4().hex}.tmp"
    installed = False
    try:
        plan, stage_fd = _authenticate_contribution_session(anchor, session)
        if payload.rank < 0 or payload.rank >= plan.world_size:
            _fail(
                "rank contribution is outside the publication world",
                code="training_state.incomplete_rank_set",
                context={"rank": payload.rank, "world_size": plan.world_size},
            )
        rank_record, components = _build_rank_record_and_components(payload, plan)
        target_name = f"rank-{payload.rank:05d}"
        if _entry_exists_at(stage_fd, target_name):
            _fail(
                "rank contribution already exists",
                code="training_state.immutable_collision",
                context={
                    "rank": payload.rank,
                    "path": str(session.stage_path / target_name),
                },
            )
        os.mkdir(temporary_name, mode=0o700, dir_fd=stage_fd)
        temporary_fd = _open_directory_at(stage_fd, temporary_name)
        try:
            for role, content in components.items():
                _write_new_file_durable_at(
                    temporary_fd, _filename_for_role(role), content
                )
            descriptor = {
                "plan_digest": plan.digest,
                "rank": rank_record.to_dict(),
                "schema": TRAINING_STATE_CONTRIBUTION_SCHEMA,
                "schema_version": TRAINING_STATE_CONTRIBUTION_SCHEMA_VERSION,
                "session_id": session.session_id,
            }
            _write_new_file_durable_at(
                temporary_fd,
                TRAINING_STATE_RANK_CONTRIBUTION,
                _canonical_json_bytes(descriptor) + b"\n",
            )
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        try:
            _rename_directory_no_replace_at(stage_fd, temporary_name, target_name)
        except FileExistsError as exc:
            _fail(
                "rank contribution was concurrently installed",
                code="training_state.immutable_collision",
                context={
                    "rank": payload.rank,
                    "path": str(session.stage_path / target_name),
                },
                cause=exc,
            )
        installed = True
        os.fsync(stage_fd)
        return PublishedRankContribution(
            path=session.stage_path / target_name,
            rank=payload.rank,
            runtime_signature=rank_record.runtime_signature,
        )
    finally:
        if stage_fd >= 0:
            if not installed:
                _remove_tree_at(stage_fd, temporary_name)
            os.close(stage_fd)
        anchor.close()


def commit_training_state_contributions(
    checkpoint_dir: Path | str,
    session: TrainingStateContributionSession,
    *,
    on_manifest_written: Callable[[], None] | None = None,
) -> PublishedTrainingState:
    """Validate ranks, then commit once; post-manifest failures are terminal."""

    anchor = _CheckpointAnchor.open(checkpoint_dir)
    stage_fd = -1
    installed = False
    try:
        plan, stage_fd = _authenticate_contribution_session(anchor, session)
        ranks = _load_complete_rank_contributions(stage_fd, session=session, plan=plan)
        runtime_signatures = {rank.runtime_signature for rank in ranks}
        if len(runtime_signatures) != 1:
            _runtime_fail(
                "runtime.signature",
                "all rank contributions must have one exact runtime-state signature",
                context={
                    "ranks": {str(rank.rank): rank.runtime_signature for rank in ranks}
                },
            )
        resolved_config_bytes = _read_stable_regular_file_at(
            stage_fd, TRAINING_STATE_RESOLVED_CONFIG
        )
        resolved_config = _decode_resolved_config(resolved_config_bytes)
        config_record = TrainingStateFile(
            path=TRAINING_STATE_RESOLVED_CONFIG,
            role="resolved_config",
            size=len(resolved_config_bytes),
            sha256=_sha256(resolved_config_bytes),
        )
        if (
            resolved_config != plan.resolved_config
            or config_record.sha256 != plan.identities["resolved_config"]
        ):
            _fail(
                "staged resolved configuration differs from the contribution plan",
                code="training_state.incompatible",
                context={"field": "resolved_config"},
            )
        resume_compatibility_bytes = _read_stable_regular_file_at(
            stage_fd, TRAINING_STATE_RESUME_COMPATIBILITY
        )
        resume_compatibility = _decode_resume_compatibility(resume_compatibility_bytes)
        compatibility_record = TrainingStateFile(
            path=TRAINING_STATE_RESUME_COMPATIBILITY,
            role="resume_compatibility",
            size=len(resume_compatibility_bytes),
            sha256=_sha256(resume_compatibility_bytes),
        )
        if (
            resume_compatibility != plan.resume_compatibility
            or compatibility_record.sha256 != plan.identities["resume_compatibility"]
        ):
            _fail(
                "staged resume compatibility differs from the contribution plan",
                code="training_state.incompatible",
                context={"field": "resume_compatibility"},
            )
        manifest = _manifest_from_plan(
            plan,
            ranks=ranks,
            resolved_config=config_record,
            resume_compatibility=compatibility_record,
        )
        _write_new_file_durable_at(
            stage_fd,
            TRAINING_STATE_MANIFEST,
            _canonical_json_bytes(manifest.to_dict()) + b"\n",
        )
        terminal_record = {
            "reason": "post_manifest_stage_is_not_retryable",
            "session_id": session.session_id,
        }
        _write_new_file_durable_at(
            stage_fd,
            TRAINING_STATE_TERMINAL_FORENSIC,
            _canonical_json_bytes(terminal_record) + b"\n",
        )
        os.fsync(stage_fd)
        if on_manifest_written is not None:
            on_manifest_written()
        for rank in ranks:
            rank_name = f"rank-{rank.rank:05d}"
            rank_fd = _open_directory_at(stage_fd, rank_name)
            try:
                os.unlink(TRAINING_STATE_RANK_CONTRIBUTION, dir_fd=rank_fd)
                os.fsync(rank_fd)
            finally:
                os.close(rank_fd)
        os.unlink(TRAINING_STATE_CONTRIBUTION_PLAN, dir_fd=stage_fd)
        os.unlink(TRAINING_STATE_TERMINAL_FORENSIC, dir_fd=stage_fd)
        os.fsync(stage_fd)
        anchor.assert_path_owner_unchanged()
        try:
            _rename_directory_no_replace_at(
                anchor.descriptor, session.stage_name, TRAINING_STATE_DIRECTORY
            )
        except FileExistsError as exc:
            _fail(
                "exact training-state target was concurrently published",
                code="training_state.immutable_collision",
                context={"path": str(anchor.path / TRAINING_STATE_DIRECTORY)},
                cause=exc,
            )
        installed = True
        os.close(stage_fd)
        stage_fd = -1
        os.fsync(anchor.descriptor)
        reloaded = load_training_state_manifest(anchor.path)
        if reloaded.to_dict() != manifest.to_dict():
            _fail(
                "committed contribution manifest did not reload exactly",
                code="training_state.publication_reload_mismatch",
            )
        return PublishedTrainingState(
            path=anchor.path / TRAINING_STATE_DIRECTORY,
            manifest=reloaded,
        )
    except BaseException as exc:
        if isinstance(exc, TrainingStateContributionError):
            raise
        code = (
            exc.code
            if isinstance(exc, ArtifactContractError)
            else ("training_state.contribution_commit_failed")
        )
        terminal_forensic_only = (
            not installed
            and session.stage_path.is_dir()
            and (session.stage_path / TRAINING_STATE_MANIFEST).is_file()
        )
        raise TrainingStateContributionError(
            "training-state contribution commit failed",
            code=code,
            stage_path=session.stage_path,
            session_id=session.session_id,
            owns_stage=(not installed and session.stage_path.is_dir()),
            terminal_forensic_only=terminal_forensic_only,
            cause=exc,
            context={"installed": installed, "error_type": type(exc).__name__},
        ) from exc
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        anchor.close()


def build_training_state_manifest(
    publication: TrainingStatePublication,
) -> tuple[TrainingStateManifest, Mapping[str, bytes]]:
    """Validate caller inputs and construct deterministic manifest/file bytes."""

    _validate_publication(publication)
    assert publication.resolved_config is not None
    resolved_config_bytes = _canonical_json_bytes(publication.resolved_config) + b"\n"
    resolved_config_digest = _sha256(resolved_config_bytes)
    if publication.identities["resolved_config"] != resolved_config_digest:
        _fail(
            "resolved configuration content does not match its identity",
            code="training_state.incompatible",
            context={
                "mismatches": [
                    {
                        "checkpoint": resolved_config_digest,
                        "current": publication.identities["resolved_config"],
                        "field": "identities.resolved_config",
                    }
                ]
            },
        )
    assert publication.resume_compatibility is not None
    resume_compatibility_bytes = (
        _canonical_json_bytes(publication.resume_compatibility) + b"\n"
    )
    resume_compatibility_digest = _sha256(resume_compatibility_bytes)
    if publication.identities["resume_compatibility"] != resume_compatibility_digest:
        _fail(
            "resume compatibility content does not match its identity",
            code="training_state.incompatible",
            context={"field": "identities.resume_compatibility"},
        )
    resolved_config_record = TrainingStateFile(
        path=TRAINING_STATE_RESOLVED_CONFIG,
        role="resolved_config",
        size=len(resolved_config_bytes),
        sha256=resolved_config_digest,
    )
    resume_compatibility_record = TrainingStateFile(
        path=TRAINING_STATE_RESUME_COMPATIBILITY,
        role="resume_compatibility",
        size=len(resume_compatibility_bytes),
        sha256=resume_compatibility_digest,
    )
    file_bytes: dict[str, bytes] = {
        TRAINING_STATE_RESOLVED_CONFIG: resolved_config_bytes,
        TRAINING_STATE_RESUME_COMPATIBILITY: resume_compatibility_bytes,
    }
    ranks: list[TrainingStateRank] = []
    for payload in sorted(publication.rank_payloads, key=lambda item: item.rank):
        decoded = _decode_rank_payload(payload, world_size=publication.world_size)
        prefix = f"rank-{payload.rank:05d}"
        components: dict[str, bytes] = {
            "optimizer": bytes(payload.optimizer),
            "trainable_model": bytes(payload.trainable_model),
        }
        if publication.scheduler_applicable:
            assert payload.scheduler is not None
            components["scheduler"] = bytes(payload.scheduler)
        if publication.scaler_applicable:
            assert payload.scaler is not None
            components["scaler"] = bytes(payload.scaler)
        for kind in REQUIRED_RNG_KINDS:
            components[f"rng:{kind}"] = bytes(payload.rng[kind])
        cursor_bytes = _cursor_envelope_bytes(payload)
        components["cursor"] = cursor_bytes

        records: list[TrainingStateFile] = []
        for role, content in components.items():
            name = _filename_for_role(role)
            relative_path = f"{prefix}/{name}"
            file_bytes[relative_path] = content
            records.append(
                TrainingStateFile(
                    path=relative_path,
                    role=role,
                    size=len(content),
                    sha256=_sha256(content),
                )
            )
        ranks.append(
            TrainingStateRank(
                rank=payload.rank,
                rng_kinds=REQUIRED_RNG_KINDS,
                next_rank_local_micro_step=payload.next_rank_local_micro_step,
                runtime_signature=decoded.signature,
                files=tuple(sorted(records, key=lambda item: item.path)),
            )
        )
    runtime_signatures = {rank.runtime_signature for rank in ranks}
    if len(runtime_signatures) != 1:
        _runtime_fail(
            "runtime.signature",
            "all rank payloads must have one exact runtime-state signature",
            context={
                "ranks": {str(rank.rank): rank.runtime_signature for rank in ranks}
            },
        )

    manifest = TrainingStateManifest(
        schema=TRAINING_STATE_SCHEMA,
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        artifact_type=TRAINING_STATE_ARTIFACT_TYPE,
        commit_status=TRAINING_STATE_COMMIT_STATUS,
        parent_run_id=publication.parent_run_id,
        parent_segment_id=publication.parent_segment_id,
        checkpoint_step=publication.checkpoint_step,
        continuation_index=publication.continuation_index,
        world_size=publication.world_size,
        save_boundary=TRAINING_STATE_SAVE_BOUNDARY,
        identities=MappingProxyType(dict(sorted(publication.identities.items()))),
        optimizer_applicable=True,
        scheduler_applicable=publication.scheduler_applicable,
        scaler_applicable=publication.scaler_applicable,
        resolved_config=resolved_config_record,
        resume_compatibility=resume_compatibility_record,
        ranks=tuple(ranks),
        aggregate_digest="0" * _SHA256_LENGTH,
    )
    manifest = replace(
        manifest,
        aggregate_digest=manifest.computed_aggregate_digest(),
    )
    # Round-trip through the strict loader before any filesystem mutation.
    manifest = TrainingStateManifest.from_dict(manifest.to_dict())
    return manifest, MappingProxyType(file_bytes)


def publish_training_state(
    checkpoint_dir: Path | str,
    publication: TrainingStatePublication,
    *,
    before_install: Callable[[], None] | None = None,
    on_installed: Callable[[], None] | None = None,
) -> PublishedTrainingState:
    """Atomically install a complete ``training_state/`` child without replacement."""

    anchor = _CheckpointAnchor.open(checkpoint_dir)
    target = anchor.path / TRAINING_STATE_DIRECTORY
    stage_name = f".{TRAINING_STATE_DIRECTORY}.{uuid.uuid4().hex}.tmp"
    installed = False
    manifest: TrainingStateManifest | None = None
    try:
        if _entry_exists_at(anchor.descriptor, TRAINING_STATE_DIRECTORY):
            _fail(
                "exact training-state target already exists",
                code="training_state.immutable_collision",
                context={"path": str(target)},
            )
        manifest, payloads = build_training_state_manifest(publication)
        os.mkdir(stage_name, mode=0o700, dir_fd=anchor.descriptor)
        stage_fd = _open_directory_at(anchor.descriptor, stage_name)
        rank_directories = sorted(
            {
                PurePosixPath(relative).parts[0]
                for relative in payloads
                if len(PurePosixPath(relative).parts) > 1
            }
        )
        try:
            for rank_directory in rank_directories:
                os.mkdir(rank_directory, mode=0o700, dir_fd=stage_fd)
            for relative_path, content in payloads.items():
                _write_new_file_durable_at(stage_fd, relative_path, content)
            _write_new_file_durable_at(
                stage_fd,
                TRAINING_STATE_MANIFEST,
                _canonical_json_bytes(manifest.to_dict()) + b"\n",
            )
            for rank_directory in rank_directories:
                rank_fd = _open_directory_at(stage_fd, rank_directory)
                try:
                    os.fsync(rank_fd)
                finally:
                    os.close(rank_fd)
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        if before_install is not None:
            before_install()
        anchor.assert_path_owner_unchanged()
        try:
            _rename_directory_no_replace_at(
                anchor.descriptor,
                stage_name,
                TRAINING_STATE_DIRECTORY,
            )
        except FileExistsError as exc:
            _fail(
                "exact training-state target was concurrently published",
                code="training_state.immutable_collision",
                context={"path": str(target)},
                cause=exc,
            )
        installed = True
        if on_installed is not None:
            on_installed()
        os.fsync(anchor.descriptor)
        admitted = _admit_training_state_at(
            anchor,
            _expectations_from_manifest(manifest),
            current_rank=0,
            expected_manifest=manifest,
        )
        if admitted.manifest.aggregate_digest != manifest.aggregate_digest:
            _fail(
                "published training state did not reload exactly",
                code="training_state.publication_reload_mismatch",
            )
        return PublishedTrainingState(path=target, manifest=manifest)
    except BaseException as exc:
        if isinstance(exc, ArtifactContractError) and not installed:
            raise
        reloaded_exact = (
            installed
            and manifest is not None
            and _reload_matches_manifest_at(anchor, manifest)
        )
        raise TrainingStatePublicationError(
            "exact training-state publication failed",
            installed_by_this_call=installed,
            reloaded_exact=reloaded_exact,
            cause=exc,
            context={"path": str(target), "error_type": type(exc).__name__},
        ) from exc
    finally:
        if not installed:
            _remove_tree_at(anchor.descriptor, stage_name)
        anchor.close()


def admit_training_state(
    checkpoint_dir: Path | str,
    expectations: TrainingStateExpectations,
    *,
    current_rank: int,
    on_admitted: Callable[[AdmittedTrainingState], _T] | None = None,
) -> AdmittedTrainingState | _T:
    """Authenticate all ranks while retaining only ``current_rank`` state."""

    if on_admitted is not None and expectations.runtime_state is None:
        _fail(
            "a mutation callback requires current runtime-state expectations",
            code="training_state.incomplete",
            context={"field": "expected.runtime_state"},
        )
    if on_admitted is not None and expectations.resolved_config is None:
        _fail(
            "a mutation callback requires the current resolved configuration",
            code="training_state.incomplete",
            context={"field": "expected.resolved_config"},
        )
    if on_admitted is not None and expectations.resume_compatibility is None:
        _fail(
            "a mutation callback requires the current resume compatibility projection",
            code="training_state.incomplete",
            context={"field": "expected.resume_compatibility"},
        )
    anchor = _CheckpointAnchor.open(checkpoint_dir)
    try:
        admitted = _admit_training_state_at(
            anchor, expectations, current_rank=current_rank
        )
        if on_admitted is None:
            return admitted
        return on_admitted(admitted)
    finally:
        anchor.close()


def restore_decoded_rank_training_state(
    decoded: DecodedRankTrainingState,
    *,
    current_rank: int,
    current_world_size: int,
    current_cuda_device_topology: Sequence[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    set_python_rng_state: Callable[[tuple[Any, ...]], None] = random.setstate,
    set_numpy_rng_state: Callable[[tuple[Any, ...]], None] = np.random.set_state,
    set_torch_cpu_rng_state: Callable[[torch.Tensor], None] = torch.set_rng_state,
    get_torch_cuda_rng_states: Callable[[], Sequence[torch.Tensor]] | None = None,
    set_torch_cuda_rng_states: Callable[[Sequence[torch.Tensor]], None] | None = None,
) -> RestoredRankTrainingState:
    """Restore one decoded rank atomically at the repository-owned state boundary."""

    if not isinstance(decoded, DecodedRankTrainingState):
        _fail(
            "decoded state has an unsupported type",
            code="training_state.schema",
            context={"field": "decoded"},
        )
    _require_nonnegative_int(current_rank, field="current_rank")
    _require_positive_int(current_world_size, field="current_world_size")
    current_topology = _require_cuda_device_topology(
        current_cuda_device_topology, field="current_cuda_device_topology"
    )
    checkpoint_topology = _require_cuda_device_topology(
        decoded.cuda_device_topology, field="decoded.cuda_device_topology"
    )
    if (
        decoded.cuda_device_count != len(checkpoint_topology)
        or len(decoded.torch_cuda_rng_states) != decoded.cuda_device_count
    ):
        _fail(
            "decoded CUDA RNG inventory is inconsistent with its topology",
            code="training_state.corrupt_component",
            context={"field": "decoded.cuda_device_count"},
        )
    identity_mismatches: list[dict[str, Any]] = []
    for field, checkpoint_value, current_value in (
        ("rank", decoded.rank, current_rank),
        ("world_size", decoded.world_size, current_world_size),
        ("cuda_device_topology", list(checkpoint_topology), list(current_topology)),
    ):
        if checkpoint_value != current_value:
            identity_mismatches.append(
                {
                    "checkpoint": checkpoint_value,
                    "current": current_value,
                    "field": field,
                }
            )
    if identity_mismatches:
        _fail(
            "current rank topology cannot restore the decoded state",
            code="training_state.incompatible",
            context={"mismatches": identity_mismatches},
        )
    observed_signature = _sha256(_canonical_json_bytes(decoded.structure))
    if observed_signature != decoded.signature:
        _fail(
            "decoded runtime-state signature is invalid",
            code="training_state.corrupt_component",
            context={"field": "decoded.signature"},
        )
    current = capture_runtime_state_expectations(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    mismatches = _runtime_state_mismatches(
        decoded.structure, current.structure, rank=decoded.rank
    )
    if mismatches:
        _fail(
            "current runtime cannot restore the decoded state",
            code="training_state.incompatible",
            context={"mismatches": mismatches},
        )
    next_micro_step = _require_nonnegative_int(
        decoded.cursor["next_rank_local_micro_step"],
        field="decoded.cursor.next_rank_local_micro_step",
    )
    model_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer_state = _optimizer_load_state_dict(
        decoded.optimizer, model=model, optimizer=optimizer
    )
    cuda_rng_getter = get_torch_cuda_rng_states or _get_current_cuda_rng_states
    cuda_rng_setter = set_torch_cuda_rng_states or _set_current_cuda_rng_states
    model_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model_parameters.items()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = (
        copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    )
    scaler_before = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_cpu_before = torch.get_rng_state()
    torch_cuda_before = _clone_rank_owned_cuda_rng_states(
        cuda_rng_getter(),
        expected_count=decoded.cuda_device_count,
        field="runtime.rng.torch_cuda.rollback",
    )
    try:
        with torch.no_grad():
            for name, value in decoded.trainable_model.items():
                model_parameters[name].copy_(
                    value.to(
                        device=model_parameters[name].device,
                        dtype=model_parameters[name].dtype,
                    )
                )
        optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            assert decoded.scheduler is not None
            scheduler.load_state_dict(
                _clone_state_value(
                    decoded.scheduler["state"], field="restore.scheduler.state"
                )
            )
        if scaler is not None:
            assert decoded.scaler is not None
            scaler.load_state_dict(
                _clone_state_value(
                    decoded.scaler["state"], field="restore.scaler.state"
                )
            )
        set_python_rng_state(decoded.python_rng_state)
        set_numpy_rng_state(decoded.numpy_rng_state)
        set_torch_cpu_rng_state(decoded.torch_cpu_rng_state.clone())
        cuda_rng_setter(tuple(state.clone() for state in decoded.torch_cuda_rng_states))
    except BaseException as exc:
        rollback_errors: list[str] = []

        def rollback(owner: str, action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as rollback_exc:
                rollback_errors.append(f"{owner}:{type(rollback_exc).__name__}")

        def restore_model() -> None:
            with torch.no_grad():
                for name, value in model_before.items():
                    model_parameters[name].copy_(
                        value.to(
                            device=model_parameters[name].device,
                            dtype=model_parameters[name].dtype,
                        )
                    )

        rollback("model", restore_model)
        rollback("optimizer", lambda: optimizer.load_state_dict(optimizer_before))
        if scheduler is not None:
            assert scheduler_before is not None
            rollback("scheduler", lambda: scheduler.load_state_dict(scheduler_before))
        if scaler is not None:
            assert scaler_before is not None
            rollback("scaler", lambda: scaler.load_state_dict(scaler_before))
        rollback("python_rng", lambda: set_python_rng_state(python_before))
        rollback("numpy_rng", lambda: set_numpy_rng_state(numpy_before))
        rollback("torch_cpu_rng", lambda: set_torch_cpu_rng_state(torch_cpu_before))
        rollback(
            "torch_cuda_rng",
            lambda: cuda_rng_setter(torch_cuda_before),
        )
        raise ArtifactContractError(
            "exact training-state restore failed",
            code="training_state.restore_failed",
            context={
                "original_error_code": getattr(exc, "code", None),
                "original_error_type": type(exc).__name__,
                "rollback_complete": not rollback_errors,
                "rollback_errors": rollback_errors,
            },
            cause=exc,
        ) from exc
    return RestoredRankTrainingState(
        cursor=decoded.cursor,
        next_rank_local_micro_step=next_micro_step,
        runtime_signature=decoded.signature,
    )


def _get_current_cuda_rng_states() -> tuple[torch.Tensor, ...]:
    device = torch.cuda.current_device()
    return (torch.cuda.get_rng_state(device).detach().cpu().clone(),)


def _set_current_cuda_rng_states(states: Sequence[torch.Tensor]) -> None:
    selected = _clone_rank_owned_cuda_rng_states(
        states,
        expected_count=1,
        field="runtime.rng.torch_cuda.current_device_restore",
    )
    torch.cuda.set_rng_state(selected[0], torch.cuda.current_device())


def _clone_rank_owned_cuda_rng_states(
    states: Sequence[torch.Tensor],
    *,
    expected_count: int,
    field: str,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes, bytearray)):
        _runtime_fail(field, "rank-owned CUDA RNG states must be a sequence")
    if len(states) != expected_count:
        _runtime_fail(
            field,
            "rank-owned CUDA RNG state count differs from its local topology",
            context={"expected_count": expected_count, "observed_count": len(states)},
        )
    cloned: list[torch.Tensor] = []
    for index, state in enumerate(states):
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
        ):
            _runtime_fail(
                f"{field}.{index}",
                "rank-owned CUDA RNG state must be a nonempty uint8 vector",
            )
        cloned.append(state.detach().cpu().clone())
    return tuple(cloned)


def load_training_state_manifest(checkpoint_dir: Path | str) -> TrainingStateManifest:
    """Strictly load the committed manifest without admitting component files."""

    anchor = _CheckpointAnchor.open(checkpoint_dir)
    try:
        state_fd = _open_training_state_at(anchor)
        try:
            manifest = _load_training_state_manifest_at(state_fd)
            anchor.assert_path_owner_unchanged()
            return manifest
        finally:
            os.close(state_fd)
    finally:
        anchor.close()


def _admit_training_state_at(
    anchor: _CheckpointAnchor,
    expectations: TrainingStateExpectations,
    *,
    current_rank: int,
    expected_manifest: TrainingStateManifest | None = None,
) -> AdmittedTrainingState:
    _validate_expectations(expectations)
    _require_nonnegative_int(current_rank, field="current_rank")
    if current_rank >= expectations.world_size:
        _fail(
            "current rank is outside the expected world",
            code="training_state.incompatible",
            context={
                "mismatches": [
                    {
                        "checkpoint": f"0..{expectations.world_size - 1}",
                        "current": current_rank,
                        "field": "rank",
                    }
                ]
            },
        )
    state_fd = _open_training_state_at(anchor)
    try:
        manifest = _load_training_state_manifest_at(state_fd)
        mismatches = _compatibility_mismatches(manifest, expectations)
        if (
            expected_manifest is not None
            and manifest.to_dict() != expected_manifest.to_dict()
        ):
            _fail(
                "training-state manifest differs from the publication input",
                code="training_state.publication_reload_mismatch",
            )
        expected_paths = {
            file.path for rank in manifest.ranks for file in rank.files
        } | {
            TRAINING_STATE_MANIFEST,
            manifest.resolved_config.path,
            manifest.resume_compatibility.path,
        }
        expected_directories = {f"rank-{rank.rank:05d}" for rank in manifest.ranks}
        observed_paths, observed_directories = _tree_inventory_at(state_fd)
        if (
            observed_paths != expected_paths
            or observed_directories != expected_directories
        ):
            _fail(
                "training-state inventory is incomplete or contains undeclared entries",
                code="training_state.incomplete",
                context={
                    "missing_directories": sorted(
                        expected_directories - observed_directories
                    ),
                    "missing_files": sorted(expected_paths - observed_paths),
                    "unexpected_directories": sorted(
                        observed_directories - expected_directories
                    ),
                    "unexpected_files": sorted(observed_paths - expected_paths),
                },
            )
        snapshots: dict[str, bytes] = {}
        resolved_config_bytes = _read_stable_regular_file_at(
            state_fd, manifest.resolved_config.path
        )
        _validate_file_snapshot(manifest.resolved_config, resolved_config_bytes)
        if manifest.resolved_config.sha256 != manifest.identities["resolved_config"]:
            _fail(
                "resolved configuration file is not cross-bound to its identity",
                code="training_state.corrupt_manifest",
                context={
                    "file_sha256": manifest.resolved_config.sha256,
                    "identity": manifest.identities["resolved_config"],
                },
            )
        resolved_config = _decode_resolved_config(resolved_config_bytes)
        snapshots[manifest.resolved_config.path] = resolved_config_bytes
        resume_compatibility_bytes = _read_stable_regular_file_at(
            state_fd, manifest.resume_compatibility.path
        )
        _validate_file_snapshot(
            manifest.resume_compatibility, resume_compatibility_bytes
        )
        resume_compatibility = _decode_resume_compatibility(resume_compatibility_bytes)
        _validate_resume_compatibility_binding(
            resolved_config=resolved_config,
            resume_compatibility=resume_compatibility,
            identities=manifest.identities,
            field="manifest",
        )
        snapshots[manifest.resume_compatibility.path] = resume_compatibility_bytes
        decoded_ranks: dict[int, DecodedRankTrainingState] = {}
        for rank in manifest.ranks:
            rank_snapshots: dict[str, bytes] = {}
            for file in rank.files:
                content = _read_stable_regular_file_at(state_fd, file.path)
                _validate_file_snapshot(file, content)
                rank_snapshots[file.path] = content
            if rank.rank != current_rank:
                continue
            snapshots.update(rank_snapshots)
            decoded = _decode_rank_snapshots(
                rank, rank_snapshots, world_size=manifest.world_size
            )
            decoded_ranks[rank.rank] = decoded
            if decoded.signature != rank.runtime_signature:
                _fail(
                    "rank runtime-state structure disagrees with its manifest signature",
                    code="training_state.corrupt_component",
                    context={
                        "rank": rank.rank,
                        "manifest": rank.runtime_signature,
                        "observed": decoded.signature,
                    },
                )
            if expectations.runtime_state is not None:
                mismatches.extend(
                    _runtime_state_mismatches(
                        decoded.structure,
                        expectations.runtime_state.structure,
                        rank=rank.rank,
                    )
                )
        if expectations.resume_compatibility is not None:
            mismatches.extend(
                _json_field_mismatches(
                    resume_compatibility,
                    expectations.resume_compatibility,
                    field="resume_compatibility",
                )
            )
        if mismatches:
            _fail(
                "training-state compatibility admission failed",
                code="training_state.incompatible",
                context={"mismatches": mismatches},
            )
        anchor.assert_path_owner_unchanged()
        return AdmittedTrainingState(
            path=anchor.path / TRAINING_STATE_DIRECTORY,
            manifest=manifest,
            files=MappingProxyType(snapshots),
            resolved_config=resolved_config,
            resume_compatibility=resume_compatibility,
            decoded_ranks=MappingProxyType(decoded_ranks),
        )
    finally:
        os.close(state_fd)


def _open_training_state_at(anchor: _CheckpointAnchor) -> int:
    try:
        return _open_directory_at(anchor.descriptor, TRAINING_STATE_DIRECTORY)
    except FileNotFoundError as exc:
        _fail(
            "checkpoint has no exact training state",
            code="training_state.model_only",
            context={"path": str(anchor.path)},
            cause=exc,
        )
    except OSError as exc:
        _fail(
            "training_state must be a real directory",
            code="training_state.unsafe_path",
            context={"path": str(anchor.path / TRAINING_STATE_DIRECTORY)},
            cause=exc,
        )
    raise AssertionError("unreachable")


def _load_training_state_manifest_at(state_fd: int) -> TrainingStateManifest:
    try:
        encoded = _read_stable_regular_file_at(state_fd, TRAINING_STATE_MANIFEST)
    except FileNotFoundError as exc:
        _fail(
            "training-state directory has no committed manifest",
            code="training_state.uncommitted",
            cause=exc,
        )
    try:
        raw = _strict_json_loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            "training-state manifest is not strict JSON",
            code="training_state.corrupt_manifest",
            context={"error_type": type(exc).__name__},
            cause=exc,
        )
    return TrainingStateManifest.from_dict(_require_mapping(raw, field="manifest"))


def _validate_publication(publication: TrainingStatePublication) -> None:
    _require_nonempty_string(publication.parent_run_id, field="parent_run_id")
    _require_nonempty_string(publication.parent_segment_id, field="parent_segment_id")
    _require_positive_int(publication.checkpoint_step, field="checkpoint_step")
    _require_nonnegative_int(publication.continuation_index, field="continuation_index")
    _require_positive_int(publication.world_size, field="world_size")
    _validate_identities(publication.identities)
    if not isinstance(publication.resolved_config, Mapping):
        _fail(
            "resolved_config must be a string-keyed mapping",
            code="training_state.incomplete",
        )
    _normalize_json_value(publication.resolved_config, field="resolved_config")
    _validate_resume_compatibility_binding(
        resolved_config=publication.resolved_config,
        resume_compatibility=publication.resume_compatibility,
        identities=publication.identities,
        field="publication",
    )
    _require_bool(publication.scheduler_applicable, field="scheduler_applicable")
    _require_bool(publication.scaler_applicable, field="scaler_applicable")
    accumulation = _require_nonnegative_int(
        publication.accumulation_microstep, field="accumulation_microstep"
    )
    if accumulation != 0:
        _fail(
            "mid-accumulation exact training-state publication is unsupported",
            code="training_state.unsupported_mid_accumulation",
            context={"accumulation_microstep": accumulation},
        )
    if not isinstance(publication.rank_payloads, Sequence) or isinstance(
        publication.rank_payloads, (str, bytes, bytearray)
    ):
        _fail("rank_payloads must be a sequence", code="training_state.schema")
    for payload in publication.rank_payloads:
        if not isinstance(payload, RankTrainingStatePayload):
            _fail(
                "rank_payloads must contain RankTrainingStatePayload values",
                code="training_state.schema",
                context={"value_type": type(payload).__name__},
            )
    observed_ranks = [payload.rank for payload in publication.rank_payloads]
    if sorted(observed_ranks) != list(range(publication.world_size)):
        _fail(
            "publication requires exactly one payload from every rank",
            code="training_state.incomplete_rank_set",
            context={
                "expected_ranks": list(range(publication.world_size)),
                "observed_ranks": observed_ranks,
            },
        )
    for payload in publication.rank_payloads:
        _validate_rank_payload(payload, publication=publication)


def _validate_publication_plan(plan: TrainingStatePublicationPlan) -> None:
    if not isinstance(plan, TrainingStatePublicationPlan):
        _fail(
            "publication plan has an unsupported type",
            code="training_state.schema",
        )
    _require_nonempty_string(plan.parent_run_id, field="publication_plan.parent_run_id")
    _require_nonempty_string(
        plan.parent_segment_id, field="publication_plan.parent_segment_id"
    )
    _require_positive_int(
        plan.checkpoint_step, field="publication_plan.checkpoint_step"
    )
    _require_nonnegative_int(
        plan.continuation_index, field="publication_plan.continuation_index"
    )
    _require_positive_int(plan.world_size, field="publication_plan.world_size")
    identities = _validate_identities(plan.identities)
    _require_bool(
        plan.scheduler_applicable, field="publication_plan.scheduler_applicable"
    )
    _require_bool(plan.scaler_applicable, field="publication_plan.scaler_applicable")
    accumulation = _require_nonnegative_int(
        plan.accumulation_microstep,
        field="publication_plan.accumulation_microstep",
    )
    if accumulation != 0:
        _fail(
            "mid-accumulation exact training-state publication is unsupported",
            code="training_state.unsupported_mid_accumulation",
            context={"accumulation_microstep": accumulation},
        )
    resolved_config = _require_mapping(
        _normalize_json_value(
            plan.resolved_config, field="publication_plan.resolved_config"
        ),
        field="publication_plan.resolved_config",
    )
    if not resolved_config:
        _fail(
            "resolved configuration must be nonempty",
            code="training_state.incomplete",
        )
    config_digest = _sha256(_canonical_json_bytes(resolved_config) + b"\n")
    if identities["resolved_config"] != config_digest:
        _fail(
            "resolved configuration content does not match its identity",
            code="training_state.incompatible",
            context={"field": "identities.resolved_config"},
        )
    _validate_resume_compatibility_binding(
        resolved_config=resolved_config,
        resume_compatibility=plan.resume_compatibility,
        identities=identities,
        field="publication_plan",
    )


def _validate_resume_compatibility_binding(
    *,
    resolved_config: Mapping[str, Any],
    resume_compatibility: Mapping[str, Any] | None,
    identities: Mapping[str, str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(resume_compatibility, Mapping):
        _fail(
            "resume compatibility projection must be present",
            code="training_state.incomplete",
            context={"field": f"{field}.resume_compatibility"},
        )
    normalized = _require_mapping(
        _normalize_json_value(
            resume_compatibility, field=f"{field}.resume_compatibility"
        ),
        field=f"{field}.resume_compatibility",
    )
    expected = build_resume_compatibility_projection(resolved_config)
    mismatches = _json_field_mismatches(
        expected,
        normalized,
        field="resume_compatibility",
    )
    if mismatches:
        _fail(
            "resume compatibility is not the exact semantic projection",
            code="training_state.incompatible",
            context={"mismatches": mismatches},
        )
    digest = _sha256(_canonical_json_bytes(normalized) + b"\n")
    if identities["resume_compatibility"] != digest:
        _fail(
            "resume compatibility content does not match its identity",
            code="training_state.incompatible",
            context={
                "mismatches": [
                    {
                        "checkpoint": digest,
                        "current": identities["resume_compatibility"],
                        "field": "identities.resume_compatibility",
                    }
                ]
            },
        )
    return MappingProxyType(dict(normalized))


def _contribution_session_record(
    *,
    anchor: _CheckpointAnchor,
    plan: TrainingStatePublicationPlan,
    session_id: str,
    stage_name: str,
) -> dict[str, Any]:
    return {
        "checkpoint_device": anchor.device,
        "checkpoint_inode": anchor.inode,
        "checkpoint_path": str(anchor.path),
        "plan": plan.to_dict(),
        "plan_digest": plan.digest,
        "schema": TRAINING_STATE_CONTRIBUTION_SCHEMA,
        "schema_version": TRAINING_STATE_CONTRIBUTION_SCHEMA_VERSION,
        "session_id": session_id,
        "stage_name": stage_name,
    }


def _authenticate_contribution_session(
    anchor: _CheckpointAnchor, session: TrainingStateContributionSession
) -> tuple[TrainingStatePublicationPlan, int]:
    if not isinstance(session, TrainingStateContributionSession):
        _fail(
            "contribution session has an unsupported type",
            code="training_state.contribution_session_mismatch",
        )
    expected_stage_name = f".{TRAINING_STATE_DIRECTORY}.{session.session_id}.tmp"
    if (
        session.checkpoint_path != anchor.path
        or session.checkpoint_device != anchor.device
        or session.checkpoint_inode != anchor.inode
        or session.stage_name != expected_stage_name
        or session.plan_digest
        != _require_sha256(
            session.plan_digest, field="contribution_session.plan_digest"
        )
    ):
        _fail(
            "contribution session is not bound to the selected checkpoint owner",
            code="training_state.contribution_session_mismatch",
            context={
                "checkpoint_path": str(anchor.path),
                "session_checkpoint_path": str(session.checkpoint_path),
            },
        )
    try:
        stage_fd = _open_directory_at(anchor.descriptor, session.stage_name)
    except OSError as exc:
        _fail(
            "contribution stage is absent or unsafe",
            code="training_state.contribution_session_mismatch",
            context={"stage_path": str(session.stage_path)},
            cause=exc,
        )
    try:
        if _entry_exists_at(stage_fd, TRAINING_STATE_TERMINAL_FORENSIC) or (
            _entry_exists_at(stage_fd, TRAINING_STATE_MANIFEST)
        ):
            _fail(
                "post-manifest contribution stage is terminal forensic evidence",
                code="training_state.terminal_forensic_only",
                context={"stage_path": str(session.stage_path)},
            )
        encoded = _read_stable_regular_file_at(
            stage_fd, TRAINING_STATE_CONTRIBUTION_PLAN
        )
        raw = _strict_json_loads(encoded)
        record = _require_mapping(raw, field="contribution_session_record")
        _require_exact_fields(
            record,
            frozenset(
                {
                    "checkpoint_device",
                    "checkpoint_inode",
                    "checkpoint_path",
                    "plan",
                    "plan_digest",
                    "schema",
                    "schema_version",
                    "session_id",
                    "stage_name",
                }
            ),
            field="contribution_session_record",
        )
        if _canonical_json_bytes(record) + b"\n" != encoded:
            _fail(
                "contribution session record is not canonical",
                code="training_state.contribution_session_mismatch",
            )
        plan = TrainingStatePublicationPlan.from_dict(
            _require_mapping(record["plan"], field="contribution_session_record.plan")
        )
        expected = _contribution_session_record(
            anchor=anchor,
            plan=plan,
            session_id=session.session_id,
            stage_name=session.stage_name,
        )
        if (
            record != expected
            or plan.digest != session.plan_digest
            or (plan.world_size != session.world_size)
        ):
            _fail(
                "contribution session record differs from its descriptor",
                code="training_state.contribution_session_mismatch",
            )
        return plan, stage_fd
    except BaseException:
        os.close(stage_fd)
        raise


def _build_rank_record_and_components(
    payload: RankTrainingStatePayload, plan: TrainingStatePublicationPlan
) -> tuple[TrainingStateRank, Mapping[str, bytes]]:
    publication = TrainingStatePublication(
        parent_run_id=plan.parent_run_id,
        parent_segment_id=plan.parent_segment_id,
        checkpoint_step=plan.checkpoint_step,
        continuation_index=plan.continuation_index,
        world_size=plan.world_size,
        identities=plan.identities,
        scheduler_applicable=plan.scheduler_applicable,
        scaler_applicable=plan.scaler_applicable,
        rank_payloads=(payload,),
        accumulation_microstep=plan.accumulation_microstep,
        resolved_config=plan.resolved_config,
        resume_compatibility=plan.resume_compatibility,
    )
    _validate_rank_payload(payload, publication=publication)
    decoded = _decode_rank_payload(payload, world_size=plan.world_size)
    components: dict[str, bytes] = {
        "optimizer": bytes(payload.optimizer),
        "trainable_model": bytes(payload.trainable_model),
        "cursor": _cursor_envelope_bytes(payload),
    }
    if plan.scheduler_applicable:
        assert payload.scheduler is not None
        components["scheduler"] = bytes(payload.scheduler)
    if plan.scaler_applicable:
        assert payload.scaler is not None
        components["scaler"] = bytes(payload.scaler)
    for kind in REQUIRED_RNG_KINDS:
        components[f"rng:{kind}"] = bytes(payload.rng[kind])
    prefix = f"rank-{payload.rank:05d}"
    files = tuple(
        sorted(
            (
                TrainingStateFile(
                    path=f"{prefix}/{_filename_for_role(role)}",
                    role=role,
                    size=len(content),
                    sha256=_sha256(content),
                )
                for role, content in components.items()
            ),
            key=lambda item: item.path,
        )
    )
    return (
        TrainingStateRank(
            rank=payload.rank,
            rng_kinds=REQUIRED_RNG_KINDS,
            next_rank_local_micro_step=payload.next_rank_local_micro_step,
            runtime_signature=decoded.signature,
            files=files,
        ),
        MappingProxyType(components),
    )


def _load_complete_rank_contributions(
    stage_fd: int,
    *,
    session: TrainingStateContributionSession,
    plan: TrainingStatePublicationPlan,
) -> tuple[TrainingStateRank, ...]:
    expected_directories = {f"rank-{rank:05d}" for rank in range(plan.world_size)}
    expected_files = {
        TRAINING_STATE_CONTRIBUTION_PLAN,
        TRAINING_STATE_RESOLVED_CONFIG,
        TRAINING_STATE_RESUME_COMPATIBILITY,
    }
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    for name in os.listdir(stage_fd):
        descriptor = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
        if stat.S_ISDIR(descriptor.st_mode) and not stat.S_ISLNK(descriptor.st_mode):
            observed_directories.add(name)
        elif stat.S_ISREG(descriptor.st_mode):
            observed_files.add(name)
        else:
            _fail(
                "contribution stage contains an unsafe entry",
                code="training_state.unsafe_path",
                context={"path": name},
            )
    if observed_directories != expected_directories or observed_files != expected_files:
        _fail(
            "contribution stage does not contain the exact complete rank set",
            code="training_state.incomplete_rank_set",
            context={
                "missing_ranks": sorted(expected_directories - observed_directories),
                "unexpected_directories": sorted(
                    observed_directories - expected_directories
                ),
                "unexpected_files": sorted(observed_files - expected_files),
            },
        )
    ranks: list[TrainingStateRank] = []
    descriptor_fields = frozenset(
        {"plan_digest", "rank", "schema", "schema_version", "session_id"}
    )
    for expected_rank in range(plan.world_size):
        rank_name = f"rank-{expected_rank:05d}"
        rank_fd = _open_directory_at(stage_fd, rank_name)
        try:
            encoded = _read_stable_regular_file_at(
                rank_fd, TRAINING_STATE_RANK_CONTRIBUTION
            )
            raw = _strict_json_loads(encoded)
            descriptor = _require_mapping(raw, field="rank_contribution")
            _require_exact_fields(
                descriptor, descriptor_fields, field="rank_contribution"
            )
            if _canonical_json_bytes(descriptor) + b"\n" != encoded:
                _fail(
                    "rank contribution descriptor is not canonical",
                    code="training_state.corrupt_component",
                    context={"rank": expected_rank},
                )
            if (
                descriptor["schema"] != TRAINING_STATE_CONTRIBUTION_SCHEMA
                or descriptor["schema_version"]
                != TRAINING_STATE_CONTRIBUTION_SCHEMA_VERSION
                or descriptor["session_id"] != session.session_id
                or descriptor["plan_digest"] != plan.digest
            ):
                _fail(
                    "rank contribution is not bound to this session and plan",
                    code="training_state.contribution_session_mismatch",
                    context={"rank": expected_rank},
                )
            rank = TrainingStateRank.from_dict(
                _require_mapping(descriptor["rank"], field="rank_contribution.rank")
            )
            if rank.rank != expected_rank:
                _fail(
                    "rank contribution descriptor has the wrong rank",
                    code="training_state.incomplete_rank_set",
                    context={"expected": expected_rank, "observed": rank.rank},
                )
            expected_component_files = {
                _filename_for_role(file.role) for file in rank.files
            }
            observed_rank_files, observed_rank_directories = _tree_inventory_at(rank_fd)
            if observed_rank_directories or observed_rank_files != (
                expected_component_files | {TRAINING_STATE_RANK_CONTRIBUTION}
            ):
                _fail(
                    "rank contribution inventory is incomplete or unexpected",
                    code="training_state.incomplete",
                    context={"rank": expected_rank},
                )
            components: dict[str, bytes] = {}
            for file in rank.files:
                content = _read_stable_regular_file_at(
                    rank_fd, _filename_for_role(file.role)
                )
                _validate_file_snapshot(file, content)
                components[file.role] = content
            decoded = _decode_runtime_components(
                rank, components, world_size=plan.world_size
            )
            if decoded.signature != rank.runtime_signature:
                _fail(
                    "rank contribution runtime signature is invalid",
                    code="training_state.corrupt_component",
                    context={"rank": expected_rank},
                )
            ranks.append(rank)
        finally:
            os.close(rank_fd)
    return tuple(ranks)


def _manifest_from_plan(
    plan: TrainingStatePublicationPlan,
    *,
    ranks: Sequence[TrainingStateRank],
    resolved_config: TrainingStateFile,
    resume_compatibility: TrainingStateFile,
) -> TrainingStateManifest:
    manifest = TrainingStateManifest(
        schema=TRAINING_STATE_SCHEMA,
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        artifact_type=TRAINING_STATE_ARTIFACT_TYPE,
        commit_status=TRAINING_STATE_COMMIT_STATUS,
        parent_run_id=plan.parent_run_id,
        parent_segment_id=plan.parent_segment_id,
        checkpoint_step=plan.checkpoint_step,
        continuation_index=plan.continuation_index,
        world_size=plan.world_size,
        save_boundary=TRAINING_STATE_SAVE_BOUNDARY,
        identities=MappingProxyType(dict(sorted(plan.identities.items()))),
        optimizer_applicable=True,
        scheduler_applicable=plan.scheduler_applicable,
        scaler_applicable=plan.scaler_applicable,
        resolved_config=resolved_config,
        resume_compatibility=resume_compatibility,
        ranks=tuple(ranks),
        aggregate_digest="0" * _SHA256_LENGTH,
    )
    manifest = replace(manifest, aggregate_digest=manifest.computed_aggregate_digest())
    return TrainingStateManifest.from_dict(manifest.to_dict())


def _validate_rank_payload(
    payload: RankTrainingStatePayload, *, publication: TrainingStatePublication
) -> None:
    _require_nonnegative_int(payload.rank, field="rank_payload.rank")
    _require_bytes(payload.trainable_model, field="rank_payload.trainable_model")
    _require_bytes(payload.optimizer, field="rank_payload.optimizer")
    _validate_optional_component(
        payload.scheduler,
        applicable=publication.scheduler_applicable,
        field="rank_payload.scheduler",
    )
    _validate_optional_component(
        payload.scaler,
        applicable=publication.scaler_applicable,
        field="rank_payload.scaler",
    )
    if not isinstance(payload.rng, Mapping) or set(payload.rng) != set(
        REQUIRED_RNG_KINDS
    ):
        _fail(
            "rank RNG state must contain every required RNG kind",
            code="training_state.incomplete",
            context={
                "rank": payload.rank,
                "expected_rng_kinds": list(REQUIRED_RNG_KINDS),
                "observed_rng_kinds": sorted(str(key) for key in payload.rng),
            },
        )
    for kind in REQUIRED_RNG_KINDS:
        _require_bytes(payload.rng[kind], field=f"rank_payload.rng.{kind}")
    if not isinstance(payload.cursor, Mapping):
        _fail("rank cursor must be a mapping", code="training_state.schema")
    observed_cursor_fields = set(payload.cursor)
    expected_cursor_fields = {"data", "pack"}
    if observed_cursor_fields != expected_cursor_fields:
        _fail(
            "rank cursor must contain exactly data and pack owner states",
            code="training_state.schema",
            context={
                "missing": sorted(expected_cursor_fields - observed_cursor_fields),
                "unexpected": sorted(observed_cursor_fields - expected_cursor_fields),
                "rank": payload.rank,
            },
        )
    for owner in ("data", "pack"):
        state = payload.cursor[owner]
        if not isinstance(state, Mapping) or not state:
            _fail(
                "cursor owner state must be a nonempty mapping",
                code="training_state.schema",
                context={"owner": owner, "rank": payload.rank},
            )
        _normalize_json_value(dict(state), field=f"rank_payload.cursor.{owner}")
    _require_nonnegative_int(
        payload.next_rank_local_micro_step,
        field="rank_payload.next_rank_local_micro_step",
    )


def _validate_expectations(expectations: TrainingStateExpectations) -> None:
    _require_positive_int(
        expectations.checkpoint_step, field="expected.checkpoint_step"
    )
    _require_positive_int(expectations.world_size, field="expected.world_size")
    _validate_identities(expectations.identities)
    if expectations.resolved_config is not None:
        normalized_config = _normalize_json_value(
            expectations.resolved_config, field="expected.resolved_config"
        )
        config_digest = _sha256(_canonical_json_bytes(normalized_config) + b"\n")
        if expectations.identities["resolved_config"] != config_digest:
            _fail(
                "current resolved configuration does not match its identity",
                code="training_state.incompatible",
                context={
                    "mismatches": [
                        {
                            "checkpoint": config_digest,
                            "current": expectations.identities["resolved_config"],
                            "field": "identities.resolved_config",
                        }
                    ]
                },
            )
        _validate_resume_compatibility_binding(
            resolved_config=normalized_config,
            resume_compatibility=expectations.resume_compatibility,
            identities=expectations.identities,
            field="expected",
        )
    elif expectations.resume_compatibility is not None:
        _fail(
            "resume compatibility requires the full current resolved configuration",
            code="training_state.incomplete",
            context={"field": "expected.resolved_config"},
        )
    if expectations.runtime_state is not None:
        if not isinstance(expectations.runtime_state, RuntimeStateExpectations):
            _fail(
                "runtime_state must be RuntimeStateExpectations",
                code="training_state.schema",
                context={"field": "expected.runtime_state"},
            )
        normalized_structure = _normalize_json_value(
            expectations.runtime_state.structure,
            field="expected.runtime_state.structure",
        )
        observed_signature = _sha256(_canonical_json_bytes(normalized_structure))
        expected_signature = _require_sha256(
            expectations.runtime_state.signature,
            field="expected.runtime_state.signature",
        )
        if observed_signature != expected_signature:
            _fail(
                "runtime-state expectation signature is invalid",
                code="training_state.schema",
                context={"field": "expected.runtime_state.signature"},
            )
    _require_bool(
        expectations.scheduler_applicable, field="expected.scheduler_applicable"
    )
    _require_bool(expectations.scaler_applicable, field="expected.scaler_applicable")
    rng_kinds = _require_string_sequence(
        expectations.rng_kinds, field="expected.rng_kinds"
    )
    if rng_kinds != REQUIRED_RNG_KINDS:
        _fail(
            "current runtime RNG kinds do not match the exact-resume contract",
            code="training_state.incompatible",
            context={
                "expected": list(REQUIRED_RNG_KINDS),
                "observed": list(rng_kinds),
            },
        )


def _compatibility_mismatches(
    manifest: TrainingStateManifest, expectations: TrainingStateExpectations
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    checks = {
        "checkpoint_step": (manifest.checkpoint_step, expectations.checkpoint_step),
        "world_size": (manifest.world_size, expectations.world_size),
        "applicability.scheduler": (
            manifest.scheduler_applicable,
            expectations.scheduler_applicable,
        ),
        "applicability.scaler": (
            manifest.scaler_applicable,
            expectations.scaler_applicable,
        ),
    }
    for field, (checkpoint_value, current_value) in checks.items():
        if checkpoint_value != current_value:
            mismatches.append(
                {
                    "checkpoint": checkpoint_value,
                    "current": current_value,
                    "field": field,
                }
            )
    for name in REQUIRED_IDENTITY_KINDS:
        if name == "resolved_config":
            continue
        checkpoint_value = manifest.identities[name]
        current_value = expectations.identities[name]
        if checkpoint_value != current_value:
            mismatches.append(
                {
                    "checkpoint": checkpoint_value,
                    "current": current_value,
                    "field": f"identities.{name}",
                }
            )
    for rank in manifest.ranks:
        if rank.rng_kinds != tuple(expectations.rng_kinds):
            mismatches.append(
                {
                    "checkpoint": list(rank.rng_kinds),
                    "current": list(expectations.rng_kinds),
                    "field": f"ranks.{rank.rank}.rng_kinds",
                }
            )
    return mismatches


def _validate_manifest_roles(manifest: TrainingStateManifest) -> None:
    if (
        manifest.resolved_config.path != TRAINING_STATE_RESOLVED_CONFIG
        or manifest.resolved_config.role != "resolved_config"
        or manifest.resolved_config.sha256 != manifest.identities["resolved_config"]
    ):
        _fail(
            "resolved configuration manifest record is not canonical or cross-bound",
            code="training_state.corrupt_manifest",
            context={"record": manifest.resolved_config.to_dict()},
        )
    if (
        manifest.resume_compatibility.path != TRAINING_STATE_RESUME_COMPATIBILITY
        or manifest.resume_compatibility.role != "resume_compatibility"
        or manifest.resume_compatibility.sha256
        != manifest.identities["resume_compatibility"]
    ):
        _fail(
            "resume compatibility manifest record is not canonical or cross-bound",
            code="training_state.corrupt_manifest",
            context={"record": manifest.resume_compatibility.to_dict()},
        )
    expected_roles = {"cursor", "optimizer", "trainable_model"}
    if manifest.scheduler_applicable:
        expected_roles.add("scheduler")
    if manifest.scaler_applicable:
        expected_roles.add("scaler")
    expected_roles.update(f"rng:{kind}" for kind in REQUIRED_RNG_KINDS)
    all_paths: set[str] = set()
    for rank in manifest.ranks:
        if rank.rng_kinds != REQUIRED_RNG_KINDS:
            _fail(
                "rank RNG kinds do not match the supported exact-resume contract",
                code="training_state.incomplete",
                context={"rank": rank.rank, "rng_kinds": list(rank.rng_kinds)},
            )
        roles = {file.role for file in rank.files}
        if roles != expected_roles:
            _fail(
                "rank training-state component set is incomplete",
                code="training_state.incomplete",
                context={
                    "rank": rank.rank,
                    "missing": sorted(expected_roles - roles),
                    "unexpected": sorted(roles - expected_roles),
                },
            )
        prefix = f"rank-{rank.rank:05d}/"
        for file in rank.files:
            if not file.path.startswith(prefix) or file.path != (
                prefix + _filename_for_role(file.role)
            ):
                _fail(
                    "rank training-state file path is noncanonical",
                    code="training_state.unsafe_path",
                    context={"path": file.path, "rank": rank.rank},
                )
            if file.path in all_paths:
                _fail(
                    "training-state manifest repeats a file path",
                    code="training_state.schema",
                    context={"path": file.path},
                )
            all_paths.add(file.path)


def _cursor_envelope_bytes(payload: RankTrainingStatePayload) -> bytes:
    next_micro_step = payload.next_rank_local_micro_step
    value = {
        "data": {
            "next_rank_local_micro_step": next_micro_step,
            "owner": "data",
            "schema": TRAINING_STATE_DATA_CURSOR_SCHEMA,
            "schema_version": TRAINING_STATE_CURSOR_SCHEMA_VERSION,
            "state": dict(payload.cursor["data"]),
        },
        "next_rank_local_micro_step": next_micro_step,
        "pack": {
            "next_rank_local_micro_step": next_micro_step,
            "owner": "pack",
            "schema": TRAINING_STATE_PACK_CURSOR_SCHEMA,
            "schema_version": TRAINING_STATE_CURSOR_SCHEMA_VERSION,
            "state": dict(payload.cursor["pack"]),
        },
        "schema": TRAINING_STATE_CURSOR_SCHEMA,
        "schema_version": TRAINING_STATE_CURSOR_SCHEMA_VERSION,
    }
    return _canonical_json_bytes(value) + b"\n"


def _validate_cursor_snapshot(
    rank: TrainingStateRank, encoded: bytes
) -> Mapping[str, Any]:
    try:
        raw = _strict_json_loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            "rank cursor is not strict JSON",
            code="training_state.corrupt_component",
            context={"rank": rank.rank},
            cause=exc,
        )
    normalized = _normalize_json_value(raw, field="cursor")
    value = _require_mapping(normalized, field="cursor")
    _require_exact_fields(value, _CURSOR_FIELDS, field="cursor")
    if value["schema"] != TRAINING_STATE_CURSOR_SCHEMA or value["schema_version"] != (
        TRAINING_STATE_CURSOR_SCHEMA_VERSION
    ):
        _fail(
            "rank cursor schema is unsupported",
            code="training_state.unsupported_schema",
            context={
                "rank": rank.rank,
                "schema": value["schema"],
                "schema_version": value["schema_version"],
            },
        )
    observed = _require_nonnegative_int(
        value["next_rank_local_micro_step"],
        field="cursor.next_rank_local_micro_step",
    )
    if observed != rank.next_rank_local_micro_step:
        _fail(
            "rank cursor disagrees with its manifest descriptor",
            code="training_state.corrupt_component",
            context={
                "rank": rank.rank,
                "manifest": rank.next_rank_local_micro_step,
                "cursor": observed,
            },
        )
    for owner, expected_schema in (
        ("data", TRAINING_STATE_DATA_CURSOR_SCHEMA),
        ("pack", TRAINING_STATE_PACK_CURSOR_SCHEMA),
    ):
        owner_value = _require_mapping(value[owner], field=f"cursor.{owner}")
        _require_exact_fields(
            owner_value, _CURSOR_OWNER_FIELDS, field=f"cursor.{owner}"
        )
        if (
            owner_value["owner"] != owner
            or owner_value["schema"] != expected_schema
            or owner_value["schema_version"] != TRAINING_STATE_CURSOR_SCHEMA_VERSION
        ):
            _fail(
                "cursor owner identity or schema is invalid",
                code="training_state.schema",
                context={"owner": owner, "rank": rank.rank},
            )
        owner_next = _require_nonnegative_int(
            owner_value["next_rank_local_micro_step"],
            field=f"cursor.{owner}.next_rank_local_micro_step",
        )
        if owner_next != observed:
            _fail(
                "cursor owner next micro-step disagrees with the rank cursor",
                code="training_state.corrupt_component",
                context={
                    "owner": owner,
                    "owner_next_rank_local_micro_step": owner_next,
                    "rank": rank.rank,
                    "rank_next_rank_local_micro_step": observed,
                },
            )
        state = owner_value["state"]
        if not isinstance(state, Mapping) or not state:
            _fail(
                "cursor owner state must be a nonempty mapping",
                code="training_state.schema",
                context={"owner": owner, "rank": rank.rank},
            )
    return MappingProxyType(dict(value))


def _trainable_model_envelope(model: torch.nn.Module) -> dict[str, Any]:
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    }
    if not parameters:
        _fail(
            "exact training state requires at least one trainable parameter",
            code="training_state.runtime_state",
            context={"field": "runtime.trainable_model.parameters"},
        )
    return {
        "parameters": parameters,
        "schema": TRAINING_STATE_TRAINABLE_MODEL_SCHEMA,
        "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
    }


def _optimizer_envelope(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    groups: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    observed_names: list[str] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters:
            _runtime_fail(
                f"runtime.optimizer.param_groups.{group_index}.params",
                "optimizer parameter group must be nonempty",
            )
        names: list[str] = []
        for parameter in parameters:
            name = names_by_id.get(id(parameter))
            if name is None:
                _runtime_fail(
                    f"runtime.optimizer.param_groups.{group_index}.params",
                    "optimizer owns an unknown or frozen parameter",
                )
            names.append(name)
            observed_names.append(name)
            if parameter not in optimizer.state:
                continue
            slots = optimizer.state[parameter]
            if not isinstance(slots, Mapping):
                _runtime_fail(
                    f"runtime.optimizer.state.{name}",
                    "optimizer state slots must be a mapping",
                )
            if not slots:
                state[name] = {}
                continue
            state[name] = {
                str(slot_name): _clone_state_value(
                    slot_value,
                    field=f"runtime.optimizer.state.{name}.{slot_name}",
                )
                for slot_name, slot_value in sorted(
                    slots.items(), key=lambda item: str(item[0])
                )
            }
        options = {
            str(key): _clone_state_value(
                value, field=f"runtime.optimizer.param_groups.{group_index}.{key}"
            )
            for key, value in sorted(group.items(), key=lambda item: str(item[0]))
            if key != "params"
        }
        groups.append({"options": options, "params": names})
    expected_names = sorted(names_by_id.values())
    if sorted(observed_names) != expected_names or len(observed_names) != len(
        set(observed_names)
    ):
        _runtime_fail(
            "runtime.optimizer.param_groups",
            "optimizer parameter groups do not cover the trainable surface exactly once",
            context={
                "expected": expected_names,
                "observed": observed_names,
            },
        )
    return {
        "class": _qualified_class_name(optimizer),
        "param_groups": groups,
        "schema": TRAINING_STATE_OPTIMIZER_SCHEMA,
        "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
        "state": dict(sorted(state.items())),
    }


def _stateful_envelope(schema: str, owner: Any) -> dict[str, Any]:
    if owner is None or not callable(getattr(owner, "state_dict", None)):
        _runtime_fail(f"runtime.{schema}.owner", "state owner must expose state_dict()")
    state = owner.state_dict()
    if not isinstance(state, Mapping):
        _runtime_fail(f"runtime.{schema}.state", "state_dict() must return a mapping")
    return {
        "class": _qualified_class_name(owner),
        "schema": schema,
        "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
        "state": _clone_state_value(state, field=f"runtime.{schema}.state"),
    }


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _clone_state_value(value: Any, *, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().clone()
        if tensor.layout != torch.strided:
            _runtime_fail(field, "only strided tensors are supported")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            _runtime_fail(field, "runtime state tensor must be finite")
        return tensor
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _runtime_fail(field, "runtime state scalar must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(
            _clone_state_value(item, field=f"{field}.{index}")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _clone_state_value(item, field=f"{field}.{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _clone_state_value(item, field=f"{field}.{key}")
            for key, item in sorted(value.items())
        }
    _runtime_fail(
        field,
        "runtime state contains an unsupported value",
        context={"value_type": type(value).__name__},
    )
    raise AssertionError("unreachable")


def _torch_envelope_bytes(value: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    return buffer.getvalue()


def _load_torch_envelope(
    encoded: bytes,
    *,
    schema: str,
    field: str,
    schema_version: int = TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
) -> Mapping[str, Any]:
    try:
        value = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except BaseException as exc:
        _fail(
            f"{field} cannot be restricted-decoded",
            code="training_state.corrupt_component",
            context={"field": field, "error_type": type(exc).__name__},
            cause=exc,
        )
    envelope = _require_mapping(value, field=field)
    if (
        envelope.get("schema") != schema
        or envelope.get("schema_version") != schema_version
    ):
        _fail(
            f"{field} schema is unsupported",
            code="training_state.unsupported_schema",
            context={
                "field": field,
                "schema": envelope.get("schema"),
                "schema_version": envelope.get("schema_version"),
            },
        )
    return envelope


def _decode_trainable_model(encoded: bytes) -> Mapping[str, Any]:
    envelope = _load_torch_envelope(
        encoded,
        schema=TRAINING_STATE_TRAINABLE_MODEL_SCHEMA,
        field="runtime.trainable_model",
    )
    _require_exact_fields(
        envelope,
        frozenset({"parameters", "schema", "schema_version"}),
        field="runtime.trainable_model",
    )
    parameters = _require_mapping(
        envelope["parameters"], field="runtime.trainable_model.parameters"
    )
    if not parameters or tuple(parameters) != tuple(sorted(parameters)):
        _runtime_fail(
            "runtime.trainable_model.parameters",
            "trainable parameter mapping must be nonempty and sorted",
        )
    decoded: dict[str, torch.Tensor] = {}
    for name, value in parameters.items():
        _require_nonempty_string(name, field="runtime.trainable_model.parameters[]")
        if not isinstance(value, torch.Tensor):
            _runtime_fail(
                f"runtime.trainable_model.parameters.{name}",
                "trainable parameter state must be a tensor",
            )
        decoded[name] = _clone_state_value(
            value, field=f"runtime.trainable_model.parameters.{name}"
        )
    return {
        "parameters": MappingProxyType(decoded),
        "schema": envelope["schema"],
        "schema_version": envelope["schema_version"],
    }


def _decode_optimizer(
    encoded: bytes, *, trainable: Mapping[str, Any]
) -> Mapping[str, Any]:
    envelope = _load_torch_envelope(
        encoded,
        schema=TRAINING_STATE_OPTIMIZER_SCHEMA,
        field="runtime.optimizer",
    )
    _require_exact_fields(
        envelope,
        frozenset({"class", "param_groups", "schema", "schema_version", "state"}),
        field="runtime.optimizer",
    )
    optimizer_class = _require_nonempty_string(
        envelope["class"], field="runtime.optimizer.class"
    )
    raw_groups = envelope["param_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        _runtime_fail(
            "runtime.optimizer.param_groups", "optimizer groups must be nonempty"
        )
    observed_names: list[str] = []
    groups: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(raw_groups):
        group = _require_mapping(
            raw_group, field=f"runtime.optimizer.param_groups.{group_index}"
        )
        _require_exact_fields(
            group,
            frozenset({"options", "params"}),
            field=f"runtime.optimizer.param_groups.{group_index}",
        )
        names = _require_string_sequence_in_order(
            group["params"],
            field=f"runtime.optimizer.param_groups.{group_index}.params",
        )
        observed_names.extend(names)
        options = _require_mapping(
            group["options"],
            field=f"runtime.optimizer.param_groups.{group_index}.options",
        )
        normalized_options = _clone_state_value(
            options, field=f"runtime.optimizer.param_groups.{group_index}.options"
        )
        groups.append({"options": normalized_options, "params": list(names)})
    parameters = trainable["parameters"]
    if sorted(observed_names) != sorted(parameters) or len(observed_names) != len(
        set(observed_names)
    ):
        _runtime_fail(
            "runtime.optimizer.param_groups",
            "optimizer groups do not match the decoded trainable surface",
            context={"expected": sorted(parameters), "observed": observed_names},
        )
    raw_state = _require_mapping(envelope["state"], field="runtime.optimizer.state")
    if tuple(raw_state) != tuple(sorted(raw_state)) or not set(raw_state).issubset(
        parameters
    ):
        _runtime_fail(
            "runtime.optimizer.state",
            "optimizer state may contain only sorted trainable-parameter slots",
            context={"expected": sorted(parameters), "observed": sorted(raw_state)},
        )
    state: dict[str, dict[str, Any]] = {}
    for name, raw_slot_value in raw_state.items():
        parameter = parameters[name]
        raw_slots = _require_mapping(
            raw_slot_value, field=f"runtime.optimizer.state.{name}"
        )
        if tuple(raw_slots) != tuple(sorted(raw_slots)):
            _runtime_fail(
                f"runtime.optimizer.state.{name}",
                "optimizer slots must be sorted",
            )
        slots: dict[str, Any] = {}
        for slot_name, slot_value in raw_slots.items():
            field = f"runtime.optimizer.state.{name}.{slot_name}"
            normalized = _clone_state_value(slot_value, field=field)
            if isinstance(normalized, torch.Tensor) and slot_name != "step":
                if tuple(normalized.shape) != tuple(parameter.shape):
                    _runtime_fail(
                        field,
                        "optimizer tensor slot shape differs from its parameter",
                        context={
                            "expected": list(parameter.shape),
                            "observed": list(normalized.shape),
                        },
                    )
                if normalized.dtype != parameter.dtype:
                    _runtime_fail(
                        field,
                        "optimizer tensor slot dtype differs from its parameter",
                        context={
                            "expected": str(parameter.dtype),
                            "observed": str(normalized.dtype),
                        },
                    )
            if (
                isinstance(normalized, torch.Tensor)
                and slot_name == "step"
                and (normalized.numel() != 1)
            ):
                _runtime_fail(field, "optimizer step slot must be scalar")
            slots[slot_name] = normalized
        state[name] = slots
    return {
        "class": optimizer_class,
        "param_groups": groups,
        "schema": envelope["schema"],
        "schema_version": envelope["schema_version"],
        "state": MappingProxyType(state),
    }


def _decode_stateful_component(
    encoded: bytes, *, schema: str, field: str
) -> Mapping[str, Any]:
    envelope = _load_torch_envelope(encoded, schema=schema, field=field)
    _require_exact_fields(
        envelope,
        frozenset({"class", "schema", "schema_version", "state"}),
        field=field,
    )
    owner_class = _require_nonempty_string(envelope["class"], field=f"{field}.class")
    state = _require_mapping(envelope["state"], field=f"{field}.state")
    normalized = _clone_state_value(state, field=f"{field}.state")
    return {
        "class": owner_class,
        "schema": envelope["schema"],
        "schema_version": envelope["schema_version"],
        "state": normalized,
    }


def _python_rng_bytes(state: tuple[Any, ...]) -> bytes:
    if not isinstance(state, tuple) or len(state) != 3:
        _runtime_fail("runtime.rng.python", "Python RNG state has an invalid shape")
    internal = state[1]
    if not isinstance(internal, tuple):
        _runtime_fail(
            "runtime.rng.python.internal", "Python RNG internal state must be a tuple"
        )
    value = {
        "schema": TRAINING_STATE_PYTHON_RNG_SCHEMA,
        "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
        "state": {
            "gauss": state[2],
            "internal": list(internal),
            "version": state[0],
        },
    }
    encoded = _canonical_json_bytes(value) + b"\n"
    _decode_python_rng(encoded)
    return encoded


def _decode_python_rng(encoded: bytes) -> tuple[Any, ...]:
    envelope = _decode_json_envelope(
        encoded, schema=TRAINING_STATE_PYTHON_RNG_SCHEMA, field="runtime.rng.python"
    )
    state = _require_mapping(envelope["state"], field="runtime.rng.python.state")
    _require_exact_fields(
        state,
        frozenset({"gauss", "internal", "version"}),
        field="runtime.rng.python.state",
    )
    version = _require_positive_int(
        state["version"], field="runtime.rng.python.version"
    )
    internal = state["internal"]
    if (
        version != 3
        or not isinstance(internal, list)
        or len(internal) != 625
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in internal
        )
        or internal[-1] < 0
        or internal[-1] > 624
    ):
        _runtime_fail(
            "runtime.rng.python.internal", "Python RNG internal state is invalid"
        )
    gauss = state["gauss"]
    if gauss is not None and (
        isinstance(gauss, bool)
        or not isinstance(gauss, (int, float))
        or not math.isfinite(float(gauss))
    ):
        _runtime_fail("runtime.rng.python.gauss", "Python Gaussian cache is invalid")
    return (version, tuple(internal), gauss)


def _numpy_rng_bytes(state: tuple[Any, ...]) -> bytes:
    if not isinstance(state, tuple) or len(state) != 5:
        _runtime_fail("runtime.rng.numpy", "NumPy RNG state has an invalid shape")
    keys = state[1]
    if not isinstance(keys, np.ndarray):
        _runtime_fail("runtime.rng.numpy.keys", "NumPy RNG keys must be an ndarray")
    value = {
        "schema": TRAINING_STATE_NUMPY_RNG_SCHEMA,
        "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
        "state": {
            "algorithm": state[0],
            "cached_gaussian": state[4],
            "has_gaussian": state[3],
            "keys": [int(item) for item in keys.tolist()],
            "position": state[2],
        },
    }
    encoded = _canonical_json_bytes(value) + b"\n"
    _decode_numpy_rng(encoded)
    return encoded


def _decode_numpy_rng(encoded: bytes) -> tuple[Any, ...]:
    envelope = _decode_json_envelope(
        encoded, schema=TRAINING_STATE_NUMPY_RNG_SCHEMA, field="runtime.rng.numpy"
    )
    state = _require_mapping(envelope["state"], field="runtime.rng.numpy.state")
    _require_exact_fields(
        state,
        frozenset({"algorithm", "cached_gaussian", "has_gaussian", "keys", "position"}),
        field="runtime.rng.numpy.state",
    )
    algorithm = _require_nonempty_string(
        state["algorithm"], field="runtime.rng.numpy.algorithm"
    )
    if algorithm != "MT19937":
        _runtime_fail(
            "runtime.rng.numpy.algorithm", "only MT19937 NumPy state is supported"
        )
    keys = state["keys"]
    if (
        not isinstance(keys, list)
        or len(keys) != 624
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > (2**32 - 1)
            for value in keys
        )
    ):
        _runtime_fail("runtime.rng.numpy.keys", "NumPy RNG keys are invalid")
    position = _require_nonnegative_int(
        state["position"], field="runtime.rng.numpy.position"
    )
    if position > len(keys):
        _runtime_fail("runtime.rng.numpy.position", "NumPy RNG position is invalid")
    has_gaussian = state["has_gaussian"]
    if isinstance(has_gaussian, bool) or has_gaussian not in (0, 1):
        _runtime_fail(
            "runtime.rng.numpy.has_gaussian", "NumPy Gaussian flag is invalid"
        )
    cached_gaussian = state["cached_gaussian"]
    if (
        isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        _runtime_fail(
            "runtime.rng.numpy.cached_gaussian", "NumPy Gaussian cache is invalid"
        )
    return (
        algorithm,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gaussian,
        float(cached_gaussian),
    )


def _torch_rng_bytes(
    schema: str,
    states: Sequence[torch.Tensor],
    *,
    cuda_device_topology: Sequence[str] | None = None,
) -> bytes:
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes, bytearray)):
        _runtime_fail(f"runtime.rng.{schema}", "Torch RNG states must be a sequence")
    cloned_states = [
        _clone_state_value(state, field=f"runtime.rng.{schema}.{index}")
        for index, state in enumerate(states)
    ]
    if schema == TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA:
        topology = _require_cuda_device_topology(
            cuda_device_topology, field="runtime.rng.torch_cuda.device_topology"
        )
        if len(topology) != len(cloned_states):
            _runtime_fail(
                "runtime.rng.torch_cuda.device_count",
                "CUDA RNG state count must match the explicit device topology",
            )
        envelope = {
            "device_count": len(topology),
            "device_states": [
                {"device": device, "state": state}
                for device, state in zip(topology, cloned_states, strict=True)
            ],
            "schema": schema,
            "schema_version": TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA_VERSION,
        }
    else:
        if cuda_device_topology is not None:
            _runtime_fail(
                f"runtime.rng.{schema}",
                "CUDA topology is valid only for CUDA RNG state",
            )
        envelope = {
            "schema": schema,
            "schema_version": TRAINING_STATE_RUNTIME_COMPONENT_VERSION,
            "states": cloned_states,
        }
    encoded = _torch_envelope_bytes(envelope)
    if schema == TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA:
        _decode_torch_cuda_rng(encoded)
    else:
        _decode_torch_rng(encoded, schema=schema)
    return encoded


def _decode_torch_rng(encoded: bytes, *, schema: str) -> tuple[torch.Tensor, ...]:
    field = f"runtime.rng.{schema}"
    envelope = _load_torch_envelope(encoded, schema=schema, field=field)
    _require_exact_fields(
        envelope,
        frozenset({"schema", "schema_version", "states"}),
        field=field,
    )
    states = envelope["states"]
    expected_count = 1 if schema == TRAINING_STATE_TORCH_CPU_RNG_SCHEMA else None
    if not isinstance(states, list) or not states:
        _runtime_fail(field, "Torch RNG state list must be nonempty")
    if expected_count is not None and len(states) != expected_count:
        _runtime_fail(field, "Torch CPU RNG must contain exactly one state")
    decoded: list[torch.Tensor] = []
    for index, state in enumerate(states):
        state_field = f"{field}.{index}"
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
        ):
            _runtime_fail(
                state_field, "Torch RNG state must be a nonempty uint8 vector"
            )
        decoded.append(state.detach().cpu().clone())
    return tuple(decoded)


def _decode_torch_cuda_rng(
    encoded: bytes,
) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
    field = "runtime.rng.torch_cuda"
    envelope = _load_torch_envelope(
        encoded,
        schema=TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA,
        field=field,
        schema_version=TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA_VERSION,
    )
    _require_exact_fields(
        envelope,
        frozenset({"device_count", "device_states", "schema", "schema_version"}),
        field=field,
    )
    device_count = _require_positive_int(
        envelope["device_count"], field=f"{field}.device_count"
    )
    rows = envelope["device_states"]
    if not isinstance(rows, list) or len(rows) != device_count:
        _runtime_fail(
            f"{field}.device_states",
            "CUDA device-state inventory must match device_count",
        )
    devices: list[str] = []
    states: list[torch.Tensor] = []
    for index, raw_row in enumerate(rows):
        row_field = f"{field}.device_states.{index}"
        row = _require_mapping(raw_row, field=row_field)
        _require_exact_fields(row, frozenset({"device", "state"}), field=row_field)
        devices.append(
            _require_nonempty_string(row["device"], field=f"{row_field}.device")
        )
        state = row["state"]
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
        ):
            _runtime_fail(
                f"{row_field}.state", "Torch RNG state must be a nonempty uint8 vector"
            )
        states.append(state.detach().cpu().clone())
    topology = _require_cuda_device_topology(devices, field=f"{field}.device_topology")
    return topology, tuple(states)


def _require_cuda_device_topology(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        _runtime_fail(field, "CUDA device topology must be a nonempty sequence")
    topology: list[str] = []
    for index, raw_device in enumerate(value):
        device = _require_nonempty_string(raw_device, field=f"{field}.{index}")
        prefix, separator, suffix = device.partition(":")
        if (
            prefix != "cuda"
            or separator != ":"
            or not suffix.isdigit()
            or str(int(suffix)) != suffix
        ):
            _runtime_fail(
                f"{field}.{index}",
                "CUDA devices must use canonical cuda:<local-index> names",
            )
        topology.append(device)
    if len(topology) != len(set(topology)):
        _runtime_fail(field, "CUDA device topology must not contain duplicates")
    return tuple(topology)


def _decode_json_envelope(
    encoded: bytes, *, schema: str, field: str
) -> Mapping[str, Any]:
    try:
        raw = _strict_json_loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            f"{field} cannot be strict-decoded",
            code="training_state.corrupt_component",
            context={"field": field},
            cause=exc,
        )
    envelope = _require_mapping(_normalize_json_value(raw, field=field), field=field)
    _require_exact_fields(
        envelope,
        frozenset({"schema", "schema_version", "state"}),
        field=field,
    )
    if envelope["schema"] != schema or envelope["schema_version"] != (
        TRAINING_STATE_RUNTIME_COMPONENT_VERSION
    ):
        _fail(
            f"{field} schema is unsupported",
            code="training_state.unsupported_schema",
            context={
                "field": field,
                "schema": envelope["schema"],
                "schema_version": envelope["schema_version"],
            },
        )
    if _canonical_json_bytes(envelope) + b"\n" != encoded:
        _fail(
            f"{field} bytes are not canonical",
            code="training_state.corrupt_component",
            context={"field": field},
        )
    return envelope


def _decode_rank_payload(
    payload: RankTrainingStatePayload, *, world_size: int | None = None
) -> DecodedRankTrainingState:
    components: dict[str, bytes] = {
        "cursor": _cursor_envelope_bytes(payload),
        "optimizer": payload.optimizer,
        "trainable_model": payload.trainable_model,
        **{f"rng:{kind}": payload.rng[kind] for kind in REQUIRED_RNG_KINDS},
    }
    if payload.scheduler is not None:
        components["scheduler"] = payload.scheduler
    if payload.scaler is not None:
        components["scaler"] = payload.scaler
    rank = TrainingStateRank(
        rank=payload.rank,
        rng_kinds=REQUIRED_RNG_KINDS,
        next_rank_local_micro_step=payload.next_rank_local_micro_step,
        runtime_signature="0" * _SHA256_LENGTH,
        files=(),
    )
    return _decode_runtime_components(
        rank,
        components,
        world_size=(payload.rank + 1 if world_size is None else world_size),
    )


def _decode_rank_snapshots(
    rank: TrainingStateRank,
    snapshots: Mapping[str, bytes],
    *,
    world_size: int,
) -> DecodedRankTrainingState:
    components = {file.role: snapshots[file.path] for file in rank.files}
    return _decode_runtime_components(rank, components, world_size=world_size)


def _decode_runtime_components(
    rank: TrainingStateRank,
    components: Mapping[str, bytes],
    *,
    world_size: int,
) -> DecodedRankTrainingState:
    trainable = _decode_trainable_model(components["trainable_model"])
    optimizer = _decode_optimizer(components["optimizer"], trainable=trainable)
    scheduler = (
        _decode_stateful_component(
            components["scheduler"],
            schema=TRAINING_STATE_SCHEDULER_SCHEMA,
            field="runtime.scheduler",
        )
        if "scheduler" in components
        else None
    )
    scaler = (
        _decode_stateful_component(
            components["scaler"],
            schema=TRAINING_STATE_SCALER_SCHEMA,
            field="runtime.scaler",
        )
        if "scaler" in components
        else None
    )
    python_rng = _decode_python_rng(components["rng:python"])
    numpy_rng = _decode_numpy_rng(components["rng:numpy"])
    torch_cpu_states = _decode_torch_rng(
        components["rng:torch_cpu"], schema=TRAINING_STATE_TORCH_CPU_RNG_SCHEMA
    )
    cuda_device_topology, torch_cuda_states = _decode_torch_cuda_rng(
        components["rng:torch_cuda"]
    )
    cursor = _validate_cursor_snapshot(rank, components["cursor"])
    structure = _runtime_structure(
        trainable=trainable,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    signature = _sha256(_canonical_json_bytes(structure))
    return DecodedRankTrainingState(
        rank=rank.rank,
        world_size=world_size,
        trainable_model=trainable["parameters"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        python_rng_state=python_rng,
        numpy_rng_state=numpy_rng,
        torch_cpu_rng_state=torch_cpu_states[0],
        torch_cuda_rng_states=torch_cuda_states,
        cuda_device_topology=cuda_device_topology,
        cuda_device_count=len(cuda_device_topology),
        cursor=cursor,
        structure=MappingProxyType(structure),
        signature=signature,
    )


def _runtime_structure(
    *,
    trainable: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    scheduler: Mapping[str, Any] | None,
    scaler: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parameters = trainable["parameters"]
    parameter_rows = {
        name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        for name, tensor in parameters.items()
    }
    groups = [
        {
            "option_keys": sorted(group["options"]),
            "option_types": {
                key: _state_value_structure(group["options"][key])
                for key in sorted(group["options"])
            },
            "params": list(group["params"]),
        }
        for group in optimizer["param_groups"]
    ]
    slots = {
        name: {
            slot_name: _state_value_structure(slot_value)
            for slot_name, slot_value in sorted(values.items())
        }
        for name, values in sorted(optimizer["state"].items())
        if values
    }
    return {
        "optimizer": {
            "class": optimizer["class"],
            "param_groups": groups,
            "slots": slots,
        },
        "scaler": _stateful_structure(scaler),
        "scheduler": _stateful_structure(scheduler),
        "trainable_model": {"parameters": parameter_rows},
    }


def _optimizer_load_state_dict(
    decoded: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    current = optimizer.state_dict()
    current_groups = current["param_groups"]
    decoded_groups = decoded["param_groups"]
    if len(current_groups) != len(decoded_groups):
        _fail(
            "optimizer group count changed after restore admission",
            code="training_state.incompatible",
            context={"field": "runtime.optimizer.param_groups"},
        )
    state: dict[int, Any] = {}
    groups: list[dict[str, Any]] = []
    for group_index, (runtime_group, current_group, decoded_group) in enumerate(
        zip(optimizer.param_groups, current_groups, decoded_groups, strict=True)
    ):
        runtime_parameters = runtime_group["params"]
        state_ids = current_group["params"]
        names = decoded_group["params"]
        observed_names = [
            names_by_id[id(parameter)] for parameter in runtime_parameters
        ]
        if observed_names != names or len(state_ids) != len(names):
            _fail(
                "optimizer parameter group changed after restore admission",
                code="training_state.incompatible",
                context={
                    "field": f"runtime.optimizer.param_groups.{group_index}.params",
                    "checkpoint": names,
                    "current": observed_names,
                },
            )
        options = _clone_state_value(
            decoded_group["options"],
            field=f"restore.optimizer.param_groups.{group_index}.options",
        )
        options["params"] = list(state_ids)
        groups.append(options)
        for name, state_id in zip(names, state_ids, strict=True):
            if name in decoded["state"]:
                state[state_id] = _clone_state_value(
                    decoded["state"][name],
                    field=f"restore.optimizer.state.{name}",
                )
    return {"param_groups": groups, "state": state}


def _stateful_structure(value: Mapping[str, Any] | None) -> Any:
    if value is None:
        return None
    state = value["state"]
    return {
        "class": value["class"],
        "state": {
            key: _state_value_structure(item) for key, item in sorted(state.items())
        },
    }


def _state_value_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "kind": "tensor", "shape": list(value.shape)}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool"}
    if isinstance(value, int):
        return {"kind": "int"}
    if isinstance(value, float):
        return {"kind": "float"}
    if isinstance(value, str):
        return {"kind": "str"}
    if isinstance(value, (list, tuple)):
        return {
            "items": [_state_value_structure(item) for item in value],
            "kind": "sequence",
        }
    if isinstance(value, Mapping):
        return {
            "fields": {
                key: _state_value_structure(item) for key, item in sorted(value.items())
            },
            "kind": "mapping",
        }
    _runtime_fail(
        "runtime.structure", "runtime structure contains an unsupported value"
    )
    raise AssertionError("unreachable")


def _runtime_state_mismatches(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any], *, rank: int
) -> list[dict[str, Any]]:
    checkpoint_compatibility = copy.deepcopy(dict(checkpoint))
    current_compatibility = copy.deepcopy(dict(current))
    checkpoint_compatibility["optimizer"].pop("slots", None)
    current_compatibility["optimizer"].pop("slots", None)
    return _json_field_mismatches(
        checkpoint_compatibility,
        current_compatibility,
        field=f"ranks.{rank}.runtime",
    )


def _require_string_sequence_in_order(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        _runtime_fail(field, "value must be a nonempty string sequence")
    items = tuple(_require_nonempty_string(item, field=f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        _runtime_fail(field, "value must not contain duplicates")
    return items


def _runtime_fail(
    field: str, message: str, *, context: Mapping[str, Any] | None = None
) -> None:
    _fail(
        message,
        code="training_state.runtime_state",
        context={"field": field, **dict(context or {})},
    )


def _validate_file_snapshot(record: TrainingStateFile, content: bytes) -> None:
    observed_digest = _sha256(content)
    if len(content) != record.size or observed_digest != record.sha256:
        _fail(
            "training-state component failed size or digest validation",
            code="training_state.corrupt_component",
            context={
                "path": record.path,
                "expected_size": record.size,
                "observed_size": len(content),
                "expected_sha256": record.sha256,
                "observed_sha256": observed_digest,
            },
        )


def _decode_resolved_config(encoded: bytes) -> Mapping[str, Any]:
    try:
        raw = _strict_json_loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            "resolved configuration is not strict JSON",
            code="training_state.corrupt_component",
            context={"path": TRAINING_STATE_RESOLVED_CONFIG},
            cause=exc,
        )
    normalized = _normalize_json_value(raw, field="resolved_config")
    value = _require_mapping(normalized, field="resolved_config")
    if not value:
        _fail(
            "resolved configuration must be a nonempty mapping",
            code="training_state.schema",
        )
    if _canonical_json_bytes(value) + b"\n" != encoded:
        _fail(
            "resolved configuration bytes are not canonical",
            code="training_state.corrupt_component",
            context={"path": TRAINING_STATE_RESOLVED_CONFIG},
        )
    return MappingProxyType(dict(value))


def _decode_resume_compatibility(encoded: bytes) -> Mapping[str, Any]:
    try:
        raw = _strict_json_loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(
            "resume compatibility is not strict JSON",
            code="training_state.corrupt_component",
            context={"path": TRAINING_STATE_RESUME_COMPATIBILITY},
            cause=exc,
        )
    value = _require_mapping(
        _normalize_json_value(raw, field="resume_compatibility"),
        field="resume_compatibility",
    )
    _require_exact_fields(
        value,
        frozenset({"schema", "schema_version", "semantic_config"}),
        field="resume_compatibility",
    )
    if (
        value["schema"] != TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA
        or value["schema_version"] != TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA_VERSION
    ):
        _fail(
            "resume compatibility schema is unsupported",
            code="training_state.unsupported_schema",
            context={
                "schema": value["schema"],
                "schema_version": value["schema_version"],
            },
        )
    semantic_config = _require_mapping(
        value["semantic_config"],
        field="resume_compatibility.semantic_config",
    )
    if not semantic_config:
        _fail(
            "resume compatibility semantic configuration must be nonempty",
            code="training_state.incomplete",
        )
    if _canonical_json_bytes(value) + b"\n" != encoded:
        _fail(
            "resume compatibility bytes are not canonical",
            code="training_state.corrupt_component",
            context={"path": TRAINING_STATE_RESUME_COMPATIBILITY},
        )
    return MappingProxyType(dict(value))


def _json_field_mismatches(
    checkpoint: Any,
    current: Any,
    *,
    field: str,
) -> list[dict[str, Any]]:
    if isinstance(checkpoint, Mapping) and isinstance(current, Mapping):
        mismatches: list[dict[str, Any]] = []
        for key in sorted(set(checkpoint) | set(current)):
            child = f"{field}.{key}"
            if key not in checkpoint:
                mismatches.append(
                    {"checkpoint": "<missing>", "current": current[key], "field": child}
                )
            elif key not in current:
                mismatches.append(
                    {
                        "checkpoint": checkpoint[key],
                        "current": "<missing>",
                        "field": child,
                    }
                )
            else:
                mismatches.extend(
                    _json_field_mismatches(checkpoint[key], current[key], field=child)
                )
        return mismatches
    if isinstance(checkpoint, list) and isinstance(current, (list, tuple)):
        mismatches = []
        if len(checkpoint) != len(current):
            return [
                {
                    "checkpoint": len(checkpoint),
                    "current": len(current),
                    "field": f"{field}.length",
                }
            ]
        for index, (checkpoint_item, current_item) in enumerate(
            zip(checkpoint, current, strict=True)
        ):
            mismatches.extend(
                _json_field_mismatches(
                    checkpoint_item, current_item, field=f"{field}.{index}"
                )
            )
        return mismatches
    if type(checkpoint) is not type(current) or checkpoint != current:
        return [{"checkpoint": checkpoint, "current": current, "field": field}]
    return []


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _open_directory_at(directory_fd: int, name: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(name, flags, dir_fd=directory_fd)


def _tree_inventory_at(root_fd: int) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk(directory_fd: int, prefix: str) -> None:
        for name in os.listdir(directory_fd):
            relative = f"{prefix}/{name}" if prefix else name
            try:
                descriptor = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                _fail(
                    "training-state inventory cannot be read",
                    code="training_state.unsafe_path",
                    context={"path": relative, "error_type": type(exc).__name__},
                    cause=exc,
                )
            if stat.S_ISLNK(descriptor.st_mode):
                _fail(
                    "training-state paths must not contain symlinks",
                    code="training_state.unsafe_path",
                    context={"path": relative},
                )
            if stat.S_ISDIR(descriptor.st_mode):
                directories.add(relative)
                child_fd = _open_directory_at(directory_fd, name)
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(descriptor.st_mode):
                files.add(relative)
            else:
                _fail(
                    "training-state inventory contains an unsupported entry",
                    code="training_state.unsafe_path",
                    context={"path": relative},
                )

    walk(root_fd, "")
    return files, directories


def _open_relative_file_at(root_fd: int, relative_path: str, flags: int) -> int:
    parts = PurePosixPath(relative_path).parts
    if not parts:
        raise FileNotFoundError(relative_path)
    current_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_directory_at(current_fd, part)
            os.close(current_fd)
            current_fd = next_fd
        return os.open(
            parts[-1],
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=current_fd,
        )
    finally:
        os.close(current_fd)


def _read_stable_regular_file_at(root_fd: int, relative_path: str) -> bytes:
    descriptor = _open_relative_file_at(root_fd, relative_path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(
                "training-state component is not a regular file",
                code="training_state.unsafe_path",
                context={"path": relative_path},
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(
                "training-state component changed while being read",
                code="training_state.corrupt_component",
                context={"path": relative_path},
            )
        content = b"".join(chunks)
        if len(content) != before.st_size:
            _fail(
                "training-state component size changed while being read",
                code="training_state.corrupt_component",
                context={"path": relative_path},
            )
        return content
    finally:
        os.close(descriptor)


def _write_new_file_durable_at(
    root_fd: int, relative_path: str, content: bytes
) -> None:
    descriptor = _open_relative_file_at(
        root_fd,
        relative_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "training-state file write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace_at(
    checkpoint_fd: int, stage_name: str, target_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace training-state publication is unavailable",
            target_name,
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        checkpoint_fd,
        os.fsencode(stage_name),
        checkpoint_fd,
        os.fsencode(target_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target_name)
    raise OSError(error_number, os.strerror(error_number), target_name)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    try:
        descriptor = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(descriptor.st_mode) or stat.S_ISLNK(descriptor.st_mode):
        return
    directory_fd = _open_directory_at(parent_fd, name)
    try:
        for child in os.listdir(directory_fd):
            child_stat = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_stat.st_mode) and not stat.S_ISLNK(
                child_stat.st_mode
            ):
                _remove_tree_at(directory_fd, child)
            else:
                os.unlink(child, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(
                "path components cannot be inspected safely",
                code="training_state.unsafe_path",
                context={"path": str(current), "error_type": type(exc).__name__},
                cause=exc,
            )
        if stat.S_ISLNK(mode):
            _fail(
                "training-state paths must not contain symlinks",
                code="training_state.unsafe_path",
                context={"path": str(current)},
            )


def _require_safe_relative_path(value: Any, *, field: str) -> str:
    path = _require_nonempty_string(value, field=field)
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path != pure.as_posix():
        _fail(
            f"{field} must be a normalized relative path",
            code="training_state.unsafe_path",
            context={"path": path},
        )
    return path


def _validate_identities(value: Any) -> dict[str, str]:
    mapping = _require_mapping(value, field="identities")
    _require_exact_fields(
        mapping, frozenset(REQUIRED_IDENTITY_KINDS), field="identities"
    )
    return {
        name: _require_sha256(mapping[name], field=f"identities.{name}")
        for name in REQUIRED_IDENTITY_KINDS
    }


def _validate_optional_component(
    value: bytes | None, *, applicable: bool, field: str
) -> None:
    if applicable:
        _require_bytes(value, field=field)
    elif value is not None:
        _fail(
            f"{field} must be absent when not applicable",
            code="training_state.schema",
        )


def _require_bytes(value: Any, *, field: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        _fail(
            f"{field} must be nonempty bytes",
            code="training_state.incomplete",
        )
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be a string-keyed mapping", code="training_state.schema")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, field: str
) -> None:
    observed = set(value)
    if observed != expected:
        _fail(
            f"{field} fields do not match the strict schema",
            code="training_state.schema",
            context={
                "field": field,
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(
            f"{field} must be a nonempty trimmed string", code="training_state.schema"
        )
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean", code="training_state.schema")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer", code="training_state.schema")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a nonnegative integer", code="training_state.schema")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            f"{field} must be a lowercase SHA-256 digest",
            code="training_state.schema",
        )
    return value


def _require_string_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        _fail(
            f"{field} must be a nonempty string sequence", code="training_state.schema"
        )
    items = tuple(_require_nonempty_string(item, field=f"{field}[]") for item in value)
    if items != tuple(sorted(set(items))):
        _fail(f"{field} must be sorted and unique", code="training_state.schema")
    return items


def _normalize_json_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{field} contains a non-finite float", code="training_state.schema")
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item, field=f"{field}[]") for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _normalize_json_value(item, field=f"{field}.{key}")
            for key, item in sorted(value.items())
        }
    _fail(
        f"{field} contains a value outside strict JSON",
        code="training_state.schema",
        context={"value_type": type(value).__name__},
    )
    raise AssertionError("unreachable")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize_json_value(value, field="json")
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_loads(encoded: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _filename_for_role(role: str) -> str:
    if role.startswith("rng:"):
        return f"rng-{role.removeprefix('rng:')}.bin"
    return {
        "cursor": "cursor.json",
        "optimizer": "optimizer.bin",
        "scaler": "scaler.bin",
        "scheduler": "scheduler.bin",
        "trainable_model": "trainable-model.bin",
    }[role]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _expectations_from_manifest(
    manifest: TrainingStateManifest,
) -> TrainingStateExpectations:
    return TrainingStateExpectations(
        checkpoint_step=manifest.checkpoint_step,
        world_size=manifest.world_size,
        identities=manifest.identities,
        scheduler_applicable=manifest.scheduler_applicable,
        scaler_applicable=manifest.scaler_applicable,
        rng_kinds=REQUIRED_RNG_KINDS,
    )


def _reload_matches_manifest_at(
    anchor: _CheckpointAnchor, expected_manifest: TrainingStateManifest
) -> bool:
    try:
        admitted = _admit_training_state_at(
            anchor,
            _expectations_from_manifest(expected_manifest),
            current_rank=0,
            expected_manifest=expected_manifest,
        )
    except BaseException:
        return False
    return admitted.manifest.to_dict() == expected_manifest.to_dict()


def _fail(
    message: str,
    *,
    code: str,
    context: Mapping[str, Any] | None = None,
    cause: BaseException | None = None,
) -> None:
    raise ArtifactContractError(message, code=code, context=context, cause=cause)


__all__ = [
    "AdmittedTrainingState",
    "DecodedRankTrainingState",
    "PublishedRankContribution",
    "PublishedTrainingState",
    "REQUIRED_IDENTITY_KINDS",
    "REQUIRED_RNG_KINDS",
    "RankTrainingStatePayload",
    "RestoredRankTrainingState",
    "RuntimeStateExpectations",
    "TRAINING_STATE_ARTIFACT_TYPE",
    "TRAINING_STATE_COMMIT_STATUS",
    "TRAINING_STATE_DIRECTORY",
    "TRAINING_STATE_MANIFEST",
    "TRAINING_STATE_RESOLVED_CONFIG",
    "TRAINING_STATE_RESUME_COMPATIBILITY",
    "TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA",
    "TRAINING_STATE_RESUME_COMPATIBILITY_SCHEMA_VERSION",
    "TRAINING_STATE_SAVE_BOUNDARY",
    "TRAINING_STATE_SCHEMA",
    "TRAINING_STATE_SCHEMA_VERSION",
    "TRAINING_STATE_TORCH_CUDA_RNG_SCHEMA_VERSION",
    "TrainingStateExpectations",
    "TrainingStateContributionError",
    "TrainingStateContributionSession",
    "TrainingStateFile",
    "TrainingStateManifest",
    "TrainingStatePublication",
    "TrainingStatePublicationError",
    "TrainingStatePublicationPlan",
    "TrainingStateRank",
    "admit_training_state",
    "begin_training_state_contributions",
    "build_resume_compatibility_projection",
    "build_training_state_manifest",
    "capture_runtime_state_expectations",
    "commit_training_state_contributions",
    "load_training_state_manifest",
    "publish_rank_training_state_contribution",
    "publish_training_state",
    "restore_decoded_rank_training_state",
    "serialize_rank_training_state",
]
