#!/usr/bin/env python3
"""Immutable, content-addressed record contracts for the Wave 5 matrix.

This module is record/admission-only: it performs no model, GPU, subprocess,
or artifact write.  It freezes the identity of the canonical
``{1.0,1.10} x {matrix_a,matrix_b,matrix_c} x {A,B,C}`` screen -- exactly six
matrix acquisitions and eighteen cells -- so that the three later lanes (the
dry-run materializer, the private runtime, and the dual-RP analyzer) consume
one shared schema and one shared admission choke point rather than defining
parallel record shapes.

Every cross-artifact relationship is content-addressed: records reference
each other by SHA-256 digest of a canonical payload, never by mutable object
identity.  ``MatrixPlan`` is the one aggregate admission choke point; every
supported constructor path (direct construction, ``from_dict``) funnels
through its ``__post_init__``, which freezes tuples before hashing and fails
closed on missing, mixed, duplicated, adaptively retried, or noncanonical
records.  Qualification identities are representable through the same leaf
types but are never admitted into the eighteen-cell matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any

from scripts.research.build_human13_k_union_manifest import EXPECTED_IMAGE_IDENTITIES


_SHA256 = re.compile(r"^[0-9a-f]{64}$")

TRAINING_RPS: tuple[float, ...] = (1.0, 1.10)
EVALUATION_RPS: tuple[float, ...] = (1.0, 1.10)
MATRIX_SEED_GROUPS: tuple[str, ...] = ("matrix_a", "matrix_b", "matrix_c")
QUALIFICATION_SEED_GROUP = "qualification"
ARM_IDS: tuple[str, ...] = ("A", "B", "C")
PHASE_MATRIX = "matrix"
PHASE_QUALIFICATION = "qualification"
CANONICAL_IMAGE_IDS: tuple[int, ...] = tuple(
    sorted(image_id for image_id, _ in EXPECTED_IMAGE_IDENTITIES)
)
DRY_RUN_COUNTER_KEYS: tuple[str, ...] = (
    "model_loads",
    "gpu_allocations",
    "subprocess_launches",
    "output_roots_created",
)

_CANONICAL_IMAGE_ID_SET = frozenset(CANONICAL_IMAGE_IDS)
_ARM_OBJECTIVE_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler", "preservation"),
    }
)
_STATUSES = ("succeeded", "failed")

ACQUISITION_KEY_SCHEMA = "human13_rp_crossover_acquisition_key.v1"
CELL_KEY_SCHEMA = "human13_rp_crossover_cell_key.v1"
SHARED_EVIDENCE_SCHEMA = "human13_rp_crossover_shared_evidence.v1"
SOURCE_BASELINE_SCHEMA = "human13_rp_crossover_source_baseline.v1"
CELL_SPEC_SCHEMA = "human13_rp_crossover_cell_spec.v1"
AUDIT_REF_SCHEMA = "human13_rp_crossover_audit_ref.v1"
CELL_RECEIPT_SCHEMA = "human13_rp_crossover_cell_receipt.v1"
MATRIX_PLAN_SCHEMA = "human13_rp_crossover_matrix_plan.v1"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _digest(value: object, *, field: str) -> str:
    text = _nonempty(value, field=field)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _optional_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field=field)


def _training_rp(value: Any) -> float:
    rp = float(value)
    if rp not in TRAINING_RPS or isinstance(value, bool):
        raise ValueError("training_rp must be exactly 1.0 or 1.10")
    return rp


def _evaluation_rp(value: Any) -> float:
    rp = float(value)
    if rp not in EVALUATION_RPS or isinstance(value, bool):
        raise ValueError("evaluation_rp must be exactly 1.0 or 1.10")
    return rp


def _image_ids(value: Any, *, field: str) -> tuple[int, ...]:
    ids = tuple(int(item) for item in value)
    if len(ids) != len(CANONICAL_IMAGE_IDS) or set(ids) != _CANONICAL_IMAGE_ID_SET:
        raise ValueError(f"{field} must cover the sealed thirteen-image panel exactly")
    return ids


@dataclass(frozen=True)
class AcquisitionKey:
    """One ``(training RP, seed group, phase)`` acquisition identity."""

    training_rp: float
    seed_group_id: str
    phase: str

    def __post_init__(self) -> None:
        rp = _training_rp(self.training_rp)
        object.__setattr__(self, "training_rp", rp)
        phase = _nonempty(self.phase, field="phase")
        if phase == PHASE_MATRIX:
            if self.seed_group_id not in MATRIX_SEED_GROUPS:
                raise ValueError(
                    "matrix-phase acquisitions require a matrix_a/b/c seed group"
                )
        elif phase == PHASE_QUALIFICATION:
            if self.seed_group_id != QUALIFICATION_SEED_GROUP:
                raise ValueError(
                    "qualification-phase acquisitions require the qualification seed group"
                )
        else:
            raise ValueError("phase must be exactly 'matrix' or 'qualification'")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self, "seed_group_id", _nonempty(self.seed_group_id, field="seed_group_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACQUISITION_KEY_SCHEMA,
            "training_rp": self.training_rp,
            "seed_group_id": self.seed_group_id,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionKey":
        if value.get("schema_version") != ACQUISITION_KEY_SCHEMA:
            raise ValueError("acquisition key schema_version differs")
        return cls(
            training_rp=value["training_rp"],
            seed_group_id=value["seed_group_id"],
            phase=value["phase"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CellKey:
    """One arm of one acquisition: ``(acquisition_key, arm_id)``."""

    acquisition_key: AcquisitionKey
    arm_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.acquisition_key, AcquisitionKey):
            raise ValueError("cell key acquisition_key must be an AcquisitionKey")
        arm_id = _nonempty(self.arm_id, field="arm_id")
        if arm_id not in ARM_IDS:
            raise ValueError("arm_id must be exactly A, B, or C")
        object.__setattr__(self, "arm_id", arm_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_KEY_SCHEMA,
            "acquisition_key": self.acquisition_key.to_dict(),
            "arm_id": self.arm_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellKey":
        if value.get("schema_version") != CELL_KEY_SCHEMA:
            raise ValueError("cell key schema_version differs")
        return cls(
            acquisition_key=AcquisitionKey.from_dict(value["acquisition_key"]),
            arm_id=value["arm_id"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class SharedEvidenceRef:
    """Byte-identical acquisition/matching/credit/policy identity for one acquisition key."""

    source_sha256: str
    manifest_sha256: str
    acquisition_path: str
    acquisition_sha256: str
    trajectory_credit_acquisition_sha256: str
    credit_ledger_sha256: str
    compiler_ledger_sha256: str
    policy_contract_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_sha256", _digest(self.source_sha256, field="source_sha256")
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _digest(self.manifest_sha256, field="manifest_sha256"),
        )
        object.__setattr__(
            self,
            "acquisition_path",
            _nonempty(self.acquisition_path, field="acquisition_path"),
        )
        object.__setattr__(
            self,
            "acquisition_sha256",
            _digest(self.acquisition_sha256, field="acquisition_sha256"),
        )
        object.__setattr__(
            self,
            "trajectory_credit_acquisition_sha256",
            _digest(
                self.trajectory_credit_acquisition_sha256,
                field="trajectory_credit_acquisition_sha256",
            ),
        )
        object.__setattr__(
            self,
            "credit_ledger_sha256",
            _digest(self.credit_ledger_sha256, field="credit_ledger_sha256"),
        )
        object.__setattr__(
            self,
            "compiler_ledger_sha256",
            _digest(self.compiler_ledger_sha256, field="compiler_ledger_sha256"),
        )
        object.__setattr__(
            self,
            "policy_contract_sha256",
            _digest(self.policy_contract_sha256, field="policy_contract_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SHARED_EVIDENCE_SCHEMA,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_path": self.acquisition_path,
            "acquisition_sha256": self.acquisition_sha256,
            "trajectory_credit_acquisition_sha256": self.trajectory_credit_acquisition_sha256,
            "credit_ledger_sha256": self.credit_ledger_sha256,
            "compiler_ledger_sha256": self.compiler_ledger_sha256,
            "policy_contract_sha256": self.policy_contract_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedEvidenceRef":
        if value.get("schema_version") != SHARED_EVIDENCE_SCHEMA:
            raise ValueError("shared evidence schema_version differs")
        return cls(
            source_sha256=value["source_sha256"],
            manifest_sha256=value["manifest_sha256"],
            acquisition_path=value["acquisition_path"],
            acquisition_sha256=value["acquisition_sha256"],
            trajectory_credit_acquisition_sha256=value[
                "trajectory_credit_acquisition_sha256"
            ],
            credit_ledger_sha256=value["credit_ledger_sha256"],
            compiler_ledger_sha256=value["compiler_ledger_sha256"],
            policy_contract_sha256=value["policy_contract_sha256"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class SourceBaselineRef:
    """Two deterministic Source outputs for one evaluation RP over the sealed panel."""

    evaluation_rp: float
    output_a_sha256: str
    output_b_sha256: str
    checkpoint_sha256: str
    image_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_rp", _evaluation_rp(self.evaluation_rp))
        object.__setattr__(
            self,
            "output_a_sha256",
            _digest(self.output_a_sha256, field="output_a_sha256"),
        )
        object.__setattr__(
            self,
            "output_b_sha256",
            _digest(self.output_b_sha256, field="output_b_sha256"),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _digest(self.checkpoint_sha256, field="checkpoint_sha256"),
        )
        object.__setattr__(
            self, "image_ids", _image_ids(self.image_ids, field="image_ids")
        )
        if self.output_a_sha256 != self.output_b_sha256:
            raise ValueError(
                "deterministic Source baseline outputs must be byte-identical"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_BASELINE_SCHEMA,
            "evaluation_rp": self.evaluation_rp,
            "output_a_sha256": self.output_a_sha256,
            "output_b_sha256": self.output_b_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "image_ids": list(self.image_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceBaselineRef":
        if value.get("schema_version") != SOURCE_BASELINE_SCHEMA:
            raise ValueError("source baseline schema_version differs")
        return cls(
            evaluation_rp=value["evaluation_rp"],
            output_a_sha256=value["output_a_sha256"],
            output_b_sha256=value["output_b_sha256"],
            checkpoint_sha256=value["checkpoint_sha256"],
            image_ids=tuple(value["image_ids"]),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CellSpec:
    """One planned cell: its evidence, leaf config, and exact single-update contract."""

    cell_key: CellKey
    shared_evidence: SharedEvidenceRef
    leaf_config_sha256: str
    source_checkpoint_sha256: str
    expected_objective_components: tuple[str, ...]
    fresh_adamw_fingerprint_sha256: str
    evaluation_rps: tuple[float, ...]
    output_root: str
    max_updates: int = 1
    retry_policy: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.cell_key, CellKey):
            raise ValueError("cell spec cell_key must be a CellKey")
        if not isinstance(self.shared_evidence, SharedEvidenceRef):
            raise ValueError("cell spec shared_evidence must be a SharedEvidenceRef")
        object.__setattr__(
            self,
            "leaf_config_sha256",
            _digest(self.leaf_config_sha256, field="leaf_config_sha256"),
        )
        object.__setattr__(
            self,
            "source_checkpoint_sha256",
            _digest(self.source_checkpoint_sha256, field="source_checkpoint_sha256"),
        )
        components = tuple(self.expected_objective_components)
        if components != _ARM_OBJECTIVE_COMPONENTS[self.cell_key.arm_id]:
            raise ValueError(
                "expected_objective_components differs from the canonical arm-nested surface"
            )
        object.__setattr__(self, "expected_objective_components", components)
        object.__setattr__(
            self,
            "fresh_adamw_fingerprint_sha256",
            _digest(
                self.fresh_adamw_fingerprint_sha256,
                field="fresh_adamw_fingerprint_sha256",
            ),
        )
        evaluation_rps = tuple(_evaluation_rp(rp) for rp in self.evaluation_rps)
        if evaluation_rps != EVALUATION_RPS:
            raise ValueError(
                "cell spec must declare exactly the two evaluation RPs in canonical order"
            )
        object.__setattr__(self, "evaluation_rps", evaluation_rps)
        object.__setattr__(
            self, "output_root", _nonempty(self.output_root, field="output_root")
        )
        if self.max_updates != 1:
            raise ValueError("max_updates must be exactly 1")
        if self.retry_policy != "none":
            raise ValueError("retry_policy must be exactly 'none'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_SPEC_SCHEMA,
            "cell_key": self.cell_key.to_dict(),
            "shared_evidence": self.shared_evidence.to_dict(),
            "leaf_config_sha256": self.leaf_config_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "expected_objective_components": list(self.expected_objective_components),
            "fresh_adamw_fingerprint_sha256": self.fresh_adamw_fingerprint_sha256,
            "evaluation_rps": list(self.evaluation_rps),
            "output_root": self.output_root,
            "max_updates": self.max_updates,
            "retry_policy": self.retry_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellSpec":
        if value.get("schema_version") != CELL_SPEC_SCHEMA:
            raise ValueError("cell spec schema_version differs")
        return cls(
            cell_key=CellKey.from_dict(value["cell_key"]),
            shared_evidence=SharedEvidenceRef.from_dict(value["shared_evidence"]),
            leaf_config_sha256=value["leaf_config_sha256"],
            source_checkpoint_sha256=value["source_checkpoint_sha256"],
            expected_objective_components=tuple(value["expected_objective_components"]),
            fresh_adamw_fingerprint_sha256=value["fresh_adamw_fingerprint_sha256"],
            evaluation_rps=tuple(value["evaluation_rps"]),
            output_root=value["output_root"],
            max_updates=value["max_updates"],
            retry_policy=value["retry_policy"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class AuditRef:
    """One clean-greedy behavioral audit at one evaluation RP over the sealed panel."""

    evaluation_rp: float
    evaluated_checkpoint_sha256: str
    output_path: str
    output_sha256: str
    row_count: int
    image_ids: tuple[int, ...]
    generation_policy_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_rp", _evaluation_rp(self.evaluation_rp))
        object.__setattr__(
            self,
            "evaluated_checkpoint_sha256",
            _digest(
                self.evaluated_checkpoint_sha256, field="evaluated_checkpoint_sha256"
            ),
        )
        object.__setattr__(
            self, "output_path", _nonempty(self.output_path, field="output_path")
        )
        object.__setattr__(
            self, "output_sha256", _digest(self.output_sha256, field="output_sha256")
        )
        if self.row_count != len(CANONICAL_IMAGE_IDS):
            raise ValueError(
                "audit row_count must equal the sealed thirteen-image panel size"
            )
        object.__setattr__(
            self, "image_ids", _image_ids(self.image_ids, field="image_ids")
        )
        object.__setattr__(
            self,
            "generation_policy_receipt_sha256",
            _digest(
                self.generation_policy_receipt_sha256,
                field="generation_policy_receipt_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_REF_SCHEMA,
            "evaluation_rp": self.evaluation_rp,
            "evaluated_checkpoint_sha256": self.evaluated_checkpoint_sha256,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "row_count": self.row_count,
            "image_ids": list(self.image_ids),
            "generation_policy_receipt_sha256": self.generation_policy_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuditRef":
        if value.get("schema_version") != AUDIT_REF_SCHEMA:
            raise ValueError("audit ref schema_version differs")
        return cls(
            evaluation_rp=value["evaluation_rp"],
            evaluated_checkpoint_sha256=value["evaluated_checkpoint_sha256"],
            output_path=value["output_path"],
            output_sha256=value["output_sha256"],
            row_count=value["row_count"],
            image_ids=tuple(value["image_ids"]),
            generation_policy_receipt_sha256=value["generation_policy_receipt_sha256"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CellReceipt:
    """The immutable outcome of one cell: proposal, audits, transaction, and status."""

    cell_key: CellKey
    shared_evidence: SharedEvidenceRef
    objective_components: tuple[str, ...]
    before_transaction_digest: str
    after_transaction_digest: str
    status: str
    audits: tuple[AuditRef, ...] = ()
    adamw_proposal_sha256: str | None = None
    projection_receipt_sha256: str | None = None
    apply_receipt_sha256: str | None = None
    update_count: int = 1
    retry_policy: str = "none"
    rollback_confirmed: bool = True
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell_key, CellKey):
            raise ValueError("cell receipt cell_key must be a CellKey")
        if not isinstance(self.shared_evidence, SharedEvidenceRef):
            raise ValueError("cell receipt shared_evidence must be a SharedEvidenceRef")
        components = tuple(self.objective_components)
        if components != _ARM_OBJECTIVE_COMPONENTS[self.cell_key.arm_id]:
            raise ValueError(
                "objective_components differs from the canonical arm-nested surface"
            )
        object.__setattr__(self, "objective_components", components)
        object.__setattr__(
            self,
            "before_transaction_digest",
            _digest(self.before_transaction_digest, field="before_transaction_digest"),
        )
        object.__setattr__(
            self,
            "after_transaction_digest",
            _digest(self.after_transaction_digest, field="after_transaction_digest"),
        )
        if self.before_transaction_digest != self.after_transaction_digest:
            raise ValueError(
                "every cell receipt must restore the pre-proposal transaction state exactly"
            )
        if self.rollback_confirmed is not True:
            raise ValueError(
                "rollback_confirmed must be exactly True on every cell receipt"
            )
        status = _nonempty(self.status, field="status")
        if status not in _STATUSES:
            raise ValueError("status must be exactly 'succeeded' or 'failed'")
        object.__setattr__(self, "status", status)
        if self.update_count != 1:
            raise ValueError("update_count must be exactly 1")
        if self.retry_policy != "none":
            raise ValueError("retry_policy must be exactly 'none'")

        requires_projection = "preservation" in components
        object.__setattr__(
            self,
            "adamw_proposal_sha256",
            _optional_digest(self.adamw_proposal_sha256, field="adamw_proposal_sha256"),
        )
        object.__setattr__(
            self,
            "apply_receipt_sha256",
            _optional_digest(self.apply_receipt_sha256, field="apply_receipt_sha256"),
        )
        projection = _optional_digest(
            self.projection_receipt_sha256, field="projection_receipt_sha256"
        )
        if not requires_projection and projection is not None:
            raise ValueError(
                "only the preservation arm (C) may bind a projection receipt"
            )
        object.__setattr__(self, "projection_receipt_sha256", projection)

        audits = tuple(self.audits)
        if any(not isinstance(audit, AuditRef) for audit in audits):
            raise ValueError("audits must contain only AuditRef records")
        audit_rps = tuple(audit.evaluation_rp for audit in audits)
        if len(set(audit_rps)) != len(audit_rps) or set(audit_rps) - set(
            EVALUATION_RPS
        ):
            raise ValueError(
                "audits must not duplicate or use a non-canonical evaluation RP"
            )
        object.__setattr__(self, "audits", audits)

        if status == "succeeded":
            if self.failure_reason is not None:
                raise ValueError(
                    "a succeeded cell receipt must not carry a failure_reason"
                )
            if self.adamw_proposal_sha256 is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its exact AdamW proposal identity"
                )
            if self.apply_receipt_sha256 is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its apply evidence"
                )
            if requires_projection and self.projection_receipt_sha256 is None:
                raise ValueError(
                    "a succeeded preservation-arm receipt must bind its projection evidence"
                )
            if set(audit_rps) != set(EVALUATION_RPS):
                raise ValueError(
                    "a succeeded cell receipt must bind exactly one audit per evaluation RP"
                )
        else:
            if not _nonempty(self.failure_reason or "", field="failure_reason"):
                raise ValueError(
                    "a failed cell receipt must carry a nonempty failure_reason"
                )
            object.__setattr__(self, "failure_reason", self.failure_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_RECEIPT_SCHEMA,
            "cell_key": self.cell_key.to_dict(),
            "shared_evidence": self.shared_evidence.to_dict(),
            "objective_components": list(self.objective_components),
            "before_transaction_digest": self.before_transaction_digest,
            "after_transaction_digest": self.after_transaction_digest,
            "status": self.status,
            "audits": [audit.to_dict() for audit in self.audits],
            "adamw_proposal_sha256": self.adamw_proposal_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "apply_receipt_sha256": self.apply_receipt_sha256,
            "update_count": self.update_count,
            "retry_policy": self.retry_policy,
            "rollback_confirmed": self.rollback_confirmed,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellReceipt":
        if value.get("schema_version") != CELL_RECEIPT_SCHEMA:
            raise ValueError("cell receipt schema_version differs")
        return cls(
            cell_key=CellKey.from_dict(value["cell_key"]),
            shared_evidence=SharedEvidenceRef.from_dict(value["shared_evidence"]),
            objective_components=tuple(value["objective_components"]),
            before_transaction_digest=value["before_transaction_digest"],
            after_transaction_digest=value["after_transaction_digest"],
            status=value["status"],
            audits=tuple(AuditRef.from_dict(item) for item in value["audits"]),
            adamw_proposal_sha256=value["adamw_proposal_sha256"],
            projection_receipt_sha256=value["projection_receipt_sha256"],
            apply_receipt_sha256=value["apply_receipt_sha256"],
            update_count=value["update_count"],
            retry_policy=value["retry_policy"],
            rollback_confirmed=value["rollback_confirmed"],
            failure_reason=value["failure_reason"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


def _validate_matrix_plan(
    acquisitions: tuple[AcquisitionKey, ...],
    cells: tuple[CellSpec, ...],
    source_baselines: tuple[SourceBaselineRef, ...],
    dependency_edges: tuple[tuple[str, str], ...],
    concurrency_cap: int,
    dry_run_counters: Mapping[str, int],
) -> None:
    """The one aggregate admission choke point for the canonical matrix shape.

    Every supported construction path -- direct construction and
    ``from_dict`` -- reaches this function through ``MatrixPlan.__post_init__``
    so no path can admit a differently-shaped or cross-artifact-mixed plan.
    """

    expected_acquisition_count = len(TRAINING_RPS) * len(MATRIX_SEED_GROUPS)
    if len(acquisitions) != expected_acquisition_count:
        raise ValueError("matrix plan must bind exactly six matrix acquisitions")
    if any(not isinstance(item, AcquisitionKey) for item in acquisitions):
        raise ValueError("matrix plan acquisitions must be AcquisitionKey records")
    if any(item.phase != PHASE_MATRIX for item in acquisitions):
        raise ValueError(
            "qualification acquisitions must never be pooled into the matrix"
        )
    expected_pairs = {
        (rp, group) for rp in TRAINING_RPS for group in MATRIX_SEED_GROUPS
    }
    actual_pairs = {(item.training_rp, item.seed_group_id) for item in acquisitions}
    if actual_pairs != expected_pairs or len(actual_pairs) != len(acquisitions):
        raise ValueError(
            "matrix plan acquisitions must cover the canonical 2x3 RP/seed-group surface exactly once"
        )

    expected_cell_count = expected_acquisition_count * len(ARM_IDS)
    if len(cells) != expected_cell_count:
        raise ValueError("matrix plan must bind exactly eighteen cells")
    if any(not isinstance(cell, CellSpec) for cell in cells):
        raise ValueError("matrix plan cells must be CellSpec records")

    acquisitions_by_content = {item.content_sha256: item for item in acquisitions}
    cells_by_acquisition: dict[str, dict[str, CellSpec]] = {}
    output_roots: set[str] = set()
    proposal_identities: set[str] = set()
    for cell in cells:
        acquisition = cell.cell_key.acquisition_key
        bound = acquisitions_by_content.get(acquisition.content_sha256)
        if bound is None or bound != acquisition:
            raise ValueError(
                "every cell must reference exactly one of the six bound acquisitions"
            )
        arm_map = cells_by_acquisition.setdefault(acquisition.content_sha256, {})
        if cell.cell_key.arm_id in arm_map:
            raise ValueError("one acquisition must not repeat an arm across cells")
        arm_map[cell.cell_key.arm_id] = cell
        if cell.output_root in output_roots:
            raise ValueError("every cell output_root must be unique")
        output_roots.add(cell.output_root)
        if cell.fresh_adamw_fingerprint_sha256 in proposal_identities:
            raise ValueError(
                "every cell must have an independent fresh AdamW proposal identity"
            )
        proposal_identities.add(cell.fresh_adamw_fingerprint_sha256)

    if len(cells_by_acquisition) != len(acquisitions):
        raise ValueError("every bound acquisition must own exactly one cell group")
    for arm_map in cells_by_acquisition.values():
        if set(arm_map) != set(ARM_IDS):
            raise ValueError("every acquisition must bind exactly arms A, B, and C")
        shared_refs = {arm_map[arm].shared_evidence for arm in ARM_IDS}
        if len(shared_refs) != 1:
            raise ValueError(
                "arms A, B, and C under one acquisition key must share byte-identical evidence"
            )

    if len(source_baselines) != len(EVALUATION_RPS):
        raise ValueError("matrix plan must bind exactly two RP source baselines")
    if any(not isinstance(item, SourceBaselineRef) for item in source_baselines):
        raise ValueError(
            "matrix plan source baselines must be SourceBaselineRef records"
        )
    if {item.evaluation_rp for item in source_baselines} != set(EVALUATION_RPS):
        raise ValueError(
            "matrix plan source baselines must cover RP 1.0 and RP 1.10 exactly once"
        )

    expected_edges = {
        (cell.cell_key.acquisition_key.content_sha256, cell.content_sha256)
        for cell in cells
    }
    actual_edges = set(dependency_edges)
    if (
        len(dependency_edges) != len(cells)
        or actual_edges != expected_edges
        or len(actual_edges) != len(dependency_edges)
    ):
        raise ValueError(
            "dependency edges must bind exactly each cell to its own acquisition, once each"
        )

    if (
        isinstance(concurrency_cap, bool)
        or not isinstance(concurrency_cap, int)
        or not (1 <= concurrency_cap <= 8)
    ):
        raise ValueError("concurrency_cap must be an integer between 1 and 8")

    if set(dry_run_counters) != set(DRY_RUN_COUNTER_KEYS):
        raise ValueError(
            "dry_run_counters schema differs from the canonical zero-action counters"
        )
    if any(count != 0 for count in dry_run_counters.values()):
        raise ValueError(
            "dry_run_counters must be all zero for a materialized-but-unexecuted plan"
        )


@dataclass(frozen=True)
class MatrixPlan:
    """The exact six-acquisition / eighteen-cell canonical matrix and its baselines."""

    acquisitions: tuple[AcquisitionKey, ...]
    cells: tuple[CellSpec, ...]
    source_baselines: tuple[SourceBaselineRef, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    concurrency_cap: int
    dry_run_counters: Mapping[str, int]

    def __post_init__(self) -> None:
        acquisitions = tuple(self.acquisitions)
        cells = tuple(self.cells)
        source_baselines = tuple(self.source_baselines)
        dependency_edges = tuple(
            (str(edge[0]), str(edge[1])) for edge in self.dependency_edges
        )
        dry_run_counters = MappingProxyType(dict(self.dry_run_counters))
        object.__setattr__(self, "acquisitions", acquisitions)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "source_baselines", source_baselines)
        object.__setattr__(self, "dependency_edges", dependency_edges)
        object.__setattr__(self, "dry_run_counters", dry_run_counters)
        _validate_matrix_plan(
            acquisitions,
            cells,
            source_baselines,
            dependency_edges,
            self.concurrency_cap,
            dry_run_counters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATRIX_PLAN_SCHEMA,
            "acquisitions": [item.to_dict() for item in self.acquisitions],
            "cells": [item.to_dict() for item in self.cells],
            "source_baselines": [item.to_dict() for item in self.source_baselines],
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "concurrency_cap": self.concurrency_cap,
            "dry_run_counters": dict(self.dry_run_counters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatrixPlan":
        if value.get("schema_version") != MATRIX_PLAN_SCHEMA:
            raise ValueError("matrix plan schema_version differs")
        return cls(
            acquisitions=tuple(
                AcquisitionKey.from_dict(item) for item in value["acquisitions"]
            ),
            cells=tuple(CellSpec.from_dict(item) for item in value["cells"]),
            source_baselines=tuple(
                SourceBaselineRef.from_dict(item) for item in value["source_baselines"]
            ),
            dependency_edges=tuple(tuple(edge) for edge in value["dependency_edges"]),
            concurrency_cap=value["concurrency_cap"],
            dry_run_counters=dict(value["dry_run_counters"]),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


__all__ = [
    "ACQUISITION_KEY_SCHEMA",
    "ARM_IDS",
    "AUDIT_REF_SCHEMA",
    "AcquisitionKey",
    "AuditRef",
    "CANONICAL_IMAGE_IDS",
    "CELL_KEY_SCHEMA",
    "CELL_RECEIPT_SCHEMA",
    "CELL_SPEC_SCHEMA",
    "CellKey",
    "CellReceipt",
    "CellSpec",
    "DRY_RUN_COUNTER_KEYS",
    "EVALUATION_RPS",
    "MATRIX_PLAN_SCHEMA",
    "MATRIX_SEED_GROUPS",
    "MatrixPlan",
    "PHASE_MATRIX",
    "PHASE_QUALIFICATION",
    "QUALIFICATION_SEED_GROUP",
    "SHARED_EVIDENCE_SCHEMA",
    "SOURCE_BASELINE_SCHEMA",
    "SharedEvidenceRef",
    "SourceBaselineRef",
    "TRAINING_RPS",
]
