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
identity.  ``MatrixPlan`` is the one aggregate admission choke point for the
planned matrix and ``validate_matrix_receipts`` is the one aggregate choke
point for its outcome; every supported constructor path (direct construction,
``from_dict``, the node terminal, and the analysis publisher) funnels through
them, freezes tuples before hashing, and fails closed on missing, mixed,
duplicated, adaptively retried, or noncanonical records.  Qualification
identities are representable through the same leaf types but are never
admitted into the eighteen-cell matrix.

Three identity families are deliberately distinguished and must not be
conflated:

* **acquisition identity** -- the sealed numeric seed tuple, collector
  acquisition/native-receipt bytes, credit ledgers, and policy contract of one
  ``(training RP, seed group)`` acquisition.  Matrix groups and qualification
  carry pairwise distinct acquisition and credit bytes; labels alone never
  separate them.
* **declared AdamW configuration identity** -- one frozen optimizer
  configuration shared by all eighteen cells.
* **per-cell optimizer/proposal/transaction identity** -- unique to each cell,
  because every cell assembles its own fresh Source, optimizer, and
  transaction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from scripts.research.build_human13_k_union_manifest import EXPECTED_IMAGE_IDENTITIES
from scripts.research.collect_human13_rp_crossover import (
    MATRIX_SEED_GROUPS as _COLLECTOR_MATRIX_SEED_GROUPS,
    QUALIFICATION_SEEDS as _COLLECTOR_QUALIFICATION_SEEDS,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")

TRAINING_RPS: tuple[float, ...] = (1.0, 1.10)
EVALUATION_RPS: tuple[float, ...] = (1.0, 1.10)
MATRIX_SEED_GROUPS: tuple[str, ...] = ("matrix_a", "matrix_b", "matrix_c")
QUALIFICATION_SEED_GROUP = "qualification"
QUALIFICATION_LEARNING_RATE_RAY: tuple[float, ...] = (
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
)
ARM_IDS: tuple[str, ...] = ("A", "B", "C")
PHASE_MATRIX = "matrix"
PHASE_QUALIFICATION = "qualification"
SEEDS_PER_GROUP = 16
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
# ``preservation`` is a projection stage over the arm-B proposal, not a loss
# term, so only these components carry their own objective bytes.
PROPOSAL_COMPONENTS_BY_ARM: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "A": ("trajectory",),
        "B": ("trajectory", "compiler"),
        "C": ("trajectory", "compiler"),
    }
)
_STATUSES = ("succeeded", "failed")

ACQUISITION_KEY_SCHEMA = "human13_rp_crossover_acquisition_key.v1"
CELL_KEY_SCHEMA = "human13_rp_crossover_cell_key.v1"
SHARED_EVIDENCE_SCHEMA = "human13_rp_crossover_shared_evidence.v1"
SOURCE_BASELINE_SCHEMA = "human13_rp_crossover_source_baseline.v1"
CELL_SPEC_SCHEMA = "human13_rp_crossover_cell_spec.v1"
AUDIT_REF_SCHEMA = "human13_rp_crossover_audit_ref.v1"
CELL_RECEIPT_SCHEMA = "human13_rp_crossover_cell_receipt.v2"
MATRIX_PLAN_SCHEMA = "human13_rp_crossover_matrix_plan.v1"
NODE_TERMINAL_RECEIPT_SCHEMA = "human13_rp_crossover_node_terminal_receipt.v2"
AGGREGATE_RESOURCE_RECEIPT_SCHEMA = "human13_rp_crossover_resources.v1"
DOSE_MECHANICAL_RECEIPT_SCHEMA = "human13_rp_crossover_dose_mechanics.v1"

# The exact sealed seed ranges of the owning research unit.  They are bound to
# the collector that actually draws them, and re-checked here so that a drift
# on either side fails closed instead of silently re-labelling one acquisition.
_EXPECTED_SEED_STARTS: Mapping[str, int] = MappingProxyType(
    {
        QUALIFICATION_SEED_GROUP: 30001,
        "matrix_a": 31001,
        "matrix_b": 32001,
        "matrix_c": 33001,
    }
)


def _frozen_seed_groups() -> Mapping[str, tuple[int, ...]]:
    collected = {
        QUALIFICATION_SEED_GROUP: tuple(_COLLECTOR_QUALIFICATION_SEEDS),
        **{
            group: tuple(seeds)
            for group, seeds in _COLLECTOR_MATRIX_SEED_GROUPS.items()
        },
    }
    if set(collected) != set(_EXPECTED_SEED_STARTS) or set(
        _COLLECTOR_MATRIX_SEED_GROUPS
    ) != set(MATRIX_SEED_GROUPS):
        raise ValueError("collector seed groups differ from the sealed matrix surface")
    for group, start in _EXPECTED_SEED_STARTS.items():
        if collected[group] != tuple(range(start, start + SEEDS_PER_GROUP)):
            raise ValueError(f"seed group {group} drifted from its sealed seed range")
    drawn = [seed for seeds in collected.values() for seed in seeds]
    if len(set(drawn)) != len(drawn):
        raise ValueError("matrix and qualification seed groups must be disjoint")
    return MappingProxyType({group: seeds for group, seeds in collected.items()})


CANONICAL_SEED_GROUPS: Mapping[str, tuple[int, ...]] = _frozen_seed_groups()


def canonical_seeds(seed_group_id: str) -> tuple[int, ...]:
    """Return the sealed numeric seed tuple of one seed group."""

    try:
        return CANONICAL_SEED_GROUPS[seed_group_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown seed group: {seed_group_id!r}") from error


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


def _optional_artifact_path(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    text = _nonempty(value, field=field)
    if not Path(text).is_absolute():
        raise ValueError(f"{field} must be an absolute immutable artifact path")
    return text


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


def _seeds(value: Any, *, seed_group_id: str, field: str) -> tuple[int, ...]:
    expected = canonical_seeds(seed_group_id)
    if value is None:
        return expected
    seeds = tuple(
        int(item) if not isinstance(item, bool) else -1 for item in tuple(value)
    )
    if seeds != expected:
        raise ValueError(
            f"{field} must be exactly the sealed seed tuple of {seed_group_id}"
        )
    return seeds


def _image_ids(value: Any, *, field: str) -> tuple[int, ...]:
    ids = tuple(int(item) for item in value)
    if len(ids) != len(CANONICAL_IMAGE_IDS) or set(ids) != _CANONICAL_IMAGE_ID_SET:
        raise ValueError(f"{field} must cover the sealed thirteen-image panel exactly")
    return ids


def _objective_component_hashes(
    value: Any, *, arm_id: str, field: str
) -> tuple[tuple[str, str], ...]:
    items = tuple((str(name), digest) for name, digest in tuple(value))
    if tuple(name for name, _ in items) != PROPOSAL_COMPONENTS_BY_ARM[arm_id]:
        raise ValueError(
            f"{field} must bind exactly the canonical proposal components of this arm"
        )
    return tuple(
        (name, _digest(digest, field=f"{field}[{name}]")) for name, digest in items
    )


@dataclass(frozen=True)
class AcquisitionKey:
    """One ``(training RP, seed group, phase)`` acquisition identity.

    The sealed numeric seed tuple is part of the identity, so a relabelled or
    reused acquisition cannot pass as another seed group.
    """

    training_rp: float
    seed_group_id: str
    phase: str
    seeds: tuple[int, ...] | None = None

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
        seed_group_id = _nonempty(self.seed_group_id, field="seed_group_id")
        object.__setattr__(self, "seed_group_id", seed_group_id)
        object.__setattr__(
            self,
            "seeds",
            _seeds(self.seeds, seed_group_id=seed_group_id, field="acquisition seeds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACQUISITION_KEY_SCHEMA,
            "training_rp": self.training_rp,
            "seed_group_id": self.seed_group_id,
            "phase": self.phase,
            "seeds": list(self.seeds or ()),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionKey":
        if value.get("schema_version") != ACQUISITION_KEY_SCHEMA:
            raise ValueError("acquisition key schema_version differs")
        if "seeds" not in value:
            raise ValueError("acquisition key payload must bind its sealed seed tuple")
        return cls(
            training_rp=value["training_rp"],
            seed_group_id=value["seed_group_id"],
            phase=value["phase"],
            seeds=tuple(value["seeds"]),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CellKey:
    """One arm, plus the exact dose for qualification proposals only."""

    acquisition_key: AcquisitionKey
    arm_id: str
    qualification_learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.acquisition_key, AcquisitionKey):
            raise ValueError("cell key acquisition_key must be an AcquisitionKey")
        arm_id = _nonempty(self.arm_id, field="arm_id")
        if arm_id not in ARM_IDS:
            raise ValueError("arm_id must be exactly A, B, or C")
        object.__setattr__(self, "arm_id", arm_id)
        dose = self.qualification_learning_rate
        if self.acquisition_key.phase == PHASE_QUALIFICATION:
            if arm_id != "C":
                raise ValueError("qualification cell keys may bind only arm C")
            if (
                isinstance(dose, bool)
                or not isinstance(dose, (int, float))
                or float(dose) not in QUALIFICATION_LEARNING_RATE_RAY
            ):
                raise ValueError(
                    "qualification cell key must bind one exact dose-ray point"
                )
            object.__setattr__(self, "qualification_learning_rate", float(dose))
        elif dose is not None:
            raise ValueError("matrix cell keys must not carry a qualification dose")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_KEY_SCHEMA,
            "acquisition_key": self.acquisition_key.to_dict(),
            "arm_id": self.arm_id,
            "qualification_learning_rate": self.qualification_learning_rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellKey":
        if value.get("schema_version") != CELL_KEY_SCHEMA:
            raise ValueError("cell key schema_version differs")
        return cls(
            acquisition_key=AcquisitionKey.from_dict(value["acquisition_key"]),
            arm_id=value["arm_id"],
            qualification_learning_rate=value.get("qualification_learning_rate"),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class SharedEvidenceRef:
    """Byte-identical acquisition/matching/credit/policy identity for one acquisition key.

    The record self-describes the acquisition it came from -- its training RP,
    seed group, sealed seed tuple, and collector receipts -- so admission can
    reject evidence that was actually drawn for another acquisition.
    """

    source_sha256: str
    manifest_sha256: str
    acquisition_path: str
    acquisition_sha256: str
    trajectory_credit_acquisition_sha256: str
    credit_ledger_sha256: str
    compiler_ledger_sha256: str
    policy_contract_sha256: str
    native_receipts_sha256: str
    training_rp: float
    seed_group_id: str
    seeds: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "source_sha256",
            "manifest_sha256",
            "acquisition_sha256",
            "trajectory_credit_acquisition_sha256",
            "credit_ledger_sha256",
            "compiler_ledger_sha256",
            "policy_contract_sha256",
            "native_receipts_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "acquisition_path",
            _nonempty(self.acquisition_path, field="acquisition_path"),
        )
        object.__setattr__(self, "training_rp", _training_rp(self.training_rp))
        seed_group_id = _nonempty(self.seed_group_id, field="seed_group_id")
        if seed_group_id not in CANONICAL_SEED_GROUPS:
            raise ValueError(
                "shared evidence seed_group_id is outside the sealed groups"
            )
        object.__setattr__(self, "seed_group_id", seed_group_id)
        object.__setattr__(
            self,
            "seeds",
            _seeds(
                self.seeds,
                seed_group_id=seed_group_id,
                field="shared evidence seeds",
            ),
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
            "native_receipts_sha256": self.native_receipts_sha256,
            "training_rp": self.training_rp,
            "seed_group_id": self.seed_group_id,
            "seeds": list(self.seeds or ()),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedEvidenceRef":
        if value.get("schema_version") != SHARED_EVIDENCE_SCHEMA:
            raise ValueError("shared evidence schema_version differs")
        if "seeds" not in value:
            raise ValueError("shared evidence payload must bind its sealed seed tuple")
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
            native_receipts_sha256=value["native_receipts_sha256"],
            training_rp=value["training_rp"],
            seed_group_id=value["seed_group_id"],
            seeds=tuple(value["seeds"]),
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
    """One planned cell: its evidence, leaf config, and exact single-update contract.

    ``adamw_config_sha256`` is the one declared optimizer configuration shared
    by every cell; ``fresh_optimizer_identity_sha256`` is this cell's own fresh
    optimizer/proposal identity and is never shared.
    """

    cell_key: CellKey
    shared_evidence: SharedEvidenceRef
    leaf_config_sha256: str
    source_checkpoint_sha256: str
    expected_objective_components: tuple[str, ...]
    objective_component_hashes: tuple[tuple[str, str], ...]
    adamw_config_sha256: str
    fresh_optimizer_identity_sha256: str
    evaluation_rps: tuple[float, ...]
    output_root: str
    max_updates: int = 1
    retry_policy: str = "none"
    learning_rate: float = 3.0e-6
    global_learning_rate_decision_sha256: str | None = None
    resolved_leaf_config_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell_key, CellKey):
            raise ValueError("cell spec cell_key must be a CellKey")
        if not isinstance(self.shared_evidence, SharedEvidenceRef):
            raise ValueError("cell spec shared_evidence must be a SharedEvidenceRef")
        acquisition = self.cell_key.acquisition_key
        if self.shared_evidence.training_rp != acquisition.training_rp:
            raise ValueError(
                "cell shared evidence training RP differs from its acquisition key"
            )
        if (
            self.shared_evidence.seed_group_id != acquisition.seed_group_id
            or self.shared_evidence.seeds != acquisition.seeds
        ):
            raise ValueError(
                "cell shared evidence seed group differs from its acquisition key"
            )
        for field in ("leaf_config_sha256", "adamw_config_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "fresh_optimizer_identity_sha256",
            _digest(
                self.fresh_optimizer_identity_sha256,
                field="fresh_optimizer_identity_sha256",
            ),
        )
        object.__setattr__(
            self,
            "source_checkpoint_sha256",
            _digest(self.source_checkpoint_sha256, field="source_checkpoint_sha256"),
        )
        if self.source_checkpoint_sha256 != self.shared_evidence.source_sha256:
            raise ValueError(
                "cell Source checkpoint differs from its shared acquisition evidence"
            )
        components = tuple(self.expected_objective_components)
        if components != _ARM_OBJECTIVE_COMPONENTS[self.cell_key.arm_id]:
            raise ValueError(
                "expected_objective_components differs from the canonical arm-nested surface"
            )
        object.__setattr__(self, "expected_objective_components", components)
        object.__setattr__(
            self,
            "objective_component_hashes",
            _objective_component_hashes(
                self.objective_component_hashes,
                arm_id=self.cell_key.arm_id,
                field="objective_component_hashes",
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
        learning_rate = float(self.learning_rate)
        if learning_rate not in QUALIFICATION_LEARNING_RATE_RAY:
            raise ValueError("cell learning_rate is outside the qualification ray")
        object.__setattr__(self, "learning_rate", learning_rate)
        decision = _optional_digest(
            self.global_learning_rate_decision_sha256,
            field="global_learning_rate_decision_sha256",
        )
        resolved_leaf = _optional_digest(
            self.resolved_leaf_config_sha256,
            field="resolved_leaf_config_sha256",
        )
        if acquisition.phase == PHASE_QUALIFICATION:
            if self.cell_key.qualification_learning_rate != learning_rate:
                raise ValueError(
                    "qualification CellSpec dose differs from its cell key"
                )
            if decision is not None:
                raise ValueError(
                    "qualification proposals precede the global learning-rate decision"
                )
        elif (decision is None) != (resolved_leaf is None):
            raise ValueError(
                "matrix selected-LR decision and resolved leaf hash must be paired"
            )
        object.__setattr__(self, "global_learning_rate_decision_sha256", decision)
        object.__setattr__(self, "resolved_leaf_config_sha256", resolved_leaf)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_SPEC_SCHEMA,
            "cell_key": self.cell_key.to_dict(),
            "shared_evidence": self.shared_evidence.to_dict(),
            "leaf_config_sha256": self.leaf_config_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "expected_objective_components": list(self.expected_objective_components),
            "objective_component_hashes": [
                list(item) for item in self.objective_component_hashes
            ],
            "adamw_config_sha256": self.adamw_config_sha256,
            "fresh_optimizer_identity_sha256": self.fresh_optimizer_identity_sha256,
            "evaluation_rps": list(self.evaluation_rps),
            "output_root": self.output_root,
            "max_updates": self.max_updates,
            "retry_policy": self.retry_policy,
            "learning_rate": self.learning_rate,
            "global_learning_rate_decision_sha256": (
                self.global_learning_rate_decision_sha256
            ),
            "resolved_leaf_config_sha256": self.resolved_leaf_config_sha256,
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
            objective_component_hashes=tuple(
                tuple(item) for item in value["objective_component_hashes"]
            ),
            adamw_config_sha256=value["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=value["fresh_optimizer_identity_sha256"],
            evaluation_rps=tuple(value["evaluation_rps"]),
            output_root=value["output_root"],
            max_updates=value["max_updates"],
            retry_policy=value["retry_policy"],
            learning_rate=value.get("learning_rate", 3.0e-6),
            global_learning_rate_decision_sha256=value.get(
                "global_learning_rate_decision_sha256"
            ),
            resolved_leaf_config_sha256=value.get("resolved_leaf_config_sha256"),
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
class AggregateResourceReceipt:
    """Complete measured count/resource envelope for one private proposal."""

    measurement_scope: str
    wall_time_seconds: float
    peak_host_rss_bytes: int
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    acquisition_request_count: int
    acquisition_batch_count: int
    acquisition_token_count: int
    decode_request_count: int
    decode_batch_count: int
    decode_token_count: int
    packed_token_count: int
    logical_token_count: int
    forward_count: int
    backward_count: int
    row_bytes: int
    artifact_bytes: int
    update_count: int
    audit_count: int
    rollback_count: int

    def __post_init__(self) -> None:
        if self.measurement_scope not in {"injected_cpu", "live"}:
            raise ValueError("resource measurement_scope must be injected_cpu or live")
        wall = float(self.wall_time_seconds)
        if not math.isfinite(wall) or wall < 0.0:
            raise ValueError("resource wall time must be finite and nonnegative")
        object.__setattr__(self, "wall_time_seconds", wall)
        integer_fields = (
            "peak_host_rss_bytes",
            "acquisition_request_count",
            "acquisition_batch_count",
            "acquisition_token_count",
            "decode_request_count",
            "decode_batch_count",
            "decode_token_count",
            "packed_token_count",
            "logical_token_count",
            "forward_count",
            "backward_count",
            "row_bytes",
            "artifact_bytes",
            "update_count",
            "audit_count",
            "rollback_count",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resource {field} must be a nonnegative integer")
        cuda_values = (
            self.cuda_peak_allocated_bytes,
            self.cuda_peak_reserved_bytes,
        )
        if (cuda_values[0] is None) != (cuda_values[1] is None):
            raise ValueError("CUDA allocated/reserved measurements must be paired")
        for value in cuda_values:
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("CUDA resource bytes must be nonnegative integers")
        if self.backward_count <= 0 or (
            self.update_count,
            self.audit_count,
            self.rollback_count,
        ) != (1, 2, 1):
            raise ValueError(
                "proposal resources require positive backward count, one update, "
                "two audits, and one rollback"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGGREGATE_RESOURCE_RECEIPT_SCHEMA,
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AggregateResourceReceipt":
        if value.get("schema_version") != AGGREGATE_RESOURCE_RECEIPT_SCHEMA:
            raise ValueError("aggregate resource receipt schema_version differs")
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class DoseMechanicalReceipt:
    """Owner-outcome-free mechanics used by the pure global dose selector."""

    cell_key: CellKey
    proposal_sha256: str
    private_checkpoint_sha256: str
    audit_checkpoint_sha256s: tuple[str, str]
    greedy_decision_change_count: int
    malformed_output_delta_count: int
    cap_terminated_output_delta_count: int
    unparseable_output_delta_count: int
    active_witness_count: int
    jvp_fd_max_abs_error: float
    jvp_fd_tolerance: float
    median_abs_decision_margin_displacement: float
    median_abs_source_decision_margin: float
    rollback_reproduced: bool
    resources: AggregateResourceReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_key, CellKey)
            or self.cell_key.acquisition_key.phase != PHASE_QUALIFICATION
            or self.cell_key.arm_id != "C"
        ):
            raise ValueError("dose mechanics require one qualification C cell key")
        for field in ("proposal_sha256", "private_checkpoint_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        audits = tuple(
            _digest(value, field="audit_checkpoint_sha256")
            for value in self.audit_checkpoint_sha256s
        )
        if len(audits) != 2 or audits != (self.private_checkpoint_sha256,) * 2:
            raise ValueError("both dose audits must use one private checkpoint")
        object.__setattr__(self, "audit_checkpoint_sha256s", audits)
        for field in (
            "greedy_decision_change_count",
            "malformed_output_delta_count",
            "cap_terminated_output_delta_count",
            "unparseable_output_delta_count",
            "active_witness_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in (
            "jvp_fd_max_abs_error",
            "jvp_fd_tolerance",
            "median_abs_decision_margin_displacement",
            "median_abs_source_decision_margin",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and nonnegative")
            object.__setattr__(self, field, value)
        if self.jvp_fd_tolerance <= 0.0:
            raise ValueError("JVP/finite-difference tolerance must be positive")
        if self.rollback_reproduced is not True:
            raise ValueError("dose mechanics require exact rollback reproduction")
        if not isinstance(self.resources, AggregateResourceReceipt):
            raise ValueError("dose mechanics require aggregate resource evidence")

    @property
    def learning_rate(self) -> float:
        value = self.cell_key.qualification_learning_rate
        assert value is not None
        return value

    @property
    def floor_passed(self) -> bool:
        return self.greedy_decision_change_count > 0 and self.active_witness_count > 0

    @property
    def ceiling_passed(self) -> bool:
        return (
            self.malformed_output_delta_count == 0
            and self.cap_terminated_output_delta_count == 0
            and self.unparseable_output_delta_count == 0
            and self.jvp_fd_max_abs_error <= self.jvp_fd_tolerance
            and self.median_abs_decision_margin_displacement
            <= self.median_abs_source_decision_margin
        )

    @property
    def mechanically_admissible(self) -> bool:
        return self.floor_passed and self.ceiling_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DOSE_MECHANICAL_RECEIPT_SCHEMA,
            "cell_key": self.cell_key.to_dict(),
            "proposal_sha256": self.proposal_sha256,
            "private_checkpoint_sha256": self.private_checkpoint_sha256,
            "audit_checkpoint_sha256s": list(self.audit_checkpoint_sha256s),
            "greedy_decision_change_count": self.greedy_decision_change_count,
            "malformed_output_delta_count": self.malformed_output_delta_count,
            "cap_terminated_output_delta_count": self.cap_terminated_output_delta_count,
            "unparseable_output_delta_count": self.unparseable_output_delta_count,
            "active_witness_count": self.active_witness_count,
            "jvp_fd_max_abs_error": self.jvp_fd_max_abs_error,
            "jvp_fd_tolerance": self.jvp_fd_tolerance,
            "median_abs_decision_margin_displacement": (
                self.median_abs_decision_margin_displacement
            ),
            "median_abs_source_decision_margin": self.median_abs_source_decision_margin,
            "rollback_reproduced": self.rollback_reproduced,
            "resources": self.resources.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoseMechanicalReceipt":
        if value.get("schema_version") != DOSE_MECHANICAL_RECEIPT_SCHEMA:
            raise ValueError("dose mechanical receipt schema_version differs")
        return cls(
            cell_key=CellKey.from_dict(value["cell_key"]),
            proposal_sha256=value["proposal_sha256"],
            private_checkpoint_sha256=value["private_checkpoint_sha256"],
            audit_checkpoint_sha256s=tuple(value["audit_checkpoint_sha256s"]),
            greedy_decision_change_count=value["greedy_decision_change_count"],
            malformed_output_delta_count=value["malformed_output_delta_count"],
            cap_terminated_output_delta_count=value[
                "cap_terminated_output_delta_count"
            ],
            unparseable_output_delta_count=value["unparseable_output_delta_count"],
            active_witness_count=value["active_witness_count"],
            jvp_fd_max_abs_error=value["jvp_fd_max_abs_error"],
            jvp_fd_tolerance=value["jvp_fd_tolerance"],
            median_abs_decision_margin_displacement=value[
                "median_abs_decision_margin_displacement"
            ],
            median_abs_source_decision_margin=value[
                "median_abs_source_decision_margin"
            ],
            rollback_reproduced=value["rollback_reproduced"],
            resources=AggregateResourceReceipt.from_dict(value["resources"]),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CellReceipt:
    """The immutable outcome of one cell: proposal, audits, transaction, and status.

    ``transaction_id`` is this cell's own transaction identity.  The
    before/after state digests are *content* digests of the fresh Source state,
    so they legitimately coincide across cells and never carry identity.
    ``proposal_delta_sha256`` is the arm-independent identity of the measured
    AdamW proposal, which is what arm C projects.
    """

    cell_key: CellKey
    shared_evidence: SharedEvidenceRef
    objective_components: tuple[str, ...]
    adamw_config_sha256: str
    fresh_optimizer_identity_sha256: str
    transaction_id: str
    before_transaction_digest: str
    after_transaction_digest: str
    status: str
    objective_component_hashes: tuple[tuple[str, str], ...] = ()
    audits: tuple[AuditRef, ...] = ()
    adamw_proposal_sha256: str | None = None
    proposal_delta_sha256: str | None = None
    projection_receipt_sha256: str | None = None
    apply_receipt_sha256: str | None = None
    adamw_proposal_artifact_path: str | None = None
    witness_bank_artifact_path: str | None = None
    witness_bank_sha256: str | None = None
    projection_receipt_artifact_path: str | None = None
    apply_receipt_artifact_path: str | None = None
    update_count: int = 1
    retry_policy: str = "none"
    rollback_confirmed: bool = True
    failure_reason: str | None = None
    learning_rate: float = 3.0e-6
    global_learning_rate_decision_sha256: str | None = None
    resolved_leaf_config_sha256: str | None = None
    aggregate_resources: AggregateResourceReceipt | None = None
    dose_mechanics: DoseMechanicalReceipt | None = None

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
        for field in ("adamw_config_sha256", "fresh_optimizer_identity_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        transaction_id = _nonempty(self.transaction_id, field="transaction_id")
        if not _TRANSACTION_ID.fullmatch(transaction_id):
            raise ValueError("transaction_id must be a 32-character hex identity")
        object.__setattr__(self, "transaction_id", transaction_id)
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
        learning_rate = float(self.learning_rate)
        if learning_rate not in QUALIFICATION_LEARNING_RATE_RAY:
            raise ValueError("cell receipt learning_rate is outside the sealed ray")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(
            self,
            "global_learning_rate_decision_sha256",
            _optional_digest(
                self.global_learning_rate_decision_sha256,
                field="global_learning_rate_decision_sha256",
            ),
        )
        object.__setattr__(
            self,
            "resolved_leaf_config_sha256",
            _optional_digest(
                self.resolved_leaf_config_sha256,
                field="resolved_leaf_config_sha256",
            ),
        )
        if self.aggregate_resources is not None and not isinstance(
            self.aggregate_resources, AggregateResourceReceipt
        ):
            raise ValueError("cell aggregate_resources must be a typed receipt")
        if self.dose_mechanics is not None:
            if not isinstance(self.dose_mechanics, DoseMechanicalReceipt):
                raise ValueError("cell dose_mechanics must be a typed receipt")
            if self.dose_mechanics.cell_key != self.cell_key:
                raise ValueError("dose mechanics differ from the cell receipt key")
            if self.dose_mechanics.resources != self.aggregate_resources:
                raise ValueError("dose mechanics and cell resources differ")

        requires_projection = "preservation" in components
        for field in (
            "adamw_proposal_sha256",
            "proposal_delta_sha256",
            "apply_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _optional_digest(getattr(self, field), field=field)
            )
        projection = _optional_digest(
            self.projection_receipt_sha256, field="projection_receipt_sha256"
        )
        if not requires_projection and projection is not None:
            raise ValueError(
                "only the preservation arm (C) may bind a projection receipt"
            )
        object.__setattr__(self, "projection_receipt_sha256", projection)
        proposal_path = _optional_artifact_path(
            self.adamw_proposal_artifact_path,
            field="adamw_proposal_artifact_path",
        )
        witness_path = _optional_artifact_path(
            self.witness_bank_artifact_path,
            field="witness_bank_artifact_path",
        )
        witness_sha256 = _optional_digest(
            self.witness_bank_sha256, field="witness_bank_sha256"
        )
        projection_path = _optional_artifact_path(
            self.projection_receipt_artifact_path,
            field="projection_receipt_artifact_path",
        )
        apply_path = _optional_artifact_path(
            self.apply_receipt_artifact_path,
            field="apply_receipt_artifact_path",
        )
        if (proposal_path is None) != (self.adamw_proposal_sha256 is None):
            raise ValueError(
                "exact AdamW proposal artifact path/hash evidence must be paired"
            )
        if (witness_path is None) != (witness_sha256 is None):
            raise ValueError("witness-bank artifact path/hash evidence must be paired")
        if (projection_path is None) != (projection is None):
            raise ValueError(
                "projection receipt artifact path/hash evidence must be paired"
            )
        if requires_projection:
            if (apply_path is None) != (self.apply_receipt_sha256 is None):
                raise ValueError(
                    "projected-apply artifact path/hash evidence must be paired"
                )
        elif any(
            value is not None
            for value in (witness_path, witness_sha256, projection_path, apply_path)
        ):
            raise ValueError(
                "only preservation-arm receipts may bind witness/projection artifacts"
            )
        for path, digest, field_name, directory in (
            (
                proposal_path,
                self.adamw_proposal_sha256,
                "adamw_proposal_artifact_path",
                False,
            ),
            (witness_path, witness_sha256, "witness_bank_artifact_path", True),
            (
                projection_path,
                projection,
                "projection_receipt_artifact_path",
                False,
            ),
            (
                apply_path,
                self.apply_receipt_sha256,
                "apply_receipt_artifact_path",
                False,
            ),
        ):
            if path is None or digest is None:
                continue
            expected_name = digest if directory else f"{digest}.json"
            if Path(path).name != expected_name:
                raise ValueError(
                    f"{field_name} does not carry its content-addressed hash"
                )
        object.__setattr__(self, "adamw_proposal_artifact_path", proposal_path)
        object.__setattr__(self, "witness_bank_artifact_path", witness_path)
        object.__setattr__(self, "witness_bank_sha256", witness_sha256)
        object.__setattr__(self, "projection_receipt_artifact_path", projection_path)
        object.__setattr__(self, "apply_receipt_artifact_path", apply_path)

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
            object.__setattr__(
                self,
                "objective_component_hashes",
                _objective_component_hashes(
                    self.objective_component_hashes,
                    arm_id=self.cell_key.arm_id,
                    field="objective_component_hashes",
                ),
            )
            if self.failure_reason is not None:
                raise ValueError(
                    "a succeeded cell receipt must not carry a failure_reason"
                )
            if self.adamw_proposal_sha256 is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its exact AdamW proposal identity"
                )
            if self.adamw_proposal_artifact_path is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its exact proposal artifact"
                )
            if self.proposal_delta_sha256 is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its measured proposal delta identity"
                )
            if self.apply_receipt_sha256 is None:
                raise ValueError(
                    "a succeeded cell receipt must bind its apply evidence"
                )
            if requires_projection and self.projection_receipt_sha256 is None:
                raise ValueError(
                    "a succeeded preservation-arm receipt must bind its projection evidence"
                )
            if requires_projection and any(
                value is None
                for value in (
                    self.witness_bank_artifact_path,
                    self.witness_bank_sha256,
                    self.projection_receipt_artifact_path,
                    self.apply_receipt_artifact_path,
                )
            ):
                raise ValueError(
                    "a succeeded preservation arm must bind all decision artifacts"
                )
            if set(audit_rps) != set(EVALUATION_RPS):
                raise ValueError(
                    "a succeeded cell receipt must bind exactly one audit per evaluation RP"
                )
        else:
            hashes = tuple(tuple(item) for item in self.objective_component_hashes)
            if hashes:
                hashes = _objective_component_hashes(
                    hashes,
                    arm_id=self.cell_key.arm_id,
                    field="objective_component_hashes",
                )
            object.__setattr__(self, "objective_component_hashes", hashes)
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
            "objective_component_hashes": [
                list(item) for item in self.objective_component_hashes
            ],
            "adamw_config_sha256": self.adamw_config_sha256,
            "fresh_optimizer_identity_sha256": self.fresh_optimizer_identity_sha256,
            "transaction_id": self.transaction_id,
            "before_transaction_digest": self.before_transaction_digest,
            "after_transaction_digest": self.after_transaction_digest,
            "status": self.status,
            "audits": [audit.to_dict() for audit in self.audits],
            "adamw_proposal_sha256": self.adamw_proposal_sha256,
            "proposal_delta_sha256": self.proposal_delta_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "apply_receipt_sha256": self.apply_receipt_sha256,
            "adamw_proposal_artifact_path": self.adamw_proposal_artifact_path,
            "witness_bank_artifact_path": self.witness_bank_artifact_path,
            "witness_bank_sha256": self.witness_bank_sha256,
            "projection_receipt_artifact_path": (self.projection_receipt_artifact_path),
            "apply_receipt_artifact_path": self.apply_receipt_artifact_path,
            "update_count": self.update_count,
            "retry_policy": self.retry_policy,
            "rollback_confirmed": self.rollback_confirmed,
            "failure_reason": self.failure_reason,
            "learning_rate": self.learning_rate,
            "global_learning_rate_decision_sha256": (
                self.global_learning_rate_decision_sha256
            ),
            "resolved_leaf_config_sha256": self.resolved_leaf_config_sha256,
            "aggregate_resources": (
                None
                if self.aggregate_resources is None
                else self.aggregate_resources.to_dict()
            ),
            "dose_mechanics": (
                None if self.dose_mechanics is None else self.dose_mechanics.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellReceipt":
        if value.get("schema_version") != CELL_RECEIPT_SCHEMA:
            raise ValueError("cell receipt schema_version differs")
        return cls(
            cell_key=CellKey.from_dict(value["cell_key"]),
            shared_evidence=SharedEvidenceRef.from_dict(value["shared_evidence"]),
            objective_components=tuple(value["objective_components"]),
            objective_component_hashes=tuple(
                tuple(item) for item in value["objective_component_hashes"]
            ),
            adamw_config_sha256=value["adamw_config_sha256"],
            fresh_optimizer_identity_sha256=value["fresh_optimizer_identity_sha256"],
            transaction_id=value["transaction_id"],
            before_transaction_digest=value["before_transaction_digest"],
            after_transaction_digest=value["after_transaction_digest"],
            status=value["status"],
            audits=tuple(AuditRef.from_dict(item) for item in value["audits"]),
            adamw_proposal_sha256=value["adamw_proposal_sha256"],
            proposal_delta_sha256=value["proposal_delta_sha256"],
            projection_receipt_sha256=value["projection_receipt_sha256"],
            apply_receipt_sha256=value["apply_receipt_sha256"],
            adamw_proposal_artifact_path=value["adamw_proposal_artifact_path"],
            witness_bank_artifact_path=value["witness_bank_artifact_path"],
            witness_bank_sha256=value["witness_bank_sha256"],
            projection_receipt_artifact_path=value["projection_receipt_artifact_path"],
            apply_receipt_artifact_path=value["apply_receipt_artifact_path"],
            update_count=value["update_count"],
            retry_policy=value["retry_policy"],
            rollback_confirmed=value["rollback_confirmed"],
            failure_reason=value["failure_reason"],
            learning_rate=value.get("learning_rate", 3.0e-6),
            global_learning_rate_decision_sha256=value.get(
                "global_learning_rate_decision_sha256"
            ),
            resolved_leaf_config_sha256=value.get("resolved_leaf_config_sha256"),
            aggregate_resources=(
                None
                if value.get("aggregate_resources") is None
                else AggregateResourceReceipt.from_dict(value["aggregate_resources"])
            ),
            dose_mechanics=(
                None
                if value.get("dose_mechanics") is None
                else DoseMechanicalReceipt.from_dict(value["dose_mechanics"])
            ),
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


def matrix_identity(cell: CellSpec | CellReceipt) -> tuple[float, str, str]:
    """The ``(training RP, seed group, arm)`` identity of one matrix cell."""

    acquisition = cell.cell_key.acquisition_key
    return acquisition.training_rp, acquisition.seed_group_id, cell.cell_key.arm_id


def _validate_group_objective_identity(arm_map: Mapping[str, CellSpec]) -> None:
    hashes = {arm: dict(arm_map[arm].objective_component_hashes) for arm in arm_map}
    trajectories = {value["trajectory"] for value in hashes.values()}
    if len(trajectories) != 1:
        raise ValueError(
            "arms under one acquisition must share one trajectory objective identity"
        )
    compilers = {value["compiler"] for value in hashes.values() if "compiler" in value}
    if len(compilers) > 1:
        raise ValueError(
            "arms B and C under one acquisition must share one compiler objective identity"
        )


def validate_node_cell_specs(
    acquisition_key: AcquisitionKey, phase: str, specs: Sequence[CellSpec]
) -> tuple[CellSpec, ...]:
    """Admit the exact cell specs one acquisition node may own.

    The node runner and the durable node terminal share this choke point, so a
    node cannot execute a cell set that its own terminal would reject.
    """

    bound = tuple(specs)
    if any(not isinstance(item, CellSpec) for item in bound):
        raise ValueError("node cell specs must be CellSpec records")
    expected_arms = ARM_IDS if phase == PHASE_MATRIX else ("C",) * 5
    if tuple(spec.cell_key.arm_id for spec in bound) != expected_arms:
        raise ValueError(
            "node cell specs must bind every planned arm/dose of its phase"
        )
    if (
        phase == PHASE_QUALIFICATION
        and tuple(spec.learning_rate for spec in bound)
        != QUALIFICATION_LEARNING_RATE_RAY
    ):
        raise ValueError("qualification node must bind the exact five-dose ray")
    for spec in bound:
        if spec.cell_key.acquisition_key != acquisition_key:
            raise ValueError("every node CellSpec must bind this acquisition key")
    if len({spec.shared_evidence for spec in bound}) != 1:
        raise ValueError(
            "one node acquires one byte-identical shared evidence identity"
        )
    if len({spec.source_checkpoint_sha256 for spec in bound}) != 1:
        raise ValueError("one node begins from one Source checkpoint identity")
    if phase == PHASE_MATRIX and len({spec.adamw_config_sha256 for spec in bound}) != 1:
        raise ValueError(
            "every cell must bind one declared AdamW configuration identity"
        )
    identities = [spec.fresh_optimizer_identity_sha256 for spec in bound]
    if len(set(identities)) != len(identities):
        raise ValueError("every cell must have an independent fresh optimizer identity")
    if phase == PHASE_MATRIX:
        decisions = {spec.global_learning_rate_decision_sha256 for spec in bound}
        if None in decisions or len(decisions) != 1:
            raise ValueError(
                "matrix cells require one selected global learning-rate decision"
            )
        if len({spec.learning_rate for spec in bound}) != 1:
            raise ValueError("matrix cells must consume one selected learning rate")
        _validate_group_objective_identity(
            {spec.cell_key.arm_id: spec for spec in bound}
        )
    elif len({spec.objective_component_hashes for spec in bound}) != 1:
        raise ValueError("qualification doses must reuse identical C objective bytes")
    return bound


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
    optimizer_identities: set[str] = set()
    config_by_surface: dict[tuple[float, str], set[str]] = {}
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
        if cell.fresh_optimizer_identity_sha256 in optimizer_identities:
            raise ValueError(
                "every cell must have an independent fresh optimizer identity"
            )
        optimizer_identities.add(cell.fresh_optimizer_identity_sha256)
        config_by_surface.setdefault(
            (acquisition.training_rp, cell.cell_key.arm_id), set()
        ).add(cell.leaf_config_sha256)

    if len(cells_by_acquisition) != len(acquisitions):
        raise ValueError("every bound acquisition must own exactly one cell group")
    group_evidence: list[SharedEvidenceRef] = []
    for arm_map in cells_by_acquisition.values():
        if set(arm_map) != set(ARM_IDS):
            raise ValueError("every acquisition must bind exactly arms A, B, and C")
        shared_refs = {arm_map[arm].shared_evidence for arm in ARM_IDS}
        if len(shared_refs) != 1:
            raise ValueError(
                "arms A, B, and C under one acquisition key must share byte-identical evidence"
            )
        group_evidence.append(arm_map["A"].shared_evidence)
        _validate_group_objective_identity(arm_map)

    for field in (
        "acquisition_path",
        "acquisition_sha256",
        "trajectory_credit_acquisition_sha256",
        "credit_ledger_sha256",
        "native_receipts_sha256",
    ):
        values = [getattr(ref, field) for ref in group_evidence]
        if len(set(values)) != len(values):
            raise ValueError(
                "matrix seed groups must have distinct acquisition and credit identities"
            )
    if (
        len({ref.source_sha256 for ref in group_evidence}) != 1
        or len({ref.manifest_sha256 for ref in group_evidence}) != 1
    ):
        raise ValueError("matrix Source or manifest lineage is mixed")
    policy_by_rp: dict[float, set[str]] = {}
    for ref in group_evidence:
        policy_by_rp.setdefault(ref.training_rp, set()).add(ref.policy_contract_sha256)
    if any(len(values) != 1 for values in policy_by_rp.values()) or len(
        {next(iter(values)) for values in policy_by_rp.values()}
    ) != len(TRAINING_RPS):
        raise ValueError("training-RP policy contract lineage is mixed")

    if (
        len(config_by_surface) != len(TRAINING_RPS) * len(ARM_IDS)
        or any(len(values) != 1 for values in config_by_surface.values())
        or len({next(iter(values)) for values in config_by_surface.values()})
        != len(config_by_surface)
    ):
        raise ValueError("leaf config lineage is mixed")
    if len({cell.adamw_config_sha256 for cell in cells}) != 1:
        raise ValueError(
            "every cell must bind one declared AdamW configuration identity"
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


def validate_matrix_receipts(
    plan: MatrixPlan, receipts: Sequence[CellReceipt]
) -> dict[tuple[float, str, str], CellReceipt]:
    """The one aggregate admission choke point for a complete matrix outcome.

    Every supported path -- the node terminal aggregate publisher, direct
    analysis-input construction, and reload through ``from_dict`` -- reaches
    this validator, so no path can pool a mixed, reused, or nested-objective
    divergent outcome.
    """

    if not isinstance(plan, MatrixPlan):
        raise ValueError("matrix receipt admission requires a MatrixPlan")
    bound = tuple(receipts)
    if any(not isinstance(item, CellReceipt) for item in bound):
        raise ValueError("matrix receipts must be CellReceipt records")
    if any(item.cell_key.acquisition_key.phase != PHASE_MATRIX for item in bound):
        raise ValueError("qualification cells must never enter matrix analysis")
    if len(bound) != len(plan.cells):
        raise ValueError("analysis requires exactly eighteen cell receipts")
    keys = [matrix_identity(item) for item in bound]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "analysis requires a unique cell receipt for every matrix cell"
        )
    planned = {matrix_identity(cell): cell for cell in plan.cells}
    if set(keys) != set(planned):
        raise ValueError("cell receipts differ from the exact eighteen-cell plan")

    admitted: dict[tuple[float, str, str], CellReceipt] = {}
    proposal_ids: set[str] = set()
    apply_ids: set[str] = set()
    transaction_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    audit_paths: set[str] = set()
    generation_ids: set[str] = set()
    for receipt in bound:
        key = matrix_identity(receipt)
        cell = planned[key]
        if receipt.status != "succeeded":
            raise ValueError("failed scientific cell cannot enter pooled success")
        if receipt.shared_evidence != cell.shared_evidence:
            raise ValueError("cell receipt shared evidence differs from its plan")
        if receipt.objective_components != cell.expected_objective_components:
            raise ValueError("cell receipt objective differs from its plan")
        if receipt.objective_component_hashes != cell.objective_component_hashes:
            raise ValueError(
                "cell receipt objective component bytes differ from its plan"
            )
        if receipt.adamw_config_sha256 != cell.adamw_config_sha256:
            raise ValueError(
                "cell receipt AdamW configuration identity differs from its plan"
            )
        if (
            receipt.fresh_optimizer_identity_sha256
            != cell.fresh_optimizer_identity_sha256
        ):
            raise ValueError("cell receipt optimizer identity differs from its plan")
        if receipt.update_count != 1 or receipt.retry_policy != "none":
            raise ValueError("cell receipt must bind one update and zero retries")
        if not receipt.rollback_confirmed or (
            receipt.before_transaction_digest != receipt.after_transaction_digest
        ):
            raise ValueError("cell receipt lacks complete rollback")
        assert receipt.adamw_proposal_sha256 is not None
        assert receipt.apply_receipt_sha256 is not None
        if receipt.transaction_id in transaction_ids:
            raise ValueError("cells require an independent transaction identity")
        if receipt.adamw_proposal_sha256 in proposal_ids:
            raise ValueError("cells require independent proposal identities")
        if receipt.apply_receipt_sha256 in apply_ids:
            raise ValueError("cells require independent apply identities")
        transaction_ids.add(receipt.transaction_id)
        proposal_ids.add(receipt.adamw_proposal_sha256)
        apply_ids.add(receipt.apply_receipt_sha256)
        audits = {audit.evaluation_rp: audit for audit in receipt.audits}
        if set(audits) != set(EVALUATION_RPS) or len(receipt.audits) != 2:
            raise ValueError("successful cells require exactly two clean-greedy audits")
        evaluated = {audit.evaluated_checkpoint_sha256 for audit in receipt.audits}
        if len(evaluated) != 1:
            raise ValueError("both audits must evaluate the same private proposal")
        checkpoint = evaluated.pop()
        if checkpoint in checkpoint_ids:
            raise ValueError("cells require independent evaluated proposal identities")
        checkpoint_ids.add(checkpoint)
        for audit in receipt.audits:
            if audit.output_path in audit_paths:
                raise ValueError("audit output paths must be unique")
            if audit.generation_policy_receipt_sha256 in generation_ids:
                raise ValueError("audit generation-policy receipts must be unique")
            audit_paths.add(audit.output_path)
            generation_ids.add(audit.generation_policy_receipt_sha256)
        admitted[key] = receipt

    # Arm C projects the arm-B proposal, so their measured deltas must be the
    # same bytes.  Deltas are deliberately not compared across arms A/B or
    # across acquisitions: a single fresh-AdamW step is sign-dominated, so
    # distinctness there is a statistical accident, not a contract.  Arm
    # separation is carried by the objective component bytes above.
    for training_rp in TRAINING_RPS:
        for seed_group in MATRIX_SEED_GROUPS:
            base = admitted[(training_rp, seed_group, "B")].proposal_delta_sha256
            if admitted[(training_rp, seed_group, "C")].proposal_delta_sha256 != base:
                raise ValueError(
                    "arm C must project the exact admitted arm-B proposal of its acquisition"
                )
    return admitted


@dataclass(frozen=True)
class NodeTerminalReceipt:
    """One acquisition node's terminal: its exact acquired specs and outcomes.

    The terminal is the only durable hand-off between the node runner and the
    aggregate publisher, so it persists the typed ``CellSpec`` records that the
    node actually acquired next to the ``CellReceipt`` records they produced.
    """

    node_id: str
    phase: str
    acquisition_key: AcquisitionKey
    status: str
    cell_specs: tuple[CellSpec, ...] = ()
    cell_receipts: tuple[CellReceipt, ...] = ()
    retry_policy: str = "none"
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.acquisition_key, AcquisitionKey):
            raise ValueError("node terminal acquisition_key must be an AcquisitionKey")
        object.__setattr__(self, "node_id", _nonempty(self.node_id, field="node_id"))
        phase = _nonempty(self.phase, field="phase")
        if phase != self.acquisition_key.phase:
            raise ValueError("node terminal phase differs from its acquisition key")
        object.__setattr__(self, "phase", phase)
        status = _nonempty(self.status, field="status")
        if status not in _STATUSES:
            raise ValueError("status must be exactly 'succeeded' or 'failed'")
        object.__setattr__(self, "status", status)
        if self.retry_policy != "none":
            raise ValueError("retry_policy must be exactly 'none'")

        specs = tuple(self.cell_specs)
        receipts = tuple(self.cell_receipts)
        if any(not isinstance(item, CellSpec) for item in specs):
            raise ValueError("node terminal cell_specs must be CellSpec records")
        if any(not isinstance(item, CellReceipt) for item in receipts):
            raise ValueError("node terminal cell_receipts must be CellReceipt records")
        expected_arms = ARM_IDS if phase == PHASE_MATRIX else ("C",) * 5
        if specs:
            validate_node_cell_specs(self.acquisition_key, phase, specs)
        if len(receipts) > len(specs):
            raise ValueError("node terminal bound more receipts than acquired cells")
        for receipt, spec in zip(receipts, specs, strict=False):
            if receipt.cell_key != spec.cell_key:
                raise ValueError(
                    "node terminal receipts must follow their acquired cell order"
                )
            if receipt.shared_evidence != spec.shared_evidence:
                raise ValueError("node terminal receipt evidence differs from its spec")
            if (
                receipt.status == "succeeded"
                and receipt.objective_component_hashes
                != spec.objective_component_hashes
            ):
                raise ValueError(
                    "node terminal receipt objective component bytes differ from its spec"
                )
            if (
                receipt.adamw_config_sha256 != spec.adamw_config_sha256
                or receipt.fresh_optimizer_identity_sha256
                != spec.fresh_optimizer_identity_sha256
                or receipt.learning_rate != spec.learning_rate
                or receipt.global_learning_rate_decision_sha256
                != spec.global_learning_rate_decision_sha256
                or receipt.resolved_leaf_config_sha256
                != spec.resolved_leaf_config_sha256
            ):
                raise ValueError(
                    "node terminal receipt optimizer identity differs from its spec"
                )
        object.__setattr__(self, "cell_specs", specs)
        object.__setattr__(self, "cell_receipts", receipts)

        if status == "succeeded":
            if self.failure_reason is not None:
                raise ValueError(
                    "a succeeded node terminal must not carry a failure_reason"
                )
            if len(receipts) != len(expected_arms) or len(specs) != len(expected_arms):
                raise ValueError(
                    "a succeeded node terminal must bind every planned arm exactly once"
                )
            if any(receipt.status != "succeeded" for receipt in receipts):
                raise ValueError(
                    "a succeeded node terminal must bind only succeeded cells"
                )
        elif not _nonempty(self.failure_reason or "", field="failure_reason"):
            raise ValueError(
                "a failed node terminal must carry a nonempty failure_reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NODE_TERMINAL_RECEIPT_SCHEMA,
            "node_id": self.node_id,
            "phase": self.phase,
            "acquisition_key": self.acquisition_key.to_dict(),
            "acquisition_key_sha256": self.acquisition_key.content_sha256,
            "status": self.status,
            "cell_specs": [item.to_dict() for item in self.cell_specs],
            "cell_receipts": [item.to_dict() for item in self.cell_receipts],
            "retry_policy": self.retry_policy,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NodeTerminalReceipt":
        if value.get("schema_version") != NODE_TERMINAL_RECEIPT_SCHEMA:
            raise ValueError("node terminal receipt schema_version differs")
        acquisition_key = AcquisitionKey.from_dict(value["acquisition_key"])
        if value.get("acquisition_key_sha256") != acquisition_key.content_sha256:
            raise ValueError("node terminal acquisition identity differs")
        return cls(
            node_id=value["node_id"],
            phase=value["phase"],
            acquisition_key=acquisition_key,
            status=value["status"],
            cell_specs=tuple(CellSpec.from_dict(item) for item in value["cell_specs"]),
            cell_receipts=tuple(
                CellReceipt.from_dict(item) for item in value["cell_receipts"]
            ),
            retry_policy=value["retry_policy"],
            failure_reason=value["failure_reason"],
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict())


__all__ = [
    "ACQUISITION_KEY_SCHEMA",
    "AGGREGATE_RESOURCE_RECEIPT_SCHEMA",
    "ARM_IDS",
    "AUDIT_REF_SCHEMA",
    "AcquisitionKey",
    "AggregateResourceReceipt",
    "AuditRef",
    "CANONICAL_IMAGE_IDS",
    "CANONICAL_SEED_GROUPS",
    "CELL_KEY_SCHEMA",
    "CELL_RECEIPT_SCHEMA",
    "CELL_SPEC_SCHEMA",
    "CellKey",
    "CellReceipt",
    "CellSpec",
    "DRY_RUN_COUNTER_KEYS",
    "DOSE_MECHANICAL_RECEIPT_SCHEMA",
    "DoseMechanicalReceipt",
    "EVALUATION_RPS",
    "MATRIX_PLAN_SCHEMA",
    "MATRIX_SEED_GROUPS",
    "MatrixPlan",
    "NODE_TERMINAL_RECEIPT_SCHEMA",
    "NodeTerminalReceipt",
    "PHASE_MATRIX",
    "PHASE_QUALIFICATION",
    "PROPOSAL_COMPONENTS_BY_ARM",
    "QUALIFICATION_SEED_GROUP",
    "QUALIFICATION_LEARNING_RATE_RAY",
    "SEEDS_PER_GROUP",
    "SHARED_EVIDENCE_SCHEMA",
    "SOURCE_BASELINE_SCHEMA",
    "SharedEvidenceRef",
    "SourceBaselineRef",
    "TRAINING_RPS",
    "canonical_seeds",
    "matrix_identity",
    "validate_node_cell_specs",
    "validate_matrix_receipts",
]
