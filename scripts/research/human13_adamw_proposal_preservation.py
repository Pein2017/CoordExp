#!/usr/bin/env python3
"""Exact fresh-AdamW proposal capture and owner-wise preservation projection.

The screen never hand-derives an optimizer update.  Capture runs the real
configured optimizer inside the existing full training-state transaction,
measures the parameter delta it actually produced, derives the bias-corrected
diagonal metric ``D_j = sqrt(v_hat_j) + eps`` from that same private state, and
restores everything before returning.

Preservation then solves the frozen research-unit equation

    min_delta  (delta - delta_0)^T D (delta - delta_0)
    s.t.       J_w delta >= -1e-4     for every frozen owner witness
               delta^T D delta <= delta_0^T D delta_0

with a bounded deterministic active-set dual solve and streamed witness
screening.  Two properties of that frozen equation are worth stating because
they bound what the certifications can mean: ``delta = 0`` always satisfies
every witness inequality, so the feasible set is never empty; and because the
feasible polyhedron contains the origin, the exact projection can never
increase the ``D`` norm, so the trust radius is a certified but redundant
constraint at convergence (it is tight, not slack, when no witness is active).
Both remain fail-closed checks: an uncertified solve, a non-finite value, a
radius breach, or a wrongly applied delta stops the cell instead of silently
falling back to the unprojected update.

Projected optimizer moments are deliberately not defined here: this screen
applies exactly one update and never continues from it.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from scripts.research.human13_training_transaction import (
    TrainingStateTransaction,
    UpdateCounter,
)


SCHEMA_VERSION = "human13_adamw_proposal_preservation.v1"
LAYOUT_SCHEMA_VERSION = "human13_trainable_parameter_layout.v1"
PROPOSAL_SCHEMA_VERSION = "human13_exact_adamw_proposal.v1"
WITNESS_BANK_SCHEMA_VERSION = "human13_frozen_owner_witness_bank.v1"
EVIDENCE_SCHEMA_VERSION = "human13_preservation_evidence.v1"
PROJECTION_SCHEMA_VERSION = "human13_adamw_projection_receipt.v1"
APPLY_SCHEMA_VERSION = "human13_projected_apply_receipt.v1"
JACOBIAN_SCHEMA_VERSION = "human13_owner_witness_jacobian.v1"
PARAMETER_STATE_SCHEMA_VERSION = "human13_parameter_state.v1"

FROZEN_LEARNING_RATE = 3e-6
QUALIFICATION_LEARNING_RATE_RAY = (3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
FROZEN_BETAS = (0.9, 0.999)
FROZEN_EPSILON = 1e-8
FROZEN_WEIGHT_DECAY = 0.0

WITNESS_FIRST_ORDER_TOLERANCE = 1e-4
TRAINING_RP_CONTRACTS = ("1.0", "1.10")
SOURCE_MEMBERSHIPS = ("u_intersect_s_1.0", "u_intersect_s_1.10")
LEGACY_M_MEMBERSHIP = "legacy_m_baseline_owner"
TRUSTED_OWNER_CLASS = "source_emitted_trusted_owner"
LEGACY_M_OWNER_CLASS = "legacy_m_audit_only"
WEAKEST_MARGIN_SELECTION = "weakest_detached_processed_logit_token_margin"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOLVE_RELATIVE = 1e-10
_MULTIPLIER_RELATIVE = 1e-10
_DUAL_RESIDUAL_RELATIVE = 1e-8
_MINIMUM_PIVOT = 1e-7
_RADIUS_RELATIVE = 1e-9
_APPLIED_RELATIVE = 1e-6
_CANONICAL_RELATIVE = 1e-9
_ACCUMULATION_GUARD = 8.0


class ProposalPreservationError(ValueError):
    """Base class for every fail-closed preservation disposition."""

    def __init__(self, message: str, *, disposition: str) -> None:
        super().__init__(f"{disposition}: {message}")
        self.disposition = disposition


class ProposalCaptureError(ProposalPreservationError):
    """Raised when the exact optimizer proposal cannot be certified."""


class ProposalAdmissionError(ProposalPreservationError):
    """Raised when artifact or cross-artifact identity cannot be certified."""


class ProposalProjectionError(ProposalPreservationError):
    """Raised when the owner-wise projection cannot be certified."""


class ProjectedApplyError(ProposalPreservationError):
    """Raised when the certified projected delta cannot be applied exactly."""


# --- canonical payloads and digests ------------------------------------------


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def _digest(
    value: object,
    *,
    field: str,
    error: type[ProposalPreservationError] = ProposalAdmissionError,
) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise error(
            f"{field} must be a lowercase SHA-256 digest", disposition="invalid_digest"
        )
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalAdmissionError(
            f"{field} must be a nonempty string", disposition="invalid_field"
        )
    return value


def _finite(
    value: object,
    *,
    field: str,
    error: type[ProposalPreservationError] = ProposalAdmissionError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{field} must be a finite number", disposition="invalid_field")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{field} must be a finite number", disposition="invalid_field")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProposalAdmissionError(
            f"{field} must be an integer >= {minimum}", disposition="invalid_field"
        )
    return value


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _float64_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().to(torch.float64).reshape(-1).contiguous().numpy()
    return np.ascontiguousarray(array, dtype="<f8").tobytes()


def _encode_float64(tensor: torch.Tensor) -> str:
    return base64.b64encode(_float64_bytes(tensor)).decode("ascii")


def _decode_float64(value: object, *, field: str) -> torch.Tensor:
    if not isinstance(value, str):
        raise ProposalAdmissionError(
            f"{field} must be a base64 float64 payload", disposition="invalid_field"
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ProposalAdmissionError(
            f"{field} is not valid base64", disposition="invalid_field"
        ) from error
    if len(raw) % 8 != 0:
        raise ProposalAdmissionError(
            f"{field} is not a float64 payload", disposition="invalid_field"
        )
    return torch.from_numpy(np.frombuffer(raw, dtype="<f8").copy())


def jacobian_sha256(jacobian: torch.Tensor) -> str:
    """Content address one witness Jacobian in its exact float64 layout."""

    flat = _as_flat_float64(jacobian, field="jacobian")
    hasher = hashlib.sha256()
    hasher.update(JACOBIAN_SCHEMA_VERSION.encode())
    hasher.update(b"\0")
    hasher.update(str(flat.numel()).encode())
    hasher.update(b"\0")
    hasher.update(_float64_bytes(flat))
    return hasher.hexdigest()


def _as_flat_float64(
    value: object,
    *,
    field: str,
    error: type[ProposalPreservationError] = ProposalAdmissionError,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise error(f"{field} must be a tensor", disposition="invalid_field")
    if not value.is_floating_point():
        raise error(f"{field} must be a floating tensor", disposition="invalid_field")
    return value.detach().cpu().to(torch.float64).reshape(-1).contiguous()


def _metric_quadratic(metric: torch.Tensor, vector: torch.Tensor) -> float:
    return float((metric * vector * vector).sum())


def _representation_ulp(magnitude: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Exact unit in the last place of ``dtype`` at each given magnitude.

    Derived from the dtype's own significand width rather than from a chosen
    percentage, so one rule states what an fp32, bf16, or fp16 parameter can
    and cannot represent.
    """

    finfo = torch.finfo(dtype)
    values = magnitude.abs().to(torch.float64)
    _, exponent = torch.frexp(values)
    ulp = torch.ldexp(torch.ones_like(values), exponent - 1) * float(finfo.eps)
    smallest = float(finfo.tiny) * float(finfo.eps)
    return torch.where(values > 0.0, ulp, torch.full_like(ulp, smallest))


# --- frozen scientific records ------------------------------------------------


@dataclass(frozen=True)
class ParameterLayoutEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        _text(self.name, field="parameter name")
        if not isinstance(self.shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ProposalAdmissionError(
                "parameter shape must be a tuple of non-negative integers",
                disposition="invalid_field",
            )
        _text(self.dtype, field="parameter dtype")
        expected = 1
        for size in self.shape:
            expected *= size
        if self.numel != expected:
            raise ProposalAdmissionError(
                f"parameter {self.name} numel disagrees with its shape",
                disposition="parameter_layout_mismatch",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterLayoutEntry:
        if not isinstance(payload, Mapping) or set(payload) != {
            "name",
            "shape",
            "dtype",
            "numel",
        }:
            raise ProposalAdmissionError(
                "parameter layout entry schema differs", disposition="invalid_field"
            )
        shape = payload["shape"]
        if not isinstance(shape, (list, tuple)):
            raise ProposalAdmissionError(
                "parameter shape must be a sequence", disposition="invalid_field"
            )
        return cls(
            name=_text(payload["name"], field="parameter name"),
            shape=tuple(
                _integer(size, field="parameter shape entry") for size in shape
            ),
            dtype=_text(payload["dtype"], field="parameter dtype"),
            numel=_integer(payload["numel"], field="parameter numel"),
        )


@dataclass(frozen=True)
class ParameterLayout:
    """Frozen trainable-parameter order, shape, and dtype binding."""

    entries: tuple[ParameterLayoutEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ProposalAdmissionError(
                "a parameter layout requires at least one entry",
                disposition="parameter_layout_mismatch",
            )
        names = tuple(entry.name for entry in self.entries)
        if len(set(names)) != len(names):
            raise ProposalAdmissionError(
                "parameter layout names must be unique",
                disposition="parameter_layout_mismatch",
            )

    @property
    def total_numel(self) -> int:
        return sum(entry.numel for entry in self.entries)

    @property
    def layout_sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterLayout:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "entries",
        }:
            raise ProposalAdmissionError(
                "parameter layout schema differs", disposition="invalid_field"
            )
        if payload["schema_version"] != LAYOUT_SCHEMA_VERSION:
            raise ProposalAdmissionError(
                "parameter layout schema version differs",
                disposition="schema_version_mismatch",
            )
        entries = payload["entries"]
        if not isinstance(entries, (list, tuple)):
            raise ProposalAdmissionError(
                "parameter layout entries must be a sequence",
                disposition="invalid_field",
            )
        return cls(
            entries=tuple(ParameterLayoutEntry.from_dict(entry) for entry in entries)
        )

    @classmethod
    def from_named_parameters(
        cls, named_parameters: Sequence[tuple[str, torch.Tensor]]
    ) -> ParameterLayout:
        bound = tuple(named_parameters)
        entries = []
        for item in bound:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ProposalAdmissionError(
                    "named parameters must be (name, tensor) pairs",
                    disposition="parameter_layout_mismatch",
                )
            name, tensor = item
            if not isinstance(tensor, torch.Tensor):
                raise ProposalAdmissionError(
                    f"parameter {name} is not a tensor",
                    disposition="parameter_layout_mismatch",
                )
            entries.append(
                ParameterLayoutEntry(
                    name=_text(name, field="parameter name"),
                    shape=tuple(int(size) for size in tensor.shape),
                    dtype=str(tensor.dtype),
                    numel=int(tensor.numel()),
                )
            )
        return cls(entries=tuple(entries))


def parameter_state_sha256(
    named_parameters: Sequence[tuple[str, torch.Tensor]], layout: ParameterLayout
) -> str:
    """Content address the live parameter values bound to one frozen layout."""

    if not isinstance(layout, ParameterLayout):
        raise ProposalAdmissionError(
            "a frozen parameter layout is required",
            disposition="parameter_layout_mismatch",
        )
    bound = tuple(named_parameters)
    if len(bound) != len(layout.entries):
        raise ProposalAdmissionError(
            "live parameter count differs from the frozen layout",
            disposition="parameter_layout_mismatch",
        )
    hasher = hashlib.sha256()
    hasher.update(PARAMETER_STATE_SCHEMA_VERSION.encode())
    hasher.update(b"\0")
    hasher.update(layout.layout_sha256.encode())
    for entry, item in zip(layout.entries, bound):
        name, tensor = item
        if (
            name != entry.name
            or not isinstance(tensor, torch.Tensor)
            or tuple(int(size) for size in tensor.shape) != entry.shape
            or str(tensor.dtype) != entry.dtype
        ):
            raise ProposalAdmissionError(
                f"live parameter {name} differs from the frozen layout",
                disposition="parameter_layout_mismatch",
            )
        hasher.update(b"\0")
        hasher.update(entry.name.encode())
        hasher.update(b"\0")
        hasher.update(entry.dtype.encode())
        hasher.update(b"\0")
        hasher.update(repr(entry.shape).encode())
        hasher.update(b"\0")
        hasher.update(_tensor_bytes(tensor))
    return hasher.hexdigest()


@dataclass(frozen=True)
class AdamWProposalConfig:
    """The frozen fresh-AdamW contract declared by the research unit."""

    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float

    def __post_init__(self) -> None:
        _finite(self.learning_rate, field="learning_rate")
        _finite(self.epsilon, field="epsilon")
        _finite(self.weight_decay, field="weight_decay")
        if self.learning_rate <= 0.0 or self.epsilon <= 0.0 or self.weight_decay < 0.0:
            raise ProposalAdmissionError(
                "optimizer configuration is outside its valid range",
                disposition="invalid_field",
            )
        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            raise ProposalAdmissionError(
                "betas must be a pair", disposition="invalid_field"
            )
        for beta in self.betas:
            value = _finite(beta, field="beta")
            if not 0.0 < value < 1.0:
                raise ProposalAdmissionError(
                    "betas must lie in (0, 1)", disposition="invalid_field"
                )

    @classmethod
    def frozen(
        cls, *, learning_rate: float = FROZEN_LEARNING_RATE
    ) -> AdamWProposalConfig:
        if learning_rate not in QUALIFICATION_LEARNING_RATE_RAY:
            raise ProposalAdmissionError(
                "learning rate is outside the sealed qualification dose ray",
                disposition="invalid_field",
            )
        return cls(
            learning_rate=learning_rate,
            betas=FROZEN_BETAS,
            epsilon=FROZEN_EPSILON,
            weight_decay=FROZEN_WEIGHT_DECAY,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdamWProposalConfig:
        if not isinstance(payload, Mapping) or set(payload) != {
            "learning_rate",
            "betas",
            "epsilon",
            "weight_decay",
        }:
            raise ProposalAdmissionError(
                "optimizer configuration schema differs", disposition="invalid_field"
            )
        betas = payload["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ProposalAdmissionError(
                "betas must be a pair", disposition="invalid_field"
            )
        return cls(
            learning_rate=_finite(payload["learning_rate"], field="learning_rate"),
            betas=(
                _finite(betas[0], field="beta_1"),
                _finite(betas[1], field="beta_2"),
            ),
            epsilon=_finite(payload["epsilon"], field="epsilon"),
            weight_decay=_finite(payload["weight_decay"], field="weight_decay"),
        )


@dataclass(frozen=True)
class ProposalBinding:
    """Scientific lineage of one private one-update proposal."""

    unit_id: str
    arm_id: str
    training_rp: str
    seed_group: str
    source_checkpoint_sha256: str
    manifest_sha256: str
    objective_ledger_sha256: str

    def __post_init__(self) -> None:
        _text(self.unit_id, field="unit_id")
        _text(self.arm_id, field="arm_id")
        _text(self.seed_group, field="seed_group")
        if self.training_rp not in TRAINING_RP_CONTRACTS:
            raise ProposalAdmissionError(
                "training_rp must be a declared RP contract",
                disposition="invalid_field",
            )
        _digest(self.source_checkpoint_sha256, field="source_checkpoint_sha256")
        _digest(self.manifest_sha256, field="manifest_sha256")
        _digest(self.objective_ledger_sha256, field="objective_ledger_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "arm_id": self.arm_id,
            "training_rp": self.training_rp,
            "seed_group": self.seed_group,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "objective_ledger_sha256": self.objective_ledger_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProposalBinding:
        keys = {
            "unit_id",
            "arm_id",
            "training_rp",
            "seed_group",
            "source_checkpoint_sha256",
            "manifest_sha256",
            "objective_ledger_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise ProposalAdmissionError(
                "proposal binding schema differs", disposition="invalid_field"
            )
        return cls(**{key: payload[key] for key in keys})


@dataclass(frozen=True)
class WitnessBinding:
    """Lineage of the witness set frozen from Source before acquisition."""

    unit_id: str
    source_checkpoint_sha256: str
    manifest_sha256: str
    frozen_before_acquisition: bool

    def __post_init__(self) -> None:
        _text(self.unit_id, field="unit_id")
        _digest(self.source_checkpoint_sha256, field="source_checkpoint_sha256")
        _digest(self.manifest_sha256, field="manifest_sha256")
        if self.frozen_before_acquisition is not True:
            raise ProposalAdmissionError(
                "witnesses must be frozen from Source before acquisition",
                disposition="witness_not_frozen",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "frozen_before_acquisition": self.frozen_before_acquisition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WitnessBinding:
        keys = {
            "unit_id",
            "source_checkpoint_sha256",
            "manifest_sha256",
            "frozen_before_acquisition",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise ProposalAdmissionError(
                "witness binding schema differs", disposition="invalid_field"
            )
        return cls(**{key: payload[key] for key in keys})


@dataclass(frozen=True)
class OwnerWitness:
    """One typed owner-wise preservation witness.

    Membership, owner class, and margin selection are receipted input
    semantics: this module never re-derives owners, matches boxes, or picks
    tokens.  Legacy-M owners are admitted as audit-only records and can never
    enter the projection constraints.
    """

    image_id: str
    owner_id: str
    source_membership: str
    owner_class: str
    token_id: int
    margin_selection: str
    margin_value: float
    detached: bool
    jacobian_sha256: str | None

    def __post_init__(self) -> None:
        _text(self.image_id, field="image_id")
        _text(self.owner_id, field="owner_id")
        _integer(self.token_id, field="token_id")
        _finite(self.margin_value, field="margin_value")
        if self.margin_selection != WEAKEST_MARGIN_SELECTION:
            raise ProposalAdmissionError(
                "witness margins must be the weakest detached processed-logit "
                "token margin",
                disposition="witness_margin_selection",
            )
        if self.detached is not True:
            raise ProposalAdmissionError(
                "witness margins must be detached from acquisition outcomes",
                disposition="witness_not_detached",
            )
        if self.owner_class == TRUSTED_OWNER_CLASS:
            if self.source_membership not in SOURCE_MEMBERSHIPS:
                raise ProposalAdmissionError(
                    "constraint witnesses must be Source-emitted owners in "
                    "U intersect S_1.0 or U intersect S_1.10",
                    disposition="witness_membership",
                )
            _digest(self.jacobian_sha256, field="jacobian_sha256")
        elif self.owner_class == LEGACY_M_OWNER_CLASS:
            if self.source_membership != LEGACY_M_MEMBERSHIP:
                raise ProposalAdmissionError(
                    "legacy-M owners cannot claim a trusted Source membership",
                    disposition="witness_membership",
                )
            if self.jacobian_sha256 is not None:
                raise ProposalAdmissionError(
                    "legacy-M owners are audit-only and carry no Jacobian",
                    disposition="witness_membership",
                )
        else:
            raise ProposalAdmissionError(
                "unknown witness owner class", disposition="witness_membership"
            )

    @property
    def canonical_key(self) -> str:
        return f"{self.source_membership}|{self.image_id}|{self.owner_id}"

    @property
    def is_constraint(self) -> bool:
        return self.owner_class == TRUSTED_OWNER_CLASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "owner_id": self.owner_id,
            "source_membership": self.source_membership,
            "owner_class": self.owner_class,
            "token_id": self.token_id,
            "margin_selection": self.margin_selection,
            "margin_value": self.margin_value,
            "detached": self.detached,
            "jacobian_sha256": self.jacobian_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OwnerWitness:
        keys = {
            "image_id",
            "owner_id",
            "source_membership",
            "owner_class",
            "token_id",
            "margin_selection",
            "margin_value",
            "detached",
            "jacobian_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise ProposalAdmissionError(
                "owner witness schema differs", disposition="invalid_field"
            )
        return cls(**{key: payload[key] for key in keys})


@dataclass(frozen=True)
class FrozenWitnessBank:
    """Canonically ordered witness set with a streamed Jacobian provider.

    The provider is deliberately outside the bank digest: every streamed
    Jacobian is re-verified against its frozen content address, so a durable or
    runtime substitution fails before it can reach the solver.
    """

    binding: WitnessBinding
    layout: ParameterLayout
    constraints: tuple[OwnerWitness, ...]
    audit_only: tuple[OwnerWitness, ...]
    jacobian_provider: Callable[[OwnerWitness], torch.Tensor] = dataclass_field(
        compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.binding, WitnessBinding) or not isinstance(
            self.layout, ParameterLayout
        ):
            raise ProposalAdmissionError(
                "witness bank requires a frozen binding and layout",
                disposition="invalid_field",
            )
        if not isinstance(self.constraints, tuple) or not isinstance(
            self.audit_only, tuple
        ):
            raise ProposalAdmissionError(
                "witness bank records must be tuples", disposition="invalid_field"
            )
        if not callable(self.jacobian_provider):
            raise ProposalAdmissionError(
                "witness bank requires a Jacobian provider",
                disposition="invalid_field",
            )
        keys = [witness.canonical_key for witness in self.constraints + self.audit_only]
        if len(set(keys)) != len(keys):
            raise ProposalAdmissionError(
                "witness owners must be unique", disposition="witness_duplicate_owner"
            )
        for group in (self.constraints, self.audit_only):
            ordered = tuple(sorted(group, key=_witness_sort_key))
            if group != ordered:
                raise ProposalAdmissionError(
                    "witness records must use the canonical order",
                    disposition="witness_order",
                )
        if any(not witness.is_constraint for witness in self.constraints):
            raise ProposalAdmissionError(
                "only Source-emitted trusted owners constrain the projection",
                disposition="witness_membership",
            )
        if any(witness.is_constraint for witness in self.audit_only):
            raise ProposalAdmissionError(
                "audit-only records cannot be trusted constraint owners",
                disposition="witness_membership",
            )

    @property
    def bank_sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WITNESS_BANK_SCHEMA_VERSION,
            "binding": self.binding.to_dict(),
            "layout": self.layout.to_dict(),
            "first_order_tolerance": WITNESS_FIRST_ORDER_TOLERANCE,
            "constraints": [witness.to_dict() for witness in self.constraints],
            "audit_only": [witness.to_dict() for witness in self.audit_only],
        }

    def stream_constraints(self) -> Iterator[tuple[int, OwnerWitness, torch.Tensor]]:
        """Yield one certified constraint Jacobian at a time.

        Only the caller's working set is retained, so screening never
        materializes the full witness Jacobian matrix.
        """

        expected = self.layout.total_numel
        for index, witness in enumerate(self.constraints):
            jacobian = self.jacobian_provider(witness)
            flat = _as_flat_float64(
                jacobian, field=f"jacobian for {witness.canonical_key}"
            )
            if flat.numel() != expected:
                raise ProposalAdmissionError(
                    f"jacobian for {witness.canonical_key} does not match the "
                    "frozen parameter layout",
                    disposition="jacobian_layout_mismatch",
                )
            if jacobian_sha256(flat) != witness.jacobian_sha256:
                raise ProposalAdmissionError(
                    f"jacobian for {witness.canonical_key} does not match its "
                    "frozen content address",
                    disposition="jacobian_digest_mismatch",
                )
            if not bool(torch.isfinite(flat).all()):
                raise ProposalProjectionError(
                    f"jacobian for {witness.canonical_key} is not finite",
                    disposition="non_finite_jacobian",
                )
            yield index, witness, flat

    @classmethod
    def from_witnesses(
        cls,
        witnesses: Sequence[OwnerWitness],
        *,
        jacobians: Mapping[str, torch.Tensor],
        binding: WitnessBinding,
        layout: ParameterLayout,
    ) -> FrozenWitnessBank:
        bound = tuple(witnesses)
        if any(not isinstance(witness, OwnerWitness) for witness in bound):
            raise ProposalAdmissionError(
                "witness records must be typed owner witnesses",
                disposition="invalid_field",
            )
        constraints = tuple(
            sorted((w for w in bound if w.is_constraint), key=_witness_sort_key)
        )
        audit_only = tuple(
            sorted((w for w in bound if not w.is_constraint), key=_witness_sort_key)
        )
        if not isinstance(jacobians, Mapping):
            raise ProposalAdmissionError(
                "jacobians must be a mapping", disposition="invalid_field"
            )
        expected_keys = {witness.canonical_key for witness in constraints}
        if set(jacobians) != expected_keys:
            raise ProposalAdmissionError(
                "every constraint witness needs exactly one Jacobian",
                disposition="witness_jacobian_coverage",
            )
        stored: dict[str, torch.Tensor] = {}
        for witness in constraints:
            flat = _as_flat_float64(
                jacobians[witness.canonical_key],
                field=f"jacobian for {witness.canonical_key}",
            )
            if flat.numel() != layout.total_numel:
                raise ProposalAdmissionError(
                    f"jacobian for {witness.canonical_key} does not match the "
                    "frozen parameter layout",
                    disposition="jacobian_layout_mismatch",
                )
            if jacobian_sha256(flat) != witness.jacobian_sha256:
                raise ProposalAdmissionError(
                    f"jacobian for {witness.canonical_key} does not match its "
                    "frozen content address",
                    disposition="jacobian_digest_mismatch",
                )
            stored[witness.canonical_key] = flat
        return cls(
            binding=binding,
            layout=layout,
            constraints=constraints,
            audit_only=audit_only,
            jacobian_provider=lambda witness: stored[witness.canonical_key],
        )


def _witness_sort_key(witness: OwnerWitness) -> tuple[str, str, str]:
    return (witness.source_membership, witness.image_id, witness.owner_id)


@dataclass(frozen=True)
class ExactAdamWProposal:
    """The measured parameter delta of one real fresh-AdamW step."""

    binding: ProposalBinding
    config: AdamWProposalConfig
    layout: ParameterLayout
    pre_parameter_sha256: str
    post_parameter_sha256: str
    gradient_sha256: str
    gradient_norm: float
    delta: tuple[torch.Tensor, ...]
    metric_denominator: tuple[torch.Tensor, ...]
    trust_radius: float
    delta_norm: float
    delta_reconstruction_residual: float
    delta_reconstruction_allowance: float
    optimizer_class: str
    optimizer_step_count_before: int
    optimizer_step_count_after: int
    update_count_before: int
    transaction_before_digest: str
    transaction_after_digest: str
    delta_sha256: str = dataclass_field(init=False, repr=False, compare=False)
    metric_denominator_sha256: str = dataclass_field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.binding, ProposalBinding)
            or not isinstance(self.config, AdamWProposalConfig)
            or not isinstance(self.layout, ParameterLayout)
        ):
            raise ProposalAdmissionError(
                "proposal requires typed binding, config, and layout",
                disposition="invalid_field",
            )
        _digest(self.pre_parameter_sha256, field="pre_parameter_sha256")
        _digest(self.post_parameter_sha256, field="post_parameter_sha256")
        _digest(self.gradient_sha256, field="gradient_sha256")
        _digest(self.transaction_before_digest, field="transaction_before_digest")
        _digest(self.transaction_after_digest, field="transaction_after_digest")
        _text(self.optimizer_class, field="optimizer_class")
        _integer(self.optimizer_step_count_before, field="optimizer_step_count_before")
        _integer(self.optimizer_step_count_after, field="optimizer_step_count_after")
        _integer(self.update_count_before, field="update_count_before")
        for field_name in (
            "gradient_norm",
            "trust_radius",
            "delta_norm",
            "delta_reconstruction_residual",
            "delta_reconstruction_allowance",
        ):
            value = _finite(getattr(self, field_name), field=field_name)
            if value < 0.0:
                raise ProposalAdmissionError(
                    f"{field_name} must be non-negative", disposition="invalid_field"
                )
        if self.transaction_before_digest != self.transaction_after_digest:
            raise ProposalAdmissionError(
                "capture must restore the exact training state it snapshotted",
                disposition="transaction_restore_uncertified",
            )
        if self.delta_reconstruction_residual > self.delta_reconstruction_allowance:
            raise ProposalAdmissionError(
                "the measured delta is not the configured AdamW step",
                disposition="delta_reconstruction_uncertified",
            )
        for name, group in (
            ("delta", self.delta),
            ("metric_denominator", self.metric_denominator),
        ):
            if not isinstance(group, tuple) or len(group) != len(self.layout.entries):
                raise ProposalAdmissionError(
                    f"{name} must hold one tensor per frozen parameter",
                    disposition="parameter_layout_mismatch",
                )
            for entry, tensor in zip(self.layout.entries, group):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype is not torch.float64
                    or tensor.dim() != 1
                    or tensor.numel() != entry.numel
                ):
                    raise ProposalAdmissionError(
                        f"{name} for {entry.name} does not match the frozen layout",
                        disposition="parameter_layout_mismatch",
                    )
        flat_delta = self.flat_delta()
        flat_metric = self.flat_metric_denominator()
        if not bool(torch.isfinite(flat_delta).all()) or not bool(
            torch.isfinite(flat_metric).all()
        ):
            raise ProposalAdmissionError(
                "proposal delta and metric must be finite",
                disposition="non_finite_proposal",
            )
        if float(flat_metric.min()) < self.config.epsilon * (1.0 - 1e-12):
            raise ProposalAdmissionError(
                "the bias-corrected metric must be at least the AdamW epsilon",
                disposition="singular_metric",
            )
        if self.trust_radius != _metric_quadratic(flat_metric, flat_delta):
            raise ProposalAdmissionError(
                "the trust radius is not delta_0^T D delta_0",
                disposition="trust_radius_mismatch",
            )
        object.__setattr__(
            self,
            "delta_sha256",
            _hash_flat_tensors(self.delta, layout=self.layout, context="delta"),
        )
        object.__setattr__(
            self,
            "metric_denominator_sha256",
            _hash_flat_tensors(
                self.metric_denominator, layout=self.layout, context="metric"
            ),
        )

    @property
    def schema_version(self) -> str:
        return PROPOSAL_SCHEMA_VERSION

    def flat_delta(self) -> torch.Tensor:
        return torch.cat(self.delta)

    def flat_metric_denominator(self) -> torch.Tensor:
        return torch.cat(self.metric_denominator)

    @property
    def proposal_sha256(self) -> str:
        return _sha256(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "binding": self.binding.to_dict(),
            "config": self.config.to_dict(),
            "layout": self.layout.to_dict(),
            "pre_parameter_sha256": self.pre_parameter_sha256,
            "post_parameter_sha256": self.post_parameter_sha256,
            "gradient_sha256": self.gradient_sha256,
            "gradient_norm": self.gradient_norm,
            "delta_sha256": self.delta_sha256,
            "metric_denominator_sha256": self.metric_denominator_sha256,
            "trust_radius": self.trust_radius,
            "delta_norm": self.delta_norm,
            "delta_reconstruction_residual": self.delta_reconstruction_residual,
            "delta_reconstruction_allowance": self.delta_reconstruction_allowance,
            "optimizer_class": self.optimizer_class,
            "optimizer_step_count_before": self.optimizer_step_count_before,
            "optimizer_step_count_after": self.optimizer_step_count_after,
            "update_count_before": self.update_count_before,
            "transaction_before_digest": self.transaction_before_digest,
            "transaction_after_digest": self.transaction_after_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["delta_base64"] = [_encode_float64(tensor) for tensor in self.delta]
        payload["metric_denominator_base64"] = [
            _encode_float64(tensor) for tensor in self.metric_denominator
        ]
        payload["proposal_sha256"] = self.proposal_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExactAdamWProposal:
        if not isinstance(payload, Mapping):
            raise ProposalAdmissionError(
                "proposal payload must be a mapping", disposition="invalid_field"
            )
        if payload.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
            raise ProposalAdmissionError(
                "proposal schema version differs",
                disposition="schema_version_mismatch",
            )
        layout = ParameterLayout.from_dict(payload["layout"])
        proposal = cls(
            binding=ProposalBinding.from_dict(payload["binding"]),
            config=AdamWProposalConfig.from_dict(payload["config"]),
            layout=layout,
            pre_parameter_sha256=payload["pre_parameter_sha256"],
            post_parameter_sha256=payload["post_parameter_sha256"],
            gradient_sha256=payload["gradient_sha256"],
            gradient_norm=payload["gradient_norm"],
            delta=_decode_group(payload, "delta_base64", layout, "delta"),
            metric_denominator=_decode_group(
                payload, "metric_denominator_base64", layout, "metric_denominator"
            ),
            trust_radius=payload["trust_radius"],
            delta_norm=payload["delta_norm"],
            delta_reconstruction_residual=payload["delta_reconstruction_residual"],
            delta_reconstruction_allowance=payload["delta_reconstruction_allowance"],
            optimizer_class=payload["optimizer_class"],
            optimizer_step_count_before=payload["optimizer_step_count_before"],
            optimizer_step_count_after=payload["optimizer_step_count_after"],
            update_count_before=payload["update_count_before"],
            transaction_before_digest=payload["transaction_before_digest"],
            transaction_after_digest=payload["transaction_after_digest"],
        )
        if (
            payload.get("delta_sha256") != proposal.delta_sha256
            or payload.get("metric_denominator_sha256")
            != proposal.metric_denominator_sha256
        ):
            raise ProposalAdmissionError(
                "proposal tensors do not match their frozen content addresses",
                disposition="proposal_digest_mismatch",
            )
        if payload.get("proposal_sha256") != proposal.proposal_sha256:
            raise ProposalAdmissionError(
                "proposal payload does not match its frozen content address",
                disposition="proposal_digest_mismatch",
            )
        return proposal


def _decode_group(
    payload: Mapping[str, Any], key: str, layout: ParameterLayout, field: str
) -> tuple[torch.Tensor, ...]:
    encoded = payload.get(key)
    if not isinstance(encoded, (list, tuple)) or len(encoded) != len(layout.entries):
        raise ProposalAdmissionError(
            f"{field} must hold one payload per frozen parameter",
            disposition="parameter_layout_mismatch",
        )
    return tuple(
        _decode_float64(item, field=f"{field}[{index}]")
        for index, item in enumerate(encoded)
    )


def _hash_flat_tensors(
    tensors: Sequence[torch.Tensor], *, layout: ParameterLayout, context: str
) -> str:
    hasher = hashlib.sha256()
    hasher.update(context.encode())
    hasher.update(b"\0")
    hasher.update(layout.layout_sha256.encode())
    for entry, tensor in zip(layout.entries, tensors):
        hasher.update(b"\0")
        hasher.update(entry.name.encode())
        hasher.update(b"\0")
        hasher.update(_float64_bytes(tensor))
    return hasher.hexdigest()


# --- exact optimizer proposal capture ----------------------------------------


def _certify_transaction_ownership(
    transaction: TrainingStateTransaction,
    bound: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    optimizer: torch.optim.Optimizer,
    update_counter: UpdateCounter | None,
    error: type[ProposalPreservationError],
) -> None:
    """Certify that the transaction owns the exact objects about to change.

    Value equality is not enough: a foreign stack with the same seed holds
    equal parameters, so a rollback bound to it would silently leave this
    surface mutated.  The bound surface is read defensively and any change to
    the transaction's attribute contract fails closed here, before the
    transaction is opened or anything is stepped.
    """

    try:
        owned = tuple(transaction._named_parameters)
        owned_optimizer = transaction._optimizer
        owned_counter = transaction._update_counter
    except AttributeError as attribute_error:
        raise error(
            "the bound transaction does not expose its owned surface",
            disposition="transaction_binding_uncertified",
        ) from attribute_error
    if {id(parameter) for _, parameter in owned} != {
        id(parameter) for _, parameter in bound
    } or len(owned) != len(bound):
        raise error(
            "the transaction does not own these exact trainable parameters",
            disposition="transaction_binding_uncertified",
        )
    if owned_optimizer is not optimizer:
        raise error(
            "the transaction does not own this optimizer",
            disposition="transaction_binding_uncertified",
        )
    if update_counter is not None and owned_counter is not update_counter:
        raise error(
            "the transaction does not own this update counter",
            disposition="transaction_binding_uncertified",
        )
    if tuple((name, id(parameter)) for name, parameter in owned) != tuple(
        (name, id(parameter)) for name, parameter in bound
    ):
        raise error(
            "the bound trainable surface differs from the transaction surface",
            disposition="parameter_layout_mismatch",
        )


def _require_frozen_optimizer(
    optimizer: torch.optim.Optimizer, config: AdamWProposalConfig
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ProposalCaptureError(
            "a real torch optimizer is required",
            disposition="optimizer_config_mismatch",
        )
    if config != AdamWProposalConfig.frozen():
        raise ProposalCaptureError(
            "the screen admits only the frozen fresh-AdamW contract",
            disposition="optimizer_config_mismatch",
        )
    for group in optimizer.param_groups:
        betas = tuple(group.get("betas", ()))
        if (
            float(group.get("lr", float("nan"))) != config.learning_rate
            or len(betas) != 2
            or float(betas[0]) != config.betas[0]
            or float(betas[1]) != config.betas[1]
            or float(group.get("eps", float("nan"))) != config.epsilon
            or float(group.get("weight_decay", float("nan"))) != config.weight_decay
        ):
            raise ProposalCaptureError(
                "the live optimizer does not carry the frozen AdamW contract",
                disposition="optimizer_config_mismatch",
            )
        if bool(group.get("amsgrad", False)) or bool(group.get("maximize", False)):
            raise ProposalCaptureError(
                "amsgrad and maximize are outside the frozen contract",
                disposition="optimizer_config_mismatch",
            )


def capture_exact_adamw_proposal(
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    optimizer: torch.optim.Optimizer,
    transaction: TrainingStateTransaction,
    config: AdamWProposalConfig,
    binding: ProposalBinding,
) -> ExactAdamWProposal:
    """Measure the exact parameter delta of one real fresh-AdamW step.

    The step runs inside the caller's full training-state transaction and is
    always rolled back: no accepted update, optimizer moment, counter, or RNG
    draw survives this call.
    """

    bound = tuple(named_trainable_parameters)
    if not bound:
        raise ProposalCaptureError(
            "capture requires bound trainable parameters",
            disposition="parameter_layout_mismatch",
        )
    names = tuple(name for name, _ in bound)
    parameters = tuple(parameter for _, parameter in bound)
    if len(set(names)) != len(names) or len({id(p) for p in parameters}) != len(
        parameters
    ):
        raise ProposalCaptureError(
            "bound trainable parameters must be unique",
            disposition="parameter_layout_mismatch",
        )
    if not isinstance(transaction, TrainingStateTransaction):
        raise ProposalCaptureError(
            "capture requires the full training-state transaction",
            disposition="invalid_field",
        )
    if not isinstance(binding, ProposalBinding):
        raise ProposalCaptureError(
            "capture requires a typed proposal binding", disposition="invalid_field"
        )
    _require_frozen_optimizer(optimizer, config)
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if {id(p) for p in optimizer_parameters} != {id(p) for p in parameters} or len(
        optimizer_parameters
    ) != len(parameters):
        raise ProposalCaptureError(
            "the optimizer surface differs from the bound trainable surface",
            disposition="optimizer_surface_mismatch",
        )
    if len(optimizer.state) != 0:
        raise ProposalCaptureError(
            "every arm starts from a fresh optimizer",
            disposition="optimizer_is_not_fresh",
        )

    layout = ParameterLayout.from_named_parameters(bound)
    gradients: list[torch.Tensor] = []
    for name, parameter in bound:
        gradient = parameter.grad
        if gradient is None:
            raise ProposalCaptureError(
                f"parameter {name} carries no gradient", disposition="missing_gradient"
            )
        if gradient.shape != parameter.shape:
            raise ProposalCaptureError(
                f"gradient for {name} does not match its parameter",
                disposition="parameter_layout_mismatch",
            )
        flat = _as_flat_float64(
            gradient, field=f"gradient for {name}", error=ProposalCaptureError
        )
        if not bool(torch.isfinite(flat).all()):
            raise ProposalCaptureError(
                f"gradient for {name} is not finite", disposition="non_finite_gradient"
            )
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise ProposalCaptureError(
                f"parameter {name} is not finite", disposition="non_finite_parameter"
            )
        gradients.append(flat)

    pre_parameter_sha256 = parameter_state_sha256(bound, layout)
    pre_values = tuple(
        parameter.detach().clone().to(torch.float64) for _, parameter in bound
    )
    _certify_transaction_ownership(
        transaction,
        bound,
        optimizer=optimizer,
        update_counter=None,
        error=ProposalCaptureError,
    )

    snapshot = transaction.begin()
    try:
        optimizer.step()
        post_parameter_sha256 = parameter_state_sha256(bound, layout)
        deltas: list[torch.Tensor] = []
        metrics: list[torch.Tensor] = []
        residual = 0.0
        parameter_scale = 0.0
        predicted_scale = 0.0
        dtype_epsilon = 0.0
        for index, (name, parameter) in enumerate(bound):
            state = optimizer.state.get(parameter, {})
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ProposalCaptureError(
                    f"optimizer state for {name} is not a fresh AdamW state",
                    disposition="unexpected_optimizer_state",
                )
            step_count = float(state["step"])
            if step_count != 1.0:
                raise ProposalCaptureError(
                    f"optimizer state for {name} did not take exactly one step",
                    disposition="unexpected_optimizer_state",
                )
            post = parameter.detach().to(torch.float64)
            delta = (post - pre_values[index]).reshape(-1).contiguous()
            first = state["exp_avg"].detach().to(torch.float64).reshape(-1)
            second = state["exp_avg_sq"].detach().to(torch.float64).reshape(-1)
            bias_correction1 = 1.0 - config.betas[0] ** step_count
            bias_correction2 = 1.0 - config.betas[1] ** step_count
            metric = (second / bias_correction2).sqrt() + config.epsilon
            predicted = -config.learning_rate * (first / bias_correction1) / metric
            residual = max(residual, float((delta - predicted).abs().max()))
            predicted_scale = max(predicted_scale, float(predicted.abs().max()))
            parameter_scale = max(
                parameter_scale,
                float(pre_values[index].abs().max()),
                float(post.abs().max()),
            )
            dtype_epsilon = max(dtype_epsilon, float(torch.finfo(parameter.dtype).eps))
            deltas.append(delta)
            metrics.append(metric)
        allowance = (
            4.0 * dtype_epsilon * parameter_scale
            + 16.0 * float(torch.finfo(torch.float64).eps) * predicted_scale
        )
        if not math.isfinite(residual) or residual > allowance:
            raise ProposalCaptureError(
                "the measured delta is not the configured AdamW step",
                disposition="delta_reconstruction_uncertified",
            )
        flat_delta = torch.cat(deltas)
        flat_metric = torch.cat(metrics)
        if not bool(torch.isfinite(flat_delta).all()) or not bool(
            torch.isfinite(flat_metric).all()
        ):
            raise ProposalCaptureError(
                "the measured proposal is not finite", disposition="non_finite_proposal"
            )
        flat_gradient = torch.cat(gradients)
        captured = {
            "post_parameter_sha256": post_parameter_sha256,
            "gradient_sha256": _hash_flat_tensors(
                tuple(gradients), layout=layout, context="gradient"
            ),
            "gradient_norm": float(flat_gradient.norm()),
            "delta": tuple(deltas),
            "metric_denominator": tuple(metrics),
            "trust_radius": _metric_quadratic(flat_metric, flat_delta),
            "delta_norm": float(flat_delta.norm()),
            "delta_reconstruction_residual": residual,
            "delta_reconstruction_allowance": allowance,
            "optimizer_class": type(optimizer).__name__,
        }
    finally:
        restore = transaction.reject(snapshot)

    if (
        parameter_state_sha256(bound, layout) != pre_parameter_sha256
        or len(optimizer.state) != 0
    ):
        raise ProposalCaptureError(
            "the private proposal did not restore the Source training state",
            disposition="transaction_restore_uncertified",
        )
    return ExactAdamWProposal(
        binding=binding,
        config=config,
        layout=layout,
        pre_parameter_sha256=pre_parameter_sha256,
        optimizer_step_count_before=0,
        optimizer_step_count_after=1,
        update_count_before=int(snapshot.update_count),
        transaction_before_digest=snapshot.state_digest,
        transaction_after_digest=restore.after_state_digest,
        **captured,
    )


# --- aggregate scientific admission ------------------------------------------


@dataclass(frozen=True)
class PreservationEvidence:
    """The admitted proposal/witness pair that every scientific path shares."""

    proposal: ExactAdamWProposal
    witness_bank: FrozenWitnessBank

    @property
    def evidence_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "proposal_sha256": self.proposal.proposal_sha256,
                "bank_sha256": self.witness_bank.bank_sha256,
            }
        )


def _admit_evidence(
    proposal: ExactAdamWProposal, witness_bank: FrozenWitnessBank
) -> PreservationEvidence:
    """One aggregate choke point for every public scientific entry."""

    if not isinstance(proposal, ExactAdamWProposal):
        raise ProposalAdmissionError(
            "an admitted exact AdamW proposal is required", disposition="invalid_field"
        )
    if not isinstance(witness_bank, FrozenWitnessBank):
        raise ProposalAdmissionError(
            "an admitted frozen witness bank is required", disposition="invalid_field"
        )
    if (
        _hash_flat_tensors(proposal.delta, layout=proposal.layout, context="delta")
        != proposal.delta_sha256
        or _hash_flat_tensors(
            proposal.metric_denominator, layout=proposal.layout, context="metric"
        )
        != proposal.metric_denominator_sha256
    ):
        raise ProposalAdmissionError(
            "proposal tensors no longer match their admitted content addresses",
            disposition="proposal_digest_mismatch",
        )
    if proposal.trust_radius != _metric_quadratic(
        proposal.flat_metric_denominator(), proposal.flat_delta()
    ):
        raise ProposalAdmissionError(
            "the trust radius is not delta_0^T D delta_0",
            disposition="trust_radius_mismatch",
        )
    if witness_bank.layout.layout_sha256 != proposal.layout.layout_sha256:
        raise ProposalAdmissionError(
            "witness Jacobians and the proposal use different parameter layouts",
            disposition="layout_mismatch",
        )
    if (
        witness_bank.binding.unit_id != proposal.binding.unit_id
        or witness_bank.binding.source_checkpoint_sha256
        != proposal.binding.source_checkpoint_sha256
        or witness_bank.binding.manifest_sha256 != proposal.binding.manifest_sha256
    ):
        raise ProposalAdmissionError(
            "the witness bank was not frozen from this proposal's Source",
            disposition="source_binding_mismatch",
        )
    if not witness_bank.constraints:
        raise ProposalAdmissionError(
            "preservation requires at least one frozen owner witness",
            disposition="empty_witness_constraint_set",
        )
    return PreservationEvidence(proposal=proposal, witness_bank=witness_bank)


def admit_preservation_evidence(
    proposal: ExactAdamWProposal, witness_bank: FrozenWitnessBank
) -> PreservationEvidence:
    """Public entry into the shared preservation admission choke point."""

    return _admit_evidence(proposal, witness_bank)


# --- bounded deterministic projection ----------------------------------------


@dataclass(frozen=True)
class _WorkingSetSolution:
    delta: torch.Tensor
    multipliers: tuple[float, ...]
    dual_residual: float


def _solve_working_set(
    delta0: torch.Tensor,
    rows: Sequence[torch.Tensor],
    *,
    inverse_metric: torch.Tensor,
    tolerance: float,
) -> _WorkingSetSolution:
    """Solve the equality-constrained subproblem for one working set.

    ``min (d-d0)^T D (d-d0)`` subject to ``a_w^T d = -tolerance`` has the dual
    solution ``d = d0 - D^-1 A^T lambda`` with ``(A D^-1 A^T) lambda = A d0 +
    tolerance``.  A dependent or ill-conditioned working set fails closed
    instead of returning a plausible but uncertified vector.
    """

    if not rows:
        return _WorkingSetSolution(
            delta=delta0.clone(), multipliers=(), dual_residual=0.0
        )
    matrix = torch.stack(tuple(rows))
    scaled = matrix * inverse_metric
    gram = scaled @ matrix.T
    right = matrix @ delta0 + tolerance
    if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(right).all()):
        raise ProposalProjectionError(
            "the dual system is not finite", disposition="projection_non_finite"
        )
    # Solve the unit-diagonal (correlation) form so that conditioning, pivots,
    # and the residual bound are all scale free.
    diagonal = torch.diagonal(gram)
    if float(diagonal.min()) <= 0.0:
        raise ProposalProjectionError(
            "an active witness Jacobian vanishes in the frozen metric",
            disposition="singular_dual_system",
        )
    scaling = diagonal.sqrt()
    normalized = gram / torch.outer(scaling, scaling)
    normalized_right = right / scaling
    factor, info = torch.linalg.cholesky_ex(normalized)
    if int(info) != 0 or float(torch.diagonal(factor).min()) < _MINIMUM_PIVOT:
        raise ProposalProjectionError(
            "the active witness Jacobians form a singular dual system",
            disposition="singular_dual_system",
        )
    normalized_lagrange = torch.cholesky_solve(
        normalized_right.unsqueeze(1), factor
    ).squeeze(1)
    residual = float((normalized @ normalized_lagrange - normalized_right).abs().max())
    scale = (
        1.0
        + float(normalized_right.abs().max())
        + float(normalized_lagrange.abs().max())
    )
    if not math.isfinite(residual) or residual > _DUAL_RESIDUAL_RELATIVE * scale:
        raise ProposalProjectionError(
            "the dual solve could not be certified",
            disposition="singular_dual_system",
        )
    lagrange = normalized_lagrange / scaling
    delta = delta0 - scaled.T @ lagrange
    if not bool(torch.isfinite(delta).all()):
        raise ProposalProjectionError(
            "the projected delta is not finite", disposition="projection_non_finite"
        )
    return _WorkingSetSolution(
        delta=delta,
        multipliers=tuple(float(-2.0 * value) for value in lagrange),
        dual_residual=residual,
    )


@dataclass(frozen=True)
class _MetricSolveResult:
    delta: torch.Tensor
    active: tuple[int, ...]
    multipliers: tuple[float, ...]
    iterations: int
    dual_residual: float
    minimum_slack: float
    metric_radius_value: float
    trust_radius_active: bool


def _solve_metric_projection(
    delta0: torch.Tensor,
    row_factory: Callable[[], Iterator[tuple[int, torch.Tensor]]],
    *,
    metric: torch.Tensor,
    tolerance: float,
    radius: float,
    max_iterations: int,
) -> _MetricSolveResult:
    """Bounded deterministic active-set projection in the frozen ``D`` metric.

    Witnesses are screened one streamed row at a time; only active rows are
    materialized.  Constraint order is the canonical witness order, the most
    violated constraint enters first with a lowest-index tie-break, and the
    most negative multiplier leaves first with the same tie-break, so the
    result cannot depend on the caller's input order.
    """

    inverse_metric = 1.0 / metric
    active: list[int] = []
    rows: dict[int, torch.Tensor] = {}
    for iteration in range(1, int(max_iterations) + 1):
        solution = _solve_working_set(
            delta0,
            tuple(rows[index] for index in active),
            inverse_metric=inverse_metric,
            tolerance=tolerance,
        )
        worst_index: int | None = None
        worst_slack = math.inf
        worst_row: torch.Tensor | None = None
        minimum_slack = math.inf
        for index, row in row_factory():
            slack = float(row @ solution.delta) + tolerance
            noise = _SOLVE_RELATIVE * max(
                float(row.abs() @ solution.delta.abs()), abs(tolerance)
            )
            minimum_slack = min(minimum_slack, slack)
            if slack < -noise and slack < worst_slack:
                worst_index, worst_slack, worst_row = index, slack, row
        if worst_index is not None and worst_row is not None:
            if worst_index in rows:
                raise ProposalProjectionError(
                    "an active witness constraint remained violated",
                    disposition="first_order_uncertified",
                )
            rows[worst_index] = worst_row
            active = sorted(active + [worst_index])
            continue
        if solution.multipliers:
            scale = max(1.0, max(abs(value) for value in solution.multipliers))
            negative = [
                (value, index)
                for value, index in zip(solution.multipliers, active)
                if value < -_MULTIPLIER_RELATIVE * scale
            ]
            if negative:
                _, drop = min(negative)
                active = [index for index in active if index != drop]
                rows.pop(drop, None)
                continue
        radius_value = _metric_quadratic(metric, solution.delta)
        if not math.isfinite(radius_value):
            raise ProposalProjectionError(
                "the projected trust-radius value is not finite",
                disposition="projection_non_finite",
            )
        if radius_value > radius * (1.0 + _RADIUS_RELATIVE):
            raise ProposalProjectionError(
                "the projected delta breaches the frozen D trust radius",
                disposition="trust_radius_breach",
            )
        return _MetricSolveResult(
            delta=solution.delta,
            active=tuple(active),
            multipliers=solution.multipliers,
            iterations=iteration,
            dual_residual=solution.dual_residual,
            minimum_slack=minimum_slack,
            metric_radius_value=radius_value,
            trust_radius_active=radius_value >= radius * (1.0 - _RADIUS_RELATIVE),
        )
    raise ProposalProjectionError(
        "the active-set solve exceeded its bounded iteration budget",
        disposition="projection_iteration_bound",
    )


@dataclass(frozen=True)
class WitnessChange:
    """One directional witness change under a proposed parameter delta."""

    canonical_key: str
    image_id: str
    owner_id: str
    source_membership: str
    change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "image_id": self.image_id,
            "owner_id": self.owner_id,
            "source_membership": self.source_membership,
            "change": self.change,
        }


@dataclass(frozen=True)
class ProjectionReceipt:
    """The certified owner-wise projection of one exact AdamW proposal."""

    disposition: str
    evidence_sha256: str
    layout: ParameterLayout
    projected_delta: tuple[torch.Tensor, ...]
    projected_delta_sha256: str
    active_witnesses: tuple[str, ...]
    multipliers: tuple[float, ...]
    iterations: int
    iteration_bound: int
    dual_residual: float
    tolerance: float
    trust_radius: float
    projected_trust_radius_value: float
    trust_radius_active: bool
    correction_metric_norm: float
    objective_value: float
    minimum_predicted_change: float
    minimum_unprojected_change: float
    predicted_changes: tuple[WitnessChange, ...]
    unprojected_changes: tuple[WitnessChange, ...]
    constraint_count: int
    audit_only_owner_count: int
    certified_first_order: bool
    all_finite: bool

    def flat_projected_delta(self) -> torch.Tensor:
        return torch.cat(self.projected_delta)

    @property
    def schema_version(self) -> str:
        return PROJECTION_SCHEMA_VERSION

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "disposition": self.disposition,
            "evidence_sha256": self.evidence_sha256,
            "layout": self.layout.to_dict(),
            "projected_delta_sha256": self.projected_delta_sha256,
            "active_witnesses": list(self.active_witnesses),
            "multipliers": list(self.multipliers),
            "iterations": self.iterations,
            "iteration_bound": self.iteration_bound,
            "dual_residual": self.dual_residual,
            "tolerance": self.tolerance,
            "trust_radius": self.trust_radius,
            "projected_trust_radius_value": self.projected_trust_radius_value,
            "trust_radius_active": self.trust_radius_active,
            "correction_metric_norm": self.correction_metric_norm,
            "objective_value": self.objective_value,
            "minimum_predicted_change": self.minimum_predicted_change,
            "minimum_unprojected_change": self.minimum_unprojected_change,
            "predicted_changes": [item.to_dict() for item in self.predicted_changes],
            "unprojected_changes": [
                item.to_dict() for item in self.unprojected_changes
            ],
            "constraint_count": self.constraint_count,
            "audit_only_owner_count": self.audit_only_owner_count,
            "certified_first_order": self.certified_first_order,
            "all_finite": self.all_finite,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["projected_delta_base64"] = [
            _encode_float64(tensor) for tensor in self.projected_delta
        ]
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _split_by_layout(
    flat: torch.Tensor, layout: ParameterLayout
) -> tuple[torch.Tensor, ...]:
    pieces: list[torch.Tensor] = []
    offset = 0
    for entry in layout.entries:
        pieces.append(flat[offset : offset + entry.numel].clone())
        offset += entry.numel
    return tuple(pieces)


def project_adamw_proposal(
    *, proposal: ExactAdamWProposal, witness_bank: FrozenWitnessBank
) -> ProjectionReceipt:
    """Project one exact AdamW proposal onto the frozen witness constraints."""

    return _project_admitted(_admit_evidence(proposal, witness_bank))


def _project_admitted(evidence: PreservationEvidence) -> ProjectionReceipt:
    """Derive the one canonical projection of an admitted proposal/witness pair.

    Both the public projection entry and the apply path re-enter this owner, so
    the applied delta is always the internally rederived optimum rather than a
    caller-supplied vector that merely looks certified.
    """

    proposal = evidence.proposal
    witness_bank = evidence.witness_bank
    delta0 = proposal.flat_delta()
    metric = proposal.flat_metric_denominator()

    def row_factory() -> Iterator[tuple[int, torch.Tensor]]:
        for index, _, jacobian in witness_bank.stream_constraints():
            yield index, jacobian

    iteration_bound = 4 * len(witness_bank.constraints) + 16
    result = _solve_metric_projection(
        delta0,
        row_factory,
        metric=metric,
        tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
        radius=proposal.trust_radius,
        max_iterations=iteration_bound,
    )
    predicted: list[WitnessChange] = []
    unprojected: list[WitnessChange] = []
    for _, witness, jacobian in witness_bank.stream_constraints():
        predicted.append(_change(witness, float(jacobian @ result.delta)))
        unprojected.append(_change(witness, float(jacobian @ delta0)))
    minimum_predicted = min(item.change for item in predicted)
    if minimum_predicted < -WITNESS_FIRST_ORDER_TOLERANCE * (1.0 + _APPLIED_RELATIVE):
        raise ProposalProjectionError(
            "a frozen witness constraint remains violated after projection",
            disposition="first_order_uncertified",
        )
    correction = result.delta - delta0
    objective = _metric_quadratic(metric, correction)
    values = (
        objective,
        result.dual_residual,
        result.metric_radius_value,
        minimum_predicted,
    )
    if not all(math.isfinite(value) for value in values):
        raise ProposalProjectionError(
            "the projection receipt is not finite", disposition="projection_non_finite"
        )
    projected = _split_by_layout(result.delta, proposal.layout)
    return ProjectionReceipt(
        disposition="projected_certified",
        evidence_sha256=evidence.evidence_sha256,
        layout=proposal.layout,
        projected_delta=projected,
        projected_delta_sha256=_hash_flat_tensors(
            projected, layout=proposal.layout, context="delta"
        ),
        active_witnesses=tuple(
            witness_bank.constraints[index].canonical_key for index in result.active
        ),
        multipliers=result.multipliers,
        iterations=result.iterations,
        iteration_bound=iteration_bound,
        dual_residual=result.dual_residual,
        tolerance=WITNESS_FIRST_ORDER_TOLERANCE,
        trust_radius=proposal.trust_radius,
        projected_trust_radius_value=result.metric_radius_value,
        trust_radius_active=result.trust_radius_active,
        correction_metric_norm=math.sqrt(max(objective, 0.0)),
        objective_value=objective,
        minimum_predicted_change=minimum_predicted,
        minimum_unprojected_change=min(item.change for item in unprojected),
        predicted_changes=tuple(predicted),
        unprojected_changes=tuple(unprojected),
        constraint_count=len(witness_bank.constraints),
        audit_only_owner_count=len(witness_bank.audit_only),
        certified_first_order=True,
        all_finite=True,
    )


def _change(witness: OwnerWitness, value: float) -> WitnessChange:
    return WitnessChange(
        canonical_key=witness.canonical_key,
        image_id=witness.image_id,
        owner_id=witness.owner_id,
        source_membership=witness.source_membership,
        change=value,
    )


# --- exact projected apply ----------------------------------------------------


@dataclass(frozen=True)
class ProjectedApplyReceipt:
    """The exact applied parameter change of one certified projected delta."""

    disposition: str
    evidence_sha256: str
    projection_sha256: str
    pre_parameter_sha256: str
    post_parameter_sha256: str
    applied_delta_sha256: str
    applied_delta_residual: float
    applied_representation_error: float
    applied_representation_allowance: float
    applied_undeliverable_dose: float
    applied_deliverable_allowance: float
    applied_trust_radius_value: float
    applied_minimum_first_order_change: float
    realized_changes: tuple[WitnessChange, ...]
    realized_minimum_change: float
    realized_witness_violation: bool
    degraded_witnesses: tuple[str, ...]
    eligible_for_behavioral_audits: bool
    update_count_before: int
    update_count_after: int
    optimizer_state_entries: int
    transaction_before_digest: str
    transaction_after_digest: str

    @property
    def schema_version(self) -> str:
        return APPLY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": APPLY_SCHEMA_VERSION,
            "disposition": self.disposition,
            "evidence_sha256": self.evidence_sha256,
            "projection_sha256": self.projection_sha256,
            "pre_parameter_sha256": self.pre_parameter_sha256,
            "post_parameter_sha256": self.post_parameter_sha256,
            "applied_delta_sha256": self.applied_delta_sha256,
            "applied_delta_residual": self.applied_delta_residual,
            "applied_representation_error": self.applied_representation_error,
            "applied_representation_allowance": (self.applied_representation_allowance),
            "applied_undeliverable_dose": self.applied_undeliverable_dose,
            "applied_deliverable_allowance": self.applied_deliverable_allowance,
            "applied_trust_radius_value": self.applied_trust_radius_value,
            "applied_minimum_first_order_change": (
                self.applied_minimum_first_order_change
            ),
            "realized_changes": [item.to_dict() for item in self.realized_changes],
            "realized_minimum_change": self.realized_minimum_change,
            "realized_witness_violation": self.realized_witness_violation,
            "degraded_witnesses": list(self.degraded_witnesses),
            "eligible_for_behavioral_audits": self.eligible_for_behavioral_audits,
            "update_count_before": self.update_count_before,
            "update_count_after": self.update_count_after,
            "optimizer_state_entries": self.optimizer_state_entries,
            "transaction_before_digest": self.transaction_before_digest,
            "transaction_after_digest": self.transaction_after_digest,
        }


def _representation_allowance(
    *,
    layout: ParameterLayout,
    bound: Sequence[tuple[str, torch.nn.Parameter]],
    pre_values: Sequence[torch.Tensor],
    certified: Sequence[torch.Tensor],
    cast_delta: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the exact per-coordinate application allowance and deliverability.

    ``allowance`` is the largest error the parameter dtype's own addition could
    have rounded away: half an ulp of the realized value, plus the exact error
    of casting the certified delta into that dtype, plus the declared float64
    accumulation guard of the comparison itself.  ``deliverable`` marks the
    coordinates whose certified update is large enough for that dtype's
    addition to move the parameter at all.
    """

    allowances: list[torch.Tensor] = []
    deliverable: list[torch.Tensor] = []
    guard = _ACCUMULATION_GUARD * float(torch.finfo(torch.float64).eps)
    for entry, (_, parameter), saved, exact, cast in zip(
        layout.entries, bound, pre_values, certified, cast_delta
    ):
        dtype = parameter.dtype
        pre_flat = saved.to(torch.float64).reshape(-1)
        post_flat = parameter.detach().to(torch.float64).reshape(-1)
        cast_flat = cast.detach().to(torch.float64).reshape(-1)
        realized_ulp = _representation_ulp(
            torch.maximum(pre_flat.abs(), post_flat.abs()), dtype
        )
        allowances.append(
            0.5 * realized_ulp
            + (exact - cast_flat).abs()
            + guard * (pre_flat.abs() + post_flat.abs() + exact.abs())
        )
        deliverable.append(exact.abs() >= 0.5 * _representation_ulp(pre_flat, dtype))
    return torch.cat(allowances), torch.cat(deliverable)


@torch.no_grad()
def apply_projected_delta(
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    proposal: ExactAdamWProposal,
    witness_bank: FrozenWitnessBank,
    projection: ProjectionReceipt,
    optimizer: torch.optim.Optimizer,
    transaction: TrainingStateTransaction,
    update_counter: UpdateCounter,
    realized_margin_probe: Callable[[], Mapping[str, float]],
) -> ProjectedApplyReceipt:
    """Apply only the canonical projected delta and rehash the physical change.

    The submitted receipt is compared against an internally rederived
    projection of the same admitted evidence, and only that rederived optimum
    is applied.  What is then hashed, radius-checked, and witness-checked is
    the physical post-minus-pre change in the parameter dtype, so an update the
    storage cannot represent fails and reverts instead of being receipted as
    applied.  Both realization gates are derived from the parameter dtype's own
    ulp at the actual pre and post values rather than from a chosen relative
    percentage.

    The optimizer is never stepped here: this screen applies one manual update
    and never synthesizes or continues projected AdamW moments.
    """

    evidence = _admit_evidence(proposal, witness_bank)
    if not isinstance(projection, ProjectionReceipt):
        raise ProjectedApplyError(
            "a certified projection receipt is required", disposition="invalid_field"
        )
    if (
        projection.disposition != "projected_certified"
        or projection.evidence_sha256 != evidence.evidence_sha256
        or projection.layout.layout_sha256 != proposal.layout.layout_sha256
    ):
        raise ProjectedApplyError(
            "the projection receipt does not belong to this admitted evidence",
            disposition="projection_evidence_mismatch",
        )
    if (
        _hash_flat_tensors(
            projection.projected_delta, layout=proposal.layout, context="delta"
        )
        != projection.projected_delta_sha256
    ):
        raise ProjectedApplyError(
            "the projected delta no longer matches its certified content address",
            disposition="projected_delta_digest_mismatch",
        )
    if not isinstance(transaction, TrainingStateTransaction) or not isinstance(
        update_counter, UpdateCounter
    ):
        raise ProjectedApplyError(
            "the apply path requires the bound transaction and update counter",
            disposition="invalid_field",
        )
    if not callable(realized_margin_probe):
        raise ProjectedApplyError(
            "a realized witness margin probe is required", disposition="invalid_field"
        )
    bound = tuple(named_trainable_parameters)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ProjectedApplyError(
            "a real torch optimizer is required", disposition="invalid_field"
        )
    metric = proposal.flat_metric_denominator()
    canonical = _project_admitted(evidence)
    submitted_delta = projection.flat_projected_delta()
    canonical_delta = canonical.flat_projected_delta()
    disagreement = _metric_quadratic(metric, submitted_delta - canonical_delta)
    if (
        projection.active_witnesses != canonical.active_witnesses
        or not math.isfinite(disagreement)
        or disagreement > (_CANONICAL_RELATIVE**2) * max(proposal.trust_radius, 0.0)
    ):
        raise ProjectedApplyError(
            "the submitted receipt is not the canonical minimum-change "
            "projection of this admitted evidence",
            disposition="projection_not_canonical",
        )
    _certify_transaction_ownership(
        transaction,
        bound,
        optimizer=optimizer,
        update_counter=update_counter,
        error=ProjectedApplyError,
    )
    if len(optimizer.state) != 0:
        raise ProjectedApplyError(
            "a projected apply never continues from existing optimizer moments",
            disposition="optimizer_moment_continuation",
        )
    if parameter_state_sha256(bound, proposal.layout) != proposal.pre_parameter_sha256:
        raise ProjectedApplyError(
            "the live parameters are not the captured Source state",
            disposition="source_parameter_mismatch",
        )

    # Only the internally rederived optimum is ever applied.
    applied: list[torch.Tensor] = []
    for entry, parameter, piece in zip(
        proposal.layout.entries, (p for _, p in bound), canonical.projected_delta
    ):
        applied.append(piece.reshape(entry.shape).to(parameter.dtype))
    applied_flat = torch.cat(
        [tensor.detach().to(torch.float64).reshape(-1) for tensor in applied]
    )
    if not bool(torch.isfinite(applied_flat).all()):
        raise ProjectedApplyError(
            "the applied delta is not finite", disposition="applied_non_finite"
        )

    pre_values = tuple(parameter.detach().clone() for _, parameter in bound)
    expected_post = tuple(saved + change for saved, change in zip(pre_values, applied))
    expected_sha = parameter_state_sha256(
        tuple(
            (entry.name, tensor)
            for entry, tensor in zip(proposal.layout.entries, expected_post)
        ),
        proposal.layout,
    )
    transaction_before_digest = transaction.state_digest()
    for (_, parameter), change in zip(bound, applied):
        parameter.add_(change)
    try:
        post_sha = parameter_state_sha256(bound, proposal.layout)
        if post_sha != expected_sha:
            raise ProjectedApplyError(
                "the realized parameter change is not the certified projected delta",
                disposition="applied_delta_mismatch",
            )
        # The physical delta is measured in the parameter dtype itself, so an
        # update the storage cannot represent shows up as the no-op it is.
        physical = tuple(
            (parameter.detach() - saved).to(torch.float64).reshape(-1)
            for (_, parameter), saved in zip(bound, pre_values)
        )
        physical_flat = torch.cat(physical)
        if not bool(torch.isfinite(physical_flat).all()):
            raise ProjectedApplyError(
                "the physical parameter change is not finite",
                disposition="applied_non_finite",
            )
        certified_flat = canonical.flat_projected_delta()
        applied_residual = float((physical_flat - certified_flat).abs().max())
        allowance, deliverable = _representation_allowance(
            layout=proposal.layout,
            bound=bound,
            pre_values=pre_values,
            certified=canonical.projected_delta,
            cast_delta=applied,
        )
        zero = torch.zeros_like(allowance)
        representation_error = math.sqrt(
            max(_metric_quadratic(metric, physical_flat - certified_flat), 0.0)
        )
        representation_allowance = math.sqrt(
            max(_metric_quadratic(metric, allowance), 0.0)
        )
        undeliverable_dose = math.sqrt(
            max(
                _metric_quadratic(
                    metric, torch.where(deliverable, zero, certified_flat)
                ),
                0.0,
            )
        )
        deliverable_allowance = math.sqrt(
            max(
                _metric_quadratic(metric, torch.where(deliverable, allowance, zero)),
                0.0,
            )
        )
        if not all(
            math.isfinite(value)
            for value in (
                representation_error,
                representation_allowance,
                undeliverable_dose,
                deliverable_allowance,
            )
        ):
            raise ProjectedApplyError(
                "the physical realization measurement is not finite",
                disposition="applied_non_finite",
            )
        if representation_error > representation_allowance:
            raise ProjectedApplyError(
                "the physical parameter change differs from the certified delta "
                "by more than this parameter dtype's addition could round away",
                disposition="applied_delta_mismatch",
            )
        if undeliverable_dose > deliverable_allowance:
            raise ProjectedApplyError(
                "the certified update carries more dose on coordinates this "
                "parameter dtype cannot move than its representable rounding "
                "could account for",
                disposition="applied_delta_mismatch",
            )
        applied_radius = _metric_quadratic(metric, physical_flat)
        if applied_radius > proposal.trust_radius * (1.0 + _APPLIED_RELATIVE):
            raise ProjectedApplyError(
                "the physical parameter change breaches the frozen D trust radius",
                disposition="applied_trust_radius_breach",
            )
        applied_minimum = math.inf
        for _, _witness, jacobian in witness_bank.stream_constraints():
            applied_minimum = min(applied_minimum, float(jacobian @ physical_flat))
        if applied_minimum < -WITNESS_FIRST_ORDER_TOLERANCE * (1.0 + _APPLIED_RELATIVE):
            raise ProjectedApplyError(
                "the physical parameter change is not first-order feasible",
                disposition="applied_first_order_uncertified",
            )
        realized = _realized_changes(witness_bank, realized_margin_probe)
    except ProjectedApplyError:
        for (_, parameter), saved in zip(bound, pre_values):
            parameter.copy_(saved)
        if parameter_state_sha256(bound, proposal.layout) != (
            proposal.pre_parameter_sha256
        ):
            raise ProjectedApplyError(
                "a failed projected apply could not be reverted",
                disposition="applied_revert_uncertified",
            ) from None
        raise

    # Any finite negative realized change is a violation; zero and positive
    # changes are not.  Every finite case stays audit eligible.
    degraded = tuple(item.canonical_key for item in realized if item.change < 0.0)
    minimum_realized = min(item.change for item in realized)
    update_count_before = int(update_counter.value)
    update_counter.value = update_count_before + 1
    return ProjectedApplyReceipt(
        disposition=(
            "applied_with_realized_witness_violation"
            if degraded
            else "applied_certified"
        ),
        evidence_sha256=evidence.evidence_sha256,
        projection_sha256=canonical.receipt_sha256,
        pre_parameter_sha256=proposal.pre_parameter_sha256,
        post_parameter_sha256=post_sha,
        applied_delta_sha256=_hash_flat_tensors(
            physical, layout=proposal.layout, context="delta"
        ),
        applied_delta_residual=applied_residual,
        applied_representation_error=representation_error,
        applied_representation_allowance=representation_allowance,
        applied_undeliverable_dose=undeliverable_dose,
        applied_deliverable_allowance=deliverable_allowance,
        applied_trust_radius_value=applied_radius,
        applied_minimum_first_order_change=applied_minimum,
        realized_changes=realized,
        realized_minimum_change=minimum_realized,
        realized_witness_violation=bool(degraded),
        degraded_witnesses=degraded,
        eligible_for_behavioral_audits=True,
        update_count_before=update_count_before,
        update_count_after=int(update_counter.value),
        optimizer_state_entries=len(optimizer.state),
        transaction_before_digest=transaction_before_digest,
        transaction_after_digest=transaction.state_digest(),
    )


def _realized_changes(
    witness_bank: FrozenWitnessBank,
    realized_margin_probe: Callable[[], Mapping[str, float]],
) -> tuple[WitnessChange, ...]:
    measured = realized_margin_probe()
    expected = {witness.canonical_key for witness in witness_bank.constraints}
    if not isinstance(measured, Mapping) or set(measured) != expected:
        raise ProjectedApplyError(
            "realized witness margins must cover exactly the frozen constraints",
            disposition="realized_measurement_uncertified",
        )
    changes: list[WitnessChange] = []
    for witness in witness_bank.constraints:
        value = measured[witness.canonical_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProjectedApplyError(
                f"realized margin for {witness.canonical_key} is not a number",
                disposition="realized_measurement_uncertified",
            )
        change = float(value) - witness.margin_value
        if not math.isfinite(change):
            raise ProjectedApplyError(
                f"realized margin for {witness.canonical_key} is not finite",
                disposition="realized_measurement_uncertified",
            )
        changes.append(_change(witness, change))
    return tuple(changes)


# --- durable artifacts --------------------------------------------------------


def write_exact_adamw_proposal(proposal: ExactAdamWProposal, path: Path) -> Path:
    """Write one private proposal artifact through the admitted payload."""

    if not isinstance(proposal, ExactAdamWProposal):
        raise ProposalAdmissionError(
            "an admitted exact AdamW proposal is required", disposition="invalid_field"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_payload(proposal.to_dict()))
    return target


def load_exact_adamw_proposal(path: Path) -> ExactAdamWProposal:
    """Load and revalidate one private proposal artifact."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAdmissionError(
            f"proposal artifact {target} is unreadable", disposition="invalid_artifact"
        ) from error
    return ExactAdamWProposal.from_dict(payload)


def write_frozen_witness_bank(bank: FrozenWitnessBank, directory: Path) -> Path:
    """Write the frozen witness bank and its streamed Jacobian store."""

    if not isinstance(bank, FrozenWitnessBank):
        raise ProposalAdmissionError(
            "an admitted frozen witness bank is required", disposition="invalid_field"
        )
    root = Path(directory)
    (root / "jacobians").mkdir(parents=True, exist_ok=True)
    for _, witness, jacobian in bank.stream_constraints():
        (root / "jacobians" / f"{witness.jacobian_sha256}.f64").write_bytes(
            _float64_bytes(jacobian)
        )
    (root / "bank.json").write_bytes(_canonical_payload(bank.to_dict()))
    return root


def load_frozen_witness_bank(directory: Path) -> FrozenWitnessBank:
    """Load one frozen witness bank with a lazy per-witness Jacobian reader."""

    root = Path(directory)
    try:
        payload = json.loads((root / "bank.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAdmissionError(
            f"witness bank artifact {root} is unreadable",
            disposition="invalid_artifact",
        ) from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != (
        WITNESS_BANK_SCHEMA_VERSION
    ):
        raise ProposalAdmissionError(
            "witness bank schema version differs",
            disposition="schema_version_mismatch",
        )
    if payload.get("first_order_tolerance") != WITNESS_FIRST_ORDER_TOLERANCE:
        raise ProposalAdmissionError(
            "witness bank declares a different first-order tolerance",
            disposition="witness_tolerance_mismatch",
        )

    def provider(witness: OwnerWitness) -> torch.Tensor:
        source = root / "jacobians" / f"{witness.jacobian_sha256}.f64"
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise ProposalAdmissionError(
                f"jacobian for {witness.canonical_key} is unreadable",
                disposition="invalid_artifact",
            ) from error
        if not raw or len(raw) % 8 != 0:
            raise ProposalAdmissionError(
                f"jacobian for {witness.canonical_key} is not a float64 payload",
                disposition="invalid_artifact",
            )
        return torch.from_numpy(np.frombuffer(raw, dtype="<f8").copy())

    return FrozenWitnessBank(
        binding=WitnessBinding.from_dict(payload["binding"]),
        layout=ParameterLayout.from_dict(payload["layout"]),
        constraints=tuple(
            OwnerWitness.from_dict(item) for item in payload["constraints"]
        ),
        audit_only=tuple(
            OwnerWitness.from_dict(item) for item in payload["audit_only"]
        ),
        jacobian_provider=provider,
    )


__all__ = [
    "APPLY_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "FROZEN_BETAS",
    "FROZEN_EPSILON",
    "FROZEN_LEARNING_RATE",
    "FROZEN_WEIGHT_DECAY",
    "LEGACY_M_MEMBERSHIP",
    "LEGACY_M_OWNER_CLASS",
    "PROJECTION_SCHEMA_VERSION",
    "PROPOSAL_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOURCE_MEMBERSHIPS",
    "TRAINING_RP_CONTRACTS",
    "TRUSTED_OWNER_CLASS",
    "WEAKEST_MARGIN_SELECTION",
    "WITNESS_BANK_SCHEMA_VERSION",
    "WITNESS_FIRST_ORDER_TOLERANCE",
    "AdamWProposalConfig",
    "ExactAdamWProposal",
    "FrozenWitnessBank",
    "OwnerWitness",
    "ParameterLayout",
    "ParameterLayoutEntry",
    "PreservationEvidence",
    "ProjectedApplyError",
    "ProjectedApplyReceipt",
    "ProjectionReceipt",
    "ProposalAdmissionError",
    "ProposalBinding",
    "ProposalCaptureError",
    "ProposalPreservationError",
    "ProposalProjectionError",
    "WitnessBinding",
    "WitnessChange",
    "admit_preservation_evidence",
    "apply_projected_delta",
    "capture_exact_adamw_proposal",
    "jacobian_sha256",
    "load_exact_adamw_proposal",
    "load_frozen_witness_bank",
    "parameter_state_sha256",
    "project_adamw_proposal",
    "write_exact_adamw_proposal",
    "write_frozen_witness_bank",
]
