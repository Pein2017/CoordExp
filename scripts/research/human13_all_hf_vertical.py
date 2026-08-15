"""CPU/injected composition for one private Human-13 all-HF proposal.

This owner performs no model forward, checkpoint, filesystem, CUDA, or network
action.  It joins already-admitted shared-surface tensors and detached ledgers
to the existing trajectory, compiler, fresh-AdamW, preservation, and training
transaction owners.  Only detached scalar/value evidence crosses its receipt
boundary; live autograd tensors remain private to ``PreparedAllHFVertical``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, TypeVar, cast
from weakref import ReferenceType, ref

import torch

import scripts.research.human13_greedy_compiler as compiler_owner
from scripts.research.human13_adamw_proposal_preservation import (
    FROZEN_BETAS,
    FROZEN_EPSILON,
    FROZEN_LEARNING_RATE,
    FROZEN_WEIGHT_DECAY,
    WITNESS_FIRST_ORDER_TOLERANCE,
    AdamWProposalConfig,
    FrozenWitnessBank,
    ParameterLayout,
    ProposalBinding,
    apply_projected_delta,
    capture_exact_adamw_proposal,
    parameter_state_sha256,
    project_adamw_proposal,
)
from scripts.research.human13_greedy_compiler import (
    KAPPA,
    MARGIN,
    AdmittedCompilerCompactLogits,
    CompilerLedger,
    greedy_compiler_numerator,
)
from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFSharedSurfaceIdentity,
    SampledHFGroup,
    plan_image1584_k16,
)
from scripts.research.human13_training_transaction import (
    TrainingStateSnapshot,
    TrainingStateTransaction,
    TransactionReceipt,
    UpdateCounter,
)
from scripts.research.human13_trajectory_credit import (
    TrajectoryCreditLedger,
    _require_scientific_ledger_admission,
    trajectory_score_function_numerator,
)
from src.artifacts.json_values import json_sha256


UNIT_ID = "2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical"
ARM_ID = "C-One-Image"
COMPONENT_NAMES = (
    "trajectory_score_function",
    "greedy_compiler",
    "owner_preservation_projection",
)
COMPILER_COEFFICIENT = 1.0

_RECEIPT_SEALS: dict[int, tuple[ReferenceType[object], str]] = {}
_R = TypeVar("_R", bound="_SealedReceipt")


class AllHFVerticalError(RuntimeError):
    """A private proposal failed; rollback evidence is attached when available."""

    def __init__(
        self,
        message: str,
        *,
        rollback_receipt: RollbackReceipt | None = None,
    ) -> None:
        super().__init__(message)
        self.rollback_receipt = rollback_receipt


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(detached.dtype).encode())
    hasher.update(b"\0")
    hasher.update(repr(tuple(detached.shape)).encode())
    hasher.update(b"\0")
    hasher.update(detached.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _rng_sha256(value: torch.Tensor) -> str:
    return _tensor_sha256(value)


def _cuda_rng_sha256s(values: tuple[torch.Tensor, ...] | None) -> tuple[str, ...]:
    return () if values is None else tuple(_rng_sha256(value) for value in values)


def _autograd_leaf_ids(value: torch.Tensor) -> frozenset[int]:
    """Return every requires-grad leaf identity reachable from ``value``."""

    if not value.requires_grad:
        return frozenset()
    if value.is_leaf:
        return frozenset((id(value),))
    pending = [value.grad_fn]
    # Keep node wrappers alive while traversing.  Comparing only ``id(node)``
    # is unsafe because PyTorch can recreate short-lived Python wrappers for
    # successive C++ autograd nodes and Python may then reuse their ids.
    visited: set[object] = set()
    leaves: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.requires_grad:
            leaves.add(id(variable))
        pending.extend(next_node for next_node, _ in node.next_functions)
    return frozenset(leaves)


def _require_graph_owner(
    values: Sequence[torch.Tensor],
    *,
    bound_parameter_ids: frozenset[int],
    label: str,
) -> tuple[tuple[int, ...], ...]:
    fingerprints: list[tuple[int, ...]] = []
    for value in values:
        leaves = _autograd_leaf_ids(value)
        if not leaves or not leaves.issubset(bound_parameter_ids):
            raise ValueError(
                f"{label} graph owner differs from the exact model trainable surface"
            )
        fingerprints.append(tuple(sorted(leaves)))
    return tuple(fingerprints)


def _full_model_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    if not isinstance(model, torch.nn.Module):
        raise ValueError("shared-surface model must be a live module")
    full_named = tuple(model.named_parameters())
    if (
        not full_named
        or len({name for name, _ in full_named}) != len(full_named)
        or len({id(parameter) for _, parameter in full_named}) != len(full_named)
        or any(parameter.device.type != "cpu" for _, parameter in full_named)
    ):
        raise ValueError("full model parameter registry must be unique and CPU-resident")
    return full_named


def _full_model_state_sha256(
    full_named: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    return json_sha256(
        {
            "schema_version": "human13_all_hf_full_model_state.v1",
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": parameter.requires_grad,
                    "content_sha256": _tensor_sha256(parameter),
                }
                for name, parameter in full_named
            ],
        }
    )


def _surface_fingerprint(
    model: torch.nn.Module,
    full_named: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[tuple[str, int, tuple[int, ...], str, bool, int, str], ...]:
    if model.training:
        raise ValueError("shared-surface model must be the live eval-mode module")
    current = _full_model_parameters(model)
    if tuple((name, id(parameter)) for name, parameter in current) != tuple(
        (name, id(parameter)) for name, parameter in full_named
    ):
        raise ValueError("full model parameter registry was substituted")
    return tuple(
        (
            name,
            id(parameter),
            tuple(parameter.shape),
            str(parameter.dtype),
            parameter.requires_grad,
            parameter._version,
            _tensor_sha256(parameter),
        )
        for name, parameter in current
    )


class _SealedReceipt:
    def _payload(self) -> dict[str, object]:
        raise NotImplementedError

    @property
    def content_sha256(self) -> str:
        _require_receipt(self)
        return json_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        _require_receipt(self)
        return self._payload() | {"content_sha256": self.content_sha256}


def _seal_receipt(value: _R) -> _R:
    fingerprint = json_sha256(value._payload())
    identity = id(value)

    def cleanup(_: ReferenceType[object], *, key: int = identity) -> None:
        _RECEIPT_SEALS.pop(key, None)

    _RECEIPT_SEALS[identity] = (ref(value, cleanup), fingerprint)
    return value


def _require_receipt(value: _SealedReceipt) -> None:
    entry = _RECEIPT_SEALS.get(id(value))
    if (
        entry is None
        or entry[0]() is not value
        or entry[1] != json_sha256(value._payload())
    ):
        raise ValueError("vertical receipt was not admitted or was mutated")


@dataclass(frozen=True)
class PrivateProposalReceipt(_SealedReceipt):
    surface_identity_sha256: str
    source_checkpoint_sha256: str
    surface_parameter_sha256: str
    full_model_parameter_sha256: str
    sampled_group_sha256s: tuple[str, ...]
    replay_group_sha256s: tuple[str, ...]
    request_ids: tuple[str, ...]
    trajectory_ledger_sha256: str
    compiler_ledger_sha256: str
    compiler_evidence_sha256: str
    objective_ledger_sha256: str
    component_names: tuple[str, str, str]
    trajectory_numerator: float
    compiler_numerator: float
    compiler_coefficient: float
    global_denominator: int
    total_loss: float
    compiler_kappa: float
    compiler_margin: float
    learning_rate: float
    backward_count: int
    proposal_attempt_count: int
    projected_apply_attempt_count: int
    gradient_norm: float
    gradient_all_finite: bool
    unprojected_delta_norm: float
    unprojected_delta_all_finite: bool
    projected_delta_norm: float
    projected_delta_all_finite: bool
    actual_delta_norm: float
    actual_delta_all_finite: bool
    projection_correction_metric_norm: float
    active_constraints: tuple[str, ...]
    exact_adamw_proposal_sha256: str
    projection_sha256: str
    projected_apply_sha256: str
    projected_delta_sha256: str
    actual_delta_sha256: str
    update_count_before: int
    update_count_after: int
    transaction_id: str
    transaction_before_state_digest: str
    transaction_applied_state_digest: str
    cpu_rng_before_sha256: str
    cpu_rng_applied_sha256: str
    cuda_rng_before_sha256s: tuple[str, ...]
    cuda_rng_applied_sha256s: tuple[str, ...]
    promoted_checkpoint: Literal[False]

    def __post_init__(self) -> None:
        for field in (
            "surface_identity_sha256",
            "source_checkpoint_sha256",
            "surface_parameter_sha256",
            "full_model_parameter_sha256",
            "trajectory_ledger_sha256",
            "compiler_ledger_sha256",
            "compiler_evidence_sha256",
            "objective_ledger_sha256",
            "exact_adamw_proposal_sha256",
            "projection_sha256",
            "projected_apply_sha256",
            "projected_delta_sha256",
            "actual_delta_sha256",
            "transaction_before_state_digest",
            "transaction_applied_state_digest",
            "cpu_rng_before_sha256",
            "cpu_rng_applied_sha256",
        ):
            _digest(getattr(self, field), label=field)
        for group in (
            self.sampled_group_sha256s,
            self.replay_group_sha256s,
            self.cuda_rng_before_sha256s,
            self.cuda_rng_applied_sha256s,
        ):
            for value in group:
                _digest(value, label="receipt digest")
        if (
            len(self.sampled_group_sha256s) != 4
            or len(self.replay_group_sha256s) != 4
            or len(self.request_ids) != 16
            or len(set(self.request_ids)) != 16
            or self.component_names != COMPONENT_NAMES
        ):
            raise ValueError("proposal receipt does not bind exact K16 components")
        if (
            self.compiler_coefficient != COMPILER_COEFFICIENT
            or self.compiler_kappa != KAPPA
            or self.compiler_margin != MARGIN
            or self.learning_rate != FROZEN_LEARNING_RATE
            or self.global_denominator != 16
        ):
            raise ValueError("proposal receipt constants differ from the frozen vertical")
        if (
            self.backward_count != 1
            or self.proposal_attempt_count != 1
            or self.projected_apply_attempt_count != 1
            or self.update_count_after != self.update_count_before + 1
            or self.promoted_checkpoint is not False
        ):
            raise ValueError("proposal receipt update cardinality differs")
        for field in (
            "trajectory_numerator",
            "compiler_numerator",
            "total_loss",
            "gradient_norm",
            "unprojected_delta_norm",
            "projected_delta_norm",
            "actual_delta_norm",
            "projection_correction_metric_norm",
        ):
            if _finite(getattr(self, field), label=field) < 0.0 and field.endswith("norm"):
                raise ValueError(f"{field} must be non-negative")
        if not all(
            (
                self.gradient_all_finite,
                self.unprojected_delta_all_finite,
                self.projected_delta_all_finite,
                self.actual_delta_all_finite,
            )
        ):
            raise ValueError("proposal receipt contains a non-finite diagnostic")
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("proposal receipt transaction id must be nonempty")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "human13_all_hf_private_proposal.v1",
            **{
                field: list(value) if isinstance(value, tuple) else value
                for field, value in (
                    (name, getattr(self, name))
                    for name in self.__dataclass_fields__
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PrivateProposalReceipt:
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields | {"schema_version", "content_sha256"} or value.get(
            "schema_version"
        ) != "human13_all_hf_private_proposal.v1":
            raise ValueError("private proposal receipt fields differ")
        payload = {field: value[field] for field in fields}
        for field in (
            "sampled_group_sha256s",
            "replay_group_sha256s",
            "request_ids",
            "component_names",
            "active_constraints",
            "cuda_rng_before_sha256s",
            "cuda_rng_applied_sha256s",
        ):
            payload[field] = tuple(payload[field])
        loaded = _seal_receipt(cls(**payload))
        if value["content_sha256"] != loaded.content_sha256:
            raise ValueError("private proposal receipt content hash differs")
        return loaded


@dataclass(frozen=True)
class RollbackReceipt(_SealedReceipt):
    decision: Literal["rejected_restored"]
    transaction_id: str
    before_state_digest: str
    applied_state_digest: str
    after_state_digest: str
    source_parameter_sha256: str
    restored_parameter_sha256: str
    full_model_source_sha256: str
    full_model_restored_sha256: str
    cpu_rng_before_sha256: str
    cpu_rng_after_sha256: str
    cuda_rng_before_sha256s: tuple[str, ...]
    cuda_rng_after_sha256s: tuple[str, ...]
    update_count_before: int
    update_count_after: int
    optimizer_state_entries_after: int
    live_gradient_count_after: int
    rollback_count: Literal[1]
    promoted_checkpoint: Literal[False]

    def __post_init__(self) -> None:
        for field in (
            "before_state_digest",
            "applied_state_digest",
            "after_state_digest",
            "source_parameter_sha256",
            "restored_parameter_sha256",
            "full_model_source_sha256",
            "full_model_restored_sha256",
            "cpu_rng_before_sha256",
            "cpu_rng_after_sha256",
        ):
            _digest(getattr(self, field), label=field)
        for group in (self.cuda_rng_before_sha256s, self.cuda_rng_after_sha256s):
            for value in group:
                _digest(value, label="rollback RNG digest")
        if (
            self.decision != "rejected_restored"
            or self.before_state_digest != self.after_state_digest
            or self.source_parameter_sha256 != self.restored_parameter_sha256
            or self.full_model_source_sha256 != self.full_model_restored_sha256
            or self.cpu_rng_before_sha256 != self.cpu_rng_after_sha256
            or self.cuda_rng_before_sha256s != self.cuda_rng_after_sha256s
            or self.update_count_after != self.update_count_before
            or self.optimizer_state_entries_after != 0
            or self.live_gradient_count_after != 0
            or self.rollback_count != 1
            or self.promoted_checkpoint is not False
        ):
            raise ValueError("rollback receipt does not certify exact Source restore")
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("rollback transaction id must be nonempty")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "human13_all_hf_rollback.v1",
            **{
                field: list(value) if isinstance(value, tuple) else value
                for field, value in (
                    (name, getattr(self, name))
                    for name in self.__dataclass_fields__
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RollbackReceipt:
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields | {"schema_version", "content_sha256"} or value.get(
            "schema_version"
        ) != "human13_all_hf_rollback.v1":
            raise ValueError("rollback receipt fields differ")
        payload = {field: value[field] for field in fields}
        payload["cuda_rng_before_sha256s"] = tuple(payload["cuda_rng_before_sha256s"])
        payload["cuda_rng_after_sha256s"] = tuple(payload["cuda_rng_after_sha256s"])
        loaded = _seal_receipt(cls(**payload))
        if value["content_sha256"] != loaded.content_sha256:
            raise ValueError("rollback receipt content hash differs")
        return loaded


def _exact_named_parameters(
    model: torch.nn.Module,
    named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    if not isinstance(model, torch.nn.Module) or model.training:
        raise ValueError("shared-surface model must be the live eval-mode module")
    named = tuple(named_trainable_parameters)
    if not named or any(
        not isinstance(name, str)
        or not name
        or not isinstance(parameter, torch.nn.Parameter)
        or not parameter.requires_grad
        or parameter.device.type != "cpu"
        for name, parameter in named
    ):
        raise ValueError("vertical trainable surface must contain CPU parameters")
    if len({name for name, _ in named}) != len(named) or len(
        {id(parameter) for _, parameter in named}
    ) != len(named):
        raise ValueError("vertical trainable parameters must be unique")
    model_trainable = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if tuple((name, id(parameter)) for name, parameter in model_trainable) != tuple(
        (name, id(parameter)) for name, parameter in named
    ):
        raise ValueError("model trainable surface was substituted")
    return named


def _require_optimizer_and_transaction(
    named: tuple[tuple[str, torch.nn.Parameter], ...],
    *,
    optimizer: torch.optim.Optimizer,
    transaction: TrainingStateTransaction,
    update_counter: UpdateCounter,
) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("vertical requires an exact fresh torch AdamW")
    if optimizer.state or update_counter.value != 0:
        raise ValueError("vertical requires fresh AdamW and zero updates")
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if tuple(id(parameter) for parameter in optimizer_parameters) != tuple(
        id(parameter) for _, parameter in named
    ):
        raise ValueError("optimizer parameter surface was substituted")
    for group in optimizer.param_groups:
        if (
            float(group.get("lr", float("nan"))) != FROZEN_LEARNING_RATE
            or tuple(group.get("betas", ())) != FROZEN_BETAS
            or float(group.get("eps", float("nan"))) != FROZEN_EPSILON
            or float(group.get("weight_decay", float("nan"))) != FROZEN_WEIGHT_DECAY
            or bool(group.get("amsgrad", False))
            or bool(group.get("maximize", False))
        ):
            raise ValueError("optimizer differs from fixed fresh AdamW 3e-6")
    if not isinstance(transaction, TrainingStateTransaction):
        raise ValueError("vertical requires TrainingStateTransaction")
    try:
        owned = tuple(transaction._named_parameters)
        owner_optimizer = transaction._optimizer
        owner_counter = transaction._update_counter
        active = transaction._active_transaction_id
        released = transaction._released
    except AttributeError as error:
        raise ValueError("transaction does not expose its exact owned surface") from error
    if (
        tuple((name, id(parameter)) for name, parameter in owned)
        != tuple((name, id(parameter)) for name, parameter in named)
        or owner_optimizer is not optimizer
        or owner_counter is not update_counter
        or active is not None
        or released
    ):
        raise ValueError("transaction parameter/optimizer surface was substituted")


class AllHFVerticalServices:
    """Factory for one fail-closed private objective/proposal owner."""

    def prepare(
        self,
        *,
        model: torch.nn.Module,
        named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        optimizer: torch.optim.Optimizer,
        transaction: TrainingStateTransaction,
        update_counter: UpdateCounter,
        sampled_groups: Sequence[SampledHFGroup],
        replay_groups: Sequence[GradientReplayGroup],
        replay_logprob_tensors: Mapping[str, torch.Tensor],
        trajectory_ledger: TrajectoryCreditLedger,
        compiler_ledger: CompilerLedger,
        compiler_compact_logits: AdmittedCompilerCompactLogits | None,
        witness_bank: FrozenWitnessBank,
        realized_margin_probe: Callable[[], Mapping[str, float]],
    ) -> PreparedAllHFVertical:
        return PreparedAllHFVertical(
            model=model,
            named_trainable_parameters=named_trainable_parameters,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=update_counter,
            sampled_groups=sampled_groups,
            replay_groups=replay_groups,
            replay_logprob_tensors=replay_logprob_tensors,
            trajectory_ledger=trajectory_ledger,
            compiler_ledger=compiler_ledger,
            compiler_compact_logits=compiler_compact_logits,
            witness_bank=witness_bank,
            realized_margin_probe=realized_margin_probe,
        )


class PreparedAllHFVertical:
    """Live one-shot graph owner; every terminal path restores or retains Source."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        named_trainable_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        optimizer: torch.optim.Optimizer,
        transaction: TrainingStateTransaction,
        update_counter: UpdateCounter,
        sampled_groups: Sequence[SampledHFGroup],
        replay_groups: Sequence[GradientReplayGroup],
        replay_logprob_tensors: Mapping[str, torch.Tensor],
        trajectory_ledger: TrajectoryCreditLedger,
        compiler_ledger: CompilerLedger,
        compiler_compact_logits: AdmittedCompilerCompactLogits | None,
        witness_bank: FrozenWitnessBank,
        realized_margin_probe: Callable[[], Mapping[str, float]],
    ) -> None:
        if KAPPA != 1.0 or MARGIN != 1e-4 or WITNESS_FIRST_ORDER_TOLERANCE != 1e-4:
            raise ValueError("compiler/preservation constants drifted from the vertical")
        named = _exact_named_parameters(model, named_trainable_parameters)
        full_named = _full_model_parameters(model)
        _require_optimizer_and_transaction(
            named,
            optimizer=optimizer,
            transaction=transaction,
            update_counter=update_counter,
        )
        if any(parameter.grad is not None for _, parameter in named):
            raise ValueError("vertical Source parameters carry stale gradients")
        sampled = tuple(sampled_groups)
        replays = tuple(replay_groups)
        if len(sampled) != 4 or len(replays) != 4:
            raise ValueError("vertical requires four sampled/replay groups")
        plan = plan_image1584_k16()
        identities: list[HFSharedSurfaceIdentity] = []
        request_ids: list[str] = []
        sampled_hashes: list[str] = []
        replay_hashes: list[str] = []
        for index, (sampled_group, replay_group) in enumerate(
            zip(sampled, replays, strict=True)
        ):
            if type(sampled_group) is not SampledHFGroup:
                raise ValueError("sampled group admission/type is missing")
            if type(replay_group) is not GradientReplayGroup:
                raise ValueError("replay group admission/type is missing")
            sampled_group.to_dict()
            replay_group.to_dict()
            if (
                sampled_group.plan != plan
                or sampled_group.group_index != index
                or replay_group.sampled_group is not sampled_group
                or tuple(request.seed for request in sampled_group.requests)
                != plan.seed_groups[index]
            ):
                raise ValueError("sampled/replay K16 group lineage differs")
            identities.append(sampled_group.identity)
            request_ids.extend(request.request_id for request in sampled_group.requests)
            sampled_hashes.append(sampled_group.content_sha256)
            replay_hashes.append(replay_group.content_sha256)
        if (
            any(identity != identities[0] for identity in identities)
            or identities[0].model_object_id != id(model)
            or len(set(sampled_hashes)) != 4
            or len(set(replay_hashes)) != 4
            or len(request_ids) != 16
            or len(set(request_ids)) != 16
        ):
            raise ValueError("shared surface or K16 request/group identity differs")
        layout = ParameterLayout.from_named_parameters(named)
        source_parameter_sha256 = parameter_state_sha256(named, layout)
        full_model_parameter_sha256 = _full_model_state_sha256(full_named)
        if identities[0].parameter_state_sha256 != source_parameter_sha256:
            raise ValueError("shared-surface parameter state differs from live Source")

        admitted_trajectory = _require_scientific_ledger_admission(trajectory_ledger)
        if (
            admitted_trajectory.source_sha256
            != identities[0].checkpoint_payload_sha256
        ):
            raise ValueError(
                "trajectory Source differs from the exact HF shared-surface Source"
            )
        ledger_request_ids = tuple(
            trajectory.request_id
            for image in admitted_trajectory.images
            for trajectory in image.trajectories
        )
        if (
            admitted_trajectory.logical_image_count != 1
            or admitted_trajectory.logical_k != 16
            or admitted_trajectory.logical_denominator != 16
            or tuple(image.image_id for image in admitted_trajectory.images) != (1584,)
            or ledger_request_ids != tuple(request_ids)
            or admitted_trajectory.training_repetition_penalty != 1.0
            or admitted_trajectory.seed_group_id != "35001..35016"
        ):
            raise ValueError("trajectory ledger differs from admitted image-1584 K16")

        tensors = dict(replay_logprob_tensors)
        if set(tensors) != set(replay_hashes):
            raise ValueError("live replay tensors must cover each admitted group once")
        request_tensors: dict[str, torch.Tensor] = {}
        replay_tensor_sha256s: list[str] = []
        bound_parameter_ids = frozenset(id(parameter) for _, parameter in named)
        for replay_group in replays:
            value = tensors[replay_group.content_sha256]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.ndim != 1
                or not value.requires_grad
                or not bool(torch.isfinite(value.detach()).all().item())
            ):
                raise ValueError("live replay evidence must be a finite CPU graph tensor")
            expected_tokens = replay_group.replayed_tokens
            if value.numel() != len(expected_tokens):
                raise ValueError("live replay tensor length differs from admitted tokens")
            expected = torch.tensor(
                [token.processed_logp for token in expected_tokens], dtype=value.dtype
            )
            if not torch.equal(value.detach().cpu(), expected):
                raise ValueError("live replay tensor values differ from admitted replay")
            offset = 0
            for request in replay_group.sampled_group.requests:
                end = offset + len(request.tokens)
                request_tensors[request.request_id] = value[offset:end]
                offset = end
            replay_tensor_sha256s.append(_tensor_sha256(value))
        replay_graph_fingerprints = _require_graph_owner(
            tuple(tensors[receipt_sha256] for receipt_sha256 in replay_hashes),
            bound_parameter_ids=bound_parameter_ids,
            label="trajectory",
        )

        admitted_compiler = compiler_owner._require_compiler_admission(compiler_ledger)
        if (
            admitted_compiler.source_sha256 != admitted_trajectory.source_sha256
            or admitted_compiler.manifest_sha256 != admitted_trajectory.manifest_sha256
            or admitted_compiler.acquisition_sha256
            != admitted_trajectory.acquisition_sha256
            or admitted_compiler.trajectory_credit_sha256
            != admitted_trajectory.content_sha256
            or admitted_compiler.logical_image_count
            != admitted_trajectory.logical_image_count
            or admitted_compiler.repetition_penalty
            != admitted_trajectory.training_repetition_penalty
            or tuple(image.image_id for image in admitted_compiler.images) != (1584,)
        ):
            raise ValueError("compiler ledger differs from trajectory/source lineage")
        compiler_sites = tuple(
            image.site for image in admitted_compiler.images if image.site is not None
        )
        if compiler_sites:
            if type(compiler_compact_logits) is not AdmittedCompilerCompactLogits:
                raise ValueError("compiler evidence is required before backward")
            compact = compiler_owner._require_compact_logits(
                compiler_compact_logits, admitted_compiler
            )
            raw_logits = tuple(compact._raw_logits.values())
            if any(
                value.device.type != "cpu"
                or not value.requires_grad
                or not bool(torch.isfinite(value.detach()).all().item())
                for value in raw_logits
            ):
                raise ValueError("compiler logits must be finite CPU graph tensors")
            compiler_numerator = greedy_compiler_numerator(compact, admitted_compiler)
            compiler_evidence_sha256 = compact.admission_sha256
            compiler_tensor_sha256s = tuple(_tensor_sha256(value) for value in raw_logits)
            compiler_graph_fingerprints = _require_graph_owner(
                raw_logits,
                bound_parameter_ids=bound_parameter_ids,
                label="compiler",
            )
        else:
            if compiler_compact_logits is not None:
                raise ValueError("absent compiler sites must not carry compact logits")
            compiler_numerator = compiler_owner._greedy_compiler_numerator_impl(
                {}, admitted_compiler
            )
            compiler_evidence_sha256 = json_sha256(
                {
                    "compiler_ledger_sha256": admitted_compiler.content_sha256,
                    "site_ids": [],
                    "absent_reasons": [
                        image.absent_reason for image in admitted_compiler.images
                    ],
                }
            )
            compiler_tensor_sha256s = ()
            compiler_graph_fingerprints = ()

        if type(witness_bank) is not FrozenWitnessBank or not witness_bank.constraints:
            raise ValueError("preservation evidence is required before backward")
        if (
            witness_bank.layout.layout_sha256 != layout.layout_sha256
            or witness_bank.binding.unit_id != UNIT_ID
            or witness_bank.binding.source_checkpoint_sha256
            != identities[0].checkpoint_payload_sha256
            or witness_bank.binding.manifest_sha256
            != admitted_trajectory.manifest_sha256
        ):
            raise ValueError("preservation evidence differs from the shared Source")
        for _, _, jacobian in witness_bank.stream_constraints():
            if jacobian.device.type != "cpu":
                raise ValueError("injected preservation Jacobians must remain on CPU")
        if not callable(realized_margin_probe):
            raise ValueError("preservation requires a realized margin probe")

        trajectory_numerator = trajectory_score_function_numerator(
            request_tensors, admitted_trajectory
        )
        if (
            trajectory_numerator.ndim != 0
            or compiler_numerator.ndim != 0
            or not trajectory_numerator.requires_grad
            or not bool(torch.isfinite(trajectory_numerator.detach()).item())
            or not bool(torch.isfinite(compiler_numerator.detach()).item())
        ):
            raise ValueError("objective components must be finite scalar graph values")
        trajectory_component_graph_fingerprint = _require_graph_owner(
            (trajectory_numerator,),
            bound_parameter_ids=bound_parameter_ids,
            label="trajectory component",
        )
        compiler_component_graph_fingerprint = (
            _require_graph_owner(
                (compiler_numerator,),
                bound_parameter_ids=bound_parameter_ids,
                label="compiler component",
            )
            if compiler_sites
            else ()
        )
        objective = (
            trajectory_numerator + COMPILER_COEFFICIENT * compiler_numerator
        ) / admitted_trajectory.logical_denominator
        if not objective.requires_grad or not bool(torch.isfinite(objective.detach()).item()):
            raise ValueError("complete vertical objective must be finite and differentiable")

        surface_identity_sha256 = json_sha256(identities[0].to_dict())
        objective_ledger_sha256 = json_sha256(
            {
                "schema_version": "human13_all_hf_objective_binding.v1",
                "surface_identity_sha256": surface_identity_sha256,
                "source_checkpoint_sha256": identities[0].checkpoint_payload_sha256,
                "full_model_parameter_sha256": full_model_parameter_sha256,
                "trajectory_ledger_sha256": admitted_trajectory.content_sha256,
                "compiler_ledger_sha256": admitted_compiler.content_sha256,
                "compiler_evidence_sha256": compiler_evidence_sha256,
                "sampled_group_sha256s": sampled_hashes,
                "replay_group_sha256s": replay_hashes,
                "global_denominator": admitted_trajectory.logical_denominator,
                "compiler_coefficient": COMPILER_COEFFICIENT,
            }
        )
        self._model = model
        self._named = named
        self._full_named = full_named
        self._optimizer = optimizer
        self._transaction = transaction
        self._update_counter = update_counter
        self._sampled = sampled
        self._replays = replays
        self._replay_tensors = tensors
        self._replay_tensor_sha256s = tuple(replay_tensor_sha256s)
        self._replay_graph_fingerprints = replay_graph_fingerprints
        self._compiler_compact_logits = compiler_compact_logits
        self._compiler_tensor_sha256s = compiler_tensor_sha256s
        self._compiler_graph_fingerprints = compiler_graph_fingerprints
        self._trajectory_ledger = admitted_trajectory
        self._compiler_ledger = admitted_compiler
        self._witness_bank = witness_bank
        self._realized_margin_probe = realized_margin_probe
        self._trajectory_numerator: torch.Tensor | None = trajectory_numerator
        self._compiler_numerator: torch.Tensor | None = compiler_numerator
        self._trajectory_component_graph_fingerprint = (
            trajectory_component_graph_fingerprint
        )
        self._compiler_component_graph_fingerprint = (
            compiler_component_graph_fingerprint
        )
        self._objective: torch.Tensor | None = objective
        self._identity = identities[0]
        self._surface_identity_sha256 = surface_identity_sha256
        self._surface_fingerprint = _surface_fingerprint(model, full_named)
        self._full_model_parameter_sha256 = full_model_parameter_sha256
        self._full_model_source_values = tuple(
            (name, parameter, parameter.detach().clone())
            for name, parameter in full_named
        )
        self._sampled_hashes = tuple(sampled_hashes)
        self._replay_hashes = tuple(replay_hashes)
        self._request_ids = tuple(request_ids)
        self._source_parameter_sha256 = source_parameter_sha256
        self._source_state_digest = transaction.state_digest()
        self._compiler_evidence_sha256 = compiler_evidence_sha256
        self._objective_ledger_sha256 = objective_ledger_sha256
        self._state: Literal["prepared", "applied", "rolled_back"] = "prepared"
        self._snapshot: TrainingStateSnapshot | None = None
        self._proposal_receipt: PrivateProposalReceipt | None = None
        self._rollback_receipt: RollbackReceipt | None = None

    def _revalidate(self) -> None:
        if self._state != "prepared":
            raise RuntimeError("vertical proposal is one-shot")
        if id(self._model) != self._identity.model_object_id:
            raise ValueError("shared-surface model object was substituted")
        _exact_named_parameters(self._model, self._named)
        layout = ParameterLayout.from_named_parameters(self._named)
        if (
            parameter_state_sha256(self._named, layout)
            != self._source_parameter_sha256
            or self._transaction.state_digest() != self._source_state_digest
        ):
            raise ValueError("Source state drifted after vertical preparation")
        if (
            _surface_fingerprint(self._model, self._full_named)
            != self._surface_fingerprint
            or _full_model_state_sha256(self._full_named)
            != self._full_model_parameter_sha256
        ):
            raise ValueError("full model Source identity/version/content drifted")
        if any(parameter.grad is not None for _, parameter in self._named):
            raise ValueError("Source gradients drifted after vertical preparation")
        for group, expected in zip(self._sampled, self._sampled_hashes, strict=True):
            if group.content_sha256 != expected:
                raise ValueError("sampled group was mutated after admission")
        for group, expected in zip(self._replays, self._replay_hashes, strict=True):
            if group.content_sha256 != expected:
                raise ValueError("replay group was mutated after admission")
        _require_scientific_ledger_admission(self._trajectory_ledger)
        compiler_owner._require_compiler_admission(self._compiler_ledger)
        if self._compiler_compact_logits is not None:
            compact = compiler_owner._require_compact_logits(
                self._compiler_compact_logits, self._compiler_ledger
            )
            if tuple(
                _tensor_sha256(value) for value in compact._raw_logits.values()
            ) != self._compiler_tensor_sha256s:
                raise ValueError("compiler live tensors drifted after preparation")
            if _require_graph_owner(
                tuple(compact._raw_logits.values()),
                bound_parameter_ids=frozenset(
                    id(parameter) for _, parameter in self._named
                ),
                label="compiler",
            ) != self._compiler_graph_fingerprints:
                raise ValueError("compiler graph owner drifted after preparation")
        if tuple(
            _tensor_sha256(self._replay_tensors[receipt_sha])
            for receipt_sha in self._replay_hashes
        ) != self._replay_tensor_sha256s:
            raise ValueError("replay live tensors drifted after preparation")
        if _require_graph_owner(
            tuple(
                self._replay_tensors[receipt_sha]
                for receipt_sha in self._replay_hashes
            ),
            bound_parameter_ids=frozenset(
                id(parameter) for _, parameter in self._named
            ),
            label="trajectory",
        ) != self._replay_graph_fingerprints:
            raise ValueError("trajectory graph owner drifted after preparation")
        if _require_graph_owner(
            (cast(torch.Tensor, self._trajectory_numerator),),
            bound_parameter_ids=frozenset(
                id(parameter) for _, parameter in self._named
            ),
            label="trajectory component",
        ) != self._trajectory_component_graph_fingerprint:
            raise ValueError("trajectory component graph drifted after preparation")
        if self._compiler_compact_logits is not None and _require_graph_owner(
            (cast(torch.Tensor, self._compiler_numerator),),
            bound_parameter_ids=frozenset(
                id(parameter) for _, parameter in self._named
            ),
            label="compiler component",
        ) != self._compiler_component_graph_fingerprint:
            raise ValueError("compiler component graph drifted after preparation")

    def _revalidate_before_apply(
        self,
        expected_surface_fingerprint: tuple[
            tuple[str, int, tuple[int, ...], str, bool, int, str], ...
        ],
    ) -> None:
        if (
            _surface_fingerprint(self._model, self._full_named)
            != expected_surface_fingerprint
        ):
            raise ValueError("full model surface drifted before projected apply")
        layout = ParameterLayout.from_named_parameters(self._named)
        if parameter_state_sha256(self._named, layout) != self._source_parameter_sha256:
            raise ValueError("Source parameter state drifted before projected apply")
        if _full_model_state_sha256(self._full_named) != self._full_model_parameter_sha256:
            raise ValueError("full model Source drifted before projected apply")
        bound_parameter_ids = frozenset(id(parameter) for _, parameter in self._named)
        if _require_graph_owner(
            tuple(
                self._replay_tensors[receipt_sha]
                for receipt_sha in self._replay_hashes
            ),
            bound_parameter_ids=bound_parameter_ids,
            label="trajectory",
        ) != self._replay_graph_fingerprints:
            raise ValueError("trajectory graph owner drifted before projected apply")
        if _require_graph_owner(
            (cast(torch.Tensor, self._trajectory_numerator),),
            bound_parameter_ids=bound_parameter_ids,
            label="trajectory component",
        ) != self._trajectory_component_graph_fingerprint:
            raise ValueError("trajectory component graph drifted before projected apply")
        if self._compiler_compact_logits is not None:
            compact = compiler_owner._require_compact_logits(
                self._compiler_compact_logits, self._compiler_ledger
            )
            if _require_graph_owner(
                tuple(compact._raw_logits.values()),
                bound_parameter_ids=bound_parameter_ids,
                label="compiler",
            ) != self._compiler_graph_fingerprints:
                raise ValueError("compiler graph owner drifted before projected apply")
            if _require_graph_owner(
                (cast(torch.Tensor, self._compiler_numerator),),
                bound_parameter_ids=bound_parameter_ids,
                label="compiler component",
            ) != self._compiler_component_graph_fingerprint:
                raise ValueError("compiler component graph drifted before projected apply")

    def _capture_transaction(self) -> TrainingStateTransaction:
        return TrainingStateTransaction(
            self._named,
            optimizer=self._optimizer,
            scheduler=self._transaction._scheduler,
            update_counter=self._update_counter,
            runtime=self._transaction._runtime,
            capture_cuda=self._transaction._capture_cuda,
        )

    def _proposal_binding(self) -> ProposalBinding:
        return ProposalBinding(
            unit_id=UNIT_ID,
            arm_id=ARM_ID,
            training_rp="1.0",
            seed_group="35001..35016",
            source_checkpoint_sha256=self._identity.checkpoint_payload_sha256,
            manifest_sha256=self._trajectory_ledger.manifest_sha256,
            objective_ledger_sha256=self._objective_ledger_sha256,
        )

    def backward_and_propose(self) -> PrivateProposalReceipt:
        if self._state != "prepared":
            raise RuntimeError("vertical proposal is one-shot")
        snapshot: TrainingStateSnapshot | None = None
        try:
            _require_optimizer_and_transaction(
                self._named,
                optimizer=self._optimizer,
                transaction=self._transaction,
                update_counter=self._update_counter,
            )
            snapshot = self._transaction.begin()
            self._revalidate()
            objective = cast(torch.Tensor, self._objective)
            trajectory_numerator = cast(torch.Tensor, self._trajectory_numerator)
            compiler_numerator = cast(torch.Tensor, self._compiler_numerator)
            objective.backward()
            gradients = tuple(parameter.grad for _, parameter in self._named)
            if any(gradient is None for gradient in gradients):
                raise ValueError("complete objective did not reach every trainable parameter")
            flat_gradient = torch.cat(
                [
                    cast(torch.Tensor, gradient).detach().to(torch.float64).reshape(-1)
                    for gradient in gradients
                ]
            )
            if not bool(torch.isfinite(flat_gradient).all().item()):
                raise ValueError("complete objective gradient is not finite")
            proposal = capture_exact_adamw_proposal(
                self._named,
                optimizer=self._optimizer,
                transaction=self._capture_transaction(),
                config=AdamWProposalConfig.frozen(),
                binding=self._proposal_binding(),
            )
            post_capture_surface_fingerprint = _surface_fingerprint(
                self._model, self._full_named
            )
            projection = project_adamw_proposal(
                proposal=proposal, witness_bank=self._witness_bank
            )
            self._revalidate_before_apply(post_capture_surface_fingerprint)
            applied = apply_projected_delta(
                self._named,
                proposal=proposal,
                witness_bank=self._witness_bank,
                projection=projection,
                optimizer=self._optimizer,
                transaction=self._transaction,
                update_counter=self._update_counter,
                realized_margin_probe=self._realized_margin_probe,
            )
            if (
                self._update_counter.value != 1
                or applied.update_count_before != 0
                or applied.update_count_after != 1
                or applied.optimizer_state_entries != 0
            ):
                raise ValueError("vertical did not apply exactly one private update")
            projected_flat = projection.flat_projected_delta()
            actual_pieces = tuple(
                (parameter.detach() - saved.to(parameter)).to(torch.float64).reshape(-1)
                for (_, parameter), (_, saved) in zip(
                    self._named, snapshot.parameter_values, strict=True
                )
            )
            actual_flat = torch.cat(actual_pieces)
            applied_state_digest = self._transaction.state_digest()
            receipt = _seal_receipt(
                PrivateProposalReceipt(
                    surface_identity_sha256=self._surface_identity_sha256,
                    source_checkpoint_sha256=self._identity.checkpoint_payload_sha256,
                    surface_parameter_sha256=self._source_parameter_sha256,
                    full_model_parameter_sha256=self._full_model_parameter_sha256,
                    sampled_group_sha256s=self._sampled_hashes,
                    replay_group_sha256s=self._replay_hashes,
                    request_ids=self._request_ids,
                    trajectory_ledger_sha256=self._trajectory_ledger.content_sha256,
                    compiler_ledger_sha256=self._compiler_ledger.content_sha256,
                    compiler_evidence_sha256=self._compiler_evidence_sha256,
                    objective_ledger_sha256=self._objective_ledger_sha256,
                    component_names=COMPONENT_NAMES,
                    trajectory_numerator=float(trajectory_numerator.detach().item()),
                    compiler_numerator=float(compiler_numerator.detach().item()),
                    compiler_coefficient=COMPILER_COEFFICIENT,
                    global_denominator=self._trajectory_ledger.logical_denominator,
                    total_loss=float(objective.detach().item()),
                    compiler_kappa=KAPPA,
                    compiler_margin=MARGIN,
                    learning_rate=FROZEN_LEARNING_RATE,
                    backward_count=1,
                    proposal_attempt_count=1,
                    projected_apply_attempt_count=1,
                    gradient_norm=float(flat_gradient.norm().item()),
                    gradient_all_finite=bool(torch.isfinite(flat_gradient).all().item()),
                    unprojected_delta_norm=proposal.delta_norm,
                    unprojected_delta_all_finite=bool(
                        torch.isfinite(proposal.flat_delta()).all().item()
                    ),
                    projected_delta_norm=float(projected_flat.norm().item()),
                    projected_delta_all_finite=bool(
                        torch.isfinite(projected_flat).all().item()
                    ),
                    actual_delta_norm=float(actual_flat.norm().item()),
                    actual_delta_all_finite=bool(torch.isfinite(actual_flat).all().item()),
                    projection_correction_metric_norm=projection.correction_metric_norm,
                    active_constraints=projection.active_witnesses,
                    exact_adamw_proposal_sha256=proposal.proposal_sha256,
                    projection_sha256=projection.receipt_sha256,
                    projected_apply_sha256=applied.receipt_sha256,
                    projected_delta_sha256=projection.projected_delta_sha256,
                    actual_delta_sha256=applied.applied_delta_sha256,
                    update_count_before=applied.update_count_before,
                    update_count_after=applied.update_count_after,
                    transaction_id=snapshot.transaction_id,
                    transaction_before_state_digest=snapshot.state_digest,
                    transaction_applied_state_digest=applied_state_digest,
                    cpu_rng_before_sha256=_rng_sha256(snapshot.cpu_rng_state),
                    cpu_rng_applied_sha256=_rng_sha256(torch.get_rng_state()),
                    cuda_rng_before_sha256s=_cuda_rng_sha256s(
                        snapshot.cuda_rng_states
                    ),
                    cuda_rng_applied_sha256s=_cuda_rng_sha256s(
                        self._transaction._cuda_rng_states()
                    ),
                    promoted_checkpoint=False,
                )
            )
            self._snapshot = snapshot
            self._proposal_receipt = receipt
            self._state = "applied"
            return receipt
        except Exception as error:
            rollback: RollbackReceipt | None = None
            if snapshot is not None:
                try:
                    rollback = self._reject(snapshot, applied_state_digest=None)
                except Exception as rollback_error:
                    self._state = "rolled_back"
                    raise AllHFVerticalError(
                        f"rollback failed after {type(error).__name__}: {error}: {rollback_error}"
                    ) from rollback_error
            self._state = "rolled_back"
            self._release_live_graphs()
            raise AllHFVerticalError(str(error), rollback_receipt=rollback) from error

    def _reject(
        self,
        snapshot: TrainingStateSnapshot,
        *,
        applied_state_digest: str | None,
    ) -> RollbackReceipt:
        observed_applied_digest = (
            self._transaction.state_digest()
            if applied_state_digest is None
            else applied_state_digest
        )
        transaction_receipt: TransactionReceipt = self._transaction.reject(snapshot)
        self._optimizer.zero_grad(set_to_none=True)
        self._restore_full_model_source()
        restored_parameter_sha256 = parameter_state_sha256(
            self._named, ParameterLayout.from_named_parameters(self._named)
        )
        after_digest = self._transaction.state_digest()
        receipt = _seal_receipt(
            RollbackReceipt(
                decision=cast(Literal["rejected_restored"], transaction_receipt.decision),
                transaction_id=snapshot.transaction_id,
                # The vertical receipt binds restoration to the Source captured
                # at preparation, even when the transaction had to begin after
                # detecting an externally drifted trainable value.
                before_state_digest=self._source_state_digest,
                applied_state_digest=observed_applied_digest,
                after_state_digest=after_digest,
                source_parameter_sha256=self._source_parameter_sha256,
                restored_parameter_sha256=restored_parameter_sha256,
                full_model_source_sha256=self._full_model_parameter_sha256,
                full_model_restored_sha256=_full_model_state_sha256(self._full_named),
                cpu_rng_before_sha256=_rng_sha256(snapshot.cpu_rng_state),
                cpu_rng_after_sha256=_rng_sha256(torch.get_rng_state()),
                cuda_rng_before_sha256s=_cuda_rng_sha256s(snapshot.cuda_rng_states),
                cuda_rng_after_sha256s=_cuda_rng_sha256s(
                    self._transaction._cuda_rng_states()
                ),
                update_count_before=snapshot.update_count,
                update_count_after=self._update_counter.value,
                optimizer_state_entries_after=len(self._optimizer.state),
                live_gradient_count_after=sum(
                    parameter.grad is not None for _, parameter in self._named
                ),
                rollback_count=1,
                promoted_checkpoint=False,
            )
        )
        self._rollback_receipt = receipt
        return receipt

    @torch.no_grad()
    def _restore_full_model_source(self) -> None:
        current = _full_model_parameters(self._model)
        if tuple((name, id(parameter)) for name, parameter in current) != tuple(
            (name, id(parameter)) for name, parameter, _ in self._full_model_source_values
        ):
            raise RuntimeError("full model registry drift prevents exact Source restore")
        for name, parameter, saved in self._full_model_source_values:
            if name not in dict(current):
                raise RuntimeError("full model Source parameter is missing during restore")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
        if _full_model_state_sha256(self._full_named) != self._full_model_parameter_sha256:
            raise RuntimeError("full model Source restore digest differs")

    def _release_live_graphs(self) -> None:
        self._objective = None
        self._trajectory_numerator = None
        self._compiler_numerator = None
        self._replay_tensors = {}
        self._compiler_compact_logits = None
        self._full_model_source_values = ()

    def rollback(self) -> RollbackReceipt:
        if self._state == "rolled_back":
            raise RuntimeError("vertical proposal was already rolled back")
        if self._state != "applied" or self._snapshot is None:
            raise RuntimeError("vertical proposal has not been applied")
        applied_digest = self._transaction.state_digest()
        receipt = self._reject(self._snapshot, applied_state_digest=applied_digest)
        self._state = "rolled_back"
        self._snapshot = None
        self._release_live_graphs()
        return receipt


__all__ = [
    "AllHFVerticalError",
    "AllHFVerticalServices",
    "PreparedAllHFVertical",
    "PrivateProposalReceipt",
    "RollbackReceipt",
]
