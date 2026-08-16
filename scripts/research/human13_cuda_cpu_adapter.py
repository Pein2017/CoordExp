"""Explicit CUDA Task-2 to Task-3 proposal adapter.

Task 3's :class:`PreparedAllHFVertical` remains deliberately CPU-only.  This
module reuses its lower-level scientific owners on a live CUDA parameter
surface without changing that CPU contract: trajectory and compiler numerators
stay attached to the Task-2 replay graph, AdamW capture and the full training
transaction remain bound to the CUDA objects, and only the detached witness
solver evidence is moved into the requested device for projection.  The
adapter always rolls its one private update back; checkpoint/output ownership
is outside this seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
import hashlib
from typing import Literal, NoReturn
import weakref

import torch

import scripts.research.human13_greedy_compiler as compiler_owner
from scripts.research.human13_adamw_proposal_preservation import (
    FROZEN_BETAS,
    FROZEN_EPSILON,
    FROZEN_LEARNING_RATE,
    FROZEN_WEIGHT_DECAY,
    AdamWProposalConfig,
    FrozenWitnessBank,
    ParameterLayout,
    ProposalBinding,
    ProjectedApplyReceipt,
    apply_projected_delta_on_device,
    capture_exact_adamw_proposal,
    parameter_state_sha256,
    project_adamw_proposal_on_device,
)
from scripts.research.human13_greedy_compiler import (
    AdmittedCompilerCompactLogits,
    CompilerLedger,
    greedy_compiler_numerator,
)
from scripts.research.human13_hf_shared_surface import (
    GradientReplayGroup,
    HFSharedSurfaceIdentity,
    SampledHFGroup,
)
from scripts.research.human13_hf_shared_surface_live import _observed_attention_backends
from scripts.research.human13_training_transaction import (
    TrainingStateSnapshot,
    TrainingStateTransaction,
    UpdateCounter,
)
from scripts.research.human13_trajectory_credit import (
    TrajectoryCreditLedger,
    _require_scientific_ledger_admission,
    trajectory_score_function_numerator,
)
from src.artifacts.json_values import json_sha256


class CudaAdapterError(RuntimeError):
    """The CUDA/CPU adapter seam cannot be certified."""


@dataclass(frozen=True)
class CudaRollbackFailureReceipt:
    """Primary rollback failure evidence; no second rollback is attempted."""

    phase: str
    source_state_digest: str
    observed_state_digest: str
    source_parameter_sha256: str
    observed_parameter_sha256: str
    source_version_counters: tuple[tuple[str, str, int], ...]
    observed_version_counters: tuple[tuple[str, str, int], ...]
    source_version_sha256: str
    observed_version_sha256: str
    source_cuda_rng_sha256: str | None
    observed_cuda_rng_sha256: str | None
    error: str

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                "phase": self.phase,
                "source_state_digest": self.source_state_digest,
                "observed_state_digest": self.observed_state_digest,
                "source_parameter_sha256": self.source_parameter_sha256,
                "observed_parameter_sha256": self.observed_parameter_sha256,
                "source_version_counters": [
                    list(item) for item in self.source_version_counters
                ],
                "observed_version_counters": [
                    list(item) for item in self.observed_version_counters
                ],
                "source_version_sha256": self.source_version_sha256,
                "observed_version_sha256": self.observed_version_sha256,
                "source_cuda_rng_sha256": self.source_cuda_rng_sha256,
                "observed_cuda_rng_sha256": self.observed_cuda_rng_sha256,
                "error": self.error,
            }
        )


class CudaAdapterRollbackError(CudaAdapterError):
    """The one permitted rollback attempt failed and is terminal."""

    def __init__(self, message: str, *, receipt: CudaRollbackFailureReceipt) -> None:
        super().__init__(message)
        self.rollback_receipt = receipt


def _graph_leaf_ids(value: torch.Tensor) -> frozenset[int]:
    if not value.requires_grad:
        return frozenset()
    if value.is_leaf:
        return frozenset((id(value),))
    pending = [value.grad_fn]
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


def _tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(detached.dtype).encode())
    hasher.update(b"\0")
    hasher.update(repr(tuple(detached.shape)).encode())
    hasher.update(b"\0")
    hasher.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _model_version_counters(
    model: torch.nn.Module,
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (kind, name, int(tensor._version))
        for kind, tensors in (
            ("parameter", model.named_parameters()),
            ("buffer", model.named_buffers()),
        )
        for name, tensor in tensors
    )


def _version_counters_sha256(
    counters: tuple[tuple[str, str, int], ...],
) -> str:
    return json_sha256(
        {
            "version_counters": [
                {"kind": kind, "name": name, "version": version}
                for kind, name, version in counters
            ]
        }
    )


def _safe_parameter_state_sha256(
    named: tuple[tuple[str, torch.nn.Parameter], ...],
    layout: ParameterLayout,
) -> str:
    try:
        return parameter_state_sha256(named, layout)
    except Exception:
        return json_sha256(
            {
                "parameter_state_metadata": [
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "value_sha256": _tensor_sha256(parameter),
                    }
                    for name, parameter in named
                ]
            }
        )


def _cuda_rng_sha256(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise CudaAdapterError("CUDA RNG evidence is unavailable")
    return json_sha256(
        {
            "cuda_rng_state_sha256s": [
                _tensor_sha256(state) for state in torch.cuda.get_rng_state_all()
            ]
        }
    )


def _restore_model_version_counters(
    model: torch.nn.Module,
    expected: tuple[tuple[str, str, int], ...],
) -> None:
    current_tensors = tuple(
        (kind, name, tensor)
        for kind, tensors in (
            ("parameter", model.named_parameters()),
            ("buffer", model.named_buffers()),
        )
        for name, tensor in tensors
    )
    if tuple((kind, name) for kind, name, _ in current_tensors) != tuple(
        (kind, name) for kind, name, _ in expected
    ):
        raise CudaAdapterError("full model version-counter registry drifted")
    setter = getattr(
        getattr(torch._C, "_autograd", None), "_unsafe_set_version_counter", None
    )
    if not callable(setter):
        raise CudaAdapterError("exact model version-counter restoration is unavailable")
    try:
        setter(
            [tensor for _, _, tensor in current_tensors],
            [version for _, _, version in expected],
        )
    except Exception as error:
        raise CudaAdapterError(
            "exact model version-counter restoration failed"
        ) from error
    if _model_version_counters(model) != expected:
        raise CudaAdapterError("full model version counters were not restored")


def _model_config_snapshots(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Module, object, bool, object, bool, object], ...]:
    snapshots: list[tuple[str, torch.nn.Module, object, bool, object, bool, object]] = []
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        has_attention = hasattr(config, "_attn_implementation")
        has_cache = hasattr(config, "use_cache")
        snapshots.append(
            (
                name,
                module,
                config,
                has_attention,
                getattr(config, "_attn_implementation", None),
                has_cache,
                getattr(config, "use_cache", None),
            )
        )
    return tuple(snapshots)


def _model_config_fingerprint_payload(model: torch.nn.Module) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "object_id": id(config),
            "attention_present": has_attention,
            "attention": repr(attention) if has_attention else None,
            "cache_present": has_cache,
            "use_cache": repr(use_cache) if has_cache else None,
        }
        for name, _module, config, has_attention, attention, has_cache, use_cache in
        _model_config_snapshots(model)
    ]


def _replay_tensor_snapshot(
    surface: CudaProposalInput,
) -> tuple[
    tuple[
        str,
        torch.Tensor,
        torch.Tensor,
        tuple[int, ...],
        tuple[int, ...],
        torch.dtype,
        torch.device,
        bool,
        frozenset[int],
        str,
    ],
    ...,
]:
    return tuple(
        (
            key,
            tensor,
            tensor.detach().clone(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            _graph_leaf_ids(tensor),
            _tensor_sha256(tensor),
        )
        for key, tensor in sorted(surface.replay_logprob_tensors.items())
    )


def _full_model_fingerprint(model: torch.nn.Module) -> str:
    modules = tuple(model.named_modules())
    parameters = tuple(model.named_parameters())
    buffers = tuple(model.named_buffers())
    if (
        not modules
        or len({id(module) for _, module in modules}) != len(modules)
        or len({name for name, _ in parameters}) != len(parameters)
        or len({id(parameter) for _, parameter in parameters}) != len(parameters)
        or len({name for name, _ in buffers}) != len(buffers)
        or len({id(buffer) for _, buffer in buffers}) != len(buffers)
    ):
        raise CudaAdapterError("full model registry is not unique")
    return json_sha256(
        {
            "modules": [
                {"name": name, "object_id": id(module), "training": module.training}
                for name, module in modules
            ],
            "parameters": [
                {
                    "name": name,
                    "object_id": id(parameter),
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "device": str(parameter.device),
                    "requires_grad": parameter.requires_grad,
                    "version": int(parameter._version),
                    "value_sha256": _tensor_sha256(parameter),
                }
                for name, parameter in parameters
            ],
            "buffers": [
                {
                    "name": name,
                    "object_id": id(buffer),
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                    "device": str(buffer.device),
                    "version": int(buffer._version),
                    "value_sha256": _tensor_sha256(buffer),
                }
                for name, buffer in buffers
            ],
            "configs": _model_config_fingerprint_payload(model),
        }
    )


def _objective_binding_sha256(
    surface: CudaProposalInput,
    *,
    full_model_source_sha256: str | None = None,
) -> str:
    """Address the exact admitted Task-2/Task-3 objective evidence."""

    if full_model_source_sha256 is None:
        full_model_source_sha256 = _full_model_fingerprint(surface.model)
    if (
        surface.surface_identity is None
        or surface.trajectory_ledger is None
        or surface.compiler_ledger is None
    ):
        raise CudaAdapterError("objective binding requires admitted Task2/Task3 lineage")
    admitted = _require_scientific_ledger_admission(surface.trajectory_ledger)
    compiler = compiler_owner._require_compiler_admission(surface.compiler_ledger)
    compact_payload: object = None
    compact = surface.compiler_compact_logits
    if any(image.site is not None for image in compiler.images):
        if type(compact) is not AdmittedCompilerCompactLogits:
            raise CudaAdapterError("objective binding requires compact logits")
        compact = compiler_owner._require_compact_logits(compact, compiler)
        compact_payload = {
            "admission_sha256": compact.admission_sha256,
            "site_ids": list(compact.site_ids),
            "tensor_sha256s": list(compact.compact_tensor_sha256s),
        }
    return json_sha256(
        {
            "schema_version": "human13_cuda_objective_binding.v1",
            "surface_identity": surface.surface_identity.to_dict(),
            "full_model_source_sha256": full_model_source_sha256,
            "witness_bank_sha256": surface.witness_bank.bank_sha256,
            "sampled_group_sha256s": [
                group.content_sha256 for group in surface.sampled_groups
            ],
            "replay_group_sha256s": [
                group.content_sha256 for group in surface.replay_groups
            ],
            "replay_tensor_sha256s": [
                _tensor_sha256(surface.replay_logprob_tensors[group.content_sha256])
                for group in surface.replay_groups
            ],
            "trajectory_ledger_sha256": admitted.content_sha256,
            "compiler_ledger_sha256": compiler.content_sha256,
            "logical_denominator": admitted.logical_denominator,
            "coefficients": {"trajectory": 1.0, "compiler": 1.0},
            "compiler_compact_logits": compact_payload,
        }
    )


def compute_cuda_objective_binding_sha256(
    surface: CudaProposalInput,
    *,
    full_model_source_sha256: str | None = None,
) -> str:
    """Compute the binding digest a production caller must place in its receipt."""

    return _objective_binding_sha256(
        surface, full_model_source_sha256=full_model_source_sha256
    )


def _require_finite_tensor(value: object, *, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CudaAdapterError(f"{field} must be a tensor")
    if not value.requires_grad or not bool(torch.isfinite(value.detach()).all().item()):
        raise CudaAdapterError(f"{field} must be finite and differentiable")
    return value


@dataclass(frozen=True)
class CudaProposalInput:
    """The exact object boundary between Task 2 replay and Task 3 proposal.

    ``objective`` is a lower-level injected seam for focused CPU/CUDA tests.
    A live caller should provide the admitted Task-2 groups/tensors and the
    admitted Task-3 ledgers instead; the adapter then derives the objective
    from those receipts and refuses a caller-supplied replacement.
    """

    model: torch.nn.Module
    named_trainable_parameters: tuple[tuple[str, torch.nn.Parameter], ...]
    optimizer: torch.optim.Optimizer
    transaction: TrainingStateTransaction
    update_counter: UpdateCounter
    objective: torch.Tensor | None
    witness_bank: FrozenWitnessBank
    proposal_binding: ProposalBinding
    realized_margin_probe: Callable[[], Mapping[str, float]]
    surface_identity: HFSharedSurfaceIdentity | None = None
    sampled_groups: tuple[SampledHFGroup, ...] = ()
    replay_groups: tuple[GradientReplayGroup, ...] = ()
    replay_logprob_tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)
    trajectory_ledger: TrajectoryCreditLedger | None = None
    compiler_ledger: CompilerLedger | None = None
    compiler_compact_logits: AdmittedCompilerCompactLogits | None = None


@dataclass(frozen=True)
class CudaAdapterReceipt:
    status: Literal["applied_and_rolled_back"]
    device: str
    model_object_id: int
    optimizer_object_id: int
    transaction_object_id: int
    parameter_object_ids: tuple[int, ...]
    source_parameter_sha256: str
    full_model_source_sha256: str
    full_model_applied_sha256: str
    full_model_restored_sha256: str
    applied_parameter_sha256: str
    restored_parameter_sha256: str
    source_version_counters: tuple[tuple[str, str, int], ...]
    applied_version_counters: tuple[tuple[str, str, int], ...]
    restored_version_counters: tuple[tuple[str, str, int], ...]
    source_version_sha256: str
    applied_version_sha256: str
    restored_version_sha256: str
    source_cuda_rng_sha256: str | None
    applied_cuda_rng_sha256: str | None
    restored_cuda_rng_sha256: str | None
    objective_binding_sha256: str
    source_state_digest: str
    applied_state_digest: str
    restored_state_digest: str
    proposal_sha256: str
    projection_sha256: str
    projected_apply_sha256: str
    rollback_decision: Literal["rejected_restored"]
    update_count_before: int
    update_count_after: int

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                "status": self.status,
                "device": self.device,
                "model_object_id": self.model_object_id,
                "optimizer_object_id": self.optimizer_object_id,
                "transaction_object_id": self.transaction_object_id,
                "parameter_object_ids": list(self.parameter_object_ids),
                "source_parameter_sha256": self.source_parameter_sha256,
                "full_model_source_sha256": self.full_model_source_sha256,
                "full_model_applied_sha256": self.full_model_applied_sha256,
                "full_model_restored_sha256": self.full_model_restored_sha256,
                "applied_parameter_sha256": self.applied_parameter_sha256,
                "restored_parameter_sha256": self.restored_parameter_sha256,
                "source_version_counters": [
                    list(item) for item in self.source_version_counters
                ],
                "applied_version_counters": [
                    list(item) for item in self.applied_version_counters
                ],
                "restored_version_counters": [
                    list(item) for item in self.restored_version_counters
                ],
                "source_version_sha256": self.source_version_sha256,
                "applied_version_sha256": self.applied_version_sha256,
                "restored_version_sha256": self.restored_version_sha256,
                "source_cuda_rng_sha256": self.source_cuda_rng_sha256,
                "applied_cuda_rng_sha256": self.applied_cuda_rng_sha256,
                "restored_cuda_rng_sha256": self.restored_cuda_rng_sha256,
                "objective_binding_sha256": self.objective_binding_sha256,
                "source_state_digest": self.source_state_digest,
                "applied_state_digest": self.applied_state_digest,
                "restored_state_digest": self.restored_state_digest,
                "proposal_sha256": self.proposal_sha256,
                "projection_sha256": self.projection_sha256,
                "projected_apply_sha256": self.projected_apply_sha256,
                "rollback_decision": self.rollback_decision,
                "update_count_before": self.update_count_before,
                "update_count_after": self.update_count_after,
            }
        )


@dataclass(frozen=True)
class CudaPrivateProposalReceipt:
    """Detached evidence for the one proposal while it remains privately applied."""

    status: Literal["private_proposal_applied"]
    device: str
    objective_binding_sha256: str
    source_parameter_sha256: str
    applied_parameter_sha256: str
    full_model_source_sha256: str
    full_model_applied_sha256: str
    source_state_digest: str
    applied_state_digest: str
    source_version_sha256: str
    applied_version_sha256: str
    source_cuda_rng_sha256: str | None
    applied_cuda_rng_sha256: str | None
    proposal_sha256: str
    projection_sha256: str
    projected_apply_sha256: str
    update_count_before: int
    update_count_after: int

    @property
    def content_sha256(self) -> str:
        return json_sha256(
            {
                field: getattr(self, field)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class _CudaSourceState:
    layout: ParameterLayout
    source_parameter_sha256: str
    source_version_counters: tuple[tuple[str, str, int], ...]
    source_version_sha256: str
    source_cuda_rng_sha256: str | None
    source_state_digest: str
    snapshot: TrainingStateSnapshot


@dataclass(frozen=True)
class _CudaPendingProposal:
    source: _CudaSourceState
    proposal_sha256: str
    projection_sha256: str
    projected_apply_sha256: str
    applied_parameter_sha256: str
    applied_version_counters: tuple[tuple[str, str, int], ...]
    applied_version_sha256: str
    applied_cuda_rng_sha256: str | None
    applied_state_digest: str
    full_model_applied_sha256: str
    update_count_before: int
    update_count_after: int


_RECEIPT_ISSUER = object()
_RECEIPT_ADMISSIONS: weakref.WeakValueDictionary[str, CudaAdapterReceipt] = (
    weakref.WeakValueDictionary()
)


def _seal_receipt(receipt: CudaAdapterReceipt, issuer: object) -> None:
    if issuer is not _RECEIPT_ISSUER:
        raise CudaAdapterError("adapter receipt issuer is not authorized")
    _RECEIPT_ADMISSIONS[receipt.content_sha256] = receipt


def require_admitted_receipt(value: object) -> CudaAdapterReceipt:
    """Require the exact in-process receipt issued by this adapter run."""

    if type(value) is not CudaAdapterReceipt:
        raise CudaAdapterError("a sealed CUDA adapter receipt is required")
    admitted = _RECEIPT_ADMISSIONS.get(value.content_sha256)
    if admitted is not value:
        raise CudaAdapterError("CUDA adapter receipt was not issued by this run")
    return value


class CudaHFVerticalAdapter:
    """Compose one CUDA objective/proposal while preserving exact rollback."""

    def __init__(self, surface: CudaProposalInput) -> None:
        if not isinstance(surface, CudaProposalInput):
            raise CudaAdapterError("adapter requires CudaProposalInput")
        self._surface = surface
        self._named = tuple(surface.named_trainable_parameters)
        self._device = self._validate_surface()
        self._full_model_source_sha256 = _full_model_fingerprint(surface.model)
        self._full_model_source_values = tuple(
            (name, parameter, parameter.detach().clone(), parameter.requires_grad)
            for name, parameter in surface.model.named_parameters()
        )
        self._full_model_source_buffers = tuple(
            (name, buffer, buffer.detach().clone())
            for name, buffer in surface.model.named_buffers()
        )
        self._full_model_source_versions = _model_version_counters(surface.model)
        self._full_model_source_modes = tuple(
            (name, module, module.training)
            for name, module in surface.model.named_modules()
        )
        self._full_model_source_configs = _model_config_snapshots(surface.model)
        self._task2_replay_source = _replay_tensor_snapshot(surface)
        self._objective = self._derive_objective()
        if self._has_task2_lineage():
            expected_binding = _objective_binding_sha256(
                surface, full_model_source_sha256=self._full_model_source_sha256
            )
            if surface.proposal_binding.objective_ledger_sha256 != expected_binding:
                raise CudaAdapterError("Task2/Task3 objective binding differs")
            self._objective_binding_sha256 = expected_binding
        else:
            self._objective_binding_sha256 = surface.proposal_binding.objective_ledger_sha256
        self._lifecycle_state: Literal["prepared", "applied", "rolled_back"] = (
            "prepared"
        )
        self._pending_proposal: _CudaPendingProposal | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def objective(self) -> torch.Tensor:
        return self._objective

    def _has_task2_lineage(self) -> bool:
        surface = self._surface
        return bool(
            surface.sampled_groups
            or surface.replay_groups
            or surface.replay_logprob_tensors
            or surface.surface_identity is not None
            or surface.trajectory_ledger is not None
            or surface.compiler_ledger is not None
        )

    def _validate_surface(self) -> torch.device:
        surface = self._surface
        model = surface.model
        if not isinstance(model, torch.nn.Module) or model.training:
            raise CudaAdapterError("CUDA proposal model must be an eval-mode module")
        if not self._named:
            raise CudaAdapterError("CUDA proposal requires trainable parameters")
        model_trainable = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if tuple((name, id(parameter)) for name, parameter in model_trainable) != tuple(
            (name, id(parameter)) for name, parameter in self._named
        ):
            raise CudaAdapterError("model trainable parameter identity differs")
        devices = {parameter.device for _, parameter in self._named}
        if len(devices) != 1:
            raise CudaAdapterError("trainable parameters must share one device")
        device = next(iter(devices))
        if device.type not in {"cpu", "cuda"}:
            raise CudaAdapterError("proposal surface must be CPU or CUDA")
        if any(not parameter.requires_grad for _, parameter in self._named):
            raise CudaAdapterError("all adapter parameters must require gradients")
        optimizer = surface.optimizer
        if type(optimizer) is not torch.optim.AdamW:
            raise CudaAdapterError("adapter requires exact fresh AdamW")
        if optimizer.state:
            raise CudaAdapterError("adapter requires a fresh optimizer")
        optimizer_parameters = tuple(
            parameter for group in optimizer.param_groups for parameter in group["params"]
        )
        if tuple(id(parameter) for parameter in optimizer_parameters) != tuple(
            id(parameter) for _, parameter in self._named
        ):
            raise CudaAdapterError("optimizer parameter identity differs")
        for group in optimizer.param_groups:
            if (
                float(group.get("lr", float("nan"))) != FROZEN_LEARNING_RATE
                or tuple(group.get("betas", ())) != FROZEN_BETAS
                or float(group.get("eps", float("nan"))) != FROZEN_EPSILON
                or float(group.get("weight_decay", float("nan"))) != FROZEN_WEIGHT_DECAY
                or bool(group.get("amsgrad", False))
                or bool(group.get("maximize", False))
            ):
                raise CudaAdapterError("optimizer differs from fixed fresh AdamW")
        transaction = surface.transaction
        if not isinstance(transaction, TrainingStateTransaction):
            raise CudaAdapterError("adapter requires TrainingStateTransaction")
        try:
            owned = tuple(transaction._named_parameters)
            if transaction._optimizer is not optimizer:
                raise CudaAdapterError("transaction optimizer ownership differs")
            if tuple((name, id(parameter)) for name, parameter in owned) != tuple(
                (name, id(parameter)) for name, parameter in self._named
            ):
                raise CudaAdapterError("transaction parameter ownership differs")
            if transaction._update_counter is not surface.update_counter:
                raise CudaAdapterError("transaction counter ownership differs")
            if transaction._active_transaction_id is not None or transaction._released:
                raise CudaAdapterError("transaction ownership differs")
        except AttributeError as error:
            raise CudaAdapterError("transaction ownership is not inspectable") from error
        if surface.update_counter.value != 0:
            raise CudaAdapterError("adapter requires zero prior updates")
        if device.type == "cuda" and not transaction._capture_cuda:
            raise CudaAdapterError("CUDA transaction must capture CUDA RNG state")
        if any(parameter.grad is not None for _, parameter in self._named):
            raise CudaAdapterError("adapter source parameters carry stale gradients")
        if not callable(surface.realized_margin_probe):
            raise CudaAdapterError("adapter requires a realized margin probe")
        layout = ParameterLayout.from_named_parameters(self._named)
        current_parameter_sha256 = parameter_state_sha256(self._named, layout)
        if surface.surface_identity is not None:
            if surface.surface_identity.model_object_id != id(model):
                raise CudaAdapterError("Task2 model object identity differs")
            if surface.surface_identity.parameter_state_sha256 != current_parameter_sha256:
                raise CudaAdapterError("Task2 parameter state identity differs")
            if any(parameter.dtype != torch.bfloat16 for _, parameter in self._named):
                raise CudaAdapterError("Task2 model parameters must remain bfloat16")
            backends = _observed_attention_backends(model)
            if not backends or any(value != "flash_attention_2" for value in backends):
                raise CudaAdapterError(
                    "Task2 model attention backend must remain flash_attention_2"
                )
            if getattr(getattr(model, "config", None), "use_cache", False) is not False:
                raise CudaAdapterError("Task2 model cache configuration must remain disabled")
        if surface.witness_bank.layout.layout_sha256 != layout.layout_sha256:
            raise CudaAdapterError("witness layout differs from live parameters")
        if surface.proposal_binding.source_checkpoint_sha256 != (
            surface.witness_bank.binding.source_checkpoint_sha256
        ) or surface.proposal_binding.manifest_sha256 != surface.witness_bank.binding.manifest_sha256:
            raise CudaAdapterError("proposal and witness Source lineage differs")
        if surface.objective is not None:
            if surface.surface_identity is not None or surface.sampled_groups:
                raise CudaAdapterError(
                    "Task2/Task3 lineage must derive the objective, not replace it"
                )
            objective = _require_finite_tensor(surface.objective, field="objective")
            if not _graph_leaf_ids(objective) <= frozenset(
                id(parameter) for _, parameter in self._named
            ):
                raise CudaAdapterError("objective graph owner differs from live parameters")
        self._validate_task2_lineage(device)
        return device

    def _validate_task2_lineage(self, device: torch.device) -> None:
        surface = self._surface
        has_any = bool(
            surface.sampled_groups
            or surface.replay_groups
            or surface.replay_logprob_tensors
            or surface.surface_identity is not None
            or surface.trajectory_ledger is not None
            or surface.compiler_ledger is not None
        )
        if not has_any:
            return
        if (
            surface.surface_identity is None
            or len(surface.sampled_groups) != 4
            or len(surface.replay_groups) != 4
            or surface.trajectory_ledger is None
            or surface.compiler_ledger is None
        ):
            raise CudaAdapterError("Task2/Task3 admitted lineage is incomplete")
        identities: list[HFSharedSurfaceIdentity] = []
        replay_hashes: list[str] = []
        request_ids: list[str] = []
        for index, (sampled, replay) in enumerate(
            zip(surface.sampled_groups, surface.replay_groups, strict=True)
        ):
            if type(sampled) is not SampledHFGroup or type(replay) is not GradientReplayGroup:
                raise CudaAdapterError("Task2 groups must be exact admitted receipts")
            sampled.to_dict()
            replay.to_dict()
            if (
                sampled.group_index != index
                or replay.sampled_group is not sampled
                or sampled.identity != surface.surface_identity
                or replay.sampled_group.identity != surface.surface_identity
            ):
                raise CudaAdapterError("Task2 group identity or order differs")
            identities.append(sampled.identity)
            replay_hashes.append(replay.content_sha256)
            request_ids.extend(request.request_id for request in sampled.requests)
        if (
            surface.surface_identity.model_object_id != id(surface.model)
            or any(identity != identities[0] for identity in identities)
            or len(set(request_ids)) != len(request_ids)
        ):
            raise CudaAdapterError("Task2 shared-surface model identity differs")
        tensors = dict(surface.replay_logprob_tensors)
        if set(tensors) != set(replay_hashes):
            raise CudaAdapterError("Task2 replay tensor keys differ from admitted groups")
        parameter_ids = frozenset(id(parameter) for _, parameter in self._named)
        for replay in surface.replay_groups:
            tensor = _require_finite_tensor(
                tensors[replay.content_sha256], field="Task2 replay logprob tensor"
            )
            if tensor.device != device or tensor.ndim != 1:
                raise CudaAdapterError("Task2 replay tensor device/shape differs")
            expected = torch.tensor(
                [token.processed_logp for token in replay.replayed_tokens],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if tensor.numel() != expected.numel() or not torch.equal(
                tensor.detach(), expected
            ):
                raise CudaAdapterError("Task2 replay tensor values differ from receipt")
            if not _graph_leaf_ids(tensor) <= parameter_ids:
                raise CudaAdapterError("Task2 replay graph owner differs from parameters")
        admitted = _require_scientific_ledger_admission(surface.trajectory_ledger)
        if (
            admitted.source_sha256 != surface.surface_identity.checkpoint_payload_sha256
            or admitted.logical_image_count != 1
            or admitted.logical_k != 16
            or tuple(image.image_id for image in admitted.images) != (1584,)
            or tuple(
                trajectory.request_id
                for image in admitted.images
                for trajectory in image.trajectories
            ) != tuple(request_ids)
        ):
            raise CudaAdapterError("Task2 trajectory ledger lineage differs")
        compiler = compiler_owner._require_compiler_admission(surface.compiler_ledger)
        if (
            compiler.source_sha256 != admitted.source_sha256
            or compiler.manifest_sha256 != admitted.manifest_sha256
            or compiler.trajectory_credit_sha256 != admitted.content_sha256
        ):
            raise CudaAdapterError("Task3 compiler ledger lineage differs")
        sites = tuple(image.site for image in compiler.images if image.site is not None)
        if sites:
            if type(surface.compiler_compact_logits) is not AdmittedCompilerCompactLogits:
                raise CudaAdapterError("Task3 compiler compact logits are required")
            compact = compiler_owner._require_compact_logits(
                surface.compiler_compact_logits, compiler
            )
            for value in compact._raw_logits.values():
                try:
                    _require_finite_tensor(value, field="Task3 compiler compact logits")
                except CudaAdapterError as error:
                    raise CudaAdapterError(
                        "Task3 compiler compact logits must retain a live graph"
                    ) from error
                leaves = _graph_leaf_ids(value)
                if (
                    value.device != device
                    or not leaves
                    or not leaves <= parameter_ids
                ):
                    raise CudaAdapterError("Task3 compiler graph/device owner differs")
        elif surface.compiler_compact_logits is not None:
            raise CudaAdapterError("Task3 compiler compact logits are stale without sites")

    def _derive_objective(self) -> torch.Tensor:
        surface = self._surface
        if surface.objective is not None:
            return surface.objective
        if surface.trajectory_ledger is None or surface.compiler_ledger is None:
            raise CudaAdapterError("objective or admitted Task2/Task3 evidence is required")
        admitted = _require_scientific_ledger_admission(surface.trajectory_ledger)
        tensors = dict(surface.replay_logprob_tensors)
        request_tensors: dict[str, torch.Tensor] = {}
        for replay in surface.replay_groups:
            value = tensors[replay.content_sha256]
            offset = 0
            for request in replay.sampled_group.requests:
                end = offset + len(request.tokens)
                request_tensors[request.request_id] = value[offset:end]
                offset = end
        trajectory = trajectory_score_function_numerator(
            request_tensors, surface.trajectory_ledger
        )
        compiler = surface.compiler_ledger
        sites = tuple(image.site for image in compiler.images if image.site is not None)
        if sites:
            if surface.compiler_compact_logits is None:
                raise CudaAdapterError("compiler compact logits are required")
            compiler_term = greedy_compiler_numerator(
                surface.compiler_compact_logits, compiler
            )
        else:
            compiler_term = trajectory.new_zeros(())
        objective = (trajectory + compiler_term) / admitted.logical_denominator
        return _require_finite_tensor(objective, field="derived objective")

    def _capture_transaction(self) -> TrainingStateTransaction:
        transaction = self._surface.transaction
        return TrainingStateTransaction(
            self._named,
            optimizer=self._surface.optimizer,
            scheduler=transaction._scheduler,
            update_counter=self._surface.update_counter,
            runtime=transaction._runtime,
            capture_cuda=transaction._capture_cuda,
        )

    def _restore_full_model_source(self) -> None:
        model = self._surface.model
        if _full_model_fingerprint(model) == self._full_model_source_sha256:
            return
        current_parameters = tuple(model.named_parameters())
        if tuple((name, id(parameter)) for name, parameter in current_parameters) != tuple(
            (name, id(parameter)) for name, parameter, _, _ in self._full_model_source_values
        ):
            raise CudaAdapterError("full model parameter registry drifted")
        current_modules = tuple(model.named_modules())
        if tuple((name, id(module)) for name, module in current_modules) != tuple(
            (name, id(module)) for name, module, _ in self._full_model_source_modes
        ):
            raise CudaAdapterError("full model module registry drifted")
        current_buffers = tuple(model.named_buffers())
        if tuple((name, id(buffer)) for name, buffer in current_buffers) != tuple(
            (name, id(buffer)) for name, buffer, _ in self._full_model_source_buffers
        ):
            raise CudaAdapterError("full model buffer registry drifted")
        current_configs = _model_config_snapshots(model)
        if tuple((name, id(config)) for name, _module, config, *_ in current_configs) != tuple(
            (name, id(config))
            for name, _module, config, *_ in self._full_model_source_configs
        ):
            raise CudaAdapterError("full model config registry drifted")
        with torch.no_grad():
            for name, parameter, saved, requires_grad in self._full_model_source_values:
                del name
                if (
                    tuple(parameter.shape) != tuple(saved.shape)
                    or parameter.dtype != saved.dtype
                    or parameter.device != saved.device
                    or parameter.stride() != saved.stride()
                ):
                    parameter.data = saved.detach().clone(memory_format=torch.preserve_format)
                else:
                    parameter.copy_(saved)
                parameter.requires_grad_(requires_grad)
            for name, buffer, saved in self._full_model_source_buffers:
                del name
                if (
                    tuple(buffer.shape) != tuple(saved.shape)
                    or buffer.dtype != saved.dtype
                    or buffer.device != saved.device
                    or buffer.stride() != saved.stride()
                ):
                    buffer.data = saved.detach().clone(memory_format=torch.preserve_format)
                else:
                    buffer.copy_(saved)
        _restore_model_version_counters(model, self._full_model_source_versions)
        # Assign the flag directly rather than calling ``Module.train``: the
        # latter recursively rewrites child modes and can destroy a valid
        # mixed train/eval module-mode registry captured at the Source seam.
        for _name, module, training in self._full_model_source_modes:
            module.training = training
        for (
            name,
            module,
            config,
            has_attention,
            attention,
            has_cache,
            use_cache,
        ) in self._full_model_source_configs:
            del name
            current_config = getattr(module, "config", None)
            if current_config is not config:
                raise CudaAdapterError("full model config registry drifted")
            if has_attention:
                setattr(config, "_attn_implementation", attention)
            elif hasattr(config, "_attn_implementation"):
                delattr(config, "_attn_implementation")
            if has_cache:
                setattr(config, "use_cache", use_cache)
            elif hasattr(config, "use_cache"):
                delattr(config, "use_cache")
        if _full_model_fingerprint(model) != self._full_model_source_sha256:
            raise CudaAdapterError("full model Source restoration differs")

    def _full_model_storage_metadata_drifted(self) -> bool:
        model = self._surface.model
        current_parameters = dict(model.named_parameters())
        current_buffers = dict(model.named_buffers())
        source_parameters = {
            name: saved for name, _parameter, saved, _requires_grad in self._full_model_source_values
        }
        source_buffers = {
            name: saved for name, _buffer, saved in self._full_model_source_buffers
        }
        if set(current_parameters) != set(source_parameters) or set(current_buffers) != set(
            source_buffers
        ):
            return True
        return any(
            tuple(current_parameters[name].shape) != tuple(source_parameters[name].shape)
            or current_parameters[name].dtype != source_parameters[name].dtype
            or current_parameters[name].device != source_parameters[name].device
            or current_parameters[name].stride() != source_parameters[name].stride()
            for name in source_parameters
        ) or any(
            tuple(current_buffers[name].shape) != tuple(source_buffers[name].shape)
            or current_buffers[name].dtype != source_buffers[name].dtype
            or current_buffers[name].device != source_buffers[name].device
            or current_buffers[name].stride() != source_buffers[name].stride()
            for name in source_buffers
        )

    def _validate_task2_replay_source(self) -> None:
        expected = self._task2_replay_source
        if not expected:
            return
        current = _replay_tensor_snapshot(self._surface)
        if len(current) != len(expected):
            raise CudaAdapterError("Task2 replay evidence changed during probe")
        for current_item, expected_item in zip(current, expected, strict=True):
            if (
                current_item[0] != expected_item[0]
                or current_item[1] is not expected_item[1]
                or current_item[3:] != expected_item[3:]
            ):
                raise CudaAdapterError("Task2 replay evidence changed during probe")

    def _restore_task2_replay_source(self) -> None:
        if not self._task2_replay_source:
            return
        mapping = self._surface.replay_logprob_tensors
        for (
            key,
            tensor,
            saved,
            shape,
            stride,
            dtype,
            device,
            requires_grad,
            _leaves,
            _digest,
        ) in self._task2_replay_source:
            current = mapping.get(key)
            if current is not tensor:
                if not isinstance(mapping, MutableMapping):
                    raise CudaAdapterError("Task2 replay mapping identity drifted")
                mapping[key] = tensor
                current = tensor
            if not isinstance(current, torch.Tensor):
                raise CudaAdapterError("Task2 replay tensor mapping is invalid")
            with torch.no_grad():
                if (
                    tuple(current.shape) != shape
                    or tuple(current.stride()) != stride
                    or current.dtype != dtype
                    or current.device != device
                ):
                    current.data = saved.detach().clone(memory_format=torch.preserve_format)
                else:
                    current.copy_(saved)
                current.requires_grad_(requires_grad)
        self._validate_task2_replay_source()

    def _source_state(self) -> _CudaSourceState:
        surface = self._surface
        transaction = surface.transaction
        layout = ParameterLayout.from_named_parameters(self._named)
        source_version_counters = _model_version_counters(surface.model)
        return _CudaSourceState(
            layout=layout,
            source_parameter_sha256=parameter_state_sha256(self._named, layout),
            source_version_counters=source_version_counters,
            source_version_sha256=_version_counters_sha256(source_version_counters),
            source_cuda_rng_sha256=_cuda_rng_sha256(self._device),
            source_state_digest=transaction.state_digest(),
            snapshot=transaction.begin(),
        )

    def _failure_receipt(
        self,
        *,
        phase: str,
        detail: Exception,
        source: _CudaSourceState,
    ) -> CudaRollbackFailureReceipt:
        surface = self._surface
        observed_versions = _model_version_counters(surface.model)
        return CudaRollbackFailureReceipt(
            phase=phase,
            source_state_digest=source.source_state_digest,
            observed_state_digest=surface.transaction.state_digest(),
            source_parameter_sha256=source.source_parameter_sha256,
            observed_parameter_sha256=_safe_parameter_state_sha256(
                self._named, source.layout
            ),
            source_version_counters=source.source_version_counters,
            observed_version_counters=observed_versions,
            source_version_sha256=source.source_version_sha256,
            observed_version_sha256=_version_counters_sha256(observed_versions),
            source_cuda_rng_sha256=source.source_cuda_rng_sha256,
            observed_cuda_rng_sha256=_cuda_rng_sha256(self._device),
            error=f"{type(detail).__name__}: {detail}",
        )

    def _raise_after_failure(
        self,
        error: Exception,
        *,
        source: _CudaSourceState,
        rollback_attempted: bool,
        full_model_restore_attempted: bool,
    ) -> NoReturn:
        surface = self._surface
        transaction = surface.transaction
        snapshot = source.snapshot
        rollback_error: Exception | None = None
        restore_error: Exception | None = None
        replay_restore_error: Exception | None = None
        try:
            self._restore_task2_replay_source()
        except Exception as replay_exc:
            replay_restore_error = replay_exc
        storage_metadata_drifted = self._full_model_storage_metadata_drifted()
        if not full_model_restore_attempted:
            try:
                self._restore_full_model_source()
            except Exception as restore_exc:
                restore_error = restore_exc
            else:
                if storage_metadata_drifted:
                    restore_error = CudaAdapterError(
                        "full model storage metadata drift was repaired after failure"
                    )
        elif isinstance(error, CudaAdapterError):
            restore_error = error
        if (
            transaction._active_transaction_id == snapshot.transaction_id
            and not rollback_attempted
        ):
            try:
                transaction.reject(snapshot)
            except Exception as rollback_exc:
                rollback_error = rollback_exc
        elif (
            transaction._active_transaction_id == snapshot.transaction_id
            and rollback_attempted
        ):
            rollback_error = error
        try:
            self._restore_full_model_source()
        except Exception as restore_exc:
            if restore_error is None:
                restore_error = restore_exc
        if transaction._active_transaction_id == snapshot.transaction_id:
            transaction._active_transaction_id = None
        surface.optimizer.zero_grad(set_to_none=True)
        self._pending_proposal = None
        self._lifecycle_state = "rolled_back"
        if restore_error is not None:
            receipt = self._failure_receipt(
                phase="full_model_restore", detail=restore_error, source=source
            )
            raise CudaAdapterRollbackError(
                f"full model rollback failed after {error}: {restore_error}",
                receipt=receipt,
            ) from restore_error
        if replay_restore_error is not None:
            receipt = self._failure_receipt(
                phase="task2_replay_restore",
                detail=replay_restore_error,
                source=source,
            )
            raise CudaAdapterRollbackError(
                f"Task2 replay rollback failed after {error}: {replay_restore_error}",
                receipt=receipt,
            ) from replay_restore_error
        if rollback_error is not None:
            receipt = self._failure_receipt(
                phase="transaction_reject", detail=rollback_error, source=source
            )
            raise CudaAdapterRollbackError(
                f"adapter rollback failed after {error}: {rollback_error}",
                receipt=receipt,
            ) from rollback_error
        if isinstance(error, CudaAdapterError):
            raise error
        raise CudaAdapterError(str(error)) from error

    def apply_private_proposal(self) -> CudaPrivateProposalReceipt:
        """Apply exactly one proposal and retain it for private checkpoint/audit."""

        if self._lifecycle_state == "applied":
            raise CudaAdapterError("private proposal was already applied")
        if self._lifecycle_state == "rolled_back":
            raise CudaAdapterError("private proposal was already rolled back")
        surface = self._surface
        source = self._source_state()
        try:
            self._objective.backward()
            if any(parameter.grad is None for _, parameter in self._named):
                raise CudaAdapterError("objective did not reach every trainable parameter")
            proposal = capture_exact_adamw_proposal(
                self._named,
                optimizer=surface.optimizer,
                transaction=self._capture_transaction(),
                config=AdamWProposalConfig.frozen(),
                binding=surface.proposal_binding,
            )
            projection = project_adamw_proposal_on_device(
                proposal=proposal,
                witness_bank=surface.witness_bank,
                device=self._device,
            )
            applied: ProjectedApplyReceipt = apply_projected_delta_on_device(
                self._named,
                proposal=proposal,
                witness_bank=surface.witness_bank,
                projection=projection,
                optimizer=surface.optimizer,
                transaction=surface.transaction,
                update_counter=surface.update_counter,
                realized_margin_probe=surface.realized_margin_probe,
                device=self._device,
            )
            if applied.update_count_before != 0 or applied.update_count_after != 1:
                raise CudaAdapterError("adapter did not apply exactly one update")
            self._validate_task2_replay_source()
            applied_versions = _model_version_counters(surface.model)
            pending = _CudaPendingProposal(
                source=source,
                proposal_sha256=proposal.proposal_sha256,
                projection_sha256=projection.receipt_sha256,
                projected_apply_sha256=applied.receipt_sha256,
                applied_parameter_sha256=parameter_state_sha256(
                    self._named, source.layout
                ),
                applied_version_counters=applied_versions,
                applied_version_sha256=_version_counters_sha256(applied_versions),
                applied_cuda_rng_sha256=_cuda_rng_sha256(self._device),
                applied_state_digest=surface.transaction.state_digest(),
                full_model_applied_sha256=_full_model_fingerprint(surface.model),
                update_count_before=applied.update_count_before,
                update_count_after=applied.update_count_after,
            )
            self._pending_proposal = pending
            self._lifecycle_state = "applied"
            return CudaPrivateProposalReceipt(
                status="private_proposal_applied",
                device=str(self._device),
                objective_binding_sha256=self._objective_binding_sha256,
                source_parameter_sha256=source.source_parameter_sha256,
                applied_parameter_sha256=pending.applied_parameter_sha256,
                full_model_source_sha256=self._full_model_source_sha256,
                full_model_applied_sha256=pending.full_model_applied_sha256,
                source_state_digest=source.source_state_digest,
                applied_state_digest=pending.applied_state_digest,
                source_version_sha256=source.source_version_sha256,
                applied_version_sha256=pending.applied_version_sha256,
                source_cuda_rng_sha256=source.source_cuda_rng_sha256,
                applied_cuda_rng_sha256=pending.applied_cuda_rng_sha256,
                proposal_sha256=pending.proposal_sha256,
                projection_sha256=pending.projection_sha256,
                projected_apply_sha256=pending.projected_apply_sha256,
                update_count_before=pending.update_count_before,
                update_count_after=pending.update_count_after,
            )
        except Exception as error:
            self._raise_after_failure(
                error,
                source=source,
                rollback_attempted=False,
                full_model_restore_attempted=False,
            )

    def rollback_private_proposal(self) -> CudaAdapterReceipt:
        """Reject the one applied private proposal and certify exact Source."""

        if self._lifecycle_state == "rolled_back":
            raise CudaAdapterError("private proposal was already rolled back")
        pending = self._pending_proposal
        if self._lifecycle_state != "applied" or pending is None:
            raise CudaAdapterError("private proposal has not been applied")
        surface = self._surface
        source = pending.source
        try:
            rollback = surface.transaction.reject(source.snapshot)
            surface.optimizer.zero_grad(set_to_none=True)
            self._restore_full_model_source()
            restored_parameter_sha256 = parameter_state_sha256(
                self._named, source.layout
            )
            restored_versions = _model_version_counters(surface.model)
            restored_version_sha256 = _version_counters_sha256(restored_versions)
            restored_cuda_rng_sha256 = _cuda_rng_sha256(self._device)
            restored_state_digest = surface.transaction.state_digest()
            full_model_restored_sha256 = _full_model_fingerprint(surface.model)
            if (
                rollback.decision != "rejected_restored"
                or restored_parameter_sha256 != source.source_parameter_sha256
                or restored_state_digest != source.source_state_digest
                or restored_versions != source.source_version_counters
                or restored_cuda_rng_sha256 != source.source_cuda_rng_sha256
            ):
                raise CudaAdapterError("adapter rollback did not restore Source exactly")
            receipt = CudaAdapterReceipt(
                status="applied_and_rolled_back",
                device=str(self._device),
                model_object_id=id(surface.model),
                optimizer_object_id=id(surface.optimizer),
                transaction_object_id=id(surface.transaction),
                parameter_object_ids=tuple(
                    id(parameter) for _, parameter in self._named
                ),
                source_parameter_sha256=source.source_parameter_sha256,
                full_model_source_sha256=self._full_model_source_sha256,
                full_model_applied_sha256=pending.full_model_applied_sha256,
                full_model_restored_sha256=full_model_restored_sha256,
                applied_parameter_sha256=pending.applied_parameter_sha256,
                restored_parameter_sha256=restored_parameter_sha256,
                source_version_counters=source.source_version_counters,
                applied_version_counters=pending.applied_version_counters,
                restored_version_counters=restored_versions,
                source_version_sha256=source.source_version_sha256,
                applied_version_sha256=pending.applied_version_sha256,
                restored_version_sha256=restored_version_sha256,
                source_cuda_rng_sha256=source.source_cuda_rng_sha256,
                applied_cuda_rng_sha256=pending.applied_cuda_rng_sha256,
                restored_cuda_rng_sha256=restored_cuda_rng_sha256,
                objective_binding_sha256=self._objective_binding_sha256,
                source_state_digest=source.source_state_digest,
                applied_state_digest=pending.applied_state_digest,
                restored_state_digest=restored_state_digest,
                proposal_sha256=pending.proposal_sha256,
                projection_sha256=pending.projection_sha256,
                projected_apply_sha256=pending.projected_apply_sha256,
                rollback_decision="rejected_restored",
                update_count_before=pending.update_count_before,
                update_count_after=pending.update_count_after,
            )
            _seal_receipt(receipt, _RECEIPT_ISSUER)
            self._pending_proposal = None
            self._lifecycle_state = "rolled_back"
            return receipt
        except Exception as error:
            self._raise_after_failure(
                error,
                source=source,
                rollback_attempted=True,
                full_model_restore_attempted=False,
            )

    def apply_and_rollback(self) -> CudaAdapterReceipt:
        """Backward-compatible one-call wrapper around the split lifecycle."""

        self.apply_private_proposal()
        return self.rollback_private_proposal()


__all__ = [
    "CudaAdapterError",
    "CudaAdapterRollbackError",
    "CudaRollbackFailureReceipt",
    "CudaAdapterReceipt",
    "CudaPrivateProposalReceipt",
    "CudaHFVerticalAdapter",
    "CudaProposalInput",
    "compute_cuda_objective_binding_sha256",
    "require_admitted_receipt",
]
