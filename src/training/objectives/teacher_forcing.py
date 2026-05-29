"""Teacher-forcing objective over Task-1 target IR atoms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import torch

from src.training.objectives.types import (
    DEFAULT_PRECISION_POLICY,
    LabelLogitRow,
    ObjectiveResult,
    ObjectiveSpec,
    ResolvedObjectiveSpan,
    config_float,
    config_tensor,
)
from src.training.supervision.distributions import TeacherForcingTargetDistribution
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.metrics import teacher_forcing_loss_events
from src.training.teacher_forcing.probabilities import (
    TeacherForcingAtomLoss,
    teacher_forcing_atom_loss,
)
from src.training.teacher_forcing.vocab import RoleVocab


class TeacherForcingObjective:
    """Teacher-forcing marginal/coverage objective over resolved logit rows."""

    objective_id = "teacher_forcing"

    def run(
        self,
        *,
        spec: ObjectiveSpec,
        spans: tuple[ResolvedObjectiveSpan, ...],
        logits: torch.Tensor,
    ) -> ObjectiveResult:
        """Compute objective-local normalized teacher-forcing loss."""

        if len(spans) == 0:
            return ObjectiveResult.zero(
                objective_id=spec.objective_id,
                weight=spec.weight,
                logits=logits,
            )

        input_ids = _require_input_ids(spec.config)
        role_vocab = _require_role_vocab(spec.config)
        coverage_strength = config_float(
            spec.config,
            "coverage_strength",
            default=0.0,
            minimum=0.0,
        )

        numerators: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        component_totals: dict[str, torch.Tensor] = {}
        atom_count = 0
        for resolved in spans:
            distribution = resolved.span.distribution
            if type(distribution) is not TeacherForcingTargetDistribution:
                raise TypeError(
                    "teacher_forcing spans must carry TeacherForcingTargetDistribution"
                )
            target_ir = distribution.target_ir
            _validate_ir_rows(
                target_ir,
                rows=resolved.rows,
                input_ids=input_ids,
                role_vocab=role_vocab,
            )

            losses: list[torch.Tensor] = []
            for atom, row, row_logits in zip(
                target_ir.atoms,
                resolved.rows,
                resolved.logits,
                strict=True,
            ):
                atom_loss = teacher_forcing_atom_loss(
                    row_logits,
                    atom=atom,
                    role_vocab=role_vocab,
                    coverage_strength=coverage_strength,
                )
                weighted = atom_loss.total * float(atom.loss_weight)
                losses.append(weighted)
                atom_count += 1
                _add_component(component_totals, "type", atom_loss.type)
                _add_component(component_totals, "valid", atom_loss.valid)
                _add_component(component_totals, "coverage", atom_loss.coverage)

            span_losses = torch.stack(losses).to(dtype=torch.float32)
            numerator = span_losses.sum()
            denominator = span_losses.new_tensor(float(len(losses)))
            numerators.append(numerator)
            denominators.append(denominator)

        numerator = torch.stack(numerators).sum().to(dtype=torch.float32)
        denominator = torch.stack(denominators).sum().to(dtype=torch.float32)
        loss = _safe_normalize(numerator, denominator)
        if not bool(torch.isfinite(loss).all().detach().cpu().item()):
            raise FloatingPointError("teacher_forcing loss contains non-finite values")
        weighted_loss = loss * float(spec.weight)

        metric_events = teacher_forcing_loss_events(
            loss=loss,
            denominator=denominator,
            span_count=len(spans),
            atom_count=atom_count,
        )
        state = {
            "atom_count": atom_count,
            "component_totals": MappingProxyType(component_totals),
        }

        return ObjectiveResult(
            objective_id=spec.objective_id,
            loss=loss,
            weighted_loss=weighted_loss,
            numerator=numerator,
            denominator=denominator,
            span_count=len(spans),
            weight=spec.weight,
            precision_policy=DEFAULT_PRECISION_POLICY,
            metric_events=metric_events,
            state=MappingProxyType(state),
        )


def _require_input_ids(config: Mapping[str, object]) -> torch.Tensor:
    input_ids = config_tensor(config, "input_ids", ndim=2)
    if input_ids.requires_grad:
        raise ValueError(
            "teacher_forcing config['input_ids'] must not have requires_grad"
        )
    if input_ids.dtype is not torch.long:
        raise TypeError("teacher_forcing config['input_ids'] must use dtype torch.long")

    return input_ids.detach().clone()


def _require_role_vocab(config: Mapping[str, object]) -> RoleVocab:
    try:
        role_vocab = config["role_vocab"]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError("teacher_forcing requires config['role_vocab']") from exc
    if type(role_vocab) is not RoleVocab:
        raise TypeError("teacher_forcing config['role_vocab'] must be a RoleVocab")
    return role_vocab


def _validate_ir_rows(
    target_ir: TeacherForcingTargetIR,
    *,
    rows: tuple[LabelLogitRow, ...],
    input_ids: torch.Tensor,
    role_vocab: RoleVocab,
) -> None:
    if len(target_ir.atoms) != len(rows):
        raise ValueError("teacher_forcing target IR atoms must align with label rows")
    for atom_index, (atom, row) in enumerate(zip(target_ir.atoms, rows, strict=True)):
        _validate_atom_structure(
            atom_index,
            atom,
            input_ids=input_ids,
            role_vocab=role_vocab,
        )
        _validate_row_identity(atom, row)


def _validate_atom_structure(
    atom_index: int,
    atom: SupervisionAtom,
    *,
    input_ids: torch.Tensor,
    role_vocab: RoleVocab,
) -> None:
    prefix = f"teacher_forcing_target_ir.atoms[{atom_index}]"
    _validate_atom_loss_weight(atom.loss_weight, prefix=prefix)
    if not atom.allowed_token_roles:
        raise ValueError(f"{prefix}: allowed_token_roles must be nonempty")
    if atom.selected_token_role not in atom.allowed_token_roles:
        raise ValueError(
            f"{prefix}: selected_token_role must be in allowed_token_roles"
        )
    if not atom.valid_token_ids:
        raise ValueError(f"{prefix}: valid_token_ids must be nonempty")

    _validate_input_position(
        input_ids,
        atom.batch_index,
        atom.logit_position,
        prefix=prefix,
        field_name="logit position",
    )
    live_token_id = _validate_input_position(
        input_ids,
        atom.batch_index,
        atom.target_position,
        prefix=prefix,
        field_name="target position",
    )
    if atom.selected_token_id != live_token_id:
        if not bool(atom.provenance.get("allow_target_token_mismatch", False)):
            raise ValueError(
                f"{prefix}: selected_token_id must match input_ids at target_position"
            )
    if atom.selected_token_id not in atom.valid_token_ids:
        raise ValueError(f"{prefix}: selected_token_id must be in valid_token_ids")

    allowed_role_vocab = role_vocab.token_ids_for_roles(atom.allowed_token_roles)
    if not atom.valid_token_ids.issubset(allowed_role_vocab):
        raise ValueError(
            f"{prefix}: valid_token_ids must be inside allowed role vocab"
        )


def _validate_atom_loss_weight(value: object, *, prefix: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{prefix}: loss_weight must be finite and nonnegative")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{prefix}: loss_weight must be finite and nonnegative")


def _validate_input_position(
    input_ids: torch.Tensor,
    batch_index: int,
    position: int,
    *,
    prefix: str,
    field_name: str,
) -> int:
    if batch_index < 0 or batch_index >= int(input_ids.shape[0]):
        raise ValueError(f"{prefix}: batch_index is out of bounds for input_ids")
    if position < 0 or position >= int(input_ids.shape[1]):
        raise ValueError(f"{prefix}: {field_name} is out of bounds for input_ids")
    value = int(input_ids[batch_index, position].detach().cpu().item())
    if value == -100:
        raise ValueError(f"{prefix}: masked {field_name} is not valid")
    return value


def _validate_row_identity(atom: SupervisionAtom, row: LabelLogitRow) -> None:
    if row.label_position != atom.target_position:
        raise ValueError(
            "teacher_forcing target_position must match LabelLogitRowMap label position"
        )
    if row.row_index != atom.logit_position:
        raise ValueError("teacher_forcing logit_position must match LabelLogitRowMap row")
    if row.batch_index is not None and row.batch_index != atom.batch_index:
        raise ValueError("teacher_forcing batch_index must match LabelLogitRowMap batch")


def _add_component(
    component_totals: dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    component_totals[key] = component_totals.get(key, value.new_tensor(0.0)) + value


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0
    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)


__all__ = [
    "TeacherForcingAtomLoss",
    "TeacherForcingObjective",
    "teacher_forcing_atom_loss",
]
