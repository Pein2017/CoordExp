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
from src.training.teacher_forcing.metrics import (
    teacher_forcing_component_loss_events,
    teacher_forcing_loss_events,
)
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
        token_type_mass_weight = config_float(
            spec.config,
            "token_type_mass_weight",
            default=0.0,
            minimum=0.0,
        )
        continuation_margin_weight = config_float(
            spec.config,
            "continuation_margin_weight",
            default=0.0,
            minimum=0.0,
        )
        bbox_positive_area_weight = config_float(
            spec.config,
            "bbox_positive_area_weight",
            default=0.0,
            minimum=0.0,
        )

        valid_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        valid_denominator = logits.new_tensor(0.0, dtype=torch.float32)
        coverage_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        coverage_denominator = logits.new_tensor(0.0, dtype=torch.float32)
        token_type_mass_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        token_type_mass_denominator = logits.new_tensor(0.0, dtype=torch.float32)
        continuation_margin_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        continuation_margin_denominator = logits.new_tensor(0.0, dtype=torch.float32)
        bbox_positive_area_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        bbox_positive_area_denominator = logits.new_tensor(0.0, dtype=torch.float32)
        continuation_margin_diagnostic_numerator = logits.new_tensor(
            0.0, dtype=torch.float32
        )
        continuation_continue_mass_numerator = logits.new_tensor(
            0.0, dtype=torch.float32
        )
        continuation_stop_mass_numerator = logits.new_tensor(0.0, dtype=torch.float32)
        bbox_positive_area_invalid_mass_numerator = logits.new_tensor(
            0.0, dtype=torch.float32
        )
        continuation_correct_count = 0
        continuation_boundary_count = 0
        bbox_positive_area_eligible_count = 0
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
                atom_weight = float(atom.loss_weight)
                atom_count += 1
                valid_numerator = valid_numerator + atom_loss.valid * atom_weight
                valid_denominator = valid_denominator + valid_denominator.new_tensor(1.0)
                coverage_numerator = coverage_numerator + atom_loss.coverage * atom_weight
                coverage_denominator = coverage_denominator + coverage_denominator.new_tensor(1.0)
                token_type_mass_numerator = (
                    token_type_mass_numerator
                    + atom_loss.token_type_mass * atom_weight
                )
                token_type_mass_denominator = (
                    token_type_mass_denominator
                    + token_type_mass_denominator.new_tensor(1.0)
                )

                if _is_continuation_boundary_atom(atom):
                    continuation_margin_numerator = (
                        continuation_margin_numerator
                        + atom_loss.continuation_margin * atom_weight
                    )
                    continuation_margin_denominator = (
                        continuation_margin_denominator
                        + continuation_margin_denominator.new_tensor(1.0)
                    )
                    continuation_diagnostics = _continuation_boundary_diagnostics(
                        row_logits,
                        atom,
                    )
                    continuation_margin_diagnostic_numerator = (
                        continuation_margin_diagnostic_numerator
                        + continuation_diagnostics["continue_minus_stop_margin"]
                    )
                    continuation_continue_mass_numerator = (
                        continuation_continue_mass_numerator
                        + continuation_diagnostics["continue_mass"]
                    )
                    continuation_stop_mass_numerator = (
                        continuation_stop_mass_numerator
                        + continuation_diagnostics["stop_mass"]
                    )
                    continuation_correct_count += int(
                        continuation_diagnostics["correct"]
                    )
                    continuation_boundary_count += 1

                if _is_bbox_positive_area_atom(atom):
                    bbox_positive_area_numerator = (
                        bbox_positive_area_numerator
                        + atom_loss.bbox_positive_area * atom_weight
                    )
                    bbox_positive_area_denominator = (
                        bbox_positive_area_denominator
                        + bbox_positive_area_denominator.new_tensor(1.0)
                    )
                    bbox_positive_area_invalid_mass_numerator = (
                        bbox_positive_area_invalid_mass_numerator
                        + _bbox_positive_area_invalid_mass(
                            row_logits,
                            atom=atom,
                            role_vocab=role_vocab,
                        )
                    )
                    bbox_positive_area_eligible_count += 1

                _add_component(component_totals, "type", atom_loss.token_type_mass)
                _add_component(
                    component_totals,
                    "token_type_mass",
                    atom_loss.token_type_mass,
                )
                _add_component(component_totals, "valid", atom_loss.valid)
                _add_component(component_totals, "coverage", atom_loss.coverage)
                _add_component(
                    component_totals,
                    "continuation_margin",
                    atom_loss.continuation_margin,
                )
                _add_component(
                    component_totals,
                    "bbox_positive_area",
                    atom_loss.bbox_positive_area,
                )

        if atom_count == 0:
            raise ValueError("teacher_forcing spans must contain supervised atoms")
        if token_type_mass_weight > 0.0:
            _require_positive_denominator(
                token_type_mass_denominator,
                field_name="token_type_mass",
            )
        if continuation_margin_weight > 0.0:
            _require_positive_denominator(
                continuation_margin_denominator,
                field_name="continuation_margin",
            )
        if bbox_positive_area_weight > 0.0:
            _require_positive_denominator(
                bbox_positive_area_denominator,
                field_name="bbox_positive_area",
            )

        valid_mean = _safe_normalize(valid_numerator, valid_denominator)
        coverage_mean = _safe_normalize(coverage_numerator, coverage_denominator)
        token_type_mass_mean = _safe_normalize(
            token_type_mass_numerator,
            token_type_mass_denominator,
        )
        continuation_margin_mean = _safe_normalize(
            continuation_margin_numerator,
            continuation_margin_denominator,
        )
        bbox_positive_area_mean = _safe_normalize(
            bbox_positive_area_numerator,
            bbox_positive_area_denominator,
        )
        continuation_boundary_count_tensor = logits.new_tensor(
            float(continuation_boundary_count),
            dtype=torch.float32,
        )
        bbox_positive_area_eligible_count_tensor = logits.new_tensor(
            float(bbox_positive_area_eligible_count),
            dtype=torch.float32,
        )
        continue_minus_stop_margin_mean = _safe_normalize(
            continuation_margin_diagnostic_numerator,
            continuation_boundary_count_tensor,
        )
        continue_mass_mean = _safe_normalize(
            continuation_continue_mass_numerator,
            continuation_boundary_count_tensor,
        )
        stop_mass_mean = _safe_normalize(
            continuation_stop_mass_numerator,
            continuation_boundary_count_tensor,
        )
        bbox_positive_area_invalid_mass_mean = _safe_normalize(
            bbox_positive_area_invalid_mass_numerator,
            bbox_positive_area_eligible_count_tensor,
        )
        loss = (
            valid_mean
            + float(coverage_strength) * coverage_mean
            + float(token_type_mass_weight) * token_type_mass_mean
            + float(continuation_margin_weight) * continuation_margin_mean
            + float(bbox_positive_area_weight) * bbox_positive_area_mean
        ).to(dtype=torch.float32)
        if not bool(torch.isfinite(loss).all().detach().cpu().item()):
            raise FloatingPointError("teacher_forcing loss contains non-finite values")
        weighted_loss = loss * float(spec.weight)
        numerator = valid_numerator.to(dtype=torch.float32)
        denominator = valid_denominator.to(dtype=torch.float32)

        metric_events = teacher_forcing_loss_events(
            loss=loss,
            denominator=denominator,
            span_count=len(spans),
            atom_count=atom_count,
        ) + teacher_forcing_component_loss_events(
            token_type_mass=token_type_mass_mean,
            token_type_mass_contribution=(
                float(token_type_mass_weight) * token_type_mass_mean
            ),
            continuation_margin=(
                continuation_margin_mean
                if continuation_boundary_count > 0
                else None
            ),
            continuation_margin_contribution=(
                float(continuation_margin_weight) * continuation_margin_mean
                if continuation_boundary_count > 0
                else None
            ),
            bbox_positive_area=(
                bbox_positive_area_mean
                if bbox_positive_area_eligible_count > 0
                else None
            ),
            bbox_positive_area_contribution=(
                float(bbox_positive_area_weight) * bbox_positive_area_mean
                if bbox_positive_area_eligible_count > 0
                else None
            ),
            continue_minus_stop_margin=(
                continue_minus_stop_margin_mean
                if continuation_boundary_count > 0
                else None
            ),
            continue_correct=(
                continuation_correct_count
                if continuation_boundary_count > 0
                else None
            ),
            continuation_boundary_count=(
                continuation_boundary_count
                if continuation_boundary_count > 0
                else None
            ),
            continue_mass=continue_mass_mean if continuation_boundary_count > 0 else None,
            stop_mass=stop_mass_mean if continuation_boundary_count > 0 else None,
            bbox_positive_area_invalid_mass=(
                bbox_positive_area_invalid_mass_mean
                if bbox_positive_area_eligible_count > 0
                else None
            ),
            bbox_positive_area_eligible_count=(
                bbox_positive_area_eligible_count
                if bbox_positive_area_eligible_count > 0
                else None
            ),
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


def _is_continuation_boundary_atom(atom: SupervisionAtom) -> bool:
    return atom.provenance.get("continuation_boundary") is True


def _is_bbox_positive_area_atom(atom: SupervisionAtom) -> bool:
    return (
        atom.coord_role in {"x2", "y2"}
        and atom.provenance.get("bbox_positive_area") is True
    )


def _continuation_boundary_diagnostics(
    row_logits: torch.Tensor,
    atom: SupervisionAtom,
) -> dict[str, torch.Tensor | bool]:
    log_probs = torch.log_softmax(row_logits.to(dtype=torch.float32), dim=-1)
    continuation_ids = _provenance_token_ids(
        atom.provenance,
        "continuation_token_ids",
    )
    stop_token_id = int(atom.provenance["stop_token_id"])
    continuation_tensor = _token_ids_tensor(
        continuation_ids,
        device=row_logits.device,
        vocab_size=int(row_logits.shape[-1]),
        field_name="continuation_token_ids",
    )
    stop_tensor = _token_ids_tensor(
        frozenset({stop_token_id}),
        device=row_logits.device,
        vocab_size=int(row_logits.shape[-1]),
        field_name="stop_token_id",
    )
    continue_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=continuation_tensor),
        dim=-1,
    ).exp()
    stop_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=stop_tensor),
        dim=-1,
    ).exp()
    target = atom.provenance["continuation_target"]
    if target == "continue":
        correct = bool((continue_mass >= stop_mass).detach().cpu().item())
    else:
        correct = bool((stop_mass >= continue_mass).detach().cpu().item())
    return {
        "continue_minus_stop_margin": continue_mass - stop_mass,
        "continue_mass": continue_mass,
        "stop_mass": stop_mass,
        "correct": correct,
    }


def _bbox_positive_area_invalid_mass(
    row_logits: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
) -> torch.Tensor:
    probs = torch.softmax(row_logits.to(dtype=torch.float32), dim=-1)
    invalid_tensor = _token_ids_tensor(
        _provenance_token_ids(
            atom.provenance,
            "bbox_positive_area_invalid_token_ids",
        ),
        device=row_logits.device,
        vocab_size=int(row_logits.shape[-1]),
        field_name="bbox_positive_area_invalid_token_ids",
    )
    coord_tensor = _token_ids_tensor(
        role_vocab.coord_token_ids,
        device=row_logits.device,
        vocab_size=int(row_logits.shape[-1]),
        field_name="coord_token_ids",
    )
    invalid_mass = probs.index_select(dim=-1, index=invalid_tensor).sum()
    coord_mass = probs.index_select(dim=-1, index=coord_tensor).sum()
    return invalid_mass / coord_mass.clamp(min=1e-12)


def _provenance_token_ids(
    provenance: Mapping[str, object],
    field_name: str,
) -> frozenset[int]:
    value = provenance[field_name]
    if isinstance(value, (str, bytes)) or not isinstance(value, frozenset | set | tuple | list):
        raise TypeError(f"{field_name} must be a collection of token ids")
    token_ids = frozenset(value)
    for token_id in token_ids:
        if type(token_id) is not int:
            raise TypeError(f"{field_name} must contain integer token ids")
    return token_ids


def _token_ids_tensor(
    token_ids: frozenset[int],
    *,
    device: torch.device,
    vocab_size: int,
    field_name: str,
) -> torch.Tensor:
    if len(token_ids) == 0:
        raise ValueError(f"{field_name} must be non-empty")
    ordered = sorted(token_ids)
    for token_id in ordered:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{field_name} contains an id outside logits vocab")
    return torch.tensor(ordered, device=device, dtype=torch.long)


def _require_positive_denominator(
    denominator: torch.Tensor,
    *,
    field_name: str,
) -> None:
    if float(denominator.detach().cpu().item()) <= 0.0:
        raise ValueError(f"{field_name} enabled but no eligible atoms were found")


def _safe_normalize(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if float(denominator.detach().cpu().item()) <= 0.0:
        return numerator * 0.0
    return (numerator / denominator.clamp(min=1e-6)).to(dtype=torch.float32)


__all__ = [
    "TeacherForcingAtomLoss",
    "TeacherForcingObjective",
    "teacher_forcing_atom_loss",
]
