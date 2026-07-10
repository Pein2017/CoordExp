"""Teacher-forcing probability decomposition helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.nn.functional as F

from src.training.teacher_forcing.ir import SupervisionAtom
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


_TOKEN_TYPE_FAMILIES = (
    (TokenRole.SCHEMA, "schema"),
    (TokenRole.TEXT, "desc"),
    (TokenRole.COORD, "coord"),
    (TokenRole.STOP, "stop"),
)


@dataclass(frozen=True, slots=True)
class TeacherForcingAtomLoss:
    """Loss components for one teacher-forcing atom."""

    total: torch.Tensor
    type: torch.Tensor
    valid: torch.Tensor
    coverage: torch.Tensor
    token_type_mass: torch.Tensor
    token_type_mass_contribution: torch.Tensor
    target_family_mass: torch.Tensor
    target_family: str | None
    family_masses: Mapping[str, torch.Tensor]
    allowed_probability: torch.Tensor
    valid_probability: torch.Tensor


def teacher_forcing_atom_loss(
    logits_row: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    coverage_strength: float,
    token_type_mass_enabled: bool = False,
    token_type_mass_weight: float = 1.0,
) -> TeacherForcingAtomLoss:
    """Return the atom-local teacher-forcing loss decomposition."""

    if logits_row.ndim != 1:
        raise ValueError("teacher_forcing atom loss expects a 1D logits row")
    if not bool(torch.isfinite(logits_row).all().detach().cpu().item()):
        raise FloatingPointError("teacher_forcing logits contain non-finite values")
    if coverage_strength < 0.0 or not math.isfinite(float(coverage_strength)):
        raise ValueError("coverage_strength must be finite and >= 0")
    parsed_token_type_mass_weight = 0.0
    if token_type_mass_enabled:
        parsed_token_type_mass_weight = _parse_token_type_mass_weight(
            token_type_mass_weight
        )

    device = logits_row.device
    vocab_size = int(logits_row.shape[-1])
    allowed_ids = _token_ids_tensor(
        role_vocab.token_ids_for_roles(atom.allowed_token_roles),
        device=device,
        vocab_size=vocab_size,
        field_name="allowed_token_roles",
    )
    valid_ids = _token_ids_tensor(
        atom.valid_token_ids,
        device=device,
        vocab_size=vocab_size,
        field_name="valid_token_ids",
    )

    if device.type in {"cpu", "cuda"}:
        context = torch.autocast(device_type=device.type, enabled=False)
    else:
        context = nullcontext()

    with context:
        logits_fp32 = logits_row.to(dtype=torch.float32)
        log_probs = F.log_softmax(logits_fp32, dim=-1)
        allowed_log_probs = log_probs.index_select(dim=-1, index=allowed_ids)
        valid_log_probs = log_probs.index_select(dim=-1, index=valid_ids)

        log_allowed = torch.logsumexp(allowed_log_probs, dim=-1)
        log_valid = torch.logsumexp(valid_log_probs, dim=-1)
        type_loss = -log_allowed
        valid_loss = -(log_valid - log_allowed)

        if float(coverage_strength) == 0.0:
            coverage_loss = type_loss.new_tensor(0.0)
        else:
            target = _coverage_target(
                atom.coverage_target_weights,
                valid_token_ids=tuple(atom.valid_token_ids),
                ordered_valid_ids=valid_ids,
                device=device,
            )
            within_valid_log_probs = valid_log_probs - log_valid
            coverage_loss = -(target * within_valid_log_probs).sum()

        zero = type_loss * 0.0
        token_type_mass = zero
        token_type_mass_contribution = zero
        target_family_mass = zero
        target_family: str | None = None
        family_masses: Mapping[str, torch.Tensor] = _zero_family_masses(zero)
        if token_type_mass_enabled:
            family_names, family_logits = _family_logits(
                log_probs,
                role_vocab=role_vocab,
                device=device,
                vocab_size=vocab_size,
            )
            family_log_probs = torch.log_softmax(family_logits, dim=-1)
            family_probs = family_log_probs.exp()
            target_family_index = _target_family_index(atom.selected_token_role)
            target_family = family_names[target_family_index]
            token_type_mass = -family_log_probs[target_family_index]
            token_type_mass_contribution = parsed_token_type_mass_weight * token_type_mass
            target_family_mass = family_probs[target_family_index]
            family_masses = MappingProxyType(
                {
                    name: family_probs[index].to(dtype=torch.float32)
                    for index, name in enumerate(family_names)
                }
            )

        total = (
            type_loss
            + valid_loss
            + float(coverage_strength) * coverage_loss
            + token_type_mass_contribution
        )
        if not bool(torch.isfinite(total).all().detach().cpu().item()):
            raise FloatingPointError(
                "teacher_forcing atom loss contains non-finite values"
            )

    return TeacherForcingAtomLoss(
        total=total.to(dtype=torch.float32),
        type=type_loss.to(dtype=torch.float32),
        valid=valid_loss.to(dtype=torch.float32),
        coverage=coverage_loss.to(dtype=torch.float32),
        token_type_mass=token_type_mass.to(dtype=torch.float32),
        token_type_mass_contribution=token_type_mass_contribution.to(
            dtype=torch.float32
        ),
        target_family_mass=target_family_mass.to(dtype=torch.float32),
        target_family=target_family,
        family_masses=family_masses,
        allowed_probability=log_allowed.exp().to(dtype=torch.float32),
        valid_probability=log_valid.exp().to(dtype=torch.float32),
    )


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
        if type(token_id) is not int:
            raise TypeError(f"{field_name} must contain integer token ids")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{field_name} contains an id outside logits vocab")
    return torch.tensor(ordered, device=device, dtype=torch.long)


def _parse_token_type_mass_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("token_type_mass_weight must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("token_type_mass_weight must be finite and >= 0")
    return parsed


def _family_logits(
    log_probs: torch.Tensor,
    *,
    role_vocab: RoleVocab,
    device: torch.device,
    vocab_size: int,
) -> tuple[tuple[str, ...], torch.Tensor]:
    names: list[str] = []
    logits: list[torch.Tensor] = []
    for role, name in _TOKEN_TYPE_FAMILIES:
        ids = _token_ids_tensor(
            role_vocab.token_ids_for_role(role),
            device=device,
            vocab_size=vocab_size,
            field_name=f"{name}_token_ids",
        )
        names.append(name)
        logits.append(
            torch.logsumexp(log_probs.index_select(dim=-1, index=ids), dim=-1)
        )
    return tuple(names), torch.stack(logits)


def _target_family_index(role: TokenRole) -> int:
    for index, (candidate, _name) in enumerate(_TOKEN_TYPE_FAMILIES):
        if role is candidate:
            return index
    raise ValueError(f"unsupported token role: {role!r}")


def _zero_family_masses(zero: torch.Tensor) -> Mapping[str, torch.Tensor]:
    return MappingProxyType(
        {name: zero.to(dtype=torch.float32) for _role, name in _TOKEN_TYPE_FAMILIES}
    )


def _coverage_target(
    coverage_target_weights: Mapping[int, float] | None,
    *,
    valid_token_ids: tuple[int, ...],
    ordered_valid_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if coverage_target_weights is None:
        return torch.full(
            (int(ordered_valid_ids.numel()),),
            1.0 / float(ordered_valid_ids.numel()),
            device=device,
            dtype=torch.float32,
        )

    valid_set = frozenset(valid_token_ids)
    weights_by_id: dict[int, float] = {}
    for token_id, weight in coverage_target_weights.items():
        if type(token_id) is not int:
            raise TypeError("coverage_target_weights keys must be token ids")
        if token_id not in valid_set:
            raise ValueError(
                "coverage_target_weights keys must be inside valid_token_ids"
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("coverage_target_weights values must be numeric scalars")
        parsed = float(weight)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                "coverage_target_weights values must be finite and non-negative"
            )
        weights_by_id[token_id] = parsed

    ordered_weights = [
        weights_by_id.get(int(token_id), 0.0)
        for token_id in ordered_valid_ids.detach().cpu().tolist()
    ]
    total = sum(ordered_weights)
    if total <= 0.0:
        raise ValueError("coverage_target_weights must have positive mass")

    target = torch.tensor(ordered_weights, device=device, dtype=torch.float32)
    return target / target.sum().clamp(min=1e-12)
