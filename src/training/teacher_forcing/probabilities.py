"""Teacher-forcing probability decomposition helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.training.teacher_forcing.ir import SupervisionAtom
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.vocab import RoleVocab


@dataclass(frozen=True, slots=True)
class TeacherForcingAtomLoss:
    """Loss components for one teacher-forcing atom."""

    total: torch.Tensor
    token_type_mass: torch.Tensor
    valid: torch.Tensor
    coverage: torch.Tensor
    continuation_margin: torch.Tensor
    bbox_positive_area: torch.Tensor
    allowed_probability: torch.Tensor
    valid_probability: torch.Tensor

    @property
    def type(self) -> torch.Tensor:
        """Backward-compatible alias for the token-type mass component."""

        return self.token_type_mass


def teacher_forcing_atom_loss(
    logits_row: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    coverage_strength: float,
) -> TeacherForcingAtomLoss:
    """Return the atom-local teacher-forcing loss decomposition."""

    if logits_row.ndim != 1:
        raise ValueError("teacher_forcing atom loss expects a 1D logits row")
    if not bool(torch.isfinite(logits_row).all().detach().cpu().item()):
        raise FloatingPointError("teacher_forcing logits contain non-finite values")
    if coverage_strength < 0.0 or not math.isfinite(float(coverage_strength)):
        raise ValueError("coverage_strength must be finite and >= 0")

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
        valid_loss = -log_valid
        token_type_mass_loss = _token_type_mass_loss(
            log_probs,
            atom=atom,
            role_vocab=role_vocab,
            device=device,
            vocab_size=vocab_size,
        )
        continuation_margin_loss = _continuation_margin_loss(
            log_probs,
            atom=atom,
            device=device,
            vocab_size=vocab_size,
        )
        bbox_positive_area_loss = _bbox_positive_area_loss(
            log_probs,
            atom=atom,
            role_vocab=role_vocab,
            device=device,
            vocab_size=vocab_size,
        )

        if float(coverage_strength) == 0.0:
            coverage_loss = valid_loss.new_tensor(0.0)
        else:
            target = _coverage_target(
                atom.coverage_target_weights,
                valid_token_ids=tuple(atom.valid_token_ids),
                ordered_valid_ids=valid_ids,
                device=device,
            )
            within_valid_log_probs = valid_log_probs - log_valid
            coverage_loss = -(target * within_valid_log_probs).sum()

        total = valid_loss + float(coverage_strength) * coverage_loss
        if not bool(torch.isfinite(total).all().detach().cpu().item()):
            raise FloatingPointError(
                "teacher_forcing atom loss contains non-finite values"
            )

    return TeacherForcingAtomLoss(
        total=total.to(dtype=torch.float32),
        token_type_mass=token_type_mass_loss.to(dtype=torch.float32),
        valid=valid_loss.to(dtype=torch.float32),
        coverage=coverage_loss.to(dtype=torch.float32),
        continuation_margin=continuation_margin_loss.to(dtype=torch.float32),
        bbox_positive_area=bbox_positive_area_loss.to(dtype=torch.float32),
        allowed_probability=log_allowed.exp().to(dtype=torch.float32),
        valid_probability=log_valid.exp().to(dtype=torch.float32),
    )


def _token_type_mass_loss(
    log_probs: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    family_log_masses = []
    for role in (TokenRole.SCHEMA, TokenRole.COORD, TokenRole.TEXT, TokenRole.STOP):
        family_log_masses.append(
            _log_mass_for_token_ids(
                log_probs,
                role_vocab.token_ids_for_role(role),
                device=device,
                vocab_size=vocab_size,
                field_name=f"{role.value.lower()} token ids",
            )
        )

    selected_log_mass = family_log_masses[
        (TokenRole.SCHEMA, TokenRole.COORD, TokenRole.TEXT, TokenRole.STOP).index(
            atom.selected_token_role
        )
    ]
    family_denominator = torch.logsumexp(torch.stack(family_log_masses), dim=-1)
    return -(selected_log_mass - family_denominator)


def _continuation_margin_loss(
    log_probs: torch.Tensor,
    *,
    atom: SupervisionAtom,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    if atom.provenance.get("continuation_boundary") is not True:
        return log_probs.new_tensor(0.0)

    continuation_token_ids = _provenance_token_ids(
        atom.provenance,
        "continuation_token_ids",
    )
    if not continuation_token_ids:
        raise ValueError("continuation_token_ids must be non-empty")

    stop_token_id = atom.provenance.get("stop_token_id")
    if type(stop_token_id) is not int:
        raise ValueError("stop_token_id must be present for continuation boundaries")
    if stop_token_id in continuation_token_ids:
        raise ValueError("stop_token_id must not appear in continuation_token_ids")

    continuation_ids = _token_ids_tensor(
        continuation_token_ids,
        device=device,
        vocab_size=vocab_size,
        field_name="continuation_token_ids",
    )
    stop_ids = _token_ids_tensor(
        frozenset({stop_token_id}),
        device=device,
        vocab_size=vocab_size,
        field_name="stop_token_id",
    )
    continuation_log_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=continuation_ids),
        dim=-1,
    )
    stop_log_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=stop_ids),
        dim=-1,
    )
    denominator = torch.logsumexp(
        torch.stack((continuation_log_mass, stop_log_mass)),
        dim=-1,
    )

    target = atom.provenance.get("continuation_target")
    if target == "continue":
        return -(continuation_log_mass - denominator)
    if target == "stop":
        return -(stop_log_mass - denominator)
    raise ValueError("continuation_target must be 'continue' or 'stop'")


def _bbox_positive_area_loss(
    log_probs: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    if not _is_bbox_positive_area_atom(atom):
        return log_probs.new_tensor(0.0)

    valid_ids = _provenance_token_ids(
        atom.provenance,
        "bbox_positive_area_valid_token_ids",
    )
    invalid_ids = _provenance_token_ids(
        atom.provenance,
        "bbox_positive_area_invalid_token_ids",
    )
    if not valid_ids:
        raise ValueError("bbox_positive_area_valid_token_ids must be non-empty")
    if not invalid_ids:
        raise ValueError("bbox_positive_area_invalid_token_ids must be non-empty")
    if valid_ids & invalid_ids:
        raise ValueError("bbox positive-area valid and invalid token ids must be disjoint")

    coord_ids = role_vocab.coord_token_ids
    if not valid_ids.issubset(coord_ids) or not invalid_ids.issubset(coord_ids):
        raise ValueError("bbox positive-area token ids must be coord token ids")
    if valid_ids | invalid_ids != coord_ids:
        raise ValueError("bbox positive-area token ids must partition all coord token ids")
    if atom.selected_token_id not in valid_ids:
        raise ValueError(
            "selected_token_id must be in bbox_positive_area_valid_token_ids"
        )

    valid_tensor = _token_ids_tensor(
        valid_ids,
        device=device,
        vocab_size=vocab_size,
        field_name="bbox_positive_area_valid_token_ids",
    )
    coord_tensor = _token_ids_tensor(
        coord_ids,
        device=device,
        vocab_size=vocab_size,
        field_name="coord_token_ids",
    )
    valid_log_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=valid_tensor),
        dim=-1,
    )
    coord_log_mass = torch.logsumexp(
        log_probs.index_select(dim=-1, index=coord_tensor),
        dim=-1,
    )
    return -(valid_log_mass - coord_log_mass)


def _is_bbox_positive_area_atom(atom: SupervisionAtom) -> bool:
    return (
        atom.coord_role in {"x2", "y2"}
        and atom.provenance.get("bbox_positive_area") is True
    )


def _log_mass_for_token_ids(
    log_probs: torch.Tensor,
    token_ids: frozenset[int],
    *,
    device: torch.device,
    vocab_size: int,
    field_name: str,
) -> torch.Tensor:
    in_vocab_token_ids = frozenset(token_id for token_id in token_ids if token_id < vocab_size)
    if not in_vocab_token_ids:
        return log_probs.new_tensor(float("-inf"))
    token_ids_tensor = _token_ids_tensor(
        in_vocab_token_ids,
        device=device,
        vocab_size=vocab_size,
        field_name=field_name,
    )
    return torch.logsumexp(log_probs.index_select(dim=-1, index=token_ids_tensor), dim=-1)


def _provenance_token_ids(
    provenance: Mapping[str, Any],
    field_name: str,
) -> frozenset[int]:
    value = provenance.get(field_name)
    if value is None:
        raise ValueError(f"{field_name} must be present")
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
        if type(token_id) is not int:
            raise TypeError(f"{field_name} must contain integer token ids")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{field_name} contains an id outside logits vocab")
    return torch.tensor(ordered, device=device, dtype=torch.long)


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
