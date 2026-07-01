"""Teacher-forcing probability decomposition helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.training.teacher_forcing.ir import SupervisionAtom
from src.training.teacher_forcing.vocab import RoleVocab


@dataclass(frozen=True, slots=True)
class TeacherForcingAtomLoss:
    """Loss components for one teacher-forcing atom."""

    total: torch.Tensor
    type: torch.Tensor
    valid: torch.Tensor
    coverage: torch.Tensor
    allowed_probability: torch.Tensor
    valid_probability: torch.Tensor


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

        total = type_loss + valid_loss + float(coverage_strength) * coverage_loss
        if not bool(torch.isfinite(total).all().detach().cpu().item()):
            raise FloatingPointError(
                "teacher_forcing atom loss contains non-finite values"
            )

    return TeacherForcingAtomLoss(
        total=total.to(dtype=torch.float32),
        type=type_loss.to(dtype=torch.float32),
        valid=valid_loss.to(dtype=torch.float32),
        coverage=coverage_loss.to(dtype=torch.float32),
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
