"""Differentiable CE adapter for recursive detection targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.detection.objective import RecursiveDetectionTargets, SemanticRole

_OBJECT_ROLE_WEIGHTS = {
    SemanticRole.DESC_IDENTITY: 0.35,
    SemanticRole.BBOX_COORD: 0.45,
    SemanticRole.ENTRY_TRIE_DECISION: 0.15,
    SemanticRole.OBJECT_CONTROL: 0.05,
}
_BOUNDARY_ROLE_WEIGHTS = {
    "separator_continue": 0.50,
    "terminal_stop": 0.50,
}
_IMAGE_MIXTURE_WEIGHTS = {
    "objects": 1.00,
    "boundary": 0.30,
    "schema": 0.10,
}


@dataclass(frozen=True)
class RecursiveDetectionLossWeights:
    branch_support_weight: float = 1.0
    branch_balance_weight: float = 1.0


@dataclass(frozen=True)
class RecursiveDetectionLossResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    per_position_losses: tuple[dict[int, torch.Tensor], ...]


def compute_recursive_detection_ce_batch_loss(
    *,
    logits: torch.Tensor,
    targets: Sequence[RecursiveDetectionTargets],
    weights: RecursiveDetectionLossWeights | None = None,
) -> RecursiveDetectionLossResult:
    """Compute the differentiable recursive-detection CE loss over a batch."""

    if weights is None:
        weights = RecursiveDetectionLossWeights()

    batch_logits = _normalize_logits_shape(logits)
    batch_size, time_steps, vocab_size = batch_logits.shape
    if len(targets) != batch_size:
        raise ValueError(
            f"targets length {len(targets)} must match logits batch size {batch_size}"
        )

    sample_losses: list[torch.Tensor] = []
    per_position_losses: list[dict[int, torch.Tensor]] = []
    for batch_index, recursive_targets in enumerate(targets):
        sample_loss, sample_position_losses = _compute_sample_loss(
            logits=batch_logits[batch_index],
            recursive_targets=recursive_targets,
            vocab_size=vocab_size,
            time_steps=time_steps,
            weights=weights,
        )
        sample_losses.append(sample_loss)
        per_position_losses.append(sample_position_losses)

    loss = torch.stack(sample_losses).mean() if sample_losses else batch_logits.sum() * 0.0
    metrics = {
        "batch_loss": float(loss.detach().item()),
        "batch_size": float(batch_size),
    }
    return RecursiveDetectionLossResult(
        loss=loss,
        metrics=metrics,
        per_position_losses=tuple(per_position_losses),
    )


def _normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(0)
    if logits.ndim == 3:
        return logits
    raise ValueError(
        f"logits must have shape [T, V] or [B, T, V], got {tuple(logits.shape)}"
    )


def _compute_sample_loss(
    *,
    logits: torch.Tensor,
    recursive_targets: RecursiveDetectionTargets,
    vocab_size: int,
    time_steps: int,
    weights: RecursiveDetectionLossWeights,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    per_position_losses: dict[int, torch.Tensor] = {}
    seen_positions: set[int] = set()
    for target in recursive_targets.token_targets:
        if target.position <= 0 or target.position > time_steps:
            raise ValueError(
                "TokenTarget.position must be in [1, logits_time_dim]; "
                f"got {target.position} for logits time dimension {time_steps}"
            )
        if target.position in seen_positions:
            raise ValueError(f"Duplicate TokenTarget.position {target.position}")
        seen_positions.add(target.position)
        _validate_token_id(
            token_id=target.teacher_token_id,
            vocab_size=vocab_size,
            label="teacher token id",
        )
        step_log_probs = F.log_softmax(logits[target.position - 1].float(), dim=-1)
        if target.kind == "hard_ce":
            per_position_losses[target.position] = -step_log_probs[target.teacher_token_id]
            continue

        if not target.trie_branch_targets:
            raise ValueError(
                f"trie_multi_positive target at position {target.position} needs children"
            )
        child_token_ids: list[int] = []
        child_weights: list[float] = []
        for branch_target in target.trie_branch_targets:
            _validate_token_id(
                token_id=branch_target.token_id,
                vocab_size=vocab_size,
                label="branch child token id",
            )
            if int(branch_target.multiplicity) <= 0:
                raise ValueError(
                    "trie branch child multiplicity must be positive; "
                    f"got {branch_target.multiplicity} at position {target.position}"
                )
            if not math.isfinite(float(branch_target.probability)) or float(
                branch_target.probability
            ) <= 0.0:
                raise ValueError(
                    "trie branch child probability must be positive and finite; "
                    f"got {branch_target.probability!r} at position {target.position}"
                )
            child_token_ids.append(int(branch_target.token_id))
            child_weights.append(float(branch_target.multiplicity))
        if target.teacher_token_id not in child_token_ids:
            raise ValueError(
                "trie_multi_positive teacher token must be one of the valid child tokens; "
                f"got teacher token {target.teacher_token_id} with children {child_token_ids}"
            )

        child_ids = torch.tensor(child_token_ids, device=step_log_probs.device, dtype=torch.long)
        q = torch.tensor(child_weights, device=step_log_probs.device, dtype=torch.float32)
        q = q / q.sum().clamp_min(1e-12)
        child_log_probs = step_log_probs.index_select(0, child_ids)
        valid_log_mass = torch.logsumexp(child_log_probs, dim=0)
        support_loss = -valid_log_mass
        balance_loss = -(q * (child_log_probs - valid_log_mass)).sum()
        per_position_losses[target.position] = (
            float(weights.branch_support_weight) * support_loss
            + float(weights.branch_balance_weight) * balance_loss
        )

    sample_loss = _normalize_sample_loss(
        recursive_targets=recursive_targets,
        per_position_losses=per_position_losses,
        device=logits.device,
    )
    return sample_loss, per_position_losses


def _validate_token_id(*, token_id: int, vocab_size: int, label: str) -> None:
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} {token_id} is outside vocab size {vocab_size}")


def _normalize_sample_loss(
    *,
    recursive_targets: RecursiveDetectionTargets,
    per_position_losses: Mapping[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if recursive_targets.normalization == "legacy_row_mean_equivalence":
        weighted_losses: list[torch.Tensor] = []
        state_weight_sum = 0.0
        for target in recursive_targets.token_targets:
            weighted_losses.append(per_position_losses[target.position] * float(target.state_weight))
            state_weight_sum += float(target.state_weight)
        denominator = torch.tensor(
            max(state_weight_sum, 1e-12),
            device=device,
            dtype=torch.float32,
        )
        return torch.stack(weighted_losses).sum() / denominator

    if recursive_targets.normalization != "semantic_image_bucket_balanced":
        raise ValueError(
            "unsupported recursive detection normalization "
            f"{recursive_targets.normalization!r}"
        )

    atom_losses = {
        atom.atom_id: torch.stack(
            [per_position_losses[position] for position in atom.token_positions]
        ).mean()
        for atom in recursive_targets.loss_atoms
    }

    component_losses: dict[str, torch.Tensor] = {}
    component_weights: dict[str, float] = {}

    object_atoms: dict[str, list[tuple[SemanticRole, torch.Tensor]]] = {}
    for atom in recursive_targets.loss_atoms:
        if atom.object_instance_id is None:
            continue
        object_atoms.setdefault(atom.object_instance_id, []).append(
            (atom.semantic_role, atom_losses[atom.atom_id])
        )
    if object_atoms:
        object_losses = [
            _weighted_tensor_mean(
                [loss for _, loss in atoms],
                [float(_OBJECT_ROLE_WEIGHTS[role]) for role, _ in atoms],
            )
            for atoms in object_atoms.values()
        ]
        component_losses["objects"] = torch.stack(object_losses).mean()
        component_weights["objects"] = _IMAGE_MIXTURE_WEIGHTS["objects"]

    separator_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SEPARATOR_CONTINUE
    ]
    stop_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role in {SemanticRole.TERMINAL_STOP, SemanticRole.CHAT_STOP}
    ]
    if separator_losses or stop_losses:
        boundary_terms: list[torch.Tensor] = []
        boundary_weights: list[float] = []
        if separator_losses:
            boundary_terms.append(torch.stack(separator_losses).mean())
            boundary_weights.append(_BOUNDARY_ROLE_WEIGHTS["separator_continue"])
        if stop_losses:
            boundary_terms.append(torch.stack(stop_losses).mean())
            boundary_weights.append(_BOUNDARY_ROLE_WEIGHTS["terminal_stop"])
        component_losses["boundary"] = _weighted_tensor_mean(boundary_terms, boundary_weights)
        component_weights["boundary"] = _IMAGE_MIXTURE_WEIGHTS["boundary"]

    schema_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SCHEMA_CONTROL
    ]
    if schema_losses:
        component_losses["schema"] = torch.stack(schema_losses).mean()
        component_weights["schema"] = _IMAGE_MIXTURE_WEIGHTS["schema"]

    if not component_losses:
        raise ValueError(
            "semantic_image_bucket_balanced requires at least one loss component "
            "(object, boundary, or schema)"
        )
    return _weighted_tensor_mean(
        list(component_losses.values()),
        [component_weights[name] for name in component_losses],
    )


def _weighted_tensor_mean(values: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    numerator: torch.Tensor | None = None
    denominator = 0.0
    for value, weight in zip(values, weights, strict=True):
        weighted = value * float(weight)
        numerator = weighted if numerator is None else numerator + weighted
        denominator += float(weight)
    if numerator is None:
        raise ValueError("weighted tensor mean requires at least one value")
    return numerator / max(denominator, 1e-12)
