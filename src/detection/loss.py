"""Differentiable CE adapter for recursive detection targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.detection.objective import RecursiveDetectionTargets, SemanticRole
from src.metrics.detection_sequence import (
    coordinate_token_accuracy_event,
    coordinate_token_cross_entropy_event,
    description_token_accuracy_event,
    description_token_cross_entropy_event,
    object_entry_exact_match_event,
    schema_token_accuracy_event,
    schema_token_cross_entropy_event,
)
from src.metrics.events import MetricEvent, last_event, ratio_event, weighted_mean_event

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
_SPAN_CATEGORY_TO_CANONICAL_SEGMENT = {
    "schema": "schema",
    "desc_text": "description",
    "coord": "coordinate",
    "object_control": "object_control",
    "separator": "separator",
    "stop": "stop",
    "other": "other",
}
_SEMANTIC_ROLE_TO_SPAN_CATEGORY = {
    SemanticRole.SCHEMA_CONTROL.value: "schema",
    SemanticRole.DESC_IDENTITY.value: "desc_text",
    SemanticRole.BBOX_COORD.value: "coord",
    SemanticRole.ENTRY_TRIE_DECISION.value: "object_control",
    SemanticRole.OBJECT_CONTROL.value: "object_control",
    SemanticRole.SEPARATOR_CONTINUE.value: "separator",
    SemanticRole.TERMINAL_STOP.value: "stop",
    SemanticRole.CHAT_STOP.value: "stop",
}
_COMPACT_METRIC_SUMMARY_CHUNK_SIZE = 64


@dataclass(frozen=True)
class RecursiveDetectionLossWeights:
    support_weight: float = 1.0
    balance_weight: float = 1.0


@dataclass(frozen=True)
class RecursiveDetectionLossResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    per_position_losses: tuple[dict[int, torch.Tensor], ...]
    metric_events: tuple[MetricEvent, ...] = ()


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
    metric_events = (
        weighted_mean_event(
            "detection_sequence/objective/recursive_detection_ce/loss_per_sample",
            float(loss.detach().item()),
            float(batch_size),
            unit="sample",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
        ),
        last_event(
            "detection_sequence/objective/recursive_detection_ce/batch_size",
            float(batch_size),
            unit="batch",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
        ),
    ) + _summarize_compact_recursive_detection_metric_events(
        logits=batch_logits,
        targets=targets,
    )
    return RecursiveDetectionLossResult(
        loss=loss,
        metrics=metrics,
        per_position_losses=tuple(per_position_losses),
        metric_events=metric_events,
    )


def _summarize_compact_recursive_detection_metric_events(
    *,
    logits: torch.Tensor,
    targets: Sequence[RecursiveDetectionTargets],
) -> tuple[MetricEvent, ...]:
    """Summarize Phase-1 compact recursive-detection diagnostics from logits."""

    category_stats = {
        category: {
            "total": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "ce_sum": 0.0,
        }
        for category in _SPAN_CATEGORY_TO_CANONICAL_SEGMENT
    }
    object_token_correct: dict[str, list[bool]] = {}

    token_batch_indices: list[int] = []
    token_time_indices: list[int] = []
    teacher_token_ids: list[int] = []
    category_ids: list[int] = []
    object_instance_ids: list[str | None] = []
    category_to_id = {
        category: index
        for index, category in enumerate(_SPAN_CATEGORY_TO_CANONICAL_SEGMENT)
    }

    for batch_index, recursive_targets in enumerate(targets):
        atom_by_id = {atom.atom_id: atom for atom in recursive_targets.loss_atoms}
        atom_by_position: dict[int, object] = {}
        for atom in recursive_targets.loss_atoms:
            for position in atom.token_positions:
                atom_by_position.setdefault(position, atom)

        for target in recursive_targets.token_targets:
            atom = None
            if target.loss_atom_id is not None:
                atom = atom_by_id.get(target.loss_atom_id)
            if atom is None:
                atom = atom_by_position.get(target.position)

            role = target.semantic_role
            if role is None and atom is not None:
                role = getattr(atom, "semantic_role", None)
            category = _span_category_for_semantic_role(role)

            token_batch_indices.append(batch_index)
            token_time_indices.append(int(target.position) - 1)
            teacher_token_ids.append(int(target.teacher_token_id))
            category_ids.append(category_to_id[category])
            object_instance_ids.append(_object_instance_id_for_target(target, atom))

    with torch.no_grad():
        metric_logits = logits.detach()
        if token_batch_indices:
            device = metric_logits.device
            batch_index_tensor = torch.tensor(
                token_batch_indices,
                device=device,
                dtype=torch.long,
            )
            time_index_tensor = torch.tensor(
                token_time_indices,
                device=device,
                dtype=torch.long,
            )
            teacher_id_tensor = torch.tensor(
                teacher_token_ids,
                device=device,
                dtype=torch.long,
            )
            category_id_tensor = torch.tensor(
                category_ids,
                device=device,
                dtype=torch.long,
            )
            for start in range(
                0,
                len(token_batch_indices),
                _COMPACT_METRIC_SUMMARY_CHUNK_SIZE,
            ):
                end = min(
                    start + _COMPACT_METRIC_SUMMARY_CHUNK_SIZE,
                    len(token_batch_indices),
                )
                chunk_logits = metric_logits[
                    batch_index_tensor[start:end],
                    time_index_tensor[start:end],
                ].float()
                chunk_teacher_ids = teacher_id_tensor[start:end]
                chunk_category_ids = category_id_tensor[start:end]
                teacher_logits = chunk_logits.gather(
                    1,
                    chunk_teacher_ids.unsqueeze(1),
                ).squeeze(1)
                teacher_ce = torch.logsumexp(chunk_logits, dim=-1) - teacher_logits
                top1_correct = chunk_logits.argmax(dim=-1).eq(chunk_teacher_ids)
                top_k = min(5, int(chunk_logits.shape[-1]))
                top5_correct = torch.topk(chunk_logits, k=top_k, dim=-1).indices.eq(
                    chunk_teacher_ids.unsqueeze(1)
                ).any(dim=-1)

                for category, category_id in category_to_id.items():
                    mask = chunk_category_ids.eq(category_id)
                    total = float(mask.sum().item())
                    if total == 0.0:
                        continue
                    stats = category_stats[category]
                    stats["total"] += total
                    stats["top1"] += float(top1_correct[mask].sum().item())
                    stats["top5"] += float(top5_correct[mask].sum().item())
                    stats["ce_sum"] += float(teacher_ce[mask].sum().item())

                top1_correct_values = tuple(
                    bool(value) for value in top1_correct.cpu().tolist()
                )
                for object_instance_id, is_top1 in zip(
                    object_instance_ids[start:end],
                    top1_correct_values,
                    strict=True,
                ):
                    if object_instance_id is not None:
                        object_token_correct.setdefault(object_instance_id, []).append(
                            is_top1
                        )

    events: list[MetricEvent] = []
    for category in _SPAN_CATEGORY_TO_CANONICAL_SEGMENT:
        stats = category_stats[category]
        total = stats["total"]
        events.append(
            _span_token_accuracy_event(
                category,
                stats["top1"],
                total,
                top_k=1,
            )
        )
        events.append(
            _span_token_accuracy_event(
                category,
                stats["top5"],
                total,
                top_k=5,
            )
        )
        ce_value = stats["ce_sum"] / total if total > 0.0 else 0.0
        events.append(_span_token_cross_entropy_event(category, ce_value, total))

    object_total = float(len(object_token_correct))
    object_correct = float(
        sum(1 for token_results in object_token_correct.values() if all(token_results))
    )
    events.append(
        object_entry_exact_match_event(
            object_correct,
            object_total,
            object_scope="object_entry",
            metric_surface="training_logits",
        )
    )
    return tuple(events)


def _span_category_for_semantic_role(role: object) -> str:
    if isinstance(role, SemanticRole):
        role_value = role.value
    elif role is None:
        role_value = None
    else:
        role_value = str(role)
    if role_value is None:
        return "other"
    return _SEMANTIC_ROLE_TO_SPAN_CATEGORY.get(role_value, "other")


def _object_instance_id_for_target(target: object, atom: object | None) -> str | None:
    atom_object_id = getattr(atom, "object_instance_id", None) if atom is not None else None
    if atom_object_id is not None:
        return str(atom_object_id)
    target_object_id = getattr(target, "object_instance_id", None)
    if target_object_id is not None:
        return str(target_object_id)
    return None


def _span_token_accuracy_event(
    category: str,
    correct: float,
    total: float,
    *,
    top_k: int,
) -> MetricEvent:
    if category == "schema":
        return schema_token_accuracy_event(correct, total, top_k=top_k)
    if category == "desc_text":
        return description_token_accuracy_event(correct, total, top_k=top_k)
    if category == "coord":
        return coordinate_token_accuracy_event(correct, total, top_k=top_k)
    segment = _SPAN_CATEGORY_TO_CANONICAL_SEGMENT[category]
    return ratio_event(
        f"detection_sequence/{segment}/token_acc/full_vocab/top{top_k}",
        correct,
        total,
        unit="token",
        semantic_role=segment,
        token_role=segment,
        vocab_scope="full_vocab",
        metric_surface="training_logits",
    )


def _span_token_cross_entropy_event(
    category: str,
    value: float,
    total: float,
) -> MetricEvent:
    if category == "schema":
        return schema_token_cross_entropy_event(value, total)
    if category == "desc_text":
        return description_token_cross_entropy_event(value, total)
    if category == "coord":
        return coordinate_token_cross_entropy_event(value, total)
    segment = _SPAN_CATEGORY_TO_CANONICAL_SEGMENT[category]
    return weighted_mean_event(
        f"detection_sequence/{segment}/token_ce/full_vocab",
        value,
        total,
        unit="token",
        semantic_role=segment,
        token_role=segment,
        vocab_scope="full_vocab",
        metric_surface="training_logits",
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
            float(weights.support_weight) * support_loss
            + float(weights.balance_weight) * balance_loss
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
            state_weight = float(target.state_weight)
            loss_weight = _target_loss_weight(target)
            weighted_losses.append(
                per_position_losses[target.position] * state_weight * loss_weight
            )
            state_weight_sum += state_weight
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

    target_by_position = {
        target.position: target for target in recursive_targets.token_targets
    }
    atom_losses = {
        atom.atom_id: torch.stack(
            [
                per_position_losses[position]
                * _target_loss_weight(target_by_position[position])
                for position in atom.token_positions
            ]
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


def _target_loss_weight(target: object) -> float:
    weight = float(getattr(target, "loss_weight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("TokenTarget.loss_weight must be a non-negative finite float")
    return weight
