"""Differentiable CE adapter for recursive detection targets."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    full_vocab_coord_soft_ce,
    full_vocab_coord_support_balance_ce,
)
from src.detection.objective import RecursiveDetectionTargets, SemanticRole
from src.detection.tokenization import TokenRole
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
from src.training.teacher_forcing.ir import TeacherForcingTargetIR

_OBJECT_ROLE_WEIGHTS = {
    SemanticRole.DESC_IDENTITY: 0.35,
    SemanticRole.BBOX_COORD: 0.45,
    SemanticRole.ENTRY_TRIE_DECISION: 0.15,
    SemanticRole.OBJECT_CONTROL: 0.05,
}
_IMAGE_MIXTURE_WEIGHTS = {
    "objects": 1.00,
    "schema": 0.10,
}
_COORD_SLOT_INDEX = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
_SPAN_CATEGORY_TO_CANONICAL_SEGMENT = {
    "schema": "schema",
    "desc_text": "description",
    "coord": "coordinate",
    "object_control": "object_control",
    "separator": "separator",
    "stop": "stop",
    "other": "other",
}
_SPAN_CATEGORY_TO_TARGET_MIX_NAME = {
    "schema": "schema",
    "desc_text": "desc",
    "coord": "coord",
    "object_control": "object_control",
    "separator": "separator",
    "stop": "eos",
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
    coord_soft_ce: CoordSoftTargetRuntimeConfig | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "support_weight",
            "balance_weight",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"RecursiveDetectionLossWeights.{field_name} must be finite and >= 0"
                )
        if float(self.support_weight) + float(self.balance_weight) <= 0.0:
            raise ValueError(
                "RecursiveDetectionLossWeights support and balance weights must sum to > 0"
            )


@dataclass(frozen=True)
class RecursiveDetectionLossResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    per_position_losses: tuple[dict[int, torch.Tensor], ...]
    metric_events: tuple[MetricEvent, ...] = ()


def _loss_precision_context(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _loss_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32)


def _log_softmax_loss(logits: torch.Tensor, *, dim: int) -> torch.Tensor:
    with _loss_precision_context(logits):
        return F.log_softmax(_loss_float(logits), dim=dim)


def _logsumexp_loss(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    with _loss_precision_context(values):
        return torch.logsumexp(_loss_float(values), dim=dim)


def _stack_loss_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([_loss_float(value) for value in values])


def support_balance_loss(
    logits: torch.Tensor,
    positive_token_ids: torch.Tensor,
    q: torch.Tensor,
    *,
    support_weight: float,
    balance_weight: float,
) -> torch.Tensor:
    """Sparse local multi-positive support/balance loss for one logit row or batch."""

    positive_token_ids = positive_token_ids.to(device=logits.device, dtype=torch.long)
    q = q.to(device=logits.device, dtype=torch.float32)
    if positive_token_ids.ndim != 1 or positive_token_ids.numel() == 0:
        raise ValueError("positive_token_ids must be a non-empty 1D tensor")
    if positive_token_ids.unique().numel() != positive_token_ids.numel():
        raise ValueError("positive_token_ids must contain unique vocabulary token ids")
    if q.ndim != 1 or q.numel() != positive_token_ids.numel():
        raise ValueError("q must be a 1D tensor aligned with positive_token_ids")
    if not torch.isfinite(q).all() or torch.any(q < 0):
        raise ValueError("q must contain finite non-negative weights")
    with _loss_precision_context(logits):
        q_sum = q.sum()
    if float(q_sum.detach().cpu().item()) <= 0.0:
        raise ValueError("q must have positive total mass")
    with _loss_precision_context(logits):
        q = q / q_sum.clamp_min(1e-12)
    for name, value in (
        ("support_weight", support_weight),
        ("balance_weight", balance_weight),
    ):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")

    log_probs = _log_softmax_loss(logits, dim=-1)
    valid_log_probs = log_probs.index_select(dim=-1, index=positive_token_ids)
    log_valid_mass = _logsumexp_loss(valid_log_probs, dim=-1)
    support = -log_valid_mass
    with _loss_precision_context(valid_log_probs):
        balance = -(q * (valid_log_probs - log_valid_mass.unsqueeze(-1))).sum(dim=-1)
    return float(support_weight) * support + float(balance_weight) * balance


def _apply_type_gate_loss(
    position_loss: torch.Tensor,
    *,
    step_log_probs: torch.Tensor,
    target: object,
    vocab_size: int,
) -> torch.Tensor:
    return _loss_float(position_loss) + _compute_type_gate_loss(
        step_log_probs=step_log_probs,
        target=target,
        vocab_size=vocab_size,
    )


def _compute_type_gate_loss(
    *,
    step_log_probs: torch.Tensor,
    target: object,
    vocab_size: int,
) -> torch.Tensor:
    type_gate_token_ids = tuple(int(token_id) for token_id in getattr(target, "type_gate_token_ids", ()))
    if not type_gate_token_ids:
        return _loss_float(step_log_probs).sum() * 0.0
    type_gate_weight = float(getattr(target, "type_gate_weight", 0.0))
    if not math.isfinite(type_gate_weight) or type_gate_weight < 0.0:
        raise ValueError("TokenTarget.type_gate_weight must be finite and >= 0")
    if type_gate_weight == 0.0:
        return _loss_float(step_log_probs).sum() * 0.0
    if len(set(type_gate_token_ids)) != len(type_gate_token_ids):
        raise ValueError("TokenTarget.type_gate_token_ids must be unique")
    for token_id in type_gate_token_ids:
        _validate_token_id(
            token_id=token_id,
            vocab_size=vocab_size,
            label="type-gate token id",
        )
    type_ids = torch.tensor(
        type_gate_token_ids,
        device=step_log_probs.device,
        dtype=torch.long,
    )
    type_loss = -_logsumexp_loss(step_log_probs.index_select(0, type_ids), dim=0)
    return type_gate_weight * type_loss


def compute_recursive_detection_ce_batch_loss(
    *,
    logits: torch.Tensor,
    targets: Sequence[RecursiveDetectionTargets],
    weights: RecursiveDetectionLossWeights | None = None,
) -> RecursiveDetectionLossResult:
    """Compute the differentiable recursive-detection CE loss over a batch."""

    if weights is None:
        weights = RecursiveDetectionLossWeights()

    allow_position_at_time_dim = logits.ndim == 2
    batch_logits = _normalize_logits_shape(logits)
    batch_size, time_steps, vocab_size = batch_logits.shape
    if len(targets) != batch_size:
        raise ValueError(
            f"targets length {len(targets)} must match logits batch size {batch_size}"
        )

    sample_losses: list[torch.Tensor] = []
    per_position_losses: list[dict[int, torch.Tensor]] = []
    for batch_index, recursive_targets in enumerate(targets):
        if isinstance(recursive_targets, TeacherForcingTargetIR):
            raise TypeError(
                "teacher_forcing_target_ir is not a recursive_detection_ce "
                "target; route it through the teacher_forcing objective runner"
            )
        sample_loss, sample_position_losses = _compute_sample_loss(
            logits=batch_logits[batch_index],
            recursive_targets=recursive_targets,
            vocab_size=vocab_size,
            time_steps=time_steps,
            weights=weights,
            allow_position_at_time_dim=allow_position_at_time_dim,
        )
        sample_losses.append(sample_loss)
        per_position_losses.append(sample_position_losses)

    loss = (
        _stack_loss_values(sample_losses).mean()
        if sample_losses
        else _loss_float(batch_logits).sum() * 0.0
    )
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
    ) + _recursive_objective_diagnostic_events(
        logits=batch_logits,
        targets=targets,
        weights=weights,
    ) + _recursive_target_mix_events(
        targets=targets,
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


def _recursive_objective_diagnostic_events(
    *,
    logits: torch.Tensor,
    targets: Sequence[RecursiveDetectionTargets],
    weights: RecursiveDetectionLossWeights,
) -> tuple[MetricEvent, ...]:
    """Summarize objective-internal diagnostics without affecting gradients."""

    def _event(
        key: str,
        value: torch.Tensor | float,
        weight: float = 1.0,
    ) -> MetricEvent:
        value_float = float(torch.as_tensor(value).detach().cpu().item())
        return weighted_mean_event(
            key,
            value_float,
            float(weight),
            unit="token",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
            diagnostic_only=True,
        )

    events: list[MetricEvent] = []
    with torch.no_grad():
        metric_logits = logits.detach()
        _, time_steps, vocab_size = metric_logits.shape
        for batch_index, recursive_targets in enumerate(targets):
            atom_by_id = {atom.atom_id: atom for atom in recursive_targets.loss_atoms}
            atom_by_position: dict[int, object] = {}
            for atom in recursive_targets.loss_atoms:
                for position in atom.token_positions:
                    atom_by_position.setdefault(position, atom)

            def _role_for_target(target: object) -> object:
                atom = None
                loss_atom_id = getattr(target, "loss_atom_id", None)
                if loss_atom_id is not None:
                    atom = atom_by_id.get(loss_atom_id)
                if atom is None:
                    atom = atom_by_position.get(int(getattr(target, "position")))
                role = getattr(target, "semantic_role", None)
                if role is None and atom is not None:
                    role = getattr(atom, "semantic_role", None)
                return role

            for target in recursive_targets.token_targets:
                if target.position <= 0 or target.position > time_steps:
                    continue
                step_log_probs = _log_softmax_loss(
                    metric_logits[batch_index, target.position - 1],
                    dim=-1,
                )
                span_category = _span_category_for_semantic_role(
                    _role_for_target(target)
                )
                if weights.coord_soft_ce is not None and _coord_soft_ce_applies(
                    target, weights.coord_soft_ce
                ):
                    if _coord_soft_target_distribution_name(
                        weights.coord_soft_ce
                    ) == "instance_trie_gaussian":
                        (
                            candidates,
                            current_slot,
                            teacher_prefix_values,
                            teacher_coord_value,
                        ) = _instance_trie_gaussian_candidates(
                            target,
                            weights.coord_soft_ce,
                        )
                        coord_result = full_vocab_coord_soft_ce(
                            metric_logits[batch_index, target.position - 1],
                            candidates,
                            weights.coord_soft_ce,
                            current_slot=current_slot,
                            teacher_prefix_values=teacher_prefix_values,
                            teacher_coord_value=teacher_coord_value,
                        )
                    else:
                        if target.kind == "trie_multi_positive":
                            _validate_trie_multi_positive_target(
                                target,
                                vocab_size=vocab_size,
                            )
                        candidates = _coord_soft_target_candidates(target)
                        current_slot = None
                        coord_result = full_vocab_coord_support_balance_ce(
                            metric_logits[batch_index, target.position - 1],
                            candidates,
                            weights.coord_soft_ce,
                            support_weight=float(weights.support_weight),
                            balance_weight=float(weights.balance_weight),
                        )
                    for metric_name, metric_value in (
                        ("weighted_loss", coord_result.weighted_loss),
                        ("support_loss", coord_result.support_loss),
                        ("support_mass", coord_result.support_mass),
                        ("outside_support_mass", coord_result.outside_support_mass),
                        ("balance_loss", coord_result.balance_loss),
                        ("pure_soft_ce_equiv", coord_result.pure_soft_ce_equiv),
                        ("target_entropy", coord_result.target_entropy),
                        ("kl_like", coord_result.kl_like),
                        ("target_peak_prob", coord_result.peak_prob),
                        ("target_perplexity", coord_result.perplexity),
                        (
                            "target_effective_support_size",
                            coord_result.effective_support_size,
                        ),
                        ("target_std", coord_result.target_std),
                        ("target_r95_radius", coord_result.target_r95_radius),
                        ("candidate_count", coord_result.candidate_count),
                        ("support_bin_count", coord_result.support_bin_count),
                    ):
                        events.append(
                            _event(
                                f"recursive_detection_ce/coord_soft_ce/{metric_name}",
                                metric_value,
                            )
                        )
                    if current_slot is not None:
                        for metric_name, metric_value in (
                            ("posterior_entropy", coord_result.posterior_entropy),
                            ("posterior_top1", coord_result.posterior_top1),
                            (
                                "effective_candidate_count",
                                coord_result.effective_candidate_count,
                            ),
                            ("target_entropy", coord_result.target_entropy),
                            ("target_peak_prob", coord_result.peak_prob),
                            ("target_r95_radius", coord_result.target_r95_radius),
                        ):
                            if metric_value is None:
                                continue
                            events.append(
                                _event(
                                    "recursive_detection_ce/coord_soft_ce/"
                                    f"{current_slot}/{metric_name}",
                                    metric_value,
                                )
                            )
                    events.append(
                        _event("recursive_detection_ce/coord_soft_ce/enabled", 1.0)
                    )
                    events.append(
                        _event(
                            "recursive_detection_ce/coord_soft_ce/support_mixture",
                            1.0 if coord_result.support_mixture else 0.0,
                        )
                    )
                    continue

                if target.kind == "trie_multi_positive":
                    child_token_ids = tuple(
                        int(branch_target.token_id)
                        for branch_target in target.trie_branch_targets
                    )
                    if child_token_ids:
                        child_ids = torch.tensor(
                            child_token_ids,
                            device=metric_logits.device,
                            dtype=torch.long,
                        )
                        valid_log_probs = step_log_probs.index_select(
                            dim=-1,
                            index=child_ids,
                        )
                        log_valid_mass = _logsumexp_loss(valid_log_probs, dim=-1)
                        child_weights = torch.tensor(
                            [
                                float(branch_target.multiplicity)
                                for branch_target in target.trie_branch_targets
                            ],
                            device=metric_logits.device,
                            dtype=torch.float32,
                        )
                        q = child_weights / child_weights.sum().clamp_min(1e-12)
                        support_loss = -log_valid_mass
                        with _loss_precision_context(valid_log_probs):
                            valid_child_log_probs = valid_log_probs - log_valid_mass
                            valid_child_probs = torch.exp(valid_child_log_probs)
                            balance_loss = -(
                                q * valid_child_log_probs
                            ).sum(dim=-1)
                            valid_child_entropy = -(
                                valid_child_probs * valid_child_log_probs
                            ).sum(dim=-1)
                            uniform_log_prob = -math.log(float(len(child_token_ids)))
                            valid_child_kl_to_uniform = (
                                uniform_log_prob - valid_child_log_probs
                            ).mean(dim=-1)
                        events.append(
                            _event(
                                "recursive_detection_ce/trie_valid_mass",
                                torch.exp(log_valid_mass),
                            )
                        )
                        events.append(
                            _event("recursive_detection_ce/support_loss", support_loss)
                        )
                        events.append(
                            _event("recursive_detection_ce/balance_loss", balance_loss)
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/trie_valid_children",
                                float(len(child_token_ids)),
                            )
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/entry/valid_child_entropy",
                                valid_child_entropy,
                            )
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/entry/valid_child_kl_to_uniform",
                                valid_child_kl_to_uniform,
                            )
                        )
                if target.type_gate_token_ids and float(target.type_gate_weight) > 0.0:
                    unique_type_gate_ids = tuple(
                        dict.fromkeys(
                            int(token_id) for token_id in target.type_gate_token_ids
                        )
                    )
                    type_gate_ids = torch.tensor(
                        unique_type_gate_ids,
                        device=metric_logits.device,
                        dtype=torch.long,
                    )
                    if int(type_gate_ids.numel()) > 0:
                        invalid_type_gate_id = torch.any(type_gate_ids < 0) or torch.any(
                            type_gate_ids >= vocab_size
                        )
                        if invalid_type_gate_id:
                            continue
                        type_gate_log_mass = _logsumexp_loss(
                            step_log_probs.index_select(dim=-1, index=type_gate_ids),
                            dim=-1,
                        )
                        type_gate_loss = -type_gate_log_mass * float(target.type_gate_weight)
                        events.append(
                            _event(
                                "recursive_detection_ce/type_gate_loss",
                                type_gate_loss,
                            )
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/type_gate_allowed_mass",
                                torch.exp(type_gate_log_mass),
                            )
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/type_gate_allowed_tokens",
                                float(type_gate_ids.numel()),
                            )
                        )
                        events.append(
                            _event(
                                "recursive_detection_ce/type_gate_weight",
                                float(target.type_gate_weight),
                            )
                        )

                if span_category == "stop":
                    unweighted_ce = -step_log_probs[int(target.teacher_token_id)]
                    events.append(
                        _event("recursive_detection_ce/eos_unweighted_ce", unweighted_ce)
                    )
    return tuple(events)


def _recursive_target_mix_events(
    *,
    targets: Sequence[RecursiveDetectionTargets],
) -> tuple[MetricEvent, ...]:
    """Summarize which recursive targets contributed to a logged batch."""

    events: list[MetricEvent] = []
    target_count = 0.0
    kind_counts = {
        "hard_ce": 0.0,
        "trie_multi_positive": 0.0,
    }
    category_counts = {
        category: 0.0
        for category in _SPAN_CATEGORY_TO_CANONICAL_SEGMENT
    }
    positive_children_total = 0.0
    positive_children_count = 0.0
    loss_weight_total = 0.0
    state_weight_total = 0.0

    for recursive_targets in targets:
        sample_target_count = float(len(recursive_targets.token_targets))
        events.append(
            weighted_mean_event(
                "recursive_detection_ce/target_mix/targets_per_sample",
                sample_target_count,
                1.0,
                unit="sample",
                semantic_role="recursive_detection_ce",
                metric_surface="training_logits",
                diagnostic_only=True,
            )
        )
        for target in recursive_targets.token_targets:
            target_count += 1.0
            kind_counts[str(target.kind)] = kind_counts.get(str(target.kind), 0.0) + 1.0
            category = _span_category_for_semantic_role(target.semantic_role)
            category_counts[category] += 1.0
            loss_weight_total += float(target.loss_weight)
            state_weight_total += float(target.state_weight)

            if target.kind == "trie_multi_positive":
                positive_children_total += float(len(target.trie_branch_targets))
                positive_children_count += 1.0

    def _append_fraction(name: str, numerator: float) -> None:
        events.append(
            ratio_event(
                f"recursive_detection_ce/target_mix/{name}_fraction",
                numerator,
                target_count,
                unit="token",
                semantic_role="recursive_detection_ce",
                metric_surface="training_logits",
                diagnostic_only=True,
            )
        )

    _append_fraction("hard_ce", kind_counts.get("hard_ce", 0.0))
    _append_fraction(
        "trie_multi_positive",
        kind_counts.get("trie_multi_positive", 0.0),
    )
    for category, metric_name in _SPAN_CATEGORY_TO_TARGET_MIX_NAME.items():
        _append_fraction(metric_name, category_counts[category])
    _append_fraction(
        "non_eos",
        target_count - category_counts["stop"],
    )

    events.append(
        weighted_mean_event(
            "recursive_detection_ce/target_mix/positive_children_per_trie_target",
            (
                positive_children_total / positive_children_count
                if positive_children_count > 0.0
                else 0.0
            ),
            positive_children_count,
            unit="token",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
            diagnostic_only=True,
        )
    )
    events.append(
        weighted_mean_event(
            "recursive_detection_ce/target_mix/effective_loss_weight_mean",
            loss_weight_total / target_count if target_count > 0.0 else 0.0,
            target_count,
            unit="token",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
            diagnostic_only=True,
        )
    )
    events.append(
        weighted_mean_event(
            "recursive_detection_ce/target_mix/state_weight_mean",
            state_weight_total / target_count if target_count > 0.0 else 0.0,
            target_count,
            unit="token",
            semantic_role="recursive_detection_ce",
            metric_surface="training_logits",
            diagnostic_only=True,
        )
    )
    return tuple(events)


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
                chunk_logits = _loss_float(metric_logits[
                    batch_index_tensor[start:end],
                    time_index_tensor[start:end],
                ])
                chunk_teacher_ids = teacher_id_tensor[start:end]
                chunk_category_ids = category_id_tensor[start:end]
                teacher_logits = chunk_logits.gather(
                    1,
                    chunk_teacher_ids.unsqueeze(1),
                ).squeeze(1)
                teacher_ce = _logsumexp_loss(chunk_logits, dim=-1) - teacher_logits
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
    allow_position_at_time_dim: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    per_position_losses: dict[int, torch.Tensor] = {}
    per_position_main_losses: dict[int, torch.Tensor] = {}
    seen_positions: set[int] = set()
    for target in recursive_targets.token_targets:
        at_or_past_boundary = (
            target.position > time_steps
            if allow_position_at_time_dim
            else target.position >= time_steps
        )
        if target.position <= 0 or at_or_past_boundary:
            raise ValueError(
                "TokenTarget.position must satisfy 0 < position < logits_time_dim; "
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
        step_log_probs = _log_softmax_loss(logits[target.position - 1], dim=-1)
        if weights.coord_soft_ce is not None and _coord_soft_ce_applies(
            target, weights.coord_soft_ce
        ):
            if _coord_soft_target_distribution_name(
                weights.coord_soft_ce
            ) == "instance_trie_gaussian":
                (
                    coord_candidates,
                    current_slot,
                    teacher_prefix_values,
                    teacher_coord_value,
                ) = _instance_trie_gaussian_candidates(target, weights.coord_soft_ce)
                coord_result = full_vocab_coord_soft_ce(
                    logits[target.position - 1],
                    coord_candidates,
                    weights.coord_soft_ce,
                    current_slot=current_slot,
                    teacher_prefix_values=teacher_prefix_values,
                    teacher_coord_value=teacher_coord_value,
                )
            else:
                if target.kind == "trie_multi_positive":
                    _validate_trie_multi_positive_target(target, vocab_size=vocab_size)
                coord_result = full_vocab_coord_support_balance_ce(
                    logits[target.position - 1],
                    _coord_soft_target_candidates(target),
                    weights.coord_soft_ce,
                    support_weight=float(weights.support_weight),
                    balance_weight=float(weights.balance_weight),
                )
            position_loss = coord_result.weighted_loss
            per_position_main_losses[target.position] = _loss_float(position_loss)
            per_position_losses[target.position] = _apply_type_gate_loss(
                position_loss,
                step_log_probs=step_log_probs,
                target=target,
                vocab_size=vocab_size,
            )
            continue

        if target.kind == "hard_ce":
            position_loss = -step_log_probs[target.teacher_token_id]
            per_position_main_losses[target.position] = _loss_float(position_loss)
            per_position_losses[target.position] = _apply_type_gate_loss(
                position_loss,
                step_log_probs=step_log_probs,
                target=target,
                vocab_size=vocab_size,
            )
            continue

        child_token_ids, child_weights = _validate_trie_multi_positive_target(
            target,
            vocab_size=vocab_size,
        )

        child_ids = torch.tensor(child_token_ids, device=logits.device, dtype=torch.long)
        q = torch.tensor(child_weights, device=logits.device, dtype=torch.float32)
        position_loss = support_balance_loss(
            logits[target.position - 1],
            child_ids,
            q,
            support_weight=float(weights.support_weight),
            balance_weight=float(weights.balance_weight),
        )
        if (
            getattr(target, "token_role", None) is TokenRole.DESC
            and getattr(target, "semantic_role", None) is SemanticRole.DESC_IDENTITY
        ):
            position_loss = position_loss + (-step_log_probs[target.teacher_token_id])
        per_position_main_losses[target.position] = _loss_float(position_loss)
        per_position_losses[target.position] = _apply_type_gate_loss(
            position_loss,
            step_log_probs=step_log_probs,
            target=target,
            vocab_size=vocab_size,
        )

    sample_loss = _normalize_sample_loss(
        recursive_targets=recursive_targets,
        per_position_losses=per_position_losses,
        per_position_main_losses=per_position_main_losses,
        weights=weights,
        device=logits.device,
    )
    return sample_loss, per_position_losses


def _coord_soft_target_candidates(
    target: object,
) -> tuple[CoordSoftTargetCandidate, ...]:
    specs = tuple(getattr(target, "coord_soft_targets", ()) or ())
    if not specs:
        raise ValueError(
            "coord_soft_ce is enabled for a coordinate TokenTarget, but "
            "TokenTarget.coord_soft_targets is empty"
        )
    return tuple(
        CoordSoftTargetCandidate(
            object_instance_id=str(getattr(spec, "object_instance_id")),
            slot_name=getattr(spec, "slot_name"),
            bbox_xyxy=tuple(int(value) for value in getattr(spec, "bbox_xyxy")),
            probability=float(getattr(spec, "probability")),
        )
        for spec in specs
    )


def _coord_soft_target_distribution_name(
    cfg: CoordSoftTargetRuntimeConfig,
) -> str:
    return str(getattr(cfg, "target_distribution", ""))


def _coord_soft_ce_applies(
    target: object,
    cfg: CoordSoftTargetRuntimeConfig,
) -> bool:
    if _coord_soft_target_distribution_name(cfg) == "instance_trie_gaussian":
        token_role = _target_token_role_value(target)
        specs = tuple(getattr(target, "coord_instance_candidates", ()) or ())
        if specs and token_role != "coord":
            raise ValueError(
                "TokenTarget.coord_instance_candidates may only be attached to "
                "coord token targets; got token_role="
                f"{token_role!r} at position {getattr(target, 'position', '?')}"
            )
        return token_role == "coord"

    specs = tuple(getattr(target, "coord_soft_targets", ()) or ())
    token_role = _target_token_role_value(target)
    if specs:
        if token_role != "coord":
            raise ValueError(
                "TokenTarget.coord_soft_targets may only be attached to coord token "
                f"targets; got token_role={token_role!r} at position "
                f"{getattr(target, 'position', '?')}"
            )
        return True
    if token_role == "coord":
        raise ValueError(
            "coord_soft_ce is enabled for a coordinate TokenTarget, but "
            "TokenTarget.coord_soft_targets is empty"
        )
    return False


def _instance_trie_gaussian_candidates(
    target: object,
    cfg: CoordSoftTargetRuntimeConfig,
) -> tuple[tuple[CoordSoftTargetCandidate, ...], str, dict[str, int], int]:
    specs = tuple(getattr(target, "coord_instance_candidates", ()) or ())
    if not specs:
        raise ValueError(
            "instance_trie_gaussian coordinate TokenTarget.coord_instance_candidates "
            "is empty"
        )

    slot_name = getattr(target, "coord_slot_name", None)
    if slot_name is None:
        raise ValueError(
            "TokenTarget.coord_slot_name is required for instance_trie_gaussian"
        )
    slot_name = str(getattr(slot_name, "value", slot_name))
    if slot_name not in _COORD_SLOT_INDEX:
        raise ValueError(
            f"TokenTarget.coord_slot_name unsupported for instance_trie_gaussian: "
            f"{slot_name!r}"
        )

    teacher_object_id = getattr(target, "object_instance_id", None)
    if not isinstance(teacher_object_id, str) or not teacher_object_id:
        raise ValueError(
            "TokenTarget.object_instance_id is required for instance_trie_gaussian "
            "coordinate targets"
        )

    teacher_token_id = int(getattr(target, "teacher_token_id"))
    coord_token_start = int(cfg.coord_token_start)
    coord_token_end = int(cfg.coord_token_end)
    if teacher_token_id < coord_token_start or teacher_token_id > coord_token_end:
        raise ValueError(
            "instance_trie_gaussian teacher token id is outside the resolved "
            f"coord token range [{coord_token_start}, {coord_token_end}]"
        )

    coord_value_min = 0
    coord_value_max = int(cfg.coord_bins) - 1
    candidates: list[CoordSoftTargetCandidate] = []
    teacher_bboxes: list[tuple[int, int, int, int]] = []
    object_id_counts: dict[str, int] = {}
    for spec in specs:
        object_id = getattr(spec, "object_instance_id", None)
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(
                "CoordInstanceCandidateSpec candidate object_instance_id must be "
                "a non-empty string"
            )
        object_id_counts[object_id] = object_id_counts.get(object_id, 0) + 1

        bbox = getattr(spec, "bbox_xyxy", None)
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError(
                "CoordInstanceCandidateSpec candidate bbox must be a "
                "four-integer xyxy tuple"
            )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in bbox
        ):
            raise ValueError(
                "CoordInstanceCandidateSpec candidate bbox has non-integer "
                "coordinates"
            )
        bbox_xyxy = tuple(int(value) for value in bbox)
        if not all(coord_value_min <= value <= coord_value_max for value in bbox_xyxy):
            raise ValueError(
                "CoordInstanceCandidateSpec candidate bbox coordinate outside "
                "resolved value domain "
                f"[{coord_value_min}, {coord_value_max}]"
            )
        x1, y1, x2, y2 = bbox_xyxy
        if not (x1 < x2 and y1 < y2):
            raise ValueError(
                "CoordInstanceCandidateSpec candidate bbox violates x1<x2/y1<y2"
            )

        if object_id == teacher_object_id:
            teacher_bboxes.append(bbox_xyxy)
        candidates.append(
            CoordSoftTargetCandidate(
                object_instance_id=object_id,
                slot_name=slot_name,  # type: ignore[arg-type]
                bbox_xyxy=bbox_xyxy,
                probability=1.0,
            )
        )

    duplicate_object_ids = tuple(
        object_id for object_id, count in object_id_counts.items() if count > 1
    )
    if duplicate_object_ids and any(
        object_id != teacher_object_id for object_id in duplicate_object_ids
    ):
        raise ValueError(
            "TokenTarget.coord_instance_candidates object_instance_id values "
            f"must be unique; duplicate ids: {duplicate_object_ids}"
        )
    if not teacher_bboxes:
        raise ValueError(
            "instance_trie_gaussian teacher object_instance_id is absent from "
            "coord_instance_candidates"
        )
    if len(teacher_bboxes) > 1:
        raise ValueError(
            "instance_trie_gaussian teacher object_instance_id appears more than "
            "once in coord_instance_candidates"
        )

    teacher_bbox = teacher_bboxes[0]
    teacher_coord_value = teacher_token_id - coord_token_start
    expected_coord_value = teacher_bbox[_COORD_SLOT_INDEX[slot_name]]
    if teacher_coord_value != expected_coord_value:
        raise ValueError(
            "instance_trie_gaussian teacher token id does not match teacher "
            f"candidate bbox {slot_name}: token value {teacher_coord_value}, "
            f"bbox value {expected_coord_value}"
        )

    return (
        tuple(candidates),
        slot_name,
        _teacher_prefix_values_for_slot(slot_name, teacher_bbox),
        teacher_coord_value,
    )


def _teacher_prefix_values_for_slot(
    slot_name: str,
    teacher_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    x1, y1, x2, _y2 = teacher_bbox
    if slot_name == "x1":
        return {}
    if slot_name == "y1":
        return {"x1": x1}
    if slot_name == "x2":
        return {"x1": x1, "y1": y1}
    if slot_name == "y2":
        return {"x1": x1, "y1": y1, "x2": x2}
    raise ValueError(f"unsupported coordinate slot {slot_name!r}")


def _target_token_role_value(target: object) -> str | None:
    role = getattr(target, "token_role", None)
    if role is None:
        return None
    return str(getattr(role, "value", role))


def _validate_trie_multi_positive_target(
    target: object,
    *,
    vocab_size: int,
) -> tuple[list[int], list[float]]:
    if not getattr(target, "trie_branch_targets", ()):
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
    return child_token_ids, child_weights


def _validate_token_id(*, token_id: int, vocab_size: int, label: str) -> None:
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} {token_id} is outside vocab size {vocab_size}")


def _normalize_sample_loss(
    *,
    recursive_targets: RecursiveDetectionTargets,
    per_position_losses: Mapping[int, torch.Tensor],
    per_position_main_losses: Mapping[int, torch.Tensor],
    weights: RecursiveDetectionLossWeights,
    device: torch.device,
) -> torch.Tensor:
    def _weighted_position_loss(target: object) -> torch.Tensor:
        total_loss = _loss_float(per_position_losses[target.position])
        main_loss = _loss_float(per_position_main_losses[target.position])
        type_gate_loss = total_loss - main_loss
        return main_loss * _target_loss_weight(target) + type_gate_loss

    if recursive_targets.normalization == "legacy_row_mean_equivalence":
        weighted_losses: list[torch.Tensor] = []
        state_weight_sum = 0.0
        for target in recursive_targets.token_targets:
            state_weight = float(target.state_weight)
            weighted_losses.append(_weighted_position_loss(target) * state_weight)
            state_weight_sum += state_weight
        denominator = torch.tensor(
            max(state_weight_sum, 1e-12),
            device=device,
            dtype=torch.float32,
        )
        return _stack_loss_values(weighted_losses).sum() / denominator

    if recursive_targets.normalization != "semantic_image_bucket_balanced":
        raise ValueError(
            "unsupported recursive detection normalization "
            f"{recursive_targets.normalization!r}"
        )

    target_by_position = {
        target.position: target for target in recursive_targets.token_targets
    }
    atom_losses = {
        atom.atom_id: _stack_loss_values(
            [
                _weighted_position_loss(target_by_position[position])
                for position in atom.token_positions
            ]
        ).mean()
        for atom in recursive_targets.loss_atoms
    }

    loss_terms: list[torch.Tensor] = []
    loss_weights: list[float] = []

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
        loss_terms.append(_stack_loss_values(object_losses).mean())
        loss_weights.append(_IMAGE_MIXTURE_WEIGHTS["objects"])

    ordinary_boundary_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role
        in {
            SemanticRole.SEPARATOR_CONTINUE,
            SemanticRole.TERMINAL_STOP,
            SemanticRole.CHAT_STOP,
        }
    ]
    for boundary_loss in ordinary_boundary_losses:
        loss_terms.append(boundary_loss)
        loss_weights.append(1.0)

    schema_losses = [
        atom_losses[atom.atom_id]
        for atom in recursive_targets.loss_atoms
        if atom.semantic_role is SemanticRole.SCHEMA_CONTROL
    ]
    if schema_losses:
        loss_terms.append(_stack_loss_values(schema_losses).mean())
        loss_weights.append(_IMAGE_MIXTURE_WEIGHTS["schema"])

    if not loss_terms:
        raise ValueError(
            "semantic_image_bucket_balanced requires at least one loss term "
            "(object, separator/stop, or schema)"
        )
    return _weighted_tensor_mean(loss_terms, loss_weights)


def _weighted_tensor_mean(values: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    numerator: torch.Tensor | None = None
    denominator = 0.0
    for value, weight in zip(values, weights, strict=True):
        weighted = _loss_float(value) * float(weight)
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
