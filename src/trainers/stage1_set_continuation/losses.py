"""Pure loss helpers for Stage-1 set-continuation training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CandidateLogProbResult:
    score: torch.Tensor
    coord_score: torch.Tensor
    non_coord_score: torch.Tensor
    schema_score: torch.Tensor
    json_structural_score: torch.Tensor
    coord_gate_loss: torch.Tensor
    text_gate_loss: torch.Tensor
    coord_gate_nll_sum: torch.Tensor
    text_gate_nll_sum: torch.Tensor
    coord_gate_coord_mass_sum: torch.Tensor
    text_gate_coord_mass_sum: torch.Tensor
    tokens: int
    coord_tokens: int
    non_coord_tokens: int
    schema_tokens: int
    json_structural_tokens: int
    coord_gate_tokens: int
    text_gate_tokens: int


@dataclass(frozen=True)
class BidirectionalTokenGateResult:
    coord_loss: torch.Tensor
    text_loss: torch.Tensor
    coord_nll_sum: torch.Tensor
    text_nll_sum: torch.Tensor
    coord_mass_sum: torch.Tensor
    text_mass_sum: torch.Tensor
    coord_tokens: int
    text_tokens: int


@dataclass(frozen=True)
class MultiPositiveLossResult:
    total_objective: torch.Tensor
    loss_mp: torch.Tensor
    loss_candidate_balanced: torch.Tensor
    loss_pem: torch.Tensor
    log_z_remaining: torch.Tensor
    responsibilities: torch.Tensor
    denominator: int
    log_z_estimator: str
    metrics: dict[str, Any]


def _zero_like_scores(scores: torch.Tensor) -> torch.Tensor:
    return (
        scores.new_zeros(()) if isinstance(scores, torch.Tensor) else torch.tensor(0.0)
    )


def _shift_for_next_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError("logits must be rank-3 [batch, seq, vocab]")
    if labels.ndim != 2 or mask.ndim != 2:
        raise ValueError("labels and masks must be rank-2 [batch, seq]")
    if labels.shape != mask.shape or logits.shape[:2] != labels.shape:
        raise ValueError("logits, labels, and masks must share batch/sequence shape")
    if labels.shape[1] < 2:
        empty_logits = logits[:, :0, :]
        empty_labels = labels[:, :0]
        empty_mask = mask[:, :0].bool()
        return empty_logits, empty_labels, empty_mask
    return logits[:, :-1, :], labels[:, 1:], mask[:, 1:].bool()


def _gather_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    safe_labels = labels.clamp_min(0)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)


def _gather_coord_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_token_ids: torch.Tensor,
) -> torch.Tensor:
    if coord_token_ids.ndim != 1 or int(coord_token_ids.numel()) <= 0:
        raise ValueError("coord_token_ids must be a non-empty rank-1 tensor")
    coord_ids = coord_token_ids.to(device=logits.device, dtype=torch.long)
    coord_logits = logits.index_select(dim=-1, index=coord_ids)
    matches = labels.unsqueeze(-1).eq(coord_ids.view(1, 1, -1))
    if not bool(matches.any().item()):
        raise ValueError("coord-labeled positions must use ids from coord_token_ids")
    local_labels = matches.float().argmax(dim=-1).long()
    return (
        F.log_softmax(coord_logits, dim=-1)
        .gather(
            dim=-1,
            index=local_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def _validate_coord_token_ids(
    coord_token_ids: torch.Tensor,
    *,
    vocab_size: int,
    require_1000: bool = False,
) -> torch.Tensor:
    if coord_token_ids.ndim != 1 or int(coord_token_ids.numel()) <= 0:
        raise ValueError("coord_token_ids must be a non-empty rank-1 tensor")
    coord_ids = coord_token_ids.to(dtype=torch.long)
    if bool((coord_ids < 0).any().item()):
        raise ValueError("coord_token_ids must be non-negative")
    if bool((coord_ids >= int(vocab_size)).any().item()):
        raise ValueError("coord_token_ids must be within logits vocabulary")
    coord_id_values = [int(value) for value in coord_ids.detach().cpu().tolist()]
    if len(set(coord_id_values)) != len(coord_id_values):
        raise ValueError("coord_token_ids must be unique")
    if require_1000 and int(coord_ids.numel()) != 1000:
        raise ValueError("coord_token_ids must contain exactly 1000 ids")
    return coord_ids


def _log1mexp(log_x: torch.Tensor) -> torch.Tensor:
    """Stable log(1 - exp(log_x)) for log_x <= 0."""

    log_x = log_x.clamp(max=-1e-7)
    threshold = math.log(0.5)
    return torch.where(
        log_x < threshold,
        torch.log1p(-torch.exp(log_x)),
        torch.log(-torch.expm1(log_x)),
    )


def compute_bidirectional_token_gate_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    special_token_ids: torch.Tensor | None = None,
    temperature: float = 1.0,
    require_1000_coord_ids: bool = False,
) -> BidirectionalTokenGateResult:
    if float(temperature) <= 0.0:
        raise ValueError("bidirectional token gate temperature must be > 0")
    shift_logits, shift_labels, objective_mask = _shift_for_next_token(
        logits, labels, objective_label_mask
    )
    _, _, coord_mask = _shift_for_next_token(logits, labels, coord_label_mask)
    zero = logits.new_zeros(())
    coord_ids = _validate_coord_token_ids(
        coord_token_ids.to(device=logits.device),
        vocab_size=int(logits.shape[-1]),
        require_1000=bool(require_1000_coord_ids),
    ).to(device=logits.device)
    valid_mask = objective_mask & shift_labels.ne(-100)
    if special_token_ids is not None and int(special_token_ids.numel()) > 0:
        special_ids = special_token_ids.to(device=shift_labels.device, dtype=torch.long)
        non_coord_special_ids = special_ids[
            ~special_ids.unsqueeze(-1).eq(coord_ids.view(1, -1)).any(dim=-1)
        ]
        if int(non_coord_special_ids.numel()) > 0:
            special_match = shift_labels.unsqueeze(-1).eq(
                non_coord_special_ids.view(1, 1, -1)
            )
            valid_mask = valid_mask & ~special_match.any(dim=-1)
    coord_gate_mask = valid_mask & coord_mask
    text_gate_mask = valid_mask & ~coord_mask
    if coord_gate_mask.any():
        coord_targets = shift_labels[coord_gate_mask]
        if not coord_targets.unsqueeze(-1).eq(coord_ids.view(1, -1)).any(dim=-1).all():
            raise ValueError("coord-gate labels must be contained in coord_token_ids")
    if not bool((coord_gate_mask | text_gate_mask).any().item()):
        return BidirectionalTokenGateResult(
            coord_loss=zero,
            text_loss=zero,
            coord_nll_sum=zero,
            text_nll_sum=zero,
            coord_mass_sum=zero,
            text_mass_sum=zero,
            coord_tokens=0,
            text_tokens=0,
        )

    full = torch.nan_to_num(
        shift_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-1e4, max=1e4) / float(temperature)
    coord_logits = full.index_select(dim=-1, index=coord_ids)
    log_p_coord = torch.logsumexp(coord_logits, dim=-1) - torch.logsumexp(full, dim=-1)
    log_p_coord = log_p_coord.clamp(max=0.0)
    p_coord = torch.exp(log_p_coord.clamp(min=-50.0, max=0.0))
    coord_nll = -log_p_coord
    text_nll = -_log1mexp(log_p_coord)
    coord_tokens = int(coord_gate_mask.sum().detach().item())
    text_tokens = int(text_gate_mask.sum().detach().item())
    coord_nll_sum = coord_nll[coord_gate_mask].sum() if coord_tokens else zero
    text_nll_sum = text_nll[text_gate_mask].sum() if text_tokens else zero
    coord_mass_sum = (
        p_coord[coord_gate_mask].sum().to(dtype=zero.dtype) if coord_tokens else zero
    )
    text_mass_sum = (
        p_coord[text_gate_mask].sum().to(dtype=zero.dtype) if text_tokens else zero
    )
    return BidirectionalTokenGateResult(
        coord_loss=torch.nan_to_num(
            coord_nll_sum / max(coord_tokens, 1), nan=0.0, posinf=1e4
        ),
        text_loss=torch.nan_to_num(
            text_nll_sum / max(text_tokens, 1), nan=0.0, posinf=1e4
        ),
        coord_nll_sum=torch.nan_to_num(coord_nll_sum, nan=0.0, posinf=1e4),
        text_nll_sum=torch.nan_to_num(text_nll_sum, nan=0.0, posinf=1e4),
        coord_mass_sum=torch.nan_to_num(coord_mass_sum, nan=0.0, posinf=1.0),
        text_mass_sum=torch.nan_to_num(text_mass_sum, nan=0.0, posinf=1.0),
        coord_tokens=coord_tokens,
        text_tokens=text_tokens,
    )


def compute_candidate_full_entry_logprob(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    objective_label_mask: torch.Tensor | None = None,
    schema_open_label_mask: torch.Tensor | None = None,
    json_structural_label_mask: torch.Tensor | None = None,
    bidirectional_gate_enabled: bool = False,
    bidirectional_gate_temperature: float = 1.0,
    special_token_ids: torch.Tensor | None = None,
) -> CandidateLogProbResult:
    """Score one full serialized candidate entry.

    Text/structure slots use full-vocabulary probabilities. Coord slots use a
    coord-vocabulary normalization so non-coordinate tokens do not compete with
    coordinate bins.
    """

    shift_logits, shift_labels, candidate_mask = _shift_for_next_token(
        logits,
        labels,
        candidate_entry_label_mask,
    )
    _, _, coord_mask = _shift_for_next_token(logits, labels, coord_label_mask)
    if schema_open_label_mask is None:
        schema_mask = torch.zeros_like(candidate_mask)
    else:
        _, _, schema_mask = _shift_for_next_token(
            logits, labels, schema_open_label_mask
        )
    if json_structural_label_mask is None:
        json_structural_mask = torch.zeros_like(candidate_mask)
    else:
        _, _, json_structural_mask = _shift_for_next_token(
            logits, labels, json_structural_label_mask
        )
    valid_mask = candidate_mask & shift_labels.ne(-100)
    coord_mask = coord_mask & valid_mask
    schema_mask = schema_mask & valid_mask
    json_structural_mask = json_structural_mask & valid_mask
    non_coord_mask = valid_mask & ~coord_mask

    zero = logits.new_zeros(())
    full_logprob = _gather_logprob(shift_logits, shift_labels)
    non_coord_score = (
        full_logprob[non_coord_mask].sum() if non_coord_mask.any() else zero
    )

    if coord_mask.any():
        coord_logprob = _gather_coord_logprob(
            shift_logits,
            shift_labels,
            coord_token_ids,
        )
        coord_score = coord_logprob[coord_mask].sum()
    else:
        coord_score = zero

    score = coord_score + non_coord_score
    schema_score = full_logprob[schema_mask].sum() if schema_mask.any() else zero
    json_structural_score = (
        full_logprob[json_structural_mask].sum() if json_structural_mask.any() else zero
    )
    if bidirectional_gate_enabled:
        gate = compute_bidirectional_token_gate_loss(
            logits=logits,
            labels=labels,
            objective_label_mask=(
                candidate_entry_label_mask
                if objective_label_mask is None
                else objective_label_mask
            ),
            coord_label_mask=coord_label_mask,
            coord_token_ids=coord_token_ids,
            special_token_ids=special_token_ids,
            temperature=float(bidirectional_gate_temperature),
        )
    else:
        gate = BidirectionalTokenGateResult(
            coord_loss=zero,
            text_loss=zero,
            coord_nll_sum=zero,
            text_nll_sum=zero,
            coord_mass_sum=zero,
            text_mass_sum=zero,
            coord_tokens=0,
            text_tokens=0,
        )
    return CandidateLogProbResult(
        score=score,
        coord_score=coord_score,
        non_coord_score=non_coord_score,
        schema_score=schema_score,
        json_structural_score=json_structural_score,
        coord_gate_loss=gate.coord_loss,
        text_gate_loss=gate.text_loss,
        coord_gate_nll_sum=gate.coord_nll_sum,
        text_gate_nll_sum=gate.text_nll_sum,
        coord_gate_coord_mass_sum=gate.coord_mass_sum,
        text_gate_coord_mass_sum=gate.text_mass_sum,
        tokens=int(valid_mask.sum().item()),
        coord_tokens=int(coord_mask.sum().item()),
        non_coord_tokens=int(non_coord_mask.sum().item()),
        schema_tokens=int(schema_mask.sum().item()),
        json_structural_tokens=int(json_structural_mask.sum().item()),
        coord_gate_tokens=int(gate.coord_tokens),
        text_gate_tokens=int(gate.text_tokens),
    )


def _resolve_log_rho(
    *,
    rho: float | None,
    log_rho: torch.Tensor | float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (rho is None) == (log_rho is None):
        raise ValueError("PEM threshold_loss requires exactly one of rho/log_rho")
    if rho is not None:
        if not (0.0 < float(rho) <= 1.0):
            raise ValueError("rho must satisfy 0 < rho <= 1")
        return torch.tensor(math.log(float(rho)), device=device, dtype=dtype)
    if isinstance(log_rho, torch.Tensor):
        return log_rho.to(device=device, dtype=dtype)
    return torch.tensor(float(log_rho), device=device, dtype=dtype)


def _normalize_pem_objective(pem_mode: str) -> str:
    if pem_mode == "replace_mp":
        return "threshold_loss"
    return pem_mode


def _estimate_log_z(
    *,
    scores: torch.Tensor,
    estimator: str,
    remaining_count: int,
    scored_count: int,
) -> torch.Tensor:
    log_z = torch.logsumexp(scores, dim=0)
    if estimator == "exact" or estimator == "sampled_raw":
        return log_z
    if estimator == "uniform_importance":
        if remaining_count <= 0 or scored_count <= 0:
            raise ValueError("uniform_importance logZ requires positive counts")
        return log_z + scores.new_tensor(
            math.log(float(remaining_count) / scored_count)
        )
    raise ValueError(
        "logZ estimator must be one of {'exact', 'sampled_raw', 'uniform_importance'}"
    )


def summarize_candidate_scores(
    *,
    scores: torch.Tensor,
    candidate_lengths: torch.Tensor | None = None,
) -> dict[str, Any]:
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    if int(scores.numel()) == 0:
        return {
            "mp/responsibility_entropy": 0.0,
            "mp/effective_candidate_count": 0.0,
            "mp/effective_candidate_fraction": 0.0,
            "mp/max_responsibility": 0.0,
            "mp/min_responsibility": 0.0,
            "mp/candidate_score_mean": 0.0,
            "mp/candidate_score_std": 0.0,
            "mp/responsibility_length_corr_valid": 0,
        }
    responsibilities = torch.softmax(scores, dim=0)
    entropy = -(responsibilities * torch.log(responsibilities.clamp_min(1e-30))).sum()
    std = (
        scores.std(unbiased=False) if int(scores.numel()) > 1 else scores.new_zeros(())
    )
    out: dict[str, Any] = {
        "mp/responsibility_entropy": float(entropy.detach().item()),
        "mp/effective_candidate_count": float(torch.exp(entropy).detach().item()),
        "mp/effective_candidate_fraction": float(
            (torch.exp(entropy) / float(scores.numel())).detach().item()
        ),
        "mp/max_responsibility": float(responsibilities.max().detach().item()),
        "mp/min_responsibility": float(responsibilities.min().detach().item()),
        "mp/candidate_score_mean": float(scores.mean().detach().item()),
        "mp/candidate_score_std": float(std.detach().item()),
        "mp/responsibility_length_corr_valid": 0,
    }
    if candidate_lengths is not None and int(scores.numel()) >= 2:
        lengths = candidate_lengths.to(device=scores.device, dtype=scores.dtype)
        if (
            lengths.numel() == scores.numel()
            and float(lengths.std(unbiased=False)) > 0.0
        ):
            resp_centered = responsibilities - responsibilities.mean()
            len_centered = lengths - lengths.mean()
            denom = resp_centered.norm() * len_centered.norm()
            if float(denom.detach().item()) > 0.0:
                corr = (resp_centered * len_centered).sum() / denom
                out["mp/responsibility_length_corr"] = float(corr.detach().item())
                out["mp/responsibility_length_corr_valid"] = 1
    return out


def _candidate_balanced_loss(
    *,
    scores: torch.Tensor,
    candidate_lengths: torch.Tensor | None,
) -> torch.Tensor:
    if candidate_lengths is None:
        return -scores.mean()
    lengths = candidate_lengths.to(device=scores.device, dtype=scores.dtype).reshape(-1)
    if int(lengths.numel()) != int(scores.numel()):
        raise ValueError("candidate_lengths must have one entry per score")
    token_normalized_scores = scores / lengths.clamp_min(1.0)
    return -token_normalized_scores.mean()


def compute_mp_pem_losses(
    *,
    scores: torch.Tensor,
    pem_mode: str,
    rho: float | None = None,
    log_rho: torch.Tensor | float | None = None,
    estimator: str = "exact",
    remaining_count: int | None = None,
    scored_count: int | None = None,
    candidate_lengths: torch.Tensor | None = None,
) -> MultiPositiveLossResult:
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    pem_mode = _normalize_pem_objective(str(pem_mode))
    if pem_mode not in {"disabled", "threshold_loss"}:
        raise ValueError("pem_mode must be one of {'disabled', 'threshold_loss'}")
    if int(scores.numel()) == 0:
        zero = _zero_like_scores(scores)
        return MultiPositiveLossResult(
            total_objective=zero,
            loss_mp=zero,
            loss_candidate_balanced=zero,
            loss_pem=zero,
            log_z_remaining=zero,
            responsibilities=scores.new_empty((0,)),
            denominator=0,
            log_z_estimator=str(estimator),
            metrics={
                **summarize_candidate_scores(
                    scores=scores, candidate_lengths=candidate_lengths
                ),
                "loss/candidate_balanced": 0.0,
                "mp/logZ_remaining": 0.0,
                "mp/logZ_estimator": str(estimator),
            },
        )

    resolved_remaining_count = int(
        remaining_count if remaining_count is not None else scores.numel()
    )
    resolved_scored_count = int(
        scored_count if scored_count is not None else scores.numel()
    )
    log_z = _estimate_log_z(
        scores=scores,
        estimator=str(estimator),
        remaining_count=resolved_remaining_count,
        scored_count=resolved_scored_count,
    )
    loss_mp = -log_z
    loss_candidate_balanced = _candidate_balanced_loss(
        scores=scores,
        candidate_lengths=candidate_lengths,
    )
    if pem_mode == "threshold_loss":
        threshold = _resolve_log_rho(
            rho=rho,
            log_rho=log_rho,
            device=scores.device,
            dtype=scores.dtype,
        )
        loss_pem = torch.clamp(threshold - log_z, min=0.0)
        total = loss_pem
    else:
        loss_pem = scores.new_zeros(())
        total = loss_candidate_balanced

    metrics = summarize_candidate_scores(
        scores=scores,
        candidate_lengths=candidate_lengths,
    )
    metrics.update(
        {
            "loss/candidate_balanced": float(loss_candidate_balanced.detach().item()),
            "mp/logZ_remaining": float(log_z.detach().item()),
            "mp/logZ_estimator": str(estimator),
        }
    )
    return MultiPositiveLossResult(
        total_objective=total,
        loss_mp=loss_mp,
        loss_candidate_balanced=loss_candidate_balanced,
        loss_pem=loss_pem,
        log_z_remaining=log_z,
        responsibilities=torch.softmax(scores, dim=0),
        denominator=resolved_scored_count,
        log_z_estimator=str(estimator),
        metrics=metrics,
    )


def _masked_shifted_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits, shift_labels, shift_mask = _shift_for_next_token(logits, labels, mask)
    valid = shift_mask & shift_labels.ne(-100)
    if not valid.any():
        return logits.new_zeros(())
    token_logprob = _gather_logprob(shift_logits, shift_labels)
    return -token_logprob[valid].sum()


def compute_close_start_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    structural_close_start_mask: torch.Tensor,
) -> torch.Tensor:
    return _masked_shifted_nll(
        logits=logits,
        labels=labels,
        mask=structural_close_start_mask,
    )


def compute_close_sequence_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    structural_close_sequence_mask: torch.Tensor,
) -> torch.Tensor:
    return _masked_shifted_nll(
        logits=logits,
        labels=labels,
        mask=structural_close_sequence_mask,
    )


__all__ = [
    "BidirectionalTokenGateResult",
    "CandidateLogProbResult",
    "MultiPositiveLossResult",
    "compute_bidirectional_token_gate_loss",
    "compute_candidate_full_entry_logprob",
    "compute_close_sequence_nll",
    "compute_close_start_nll",
    "compute_mp_pem_losses",
    "summarize_candidate_scores",
]
