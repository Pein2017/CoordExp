"""Exact FP32 repetition-penalty policy transform and replay gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from scripts.research.human13_k_trajectory_contracts import (
    CompleteTrajectoryEvidence,
    PolicyContract,
    ReplayTolerance,
)


class PolicyReplayError(ValueError):
    """Raised when sealed sampling evidence cannot be admitted for replay."""


@dataclass(frozen=True)
class PolicyReplayReceipt:
    admitted: bool
    per_token_absolute_error_nats: tuple[float, ...]
    mean_absolute_error_nats: float
    sampled_trajectory_sha256: str
    replayed_trajectory_sha256: str


def processed_policy_logprobs(
    raw_logits: torch.Tensor,
    history_token_ids: tuple[int, ...],
    contract: PolicyContract,
) -> torch.Tensor:
    """Return full-vocabulary FP32 policy log probabilities for one causal row.

    ``history_token_ids`` is exactly the prompt plus already-generated tokens;
    it excludes the token chosen from this returned distribution.  The literal
    processor order is repetition penalty on raw logits, temperature division,
    then full-vocabulary log-softmax.
    """

    if not isinstance(contract, PolicyContract):
        raise ValueError("contract must be a sealed PolicyContract")
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 1:
        raise ValueError("raw_logits must be a one-dimensional tensor")
    logits = raw_logits.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("raw_logits must be finite")
    history = tuple(history_token_ids)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in history):
        raise ValueError("history_token_ids must contain nonnegative integer token ids")
    vocab_size = int(logits.numel())
    if any(token >= vocab_size for token in history):
        raise ValueError("history token is outside the logits vocabulary")

    # Use unique indexes: repetition processors apply once per token type, not
    # once per occurrence in its causal history.
    if contract.repetition_penalty != 1.0 and history:
        indices = torch.tensor(sorted(set(history)), device=logits.device, dtype=torch.long)
        selected = logits.index_select(0, indices)
        penalized = torch.where(
            selected < 0,
            selected * contract.repetition_penalty,
            selected / contract.repetition_penalty,
        )
        logits = logits.scatter(0, indices, penalized)
    logits = logits / contract.temperature
    logprobs = torch.log_softmax(logits, dim=-1)
    if not bool(torch.isfinite(logprobs).all().item()):
        raise ValueError("processed policy log probabilities must be finite")
    return logprobs


def validate_policy_replay(
    sampled: CompleteTrajectoryEvidence,
    replayed: CompleteTrajectoryEvidence,
    tolerance: ReplayTolerance,
) -> PolicyReplayReceipt:
    """Fail closed unless every sealed token and both numeric gates agree."""

    if not isinstance(sampled, CompleteTrajectoryEvidence) or not isinstance(replayed, CompleteTrajectoryEvidence):
        raise PolicyReplayError("sampled and replayed evidence must be complete trajectories")
    if not isinstance(tolerance, ReplayTolerance):
        raise ValueError("tolerance must be the sealed ReplayTolerance")
    if sampled.policy_contract != replayed.policy_contract:
        raise PolicyReplayError("policy contract differs between sampled and replayed evidence")
    if sampled.identity != replayed.identity or sampled.terminal_kind != replayed.terminal_kind:
        raise PolicyReplayError("trajectory identity or terminal kind differs")
    if len(sampled.generated_tokens) != len(replayed.generated_tokens):
        raise PolicyReplayError("generated token count differs")

    errors: list[float] = []
    for sampled_token, replayed_token in zip(sampled.generated_tokens, replayed.generated_tokens, strict=True):
        if (
            sampled_token.token_index != replayed_token.token_index
            or sampled_token.history_token_ids != replayed_token.history_token_ids
            or sampled_token.chosen_token_id != replayed_token.chosen_token_id
            or sampled_token.policy_contract_sha256 != replayed_token.policy_contract_sha256
        ):
            raise PolicyReplayError("generated token contract or history differs")
        error = abs(sampled_token.processed_logprob - replayed_token.processed_logprob)
        if not math.isfinite(error):
            raise PolicyReplayError("processed log probability error is non-finite")
        if error > tolerance.per_token_nats:
            raise PolicyReplayError("per-token replay error exceeds the sealed tolerance")
        errors.append(error)
    mean_error = sum(errors) / len(errors)
    if mean_error > tolerance.group_mean_nats:
        raise PolicyReplayError("group mean replay error exceeds the sealed tolerance")
    return PolicyReplayReceipt(
        admitted=True,
        per_token_absolute_error_nats=tuple(errors),
        mean_absolute_error_nats=mean_error,
        sampled_trajectory_sha256=sampled.content_sha256,
        replayed_trajectory_sha256=replayed.content_sha256,
    )


__all__ = [
    "PolicyReplayError",
    "PolicyReplayReceipt",
    "processed_policy_logprobs",
    "validate_policy_replay",
]
