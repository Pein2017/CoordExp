"""Exact FP32 repetition-penalty policy transform and replay gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from scripts.research.human13_k_trajectory_contracts import (
    AcquisitionGroup,
    CompleteTrajectoryEvidence,
    GeneratedTokenEvidence,
    PolicyContract,
    ReplayTolerance,
)


class PolicyReplayError(ValueError):
    """Raised when sealed sampling evidence cannot be admitted for replay."""


@dataclass(frozen=True)
class PolicyReplayReceipt:
    admitted: bool
    per_token_absolute_error_nats: tuple[float, ...]
    trajectory_mean_absolute_error_nats: float
    sampled_trajectory_sha256: str
    replayed_trajectory_sha256: str


@dataclass(frozen=True)
class AcquisitionGroupReplayReceipt:
    """Content-addressed parity receipt across the complete K-trajectory group."""

    admitted: bool
    tolerance_sha256: str
    sampled_group_sha256: str
    replayed_group_sha256: str
    request_ids: tuple[str, ...]
    per_token_absolute_error_nats: tuple[float, ...]
    group_mean_absolute_error_nats: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "human13_acquisition_group_replay_receipt.v1",
            "admitted": self.admitted,
            "tolerance_sha256": self.tolerance_sha256,
            "sampled_group_sha256": self.sampled_group_sha256,
            "replayed_group_sha256": self.replayed_group_sha256,
            "request_ids": list(self.request_ids),
            "per_token_absolute_error_nats": list(self.per_token_absolute_error_nats),
            "group_mean_absolute_error_nats": self.group_mean_absolute_error_nats,
        }

    @property
    def content_sha256(self) -> str:
        import hashlib
        import json

        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def processed_policy_logprobs(
    raw_logits: torch.Tensor,
    token_evidence: GeneratedTokenEvidence,
    contract: PolicyContract,
) -> torch.Tensor:
    """Return full-vocabulary FP32 policy log probabilities for one causal row.

    The history is read from the sealed token evidence, not caller input.  It
    must equal the contract prompt plus the evidence identity's generated
    prefix, and excludes the token chosen from this returned distribution.
    """

    if not isinstance(contract, PolicyContract):
        raise ValueError("contract must be a sealed PolicyContract")
    if not isinstance(token_evidence, GeneratedTokenEvidence):
        raise ValueError("token_evidence must be sealed GeneratedTokenEvidence")
    if token_evidence.policy_contract_sha256 != contract.content_sha256:
        raise ValueError("token evidence policy contract differs")
    if token_evidence.identity.prompt_token_ids != contract.identity.prompt_token_ids:
        raise ValueError("token evidence prompt differs from the sealed contract prompt")
    if token_evidence.token_index >= len(token_evidence.identity.generated_token_ids):
        raise ValueError("token evidence index is outside its generated history")
    if token_evidence.chosen_token_id != token_evidence.identity.generated_token_ids[token_evidence.token_index]:
        raise ValueError("token evidence chosen token differs from generated history")
    expected_history = (
        *contract.identity.prompt_token_ids,
        *token_evidence.identity.generated_token_ids[: token_evidence.token_index],
    )
    if token_evidence.history_token_ids != expected_history:
        if token_evidence.history_token_ids[: len(contract.identity.prompt_token_ids)] != contract.identity.prompt_token_ids:
            raise ValueError("token evidence prompt history differs from the sealed prompt")
        raise ValueError("token evidence generated prefix differs from the sealed history")
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 1:
        raise ValueError("raw_logits must be a one-dimensional tensor")
    logits = raw_logits.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("raw_logits must be finite")
    history = token_evidence.history_token_ids
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


def _trajectory_replay_errors(
    sampled: CompleteTrajectoryEvidence,
    replayed: CompleteTrajectoryEvidence,
    tolerance: ReplayTolerance,
) -> tuple[float, ...]:

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
    return tuple(errors)


def validate_policy_replay(
    sampled: CompleteTrajectoryEvidence,
    replayed: CompleteTrajectoryEvidence,
    tolerance: ReplayTolerance,
) -> PolicyReplayReceipt:
    """Validate one trajectory's identity and token gates, without a group claim."""

    errors = _trajectory_replay_errors(sampled, replayed, tolerance)
    mean_error = sum(errors) / len(errors)
    return PolicyReplayReceipt(
        admitted=True,
        per_token_absolute_error_nats=errors,
        trajectory_mean_absolute_error_nats=mean_error,
        sampled_trajectory_sha256=sampled.content_sha256,
        replayed_trajectory_sha256=replayed.content_sha256,
    )


def validate_acquisition_group_replay(
    sampled: AcquisitionGroup,
    replayed: AcquisitionGroup,
    tolerance: ReplayTolerance,
) -> AcquisitionGroupReplayReceipt:
    """Fail closed on any K-member mismatch or one sealed group-wide mean breach."""

    if not isinstance(sampled, AcquisitionGroup) or not isinstance(replayed, AcquisitionGroup):
        raise PolicyReplayError("sampled and replayed evidence must be acquisition groups")
    if not isinstance(tolerance, ReplayTolerance):
        raise ValueError("tolerance must be the sealed ReplayTolerance")
    if (
        sampled.identity != replayed.identity
        or sampled.policy_contract != replayed.policy_contract
        or sampled.seed_group_id != replayed.seed_group_id
    ):
        raise PolicyReplayError("acquisition group lineage differs")
    sampled_by_request = {item.identity.request_id: item for item in sampled.trajectories}
    replayed_by_request = {item.identity.request_id: item for item in replayed.trajectories}
    if set(sampled_by_request) != set(replayed_by_request):
        raise PolicyReplayError("acquisition group request identities differ")
    errors: list[float] = []
    request_ids = tuple(sorted(sampled_by_request))
    for request_id in request_ids:
        errors.extend(
            _trajectory_replay_errors(
                sampled_by_request[request_id], replayed_by_request[request_id], tolerance
            )
        )
    mean_error = sum(errors) / len(errors)
    if mean_error > tolerance.group_mean_nats:
        raise PolicyReplayError("group mean replay error exceeds the sealed tolerance")
    return AcquisitionGroupReplayReceipt(
        admitted=True,
        tolerance_sha256=tolerance.content_sha256,
        sampled_group_sha256=sampled.content_sha256,
        replayed_group_sha256=replayed.content_sha256,
        request_ids=request_ids,
        per_token_absolute_error_nats=tuple(errors),
        group_mean_absolute_error_nats=mean_error,
    )


__all__ = [
    "PolicyReplayError",
    "PolicyReplayReceipt",
    "AcquisitionGroupReplayReceipt",
    "processed_policy_logprobs",
    "validate_acquisition_group_replay",
    "validate_policy_replay",
]
