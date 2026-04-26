from __future__ import annotations

import hashlib
import random
from typing import Sequence, cast

from src.config.schema import Stage1SetContinuationConfig

from .runtime import (
    CandidateExecutionPlan,
    FallbackReason,
    LogZEstimator,
    ObjectiveFidelity,
)
from .sampling import Stage1SetContinuationSample


def _stable_rng(parts: Sequence[object]) -> random.Random:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return random.Random(seed)


def _planned_branch_tokens(
    *,
    prefix_tokens: int,
    candidate_count: int,
    candidate_token_lengths: Sequence[int],
) -> int:
    return int(prefix_tokens) * max(1, int(candidate_count)) + sum(
        int(length) for length in candidate_token_lengths
    )


def _budget_reason(
    *,
    exact_candidate_count: int,
    prefix_tokens: int,
    candidate_token_lengths: Sequence[int],
    max_candidates: int | None,
    max_branch_tokens_per_sample: int | None,
    min_free_memory_gib: float | None,
    memory_free_gib: float | None,
    enabled_budget_kinds: Sequence[str],
) -> FallbackReason:
    enabled = set(enabled_budget_kinds)
    if (
        "candidate" in enabled
        and max_candidates is not None
        and exact_candidate_count > max_candidates
    ):
        return "candidate_budget"
    planned_tokens = _planned_branch_tokens(
        prefix_tokens=prefix_tokens,
        candidate_count=exact_candidate_count,
        candidate_token_lengths=candidate_token_lengths,
    )
    if (
        "token" in enabled
        and max_branch_tokens_per_sample is not None
        and planned_tokens > max_branch_tokens_per_sample
    ):
        return "token_budget"
    if (
        "memory" in enabled
        and min_free_memory_gib is not None
        and memory_free_gib is not None
        and memory_free_gib < min_free_memory_gib
    ):
        return "memory_budget"
    return "none"


def _no_fallback_fidelity(
    *,
    authored_mode: str,
    exact_indices: tuple[int, ...],
    remaining_indices: tuple[int, ...],
    pem_enabled: bool,
) -> tuple[ObjectiveFidelity, LogZEstimator]:
    authored_is_exact = authored_mode == "exact" and len(exact_indices) == len(
        remaining_indices
    )
    if authored_is_exact:
        return "exact", "exact"
    return (
        "approximate_uniform_subsample",
        "uniform_importance" if pem_enabled else "sampled_raw",
    )


def plan_candidate_execution(
    *,
    sample: Stage1SetContinuationSample,
    cfg: Stage1SetContinuationConfig,
    sample_seed_parts: Sequence[object],
    prefix_tokens: int,
    candidate_token_lengths: Sequence[int],
    memory_free_gib: float | None,
    enabled_budget_kinds: Sequence[str] = ("candidate", "token", "memory"),
) -> CandidateExecutionPlan:
    train_forward = cfg.train_forward
    exact_indices = tuple(int(idx) for idx in sample.candidate_indices)
    authored_mode = str(sample.candidate_scoring_mode)
    remaining_indices = tuple(int(idx) for idx in sample.remaining_indices)
    pem_enabled = cfg.positive_evidence_margin.objective == "threshold_loss"
    objective_fidelity, logz_estimator = _no_fallback_fidelity(
        authored_mode=authored_mode,
        exact_indices=exact_indices,
        remaining_indices=remaining_indices,
        pem_enabled=pem_enabled,
    )
    planned_tokens = _planned_branch_tokens(
        prefix_tokens=prefix_tokens,
        candidate_count=len(exact_indices),
        candidate_token_lengths=candidate_token_lengths,
    )

    if not train_forward.budget_policy.enabled or not exact_indices:
        return CandidateExecutionPlan(
            selected_candidate_indices=exact_indices,
            remaining_candidate_count=len(remaining_indices),
            authored_candidate_scoring_mode=authored_mode,
            objective_fidelity=objective_fidelity,
            fallback_applied=False,
            fallback_reason="none",
            logz_estimator=logz_estimator,
            prefix_tokens=int(prefix_tokens),
            planned_candidate_tokens=planned_tokens,
            exact_candidate_count=len(exact_indices),
        )

    exact_until = train_forward.budget_policy.exact_until
    fallback = train_forward.budget_policy.fallback
    fallback_reason = _budget_reason(
        exact_candidate_count=len(exact_indices),
        prefix_tokens=int(prefix_tokens),
        candidate_token_lengths=candidate_token_lengths,
        max_candidates=exact_until.max_candidates,
        max_branch_tokens_per_sample=exact_until.max_branch_tokens_per_sample,
        min_free_memory_gib=exact_until.min_free_memory_gib,
        memory_free_gib=memory_free_gib,
        enabled_budget_kinds=enabled_budget_kinds,
    )
    if fallback_reason == "none":
        return CandidateExecutionPlan(
            selected_candidate_indices=exact_indices,
            remaining_candidate_count=len(remaining_indices),
            authored_candidate_scoring_mode=authored_mode,
            objective_fidelity=objective_fidelity,
            fallback_applied=False,
            fallback_reason="none",
            logz_estimator=logz_estimator,
            prefix_tokens=int(prefix_tokens),
            planned_candidate_tokens=planned_tokens,
            exact_candidate_count=len(exact_indices),
        )

    if fallback.mode != "approximate_uniform_subsample":
        raise RuntimeError(
            "Stage-1 set-continuation budget exceeded with fallback disabled: "
            f"reason={fallback_reason}, candidates={len(exact_indices)}, "
            f"planned_candidate_tokens={planned_tokens}. "
            "Enable train_forward.budget_policy.fallback.mode="
            "approximate_uniform_subsample or relax the exact_until budget."
        )

    limit = int(cast(int, fallback.max_candidates))
    if len(exact_indices) <= limit:
        selected = exact_indices
    else:
        shuffled = list(exact_indices)
        _stable_rng(tuple(sample_seed_parts) + ("fallback", fallback_reason)).shuffle(
            shuffled
        )
        selected = tuple(shuffled[:limit])

    selected_all = set(selected) == set(remaining_indices) and len(selected) == len(
        remaining_indices
    )
    if selected_all and authored_mode == "exact":
        fallback_fidelity: ObjectiveFidelity = "exact"
        fallback_estimator: LogZEstimator = "exact"
    else:
        fallback_fidelity = "approximate_uniform_subsample"
        fallback_estimator = "uniform_importance" if pem_enabled else "sampled_raw"

    return CandidateExecutionPlan(
        selected_candidate_indices=tuple(int(idx) for idx in selected),
        remaining_candidate_count=len(remaining_indices),
        authored_candidate_scoring_mode=authored_mode,
        objective_fidelity=fallback_fidelity,
        fallback_applied=True,
        fallback_reason=fallback_reason,
        logz_estimator=fallback_estimator,
        prefix_tokens=int(prefix_tokens),
        planned_candidate_tokens=planned_tokens,
        exact_candidate_count=len(exact_indices),
    )


__all__ = ["plan_candidate_execution"]
