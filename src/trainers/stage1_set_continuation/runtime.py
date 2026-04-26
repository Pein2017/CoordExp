from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ObjectiveFidelity = Literal["exact", "approximate_uniform_subsample"]
FallbackReason = Literal["none", "candidate_budget", "token_budget", "memory_budget"]
LogZEstimator = Literal["exact", "sampled_raw", "uniform_importance"]


@dataclass(frozen=True)
class CandidateExecutionPlan:
    selected_candidate_indices: tuple[int, ...]
    remaining_candidate_count: int
    authored_candidate_scoring_mode: str
    objective_fidelity: ObjectiveFidelity
    fallback_applied: bool
    fallback_reason: FallbackReason
    logz_estimator: LogZEstimator
    prefix_tokens: int
    planned_candidate_tokens: int
    exact_candidate_count: int


__all__ = [
    "CandidateExecutionPlan",
    "FallbackReason",
    "LogZEstimator",
    "ObjectiveFidelity",
]
