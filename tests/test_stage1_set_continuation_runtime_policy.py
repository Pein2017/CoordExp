from __future__ import annotations

import pytest

from src.config.schema import Stage1SetContinuationConfig
from src.trainers.stage1_set_continuation.budget import plan_candidate_execution
from src.trainers.stage1_set_continuation.sampling import Stage1SetContinuationSample


def _sample(candidate_count: int) -> Stage1SetContinuationSample:
    candidates = tuple(range(candidate_count))
    return Stage1SetContinuationSample(
        selected_mode="empty_prefix",
        configured_mixture={"empty_prefix": 1.0},
        resolved_valid_mixture={"empty_prefix": 1.0},
        prefix_indices=(),
        remaining_indices=candidates,
        candidate_indices=candidates,
        candidate_scoring_mode="exact",
        scored_candidate_fraction=1.0,
    )


def _cfg(payload: dict) -> Stage1SetContinuationConfig:
    base = {
        "subset_sampling": {
            "empty_prefix_ratio": 1.0,
            "random_subset_ratio": 0.0,
            "leave_one_out_ratio": 0.0,
            "full_prefix_ratio": 0.0,
            "prefix_order": "dataset",
        },
        "candidates": {"mode": "exact"},
        "structural_close": {
            "close_start_suppression_weight": 0.0,
            "final_schema_close_weight": 0.0,
        },
        "positive_evidence_margin": {"objective": "disabled"},
    }
    base["train_forward"] = payload
    return Stage1SetContinuationConfig.from_mapping(base)


def test_disabled_budget_keeps_exact_candidate_plan() -> None:
    cfg = _cfg({"budget_policy": {"enabled": False}})

    plan = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 0),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )

    assert plan.objective_fidelity == "exact"
    assert plan.authored_candidate_scoring_mode == "exact"
    assert plan.selected_candidate_indices == (0, 1, 2, 3, 4)
    assert plan.fallback_applied is False
    assert plan.logz_estimator == "exact"


def test_candidate_budget_falls_back_to_uniform_subset() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_candidates": 3},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 2,
                    "estimator": "uniform_importance",
                },
            }
        }
    )

    first = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 17),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )
    second = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 17),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )

    assert first.selected_candidate_indices == second.selected_candidate_indices
    assert len(first.selected_candidate_indices) == 2
    assert first.remaining_candidate_count == 5
    assert first.authored_candidate_scoring_mode == "exact"
    assert first.objective_fidelity == "approximate_uniform_subsample"
    assert first.fallback_applied is True
    assert first.fallback_reason == "candidate_budget"
    assert first.logz_estimator == "sampled_raw"


def test_candidate_budget_fallback_keeps_tail_positive_candidate() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_candidates": 3},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 2,
                    "estimator": "uniform_importance",
                },
            }
        }
    )

    plan = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 0),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )

    assert len(plan.selected_candidate_indices) == 2
    assert 4 in plan.selected_candidate_indices
    assert plan.objective_fidelity == "approximate_uniform_subsample"


def test_token_budget_fallback_records_token_reason() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_branch_tokens_per_sample": 110},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 1,
                    "estimator": "uniform_importance",
                },
            }
        }
    )

    plan = plan_candidate_execution(
        sample=_sample(3),
        cfg=cfg,
        sample_seed_parts=("unit", 3),
        prefix_tokens=100,
        candidate_token_lengths=[20, 20, 20],
        memory_free_gib=None,
    )

    assert plan.objective_fidelity == "approximate_uniform_subsample"
    assert plan.fallback_reason == "token_budget"
    assert len(plan.selected_candidate_indices) == 1


def test_budget_exceed_without_fallback_raises_actionable_error() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_candidates": 3},
                "fallback": {"mode": "disabled"},
            }
        }
    )

    with pytest.raises(RuntimeError, match="candidate_budget"):
        plan_candidate_execution(
            sample=_sample(5),
            cfg=cfg,
            sample_seed_parts=("unit", "disabled-fallback"),
            prefix_tokens=100,
            candidate_token_lengths=[10, 10, 10, 10, 10],
            memory_free_gib=None,
        )


def test_authored_uniform_subsample_remains_approximate_without_budget_fallback() -> (
    None
):
    sample = _sample(2)
    sample = Stage1SetContinuationSample(
        selected_mode=sample.selected_mode,
        configured_mixture=sample.configured_mixture,
        resolved_valid_mixture=sample.resolved_valid_mixture,
        prefix_indices=sample.prefix_indices,
        remaining_indices=(0, 1, 2, 3),
        candidate_indices=(1, 3),
        candidate_scoring_mode="uniform_subsample",
        scored_candidate_fraction=0.5,
    )
    cfg = _cfg({"budget_policy": {"enabled": False}})

    plan = plan_candidate_execution(
        sample=sample,
        cfg=cfg,
        sample_seed_parts=("unit", 5),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10],
        memory_free_gib=None,
    )

    assert plan.authored_candidate_scoring_mode == "uniform_subsample"
    assert plan.objective_fidelity == "approximate_uniform_subsample"
    assert plan.fallback_applied is False
    assert plan.logz_estimator == "sampled_raw"
