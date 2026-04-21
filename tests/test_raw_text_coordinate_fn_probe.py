from __future__ import annotations

from src.analysis.raw_text_coordinate_fn_probe import (
    rank_fn_suppression_candidates,
    summarize_fn_suppression_margin_rows,
)


def test_rank_fn_suppression_candidates_prefers_recovered_supported_cases() -> None:
    rows = [
        {
            "case_id": "weak",
            "recover_fraction_full": 0.25,
            "teacher_forced_support": 0.1,
            "proposal_support": 0.2,
            "competitor_margin": 1.0,
        },
        {
            "case_id": "strong",
            "recover_fraction_full": 1.0,
            "teacher_forced_support": 1.8,
            "proposal_support": 1.4,
            "competitor_margin": -0.5,
        },
        {
            "case_id": "medium",
            "recover_fraction_full": 0.75,
            "teacher_forced_support": 0.9,
            "proposal_support": 0.5,
            "competitor_margin": 0.0,
        },
    ]

    ranked = rank_fn_suppression_candidates(rows=rows, max_cases=2)

    assert [row["case_id"] for row in ranked] == ["strong", "medium"]


def test_summarize_fn_suppression_margin_rows_tracks_stop_pressure_signature() -> None:
    rows = [
        {
            "model_alias": "base_only",
            "continue_minus_eos_sum_logprob": -3.0,
            "continue_minus_eos_mean_logprob": 0.2,
        },
        {
            "model_alias": "base_only",
            "continue_minus_eos_sum_logprob": -1.0,
            "continue_minus_eos_mean_logprob": -0.1,
        },
        {
            "model_alias": "base_plus_adapter",
            "continue_minus_eos_sum_logprob": -0.5,
            "continue_minus_eos_mean_logprob": 0.4,
        },
    ]

    summary = summarize_fn_suppression_margin_rows(rows)

    metrics = {row["model_alias"]: row for row in summary["model_metrics"]}
    base = metrics["base_only"]
    adapter = metrics["base_plus_adapter"]

    assert summary["num_case_model_rows"] == 3
    assert base["stop_pressure_rate"] == 0.5
    assert base["positive_continue_minus_eos_mean_rate"] == 0.5
    assert adapter["stop_pressure_rate"] == 1.0
