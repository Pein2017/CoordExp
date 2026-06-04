from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.residual_row_scoring import (
    score_residual_rows,
)


def test_length_normalized_mean_selects_best_residual_not_summed_logprob() -> None:
    rows = [
        {
            "residual_row_id": "short_teacher",
            "is_teacher_next": True,
            "desc_logprobs": [-0.15],
            "x1_logprob": -0.15,
            "bbox_tail_logprobs": [-0.15],
            "eos_logprob": -0.70,
        },
        {
            "residual_row_id": "long_model_pref",
            "is_teacher_next": False,
            "desc_logprobs": [-0.10, -0.10],
            "x1_logprob": -0.10,
            "bbox_tail_logprobs": [-0.10, -0.10, -0.10],
            "eos_logprob": -0.70,
        },
    ]

    scored_rows, summary = score_residual_rows(rows)

    by_id = {row["residual_row_id"]: row for row in scored_rows}
    assert by_id["long_model_pref"]["logp_row_sum"] < by_id["short_teacher"]["logp_row_sum"]
    assert by_id["long_model_pref"]["logp_row_mean"] > by_id["short_teacher"]["logp_row_mean"]
    assert summary["best_residual_row_id"] == "long_model_pref"
    assert summary["teacher_next_row_id"] == "short_teacher"
    assert summary["preference_violation"] is True
    assert summary["margin_best_residual_vs_teacher_next"] > 0


def test_eos_margin_uses_boundary_logprob_from_same_prompt() -> None:
    rows = [
        {
            "residual_row_id": "candidate",
            "is_teacher_next": True,
            "desc_logprobs": [-0.50],
            "x1_logprob": -0.50,
            "bbox_tail_logprobs": [-0.50, -0.50, -0.50],
            "eos_logprob": -0.10,
        }
    ]

    scored_rows, summary = score_residual_rows(rows)

    assert scored_rows[0]["logp_eos_at_boundary"] == -0.10
    assert summary["margin_best_residual_vs_eos"] == -0.40
    assert summary["row_score_policy_id"] == "residual_row_mean_v1"
    assert summary["score_token_reduction"] == "mean_logprob"
