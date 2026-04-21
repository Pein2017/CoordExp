"""Selection and summary helpers for raw-text FN suppression probes."""

from __future__ import annotations

from statistics import mean
from typing import Sequence


def rank_fn_suppression_candidates(
    rows: Sequence[dict[str, object]],
    *,
    max_cases: int,
) -> list[dict[str, object]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row.get("recover_fraction_full") or 0.0),
            max(
                float(row.get("teacher_forced_support") or 0.0),
                float(row.get("proposal_support") or 0.0),
            ),
            -float(row.get("competitor_margin") or 0.0),
            str(row.get("case_id") or ""),
        ),
        reverse=True,
    )
    return ranked[: max(0, int(max_cases))]


def summarize_fn_suppression_margin_rows(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_alias"]), []).append(dict(row))

    model_metrics: list[dict[str, object]] = []
    for model_alias, group_rows in sorted(grouped.items()):
        stop_pressure_flags = [
            1.0
            if float(row["continue_minus_eos_sum_logprob"]) < 0.0
            and float(row["continue_minus_eos_mean_logprob"]) > 0.0
            else 0.0
            for row in group_rows
        ]
        model_metrics.append(
            {
                "model_alias": model_alias,
                "num_cases": len(group_rows),
                "mean_continue_minus_eos_sum_logprob": float(
                    mean(float(row["continue_minus_eos_sum_logprob"]) for row in group_rows)
                ),
                "mean_continue_minus_eos_mean_logprob": float(
                    mean(float(row["continue_minus_eos_mean_logprob"]) for row in group_rows)
                ),
                "positive_continue_minus_eos_sum_rate": float(
                    mean(
                        1.0
                        if float(row["continue_minus_eos_sum_logprob"]) > 0.0
                        else 0.0
                        for row in group_rows
                    )
                ),
                "positive_continue_minus_eos_mean_rate": float(
                    mean(
                        1.0
                        if float(row["continue_minus_eos_mean_logprob"]) > 0.0
                        else 0.0
                        for row in group_rows
                    )
                ),
                "stop_pressure_rate": float(mean(stop_pressure_flags)),
            }
        )
    return {
        "num_case_model_rows": len(rows),
        "model_metrics": model_metrics,
    }
