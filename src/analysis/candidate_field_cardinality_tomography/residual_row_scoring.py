from __future__ import annotations

from typing import Any, Mapping, Sequence


ROW_SCORE_POLICY_ID = "residual_row_mean_v1"
SCORE_TOKEN_REDUCTION = "mean_logprob"


def score_residual_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_rows = [_score_one_row(row) for row in rows]
    if not scored_rows:
        return [], {
            "row_score_policy_id": ROW_SCORE_POLICY_ID,
            "score_token_reduction": SCORE_TOKEN_REDUCTION,
            "best_residual_row_id": None,
            "teacher_next_row_id": None,
            "preference_violation": False,
            "margin_best_residual_vs_teacher_next": None,
            "margin_best_residual_vs_eos": None,
        }
    best = max(scored_rows, key=lambda row: float(row["logp_row_mean"]))
    teacher = next((row for row in scored_rows if row.get("is_teacher_next") is True), None)
    eos_logprob = float(best["logp_eos_at_boundary"])
    summary = {
        "row_score_policy_id": ROW_SCORE_POLICY_ID,
        "score_token_reduction": SCORE_TOKEN_REDUCTION,
        "best_residual_row_id": best.get("residual_row_id"),
        "teacher_next_row_id": None if teacher is None else teacher.get("residual_row_id"),
        "preference_violation": False if teacher is None else best.get("residual_row_id") != teacher.get("residual_row_id"),
        "margin_best_residual_vs_teacher_next": (
            None if teacher is None else float(best["logp_row_mean"]) - float(teacher["logp_row_mean"])
        ),
        "margin_best_residual_vs_eos": float(best["logp_row_mean"]) - eos_logprob,
    }
    return scored_rows, summary


def _score_one_row(row: Mapping[str, Any]) -> dict[str, Any]:
    desc = _float_list(row.get("desc_logprobs"))
    bbox_tail = _float_list(row.get("bbox_tail_logprobs"))
    x1 = float(row.get("x1_logprob", 0.0))
    tokens = [*desc, x1, *bbox_tail]
    row_sum = sum(tokens)
    row_mean = row_sum / len(tokens) if tokens else 0.0
    scored = dict(row)
    scored.update(
        {
            "row_score_policy_id": ROW_SCORE_POLICY_ID,
            "score_token_reduction": SCORE_TOKEN_REDUCTION,
            "logp_row_mean": row_mean,
            "logp_row_sum": row_sum,
            "logp_desc_span_mean": sum(desc) / len(desc) if desc else 0.0,
            "logp_x1_token": x1,
            "logp_bbox_tail_mean": sum(bbox_tail) / len(bbox_tail) if bbox_tail else 0.0,
            "logp_eos_at_boundary": float(row.get("eos_logprob", 0.0)),
        }
    )
    return scored


def _float_list(value: object) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("logprob spans must be lists or tuples")
    return [float(item) for item in value]
