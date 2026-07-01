from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


BOUNDARY_SCORE_POLICY_ID = "boundary_desc_span_mean_v1"
SCORE_TOKEN_REDUCTION = "mean_logprob"
ALIGNMENT_LABELS = {
    "residual_favored",
    "emitted_favored",
    "eos_favored",
    "mixed_or_tied",
    "no_residual_candidate",
}
_TIE_EPSILON = 1e-12


def length_normalized_logprob(token_logprobs: Sequence[float]) -> float:
    values = [_finite_float(value, "token_logprobs") for value in token_logprobs]
    if not values:
        return 0.0
    return sum(values) / len(values)


def rank_desc_scores(desc_scores: Mapping[str, float] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _coerce_desc_score_rows(desc_scores)
    rows.sort(key=lambda row: (-float(row["score"]), str(row["desc"]), str(row.get("role", ""))))
    return [dict(row, rank=rank) for rank, row in enumerate(rows, start=1)]


def summarize_boundary_alignment(
    desc_scores: Mapping[str, float] | Sequence[Mapping[str, Any]],
    eos_score: float,
) -> dict[str, Any]:
    ranked = rank_desc_scores(desc_scores)
    eos = _finite_float(eos_score, "eos_score")
    residual_scores = [float(row["score"]) for row in ranked if row.get("role") == "residual"]
    emitted_scores = [float(row["score"]) for row in ranked if row.get("role") == "emitted"]
    best_residual = max(residual_scores) if residual_scores else None
    best_emitted = max(emitted_scores) if emitted_scores else None
    best_desc_row = ranked[0] if ranked else None
    alignment = _classify_alignment(ranked=ranked, eos_score=eos, best_residual=best_residual)
    summary = {
        "boundary_score_policy_id": BOUNDARY_SCORE_POLICY_ID,
        "score_token_reduction": SCORE_TOKEN_REDUCTION,
        "boundary_alignment": alignment,
        "ranked_desc_scores": ranked,
        "eos_score": eos,
        "best_desc": None if best_desc_row is None else best_desc_row["desc"],
        "best_role": None if best_desc_row is None else best_desc_row.get("role"),
        "best_desc_score": None if best_desc_row is None else float(best_desc_row["score"]),
        "best_residual_score": best_residual,
        "best_emitted_score": best_emitted,
        "margin_best_residual_vs_best_emitted": (
            None if best_residual is None or best_emitted is None else best_residual - best_emitted
        ),
        "margin_best_residual_vs_eos": None if best_residual is None else best_residual - eos,
    }
    _assert_json_safe_summary(summary)
    return summary


def score_boundary_desc_spans(
    *,
    model_handle: Mapping[str, Any],
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    boundary_assistant_text: str,
    candidate_descs: Sequence[str],
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    del model_handle, image_path, system_prompt, user_prompt, boundary_assistant_text, candidate_descs, eos_token_ids
    raise NotImplementedError(
        "GPU boundary scoring runtime is not implemented in the pure utility skeleton. "
        "Wire tokenizer/model calls here without importing heavy HF modules at module import time."
    )


def _coerce_desc_score_rows(desc_scores: Mapping[str, float] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(desc_scores, Mapping):
        return [
            {
                "desc": str(desc),
                "score": _finite_float(score, f"desc_scores[{desc!r}]"),
            }
            for desc, score in desc_scores.items()
        ]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(desc_scores):
        desc = row.get("desc")
        if desc is None:
            raise KeyError(f"desc_scores[{idx}] is missing 'desc'")
        if "score" not in row:
            raise KeyError(f"desc_scores[{idx}] is missing 'score'")
        output = {
            "desc": str(desc),
            "score": _finite_float(row["score"], f"desc_scores[{idx}]['score']"),
        }
        if row.get("role") is not None:
            output["role"] = str(row["role"])
        rows.append(output)
    return rows


def _classify_alignment(
    *,
    ranked: Sequence[Mapping[str, Any]],
    eos_score: float,
    best_residual: float | None,
) -> str:
    if best_residual is None:
        return "no_residual_candidate"
    candidate_scores = [float(row["score"]) for row in ranked]
    best_score = max([*candidate_scores, eos_score])
    leaders: list[str] = []
    if _close(eos_score, best_score):
        leaders.append("eos")
    leaders.extend(str(row.get("role", "unknown")) for row in ranked if _close(float(row["score"]), best_score))
    unique_leaders = set(leaders)
    if len(unique_leaders) != 1:
        return "mixed_or_tied"
    leader = unique_leaders.pop()
    if leader == "residual":
        return "residual_favored"
    if leader == "emitted":
        return "emitted_favored"
    if leader == "eos":
        return "eos_favored"
    return "mixed_or_tied"


def _finite_float(value: object, name: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TIE_EPSILON


def _assert_json_safe_summary(summary: Mapping[str, Any]) -> None:
    stack: list[Any] = [summary]
    while stack:
        value = stack.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("summary contains a non-finite float")
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
