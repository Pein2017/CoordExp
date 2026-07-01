from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


EOS_DESC = "<eos>"

ALLOWED_BOUNDARY_WINNER_CLASSES = (
    "residual_same_desc_favored",
    "residual_other_desc_favored",
    "emitted_same_desc_favored",
    "emitted_other_desc_favored",
    "hard_competitor_favored",
    "other_gt_desc_favored",
    "eos_favored",
    "mixed_tie",
    "no_residual_candidate",
)

_RESIDUAL_ROLES = frozenset(("residual_same_desc", "residual_other_desc"))
_WINNER_CLASS_BY_ROLE = (
    ("residual_same_desc", "residual_same_desc_favored"),
    ("residual_other_desc", "residual_other_desc_favored"),
    ("emitted_same_desc", "emitted_same_desc_favored"),
    ("emitted_other_desc", "emitted_other_desc_favored"),
    ("hard_competitor", "hard_competitor_favored"),
    ("other_gt_desc", "other_gt_desc_favored"),
)


def candidate_roles(
    desc: str,
    target_fn_desc: str,
    emitted_same_desc_gt_indices: Iterable[int],
    residual_same_desc_gt_indices: Iterable[int],
    selected_hard_competitor_desc: str | None,
) -> list[str]:
    roles: list[str] = []
    if desc == target_fn_desc:
        roles.append("target_fn_desc")
    if _has_any(emitted_same_desc_gt_indices):
        roles.append("emitted_same_desc")
    if _has_any(residual_same_desc_gt_indices):
        roles.append("residual_same_desc")
    if selected_hard_competitor_desc is not None and desc == selected_hard_competitor_desc:
        roles.append("hard_competitor")
    if not roles and desc != EOS_DESC:
        roles.append("other_gt_desc")
    return roles


def summarize_boundary(
    candidates: Sequence[Mapping[str, Any]],
    eos_score: float,
    *,
    low_margin_threshold: float = 0.05,
    tie_epsilon: float = 1e-9,
) -> dict[str, Any]:
    eos_score_value = _finite_float(eos_score, "eos_score")
    low_margin_threshold_value = _finite_float(
        low_margin_threshold,
        "low_margin_threshold",
    )
    tie_epsilon_value = _finite_float(tie_epsilon, "tie_epsilon")
    normalized = [_normalize_candidate(candidate) for candidate in candidates]
    candidate_descs_with_roles = [
        {
            "desc": candidate["desc"],
            "roles": list(candidate["roles"]),
            "score": candidate["score"],
        }
        for candidate in normalized
    ]
    residual_candidates = [
        candidate
        for candidate in normalized
        if _has_residual_role(candidate["roles"])
    ]
    best_residual_score = (
        max(candidate["score"] for candidate in residual_candidates)
        if residual_candidates
        else None
    )

    winner = _winner(normalized, eos_score_value, tie_epsilon=tie_epsilon_value)
    residual_vs_eos_margin = _margin(best_residual_score, eos_score_value)
    residual_vs_winner_margin = _margin(best_residual_score, winner["score"])

    if not residual_candidates:
        boundary_winner_class = "no_residual_candidate"
    else:
        boundary_winner_class = winner["boundary_winner_class"]

    summary = {
        "winner_desc": winner["winner_desc"],
        "winner_roles": list(winner["winner_roles"]),
        "boundary_winner_class": boundary_winner_class,
        "residual_vs_eos_margin": residual_vs_eos_margin,
        "residual_vs_winner_margin": residual_vs_winner_margin,
        "low_margin_flag": _low_margin_flag(
            residual_vs_eos_margin,
            residual_vs_winner_margin,
            threshold=low_margin_threshold_value,
            tie_epsilon=tie_epsilon_value,
        ),
        "candidate_descs_with_roles": candidate_descs_with_roles,
    }
    if winner["boundary_winner_class"] == "mixed_tie":
        summary["tied_winners"] = list(winner["tied_winners"])
    return summary


def _winner(
    candidates: Sequence[dict[str, Any]],
    eos_score: float,
    *,
    tie_epsilon: float,
) -> dict[str, Any]:
    if not candidates:
        return {
            "winner_desc": EOS_DESC,
            "winner_roles": [],
            "boundary_winner_class": "eos_favored",
            "score": eos_score,
            "tied_winners": [],
        }

    max_candidate_score = max(candidate["score"] for candidate in candidates)
    max_score = max(max_candidate_score, eos_score)
    top_candidates = [
        candidate
        for candidate in candidates
        if _is_tie(candidate["score"], max_score, tie_epsilon=tie_epsilon)
    ]

    if not top_candidates:
        return {
            "winner_desc": EOS_DESC,
            "winner_roles": [],
            "boundary_winner_class": "eos_favored",
            "score": eos_score,
            "tied_winners": [],
        }

    tied_winners = [_tied_winner(candidate) for candidate in top_candidates]
    top_classes = {winner["winner_class"] for winner in tied_winners}
    if len(top_classes) > 1:
        return {
            "winner_desc": None,
            "winner_roles": [],
            "boundary_winner_class": "mixed_tie",
            "score": max_score,
            "tied_winners": tied_winners,
        }

    first = top_candidates[0]
    return {
        "winner_desc": first["desc"],
        "winner_roles": list(first["roles"]),
        "boundary_winner_class": _winner_class(first["roles"]),
        "score": first["score"],
        "tied_winners": tied_winners,
    }


def _normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    desc = candidate.get("desc")
    if desc is None:
        raise ValueError("candidate is missing desc")
    if "score" not in candidate:
        raise ValueError(f"candidate is missing score: {desc}")
    roles = candidate.get("roles", [])
    if isinstance(roles, str):
        raise ValueError(f"candidate roles must be a list, not a string: {desc}")
    return {
        "desc": str(desc),
        "roles": [str(role) for role in roles],
        "score": _finite_float(candidate["score"], "candidate score"),
    }


def _tied_winner(candidate: Mapping[str, Any]) -> dict[str, Any]:
    roles = [str(role) for role in candidate["roles"]]
    return {
        "desc": str(candidate["desc"]),
        "roles": roles,
        "score": candidate["score"],
        "winner_class": _winner_class(roles),
    }


def _winner_class(roles: Sequence[str]) -> str:
    role_set = set(roles)
    for role, winner_class in _WINNER_CLASS_BY_ROLE:
        if role in role_set:
            return winner_class
    return "other_gt_desc_favored"


def _has_residual_role(roles: Sequence[str]) -> bool:
    return bool(_RESIDUAL_ROLES.intersection(roles))


def _has_any(values: Iterable[int]) -> bool:
    return any(True for _ in values)


def _margin(left: float | None, right: float) -> float | None:
    if left is None:
        return None
    margin = round(float(left) - float(right), 12)
    if not math.isfinite(margin):
        raise ValueError("derived margin must be finite")
    if margin == -0.0:
        return 0.0
    return margin


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _low_margin_flag(
    residual_vs_eos_margin: float | None,
    residual_vs_winner_margin: float | None,
    *,
    threshold: float,
    tie_epsilon: float,
) -> bool:
    if threshold <= 0.0:
        return False
    for margin in (residual_vs_eos_margin, residual_vs_winner_margin):
        if margin is None:
            continue
        magnitude = abs(margin)
        if tie_epsilon < magnitude <= threshold:
            return True
    return False


def _is_tie(left: float, right: float, *, tie_epsilon: float) -> bool:
    return abs(float(left) - float(right)) <= tie_epsilon


__all__ = [
    "ALLOWED_BOUNDARY_WINNER_CLASSES",
    "EOS_DESC",
    "candidate_roles",
    "summarize_boundary",
]
