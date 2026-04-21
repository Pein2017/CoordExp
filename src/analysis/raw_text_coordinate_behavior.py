from __future__ import annotations

from typing import Mapping, Sequence


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    prev = list(range(len(right) + 1))
    for left_idx, left_value in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_value in enumerate(right, start=1):
            cost = 0 if left_value == right_value else 1
            current.append(
                min(
                    prev[right_idx] + 1,
                    current[right_idx - 1] + 1,
                    prev[right_idx - 1] + cost,
                )
            )
        prev = current
    return int(prev[-1])


def lexical_control_features(
    *,
    candidate_value: int,
    center_value: int,
    gt_value: int,
    candidate_tokens: Sequence[str],
    center_tokens: Sequence[str],
) -> dict[str, int]:
    candidate_text = str(candidate_value)
    center_text = str(center_value)
    shared_prefix_length = 0
    for left, right in zip(candidate_text, center_text):
        if left != right:
            break
        shared_prefix_length += 1
    return {
        "numeric_distance_to_center": abs(candidate_value - center_value),
        "numeric_distance_to_gt": abs(candidate_value - gt_value),
        "char_edit_distance": _edit_distance(
            tuple(candidate_text),
            tuple(center_text),
        ),
        "token_edit_distance": _edit_distance(candidate_tokens, center_tokens),
        "digit_length_match": int(len(candidate_text) == len(center_text)),
        "token_count": len(candidate_tokens),
        "shared_prefix_length": shared_prefix_length,
    }


def summarize_choice_margin(
    *,
    choice_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float | str]:
    ranked = sorted(
        (
            (label, float(payload["logprob_sum"]))
            for label, payload in choice_scores.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1]
    return {
        "winner": winner,
        "margin": winner_score - runner_up_score,
    }
