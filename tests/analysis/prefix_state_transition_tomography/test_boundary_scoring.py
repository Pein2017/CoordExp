from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.prefix_state_transition_tomography.boundary_scoring import (
    length_normalized_logprob,
    rank_desc_scores,
    score_boundary_desc_spans,
    summarize_boundary_alignment,
)


def test_length_normalized_logprob_uses_mean_logprob() -> None:
    assert length_normalized_logprob([-1.0, -3.0]) == -2.0


def test_rank_desc_scores_accepts_mapping_and_orders_highest_first() -> None:
    ranked = rank_desc_scores({"person": -1.0, "chair": -2.0})

    assert ranked[0]["desc"] == "person"
    assert ranked[0]["score"] == -1.0
    assert ranked[0]["rank"] == 1
    assert ranked[1]["desc"] == "chair"
    assert ranked[1]["rank"] == 2


def test_rank_desc_scores_accepts_rows_and_is_json_safe() -> None:
    ranked = rank_desc_scores(
        [
            {"desc": "chair", "role": "emitted", "score": -2.0},
            {"desc": "person", "role": "residual", "score": -1.0},
        ]
    )

    assert ranked[0] == {"desc": "person", "role": "residual", "score": -1.0, "rank": 1}
    json.dumps(ranked, allow_nan=False)


def test_summarize_boundary_alignment_labels_residual_favored() -> None:
    summary = summarize_boundary_alignment(
        desc_scores=[
            {"desc": "person", "role": "residual", "score": -1.0},
            {"desc": "chair", "role": "emitted", "score": -3.0},
        ],
        eos_score=-2.5,
    )

    assert summary["boundary_alignment"] == "residual_favored"
    assert summary["best_desc"] == "person"
    assert summary["best_role"] == "residual"
    assert summary["margin_best_residual_vs_eos"] == 1.5
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    ("desc_scores", "eos_score", "expected"),
    [
        (
            [
                {"desc": "person", "role": "residual", "score": -3.0},
                {"desc": "chair", "role": "emitted", "score": -1.0},
            ],
            -2.5,
            "emitted_favored",
        ),
        (
            [
                {"desc": "person", "role": "residual", "score": -3.0},
                {"desc": "chair", "role": "emitted", "score": -2.0},
            ],
            -1.0,
            "eos_favored",
        ),
        (
            [
                {"desc": "person", "role": "residual", "score": -1.0},
                {"desc": "chair", "role": "emitted", "score": -1.0},
            ],
            -3.0,
            "mixed_or_tied",
        ),
        (
            [{"desc": "chair", "role": "emitted", "score": -1.0}],
            -3.0,
            "no_residual_candidate",
        ),
    ],
)
def test_summarize_boundary_alignment_labels_other_cases(
    desc_scores: list[dict[str, object]],
    eos_score: float,
    expected: str,
) -> None:
    summary = summarize_boundary_alignment(desc_scores=desc_scores, eos_score=eos_score)

    assert summary["boundary_alignment"] == expected
    json.dumps(summary, allow_nan=False)


def test_score_boundary_desc_spans_runtime_skeleton_is_explicit_and_lightweight() -> None:
    with pytest.raises(NotImplementedError, match="GPU boundary scoring runtime"):
        score_boundary_desc_spans(
            model_handle={},
            image_path=Path("/tmp/example.jpg"),
            system_prompt="system",
            user_prompt="user",
            boundary_assistant_text="assistant",
            candidate_descs=["person"],
            eos_token_ids=[1],
        )
