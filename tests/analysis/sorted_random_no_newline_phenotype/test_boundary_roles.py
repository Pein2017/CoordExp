from __future__ import annotations

import json
import math

import pytest

from src.analysis.sorted_random_no_newline_phenotype.boundary_roles import (
    ALLOWED_BOUNDARY_WINNER_CLASSES,
    EOS_DESC,
    candidate_roles,
    summarize_boundary,
)


def test_candidate_roles_preserves_overlapping_role_order_from_roadmap() -> None:
    roles = candidate_roles(
        desc="chair",
        target_fn_desc="chair",
        emitted_same_desc_gt_indices=[1],
        residual_same_desc_gt_indices=[2],
        selected_hard_competitor_desc=None,
    )

    assert roles == [
        "target_fn_desc",
        "emitted_same_desc",
        "residual_same_desc",
    ]


def test_candidate_roles_appends_hard_competitor_before_other_gt_fallback() -> None:
    assert candidate_roles(
        desc="chair",
        target_fn_desc="chair",
        emitted_same_desc_gt_indices=[1],
        residual_same_desc_gt_indices=[2],
        selected_hard_competitor_desc="chair",
    ) == [
        "target_fn_desc",
        "emitted_same_desc",
        "residual_same_desc",
        "hard_competitor",
    ]
    assert candidate_roles(
        desc="lamp",
        target_fn_desc="chair",
        emitted_same_desc_gt_indices=[],
        residual_same_desc_gt_indices=[],
        selected_hard_competitor_desc="lamp",
    ) == ["hard_competitor"]
    assert candidate_roles(
        desc="table",
        target_fn_desc="chair",
        emitted_same_desc_gt_indices=[],
        residual_same_desc_gt_indices=[],
        selected_hard_competitor_desc=None,
    ) == ["other_gt_desc"]


def test_boundary_summary_keeps_role_lists_and_calculates_residual_margins() -> None:
    summary = summarize_boundary(
        candidates=[
            {
                "desc": "chair",
                "score": 2.0,
                "roles": [
                    "target_fn_desc",
                    "emitted_same_desc",
                    "residual_same_desc",
                ],
            },
            {
                "desc": "lamp",
                "score": 1.4,
                "roles": ["hard_competitor"],
            },
        ],
        eos_score=2.0,
        low_margin_threshold=0.10,
    )

    assert set(ALLOWED_BOUNDARY_WINNER_CLASSES) == {
        "residual_same_desc_favored",
        "residual_other_desc_favored",
        "emitted_same_desc_favored",
        "emitted_other_desc_favored",
        "hard_competitor_favored",
        "other_gt_desc_favored",
        "eos_favored",
        "mixed_tie",
        "no_residual_candidate",
    }
    assert summary["winner_desc"] == "chair"
    assert summary["winner_roles"] == [
        "target_fn_desc",
        "emitted_same_desc",
        "residual_same_desc",
    ]
    assert summary["boundary_winner_class"] == "residual_same_desc_favored"
    assert summary["residual_vs_eos_margin"] == 0.0
    assert summary["residual_vs_winner_margin"] == 0.0
    assert summary["low_margin_flag"] is False
    assert summary["candidate_descs_with_roles"] == [
        {
            "desc": "chair",
            "roles": [
                "target_fn_desc",
                "emitted_same_desc",
                "residual_same_desc",
            ],
            "score": 2.0,
        },
        {"desc": "lamp", "roles": ["hard_competitor"], "score": 1.4},
    ]
    json.dumps(summary, allow_nan=False)


def test_boundary_summary_flags_low_margin_when_residual_barely_loses() -> None:
    summary = summarize_boundary(
        candidates=[
            {"desc": "chair", "score": 0.94, "roles": ["residual_same_desc"]},
            {"desc": "cup", "score": 1.0, "roles": ["emitted_other_desc"]},
        ],
        eos_score=0.50,
        low_margin_threshold=0.10,
    )

    assert summary["winner_desc"] == "cup"
    assert summary["winner_roles"] == ["emitted_other_desc"]
    assert summary["boundary_winner_class"] == "emitted_other_desc_favored"
    assert summary["residual_vs_eos_margin"] == 0.44
    assert summary["residual_vs_winner_margin"] == -0.06
    assert summary["low_margin_flag"] is True


def test_boundary_summary_handles_eos_favored_with_residual_candidate() -> None:
    summary = summarize_boundary(
        candidates=[
            {"desc": "chair", "score": 0.40, "roles": ["residual_other_desc"]},
            {"desc": "cup", "score": 0.30, "roles": ["emitted_same_desc"]},
        ],
        eos_score=0.80,
    )

    assert summary["winner_desc"] == EOS_DESC
    assert summary["winner_roles"] == []
    assert summary["boundary_winner_class"] == "eos_favored"
    assert summary["residual_vs_eos_margin"] == -0.40
    assert summary["residual_vs_winner_margin"] == -0.40
    assert summary["low_margin_flag"] is False


def test_boundary_summary_reports_no_residual_candidate_separately() -> None:
    summary = summarize_boundary(
        candidates=[
            {"desc": "cup", "score": 0.90, "roles": ["emitted_same_desc"]},
            {"desc": "lamp", "score": 0.80, "roles": ["hard_competitor"]},
        ],
        eos_score=0.10,
    )

    assert summary["winner_desc"] == "cup"
    assert summary["winner_roles"] == ["emitted_same_desc"]
    assert summary["boundary_winner_class"] == "no_residual_candidate"
    assert summary["residual_vs_eos_margin"] is None
    assert summary["residual_vs_winner_margin"] is None
    assert summary["low_margin_flag"] is False
    json.dumps(summary, allow_nan=False)


def test_boundary_summary_reports_mixed_tie_across_classes() -> None:
    summary = summarize_boundary(
        candidates=[
            {"desc": "chair", "score": 1.0, "roles": ["residual_same_desc"]},
            {"desc": "lamp", "score": 1.0, "roles": ["hard_competitor"]},
        ],
        eos_score=0.2,
    )

    assert summary["winner_desc"] is None
    assert summary["winner_roles"] == []
    assert summary["boundary_winner_class"] == "mixed_tie"
    assert summary["residual_vs_eos_margin"] == 0.8
    assert summary["residual_vs_winner_margin"] == 0.0
    assert summary["low_margin_flag"] is False
    assert summary["tied_winners"] == [
        {
            "desc": "chair",
            "roles": ["residual_same_desc"],
            "score": 1.0,
            "winner_class": "residual_same_desc_favored",
        },
        {
            "desc": "lamp",
            "roles": ["hard_competitor"],
            "score": 1.0,
            "winner_class": "hard_competitor_favored",
        },
    ]
    json.dumps(summary, allow_nan=False)


def test_boundary_summary_rejects_non_finite_candidate_score() -> None:
    with pytest.raises(ValueError, match="candidate score must be finite"):
        summarize_boundary(
            candidates=[
                {
                    "desc": "chair",
                    "score": math.nan,
                    "roles": ["residual_same_desc"],
                }
            ],
            eos_score=0.0,
        )


def test_boundary_summary_rejects_non_finite_eos_score() -> None:
    with pytest.raises(ValueError, match="eos_score must be finite"):
        summarize_boundary(
            candidates=[
                {
                    "desc": "chair",
                    "score": 1.0,
                    "roles": ["residual_same_desc"],
                }
            ],
            eos_score=math.inf,
        )


def test_boundary_summary_rejects_non_finite_derived_margin() -> None:
    with pytest.raises(ValueError, match="derived margin must be finite"):
        summarize_boundary(
            candidates=[
                {
                    "desc": "chair",
                    "score": 1e308,
                    "roles": ["residual_same_desc"],
                }
            ],
            eos_score=-1e308,
        )
