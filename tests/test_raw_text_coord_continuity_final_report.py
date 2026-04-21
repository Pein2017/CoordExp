from __future__ import annotations

import math

from src.analysis.raw_text_coord_continuity_final_report import (
    derive_core_verdicts,
    summarize_bad_basin,
    summarize_perturbation,
    summarize_vision_lift,
)


def test_summarize_vision_lift_aggregates_gt_rows() -> None:
    correct_rows = [
        {
            "model_alias": "base",
            "image_id": 1,
            "object_index": 0,
            "slot": "x1",
            "candidate_value": 10,
            "gt_value": 10,
            "sum_logprob": -2.0,
        }
    ]
    swapped_rows = [
        {
            "model_alias": "base",
            "image_id": 1,
            "object_index": 0,
            "slot": "x1",
            "candidate_value": 10,
            "gt_value": 10,
            "sum_logprob": -5.0,
        }
    ]
    correct_summary = {
        "slot_metrics": [
            {
                "model_alias": "base",
                "slot": "x1",
                "mass_at_4": 0.7,
                "local_expected_abs_error": 3.0,
            }
        ]
    }
    swapped_summary = {
        "slot_metrics": [
            {
                "model_alias": "base",
                "slot": "x1",
                "mass_at_4": 0.4,
                "local_expected_abs_error": 4.5,
            }
        ]
    }
    rows = summarize_vision_lift(
        correct_rows=correct_rows,
        swapped_rows=swapped_rows,
        correct_summary=correct_summary,
        swapped_summary=swapped_summary,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["model_alias"] == "base"
    assert row["slot"] == "x1"
    assert math.isclose(row["avg_gt_score_lift"], 3.0)
    assert math.isclose(row["mass_at_4_lift"], 0.3)
    assert math.isclose(row["local_expected_abs_error_delta"], -1.5)


def test_summarize_bad_basin_compares_pred_and_gt_centers() -> None:
    summary = {
        "center_metrics": [
            {
                "model_alias": "pure_ce",
                "center_kind": "gt",
                "slot": "y1",
                "num_probes": 6,
                "mass_at_4": 0.4,
                "local_expected_abs_error": 8.0,
            },
            {
                "model_alias": "pure_ce",
                "center_kind": "pred",
                "slot": "y1",
                "num_probes": 6,
                "mass_at_4": 0.8,
                "local_expected_abs_error": 3.0,
            },
        ]
    }
    rows = summarize_bad_basin(summary)
    assert rows == [
        {
            "model_alias": "pure_ce",
            "slot": "y1",
            "num_probes": 6,
            "pred_minus_gt_mass_at_4": 0.4,
            "pred_minus_gt_local_expected_abs_error": -5.0,
            "pred_center_mass_at_4": 0.8,
            "gt_center_mass_at_4": 0.4,
        }
    ]


def test_summarize_perturbation_extracts_variant_metrics() -> None:
    summary = {
        "results": {
            "variant_metrics": [
                {
                    "model_alias": "base",
                    "variant": "source_x1y1_from_gt_next",
                    "slot": "y1",
                    "num_cases": 2,
                    "avg_delta_wrong_anchor_advantage_at_4": -0.2,
                    "avg_delta_pred_center_mass_at_4": -0.1,
                    "avg_delta_gt_center_mass_at_4": 0.1,
                }
            ]
        }
    }
    rows = summarize_perturbation(summary)
    assert rows[0]["variant"] == "source_x1y1_from_gt_next"
    assert rows[0]["slot"] == "y1"
    assert rows[0]["avg_delta_wrong_anchor_advantage_at_4"] == -0.2


def test_derive_core_verdicts_marks_coord_token_as_not_supported() -> None:
    broad_metrics = {
        ("base", "x1"): {"mass_at_4": 0.7, "local_expected_abs_error": 3.5},
        ("base", "y1"): {"mass_at_4": 0.72, "local_expected_abs_error": 3.2},
        ("pure_ce", "x1"): {"mass_at_4": 0.69, "local_expected_abs_error": 3.3},
        ("pure_ce", "y1"): {"mass_at_4": 0.75, "local_expected_abs_error": 3.0},
    }
    crowded_metrics = {
        ("base", "x1"): {"mass_at_4": 0.7, "local_expected_abs_error": 3.5},
        ("base", "y1"): {"mass_at_4": 0.71, "local_expected_abs_error": 3.2},
        ("pure_ce", "x1"): {"mass_at_4": 0.73, "local_expected_abs_error": 3.1},
        ("pure_ce", "y1"): {"mass_at_4": 0.8, "local_expected_abs_error": 2.6},
    }
    model_mined_metrics = {
        ("base", "x1"): {"mass_at_4": 0.67, "local_expected_abs_error": 3.7},
        ("base", "y1"): {"mass_at_4": 0.7, "local_expected_abs_error": 3.5},
        ("pure_ce", "x1"): {"mass_at_4": 0.79, "local_expected_abs_error": 2.9},
        ("pure_ce", "y1"): {"mass_at_4": 0.81, "local_expected_abs_error": 2.5},
    }
    lexical_summary = {
        "coefficients": {
            "numeric_distance_to_center": {"coef": -0.14, "pvalue": 1e-12}
        }
    }
    vision_lift_rows = [
        {
            "model_alias": "base",
            "slot": "x1",
            "avg_gt_score_lift": 4.0,
            "mass_at_4_lift": 0.08,
        },
        {
            "model_alias": "base",
            "slot": "y1",
            "avg_gt_score_lift": 3.0,
            "mass_at_4_lift": 0.04,
        },
    ]
    bad_basin_rows = [
        {"pred_minus_gt_mass_at_4": 0.3},
    ]
    perturb_rows = [
        {
            "slot": "y1",
            "variant": "source_x1y1_from_gt_next",
            "avg_delta_wrong_anchor_advantage_at_4": -0.1,
        }
    ]
    verdicts = derive_core_verdicts(
        broad_metrics=broad_metrics,
        crowded_metrics=crowded_metrics,
        model_mined_metrics=model_mined_metrics,
        lexical_summary=lexical_summary,
        vision_lift_rows=vision_lift_rows,
        bad_basin_rows=bad_basin_rows,
        perturb_rows=perturb_rows,
    )
    assert verdicts["q1_base_numeric_continuity"]["verdict"] == "strongly supported"
    assert verdicts["q2_pure_ce_enhances_continuity"]["verdict"] == "partially supported"
    assert verdicts["q3_visual_condition_modulates_continuity"]["verdict"] == "strongly supported"
    assert verdicts["q4_wrong_prefix_bad_basin"]["verdict"] == "strongly supported"
    assert verdicts["q5_coord_token_needed_for_continuity"]["verdict"] == "not supported"
