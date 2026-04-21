from __future__ import annotations

from src.analysis.raw_text_coordinate_preburst_probe import (
    build_preburst_variants,
    summarize_preburst_margin_rows,
)


def test_build_preburst_variants_modifies_only_source_geometry() -> None:
    prefix_objects = [
        {"desc": "chair", "bbox_2d": [10, 20, 30, 40]},
        {"desc": "person", "bbox_2d": [100, 200, 300, 400]},
        {"desc": "table", "bbox_2d": [50, 60, 70, 80]},
    ]
    gt_next = {"desc": "person", "bbox_2d": [111, 222, 333, 444]}

    variants = {
        row["variant_label"]: row
        for row in build_preburst_variants(
            prefix_objects=prefix_objects,
            source_object_index=1,
            gt_next=gt_next,
        )
    }

    assert list(variants) == [
        "baseline",
        "drop_source",
        "source_x1y1_from_gt_next",
        "source_bbox_from_gt_next",
    ]
    assert variants["baseline"]["prefix_objects"] == prefix_objects
    assert variants["baseline"]["duplicate_object"] == {
        "desc": "person",
        "bbox_2d": [100, 200, 300, 400],
    }
    assert variants["drop_source"]["prefix_objects"] == [
        {"desc": "chair", "bbox_2d": [10, 20, 30, 40]},
        {"desc": "table", "bbox_2d": [50, 60, 70, 80]},
    ]
    assert variants["drop_source"]["duplicate_object"] == {
        "desc": "person",
        "bbox_2d": [100, 200, 300, 400],
    }
    assert variants["source_x1y1_from_gt_next"]["prefix_objects"][1]["bbox_2d"] == [
        111,
        222,
        300,
        400,
    ]
    assert variants["source_x1y1_from_gt_next"]["duplicate_object"]["bbox_2d"] == [
        111,
        222,
        300,
        400,
    ]
    assert variants["source_bbox_from_gt_next"]["prefix_objects"][1]["bbox_2d"] == [
        111,
        222,
        333,
        444,
    ]
    assert variants["source_bbox_from_gt_next"]["duplicate_object"]["bbox_2d"] == [
        111,
        222,
        333,
        444,
    ]


def test_summarize_preburst_margin_rows_computes_variant_deltas() -> None:
    rows = [
        {
            "model_alias": "base_only",
            "case_id": "case-a",
            "variant_label": "baseline",
            "margin_sum_logprob": -0.4,
            "margin_mean_logprob": -0.2,
        },
        {
            "model_alias": "base_only",
            "case_id": "case-a",
            "variant_label": "source_x1y1_from_gt_next",
            "margin_sum_logprob": 0.6,
            "margin_mean_logprob": 0.5,
        },
        {
            "model_alias": "base_only",
            "case_id": "case-b",
            "variant_label": "baseline",
            "margin_sum_logprob": -0.1,
            "margin_mean_logprob": -0.05,
        },
        {
            "model_alias": "base_only",
            "case_id": "case-b",
            "variant_label": "source_x1y1_from_gt_next",
            "margin_sum_logprob": 0.2,
            "margin_mean_logprob": 0.1,
        },
    ]

    summary = summarize_preburst_margin_rows(rows)

    variant_rows = {
        (row["model_alias"], row["variant_label"]): row
        for row in summary["variant_metrics"]
    }
    baseline = variant_rows[("base_only", "baseline")]
    x1y1 = variant_rows[("base_only", "source_x1y1_from_gt_next")]

    assert summary["num_case_variant_rows"] == 4
    assert baseline["num_cases"] == 2
    assert baseline["positive_margin_mean_rate"] == 0.0
    assert x1y1["positive_margin_mean_rate"] == 1.0
    assert x1y1["mean_delta_from_baseline_mean_logprob"] == 0.425
    assert x1y1["mean_delta_from_baseline_sum_logprob"] == 0.65
