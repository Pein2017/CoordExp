from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.post_x1_instance_basin_tomography.merge_report import (
    build_report_markdown,
    build_summary,
    validate_report_language,
    write_report,
)


def test_report_preserves_reference_anchor_caveat() -> None:
    summary = build_summary(
        slot_rows=[
            {
                "checkpoint_role": "et_rmp_ce_ckpt3664",
                "comparison_role": "reference_anchor",
                "winner_bucket": "target_instance",
                "slot": "y1",
                "boundary_extreme_flag": False,
            },
            {
                "checkpoint_role": "fullobj_sorted_pure_ce_ckpt3668",
                "comparison_role": "clean_pair",
                "winner_bucket": "same_desc_competitor",
                "slot": "y1",
                "boundary_extreme_flag": True,
            },
        ],
        trajectory_rows=[],
        prefix_sensitivity_rows=[],
        greedy_rows=[],
    )

    report = build_report_markdown(summary)

    assert summary["comparison_semantics"] == "mechanism_traits_not_detector_accuracy"
    assert summary["boundary_extreme_counts_by_checkpoint"]["fullobj_sorted_pure_ce_ckpt3668"] == 1
    assert "not by final detector accuracy" in report
    assert "template_objective_confounded_reference" in report
    assert "reference_anchor" in report


def test_report_linter_rejects_uncaveated_metric_ranking_language() -> None:
    with pytest.raises(ValueError, match="detector ranking"):
        validate_report_language("sorted is best and wins AP50")


def test_write_report_materializes_summary_and_report(tmp_path: Path) -> None:
    summary = build_summary(
        slot_rows=[
            {
                "checkpoint_role": "fullobj_random_pure_ce_ckpt3668",
                "comparison_role": "clean_pair",
                "winner_bucket": "target_instance",
                "slot": "y1",
            },
        ],
        trajectory_rows=[],
        prefix_sensitivity_rows=[],
        greedy_rows=[],
        config_path="config.yaml",
        config_sha256="a" * 64,
    )

    result = write_report(tmp_path, summary)

    assert result["summary_path"] == str(tmp_path / "summary.json")
    assert (tmp_path / "summary.json").is_file()
    assert "not by final detector accuracy" in (tmp_path / "report.md").read_text(encoding="utf-8")
