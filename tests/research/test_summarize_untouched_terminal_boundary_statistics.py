from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "summarize_untouched_terminal_boundary_statistics.py"
)
_SPEC = importlib.util.spec_from_file_location("terminal_boundary_statistics", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_wilson_interval_and_fraction_keep_declared_denominator() -> None:
    value = summary._fraction(45, 200)

    assert value["success_count"] == 45
    assert value["count"] == 200
    assert value["fraction"] == pytest.approx(0.225)
    lower, upper = value["wilson_95_interval"]
    assert lower < 0.225 < upper


def test_threshold_summary_keeps_pairwise_and_robust_thresholds_separate() -> None:
    value = summary._threshold_summary([-2.0, -0.25, 0.1, 0.75, 1.5])

    assert value["row_opener_over_terminal"]["success_count"] == 3
    assert value["margin_greater_than_0_5"]["success_count"] == 2
    assert value["margin_greater_than_1_0"]["success_count"] == 1
    assert value["within_0_5_of_pairwise_boundary"]["success_count"] == 2


def test_paired_sign_summary_reports_both_crossing_directions() -> None:
    value = summary._paired_sign_summary([-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0])

    assert value == {
        "source_nonpositive_to_checkpoint_nonpositive": 1,
        "source_nonpositive_to_checkpoint_positive": 1,
        "source_positive_to_checkpoint_nonpositive": 1,
        "source_positive_to_checkpoint_positive": 1,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "depth_1_to_2"),
        (3, "depth_3_to_5"),
        (9, "depth_6_to_9"),
        (10, "depth_10_plus"),
    ],
)
def test_depth_bins_are_exhaustive(value: int, expected: str) -> None:
    assert summary._depth_bin(value) == expected


def test_source_margin_bins_do_not_relabel_observed_stops() -> None:
    assert summary._source_margin_bin(0.1) == "source_diagnostic_positive_despite_observed_stop"
    assert summary._source_margin_bin(-0.25) == "source_terminal_ahead_by_less_than_0_5"
    assert summary._source_margin_bin(-1.0) == "source_terminal_ahead_by_0_5_to_2_0"
    assert summary._source_margin_bin(-2.0) == "source_terminal_ahead_by_at_least_2_0"
