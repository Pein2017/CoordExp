from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.controls import (
    headline_eligibility_status,
)


def test_missing_required_control_blocks_headline() -> None:
    controls = {
        "same_desc_count_1_control": "pass",
        "wrong_desc_same_image": "missing",
    }

    assert headline_eligibility_status(controls, valid_rate=1.0) == "ineligible_missing_controls"


def test_low_valid_rate_blocks_headline() -> None:
    controls = {"same_desc_count_1_control": "pass", "wrong_desc_same_image": "pass"}

    assert headline_eligibility_status(controls, valid_rate=0.2) == "ineligible_low_coverage"
