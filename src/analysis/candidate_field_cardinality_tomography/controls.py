from __future__ import annotations

from typing import Mapping


REQUIRED_CONTROL_TYPES = (
    "same_desc_count_1_control",
    "same_desc_count_2_control",
    "wrong_desc_same_image",
    "wrong_image_same_desc",
    "x1_projection_collision_slice",
    "gt_x1_jitter",
    "competitor_x1_control",
    "merge_radius_sensitivity",
    "mass_floor_sensitivity",
    "p_cond_vs_coord_vocab_mass",
)


def headline_eligibility_status(
    control_status_by_type: Mapping[str, str],
    *,
    valid_rate: float,
    minimum_valid_rate: float = 0.5,
) -> str:
    if valid_rate < minimum_valid_rate:
        return "ineligible_low_coverage"
    if any(control not in control_status_by_type for control in REQUIRED_CONTROL_TYPES):
        return "ineligible_missing_controls"
    if any(status in {"missing", "fail"} for status in control_status_by_type.values()):
        return "ineligible_missing_controls"
    return "eligible"
