from __future__ import annotations

import copy

import pytest
import torch

from scripts.research.run_repeated_first_differing_slot_coordinate_logits import (
    COORDINATE_TOKEN_START,
    HOMOGENEOUS_FOUR_COPY_LAYOUT,
    LOCAL_WINDOW_RADIUS,
    SINGLE_RECIPIENT_LAYOUT,
    build_case_comparisons,
    classify_bfloat_case,
    compare_coordinate_summaries,
    derive_case_contracts,
    first_differing_index,
    local_window,
    summarize_coordinate_logits,
)


def _full_logits(peaks: dict[int, float] | None = None) -> torch.Tensor:
    logits = torch.zeros(152700, dtype=torch.float32)
    for coordinate_bin, value in (peaks or {}).items():
        logits[COORDINATE_TOKEN_START + coordinate_bin] = value
    return logits


def _source_record(tokens: list[int]) -> dict:
    return {"generated_token_ids": tokens}


def test_first_differing_index_detects_token_and_length_differences() -> None:
    assert first_differing_index([1, 2, 3], [1, 2, 4]) == 2
    assert first_differing_index([1, 2], [1, 2, 3]) == 2
    with pytest.raises(ValueError, match="do not differ"):
        first_differing_index([1, 2], [1, 2])


def test_derive_case_contract_freezes_common_prefix_and_local_center() -> None:
    single = [10, 11, COORDINATE_TOKEN_START + 200, 99]
    homogeneous = [10, 11, COORDINATE_TOKEN_START + 201, 99]
    receipt = {
        "requested_model_execution_dtype": "bfloat16",
        "cases": [
            {
                "image_id": "8629",
                "source_bundle_path": "/tmp/source.json",
                "recipient_prompt_token_ids_sha256": "recipient",
                "source_first_row": {"row_token_ids_sha256": "row"},
                "comparison": {"primary_first_action_divergence": True},
                "single_recipient": _source_record(single),
                "homogeneous_four_copy_recipients": [
                    _source_record(homogeneous) for _ in range(4)
                ],
            }
        ],
    }

    contract = derive_case_contracts(receipt, ["8629"])[0]

    assert contract["first_differing_generated_token_index"] == 2
    assert contract["common_generated_prefix_token_ids"] == [10, 11]
    assert contract["source_single_selected_coordinate_bin"] == 200
    assert contract["source_homogeneous_selected_coordinate_bin"] == 201


def test_local_window_is_clipped_and_radius_is_frozen() -> None:
    assert local_window(10, LOCAL_WINDOW_RADIUS) == (0, 26)
    assert local_window(990, LOCAL_WINDOW_RADIUS) == (974, 999)
    with pytest.raises(ValueError, match="freezes"):
        local_window(500, 8)


def test_summary_preserves_complete_vector_and_dynamic_window_mass() -> None:
    summary = summarize_coordinate_logits(
        _full_logits({200: 6.0, 201: 5.0, 800: 4.0}), window=(184, 216)
    )

    assert len(summary["coordinate_raw_logits_float32"]) == 1000
    assert len(summary["coordinate_conditional_log_probabilities_float32"]) == 1000
    assert summary["selected_coordinate_bin"] == 200
    assert summary["local_coordinate_window_inclusive"] == [184, 216]
    assert summary["local_window_conditional_probability_mass_float32"] > 0
    assert len(summary["top_coordinates"]) == 20
    assert len(summary["coordinate_raw_logits_float32_sha256"]) == 64


def test_comparison_ignores_uniform_shift_and_detects_window_escape() -> None:
    reference = summarize_coordinate_logits(_full_logits({200: 6.0}), window=(184, 216))
    uniform = copy.deepcopy(reference)
    uniform["coordinate_raw_logits_float32"] = [
        value + 3.0 for value in reference["coordinate_raw_logits_float32"]
    ]
    unchanged = compare_coordinate_summaries(reference, uniform)
    assert unchanged["centered_root_mean_square_difference_float32"] == pytest.approx(
        0.0, abs=1e-7
    )

    escaped = summarize_coordinate_logits(_full_logits({800: 6.0}), window=(184, 216))
    changed = compare_coordinate_summaries(reference, escaped)
    assert changed["both_selected_bins_inside_local_window"] is False
    assert changed["jensen_shannon_divergence_float32"] > 0
    assert changed["outside_window_centered_root_mean_square_difference_float32"] > 0


def _execution(
    *, layout: str, repeat_index: int, summary: dict, reached: bool = True
) -> dict:
    rows = [{"batch_position": 0, "logit_summary": copy.deepcopy(summary)}]
    return {
        "layout_name": layout,
        "repeat_index": repeat_index,
        "cached_natural_replay": {
            "recipient_reached_by_every_request": reached,
            "rows": rows if reached else [],
        },
        "direct_full_prefix": {
            "recipient_reached_by_every_request": True,
            "rows": rows,
        },
    }


def test_build_case_comparisons_measures_batch_shift_and_repeat_floor() -> None:
    single = summarize_coordinate_logits(_full_logits({200: 6.0}), window=(184, 216))
    homogeneous = summarize_coordinate_logits(
        _full_logits({201: 6.0, 700: 2.0}), window=(184, 216)
    )
    executions = []
    for repeat_index in range(2):
        executions.extend(
            [
                _execution(
                    layout=SINGLE_RECIPIENT_LAYOUT,
                    repeat_index=repeat_index,
                    summary=single,
                ),
                _execution(
                    layout=HOMOGENEOUS_FOUR_COPY_LAYOUT,
                    repeat_index=repeat_index,
                    summary=homogeneous,
                ),
            ]
        )

    comparison = build_case_comparisons(
        executions, path_name="cached_natural_replay"
    )

    assert comparison["eligible"] is True
    assert comparison["maximum_same_layout_repeat_noise"][
        "centered_root_mean_square_difference_float32"
    ] == pytest.approx(0.0)
    assert comparison["single_vs_homogeneous_comparisons"][0][
        "centered_root_mean_square_difference_float32"
    ] > 0


def _classification_comparison(*, broad: bool) -> dict:
    fraction = 0.50 if broad else 0.01
    mass_shift = 0.06 if broad else 0.001
    repeat_ratio = 20.0 if broad else 2.0
    rows = []
    for repeat_index in range(2):
        rows.append(
            {
                "repeat_index": repeat_index,
                "both_selected_bins_inside_local_window": True,
                "local_window_probability_mass_shift_float32": mass_shift,
                "fraction_of_predecessor_broad_shift_anchor": {
                    "centered_root_mean_square_difference_float32": fraction,
                    "outside_window_centered_root_mean_square_difference_float32": fraction,
                    "jensen_shannon_divergence_float32": fraction,
                },
                "between_layout_divided_by_same_layout_repeat_noise": {
                    "centered_root_mean_square_difference_float32": repeat_ratio,
                    "outside_window_centered_root_mean_square_difference_float32": repeat_ratio,
                    "jensen_shannon_divergence_float32": repeat_ratio,
                },
            }
        )
    return {
        "eligible": True,
        "maximum_same_layout_repeat_noise": {
            "centered_root_mean_square_difference_float32": 0.0,
            "outside_window_centered_root_mean_square_difference_float32": 0.0,
            "jensen_shannon_divergence_float32": 0.0,
        },
        "single_vs_homogeneous_comparisons": rows,
    }


def test_classification_separates_broad_and_local_patterns() -> None:
    assert classify_bfloat_case(
        _classification_comparison(broad=True), source_split_reproduced=True
    ) == "broad_repeat_stable_shift_candidate"
    assert classify_bfloat_case(
        _classification_comparison(broad=False), source_split_reproduced=True
    ) == "local_same_basin_numeric_jitter_candidate"
    assert classify_bfloat_case(
        _classification_comparison(broad=False), source_split_reproduced=False
    ) == "inconclusive_recipient_or_source_split_failure"


def test_close_classification_rejects_larger_repeat_noise() -> None:
    comparison = _classification_comparison(broad=False)
    comparison["maximum_same_layout_repeat_noise"][
        "centered_root_mean_square_difference_float32"
    ] = 1.0

    assert classify_bfloat_case(
        comparison, source_split_reproduced=True
    ) == "bounded_intermediate_pattern"
