from __future__ import annotations

import pytest
import torch

from scripts.research.run_fixed_encoding_object_centered_spatial_eligibility_crossover import (
    BOX_END,
    BOX_START,
    COORDINATE_TOKEN_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    assess_noop_trust_gate,
    build_causal_key_eligibility_mask,
    build_row_tokens,
    build_translated_competitor_mask,
    classify_case,
    classify_panel,
    compute_crossover_and_release,
    first_differing_description_index,
    _phase_score_map,
    score_row_log_likelihoods,
    select_highest_unrestricted_competitor,
    split_row_phases,
    translate_mask_to_center,
)


def test_causal_mask_keeps_non_image_keys_and_blocks_only_image_keys() -> None:
    mask = build_causal_key_eligibility_mask(
        sequence_length=6,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
    )[0, 0]
    assert bool(mask[5, 0])
    assert not bool(mask[5, 1])
    assert bool(mask[5, 2])
    assert not bool(mask[5, 3])
    assert not bool(mask[1, 2])  # future keys remain causal even if eligible


def test_translation_preserves_shape_count_and_center_ownership() -> None:
    source = torch.zeros(1 * 8 * 8, dtype=torch.bool).reshape(1, 8, 8)
    source[0, 1:3, 2:5] = True
    translated = translate_mask_to_center(
        source.reshape(-1),
        temporal=1,
        merged_height=8,
        merged_width=8,
        source_center=(1, 3),
        destination_center=(5, 6),
    )
    assert int(translated.sum()) == int(source.sum()) == 6
    assert translated.reshape(1, 8, 8)[0, 5, 6]
    assert translated.reshape(1, 8, 8)[0, 5, 5]


def test_competitor_translation_rejects_overlap_or_clip() -> None:
    target = torch.zeros(1 * 6 * 6, dtype=torch.bool).reshape(1, 6, 6)
    target[0, 1:3, 1:3] = True
    with pytest.raises(ValueError, match="overlaps"):
        build_translated_competitor_mask(
            target.reshape(-1),
            target_bbox_xyxy=(10, 10, 30, 30),
            competitor_bbox_xyxy=(10, 10, 30, 30),
            image_width=60,
            image_height=60,
            temporal=1,
            merged_height=6,
            merged_width=6,
        )


def test_row_tokens_and_phase_slicing_are_canonical() -> None:
    row = build_row_tokens(description_token_ids=[42, 43], coordinate_bins=[1, 2, 3, 4])
    assert row == [OBJECT_REF_START, 42, 43, OBJECT_REF_END, BOX_START, COORDINATE_TOKEN_START + 1, COORDINATE_TOKEN_START + 2, COORDINATE_TOKEN_START + 3, COORDINATE_TOKEN_START + 4, BOX_END]
    phases = split_row_phases(row, description_length=2)
    assert phases["row_entry"] == [0]
    assert phases["description"] == [1, 2]
    assert phases["x1"] == [5]
    assert phases["geometry"] == [4, 5, 6, 7, 8, 9]
    assert first_differing_description_index(row, build_row_tokens(description_token_ids=[42, 99], coordinate_bins=[1, 2, 3, 4]), description_length=2) == 2
    assert first_differing_description_index(
        build_row_tokens(description_token_ids=[42], coordinate_bins=[1, 2, 3, 4]),
        build_row_tokens(description_token_ids=[42, 99], coordinate_bins=[1, 2, 3, 4]),
        description_length=1,
    ) is None


def test_teacher_forced_phase_score_preserves_sum_and_mean() -> None:
    row = build_row_tokens(description_token_ids=[42], coordinate_bins=[1, 2, 3, 4])
    prefix_length = 3
    logits = torch.zeros(prefix_length + len(row), 200000)
    for index, token in enumerate(row):
        logits[prefix_length + index - 1, token] = float(index + 1)
    scores = score_row_log_likelihoods(logits, prefix_length=prefix_length, row_tokens=row, description_length=1, terminal_token_id=7)
    assert scores["full_row"]["count"] == len(row)
    assert scores["full_row"]["sum"] == pytest.approx(sum(scores["token_log_probabilities"]))
    assert scores["full_row"]["mean"] == pytest.approx(scores["full_row"]["sum"] / len(row))
    assert "row_entry_vs_terminal" in scores


def test_noop_gate_requires_equal_ranks_and_small_drift() -> None:
    passed = assess_noop_trust_gate(
        implicit_token_log_probs=[-1.0, -2.0],
        explicit_token_log_probs=[-1.0 + 1e-6, -2.0],
        all_allowed_token_log_probs=[-1.0, -2.0 - 1e-6],
        implicit_ranks=[1, 2], explicit_ranks=[1, 2], all_allowed_ranks=[1, 2],
    )
    assert passed["passed"] is True
    failed = assess_noop_trust_gate(
        implicit_token_log_probs=[-1.0], explicit_token_log_probs=[-1.1], all_allowed_token_log_probs=[-1.0],
        implicit_ranks=[1], explicit_ranks=[2], all_allowed_ranks=[1],
    )
    assert failed["passed"] is False


def test_crossover_release_and_classification() -> None:
    def score(target: float, competitor: float) -> dict[str, dict[str, float]]:
        return {"target": {"mean": target}, "competitor": {"mean": competitor}}

    full = {"full_row": score(-2.0, -1.0), "geometry": score(-1.0, -0.8), "description": score(-1.0, -0.9)}
    target = {"full_row": score(-1.0, -2.0), "geometry": score(-0.2, -1.0), "description": score(-0.1, -1.0)}
    competitor = {"full_row": score(-2.0, -1.0), "geometry": score(-1.0, -0.1), "description": score(-1.0, -0.2)}
    result = compute_crossover_and_release(full_scores=full, target_scores=target, competitor_scores=competitor)
    assert result["full_row"]["crossover"] == pytest.approx(2.0)
    assert result["full_row"]["target_release"] == pytest.approx(1.0)
    assert classify_case(result, no_op_drift=1e-4) == "promote_bounded_free_row_switch_replay"


def test_case_requires_symmetric_owner_reversal_not_only_positive_crossover() -> None:
    crossover = {
        "full_row": {
            "crossover": 2.0,
            "gamma_target": -0.01,
            "gamma_competitor": -2.01,
            "target_release": 1.0,
            "competitor_release": 0.0,
        },
        "geometry": {
            "crossover": 2.0,
            "gamma_target": -0.01,
            "gamma_competitor": -2.01,
            "target_release": 1.0,
            "competitor_release": 0.0,
        },
    }
    assert classify_case(crossover, no_op_drift=1e-5) == "inconclusive"


def test_competitor_preselection_requires_b_to_outrank_a() -> None:
    candidates = [
        {"object_id": "low", "unrestricted_mean_full_row": -2.0},
        {"object_id": "winner", "unrestricted_mean_full_row": -0.5},
    ]
    selected = select_highest_unrestricted_competitor(candidates, target_mean_full_row=-1.0)
    assert selected is not None and selected["object_id"] == "winner"
    assert select_highest_unrestricted_competitor(candidates, target_mean_full_row=0.0) is None


def test_panel_requires_two_independent_promoting_cases() -> None:
    def result(image_id: str, classification: str, value: float = 0.2) -> dict[str, object]:
        return {
            "image_id": image_id,
            "classification": classification,
            "crossover": {"full_row": {"crossover": value}},
        }

    one = classify_panel([result("1", "promote_bounded_free_row_switch_replay")])
    assert one["classification"] == "inconclusive_isolated_owner_specific_case"
    two = classify_panel(
        [
            result("1", "promote_bounded_free_row_switch_replay", 0.2),
            result("2", "promote_bounded_free_row_switch_replay", 0.4),
        ]
    )
    assert two["classification"] == "promote_one_bounded_free_row_switch_replay"
    assert two["strongest_case_image_id"] == "2"
    with_invalid = classify_panel(
        [
            result("1", "promote_bounded_free_row_switch_replay", 0.2),
            result("2", "promote_bounded_free_row_switch_replay", 0.4),
            {"image_id": "3", "classification": "invalid_no_op_trust_gate"},
        ]
    )
    assert with_invalid["classification"] == "promote_one_bounded_free_row_switch_replay"
    assert with_invalid["invalid_no_op_case_ids"] == ["3"]


def test_phase_score_map_preserves_first_differing_description() -> None:
    base = {
        "full_row": {"mean": -1.0},
        "token_log_probabilities": [-1.0],
        "selected_token_ranks": [1],
        "top_prediction_token_ids": [7],
    }
    mapped = _phase_score_map(
        {
            "target": dict(base),
            "competitor": dict(base),
            "first_differing_description": {
                "target": {"sum": -0.1, "mean": -0.1, "count": 1},
                "competitor": {"sum": -2.0, "mean": -2.0, "count": 1},
            },
        }
    )
    assert mapped["first_differing_description"]["target"]["mean"] == -0.1
    assert mapped["first_differing_description"]["competitor"]["mean"] == -2.0
