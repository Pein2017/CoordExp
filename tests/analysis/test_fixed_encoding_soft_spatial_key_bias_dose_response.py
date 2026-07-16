from __future__ import annotations

import pytest
import torch

from scripts.research.run_fixed_encoding_soft_spatial_key_bias_dose_response import (
    apply_image139_actuation_gate,
    assess_parent_all_allowed_continuity,
    assess_lambda05_actuation_gate,
    build_soft_spatial_key_bias_mask,
    classify_case_at_dose,
    classify_shared_dose_panel,
    collect_panel_rows_by_dose,
    compare_feature_fingerprints,
    signed_distance_toward_hard_endpoint,
)


def _scores(*, row: float, geometry: float, target: float, competitor: float, target_release: float = 0.1, competitor_release: float = 0.0):
    return {
        "full_row": {"crossover": row, "gamma_target": target, "gamma_competitor": competitor, "target_release": target_release, "competitor_release": competitor_release},
        "geometry": {"crossover": geometry, "gamma_target": target, "gamma_competitor": competitor},
        "description": {"crossover": row, "gamma_target": target, "gamma_competitor": competitor},
        "first_differing_description": {"crossover": row, "gamma_target": target, "gamma_competitor": competitor},
    }


def test_soft_mask_is_float32_additive_and_causal() -> None:
    mask = build_soft_spatial_key_bias_mask(
        sequence_length=6,
        image_key_positions=[1, 2, 4],
        selected_image_positions=[2, 4],
        bias=1.0,
    )
    assert mask.shape == (1, 1, 6, 6)
    assert mask.dtype == torch.float32
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 5, 2].item() == 1.0
    assert mask[0, 0, 5, 4].item() == 1.0
    assert mask[0, 0, 5, 1].item() == 0.0
    assert torch.isneginf(mask[0, 0, 1, 2])
    assert torch.isneginf(mask[0, 0, 0, 2])


def test_soft_mask_rejects_invalid_values_and_positions() -> None:
    with pytest.raises(ValueError):
        build_soft_spatial_key_bias_mask(sequence_length=3, image_key_positions=[1], selected_image_positions=[2], bias=1.0)
    with pytest.raises(ValueError):
        build_soft_spatial_key_bias_mask(sequence_length=3, image_key_positions=[1], selected_image_positions=[1], bias=-1.0)


def test_feature_fingerprint_ignores_device_but_requires_content_shape_dtype() -> None:
    expected = {"primary": [{"sha256": "a", "shape": [2, 3], "dtype": "torch.float32", "device": "cuda:0"}], "deepstack": []}
    observed = {"primary": [{"sha256": "a", "shape": [2, 3], "dtype": "torch.float32", "device": "cpu"}], "deepstack": []}
    assert compare_feature_fingerprints(expected, observed)["passed"]
    observed["primary"][0]["sha256"] = "b"
    assert not compare_feature_fingerprints(expected, observed)["passed"]


def test_parent_continuity_requires_full_vector_and_ranks() -> None:
    parent = {"target": {"token_log_probabilities": [-1.0, -2.0], "selected_token_ranks": [1, 3]}, "competitor": {"token_log_probabilities": [-3.0, -4.0], "selected_token_ranks": [2, 4]}}
    observed = {"target": {"token_log_probabilities": [-1.0, -2.0], "selected_token_ranks": [1, 3]}, "competitor": {"token_log_probabilities": [-3.0, -4.0], "selected_token_ranks": [2, 4]}}
    assert assess_parent_all_allowed_continuity(parent_all_allowed=parent, observed_all_allowed=observed)["passed"]
    observed["target"]["selected_token_ranks"] = [2, 3]
    assert not assess_parent_all_allowed_continuity(parent_all_allowed=parent, observed_all_allowed=observed)["passed"]


def test_hard_endpoint_uses_exact_movement_and_signed_distance() -> None:
    finite = {"full_row": {"gamma_target": 1.0, "gamma_competitor": -0.5, "crossover": 1.5}}
    zero = {"full_row": {"gamma_full": 0.2, "gamma_target": 9.0, "gamma_competitor": -9.0, "crossover": 0.0}}
    hard = {"full_row": {"gamma_full": 7.0, "gamma_target": 1.4, "gamma_competitor": -0.8, "crossover": 2.2}}
    result = signed_distance_toward_hard_endpoint(finite=finite, full=zero, hard=hard)["full_row"]
    assert result["gamma_target_movement_from_zero"] == pytest.approx(0.8)
    assert result["gamma_target_signed_distance_to_hard"] == pytest.approx(-0.4)
    assert result["gamma_target_sign_agreement"]
    assert result["gamma_competitor_movement_from_zero"] == pytest.approx(-0.7)
    assert result["gamma_competitor_signed_distance_to_hard"] == pytest.approx(0.3)
    assert result["gamma_competitor_sign_agreement"]


def test_geometry_only_reversal_routes_even_when_full_row_crossover_is_small() -> None:
    scores = _scores(row=0.01, geometry=0.20, target=0.20, competitor=-0.20)
    scores["full_row"]["gamma_target"] = 0.0
    scores["full_row"]["gamma_competitor"] = 0.0
    assert (
        classify_case_at_dose(
            scores,
            no_op_drift=0.0,
            different_category=False,
            dose=0.5,
        )
        == "route_to_phase_specific_discriminator"
    )


def test_lambda05_actuation_gate_is_strictly_above_ten_times_noop() -> None:
    common = dict(
        target_bias_target=[-1.0, -2.0], target_bias_competitor=[-1.0, -2.0],
        competitor_bias_target=[-1.0, -2.0], competitor_bias_competitor=[-1.0, -2.0],
        zero_target=[-1.0, -2.0], zero_competitor=[-1.0, -2.0], no_op_drift=0.1,
    )
    assert not assess_lambda05_actuation_gate(**common)["passed"]
    common["target_bias_target"] = [-1.0, -0.0]
    assert assess_lambda05_actuation_gate(**common)["passed"]


def test_dose_two_requires_compatible_dose_one_signs() -> None:
    good = _scores(row=0.2, geometry=0.2, target=0.2, competitor=-0.2)
    weak_good_one = _scores(row=0.01, geometry=0.01, target=0.01, competitor=-0.01)
    bad_one = _scores(row=0.2, geometry=0.2, target=-0.2, competitor=0.2)
    assert classify_case_at_dose(good, no_op_drift=0.01, different_category=True, dose=2.0, dose_one_crossover=good) == "promote_bounded_free_row_switch_replay"
    assert classify_case_at_dose(good, no_op_drift=0.01, different_category=True, dose=2.0, dose_one_crossover=weak_good_one) == "promote_bounded_free_row_switch_replay"
    assert classify_case_at_dose(good, no_op_drift=0.01, different_category=True, dose=2.0, dose_one_crossover=bad_one) == "inconclusive"


def test_lambda05_actuation_is_invalidating_only_for_image139() -> None:
    failed_gate = {"passed": False}
    other = apply_image139_actuation_gate(
        {
            "image_id": "632",
            "classification": "inconclusive",
            "lambda_0.5_actuation_gate": failed_gate,
        }
    )
    assert other["classification"] == "inconclusive"
    smoke = apply_image139_actuation_gate(
        {
            "image_id": "139",
            "classification": "inconclusive",
            "lambda_0.5_actuation_gate": failed_gate,
        }
    )
    assert smoke["classification"] == "invalid_lambda_0.5_actuation_gate"


def test_invalid_top_level_case_cannot_reenter_panel_by_dose() -> None:
    promoting = _scores(
        row=0.2,
        geometry=0.2,
        target=0.2,
        competitor=-0.2,
    )
    invalid = {
        "image_id": "bad",
        "classification": "invalid_lambda_0.5_actuation_gate",
        "different_category": True,
        "no_op_max_abs_logprob_drift": 0.0,
        "doses": {"0.5": {"crossover": promoting}},
    }
    valid = {
        "image_id": "good",
        "classification": "inconclusive",
        "different_category": True,
        "no_op_max_abs_logprob_drift": 0.0,
        "doses": {"0.5": {"crossover": promoting}},
    }
    by_dose = collect_panel_rows_by_dose([invalid, valid])
    assert [row["image_id"] for row in by_dose["0.5"]] == ["good"]
    assert (
        classify_shared_dose_panel(by_dose)["classification"]
        != "promote_one_bounded_free_row_switch_replay"
    )


def test_shared_dose_requires_two_cases_and_different_category() -> None:
    def row(image_id: str, different: bool):
        return {"image_id": image_id, "different_category": different, "classification": "promote_bounded_free_row_switch_replay", "crossover": _scores(row=0.2, geometry=0.2, target=0.2, competitor=-0.2)}
    one = classify_shared_dose_panel({0.5: [row("a", True)], 1.0: [], 2.0: []})
    assert one["classification"] != "promote_one_bounded_free_row_switch_replay"
    two_same = classify_shared_dose_panel({0.5: [row("a", False), row("b", False)]})
    assert two_same["classification"] != "promote_one_bounded_free_row_switch_replay"
    two_mixed = classify_shared_dose_panel({0.5: [row("a", False), row("b", True)]})
    assert two_mixed["classification"] == "promote_one_bounded_free_row_switch_replay"
    assert two_mixed["selected_shared_bias"] == 0.5


def test_lowest_shared_dose_is_selected() -> None:
    def row(image_id: str):
        return {"image_id": image_id, "different_category": True, "classification": "promote_bounded_free_row_switch_replay", "crossover": _scores(row=0.2, geometry=0.2, target=0.2, competitor=-0.2)}
    result = classify_shared_dose_panel({0.5: [], 1.0: [row("a"), row("b")], 2.0: [row("a"), row("b")]})
    assert result["selected_shared_bias"] == 1.0


def test_phase_route_requires_two_cases_with_same_named_phase() -> None:
    def row(image_id: str, phases: list[str]):
        return {
            "image_id": image_id,
            "classification": "route_to_phase_specific_discriminator",
            "phase_specific_reversals": phases,
            "crossover": {},
        }
    mixed = classify_shared_dose_panel({0.5: [row("a", ["description"]), row("b", ["geometry"])]})
    assert mixed["classification"] == "close_uniform_soft_spatial_key_bias"
    same = classify_shared_dose_panel({0.5: [row("a", ["description"]), row("b", ["description"])]})
    assert same["classification"] == "route_to_one_phase_specific_discriminator"
    assert same["phase"] == "description"
