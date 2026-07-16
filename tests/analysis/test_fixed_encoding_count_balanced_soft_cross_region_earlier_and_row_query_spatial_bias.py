from __future__ import annotations

import math

import pytest
import torch

from scripts.research.run_fixed_encoding_count_balanced_soft_cross_region_earlier_and_row_query_spatial_bias import (
    ARM_REGION_MAPPING,
    assess_runtime_forward_semantics,
    LAMBDA_FLOAT32,
    LAMBDA_MATH,
    assess_matched_control_gate,
    build_count_balanced_soft_cross_region_mask,
    build_soft_cross_arm_plan,
    build_soft_cross_region_additive_mask,
    build_zero_float32_causal_mask,
    classify_crossed_panel,
    classify_owner_activation_suppression,
    finalize_soft_execution_gate,
    inspect_soft_cross_region_mask_structure,
    lambda_star,
    _phase_means,
)


def test_count_balanced_lambda_is_math_and_float32_value() -> None:
    assert LAMBDA_MATH == pytest.approx(math.log(994 / 20))
    assert LAMBDA_FLOAT32 == lambda_star()
    assert LAMBDA_FLOAT32 == float(torch.tensor(math.log(994 / 20), dtype=torch.float32).item())


def test_soft_mask_is_float32_additive_and_preserves_future() -> None:
    image = list(range(1014))
    earlier = list(range(20))
    row = list(range(30, 50))
    mask = build_soft_cross_region_additive_mask(
        sequence_length=1100,
        image_key_positions=image,
        earlier_selected_image_positions=earlier,
        row_selected_image_positions=row,
        prefix_length=50,
        row_length=20,
    )
    assert mask.shape == (1, 1, 1100, 1100)
    assert mask.dtype is torch.float32
    observed = mask[0, 0]
    assert torch.isneginf(observed[0, 100])
    assert observed[48, 0].item() == pytest.approx(LAMBDA_FLOAT32)
    assert observed[49, 30].item() == pytest.approx(LAMBDA_FLOAT32)
    assert observed[49, 0].item() == 0.0
    assert torch.isneginf(observed[68, 1000])
    assert torch.isneginf(observed[69, 100])


def test_soft_mask_structure_requires_exact_twenty_and_zero_visible_controls() -> None:
    image = list(range(1014))
    earlier = list(range(20))
    row = list(range(100, 120))
    mask = build_count_balanced_soft_cross_region_mask(
        sequence_length=1100,
        image_key_positions=image,
        earlier_selected_image_positions=earlier,
        row_selected_image_positions=row,
        prefix_length=50,
        row_length=20,
    )
    receipt = inspect_soft_cross_region_mask_structure(
        mask,
        sequence_length=1100,
        image_key_positions=image,
        earlier_selected_image_positions=earlier,
        row_selected_image_positions=row,
        prefix_length=50,
        row_length=20,
    )
    assert receipt["passed"]
    assert receipt["exact_1014_image_keys"]
    assert receipt["exact_twenty_selected_each"]
    assert receipt["future_changed_cell_count"] == 0
    assert receipt["changed_visible_nonselected_cell_count"] == 0
    assert receipt["changed_visible_non_image_cell_count"] == 0
    bad = mask.clone()
    bad[0, 0, 50, 500] = 0.25
    assert not inspect_soft_cross_region_mask_structure(
        bad,
        sequence_length=1100,
        image_key_positions=image,
        earlier_selected_image_positions=earlier,
        row_selected_image_positions=row,
        prefix_length=50,
        row_length=20,
    )["passed"]
    short = inspect_soft_cross_region_mask_structure(
        mask,
        sequence_length=1100,
        image_key_positions=image[:-1],
        earlier_selected_image_positions=earlier,
        row_selected_image_positions=row,
        prefix_length=50,
        row_length=20,
    )
    assert not short["passed"]


def test_zero_float32_causal_mask_has_no_finite_changes() -> None:
    mask = build_zero_float32_causal_mask(sequence_length=16)
    assert mask.dtype is torch.float32
    assert torch.all(mask[0, 0].tril() == 0.0)
    future = torch.triu(torch.ones((16, 16), dtype=torch.bool), diagonal=1)
    assert torch.all(torch.isneginf(mask[0, 0][future]))


def test_arm_plan_wires_all_four_arms_from_one_source() -> None:
    frozen = {
        "earlier_query_only_arms": {"target": {"tag": "early-t"}, "competitor": {"tag": "early-c"}},
        "arms": {
            "target_row_query_only_hard": {"tag": "row-t"},
            "competitor_row_query_only_hard": {"tag": "row-c"},
        },
        "parent_hard_endpoint": {"target_eligibility": {"tag": "all-t"}, "competitor_eligibility": {"tag": "all-c"}},
    }
    plan = build_soft_cross_arm_plan(frozen_case=frozen, target_indices=[1, 2], competitor_indices=[7, 8])
    assert set(plan) == set(ARM_REGION_MAPPING)
    assert plan["target_earlier_competitor_row"]["earlier_indices"] == [1, 2]
    assert plan["target_earlier_competitor_row"]["row_indices"] == [7, 8]
    assert plan["target_earlier_competitor_row"]["row_only_reference"]["tag"] == "row-c"
    assert plan["target_earlier_competitor_row"]["matched_all_query_reference"]["tag"] == "all-c"
    assert plan["competitor_earlier_target_row"]["earlier_indices"] == [7, 8]
    assert plan["competitor_earlier_target_row"]["row_indices"] == [1, 2]
    assert plan["competitor_earlier_target_row"]["row_only_reference"]["tag"] == "row-t"
    assert plan["competitor_earlier_target_row"]["matched_all_query_reference"]["tag"] == "all-t"


def test_owner_mechanism_table_is_exclusive_and_thresholds_are_closed() -> None:
    assert classify_owner_activation_suppression(gamma=0.1, preferred_owner_release=0.05, alternative_owner_release=-0.05)["mechanism"] == "mixed_activation_and_suppression"
    assert classify_owner_activation_suppression(gamma=0.1, preferred_owner_release=0.05, alternative_owner_release=0.0)["mechanism"] == "activation_only"
    assert classify_owner_activation_suppression(gamma=0.1, preferred_owner_release=0.0, alternative_owner_release=-0.05)["mechanism"] == "suppression_only"
    assert classify_owner_activation_suppression(gamma=0.1, preferred_owner_release=0.0, alternative_owner_release=0.0)["mechanism"] == "neither_activation_nor_suppression"
    neutral = classify_owner_activation_suppression(gamma=0.099, preferred_owner_release=1.0, alternative_owner_release=-1.0)
    assert neutral["preferred_owner"] is None
    assert neutral["mechanism"] == "neither_activation_nor_suppression"


def _signature(gamma: float, target_release: float, competitor_release: float, *, target_compat: float = 0.0, competitor_compat: float = 0.0) -> dict[str, float]:
    return {
        "gamma_vase_minus_clock": gamma,
        "Delta_vase": target_release,
        "Delta_clock": competitor_release,
        "compatibility_difference_target": target_compat,
        "compatibility_difference_competitor": competitor_compat,
    }


def _arm(top: int) -> dict[str, dict[str, list[int]]]:
    return {"target": {"top_prediction_token_ids": [top]}, "competitor": {"top_prediction_token_ids": [top]}}


def _matched_gate(phase_signatures: dict, *, first_index: int = 0, target_token: int = 11, competitor_token: int = 22, target_arm: dict | None = None, competitor_arm: dict | None = None) -> dict:
    plan = {
        "target_earlier_target_row": {},
        "competitor_earlier_competitor_row": {},
    }
    arms = {
        "target_earlier_target_row": target_arm or _arm(target_token),
        "competitor_earlier_competitor_row": competitor_arm or _arm(competitor_token),
    }
    return assess_matched_control_gate(
        plan=plan,
        arms=arms,
        phase_signatures=phase_signatures,
        target_row=[target_token],
        competitor_row=[competitor_token],
        first_differing_index=first_index,
    )


def test_matched_gate_requires_exact_top_id_agreement_and_release() -> None:
    signatures = {
        "target_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "competitor_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    assert _matched_gate(signatures)["passed"]
    bad_arm = {"target": {"top_prediction_token_ids": [11]}, "competitor": {"top_prediction_token_ids": [12]}}
    assert not _matched_gate(signatures, target_arm=bad_arm)["passed"]
    release_bad = {
        **signatures,
        "target_earlier_target_row": {"description": _signature(0.2, -0.0501, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
    }
    assert not _matched_gate(release_bad)["passed"]


def test_crossed_classifier_uses_strict_precedence_and_can_close_gate() -> None:
    rows = {"target_earlier_target_row": _arm(11), "competitor_earlier_competitor_row": _arm(22), "competitor_earlier_target_row": _arm(22), "target_earlier_competitor_row": _arm(11)}
    matched_sigs = {
        "target_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "competitor_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    gate = _matched_gate(matched_sigs)
    crossed = {
        **matched_sigs,
        "competitor_earlier_target_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "target_earlier_competitor_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    result = classify_crossed_panel(phase_signatures=crossed, arms=rows, target_row=[11], competitor_row=[22], first_differing_index=0, matched_gate=gate)
    assert result["classification"] == "bidirectional_soft_phase_split"
    assert classify_crossed_panel(phase_signatures=crossed, arms=rows, target_row=[11], competitor_row=[22], first_differing_index=0, matched_gate={"passed": False})["classification"] == "no_adjudication_close_count_balanced_soft_cross_operator"


def test_crossed_classifier_penalty_boundary_is_inclusive() -> None:
    rows = {"target_earlier_target_row": _arm(11), "competitor_earlier_competitor_row": _arm(22), "competitor_earlier_target_row": _arm(22), "target_earlier_competitor_row": _arm(11)}
    matched_sigs = {
        "target_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "competitor_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    gate = _matched_gate(matched_sigs)
    crossed = {
        **matched_sigs,
        "competitor_earlier_target_row": {"description": _signature(-0.2, 0.0, 0.0, target_compat=-0.05), "geometry": _signature(-0.2, 0.0, 0.0, target_compat=-0.05)},
        "target_earlier_competitor_row": {"description": _signature(0.2, 0.0, 0.0, competitor_compat=-0.05), "geometry": _signature(0.2, 0.0, 0.0, competitor_compat=-0.05)},
    }
    result = classify_crossed_panel(phase_signatures=crossed, arms=rows, target_row=[11], competitor_row=[22], first_differing_index=0, matched_gate=gate)
    assert result["classification"] == "hard_mismatch_penalty_persists_without_a_phase_split"


def test_crossed_classifier_row_region_and_earlier_region_owner_mapping() -> None:
    rows = {
        "target_earlier_target_row": _arm(11),
        "competitor_earlier_competitor_row": _arm(22),
        "competitor_earlier_target_row": _arm(11),
        "target_earlier_competitor_row": _arm(22),
    }
    matched = {
        "target_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "competitor_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    gate = _matched_gate(matched)
    row_control = {
        **matched,
        "competitor_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "target_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    assert classify_crossed_panel(
        phase_signatures=row_control, arms=rows, target_row=[11], competitor_row=[22],
        first_differing_index=0, matched_gate=gate,
    )["classification"] == "crossed_behavior_disappears_into_row_region_control"
    earlier_rows = {
        **rows,
        "competitor_earlier_target_row": _arm(22),
        "target_earlier_competitor_row": _arm(11),
    }
    earlier_control = {
        **matched,
        "competitor_earlier_target_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
        "target_earlier_competitor_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
    }
    assert classify_crossed_panel(
        phase_signatures=earlier_control, arms=earlier_rows, target_row=[11], competitor_row=[22],
        first_differing_index=0, matched_gate=gate,
    )["classification"] == "crossed_behavior_follows_earlier_region"


def test_asymmetric_classifier_accepts_penalty_on_either_crossed_arm() -> None:
    rows = {
        "target_earlier_target_row": _arm(11),
        "competitor_earlier_competitor_row": _arm(22),
        "competitor_earlier_target_row": _arm(22),
        "target_earlier_competitor_row": _arm(22),
    }
    matched = {
        "target_earlier_target_row": {"description": _signature(0.2, 0.0, 0.0), "geometry": _signature(0.2, 0.0, 0.0)},
        "competitor_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0), "geometry": _signature(-0.2, 0.0, 0.0)},
    }
    gate = _matched_gate(matched)
    def panel(a_penalty: float, b_penalty: float) -> dict:
        return {
            **matched,
            "competitor_earlier_target_row": {"description": _signature(-0.2, 0.0, 0.0, target_compat=a_penalty), "geometry": _signature(0.2, 0.0, 0.0, target_compat=a_penalty)},
            "target_earlier_competitor_row": {"description": _signature(-0.2, 0.0, 0.0, competitor_compat=b_penalty), "geometry": _signature(-0.2, 0.0, 0.0, competitor_compat=b_penalty)},
        }
    for a_penalty, b_penalty in ((-0.05, 0.0), (0.0, -0.05), (-0.05, -0.05)):
        assert classify_crossed_panel(
            phase_signatures=panel(a_penalty, b_penalty), arms=rows,
            target_row=[11], competitor_row=[22], first_differing_index=0, matched_gate=gate,
        )["classification"] == "asymmetric_hard_cross_phenotype_persists"


def test_finalize_soft_execution_gate_uses_late_outer_fields() -> None:
    inner = {"soft_execution_gate": {"passed": True, "deferred_outer_fields": ["feature_continuity_passed", "row_contract_passed"]}, "classification": "invalid_count_balanced_soft_cross_execution_gate", "feature_continuity": {"passed": True}, "row_contract": {"passed": True}, "crossed_panel_classification": {"classification": "mixed", "interpreted": True}}
    assert finalize_soft_execution_gate(inner)["soft_execution_gate"]["passed"]
    inner["row_contract"] = {"passed": False}
    invalid = finalize_soft_execution_gate(inner)
    assert not invalid["soft_execution_gate"]["passed"]
    assert not invalid["crossed_panel_classification"]["interpreted"]
    assert invalid["crossed_panel_classification"]["invalidated_by_execution_gate"]


def test_runtime_semantics_gate_rejects_raw_dtype_and_layer_mask_mutations() -> None:
    valid = {
        "passed_direct_forward_mask_consumption": True,
        "raw_forward_logits_dtype": "torch.float32",
        "text_layer_mask_hook": {"passed": True},
    }
    assert assess_runtime_forward_semantics(valid)["passed"]
    bad_dtype = {**valid, "raw_forward_logits_dtype": "torch.float16"}
    assert not assess_runtime_forward_semantics(bad_dtype)["passed"]
    bad_hook = {**valid, "text_layer_mask_hook": {"passed": False}}
    assert not assess_runtime_forward_semantics(bad_hook)["passed"]


def test_phase_means_ignores_non_phase_runtime_diagnostics() -> None:
    arm = {
        "target": {"description": {"mean": -1.0}, "runtime_attestation": {"passed": True}, "selected_token_logits": [1.0]},
        "competitor": {"description": {"mean": -2.0}, "runtime_attestation": {"passed": True}, "selected_token_logits": [2.0]},
    }
    assert _phase_means(arm) == {"description": {"vase": -1.0, "clock": -2.0}}
