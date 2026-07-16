from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid import (
    classify_hybrid_effect,
    build_hybrid_key_eligibility_mask,
    build_hybrid_arm_plan,
    compare_input_position_identities,
    hybrid_region_mapping,
    inspect_hybrid_mask_structure,
    validate_exact_image_ids,
    validate_factorial_receipt,
)
from scripts.research.run_fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial import (
    assess_cross_receipt_baseline,
    build_earlier_query_only_key_eligibility_mask,
)
from scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility import (
    build_query_scoped_key_eligibility_mask,
)


def _masks(earlier: list[int], row: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, dict]:
    image = list(range(20))
    kwargs = {
        "sequence_length": 40,
        "image_key_positions": image,
        "prefix_length": 8,
        "row_length": 5,
    }
    hybrid = build_hybrid_key_eligibility_mask(
        **kwargs,
        earlier_eligible_image_positions=earlier,
        row_eligible_image_positions=row,
    )
    earlier_mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=40,
        image_key_positions=image,
        eligible_image_positions=earlier,
        prefix_length=8,
    )
    row_mask = build_query_scoped_key_eligibility_mask(
        sequence_length=40,
        image_key_positions=image,
        eligible_image_positions=row,
        prefix_length=8,
        row_length=5,
    )
    baseline = torch.tril(torch.ones((40, 40), dtype=torch.bool))
    earlier_receipt = {
        "earlier_query_range": {"start_inclusive": 0, "end_inclusive": 6},
        "blocked_image_key_count": 20 - len(set(earlier)),
        "earlier_changed_cell_count": int((earlier_mask[0, 0] != baseline).sum()),
    }
    row_receipt = {
        "query_range": {"start_inclusive": 7, "end_inclusive": 11},
        "blocked_image_key_count": 20 - len(set(row)),
        "changed_cell_count": int((row_mask[0, 0] != baseline).sum()),
    }
    return hybrid, earlier_mask, row_mask, earlier_receipt, row_receipt


def test_hybrid_mask_uses_earlier_and_row_regions_without_future_changes() -> None:
    hybrid, _, _, _, _ = _masks(list(range(10)), list(range(10, 20)))
    observed = hybrid[0, 0]
    # Earlier queries q=0..6 see only the first region; row queries q=7..11
    # see only the second region. q=12 is deliberately not scored.
    assert not bool(observed[0, 10])
    assert bool(observed[0, 0])
    assert not bool(observed[7, 0])
    assert bool(observed[11, 10])
    assert torch.equal(torch.triu(observed, diagonal=1), torch.triu(torch.tril(torch.ones_like(observed)), diagonal=1))


def test_structural_proof_requires_exact_frozen_components_and_union() -> None:
    hybrid, earlier, row, earlier_receipt, row_receipt = _masks(list(range(10)), list(range(10, 20)))
    receipt = inspect_hybrid_mask_structure(
        hybrid,
        sequence_length=40,
        image_key_positions=list(range(20)),
        earlier_eligible_image_positions=list(range(10)),
        row_eligible_image_positions=list(range(10, 20)),
        prefix_length=8,
        row_length=5,
        earlier_reference_mask=earlier,
        row_reference_mask=row,
        earlier_reference_receipt=earlier_receipt,
        row_reference_receipt=row_receipt,
    )
    assert receipt["exact_twenty_eligible_image_keys_each"] is False
    assert not receipt["passed"]
    # The synthetic panel deliberately has ten keys per region, so it must not
    # pass the conclusion-level exact-twenty gate.
    corrupted = hybrid.clone()
    corrupted[0, 0, 7, 0] = True
    bad = inspect_hybrid_mask_structure(
        corrupted,
        sequence_length=40,
        image_key_positions=list(range(20)),
        earlier_eligible_image_positions=list(range(10)),
        row_eligible_image_positions=list(range(10, 20)),
        prefix_length=8,
        row_length=5,
        earlier_reference_mask=earlier,
        row_reference_mask=row,
        earlier_reference_receipt=earlier_receipt,
        row_reference_receipt=row_receipt,
    )
    assert not bad["passed"]
    corrupted_metadata = dict(earlier_receipt)
    corrupted_metadata["blocked_image_key_count"] = 999
    metadata_bad = inspect_hybrid_mask_structure(
        hybrid,
        sequence_length=40,
        image_key_positions=list(range(20)),
        earlier_eligible_image_positions=list(range(10)),
        row_eligible_image_positions=list(range(10, 20)),
        prefix_length=8,
        row_length=5,
        earlier_reference_mask=earlier,
        row_reference_mask=row,
        earlier_reference_receipt=corrupted_metadata,
        row_reference_receipt=row_receipt,
    )
    assert not metadata_bad["passed"]


def test_exact_twenty_key_panel_passes_structural_gate() -> None:
    image = list(range(40))
    earlier = list(range(20))
    row = list(range(20, 40))
    hybrid = build_hybrid_key_eligibility_mask(
        sequence_length=60,
        image_key_positions=image,
        earlier_eligible_image_positions=earlier,
        row_eligible_image_positions=row,
        prefix_length=8,
        row_length=5,
    )
    earlier_mask = build_earlier_query_only_key_eligibility_mask(
        sequence_length=60, image_key_positions=image, eligible_image_positions=earlier, prefix_length=8
    )
    row_mask = build_query_scoped_key_eligibility_mask(
        sequence_length=60, image_key_positions=image, eligible_image_positions=row, prefix_length=8, row_length=5
    )
    receipt = inspect_hybrid_mask_structure(
        hybrid,
        sequence_length=60,
        image_key_positions=image,
        earlier_eligible_image_positions=earlier,
        row_eligible_image_positions=row,
        prefix_length=8,
        row_length=5,
        earlier_reference_mask=earlier_mask,
        row_reference_mask=row_mask,
        earlier_reference_receipt={
            "earlier_query_range": {"start_inclusive": 0, "end_inclusive": 6},
            "blocked_image_key_count": 20,
            "earlier_changed_cell_count": int((earlier_mask[0, 0] != torch.tril(torch.ones((60, 60), dtype=torch.bool))).sum()),
        },
        row_reference_receipt={
            "query_range": {"start_inclusive": 7, "end_inclusive": 11},
            "blocked_image_key_count": 20,
            "changed_cell_count": int((row_mask[0, 0] != torch.tril(torch.ones((60, 60), dtype=torch.bool))).sum()),
        },
    )
    assert receipt["passed"]
    assert receipt["exact_twenty_eligible_image_keys_each"]


def test_factorial_digest_is_checked_before_json_parse(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_factorial_receipt(bad)


def test_exact_image_restriction() -> None:
    assert validate_exact_image_ids([139]) == ["139"]
    with pytest.raises(ValueError):
        validate_exact_image_ids([139, 632])


def test_frozen_factorial_baseline_parity_checks_logprobs_and_ranks() -> None:
    frozen = {
        "target": {"token_log_probabilities": [1.0, 2.0], "selected_token_ranks": [1, 2]},
        "competitor": {"token_log_probabilities": [3.0], "selected_token_ranks": [4]},
    }
    live = {
        "target": {"token_log_probabilities": [1.0, 2.00001], "selected_token_ranks": [1, 2]},
        "competitor": {"token_log_probabilities": [3.0], "selected_token_ranks": [4]},
    }
    assert assess_cross_receipt_baseline(live=live, frozen=frozen)["passed"]
    live["target"]["selected_token_ranks"] = [1, 3]
    assert not assess_cross_receipt_baseline(live=live, frozen=frozen)["passed"]


def _signature(gamma: float, delta_vase: float, delta_clock: float) -> dict[str, float]:
    return {"gamma_vase_minus_clock": gamma, "Delta_vase": delta_vase, "Delta_clock": delta_clock}


def test_classification_vase_activation() -> None:
    result = classify_hybrid_effect({"description": _signature(0.2, 0.1, 0.0), "geometry": _signature(0.2, 0.1, 0.0), "full_row": _signature(0.2, 0.1, 0.0)})
    assert result["classification"] == "constructive_vase_activation"


def test_classification_clock_activation() -> None:
    result = classify_hybrid_effect({"description": _signature(-0.2, 0.0, 0.1), "geometry": _signature(-0.2, 0.0, 0.1), "full_row": _signature(-0.2, 0.0, 0.1)})
    assert result["classification"] == "constructive_clock_activation"


def test_classification_pure_alternative_suppression() -> None:
    result = classify_hybrid_effect({"full_row": _signature(0.2, 0.0, -0.1)})
    assert result["classification"] == "clock_suppression"


def test_classification_destructive_preferred_owner_lowering() -> None:
    result = classify_hybrid_effect({"full_row": _signature(0.2, -0.1, 0.0)})
    assert result["classification"] == "destructive_preferred_owner_lowering"


def test_classification_mixed_movement_and_chimera_requires_two_phases() -> None:
    result = classify_hybrid_effect({"description": _signature(0.2, 0.1, -0.1), "geometry": _signature(-0.2, 0.0, 0.1), "full_row": _signature(0.0, 0.0, 0.0)})
    assert result["phase_labels"]["description"] == "mixed_activation_and_suppression"
    assert result["phrase_geometry_chimera"]
    assert result["full_row_is_not_sufficient_for_chimera"]


def test_classification_weak_or_ambiguous() -> None:
    result = classify_hybrid_effect({"full_row": _signature(0.01, 0.01, 0.0)})
    assert result["classification"] == "weak_or_ambiguous"


def test_classifier_infers_earlier_or_row_owner_without_hypothesis_input() -> None:
    earlier = classify_hybrid_effect({"description": _signature(0.2, 0.1, 0.0)})
    row = classify_hybrid_effect({"geometry": _signature(-0.2, 0.0, 0.1)})
    neutral = classify_hybrid_effect({"description": _signature(0.05, 0.1, 0.1)})
    assert earlier["description_owner"] == "vase"
    assert row["geometry_owner"] == "clock"
    assert neutral["phase_labels"]["description"] == "weak_or_ambiguous"


def test_identity_comparison_requires_all_four_row_hashes() -> None:
    expected = {
        "target": {"input_ids_sha256": "t-i", "position_ids_sha256": "t-p"},
        "competitor": {"input_ids_sha256": "c-i", "position_ids_sha256": "c-p"},
    }
    assert compare_input_position_identities(observed=expected, expected=expected)["passed"]
    for owner, field in (("target", "input_ids_sha256"), ("target", "position_ids_sha256"), ("competitor", "input_ids_sha256"), ("competitor", "position_ids_sha256")):
        corrupted = {name: dict(values) for name, values in expected.items()}
        corrupted[owner][field] = "corrupt"
        assert not compare_input_position_identities(observed=corrupted, expected=expected)["passed"]


def test_hybrid_region_mapping_and_secondary_deltas_are_unambiguous() -> None:
    assert hybrid_region_mapping("target_earlier_competitor_row") == ("target", "competitor")
    assert hybrid_region_mapping("competitor_earlier_target_row") == ("competitor", "target")
    with pytest.raises(ValueError):
        hybrid_region_mapping("unknown")


def test_hybrid_arm_plan_wires_regions_and_frozen_comparators_from_one_source() -> None:
    frozen = {
        "earlier_query_only_arms": {
            "target": {"name": "earlier-target"},
            "competitor": {"name": "earlier-competitor"},
        },
        "arms": {
            "target_row_query_only_hard": {"name": "row-target"},
            "competitor_row_query_only_hard": {"name": "row-competitor"},
        },
        "parent_hard_endpoint": {
            "target_eligibility": {"name": "all-target"},
            "competitor_eligibility": {"name": "all-competitor"},
        },
    }
    plan = build_hybrid_arm_plan(
        frozen_case=frozen,
        target_indices=[1, 2],
        competitor_indices=[7, 8],
    )
    target_earlier = plan["target_earlier_competitor_row"]
    competitor_earlier = plan["competitor_earlier_target_row"]
    assert target_earlier["earlier_indices"] == [1, 2]
    assert target_earlier["row_indices"] == [7, 8]
    assert target_earlier["row_only_reference"]["name"] == "row-competitor"
    assert target_earlier["matched_all_query_reference"]["name"] == "all-competitor"
    assert competitor_earlier["earlier_indices"] == [7, 8]
    assert competitor_earlier["row_indices"] == [1, 2]
    assert competitor_earlier["row_only_reference"]["name"] == "row-target"
    assert competitor_earlier["matched_all_query_reference"]["name"] == "all-target"


def test_phase_signature_keeps_row_only_and_matched_deltas_distinct() -> None:
    from scripts.research.run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid import build_phase_signatures

    def arm(target: float, competitor: float) -> dict:
        return {"target": {"full_row": {"mean": target}}, "competitor": {"full_row": {"mean": competitor}}}

    sig = build_phase_signatures(
        unrestricted=arm(1.0, 0.0),
        hybrid=arm(3.0, -1.0),
        row_only=arm(2.0, 0.5),
        matched=arm(2.5, -0.5),
    )["full_row"]
    assert sig["Delta_vase"] == pytest.approx(2.0)
    assert sig["Delta_clock"] == pytest.approx(-1.0)
    assert sig["secondary_delta_vase_to_same_row_region_row_only"] == pytest.approx(1.0)
    assert sig["secondary_delta_clock_to_same_row_region_row_only"] == pytest.approx(-1.5)
    assert sig["secondary_delta_vase_to_matched_arm"] == pytest.approx(0.5)
    assert sig["secondary_delta_clock_to_matched_arm"] == pytest.approx(-0.5)
