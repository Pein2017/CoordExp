from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import run_fixed_encoding_downstream_geometry_state_portability as geometry


EXPECTED_ROW_QUERY_PARENT_SHA256 = (
    "4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4"
)


def _mask(*, sequence_length: int = 16, prefix_length: int = 5, partial_length: int = 4) -> torch.Tensor:
    return geometry.build_partial_row_query_only_key_eligibility_mask(
        sequence_length=sequence_length,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=prefix_length,
        partial_row_length=partial_length,
    )


def test_complete_and_partial_query_ranges_are_exact_and_inclusive() -> None:
    assert geometry.query_range_for_complete_row(prefix_length=5, row_length=4) == (4, 7)
    assert geometry.query_range_for_partial_row(prefix_length=5, partial_row_length=1) == (4, 5)
    assert geometry.query_range_for_partial_row(prefix_length=5, partial_row_length=4) == (4, 8)
    with pytest.raises(ValueError):
        geometry.query_range_for_partial_row(prefix_length=0, partial_row_length=1)
    with pytest.raises(ValueError):
        geometry.query_range_for_partial_row(prefix_length=5, partial_row_length=0)


def test_partial_mask_restricts_only_dynamic_queries_and_blocked_image_keys() -> None:
    mask = _mask()
    receipt = geometry.inspect_partial_row_mask_structure(
        mask,
        sequence_length=16,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        partial_row_length=4,
    )
    assert receipt["passed"]
    assert receipt["query_range"]["indices"] == [4, 5, 6, 7, 8]
    assert receipt["changed_off_scope_query_cell_count"] == 0
    assert receipt["changed_non_image_key_cell_count"] == 0
    assert receipt["eligible_image_cells_unchanged"]
    assert receipt["exact_expected_mask_match"]
    observed = mask[0, 0]
    for query_index in range(16):
        assert bool(observed[query_index, 0]) is (0 <= query_index)
        assert bool(observed[query_index, 1]) is (query_index >= 1 and not (4 <= query_index <= 8))
        assert bool(observed[query_index, 2]) is (2 <= query_index)
        assert bool(observed[query_index, 3]) is (query_index >= 3 and not (4 <= query_index <= 8))


def test_dynamic_mask_rebuilds_from_box_start_through_each_coordinate() -> None:
    # Positions 1..3 stand in for image placeholder keys.  The five-token
    # prefix ends at position 4; BOX_START is the first materialized row token.
    ids = torch.tensor([[10, 99, 99, 99, 20, 30]])
    initial_mask, initial_receipt = geometry.build_dynamic_mask_for_ids(
        ids=ids,
        image_token_id=99,
        selected_indices=[1],
        prefix_length=5,
        device=torch.device("cpu"),
    )
    assert initial_receipt["passed"]
    assert initial_receipt["query_range"]["indices"] == [4, 5]
    assert bool(initial_mask[0, 0, 5, 1]) is False
    assert bool(initial_mask[0, 0, 5, 2]) is True
    assert bool(initial_mask[0, 0, 3, 1]) is True  # pre-row query unchanged

    after_x1 = torch.cat([ids, torch.tensor([[31]])], dim=1)
    next_mask, next_receipt = geometry.build_dynamic_mask_for_ids(
        ids=after_x1,
        image_token_id=99,
        selected_indices=[1],
        prefix_length=5,
        device=torch.device("cpu"),
    )
    assert next_receipt["passed"]
    assert next_receipt["query_range"]["indices"] == [4, 5, 6]
    assert bool(next_mask[0, 0, 6, 1]) is False
    assert bool(next_mask[0, 0, 6, 2]) is True
    assert bool(next_mask[0, 0, 3, 1]) is True


def test_all_query_regional_mask_fails_dynamic_structural_contract() -> None:
    # This is the historical all-query intervention: it blocks non-selected
    # image keys at every causal query rather than only P-1..P+K-1.
    bad = torch.tril(torch.ones((1, 1, 16, 16), dtype=torch.bool))
    for query_index in range(16):
        bad[0, 0, query_index, 1] = False
        bad[0, 0, query_index, 3] = False
    receipt = geometry.inspect_partial_row_mask_structure(
        bad,
        sequence_length=16,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        partial_row_length=4,
    )
    assert not receipt["passed"]
    assert receipt["changed_off_scope_query_cell_count"] > 0
    assert not receipt["exact_expected_mask_match"]


def test_non_image_and_future_key_access_remain_causal() -> None:
    observed = _mask(sequence_length=12, prefix_length=4, partial_length=3)[0, 0]
    baseline = torch.tril(torch.ones((12, 12), dtype=torch.bool))
    image_positions = {1, 2, 3}
    for query_index in range(12):
        for key_index in range(12):
            if key_index not in image_positions or key_index > query_index:
                assert bool(observed[query_index, key_index]) is bool(baseline[query_index, key_index])


def test_authoritative_parent_receipt_sha256_is_exact() -> None:
    assert geometry.ROW_QUERY_PARENT_SHA256 == EXPECTED_ROW_QUERY_PARENT_SHA256
    assert len(geometry.ROW_QUERY_PARENT_SHA256) == 64
    receipt_path = Path(geometry.DEFAULT_ROW_QUERY_RECEIPT)
    if receipt_path.exists():
        assert geometry.sha256_file(receipt_path) == EXPECTED_ROW_QUERY_PARENT_SHA256


def test_canonical_rows_sha256_is_exact() -> None:
    assert geometry.CANONICAL_ROWS_SHA256 == (
        "8b2a96f220e965e1822cb41f75ebcf170ad05fd6bd67cc0ae57ec69df58eec0e"
    )
    assert len(geometry.CANONICAL_ROWS_SHA256) == 64


def test_request_identity_is_frozen_to_image_632_books() -> None:
    assert geometry.build_request_identity() == {
        "image_id": "632",
        "recipient": {"object_name": "book", "annotation_id": "1661908"},
        "donors": [
            {"donor_name": "target_book", "object_name": "book", "annotation_id": "1661908"},
            {"donor_name": "competitor_book", "object_name": "book", "annotation_id": "1989419"},
        ],
    }


def _valid_geometry_suffix() -> list[int]:
    return [
        151770,
        151771,
        151870,
        151871,
        151649,  # box end
    ]


def test_strict_geometry_suffix_parser_accepts_exact_coordinate_suffix() -> None:
    parsed = geometry.parse_geometry_suffix(_valid_geometry_suffix())
    assert parsed["valid"] is True
    assert parsed["coordinate_bins"] == [100, 101, 200, 201]
    assert parsed["coordinate_token_ids"] == [151770, 151771, 151870, 151871]


@pytest.mark.parametrize(
    ("suffix", "reason"),
    [
        (_valid_geometry_suffix()[:-1], "truncated_geometry_suffix"),
        (_valid_geometry_suffix() + [151649], "trailing_token_after_geometry_suffix"),
        (_valid_geometry_suffix()[:4] + [151648], "box_end_not_final"),
        (_valid_geometry_suffix()[:3] + [151669, 151649], "coordinate_token_out_of_range"),
    ],
)
def test_strict_geometry_suffix_parser_rejects_malformed_paths(suffix: list[int], reason: str) -> None:
    parsed = geometry.parse_geometry_suffix(suffix)
    assert parsed["valid"] is False
    assert parsed["reason"] == reason


def test_geometry_ownership_requires_donor_distance_and_iou_floor() -> None:
    parsed = geometry.parse_geometry_suffix(_valid_geometry_suffix())
    donor = [100 / 999.0, 101 / 999.0, 200 / 999.0, 201 / 999.0]
    owned = geometry.geometry_ownership(
        parsed,
        donor_box_normalized=donor,
        paired_box_normalized=[0.800, 0.800, 0.900, 0.900],
        iou_floor=0.30,
    )
    assert owned["passed"] is True
    assert owned["donor_l1_distance"] == pytest.approx(0.0)
    assert owned["paired_l1_distance"] > owned["donor_l1_distance"]
    assert owned["donor_iou"] == pytest.approx(1.0)

    low_iou = geometry.geometry_ownership(
        parsed,
        donor_box_normalized=[0.000, 0.000, 0.050, 0.050],
        paired_box_normalized=[0.800, 0.800, 0.900, 0.900],
        iou_floor=0.30,
    )
    assert low_iou["passed"] is False
    assert low_iou["donor_iou"] < 0.30


def test_geometry_ownership_ties_fail_even_when_iou_is_perfect() -> None:
    parsed = geometry.parse_geometry_suffix(_valid_geometry_suffix())
    donor = [100 / 999.0, 101 / 999.0, 200 / 999.0, 201 / 999.0]
    tied = geometry.geometry_ownership(
        parsed,
        donor_box_normalized=donor,
        paired_box_normalized=donor,
        iou_floor=0.30,
    )
    assert tied["donor_iou"] == pytest.approx(1.0)
    assert tied["donor_l1_distance"] == pytest.approx(tied["paired_l1_distance"])
    assert tied["passed"] is False


def test_geometry_eligibility_and_portability_use_coordinate_release_only() -> None:
    assert geometry.assess_geometry_donor_eligibility(
        coordinate_release=0.2, valid_path=True, owner_match=True
    )["passed"]
    assert not geometry.assess_geometry_donor_eligibility(
        coordinate_release=-0.06, valid_path=True, owner_match=True
    )["passed"]
    assert not geometry.assess_geometry_donor_eligibility(
        coordinate_release=0.2, valid_path=True, owner_match=False
    )["passed"]

    passed = geometry.assess_geometry_portability(
        persistent_release=0.8,
        replacement_release=0.5,
        no_op_drift=1e-4,
        valid_path=True,
        owner_match=True,
    )
    assert passed["passed"] is True
    assert passed["box_end_excluded"] is True
    assert passed["half_positive_persistent_release_floor"] == pytest.approx(0.4)
    failed = geometry.assess_geometry_portability(
        persistent_release=0.8,
        replacement_release=0.05,
        no_op_drift=1e-4,
        valid_path=True,
        owner_match=True,
    )
    assert failed["passed"] is False


def test_self_noop_requires_teacher_forced_top_token_identity() -> None:
    accepted = geometry.assess_self_noop_trust(
        max_abs_logprob_drift=1e-5,
        teacher_forced_top_token_ids_equal=True,
        generated_token_ids_equal=True,
        replacement_count=1,
    )
    assert accepted["passed"]
    flipped_top = geometry.assess_self_noop_trust(
        max_abs_logprob_drift=1e-5,
        teacher_forced_top_token_ids_equal=False,
        generated_token_ids_equal=True,
        replacement_count=1,
    )
    assert not flipped_top["passed"]


def test_layer13_veto_applies_only_to_corresponding_layer23_donor() -> None:
    matching = geometry.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=["target"],
    )
    assert matching["classification"] == "vetoed_by_negative_control_layer_13"
    assert matching["corresponding_donor_veto_intersection"] == ["target"]

    disjoint = geometry.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=["competitor"],
    )
    assert disjoint["classification"] == "promote_bounded_one_sided_geometry_portability"
    assert disjoint["corresponding_donor_veto_intersection"] == []

    layer13_only = geometry.classify_geometry_panel(
        trust_passed=True,
        eligible_count=1,
        layer23_passed_donors=[],
        layer13_passed_donors=["target"],
    )
    assert layer13_only["classification"] == "close_one_site_conditional_downstream_portability"
