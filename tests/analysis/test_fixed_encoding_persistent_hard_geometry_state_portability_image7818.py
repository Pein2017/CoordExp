from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research import (
    run_fixed_encoding_persistent_hard_geometry_state_portability_image7818 as probe,
)


def _score(row: list[int], *, coordinate_value: float = -1.0) -> dict:
    values = [-9.0] * len(row)
    ranks = [99] * len(row)
    top_ids = [7] * len(row)
    start = len(row) - 8 + 3
    for offset in range(4):
        values[start + offset] = coordinate_value - offset
        ranks[start + offset] = offset + 1
        top_ids[start + offset] = 100 + offset
    return {
        "token_log_probabilities": values,
        "selected_token_ranks": ranks,
        "top_prediction_token_ids": top_ids,
    }


def _owner_attribution(*, donor_iou: float = 0.8) -> dict:
    return {
        "valid": True,
        "donor_annotation_id": probe.TARGET_ANNOTATION_ID,
        "donor_iou": donor_iou,
        "paired_iou": 0.0,
        "donor_l1_distance": 0.02,
        "paired_l1_distance": 1.0,
        "strictly_closer_to_donor": True,
        "all_object_ious": {
            probe.TARGET_ANNOTATION_ID: donor_iou,
            probe.PAIRED_ANNOTATION_ID: 0.1,
        },
        "strongest_competing_iou": 0.1,
        "support_envelope_donor_iou": 0.7,
        "support_envelope_donor_l1_distance": 0.04,
    }


def test_frozen_hashes_and_parent_receipts_are_exact() -> None:
    assert probe.PARENT_SPLIT_RECEIPT_SHA256 == (
        "7c4d9ea4c6a82da103453de4189ad1041f46efde3cf0333a6b17442d322aa017"
    )
    assert probe.PARENT_MERGED_RECEIPT_SHA256 == (
        "2046d5784030f255bc039b9252b2e63770e9f9b069219cfa9478a6e94c8f59b2"
    )
    if probe.PARENT_SPLIT_RECEIPT.exists() and probe.PARENT_MERGED_RECEIPT.exists():
        receipt = probe.validate_parent_receipts()
        assert receipt["split_sha256"] == probe.PARENT_SPLIT_RECEIPT_SHA256
        assert receipt["merged_sha256"] == probe.PARENT_MERGED_RECEIPT_SHA256


def test_input_hash_validation_checks_all_three_files(tmp_path: Path) -> None:
    paths = {}
    expected = {}
    for name, content in (("config", "c"), ("source_jsonl", "s"), ("audit_ledger", "l")):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path
        expected[f"{name}_sha256"] = probe.sha256_file(path)
    observed = probe.validate_input_hashes(
        config_path=paths["config"],
        source_jsonl_path=paths["source_jsonl"],
        audit_ledger_path=paths["audit_ledger"],
        expected=expected,
    )
    assert observed == expected
    paths["config"].write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256 drifted"):
        probe.validate_input_hashes(
            config_path=paths["config"],
            source_jsonl_path=paths["source_jsonl"],
            audit_ledger_path=paths["audit_ledger"],
            expected=expected,
        )


def test_multi_token_prefix_boundary_and_positions_are_exact() -> None:
    suffix = [
        probe.COORDINATE_TOKEN_START + 1,
        probe.COORDINATE_TOKEN_START + 2,
        probe.COORDINATE_TOKEN_START + 3,
        probe.COORDINATE_TOKEN_START + 4,
        probe.BOX_END,
    ]
    target = [probe.OBJECT_REF_START, 21, 22, probe.OBJECT_REF_END, probe.BOX_START, *suffix]
    paired = [probe.OBJECT_REF_START, 21, 22, probe.OBJECT_REF_END, probe.BOX_START, *suffix]
    boundary = probe.shared_boundary_contract(
        prompt_ids=[10, 11, 12], target_row=target, paired_row=paired
    )
    assert boundary["shared_prefix_token_ids"] == target[:5]
    assert boundary["boundary_pos"] == 7
    assert boundary["boundary_token_id"] == probe.BOX_START

    donor = torch.arange(27, dtype=torch.long).reshape(3, 1, 9)
    recipient = donor.clone()
    positions = probe.position_boundary_contract(
        donor_position_ids=donor,
        recipient_position_ids=recipient,
        boundary_pos=7,
    )
    assert positions["passed"]
    recipient[..., 8] += 10
    assert probe.position_boundary_contract(
        donor_position_ids=donor,
        recipient_position_ids=recipient,
        boundary_pos=7,
    )["passed"]
    recipient[..., 7] += 1
    assert not probe.position_boundary_contract(
        donor_position_ids=donor,
        recipient_position_ids=recipient,
        boundary_pos=7,
    )["passed"]


def test_canonical_gt_row_may_differ_from_realized_donor_geometry() -> None:
    canonical_target = probe.TARGET_CANONICAL_GT_ROW_TOKEN_IDS
    assert canonical_target[-5:-1] == [
        probe.COORDINATE_TOKEN_START + 213,
        probe.COORDINATE_TOKEN_START + 351,
        probe.COORDINATE_TOKEN_START + 356,
        probe.COORDINATE_TOKEN_START + 773,
    ]
    assert canonical_target != probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS
    assert probe.shared_row_prefix(
        canonical_target,
        probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS,
    ) == probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[:5]
    assert probe.shared_row_prefix(
        probe.TARGET_CANONICAL_GT_ROW_TOKEN_IDS,
        probe.PAIRED_CANONICAL_GT_ROW_TOKEN_IDS,
    ) == probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS[:5]


def test_dynamic_mask_owns_exact_partial_row_query_range() -> None:
    ids = torch.tensor([[10, 99, 99, 99, 20, probe.BOX_START]])
    mask, receipt = probe.build_dynamic_mask_for_ids(
        ids=ids,
        image_token_id=99,
        selected_indices=[1],
        prefix_length=5,
        device=torch.device("cpu"),
    )
    assert receipt["passed"]
    assert receipt["query_range"]["indices"] == [4, 5]
    assert receipt["changed_off_scope_query_cell_count"] == 0
    assert receipt["changed_non_image_key_cell_count"] == 0
    assert receipt["exact_expected_mask_match"]
    assert bool(mask[0, 0, 5, 1]) is False
    assert bool(mask[0, 0, 5, 2]) is True
    assert bool(mask[0, 0, 3, 1]) is True

    after_two_coordinates = torch.cat(
        (
            ids,
            torch.tensor(
                [[probe.COORDINATE_TOKEN_START, probe.COORDINATE_TOKEN_START + 1]]
            ),
        ),
        dim=1,
    )
    _, extended = probe.build_dynamic_mask_for_ids(
        ids=after_two_coordinates,
        image_token_id=99,
        selected_indices=[1],
        prefix_length=5,
        device=torch.device("cpu"),
    )
    assert extended["query_range"]["indices"] == [4, 5, 6, 7]


def test_suffix_is_exactly_four_coordinates_and_box_end() -> None:
    assert probe.parse_geometry_suffix(probe.TARGET_SUFFIX_TOKEN_IDS)["valid"]
    assert probe.parse_geometry_suffix(probe.PAIRED_SUFFIX_TOKEN_IDS)["valid"]
    assert not probe.parse_geometry_suffix(probe.TARGET_SUFFIX_TOKEN_IDS[:-1])["valid"]
    assert not probe.parse_geometry_suffix(
        [*probe.TARGET_SUFFIX_TOKEN_IDS, probe.BOX_END]
    )["valid"]


def test_parent_reproduction_uses_only_four_frozen_coordinate_slots() -> None:
    row = list(probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS)
    frozen = _score(row)
    live = _score(row)
    live["token_log_probabilities"][0] = 100.0
    reproduced = probe.assess_score_reproduction(live=live, frozen=frozen, row=row)
    assert reproduced["passed"]
    assert reproduced["selected_token_ids"] == row[-5:-1]
    assert len(reproduced["live_coordinate_log_probabilities"]) == 4

    live = _score(row)
    coordinate_start = len(row) - 8 + 3
    live["token_log_probabilities"][coordinate_start + 2] += 2e-4
    assert not probe.assess_score_reproduction(live=live, frozen=frozen, row=row)[
        "passed"
    ]


def test_noop_drift_and_discrete_trust_are_four_slot_and_cache_seam_tight() -> None:
    row = list(probe.TARGET_REALIZED_DONOR_ROW_TOKEN_IDS)
    baseline = _score(row)
    noop = _score(row)
    noop["token_log_probabilities"][0] = 100.0
    assert probe._coordinate_log_probs(baseline, row) == probe._coordinate_log_probs(
        noop, row
    )
    accepted = probe.assess_self_noop_trust(
        max_abs_logprob_drift=1e-5,
        teacher_forced_top_token_ids_equal=True,
        teacher_forced_selected_token_ranks_equal=True,
        generated_token_ids_equal=True,
        replacement_count=1,
        cached_replacement_count=1,
        cache_hook_removed=True,
    )
    assert accepted["passed"]
    assert not probe.assess_self_noop_trust(
        max_abs_logprob_drift=1e-5,
        teacher_forced_top_token_ids_equal=True,
        teacher_forced_selected_token_ranks_equal=False,
        generated_token_ids_equal=True,
        replacement_count=1,
        cached_replacement_count=1,
        cache_hook_removed=True,
    )["passed"]
    assert not probe.assess_self_noop_trust(
        max_abs_logprob_drift=1e-5,
        teacher_forced_top_token_ids_equal=True,
        teacher_forced_selected_token_ranks_equal=True,
        generated_token_ids_equal=True,
        replacement_count=1,
        cached_replacement_count=1,
        cache_hook_removed=False,
    )["passed"]


def test_one_shot_replacement_receipt_owns_only_boundary_and_removes_hook() -> None:
    class Layer(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + 1.0

    layer = Layer()
    replacement = probe.residual.ResidualStateReplacement(
        layer, boundary_pos=1, replacement=torch.full((3,), 9.0)
    )
    with replacement:
        first = layer(torch.zeros(1, 3, 3))
        second = layer(torch.zeros(1, 1, 3))
    assert replacement.replacement_count == 1
    assert replacement.hook_removed_inside_hook
    assert replacement.other_position_max_abs_delta == 0.0
    assert torch.equal(first[0, 1], torch.full((3,), 9.0))
    assert torch.equal(first[0, 0], torch.ones(3))
    assert torch.equal(second, torch.ones_like(second))


def test_owner_gate_requires_accepted_competitor_and_support_margins() -> None:
    parsed = {"valid": True}
    assert probe.assess_owner_match(
        parsed,
        donor_annotation_id=probe.TARGET_ANNOTATION_ID,
        attribution=_owner_attribution(),
    )["passed"]

    accepted_margin = _owner_attribution(donor_iou=0.8)
    accepted_margin["strongest_competing_iou"] = 0.650001
    assert not probe.assess_owner_match(
        parsed,
        donor_annotation_id=probe.TARGET_ANNOTATION_ID,
        attribution=accepted_margin,
    )["passed"]

    support_iou = _owner_attribution(donor_iou=0.8)
    support_iou["support_envelope_donor_iou"] = 0.750001
    assert not probe.assess_owner_match(
        parsed,
        donor_annotation_id=probe.TARGET_ANNOTATION_ID,
        attribution=support_iou,
    )["passed"]

    support_l1 = _owner_attribution(donor_iou=0.8)
    support_l1["donor_l1_distance"] = 0.030001
    assert not probe.assess_owner_match(
        parsed,
        donor_annotation_id=probe.TARGET_ANNOTATION_ID,
        attribution=support_l1,
    )["passed"]


def test_accepted_ledger_and_generated_owner_are_normalized_and_unique() -> None:
    objects = probe.build_accepted_objects(
        [
            {
                "object_identifier": f"coco:{probe.TARGET_ANNOTATION_ID}",
                "source_canvas_box_xyxy": [10, 20, 30, 40],
            },
            {
                "object_identifier": f"coco:{probe.PAIRED_ANNOTATION_ID}",
                "source_canvas_box_xyxy": [60, 60, 80, 80],
            },
        ],
        width=100,
        height=100,
    )
    assert objects[0]["normalized_box"] == [0.1, 0.2, 0.3, 0.4]
    assert probe.generated_owner(
        {"valid": True, "normalized_box": [0.1, 0.2, 0.3, 0.4]}, objects
    ) == probe.TARGET_ANNOTATION_ID


def test_classification_distinguishes_specificity_confidence_and_veto() -> None:
    confidence = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=[],
        unrestricted_baseline_owner=probe.TARGET_ANNOTATION_ID,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": None,
        },
    )
    assert confidence["classification"] == "bounded_donor_path_confidence_portability"
    assert not confidence["owner_specific_geometry_portability"]

    switched = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["paired"],
        layer13_passed_donors=[],
        unrestricted_baseline_owner=probe.TARGET_ANNOTATION_ID,
        layer23_generated_owner_by_donor={
            "target": None,
            "paired": probe.PAIRED_ANNOTATION_ID,
        },
    )
    assert switched["classification"] == "bounded_owner_specific_geometry_state_portability"
    assert switched["owner_switch_from_unrestricted_baseline"]

    unowned_baseline = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=[],
        unrestricted_baseline_owner=None,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": None,
        },
    )
    assert unowned_baseline["classification"] == (
        "bounded_owner_specific_geometry_state_portability"
    )
    assert unowned_baseline["owner_switch_from_unrestricted_baseline"]

    both = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target", "paired"],
        layer13_passed_donors=[],
        unrestricted_baseline_owner=None,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": probe.PAIRED_ANNOTATION_ID,
        },
    )
    assert both["both_donor_states_recover_distinct_intended_owners"]
    assert both["owner_specific_geometry_portability"]

    vetoed = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=["target"],
        unrestricted_baseline_owner=probe.PAIRED_ANNOTATION_ID,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": None,
        },
    )
    assert vetoed["classification"] == "vetoed_by_negative_control_layer_13"
    assert vetoed["corresponding_donor_veto_intersection"] == ["target"]

    disjoint = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target"],
        layer13_passed_donors=["paired"],
        unrestricted_baseline_owner=probe.PAIRED_ANNOTATION_ID,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": None,
        },
    )
    assert disjoint["classification"] == (
        "bounded_owner_specific_geometry_state_portability"
    )
    assert disjoint["corresponding_donor_veto_intersection"] == []

    mixed = probe.classify_geometry_panel(
        trust_passed=True,
        eligible_count=2,
        layer23_passed_donors=["target", "paired"],
        layer13_passed_donors=["target"],
        unrestricted_baseline_owner=probe.TARGET_ANNOTATION_ID,
        layer23_generated_owner_by_donor={
            "target": probe.TARGET_ANNOTATION_ID,
            "paired": probe.PAIRED_ANNOTATION_ID,
        },
    )
    assert mixed["localized_layer23_passed_donors"] == ["paired"]
    assert mixed["corresponding_donor_veto_intersection"] == ["target"]
    assert mixed["classification"] == (
        "bounded_owner_specific_geometry_state_portability"
    )


def test_parent_reproduction_gate_requires_both_modes_for_both_donors() -> None:
    persistent = {
        "owners": {
            owner: {
                "parent_hard_reproduction": {"passed": True},
                "parent_unrestricted_reproduction": {"passed": True},
            }
            for owner in ("target", "paired")
        }
    }
    assert probe.persistent_parent_reproductions_passed(persistent)
    persistent["owners"]["paired"]["parent_unrestricted_reproduction"]["passed"] = False
    assert not probe.persistent_parent_reproductions_passed(persistent)


def test_parser_contains_each_frozen_argument_once(tmp_path: Path) -> None:
    parser = probe.build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    assert args.layers == [23, 13]
    assert args.max_new_tokens == 5
