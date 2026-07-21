from __future__ import annotations

import pytest

from scripts.research.run_greedy_prefix_forced_owner_path import (
    BOX_END,
    BOX_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    compare_noop_parity,
    _covered_after_row,
    _summary,
    derive_intervention_rungs,
    first_divergence_index,
    fixed_budget_plan,
    require_target_not_covered,
    semantic_endpoint_role,
    split_generated_rows,
    target_already_covered,
)


def _row(*, description: int = 8987, coords: tuple[int, int, int, int] = (151700, 151701, 151702, 151703)) -> list[int]:
    return [OBJECT_REF_START, description, OBJECT_REF_END, BOX_START, *coords, BOX_END]


def test_rows_split_only_at_box_end_and_incomplete_or_cross_row_is_rejected() -> None:
    row = _row()
    assert split_generated_rows(row + row) == [row, row]

    with pytest.raises(ValueError, match="incomplete"):
        split_generated_rows(row[:-1])
    with pytest.raises(ValueError, match="cross-row"):
        split_generated_rows(row[:4] + [OBJECT_REF_START] + row[4:])


def test_first_divergence_and_endpoint_roles_follow_grammar() -> None:
    native = _row(coords=(151700, 151701, 151702, 151703))
    donor = _row(coords=(151700, 151701, 151710, 151711))
    assert first_divergence_index(native, donor) == 6
    assert semantic_endpoint_role(donor[:1]) == "row_opener"
    assert semantic_endpoint_role(donor[:2]) == "description"
    assert semantic_endpoint_role(donor[:3]) == "description_end"
    assert semantic_endpoint_role(donor[:4]) == "box_start"
    assert semantic_endpoint_role(donor[:5]) == "x1"
    assert semantic_endpoint_role(donor[:6]) == "y1"
    assert semantic_endpoint_role(donor[:7]) == "x2"
    assert semantic_endpoint_role(donor[:8]) == "y2"
    assert semantic_endpoint_role(donor) == "box_end"

    rungs = derive_intervention_rungs(native, donor)
    assert [rung["rung_name"] for rung in rungs] == ["native", "prefix_7", "prefix_8", "full_row"]
    assert [rung["endpoint_role"] for rung in rungs[1:]] == ["x2", "y2", "box_end"]
    assert len({tuple(rung["forced_row_prefix_token_ids"]) for rung in rungs}) == len(rungs)


def test_fixed_budget_and_stop_native_force_all_donor_positions() -> None:
    assert fixed_budget_plan(3, 8)["remaining_suffix_horizon"] == 4
    with pytest.raises(ValueError):
        fixed_budget_plan(8, 8)

    donor = _row()
    rungs = derive_intervention_rungs([], donor, native_stop=True)
    assert rungs[0]["endpoint_role"] == "native"
    assert rungs[1]["endpoint_token_count"] == 1
    assert rungs[-1]["rung_name"] == "full_row"
    assert rungs[-1]["endpoint_role"] == "box_end"


def test_target_already_covered_refusal_accepts_qualified_and_local_ids() -> None:
    assert target_already_covered("5001:1329465", ["1329465"], image_id=5001)
    with pytest.raises(ValueError, match="already covered"):
        require_target_not_covered("5001:1329465", ["5001:1329465"], image_id=5001)
    require_target_not_covered("5001:1329465", ["1329466"], image_id=5001)


def test_noop_comparison_is_pure_and_checks_suffix() -> None:
    native = {"raw_generated_token_ids": _row()}
    same = {"raw_generated_token_ids": _row()}
    suffix = [{"raw_generated_token_ids": _row()}]
    assert compare_noop_parity(native, [same], native_suffix=suffix, forced_suffixes=[suffix])["passed"]
    different = {"raw_generated_token_ids": _row(coords=(151700, 151701, 151702, 151704))}
    assert not compare_noop_parity(native, [different])["passed"]


def test_reviewed_owner_resolves_entity_without_relabelling_automatic_match_and_terminal_is_not_malformed() -> None:
    reviewed = {
        "row_index": 4,
        "status": "success",
        "accepted_complete_row": True,
        "row_stop": {"stop_reason": "complete_row"},
        "strict_matched_owner_ids": [],
        "reviewed_owner_ids": ["-169"],
        "reviewed_owner_ids_qualified": ["7511:-169"],
        "unmatched_or_ambiguous_prediction_indices": [0],
    }
    covered, receipt = _covered_after_row(["565324"], reviewed)
    assert covered == ["-169", "565324"]
    assert receipt["source"] == "frozen_human_crop_review"

    terminal = {
        "row_index": 5,
        "status": "success",
        "accepted_complete_row": False,
        "row_stop": {"stop_reason": "terminal"},
        "strict_matched_owner_ids": [],
    }
    result = _summary(
        [reviewed, terminal],
        target_owner_id="7511:-169",
        image_id=7511,
        covered_parent=["565324"],
    )
    assert result["target_owner_acquired"]
    assert not result["unresolved"]
    assert result["automatic_unresolved_present"]
    assert not result["malformed"]
