"""Pure contracts for the human-refined completion-condition runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/research/run_human_refined_greedy_set_completion_conditions.py"
SPEC = importlib.util.spec_from_file_location("human_refined_completion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows() -> list[dict[str, object]]:
    return [
        {"owner_id": "a", "category": "person", "description": "person", "bbox": [0, 0, 10, 10]},
        {"owner_id": "b", "category": "car", "description": "car", "bbox": [20, 0, 30, 10]},
        {"owner_id": "c", "category": "person", "description": "person", "bbox": [0, 20, 10, 30]},
        {"owner_id": "d", "category": "chair", "description": "chair", "bbox": [20, 20, 30, 30]},
        {"owner_id": "e", "category": "person", "description": "person", "bbox": [40, 20, 50, 30]},
    ]


def test_policy_schedules_are_deterministic_and_suffixes_are_schedule_specific() -> None:
    first = MODULE.build_policy_schedules(_rows(), random_seeds=(7,), image_id="1584")
    second = MODULE.build_policy_schedules(_rows(), random_seeds=(7,), image_id="1584")
    assert first == second
    assert first["geometry"] == ["a", "b", "c", "d", "e"]
    assert first["reverse"] == ["e", "d", "c", "b", "a"]
    assert first["category"] == ["b", "d", "a", "c", "e"]
    assert first["random:7"] != first["geometry"]


def test_same_remaining_set_freezes_geometry_suffix_and_common_final_forced_row() -> None:
    controls = MODULE.build_same_remaining_set_schedules(_rows(), 2, random_seeds=(7,), image_id="1584")
    geometry = controls["same_remaining:geometry"]
    for control in controls.values():
        assert control["remaining_owner_ids"] == ["d", "e"]
        assert set(control["forced_owner_ids"]) == {"a", "b", "c"}
        assert control["forced_owner_ids"][-1] == geometry["forced_owner_ids"][-1]
        assert control["order"][-2:] == ["d", "e"]


def test_depth_construction_skips_invalid_values_and_resolves_N() -> None:
    assert MODULE.build_remaining_depths(5) == (5, 4, 2, 1)
    assert MODULE.build_remaining_depths(5, "N,16,8,4,2,1") == (5, 4, 2, 1)
    assert MODULE.build_remaining_depths(5, (16, 8, 4)) == (4,)
    with pytest.raises(ValueError):
        MODULE.build_remaining_depths(5, (16, 8))


def test_budget_formula_is_exact() -> None:
    assert MODULE.complete_row_budget(1, "strict") == 1
    assert MODULE.complete_row_budget(1, "relaxed") == 5
    assert MODULE.complete_row_budget(16, "relaxed") == 20


def test_terminal_suppression_state_transitions_native_first_only_and_full() -> None:
    torch = pytest.importorskip("torch")

    row = [
        MODULE.OBJECT_REF_START_TOKEN_ID,
        100,
        MODULE.OBJECT_REF_END_TOKEN_ID,
        MODULE.BOX_START_TOKEN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.BOX_END_TOKEN_ID,
    ]

    def call(processor, generated: list[int], scores):
        input_ids = torch.tensor([[99, *generated]], dtype=torch.long)
        return processor(input_ids, scores)

    # Empty suffix is a row boundary.  Terminal id 4 is raw top-1 and id 8 is
    # the strongest non-terminal choice.
    native = MODULE.RowBoundaryTerminalSuppressor(
        prompt_width=1, im_end_token_id=4, box_end_token_id=3, condition="native", row_budget=2
    )
    native_scores = torch.tensor([[0.0, 1.0, 2.0, 0.0, 9.0]])
    assert call(native, [], native_scores.clone())[0, 4].item() == 9.0
    assert native.override_receipts == []

    first = MODULE.RowBoundaryTerminalSuppressor(
        prompt_width=1, im_end_token_id=4, box_end_token_id=3, condition="first_only", row_budget=2
    )
    first_scores = torch.tensor([[0.0, 1.0, 2.0, 0.0, 9.0]])
    masked = call(first, [], first_scores.clone())
    assert masked[0, 4].item() < -1e20
    assert len(first.override_receipts) == 1
    # At the next boundary, native stopping is restored.
    restored = call(first, row, first_scores.clone())
    assert restored[0, 4].item() == 9.0
    assert len(first.override_receipts) == 1

    full = MODULE.RowBoundaryTerminalSuppressor(
        prompt_width=1, im_end_token_id=4, box_end_token_id=MODULE.BOX_END_TOKEN_ID, condition="full", row_budget=2
    )
    assert call(full, [], first_scores.clone())[0, 4].item() < -1e20
    assert call(full, row, first_scores.clone())[0, 4].item() < -1e20
    assert call(full, [*row, *row], first_scores.clone())[0, 4].item() == 9.0
    assert len(full.override_receipts) == 2
    assert full.override_receipts[0]["raw_terminal_minus_selected_margin_fp32"] == pytest.approx(7.0)
    assert all(item["raw_terminal_top1"] for item in full.override_receipts)


def test_row_budget_stopping_criterion_and_termination_classification() -> None:
    torch = pytest.importorskip("torch")
    row = [
        MODULE.OBJECT_REF_START_TOKEN_ID,
        100,
        MODULE.OBJECT_REF_END_TOKEN_ID,
        MODULE.BOX_START_TOKEN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.BOX_END_TOKEN_ID,
    ]
    stopping = MODULE.RowBudgetStoppingCriteria(prompt_width=1, row_budget=2, box_end_token_id=MODULE.BOX_END_TOKEN_ID)
    assert bool(stopping(torch.tensor([[99, *row]], dtype=torch.long))[0]) is False
    assert bool(stopping(torch.tensor([[99, *row, *row]], dtype=torch.long))[0]) is True
    assert stopping.stop_reason == "row_budget"
    assert MODULE.classify_termination(
        generated_token_ids=[1, 2], complete_row_count=0, row_budget=2, im_end_token_id=9, max_new_tokens=2
    ) == "token_limit"
    assert MODULE.classify_termination(
        generated_token_ids=[3, 3], complete_row_count=2, row_budget=2, im_end_token_id=9
    ) == "row_budget"
    assert MODULE.classify_termination(
        generated_token_ids=[9], complete_row_count=0, row_budget=2, im_end_token_id=9
    ) == "native_im_end"
    assert MODULE.termination_is_valid("row_budget")
    assert not MODULE.termination_is_valid("token_limit")


def test_structural_counter_ignores_bare_box_end_and_malformed_tail() -> None:
    torch = pytest.importorskip("torch")
    valid_row = [
        MODULE.OBJECT_REF_START_TOKEN_ID,
        100,
        MODULE.OBJECT_REF_END_TOKEN_ID,
        MODULE.BOX_START_TOKEN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.COORDINATE_TOKEN_MIN_ID + 1,
        MODULE.BOX_END_TOKEN_ID,
    ]
    assert MODULE.count_complete_rows_from_token_ids([MODULE.BOX_END_TOKEN_ID]) == 0
    assert MODULE.count_complete_rows_from_token_ids([*valid_row, MODULE.BOX_END_TOKEN_ID]) == 1
    stopping = MODULE.RowBudgetStoppingCriteria(prompt_width=1, row_budget=2, box_end_token_id=MODULE.BOX_END_TOKEN_ID)
    assert bool(stopping(torch.tensor([[99, MODULE.BOX_END_TOKEN_ID]], dtype=torch.long))[0]) is False
    assert bool(stopping(torch.tensor([[99, *valid_row, MODULE.BOX_END_TOKEN_ID]], dtype=torch.long))[0]) is False
    assert stopping.complete_row_count == 1


def test_parser_receipt_preserves_drops_and_normalizes_from_coord_bins() -> None:
    text = (
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_0|><|coord_0|><|coord_10|><|coord_10|><|box_end|>"
        "<|box_end|>"
    )
    evidence = MODULE._parse_generated_text(text, image_width=100, image_height=100, row_id="test")
    assert evidence["parse_status"] == evidence["parser_artifact"]["parse_status"]
    assert evidence["dropped_prediction_count"] == len(evidence["dropped_predictions"])
    assert evidence["dropped_prediction_count"] >= 1
    assert evidence["predictions"][0]["bbox"] == [0, 0, 10, 10]
    assert evidence["predictions"][0]["bbox_pixel"] == [0.0, 0.0, 1.0, 1.0]


def test_valid_row_after_malformed_candidate_is_retained_at_budget_one() -> None:
    text = (
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_0|><|coord_0|><|box_end|>"
        "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        "<|coord_0|><|coord_0|><|coord_10|><|coord_10|><|box_end|>"
    )
    evidence = MODULE._parse_generated_text(text, image_width=100, image_height=100, row_id="test")
    assert evidence["dropped_prediction_count"] == 1
    assert len(evidence["predictions"]) == 1
    # generated_order reflects the malformed candidate before this valid row;
    # it is diagnostic metadata, not a complete-row budget index.
    assert evidence["predictions"][0]["generated_row_index"] == 1


def test_forced_repeat_partition_never_uses_total_owner_credit() -> None:
    partition = MODULE.partition_owner_matches(
        {"matched_owner_ids": ["forced", "remaining", "other"]},
        remaining_owner_ids=["remaining", "missing"],
        prefix_owner_ids=["forced"],
    )
    assert partition["remaining_owner_ids_discovered"] == ["remaining"]
    assert partition["forced_context_owner_ids_repeated"] == ["forced"]
    assert partition["remaining_conservative_lower_bound_coverage"] == 1
    assert partition["remaining_coverage_fraction"] == pytest.approx(0.5)
    assert partition["completion_flag"] is False


def test_generation_call_key_and_id_are_exact_dedup_receipts() -> None:
    first = MODULE.generation_call_key([1, 2, 3], "first_only", 4)
    second = MODULE.generation_call_key((1, 2, 3), "first-only", 4)
    assert first == second
    assert MODULE.generation_call_id(first) == MODULE.generation_call_id(second)
    assert MODULE.generation_call_key([1, 2, 4], "first_only", 4) != first


def test_global_assignment_prevents_duplicate_owner_credit_and_keeps_ambiguity_for_review() -> None:
    owners = [
        {"owner_id": "person-1", "category": "person", "bbox": [0, 0, 10, 10]},
        {"owner_id": "person-2", "category": "person", "bbox": [20, 0, 30, 10]},
    ]
    predictions = [
        {"prediction_id": "p1", "generated_row_index": 0, "category": "person", "bbox": [0, 0, 10, 10]},
        {"prediction_id": "p2", "generated_row_index": 1, "category": "person", "bbox": [0, 0, 10, 10]},
    ]
    result = MODULE.match_predictions_one_to_one(predictions, owners)
    assert result["conservative_lower_bound_coverage"] == 1
    assert result["matched_owner_ids"] == ["person-1"]
    assert len(result["duplicate_candidates"]) == 1
    assert len(result["matches"]) == 1
