from __future__ import annotations

import copy

import pytest

from probes.training_set_completion import paired_evaluation as prior_eval
from probes.training_set_completion.coco227_evaluation import (
    OWNER_COUNTS,
    _joint_partition,
    _validate_saved_rows,
    compare,
    sustained_clean_milestone,
)


def _target(owner: str, box: list[int]) -> dict:
    return {"image_id": 1, "owner_id": owner, "reference_coord_bins_1000": box, "description": "person", "class_status": "verified_coco80"}


def _prediction(name: str, box: list[int]) -> dict:
    return {"prediction_id": name, "generated_order": int(name[1:]), "coord_bins_1000": box, "description": "person", "raw_span_sha256": name}


def test_old_and_new_use_one_joint_assignment_not_two_subset_matchers() -> None:
    targets = [_target("old", [0, 0, 10, 10]), _target("new", [0, 0, 10, 10])]
    predictions = [_prediction("p0", [0, 0, 10, 10])]
    joint = prior_eval._ledger_image(targets, predictions, threshold=0.5)
    old = _joint_partition(joint=joint, targets=targets, predictions=predictions, owner_ids={"old"}, name="old218")
    new = _joint_partition(joint=joint, targets=targets, predictions=predictions, owner_ids={"new"}, name="new9")

    assert joint["matched_count"] == 1
    assert old["matched_count"] + new["matched_count"] == joint["matched_count"]
    assert old["matching_basis"] == "projection_of_joint_scoped227_iou_0_5_assignment"


def test_saved_endpoint_validation_rejects_wrong_prompt_even_with_valid_token_hash() -> None:
    route = {"prompt_token_ids": [10], "image_identity": {"executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 3]}}
    row = {
        "arm": "S", "step": 8, "checkpoint_step": 8,
        "empty_assistant_prefix": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": 3084,
        "generated_token_ids": [151645], "generated_token_ids_sha256": prior_eval.digest([151645]), "decode_stop_reason": "im_end",
        "prompt_token_ids": [10], "executed_media_sha256": "media", "observed_image_grid_thw": [1, 2, 3],
    }
    routes = {image_id: route for image_id in range(1, 12)}
    rows = [{**row, "image_id": image_id, "prompt_token_ids": [999] if image_id == 1 else [10]} for image_id in range(1, 12)]
    with pytest.raises(ValueError, match="saved prompt identity"):
        _validate_saved_rows(rows, routes=routes, arm="S", step=8)


def _score(step: int, clean: bool) -> dict:
    return {"schema": "training_set_completion.coco227_ce_normalization_evaluation.v1", "arm": "S", "step": step, "clean_completion": {"clean_complete": clean}}


def test_sustained_clean_milestone_requires_all_later_saved_points() -> None:
    scores = [_score(8, True), _score(16, True), _score(32, False), _score(64, True), _score(128, True), _score(256, True)]
    assert sustained_clean_milestone(scores, arm="S")["earliest_sustained_clean_milestone"] == 64


def test_compare_uses_image_qualified_owner_sets_for_every_ledger() -> None:
    def score(owner: str) -> dict:
        return {
            "schema": "training_set_completion.coco227_ce_normalization_evaluation.v1",
            "sources": {"preparation": {"path": "p", "sha256": "p", "size_bytes": 1}, "readback_admission": {}},
            "matching": {"primary": "joint"}, "metric_contract": {"primary": "joint"},
            "per_image": [{"image_id": 1, "ledgers_iou_0_5": {name: {"covered_owner_ids": [owner]} for name in OWNER_COUNTS}}],
        }
    result = compare(baseline=score("same-id"), endpoint=score("next-id"), label="comparison")
    assert result["per_ledger"]["old218"]["lost"] == [{"image_id": 1, "owner_id": "same-id"}]
    assert result["per_ledger"]["new9"]["gained"] == [{"image_id": 1, "owner_id": "next-id"}]
