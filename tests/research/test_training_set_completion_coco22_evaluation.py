from __future__ import annotations

import copy
import json

import pytest

from probes.training_set_completion import coco22_evaluation as e


def target(owner: str, box: list[int], image: int = 1) -> dict:
    return {"image_id": image, "owner_id": owner, "reference_coord_bins_1000": box,
            "description": "person", "class_status": "verified_coco80"}


def prediction(index: int, box: list[int], description: str = "person") -> dict:
    return {"prediction_id": f"p{index}", "generated_order": index, "coord_bins_1000": box,
            "description": description, "raw_span_sha256": f"hash-{index}"}


def ledgers() -> dict:
    old = target("old", [10, 10, 110, 110]); new = target("new", [250, 250, 350, 350])
    return {"frozen": [old, new], "old227": [old], "new_cohort": [new],
            "old218": [old], "prior_new9": [], "current_known": [old, new]}


def test_verified_extra_retains_completion_and_annotation_f1_denominator() -> None:
    rows = [prediction(0, [10, 10, 110, 110]), prediction(1, [250, 250, 350, 350]),
            prediction(2, [500, 500, 600, 600])]
    current = copy.deepcopy(ledgers())
    current["current_known"].append(target("extra", [500, 500, 600, 600]))
    result = e.score_parsed_image(1, current, rows, [], "im_end", [1, 151645])
    assert result["frozen_task_completion"]["complete"] is True
    assert result["complete_output_review"]["closed"] is True
    assert result["annotation_relative_micro"]["tp"] == 2
    assert result["annotation_relative_micro"]["prediction_denominator_all_valid_parsed_rows"] == 3
    assert result["annotation_relative_micro"]["f1"] == pytest.approx(4 / 5)
    assert result["ledgers_iou_0_5"]["old227"]["matched_count"] == 1
    assert result["ledgers_iou_0_5"]["new_cohort"]["matched_count"] == 1


def test_unknown_is_neutral_for_confirmed_fp_but_review_stays_open() -> None:
    rows = [prediction(0, [10, 10, 110, 110]), prediction(1, [250, 250, 350, 350]),
            prediction(2, [500, 500, 600, 600])]
    result = e.score_parsed_image(1, ledgers(), rows, [], "im_end", [1, 151645])
    assert result["frozen_task_completion"]["complete"] is True
    assert result["complete_output_review"]["closed"] is False
    assert result["complete_output_review"]["failure_counts"]["unresolved_owner_or_category"] == 1
    assert result["complete_output_review"]["failure_counts"].get("confirmed_false_positive", 0) == 0


def test_wrong_class_geometry_still_matches_but_completion_and_review_fail() -> None:
    rows = [prediction(0, [10, 10, 110, 110], "cat"), prediction(1, [250, 250, 350, 350])]
    result = e.score_parsed_image(1, ledgers(), rows, [], "im_end", [1, 151645])
    assert result["ledgers_iou_0_5"]["frozen"]["matched_count"] == 2
    assert result["frozen_task_completion"]["complete"] is False
    assert result["complete_output_review"]["closed"] is False
    assert result["ledgers_iou_0_5"]["frozen"]["class_wrong_count"] == 1


def test_duplicate_and_dropped_and_cap_fail_even_when_frozen_covered() -> None:
    rows = [prediction(0, [10, 10, 110, 110]), prediction(1, [250, 250, 350, 350]),
            prediction(2, [11, 11, 109, 109])]
    result = e.score_parsed_image(1, ledgers(), rows,
                                  [{"drop_reason": "malformed", "prediction_id": "p3"}],
                                  "length", [1] * e.CAP)
    assert result["frozen_task_completion"]["complete"] is True
    failure = result["complete_output_review"]["failure_counts"]
    assert failure["duplicate_candidate_pairs"] > 0
    assert failure["malformed_or_invalid_geometry"] == 1
    assert failure["eos_or_cap"] > 0


def test_partition_projects_one_joint_assignment_even_when_targets_overlap() -> None:
    old = target("old", [10, 10, 110, 110]); new = target("new", [10, 10, 110, 110])
    ledger = {"frozen": [old, new], "old227": [old], "new_cohort": [new],
              "old218": [old], "prior_new9": [], "current_known": [old, new]}
    result = e.score_parsed_image(1, ledger, [prediction(0, [10, 10, 110, 110])], [], "im_end", [1, 151645])
    primary = result["ledgers_iou_0_5"]
    assert primary["frozen"]["matched_count"] == 1
    assert primary["old227"]["matched_count"] + primary["new_cohort"]["matched_count"] == 1


def test_unreviewed_claim_cannot_promote_an_unmatched_owner() -> None:
    rows = [prediction(0, [10, 10, 110, 110]), prediction(1, [250, 250, 350, 350]),
            prediction(2, [500, 500, 600, 600])]
    with pytest.raises(ValueError, match="lead-admitted"):
        e.score_parsed_image(1, ledgers(), rows, [], "im_end", [1, 151645],
                             review_decisions={"hash-2": {"status": "verified_extra_real"}})


@pytest.mark.parametrize("kind", ["toy_or_figurine", "depicted_in_wall_photo"])
def test_individual_out_of_scope_review_blocks_extra_owner_claim(kind) -> None:
    rows = [prediction(0, [10, 10, 110, 110]), prediction(1, [250, 250, 350, 350]),
            prediction(2, [500, 500, 600, 600])]
    result = e.score_parsed_image(1, ledgers(), rows, [], "im_end", [1, 151645],
        review_decisions={"hash-2": {"status": "lead-accepted", "decision": "out_of_scope",
                                     "review_category": kind}})
    assert result["frozen_task_completion"]["complete"] is True
    assert result["complete_output_review"]["closed"] is False
    assert result["complete_output_review"]["failure_counts"]["out_of_scope"] == 1
    assert result["physical"]["rows"][2]["review_category"] == kind


def test_rejected_raw_gt_car_on_person_face_never_enters_accepted_physical_ledger(tmp_path) -> None:
    raw_car = {"coco_ann_id": 2039788, "desc": "car", "bbox_2d":
               ["<|coord_400|>", "<|coord_400|>", "<|coord_450|>", "<|coord_450|>"]}
    person = {"coco_ann_id": 2039789, "desc": "person", "bbox_2d":
              ["<|coord_100|>", "<|coord_100|>", "<|coord_500|>", "<|coord_900|>"]}
    raw = {"image_id": 196090, "objects": [raw_car, person], "unlabeled": [], "keep_source": {"untouched": True}}
    old = {"image_id": 1, "objects": [{"coco_ann_id": 10, "desc": "person", "bbox_2d":
           ["<|coord_1|>", "<|coord_1|>", "<|coord_10|>", "<|coord_10|>"]}], "unlabeled": []}
    before = copy.deepcopy(raw)
    evidence = {}
    for name in ("original", "overlay", "crop"):
        path = tmp_path / f"{name}.jpg"; path.write_bytes(b"original review image")
        evidence[name] = str(path)
    admission = {"status": "lead-accepted", "gt_owners": [
        {"image_id": 196090, "coco_ann_id": 2039788, "decision": "reject_false_gt",
         "category": "car", "bbox_2d": raw_car["bbox_2d"], "evidence": evidence},
        {"image_id": 196090, "coco_ann_id": 2039789, "decision": "verified_real_gt",
         "category": "person", "bbox_2d": person["bbox_2d"], "evidence": evidence}]}
    gt_reviews = e._gt_review_map(admission, [old, raw], new_image_ids={196090})
    physical = e._annotation_ledger([old, raw], new_image_ids={196090}, gt_reviews=gt_reviews)
    assert {(row["image_id"], row["owner_id"]) for row in physical} == {
        (1, "10"), (196090, "2039789")}
    assert raw == before
    assert e.score_parsed_image(196090,
        {"frozen": [], "old227": [], "new_cohort": [], "old218": [],
         "prior_new9": [], "current_known": [row for row in physical if row["image_id"] == 196090]},
        [prediction(1, [400, 400, 450, 450], "car")], [], "im_end", [1, 151645]
    )["complete_output_review"]["closed"] is False


def test_missing_gt_review_or_unaccepted_sidecar_does_not_promote_raw_owner(tmp_path) -> None:
    raw = {"image_id": 2, "objects": [{"coco_ann_id": 20, "desc": "person", "bbox_2d":
           ["<|coord_1|>", "<|coord_1|>", "<|coord_9|>", "<|coord_9|>"]}], "unlabeled": []}
    with pytest.raises(ValueError, match="lead-accepted"):
        e._gt_review_map({"status": "candidate", "gt_owners": []}, [raw], new_image_ids={2})
    with pytest.raises(ValueError, match="every new raw GT"):
        e._gt_review_map({"status": "lead-accepted", "gt_owners": []}, [raw], new_image_ids={2})
