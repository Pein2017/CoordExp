from probes.training_set_completion.readback_selectors import (
    flatten_raw_rows,
    one_to_one_matches,
    pairwise_iou95,
    termination_metrics,
    validate_unique_image_rows,
)
import pytest


def test_matching_is_class_agnostic_and_one_to_one():
    references = [
        {"owner_id": "a", "reference_coord_bins_1000": [0, 0, 100, 100]},
        {"owner_id": "b", "reference_coord_bins_1000": [200, 200, 300, 300]},
    ]
    predictions = [
        {"prediction_id": "p0", "generated_order": 0, "coord_bins_1000": [0, 0, 100, 100]},
        {"prediction_id": "p1", "generated_order": 1, "coord_bins_1000": [0, 0, 100, 100]},
        {"prediction_id": "p2", "generated_order": 2, "coord_bins_1000": [200, 200, 300, 300]},
    ]
    matches = one_to_one_matches(references, predictions, 0.8)
    assert [(item["reference_owner_id"], item["prediction_id"]) for item in matches] == [("a", "p0"), ("b", "p2")]


def test_pairwise_selector_does_not_label_physical_repeat():
    predictions = [
        {"prediction_id": "p0", "generated_order": 0, "coord_bins_1000": [0, 0, 100, 100]},
        {"prediction_id": "p1", "generated_order": 1, "coord_bins_1000": [0, 0, 100, 100]},
    ]
    pairs = pairwise_iou95(predictions)
    assert [(item["left_prediction_id"], item["right_prediction_id"]) for item in pairs] == [("p0", "p1")]
    assert "physical" not in pairs[0]


def test_parser_drops_remain_in_raw_rows():
    valid, dropped = flatten_raw_rows({
        "pred": [{"generated_order": 0, "description": "cup", "coord_bins": [1, 2, 10, 20], "bbox": [2, 3, 20, 30]}],
        "dropped_predictions": [{"generated_order": 1, "reason": "geometry_invalid", "code": "data.bbox_order", "context": {"bbox": [4, 5, 4, 8]} }],
    })
    assert len(valid) == 1 and len(dropped) == 1
    assert dropped[0]["status"] == "parser_dropped"
    assert dropped[0]["coord_bins_1000"] == [4, 5, 4, 8]


def test_exact_cap_length_stop_is_cap_debt_but_natural_eos_can_end_at_cap():
    capped = termination_metrics(3, [1, 2, 3], "length", cap=3)
    assert capped["capped"] and capped["cap_debt"] == 1 and capped["eos_debt"]
    natural = termination_metrics(3, [1, 2, 151645], "im_end", cap=3)
    assert natural["natural_eos"] and not natural["capped"] and natural["cap_debt"] == 0
    forced = termination_metrics(3, [1, 2, 151645], "forced_eos", cap=3)
    assert not forced["natural_eos"] and forced["eos_debt"]


def test_duplicate_readback_images_fail_closed():
    with pytest.raises(ValueError, match="duplicate readback image rows"):
        validate_unique_image_rows([{"image_id": 210457}, {"image_id": 210457}])
