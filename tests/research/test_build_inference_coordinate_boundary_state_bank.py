from scripts.research.build_inference_coordinate_boundary_state_bank import (
    first_wrong_coordinate,
    split_complete_rows,
    unique_owner_match,
)


def test_split_complete_rows_keeps_only_strict_complete_rows() -> None:
    row = [151646, 42, 151647, 151648, 151671, 151672, 151673, 151674, 151649]
    assert split_complete_rows([9, *row, 10, 151646, 11]) == [row]


def test_first_wrong_coordinate_requires_accepted_earlier_prefix() -> None:
    result = first_wrong_coordinate([101, 205, 300, 400], [100, 200, 300, 400], tolerance=3)
    assert result is not None
    index, observations = result
    assert index == 1
    assert [item["coordinate"] for item in observations] == ["x1", "y1"]


def test_unique_owner_match_refuses_a_close_second_instance() -> None:
    prediction = {
        "description": "person",
        "bbox": [0, 0, 128, 80],
        "coord_bins": [0, 0, 100, 100],
    }
    clear = [
        {"description": "person", "bbox": [0, 0, 100, 100], "object_id": "a"},
        {"description": "person", "bbox": [500, 500, 600, 600], "object_id": "b"},
    ]
    assert unique_owner_match(prediction, clear, minimum_iou=0.55, minimum_margin=0.30)[0]["object_id"] == "a"
    ambiguous = [
        {"description": "person", "bbox": [0, 0, 100, 100], "object_id": "a"},
        {"description": "person", "bbox": [5, 5, 105, 105], "object_id": "b"},
    ]
    assert unique_owner_match(prediction, ambiguous, minimum_iou=0.55, minimum_margin=0.30) is None


def test_unique_owner_match_uses_coord_bins_not_pixel_bbox() -> None:
    prediction = {
        "description": "cow",
        "bbox": [709, 337, 805, 425],
        "coord_bins": [568, 405, 645, 511],
    }
    owner = {"description": "cow", "bbox": [567, 402, 643, 511]}
    distractor = {"description": "cow", "bbox": [100, 100, 200, 200]}

    match = unique_owner_match(
        prediction,
        [owner, distractor],
        minimum_iou=0.35,
        minimum_margin=0.15,
    )

    assert match is not None
    assert match[0] is owner
    assert match[1] > 0.9
