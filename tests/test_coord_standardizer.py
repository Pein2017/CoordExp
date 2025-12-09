from __future__ import annotations

from src.common.coord_standardizer import CoordinateStandardizer


def test_standardize_gt_tokens_to_pixel_text():
    standardizer = CoordinateStandardizer("coord")
    errors: list[str] = []
    record = {
        "width": 64,
        "height": 48,
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_0|>",
                    "<|coord_0|>",
                    "<|coord_999|>",
                    "<|coord_999|>",
                ],
                "desc": "box",
            }
        ],
    }
    objs = standardizer.process_record_gt(record, width=64, height=48, errors=errors)
    assert errors == []
    assert len(objs) == 1
    assert objs[0]["points"] == [0, 0, 63, 47]
    assert objs[0]["points_text"] == "0 0 63 47"
    assert objs[0]["_coord_mode"] == "norm1000"


def test_standardize_pred_tokens_in_text_mode():
    standardizer = CoordinateStandardizer("text")
    errors: list[str] = []
    raw_text = (
        '{"obj":{"bbox_2d":["<|coord_0|>","<|coord_0|>",'
        '"<|coord_999|>","<|coord_999|>"],"desc":"car"}}'
    )
    preds = standardizer.process_prediction_text(
        raw_text, width=100, height=50, errors=errors
    )
    assert errors == []
    assert len(preds) == 1
    assert preds[0]["points"] == [0, 0, 99, 49]
    assert preds[0]["points_text"] == "0 0 99 49"
    assert preds[0]["_coord_mode"] == "norm1000"


def test_standardize_pred_pixel_passthrough():
    standardizer = CoordinateStandardizer("text")
    errors: list[str] = []
    raw_text = '{"obj":{"bbox_2d":[10,20,30,40],"desc":"car"}}'
    preds = standardizer.process_prediction_text(
        raw_text, width=120, height=80, errors=errors
    )
    assert errors == []
    assert len(preds) == 1
    assert preds[0]["points"] == [10, 20, 30, 40]
    assert preds[0]["points_text"] == "10 20 30 40"
    assert preds[0]["_coord_mode"] == "pixel"


def test_gt_mode_mismatch_flags_error():
    standardizer = CoordinateStandardizer("text")
    errors: list[str] = []
    record = {
        "width": 32,
        "height": 24,
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_10|>",
                    "<|coord_10|>",
                    "<|coord_20|>",
                    "<|coord_20|>",
                ],
                "desc": "bad",
            }
        ],
    }
    objs = standardizer.process_record_gt(record, width=32, height=24, errors=errors)
    assert objs == []
    assert "mode_gt_mismatch" in errors
