from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys
import types
pytest.importorskip("pycocotools")

if "yaml" not in sys.modules:
    yaml_stub = types.SimpleNamespace(
        safe_load=lambda *args, **kwargs: {},
        dump=lambda *args, **kwargs: "",
    )
    sys.modules["yaml"] = yaml_stub

if "swift" not in sys.modules:
    swift_utils = types.SimpleNamespace(get_dist_setting=lambda *args, **kwargs: None)
    swift_argument = types.SimpleNamespace(RLHFArguments=object, TrainArguments=object)
    swift_llm = types.SimpleNamespace(argument=swift_argument)
    swift_stub = types.SimpleNamespace(llm=swift_llm, utils=swift_utils)
    sys.modules["swift"] = swift_stub
    sys.modules["swift.llm"] = swift_llm
    sys.modules["swift.llm.argument"] = swift_argument
    sys.modules["swift.utils"] = swift_utils

if "torch" not in sys.modules:
    torch_utils_data = types.SimpleNamespace(
        Dataset=object,
        IterableDataset=object,
        get_worker_info=lambda *args, **kwargs: None,
    )
    torch_utils = types.SimpleNamespace(data=torch_utils_data)
    torch_stub = types.SimpleNamespace(utils=torch_utils)
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

from src.common.geometry import (
    bbox_from_points,
    bbox_to_quadrilateral,
    coerce_point_list,
    denorm_and_clamp,
    is_degenerate_bbox,
    line_to_bbox,
)
from src.eval.parsing import parse_prediction
from src.eval.detection import EvalOptions, evaluate_detection


def test_norm1000_to_pixels_and_bbox():
    pts_norm = [10, 20, 200, 220]
    pts_px = denorm_and_clamp(pts_norm, width=1000, height=800, coord_mode="norm1000")
    assert pts_px == [10, 16, 200, 176]
    x1, y1, x2, y2 = bbox_from_points(pts_px)
    assert not is_degenerate_bbox(x1, y1, x2, y2)


def test_line_to_bbox_and_quad():
    line_pts = [0, 0, 10, 20]
    bbox = line_to_bbox(line_pts)
    assert bbox == [0.0, 0.0, 10.0, 20.0]
    quad = bbox_to_quadrilateral(bbox)
    assert quad == [0.0, 0.0, 10.0, 0.0, 10.0, 20.0, 0.0, 20.0]


def test_degenerate_bbox_flagged():
    assert is_degenerate_bbox(0, 0, 0, 5)


def test_parse_prediction_repairs_truncated_json():
    raw = '{"object_1":{"bbox_2d":["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"], "desc":"box"'
    parsed = parse_prediction(raw)
    assert len(parsed) == 1
    assert parsed[0]["points"] == [10, 20, 30, 40]


def test_end_to_end_eval_from_inference_jsonl(tmp_path: Path):
    # Build tiny GT and pred JSONL mimicking inference output
    img_path = tmp_path / "img.png"
    try:
        from PIL import Image

        Image.new("RGB", (64, 48), color=(123, 123, 123)).save(img_path)
    except Exception:  # PIL not available
        img_path.write_bytes(b"")

    gt_record = {
        "images": [str(img_path)],
        "width": 64,
        "height": 48,
        "objects": [
            {
                "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>"],
                "desc": "box",
            }
        ],
    }
    pred_record = {
        "index": 0,
        "images": [str(img_path)],
        "width": 64,
        "height": 48,
        "coord_mode": "norm1000",
        "predictions": [
            {"type": "bbox_2d", "points": [0, 0, 999, 999], "desc": "box", "score": 1.0}
        ],
    }

    gt_path = tmp_path / "gt.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    gt_path.write_text(json.dumps(gt_record) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps(pred_record) + "\n", encoding="utf-8")

    options = EvalOptions(output_dir=tmp_path, use_segm=False)
    summary = evaluate_detection(gt_path, pred_path, options=options)
    assert summary["counters"]["invalid_json"] == 0
    assert summary["counters"]["empty_pred"] == 0
    assert summary["metrics"]["bbox_AP50"] > 0.9
