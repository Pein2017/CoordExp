from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys
import types

pytest.importorskip("pycocotools")

# Some eval modules historically imported yaml/swift/torch at import time.
# In the normal CoordExp environment these are installed; for stripped-down
# environments we fall back to lightweight stubs.
try:
    import yaml  # noqa: F401
except Exception:
    yaml_stub = types.SimpleNamespace(
        safe_load=lambda *args, **kwargs: {},
        dump=lambda *args, **kwargs: "",
    )
    sys.modules["yaml"] = yaml_stub

try:
    import swift  # noqa: F401
except Exception:
    swift_utils = types.SimpleNamespace(get_dist_setting=lambda *args, **kwargs: None)
    swift_argument = types.SimpleNamespace(RLHFArguments=object, TrainArguments=object)
    swift_llm = types.SimpleNamespace(argument=swift_argument)
    swift_stub = types.SimpleNamespace(llm=swift_llm, utils=swift_utils)
    sys.modules["swift"] = swift_stub
    sys.modules["swift.llm"] = swift_llm
    sys.modules["swift.llm.argument"] = swift_argument
    sys.modules["swift.utils"] = swift_utils

try:
    import torch  # noqa: F401
except Exception:
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
)
from src.eval.parsing import parse_prediction
from src.eval.detection import EvalOptions, evaluate_detection


def test_norm1000_to_pixels_and_bbox():
    pts_norm = [10, 20, 200, 220]
    pts_px = denorm_and_clamp(pts_norm, width=1000, height=800, coord_mode="norm1000")
    assert pts_px == [10, 16, 200, 176]
    x1, y1, x2, y2 = bbox_from_points(pts_px)
    assert not is_degenerate_bbox(x1, y1, x2, y2)


def test_bbox_to_quad():
    bbox = [0.0, 0.0, 10.0, 20.0]
    quad = bbox_to_quadrilateral(bbox)
    assert quad == [0.0, 0.0, 10.0, 0.0, 10.0, 20.0, 0.0, 20.0]


def test_coerce_point_list_preserves_order_and_token_flag() -> None:
    pts = ["<|coord_4|>", "<|coord_3|>", "<|coord_2|>", "<|coord_1|>"]
    numeric, had_tokens = coerce_point_list(pts)
    assert had_tokens is True
    assert numeric == [4.0, 3.0, 2.0, 1.0]

    numeric2, had_tokens2 = coerce_point_list([4, 3, 2, 1])
    assert had_tokens2 is False
    assert numeric2 == [4.0, 3.0, 2.0, 1.0]

    numeric3, had_tokens3 = coerce_point_list(["<|coord_1|>"])
    assert numeric3 is None
    assert had_tokens3 is False


def test_degenerate_bbox_flagged():
    assert is_degenerate_bbox(0, 0, 0, 5)


def test_parse_prediction_drops_truncated_json_without_terminator():
    raw = (
        '{"objects": [{"bbox_2d": [<|coord_10|>, <|coord_20|>, <|coord_30|>, '
        '<|coord_40|>], "desc": "box"'
    )
    parsed = parse_prediction(raw)
    assert parsed == []


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
    pred_record = {
        "index": 0,
        "images": [str(img_path)],
        "width": 64,
        "height": 48,
        "coord_mode": "norm1000",
        "pred_score_source": "confidence_postop",
        "pred_score_version": 1,
        "predictions": [
            {"type": "bbox_2d", "points": [0, 0, 999, 999], "desc": "box", "score": 1.0}
        ],
    }

    gt_path = tmp_path / "gt.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    gt_path.write_text(json.dumps(gt_record) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps(pred_record) + "\n", encoding="utf-8")

    options = EvalOptions(output_dir=tmp_path, use_segm=False, metrics="coco")
    summary = evaluate_detection(gt_path, pred_path, options=options)
    assert summary["counters"]["invalid_json"] == 0
    assert summary["counters"]["empty_pred"] == 0
    assert summary["metrics"]["bbox_AP50"] > 0.9
