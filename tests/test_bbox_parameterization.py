from __future__ import annotations

from src.common.geometry.bbox_parameterization import (
    center_log_size_norm1000_to_xyxy_norm1000,
    xyxy_norm1000_to_center_log_size_bins,
)
from src.datasets.builders.jsonlines import JSONLinesBuilder


def test_center_log_size_codec_round_trip_is_close_to_canonical_xyxy() -> None:
    xyxy = [100, 200, 400, 700]
    encoded = xyxy_norm1000_to_center_log_size_bins(xyxy)
    decoded = center_log_size_norm1000_to_xyxy_norm1000(encoded)

    assert encoded != xyxy
    for got, expected in zip(decoded, xyxy):
        assert abs(float(got) - float(expected)) <= 2.5


def test_jsonlines_builder_serializes_center_log_size_but_keeps_metadata_xyxy() -> None:
    builder = JSONLinesBuilder(
        user_prompt="Describe objects",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=True,
        bbox_format="center_log_size",
    )
    record = {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "objects": [
            {
                "bbox_2d": [100, 200, 400, 700],
                "desc": "cat",
            }
        ],
    }

    rendered = builder.build_many([record])
    assistant_bbox = rendered["assistant_payload"]["objects"][0]["bbox_2d"]

    assert assistant_bbox != [100, 200, 400, 700]
    assert all(str(token).startswith("<|coord_") for token in assistant_bbox)
    assert rendered["objects"]["bbox"][0] == [100, 200, 400, 700]
