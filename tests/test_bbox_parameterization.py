from __future__ import annotations

from src.common.geometry.bbox_parameterization import (
    center_log_size_norm1000_to_xyxy_norm1000,
    xyxy_norm1000_to_center_log_size_bins,
)
from src.datasets.dense_caption import BaseCaptionDataset
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


def test_dense_caption_dataset_applies_center_log_size_without_legacy_converter() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "center_log_size"
    record = {
        "objects": [
            {
                "bbox_2d": [100, 200, 400, 700],
                "desc": "cat",
            }
        ]
    }

    dataset._apply_bbox_format(record)

    assert record["objects"][0]["bbox_2d"] == xyxy_norm1000_to_center_log_size_bins(
        [100, 200, 400, 700]
    )
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]


def test_dense_caption_dataset_applies_center_log_size_from_coord_tokens() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "center_log_size"
    record = {
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_400|>",
                    "<|coord_700|>",
                ],
                "desc": "cat",
            }
        ]
    }

    dataset._apply_bbox_format(record)

    assert record["objects"][0]["bbox_2d"] == xyxy_norm1000_to_center_log_size_bins(
        [100, 200, 400, 700]
    )
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]
