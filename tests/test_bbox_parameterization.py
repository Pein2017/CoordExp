from __future__ import annotations

from src.common.geometry.bbox_parameterization import (
    cxcy_logw_logh_norm1000_to_xyxy_norm1000,
    xyxy_norm1000_to_cxcy_logw_logh_bins,
)
from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset
from src.datasets.builders.jsonlines import JSONLinesBuilder


def test_cxcy_logw_logh_codec_round_trip_is_close_to_canonical_xyxy() -> None:
    xyxy = [100, 200, 400, 700]
    encoded = xyxy_norm1000_to_cxcy_logw_logh_bins(xyxy)
    decoded = cxcy_logw_logh_norm1000_to_xyxy_norm1000(encoded)

    assert encoded != xyxy
    for got, expected in zip(decoded, xyxy):
        assert abs(float(got) - float(expected)) <= 2.5


def test_jsonlines_builder_serializes_cxcy_logw_logh_but_keeps_metadata_xyxy() -> None:
    builder = JSONLinesBuilder(
        user_prompt="Describe objects",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=True,
        bbox_format="cxcy_logw_logh",
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


def test_dense_caption_dataset_applies_cxcy_logw_logh_without_legacy_converter() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    record = {
        "objects": [
            {
                "bbox_2d": [100, 200, 400, 700],
                "desc": "cat",
            }
        ]
    }

    dataset._apply_bbox_format(record)

    assert record["objects"][0]["bbox_2d"] == xyxy_norm1000_to_cxcy_logw_logh_bins(
        [100, 200, 400, 700]
    )
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]


def test_dense_caption_dataset_applies_cxcy_logw_logh_from_coord_tokens() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
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

    assert record["objects"][0]["bbox_2d"] == xyxy_norm1000_to_cxcy_logw_logh_bins(
        [100, 200, 400, 700]
    )
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]


def test_dense_caption_dataset_and_builder_do_not_double_apply_cxcy_logw_logh() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    dataset.coord_tokens = CoordTokensConfig(enabled=True, skip_bbox_norm=True)
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

    dataset._apply_bbox_format(record)
    dataset._maybe_annotate_coord_tokens(record)

    builder = JSONLinesBuilder(
        user_prompt="Describe objects",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=True,
        bbox_format="cxcy_logw_logh",
    )
    rendered = builder.build_many([record])
    assistant_bbox = rendered["assistant_payload"]["objects"][0]["bbox_2d"]

    expected_bins = xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    expected_tokens = [f"<|coord_{value}|>" for value in expected_bins]

    assert record["objects"][0]["bbox_2d"] == expected_bins
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]
    assert assistant_bbox == expected_tokens
