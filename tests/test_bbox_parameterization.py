from __future__ import annotations

import pytest

from src.common.geometry.bbox_parameterization import (
    CXCY_LOGW_LOGH_CONVERSION_VERSION,
    CXCY_LOGW_LOGH_SLOT_ORDER,
    cxcy_logw_logh_norm1000_to_xyxy_norm1000,
    xyxy_norm1000_to_cxcy_logw_logh_bins,
)
from src.config.schema import CoordTokensConfig
from src.datasets.dense_caption import BaseCaptionDataset
from src.datasets.builders.jsonlines import JSONLinesBuilder


def _prepared_record(*, tokens: bool = False) -> dict:
    bins = xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    bbox = [f"<|coord_{value}|>" for value in bins] if tokens else bins
    return {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "metadata": {
            "prepared_bbox_format": "cxcy_logw_logh",
            "prepared_bbox_slot_order": CXCY_LOGW_LOGH_SLOT_ORDER,
            "prepared_bbox_source_format": "xyxy",
            "prepared_bbox_conversion_version": CXCY_LOGW_LOGH_CONVERSION_VERSION,
        },
        "objects": [
            {
                "bbox_2d": bbox,
                "desc": "cat",
            }
        ],
    }


def test_cxcy_logw_logh_codec_round_trip_is_close_to_canonical_xyxy() -> None:
    xyxy = [100, 200, 400, 700]
    encoded = xyxy_norm1000_to_cxcy_logw_logh_bins(xyxy)
    decoded = cxcy_logw_logh_norm1000_to_xyxy_norm1000(encoded)

    assert encoded != xyxy
    for got, expected in zip(decoded, xyxy):
        assert abs(float(got) - float(expected)) <= 2.5


def test_jsonlines_builder_serializes_prepared_cxcy_logw_logh_and_keeps_metadata_xyxy() -> None:
    builder = JSONLinesBuilder(
        user_prompt="Describe objects",
        emit_norm="none",
        mode="dense",
        json_format="standard",
        coord_tokens_enabled=True,
        bbox_format="cxcy_logw_logh",
    )
    record = _prepared_record()
    record["objects"][0]["_bbox_xyxy_original"] = [100, 200, 400, 700]

    rendered = builder.build_many([record])
    assistant_bbox = rendered["assistant_payload"]["objects"][0]["bbox_2d"]

    assert assistant_bbox == [
        f"<|coord_{value}|>"
        for value in xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    ]
    assert all(str(token).startswith("<|coord_") for token in assistant_bbox)
    assert rendered["objects"]["bbox"][0] == [100, 200, 400, 700]


def test_dense_caption_dataset_accepts_prepared_cxcy_logw_logh_and_restores_xyxy_sidecar() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    record = _prepared_record()

    dataset._apply_bbox_format(record)

    assert record["objects"][0]["bbox_2d"] == xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]


def test_dense_caption_dataset_accepts_prepared_cxcy_logw_logh_coord_tokens() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    record = _prepared_record(tokens=True)

    dataset._apply_bbox_format(record)

    assert record["objects"][0]["bbox_2d"] == [
        f"<|coord_{value}|>"
        for value in xyxy_norm1000_to_cxcy_logw_logh_bins([100, 200, 400, 700])
    ]
    assert record["objects"][0]["_bbox_xyxy_original"] == [100, 200, 400, 700]


def test_dense_caption_dataset_and_builder_preserve_prepared_cxcy_logw_logh_without_runtime_conversion() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    dataset.coord_tokens = CoordTokensConfig(enabled=True, skip_bbox_norm=True)
    record = _prepared_record()

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


def test_dense_caption_dataset_rejects_canonical_xyxy_for_cxcy_logw_logh() -> None:
    dataset = object.__new__(BaseCaptionDataset)
    dataset.bbox_format = "cxcy_logw_logh"
    record = {
        "images": ["img.jpg"],
        "width": 32,
        "height": 32,
        "objects": [{"bbox_2d": [100, 200, 400, 700], "desc": "cat"}],
    }

    with pytest.raises(ValueError, match="offline-prepared branch"):
        dataset._apply_bbox_format(record)


def test_jsonlines_builder_rejects_runtime_xyxy_to_cxcy_logw_logh_conversion() -> None:
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
        "objects": [{"bbox_2d": [100, 200, 400, 700], "desc": "cat"}],
    }

    with pytest.raises(ValueError, match="offline-prepared bbox records"):
        builder.build_many([record])
