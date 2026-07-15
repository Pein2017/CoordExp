from __future__ import annotations

import json

import pytest

from src.common.errors import DataContractError
from src.label_studio_coco_refinement.geometry import (
    label_studio_xywh_to_norm1000,
    norm1000_bbox_to_label_studio_xywh,
    norm1000_edge_to_percent,
    outward_quantize_norm1000_xyxy,
    parser_bins_to_canvas_xyxy,
    pixel_xyxy_to_norm1000,
)


def test_every_norm1000_edge_survives_label_studio_json_float_round_trip() -> None:
    for edge in range(999):
        bbox = (edge, edge, edge + 1, edge + 1)
        serialized = json.dumps(norm1000_bbox_to_label_studio_xywh(bbox))
        xywh = json.loads(serialized)

        assert label_studio_xywh_to_norm1000(*xywh) == bbox
        assert label_studio_xywh_to_norm1000(*xywh, previous_bbox=bbox) == bbox

    assert json.loads(json.dumps(norm1000_edge_to_percent(999))) == 100.0


@pytest.mark.parametrize(
    ("percent_xyxy", "expected"),
    [
        ((10.01, 20.01, 30.01, 40.01), (99, 199, 300, 400)),
        ((-10.0, -5.0, 110.0, 120.0), (0, 0, 999, 999)),
        ((0.0, 0.0, 100.0, 100.0), (0, 0, 999, 999)),
    ],
)
def test_edited_percentage_edges_are_outward_quantized_and_clipped(
    percent_xyxy: tuple[float, float, float, float],
    expected: tuple[int, int, int, int],
) -> None:
    assert outward_quantize_norm1000_xyxy(percent_xyxy) == expected


def test_x_plus_width_reconstruction_uses_tolerance_without_end_expansion() -> None:
    bbox = (568, 4, 717, 153)
    x, y, width, height = json.loads(
        json.dumps(norm1000_bbox_to_label_studio_xywh(bbox))
    )

    assert label_studio_xywh_to_norm1000(x, y, width, height) == bbox


@pytest.mark.parametrize(
    "xywh",
    [
        (10.0, 10.0, 0.0, 20.0),
        (10.0, 10.0, 20.0, 0.0),
        (200.0, 10.0, 1.0, 20.0),
        (float("nan"), 0.0, 10.0, 10.0),
    ],
)
def test_invalid_or_degenerate_label_studio_geometry_is_rejected(
    xywh: tuple[float, float, float, float],
) -> None:
    with pytest.raises(DataContractError):
        label_studio_xywh_to_norm1000(*xywh)


@pytest.mark.parametrize(
    ("bins", "canvas", "expected"),
    [
        ((1, 2, 998, 999), (1024, 768), (1, 2, 1022, 767)),
        ((500, 500, 999, 999), (1280, 768), (640, 384, 1279, 767)),
        ((1, 1, 2, 2), (300, 300), (0, 0, 1, 1)),
    ],
)
def test_parser_canvas_conversion_matches_executed_round_bin_extent_contract(
    bins: tuple[int, int, int, int],
    canvas: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert parser_bins_to_canvas_xyxy(
        bins,
        canvas_width=canvas[0],
        canvas_height=canvas[1],
    ) == expected


def test_pixel_to_norm1000_clips_to_image_and_requires_positive_final_area() -> None:
    assert pixel_xyxy_to_norm1000(
        (-5.0, -10.0, 1280.0, 768.0),
        image_width=1280,
        image_height=768,
    ) == (0, 0, 999, 999)

    with pytest.raises(DataContractError) as exc_info:
        pixel_xyxy_to_norm1000(
            (-5.0, 10.0, -1.0, 20.0),
            image_width=1280,
            image_height=768,
        )
    assert exc_info.value.code == "label_studio.pixel_bbox_order"
