from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from PIL import Image

from src.data.geometry import coord_bins_to_pixel_xyxy
from src.label_studio_coco_refinement.roi_transform import (
    LabelStudioRoi,
    RoiLetterboxTransform,
    RoiTransformError,
)


def test_roi_percentages_clip_then_form_exact_half_open_crop() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=100,
        source_height=80,
        roi=LabelStudioRoi(x=-10, y=25, width=30, height=100),
        canvas_width=64,
        canvas_height=64,
    )

    assert transform.roi_float_edges == (-10.0, 20.0, 20.0, 100.0)
    assert transform.clipped_float_edges == (0.0, 20.0, 20.0, 80.0)
    assert transform.crop_edges == (0, 20, 20, 80)


def test_fractional_roi_uses_floor_start_and_ceil_end() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=101,
        source_height=99,
        roi=LabelStudioRoi(x=10, y=10, width=20, height=20),
        canvas_width=64,
        canvas_height=64,
    )

    assert transform.crop_edges == (10, 9, 31, 30)


def test_realized_size_is_half_up_and_odd_padding_goes_right_bottom() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=4,
        source_height=3,
        roi=(0, 0, 100, 100),
        canvas_width=6,
        canvas_height=6,
    )

    assert (transform.realized_width, transform.realized_height) == (6, 5)
    assert (transform.pad_top, transform.pad_bottom) == (0, 1)
    assert (transform.pad_left, transform.pad_right) == (0, 0)
    assert transform.scale_x == 1.5
    assert transform.scale_y == pytest.approx(5 / 3)


def test_prepare_image_crops_resizes_once_and_pads_black() -> None:
    image = Image.new("RGB", (100, 50), (255, 0, 0))
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=100,
        source_height=50,
        roi=(0, 0, 100, 100),
        canvas_width=100,
        canvas_height=100,
    )

    canvas = transform.prepare_image(image)

    assert canvas.size == (100, 100)
    assert canvas.getpixel((50, 0)) == (0, 0, 0)
    assert canvas.getpixel((50, 25)) == (255, 0, 0)
    assert canvas.getpixel((50, 74)) == (255, 0, 0)
    assert canvas.getpixel((50, 75)) == (0, 0, 0)
    assert transform.to_receipt_dict()["processor_kwargs"] == {"do_resize": False}


def test_inverse_mapping_removes_padding_then_offsets_and_quantizes() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=100,
        source_height=50,
        roi=(0, 0, 100, 100),
        canvas_width=100,
        canvas_height=100,
    )

    mapped = transform.inverse_canvas_bbox((0, 25, 100, 75))

    assert mapped.unpadded_bbox == (0, 0, 100, 50)
    assert mapped.source_bbox == (0, 0, 100, 50)
    assert mapped.norm1000_bbox == (0, 0, 999, 999)


def test_inverse_mapping_clips_partial_padding_and_rejects_padding_only() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=100,
        source_height=50,
        roi=(0, 0, 100, 100),
        canvas_width=100,
        canvas_height=100,
    )

    mapped = transform.inverse_canvas_bbox((10, 20, 20, 30))
    assert mapped.canvas_content_intersection == (10, 25, 20, 30)
    assert mapped.source_bbox == (10, 0, 20, 5)

    with pytest.raises(RoiTransformError) as exc_info:
        transform.inverse_canvas_bbox((0, 0, 10, 20))
    assert exc_info.value.code == "mapped_entirely_in_padding"


def test_parser_canvas_conversion_remains_its_distinct_round_extent_contract() -> None:
    assert coord_bins_to_pixel_xyxy(
        [1, 1, 999, 999], image_width=100, image_height=80, field="bbox"
    ) == (0, 0, 100, 80)


def test_do_resize_false_and_canvas_identity_are_asserted() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=32,
        source_height=32,
        roi=(0, 0, 100, 100),
        canvas_width=64,
        canvas_height=64,
    )
    transform.assert_no_resize_processor_canvas(
        do_resize=False, observed_width=64, observed_height=64
    )
    assert transform.expected_grid_thw(patch_size=16) == (1, 4, 4)

    with pytest.raises(RoiTransformError, match="do_resize=false"):
        transform.assert_no_resize_processor_canvas(
            do_resize=True, observed_width=64, observed_height=64
        )
    with pytest.raises(RoiTransformError, match="does not match"):
        transform.assert_no_resize_processor_canvas(
            do_resize=False, observed_width=32, observed_height=64
        )


def test_transform_is_immutable_and_degenerate_clipped_roi_is_rejected() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=32,
        source_height=32,
        roi=(0, 0, 100, 100),
        canvas_width=32,
        canvas_height=32,
    )
    with pytest.raises(FrozenInstanceError):
        transform.pad_left = 1  # type: ignore[misc]
    with pytest.raises(RoiTransformError) as exc_info:
        RoiLetterboxTransform.from_label_studio_roi(
            source_width=32,
            source_height=32,
            roi=(200, 0, 10, 10),
            canvas_width=32,
            canvas_height=32,
        )
    assert exc_info.value.code == "roi_degenerate_after_clipping"
