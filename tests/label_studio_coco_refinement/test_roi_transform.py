from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from PIL import Image

from src.data.geometry import coord_bins_to_pixel_xyxy
from src.label_studio_coco_refinement.geometry import pixel_xyxy_to_norm1000
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

    transposed = RoiLetterboxTransform.from_label_studio_roi(
        source_width=3,
        source_height=4,
        roi=(0, 0, 100, 100),
        canvas_width=6,
        canvas_height=6,
    )
    assert (transposed.realized_width, transposed.realized_height) == (5, 6)
    assert (transposed.pad_left, transposed.pad_right) == (0, 1)
    assert (transposed.pad_top, transposed.pad_bottom) == (0, 0)


def test_prepare_image_crops_resizes_once_with_bicubic_and_pads_black(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (4, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 64),
            (64, 128, 255),
        ]
    )
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=4,
        source_height=2,
        roi=(0, 0, 100, 100),
        canvas_width=8,
        canvas_height=8,
    )
    resize_calls: list[tuple[tuple[int, int], int | None]] = []
    original_resize = Image.Image.resize

    def resize_spy(
        instance: Image.Image,
        size: tuple[int, int],
        resample: int | None = None,
        box: tuple[float, float, float, float] | None = None,
        reducing_gap: float | None = None,
    ) -> Image.Image:
        resize_calls.append((size, resample))
        return original_resize(
            instance,
            size,
            resample=resample,
            box=box,
            reducing_gap=reducing_gap,
        )

    monkeypatch.setattr(Image.Image, "resize", resize_spy)

    canvas = transform.prepare_image(image)

    assert resize_calls == [((8, 4), Image.Resampling.BICUBIC)]
    assert canvas.size == (8, 8)
    assert canvas.getpixel((4, 0)) == (0, 0, 0)
    assert canvas.getpixel((4, 1)) == (0, 0, 0)
    assert canvas.getpixel((4, 6)) == (0, 0, 0)
    assert canvas.getpixel((4, 7)) == (0, 0, 0)
    assert len({canvas.getpixel((x, y)) for x in range(8) for y in range(2, 6)}) > 1
    assert transform.to_receipt_dict()["processor_kwargs"] == {"do_resize": False}


def test_no_resize_identity_preserves_rgb_pixels_exactly() -> None:
    image = Image.new("RGB", (4, 3))
    image.putdata(
        [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
            (18, 19, 20),
            (21, 22, 23),
            (24, 25, 26),
            (27, 28, 29),
            (30, 31, 32),
            (33, 34, 35),
        ]
    )
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=4,
        source_height=3,
        roi=(0, 0, 100, 100),
        canvas_width=4,
        canvas_height=3,
    )

    canvas = transform.prepare_image(image)

    assert (transform.realized_width, transform.realized_height) == image.size
    assert (
        transform.pad_left,
        transform.pad_top,
        transform.pad_right,
        transform.pad_bottom,
    ) == (0, 0, 0, 0)
    assert transform.scale_x == transform.scale_y == 1.0
    assert canvas.mode == "RGB"
    assert canvas.tobytes() == image.tobytes()


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


def test_rectangular_canvas_golden_receipt_and_inverse_mapping() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=400,
        source_height=200,
        roi=(25, 25, 50, 50),
        canvas_width=1280,
        canvas_height=768,
    )

    assert transform.to_receipt_dict() == {
        "transform_id": "coordexp-roi-letterbox-half-up-v1",
        "source_size": [400, 200],
        "roi_percent_xywh": [25.0, 25.0, 50.0, 50.0],
        "roi_float_edges": [100.0, 50.0, 300.0, 150.0],
        "clipped_float_edges": [100.0, 50.0, 300.0, 150.0],
        "crop_edges": [100, 50, 300, 150],
        "crop_edge_convention": "half-open-floor-start-ceil-end",
        "canvas_size": [1280, 768],
        "realized_size": [1280, 640],
        "scale_x": 6.4,
        "scale_y": 6.4,
        "padding": {"left": 0, "top": 64, "right": 0, "bottom": 64},
        "resampler": "Pillow.Resampling.BICUBIC",
        "pad_value_rgb": [0, 0, 0],
        "pixel_edge_convention": "continuous-pixel-edges",
        "processor_kwargs": {"do_resize": False},
    }
    mapped = transform.inverse_canvas_bbox((320, 224, 960, 544))
    assert mapped.unpadded_bbox == (320.0, 160.0, 960.0, 480.0)
    assert mapped.crop_bbox == (50.0, 25.0, 150.0, 75.0)
    assert mapped.source_bbox == (150.0, 75.0, 250.0, 125.0)
    assert mapped.norm1000_bbox == (374, 374, 625, 625)


def test_transform_receipt_json_replay_is_deterministic_and_tamper_evident() -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=101,
        source_height=99,
        roi=LabelStudioRoi(x=-10, y=10, width=30, height=100),
        canvas_width=96,
        canvas_height=64,
    )
    receipt = json.loads(
        json.dumps(transform.to_receipt_dict(), sort_keys=True, allow_nan=False)
    )

    replayed = RoiLetterboxTransform.from_receipt_dict(receipt)

    assert replayed == transform
    assert replayed.to_receipt_dict() == receipt
    assert replayed.fingerprint == transform.fingerprint
    assert replayed.inverse_canvas_bbox(replayed.content_canvas_edges) == (
        transform.inverse_canvas_bbox(transform.content_canvas_edges)
    )
    image = Image.new("RGB", (101, 99), (7, 11, 13))
    assert (
        replayed.prepare_image(image).tobytes()
        == transform.prepare_image(image).tobytes()
    )

    receipt["scale_x"] += 0.01
    with pytest.raises(RoiTransformError) as exc_info:
        RoiLetterboxTransform.from_receipt_dict(receipt)
    assert exc_info.value.code == "receipt_mismatch"


@pytest.mark.parametrize(
    ("source_size", "roi", "canvas_size"),
    [
        ((37, 23), (0, 0, 100, 100), (64, 32)),
        ((101, 99), (10.25, 5.5, 70.25, 80.125), (1280, 768)),
        ((640, 480), (-5, 20, 70, 90), (96, 160)),
        ((7, 19), (85, -25, 40, 80), (33, 35)),
    ],
)
def test_letterbox_geometry_properties_hold_across_shapes(
    source_size: tuple[int, int],
    roi: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
) -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=source_size[0],
        source_height=source_size[1],
        roi=roi,
        canvas_width=canvas_size[0],
        canvas_height=canvas_size[1],
    )

    assert 0 < transform.realized_width <= transform.canvas_width
    assert 0 < transform.realized_height <= transform.canvas_height
    assert transform.pad_left + transform.realized_width + transform.pad_right == (
        transform.canvas_width
    )
    assert transform.pad_top + transform.realized_height + transform.pad_bottom == (
        transform.canvas_height
    )
    assert transform.pad_right - transform.pad_left in {0, 1}
    assert transform.pad_bottom - transform.pad_top in {0, 1}
    assert transform.scale_x == transform.realized_width / transform.crop_width
    assert transform.scale_y == transform.realized_height / transform.crop_height

    mapped = transform.inverse_canvas_bbox(transform.content_canvas_edges)
    expected_source = tuple(float(edge) for edge in transform.crop_edges)
    assert mapped.source_bbox == expected_source
    assert mapped.norm1000_bbox == pixel_xyxy_to_norm1000(
        expected_source,
        image_width=transform.source_width,
        image_height=transform.source_height,
    )
    assert transform.prepare_image(Image.new("RGB", source_size)).size == canvas_size


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


def test_invalid_roi_types_and_final_norm1000_degeneracy_are_rejected() -> None:
    with pytest.raises(RoiTransformError) as exc_info:
        RoiLetterboxTransform.from_label_studio_roi(
            source_width=32,
            source_height=32,
            roi=(True, 0, 10, 10),
            canvas_width=32,
            canvas_height=32,
        )
    assert exc_info.value.code == "roi_non_finite"

    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=1000,
        source_height=1000,
        roi=(0, 0, 100, 100),
        canvas_width=1000,
        canvas_height=1000,
    )
    lattice_edge = 1000 * 100 / 999
    with pytest.raises(RoiTransformError) as exc_info:
        transform.inverse_canvas_bbox(
            (lattice_edge, 100.0, lattice_edge + 1e-13, 200.0)
        )
    assert exc_info.value.code == "mapped_norm1000_degenerate"


@pytest.mark.parametrize(
    ("bbox", "expected_code"),
    [
        ((False, 0, 10, 10), "bbox_type"),
        (("0", 0, 10, 10), "bbox_type"),
        ((0, float("nan"), 10, 10), "bbox_non_finite"),
        ((0, 0, float("inf"), 10), "bbox_non_finite"),
    ],
)
def test_inverse_bbox_rejects_non_numeric_bool_and_non_finite_values(
    bbox: tuple[object, object, object, object], expected_code: str
) -> None:
    transform = RoiLetterboxTransform.from_label_studio_roi(
        source_width=32,
        source_height=32,
        roi=(0, 0, 100, 100),
        canvas_width=32,
        canvas_height=32,
    )

    with pytest.raises(RoiTransformError) as exc_info:
        transform.inverse_canvas_bbox(bbox)  # type: ignore[arg-type]

    assert exc_info.value.code == expected_code
