import pytest

from src.datasets.dense_caption import BaseCaptionDataset
from src.datasets.preprocessors.resize import smart_resize


class DummyTemplate:
    def __init__(self, *, max_pixels: int) -> None:
        self.max_pixels = max_pixels
        self.system = None


class _NoEncodeDataset(BaseCaptionDataset):
    """Dataset stub that bypasses template encoding.

    These tests exercise BaseCaptionDataset max_pixels enforcement without
    requiring real images or ms-swift template.encode behavior.
    """

    def _create_builder(self, mode):  # type: ignore[override]
        return object()

    def _encode_record(self, *, record, builder, system_prompt):  # type: ignore[override]
        return {"input_ids": [0, 1, 2]}


def _make_dataset(
    *,
    record: dict,
    max_pixels: int = 786432,
    offline_max_pixels: int | None = None,
) -> _NoEncodeDataset:
    template = DummyTemplate(max_pixels=max_pixels)
    return _NoEncodeDataset(
        base_records=[record],
        template=template,
        user_prompt="describe the image",
        emit_norm="none",
        json_format="standard",
        dataset_name="unit",
        offline_max_pixels=offline_max_pixels,
    )


def _record(*, width: int, height: int) -> dict:
    return {
        "images": ["unit.png"],
        "width": int(width),
        "height": int(height),
        "objects": [],
    }


def test_max_pixels_guard_raises_on_oversize_record() -> None:
    ds = _make_dataset(record=_record(width=2048, height=1024))
    with pytest.raises(ValueError, match=r"exceeds template\.max_pixels"):
        _ = ds[0]


def test_max_pixels_guard_allows_record_at_cap() -> None:
    ds = _make_dataset(record=_record(width=1024, height=768))
    out = ds[0]
    assert out["dataset"] == "unit"
    assert out["base_idx"] == 0
    assert "sample_id" in out


def test_max_pixels_guard_prefers_offline_contract_when_set() -> None:
    ds = _make_dataset(
        record=_record(width=1200, height=900),
        max_pixels=7864320000,
        offline_max_pixels=786432,
    )
    with pytest.raises(ValueError, match=r"custom\.offline_max_pixels"):
        _ = ds[0]


def test_smart_resize_upscales_low_resolution_images_toward_cap() -> None:
    target_h, target_w = smart_resize(
        height=96,
        width=128,
        factor=32,
        max_pixels=32 * 32 * 1024,
        min_pixels=32 * 32 * 4,
    )
    assert (target_h, target_w) == (864, 1152)
    assert target_h * target_w <= 32 * 32 * 1024
    assert target_h * target_w > 96 * 128


def test_smart_resize_keeps_aspect_ratio_for_common_low_res_shapes() -> None:
    target_h, target_w = smart_resize(
        height=480,
        width=640,
        factor=32,
        max_pixels=32 * 32 * 1024,
        min_pixels=32 * 32 * 4,
    )
    assert target_h % 32 == 0
    assert target_w % 32 == 0
    assert target_h * target_w <= 32 * 32 * 1024
    assert (target_w / target_h) == pytest.approx(640 / 480)
    assert target_h * target_w > 480 * 640


def test_dataset_requires_width_and_height_at_init() -> None:
    with pytest.raises(ValueError, match=r"missing required key 'width'"):
        _make_dataset(record={"images": ["x.png"], "objects": []})


def test_dataset_rejects_non_int_dims_at_init() -> None:
    with pytest.raises(ValueError, match=r"width must be an int"):
        _make_dataset(
            record={
                "images": ["x.png"],
                "objects": [],
                "width": 1024.5,
                "height": 768,
            }
        )
