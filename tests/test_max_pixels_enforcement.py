import pytest

from src.datasets.dense_caption import BaseCaptionDataset


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


def _make_dataset(*, record: dict, max_pixels: int = 786432) -> _NoEncodeDataset:
    template = DummyTemplate(max_pixels=max_pixels)
    return _NoEncodeDataset(
        base_records=[record],
        template=template,
        user_prompt="describe the image",
        emit_norm="none",
        json_format="standard",
        dataset_name="unit",
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
