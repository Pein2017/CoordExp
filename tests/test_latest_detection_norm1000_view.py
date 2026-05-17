from __future__ import annotations

import pytest

from src.detection.data import (
    ObjectOrderingPlan,
    normalize_detection_row,
    parse_raw_detection_row,
)
from src.detection.template import get_detection_template


def _canonical_norm1000_row() -> dict:
    return {
        "images": ["images/val2017/000000000139.jpg"],
        "width": 1248,
        "height": 832,
        "image_id": 139,
        "file_name": "000000000139.jpg",
        "metadata": {"source": "coco2017", "split": "val"},
        "objects": [
            {
                "object_id": "coco:ann:1666628",
                "bbox_2d": [699, 284, 722, 336],
                "category_id": 85,
                "category_name": "clock",
                "coco_ann_id": 1666628,
                "desc": "clock",
            }
        ],
    }


def test_parse_raw_detection_row_accepts_canonical_norm1000_integer_bbox() -> None:
    raw = parse_raw_detection_row(_canonical_norm1000_row())

    obj = raw.objects[0]
    assert obj.bbox_2d.values == (699, 284, 722, 336)
    assert obj.object_id == "coco:ann:1666628"


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ([True, 2, 3, 4], "norm1000"),
        ([1, 2.0, 3, 4], "norm1000"),
        ([1, "<|coord_2|>", 3, 4], "norm1000"),
        ([1000, 2, 3, 4], "0..999"),
        ([10, 2, 3, 4], "xyxy"),
    ],
)
def test_parse_raw_detection_row_rejects_invalid_canonical_norm1000_bbox(
    bbox: list[object], message: str
) -> None:
    row = _canonical_norm1000_row()
    row["objects"][0]["bbox_2d"] = bbox

    with pytest.raises(ValueError, match=message):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_norm1000_bbox_without_object_id() -> None:
    row = _canonical_norm1000_row()
    row["objects"][0].pop("object_id")

    with pytest.raises(ValueError, match="object_id"):
        parse_raw_detection_row(row)


def test_compact_full_renders_canonical_norm1000_bbox_as_coord_tokens() -> None:
    raw = parse_raw_detection_row(_canonical_norm1000_row())
    sample = normalize_detection_row(raw, object_ordering=ObjectOrderingPlan.sorted())

    assert sample.objects[0].object_id == "coco:ann:1666628"

    rendered = get_detection_template("compact_full").render_assistant(sample)

    assert "<|coord_699|>" in rendered.text
    assert "<|coord_336|>" in rendered.text
    assert "[699, 284, 722, 336]" not in rendered.text
