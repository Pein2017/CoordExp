#!/usr/bin/env python3
"""
Tests for LVIS pre-sorting (object ordering + polygon vertex canonicalization).

This validates the Qwen3-VL prompt-spec compatibility layer added to the LVIS
converter:
  - poly vertices: canonical order (centroid-angle sort, start at top-most/left-most)
  - objects: sorted top-to-bottom then left-to-right using bbox TL / poly first vertex
"""

import json
import sys
import tempfile
from pathlib import Path

# Add public_data to sys.path (same pattern as other tests in this folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from public_data.converters.base import ConversionConfig
from public_data.converters.lvis_converter import LVISConverter
from public_data.converters.sorting import canonicalize_poly


def _pairs(flat):
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def test_canonicalize_poly_is_deterministic():
    # Same square vertices, different input orders / start points.
    pts_a = [60, 10, 60, 20, 50, 20, 50, 10]  # start at top-right
    pts_b = [50, 20, 50, 10, 60, 10, 60, 20]  # start at bottom-left

    canon_a = canonicalize_poly(pts_a)
    canon_b = canonicalize_poly(pts_b)

    assert canon_a == canon_b
    assert canon_a[:2] == [50.0, 10.0]  # top-most then left-most
    assert _pairs(canon_a) == [(50.0, 10.0), (50.0, 20.0), (60.0, 20.0), (60.0, 10.0)]

    # Vertex set preserved (up to ordering); helpful sanity check.
    assert sorted(_pairs(canon_a)) == sorted(_pairs([float(v) for v in pts_a]))


def test_lvis_converter_presorts_objects():
    mock = {
        "images": [
            {"id": 1, "file_name": "test.jpg", "width": 300, "height": 300},
        ],
        "categories": [
            {"id": 1, "name": "thing", "frequency": "common"},
        ],
        "annotations": [
            # Intentionally scrambled input order:
            # 1) polygon lower in the image
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 200, 10, 10],
                "segmentation": [[110, 200, 110, 210, 100, 210, 100, 200]],
            },
            # 2) bbox in the middle
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 100, 10, 10],
            },
            # 3) polygon near the top
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 10, 10, 10],
                "segmentation": [[60, 10, 60, 20, 50, 20, 50, 10]],
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ann_path = tmp / "lvis_mock.json"
        img_root = tmp / "images"
        img_root.mkdir(parents=True, exist_ok=True)
        (img_root / "test.jpg").touch()

        ann_path.write_text(json.dumps(mock), encoding="utf-8")

        config = ConversionConfig(
            input_path=str(ann_path.resolve()),
            output_path=str((tmp / "out.jsonl").resolve()),
            image_root=str(img_root.resolve()),
            split="train",
        )

        converter = LVISConverter(config, use_polygon=True, poly_max_points=None)
        converter.load_annotations()

        record = converter.convert_sample(1)
        assert record is not None

        objs = record["objects"]
        assert len(objs) == 3

        # Expect ordering by anchor point (y then x):
        # - top polygon: anchor y=10
        # - middle bbox: anchor y=100
        # - bottom polygon: anchor y=200
        anchors = []
        for obj in objs:
            if "bbox_2d" in obj:
                anchors.append((obj["bbox_2d"][1], obj["bbox_2d"][0]))
            else:
                anchors.append((obj["poly"][1], obj["poly"][0]))

        assert anchors == sorted(anchors)

        # Check the first object is the top polygon and is canonicalized.
        assert "poly" in objs[0]
        assert objs[0]["poly"][:2] == [50.0, 10.0]

        # Check the last object is the bottom polygon and is canonicalized.
        assert "poly" in objs[-1]
        assert objs[-1]["poly"][:2] == [100.0, 200.0]


def main():
    print("=" * 60)
    print("LVIS Pre-sorting Tests")
    print("=" * 60)

    test_canonicalize_poly_is_deterministic()
    print("✓ canonicalize_poly deterministic")

    test_lvis_converter_presorts_objects()
    print("✓ LVISConverter pre-sorts objects")

    print("=" * 60)
    print("All pre-sorting tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

