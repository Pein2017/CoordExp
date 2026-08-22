from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from public_data.scripts.build_coco_views import build_views
from public_data.view_contracts import load_view_metadata


def test_build_current_coco_views_in_tempdir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = source / "images" / "train2017" / "one.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(image)
    row = {
        "images": ["images/train2017/one.jpg"], "width": 64, "height": 64,
        "objects": [{"bbox_2d": [1, 2, 40, 50], "desc": "person"}],
    }
    (source / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    image_store = tmp_path / "public_data" / "coco" / "images" / "res-64"
    views = tmp_path / "public_data" / "coco" / "views"

    totals = build_views(
        source_preset=source, image_store_root=image_store, views_root=views,
        splits=("train",), views=("coco80/full", "coco80/len-12000"),
        max_total_tokens=12000, image_store_mode="hardlink",
    )

    assert totals == {"coco80/full": 1, "coco80/len-12000": 1}
    output = views / "coco80" / "len-12000" / "train.jsonl"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["images"] == ["images/train2017/one.jpg"]
    assert payload["objects"][0]["bbox_2d"] == [15, 31, 635, 793]
    meta = load_view_metadata(output.parent / "meta.json")
    assert meta.sample_policy == "length_budget"
    assert meta.summary["records"] == 1
