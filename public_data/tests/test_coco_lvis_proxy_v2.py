import collections
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from public_data.scripts.build_coco_lvis_proxy_v2 import _hard_view, _refine
from src.data import iter_raw_examples


def _obj(box, name, **extra):
    return {
        "bbox_2d": [f"<|coord_{v}|>" for v in box],
        "desc": name,
        "category_id": 1,
        "category_name": name,
        **extra,
    }


def test_refined_proxy_has_aligned_weights_and_current_trainable_hard_view(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "img.png")
    row = {
        "file_name": "img.png",
        "image_id": 1,
        "images": ["img.png"],
        "width": 100,
        "height": 100,
        "objects": [
            _obj((0, 0, 300, 500), "person", coco_ann_id=1),
            _obj((300, 0, 700, 500), "laptop", coco_ann_id=2),
            _obj((20, 20, 100, 90), "person", lvis_ann_id=3, lvis_category_id=793, lvis_category_name="person"),
            _obj((400, 100, 600, 200), "keyboard", lvis_ann_id=4, lvis_category_id=296, lvis_category_name="computer_keyboard"),
            _obj((0, 600, 200, 800), "dining table", lvis_ann_id=5, lvis_category_id=1052, lvis_category_name="tablecloth"),
            _obj((200, 600, 300, 800), "bottle", lvis_ann_id=6, lvis_category_id=979, lvis_category_name="soap"),
            _obj((710, 10, 950, 400), "person", lvis_ann_id=8, lvis_category_id=793, lvis_category_name="person"),
        ],
        "metadata": {
            "source": "coco2017",
            "split": "train",
            "coordexp_proxy_supervision": {
                "object_supervision": [
                    {"source": "coco", "proxy_tier": "real", "desc_ce_weight": 1.0, "coord_weight": 1.0},
                    {"source": "coco", "proxy_tier": "real", "desc_ce_weight": 1.0, "coord_weight": 1.0},
                    *[
                        {"source": "lvis", "proxy_tier": "plausible", "desc_ce_weight": 0.25, "coord_weight": 0.0, "lvis_ann_id": i, "lvis_category_id": c}
                        for i, c in ((3, 793), (4, 296), (5, 1052), (6, 979), (8, 793))
                    ],
                ]
            },
        },
    }
    refined = _refine(
        row,
        [{"id": 7, "category_id": 207, "bbox": [70.0, 70.0, 20.0, 15.0]}],
        (100, 100),
        collections.Counter(),
    )
    entries = refined["metadata"]["coordexp_proxy_supervision"]["object_supervision"]
    assert [o.get("lvis_ann_id") for o in refined["objects"] if "lvis_ann_id" in o] == [8, 5, 7]
    assert [e["lvis_ann_id"] for e in entries if e["source"] == "lvis"] == [8, 5, 7]
    assert [e["coord_weight"] for e in entries if e["source"] == "lvis"] == [0.0, 0.0, 0.5]

    hard = _hard_view(refined, collections.Counter())
    assert hard is not None
    assert [o["coco_ann_id"] for o in hard["objects"]] == [1, 2, -7]
    path = tmp_path / "train.coord.jsonl"
    path.write_text(json.dumps(hard) + "\n")
    sample = next(iter_raw_examples(path))
    assert len(sample.objects) == 3
    assert sample.objects[-1].description == "car"
