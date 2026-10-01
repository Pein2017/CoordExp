from pathlib import Path
import hashlib
import json
import pytest
from public_data.scripts.audit_coco_lvis_proxy import aggregate, main
from public_data.scripts.coco_lvis_pair_stats import iou_and_cover, analyze


def fixture(root):
    sample = [dict(source="tablecloth", index=i, image_id=i, coco_split="val", target_lvis_status="unknown") for i in range(32)]
    dump = lambda name, rows: (root / name).write_text("".join(json.dumps(x)+"\n" for x in rows))
    dump("sample.jsonl", sample)
    (root / "contract.json").write_text(json.dumps(dict(sample_sha256=hashlib.sha256((root/"sample.jsonl").read_bytes()).hexdigest(), population={"tablecloth": {"train": 0, "val": 32}})))
    dump("tablecloth.labels.jsonl", [dict(source="tablecloth", index=i, image_id=i, presence="yes", box_proxy="yes", embedded=False, note="") for i in range(32)])
    dump("tablecloth.inference.jsonl", [dict(index=i, image_id=i, likely_dining_table="high") for i in range(32)])
    return root


def test_fixed_review_is_read_only(tmp_path):
    root = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = aggregate(root)
    assert result["counts"]["tablecloth"]["presence"]["yes"] == 32
    assert result["counts"]["tablecloth"]["scene_inference"] == {"high": 32}
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("kind", ["duplicate", "presence", "note", "bool", "sample_hash", "scene"])
def test_corruption_is_rejected(tmp_path, kind):
    root = fixture(tmp_path)
    path = root / "tablecloth.labels.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if kind == "duplicate": rows[-1] = rows[0]
    elif kind == "presence": rows[0]["presence"] = "invented"
    elif kind == "note": rows[0].update(presence="no", box_proxy="not_applicable")
    elif kind == "bool": rows[0]["embedded"] = 1
    elif kind == "sample_hash": (root/"sample.jsonl").write_text((root/"sample.jsonl").read_text()+"\n")
    elif kind == "scene":
        p = root/"tablecloth.inference.jsonl"
        p.write_text(p.read_text().replace('"high"','"invented"',1))
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    with pytest.raises(ValueError): aggregate(root)


def test_no_overwrite(tmp_path):
    output = tmp_path/"existing.json"; output.write_text("original")
    with pytest.raises(SystemExit): main(["--input", str(tmp_path/"absent"), "--output", str(output)])
    assert output.read_text() == "original"


def test_pair_geometry():
    assert iou_and_cover([0,0,10,10], [0,0,10,10]) == (1,1,1)
    assert iou_and_cover([0,0,0,0], [0,0,0,0]) == (0,0,0)
    assert iou_and_cover([0,0,5,10], [0,0,10,10]) == (.5,1,.5)


def test_pair_denominators_and_crowd(tmp_path):
    for ds in ("coco", "lvis"): (tmp_path/ds/"raw/annotations").mkdir(parents=True)
    coco = dict(images=[dict(id=1)], categories=[dict(id=1,name="target")], annotations=[dict(image_id=1,category_id=1,bbox=[0,0,10,10]),dict(image_id=1,category_id=1,bbox=[0,0,10,10],iscrowd=1)])
    lvis = dict(images=[dict(id=1)], categories=[dict(id=2,name="source")], annotations=[dict(image_id=1,category_id=2,bbox=[0,0,10,10]),dict(image_id=1,category_id=2,bbox=[0,0,5,10])])
    (tmp_path/"coco/raw/annotations/c.json").write_text(json.dumps(coco))
    (tmp_path/"lvis/raw/annotations/l.json").write_text(json.dumps(lvis))
    out = analyze("c.json", "l.json", data_root=tmp_path)
    assert out["coco_crowd_excluded"] == 1
    assert out["rows"][0]["l_instances"] == 2 and out["rows"][0]["images_both"] == 1
    assert out["rows"][0]["iou05"] == 2
