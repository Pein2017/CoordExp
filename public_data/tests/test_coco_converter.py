import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _make_80_categories() -> List[Dict[str, Any]]:
    # COCO 2017 instances uses 80 categories; ids are not required to be contiguous.
    # For tests, keep it simple and contiguous.
    return [{"id": i, "name": f"cat_{i}"} for i in range(1, 81)]


def _run_main(module, argv):
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + list(argv)
        try:
            module.main()
        except SystemExit as exc:
            return int(exc.code) if exc.code is not None else 0
        return 0
    finally:
        sys.argv = old_argv


def test_coco_converter_and_validator_smoke():
    from public_data.scripts import convert_coco2017_instances
    from public_data.scripts import validate_coco2017_instances

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        (raw_dir / "annotations").mkdir(parents=True, exist_ok=True)

        ann_path = raw_dir / "annotations" / "instances_train2017.json"
        out_jsonl = raw_dir / "train.jsonl"

        coco = {
            "images": [
                {"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 80},
            ],
            "categories": _make_80_categories(),
            "annotations": [
                # valid
                {"id": 10, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [10, 20, 30, 10]},
                # out-of-bounds (should be skipped by default)
                {"id": 11, "image_id": 1, "category_id": 2, "iscrowd": 0, "bbox": [-1, 0, 10, 10]},
                # crowd (skipped by default)
                {"id": 12, "image_id": 1, "category_id": 3, "iscrowd": 1, "bbox": [0, 0, 1, 1]},
            ],
        }
        ann_path.write_text(json.dumps(coco) + "\n", encoding="utf-8")

        rc = _run_main(
            convert_coco2017_instances,
            [
                "--split",
                "train",
                "--raw_dir",
                str(raw_dir),
                "--image_dir_name",
                "train2017",
                "--output",
                str(out_jsonl),
                "--require-80-categories",
            ],
        )
        assert rc == 0
        assert out_jsonl.exists()

        lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["images"] == ["images/train2017/000000000001.jpg"]
        assert row["width"] == 100
        assert row["height"] == 80

        objs = row["objects"]
        assert len(objs) == 1
        obj0 = objs[0]
        assert obj0["desc"] == "cat_1"
        assert obj0["bbox_2d"] == [10.0, 20.0, 40.0, 30.0]

        categories_json = raw_dir / "categories.json"
        assert categories_json.exists()
        cats = json.loads(categories_json.read_text(encoding="utf-8"))
        assert isinstance(cats, list) and len(cats) == 80

        sample_out = raw_dir / "sample_first5.jsonl"
        rc = _run_main(
            validate_coco2017_instances,
            [
                str(out_jsonl),
                "--categories_json",
                str(categories_json),
                "--require-80",
                "--sample_out",
                str(sample_out),
                "--sample-n",
                "1",
            ],
        )
        assert rc == 0
        assert sample_out.exists()


if __name__ == "__main__":
    # Keep parity with existing public_data/tests/*.py style.
    test_coco_converter_and_validator_smoke()
    print("✓ COCO converter/validator smoke test passed")
