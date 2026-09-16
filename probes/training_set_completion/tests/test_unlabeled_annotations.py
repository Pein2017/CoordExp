import copy
import json

import pytest

from probes.training_set_completion import unlabeled_annotations as u


def _original_rows():
    manifest = json.loads(u.ACQUISITION.read_text())
    return [record["case"]["input_record"] for record in manifest["records"]]


def test_real_export_preserves_original_records_and_contains_all_63_accepted_non_gt_owners(tmp_path):
    manifest = u.build(output=tmp_path / "export")
    rows = [json.loads(line) for line in (tmp_path / "export" / "annotations.jsonl").read_text().splitlines()]
    originals = {row["image_id"]: row for row in _original_rows()}
    assert manifest["counts"] == {"images": 11, "gt_objects": 169, "valid_unlabeled": 63, "class_unknown": 13, "class_verified": 50}
    assert len(rows) == 11
    assert sum(len(row["unlabeled"]) for row in rows) == 63
    source_index = json.loads((tmp_path / "export" / "review-source-index.json").read_text())
    assert len(source_index["entries"]) == 63
    assert all(entry["status"] == "complete" and not entry["exceptions"] for entry in source_index["entries"].values())
    assert all(all(entry["visual"][kind] for kind in ("original", "bbox_overlay", "crop")) for entry in source_index["entries"].values())
    for row in rows:
        expected = copy.deepcopy(originals[row["image_id"]])
        expected["unlabeled"] = row["unlabeled"]
        assert row == expected
        gt = {str(obj["coco_ann_id"]) for obj in row["objects"]}
        assert not gt.intersection(item["stable_owner_id"] for item in row["unlabeled"])
        assert all(item["provenance"]["recorded_visual_or_review"] for item in row["unlabeled"])


def test_class_mask_stays_unknown_even_when_old_teacher_literals_named_a_class(tmp_path):
    u.build(output=tmp_path / "export")
    rows = [json.loads(line) for line in (tmp_path / "export" / "annotations.jsonl").read_text().splitlines()]
    masked = set(json.loads(u.COMPLETE_ACCEPTANCE.read_text())["masked_description_owner_ids"])
    output = {item["stable_owner_id"]: item for row in rows for item in row["unlabeled"]}
    assert set(output).issuperset(masked)
    assert all(output[owner]["class_status"] == "unknown" and output[owner]["category_name"] is None and output[owner]["desc"] is None for owner in masked)
    assert output["new-25274-L08"]["bbox_2d_bins_1000"] == [341, 640, 371, 798]
    assert output["new-25274-L09"]["bbox_2d_bins_1000"] == [359, 651, 395, 798]
    assert output["stable-new-219546-transparent-serving-jar"]["bbox_2d_bins_1000"] == [564, 322, 684, 480]
    assert output["stablenew:rear-steering-wheel"]["bbox_2d_bins_1000"] == [710, 96, 828, 268]
    assert output["stableNew:right-white-spoon"]["bbox_2d_bins_1000"] == [776, 239, 865, 615]
    assert output["second-fit:new:59571:pig-chef-figurine"]["category_name"] == "figurine"


def test_invalid_area_is_rejected_before_export():
    with pytest.raises(ValueError, match="invalid bbox area"):
        u.validate_bbox([5, 20, 5, 40], owner_id="bad-owner")


def test_real_third_fit_admissions_extend_the_owner_bound_visual_export(tmp_path):
    evidence = u.B / "third-fit-root-rulings-v1/admissions-with-evidence.json"
    manifest = u.build(output=tmp_path / "export", extra_root_admissions=evidence)
    rows = [json.loads(line) for line in (tmp_path / "export" / "annotations.jsonl").read_text().splitlines()]
    index = json.loads((tmp_path / "export" / "review-source-index.json").read_text())
    assert manifest["counts"] == {"images": 11, "gt_objects": 169, "valid_unlabeled": 77, "class_unknown": 18, "class_verified": 59}
    assert len(index["entries"]) == 77
    additions = [entry for owner, entry in index["entries"].items() if owner.startswith("third-fit:new:")]
    assert len(additions) == 14
    assert all(entry["decision"]["source"]["path"] == str(u.B / "third-fit-root-rulings-v1/admissions.json") for entry in additions)
    assert all(all(entry["visual"][kind] for kind in ("original", "bbox_overlay", "crop")) for entry in additions)
    assert sum(len(row["unlabeled"]) for row in rows) == 77
