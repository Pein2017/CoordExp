from __future__ import annotations

import json

import pytest

from probes.training_set_completion import coco22_annotations as a


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _sources(tmp_path):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    historical = {"stable_owner_id": "previous", "bbox_2d": ["<|coord_1|>", "<|coord_1|>", "<|coord_9|>", "<|coord_9|>"],
                  "bbox_2d_bins_1000": [1, 1, 9, 9], "class_status": "unknown", "category_name": None,
                  "desc": None, "physical_status": "valid_unlabeled", "geometry_status": "reasonable", "provenance": {"historical": "preserve"}}
    prior = {"image_id": 1, "file_name": "old.jpg", "objects": [{"coco_ann_id": 100, "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_20|>", "<|coord_20|>"]}],
             "unlabeled": [historical], "metadata": {"raw_field": [1, 2]}, "unrelated": {"keep": True}}
    fresh = {"image_id": 12, "file_name": "new.jpg", "objects": [{"coco_ann_id": 200,
             "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_20|>", "<|coord_20|>"]}],
             "metadata": {"raw_field": [3]}}
    old_rows = [prior] + [{"image_id": image, "file_name": f"old-{image}.jpg", "objects": [], "unlabeled": []}
                          for image in range(2, 12)]
    new_rows = [fresh] + [{"image_id": image, "file_name": f"new-{image}.jpg", "objects": []}
                          for image in range(13, 23)]
    _write(old, old_rows); _write(new, new_rows); return old, new, prior, fresh


def _admission(tmp_path):
    files = {}
    for key in ("original", "overlay", "crop"):
        path = tmp_path / f"{key}.jpg"; path.write_bytes(b"reviewed evidence"); files[key] = str(path)
    path = tmp_path / "lead-admission.json"
    path.write_text(json.dumps({"status": "lead-accepted", "owners": [{"image_id": 12,
        "stable_owner_id": "visual:new:12", "bbox_2d_bins_1000": [100, 200, 300, 400],
        "category_name": "person", "class_status": "verified", "decision": "distinct_real_owner",
        "scene_scope": "real_original_scene", "review_category": "original_scene_object",
        "visual": files, "reference_status": "visually_admitted", "reference_proposal_id": "proposal:12"}]}) + "\n")
    return path


def test_versioned_writeback_roundtrip_preserves_all_existing_fields_and_owners(tmp_path) -> None:
    old, new, prior, fresh = _sources(tmp_path)
    manifest = a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                               lead_admission_path=_admission(tmp_path), output=tmp_path / "annotation-v2")
    rows = [json.loads(line) for line in (tmp_path / "annotation-v2/annotations.jsonl").read_text().splitlines()]
    assert rows[0] == prior
    assert {key: value for key, value in rows[11].items() if key != "unlabeled"} == fresh
    assert rows[11]["unlabeled"][0]["stable_owner_id"] == "visual:new:12"
    assert rows[11]["unlabeled"][0]["bbox_2d"] == [f"<|coord_{x}|>" for x in [100, 200, 300, 400]]
    assert rows[11]["unlabeled"][0]["provenance"]["recorded_visual_or_review"]
    assert rows[11]["unlabeled"][0]["scene_scope"] == "real_original_scene"
    assert manifest["sources"]["predecessor_annotations"]["sha256"]
    assert manifest["annotations"]["sha256"] == a.binding(tmp_path / "annotation-v2/annotations.jsonl")["sha256"]


def test_neutral_unknown_keeps_null_description_in_consumer_jsonl(tmp_path) -> None:
    old, new, _, _ = _sources(tmp_path); path = _admission(tmp_path)
    content = json.loads(path.read_text()); content["owners"][0]["class_status"] = "unknown"
    content["owners"][0]["category_name"] = None
    path.write_text(json.dumps(content) + "\n")
    a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                    lead_admission_path=path, output=tmp_path / "annotation-v2")
    row = json.loads((tmp_path / "annotation-v2/annotations.jsonl").read_text().splitlines()[11])
    assert row["unlabeled"][0]["class_status"] == "unknown"
    assert row["unlabeled"][0]["category_name"] is row["unlabeled"][0]["desc"] is None


def test_visually_resolved_prior_unknown_updates_same_owner_and_keeps_prior_provenance(tmp_path) -> None:
    old, new, prior, _ = _sources(tmp_path); path = _admission(tmp_path)
    entry = json.loads(path.read_text())["owners"][0]
    entry.update({"image_id": 1, "stable_owner_id": "previous", "bbox_2d_bins_1000": [1, 1, 9, 9],
                  "decision": "resolve_prior_unknown_category", "category_name": "person", "class_status": "verified"})
    path.write_text(json.dumps({"status": "lead-accepted", "owners": [entry]}) + "\n")
    a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                    lead_admission_path=path, output=tmp_path / "annotation-v2")
    resolved = json.loads((tmp_path / "annotation-v2/annotations.jsonl").read_text().splitlines()[0])
    owner = resolved["unlabeled"][0]
    assert len(resolved["unlabeled"]) == 1
    assert owner["stable_owner_id"] == "previous" and owner["class_status"] == "verified"
    assert owner["desc"] == "person"
    assert owner["provenance"]["historical"] == prior["unlabeled"][0]["provenance"]["historical"]
    assert owner["provenance"]["category_resolution_admission"]


def test_no_lead_admission_or_duplicate_owner_cannot_create_writeback(tmp_path) -> None:
    old, new, _, _ = _sources(tmp_path); admission = _admission(tmp_path)
    content = json.loads(admission.read_text()); content["status"] = "candidate"
    admission.write_text(json.dumps(content) + "\n")
    with pytest.raises(ValueError, match="lead-accepted"):
        a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                        lead_admission_path=admission, output=tmp_path / "rejected")
    content["status"] = "lead-accepted"; content["owners"][0]["stable_owner_id"] = "previous"
    admission.write_text(json.dumps(content) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                        lead_admission_path=admission, output=tmp_path / "duplicate")


@pytest.mark.parametrize("kind", ["toy_or_figurine", "depicted_in_wall_photo"])
def test_image_within_image_or_toy_cannot_enter_original_scene_owner_ledger(tmp_path, kind) -> None:
    old, new, _, _ = _sources(tmp_path); admission = _admission(tmp_path)
    content = json.loads(admission.read_text()); content["owners"][0]["scene_scope"] = "out_of_scope"
    content["owners"][0]["review_category"] = kind
    admission.write_text(json.dumps(content) + "\n")
    with pytest.raises(ValueError, match="real original scene"):
        a.build_version(predecessor_annotations_path=old, new_originals_path=new,
                        lead_admission_path=admission, output=tmp_path / "rejected")
