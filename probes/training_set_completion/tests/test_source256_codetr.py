import json

from PIL import Image

from probes.training_set_completion import source256_codetr as codetr


def _candidate(tmp_path, candidate_id="img:p0", bbox=None, image_id="img"):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (400, 300), (120, 120, 120)).save(image_path)
    return {
        "candidate_id": candidate_id,
        "image_id": image_id,
        "example_id": image_id,
        "image_path": str(image_path),
        "width": 400,
        "height": 300,
        "bbox": bbox or [100, 80, 140, 120],
        "category": "person",
        "source_kind": "greedy",
    }


def test_context_window_is_three_x_clipped_and_minimum_is_source_pixels(tmp_path):
    row = _candidate(tmp_path, bbox=[2, 3, 12, 13])
    assert codetr.context_window(row) == [0, 0, 128, 128]

    row = _candidate(tmp_path, bbox=[350, 250, 390, 290])
    assert codetr.context_window(row) == [272, 172, 400, 300]


def test_prepare_preserves_provenance_and_unpainted_crop(tmp_path):
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(_candidate(tmp_path)) + "\n")
    output = tmp_path / "prepared"
    manifest = codetr.prepare(source, output)
    assert manifest["candidate_count"] == 1
    assert manifest["resize_rule"] == {"img_scale": [2048, 1280], "keep_ratio": True}
    prepared = json.loads((output / "candidates.jsonl").read_text().splitlines()[0])
    assert prepared["context_window_xyxy"] == [56, 36, 184, 164]
    assert prepared["transform"]["detector_to_source"].endswith("exactly once after official rescale=True inference")
    with Image.open(prepared["context_path"]) as crop:
        assert crop.size == (128, 128)
        assert crop.getpixel((0, 0)) == (120, 120, 120)


def test_source_bank_hypothesis_field_aliases_are_accepted(tmp_path):
    row = _candidate(tmp_path)
    row.pop("category")
    row["description"] = "person"
    row["bbox_pixel_xyxy"] = row.pop("bbox")
    row["image_width"] = row.pop("width")
    row["image_height"] = row.pop("height")
    source = tmp_path / "source-bank.jsonl"
    source.write_text(json.dumps(row) + "\n")
    canonical = codetr.validate_candidates(source)[0]
    assert canonical["category"] == "person"
    assert canonical["bbox"] == [100.0, 80.0, 140.0, 120.0]


def test_validate_allows_detector_corpus_beyond_review_limits(tmp_path):
    rows = [_candidate(tmp_path, f"img:p{i}", bbox=[100 + i, 80, 140 + i, 120], image_id="img") for i in range(3)]
    source = tmp_path / "too_many_same_image.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(codetr.validate_candidates(source)) == 3

    rows = []
    for i in range(codetr.MAX_VISUAL_REVIEW_CANDIDATES + 1):
        rows.append(_candidate(tmp_path, f"img{i}:p0", image_id=f"img{i}"))
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(codetr.validate_candidates(source)) == codetr.MAX_VISUAL_REVIEW_CANDIDATES + 1


def test_validate_rejects_crop_filename_collision(tmp_path):
    first = _candidate(tmp_path, "img:p0")
    second = _candidate(tmp_path, "img_p0", image_id="img2")
    source = tmp_path / "collision.jsonl"
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    try:
        codetr.validate_candidates(source)
    except ValueError as error:
        assert "filename collision" in str(error)
    else:
        raise AssertionError("crop filename collision was not rejected")


def test_triage_keeps_detector_miss_unknown_and_support_is_not_admission():
    row = {"bbox": [10, 10, 50, 50], "category": "person"}
    unknown = codetr.triage(row, [])
    assert unknown["triage"] == "unknown"
    assert unknown["teacher_admission"] == "not_run"
    prediction = {"category": "person", "score": 0.9, "bbox_xyxy_source": [10, 10, 50, 50]}
    supported = codetr.triage(row, [prediction])
    assert supported["triage"] == "support"
    assert supported["support_priority"] is True
    assert supported["teacher_admission"] == "not_run"

    invalid = codetr.triage({"bbox": [10, 10, 10, 50], "category": "person", "bbox_valid": False}, [])
    assert invalid["technical_status"] == "invalid_candidate_geometry"
    assert invalid["triage"] == "unknown"


def test_visual_review_selection_is_priority_round_robin_and_bounded():
    decisions = []
    for image_id in ("a", "b", "c"):
        for index, status in enumerate(("support", "conflict", "unknown")):
            decisions.append({
                "candidate_id": f"{image_id}:{index}",
                "image_id": image_id,
                "decision": {"triage": status},
            })
    selected = codetr.select_visual_review(decisions)
    assert selected == ["a:0", "b:0", "c:0", "a:1", "b:1", "c:1"]
    assert len(selected) == len(set(selected))

    many = []
    for index in range(200):
        many.append({"candidate_id": f"i{index}:0", "image_id": f"i{index}", "decision": {"triage": "unknown"}})
    selected = codetr.select_visual_review(many)
    assert len(selected) == codetr.MAX_VISUAL_REVIEW_CANDIDATES
