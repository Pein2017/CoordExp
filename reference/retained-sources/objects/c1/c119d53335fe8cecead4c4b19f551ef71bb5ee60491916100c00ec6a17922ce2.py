import json
from pathlib import Path

from PIL import Image
import pytest

import blind_review as b


def _image(path: Path, size=(100, 80)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def _item(path: Path, image_id: int, proposals: list[dict]) -> dict:
    return {"schema": "synthetic_cpu_fixture", "review_id": f"r:{image_id}",
            "image_id": image_id, "literal_source_canvas_path": str(path),
            "image_width": 100, "image_height": 80, "source_blind": True,
            "proposals": proposals}


def _proposal(proposal_id: str, description="cup", bbox=None) -> dict:
    return {"proposal_id": proposal_id, "description": description,
            "bbox": bbox or [10, 15, 40, 50], "bbox_format": "xyxy"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_renderer_keeps_literal_dimensions_coordinates_and_aliases(tmp_path):
    source = _image(tmp_path / "source.png")
    item = _item(source, 1, [_proposal("p1"), _proposal("p2")])
    rendered = b.render_item(item, tmp_path / "render")
    assert rendered["source_dimensions"] == [100, 80]
    assert rendered["original_image"]["sha256"] == b.binding(source)["sha256"]
    assert len(rendered["candidates"]) == 1
    candidate = rendered["candidates"][0]
    assert candidate["proposal_ids"] == ["p1", "p2"]
    assert candidate["bbox"] == [10, 15, 40, 50]
    assert candidate["overlay_candidates"] == 1
    with Image.open(candidate["overlay"]["path"]) as overlay:
        assert overlay.size == (100, 80)
        assert overlay.getpixel((10, 50)) != (255, 255, 255)
    b._assert_reviewer_safe(rendered)


def test_reviewer_payload_rejects_arm_leak_and_queue_rejects_duplicate_proposals(tmp_path):
    with pytest.raises(ValueError, match="leaks source"):
        b._assert_reviewer_safe({"source_arm": "A"})
    source = _image(tmp_path / "source.png")
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(queue, [_item(source, 1, [_proposal("p1"), _proposal("p1", bbox=[50, 10, 70, 30])])])
    with pytest.raises(ValueError, match="duplicate proposal ID"):
        b.validate_queue(queue, expected_images=1)


def _fixture(tmp_path: Path):
    source1 = _image(tmp_path / "source1.png")
    source2 = _image(tmp_path / "source2.png")
    queue_rows = [
        _item(source1, 1, [_proposal("p1"), _proposal("p2"),
                          _proposal("p3", description="book", bbox=[50, 10, 90, 60])]),
        _item(source2, 2, [_proposal("q1", description="person", bbox=[5, 5, 30, 75])]),
    ]
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(queue, queue_rows)
    rendered = [b.render_item(item, tmp_path / "render" / str(item["image_id"]))
                for item in queue_rows]
    batch = {
        "schema": "owner_successor_scale.blind_review.batch.v1",
        "status": "frozen_source_blind_unreviewed",
        "batch_id": "synthetic-batch-01",
        "images": rendered,
    }
    batch_path = tmp_path / "preparation" / "batches" / "batch-01" / "input.json"
    b.publish(batch_path, batch)
    preparation = tmp_path / "preparation" / "manifest.json"
    b.publish(preparation, {
        "schema": "owner_successor_scale.blind_review.preparation.v1",
        "status": "rendered_and_batches_frozen_not_reviewed",
        "queue": b.binding(queue),
        "images": 2,
        "batches": [{"batch_id": batch["batch_id"], "input": b.binding(batch_path),
                     "images": [1, 2]}],
    })
    source_map = tmp_path / "source-map.json"
    source_map.write_text(json.dumps({"rows": [
        {"image_id": 1, "proposal_id": "p1", "source_arm": "N16-anchor", "source_prediction_index": 0},
        {"image_id": 1, "proposal_id": "p2", "source_arm": "A", "source_prediction_index": 0},
        {"image_id": 1, "proposal_id": "p3", "source_arm": "B", "source_prediction_index": 0},
        {"image_id": 2, "proposal_id": "q1", "source_arm": "N16-anchor", "source_prediction_index": 0},
        {"image_id": 2, "proposal_id": "q1", "source_arm": "A", "source_prediction_index": 0},
    ]}))
    reviews = [
        {"image_id": 1, "review_id": "r:1", "reviewer": "synthetic fixture",
         "saved_before_next_view_attestation": b.SAVE_ATTESTATION,
         "viewed": [
             {"path": rendered[0]["original_image"]["path"],
              "sha256": rendered[0]["original_image"]["sha256"], "detail": "original"},
             {"path": rendered[0]["candidates"][0]["overlay"]["path"],
              "sha256": rendered[0]["candidates"][0]["overlay"]["sha256"], "detail": "original"},
         ],
         "owners": [{"owner_id": "O1", "proposal_ids": ["p1", "p2"],
                     "extent_or_class_caveats": []}],
         "group_coverage": [{"group_id": "G1", "proposal_ids": ["p3"],
                             "extent_or_class_caveats": ["dense coherent group, not atomic"]}],
         "unresolved": [], "non_owner_evidence": []},
        {"image_id": 2, "review_id": "r:2", "reviewer": "synthetic fixture",
         "saved_before_next_view_attestation": b.SAVE_ATTESTATION,
         "viewed": [{"path": rendered[1]["original_image"]["path"],
                     "sha256": rendered[1]["original_image"]["sha256"], "detail": "original"}],
         "owners": [{"owner_id": "O1", "proposal_ids": ["q1"],
                     "extent_or_class_caveats": ["class ignored for physical presence"]}],
         "group_coverage": [], "unresolved": [], "non_owner_evidence": []},
    ]
    review = tmp_path / "review.jsonl"
    _write_jsonl(review, reviews)
    return queue, preparation, source_map, review, reviews


def test_exact_join_deduplicates_alias_owner_and_keeps_group_out_of_atomic_gain(tmp_path):
    queue, preparation, source_map, review, _ = _fixture(tmp_path)
    result = b.compare(queue_path=queue, preparation_path=preparation,
                       source_map_path=source_map, review_paths=[review],
                       output=tmp_path / "comparison.json", expected_images=2)
    assert result["denominators"]["queue_unique_proposal_ids"] == 4
    assert result["denominators"]["source_map_rows"] == 5
    assert result["denominators"]["atomic_owner_clusters"] == 2
    assert result["denominators"]["group_coverage_clusters_separate"] == 1
    assert result["per_arm"]["N16-anchor"]["atomic_physical_owners_present"] == 2
    assert result["per_arm"]["A"]["atomic_physical_owners_present"] == 2
    assert result["per_arm"]["B"]["atomic_physical_owners_present"] == 0
    assert result["per_arm"]["B"]["group_coverage_present_separate"] == 1
    assert result["comparisons_atomic_owners_only"]["N16-anchor->A"]["counts"] == {
        "gained": 0, "retained": 2, "lost": 0}
    assert result["comparisons_atomic_owners_only"]["N16-anchor->B"]["counts"] == {
        "gained": 0, "retained": 0, "lost": 2}


def test_exact_review_coverage_and_literal_alias_split_fail_closed(tmp_path):
    queue, preparation, _, review, reviews = _fixture(tmp_path)
    queue_rows = b.validate_queue(queue, expected_images=2)
    omitted = tmp_path / "omitted.jsonl"
    broken = json.loads(json.dumps(reviews))
    broken[1]["owners"][0]["proposal_ids"] = []
    _write_jsonl(omitted, broken)
    with pytest.raises(ValueError, match="empty/duplicate"):
        b.validate_reviews(queue_rows, [omitted], preparation_path=preparation,
                           queue_path=queue)

    split = tmp_path / "split.jsonl"
    broken = json.loads(json.dumps(reviews))
    broken[0]["owners"][0]["proposal_ids"] = ["p1"]
    broken[0]["unresolved"] = [{"proposal_ids": ["p2"], "axes": ["extent"],
                                "image_grounded_reason": "synthetic ambiguity"}]
    _write_jsonl(split, broken)
    with pytest.raises(ValueError, match="literal alias split"):
        b.validate_reviews(queue_rows, [split], preparation_path=preparation,
                           queue_path=queue)


def test_review_rejects_viewed_evidence_from_another_prepared_image(tmp_path):
    queue, preparation, _, _, reviews = _fixture(tmp_path)
    queue_rows = b.validate_queue(queue, expected_images=2)
    swapped = json.loads(json.dumps(reviews))
    swapped[0]["viewed"] = swapped[1]["viewed"]
    review = tmp_path / "swapped-view.jsonl"
    _write_jsonl(review, swapped)
    with pytest.raises(ValueError, match="different image or preparation"):
        b.validate_reviews(queue_rows, [review], preparation_path=preparation,
                           queue_path=queue)
