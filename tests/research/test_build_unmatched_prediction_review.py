from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.research.build_unmatched_prediction_review import (
    SCHEMA_VERSION,
    build_html,
    build_review_payload,
    parse_args,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for image_id, colour in (("one", (220, 30, 30)), ("two", (30, 30, 220))):
        Image.new("RGB", (40, 30), colour).save(image_root / f"{image_id}.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fixture.manifest.v1",
                "receipt_sha256": "receipt-digest",
                "selected_image_ids": ["one", "two"],
                "items": [
                    {"image_id": "one", "image_path": str(image_root / "one.png"), "width": 40, "height": 30, "arms": {"FULL_BAG_K": {"post_merge_prediction_count": 1}}},
                    {"image_id": "two", "image_path": str(image_root / "two.png"), "width": 40, "height": 30, "arms": {"FULL_BAG_K": {"post_merge_prediction_count": 0}}},
                ],
            }
        ),
        encoding="utf-8",
    )
    queue = tmp_path / "queue.jsonl"
    _write_jsonl(
        queue,
        [
            {"image_id": "one", "arm": "FULL_BAG_K", "prediction_id": "p-one", "category": "cup", "bbox_xyxy": [3, 4, 15, 18], "score": 0.9},
            {"image_id": "one", "arm": "MASK_RESET", "prediction_id": "p-hidden-arm", "category": "cat", "bbox_xyxy": [1, 1, 4, 4]},
            {"image_id": "other", "arm": "FULL_BAG_K", "prediction_id": "p-hidden-image", "category": "dog", "bbox_xyxy": [1, 1, 4, 4]},
        ],
    )
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {"image_id": "one", "object_identifier": "ledger-one", "normalized_category_name": "chair", "source_canvas_box_xyxy": [16, 2, 30, 20], "final_state": "accepted"},
            {"image_id": "two", "object_identifier": "ledger-two", "normalized_category_name": "book", "source_canvas_box_xyxy": [2, 2, 8, 10], "final_state": "accepted"},
        ],
    )
    return manifest, queue, ledger


def test_build_payload_filters_arm_and_preserves_zero_candidate_image(tmp_path: Path) -> None:
    manifest, queue, ledger = _fixture(tmp_path)
    payload = build_review_payload(
        manifest_path=manifest,
        queue_path=queue,
        accepted_ledger_path=ledger,
        arm="FULL_BAG_K",
        review_set_id="fixture-review",
        reviewer="alice",
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["review_set_id"] == "fixture-review"
    assert payload["selected_image_ids"] == ["one", "two"]
    assert [row["candidate_id"] for row in payload["candidates"]] == ["p-one"]
    assert [row["image_id"] for row in payload["images"]] == ["one", "two"]
    assert payload["images"][1]["image_id"] == "two"
    assert payload["accepted_ledger"][1]["image_id"] == "two"
    assert payload["source_digests"]["queue_sha256"] == hashlib.sha256(queue.read_bytes()).hexdigest()
    assert payload["hidden_provenance"]["manifest_schema_version"] == "fixture.manifest.v1"


def test_html_embeds_contract_controls_and_safe_payload(tmp_path: Path) -> None:
    manifest, queue, ledger = _fixture(tmp_path)
    payload = build_review_payload(manifest_path=manifest, queue_path=queue, accepted_ledger_path=ledger)
    html = build_html(payload)
    assert "review-payload" in html
    assert "Approve A" in html and "Reject R" in html and "Unknown U" in html
    assert "localStorage" in html
    assert "Import JSON" in html and "Export JSON" in html
    assert "entity_ref" in html and "semantic_status" in html and "geometry_status" in html
    assert "annotation_gate_summary" in html
    assert "hidden_provenance" in html
    assert "exact" in html and "wrong_category" in html and "not_applicable" in html
    assert "acceptable" in html and "localization_error" in html and "multiple_entities" in html
    assert "Consolidation" in html and "approvedCandidate" in html and "human:" in html
    assert "bbox_xy[" not in html
    assert "consolidationMode ? candidates.filter" in html
    assert "approvedCandidate" in html and 'x.bbox_xyxy[0]' in html
    assert 'x.bbox_xy[0]' not in html
    assert "source_digests" in html and "mismatch" in html
    assert "approved_invalid_entity_ref" in html and "approved_invalid_status" in html
    assert "score" not in html.split("<script id=\"review-payload\"")[0]
    # Payload is escaped before insertion into an executable script block.
    assert "\\u003c" in html or "data:image/png;base64" in html


def test_selected_image_ids_and_invalid_bbox_or_path_fail_fast(tmp_path: Path) -> None:
    manifest, queue, ledger = _fixture(tmp_path)
    payload = build_review_payload(
        manifest_path=manifest,
        queue_path=queue,
        accepted_ledger_path=ledger,
        image_ids=["two"],
    )
    assert payload["selected_image_ids"] == ["two"]
    assert payload["candidates"] == []

    bad_queue = tmp_path / "bad_queue.jsonl"
    _write_jsonl(bad_queue, [{"image_id": "one", "arm": "FULL_BAG_K", "prediction_id": "bad", "category": "cup", "bbox_xyxy": [4, 4, 3, 8]}])
    with pytest.raises(ValueError, match="Invalid bbox"):
        build_review_payload(manifest_path=manifest, queue_path=bad_queue, accepted_ledger_path=ledger)

    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({"selected_image_ids": ["one"], "items": [{"image_id": "one", "image_path": str(tmp_path / "missing.png")}]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Image path"):
        build_review_payload(manifest_path=bad_manifest, queue_path=queue, accepted_ledger_path=ledger)


def test_parse_args_accepts_repeated_image_ids_without_duplicate_registration(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--manifest", str(tmp_path / "manifest.json"),
            "--queue", str(tmp_path / "queue.jsonl"),
            "--accepted-ledger", str(tmp_path / "ledger.jsonl"),
            "--image-id", "one", "--image-id", "two", "--output", str(tmp_path / "review.html"),
        ]
    )
    assert args.image_id == ["one", "two"]
    assert args.arm == "FULL_BAG_K"


def test_html_contains_gate_and_candidate_provenance_contract(tmp_path: Path) -> None:
    manifest, queue, ledger = _fixture(tmp_path)
    payload = build_review_payload(manifest_path=manifest, queue_path=queue, accepted_ledger_path=ledger)
    html = build_html(payload)
    assert "approved_missing_entity_ref" in html
    assert "approved_missing_status" in html
    assert "unreviewed_candidates" in html
    assert "candidate_id" in html and "source_bbox_xyxy" in html
    assert "hidden_provenance:source.hidden_provenance" in html
    assert "sameObject(data.source_digests" in html
