from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.research.build_sampled_rescue_review_packet import (
    ReviewPacketError,
    SCHEMA_VERSION,
    build_packet,
    sha256_json,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path, *, seed: int, image_id: str = "7816", selected: bool = True) -> Path:
    image_path = tmp_path / f"{image_id}.jpg"
    if not image_path.exists():
        Image.new("RGB", (64, 48), (220, 230, 240)).save(image_path, format="JPEG")
    source_path = tmp_path / "val.coord.jsonl"
    source_path.write_text(json.dumps({"image_id": int(image_id), "images": [image_path.name]}) + "\n", encoding="utf-8")
    image_hash = _file_sha256(image_path)
    prompt_ids = [1, 2, 3]
    prefix_ids = [10, 11]
    ledger = [
        {"entity_id": "covered", "description": "person", "bbox_norm1000": [100, 100, 250, 500], "verification": "verified", "source": "test"},
        {"entity_id": "uncovered", "description": "person", "bbox_norm1000": [700, 200, 850, 650], "verification": "verified", "source": "test"},
    ]
    candidate = [151646, 8987, 151647, 151648, 151800 + seed, 151700, 151850, 151900, 151649]
    candidate_hash = sha256_json(candidate)
    owner_match = {
        "matched_entity_id": "uncovered",
        "predicted_bbox_norm1000": [705, 205, 845, 640],
        "predicted_bbox": [45, 10, 54, 31],
        "status": "matched",
    }
    sample = {
        "seed": seed,
        "selected_verified_uncovered_rescue": selected,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": sha256_json(prefix_ids),
        "candidate_token_ids": candidate,
        "candidate_token_ids_sha256": candidate_hash,
        "strict_matched_owner_ids": ["uncovered"],
        "verified_uncovered_owner_ids": ["uncovered"],
        "owner_matches": [owner_match],
    }
    greedy_ids = [151646, 8987, 151647, 151648, 151720, 151701, 151800, 151850, 151649]
    artifact = {
        "schema_version": "exact_prefix_sampled_rescue.v1",
        "selected_source_identity": {
            "image_id": image_id,
            "source_jsonl": str(source_path),
            "image_sha256": image_hash,
            "width": 64,
            "height": 48,
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": sha256_json(prompt_ids),
            "positive_owner_ledger": ledger,
        },
        "prefix": {
            "token_ids": prefix_ids,
            "token_ids_sha256": sha256_json(prefix_ids),
            "covered_owner_ids": ["covered"],
            "uncovered_owner_ids": ["uncovered"],
            "row_index": 5,
            "harmful_kind": "duplicate",
        },
        "frozen_source": {
            "harmful_kind": "duplicate",
            "row_index": 5,
            "prefix_token_ids_sha256": sha256_json(prefix_ids),
        },
        "greedy": {
            "candidate_token_ids": greedy_ids,
            "candidate_token_ids_sha256": sha256_json(greedy_ids),
            "strict_matched_owner_ids": ["covered"],
            "owner_matches": [{
                "matched_entity_id": "covered",
                "predicted_bbox_norm1000": [105, 105, 245, 490],
                "status": "matched",
            }],
        },
        "source_checkpoint_identity": {"checkpoint_id": "checkpoint-test"},
        "samples": [sample],
    }
    path = tmp_path / f"rescue-{seed}.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_builds_packet_with_pending_sample_and_harmful_cases(tmp_path: Path) -> None:
    first = _artifact(tmp_path, seed=28)
    second = _artifact(tmp_path, seed=39)
    output = tmp_path / "packet"
    packet = build_packet([first, second], output)
    assert packet["schema_version"] == SCHEMA_VERSION
    assert packet["never_auto_admit"] is True
    assert len(packet["groups"]) == 1
    assert len(packet["cases"]) == 3
    assert {case["role"] for case in packet["cases"]} == {
        "selected_verified_uncovered_rescue",
        "frozen_greedy_harmful_row",
    }
    for case in packet["cases"]:
        assert case["review"]["entity_review_status"] == "pending"
        assert case["review"]["geometry_review_status"] == "pending"
    artifacts = packet["groups"][0]["artifacts"]
    assert Path(artifacts["full_image_png"]).is_file()
    assert Path(artifacts["crop_png"]).is_file()
    assert len(artifacts["full_image_png_sha256"]) == 64
    assert json.loads((output / "review-packet.json").read_text()) == packet


def test_refuses_blind_image(tmp_path: Path) -> None:
    path = _artifact(tmp_path, seed=28, image_id="1584")
    with pytest.raises(ReviewPacketError, match="blind image"):
        build_packet([path], tmp_path / "packet")


def test_refuses_untrusted_selected_row_without_owner_geometry(tmp_path: Path) -> None:
    path = _artifact(tmp_path, seed=28)
    data = json.loads(path.read_text())
    data["samples"][0]["owner_matches"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReviewPacketError, match="exactly one owner match"):
        build_packet([path], tmp_path / "packet")
