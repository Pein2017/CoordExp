from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.research.build_exact_trace_coordinate_review_packet import (
    SCHEMA_VERSION,
    build_packet,
    sha256_json,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(tmp_path: Path, *, image_id: str = "9400") -> tuple[Path, Path]:
    image_path = tmp_path / f"{image_id}.jpg"
    Image.new("RGB", (32, 24), (240, 240, 240)).save(image_path, format="JPEG")
    image_hash = _file_sha256(image_path)
    raw_tokens = [151646, 8987, 151647, 151648, 152544, 151768, 152669, 151933, 151649]
    prefix = [151646, 8948]
    raw_text = "<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_874|><|coord_98|><|coord_999|><|coord_263|><|box_end|>"
    prediction = {
        "bbox": [27, 2, 32, 6],
        "bbox_format": "xyxy",
        "coord_bins": [874, 98, 999, 263],
        "description": "person",
        "generated_order": 0,
        "object_span_id": "test:row-0:span-0",
    }
    row = {
        "row_index": 0,
        "status": "success",
        "accepted_complete_row": True,
        "parse_evidence": {"parse_status": "accepted", "predictions": [prediction]},
        "parsed_predictions": [prediction],
        "strict_matched_owner_ids": ["owner-1"],
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": sha256_json(prefix),
        "raw_generated_token_ids": raw_tokens,
        "raw_generated_token_ids_sha256": sha256_json(raw_tokens),
        "raw_generated_text": raw_text,
        "raw_generated_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
    }
    trace = {
        "schema_version": "test",
        "images": [{
            "image_id": image_id,
            "prompt": {
                "image_path": str(image_path),
                "image_sha256": image_hash,
                "width": 32,
                "height": 24,
            },
            "entity_ledger": [
                {"entity_id": "owner-1", "description": "person", "bbox_norm1000": [850, 80, 960, 270]},
                {"entity_id": "owner-2", "description": "person", "bbox_norm1000": [800, 90, 870, 250]},
                {"entity_id": "other", "description": "chair", "bbox_norm1000": [100, 100, 300, 400]},
            ],
            "extended_root_greedy": {"rows": [row]},
        }],
    }
    trace_path = tmp_path / f"trace-{image_id}.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    return trace_path, image_path


def test_builds_deterministic_packet_with_exact_fields_and_pngs(tmp_path: Path) -> None:
    trace_path, _ = _trace(tmp_path)
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps({"cases": [{
        "case_id": "image-9400-row-0",
        "exact_trace_path": str(trace_path),
        "row_index": 0,
        "split_role": "train",
        "proposed_review_coordinate": "x2",
    }]}), encoding="utf-8")
    output_dir = tmp_path / "packet"
    packet = build_packet(manifest_path, output_dir)
    packet_again = build_packet(manifest_path, output_dir)
    assert packet == packet_again
    assert packet["schema_version"] == SCHEMA_VERSION
    case = packet["cases"][0]
    assert case["image_id"] == "9400"
    assert case["owner_id"] == "owner-1"
    assert case["exact_prefix_token_ids"] == [151646, 8948]
    assert case["exact_raw_generated_token_ids"][2] == 151647
    assert case["predicted_norm1000_xyxy"] == [874, 98, 999, 263]
    assert case["proposed_review_axis"] == "x2"
    assert case["reference_norm1000_xyxy"] == [850, 80, 960, 270]
    assert case["per_axis_delta_predicted_minus_reference"] == [24, 18, 39, -7]
    assert case["review"]["entity_review_status"] == "pending"
    assert case["review"]["geometry_review_status"] == "pending"
    assert case["review"]["accepted_coordinate_bins"] is None
    assert Path(case["artifacts"]["full_image_png"]).is_file()
    assert Path(case["artifacts"]["crop_png"]).is_file()
    assert Image.open(case["artifacts"]["crop_png"]).size[0] > 0
    assert case["artifacts"]["full_image_png_sha256"] == _file_sha256(Path(case["artifacts"]["full_image_png"]))


def test_refuses_blind_image(tmp_path: Path) -> None:
    trace_path, _ = _trace(tmp_path, image_id="1584")
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps({"cases": [{
        "case_id": "blind",
        "exact_trace_path": str(trace_path),
        "row_index": 0,
        "split_role": "eval",
        "proposed_review_coordinate": "x2",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="blind image"):
        build_packet(manifest_path, tmp_path / "packet")


@pytest.mark.parametrize(
    ("split_role", "axis", "message"),
    [
        ("holdout", "x2", "split_role"),
        ("train", "z1", "proposed_review_coordinate"),
    ],
)
def test_refuses_invalid_review_contract(tmp_path: Path, split_role: str, axis: str, message: str) -> None:
    trace_path, _ = _trace(tmp_path)
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps({"cases": [{
        "case_id": "bad-contract",
        "exact_trace_path": str(trace_path),
        "row_index": 0,
        "split_role": split_role,
        "proposed_review_coordinate": axis,
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_packet(manifest_path, tmp_path / "packet")
