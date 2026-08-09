from __future__ import annotations

import json
from pathlib import Path

from scripts.research.build_static_dynamic_owner_interface_inputs import build_inputs


def _row(image_id: int, objects: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": [f"images/{image_id}.jpg"],
        "file_name": f"images/{image_id}.jpg",
        "image_id": image_id,
        "width": 1024,
        "height": 1024,
        "objects": objects,
    }


def _obj(x1: int, y1: int, name: str) -> dict[str, object]:
    return {
        "bbox_2d": [
            f"<|coord_{x1}|>",
            f"<|coord_{y1}|>",
            "<|coord_900|>",
            "<|coord_999|>",
        ],
        "desc": name,
    }


def test_build_inputs_stably_sorts_and_proves_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "derived.jsonl"
    receipt = tmp_path / "derived.receipt.json"
    rows = [
        _row(1, [_obj(20, 3, "b"), _obj(2, 9, "a"), _obj(2, 9, "tie")]),
        _row(2, [_obj(8, 1, "c")]),
    ]
    (tmp_path / "images").mkdir()
    (tmp_path / "images/1.jpg").write_bytes(b"image-1")
    (tmp_path / "images/2.jpg").write_bytes(b"image-2")
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = build_inputs(source, output, receipt)
    derived_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [obj["desc"] for obj in derived_rows[0]["objects"]] == ["a", "tie", "b"]
    assert result["ordering"] == "geo_sorted_xy"
    assert result["unit_id"] == "2026-08-05-static-dynamic-owner-interface-crossover"
    assert result["row_count"] == 2
    assert result["owner_count"] == 4
    assert result["source_owner_multiset_sha256"] == result["derived_owner_multiset_sha256"]
    assert result["object_payload_identity"] == "canonical_json_sha256"
    assert result["source_sha256"]
    assert result["derived_sha256"]
    assert result["mapping_sha256"]
    assert result["image_references_sha256"]
    assert result["image_references_resolve_from_derived"] is True
    assert len(result["images_manifest"]) == 2
    assert result["source_to_derived"][0]["mapping"][0]["source_index"] == 1
    assert json.loads(receipt.read_text())["derived_sha256"] == result["derived_sha256"]


def test_build_inputs_rejects_non_quad_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    (tmp_path / "images").mkdir()
    (tmp_path / "images/1.jpg").write_bytes(b"image-1")
    source.write_text(json.dumps(_row(1, [{"bbox_2d": ["<|coord_1|>"], "desc": "bad"}])) + "\n")
    try:
        build_inputs(source, tmp_path / "out.jsonl", tmp_path / "receipt.json")
    except ValueError as exc:
        assert "arity 4" in str(exc)
    else:
        raise AssertionError("expected malformed coordinate arity to fail closed")


def test_build_inputs_rejects_relative_image_semantic_drift(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "images").mkdir()
    (source_dir / "images/1.jpg").write_bytes(b"image-1")
    source = source_dir / "source.jsonl"
    source.write_text(json.dumps(_row(1, [_obj(1, 2, "person")])) + "\n")

    output = tmp_path / "other/depth/derived.jsonl"
    try:
        build_inputs(source, output, output.with_suffix(".receipt.json"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected moved relative image reference to fail closed")
