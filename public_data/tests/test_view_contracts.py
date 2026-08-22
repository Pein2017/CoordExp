from __future__ import annotations

from pathlib import Path

import pytest

from public_data.view_contracts import load_view_metadata, write_view_metadata


def _values(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": "annotation_view", "dataset": "coco",
        "view": "coco80/full", "image_store": "public_data/coco/images/res-64",
        "path_anchor": "repository_root", "image_path_semantics": "image_store_relative",
        "coordinate_space": "norm1000", "coordinate_storage": "integer",
        "coordinate_range": (0, 999), "coordinate_chart": "xyxy",
        "assistant_coordinate_rendering": "qwen_coord_tokens",
        "primary_jsonl": {"train": "train.jsonl"}, "summary": {"records": 1},
    }


def test_view_metadata_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    write_view_metadata(path, **_values(tmp_path))
    assert load_view_metadata(path).view == "coco80/full"


def test_view_metadata_rejects_wrong_coordinate_space(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["coordinate_space"] = "pixel"
    with pytest.raises(ValueError, match="coordinate_space"):
        write_view_metadata(tmp_path / "meta.json", **values)
