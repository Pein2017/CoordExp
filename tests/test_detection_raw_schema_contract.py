from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.data import CoordinateTokenBox, parse_raw_detection_row


DATASET = Path("public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl")
MANIFEST = Path(
    "manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60.json"
)
LEGACY_MAX60_PATH = "public_data/coco/rescale_32_1024_bbox_max60"
VAL_COORD_PATH = f"{LEGACY_MAX60_PATH}/val.coord.jsonl"
TRAIN_COORD_PATH = f"{LEGACY_MAX60_PATH}/train.coord.jsonl"


def _legacy_coord_token_row() -> dict:
    return {
        "file_name": "images/val2017/000000000139.jpg",
        "height": 832,
        "image_id": 139,
        "images": ["images/val2017/000000000139.jpg"],
        "metadata": {"source": "coco2017", "split": "val"},
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_699|>",
                    "<|coord_284|>",
                    "<|coord_722|>",
                    "<|coord_336|>",
                ],
                "category_id": 85,
                "category_name": "clock",
                "coco_ann_id": 1666628,
                "desc": "clock",
            }
        ],
        "width": 1248,
    }


def _first_raw_row() -> dict:
    if not DATASET.exists():
        return _legacy_coord_token_row()
    with DATASET.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _manifest_file_by_path(manifest: dict, path: str) -> dict:
    for entry in manifest["checksums"]["files"]:
        if entry["path"] == path:
            return entry
    raise AssertionError(f"manifest is missing checksum entry for {path}")


def test_parse_raw_detection_row_preserves_legacy_coord_token_schema_and_geometry() -> None:
    raw = parse_raw_detection_row(_legacy_coord_token_row())

    assert raw.file_name == "images/val2017/000000000139.jpg"
    assert raw.images == ("images/val2017/000000000139.jpg",)
    assert raw.image_id == 139
    assert raw.width == 1248
    assert raw.height == 832
    assert raw.metadata.source == "coco2017"
    assert raw.metadata.split == "val"
    assert raw.objects

    first = raw.objects[0]
    assert first.source_object_index == 0
    assert first.desc == "clock"
    assert first.category_id == 85
    assert first.category_name == "clock"
    assert first.coco_ann_id == 1666628
    assert first.bbox_2d == CoordinateTokenBox(
        "<|coord_699|>",
        "<|coord_284|>",
        "<|coord_722|>",
        "<|coord_336|>",
    )
    assert first.bbox_2d.tokens == (
        "<|coord_699|>",
        "<|coord_284|>",
        "<|coord_722|>",
        "<|coord_336|>",
    )
    assert first.bbox_2d.values == (699, 284, 722, 336)
    assert first.object_id is None


def test_legacy_max60_manifest_records_public_data_provenance() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["relative_path"] == LEGACY_MAX60_PATH
    assert manifest["key_params"]["max_objects"] == 60

    val_coord = _manifest_file_by_path(manifest, VAL_COORD_PATH)
    assert val_coord["path"] == VAL_COORD_PATH
    assert val_coord["records"] == 4_951

    train_coord = _manifest_file_by_path(manifest, TRAIN_COORD_PATH)
    assert train_coord["path"] == TRAIN_COORD_PATH
    assert train_coord["records"] == 117_247


def test_materialized_legacy_max60_val_coord_jsonl_matches_known_schema_counts() -> None:
    if not DATASET.exists():
        pytest.skip(
            f"{DATASET} is not materialized in this worktree; this test only "
            "verifies local materialized public_data, while the manifest sentinel "
            "remains the default provenance check"
        )

    rows = 0
    object_count = 0
    min_objects = None
    max_objects = 0

    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = parse_raw_detection_row(json.loads(line))
            rows += 1
            object_count += len(raw.objects)
            min_objects = (
                len(raw.objects)
                if min_objects is None
                else min(min_objects, len(raw.objects))
            )
            max_objects = max(max_objects, len(raw.objects))

    assert rows == 4_951
    assert object_count == 36_273
    assert min_objects == 1
    assert max_objects == 55


def test_parse_raw_detection_row_rejects_missing_top_level_schema_key() -> None:
    row = _first_raw_row()
    row.pop("metadata")

    with pytest.raises(ValueError, match="missing top-level keys: metadata"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_unknown_top_level_schema_key() -> None:
    row = _first_raw_row()
    row["dataset"] = "coco"

    with pytest.raises(ValueError, match="unsupported top-level keys: dataset"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_empty_images() -> None:
    row = _first_raw_row()
    row["images"] = []

    with pytest.raises(ValueError, match="images must contain at least one image"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_unknown_object_schema_key() -> None:
    row = _first_raw_row()
    row["objects"][0]["area"] = 123

    with pytest.raises(ValueError, match=r"objects\[0\].*unsupported object keys: area"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_non_compact_v1_box() -> None:
    row = _first_raw_row()
    row["objects"][0]["bbox_2d"] = ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>"]

    with pytest.raises(ValueError, match=r"objects\[0\]\.bbox_2d.*exactly four"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_non_coordinate_token_box_item() -> None:
    row = _first_raw_row()
    row["objects"][0]["bbox_2d"] = ["<|coord_1|>", "<|coord_2|>", "3", "<|coord_4|>"]

    with pytest.raises(ValueError, match=r"objects\[0\]\.bbox_2d\[2\].*coordinate token"):
        parse_raw_detection_row(row)


def test_parse_raw_detection_row_rejects_leading_zero_coordinate_token() -> None:
    row = _first_raw_row()
    row["objects"][0]["bbox_2d"] = [
        "<|coord_001|>",
        "<|coord_2|>",
        "<|coord_3|>",
        "<|coord_4|>",
    ]

    with pytest.raises(ValueError, match=r"objects\[0\]\.bbox_2d\[0\].*canonical"):
        parse_raw_detection_row(row)


@pytest.mark.parametrize(
    "bbox",
    [
        ["<|coord_4|>", "<|coord_2|>", "<|coord_3|>", "<|coord_5|>"],
        ["<|coord_1|>", "<|coord_6|>", "<|coord_3|>", "<|coord_5|>"],
    ],
)
def test_parse_raw_detection_row_rejects_inverted_xyxy_coord_token_box(
    bbox: list[str],
) -> None:
    row = _first_raw_row()
    row["objects"][0]["bbox_2d"] = bbox

    with pytest.raises(ValueError, match=r"objects\[0\]\.bbox_2d.*xyxy"):
        parse_raw_detection_row(row)
