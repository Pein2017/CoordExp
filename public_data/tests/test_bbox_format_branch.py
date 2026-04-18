from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from public_data.scripts.convert_to_coord_tokens import convert_record_to_ints
from public_data.scripts.derive_bbox_format_branch import derive_bbox_format_branch
from src.common.geometry.bbox_parameterization import (
    CXCY_LOGW_LOGH_CONVERSION_VERSION,
    CXCY_LOGW_LOGH_SLOT_ORDER,
    CXCYWH_CONVERSION_VERSION,
    CXCYWH_SLOT_ORDER,
    cxcy_logw_logh_norm1000_to_xyxy_norm1000,
    cxcywh_norm1000_to_xyxy_norm1000,
    xyxy_norm1000_to_cxcy_logw_logh_bins,
    xyxy_norm1000_to_cxcywh_bins,
)


def _write_image(path: Path, width: int = 128, height: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(120, 140, 160)).save(path, format="JPEG")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _canonical_rows(split: str) -> list[dict]:
    return [
        {
            "images": [f"images/{split}2017/000000000001.jpg"],
            "objects": [
                {"bbox_2d": [10, 12, 80, 60], "desc": "person"},
                {"bbox_2d": [20, 16, 40, 50], "desc": "bag"},
            ],
            "width": 128,
            "height": 96,
        }
    ]


def _setup_canonical_preset(preset_dir: Path) -> None:
    for split in ("train", "val"):
        rows = _canonical_rows(split)
        _write_jsonl(preset_dir / f"{split}.jsonl", rows)
        for row in rows:
            _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])


def test_derive_bbox_format_branch_emits_numeric_coord_and_manifest(tmp_path: Path) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_bbox"
    _setup_canonical_preset(preset_dir)

    branch_root = derive_bbox_format_branch(
        preset_dir=preset_dir,
        bbox_format="cxcy_logw_logh",
    )

    train_numeric = branch_root / "train.jsonl"
    train_norm = branch_root / "train.norm.jsonl"
    train_coord = branch_root / "train.coord.jsonl"
    val_numeric = branch_root / "val.jsonl"
    val_norm = branch_root / "val.norm.jsonl"
    manifest = branch_root / "pipeline_manifest.json"
    assert train_numeric.is_file()
    assert train_norm.is_file()
    assert train_coord.is_file()
    assert val_numeric.is_file()
    assert val_norm.is_file()
    assert manifest.is_file()
    assert branch_root.name == "rescale_32_768_bbox_cxcy_logw_logh"

    train_row = json.loads(train_numeric.read_text(encoding="utf-8").splitlines()[0])
    expected_norm = convert_record_to_ints(
        {
            "images": ["images/train2017/000000000001.jpg"],
            "objects": [{"bbox_2d": [10, 12, 80, 60], "desc": "person"}],
            "width": 128,
            "height": 96,
        },
        ["bbox_2d"],
        assume_normalized=False,
    )
    expected_bins = xyxy_norm1000_to_cxcy_logw_logh_bins(
        expected_norm["objects"][0]["bbox_2d"]
    )
    assert train_row["objects"][0]["bbox_2d"] == expected_bins
    assert train_row["metadata"]["prepared_bbox_format"] == "cxcy_logw_logh"
    assert train_row["metadata"]["prepared_bbox_slot_order"] == CXCY_LOGW_LOGH_SLOT_ORDER
    assert (
        train_row["metadata"]["prepared_bbox_conversion_version"]
        == CXCY_LOGW_LOGH_CONVERSION_VERSION
    )

    coord_row = json.loads(train_coord.read_text(encoding="utf-8").splitlines()[0])
    assert coord_row["objects"][0]["bbox_2d"] == [
        f"<|coord_{value}|>" for value in expected_bins
    ]

    derived_image = branch_root / "images/train2017/000000000001.jpg"
    source_image = preset_dir / "images/train2017/000000000001.jpg"
    assert derived_image.exists()
    assert source_image.stat().st_ino == derived_image.stat().st_ino


def test_derive_bbox_format_branch_rejects_poly_sources(tmp_path: Path) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_poly"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [{"poly": [1, 1, 2, 2, 3, 3], "desc": "triangle"}],
        "width": 128,
        "height": 96,
    }
    _write_jsonl(preset_dir / "train.jsonl", [row])
    _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])

    with pytest.raises(ValueError, match="bbox_2d-only"):
        derive_bbox_format_branch(
            preset_dir=preset_dir,
            bbox_format="cxcy_logw_logh",
        )


def test_derive_bbox_format_branch_rejects_noncanonical_sources(tmp_path: Path) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_bbox"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [{"bbox_2d": [10, 12, 80, 60], "desc": "person"}],
        "width": 128,
        "height": 96,
        "metadata": {"prepared_bbox_format": "cxcy_logw_logh"},
    }
    _write_jsonl(preset_dir / "train.jsonl", [row])
    _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])

    with pytest.raises(ValueError, match="canonical sources only"):
        derive_bbox_format_branch(
            preset_dir=preset_dir,
            bbox_format="cxcy_logw_logh",
        )


def test_derive_bbox_format_branch_accepts_canonical_coord_only_preset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "public_data" / "coco"
    base_preset_dir = dataset_dir / "rescale_32_768_bbox"
    _write_image(base_preset_dir / "images/train2017/000000000001.jpg", width=128, height=96)

    preset_dir = dataset_dir / "rescale_32_768_bbox_lvis_proxy"
    row = {
        "images": ["../rescale_32_768_bbox/images/train2017/000000000001.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_400|>",
                    "<|coord_700|>",
                ],
                "desc": "person",
            }
        ],
        "width": 128,
        "height": 96,
    }
    _write_jsonl(preset_dir / "train.coord.jsonl", [row])
    _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])

    branch_root = derive_bbox_format_branch(
        preset_dir=preset_dir,
        bbox_format="cxcy_logw_logh",
    )
    train_row = json.loads((branch_root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert train_row["metadata"]["prepared_bbox_source_surface"] == "coord"
    assert train_row["images"] == ["images/train2017/000000000001.jpg"]

    derived_image = branch_root / "images/train2017/000000000001.jpg"
    source_image = base_preset_dir / "images/train2017/000000000001.jpg"
    assert derived_image.exists()
    assert source_image.stat().st_ino == derived_image.stat().st_ino


def test_convert_record_to_ints_accepts_coord_bbox_with_assume_normalized() -> None:
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_400|>",
                    "<|coord_700|>",
                ],
                "desc": "person",
            }
        ],
        "width": 128,
        "height": 96,
    }

    converted = convert_record_to_ints(
        row,
        ["bbox_2d"],
        assume_normalized=True,
    )

    assert converted["objects"][0]["bbox_2d"] == [100, 200, 400, 700]


def test_derive_bbox_format_branch_resorts_for_decoded_xyxy_order_after_conversion(
    tmp_path: Path,
) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_bbox"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [
            {"bbox_2d": [566, 462, 605, 480], "desc": "truck"},
            {"bbox_2d": [572, 462, 604, 481], "desc": "car"},
        ],
        "width": 1000,
        "height": 1000,
    }
    _write_jsonl(preset_dir / "train.jsonl", [row])
    _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])

    branch_root = derive_bbox_format_branch(
        preset_dir=preset_dir,
        bbox_format="cxcy_logw_logh",
    )

    train_row = json.loads((branch_root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    prepared_boxes = [obj["bbox_2d"] for obj in train_row["objects"]]
    decoded_anchors = [
        tuple(
            float(v)
            for v in (
                cxcy_logw_logh_norm1000_to_xyxy_norm1000(bbox)[1],
                cxcy_logw_logh_norm1000_to_xyxy_norm1000(bbox)[0],
            )
        )
        for bbox in prepared_boxes
    ]

    assert [obj["desc"] for obj in train_row["objects"]] == ["car", "truck"]
    assert decoded_anchors == sorted(decoded_anchors)


def test_derive_bbox_format_branch_emits_cxcywh_numeric_coord_and_manifest(tmp_path: Path) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_bbox"
    _setup_canonical_preset(preset_dir)

    branch_root = derive_bbox_format_branch(
        preset_dir=preset_dir,
        bbox_format="cxcywh",
    )

    train_numeric = branch_root / "train.jsonl"
    train_coord = branch_root / "train.coord.jsonl"
    manifest = branch_root / "pipeline_manifest.json"
    assert train_numeric.is_file()
    assert train_coord.is_file()
    assert manifest.is_file()
    assert branch_root.name == "rescale_32_768_bbox_cxcywh"

    train_row = json.loads(train_numeric.read_text(encoding="utf-8").splitlines()[0])
    expected_norm = convert_record_to_ints(
        {
            "images": ["images/train2017/000000000001.jpg"],
            "objects": [{"bbox_2d": [10, 12, 80, 60], "desc": "person"}],
            "width": 128,
            "height": 96,
        },
        ["bbox_2d"],
        assume_normalized=False,
    )
    expected_bins = xyxy_norm1000_to_cxcywh_bins(expected_norm["objects"][0]["bbox_2d"])
    assert train_row["objects"][0]["bbox_2d"] == expected_bins
    assert train_row["metadata"]["prepared_bbox_format"] == "cxcywh"
    assert train_row["metadata"]["prepared_bbox_slot_order"] == CXCYWH_SLOT_ORDER
    assert (
        train_row["metadata"]["prepared_bbox_conversion_version"]
        == CXCYWH_CONVERSION_VERSION
    )

    coord_row = json.loads(train_coord.read_text(encoding="utf-8").splitlines()[0])
    assert coord_row["objects"][0]["bbox_2d"] == [
        f"<|coord_{value}|>" for value in expected_bins
    ]


def test_derive_bbox_format_branch_resorts_cxcywh_for_decoded_xyxy_order_after_conversion(
    tmp_path: Path,
) -> None:
    preset_dir = tmp_path / "public_data" / "coco" / "rescale_32_768_bbox"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "objects": [
            {"bbox_2d": [566, 462, 605, 480], "desc": "truck"},
            {"bbox_2d": [572, 462, 604, 481], "desc": "car"},
        ],
        "width": 1000,
        "height": 1000,
    }
    _write_jsonl(preset_dir / "train.jsonl", [row])
    _write_image(preset_dir / row["images"][0], width=row["width"], height=row["height"])

    branch_root = derive_bbox_format_branch(
        preset_dir=preset_dir,
        bbox_format="cxcywh",
    )

    train_row = json.loads((branch_root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    prepared_boxes = [obj["bbox_2d"] for obj in train_row["objects"]]
    decoded_anchors = [
        tuple(
            float(v)
            for v in (
                cxcywh_norm1000_to_xyxy_norm1000(bbox)[1],
                cxcywh_norm1000_to_xyxy_norm1000(bbox)[0],
            )
        )
        for bbox in prepared_boxes
    ]

    assert [obj["desc"] for obj in train_row["objects"]] == ["car", "truck"]
    assert decoded_anchors == sorted(decoded_anchors)
