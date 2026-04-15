from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from public_data.scripts.validate_jsonl import JSONLValidator
from src.common.geometry.bbox_parameterization import (
    CXCY_LOGW_LOGH_CONVERSION_VERSION,
    CXCY_LOGW_LOGH_SLOT_ORDER,
    CXCYWH_CONVERSION_VERSION,
    CXCYWH_SLOT_ORDER,
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


def _cxcy_logw_logh_row(
    *,
    with_metadata: bool,
    coord_tokens: bool,
    source_xyxy: list[int] | None = None,
) -> dict:
    bins = xyxy_norm1000_to_cxcy_logw_logh_bins(source_xyxy or [100, 200, 400, 700])
    bbox = [f"<|coord_{value}|>" for value in bins] if coord_tokens else bins
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [{"bbox_2d": bbox, "desc": "person"}],
    }
    if with_metadata:
        row["metadata"] = {
            "prepared_bbox_format": "cxcy_logw_logh",
            "prepared_bbox_slot_order": CXCY_LOGW_LOGH_SLOT_ORDER,
            "prepared_bbox_source_format": "xyxy",
            "prepared_bbox_conversion_version": CXCY_LOGW_LOGH_CONVERSION_VERSION,
        }
    return row


def _cxcywh_row(
    *,
    with_metadata: bool,
    coord_tokens: bool,
    source_xyxy: list[int] | None = None,
) -> dict:
    bins = xyxy_norm1000_to_cxcywh_bins(source_xyxy or [100, 200, 400, 700])
    bbox = [f"<|coord_{value}|>" for value in bins] if coord_tokens else bins
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [{"bbox_2d": bbox, "desc": "person"}],
    }
    if with_metadata:
        row["metadata"] = {
            "prepared_bbox_format": "cxcywh",
            "prepared_bbox_slot_order": CXCYWH_SLOT_ORDER,
            "prepared_bbox_source_format": "xyxy",
            "prepared_bbox_conversion_version": CXCYWH_CONVERSION_VERSION,
        }
    return row


def test_validate_jsonl_accepts_explicit_cxcy_logw_logh_bbox_format(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _cxcy_logw_logh_row(
        with_metadata=False,
        coord_tokens=True,
        source_xyxy=[700, 100, 900, 300],
    )
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(
        check_images=True,
        check_image_sizes=True,
        bbox_format="cxcy_logw_logh",
    )

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_auto_detects_prepared_bbox_format_from_metadata(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _cxcy_logw_logh_row(with_metadata=True, coord_tokens=True)
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(check_images=True, check_image_sizes=True)

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_default_xyxy_still_rejects_cxcy_logw_logh_without_hint(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _cxcy_logw_logh_row(
        with_metadata=False,
        coord_tokens=True,
        source_xyxy=[700, 100, 900, 300],
    )
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(check_images=False)

    assert not validator.validate_file(str(jsonl_path))
    assert any("Invalid: x2" in error.message or "Invalid: y2" in error.message for error in validator.errors)


def test_validate_jsonl_accepts_explicit_cxcywh_bbox_format(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _cxcywh_row(
        with_metadata=False,
        coord_tokens=True,
        source_xyxy=[700, 100, 900, 300],
    )
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(
        check_images=True,
        check_image_sizes=True,
        bbox_format="cxcywh",
    )

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_auto_detects_prepared_cxcywh_from_metadata(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.coord.jsonl"
    row = _cxcywh_row(with_metadata=True, coord_tokens=True)
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(check_images=True, check_image_sizes=True)

    assert validator.validate_file(str(jsonl_path))
