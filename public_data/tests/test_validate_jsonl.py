from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def _write_view_meta(path: Path, image_store: Path) -> None:
    meta = {
        "schema_version": 1,
        "kind": "annotation_view",
        "dataset": "unit",
        "view": "unit/full",
        "image_store": str(image_store),
        "path_anchor": "test_absolute",
        "image_path_semantics": "image_store_relative",
        "coordinate_space": "norm1000",
        "coordinate_storage": "integer",
        "coordinate_range": [0, 999],
        "coordinate_chart": "xyxy",
        "assistant_coordinate_rendering": "qwen_coord_tokens",
        "primary_jsonl": {"train": "train.jsonl"},
        "summary": {
            "records": 1,
            "rendered_object_count": 1,
            "rendered_proxy_candidate_count": 0,
            "support_sidecar_count": 0,
            "object_supervision_count": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta), encoding="utf-8")


def _canonical_row(**overrides: object) -> dict:
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [
            {
                "object_id": "coco:1:0",
                "bbox_2d": [100, 200, 400, 700],
                "desc": "person",
            }
        ],
    }
    row.update(overrides)
    return row


def _canonical_validator(
    tmp_path: Path,
    image_store: Path,
    **kwargs: object,
) -> JSONLValidator:
    meta_path = tmp_path / "view" / "meta.json"
    _write_view_meta(meta_path, image_store)
    options = {"check_images": False, **kwargs}
    return JSONLValidator(
        view_meta_path=meta_path,
        **options,
    )


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


def test_validate_jsonl_treats_norm_jsonl_xyxy_numbers_as_norm1000(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.norm.jsonl"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [{"bbox_2d": [900, 100, 950, 300], "desc": "person"}],
    }
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(check_images=True, check_image_sizes=True)

    assert validator.validate_file(str(jsonl_path))
    assert not any("completely outside image width" in warning for warning in validator.warnings)
    assert not any("completely outside image height" in warning for warning in validator.warnings)


def test_validate_jsonl_rejects_out_of_range_norm1000_xyxy_numbers(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.norm.jsonl"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [{"bbox_2d": [1005, 100, 950, 300], "desc": "person"}],
    }
    _write_jsonl(jsonl_path, [row])
    _write_image(tmp_path / row["images"][0], width=row["width"], height=row["height"])

    validator = JSONLValidator(check_images=False)

    assert not validator.validate_file(str(jsonl_path))
    assert any("Expected norm1000 slot" in error.message for error in validator.errors)


def test_validate_jsonl_canonical_view_rejects_coord_token_strings(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": ["<|coord_100|>", 200, 400, 700],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "Coord-token strings are invalid" in error.message
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_coordinate_value_1000(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": [100, 200, 1000, 700],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "Expected bare norm1000 integer in [0,999], got 1000" in error.message
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_float_above_999(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": [100, 200, 1000.5, 700],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "Expected bare norm1000 integer" in error.message
        and "1000.5" in error.message
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_bool_coordinate(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": [False, 200, 400, 700],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "Expected bare norm1000 integer" in error.message
        and "False" in error.message
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_counts_one_invalid_bbox_per_bad_geometry(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": [1000, -1, 400.5, "<|coord_700|>"],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert validator.stats["invalid_bboxes"] == 1
    assert len(validator.errors) == 4


def test_validate_jsonl_coordinate_storage_integer_does_not_require_view_image_root(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "train.jsonl"
    row = {
        "images": ["images/train2017/000000000001.jpg"],
        "width": 128,
        "height": 96,
        "objects": [{"bbox_2d": [100, 200, 400, 700], "desc": "person"}],
    }
    _write_jsonl(jsonl_path, [row])

    validator = JSONLValidator(
        check_images=False,
        coordinate_storage="integer",
    )

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_canonical_view_accepts_valid_norm1000_integer_row(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    _write_jsonl(jsonl_path, [_canonical_row()])

    validator = _canonical_validator(tmp_path, image_store)

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_canonical_view_rejects_non_xyxy_prepared_bbox_format(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        metadata={
            "prepared_bbox_format": "cxcywh",
        },
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any("require xyxy bbox" in error.message for error in validator.errors)


def test_validate_jsonl_canonical_view_resolves_images_through_view_meta_image_store(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row()
    _write_jsonl(jsonl_path, [row])
    _write_image(
        image_store / row["images"][0],
        width=row["width"],
        height=row["height"],
    )

    validator = _canonical_validator(
        tmp_path,
        image_store,
        check_images=True,
        check_image_sizes=True,
    )

    assert validator.validate_file(str(jsonl_path))


def test_validate_jsonl_canonical_view_image_check_n_zero_checks_all_rows(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    first = _canonical_row(images=["images/train2017/000000000001.jpg"])
    second = _canonical_row(images=["images/train2017/000000000002.jpg"])
    _write_jsonl(jsonl_path, [first, second])
    _write_image(
        image_store / first["images"][0],
        width=first["width"],
        height=first["height"],
    )

    validator = _canonical_validator(
        tmp_path,
        image_store,
        check_images=True,
        image_check_mode="exists",
        image_check_n=0,
    )

    assert not validator.validate_file(str(jsonl_path))
    assert validator.stats["image_spotcheck_n"] == 2
    assert validator.stats["missing_images"] == 1


def test_validate_jsonl_canonical_view_rejects_mismatched_image_root_override(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    other_store = tmp_path / "other_store"
    meta_path = tmp_path / "view" / "meta.json"
    _write_view_meta(meta_path, image_store)

    with pytest.raises(ValueError, match="must match --view-meta image_store"):
        JSONLValidator(
            check_images=False,
            view_meta_path=meta_path,
            image_root=other_store,
        )


def test_validate_jsonl_canonical_view_requires_object_id(tmp_path: Path) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "bbox_2d": [100, 200, 400, 700],
                "desc": "person",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "require non-empty object_id" in error.message for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_missing_object_supervision_entry(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        metadata={
            "supervision": {
                "object_supervision": {
                    "coco:1:other": {"source_role": "coco_ground_truth"},
                },
            },
        },
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "missing metadata.supervision.object_supervision entry" in error.message
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_support_only_sidecar_role(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        metadata={
            "supervision": {
                "object_supervision": {
                    "coco:1:0": {"source_role": "support_anchor"},
                },
            },
        },
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any("Support-only objects are sidecars" in error.message for error in validator.errors)


def test_validate_jsonl_canonical_view_rejects_orphan_support_source_role(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        metadata={
            "supervision": {
                "object_supervision": {
                    "coco:1:0": {"source_role": "coco_ground_truth"},
                    "support:1": {"source_role": "support_anchor"},
                },
            },
        },
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "metadata.supervision.object_supervision.support:1.source_role" in error.field
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_orphan_support_target_role(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        metadata={
            "supervision": {
                "object_supervision": {
                    "coco:1:0": {"source_role": "coco_ground_truth"},
                    "support:1": {"target_role": "support_cue"},
                },
            },
        },
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any(
        "metadata.supervision.object_supervision.support:1.target_role" in error.field
        for error in validator.errors
    )


def test_validate_jsonl_canonical_view_rejects_support_only_object_marker(
    tmp_path: Path,
) -> None:
    image_store = tmp_path / "image_store"
    jsonl_path = tmp_path / "view" / "train.jsonl"
    row = _canonical_row(
        objects=[
            {
                "object_id": "coco:1:0",
                "bbox_2d": [100, 200, 400, 700],
                "desc": "person",
                "target_role": "support_anchor",
            }
        ],
    )
    _write_jsonl(jsonl_path, [row])

    validator = _canonical_validator(tmp_path, image_store)

    assert not validator.validate_file(str(jsonl_path))
    assert any("Support-only objects are sidecars" in error.message for error in validator.errors)
