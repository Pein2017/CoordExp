from __future__ import annotations

from pathlib import Path

from src.analysis.post_x1_instance_basin_tomography.case_universe import (
    build_case_universe_rows,
    normalize_sample_objects,
)


def test_normalize_sample_objects_accepts_bbox_2d_coord_tokens(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "train.coord.jsonl"
    source_jsonl.write_text('{"image_id": "demo"}\n', encoding="utf-8")
    sample = {
        "image_id": "demo",
        "image": "train/demo.jpg",
        "objects": [
            {
                "desc": " Person ",
                "bbox_2d": [
                    "<|coord_010|>",
                    "<|coord_020|>",
                    "<|coord_110|>",
                    "<|coord_220|>",
                ],
            },
        ],
    }

    objects = normalize_sample_objects(
        sample,
        split="train",
        source_line_id=7,
        source_jsonl_path=source_jsonl,
    )

    assert objects[0]["desc"] == "person"
    assert objects[0]["bbox_coord_token_xyxy"] == [10, 20, 110, 220]
    assert objects[0]["bbox_surface"] == "bbox_2d_coord_token_xyxy"
    assert objects[0]["bbox_source_field"] == "bbox_2d"
    assert objects[0]["source_line_id"] == 7
    assert objects[0]["source_jsonl_path"] == str(source_jsonl)
    assert len(objects[0]["source_jsonl_sha256"]) == 64


def test_normalize_sample_objects_accepts_numeric_fixture_surfaces() -> None:
    sample = {"objects": [{"desc": "person", "bbox": [10, 20, 110, 220]}]}

    objects = normalize_sample_objects(sample, split="unit", source_line_id=0)

    assert objects[0]["bbox_coord_token_xyxy"] == [10, 20, 110, 220]
    assert objects[0]["bbox_surface"] == "numeric_xyxy_fixture"
    assert objects[0]["bbox_source_field"] == "bbox"


def test_case_universe_selects_same_desc_targets_and_competitors(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "train.coord.jsonl"
    source_jsonl.write_text('{"image_id": "demo"}\n', encoding="utf-8")
    sample = {
        "image_id": "demo",
        "image": "train/demo.jpg",
        "width": 1000,
        "height": 800,
        "objects": [
            {"desc": "person", "bbox": [100, 100, 200, 300]},
            {"desc": "person", "bbox": [300, 120, 420, 320]},
            {"desc": "person", "bbox": [600, 130, 760, 340]},
            {"desc": "chair", "bbox": [50, 500, 160, 700]},
            {"desc": "table", "bbox": [400, 500, 700, 760]},
            {"desc": "cup", "bbox": [710, 510, 750, 570]},
        ],
    }

    rows = build_case_universe_rows(
        [sample],
        split="train",
        max_images=1,
        max_target_instances=12,
        source_jsonl_path=source_jsonl,
    )

    target_rows = [row for row in rows if row["desc"] == "person"]
    assert len(target_rows) == 3
    for row in target_rows:
        assert row["same_desc_count"] == 3
        assert row["target_gt_idx"] in {0, 1, 2}
        assert set(row["competitor_gt_indices"]) == (
            {0, 1, 2} - {row["target_gt_idx"]}
        )
        assert row["primary_basin_label_source"] == "same_desc_gt_instances"
        assert row["bbox_coord_token_xyxy"] == row["target_bbox_coord_token_xyxy"]
        assert row["bbox_surface"] == row["target_bbox_surface"]
        assert row["bbox_source_field"] == row["target_bbox_source_field"]
        assert row["source_line_id"] == 0
        assert row["source_jsonl_path"] == str(source_jsonl)
        assert len(row["source_jsonl_sha256"]) == 64
        assert row["image_id"] == "demo"
        assert row["image_path"] == "train/demo.jpg"


def test_x1_anchor_r95_flags_near_and_exact_collisions() -> None:
    sample = {
        "image_id": "ambiguous",
        "image": "train/ambiguous.jpg",
        "width": 1000,
        "height": 800,
        "objects": [
            # Width 200 gives R95=floor(min(8, 8)) = 8.
            {"desc": "person", "bbox": [100, 100, 300, 300]},
            {"desc": "person", "bbox": [107, 120, 307, 320]},
            {"desc": "person", "bbox": [100, 400, 300, 600]},
        ],
    }

    rows = build_case_universe_rows(
        [sample],
        split="unit",
        max_images=1,
        max_target_instances=12,
    )
    by_gt_idx = {row["target_gt_idx"]: row for row in rows}

    assert by_gt_idx[0]["anchor_ambiguity_bucket"] == "exact_x1_collision"
    assert by_gt_idx[0]["x1_anchor_unique_under_r95"] is False
    assert by_gt_idx[0]["primary_denominator_eligible"] is False

    assert by_gt_idx[1]["anchor_ambiguity_bucket"] == "near_collision"
    assert by_gt_idx[1]["x1_anchor_unique_under_r95"] is False
    assert by_gt_idx[1]["primary_denominator_eligible"] is False
