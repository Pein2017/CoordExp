from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_coord_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.coord.jsonl"
    rows = [
        {
            "images": ["images/a.jpg"],
            "objects": [
                {"desc": " Person ", "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_20|>", "<|coord_30|>"], "coco_ann_id": 1},
                {"desc": "person", "bbox_2d": ["<|coord_60|>", "<|coord_10|>", "<|coord_80|>", "<|coord_30|>"], "coco_ann_id": 2},
                {"desc": "PERSON", "bbox_2d": ["<|coord_120|>", "<|coord_10|>", "<|coord_140|>", "<|coord_30|>"], "coco_ann_id": 3},
                {"desc": "traffic  light", "bbox_2d": ["<|coord_220|>", "<|coord_10|>", "<|coord_230|>", "<|coord_30|>"], "coco_ann_id": 4},
                {"desc": "Traffic light", "bbox_2d": ["<|coord_260|>", "<|coord_10|>", "<|coord_280|>", "<|coord_30|>"], "coco_ann_id": 5},
            ],
            "width": 1000,
            "height": 1000,
            "image_id": 101,
            "file_name": "images/a.jpg",
            "metadata": {"split": "val"},
        },
        {
            "images": ["images/b.jpg"],
            "objects": [
                {"desc": "cat", "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_20|>", "<|coord_30|>"], "coco_ann_id": 6},
            ],
            "width": 1000,
            "height": 1000,
            "image_id": 102,
            "file_name": "images/b.jpg",
            "metadata": {"split": "val"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def minimal_base_row() -> dict[str, object]:
    return {
        "schema_version": "candidate_field_cardinality_tomography.v1",
        "project_id": "candidate_field_cardinality_tomography",
        "phase_id": "phase_a",
        "run_id": "run-test",
        "checkpoint_id": "checkpoint-3664",
        "case_id": "case-1",
        "case_index_row_id": "ci-1",
        "split": "val",
        "pool_role": "headline_crowded",
        "source_dataset_jsonl": "/tmp/val.coord.jsonl",
        "dataset_manifest_id": "manifest-test",
        "dataset_manifest_sha256": "0" * 64,
        "fn_rescue_overlay_membership": False,
    }
