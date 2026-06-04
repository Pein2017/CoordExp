from __future__ import annotations

import json
from pathlib import Path

from src.analysis.prefix_state_transition_tomography.prefix_state_index import (
    build_prefix_state_index,
    canonical_desc,
)


def test_canonical_desc_lower_strip_collapse_ws() -> None:
    assert canonical_desc(" Traffic   LIGHT ") == "traffic light"


def test_build_prefix_state_index_materializes_hard_transition_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "train.coord.jsonl"
    val_path = tmp_path / "val.coord.jsonl"
    train_path.write_text(_records_jsonl(split="train", image_id=1), encoding="utf-8")
    val_path.write_text(_records_jsonl(split="val", image_id=2), encoding="utf-8")

    rows, sampled, summary = build_prefix_state_index(
        train_jsonl=train_path,
        val_jsonl=val_path,
        run_id="test-run",
        max_prefix_states=64,
        seed=3664,
    )

    assert rows
    assert sampled
    assert summary["launch_eligible"] is True
    assert summary["failed_launch_gates"] == []
    assert any(row["transition_type"] == "same_desc_transition" for row in sampled)
    assert any(row["transition_type"] == "different_desc_transition" for row in sampled)
    assert {"shallow_1", "mid_half", "late_one_left", "class_block_done"} & {
        row["prefix_depth"] for row in sampled
    }
    assert all(row["prefix_state_id"] for row in rows)
    assert all(row["project_id"] == "prefix_state_transition_tomography" for row in rows)
    assert all(row["phase_id"] == "phase_a3_1" for row in rows)
    required_contract_fields = {
        "checkpoint_role",
        "probe_desc",
        "probe_desc_role",
        "readout_type",
        "shard_id",
    }
    assert all(required_contract_fields <= row.keys() for row in rows)
    assert all(required_contract_fields <= row.keys() for row in sampled)
    assert all(row["readout_type"] == "prefix_state_index" for row in rows)
    assert all(row["checkpoint_role"] == "paired_index" for row in rows)
    assert all(row["probe_desc_role"] == "target_residual_desc" for row in rows)
    assert all(row["shard_id"] is not None for row in sampled)
    same_hard = [
        row
        for row in sampled
        if row["transition_type"] == "same_desc_transition" and row["hardness"] == "headline_hard"
    ]
    assert same_hard
    assert all(row["residual_target_count"] >= 2 for row in same_hard)


def test_prefix_state_index_reports_failed_launch_gates_for_easy_pool(tmp_path: Path) -> None:
    train_path = tmp_path / "train.coord.jsonl"
    val_path = tmp_path / "val.coord.jsonl"
    easy = {
        "images": ["images/easy.jpg"],
        "file_name": "images/easy.jpg",
        "image_id": 10,
        "objects": [
            {
                "desc": "cat",
                "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_40|>", "<|coord_80|>"],
            }
        ],
    }
    text = json.dumps(easy) + "\n"
    train_path.write_text(text, encoding="utf-8")
    val_path.write_text(text, encoding="utf-8")

    _, sampled, summary = build_prefix_state_index(
        train_jsonl=train_path,
        val_jsonl=val_path,
        run_id="test-run",
        max_prefix_states=16,
        seed=3664,
    )

    assert sampled == []
    assert summary["launch_eligible"] is False
    assert "missing_train_same_desc_transition" in summary["failed_launch_gates"]


def _records_jsonl(*, split: str, image_id: int) -> str:
    record = {
        "images": [f"images/{split}_{image_id}.jpg"],
        "file_name": f"images/{split}_{image_id}.jpg",
        "image_id": image_id,
        "objects": [
            _obj("person", 10, 10, 30, 80),
            _obj("person", 80, 10, 110, 85),
            _obj("person", 150, 12, 180, 90),
            _obj("person", 220, 11, 250, 88),
            _obj("chair", 320, 300, 420, 460),
            _obj("chair", 450, 310, 550, 470),
            _obj("dog", 620, 500, 720, 650),
        ],
    }
    return json.dumps(record) + "\n"


def _obj(desc: str, x1: int, y1: int, x2: int, y2: int) -> dict[str, object]:
    return {
        "desc": desc,
        "bbox_2d": [
            f"<|coord_{x1}|>",
            f"<|coord_{y1}|>",
            f"<|coord_{x2}|>",
            f"<|coord_{y2}|>",
        ],
    }
