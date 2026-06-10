from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.analysis.sorted_random_no_newline_phenotype.config import load_config
from src.analysis.sorted_random_no_newline_phenotype.prefix_index import (
    build_prefix_state_index,
    canonical_sorted_objects,
)


def test_canonical_sorted_objects_catches_x_first_mistakes() -> None:
    objects = [
        {"gt_idx": 0, "desc": "person", "bbox_xyxy": [900, 10, 950, 80]},
        {"gt_idx": 1, "desc": "person", "bbox_xyxy": [20, 500, 80, 580]},
        {"gt_idx": 2, "desc": "chair", "bbox_xyxy": [40, 20, 100, 100]},
    ]

    ordered = canonical_sorted_objects(objects)

    assert [obj["gt_idx"] for obj in ordered] == [0, 2, 1]


def test_build_prefix_state_index_materializes_a3_2_rows(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            _hard_record(image_id=1, split="train"),
            _easy_record(image_id=2, split="train"),
        ],
    )
    _write_jsonl(
        val_jsonl,
        [
            _hard_record(image_id=3, split="val"),
            _malformed_record(image_id=4, split="val"),
        ],
    )
    config = load_config(_write_config(tmp_path, train_jsonl, val_jsonl))

    rows, sampled_rows, summary, sample_manifest = build_prefix_state_index(config)
    rows_again, sampled_rows_again, summary_again, sample_manifest_again = (
        build_prefix_state_index(config)
    )

    assert rows
    assert sampled_rows
    assert [row["prefix_state_id"] for row in sampled_rows] == [
        row["prefix_state_id"] for row in sampled_rows_again
    ]
    assert len({row["prefix_state_id"] for row in sampled_rows}) == len(sampled_rows)
    assert summary["launch_eligible"] is True
    assert summary["failed_launch_gates"] == []
    assert summary["total_rows"] == len(rows)
    assert summary["sampled_rows"] == len(sampled_rows)
    assert summary["num_shards"] == config.sampling.num_shards
    assert summary["evidence_scope"] == (
        "a3_2_prefix4096_hardbiased_len12000_canonical_sorted"
    )
    assert summary["template_contract"] == {"row_separator": "none"}
    assert summary["prefix_order_policy_id"] == "canonical_sorted_yx_teacher_v1"

    row = sampled_rows[0]
    assert row["project_id"] == "sorted_random_no_newline_phenotype"
    assert row["phase_id"] == "phase_a3_2"
    assert row["schema_version"] == "a3.2.v1"
    assert row["run_id"] == "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2"
    assert row["evidence_scope"] == (
        "a3_2_prefix4096_hardbiased_len12000_canonical_sorted"
    )
    assert row["prefix_source_policy"] == "canonical_sorted_teacher_prefix_readout"
    assert row["prefix_order_policy_id"] == "canonical_sorted_yx_teacher_v1"
    assert row["readout_prompt_ordering"] == "sorted"
    assert row["template_contract"] == {"row_separator": "none"}
    assert row["selected_for_probe"] is True
    assert row["planned_shard_id"] == row["shard_id"]
    assert 0 <= row["shard_id"] < config.sampling.num_shards
    assert row["canonical_sorted_gt_indices"] == [
        obj["gt_idx"] for obj in canonical_sorted_objects(row["gt_objects"])
    ]
    assert set(row["emitted_gt_indices"]) | set(row["residual_gt_indices"]) == set(
        row["canonical_sorted_gt_indices"]
    )
    assert row["candidate_descs"]
    assert row["candidate_descs_with_roles"]
    assert all(
        {"desc", "roles", "gt_indices"} <= candidate.keys()
        for candidate in row["candidate_descs_with_roles"]
    )

    assert row["rendered_prefix_sha256"]
    assert len(row["rendered_prefix_sha256"]) == 64
    assert row["rendered_prefix_char_len"] == 0 or row["rendered_prefix_char_len"] > 10
    assert "rendered_prefix" not in row
    assert any(
        {"emitted_same_desc", "residual_same_desc"} <= set(candidate["roles"])
        for sampled_row in sampled_rows
        for candidate in sampled_row["candidate_descs_with_roles"]
    )

    assert sample_manifest["available_by_split"]["train"] >= 6
    assert sample_manifest["available_by_split"]["val"] >= 6
    assert sample_manifest["selected_by_split"]
    assert sample_manifest["easy_sanity_selected_fraction"] <= 0.20
    assert "underfill_reasons" in sample_manifest
    json.dumps(rows)
    json.dumps(sampled_rows)
    json.dumps(summary)
    json.dumps(sample_manifest)
    assert summary == summary_again
    assert sample_manifest == sample_manifest_again


def test_non_selected_rows_stay_lightweight_for_high_object_records(
    tmp_path: Path,
) -> None:
    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    _write_jsonl(train_jsonl, [_large_hard_record(image_id=101, split="train")])
    _write_jsonl(val_jsonl, [_large_hard_record(image_id=102, split="val")])
    config = load_config(
        _write_config(tmp_path, train_jsonl, val_jsonl, max_prefix_states=4)
    )

    rows, sampled_rows, _, _ = build_prefix_state_index(config)

    assert len(rows) > len(sampled_rows)
    assert all("gt_objects" in row for row in sampled_rows)
    assert all("candidate_descs_with_roles" in row for row in sampled_rows)
    assert all("rendered_prefix_sha256" in row for row in sampled_rows)
    non_selected = [row for row in rows if not row["selected_for_probe"]]
    assert non_selected
    assert all("gt_objects" not in row for row in non_selected)
    assert all("candidate_descs_with_roles" not in row for row in non_selected)
    assert all("rendered_prefix_sha256" not in row for row in non_selected)


def test_image_ref_uses_later_valid_images_entry(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    record = _hard_record(image_id=9, split="train")
    record.pop("file_name")
    record["images"] = [
        123,
        {"not_file_name": "ignored.jpg"},
        {"file_name": "images/train/later-valid.jpg"},
    ]
    _write_jsonl(train_jsonl, [record])
    _write_jsonl(val_jsonl, [_hard_record(image_id=10, split="val")])
    config = load_config(_write_config(tmp_path, train_jsonl, val_jsonl))

    rows, sampled_rows, _, _ = build_prefix_state_index(config)

    assert any(row["image_path"] == "images/train/later-valid.jpg" for row in rows)
    assert any(
        row["image_path"] == "images/train/later-valid.jpg" for row in sampled_rows
    )


def test_build_prefix_state_index_skips_malformed_objects(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            {
                "file_name": "images/train/invalid.jpg",
                "image_id": 10,
                "objects": [
                    123,
                    _obj("person", 10, 20, 50, 90),
                    {"bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]},
                    {"desc": "chair", "bbox_2d": ["<|coord_1|>", "bad-token"]},
                    {"desc": "lamp", "bbox_2d": ["<|coord_1000|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]},
                ],
            }
        ],
    )
    _write_jsonl(val_jsonl, [_hard_record(image_id=11, split="val")])
    config = load_config(_write_config(tmp_path, train_jsonl, val_jsonl))

    rows, sampled_rows, summary, sample_manifest = build_prefix_state_index(config)

    assert rows
    assert sampled_rows
    assert summary["total_rows"] == len(rows)
    assert sample_manifest["skipped_object_count"] == 4
    assert sample_manifest["malformed_bbox_object_count"] == 2
    assert sample_manifest["missing_desc_object_count"] == 1
    assert sample_manifest["non_mapping_object_count"] == 1


def _write_config(
    tmp_path: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    *,
    max_prefix_states: int = 12,
) -> Path:
    random_checkpoint = tmp_path / "random" / "checkpoint-3668"
    sorted_checkpoint = tmp_path / "sorted" / "checkpoint-3668"
    random_checkpoint.mkdir(parents=True)
    sorted_checkpoint.mkdir(parents=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_id": "sorted_random_no_newline_phenotype",
                "phase_id": "phase_a3_2",
                "schema_version": "a3.2.v1",
                "run_id": "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2",
                "artifact_root": str(tmp_path / "artifacts"),
                "train_jsonl": str(train_jsonl),
                "val_jsonl": str(val_jsonl),
                "image_root": str(tmp_path / "rescale_32_1024_bbox"),
                "template_contract": {
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "none",
                },
                "sampling": {
                    "max_prefix_states": max_prefix_states,
                    "num_shards": 8,
                    "seed": 3668,
                    "easy_sanity_max_fraction": 0.20,
                },
                "rollout": {
                    "limit_images": 1024,
                    "decode_policy": "free_text_unconstrained_greedy_temp0",
                    "native_prompt_ordering": True,
                },
                "fn_probe": {
                    "max_fn_objects_per_checkpoint": 512,
                    "hint_policy_id": "desc_x1_r95_ladder_v1",
                    "strict_r95_axis_fraction": 0.04,
                    "strict_r95_cap_bins": 8,
                    "broad_x1_radius": 24,
                },
                "peak": {
                    "absolute_mass_floor": 0.002,
                    "relative_floor": 0.10,
                    "primary_merge_radius": 24,
                    "gt_x1_neighborhood_radius": 24,
                    "raw_topk_k": 32,
                },
                "checkpoints": {
                    "fullobj_random_pure_ce_ckpt3668": {
                        "training_ordering": "random_permutation",
                        "readout_prompt_ordering": "sorted",
                        "checkpoint_path": str(random_checkpoint),
                    },
                    "fullobj_sorted_pure_ce_ckpt3668": {
                        "training_ordering": "sorted",
                        "readout_prompt_ordering": "sorted",
                        "checkpoint_path": str(sorted_checkpoint),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _hard_record(*, image_id: int, split: str) -> dict[str, Any]:
    return {
        "file_name": f"images/{split}/{image_id:012d}.jpg",
        "image_id": image_id,
        "objects": [
            _obj(" Person ", 900, 10, 950, 80),
            _obj("person", 20, 500, 80, 580),
            _obj("chair", 40, 20, 100, 100),
            _obj("chair", 160, 30, 220, 130),
            _obj("traffic   light", 300, 300, 360, 420),
            _obj("cup", 520, 350, 560, 410),
        ],
    }


def _easy_record(*, image_id: int, split: str) -> dict[str, Any]:
    return {
        "file_name": f"images/{split}/{image_id:012d}.jpg",
        "image_id": image_id,
        "objects": [
            _obj("cat", 15, 20, 45, 80),
            _obj("dog", 70, 25, 120, 95),
        ],
    }


def _malformed_record(*, image_id: int, split: str) -> dict[str, Any]:
    return {
        "file_name": f"images/{split}/{image_id:012d}.jpg",
        "image_id": image_id,
        "objects": [
            _obj("person", 10, 10, 40, 90),
            {"desc": "person", "bbox_2d": ["<|coord_20|>", "<|coord_bad|>"]},
            _obj("chair", 60, 15, 120, 100),
            _obj("chair", 140, 30, 190, 120),
            _obj("book", 220, 200, 260, 250),
            _obj("cup", 300, 250, 340, 310),
            _obj("cup", 360, 255, 400, 320),
        ],
    }


def _large_hard_record(*, image_id: int, split: str) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for idx in range(60):
        desc = f"class {idx % 6}"
        x1 = 10 + (idx % 10) * 40
        y1 = 20 + idx * 10
        objects.append(_obj(desc, x1, y1, x1 + 20, y1 + 30))
    return {
        "file_name": f"images/{split}/{image_id:012d}.jpg",
        "image_id": image_id,
        "objects": objects,
    }


def _obj(desc: str, x1: int, y1: int, x2: int, y2: int) -> dict[str, Any]:
    return {
        "desc": desc,
        "bbox_2d": [
            f"<|coord_{x1}|>",
            f"<|coord_{y1}|>",
            f"<|coord_{x2}|>",
            f"<|coord_{y2}|>",
        ],
    }
