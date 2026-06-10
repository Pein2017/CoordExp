from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.analysis.sorted_random_no_newline_phenotype.config import load_config
from src.analysis.sorted_random_no_newline_phenotype.data_root_audit import (
    build_data_root_audit,
)


def test_build_data_root_audit_summarizes_jsonls_and_missing_images(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "rescale_32_1024_bbox"
    train_image_dir = image_root / "images" / "train2017"
    val_image_dir = image_root / "images" / "val2017"
    train_image_dir.mkdir(parents=True)
    val_image_dir.mkdir(parents=True)
    (train_image_dir / "000000000001.jpg").write_bytes(b"one")
    (train_image_dir / "000000000002.jpg").write_bytes(b"two")
    (val_image_dir / "000000000003.jpg").write_bytes(b"three")

    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            {
                "file_name": "images/train2017/000000000001.jpg",
                "images": ["images/train2017/missing-but-ignored.jpg"],
                "objects": [
                    {"desc": "orange"},
                    {"desc": "orange"},
                    {"desc": "cup"},
                ],
            },
            {
                "images": [
                    "../rescale_32_1024_bbox/images/train2017/000000000002.jpg"
                ],
                "objects": [{"category_name": "person"}],
            },
        ],
    )
    _write_jsonl(
        val_jsonl,
        [
            {
                "images": [{"file_name": "images/val2017/000000000003.jpg"}],
                "objects": [],
            },
            {
                "file_name": "images/val2017/000000009999.jpg",
                "images": [123],
                "objects": [{"desc": "dog"}],
            },
        ],
    )
    config = load_config(_write_config(tmp_path, train_jsonl, val_jsonl, image_root))

    audit = build_data_root_audit(config)

    assert audit["status"] == "missing_images"
    assert audit["actual_train_jsonl"] == str(train_jsonl)
    assert audit["actual_val_jsonl"] == str(val_jsonl)
    assert audit["image_root"] == str(image_root)
    assert audit["evidence_scope"] == "len12000-jsonl-local-mechanism-probe"
    assert audit["row_counts"] == {"train": 2, "val": 2}
    assert audit["jsonl_sha256"] == {
        "train": _sha256(train_jsonl),
        "val": _sha256(val_jsonl),
    }
    assert audit["object_count_histogram"] == {"0": 1, "1": 2, "3": 1}
    assert audit["desc_count_histogram"] == {"0": 1, "1": 2, "2": 1}
    assert audit["same_desc_multi_instance_count"] == 1
    assert audit["missing_image_examples"] == [
        {
            "split": "val",
            "row_index": 2,
            "image_ref": "images/val2017/000000009999.jpg",
            "resolved_path": str(image_root / "images/val2017/000000009999.jpg"),
        }
    ]
    json.dumps(audit)


def test_build_data_root_audit_does_not_hard_fail_on_unexpected_images_shape(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "rescale_32_1024_bbox"
    train_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "train.coord.jsonl"
    val_jsonl = tmp_path / "rescale_32_1024_bbox_len12000" / "val.coord.jsonl"
    _write_jsonl(train_jsonl, [{"images": [123], "objects": []}])
    _write_jsonl(val_jsonl, [{"images": "not-a-list", "objects": []}])
    config = load_config(_write_config(tmp_path, train_jsonl, val_jsonl, image_root))

    audit = build_data_root_audit(config)

    assert audit["row_counts"] == {"train": 1, "val": 1}
    assert audit["missing_image_examples"] == []


def _write_config(
    tmp_path: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    image_root: Path,
) -> Path:
    random_checkpoint = tmp_path / "random" / "checkpoint-3668"
    sorted_checkpoint = tmp_path / "sorted" / "checkpoint-3668"
    random_checkpoint.mkdir(parents=True, exist_ok=True)
    sorted_checkpoint.mkdir(parents=True, exist_ok=True)

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
                "image_root": str(image_root),
                "template_contract": {
                    "detection_sequence_format": "compact_full",
                    "coordinate_surface": "coord_token",
                    "bbox_format": "xyxy",
                    "row_separator": "none",
                },
                "sampling": {
                    "max_prefix_states": 4096,
                    "num_shards": 8,
                    "seed": 3668,
                    "easy_sanity_max_fraction": 0.20,
                },
                "rollout": {
                    "limit_images": 1024,
                    "decode_policy": "free_text_unconstrained_greedy_temp0",
                    "native_prompt_ordering": True,
                    "constraint_policy": "none",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
