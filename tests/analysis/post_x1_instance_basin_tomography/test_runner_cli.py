from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.analysis.post_x1_instance_basin_tomography.jsonl import write_jsonl
from src.analysis.post_x1_instance_basin_tomography.runner import run_stages


EXPECTED_ROLES = [
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
]


def _write_config(path: Path, artifact_root: Path, *, run_id: str = "three_ckpt_phase_a3_3_smoke") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "project_id": "post_x1_instance_basin_tomography",
                "phase_id": "phase_a3_3",
                "schema_version": "a3.3.v1",
                "run_id": run_id,
                "artifact_root": str(artifact_root),
                "train_jsonl": str(artifact_root.parent / "train.coord.jsonl"),
                "val_jsonl": str(artifact_root.parent / "val.coord.jsonl"),
                "image_root": str(artifact_root.parent / "images"),
                "checkpoints": {
                    "fullobj_random_pure_ce_ckpt3668": {
                        "checkpoint_path": "/ckpts/random/checkpoint-3668",
                        "objective_policy": "pure_ce",
                        "training_ordering": "random_permutation",
                        "comparison_role": "clean_pair",
                        "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
                        "chat_template_variant": "compact_full_no_newline_native_v1",
                        "template_contract": {
                            "template_contract_id": "compact_full_no_newline_native_v1",
                            "detection_sequence_format": "compact_full",
                            "coordinate_surface": "coord_token",
                            "bbox_format": "xyxy",
                            "row_separator": "none",
                            "contract_provenance": "user_reported_training_contract",
                        },
                    },
                    "fullobj_sorted_pure_ce_ckpt3668": {
                        "checkpoint_path": "/ckpts/sorted/checkpoint-3668",
                        "objective_policy": "pure_ce",
                        "training_ordering": "sorted",
                        "comparison_role": "clean_pair",
                        "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
                        "chat_template_variant": "compact_full_no_newline_native_v1",
                        "template_contract": {
                            "template_contract_id": "compact_full_no_newline_native_v1",
                            "detection_sequence_format": "compact_full",
                            "coordinate_surface": "coord_token",
                            "bbox_format": "xyxy",
                            "row_separator": "none",
                            "contract_provenance": "user_reported_training_contract",
                        },
                    },
                    "et_rmp_ce_ckpt3664": {
                        "checkpoint_path": "/ckpts/et/checkpoint-3664",
                        "objective_policy": "et_rmp_ce",
                        "training_ordering": "random_permutation",
                        "comparison_role": "reference_anchor",
                        "controlled_comparison_group": "reference_anchor_not_controlled",
                        "chat_template_variant": "compact_full_newline_native_v1",
                        "template_contract": {
                            "template_contract_id": "compact_full_newline_native_v1",
                            "detection_sequence_format": "compact_full",
                            "coordinate_surface": "coord_token",
                            "bbox_format": "xyxy",
                            "row_separator": "newline",
                            "contract_provenance": "legacy_compact_full_default_inferred",
                        },
                    },
                },
                "case_sampling": {
                    "max_images": 1,
                    "max_target_instances": 8,
                    "min_same_desc_count": 2,
                    "min_object_count": 2,
                    "easy_sanity_max_fraction": 0.2,
                    "seed": 3668,
                    "splits": ["train"],
                    "split_quotas": {"train": 1},
                    "desc_cap_per_split": 4,
                    "desc_count_caps": {"train": 4},
                    "object_count_buckets": ["2-4"],
                },
                "prefix": {
                    "prefix_modes_requested": ["empty", "same_desc_good_prefix"],
                    "rollout_prefix_missing_policy_smoke": "skip_with_manifest",
                    "rollout_prefix_missing_policy_full": "fail",
                },
                "posterior": {
                    "strict_r95_axis_fraction": 0.04,
                    "strict_r95_cap_bins": 8,
                    "peak_mass_floor": 0.002,
                    "low_margin_threshold": 0.25,
                    "coord_mass_low_threshold": 0.01,
                    "r95_anchor_policy": "target_axis_fraction_cap",
                },
                "greedy": {
                    "enabled": True,
                    "sample_fraction": 0.25,
                    "decode_policy": "free_text_unconstrained_greedy_temp0",
                    "constraint_policy": "none",
                },
                "runtime": {
                    "num_shards": 8,
                    "max_new_tokens": 1024,
                    "torch_dtype": "bfloat16",
                    "device_map": "single_gpu",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_runner_dry_run_lists_a3_3_stages(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)

    result = run_stages(
        config_path,
        stages="data_root_audit,case_universe,prefix_states",
        dry_run=True,
    )

    assert result["project_id"] == "post_x1_instance_basin_tomography"
    assert result["phase_id"] == "phase_a3_3"
    assert result["schema_version"] == "a3.3.v1"
    assert result["checkpoint_roles"] == EXPECTED_ROLES
    assert result["stage_results"]["case_universe"]["dry_run"] is True
    assert result["stage_results"]["prefix_states"]["planned_artifact"] == "prefix_states.jsonl"
    assert not artifact_root.exists()


def test_runner_dry_run_uses_validated_template_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["checkpoints"]["et_rmp_ce_ckpt3664"]["template_contract"][
        "row_separator"
    ] = "none"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="row_separator must be newline"):
        run_stages(config_path, stages="data_root_audit", dry_run=True)


def test_runner_cli_dry_run_returns_json_safe_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/run_post_x1_instance_basin_tomography.py",
            "--config",
            str(config_path),
            "--stages",
            "case_universe,prefix_states",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    result = json.loads(proc.stdout)
    assert result["dry_run"] is True
    assert result["checkpoint_roles"] == EXPECTED_ROLES
    assert result["stage_results"]["case_universe"]["planned_artifact"] == "case_universe.jsonl"
    assert not artifact_root.exists()


def test_runner_refuses_full_run_without_smoke_marker_or_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "full_artifacts", run_id="three_ckpt_phase_a3_3")

    with pytest.raises(PermissionError, match="smoke marker"):
        run_stages(config_path, stages="case_universe", dry_run=False)


def test_runner_exposes_gpu_not_implemented_boundary(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="mock runtime or real runtime"):
        run_stages(config_path, stages="slot_posterior", dry_run=False, shard_id=0)


def test_mock_cpu_path_materializes_stage_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    _write_config(config_path, artifact_root)

    result = run_stages(
        config_path,
        stages=(
            "data_root_audit",
            "case_universe",
            "prefix_states",
            "slot_posterior",
            "slot_merge",
            "trajectory",
            "attraction_matrix",
            "prefix_sensitivity",
            "greedy_continuation",
            "report",
            "gallery",
        ),
        dry_run=False,
        mock_runtime=True,
        allow_overwrite=True,
    )

    assert result["stage_results"]["slot_posterior"]["runtime_kind"] == "mock_cpu_slot_posterior_v1"
    assert (artifact_root / "case_universe.jsonl").is_file()
    assert (artifact_root / "prefix_states.jsonl").is_file()
    assert (artifact_root / "slot_posterior_shards" / "shard_0.jsonl").is_file()
    assert (artifact_root / "slot_posterior_shard_summaries.jsonl").is_file()
    assert (artifact_root / "merge_manifest.json").is_file()
    assert (artifact_root / "summary.json").is_file()
    assert (artifact_root / "report.md").is_file()
    assert (artifact_root / "gallery" / "gallery_summary.json").is_file()


def test_jsonl_writer_rejects_non_json_safe_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        write_jsonl(tmp_path / "bad.jsonl", [{"value": math.nan}])
