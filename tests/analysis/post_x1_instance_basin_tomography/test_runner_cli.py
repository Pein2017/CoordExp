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
                "num_shards": 8,
                "checkpoints": {
                    "fullobj_random_pure_ce_ckpt3668": {
                        "checkpoint_path": "/ckpts/random/checkpoint-3668",
                        "comparison_role": "clean_pair",
                        "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
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
                        "comparison_role": "clean_pair",
                        "controlled_comparison_group": "pure_ce_sorted_vs_random_no_newline",
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
                        "comparison_role": "reference_anchor",
                        "controlled_comparison_group": "reference_anchor_not_controlled",
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
            },
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
