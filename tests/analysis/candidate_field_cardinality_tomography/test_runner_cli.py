from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pytest
import yaml

from src.analysis.candidate_field_cardinality_tomography.artifacts import write_json, write_jsonl
from src.analysis.candidate_field_cardinality_tomography.controls import REQUIRED_CONTROL_TYPES
from src.analysis.candidate_field_cardinality_tomography.runner import run_from_config


def test_runner_dry_run_does_not_create_artifact_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("", encoding="utf-8")
    val.write_text("", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(artifact_root),
                "checkpoint_path": "/tmp/checkpoint",
                "train_jsonl": str(train),
                "val_jsonl": str(val),
                "stages": ["case_index"],
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/candidate_field_cardinality_tomography/run.py",
            "--config",
            str(config),
            "--stages",
            "case_index",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "candidate_field_cardinality_tomography" in result.stdout
    assert not artifact_root.exists()


def test_runner_taxonomy_report_consumes_existing_x1_rows(
    tmp_path: Path,
    minimal_base_row: dict[str, object],
) -> None:
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("", encoding="utf-8")
    val.write_text("", encoding="utf-8")
    artifact_root.mkdir()
    write_json(
        artifact_root / "case_index_summary.json",
        {"case_index_total_cases": 1, "case_index_total_rows": 3},
    )
    write_json(artifact_root / "probe_plan_summary.json", {"gpu_probe_planned_cases": 1})
    write_jsonl(
        artifact_root / "x1_candidate_field_rows.jsonl",
        [
            {
                **minimal_base_row,
                "probe_status": "ok",
                "same_desc_gt_count_annotated": 3,
                "gt_instance_coverage_count": 1,
                "coord_vocab_mass": 0.91,
                "x1_projection_collision": False,
                "posterior_snapshot_id": "pp-000001:pre_x1",
                "prefix_condition": "teacher_set_empty_prefix",
            }
        ],
    )
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(artifact_root),
                "checkpoint_path": "/tmp/checkpoint-3664",
                "train_jsonl": str(train),
                "val_jsonl": str(val),
                "stages": ["taxonomy", "validate", "report"],
            }
        ),
        encoding="utf-8",
    )

    run_from_config(config, stages=("taxonomy", "validate", "report"), allow_overwrite=True)

    summary = yaml.safe_load((artifact_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_id"] == "checkpoint-3664"
    assert summary["checkpoint_path"] == "/tmp/checkpoint-3664"
    assert summary["row_counts"]["taxonomy_assigned_cases"] == 1
    assert summary["denominator_counts"]["indexed_gt_rows"] == 3
    assert set(summary["control_status_by_type"]) == set(REQUIRED_CONTROL_TYPES)
    assert summary["control_status_by_type"]["wrong_desc_same_image"] == "missing"
    assert summary["control_status_by_type"]["same_desc_count_1_control"] == "missing"
    assert summary["row_count_breakdowns"]["prefix_condition"]["teacher_set_empty_prefix"]["taxonomy"] == 1
    assert (artifact_root / "manifest.json").exists()
    resolved = yaml.safe_load((artifact_root / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["sampling"]["max_cases"] is None
    assert resolved["peak"]["primary_merge_radius"] == 24
    assert resolved["effective_stages"] == ["taxonomy", "validate", "report"]
    assert "phase_a_case_taxonomy_rows.jsonl" in {
        path.name for path in artifact_root.iterdir()
    }
    report = (artifact_root / "report.md").read_text(encoding="utf-8")
    assert "Checkpoint: `checkpoint-3664`" in report
    assert "## Denominators" in report
    assert "indexed_cases" in report
    assert "Current taxonomy is based on the first x1 posterior probe only" in (
        artifact_root / "report.md"
    ).read_text(encoding="utf-8")


def test_validate_before_gpu_outputs_marks_incomplete(
    tmp_path: Path,
    tiny_coord_jsonl: Path,
) -> None:
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(artifact_root),
                "checkpoint_path": "/tmp/checkpoint",
                "train_jsonl": str(tiny_coord_jsonl),
                "val_jsonl": str(tiny_coord_jsonl),
                "stages": ["case_index", "probe_plan", "validate"],
                "sampling": {"max_cases": 2, "num_shards": 2},
            }
        ),
        encoding="utf-8",
    )

    run_from_config(config, allow_overwrite=True)

    summary = yaml.safe_load((artifact_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_status"] == "incomplete_missing_stage_outputs"
    assert summary["denominator_counts"]["sampled_gpu_cases"] == 2
    assert summary["denominator_counts"]["attempted_gpu_cases"] == 0


def test_unimplemented_stage_fails_explicitly(tmp_path: Path, tiny_coord_jsonl: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(tmp_path / "artifacts"),
                "checkpoint_path": "/tmp/checkpoint",
                "train_jsonl": str(tiny_coord_jsonl),
                "val_jsonl": str(tiny_coord_jsonl),
                "stages": ["attention_components"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="attention_components"):
        run_from_config(config, allow_overwrite=True)


def test_merge_requires_every_expected_shard_manifest(tmp_path: Path, minimal_base_row: dict[str, object]) -> None:
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "artifacts"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("", encoding="utf-8")
    val.write_text("", encoding="utf-8")
    artifact_root.mkdir()
    write_json(artifact_root / "probe_plan_summary.json", {"gpu_probe_planned_cases": 2, "num_shards": 2})
    write_jsonl(
        artifact_root / "probe_plan.jsonl",
        [
            {
                **minimal_base_row,
                "probe_sampled": True,
                "planned_shard_id": 0,
                "probe_plan_row_id": "pp-0",
            },
            {
                **minimal_base_row,
                "probe_sampled": True,
                "planned_shard_id": 1,
                "probe_plan_row_id": "pp-1",
            },
        ],
    )
    shard0 = artifact_root / "shards" / "shard_000"
    shard0.mkdir(parents=True)
    write_jsonl(shard0 / "x1_candidate_field_rows.jsonl", [{**minimal_base_row, "probe_plan_row_id": "pp-0"}])
    write_json(
        shard0 / "shard_manifest.json",
        {
            "shard_id": 0,
            "stage_status": "ok",
            "output_row_counts": {"x1_candidate_field_rows.jsonl": 1},
        },
    )
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "candidate_field_cardinality_tomography",
                "artifact_root": str(artifact_root),
                "checkpoint_path": "/tmp/checkpoint",
                "train_jsonl": str(train),
                "val_jsonl": str(val),
                "stages": ["merge"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="shard_001"):
        run_from_config(config, allow_overwrite=True)


def test_tmux_launcher_has_gpu_busy_preflight_guard() -> None:
    launcher = Path(
        "scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh"
    ).read_text(encoding="utf-8")

    assert "gpu_preflight()" in launcher
    assert "nvidia-smi -i \"$gpu\" --query-compute-apps" in launcher
    assert "refusing to launch because one or more requested GPUs are busy" in launcher
    assert "SKIP_GPU_PREFLIGHT=1" in launcher
