from __future__ import annotations

import json
from pathlib import Path

from src.bootstrap.experiment_manifest import write_experiment_manifest_file
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig
from src.sft import _resolve_authored_experiment_payload


def test_write_experiment_manifest_file_captures_soft_and_hard_context(
    tmp_path: Path,
) -> None:
    stage2_policy = {
        "assignment_strategy": "greedy_iou",
        "duplicate_filter_strategy": "rollout_correction_duplicate_control",
        "object_ordering_policy": "sorted",
    }
    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml",
        base_config_path="configs/base.yaml",
        run_name="compact_full_hf_1step",
        dataset_seed=17,
        experiment={
            "title": "Stage-2 rollout-correction smoke",
            "purpose": "Validate the compact rollout-correction smoke path.",
            "key_deviations": ["Uses the canonical residual_set_correction objective."],
        },
        effective_runtime={
            "trainer_variant": "stage2_rollout_correction",
            "checkpoint_mode": "artifact_only",
            "save_model_only": False,
            "gradient_accumulation_steps": 4,
            "packing": {"enabled": True},
            "launcher": {"COORDEXP_STAGE2_LAUNCHER": "scripts/train_stage2.sh"},
        },
        pipeline_manifest={
            "checksum": "abc123",
            "objective": [{"name": "residual_set_correction"}],
            "diagnostics": [],
        },
        run_metadata={
            "created_at": "2026-04-13T00:00:00+00:00",
            "git_sha": "deadbeef",
            "git_branch": "main",
            "git_dirty": False,
            "upstream": {"swift": {"version": "1.0"}},
        },
        manifest_files={
            "resolved_config": "resolved_config.json",
            "effective_runtime": "effective_runtime.json",
            "pipeline_manifest": "pipeline_manifest.json",
        },
        stage2_policy_provenance=stage2_policy,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path.name == "experiment_manifest.json"
    assert payload["identity"]["run_name"] == "compact_full_hf_1step"
    assert payload["experiment"]["authored"]["purpose"] == (
        "Validate the compact rollout-correction smoke path."
    )
    assert payload["runtime_summary"]["trainer_variant"] == "stage2_rollout_correction"
    assert payload["runtime_summary"]["save_model_only"] is False
    assert payload["runtime_summary"]["pipeline"]["objective"] == [
        "residual_set_correction",
    ]
    assert payload["runtime_summary"]["stage2_policy_provenance"] == stage2_policy
    assert payload["stage2_policy_provenance"] == stage2_policy
    assert payload["provenance_summary"]["git_sha"] == "deadbeef"
    assert payload["artifacts"]["run_metadata"] == "run_metadata.json"
    assert payload["artifacts"]["resolved_config"] == "resolved_config.json"


def test_write_experiment_manifest_file_marks_missing_authored_experiment(
    tmp_path: Path,
) -> None:
    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/unit.yaml",
        base_config_path=None,
        run_name="unit-run",
        dataset_seed=23,
        experiment=None,
        effective_runtime=None,
        pipeline_manifest=None,
        run_metadata=None,
        manifest_files=None,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["experiment"]["authored"] is None


def test_teacher_forcing_authored_experiment_preserves_claim_scope() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml"
    )
    assert isinstance(cfg, DetectionTrainingConfig)

    authored = _resolve_authored_experiment_payload(cfg)

    assert authored == {
        "surface": "smoke",
        "claim_scope": "smoke",
    }
