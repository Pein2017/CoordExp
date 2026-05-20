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
    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage2_two_channel/smoke/a_only.yaml",
        base_config_path="configs/base.yaml",
        run_name="smoke_20steps-stage2-a_only",
        dataset_seed=17,
        experiment={
            "title": "Stage-2 A-only smoke",
            "purpose": "Validate the compact A-only smoke path.",
            "key_deviations": ["Uses the retained canonical A-only smoke profile."],
        },
        effective_runtime={
            "trainer_variant": "stage2_two_channel",
            "checkpoint_mode": "artifact_only",
            "save_model_only": False,
            "gradient_accumulation_steps": 4,
            "packing": {"enabled": True},
            "launcher": {"COORDEXP_STAGE2_LAUNCHER": "scripts/train_stage2.sh"},
        },
        pipeline_manifest={
            "checksum": "abc123",
            "objective": [{"name": "token_ce"}, {"name": "bbox_geo"}],
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
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path.name == "experiment_manifest.json"
    assert payload["identity"]["run_name"] == "smoke_20steps-stage2-a_only"
    assert payload["experiment"]["authored"]["purpose"] == (
        "Validate the compact A-only smoke path."
    )
    assert payload["runtime_summary"]["trainer_variant"] == "stage2_two_channel"
    assert payload["runtime_summary"]["save_model_only"] is False
    assert payload["runtime_summary"]["pipeline"]["objective"] == [
        "token_ce",
        "bbox_geo",
    ]
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


def test_detection_authored_experiment_preserves_claim_scope() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_balance2.yaml"
    )
    assert isinstance(cfg, DetectionTrainingConfig)

    authored = _resolve_authored_experiment_payload(cfg)

    assert authored == {
        "surface": "ablation",
        "ablation_id": "E1",
        "claim_scope": "none",
    }
