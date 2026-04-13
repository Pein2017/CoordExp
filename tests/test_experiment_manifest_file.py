from __future__ import annotations

import json
from pathlib import Path

from src.bootstrap.experiment_manifest import write_experiment_manifest_file


def test_write_experiment_manifest_file_captures_soft_and_hard_context(
    tmp_path: Path,
) -> None:
    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml",
        base_config_path="configs/base.yaml",
        run_name="stage2_a_only_center_size_smoke",
        dataset_seed=17,
        experiment={
            "title": "Stage-2 A-only center-size smoke",
            "purpose": "Validate the center-size smoke path.",
            "key_deviations": ["Uses center-size bbox regression."],
        },
        effective_runtime={
            "trainer_variant": "stage2_two_channel",
            "checkpoint_mode": "artifact_only",
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
    assert payload["identity"]["run_name"] == "stage2_a_only_center_size_smoke"
    assert payload["experiment"]["authored"]["purpose"] == (
        "Validate the center-size smoke path."
    )
    assert payload["runtime_summary"]["trainer_variant"] == "stage2_two_channel"
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
