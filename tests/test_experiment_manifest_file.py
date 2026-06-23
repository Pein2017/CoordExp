from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bootstrap.experiment_manifest import write_experiment_manifest_file
from src.config.schema import DetectionTrainingConfig
from src.sft import _resolve_authored_experiment_payload
from test_detection_training_config_contract import _detection_payload


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
            "training_hierarchy": {"pipeline": {"id": "stage2_rollout_correction"}},
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
    assert "trainer_variant" not in payload["runtime_summary"]
    assert payload["runtime_summary"]["pipeline"]["id"] == "stage2_rollout_correction"
    assert payload["runtime_summary"]["save_model_only"] is False
    assert payload["runtime_summary"]["pipeline"]["objective"] == [
        "residual_set_correction",
    ]
    assert payload["runtime_summary"]["stage2_policy_provenance"] == stage2_policy
    assert payload["stage2_policy_provenance"] == stage2_policy
    assert payload["provenance_summary"]["git_sha"] == "deadbeef"
    assert payload["artifacts"]["run_metadata"] == "run_metadata.json"
    assert payload["artifacts"]["resolved_config"] == "resolved_config.json"


def test_experiment_manifest_runtime_summary_preserves_token_embeddings_adapter(
    tmp_path: Path,
) -> None:
    adapter_summary = {
        "enabled": True,
        "tie_head": True,
        "expected_trainable_row_count": 1002,
        "groups": {
            "coord_geometry": {
                "role": "coord_geometry",
                "expected_row_count": 1000,
            },
            "compact_structure": {
                "role": "structural_ce_only",
                "expected_row_count": 2,
            },
        },
    }

    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage1/standard_sft.yaml",
        base_config_path=None,
        run_name="stage1-standard-sft",
        dataset_seed=17,
        experiment=None,
        effective_runtime={
            "trainer_variant": "",
            "token_embeddings_adapter": adapter_summary,
        },
        pipeline_manifest=None,
        run_metadata=None,
        manifest_files={"effective_runtime": "effective_runtime.json"},
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert "trainer_variant" not in payload["runtime_summary"]
    assert payload["runtime_summary"]["token_embeddings_adapter"] == adapter_summary
    assert "token_rows" not in payload["runtime_summary"]


def test_experiment_manifest_runtime_summary_preserves_training_hierarchy(
    tmp_path: Path,
) -> None:
    hierarchy = {
        "pipeline": {"id": "stage1_standard_sft"},
        "objective": {"id": "standard_ce"},
        "sample_factory": {
            "id": "detection_sequence",
            "target_sequence": {
                "object_ordering": "sorted",
                "object_field_order": "desc_first",
                "bbox_format": "xyxy",
                "coordinate_surface": "coord_token",
                "strict_parse": True,
            },
        },
        "prompt": {"variant": "coco_80", "template_hash": "abc123"},
        "tokenizer": {"id": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct"},
        "chat_template": {"identity": "unknown_chat_template"},
        "packing": {"length": 1024},
    }

    out_path = write_experiment_manifest_file(
        output_dir=tmp_path,
        config_path="configs/stage1/standard_sft.yaml",
        base_config_path=None,
        run_name="stage1-standard-sft",
        dataset_seed=17,
        experiment=None,
        effective_runtime={"training_hierarchy": hierarchy},
        pipeline_manifest=None,
        run_metadata=None,
        manifest_files={"effective_runtime": "effective_runtime.json"},
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["runtime_summary"]["training_hierarchy"] == hierarchy


def test_experiment_manifest_runtime_summary_rejects_token_rows_alias(
    tmp_path: Path,
) -> None:
    adapter_summary = {
        "enabled": True,
        "groups": {"coord_geometry": {"expected_row_count": 1000}},
    }
    legacy_token_rows = {
        "enabled": True,
        "groups": {"legacy": {"expected_row_count": 1}},
    }

    with pytest.raises(
        ValueError,
        match=r"effective_runtime\.token_rows.*token_embeddings_adapter",
    ):
        write_experiment_manifest_file(
            output_dir=tmp_path,
            config_path="configs/stage1/standard_sft.yaml",
            base_config_path=None,
            run_name="stage1-standard-sft",
            dataset_seed=17,
            experiment=None,
            effective_runtime={
                "token_embeddings_adapter": adapter_summary,
                "token_rows": legacy_token_rows,
            },
            pipeline_manifest=None,
            run_metadata=None,
            manifest_files={"effective_runtime": "effective_runtime.json"},
        )


def test_experiment_manifest_runtime_summary_rejects_old_only_token_rows(
    tmp_path: Path,
) -> None:
    legacy_token_rows = {
        "enabled": True,
        "groups": {"coord_geometry": {"expected_row_count": 1000}},
    }

    with pytest.raises(
        ValueError,
        match=r"effective_runtime\.token_rows.*token_embeddings_adapter",
    ):
        write_experiment_manifest_file(
            output_dir=tmp_path,
            config_path="configs/legacy-effective-runtime.yaml",
            base_config_path=None,
            run_name="legacy-effective-runtime",
            dataset_seed=17,
            experiment=None,
            effective_runtime={
                "token_rows": legacy_token_rows,
            },
            pipeline_manifest=None,
            run_metadata=None,
            manifest_files={"effective_runtime": "effective_runtime.json"},
        )


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
    payload = _detection_payload()
    payload["experiment"] = {
        "surface": "smoke",
        "claim_scope": "smoke",
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)
    assert isinstance(cfg, DetectionTrainingConfig)

    authored = _resolve_authored_experiment_payload(cfg)

    assert authored == {
        "surface": "smoke",
        "claim_scope": "smoke",
    }
