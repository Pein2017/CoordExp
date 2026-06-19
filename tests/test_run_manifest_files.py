import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.config.schema import DetectionTrainingConfig
from src.utils.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    collect_runtime_env_metadata,
    serialize_resolved_training_config,
    write_run_manifest_files,
)
from test_detection_training_config_contract import _detection_payload


@dataclass
class _TinyCfg:
    output_dir: Path
    template: dict[str, Any]


def test_collect_runtime_env_metadata_is_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOT_IMAGE_DIR", "/tmp/images")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # Should not be included unless explicitly requested.
    monkeypatch.setenv("SHOULD_NOT_APPEAR", "secret")

    env = collect_runtime_env_metadata()
    assert env["ROOT_IMAGE_DIR"] == "/tmp/images"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "SHOULD_NOT_APPEAR" not in env


def test_serialize_resolved_training_config_converts_paths_to_strings() -> None:
    cfg = _TinyCfg(output_dir=Path("out"), template={"max_pixels": 10485760})
    resolved = serialize_resolved_training_config(cfg)
    assert resolved["output_dir"] == "out"
    assert resolved["template"]["max_pixels"] == 10485760


def test_write_run_manifest_files_writes_required_json(tmp_path: Path) -> None:
    cfg = _TinyCfg(output_dir=Path("out"), template={"max_pixels": 10485760})
    stage2_policy = {
        "assignment_strategy": "greedy_iou",
        "duplicate_filter_strategy": "rollout_correction_duplicate_control",
        "object_ordering_policy": "sorted",
    }
    written = write_run_manifest_files(
        output_dir=tmp_path,
        training_config=cfg,
        config_path="configs/unit.yaml",
        base_config_path="configs/base.yaml",
        dataset_seed=17,
        effective_runtime={
            "trainer_variant": "stage2_rollout_correction",
            "checkpoint_mode": "restartable",
            "save_model_only": True,
            "save_only_model": False,
            "hf_save_only_model": False,
        },
        pipeline_manifest={"checksum": "abc123", "objective": [{"name": "token_ce"}]},
        stage2_policy_provenance=stage2_policy,
        train_data_provenance={"dataset_jsonl": "train.jsonl"},
        eval_data_provenance={"dataset_jsonl": "val.jsonl"},
    )

    resolved_path = tmp_path / written["resolved_config"]
    env_path = tmp_path / written["runtime_env"]
    effective_runtime_path = tmp_path / written["effective_runtime"]
    pipeline_manifest_path = tmp_path / written["pipeline_manifest"]
    train_provenance_path = tmp_path / written["train_data_provenance"]
    eval_provenance_path = tmp_path / written["eval_data_provenance"]
    assert resolved_path.is_file()
    assert env_path.is_file()
    assert effective_runtime_path.is_file()
    assert pipeline_manifest_path.is_file()
    assert train_provenance_path.is_file()
    assert eval_provenance_path.is_file()

    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert resolved["dataset_seed"] == 17
    assert resolved["resolved"]["template"]["max_pixels"] == 10485760

    env = json.loads(env_path.read_text(encoding="utf-8"))
    assert env["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert isinstance(env["env"], dict)

    effective_runtime = json.loads(effective_runtime_path.read_text(encoding="utf-8"))
    assert "trainer_variant" not in effective_runtime["runtime"]
    assert effective_runtime["runtime"]["checkpoint_mode"] == "restartable"
    assert effective_runtime["runtime"]["save_model_only"] is True
    assert effective_runtime["runtime"]["save_only_model"] is False
    assert effective_runtime["runtime"]["hf_save_only_model"] is False
    assert effective_runtime["stage2_policy_provenance"] == stage2_policy

    pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    assert pipeline_manifest["pipeline"]["checksum"] == "abc123"
    assert pipeline_manifest["stage2_policy_provenance"] == stage2_policy

    train_provenance = json.loads(train_provenance_path.read_text(encoding="utf-8"))
    assert train_provenance["split"] == "train"
    assert train_provenance["provenance"]["dataset_jsonl"] == "train.jsonl"

    eval_provenance = json.loads(eval_provenance_path.read_text(encoding="utf-8"))
    assert eval_provenance["split"] == "eval"
    assert eval_provenance["provenance"]["dataset_jsonl"] == "val.jsonl"


def test_write_run_manifest_files_copies_training_hierarchy_to_resolved_config(
    tmp_path: Path,
) -> None:
    cfg = {
        "pipeline": {"id": "stage1_standard_sft"},
        "objective": {"id": "standard_ce"},
    }
    hierarchy = {
        "pipeline": {"id": "stage1_standard_sft"},
        "objective": {"id": "standard_ce"},
        "sample_factory": {
            "id": "detection_sequence",
            "target_sequence": {
                "object_ordering": "random_permutation",
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

    written = write_run_manifest_files(
        output_dir=tmp_path,
        training_config=cfg,
        config_path="configs/unit.yaml",
        base_config_path=None,
        dataset_seed=17,
        effective_runtime={"training_hierarchy": hierarchy},
        pipeline_manifest={"checksum": "abc123", "objective": [{"name": "standard_ce"}]},
    )

    resolved = json.loads(
        (tmp_path / written["resolved_config"]).read_text(encoding="utf-8")
    )
    effective_runtime = json.loads(
        (tmp_path / written["effective_runtime"]).read_text(encoding="utf-8")
    )
    pipeline_manifest = json.loads(
        (tmp_path / written["pipeline_manifest"]).read_text(encoding="utf-8")
    )

    assert resolved["resolved"]["pipeline"]["id"] == "stage1_standard_sft"
    assert resolved["resolved"]["objective"]["id"] == "standard_ce"
    assert resolved["training_hierarchy"] == hierarchy
    assert effective_runtime["runtime"]["training_hierarchy"] == hierarchy
    assert pipeline_manifest["training_hierarchy"] == hierarchy


def test_write_run_manifest_files_resolved_config_uses_target_hierarchy(
    tmp_path: Path,
) -> None:
    cfg = DetectionTrainingConfig.from_mapping(_detection_payload())
    hierarchy = {
        "pipeline": {"id": "stage1_standard_sft"},
        "objective": {"id": "standard_ce"},
        "sample_factory": {
            "id": "detection_sequence",
            "target_sequence": {
                "object_ordering": "random_permutation",
                "object_field_order": "desc_first",
                "bbox_format": "xyxy",
                "coordinate_surface": "coord_token",
                "strict_parse": True,
            },
        },
        "detection_template": {"id": "compact"},
        "prompt": {"variant": "coco_80", "template_hash": "abc123"},
        "tokenizer": {"id": "model_cache/models/Qwen/Qwen3-VL-2B-Instruct"},
        "chat_template": {"identity": "unknown_chat_template"},
        "packing": {"length": 1024},
    }

    written = write_run_manifest_files(
        output_dir=tmp_path,
        training_config=cfg,
        config_path="configs/stage1/standard_sft.yaml",
        base_config_path=None,
        dataset_seed=17,
        effective_runtime={"training_hierarchy": hierarchy},
    )

    resolved = json.loads(
        (tmp_path / written["resolved_config"]).read_text(encoding="utf-8")
    )

    payload = resolved["resolved"]
    assert payload["pipeline"]["id"] == "stage1_standard_sft"
    assert payload["objective"]["id"] == "standard_ce"
    assert payload["sample_factory"]["id"] == "detection_sequence"
    assert payload["sample_factory"]["target_sequence"]["object_ordering"] == (
        "random_permutation"
    )
    assert payload["sample_factory"]["target_sequence"]["object_field_order"] == (
        "desc_first"
    )
    assert payload["detection_template"]["id"] == "compact"
    assert payload["token_embeddings_adapter"]["enabled"] is True
    assert "custom" not in payload
    assert "token_rows" not in payload
    assert resolved["training_hierarchy"] == hierarchy


def test_write_run_manifest_files_tracks_source_config_copies(tmp_path: Path) -> None:
    config_path = tmp_path / "unit.yaml"
    base_config_path = tmp_path / "base.yaml"
    config_path.write_text("training:\n  output_dir: out\n", encoding="utf-8")
    base_config_path.write_text("seed: 17\n", encoding="utf-8")
    cfg = _TinyCfg(output_dir=Path("out"), template={"max_pixels": 10485760})

    written = write_run_manifest_files(
        output_dir=tmp_path / "run",
        training_config=cfg,
        config_path=str(config_path),
        base_config_path=str(base_config_path),
        dataset_seed=17,
    )

    assert written["config_source"] == "config_source.yaml"
    assert written["base_config_source"] == "base_config_source.yaml"
    assert (tmp_path / "run" / "config_source.yaml").read_text(
        encoding="utf-8"
    ) == config_path.read_text(encoding="utf-8")
    assert (tmp_path / "run" / "base_config_source.yaml").read_text(
        encoding="utf-8"
    ) == base_config_path.read_text(encoding="utf-8")
