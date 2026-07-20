from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.errors import ArtifactContractError, ConfigContractError
from src.config.fingerprint import sha256_json
from src.config.loader import load_train_config
from src.rollout_calibration.state_bank import (
    BLIND_IMAGE_IDS,
    STATE_BANK_RECORDS_NAME,
    STATE_BANK_SCHEMA_VERSION,
)


SUPERVISED_FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")
SOURCE_CHECKPOINT = {
    "adapter_fingerprint": "1" * 64,
    "embedding_delta_fingerprint": "2" * 64,
    "base_config_sha256": "3" * 64,
    "tokenizer_sha256": "4" * 64,
    "token_identity_sha256": "5" * 64,
    "special_token_identity_sha256": "6" * 64,
    "processor_identity_sha256": "7" * 64,
}
SOURCE_CHECKPOINT_ID = sha256_json(SOURCE_CHECKPOINT)


@pytest.mark.parametrize(
    ("profile", "entity_weight", "coordinate_weight"),
    [
        ("transition_only", 1.0, 0.0),
        ("coordinate_boundary_only", 0.0, 1.0),
        ("joint", 0.5, 0.5),
    ],
)
def test_rollout_calibration_profiles_load_with_exact_weights_and_paths(
    tmp_path: Path,
    profile: str,
    entity_weight: float,
    coordinate_weight: float,
) -> None:
    config_dir = tmp_path / "configs" / "profiles"
    config_path = config_dir / f"{profile}.yaml"
    manifest_path = config_dir / "banks" / "state-bank-manifest.json"
    _write_manifest(manifest_path)
    payload = _calibration_config(
        profile=profile,
        entity_weight=entity_weight,
        coordinate_weight=coordinate_weight,
    )
    _write_yaml(config_path, payload)

    resolved = load_train_config(config_path)
    config = resolved.config
    calibration = config.rollout_calibration

    assert config.training.mode == "rollout_calibration"
    assert config.data.train is None
    assert config.data.eval is None
    assert config.losses.normalizer == "event_balanced"
    assert config.losses.protected.base_ce.weight == 0.0
    assert config.losses.protected.token_type_gate.weight == 0.0
    assert config.losses.protected.coord_gaussian_rps.weight == 0.0
    assert config.losses.protected.rollout_site_token_type_gate is not None
    assert config.losses.protected.rollout_site_token_type_gate.weight == pytest.approx(
        0.1
    )
    assert calibration is not None
    assert calibration.profile == profile
    assert calibration.entity_transition.weight == entity_weight
    assert calibration.coordinate_boundary.weight == coordinate_weight
    assert calibration.incomplete_objective_policy == "fail"
    assert calibration.online_state_bank_refresh is False
    assert calibration.state_bank_manifest_path == str(manifest_path.resolve())
    assert config.adapter.source_adapter_path == str(
        (config_dir / "warm-start" / "adapter").resolve()
    )
    assert config.adapter.repaired_embedding_payload_path == str(
        (config_dir / "warm-start" / "embedding-payload").resolve()
    )
    assert (
        resolved.path_origins[
            "rollout_calibration.state_bank_manifest_path"
        ].declaring_config_path
        == config_path.resolve()
    )
    assert (
        resolved.path_origins["adapter.source_adapter_path"].declaring_config_path
        == config_path.resolve()
    )
    assert (
        resolved.path_origins[
            "adapter.repaired_embedding_payload_path"
        ].declaring_config_path
        == config_path.resolve()
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_binding = resolved.to_artifact_dict()["resolution"][
        "rollout_calibration_manifest"
    ]
    assert resolved_binding["bank_id"] == manifest_payload["bank_id"]
    assert resolved_binding["records_sha256"] == manifest_payload["records_sha256"]


def test_supervised_config_keeps_ordinary_resolved_shape_and_defaults() -> None:
    resolved = load_train_config(SUPERVISED_FIXTURE)

    assert resolved.config.training.mode == "supervised"
    assert resolved.config.rollout_calibration is None
    assert resolved.config.losses.normalizer == "segment_balanced"
    assert resolved.config.losses.protected.base_ce.weight == 1.0
    assert resolved.config.losses.protected.token_type_gate.weight == pytest.approx(0.1)
    assert resolved.config.losses.protected.rollout_site_token_type_gate is None
    assert "rollout_calibration" not in resolved.config_dict
    assert (
        "rollout_site_token_type_gate"
        not in resolved.config_dict["losses"]["protected"]
    )


def test_supervised_mode_allows_only_zero_rollout_site_gate(tmp_path: Path) -> None:
    zero_path = tmp_path / "zero.yaml"
    payload = _supervised_payload()
    payload["losses"]["protected"]["rollout_site_token_type_gate"] = {"weight": 0.0}
    _write_yaml(zero_path, payload)

    assert (
        load_train_config(
            zero_path
        ).config.losses.protected.rollout_site_token_type_gate.weight
        == 0.0
    )

    positive_path = tmp_path / "positive.yaml"
    payload["losses"]["protected"]["rollout_site_token_type_gate"]["weight"] = 0.1
    _write_yaml(positive_path, payload)

    with pytest.raises(
        ConfigContractError, match="rollout_site_token_type_gate.weight=0"
    ):
        load_train_config(positive_path)


def test_inherited_calibration_paths_resolve_from_declaring_config(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base" / "calibration.yaml"
    child_path = tmp_path / "runs" / "joint.yaml"
    manifest_path = base_path.parent / "banks" / "state-bank-manifest.json"
    _write_manifest(manifest_path)
    _write_yaml(base_path, _calibration_config())
    _write_yaml(
        child_path,
        {
            "schema_version": 1,
            "extends": "../base/calibration.yaml",
            "run": {"name": "inherited-joint"},
        },
    )

    resolved = load_train_config(child_path)

    assert resolved.config.rollout_calibration is not None
    assert resolved.config.rollout_calibration.state_bank_manifest_path == str(
        manifest_path.resolve()
    )
    for field in (
        "rollout_calibration.state_bank_manifest_path",
        "adapter.source_adapter_path",
        "adapter.repaired_embedding_payload_path",
    ):
        assert resolved.path_origins[field].declaring_config_path == base_path.resolve()


@pytest.mark.parametrize(
    ("mutation", "expected_text"),
    [
        (("data.train", {"path": "canonical-train.jsonl"}), "rejects data.train"),
        (("data.eval", {"path": "canonical-eval.jsonl"}), "data.eval"),
        (("losses.normalizer", "segment_balanced"), "event_balanced"),
        (("losses.protected.base_ce.weight", 1.0), "base_ce.weight=0"),
        (("losses.protected.token_type_gate.weight", 0.1), "token_type_gate.weight=0"),
        (
            ("losses.protected.coord_gaussian_rps.weight", 0.1),
            "coord_gaussian_rps.weight=0",
        ),
        (("losses.protected.rollout_site_token_type_gate.weight", 0.0), "positive"),
        (("adapter.seed_mode", "initialize_new"), "warm-start source paths"),
        (("rollout_calibration.online_state_bank_refresh", True), "False"),
        (("rollout_calibration.incomplete_objective_policy", "skip"), "fail"),
        (("rollout_calibration.kl_anchor_weight", 0.1), "Extra inputs"),
        (("rollout_calibration.canonical_replay_weight", 0.1), "Extra inputs"),
    ],
)
def test_rollout_calibration_rejects_forbidden_surfaces(
    tmp_path: Path,
    mutation: tuple[str, Any],
    expected_text: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_manifest(tmp_path / "banks" / "state-bank-manifest.json")
    payload = _calibration_config()
    _set_nested(payload, *mutation)
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert exc_info.value.code == "config.schema_validation"
    assert expected_text in str(exc_info.value)


@pytest.mark.parametrize(
    ("profile", "entity_weight", "coordinate_weight"),
    [
        ("transition_only", 0.5, 0.5),
        ("coordinate_boundary_only", 1.0, 0.0),
        ("joint", 1.0, 1.0),
    ],
)
def test_rollout_calibration_rejects_noncanonical_profile_weights(
    tmp_path: Path,
    profile: str,
    entity_weight: float,
    coordinate_weight: float,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_manifest(tmp_path / "banks" / "state-bank-manifest.json")
    _write_yaml(
        config_path,
        _calibration_config(
            profile=profile,
            entity_weight=entity_weight,
            coordinate_weight=coordinate_weight,
        ),
    )

    with pytest.raises(ConfigContractError, match="requires entity_transition.weight"):
        load_train_config(config_path)


def test_rollout_calibration_requires_warm_start_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_manifest(tmp_path / "banks" / "state-bank-manifest.json")
    payload = _calibration_config()
    payload["adapter"].pop("source_adapter_path")
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError, match="source_adapter_path"):
        load_train_config(config_path)


def test_state_bank_checkpoint_mismatch_fails_during_config_loading(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "banks" / "state-bank-manifest.json"
    other_checkpoint = {
        **SOURCE_CHECKPOINT,
        "adapter_fingerprint": "8" * 64,
    }
    other_checkpoint_id = _write_manifest(
        manifest_path,
        source_checkpoint=other_checkpoint,
    )
    _write_yaml(config_path, _calibration_config())

    with pytest.raises(ConfigContractError) as exc_info:
        load_train_config(config_path)

    assert exc_info.value.code == "config.state_bank_checkpoint_mismatch"
    assert exc_info.value.context == {
        "state_bank_manifest_path": str(manifest_path.resolve()),
        "configured_source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "manifest_source_checkpoint_id": other_checkpoint_id,
    }


def test_state_bank_manifest_binding_requires_strict_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "banks" / "state-bank-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    payload = _manifest_payload(SOURCE_CHECKPOINT)
    serialized = json.dumps(payload)
    needle = '"source_checkpoint_id": '
    manifest_path.write_text(
        serialized.replace(
            needle,
            f'{needle}"{SOURCE_CHECKPOINT_ID}", {needle}',
            1,
        ),
        encoding="utf-8",
    )
    _write_yaml(config_path, _calibration_config())

    with pytest.raises(ArtifactContractError) as exc_info:
        load_train_config(config_path)

    assert exc_info.value.code == "state_bank.manifest_json"


def test_supervised_mode_still_requires_train_data(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = _supervised_payload()
    payload["data"].pop("train")
    _write_yaml(config_path, payload)

    with pytest.raises(ConfigContractError, match="requires data.train"):
        load_train_config(config_path)


def _calibration_config(
    *,
    profile: str = "joint",
    entity_weight: float = 0.5,
    coordinate_weight: float = 0.5,
) -> dict[str, Any]:
    payload = _supervised_payload()
    payload["adapter"].update(
        {
            "seed_mode": "warm_start_expand_dora",
            "source_adapter_path": "warm-start/adapter",
            "repaired_embedding_payload_path": "warm-start/embedding-payload",
        }
    )
    payload["data"] = {}
    payload["losses"] = {
        "normalizer": "event_balanced",
        "protected": {
            "base_ce": {"weight": 0.0},
            "token_type_gate": {
                "weight": 0.0,
                "groups": ["desc_text", "schema", "coordinate", "eos"],
            },
            "coord_gaussian_rps": {"weight": 0.0},
            "rollout_site_token_type_gate": {"weight": 0.1},
        },
    }
    payload["training"]["mode"] = "rollout_calibration"
    payload["eval"] = {
        "forward": {"every_fraction": None, "steps": []},
        "inference": {"enabled": False},
    }
    payload["rollout_calibration"] = {
        "profile": profile,
        "state_bank_manifest_path": "banks/state-bank-manifest.json",
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "entity_transition": {
            "weight": entity_weight,
            "margin": 0.2,
            "smooth_max_temperature": 0.5,
        },
        "coordinate_boundary": {
            "weight": coordinate_weight,
            "margin": 0.2,
        },
        "incomplete_objective_policy": "fail",
        "online_state_bank_refresh": False,
    }
    return payload


def _supervised_payload() -> dict[str, Any]:
    payload = yaml.safe_load(SUPERVISED_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(
    path: Path,
    *,
    source_checkpoint: dict[str, str] | None = None,
) -> str:
    checkpoint = SOURCE_CHECKPOINT if source_checkpoint is None else source_checkpoint
    payload = _manifest_payload(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    return payload["source_checkpoint_id"]


def _manifest_payload(source_checkpoint: dict[str, str]) -> dict[str, Any]:
    source_checkpoint_id = sha256_json(source_checkpoint)
    determinants = {
        "schema_version": STATE_BANK_SCHEMA_VERSION,
        "records_file": STATE_BANK_RECORDS_NAME,
        "records_sha256": "9" * 64,
        "record_count": 1,
        "source_checkpoint_id": source_checkpoint_id,
        "source_checkpoint": source_checkpoint,
        "prompt_identity_sha256": "a" * 64,
        "split_assignments": [
            {
                "image_id": 42,
                "split": "train",
                "split_group_id": "image:42",
                "image_content_sha256": "b" * 64,
            }
        ],
        "split_counts": {"train": 1},
        "event_family_counts": {"entity_transition": 1},
        "source_artifacts": [],
        "blind_image_ids": list(sorted(BLIND_IMAGE_IDS)),
        "rejection_reasons": {},
    }
    return {"bank_id": sha256_json(determinants), **determinants}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        assert isinstance(child, dict)
        current = child
    current[parts[-1]] = value
