from __future__ import annotations

import fcntl
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.config.fingerprint import sha256_json
from src.config.inference import (
    INFER_CONFIG_LOADER_VERSION,
    InferConfig,
    ResolvedInferConfig,
)
import src.label_studio_coco_refinement.inference_profiles as profiles_module
from src.label_studio_coco_refinement.inference_profiles import (
    ALLOWED_ARTIFACT_ROLES,
    EngineProfile,
    EngineProfileStore,
    ProfileContractError,
    ProfileDriftError,
)


def _capture_inputs(
    tmp_path: Path,
    *,
    name: str = "accepted",
    with_conditionals: bool = False,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "weights.bin").write_bytes(b"weights-v1")
    model_config = tmp_path / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "vision_config": {"patch_size": 16, "spatial_merge_size": 2},
            }
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"vocab": {"a": 1}}', encoding="utf-8")
    processor = tmp_path / "preprocessor_config.json"
    processor.write_text(
        json.dumps({"patch_size": 16, "merge_size": 2}), encoding="utf-8"
    )
    artifact_paths: dict[str, Path] = {
        "base_weights": base,
        "model_config": model_config,
        "tokenizer": tokenizer,
        "processor": processor,
    }
    strict_payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "profile-test",
            "artifact_root": str(tmp_path / "outputs"),
            "collision_policy": "fail",
        },
        "model": {
            "base_model": str(base),
            "dtype": "bf16",
            "attn_implementation": "eager",
            "processor": {"do_resize": False},
            "runtime_patches": {"patch_embed_linearization": "enabled"},
        },
        "data": {"input_jsonl": str(tmp_path / "source.jsonl")},
        "template": {
            "object_field_order": "desc_first",
            "object_ordering": "source_order",
            "assistant_format": "object_box_closed",
            "prompt": {"system": "detect", "user": "find all objects"},
        },
        "backend": {"type": "hf"},
        "generation": {
            "batch_size": 1,
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
        "scoring": {"enabled": True},
        "artifacts": {
            "write_token_trace": True,
            "write_parse_diagnostics": True,
        },
        "debug": {"smoke": True, "dry_run": False},
        "adapter": None,
        "embedding_delta": None,
    }
    roi_inference = {
        "processor_factor": 32,
        "default_width": 1024,
        "default_height": 1024,
        "min_axis_pixels": 32,
        "max_axis_pixels": 2048,
        "max_total_pixels": 2_097_152,
        "deadline_seconds": 20.0,
    }
    if with_conditionals:
        for role in ("adapter", "embedding_delta"):
            path = tmp_path / role
            path.mkdir(exist_ok=True)
            (path / f"{role}.bin").write_bytes(f"{role}-v1".encode())
            artifact_paths[role] = path
            strict_payload[role] = (
                {"type": "dora", "path": str(path), "name": "default"}
                if role == "adapter"
                else {"path": str(path)}
            )
    config = InferConfig.model_validate(strict_payload)
    config_dict = config.model_dump(mode="json")
    resolved_config = ResolvedInferConfig(
        config=config,
        config_dict=config_dict,
        fingerprint=sha256_json(config_dict),
        schema_version=1,
        loader_version=INFER_CONFIG_LOADER_VERSION,
        entry_config_path=tmp_path / "infer.yaml",
        sources=(),
        path_origins={},
    )
    return {
        "name": name,
        "endpoint": "http://127.0.0.1:8123/infer",
        "artifact_paths": artifact_paths,
        "resolved_config": resolved_config,
        "roi_inference": roi_inference,
        "parser_identity": {"id": "compact-object-box-closed-v1"},
        "adapter_identity": {"id": "resident-roi-v1"},
        "transform_identity": {"id": "coordexp-roi-letterbox-half-up-v1"},
        "transformers_version": "4.57.3",
        "processor_kwargs": {"do_resize": False, "return_tensors": "pt"},
        "runtime_identity": {"backend": "hf", "device": "cuda:0"},
    }


def _profile(
    tmp_path: Path, *, name: str = "accepted", with_conditionals: bool = False
) -> EngineProfile:
    return EngineProfile.capture(
        **_capture_inputs(tmp_path, name=name, with_conditionals=with_conditionals)
    )


def test_profile_persists_complete_identity_and_one_active_binding(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    store = EngineProfileStore(tmp_path / "profiles.json")

    store.save(profile)
    activated = store.activate("train-project", profile.name)

    assert activated.fingerprint == profile.fingerprint
    assert store.active("train-project").fingerprint == profile.fingerprint
    assert list(store.profiles()) == ["accepted"]
    receipt = activated.to_receipt_dict()
    assert receipt["processor"]["do_resize"] is False
    assert receipt["processor"]["factor"] == 32
    assert receipt["processor"]["axis_bounds"] == [32, 2048]
    assert set(receipt["identity_fingerprints"]) == {
        "resolved_config",
        "prompt_policy",
        "parser",
        "adapter",
        "transform",
        "transformers",
        "processor_kwargs",
        "runtime",
    }
    assert all(receipt["identity_fingerprints"].values())
    assert "resolved_config" not in receipt
    assert "prompt_policy" not in receipt
    assert "runtime_identity" not in receipt


def test_profile_store_mutations_share_one_lock_and_fsync_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    store = EngineProfileStore(tmp_path / "profiles.json")
    lock_acquisitions: list[Path] = []
    fsynced_directories: list[Path] = []
    real_flock = fcntl.flock
    real_fsync_directory = profiles_module._fsync_directory

    def tracking_flock(descriptor: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            lock_acquisitions.append(
                Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
            )
        real_flock(descriptor, operation)

    def tracking_fsync_directory(directory: Path) -> None:
        fsynced_directories.append(directory)
        real_fsync_directory(directory)

    monkeypatch.setattr(profiles_module.fcntl, "flock", tracking_flock)
    monkeypatch.setattr(profiles_module, "_fsync_directory", tracking_fsync_directory)

    store.save(profile)
    store.activate("train-project", profile.name)
    store.save(profile)

    assert lock_acquisitions == [store.lock_path.resolve()] * 3
    assert fsynced_directories == [store.path.parent] * 3


def test_profile_store_parent_fsync_failure_is_repaired_by_idempotent_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    store = EngineProfileStore(tmp_path / "profiles.json")
    real_fsync_directory = profiles_module._fsync_directory
    fail_once = True

    def fail_parent_once(directory: Path) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("injected profile directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(profiles_module, "_fsync_directory", fail_parent_once)
    with pytest.raises(OSError, match="injected profile directory fsync failure"):
        store.save(profile)

    persisted = store.path.read_bytes()
    fsynced: list[Path] = []

    def tracking_fsync_directory(directory: Path) -> None:
        fsynced.append(directory)
        real_fsync_directory(directory)

    monkeypatch.setattr(profiles_module, "_fsync_directory", tracking_fsync_directory)
    store.save(profile)
    assert store.path.read_bytes() == persisted
    assert fsynced == [store.path.parent]


@pytest.mark.parametrize(
    "missing_role", ["base_weights", "model_config", "tokenizer", "processor"]
)
def test_each_unconditional_artifact_role_is_required(
    tmp_path: Path, missing_role: str
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["artifact_paths"].pop(missing_role)

    with pytest.raises(ProfileContractError, match="missing required artifact roles"):
        EngineProfile.capture(**inputs)


@pytest.mark.parametrize("missing_role", ["adapter", "embedding_delta"])
def test_each_configured_conditional_artifact_role_is_required(
    tmp_path: Path, missing_role: str
) -> None:
    inputs = _capture_inputs(tmp_path, with_conditionals=True)
    inputs["artifact_paths"].pop(missing_role)

    with pytest.raises(ProfileContractError, match="missing required artifact roles"):
        EngineProfile.capture(**inputs)


def test_unknown_and_unconfigured_conditional_roles_are_rejected(
    tmp_path: Path,
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["artifact_paths"]["processor_artifacts"] = inputs["artifact_paths"][
        "processor"
    ]
    with pytest.raises(ProfileContractError, match="unknown or mislabeled"):
        EngineProfile.capture(**inputs)

    inputs = _capture_inputs(tmp_path / "other")
    extra = tmp_path / "other" / "adapter-extra"
    extra.mkdir()
    (extra / "adapter.bin").write_bytes(b"adapter")
    inputs["artifact_paths"]["adapter"] = extra
    with pytest.raises(ProfileContractError, match="without matching resolved config"):
        EngineProfile.capture(**inputs)

    inputs = _capture_inputs(tmp_path / "checkpoint")
    checkpoint = tmp_path / "checkpoint" / "checkpoint-extra"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"checkpoint")
    inputs["artifact_paths"]["checkpoint"] = checkpoint
    with pytest.raises(ProfileContractError, match="unknown or mislabeled"):
        EngineProfile.capture(**inputs)


def test_mislabeled_model_and_processor_artifacts_are_rejected(tmp_path: Path) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["artifact_paths"]["model_config"], inputs["artifact_paths"]["processor"] = (
        inputs["artifact_paths"]["processor"],
        inputs["artifact_paths"]["model_config"],
    )

    with pytest.raises(ProfileContractError):
        EngineProfile.capture(**inputs)


@pytest.mark.parametrize("role", sorted(ALLOWED_ARTIFACT_ROLES))
def test_same_path_drift_is_rejected_for_every_artifact_family(
    tmp_path: Path, role: str
) -> None:
    profile = _profile(tmp_path, with_conditionals=True)
    artifact = next(item for item in profile.artifacts if item.role == role)
    path = Path(artifact.path)
    if path.is_dir():
        (path / "drift.bin").write_bytes(b"changed")
    else:
        path.write_bytes(path.read_bytes() + b"\nchanged")

    with pytest.raises(ProfileDriftError) as exc_info:
        profile.verify_artifacts()

    assert role in exc_info.value.mismatches


def test_directory_fingerprint_covers_relative_names_and_empty_directories(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    old = next(item for item in profile.artifacts if item.role == "base_weights")
    (tmp_path / "base" / "empty").mkdir()

    with pytest.raises(ProfileDriftError):
        old.verify_current()


def test_processor_factor_is_derived_and_cross_checked_with_model_and_config(
    tmp_path: Path,
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["roi_inference"]["processor_factor"] = 64
    with pytest.raises(ProfileContractError, match="processor_factor"):
        EngineProfile.capture(**inputs)

    inputs = _capture_inputs(tmp_path / "model-mismatch")
    model_path = inputs["artifact_paths"]["model_config"]
    model_path.write_text(
        json.dumps({"vision_config": {"patch_size": 14, "spatial_merge_size": 2}}),
        encoding="utf-8",
    )
    with pytest.raises(ProfileContractError, match="patch/merge"):
        EngineProfile.capture(**inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default_width", 1000, "not divisible"),
        ("min_axis_pixels", 4096, "minimum axis"),
        ("max_total_pixels", 1000, "total-pixel"),
    ],
)
def test_invalid_config_derived_bounds_and_defaults_are_rejected(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["roi_inference"][field] = value

    with pytest.raises(ProfileContractError, match=message):
        EngineProfile.capture(**inputs)


def test_recorded_constraints_cannot_be_replaced_with_caller_assertions(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)

    with pytest.raises(ProfileContractError, match="do not match captured"):
        replace(profile, processor_factor=64)
    with pytest.raises(ProfileContractError, match="do not match captured"):
        replace(profile, default_width=1280)


def test_canvas_bounds_alignment_and_total_pixels_are_fail_closed(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)

    assert profile.validate_canvas(1280, 768) == (1280, 768)
    with pytest.raises(ProfileContractError, match="divisible"):
        profile.validate_canvas(1000, 1024)
    with pytest.raises(ProfileContractError, match="axis bounds"):
        profile.validate_canvas(4096, 32)
    with pytest.raises(ProfileContractError, match="total-pixel"):
        profile.validate_canvas(2048, 2048)


@pytest.mark.parametrize(
    "nested_secret",
    [
        {"transport": {"Authorization": "Bearer abc"}},
        {"transport": {"Cookie": "session=abc"}},
        {"transport": {"headers": {"X-Safe": "value"}}},
        {"provider": {"provider_credentials": {"client_id": "x"}}},
        {"tls": {"private_key": "-----BEGIN PRIVATE KEY-----abc"}},
    ],
)
def test_nested_credential_fields_are_rejected_before_persistence_or_receipt(
    tmp_path: Path, nested_secret: dict[str, Any]
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["runtime_identity"] = nested_secret

    with pytest.raises(ProfileContractError, match="credential-bearing"):
        EngineProfile.capture(**inputs)


def test_receipt_is_allowlisted_and_never_emits_identity_payloads(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    receipt_text = json.dumps(profile.to_receipt_dict(), sort_keys=True)

    assert set(profile.to_receipt_dict()) == {
        "schema_version",
        "profile_name",
        "profile_fingerprint",
        "endpoint",
        "artifacts",
        "identity_fingerprints",
        "processor",
        "deadline_seconds",
    }
    assert "find all objects" not in receipt_text
    assert "cuda:0" not in receipt_text
    assert "max_new_tokens" not in receipt_text
    assert str(tmp_path.resolve()) not in receipt_text
    assert all(
        "path" not in artifact for artifact in profile.to_receipt_dict()["artifacts"]
    )


def test_capture_requires_canonical_resolved_config_and_executed_prompt_policy(
    tmp_path: Path,
) -> None:
    inputs = _capture_inputs(tmp_path)
    resolved = inputs["resolved_config"]
    inputs["resolved_config"] = replace(resolved, fingerprint="0" * 64)
    with pytest.raises(ProfileContractError, match="inconsistent"):
        EngineProfile.capture(**inputs)

    inputs = _capture_inputs(tmp_path / "defaults")
    resolved = inputs["resolved_config"]
    incomplete = dict(resolved.config_dict)
    incomplete["generation"] = dict(incomplete["generation"])
    incomplete["generation"].pop("repetition_penalty")
    inputs["resolved_config"] = replace(resolved, config_dict=incomplete)
    with pytest.raises(ProfileContractError, match="inconsistent"):
        EngineProfile.capture(**inputs)

    profile = _profile(tmp_path / "prompt")
    with pytest.raises(ProfileContractError, match="normalized executed"):
        replace(profile, prompt_policy_json='{"template_id":"caller-authored"}')

    resolved_payload = json.loads(profile.resolved_config_json)
    resolved_payload["checkpoint"] = {"path": "/tmp/unsupported-checkpoint"}
    with pytest.raises(ProfileContractError, match="strict InferConfig"):
        replace(
            profile,
            resolved_config_json=json.dumps(
                resolved_payload, sort_keys=True, separators=(",", ":")
            ),
        )


def test_processor_resize_and_credential_bearing_endpoint_are_forbidden(
    tmp_path: Path,
) -> None:
    inputs = _capture_inputs(tmp_path)
    inputs["processor_kwargs"]["do_resize"] = True
    with pytest.raises(ProfileContractError, match="do_resize=false"):
        EngineProfile.capture(**inputs)

    profile = _profile(tmp_path / "endpoint")
    with pytest.raises(ProfileContractError, match="credentials"):
        replace(profile, endpoint="http://user:secret@127.0.0.1:8123/infer")


def test_full_identity_fields_change_profile_fingerprint(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    changed = replace(
        profile,
        transform_identity_json='{"id":"coordexp-roi-letterbox-half-up-v2"}',
    )

    assert changed.fingerprint != profile.fingerprint
    assert (
        changed.identity_fingerprints["transform"]
        != profile.identity_fingerprints["transform"]
    )


def test_saved_profile_name_cannot_silently_change_identity(tmp_path: Path) -> None:
    store = EngineProfileStore(tmp_path / "profiles.json")
    profile = _profile(tmp_path)
    store.save(profile)

    with pytest.raises(ProfileContractError, match="immutable profile"):
        store.save(replace(profile, endpoint="http://127.0.0.1:8123/other"))
