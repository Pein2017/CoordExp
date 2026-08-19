from __future__ import annotations

import copy
import json
import io
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.artifacts import training_state
from src.artifacts.training_state import (
    REQUIRED_IDENTITY_KINDS,
    REQUIRED_RNG_KINDS,
    RankTrainingStatePayload,
    TrainingStateExpectations,
    TrainingStateManifest,
    TrainingStatePublication,
    TrainingStatePublicationError,
    admit_training_state,
    build_training_state_manifest,
    load_training_state_manifest,
    publish_training_state,
)
from src.common.errors import ArtifactContractError


def _identities(*, changed: str | None = None) -> dict[str, str]:
    result = {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(REQUIRED_IDENTITY_KINDS)
    }
    if changed is not None:
        result[changed] = "f" * 64
    return result


def _full_resolved_config() -> dict[str, Any]:
    return {
        "config": {
            "adapter": {"rank": 16},
            "checkpoint": {"save_final": True, "steps": [5]},
            "data": {"train": {"path": "/data/train.jsonl"}},
            "eval": {"forward": {"steps": [5]}},
            "losses": {"coordinate": {"weight": 1.0}},
            "model": {"dtype": "bfloat16"},
            "optimizer": {
                "lr": 0.0002,
                "scheduler": {"name": "cosine", "warmup_ratio": 0.1},
            },
            "packing": {"policy": "source_order_next_fit"},
            "resume": {"checkpoint_dir": None, "mode": "disabled"},
            "run": {
                "artifact_root": "/outputs/parent",
                "name": "parent",
                "output_dir": None,
            },
            "runtime": {"world_size": 8},
            "training": {
                "forward_input_provider_mode": "synchronous",
                "precision": "bf16",
                "seed": 17,
            },
        },
        "resolution": {
            "entry_config_path": "/configs/parent.yaml",
            "fingerprint": "parent-fingerprint",
            "loader_version": "coordexp-swift-config-v1",
            "path_origins": {},
            "schema_version": 1,
            "sources": [{"path": "/configs/parent.yaml", "sha256": "a" * 64}],
        },
    }


def test_resume_compatibility_projection_allows_only_continuation_metadata() -> None:
    parent = _full_resolved_config()
    child = copy.deepcopy(parent)
    child["config"]["run"] = {
        "artifact_root": "/outputs/child",
        "name": "child",
        "output_dir": "/outputs/child/segment",
    }
    child["config"]["resume"] = {
        "checkpoint_dir": "/outputs/parent/checkpoints/step-5",
        "mode": "exact_same_world_size",
    }
    child["resolution"] = {
        **child["resolution"],
        "entry_config_path": "/configs/child.yaml",
        "fingerprint": "child-fingerprint",
        "path_origins": {
            "resume.checkpoint_dir": {
                "declared_path": "../parent/checkpoints/step-5",
                "declaring_config_path": "/configs/child.yaml",
                "resolved_path": "/outputs/parent/checkpoints/step-5",
            }
        },
        "sources": [{"path": "/configs/child.yaml", "sha256": "b" * 64}],
    }

    assert training_state.build_resume_compatibility_projection(parent) == (
        training_state.build_resume_compatibility_projection(child)
    )


def _rank_payload(
    rank: int,
    *,
    scheduler: bool = True,
    scaler: bool = True,
) -> RankTrainingStatePayload:
    return _serialized_real_rank(
        rank,
        scheduler_applicable=scheduler,
        scaler_applicable=scaler,
    )


def _publication(
    *,
    world_size: int = 2,
    step: int = 17,
    scheduler: bool = True,
    scaler: bool = True,
    reverse_ranks: bool = False,
) -> TrainingStatePublication:
    ranks = [
        _rank_payload(rank, scheduler=scheduler, scaler=scaler)
        for rank in range(world_size)
    ]
    if reverse_ranks:
        ranks.reverse()
    resolved_config, identities = _resolved_config_and_identities()
    return TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=step,
        continuation_index=2,
        world_size=world_size,
        identities=identities,
        scheduler_applicable=scheduler,
        scaler_applicable=scaler,
        rank_payloads=ranks,
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
    )


def _expectations(
    publication: TrainingStatePublication,
    *,
    identities: dict[str, str] | None = None,
    world_size: int | None = None,
    step: int | None = None,
    scheduler: bool | None = None,
    scaler: bool | None = None,
) -> TrainingStateExpectations:
    decoded = training_state._decode_rank_payload(publication.rank_payloads[0])
    return TrainingStateExpectations(
        checkpoint_step=publication.checkpoint_step if step is None else step,
        world_size=publication.world_size if world_size is None else world_size,
        identities=publication.identities if identities is None else identities,
        scheduler_applicable=(
            publication.scheduler_applicable if scheduler is None else scheduler
        ),
        scaler_applicable=publication.scaler_applicable if scaler is None else scaler,
        resolved_config=publication.resolved_config,
        resume_compatibility=publication.resume_compatibility,
        runtime_state=training_state.RuntimeStateExpectations(
            structure=decoded.structure,
            signature=decoded.signature,
        ),
    )


def _checkpoint(tmp_path: Path, name: str = "step-17") -> Path:
    checkpoint = tmp_path / "checkpoints" / name
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"inference-still-loadable")
    return checkpoint


def _publish(
    tmp_path: Path,
    *,
    publication: TrainingStatePublication | None = None,
    name: str = "step-17",
) -> tuple[Path, TrainingStatePublication]:
    checkpoint = _checkpoint(tmp_path, name)
    selected = publication or _publication()
    publish_training_state(checkpoint, selected)
    return checkpoint, selected


def _manifest_path(checkpoint: Path) -> Path:
    return checkpoint / "training_state" / "manifest.json"


def _rewrite_manifest(checkpoint: Path, mutate: Any, *, redigest: bool = True) -> None:
    path = _manifest_path(checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    if redigest:
        unsigned = dict(payload)
        unsigned.pop("aggregate_digest", None)
        payload["aggregate_digest"] = training_state._sha256(
            training_state._canonical_json_bytes(unsigned)
        )
    path.write_bytes(training_state._canonical_json_bytes(payload) + b"\n")


def _rewrite_cursor_bytes(checkpoint: Path, rank: int, encoded: bytes) -> None:
    relative = f"rank-{rank:05d}/cursor.json"
    cursor_path = checkpoint / "training_state" / relative
    cursor_path.write_bytes(encoded)

    def update_cursor_record(payload: dict[str, Any]) -> None:
        rank_row = next(row for row in payload["ranks"] if row["rank"] == rank)
        file_row = next(row for row in rank_row["files"] if row["path"] == relative)
        file_row["size"] = len(encoded)
        file_row["sha256"] = training_state._sha256(encoded)

    _rewrite_manifest(checkpoint, update_cursor_record)


def _rewrite_role_bytes(checkpoint: Path, rank: int, role: str, encoded: bytes) -> None:
    manifest = json.loads(_manifest_path(checkpoint).read_text(encoding="utf-8"))
    rank_row = next(row for row in manifest["ranks"] if row["rank"] == rank)
    file_row = next(row for row in rank_row["files"] if row["role"] == role)
    (checkpoint / "training_state" / file_row["path"]).write_bytes(encoded)

    def update(payload: dict[str, Any]) -> None:
        selected_rank = next(row for row in payload["ranks"] if row["rank"] == rank)
        selected_file = next(
            row for row in selected_rank["files"] if row["role"] == role
        )
        selected_file["size"] = len(encoded)
        selected_file["sha256"] = training_state._sha256(encoded)

    _rewrite_manifest(checkpoint, update)


def _assert_code(
    exc_info: pytest.ExceptionInfo[ArtifactContractError], code: str
) -> None:
    assert exc_info.value.code == code


def test_manifest_strict_roundtrip_and_deterministic_digest() -> None:
    first, first_files = build_training_state_manifest(_publication())
    second, second_files = build_training_state_manifest(
        _publication(reverse_ranks=True)
    )

    roundtrip = TrainingStateManifest.from_dict(
        json.loads(json.dumps(first.to_dict(), sort_keys=True))
    )

    assert roundtrip == first
    assert first.aggregate_digest == first.computed_aggregate_digest()
    assert first.aggregate_digest == second.aggregate_digest
    assert dict(first_files) == dict(second_files)
    assert len(first.aggregate_digest) == 64
    assert [rank.rank for rank in first.ranks] == [0, 1]
    assert first.optimizer_applicable is True


def test_strict_manifest_rejects_unknown_fields_and_digest_mutation() -> None:
    manifest, _ = build_training_state_manifest(_publication())
    extra = manifest.to_dict()
    extra["future"] = True
    with pytest.raises(ArtifactContractError) as exc_info:
        TrainingStateManifest.from_dict(extra)
    _assert_code(exc_info, "training_state.schema")

    corrupt = manifest.to_dict()
    corrupt["parent_run_id"] = "different"
    with pytest.raises(ArtifactContractError) as exc_info:
        TrainingStateManifest.from_dict(corrupt)
    _assert_code(exc_info, "training_state.corrupt_manifest")


def test_publication_and_admission_authenticate_every_rank_payload(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)

    admitted = admit_training_state(
        checkpoint, _expectations(publication), current_rank=1
    )

    assert admitted.manifest.parent_run_id == "run-parent"
    assert admitted.manifest.parent_segment_id == "segment-2"
    assert admitted.manifest.continuation_index == 2
    assert admitted.manifest.save_boundary == "optimizer_step"
    assert admitted.bytes_for(rank=1, role="optimizer")
    assert set(admitted.files) == {
        training_state.TRAINING_STATE_RESOLVED_CONFIG,
        training_state.TRAINING_STATE_RESUME_COMPATIBILITY,
        *(file.path for file in admitted.manifest.ranks[1].files),
    }
    assert admitted.decoded_ranks[1].optimizer["class"].endswith(".Adam")
    cursor = json.loads(admitted.bytes_for(rank=1, role="cursor"))
    assert cursor["schema"] == training_state.TRAINING_STATE_CURSOR_SCHEMA
    assert cursor["next_rank_local_micro_step"] == 31
    assert cursor["pack"]["state"] == {"ordinal": 11, "pending": []}
    assert cursor["data"]["next_rank_local_micro_step"] == 31
    assert (
        checkpoint / "adapter.safetensors"
    ).read_bytes() == b"inference-still-loadable"
    assert load_training_state_manifest(checkpoint) == admitted.manifest


def test_scheduler_and_scaler_applicability_are_strict(tmp_path: Path) -> None:
    publication = _publication(scheduler=False, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)

    admitted = admit_training_state(
        checkpoint, _expectations(publication), current_rank=0
    )
    roles = {file.role for file in admitted.manifest.ranks[0].files}

    assert "scheduler" not in roles
    assert "scaler" not in roles
    assert admitted.manifest.scheduler_applicable is False
    assert admitted.manifest.scaler_applicable is False


def test_publication_rejects_incomplete_rank_set_before_filesystem_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = replace(_publication(), rank_payloads=(_rank_payload(0),))

    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(checkpoint, publication)

    _assert_code(exc_info, "training_state.incomplete_rank_set")
    assert not (checkpoint / "training_state").exists()
    assert not list(checkpoint.glob(".training_state.*.tmp"))


def test_publication_rejects_missing_rank_component_and_rng_kind(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    rank_zero = _rank_payload(0)
    rank_one = replace(_rank_payload(1), optimizer=b"")
    publication = replace(_publication(), rank_payloads=(rank_zero, rank_one))
    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(checkpoint, publication)
    _assert_code(exc_info, "training_state.incomplete")

    missing_rng = dict(rank_one.rng)
    missing_rng.pop("numpy")
    publication = replace(
        _publication(),
        rank_payloads=(rank_zero, replace(_rank_payload(1), rng=missing_rng)),
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(checkpoint, publication)
    _assert_code(exc_info, "training_state.incomplete")
    assert not (checkpoint / "training_state").exists()


def test_mid_accumulation_save_is_explicitly_unsupported(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = replace(_publication(), accumulation_microstep=1)

    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(checkpoint, publication)

    _assert_code(exc_info, "training_state.unsupported_mid_accumulation")
    assert not (checkpoint / "training_state").exists()


def test_model_only_and_uncommitted_checkpoints_are_rejected(tmp_path: Path) -> None:
    model_only = _checkpoint(tmp_path, "model-only")
    publication = _publication()
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(model_only, _expectations(publication), current_rank=0)
    _assert_code(exc_info, "training_state.model_only")

    uncommitted = _checkpoint(tmp_path, "uncommitted")
    (uncommitted / "training_state").mkdir()
    (uncommitted / "training_state" / "rank-00000").mkdir()
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(uncommitted, _expectations(publication), current_rank=0)
    _assert_code(exc_info, "training_state.uncommitted")


def test_manifest_with_noncommitted_status_is_rejected(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)
    _rewrite_manifest(
        checkpoint, lambda payload: payload.__setitem__("commit_status", "staging")
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(checkpoint, _expectations(publication), current_rank=0)

    _assert_code(exc_info, "training_state.uncommitted")


def test_inference_minimal_artifact_type_is_not_exact_state(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)
    _rewrite_manifest(
        checkpoint,
        lambda payload: payload.__setitem__("artifact_type", "inference_checkpoint"),
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(checkpoint, _expectations(publication), current_rank=0)

    _assert_code(exc_info, "training_state.model_only")


@pytest.mark.parametrize("identity", REQUIRED_IDENTITY_KINDS)
def test_admission_rejects_every_identity_mismatch_before_mutation(
    tmp_path: Path, identity: str
) -> None:
    checkpoint, publication = _publish(tmp_path)
    calls: list[str] = []

    current_identities = dict(publication.identities)
    current_identities[identity] = "f" * 64
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication, identities=current_identities),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert calls == []
    assert exc_info.value.context["mismatches"][0]["field"] == f"identities.{identity}"


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"world_size": 3}, "world_size"),
        ({"step": 18}, "checkpoint_step"),
        ({"scheduler": False}, "applicability.scheduler"),
        ({"scaler": False}, "applicability.scaler"),
    ],
)
def test_admission_rejects_runtime_shape_mismatch_before_mutation(
    tmp_path: Path, override: dict[str, Any], field: str
) -> None:
    checkpoint, publication = _publish(tmp_path)
    calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication, **override),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert calls == []
    assert exc_info.value.context["mismatches"][0]["field"] == field


def test_admission_callback_runs_only_after_complete_validation(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)
    calls: list[str] = []

    result = admit_training_state(
        checkpoint,
        _expectations(publication),
        current_rank=0,
        on_admitted=lambda state: calls.append(state.manifest.aggregate_digest)
        or "restored",
    )

    assert result == "restored"
    assert calls == [load_training_state_manifest(checkpoint).aggregate_digest]


def test_corrupt_and_missing_components_are_rejected_before_callback(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    calls: list[str] = []
    optimizer = checkpoint / "training_state" / "rank-00001" / "optimizer.bin"
    optimizer.write_bytes(b"tampered")

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.corrupt_component")
    assert calls == []

    optimizer.unlink()
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )
    _assert_code(exc_info, "training_state.incomplete")
    assert calls == []


def test_undeclared_file_is_rejected_before_callback(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)
    calls: list[str] = []
    (checkpoint / "training_state" / "extra.bin").write_bytes(b"undeclared")

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incomplete")
    assert calls == []


def test_manifest_path_escape_is_rejected_before_file_access(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)

    def escape(payload: dict[str, Any]) -> None:
        payload["ranks"][0]["files"][0]["path"] = "../outside.bin"

    _rewrite_manifest(checkpoint, escape)
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(checkpoint, _expectations(publication), current_rank=0)
    _assert_code(exc_info, "training_state.unsafe_path")


def test_symlinked_checkpoint_state_and_component_are_rejected(tmp_path: Path) -> None:
    real_checkpoint, publication = _publish(tmp_path, name="real")
    linked_checkpoint = tmp_path / "linked-checkpoint"
    linked_checkpoint.symlink_to(real_checkpoint, target_is_directory=True)
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            linked_checkpoint, _expectations(publication), current_rank=0
        )
    _assert_code(exc_info, "training_state.unsafe_path")

    state_link_checkpoint = _checkpoint(tmp_path, "state-link")
    (state_link_checkpoint / "training_state").symlink_to(
        real_checkpoint / "training_state", target_is_directory=True
    )
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            state_link_checkpoint, _expectations(publication), current_rank=0
        )
    _assert_code(exc_info, "training_state.unsafe_path")

    optimizer = real_checkpoint / "training_state" / "rank-00000" / "optimizer.bin"
    original = optimizer.read_bytes()
    external = tmp_path / "external.bin"
    external.write_bytes(original)
    optimizer.unlink()
    optimizer.symlink_to(external)
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            real_checkpoint, _expectations(publication), current_rank=0
        )
    _assert_code(exc_info, "training_state.unsafe_path")


def test_publication_never_replaces_same_or_different_existing_target(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    before = {
        path.relative_to(checkpoint).as_posix(): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }

    for candidate in (publication, replace(publication, parent_run_id="other-run")):
        with pytest.raises(ArtifactContractError) as exc_info:
            publish_training_state(checkpoint, candidate)
        _assert_code(exc_info, "training_state.immutable_collision")

    after = {
        path.relative_to(checkpoint).as_posix(): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_concurrent_publishers_have_exactly_one_winner(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication()

    def attempt() -> str:
        try:
            publish_training_state(checkpoint, publication)
        except ArtifactContractError as exc:
            return exc.code
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _index: attempt(), range(2)))

    assert outcomes == ["published", "training_state.immutable_collision"]
    admitted = admit_training_state(
        checkpoint, _expectations(publication), current_rank=0
    )
    assert admitted.manifest.world_size == 2
    assert not list(checkpoint.glob(".training_state.*.tmp"))


def test_preinstall_failure_leaves_no_final_target_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)

    def fail_install(_checkpoint_fd: int, _stage: str, _target: str) -> None:
        raise OSError("injected pre-install failure")

    monkeypatch.setattr(training_state, "_rename_directory_no_replace_at", fail_install)
    with pytest.raises(TrainingStatePublicationError) as exc_info:
        publish_training_state(checkpoint, _publication())

    assert exc_info.value.installed_by_this_call is False
    assert exc_info.value.reloaded_exact is False
    assert not (checkpoint / "training_state").exists()
    assert not list(checkpoint.glob(".training_state.*.tmp"))


def test_cursor_exponent_overflow_is_rejected_before_admission_callback(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    cursor_path = checkpoint / "training_state/rank-00000/cursor.json"
    encoded = cursor_path.read_bytes().replace(b'"ordinal":20', b'"ordinal":1e999')
    assert encoded != cursor_path.read_bytes()
    _rewrite_cursor_bytes(checkpoint, 0, encoded)
    callbacks = 0

    def admitted(_state: Any) -> None:
        nonlocal callbacks
        callbacks += 1

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=admitted,
        )
    _assert_code(exc_info, "training_state.schema")
    assert callbacks == 0


def test_cursor_owner_state_must_be_a_nonempty_mapping_before_callback(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    cursor_path = checkpoint / "training_state/rank-00000/cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    cursor["data"]["state"] = None
    _rewrite_cursor_bytes(
        checkpoint,
        0,
        training_state._canonical_json_bytes(cursor) + b"\n",
    )
    callbacks = 0

    def admitted(_state: Any) -> None:
        nonlocal callbacks
        callbacks += 1

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=admitted,
        )
    _assert_code(exc_info, "training_state.schema")
    assert callbacks == 0


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_checkpoint_path_replacement_before_install_never_receives_state(
    tmp_path: Path, replacement_kind: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    moved = checkpoint.with_name("selected-checkpoint-moved")

    def replace_selected_path() -> None:
        checkpoint.rename(moved)
        if replacement_kind == "directory":
            checkpoint.mkdir()
        else:
            checkpoint.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(
            checkpoint,
            _publication(),
            before_install=replace_selected_path,
        )
    _assert_code(exc_info, "training_state.path_owner_drift")
    assert not (moved / "training_state").exists()
    if replacement_kind == "directory":
        assert not (checkpoint / "training_state").exists()
    assert not list(moved.glob(".training_state.*.tmp"))


def test_postinstall_failure_reports_owned_exact_reload(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    def fail_after_install() -> None:
        raise OSError("injected post-install failure")

    with pytest.raises(TrainingStatePublicationError) as exc_info:
        publish_training_state(
            checkpoint,
            _publication(),
            on_installed=fail_after_install,
        )

    assert exc_info.value.installed_by_this_call is True
    assert exc_info.value.reloaded_exact is True
    assert load_training_state_manifest(checkpoint).commit_status == "committed"


def test_postinstall_mutation_never_claims_exact_reload(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    def mutate_after_install() -> None:
        target = checkpoint / "training_state"
        (target / "rank-00000" / "optimizer.bin").write_bytes(b"mutated")
        raise OSError("injected post-install mutation")

    with pytest.raises(TrainingStatePublicationError) as exc_info:
        publish_training_state(
            checkpoint,
            _publication(),
            on_installed=mutate_after_install,
        )

    assert exc_info.value.installed_by_this_call is True
    assert exc_info.value.reloaded_exact is False


def test_committed_exact_state_is_never_automatically_pruned(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path, "step-17")
    second = _checkpoint(tmp_path, "step-18")
    first_publication = _publication(step=17)
    second_publication = _publication(step=18)

    publish_training_state(first, first_publication)
    first_before = _manifest_path(first).read_bytes()
    publish_training_state(second, second_publication)

    assert _manifest_path(first).read_bytes() == first_before
    assert (
        admit_training_state(
            first, _expectations(first_publication), current_rank=0
        ).manifest.checkpoint_step
        == 17
    )
    assert (
        admit_training_state(
            second, _expectations(second_publication), current_rank=0
        ).manifest.checkpoint_step
        == 18
    )


def test_manifest_rank_set_mutation_is_rejected_as_incomplete(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)

    def drop_rank(payload: dict[str, Any]) -> None:
        payload["ranks"].pop()

    _rewrite_manifest(checkpoint, drop_rank)
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(checkpoint, _expectations(publication), current_rank=0)
    _assert_code(exc_info, "training_state.incomplete_rank_set")


def _expectations_with_current_config(
    publication: TrainingStatePublication,
    current_config: dict[str, Any],
) -> TrainingStateExpectations:
    projection = training_state.build_resume_compatibility_projection(current_config)
    identities = dict(publication.identities)
    identities["resolved_config"] = training_state._sha256(
        training_state._canonical_json_bytes(current_config) + b"\n"
    )
    identities["resume_compatibility"] = training_state._sha256(
        training_state._canonical_json_bytes(projection) + b"\n"
    )
    return replace(
        _expectations(publication),
        identities=identities,
        resolved_config=current_config,
        resume_compatibility=projection,
    )


def test_full_parent_config_and_semantic_projection_are_separately_authenticated(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    manifest = load_training_state_manifest(checkpoint)

    full_record = manifest.resolved_config
    full_bytes = (
        training_state._canonical_json_bytes(publication.resolved_config) + b"\n"
    )
    assert (checkpoint / "training_state" / full_record.path).read_bytes() == full_bytes
    assert full_record.role == "resolved_config"
    assert full_record.sha256 == publication.identities["resolved_config"]

    compatibility_record = manifest.resume_compatibility
    compatibility_bytes = (
        training_state._canonical_json_bytes(publication.resume_compatibility) + b"\n"
    )
    assert (
        checkpoint / "training_state" / compatibility_record.path
    ).read_bytes() == compatibility_bytes
    assert compatibility_record.role == "resume_compatibility"
    assert compatibility_record.sha256 == publication.identities["resume_compatibility"]


def test_child_may_change_only_run_resume_and_resolution_provenance(
    tmp_path: Path,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    child = copy.deepcopy(publication.resolved_config)
    child["config"]["run"] = {
        "artifact_root": "/outputs/child",
        "name": "child",
        "output_dir": "/outputs/child/segment",
    }
    child["config"]["resume"] = {
        "checkpoint_dir": str(checkpoint),
        "mode": "exact_same_world_size",
    }
    child["resolution"]["entry_config_path"] = "/configs/child.yaml"
    child["resolution"]["fingerprint"] = "child-fingerprint"
    child["resolution"]["sources"] = [
        {"path": "/configs/child.yaml", "sha256": "b" * 64}
    ]
    child["resolution"]["path_origins"] = {
        "resume.checkpoint_dir": {
            "declared_path": "../parent/checkpoint",
            "declaring_config_path": "/configs/child.yaml",
            "resolved_path": str(checkpoint),
        }
    }

    admitted = admit_training_state(
        checkpoint,
        _expectations_with_current_config(publication, child),
        current_rank=0,
    )

    assert admitted.resolved_config == publication.resolved_config
    assert admitted.resume_compatibility == publication.resume_compatibility


@pytest.mark.parametrize(
    ("path", "value", "expected_field"),
    [
        (("data", "train", "path"), "/data/other.jsonl", "data.train.path"),
        (("training", "seed"), 18, "training.seed"),
        (("packing", "policy"), "window_binpack", "packing.policy"),
        (("optimizer", "lr"), 0.0003, "optimizer.lr"),
        (
            ("optimizer", "scheduler", "name"),
            "linear",
            "optimizer.scheduler.name",
        ),
        (("losses", "coordinate", "weight"), 0.5, "losses.coordinate.weight"),
        (
            ("training", "forward_input_provider_mode"),
            "overlapped",
            "training.forward_input_provider_mode",
        ),
        (("training", "precision"), "fp32", "training.precision"),
        (("eval", "forward", "steps"), [6], "eval.forward.steps.0"),
        (("checkpoint", "steps"), [6], "checkpoint.steps.0"),
    ],
)
def test_training_semantic_config_changes_are_field_diagnostic(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
    expected_field: str,
) -> None:
    checkpoint, publication = _publish(tmp_path)
    child = copy.deepcopy(publication.resolved_config)
    owner = child["config"]
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations_with_current_config(publication, child),
            current_rank=0,
        )

    _assert_code(exc_info, "training_state.incompatible")
    fields = [row["field"] for row in exc_info.value.context["mismatches"]]
    assert "identities.resume_compatibility" in fields
    assert f"resume_compatibility.semantic_config.{expected_field}" in fields


def test_topology_identity_change_is_rejected_separately(tmp_path: Path) -> None:
    checkpoint, publication = _publish(tmp_path)
    identities = dict(publication.identities)
    identities["topology"] = "f" * 64

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            replace(_expectations(publication), identities=identities),
            current_rank=0,
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert [row["field"] for row in exc_info.value.context["mismatches"]] == [
        "identities.topology"
    ]


def test_schema_v1_manifest_is_explicitly_unsupported() -> None:
    manifest, _ = build_training_state_manifest(_publication())
    payload = manifest.to_dict()
    payload["schema_version"] = 1
    payload.pop("resume_compatibility")
    payload["identities"].pop("resume_compatibility")

    with pytest.raises(ArtifactContractError) as exc_info:
        TrainingStateManifest.from_dict(payload)

    _assert_code(exc_info, "training_state.unsupported_schema")


def test_unknown_schema_value_is_rejected() -> None:
    manifest, _ = build_training_state_manifest(_publication())

    unknown_schema_family = manifest.to_dict()
    unknown_schema_family["schema"] = "coordexp-swift-training-state-v99"
    with pytest.raises(ArtifactContractError) as exc_info:
        TrainingStateManifest.from_dict(unknown_schema_family)
    _assert_code(exc_info, "training_state.unsupported_schema")

    unknown_schema_version = manifest.to_dict()
    unknown_schema_version["schema_version"] = 99
    with pytest.raises(ArtifactContractError) as exc_info:
        TrainingStateManifest.from_dict(unknown_schema_version)
    _assert_code(exc_info, "training_state.unsupported_schema")


def _real_runtime_state() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = model(torch.arange(6, dtype=torch.float32).reshape(2, 3)).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, scheduler


def _serialized_real_rank(
    rank: int,
    *,
    dtype: torch.dtype = torch.float32,
    scheduler_applicable: bool = True,
    scaler_applicable: bool = False,
    torch_cuda_rng_states: tuple[torch.Tensor, ...] | None = None,
) -> RankTrainingStatePayload:
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).to(
        dtype=dtype
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        if scheduler_applicable
        else None
    )
    scaler = torch.amp.GradScaler("cpu") if scaler_applicable else None
    values = torch.arange(6, dtype=dtype).reshape(2, 3)
    loss = model(values).square().sum()
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return training_state.serialize_rank_training_state(
        rank=rank,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=(
            torch_cuda_rng_states
            if torch_cuda_rng_states is not None
            else (torch.arange(32, dtype=torch.uint8),)
        ),
        cursor={
            "data": {"epoch": 2, "ordinal": 20 + rank},
            "pack": {"ordinal": 10 + rank, "pending": []},
        },
        next_rank_local_micro_step=30 + rank,
    )


def _resolved_config_and_identities() -> tuple[dict[str, Any], dict[str, str]]:
    resolved_config = _full_resolved_config()
    resolved_config["config"]["model"] = {"dtype": "float32"}
    resume_compatibility = training_state.build_resume_compatibility_projection(
        resolved_config
    )
    identities = _identities()
    identities["resolved_config"] = training_state._sha256(
        training_state._canonical_json_bytes(resolved_config) + b"\n"
    )
    identities["resume_compatibility"] = training_state._sha256(
        training_state._canonical_json_bytes(resume_compatibility) + b"\n"
    )
    return resolved_config, identities


def test_repository_codec_decodes_real_runtime_state_before_callback(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _real_runtime_state()
    simulated_cuda_rng = torch.arange(32, dtype=torch.uint8)
    rank_payload = training_state.serialize_rank_training_state(
        rank=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=(simulated_cuda_rng,),
        cursor={
            "data": {"epoch": 2, "ordinal": 20},
            "pack": {"ordinal": 10, "pending": []},
        },
        next_rank_local_micro_step=30,
    )
    resolved_config, identities = _resolved_config_and_identities()
    publication = TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=17,
        continuation_index=2,
        world_size=1,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        rank_payloads=(rank_payload,),
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
    )
    checkpoint = _checkpoint(tmp_path)
    publish_training_state(checkpoint, publication)
    expectations = TrainingStateExpectations(
        checkpoint_step=17,
        world_size=1,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
        runtime_state=training_state.capture_runtime_state_expectations(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
        ),
    )
    calls: list[str] = []

    def inspect(admitted: Any) -> str:
        calls.append("decoded")
        decoded = admitted.decoded_ranks[0]
        assert tuple(decoded.trainable_model) == (
            "0.bias",
            "0.weight",
            "1.bias",
            "1.weight",
        )
        assert decoded.optimizer["class"].endswith(".Adam")
        assert decoded.scheduler["state"]["last_epoch"] == 1
        assert decoded.scaler is None
        assert decoded.python_rng_state[0] == random.getstate()[0]
        assert decoded.numpy_rng_state[0] == np.random.get_state()[0]
        assert torch.equal(decoded.torch_cpu_rng_state, torch.get_rng_state())
        assert torch.equal(decoded.torch_cuda_rng_states[0], simulated_cuda_rng)
        assert decoded.cursor["pack"]["state"] == {"ordinal": 10, "pending": []}
        return "accepted"

    assert (
        admit_training_state(
            checkpoint, expectations, current_rank=0, on_admitted=inspect
        )
        == "accepted"
    )
    assert calls == ["decoded"]


def test_publication_rejects_cross_rank_runtime_signature_drift(
    tmp_path: Path,
) -> None:
    resolved_config, identities = _resolved_config_and_identities()
    publication = TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=17,
        continuation_index=2,
        world_size=2,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        rank_payloads=(
            _serialized_real_rank(0, dtype=torch.float32),
            _serialized_real_rank(1, dtype=torch.float64),
        ),
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
    )
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(ArtifactContractError) as exc_info:
        publish_training_state(checkpoint, publication)

    _assert_code(exc_info, "training_state.runtime_state")
    assert exc_info.value.context["field"] == "runtime.signature"
    assert not (checkpoint / "training_state").exists()


def test_mutation_callback_requires_current_runtime_signature(tmp_path: Path) -> None:
    resolved_config, identities = _resolved_config_and_identities()
    publication = TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=17,
        continuation_index=2,
        world_size=1,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        rank_payloads=(_serialized_real_rank(0),),
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
    )
    checkpoint = _checkpoint(tmp_path)
    publish_training_state(checkpoint, publication)
    calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            TrainingStateExpectations(
                checkpoint_step=17,
                world_size=1,
                identities=identities,
                scheduler_applicable=True,
                scaler_applicable=False,
                resolved_config=resolved_config,
                resume_compatibility=training_state.build_resume_compatibility_projection(
                    resolved_config
                ),
            ),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incomplete")
    assert exc_info.value.context["field"] == "expected.runtime_state"
    assert calls == []


def test_mutation_callback_requires_current_resolved_config(tmp_path: Path) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            replace(_expectations(publication), resolved_config=None),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incomplete")
    assert exc_info.value.context["field"] == "expected.resolved_config"
    assert calls == []


def _runtime_expectations_variant(
    variant: str,
) -> training_state.RuntimeStateExpectations:
    torch.manual_seed(41)
    dtype = torch.float64 if variant == "dtype" else torch.float32
    if variant == "missing":
        model = torch.nn.Sequential(torch.nn.Linear(3, 4)).to(dtype=dtype)
    elif variant == "shape":
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2)).to(
            dtype=dtype
        )
    else:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).to(
            dtype=dtype
        )
    if variant == "groups":
        optimizer = torch.optim.Adam(
            [
                {"params": list(model[0].parameters())},
                {"params": list(model[1].parameters())},
            ],
            lr=0.01,
        )
    elif variant == "class":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model(torch.arange(6, dtype=dtype).reshape(2, 3)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return training_state.capture_runtime_state_expectations(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
    )


@pytest.mark.parametrize(
    ("variant", "expected_field"),
    [
        ("missing", "ranks.0.runtime.trainable_model.parameters.1.bias"),
        ("shape", "ranks.0.runtime.trainable_model.parameters.0.bias.shape.0"),
        ("dtype", "ranks.0.runtime.trainable_model.parameters.0.bias.dtype"),
        ("groups", "ranks.0.runtime.optimizer.param_groups.length"),
        ("class", "ranks.0.runtime.optimizer.class"),
    ],
)
def test_runtime_signature_drift_is_field_diagnostic_before_callback(
    tmp_path: Path, variant: str, expected_field: str
) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    calls: list[str] = []
    expectations = replace(
        _expectations(publication),
        runtime_state=_runtime_expectations_variant(variant),
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            expectations,
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert calls == []
    fields = [row["field"] for row in exc_info.value.context["mismatches"]]
    assert expected_field in fields


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_optimizer_slot_semantic_drift_is_rejected_before_callback(
    tmp_path: Path, mutation: str
) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    optimizer_path = checkpoint / "training_state/rank-00000/optimizer.bin"
    envelope = torch.load(
        io.BytesIO(optimizer_path.read_bytes()), map_location="cpu", weights_only=True
    )
    slots = envelope["state"]["0.weight"]
    if mutation == "shape":
        slots["exp_avg"] = torch.zeros(1, dtype=torch.float32)
    else:
        slots["exp_avg"] = slots["exp_avg"].to(dtype=torch.float64)
    encoded = training_state._torch_envelope_bytes(envelope)
    _rewrite_role_bytes(checkpoint, 0, "optimizer", encoded)
    calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )

    _assert_code(exc_info, "training_state.runtime_state")
    assert calls == []
    assert exc_info.value.context["field"].startswith("runtime.optimizer.state")


@pytest.mark.parametrize("role", REQUIRED_RNG_KINDS)
def test_semantically_invalid_rng_is_rejected_before_callback(
    tmp_path: Path, role: str
) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    component_role = f"rng:{role}"
    manifest = load_training_state_manifest(checkpoint)
    record = next(row for row in manifest.ranks[0].files if row.role == component_role)
    original = (checkpoint / "training_state" / record.path).read_bytes()
    if role in {"python", "numpy"}:
        envelope = json.loads(original)
        if role == "python":
            envelope["state"]["internal"] = [1, 2]
        else:
            envelope["state"]["algorithm"] = "PCG64"
        encoded = training_state._canonical_json_bytes(envelope) + b"\n"
    else:
        envelope = torch.load(
            io.BytesIO(original), map_location="cpu", weights_only=True
        )
        if role == "torch_cpu":
            envelope["states"] = [torch.zeros(8, dtype=torch.float32)]
        else:
            envelope["states"] = []
        encoded = training_state._torch_envelope_bytes(envelope)
    _rewrite_role_bytes(checkpoint, 0, component_role, encoded)
    calls: list[str] = []

    with pytest.raises(ArtifactContractError):
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=0,
            on_admitted=lambda _state: calls.append("mutated"),
        )
    assert calls == []


def test_distributed_rank_contributions_commit_one_complete_atomic_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=2, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)

    session = training_state.begin_training_state_contributions(checkpoint, plan)
    assert session.world_size == 2
    assert session.plan_digest
    assert session.stage_path.parent == checkpoint
    assert session.stage_path.name.startswith(".training_state.")
    assert not (checkpoint / "training_state").exists()

    first = training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[0]
    )
    second = training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[1]
    )
    assert first.rank == 0
    assert second.rank == 1
    assert first.path.name == "rank-00000"
    assert second.path.name == "rank-00001"
    assert session.stage_path.exists()

    published = training_state.commit_training_state_contributions(checkpoint, session)

    assert published.path == checkpoint / "training_state"
    assert not session.stage_path.exists()
    assert published.path.is_dir()
    assert not (published.path / "contribution-plan.json").exists()
    assert not list(published.path.rglob("contribution.json"))
    admitted = admit_training_state(
        checkpoint, _expectations(publication), current_rank=1
    )
    assert set(admitted.decoded_ranks) == {1}


def test_incomplete_distributed_commit_preserves_owned_forensic_stage(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=2, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    session = training_state.begin_training_state_contributions(checkpoint, plan)
    training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[0]
    )

    with pytest.raises(training_state.TrainingStateContributionError) as exc_info:
        training_state.commit_training_state_contributions(checkpoint, session)

    _assert_code(exc_info, "training_state.incomplete_rank_set")
    assert exc_info.value.stage_path == session.stage_path
    assert exc_info.value.session_id == session.session_id
    assert exc_info.value.owns_stage is True
    assert session.stage_path.is_dir()
    assert (session.stage_path / "rank-00000").is_dir()
    assert not (checkpoint / "training_state").exists()


def test_abort_removes_only_the_authenticated_pre_manifest_session(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=1, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    owned_session = training_state.begin_training_state_contributions(checkpoint, plan)
    other_session = training_state.begin_training_state_contributions(checkpoint, plan)
    training_state.publish_rank_training_state_contribution(
        checkpoint, owned_session, publication.rank_payloads[0]
    )
    other_plan_before = (
        other_session.stage_path / training_state.TRAINING_STATE_CONTRIBUTION_PLAN
    ).read_bytes()
    inference_before = (checkpoint / "adapter.safetensors").read_bytes()

    with pytest.raises(ArtifactContractError) as forged_info:
        training_state.abort_training_state_contributions(
            checkpoint,
            replace(owned_session, stage_name=other_session.stage_name),
        )
    _assert_code(forged_info, "training_state.contribution_session_mismatch")
    assert owned_session.stage_path.is_dir()
    assert other_session.stage_path.is_dir()

    training_state.abort_training_state_contributions(checkpoint, owned_session)

    assert not owned_session.stage_path.exists()
    assert other_session.stage_path.is_dir()
    assert (
        other_session.stage_path / training_state.TRAINING_STATE_CONTRIBUTION_PLAN
    ).read_bytes() == other_plan_before
    assert (checkpoint / "adapter.safetensors").read_bytes() == inference_before
    assert not (checkpoint / training_state.TRAINING_STATE_DIRECTORY).exists()


def test_abort_refuses_and_retains_a_post_manifest_forensic_session(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=1, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    session = training_state.begin_training_state_contributions(checkpoint, plan)
    training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[0]
    )

    with pytest.raises(training_state.TrainingStateContributionError) as commit_info:
        training_state.commit_training_state_contributions(
            checkpoint,
            session,
            on_manifest_written=lambda: (_ for _ in ()).throw(
                RuntimeError("injected post-manifest failure")
            ),
        )
    assert commit_info.value.terminal_forensic_only is True
    manifest_before = (
        session.stage_path / training_state.TRAINING_STATE_MANIFEST
    ).read_bytes()

    with pytest.raises(ArtifactContractError) as abort_info:
        training_state.abort_training_state_contributions(checkpoint, session)

    _assert_code(abort_info, "training_state.terminal_forensic_only")
    assert session.stage_path.is_dir()
    assert (
        session.stage_path / training_state.TRAINING_STATE_MANIFEST
    ).read_bytes() == manifest_before
    assert not (checkpoint / training_state.TRAINING_STATE_DIRECTORY).exists()


def test_contribution_session_is_bound_to_checkpoint_owner_and_never_reused(
    tmp_path: Path,
) -> None:
    first_checkpoint = _checkpoint(tmp_path, "first")
    second_checkpoint = _checkpoint(tmp_path, "second")
    publication = _publication(world_size=1, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    first_session = training_state.begin_training_state_contributions(
        first_checkpoint, plan
    )
    second_session = training_state.begin_training_state_contributions(
        first_checkpoint, plan
    )
    assert first_session.session_id != second_session.session_id
    assert first_session.stage_path != second_session.stage_path

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.publish_rank_training_state_contribution(
            second_checkpoint,
            first_session,
            publication.rank_payloads[0],
        )

    _assert_code(exc_info, "training_state.contribution_session_mismatch")
    assert not (second_checkpoint / "training_state").exists()
    assert first_session.stage_path.is_dir()


def test_rank_contribution_is_immutable_and_collision_preserves_first_bytes(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=1, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    session = training_state.begin_training_state_contributions(checkpoint, plan)
    first = training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[0]
    )
    before = {
        path.relative_to(first.path).as_posix(): path.read_bytes()
        for path in first.path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.publish_rank_training_state_contribution(
            checkpoint, session, publication.rank_payloads[0]
        )

    _assert_code(exc_info, "training_state.immutable_collision")
    after = {
        path.relative_to(first.path).as_posix(): path.read_bytes()
        for path in first.path.rglob("*")
        if path.is_file()
    }
    assert after == before


def _fresh_runtime_state() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    torch.manual_seed(99)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


class _LazyOptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.unused = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    def forward(self) -> torch.Tensor:
        return self.used.square().sum()


def _lazy_optimizer_runtime(
    *,
    explicit_empty_unused: bool,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = _LazyOptimizerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if explicit_empty_unused:
        optimizer.state[model.unused] = {}
    return model, optimizer


@pytest.mark.parametrize("explicit_empty_unused", [False, True])
def test_lazy_optimizer_missing_or_empty_state_roundtrips_exactly(
    explicit_empty_unused: bool,
) -> None:
    model, optimizer = _lazy_optimizer_runtime(
        explicit_empty_unused=explicit_empty_unused
    )

    envelope = training_state._optimizer_envelope(model, optimizer)
    encoded = training_state._torch_envelope_bytes(envelope)
    decoded = training_state._decode_optimizer(
        encoded,
        trainable=training_state._decode_trainable_model(
            training_state._torch_envelope_bytes(
                training_state._trainable_model_envelope(model)
            )
        ),
    )

    assert "used" in decoded["state"]
    assert ("unused" in decoded["state"]) is explicit_empty_unused
    if explicit_empty_unused:
        assert decoded["state"]["unused"] == {}

    fresh_model = _LazyOptimizerModel()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=0.01)
    load_state = training_state._optimizer_load_state_dict(
        decoded,
        model=fresh_model,
        optimizer=fresh_optimizer,
    )
    fresh_optimizer.load_state_dict(load_state)
    restored_names = {
        name
        for name, parameter in fresh_model.named_parameters()
        if parameter in fresh_optimizer.state
    }
    assert restored_names == ({"used", "unused"} if explicit_empty_unused else {"used"})


@pytest.mark.parametrize("mutation", ["extra_parameter", "non_mapping_slots"])
def test_lazy_optimizer_rejects_malformed_or_extra_state_slots(
    mutation: str,
) -> None:
    model, optimizer = _lazy_optimizer_runtime(explicit_empty_unused=False)
    envelope = training_state._optimizer_envelope(model, optimizer)
    if mutation == "extra_parameter":
        envelope["state"]["not_owned"] = {}
    else:
        envelope["state"]["unused"] = []
    encoded = training_state._torch_envelope_bytes(envelope)
    trainable = training_state._decode_trainable_model(
        training_state._torch_envelope_bytes(
            training_state._trainable_model_envelope(model)
        )
    )

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state._decode_optimizer(encoded, trainable=trainable)

    _assert_code(exc_info, "training_state.runtime_state")
    assert exc_info.value.context["field"].startswith("runtime.optimizer.state")


def test_runtime_expectations_support_fresh_optimizer_before_restore() -> None:
    model, optimizer, scheduler = _fresh_runtime_state()

    expectations = training_state.capture_runtime_state_expectations(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
    )

    assert expectations.structure["optimizer"]["class"].endswith(".Adam")
    assert expectations.structure["optimizer"]["slots"] == {}


def test_restore_decoded_state_owns_model_optimizer_scheduler_rng_and_cursor(
    tmp_path: Path,
) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    admitted = admit_training_state(
        checkpoint, _expectations(publication), current_rank=0
    )
    decoded = admitted.decoded_ranks[0]
    model, optimizer, scheduler = _fresh_runtime_state()
    simulated_cuda = [torch.full((32,), 199, dtype=torch.uint8)]
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_cpu = torch.get_rng_state()
    try:
        receipt = training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=1,
            current_cuda_device_topology=("cuda:0",),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            get_torch_cuda_rng_states=lambda: tuple(
                state.clone() for state in simulated_cuda
            ),
            set_torch_cuda_rng_states=lambda states: simulated_cuda.__setitem__(
                slice(None), [state.clone() for state in states]
            ),
        )

        assert receipt.runtime_signature == decoded.signature
        assert receipt.next_rank_local_micro_step == 30
        assert receipt.cursor["pack"]["state"] == {"ordinal": 10, "pending": []}
        assert set(optimizer.state) == set(model.parameters())
        assert scheduler.last_epoch == decoded.scheduler["state"]["last_epoch"]
        assert all(
            torch.equal(dict(model.named_parameters())[name], value)
            for name, value in decoded.trainable_model.items()
        )
        assert torch.equal(simulated_cuda[0], decoded.torch_cuda_rng_states[0])
        assert random.getstate() == decoded.python_rng_state
        restored_numpy = np.random.get_state()
        assert restored_numpy[0] == decoded.numpy_rng_state[0]
        assert np.array_equal(restored_numpy[1], decoded.numpy_rng_state[1])
        assert torch.equal(torch.get_rng_state(), decoded.torch_cpu_rng_state)
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_cpu)


def test_restore_failure_rolls_back_every_mutated_owner_and_rng(tmp_path: Path) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    decoded = admit_training_state(
        checkpoint, _expectations(publication), current_rank=0
    ).decoded_ranks[0]
    model, optimizer, scheduler = _fresh_runtime_state()
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer_before = optimizer.state_dict()
    scheduler_before = scheduler.state_dict()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = (torch.full((32,), 173, dtype=torch.uint8),)
    cuda_calls: list[tuple[torch.Tensor, ...]] = []

    def fail_cuda(states: Any) -> None:
        cuda_calls.append(tuple(state.clone() for state in states))
        if len(cuda_calls) == 1:
            raise RuntimeError("injected CUDA RNG restore failure")

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=1,
            current_cuda_device_topology=("cuda:0",),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            get_torch_cuda_rng_states=lambda: tuple(
                state.clone() for state in cuda_before
            ),
            set_torch_cuda_rng_states=fail_cuda,
        )

    _assert_code(exc_info, "training_state.restore_failed")
    assert exc_info.value.context["rollback_complete"] is True
    assert exc_info.value.context["original_error_type"] == "RuntimeError"
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before

    assert scheduler.state_dict() == scheduler_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert len(cuda_calls) == 2
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(cuda_calls[1], cuda_before, strict=True)
    )


def test_default_cuda_rollback_touches_only_current_rank_device_with_eight_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _publication(world_size=1, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)
    decoded = admit_training_state(
        checkpoint, _expectations(publication), current_rank=0
    ).decoded_rank
    model, optimizer, scheduler = _fresh_runtime_state()
    original_rank_rng = torch.full((32,), 211, dtype=torch.uint8)
    get_devices: list[int] = []
    set_calls: list[tuple[int, torch.Tensor]] = []

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 5)

    def get_rank_rng(device: int) -> torch.Tensor:
        get_devices.append(device)
        return original_rank_rng.clone()

    def set_rank_rng(state: torch.Tensor, device: int) -> None:
        set_calls.append((device, state.clone()))
        if len(set_calls) == 1:
            raise RuntimeError("injected current-device CUDA RNG restore failure")

    def reject_all_devices(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("all-visible-device CUDA RNG API must not be called")

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rank_rng)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rank_rng)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", reject_all_devices)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", reject_all_devices)

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=1,
            current_cuda_device_topology=("cuda:0",),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
        )

    _assert_code(exc_info, "training_state.restore_failed")
    assert exc_info.value.context["rollback_complete"] is True
    assert get_devices == [5]
    assert [device for device, _state in set_calls] == [5, 5]
    assert torch.equal(set_calls[0][1], decoded.torch_cuda_rng_states[0])
    assert torch.equal(set_calls[1][1], original_rank_rng)


def test_admission_retains_only_requested_rank_but_authenticates_every_rank(
    tmp_path: Path,
) -> None:
    publication = _publication(world_size=2, scaler=False)
    checkpoint, publication = _publish(tmp_path, publication=publication)

    admitted = admit_training_state(
        checkpoint,
        _expectations(publication),
        current_rank=1,
    )

    assert admitted.current_rank == 1
    assert set(admitted.decoded_ranks) == {1}
    assert admitted.decoded_rank is admitted.decoded_ranks[1]
    assert admitted.decoded_rank.rank == 1
    assert admitted.decoded_rank.world_size == 2
    assert admitted.decoded_rank.cursor["data"]["state"]["ordinal"] == 21
    assert all(
        path == training_state.TRAINING_STATE_RESOLVED_CONFIG
        or path == training_state.TRAINING_STATE_RESUME_COMPATIBILITY
        or path.startswith("rank-00001/")
        for path in admitted.files
    )

    rank_zero_optimizer = checkpoint / "training_state/rank-00000/optimizer.bin"
    rank_zero_optimizer.write_bytes(b"corrupt-noncurrent-rank")
    with pytest.raises(ArtifactContractError) as exc_info:
        admit_training_state(
            checkpoint,
            _expectations(publication),
            current_rank=1,
        )
    _assert_code(exc_info, "training_state.corrupt_component")


def test_wrong_rank_restore_rejects_distinct_rank_cursor_and_rng_without_mutation(
    tmp_path: Path,
) -> None:
    random.seed(101)
    rank_zero = _serialized_real_rank(0)
    random.seed(202)
    rank_one = _serialized_real_rank(1)
    resolved_config, identities = _resolved_config_and_identities()
    publication = TrainingStatePublication(
        parent_run_id="run-parent",
        parent_segment_id="segment-2",
        checkpoint_step=17,
        continuation_index=2,
        world_size=2,
        identities=identities,
        scheduler_applicable=True,
        scaler_applicable=False,
        rank_payloads=(rank_zero, rank_one),
        resolved_config=resolved_config,
        resume_compatibility=training_state.build_resume_compatibility_projection(
            resolved_config
        ),
    )
    checkpoint, publication = _publish(tmp_path, publication=publication)
    decoded = admit_training_state(
        checkpoint,
        _expectations(publication),
        current_rank=1,
    ).decoded_rank
    model, optimizer, scheduler = _fresh_runtime_state()
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    setter_calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=2,
            current_cuda_device_topology=("cuda:0",),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            set_python_rng_state=lambda _state: setter_calls.append("python"),
            set_numpy_rng_state=lambda _state: setter_calls.append("numpy"),
            set_torch_cpu_rng_state=lambda _state: setter_calls.append("cpu"),
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert exc_info.value.context["mismatches"] == [
        {"checkpoint": 1, "current": 0, "field": "rank"}
    ]
    assert decoded.cursor["data"]["state"]["ordinal"] == 21
    assert (
        decoded.python_rng_state
        != training_state._decode_rank_payload(rank_zero).python_rng_state
    )
    assert setter_calls == []
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in model_before.items()
    )
    assert optimizer.state_dict() == optimizer_before

    with pytest.raises(ArtifactContractError) as world_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=1,
            current_world_size=1,
            current_cuda_device_topology=("cuda:0",),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            set_python_rng_state=lambda _state: setter_calls.append("python"),
            set_numpy_rng_state=lambda _state: setter_calls.append("numpy"),
            set_torch_cpu_rng_state=lambda _state: setter_calls.append("cpu"),
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )
    _assert_code(world_info, "training_state.incompatible")
    assert [row["field"] for row in world_info.value.context["mismatches"]] == [
        "world_size"
    ]
    assert setter_calls == []


@pytest.mark.parametrize(
    "current_topology",
    [
        ("cuda:0",),
        ("cuda:0", "cuda:1", "cuda:2"),
        ("cuda:1", "cuda:0"),
        ("cuda:0", "cuda:2"),
    ],
)
def test_restore_rejects_missing_extra_reordered_or_wrong_cuda_devices_before_setters(
    tmp_path: Path,
    current_topology: tuple[str, ...],
) -> None:
    publication = replace(
        _publication(world_size=1, scaler=False),
        rank_payloads=(
            _serialized_real_rank(
                0,
                torch_cuda_rng_states=(
                    torch.arange(32, dtype=torch.uint8),
                    torch.arange(32, dtype=torch.uint8) + 1,
                ),
            ),
        ),
    )
    checkpoint, publication = _publish(tmp_path, publication=publication)
    decoded = admit_training_state(
        checkpoint,
        _expectations(publication),
        current_rank=0,
    ).decoded_rank
    model, optimizer, scheduler = _fresh_runtime_state()
    setter_calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=1,
            current_cuda_device_topology=current_topology,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            set_python_rng_state=lambda _state: setter_calls.append("python"),
            set_numpy_rng_state=lambda _state: setter_calls.append("numpy"),
            set_torch_cpu_rng_state=lambda _state: setter_calls.append("cpu"),
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert [row["field"] for row in exc_info.value.context["mismatches"]] == [
        "cuda_device_topology"
    ]
    assert setter_calls == []


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "wrong"])
def test_authenticated_cuda_device_state_inventory_cannot_target_current_topology(
    tmp_path: Path,
    mutation: str,
) -> None:
    publication = replace(
        _publication(world_size=1, scaler=False),
        rank_payloads=(
            _serialized_real_rank(
                0,
                torch_cuda_rng_states=(
                    torch.arange(32, dtype=torch.uint8),
                    torch.arange(32, dtype=torch.uint8) + 1,
                ),
            ),
        ),
    )
    checkpoint, publication = _publish(tmp_path, publication=publication)
    record = next(
        row
        for row in load_training_state_manifest(checkpoint).ranks[0].files
        if row.role == "rng:torch_cuda"
    )
    path = checkpoint / "training_state" / record.path
    envelope = torch.load(
        io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True
    )
    rows = envelope["device_states"]
    if mutation == "missing":
        rows.pop()
        envelope["device_count"] = 1
    elif mutation == "extra":
        rows.append(
            {
                "device": "cuda:2",
                "state": torch.arange(32, dtype=torch.uint8) + 2,
            }
        )
        envelope["device_count"] = 3
    elif mutation == "reordered":
        rows.reverse()
    else:
        rows[1]["device"] = "cuda:2"
    _rewrite_role_bytes(
        checkpoint,
        0,
        "rng:torch_cuda",
        training_state._torch_envelope_bytes(envelope),
    )
    decoded = admit_training_state(
        checkpoint,
        _expectations(publication),
        current_rank=0,
    ).decoded_rank
    model, optimizer, scheduler = _fresh_runtime_state()
    setter_calls: list[str] = []

    with pytest.raises(ArtifactContractError) as exc_info:
        training_state.restore_decoded_rank_training_state(
            decoded,
            current_rank=0,
            current_world_size=1,
            current_cuda_device_topology=("cuda:0", "cuda:1"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            set_python_rng_state=lambda _state: setter_calls.append("python"),
            set_numpy_rng_state=lambda _state: setter_calls.append("numpy"),
            set_torch_cpu_rng_state=lambda _state: setter_calls.append("cpu"),
            set_torch_cuda_rng_states=lambda _states: setter_calls.append("cuda"),
        )

    _assert_code(exc_info, "training_state.incompatible")
    assert setter_calls == []


def test_post_manifest_contribution_failure_is_terminal_forensic_only(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    publication = _publication(world_size=1, scaler=False)
    plan = training_state.TrainingStatePublicationPlan.from_publication(publication)
    session = training_state.begin_training_state_contributions(checkpoint, plan)
    training_state.publish_rank_training_state_contribution(
        checkpoint, session, publication.rank_payloads[0]
    )

    def fail_after_manifest() -> None:
        raise RuntimeError("injected post-manifest fault")

    with pytest.raises(training_state.TrainingStateContributionError) as exc_info:
        training_state.commit_training_state_contributions(
            checkpoint,
            session,
            on_manifest_written=fail_after_manifest,
        )

    assert exc_info.value.terminal_forensic_only is True
    assert exc_info.value.retryable is False
    assert session.stage_path.is_dir()
    assert (session.stage_path / training_state.TRAINING_STATE_MANIFEST).is_file()
    before_retry = {
        path.relative_to(session.stage_path).as_posix(): path.read_bytes()
        for path in session.stage_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(training_state.TrainingStateContributionError) as retry_info:
        training_state.commit_training_state_contributions(checkpoint, session)

    _assert_code(retry_info, "training_state.terminal_forensic_only")
    assert retry_info.value.terminal_forensic_only is True
    with pytest.raises(ArtifactContractError) as reuse_info:
        training_state.publish_rank_training_state_contribution(
            checkpoint,
            session,
            publication.rank_payloads[0],
        )
    _assert_code(reuse_info, "training_state.terminal_forensic_only")
    after_retry = {
        path.relative_to(session.stage_path).as_posix(): path.read_bytes()
        for path in session.stage_path.rglob("*")
        if path.is_file()
    }
    assert after_retry == before_retry
    assert not (checkpoint / "training_state").exists()
