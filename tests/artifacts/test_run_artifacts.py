from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

import src.artifacts.run_writer as run_writer_module
from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError


def test_initialize_writes_only_fixed_run_files_with_no_selector_state(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    assert writer.file_inventory() == (
        "logging.jsonl",
        "resolved_config.json",
        "run.json",
    )
    state = writer.read_run()
    assert state["resolved_config_path"] == "resolved_config.json"
    assert state["runtime"] == {"world_size": 2}
    assert not set(state).intersection({"final", "best", "selector", "selector_value"})
    assert json.loads(writer.resolved_config_path.read_text()) == {
        "training": {"seed": 7}
    }
    assert not any(
        (writer.run_dir / name).exists()
        for name in ("metrics", "receipts", "reports", "eval")
    )


def test_initialize_adds_fresh_append_only_segment_lineage_by_default(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    continuation = writer.read_run()["continuation"]
    assert continuation == {
        "schema_version": 1,
        "segment_id": continuation["segment_id"],
        "continuation_index": 0,
        "parent": None,
    }
    assert UUID(continuation["segment_id"]).version == 4


def test_initialize_exact_child_records_lineage_without_mutating_parent(
    tmp_path: Path,
) -> None:
    parent = RunWriter.initialize(
        run_dir=tmp_path / "parent",
        run_id="run-parent",
        run_name="parent",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="parent-created",
        config_fingerprint="parent-fp",
        resolved_config={"training": {"seed": 7}},
        world_size=2,
        segment_id="segment-parent",
    )
    parent_files_before = {
        path.relative_to(parent.run_dir).as_posix(): path.read_bytes()
        for path in parent.run_dir.rglob("*")
        if path.is_file()
    }
    checkpoint_identity = {
        "resolved_path": str((parent.checkpoints_dir / "step-5").resolve()),
        "checkpoint_step": 5,
        "training_state_manifest_file_sha256": "a" * 64,
        "training_state_aggregate_digest": "b" * 64,
    }

    child = RunWriter.initialize(
        run_dir=tmp_path / "child",
        run_id="run-child",
        run_name="child",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="child-created",
        config_fingerprint="child-fp",
        resolved_config={"training": {"seed": 7}},
        world_size=2,
        segment_id="segment-child",
        continuation_lineage={
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": checkpoint_identity,
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
    )

    assert child.read_run()["continuation"] == {
        "schema_version": 1,
        "segment_id": "segment-child",
        "continuation_index": 1,
        "parent": {
            "run_id": "run-parent",
            "segment_id": "segment-parent",
            "checkpoint_identity": checkpoint_identity,
            "continuation_index": 0,
        },
    }
    assert {
        path.relative_to(parent.run_dir).as_posix(): path.read_bytes()
        for path in parent.run_dir.rglob("*")
        if path.is_file()
    } == parent_files_before


def test_continuation_lineage_is_bound_only_after_parent_admission(
    tmp_path: Path,
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "child",
        run_id="run-child",
        run_name="child",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="child-created",
        config_fingerprint="child-fp",
        resolved_config={"training": {"seed": 7}},
        world_size=2,
        segment_id="segment-child",
    )
    lineage = {
        "parent_run_id": "run-parent",
        "parent_segment_id": "segment-parent",
        "parent_checkpoint_identity": {
            "resolved_path": str((tmp_path / "parent" / "step-5").resolve()),
            "checkpoint_step": 5,
            "training_state_manifest_file_sha256": "a" * 64,
            "training_state_aggregate_digest": "b" * 64,
        },
        "parent_continuation_index": 0,
        "continuation_index": 1,
    }

    assert writer.read_run()["continuation"]["parent"] is None
    writer.bind_continuation_lineage(lineage)

    assert writer.read_run()["continuation"] == {
        "schema_version": 1,
        "segment_id": "segment-child",
        "continuation_index": 1,
        "parent": {
            "run_id": "run-parent",
            "segment_id": "segment-parent",
            "checkpoint_identity": lineage["parent_checkpoint_identity"],
            "continuation_index": 0,
        },
    }
    with pytest.raises(ArtifactContractError, match="already bound"):
        writer.bind_continuation_lineage(lineage)


@pytest.mark.parametrize(
    "continuation_lineage",
    [
        {},
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "relative/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 0,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "B" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "not-a-sha",
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
                "unexpected": "field",
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-child",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-child",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 1,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 0,
            "continuation_index": 0,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 2,
            "continuation_index": 2,
        },
        {
            "parent_run_id": "run-parent",
            "parent_segment_id": "segment-parent",
            "parent_checkpoint_identity": {
                "resolved_path": "/tmp/checkpoint",
                "checkpoint_step": 1,
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            "parent_continuation_index": 2,
            "continuation_index": 4,
        },
    ],
)
def test_initialize_rejects_invalid_or_nonmonotonic_continuation_lineage(
    tmp_path: Path,
    continuation_lineage: dict[str, object],
) -> None:
    run_dir = tmp_path / "child"

    with pytest.raises(ArtifactContractError) as exc_info:
        RunWriter.initialize(
            run_dir=run_dir,
            run_id="run-child",
            run_name="child",
            artifact_root=tmp_path,
            collision_outcome="created",
            created_at="now",
            config_fingerprint="fp",
            resolved_config={},
            world_size=1,
            segment_id="segment-child",
            continuation_lineage=continuation_lineage,
        )

    assert exc_info.value.code == "run_writer.invalid_continuation_lineage"
    assert not run_dir.exists()


@pytest.mark.parametrize("segment_id", ["", "   ", 0, True])
def test_initialize_rejects_invalid_segment_identity_before_writing(
    tmp_path: Path, segment_id: object
) -> None:
    run_dir = tmp_path / "run"

    with pytest.raises(ArtifactContractError) as exc_info:
        RunWriter.initialize(
            run_dir=run_dir,
            run_id="run",
            run_name="run",
            artifact_root=tmp_path,
            collision_outcome="created",
            created_at="now",
            config_fingerprint="fp",
            resolved_config={},
            world_size=1,
            segment_id=segment_id,  # type: ignore[arg-type]
        )

    assert exc_info.value.code == "run_writer.invalid_segment_id"
    assert not run_dir.exists()


def test_atomic_json_and_run_tree_publication_fsync_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsynced_directories: list[Path] = []
    real_fsync_directory = run_writer_module._fsync_directory

    def record_directory(path: Path) -> None:
        fsynced_directories.append(path.resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(run_writer_module, "_fsync_directory", record_directory)

    writer = _writer(tmp_path)
    writer.record_warning("durability-check")

    assert writer.run_dir.parent.resolve() in fsynced_directories
    assert writer.run_dir.resolve() in fsynced_directories
    assert fsynced_directories[-1] == writer.run_dir.resolve()


def test_logging_appends_one_self_contained_train_and_eval_row(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append_logging_row(
        {
            "step": 1,
            "split": "train",
            "loss/total": 1.25,
            "acc_top1": 0.5,
            "acc_top5": 0.75,
            "learning_rates": {"adapter": 1e-5},
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        }
    )
    writer.append_logging_row(
        {
            "step": 1,
            "split": "eval",
            "example_count": 3,
            "pack_count": 2,
            "acc_top1": 0.6,
        }
    )

    rows = [json.loads(line) for line in writer.logging_path.read_text().splitlines()]
    assert [row["split"] for row in rows] == ["train", "eval"]
    assert rows[0]["learning_rates"] == {"adapter": 1e-5}
    assert rows[1]["example_count"] == 3
    assert all("non_finite_fields" in row for row in rows)


def test_logging_normalizes_nested_nonfinite_values_to_null(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append_logging_row(
        {
            "step": 2,
            "split": "train",
            "loss": float("nan"),
            "metrics": {"x": float("inf")},
        }
    )
    row = json.loads(writer.logging_path.read_text())
    assert row["loss"] is None and row["metrics"]["x"] is None
    assert row["non_finite_fields"] == ["loss", "metrics.x"]
    assert (
        "NaN" not in writer.logging_path.read_text()
        and "Infinity" not in writer.logging_path.read_text()
    )


def test_logging_rejects_non_json_values_without_appending_partial_row(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.append_logging_row({"step": 1, "split": "train", "bad": {1, 2}})
    assert exc_info.value.code == "run_writer.not_json_serializable"
    assert writer.logging_path.read_bytes() == b""


def test_warning_counts_are_grouped_and_bounded_without_contexts(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.record_warning("bad_example", count=2)
    writer.record_warning("bad_example", count=3)
    for index in range(40):
        writer.record_warning(f"code_{index}")
    counts = writer.read_run()["warning_counts"]
    assert counts["bad_example"] == 5
    assert len(counts) <= 32
    assert counts["other"] == 10
    assert all(isinstance(value, int) for value in counts.values())


def test_materialization_binding_is_one_time_and_survives_cache_deletion(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    writer.bind_materialization(
        "train",
        cache_format_version=4,
        semantic_fingerprint="fp",
        determinant_digest="digest",
    )
    cache.rmdir()
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_materialization(
            "train",
            cache_format_version=4,
            semantic_fingerprint="changed",
            determinant_digest="changed",
        )
    assert exc_info.value.code == "run_writer.materialization_already_bound"
    assert writer.read_run()["materializations"]["train"] == {
        "cache_format_version": 4,
        "semantic_fingerprint": "fp",
        "determinant_digest": "digest",
    }


def test_schedule_can_be_bound_once_after_early_initialization(tmp_path: Path) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="experiment",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
    )
    assert writer.read_run()["resolved_max_steps"] is None
    writer.bind_schedule(resolved_max_steps=5)
    assert writer.read_run()["resolved_max_steps"] == 5
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_schedule(resolved_max_steps=6)
    assert exc_info.value.code == "run_writer.schedule_already_bound"


@pytest.mark.parametrize("mode", ["legacy_fused", "overlapped", "synchronous"])
def test_forward_input_provider_mode_can_be_bound_once_per_mode(
    tmp_path: Path, mode: str
) -> None:
    writer = _writer(tmp_path)
    assert writer.read_run()["forward_input_provider_mode"] is None
    assert writer.read_run()["forward_input_provider_resolution"] is None
    resolution = {
        "configured_mode": mode,
        "resolved_mode": mode,
        "source": "strict_config",
    }
    writer.bind_forward_input_provider_mode(mode, resolution=resolution)
    state = writer.read_run()
    assert state["forward_input_provider_mode"] == mode
    assert state["forward_input_provider_resolution"] == resolution
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_forward_input_provider_mode(mode)
    assert exc_info.value.code == "run_writer.forward_input_provider_mode_already_bound"


def test_forward_input_provider_mode_rejects_unknown_values(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_forward_input_provider_mode("bogus")
    assert exc_info.value.code == "run_writer.invalid_forward_input_provider_mode"


def test_forward_input_provider_resolution_must_match_bound_mode(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_forward_input_provider_mode(
            "synchronous",
            resolution={
                "configured_mode": "synchronous",
                "resolved_mode": "overlapped",
                "source": "deprecated_environment_override",
            },
        )
    assert exc_info.value.code == (
        "run_writer.forward_input_provider_resolution_mismatch"
    )
    state = writer.read_run()
    assert state["forward_input_provider_mode"] is None
    assert state["forward_input_provider_resolution"] is None


def test_initialize_adds_provenance_measurement_and_policy_fields_without_new_files(
    tmp_path: Path,
) -> None:
    provenance = {
        "schema_version": 1,
        "repository": {"status": "available", "commit": "abc", "dirty": False},
        "dependencies": {},
    }
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="experiment",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        provenance=provenance,
        measurement_context={
            "comparison_arm": "compatibility_reference",
            "wall_clock_scope": "training_entry_to_terminal_artifact",
            "warmup_exclusion_steps": 2,
        },
    )

    state = writer.read_run()
    assert writer.file_inventory() == (
        "logging.jsonl",
        "resolved_config.json",
        "run.json",
    )
    assert state["provenance"] == provenance
    assert state["policy_identities"] == {}
    assert state["measurement"]["schema_version"] == 1
    assert (
        state["measurement"]["context"]["comparison_arm"] == "compatibility_reference"
    )
    assert state["measurement"]["active_phase"] is None
    assert state["measurement"]["terminal_phase"] is None
    assert state["measurement"]["terminal_phase_status"] is None
    assert state["measurement"]["last_completed_phase"] is None
    assert state["measurement"]["failure_phase"] is None
    assert state["measurement"]["steady_state_eligible"] is False


def test_policy_identity_binding_is_additive_immutable_and_named(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    original = writer.read_run()

    writer.bind_policy_identity(
        "packing",
        {"name": "source_order_next_fit", "schema_version": 1},
    )
    writer.bind_policy_identity(
        "resume",
        {"mode": "disabled", "schema_version": 1},
    )
    writer.bind_policy_identity(
        "eval_reduction",
        {
            "mode": "disjoint_shard",
            "schema_version": 1,
            "source": "default",
        },
    )
    writer.bind_policy_identity(
        "upstream_runtime_baseline",
        {
            "schema_version": 1,
            "baseline_sha256": "a" * 64,
            "attention_backend": "flash_attention_2",
            "admitted": True,
            "mismatches": [],
            "reference_only": {},
        },
    )

    state = writer.read_run()
    assert state["policy_identities"] == {
        "packing": {"name": "source_order_next_fit", "schema_version": 1},
        "resume": {"mode": "disabled", "schema_version": 1},
        "eval_reduction": {
            "mode": "disjoint_shard",
            "schema_version": 1,
            "source": "default",
        },
        "upstream_runtime_baseline": {
            "schema_version": 1,
            "baseline_sha256": "a" * 64,
            "attention_backend": "flash_attention_2",
            "admitted": True,
            "mismatches": [],
            "reference_only": {},
        },
    }
    for stable_field in (
        "run_id",
        "config_fingerprint",
        "runtime",
        "resolved_max_steps",
        "warning_counts",
        "materializations",
    ):
        assert state[stable_field] == original[stable_field]
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_policy_identity(
            "packing",
            {"name": "window_binpack", "schema_version": 1},
        )
    assert exc_info.value.code == "run_writer.policy_identity_already_bound"
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.bind_policy_identity("unknown", {"schema_version": 1})
    assert exc_info.value.code == "run_writer.invalid_policy_identity_name"


def test_phase_receipts_track_status_duration_and_resource_high_water(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    initial = {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 100,
            "io_read_bytes": 10,
            "io_write_bytes": 20,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 0,
            "max_memory_allocated_bytes": 30,
            "max_memory_reserved_bytes": 40,
        },
    }
    final = {
        "schema_version": 1,
        "cpu": {
            "scope": "current_process",
            "max_rss_bytes": 150,
            "io_read_bytes": 15,
            "io_write_bytes": 25,
        },
        "gpu": {
            "scope": "current_process_current_device",
            "initialized": True,
            "device_index": 0,
            "max_memory_allocated_bytes": 35,
            "max_memory_reserved_bytes": 50,
        },
    }

    writer.begin_phase("model_loading", started_at="t0", resources=initial)
    running = writer.read_run()["measurement"]
    assert running["active_phase"] == "model_loading"
    assert running["phases"]["model_loading"]["status"] == "running"

    writer.finish_phase(
        "model_loading",
        status="completed",
        completed_at="t1",
        duration_seconds=1.25,
        resources=final,
    )
    measurement = writer.read_run()["measurement"]
    assert measurement["active_phase"] is None
    assert measurement["terminal_phase"] == "model_loading"
    assert measurement["terminal_phase_status"] == "completed"
    assert measurement["last_completed_phase"] == "model_loading"
    assert measurement["failure_phase"] is None
    assert measurement["phases"]["model_loading"] == {
        "status": "completed",
        "started_at": "t0",
        "completed_at": "t1",
        "duration_seconds": 1.25,
    }
    high_water = measurement["resource_high_water"]
    assert high_water["cpu"]["max_rss_bytes"] == 150
    assert high_water["cpu"]["io_read_bytes"] == 15
    assert high_water["gpu"]["max_memory_reserved_bytes"] == 50


def test_failed_phase_is_terminal_and_cannot_claim_steady_state(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase("cache_admission", started_at="t0")
    writer.finish_phase(
        "cache_admission",
        status="failed",
        completed_at="t1",
        duration_seconds=0.5,
    )
    writer.finalize(
        status="failed",
        updated_at="t1",
        completed_steps=0,
        consumed_packs=0,
        checkpoint_event_count=0,
        optimizer_update_status=None,
        finite_status=None,
        terminal_error="cache rejected",
    )

    state = writer.read_run()
    assert state["measurement"]["terminal_phase"] == "cache_admission"
    assert state["measurement"]["terminal_phase_status"] == "failed"
    assert state["measurement"]["last_completed_phase"] is None
    assert state["measurement"]["failure_phase"] == "cache_admission"
    assert state["measurement"]["phases"]["cache_admission"]["status"] == "failed"
    assert state["measurement"]["steady_state_eligible"] is False
    assert state["status"] == "failed"


def test_failed_finalize_after_completed_phase_records_unphased_failure(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase("model_loading", started_at="t0")
    writer.finish_phase(
        "model_loading",
        status="completed",
        completed_at="t1",
        duration_seconds=1.0,
    )

    writer.finalize(
        status="failed",
        updated_at="t2",
        completed_steps=0,
        consumed_packs=0,
        checkpoint_event_count=0,
        optimizer_update_status=None,
        finite_status=None,
        terminal_error="failure between named phases",
    )

    measurement = writer.read_run()["measurement"]
    assert measurement["active_phase"] is None
    assert measurement["last_completed_phase"] == "model_loading"
    assert measurement["terminal_phase"] == "unphased_failure"
    assert measurement["terminal_phase_status"] == "failed"
    assert measurement["failure_phase"] == "unphased_failure"
    assert measurement["phase_order"] == ["model_loading", "unphased_failure"]
    assert measurement["phases"]["unphased_failure"] == {
        "status": "failed",
        "reason": "run_failed_outside_named_phase",
        "started_at": None,
        "completed_at": "t2",
        "duration_seconds": 0.0,
    }
    assert measurement["phases"]["model_loading"]["status"] == "completed"
    assert measurement["steady_state_eligible"] is False


def test_fine_grained_startup_phases_keep_order_and_rank_resource_receipts(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    rank_resources = {
        "schema_version": 1,
        "scope": "current_process_lifetime_high_water_at_phase_observation",
        "world_size": 2,
        "per_rank": {
            "0": {
                "scope": "current_process",
                "max_rss_bytes": 100,
                "io_read_bytes": 10,
                "io_write_bytes": 20,
            },
            "1": {
                "scope": "current_process",
                "max_rss_bytes": 150,
                "io_read_bytes": 30,
                "io_write_bytes": 5,
            },
        },
        "global_maxima": {
            "scope": "all_rank_deterministic_maximum",
            "max_rss_bytes": 150,
            "io_read_bytes": 30,
            "io_write_bytes": 20,
        },
    }
    phases = (
        "config_provenance_resolution",
        "cache_identity_resolution",
        "cache_publication_admission",
        "train_rank_hydration",
    )
    for phase in phases:
        writer.record_completed_phase(
            phase,
            completed_at=f"done-{phase}",
            duration_seconds=0.25,
            rank_resources=rank_resources,
            rank_details={
                "0": {"status": "completed"},
                "1": {"status": "completed"},
            },
        )

    measurement = writer.read_run()["measurement"]
    assert tuple(measurement["phase_order"]) == phases
    assert measurement["terminal_phase"] == "train_rank_hydration"
    for phase in phases:
        receipt = measurement["phases"][phase]
        assert receipt["status"] == "completed"
        assert receipt["rank_resources"]["per_rank"]["1"]["max_rss_bytes"] == 150
        assert receipt["rank_resources"]["global_maxima"]["io_write_bytes"] == 20


def test_steady_state_requires_complete_expected_measured_step_count(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase("steady_state", started_at="t0")
    with pytest.raises(ArtifactContractError) as exc_info:
        writer.finish_phase(
            "steady_state",
            status="completed",
            completed_at="t1",
            duration_seconds=1.0,
            accepted_measured_steps=0,
            expected_measured_steps=None,
        )
    assert exc_info.value.code == "run_writer.invalid_measured_step_count"

    writer.finish_phase(
        "steady_state",
        status="completed",
        completed_at="t1",
        duration_seconds=1.0,
        accepted_measured_steps=3,
        expected_measured_steps=4,
    )
    measurement = writer.read_run()["measurement"]
    assert measurement["steady_state_eligible"] is False
    assert measurement["accepted_measured_steps"] == 3
    assert measurement["expected_measured_steps"] == 4


def test_phase_summary_validates_steady_and_evaluation_aggregate_contract(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.record_phase_summary(
        "steady_state",
        status="completed",
        completed_at="t1",
        duration_seconds=12.0,
        duration_scope="sum_of_accepted_all_rank_max_step_durations",
        accepted_measured_steps=3,
        expected_measured_steps=3,
    )
    writer.record_phase_summary(
        "evaluation_execution",
        status="completed",
        completed_at="t2",
        duration_seconds=2.5,
        duration_scope="sum_of_all_rank_max_evaluation_event_durations",
        event_count=2,
        resource_observation_scope=(
            "process_lifetime_high_water_observed_after_evaluation"
        ),
        resource_high_water_observed_after_events={
            "resource/cpu_max_rss_bytes": 2048.0
        },
    )

    measurement = writer.read_run()["measurement"]
    assert measurement["steady_state_eligible"] is True
    assert measurement["accepted_measured_steps"] == 3
    assert measurement["expected_measured_steps"] == 3
    assert measurement["phases"]["steady_state"]["duration_seconds"] == 12.0
    assert measurement["phases"]["steady_state"]["acceptance_scope"] == (
        "applied_finite_post_warmup_steps_with_finite_all_rank_max_step_duration"
    )
    evaluation = measurement["phases"]["evaluation_execution"]
    assert evaluation["event_count"] == 2
    assert evaluation["resource_observation_scope"] == (
        "process_lifetime_high_water_observed_after_evaluation"
    )
    assert evaluation["resource_high_water_observed_after_events"] == {
        "resource/cpu_max_rss_bytes": 2048.0
    }
    assert measurement["terminal_phase"] == "evaluation_execution"
    assert measurement["terminal_phase_status"] == "completed"
    assert measurement["last_completed_phase"] == "evaluation_execution"
    assert measurement["failure_phase"] is None

    writer.finalize(
        status="failed",
        updated_at="t3",
        completed_steps=3,
        consumed_packs=3,
        checkpoint_event_count=0,
        optimizer_update_status="applied",
        finite_status="finite",
        terminal_error="later terminal failure",
    )
    assert writer.read_run()["measurement"]["steady_state_eligible"] is False


def test_failed_optimizer_runtime_assembly_does_not_relabel_completed_model_load(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.begin_phase("model_loading", started_at="model-start")
    writer.finish_phase(
        "model_loading",
        status="completed",
        completed_at="model-complete",
        duration_seconds=1.0,
    )
    writer.begin_phase("optimizer_runtime_assembly", started_at="runtime-start")
    writer.finish_phase(
        "optimizer_runtime_assembly",
        status="failed",
        completed_at="runtime-failed",
        duration_seconds=0.25,
    )

    measurement = writer.read_run()["measurement"]
    assert measurement["phases"]["model_loading"]["status"] == "completed"
    assert measurement["last_completed_phase"] == "model_loading"
    assert measurement["terminal_phase"] == "optimizer_runtime_assembly"
    assert measurement["terminal_phase_status"] == "failed"
    assert measurement["failure_phase"] == "optimizer_runtime_assembly"
    assert measurement["phases"]["optimizer_runtime_assembly"]["status"] == "failed"


def test_finalize_persists_entry_to_durable_terminal_state_monotonic_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = RunWriter.initialize(
        run_dir=tmp_path / "run",
        run_id="run",
        run_name="run",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="created",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
        entry_started_at="entry-utc",
    )
    monotonic_values = iter((112.5,))
    monkeypatch.setattr(
        run_writer_module.time, "monotonic", lambda: next(monotonic_values)
    )
    monkeypatch.setattr(run_writer_module, "_utc_now", lambda: "terminal-utc")

    writer.finalize(
        status="completed",
        updated_at="terminal-state-utc",
        completed_steps=1,
        consumed_packs=1,
        checkpoint_event_count=0,
        optimizer_update_status="applied",
        finite_status="finite",
        entry_started_monotonic=100.0,
    )

    receipt = writer.read_run()["measurement"]["entry_to_terminal"]
    assert receipt == {
        "status": "completed",
        "started_at": "entry-utc",
        "completed_at": "terminal-utc",
        "duration_seconds": 12.5,
        "clock": "monotonic",
        "boundary": (
            "training_entry_to_terminal_state_durable_before_measurement_annotation"
        ),
    }


def test_initialize_failure_leaves_no_partial_run_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_touch = Path.touch

    def fail_logging_touch(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "logging.jsonl":
            raise OSError("injected logging bootstrap failure")
        real_touch(path, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_logging_touch)
    run_dir = tmp_path / "run"
    with pytest.raises(OSError, match="injected logging"):
        RunWriter.initialize(
            run_dir=run_dir,
            run_id="run",
            run_name="experiment",
            artifact_root=tmp_path,
            collision_outcome="created",
            created_at="now",
            config_fingerprint="fp",
            resolved_config={},
            world_size=1,
        )
    assert not run_dir.exists()
    assert not list(tmp_path.glob(".run.*.init"))


def test_initialize_can_atomically_replace_precreated_empty_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    writer = RunWriter.initialize(
        run_dir=run_dir,
        run_id="run",
        run_name="experiment",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="now",
        config_fingerprint="fp",
        resolved_config={},
        world_size=1,
    )
    assert writer.file_inventory() == (
        "logging.jsonl",
        "resolved_config.json",
        "run.json",
    )


def test_final_and_best_aliases_are_canonical_selector_owners(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_final(step=5)
    assert writer.write_best(
        step=4,
        selector="acc_top1",
        value=0.8,
        optimizer_update_status="applied",
        finite_status="finite",
        checkpoint_committed=True,
    )
    assert json.loads((writer.checkpoints_dir / "final.json").read_text()) == {
        "checkpoint_path": "checkpoints/step-5",
        "step": 5,
    }
    assert json.loads((writer.checkpoints_dir / "best.json").read_text()) == {
        "checkpoint_path": "checkpoints/step-4",
        "selector": "acc_top1",
        "step": 4,
        "value": 0.8,
    }
    state = writer.read_run()
    assert not set(state).intersection({"final", "best", "selector", "selector_value"})


@pytest.mark.parametrize(
    ("update", "finite", "committed", "value"),
    [
        ("skipped", "finite", True, 0.9),
        ("applied", "unsafe", True, 0.9),
        ("applied", "finite", False, 0.9),
        ("applied", "finite", True, float("nan")),
        ("applied", "finite", True, float("inf")),
    ],
)
def test_ineligible_step_cannot_advance_best(
    tmp_path: Path, update: str, finite: str, committed: bool, value: float
) -> None:
    writer = _writer(tmp_path)
    assert writer.write_best(
        step=1,
        selector="acc_top1",
        value=0.5,
        optimizer_update_status="applied",
        finite_status="finite",
        checkpoint_committed=True,
    )
    assert not writer.write_best(
        step=2,
        selector="acc_top1",
        value=value,
        optimizer_update_status=update,
        finite_status=finite,
        checkpoint_committed=committed,
    )
    assert json.loads((writer.checkpoints_dir / "best.json").read_text())["step"] == 1


def test_finalize_is_atomic_compact_and_bounds_terminal_error(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.finalize(
        status="failed",
        updated_at="2026-07-11T00:01:00Z",
        completed_steps=2,
        consumed_packs=8,
        checkpoint_event_count=1,
        optimizer_update_status="skipped",
        finite_status="unsafe",
        terminal_error="x" * 5000,
    )
    state = writer.read_run()
    assert state["status"] == "failed" and state["completed_steps"] == 2
    assert len(state["terminal_error"]) == 1024
    assert not list(writer.run_dir.glob(".run.json.*"))


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_finalize_rejects_overwriting_committed_checkpoint_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    writer = _writer(tmp_path)
    _record_committed_checkpoint_event(writer, monkeypatch=monkeypatch)
    state_before = writer.run_path.read_bytes()

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.finalize(
            status=terminal_status,
            updated_at="2026-08-11T00:00:02+00:00",
            completed_steps=3,
            consumed_packs=7,
            checkpoint_event_count=0,
            optimizer_update_status="skipped",
            finite_status="non_finite",
            terminal_error="abrupt failure" if terminal_status == "failed" else None,
        )

    assert exc_info.value.code == "run_writer.finalize_checkpoint_progress_mismatch"
    assert writer.run_path.read_bytes() == state_before


def test_failed_finalize_preserves_matching_committed_checkpoint_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _writer(tmp_path)
    _record_committed_checkpoint_event(writer, monkeypatch=monkeypatch)
    writer.record_checkpoint_publication_event(
        step=4,
        status="failed",
        started_at="2026-08-11T00:00:01+00:00",
        completed_at="2026-08-11T00:00:02+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=False,
        checkpoint_identity=None,
        inference_payload_identity=None,
        committed_progress=None,
        failure_code="checkpoint.abrupt_failure",
    )

    writer.finalize(
        status="failed",
        updated_at="2026-08-11T00:00:02+00:00",
        completed_steps=3,
        consumed_packs=9,
        checkpoint_event_count=1,
        optimizer_update_status="applied",
        finite_status="finite",
        terminal_error="failure after committed checkpoint",
    )

    state = writer.read_run()
    assert state["status"] == "failed"
    assert state["completed_steps"] == 3
    assert state["consumed_packs"] == 9
    assert state["checkpoint_event_count"] == 1
    assert state["final_optimizer_update_status"] == "applied"
    assert state["final_finite_status"] == "finite"
    assert state["terminal_error"] == "failure after committed checkpoint"
    assert [
        event["status"]
        for event in state["measurement"]["checkpoint_publication_events"]
    ] == ["completed", "failed"]


def test_inference_only_publication_event_cannot_carry_exact_state_identity(
    tmp_path: Path,
) -> None:
    """`coordexp-swift-training-resume` -> Scenario: Exact training state is disabled."""

    writer = _writer(tmp_path)
    checkpoint_dir = tmp_path / "run-a" / "checkpoints" / "step-3"

    with pytest.raises(ArtifactContractError) as exc_info:
        writer.record_checkpoint_publication_event(
            step=3,
            status="completed",
            started_at="2026-08-11T00:00:00+00:00",
            completed_at="2026-08-11T00:00:01+00:00",
            duration_seconds=1.0,
            is_final=False,
            exact_training_state_enabled=False,
            checkpoint_identity={
                "checkpoint_step": 3,
                "resolved_path": str(checkpoint_dir),
                "training_state_manifest_file_sha256": "a" * 64,
                "training_state_aggregate_digest": "b" * 64,
            },
            inference_payload_identity={
                "schema": "coordexp-swift-inference-checkpoint-payload-publication",
                "schema_version": 2,
                "manifest_relative_path": "inference_payload_manifest.json",
                "manifest_file_sha256": "c" * 64,
                "aggregate_digest": "d" * 64,
            },
            committed_progress={
                "schema": "coordexp-swift-checkpoint-committed-progress",
                "schema_version": 1,
                "completed_steps": 3,
                "consumed_packs": 9,
                "optimizer_update_status": "applied",
                "finite_status": "finite",
            },
            failure_code=None,
        )

    assert exc_info.value.code == "run_writer.invalid_checkpoint_publication_event"
    state = writer.read_run()
    assert state["measurement"].get("checkpoint_publication_events", []) == []
    assert not checkpoint_dir.exists()


def _record_committed_checkpoint_event(
    writer: RunWriter,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_identity = {"schema": "test-inference-payload", "schema_version": 1}
    monkeypatch.setattr(
        run_writer_module,
        "admit_inference_checkpoint_payload_identity",
        lambda _checkpoint_dir, identity: dict(identity),
    )
    writer.record_checkpoint_publication_event(
        step=3,
        status="completed",
        started_at="2026-08-11T00:00:00+00:00",
        completed_at="2026-08-11T00:00:01+00:00",
        duration_seconds=1.0,
        is_final=False,
        exact_training_state_enabled=False,
        checkpoint_identity=None,
        inference_payload_identity=payload_identity,
        committed_progress={
            "schema": "coordexp-swift-checkpoint-committed-progress",
            "schema_version": 1,
            "completed_steps": 3,
            "consumed_packs": 9,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        },
        failure_code=None,
    )


def _writer(tmp_path: Path) -> RunWriter:
    return RunWriter.initialize(
        run_dir=tmp_path / "run-a",
        run_id="run-a",
        run_name="experiment",
        artifact_root=tmp_path,
        collision_outcome="created",
        created_at="2026-07-11T00:00:00Z",
        config_fingerprint="config-fp",
        resolved_config={"training": {"seed": 7}},
        world_size=2,
        resolved_max_steps=5,
    )


# ---------------------------------------------------------------------------
# Wave-0 pre-move characterization for
# `decompose-coordexp-swift-training-orchestration`.
#
# `tests/training/test_orchestration_compatibility.py` derives the frozen byte
# tree below from one representative RunWriter lifetime.  These additions bind
# the artifact suite to that same tree so Wave 4 cannot split RunWriter internals
# without an exact-byte comparison, and they re-derive the run-file inventory
# live so a schema drift fails here as well as there.
# ---------------------------------------------------------------------------


WAVE0_RUN_WRITER_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "training_orchestration"
    / "run_writer"
)

WAVE0_RUN_FILE_INVENTORY = (
    "checkpoints/best.json",
    "checkpoints/final.json",
    "logging.jsonl",
    "resolved_config.json",
    "run.json",
)

WAVE0_RUN_STATE_KEYS = (
    "artifact_root",
    "checkpoint_event_count",
    "collision_outcome",
    "completed_at",
    "completed_steps",
    "config_fingerprint",
    "consumed_packs",
    "continuation",
    "created_at",
    "final_finite_status",
    "final_optimizer_update_status",
    "forward_input_provider_mode",
    "forward_input_provider_resolution",
    "materializations",
    "measurement",
    "policy_identities",
    "provenance",
    "resolved_config_path",
    "resolved_max_steps",
    "run_dir",
    "run_id",
    "run_name",
    "runtime",
    "status",
    "terminal_error",
    "updated_at",
    "warning_counts",
)

WAVE0_MEASUREMENT_KEYS = (
    "accepted_measured_steps",
    "active_phase",
    "checkpoint_publication_events",
    "context",
    "entry_to_terminal",
    "expected_measured_steps",
    "failure_phase",
    "last_completed_phase",
    "phase_order",
    "phases",
    "resource_high_water",
    "schema_version",
    "steady_state_eligible",
    "terminal_phase",
    "terminal_phase_status",
)

WAVE0_RUN_PHASE_ORDER = (
    "config_provenance_resolution",
    "cache_identity_resolution",
    "cache_preparation",
    "cache_publication",
    "cache_admission",
    "first_optimizer_step",
    "steady_state",
    "evaluation_execution",
)


def _wave0_fixture_bytes() -> dict[str, bytes]:
    root = WAVE0_RUN_WRITER_FIXTURE_ROOT
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_wave0_run_writer_fixture_tree_has_the_frozen_file_inventory() -> None:
    assert tuple(sorted(_wave0_fixture_bytes())) == WAVE0_RUN_FILE_INVENTORY


def test_wave0_run_writer_fixture_run_state_schema_is_frozen() -> None:
    state = json.loads(
        (WAVE0_RUN_WRITER_FIXTURE_ROOT / "run.json").read_text(encoding="utf-8")
    )

    assert tuple(sorted(state)) == WAVE0_RUN_STATE_KEYS
    assert tuple(sorted(state["measurement"])) == WAVE0_MEASUREMENT_KEYS
    assert tuple(state["measurement"]["phase_order"]) == WAVE0_RUN_PHASE_ORDER
    assert state["status"] == "completed"
    assert state["warning_counts"] == {"characterized_warning": 2}


def test_wave0_run_writer_fixture_checkpoint_alias_bytes_are_frozen() -> None:
    tree = _wave0_fixture_bytes()

    assert tree["checkpoints/final.json"] == (
        b'{"checkpoint_path": "checkpoints/step-1", "step": 1}\n'
    )
    assert tree["checkpoints/best.json"] == (
        b'{"checkpoint_path": "checkpoints/step-1", "selector": "acc_top1", '
        b'"step": 1, "value": 0.5}\n'
    )


def test_wave0_run_writer_fixture_logging_rows_are_strict_compact_json() -> None:
    text = (WAVE0_RUN_WRITER_FIXTURE_ROOT / "logging.jsonl").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()

    assert text.endswith("\n")
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [row["split"] for row in rows] == ["train", "eval"]
    assert all(row["non_finite_fields"] == [] for row in rows)
    assert all(", " not in line and ": " not in line for line in lines)


def test_wave0_initialized_run_file_inventory_matches_the_fixture_prefix(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    assert writer.file_inventory() == (
        "logging.jsonl",
        "resolved_config.json",
        "run.json",
    )
    assert set(writer.file_inventory()) <= set(WAVE0_RUN_FILE_INVENTORY)
    assert tuple(sorted(writer.read_run())) == WAVE0_RUN_STATE_KEYS
