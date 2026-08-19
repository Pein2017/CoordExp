"""Wave-4 tests for ``src.artifacts.run_state``: pure run-state transitions
and the exact-resume checkpoint-publication admission core moved out of
``src/artifacts/run_writer.py`` (design decision 10).

``admit_exact_resume_checkpoint_publication_from_state`` is the pure
validation core of ``run_writer.admit_exact_resume_checkpoint_publication``:
the writer-level function (frozen by ``tests/training/test_exact_resume.py``
and ``tests/artifacts/test_training_state.py``, both of which must keep
passing unchanged) resolves the checkpoint directory and reads the parent
``run.json`` before delegating here. These tests exercise the pure core
directly with an already-loaded state mapping, so no filesystem I/O happens
in this module or in this test file's assertions about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.artifacts import run_state
from src.artifacts.resources import collect_resource_snapshot
from src.common.errors import ArtifactContractError


# ---------------------------------------------------------------------------
# _merge_resources
# ---------------------------------------------------------------------------


def test_merge_resources_is_a_no_op_when_resources_is_none() -> None:
    measurement: dict[str, Any] = {"resource_high_water": {"cpu": {"max_rss_bytes": 5}}}
    run_state._merge_resources(measurement, None)
    assert measurement == {"resource_high_water": {"cpu": {"max_rss_bytes": 5}}}


def test_merge_resources_seeds_high_water_from_the_first_sample() -> None:
    measurement: dict[str, Any] = {"resource_high_water": None}
    sample = collect_resource_snapshot()
    sample["cpu"]["max_rss_bytes"] = 10
    run_state._merge_resources(measurement, sample)
    assert measurement["resource_high_water"]["cpu"]["max_rss_bytes"] == 10


def test_merge_resources_keeps_the_running_maximum() -> None:
    measurement: dict[str, Any] = {"resource_high_water": None}
    low = collect_resource_snapshot()
    low["cpu"]["max_rss_bytes"] = 10
    high = collect_resource_snapshot()
    high["cpu"]["max_rss_bytes"] = 20
    run_state._merge_resources(measurement, low)
    run_state._merge_resources(measurement, high)
    run_state._merge_resources(measurement, low)
    assert measurement["resource_high_water"]["cpu"]["max_rss_bytes"] == 20


# ---------------------------------------------------------------------------
# _validate_finalization_checkpoint_progress
# ---------------------------------------------------------------------------


def _completed_event(
    *, step: int, completed_steps: int, consumed_packs: int
) -> dict[str, Any]:
    return {
        "status": "completed",
        "step": step,
        "committed_progress": {
            "schema": "coordexp-swift-checkpoint-committed-progress",
            "schema_version": 1,
            "completed_steps": completed_steps,
            "consumed_packs": consumed_packs,
            "optimizer_update_status": "applied",
            "finite_status": "finite",
        },
    }


def test_validate_finalization_checkpoint_progress_no_op_without_events() -> None:
    run_state._validate_finalization_checkpoint_progress(
        {"measurement": {"checkpoint_publication_events": []}},
        completed_steps=99,
        consumed_packs=1,
        checkpoint_event_count=0,
        optimizer_update_status=None,
        finite_status=None,
    )  # does not raise: nothing authoritative to disagree with


def test_validate_finalization_checkpoint_progress_admits_matching_terminal_state() -> (
    None
):
    state = {
        "measurement": {
            "checkpoint_publication_events": [
                _completed_event(step=1, completed_steps=1, consumed_packs=2),
                _completed_event(step=2, completed_steps=2, consumed_packs=4),
            ]
        }
    }
    run_state._validate_finalization_checkpoint_progress(
        state,
        completed_steps=2,
        consumed_packs=4,
        checkpoint_event_count=2,
        optimizer_update_status="applied",
        finite_status="finite",
    )  # does not raise


def test_validate_finalization_checkpoint_progress_rejects_disagreement() -> None:
    state = {
        "measurement": {
            "checkpoint_publication_events": [
                _completed_event(step=1, completed_steps=1, consumed_packs=2),
            ]
        }
    }
    with pytest.raises(ArtifactContractError) as exc_info:
        run_state._validate_finalization_checkpoint_progress(
            state,
            completed_steps=99,
            consumed_packs=2,
            checkpoint_event_count=1,
            optimizer_update_status="applied",
            finite_status="finite",
        )
    assert exc_info.value.code == "run_writer.finalize_checkpoint_progress_mismatch"
    assert exc_info.value.context["mismatched_fields"] == ["completed_steps"]


def test_validate_finalization_checkpoint_progress_rejects_malformed_event_inventory() -> (
    None
):
    with pytest.raises(ArtifactContractError) as exc_info:
        run_state._validate_finalization_checkpoint_progress(
            {"measurement": {"checkpoint_publication_events": "not-a-list"}},
            completed_steps=1,
            consumed_packs=1,
            checkpoint_event_count=1,
            optimizer_update_status=None,
            finite_status=None,
        )
    assert exc_info.value.code == "run_writer.finalize_checkpoint_progress_mismatch"


# ---------------------------------------------------------------------------
# admit_exact_resume_checkpoint_publication_from_state (pure admission core)
# ---------------------------------------------------------------------------


def _admitted_state(
    tmp_path: Path, *, checkpoint_step: int = 3
) -> tuple[dict[str, Any], Path, str, str]:
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dir = checkpoints_dir / f"step-{checkpoint_step}"
    checkpoint_dir.mkdir(parents=True)
    resolved_checkpoint_dir = checkpoint_dir.resolve()
    manifest_sha = "a" * 64
    aggregate_digest = "b" * 64
    checkpoint_identity = {
        "checkpoint_step": checkpoint_step,
        "resolved_path": str(resolved_checkpoint_dir),
        "training_state_manifest_file_sha256": manifest_sha,
        "training_state_aggregate_digest": aggregate_digest,
    }
    committed_progress = {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": checkpoint_step,
        "consumed_packs": checkpoint_step * 2,
        "optimizer_update_status": "applied",
        "finite_status": "finite",
    }
    event = {
        "schema": "coordexp-swift-checkpoint-publication-event",
        "schema_version": 2,
        "step": checkpoint_step,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "duration_seconds": 1.0,
        "duration_clock": "monotonic",
        "checkpoint_path": f"checkpoints/step-{checkpoint_step}",
        "is_final": False,
        "exact_training_state_enabled": True,
        "checkpoint_identity": checkpoint_identity,
        "inference_payload_identity": {"schema_version": 1, "files": {}},
        "committed_progress": committed_progress,
        "failure_code": None,
    }
    state = {
        "run_id": "parent-run",
        "run_dir": str(run_dir.resolve()),
        "continuation": {"segment_id": "parent-segment"},
        "completed_steps": checkpoint_step,
        "consumed_packs": checkpoint_step * 2,
        "checkpoint_event_count": 1,
        "final_optimizer_update_status": "applied",
        "final_finite_status": "finite",
        "measurement": {"checkpoint_publication_events": [event]},
    }
    return state, resolved_checkpoint_dir, manifest_sha, aggregate_digest


def _select(
    tmp_path: Path, *, checkpoint_step: int = 3, **state_overrides: Any
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Drive ``select_and_validate_checkpoint_publication_event`` alone.

    This is the pure phase that owns run identity, event selection, and
    checkpoint-identity validation -- everything except the committed
    inference payload identity, which the caller (``run_writer``) recomputes
    from real checkpoint files before calling
    ``admit_exact_resume_checkpoint_publication_from_state``.
    """

    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=checkpoint_step
    )
    state.update(state_overrides)
    run_dir = resolved_checkpoint_dir.parent.parent
    return run_state.select_and_validate_checkpoint_publication_event(
        state,
        checkpoint_step=checkpoint_step,
        resolved_checkpoint_dir=resolved_checkpoint_dir,
        resolved_run_dir=str(run_dir),
        parent_run_id="parent-run",
        parent_segment_id="parent-segment",
        training_state_manifest_file_sha256=manifest_sha,
        training_state_aggregate_digest=aggregate_digest,
    )


def _admit(
    tmp_path: Path, *, checkpoint_step: int = 3, **state_overrides: Any
) -> dict[str, Any]:
    """Drive the full two-phase admission the way ``run_writer`` does.

    The committed inference payload identity is admitted by
    ``src.artifacts.checkpoint_payload`` (already covered end-to-end by
    ``tests/artifacts/test_checkpoint_payload_identity.py``); here it is
    stubbed as an identity function so these tests isolate this module's own
    pure admission logic.
    """

    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=checkpoint_step
    )
    state.update(state_overrides)
    run_dir = resolved_checkpoint_dir.parent.parent
    event_index, event, admitted_checkpoint_identity = (
        run_state.select_and_validate_checkpoint_publication_event(
            state,
            checkpoint_step=checkpoint_step,
            resolved_checkpoint_dir=resolved_checkpoint_dir,
            resolved_run_dir=str(run_dir),
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
            training_state_manifest_file_sha256=manifest_sha,
            training_state_aggregate_digest=aggregate_digest,
        )
    )
    return run_state.admit_exact_resume_checkpoint_publication_from_state(
        state,
        checkpoint_step=checkpoint_step,
        event_index=event_index,
        event=event,
        admitted_checkpoint_identity=admitted_checkpoint_identity,
        admitted_payload_identity=dict(event["inference_payload_identity"]),
    )


def test_admission_returns_the_identity_progress_and_event_index(
    tmp_path: Path,
) -> None:
    result = _admit(tmp_path)
    assert result["event_index"] == 0
    assert result["checkpoint_identity"]["checkpoint_step"] == 3
    assert result["committed_progress"]["completed_steps"] == 3
    assert result["inference_payload_identity"] == {"schema_version": 1, "files": {}}


def test_admission_rejects_run_id_disagreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent run identity"):
        _select(tmp_path, run_id="a-different-run")


def test_admission_rejects_segment_disagreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent segment identity"):
        _select(tmp_path, continuation={"segment_id": "a-different-segment"})


def test_admission_rejects_run_dir_disagreement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent run path"):
        _select(tmp_path, run_dir="/somewhere/else")


def test_admission_rejects_missing_event_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event inventory is missing"):
        _select(tmp_path, measurement={})


def test_admission_rejects_when_no_event_matches_the_requested_step(
    tmp_path: Path,
) -> None:
    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=3
    )
    run_dir = resolved_checkpoint_dir.parent.parent
    with pytest.raises(ValueError, match="exactly one matching completed event"):
        run_state.select_and_validate_checkpoint_publication_event(
            state,
            checkpoint_step=4,
            resolved_checkpoint_dir=resolved_checkpoint_dir,
            resolved_run_dir=str(run_dir),
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
            training_state_manifest_file_sha256=manifest_sha,
            training_state_aggregate_digest=aggregate_digest,
        )


def test_admission_rejects_a_non_latest_completed_event(tmp_path: Path) -> None:
    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=3
    )
    later_event = dict(state["measurement"]["checkpoint_publication_events"][0])
    later_event["step"] = 4
    later_event["checkpoint_path"] = "checkpoints/step-4"
    state["measurement"]["checkpoint_publication_events"].append(later_event)
    run_dir = resolved_checkpoint_dir.parent.parent
    with pytest.raises(ValueError, match="not the latest completed publication event"):
        run_state.select_and_validate_checkpoint_publication_event(
            state,
            checkpoint_step=3,
            resolved_checkpoint_dir=resolved_checkpoint_dir,
            resolved_run_dir=str(run_dir),
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
            training_state_manifest_file_sha256=manifest_sha,
            training_state_aggregate_digest=aggregate_digest,
        )


def test_admission_rejects_checkpoint_identity_digest_mismatch(
    tmp_path: Path,
) -> None:
    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=3
    )
    run_dir = resolved_checkpoint_dir.parent.parent
    with pytest.raises(ValueError, match="checkpoint publication identity"):
        run_state.select_and_validate_checkpoint_publication_event(
            state,
            checkpoint_step=3,
            resolved_checkpoint_dir=resolved_checkpoint_dir,
            resolved_run_dir=str(run_dir),
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
            training_state_manifest_file_sha256="c" * 64,  # disagrees with the event
            training_state_aggregate_digest=aggregate_digest,
        )


def test_admission_rejects_payload_identity_disagreement(tmp_path: Path) -> None:
    state, resolved_checkpoint_dir, manifest_sha, aggregate_digest = _admitted_state(
        tmp_path, checkpoint_step=3
    )
    run_dir = resolved_checkpoint_dir.parent.parent
    event_index, event, admitted_checkpoint_identity = (
        run_state.select_and_validate_checkpoint_publication_event(
            state,
            checkpoint_step=3,
            resolved_checkpoint_dir=resolved_checkpoint_dir,
            resolved_run_dir=str(run_dir),
            parent_run_id="parent-run",
            parent_segment_id="parent-segment",
            training_state_manifest_file_sha256=manifest_sha,
            training_state_aggregate_digest=aggregate_digest,
        )
    )
    with pytest.raises(ValueError, match="inference payload identity"):
        run_state.admit_exact_resume_checkpoint_publication_from_state(
            state,
            checkpoint_step=3,
            event_index=event_index,
            event=event,
            admitted_checkpoint_identity=admitted_checkpoint_identity,
            admitted_payload_identity={"schema_version": 1, "files": {"different": True}},
        )


def test_admission_rejects_top_level_progress_not_bound_to_the_event(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not checkpoint-bound"):
        _admit(tmp_path, completed_steps=999)
