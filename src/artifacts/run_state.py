"""Pure run-state transition and exact-resume admission owner.

Per design decision 10, this module owns pure run-state transitions and the
exact-resume checkpoint-publication admission logic moved out of
``src/artifacts/run_writer.py``. Functions here receive an already-loaded
mapping (the parsed ``run.json`` state, or a fragment of it) and return a new
validated mapping or raise; none of them read or write files, resolve paths,
or read the wall clock. ``RunWriter`` remains the only owner that performs
that I/O -- it reads ``run.json``, resolves the checkpoint directory, and
sequences the atomic write, then delegates the pure validation here.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.artifacts.resources import merge_resource_high_water
from src.artifacts.run_schema import (
    _checkpoint_committed_progress,
    _checkpoint_publication_identity,
    _strict_json_dumps,
    _validate_nonnegative_finite,
    _validate_timestamp,
)
from src.common.errors import ArtifactContractError

_CHECKPOINT_PUBLICATION_EVENT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "step",
        "status",
        "started_at",
        "completed_at",
        "duration_seconds",
        "duration_clock",
        "checkpoint_path",
        "is_final",
        "exact_training_state_enabled",
        "checkpoint_identity",
        "inference_payload_identity",
        "committed_progress",
        "failure_code",
    }
)


def _merge_resources(
    measurement: dict[str, Any], resources: Mapping[str, Any] | None
) -> None:
    if resources is None:
        return
    sample = deepcopy(dict(resources))
    current = measurement.get("resource_high_water")
    measurement["resource_high_water"] = (
        merge_resource_high_water(sample, sample)
        if current is None
        else merge_resource_high_water(current, sample)
    )


def _validate_finalization_checkpoint_progress(
    state: Mapping[str, Any],
    *,
    completed_steps: int,
    consumed_packs: int,
    checkpoint_event_count: int,
    optimizer_update_status: str | None,
    finite_status: str | None,
) -> None:
    measurement = state.get("measurement")
    events = (
        measurement.get("checkpoint_publication_events", [])
        if isinstance(measurement, Mapping)
        else []
    )
    if not isinstance(events, list):
        raise ArtifactContractError(
            "checkpoint publication event inventory is invalid",
            code="run_writer.finalize_checkpoint_progress_mismatch",
        )

    authoritative: dict[str, Any] | None = None
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("status") != "completed":
            continue
        committed_progress = event.get("committed_progress")
        if not isinstance(committed_progress, Mapping):
            continue
        progress = _checkpoint_committed_progress(
            committed_progress,
            step=event.get("step"),
        )
        authoritative = {
            "completed_steps": progress["completed_steps"],
            "consumed_packs": progress["consumed_packs"],
            "checkpoint_event_count": event_index + 1,
            "optimizer_update_status": progress["optimizer_update_status"],
            "finite_status": progress["finite_status"],
        }

    if authoritative is None:
        return
    requested = {
        "completed_steps": completed_steps,
        "consumed_packs": consumed_packs,
        "checkpoint_event_count": checkpoint_event_count,
        "optimizer_update_status": optimizer_update_status,
        "finite_status": finite_status,
    }
    mismatched_fields = sorted(
        field
        for field, expected in authoritative.items()
        if type(requested[field]) is not type(expected) or requested[field] != expected
    )
    if mismatched_fields:
        raise ArtifactContractError(
            "terminal progress disagrees with the latest committed checkpoint event",
            code="run_writer.finalize_checkpoint_progress_mismatch",
            context={"mismatched_fields": mismatched_fields},
        )


def select_and_validate_checkpoint_publication_event(
    state: Mapping[str, Any],
    *,
    checkpoint_step: int,
    resolved_checkpoint_dir: Path,
    resolved_run_dir: str,
    parent_run_id: str,
    parent_segment_id: str,
    training_state_manifest_file_sha256: str,
    training_state_aggregate_digest: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Select and validate the one authoritative checkpoint-publication event.

    This is the first pure phase of
    ``run_writer.admit_exact_resume_checkpoint_publication``: the caller has
    already resolved and read the checkpoint's parent ``run.json`` into
    ``state`` (and resolved its own directory into ``resolved_run_dir``, a
    string, so this function never has to call ``Path.resolve()`` itself).
    Returns ``(event_index, event, admitted_checkpoint_identity)``.

    Recomputing the committed inference payload identity requires reading
    files under the checkpoint directory, which is I/O this module does not
    perform; the caller does that (through the same
    ``admit_inference_checkpoint_payload_identity`` binding
    ``RunWriter.record_checkpoint_publication_event`` uses, so both share one
    mockable import site) and passes the result to
    ``admit_exact_resume_checkpoint_publication_from_state`` below. Every
    ``ValueError`` here is caught and re-wrapped by the caller into the
    historical ``ArtifactContractError``.
    """

    _strict_json_dumps(state)
    if state.get("run_id") != parent_run_id:
        raise ValueError("parent run identity disagrees with training state")
    continuation = state.get("continuation")
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("segment_id") != parent_segment_id
    ):
        raise ValueError("parent segment identity disagrees with training state")
    if state.get("run_dir") != resolved_run_dir:
        raise ValueError("parent run path disagrees with checkpoint ancestry")

    measurement = state.get("measurement")
    events = (
        measurement.get("checkpoint_publication_events")
        if isinstance(measurement, Mapping)
        else None
    )
    if not isinstance(events, list):
        raise ValueError("checkpoint publication event inventory is missing")
    expected_checkpoint_path = f"checkpoints/step-{checkpoint_step}"
    candidates = [
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, Mapping)
        and event.get("status") == "completed"
        and (
            event.get("step") == checkpoint_step
            or event.get("checkpoint_path") == expected_checkpoint_path
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "parent run must contain exactly one matching completed event"
        )
    event_index, event = candidates[0]
    completed_event_indices = [
        index
        for index, inventory_event in enumerate(events)
        if isinstance(inventory_event, Mapping)
        and inventory_event.get("status") == "completed"
    ]
    if not completed_event_indices or completed_event_indices[-1] != event_index:
        raise ValueError(
            "requested checkpoint is not the latest completed publication event"
        )
    if set(event) != _CHECKPOINT_PUBLICATION_EVENT_FIELDS:
        raise ValueError("checkpoint publication event has an invalid field set")
    if (
        event.get("schema") != "coordexp-swift-checkpoint-publication-event"
        or event.get("schema_version") != 2
        or event.get("step") != checkpoint_step
        or event.get("checkpoint_path") != expected_checkpoint_path
        or event.get("status") != "completed"
        or event.get("duration_clock") != "monotonic"
        or not isinstance(event.get("is_final"), bool)
        or event.get("exact_training_state_enabled") is not True
        or event.get("failure_code") is not None
    ):
        raise ValueError("checkpoint publication event-v2 is inconsistent")
    _validate_timestamp(event["started_at"], field="started_at")
    _validate_timestamp(event["completed_at"], field="completed_at")
    _validate_nonnegative_finite(
        event["duration_seconds"], field="checkpoint_publication.duration_seconds"
    )
    expected_checkpoint_identity = {
        "checkpoint_step": checkpoint_step,
        "resolved_path": str(resolved_checkpoint_dir),
        "training_state_manifest_file_sha256": (
            training_state_manifest_file_sha256
        ),
        "training_state_aggregate_digest": training_state_aggregate_digest,
    }
    admitted_checkpoint_identity = _checkpoint_publication_identity(
        event["checkpoint_identity"],
        step=checkpoint_step,
        checkpoint_dir=resolved_checkpoint_dir,
    )
    if admitted_checkpoint_identity != expected_checkpoint_identity:
        raise ValueError(
            "checkpoint publication identity disagrees with training state"
        )
    return event_index, event, admitted_checkpoint_identity


def admit_exact_resume_checkpoint_publication_from_state(
    state: Mapping[str, Any],
    *,
    checkpoint_step: int,
    event_index: int,
    event: Mapping[str, Any],
    admitted_checkpoint_identity: Mapping[str, Any],
    admitted_payload_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Finish exact-resume checkpoint-publication admission.

    Takes the event already selected and validated by
    ``select_and_validate_checkpoint_publication_event`` above, plus the
    caller's already-recomputed committed inference payload identity (that
    I/O stays with the caller; see that function's docstring for why). Every
    ``ValueError`` here is caught and re-wrapped by the caller into the
    historical ``ArtifactContractError``.
    """

    if admitted_payload_identity != event["inference_payload_identity"]:
        raise ValueError("inference payload identity is not the committed identity")
    progress = _checkpoint_committed_progress(
        event["committed_progress"],
        step=checkpoint_step,
    )
    latest_progress = {
        "completed_steps": progress["completed_steps"],
        "consumed_packs": progress["consumed_packs"],
        "checkpoint_event_count": event_index + 1,
        "final_optimizer_update_status": progress["optimizer_update_status"],
        "final_finite_status": progress["finite_status"],
    }
    mismatched_top_level = sorted(
        field
        for field, expected in latest_progress.items()
        if type(state.get(field)) is not type(expected)
        or state.get(field) != expected
    )
    if mismatched_top_level:
        raise ValueError("parent top-level progress is not checkpoint-bound")
    return {
        "checkpoint_identity": admitted_checkpoint_identity,
        "committed_progress": progress,
        "event_index": event_index,
        "inference_payload_identity": dict(admitted_payload_identity),
    }
